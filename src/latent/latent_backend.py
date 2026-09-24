"""Experiment 13: local inference backends for text and continuous channels.

``src/llm.py`` talks to OpenRouter over a text API.  It cannot read a
transformer's activations or accept input embeddings, so a latent channel needs
a separate backend.  This module defines that interface twice:

* :class:`TransformersBackend` -- one pinned open-weight checkpoint, shared
  frozen weights for writer and reader, fresh per-agent state, and an explicit
  ``inputs_embeds`` path.  Imports ``torch``/``transformers`` lazily so nothing
  here downloads a model at import time.
* :class:`FakeBackend` -- deterministic, dependency-free, and *decodable*.  It
  exists so the entire pipeline, including the shuffle/zero/replay controls, can
  be regression-tested with no GPU and no weights.  It is a plumbing simulator,
  never a source of experimental results.

The two share one interface so the runner cannot accidentally take a different
code path under test than it takes on the cluster.

Alignment note: final-layer hidden states and input embeddings do not live in
the same space.  Feeding the former straight in is the single most common way a
latent pilot "fails" for reasons that have nothing to do with the idea.  The
mapping is therefore explicit, versioned and recorded (:data:`ALIGNMENT_VERSION`),
never implicit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import hashlib
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import numpy as np
from latent_alignment import AlignmentConfig, VERSION as ALIGNMENT_VERSION, NORM_VERSION


NOT_IN_MESSAGE = "not in the message"


# ---------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class BackendConfig:
    """Everything that changes a backend's output, and therefore its cache key."""

    model_id: str = "Qwen/Qwen3-4B"
    revision: str = ""                 # pinned commit sha; empty means unpinned
    dtype: str = "bfloat16"
    device: str = "cuda"
    enable_thinking: bool = False
    answer_max_tokens: int = 48
    transcript_max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0
    alignment: str = ALIGNMENT_VERSION
    hidden_layer: int = -1             # -1 = final layer
    alignment_regularization: float = 1e-3
    alignment_eigen_floor: float = 1e-6
    alignment_snap_ratio: float = 0.3
    alignment_vocab_chunk: int = 8192

    def __post_init__(self):
        if self.alignment not in (ALIGNMENT_VERSION, NORM_VERSION):
            raise ValueError("unknown alignment version")
        if self.temperature != 0 or self.hidden_layer != -1:
            raise ValueError("this protocol requires greedy decoding and the last block")
        AlignmentConfig(self.alignment_regularization, self.alignment_eigen_floor,
                        self.alignment_snap_ratio, self.alignment_vocab_chunk)

    def fingerprint(self) -> str:
        import json
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def record(self) -> dict:
        return {**asdict(self),
            "model_id": self.model_id, "revision": self.revision,
            "dtype": self.dtype, "device": self.device,
            "enable_thinking": self.enable_thinking,
            "answer_max_tokens": self.answer_max_tokens,
            "transcript_max_tokens": self.transcript_max_tokens,
            "temperature": self.temperature, "seed": self.seed,
            "alignment": self.alignment, "hidden_layer": self.hidden_layer,
            "backend_fingerprint": self.fingerprint(),
        }


@dataclass
class Usage:
    """Resource accounting for one backend call.

    Kept per call rather than per experiment because the design requires sender
    prefill, sender decoding, alignment and reader prefill to be reported as
    separate axes -- a total alone cannot distinguish "fewer positions" from
    "less compute".
    """

    prefill_positions: int = 0
    generated_tokens: int = 0
    prefix_positions: int = 0
    seconds: float = 0.0
    calls: int = 1
    prefill_s: float = 0.0
    decode_s: float = 0.0
    state_extract_s: float = 0.0
    peak_memory_bytes: int = 0

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            prefill_positions=self.prefill_positions + other.prefill_positions,
            generated_tokens=self.generated_tokens + other.generated_tokens,
            prefix_positions=self.prefix_positions + other.prefix_positions,
            seconds=self.seconds + other.seconds,
            calls=self.calls + other.calls,
            prefill_s=self.prefill_s + other.prefill_s,
            decode_s=self.decode_s + other.decode_s,
            state_extract_s=self.state_extract_s + other.state_extract_s,
            peak_memory_bytes=max(self.peak_memory_bytes, other.peak_memory_bytes))

    def record(self) -> dict:
        return {**asdict(self), "prefill_positions": self.prefill_positions,
                "generated_tokens": self.generated_tokens,
                "prefix_positions": self.prefix_positions,
                "seconds": round(self.seconds, 6), "calls": self.calls}


@dataclass
class Generation:
    text: str
    usage: Usage
    error: str = ""
    cache_hit: bool = False


@dataclass
class Transcript:
    """A sender's internal trajectory H, plus the states actually transmitted.

    ``states`` are the last ``k`` valid message positions.  ``realized_length``
    is how many tokens H actually ran to: a 32-vector prefix formed by
    generating 256 tokens costs 256 tokens of decoding, and reporting only the
    32 would be the accounting trap the design warns about.
    """

    text: str
    states: np.ndarray
    realized_length: int
    requested_k: int
    usage: Usage
    truncated: bool = False
    token_ids: tuple[int, ...] = ()
    reference_embeddings: np.ndarray | None = None
    tail_text: str = ""
    state_convention: str = "predicting-token-last-block-before-final-norm-v1"

    @property
    def delivered_k(self) -> int:
        return int(self.states.shape[0])

    @property
    def short(self) -> bool:
        """H produced fewer than k valid positions -- flagged, never padded."""
        return self.delivered_k < self.requested_k


# ---------------------------------------------------------------------------
# Interface


class Backend:
    """The operations a channel needs.  Both implementations provide all of them."""

    config: BackendConfig

    def count_positions(self, text: str) -> int:
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 system: str | None = None) -> Generation:
        raise NotImplementedError

    def transcript_states(self, prompt: str, k: int,
                          max_new_tokens: int | None = None,
                          system: str | None = None) -> Transcript:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        """Input-embedding rows for text.  Used by the replay control."""
        raise NotImplementedError

    def teacher_force(self, prompt: str, transcript: Transcript, k: int,
                      system: str | None = None) -> Transcript:
        raise NotImplementedError

    def read_message(self, message: str | np.ndarray, question: str,
                     schema: str = "") -> Generation:
        from latent_handoff import READER_INSTRUCTION
        if isinstance(message, str):
            return self.generate(f"{schema}\nMESSAGE FROM THE OTHER AGENT:\n{message}"
                                 f"\n\nQUESTION: {question}\nANSWER:",
                                 system=READER_INSTRUCTION)
        return self.generate_from_prefix(message, f"\n\nQUESTION: {question}\nANSWER:",
                                         system=READER_INSTRUCTION + "\n" + schema)

    def generate_from_prefix(self, prefix: np.ndarray, suffix: str,
                             max_new_tokens: int | None = None,
                             system: str | None = None) -> Generation:
        """Answer from delivered vectors plus a text suffix. Fresh reader state."""
        raise NotImplementedError

    @property
    def hidden_width(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Alignment
#
# Declared once, used by both backends, recorded in every payload's metadata.


def align_states(states: np.ndarray, target_rms: float,
                 anchor: np.ndarray | None = None) -> np.ndarray:
    """Map final-layer hidden states into the reader's input-embedding scale.

    Norm calibration only: each row is rescaled so the batch RMS matches the
    reader's input-embedding RMS, optionally after subtracting an anchor (the
    mean input embedding).  This is deliberately the weakest defensible mapping
    -- this is the separately named norm-only diagnostic. A failure here does
    not establish failure of StateBridge or of frozen interfaces generally.
    """
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected (positions, width), got {values.shape}")
    if values.size == 0:
        return values
    centred = values - anchor if anchor is not None else values
    rms = float(np.sqrt(np.mean(np.square(centred))))
    if rms <= 0 or not math.isfinite(rms):
        return np.zeros_like(centred)
    scaled = centred * (target_rms / rms)
    if anchor is not None:
        scaled = scaled + anchor
    return np.ascontiguousarray(scaled, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake backend


_WORD = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS = frozenset(
    "a an the of and or to in on at is are was were be been by for with from "
    "that this it its as into than then so such".split())


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


@dataclass
class FakeFact:
    """One thing the simulated reader is able to answer, if it receives it."""

    evidence: str
    question: str
    answer: str
    key_tokens: tuple[str, ...] = field(default=())


class FakeBackend(Backend):
    """A deterministic, decodable stand-in for a frozen local model.

    Design constraints that make it useful rather than decorative:

    * **Shared frozen weights.** Writer and reader use the same embedding table,
      derived from a seed.  That mirrors the real single-backbone setup, and it
      is what makes nearest-neighbour decoding of a delivered prefix legitimate
      rather than a hidden side channel.
    * **Source-conditioned states.** A hidden state row mixes the token's own
      embedding with a running context vector over everything read so far.  So a
      state produced while reading the source genuinely differs from one
      produced from the visible transcript alone, and the source-free replay
      control has something real to detect.
    * **Reading is gated on delivery.** The simulated reader answers a question
      only when the corresponding evidence actually arrives in its material.  A
      zeroed or cross-dossier payload therefore fails, as it must.
    """

    def __init__(self, config: BackendConfig | None = None, width: int = 64,
                 vocab_seed: str = "latent-fake-v1", context_mix: float = 0.25) -> None:
        self.config = config or BackendConfig(model_id="fake/deterministic",
                                              device="cpu", dtype="float32")
        self.width = int(width)
        self.vocab_seed = vocab_seed
        self.context_mix = float(context_mix)
        self._emb_cache: dict[str, np.ndarray] = {}
        self._vocab: dict[str, np.ndarray] = {}
        self._facts: list[FakeFact] = []
        self.call_log: list[str] = []

    # -- weights ------------------------------------------------------------

    @property
    def hidden_width(self) -> int:
        return self.width

    def _emb(self, token: str) -> np.ndarray:
        cached = self._emb_cache.get(token)
        if cached is not None:
            return cached
        digest = hashlib.sha256(f"{self.vocab_seed}\x1f{token}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vec = rng.standard_normal(self.width).astype(np.float32)
        vec /= float(np.linalg.norm(vec)) or 1.0
        self._emb_cache[token] = vec
        self._vocab[token] = vec
        return vec

    def _drops(self, token: str) -> bool:
        """Deterministic simulated compression: which tokens a writer omits.

        Function words go first, then a fixed hash-selected minority, so H is
        shorter than its source without losing the values a question needs.
        """
        if token in _STOPWORDS:
            return True
        digest = hashlib.sha256(f"{self.vocab_seed}\x1fdrop\x1f{token}".encode()).digest()
        return digest[0] % 8 == 0

    def learn_vocabulary(self, texts: Sequence[str]) -> None:
        """Pre-embed the corpus.  This is the frozen backbone's vocabulary.

        Called once at setup for both writer and reader -- they are the same
        model.  It is *not* an answer table: knowing a token exists does not
        tell the reader which fact was delivered.
        """
        for text in texts:
            for token in _tokens(text):
                self._emb(token)

    def register_facts(self, facts: Sequence[FakeFact]) -> None:
        """Teach the simulated reader what it is capable of reading.

        Answering still requires the evidence to arrive; this only defines what
        a competent reader *would* extract from evidence it actually receives.
        """
        for fact in facts:
            key = tuple(t for t in _tokens(fact.evidence) if len(t) > 3)
            self._facts.append(replace(fact, key_tokens=key))

    # -- text ---------------------------------------------------------------

    def count_positions(self, text: str) -> int:
        return len(_tokens(text))

    def embed_text(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        if not toks:
            return np.zeros((0, self.width), dtype=np.float32)
        return np.stack([self._emb(t) for t in toks]).astype(np.float32)

    def _contextual_states(self, text: str) -> np.ndarray:
        """Row i = token i's embedding mixed with the context read before it."""
        toks = _tokens(text)
        if not toks:
            return np.zeros((0, self.width), dtype=np.float32)
        rows, running = [], np.zeros(self.width, dtype=np.float32)
        for i, token in enumerate(toks):
            emb = self._emb(token)
            running = running + emb
            context = running / float(i + 1)
            rows.append((1.0 - self.context_mix) * emb + self.context_mix * context)
        return np.stack(rows).astype(np.float32)

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 system: str | None = None) -> Generation:
        t0 = time.perf_counter()
        self.call_log.append("generate")
        text = self._answer_from_material(prompt)
        usage = Usage(prefill_positions=self.count_positions(prompt),
                      generated_tokens=self.count_positions(text),
                      seconds=time.perf_counter() - t0)
        return Generation(text=text, usage=usage)

    def transcript_states(self, prompt: str, k: int,
                          max_new_tokens: int | None = None,
                          system: str | None = None) -> Transcript:
        t0 = time.perf_counter()
        self.call_log.append("transcript")
        cap = int(max_new_tokens or self.config.transcript_max_tokens)
        # H is a *written message*, not a copy of the source: a real writer
        # compresses, so H and the material it was written from must differ.
        # Without that the source-free replay control has nothing to detect,
        # because re-encoding H would be re-encoding the source.
        material = _material_block(prompt)
        toks = [t for t in _tokens(material) if not self._drops(t)][:cap]
        text = " ".join(toks)
        # States are produced *while reading the prompt*, so they are
        # source-conditioned: the running context covers the whole prompt.
        # Only the message positions are transmittable, matching the real
        # backend's `hidden[0, prompt_len:, :]` slice -- otherwise the payload
        # would carry prompt positions the sender never "said".
        states = self._contextual_states(f"{material} {text}")
        message_states = states[len(_tokens(material)):] if toks else states[:0]
        tail = message_states[-int(k):] if k > 0 else message_states[:0]
        usage = Usage(prefill_positions=self.count_positions(prompt),
                      generated_tokens=len(toks),
                      seconds=time.perf_counter() - t0)
        return Transcript(text=text, states=np.ascontiguousarray(tail),
                          realized_length=len(toks), requested_k=int(k),
                          usage=usage, truncated=len(toks) >= cap,
                          reference_embeddings=self.embed_text(" ".join(toks[-k:])) if k else self.embed_text(""),
                          tail_text=" ".join(toks[-k:]) if k else "",
                          state_convention="fake-contextual-token-v2")

    def teacher_force(self, prompt, transcript, k, system=None):
        self.call_log.append("teacher_force")
        material = _material_block(prompt)
        message = _tokens(transcript.text)
        states = self._contextual_states(material + " " + transcript.text)[len(_tokens(material)):]
        tail = states[-k:] if k else states[:0]
        return Transcript(text=transcript.text, states=tail.copy(),
                          realized_length=len(message), requested_k=k,
                          usage=Usage(prefill_positions=self.count_positions(prompt)+len(message),
                                      generated_tokens=0),
                          reference_embeddings=self.embed_text(" ".join(message[-k:])) if k else self.embed_text(""),
                          tail_text=" ".join(message[-k:]) if k else "",
                          state_convention="fake-contextual-token-v2")

    def generate_from_prefix(self, prefix: np.ndarray, suffix: str,
                             max_new_tokens: int | None = None,
                             system: str | None = None) -> Generation:
        t0 = time.perf_counter()
        self.call_log.append("generate_from_prefix")
        decoded = self.decode_prefix(prefix)
        material = f"MESSAGE FROM THE OTHER AGENT:\n{decoded}\n{suffix}"
        text = self._answer_from_material(material)
        usage = Usage(prefill_positions=self.count_positions(suffix),
                      prefix_positions=int(np.asarray(prefix).shape[0]),
                      generated_tokens=self.count_positions(text),
                      seconds=time.perf_counter() - t0)
        return Generation(text=text, usage=usage)

    def decode_prefix(self, prefix: np.ndarray) -> str:
        """Nearest-neighbour readout against the shared frozen vocabulary.

        A zeroed or cross-source prefix decodes to something, but not to the
        evidence the question needs -- which is exactly the behaviour the
        payload-dependence controls are looking for.
        """
        rows = np.asarray(prefix, dtype=np.float32)
        if rows.ndim != 2 or rows.size == 0 or not self._vocab:
            return ""
        tokens = list(self._vocab)
        table = np.stack([self._vocab[t] for t in tokens])
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        usable = norms[:, 0] > 1e-6
        scores = rows @ table.T
        out = []
        for i, ok in enumerate(usable):
            if ok:
                out.append(tokens[int(np.argmax(scores[i]))])
        return " ".join(out)

    # -- the simulated reader ----------------------------------------------

    def _answer_from_material(self, prompt: str) -> str:
        question = _question_block(prompt)
        material = _material_block(prompt)
        if not question:
            return material[:200]
        q_tokens = set(_tokens(question))
        m_tokens = set(_tokens(material))
        best, best_score = None, 0.0
        for fact in self._facts:
            if not q_tokens & set(_tokens(fact.question)):
                continue
            if not fact.key_tokens:
                continue
            covered = sum(1 for t in fact.key_tokens if t in m_tokens)
            score = covered / len(fact.key_tokens)
            overlap = len(q_tokens & set(_tokens(fact.question))) / max(len(q_tokens), 1)
            score *= 0.5 + 0.5 * overlap
            if score > best_score:
                best, best_score = fact, score
        if best is not None and best_score >= 0.5:
            return best.answer
        return NOT_IN_MESSAGE


def _material_block(prompt: str) -> str:
    """The evidence part of a prompt, excluding instructions and the question."""
    text = prompt
    for marker in ("MESSAGE FROM THE OTHER AGENT:", "EVIDENCE:", "SOURCE:"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for marker in ("\nQUESTION:", "\nANSWER:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _question_block(prompt: str) -> str:
    if "QUESTION:" not in prompt:
        return ""
    tail = prompt.split("QUESTION:", 1)[1]
    return tail.split("\nANSWER:", 1)[0].strip()


# ---------------------------------------------------------------------------
# Real backend


def __getattr__(name):
    if name == "TransformersBackend":
        from latent_transformers import TransformersBackend
        return TransformersBackend
    raise AttributeError(name)


def build_backend(config: BackendConfig, fake: bool = False) -> Backend:
    from latent_transformers import TransformersBackend
    return FakeBackend(config) if fake else TransformersBackend(config)
