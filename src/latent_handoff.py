"""Experiment 13: the sealed transport record for non-language handoffs.

``handoffs.SealedHandoff`` is four strings and stays that way.  A continuous
payload cannot be a string, so this module adds a *separate* sealed record
rather than widening the existing one.  The guarantee is restated, not relaxed:

* the reader is given exactly one payload, one evaluation question, and fixed
  public decoding instructions;
* the payload is **immutable serialized bytes**.  A frozen dataclass holding a
  live tensor is not sufficient -- it keeps a reference into the sender's
  process, so "the receiver only saw the message" would be unverifiable.  Bytes
  survive a process boundary; a tensor handle does not;
* metadata is an allowlisted public schema.  Evaluator card ids, fact ids, gold
  answers and source paths are not in it and cannot be smuggled through it.

Byte accounting lives here too, because the byte count is a property of the
serialized payload and must not be recomputed (differently) by each caller.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "latent-handoff-v1"

TEXT_DTYPE = "utf-8"
VECTOR_DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2}
ALL_DTYPES = {TEXT_DTYPE: 1, **VECTOR_DTYPES}

#: The only metadata keys a reader-visible schema block may carry.  Anything
#: else is provenance and belongs in the run record, never in the payload.
PUBLIC_METADATA_KEYS = frozenset({
    "codec", "codec_version", "dtype", "positions", "shape", "channel",
    "normalization", "alignment",
})

#: Patterns that must never appear in a sealed payload's metadata.  These are
#: the evaluator's own identifiers; if one reaches the reader it can index back
#: into the corpus and answer a question the payload never carried.
_FORBIDDEN_METADATA = (
    re.compile(r"\bC\d{2}\b"),                    # evidence-card ids
    re.compile(r"\bfic:\d+", re.IGNORECASE),      # fictional fact/dossier ids
    re.compile(r"\bside\d\b", re.IGNORECASE),     # fact roles
)


class PayloadError(ValueError):
    """A sealed payload violated the transport contract."""


# ---------------------------------------------------------------------------
# Hashing and opaque identifiers


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def opaque_id(*parts: object) -> str:
    """A reader-side id that cannot be resolved back into the corpus.

    Shuffle and zero controls are only meaningful if the reader-side id differs
    from the donor payload's id -- otherwise a cache keyed on the id restores
    the *original* answer and the control silently passes.
    """
    return "px" + canonical_hash([str(p) for p in parts])[:24]


# ---------------------------------------------------------------------------
# bfloat16 without a torch dependency
#
# numpy has no bfloat16.  bf16 is the top 16 bits of an IEEE float32, so the
# conversion is a shift, and the offline selftest stays torch-free.


def f32_to_bf16_bytes(array: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(array, dtype="<f4")
    bits = f32.view(np.uint32)
    # Round-to-nearest-even on the discarded low 16 bits.
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def bf16_bytes_to_f32(payload: bytes, shape: Sequence[int]) -> np.ndarray:
    half = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16
    return half.view(np.float32).reshape(tuple(shape)).copy()


# ---------------------------------------------------------------------------
# The sealed record


@dataclass(frozen=True)
class SealedLatentPayload:
    """Everything the reader is permitted to receive, for any channel.

    ``payload`` is the single content channel.  ``question`` is the one
    evaluation question.  ``metadata`` is the fixed public schema the reader may
    be told about the encoding.  There is no field through which the source
    document, the unselected cards, the gold answers or the sender's tensors can
    arrive.
    """

    qid: str
    question: str
    channel: str
    dtype: str
    shape: tuple[int, ...]
    positions: int
    payload: bytes
    payload_sha256: str
    metadata: tuple[tuple[str, Any], ...] = field(default=())
    schema_version: str = SCHEMA_VERSION

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise PayloadError(
                "payload must be immutable `bytes`, not "
                f"{type(self.payload).__name__} -- a live tensor, memoryview or "
                "bytearray keeps a reference into the sender's process"
            )
        for name in ("qid", "question", "channel", "dtype"):
            if not isinstance(getattr(self, name), str):
                raise PayloadError(f"{name} must be a string")
        if self.dtype not in ALL_DTYPES:
            raise PayloadError(f"unknown dtype {self.dtype!r}")
        if not isinstance(self.shape, tuple) or not all(
                isinstance(n, int) and n >= 0 for n in self.shape):
            raise PayloadError("shape must be a tuple of non-negative ints")
        if not isinstance(self.positions, int) or self.positions < 0:
            raise PayloadError("positions must be a non-negative int")
        if self.payload_sha256 != sha256_bytes(self.payload):
            raise PayloadError("payload_sha256 does not match the payload bytes")

        if self.dtype == TEXT_DTYPE:
            if len(self.shape) != 1 or self.shape[0] != len(self.payload):
                raise PayloadError("text shape must be (n_bytes,)")
        else:
            width = VECTOR_DTYPES[self.dtype]
            expected = width * int(np.prod(self.shape)) if self.shape else 0
            if expected != len(self.payload):
                raise PayloadError(
                    f"{self.dtype}{tuple(self.shape)} needs {expected} bytes, "
                    f"got {len(self.payload)}")
            if len(self.shape) != 2:
                raise PayloadError("vector shape must be (positions, width)")
            if self.shape[0] != self.positions:
                raise PayloadError(
                    f"a vector payload occupies one receiver position per row: "
                    f"shape[0]={self.shape[0]} but positions={self.positions}")

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if not isinstance(self.metadata, tuple):
            raise PayloadError("metadata must be a tuple of (key, value) pairs")
        seen = set()
        for pair in self.metadata:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise PayloadError("metadata entries must be (key, value) pairs")
            key, value = pair
            if key not in PUBLIC_METADATA_KEYS:
                raise PayloadError(
                    f"metadata key {key!r} is not in the public schema "
                    f"{sorted(PUBLIC_METADATA_KEYS)}")
            if key in seen:
                raise PayloadError(f"duplicate metadata key {key!r}")
            seen.add(key)
            if not isinstance(value, (str, int, float, bool, type(None), tuple)):
                raise PayloadError(f"metadata value for {key!r} is not a scalar")
            if isinstance(value, tuple) and any(type(v) not in (str, int, float, bool, type(None)) for v in value):
                raise PayloadError("metadata tuples must contain immutable scalars")
            for pattern in _FORBIDDEN_METADATA:
                if pattern.search(str(value)):
                    raise PayloadError(
                        f"metadata {key}={value!r} carries an evaluator identifier")
        # The reader is told the payload's size through this block, so it must
        # not be able to contradict the payload it describes.
        declared = dict(self.metadata)
        if "positions" in declared and declared["positions"] != self.positions:
            raise PayloadError(
                f"metadata declares {declared['positions']} positions but the "
                f"payload occupies {self.positions}")
        if "dtype" in declared and declared["dtype"] != self.dtype:
            raise PayloadError(
                f"metadata declares dtype {declared['dtype']!r} but the payload "
                f"is {self.dtype!r}")
        if "channel" in declared and declared["channel"] != self.channel:
            raise PayloadError(
                f"metadata declares channel {declared['channel']!r} but the "
                f"payload is {self.channel!r}")

    # -- accessors ----------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Serialized content bytes.  Schema/header bytes are counted separately."""
        return len(self.payload)

    @property
    def metadata_bytes(self) -> int:
        """Bytes the public schema block costs on the wire."""
        return len(json.dumps(dict(self.metadata), sort_keys=True,
                              separators=(",", ":")).encode("utf-8"))

    @property
    def total_bytes(self) -> int:
        return self.nbytes + self.metadata_bytes

    def as_text(self) -> str:
        if self.dtype != TEXT_DTYPE:
            raise PayloadError(f"payload is {self.dtype}, not text")
        return self.payload.decode("utf-8")

    def as_array(self) -> np.ndarray:
        """A fresh float32 copy.  Never a view into the sealed bytes."""
        if self.dtype == TEXT_DTYPE:
            raise PayloadError("payload is text, not vectors")
        if self.dtype == "bfloat16":
            return bf16_bytes_to_f32(self.payload, self.shape)
        code = {"float32": "<f4", "float16": "<f2"}[self.dtype]
        return (np.frombuffer(self.payload, dtype=code)
                .reshape(self.shape).astype(np.float32, copy=True))

    def public_schema(self) -> dict:
        """The block a reader may be shown alongside a non-text payload."""
        return dict(self.metadata)

    def record(self) -> dict:
        """Provenance for the run log.  Deliberately excludes the payload itself."""
        return {
            "schema_version": self.schema_version,
            "qid": self.qid,
            "channel": self.channel,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "positions": self.positions,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.nbytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Constructors


def _metadata_tuple(metadata: dict | None) -> tuple[tuple[str, Any], ...]:
    if not metadata:
        return ()
    return tuple(sorted((k, v) for k, v in metadata.items()))


def seal_text(qid: str, question: str, channel: str, text: str, positions: int,
              metadata: dict | None = None) -> SealedLatentPayload:
    """Seal a text channel (prose or typed records).

    ``positions`` is the tokenizer-measured receiver cost, which is *not*
    derivable from the byte count -- so it is passed in from the backend that
    owns the tokenizer, never guessed here.
    """
    if not isinstance(text, str):
        raise PayloadError("seal_text takes a string")
    blob = text.encode("utf-8")
    meta = dict(metadata or {})
    meta.setdefault("channel", channel)
    meta.setdefault("dtype", TEXT_DTYPE)
    return SealedLatentPayload(
        qid=qid, question=question, channel=channel, dtype=TEXT_DTYPE,
        shape=(len(blob),), positions=int(positions), payload=blob,
        payload_sha256=sha256_bytes(blob), metadata=_metadata_tuple(meta))


def seal_vectors(qid: str, question: str, channel: str, array: Any,
                 dtype: str = "bfloat16",
                 metadata: dict | None = None) -> SealedLatentPayload:
    """Seal a continuous prefix.

    The array is detached and serialized immediately, so the returned record
    shares no memory with the sender.  ``positions`` is the row count: one
    delivered vector is one receiver input position.
    """
    if dtype not in VECTOR_DTYPES:
        raise PayloadError(f"{dtype!r} is not a vector dtype")
    values = _to_float32_matrix(array)
    if dtype == "bfloat16":
        blob = f32_to_bf16_bytes(values)
    else:
        code = {"float32": "<f4", "float16": "<f2"}[dtype]
        blob = np.ascontiguousarray(values, dtype=code).tobytes()
    meta = dict(metadata or {})
    meta.setdefault("channel", channel)
    meta.setdefault("dtype", dtype)
    return SealedLatentPayload(
        qid=qid, question=question, channel=channel, dtype=dtype,
        shape=(int(values.shape[0]), int(values.shape[1])),
        positions=int(values.shape[0]), payload=blob,
        payload_sha256=sha256_bytes(blob), metadata=_metadata_tuple(meta))


def _to_float32_matrix(array: Any) -> np.ndarray:
    """Accept numpy or torch; always return an owned 2-D float32 copy."""
    if hasattr(array, "detach"):          # torch.Tensor, without importing torch
        array = array.detach().to("cpu").to(dtype=__import__("torch").float32).numpy()
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise PayloadError(f"expected a (positions, width) matrix, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise PayloadError("payload contains NaN or Inf")
    return np.ascontiguousarray(values)


# ---------------------------------------------------------------------------
# Round-trip across a process boundary


def to_wire(payload: SealedLatentPayload) -> bytes:
    """Serialize the whole record, including its schema, to one blob.

    The point of this function is the test that uses it: a reader in a fresh
    process must be able to reconstruct exactly what was sealed, from bytes
    alone.  If anything the reader needs is *not* reachable this way, it was
    travelling through a second channel.
    """
    header = {
        "schema_version": payload.schema_version, "qid": payload.qid,
        "question": payload.question, "channel": payload.channel,
        "dtype": payload.dtype, "shape": list(payload.shape),
        "positions": payload.positions, "payload_sha256": payload.payload_sha256,
        "metadata": dict(payload.metadata),
    }
    blob = json.dumps(header, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return len(blob).to_bytes(4, "little") + blob + payload.payload


def from_wire(wire: bytes) -> SealedLatentPayload:
    if type(wire) is not bytes or len(wire) < 4:
        raise PayloadError("truncated wire header")
    size = int.from_bytes(wire[:4], "little")
    if size > len(wire) - 4:
        raise PayloadError("truncated wire record")
    header = json.loads(wire[4:4 + size].decode("utf-8"))
    body = bytes(wire[4 + size:])
    return SealedLatentPayload(
        qid=header["qid"], question=header["question"], channel=header["channel"],
        dtype=header["dtype"], shape=tuple(header["shape"]),
        positions=header["positions"], payload=body,
        payload_sha256=header["payload_sha256"],
        metadata=_metadata_tuple(header.get("metadata") or {}),
        schema_version=header["schema_version"])


# ---------------------------------------------------------------------------
# Reader-side material


READER_INSTRUCTION = (
    "You are answering one question using only the message another agent sent you. "
    "You cannot see their source material. If the message does not contain the "
    "answer, say \"not in the message\". Answer with the shortest exact span or "
    "value that answers the question, and nothing else."
)


def reader_prompt(payload: SealedLatentPayload) -> str:
    """The text a reader sees for a *text* channel.

    For a continuous channel the delivered vectors replace this block entirely;
    only the instruction and the question are rendered as text.  Both paths are
    built here so the two arms cannot drift apart.
    """
    if payload.dtype == TEXT_DTYPE:
        body = f"MESSAGE FROM THE OTHER AGENT:\n{payload.as_text()}\n\n"
    else:
        body = ""
    return f"{READER_INSTRUCTION}\n\n{body}QUESTION: {payload.question}\nANSWER:"


def reader_suffix(payload: SealedLatentPayload) -> str:
    """The text that follows a continuous prefix in the reader's input."""
    if payload.dtype == TEXT_DTYPE:
        raise PayloadError("text payloads use reader_prompt(), not reader_suffix()")
    return f"\n\nQUESTION: {payload.question}\nANSWER:"


# ---------------------------------------------------------------------------
# Guards


def assert_sealed_handoff_untouched() -> None:
    """Experiment 13 must not widen Experiment 0-12's isolation guarantee.

    Imported and asserted by the selftest.  If someone adds a tensor field to
    ``SealedHandoff`` to make a latent mechanism easier to implement, this
    fails, which is the whole point.
    """
    import dataclasses

    import handoffs as hm

    fields = [f.name for f in dataclasses.fields(hm.SealedHandoff)]
    if fields != ["qid", "question", "handoff_text", "mechanism"]:
        raise PayloadError(f"SealedHandoff has changed shape: {fields}")
    for f in dataclasses.fields(hm.SealedHandoff):
        if f.type not in ("str", str):
            raise PayloadError(f"SealedHandoff.{f.name} is no longer a string")


def assert_no_source_leak(text: str, forbidden: Iterable[str]) -> None:
    """Fail if reader-bound text contains material it must never carry.

    ``forbidden`` is the omitted evidence plus the evaluator's ids for this
    item.  Used on every reader prompt in Panel A, where the whole claim rests
    on the reader not having seen the unselected cards.
    """
    haystack = " ".join(text.casefold().split())
    for needle in forbidden:
        probe = " ".join(str(needle).casefold().split())
        if probe and probe in haystack:
            raise PayloadError(f"reader material leaked forbidden text: {needle!r}")
