"""Experiment 13, Panel A: efficient and reusable non-language handoffs.

Selection is held fixed *before* encoding.  Every channel receives exactly the
same K=4 evidence packet -- the one Experiment 5b already selected and logged --
and the same writing instruction.  No encoder can see the omitted cards.  The
reader is fresh, gets one payload and one question, and never sees the source.

This isolates the **coding** effect.  It does not measure a selection policy,
and a win here is not a win in Panel B, where every sender sees the full source
and chooses for itself.  Do not pool the two.

Channels
    ``text_prose``   concise standalone prose -- the natural-language reference
    ``text_record``  typed key/value records with literal values, no prose
    ``latent_sb``    an aligned continuous prefix from the sender's transcript H

Behavioural controls, on every packet -- these carry interpretation
    ``direct_packet``     the reader gets the selected cards verbatim -- ceiling
    ``closed_book``       no material at all -- leakage floor for this reader
    ``shuffled_prefix``   another dossier's prefix, under an opaque reader id
    ``zero_prefix``       zeros; an ablation, not a neutral prompt
    ``source_free``       H re-encoded with no source access -- separates
                          source-conditioned state from what H already says

Infrastructure controls, on four fixed dossiers -- these are yes/no plumbing
    ``token_replay``      token ids for text T
    ``embed_replay``      input embeddings for the *same* tokens as token_replay
                          -- these two must agree, or the continuous wrapper is
                          broken and every latent result is uninterpretable
    ``transcript_text``   all of H as text
    ``transcript_tail``   the final k tokens of H as text -- distinguishes
                          continuous content from prefix plumbing

Nothing here writes to data/, runs/ or results/ under ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fictional_qa as fqa  # noqa: E402
import latent_backend as lb  # noqa: E402
import latent_handoff as lh  # noqa: E402
import latent_metrics as lm  # noqa: E402
from llm import load_config  # noqa: E402
from score import extract_short_answer, score_against_golds  # noqa: E402


SCHEMA_VERSION = "latent-reusability-v2"

TEXT_CHANNELS = ("text_prose", "text_record")
LATENT_CHANNELS = ("latent_sb", "latent_norm")
CHANNELS = TEXT_CHANNELS + ("latent_sb",)

#: Infrastructure checks: do the wrapper and the plumbing work at all.  These
#: run on a few fixed development dossiers, because they answer a yes/no
#: engineering question and do not need the full set's statistical power.
INFRASTRUCTURE_CONTROLS = ("token_replay", "embed_replay", "transcript_text",
                           "transcript_tail", "fixed_fact_swap", "fixed_fact_sham")

#: Behavioural controls: is the reader using the payload, and where does the
#: information come from.  These carry interpretation, so they run on every
#: packet alongside the channels.
BEHAVIORAL_CONTROLS = ("direct_packet", "closed_book", "shuffled_prefix",
                       "zero_prefix", "source_free")

CONTROLS = BEHAVIORAL_CONTROLS + INFRASTRUCTURE_CONTROLS

WRITER_SYSTEM = (
    "You write a handoff message for another agent who cannot see your source. "
    "Use only the evidence you were given. Never invent, merge or infer facts, "
    "and never write an evidence-card label."
)

PROSE_INSTRUCTION = """Write a standalone message carrying the evidence below to another agent.
They cannot see this evidence and may be asked about any part of it. Write prose.
Do not answer any question yourself. Do not mention card labels."""

RECORD_INSTRUCTION = """Rewrite the evidence below as short typed records for another agent.
They cannot see this evidence and may be asked about any part of it.
Use one `field: value` line per fact, with literal values copied from the evidence.
The records must be understandable on their own from the values you write.
Never write an evidence-card label as a substitute for its content."""

TRANSCRIPT_INSTRUCTION = """Write research notes on the evidence below for another agent.
They cannot see this evidence and may be asked about any part of it.
Do not answer any question yourself. Do not mention card labels."""

#: Fixed public schema shown to a reader receiving typed records.  Charged to
#: `text_record`'s position budget: a legend the reader needs is part of the
#: message cost, not free context.
RECORD_SCHEMA = "The message is a list of `field: value` records copied from source evidence."

#: The preamble a reader sees before a continuous prefix.  Charged to
#: `latent_sb` for the same reason.
LATENT_SCHEMA = "The message arrives as encoded vectors before this question."


# ---------------------------------------------------------------------------
# I/O


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Packets


@dataclass(frozen=True)
class Packet:
    """One fixed selection: the entire source available to every encoder."""

    packet_id: str
    dossier_id: str
    rotation_id: str
    mode: str                       # generic | conditioned
    announced_question: str         # "" for generic
    selected_fact_ids: tuple[str, ...]
    packet_text: str
    packet_hash: str
    omitted_fact_ids: tuple[str, ...]
    omitted_evidence: tuple[str, ...]
    panel: str = "A"
    evaluation_ids: tuple[str, ...] = ()
    current_fact_id: str = ""
    relation_labels: tuple[tuple[str, str], ...] = ()

    def record(self) -> dict:
        return {"packet_id": self.packet_id, "dossier_id": self.dossier_id,
                "rotation_id": self.rotation_id, "mode": self.mode,
                "selected_fact_ids": list(self.selected_fact_ids),
                "omitted_fact_ids": list(self.omitted_fact_ids),
                "packet_hash": self.packet_hash, "panel": self.panel,
                "evaluation_ids": list(self.evaluation_ids),
                "current_fact_id": self.current_fact_id}


def build_packets(dossiers, selections: list[dict], k: int) -> list[Packet]:
    """Project Experiment 5b's logged selections into fixed encoder inputs.

    Provenance is inherited, not recomputed: this panel is not a test of the new
    model's selection policy, so the selector that produced these ids stays
    exactly as it was.
    """
    by_id = {d.dossier_id: d for d in dossiers}
    packets = []
    for row in selections:
        if row.get("slots") != k or not row.get("selection_valid"):
            continue
        dossier = by_id.get(row["dossier_id"])
        if dossier is None:
            continue
        cards = fqa.evidence_cards(dossier)
        selected = tuple(row["selected_fact_ids"])
        text = fqa.render_packet_slots(cards, selected, k)
        omitted = tuple(c.fact_id for c in cards if c.fact_id not in set(selected))
        omitted_text = tuple(c.evidence_text for c in cards
                             if c.fact_id in set(omitted))
        announced = ""
        if row["mode"] == "conditioned":
            fact = dossier.fact(row["announced_fact_id"])
            announced = fact.question
        packets.append(Packet(
            packet_id=lh.opaque_id(row["dossier_id"], row["rotation_id"],
                                   row["mode"], k, row["handoff_hash"]),
            dossier_id=row["dossier_id"], rotation_id=row["rotation_id"],
            mode=row["mode"], announced_question=announced,
            selected_fact_ids=selected, packet_text=text,
            packet_hash=row["handoff_hash"],
            omitted_fact_ids=omitted, omitted_evidence=omitted_text))
    return packets


def writer_prompt(channel: str, packet: Packet, band: lm.PositionBand) -> str:
    """The one allowed writing instruction, identical across channels but for form.

    The announced question is shown only when the packet is a conditioned one,
    and hidden questions never appear in a writer prompt.
    """
    instruction = {"text_prose": PROSE_INSTRUCTION,
                   "text_record": RECORD_INSTRUCTION}.get(channel, TRANSCRIPT_INSTRUCTION)
    ask = (f"\n\nThe next agent's current question is: {packet.announced_question}"
           if packet.announced_question else "")
    length = (f"\n\nYour message must be between {band.floor} and {band.target} "
              f"tokens long. Use most of that room.")
    return f"{instruction}{ask}{length}\n\nEVIDENCE:\n{packet.packet_text}\n"


# ---------------------------------------------------------------------------
# Encoding


@dataclass
class Encoded:
    payload_text: str
    positions: lm.PositionAccount
    cost: lm.CostBreakdown
    vectors: np.ndarray | None = None
    transcript: lb.Transcript | None = None
    attempts: int = 1
    band_ok: bool = True
    alignment: str = "none"
    diagnostics: dict = field(default_factory=dict)
    error: str = ""


def _fit_text(backend: lb.Backend, prompt: str, band: lm.PositionBand,
              cost: lm.CostBreakdown, max_attempts: int = 3) -> tuple[str, int, bool]:
    """Generate into the two-sided band, charging every attempt.

    ``max_new_tokens`` is not a length control and a cap alone is not a band --
    Experiment 10 established that the hard way.  Failures are kept and counted;
    they are not silently regenerated until one fits.
    """
    text, attempts = "", 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        nudge = ""
        if attempt > 1:
            got = backend.count_positions(text)
            nudge = (f"\n\nYour previous message was {got} tokens. "
                     f"Rewrite it to between {band.floor} and {band.target} tokens, "
                     f"keeping every fact.")
        gen = backend.generate(prompt + nudge, max_new_tokens=band.target * 2,
                               system=WRITER_SYSTEM)
        cost.sender_prefill_s += gen.usage.prefill_s
        cost.sender_decode_s += gen.usage.decode_s
        cost.sender_generated_tokens += gen.usage.generated_tokens
        if attempt > 1:
            cost.discarded_tokens += backend.count_positions(text)
            cost.retries += 1
        text = gen.text.strip()
        if band.fits(backend.count_positions(text)):
            return text, attempts, True
    return text, attempts, band.fits(backend.count_positions(text))


def encode(backend: lb.Backend, channel: str, packet: Packet,
           band: lm.PositionBand, k: int, max_attempts: int = 3) -> Encoded:
    """Produce one message for one packet on one channel."""
    cost = lm.CostBreakdown()
    schema = RECORD_SCHEMA if channel == "text_record" else (LATENT_SCHEMA if channel in LATENT_CHANNELS else "")
    overhead = backend.count_positions(schema + "\n") if schema else 0
    capacity = band.target - overhead
    if capacity < 1:
        raise ValueError("position cap leaves no payload capacity after schema")
    payload_floor = max(1, band.floor - overhead)
    payload_band = lm.PositionBand(target=capacity, floor_ratio=payload_floor / capacity)
    prompt = writer_prompt(channel, packet, payload_band if channel in TEXT_CHANNELS else
                           lm.PositionBand(backend.config.transcript_max_tokens, 0.85))
    instruction_positions = backend.count_positions(lh.READER_INSTRUCTION)

    if channel in TEXT_CHANNELS:
        text, attempts, ok = _fit_text(backend, prompt, payload_band, cost, max_attempts)
        positions = lm.PositionAccount(
            payload=backend.count_positions(text),
            schema=overhead,
            instruction=instruction_positions)
        return Encoded(payload_text=text, positions=positions, cost=cost,
                       attempts=attempts, band_ok=ok)

    if channel not in LATENT_CHANNELS:
        raise ValueError(f"unknown channel {channel!r}")

    # k historically meant evidence cards in the CLI. It must never silently
    # become the number of vectors. The delivered vector budget is B-overhead.
    transcript = backend.transcript_states(prompt, k=capacity, system=WRITER_SYSTEM)
    cost.sender_prefill_s += transcript.usage.prefill_s
    cost.sender_decode_s += transcript.usage.decode_s
    cost.sender_generated_tokens += transcript.usage.generated_tokens
    # Every token of H is charged, not only the k rows that travel.
    cost.discarded_tokens += max(0, transcript.realized_length - transcript.delivered_k)

    t0 = time.perf_counter()
    mode = lb.NORM_VERSION if channel == "latent_norm" else lb.ALIGNMENT_VERSION
    vectors = _align(backend, transcript.states, transcript.reference_embeddings, mode)
    cost.align_s += time.perf_counter() - t0

    positions = lm.PositionAccount(
        payload=int(vectors.shape[0]),
        schema=overhead,
        instruction=instruction_positions)
    return Encoded(payload_text=transcript.text, positions=positions, cost=cost,
                   vectors=vectors, transcript=transcript,
                   band_ok=band.fits(positions.added), alignment=mode,
                   diagnostics=dict(getattr(backend, "last_alignment", {})))


def _align(backend: lb.Backend, states: np.ndarray, references=None, mode=None) -> np.ndarray:
    """Use the backend's own calibration when it has one, else identity scale."""
    if hasattr(backend, "align"):
        return backend.align(states, references, mode)
    return np.ascontiguousarray(np.asarray(states, dtype=np.float32))


def seal(channel: str, packet: Packet, question: str, encoded: Encoded,
         dtype: str, tag: str = "") -> lh.SealedLatentPayload:
    """Freeze one message for one question.

    The reader-side id is opaque and includes the question and any control tag,
    so a shuffled or zeroed control cannot collide with the original in a cache
    and quietly return the original's answer.
    """
    qid = lh.opaque_id(packet.packet_id, channel, question, tag)
    # A payload's `positions` is the payload's own receiver cost. The schema
    # block it travels with is charged in PositionAccount.schema and reported as
    # `added_positions`; folding it in here would make the vector case
    # self-contradictory, since one delivered row is exactly one position.
    meta = {"codec": channel, "codec_version": SCHEMA_VERSION,
            "positions": encoded.positions.payload}
    if channel in TEXT_CHANNELS:
        if channel == "text_record":
            meta["alignment"] = "none"
        return lh.seal_text(qid, question, channel, encoded.payload_text,
                            positions=encoded.positions.payload, metadata=meta)
    meta["alignment"] = encoded.alignment
    meta["normalization"] = encoded.alignment
    return lh.seal_vectors(qid, question, channel, encoded.vectors,
                           dtype=dtype, metadata=meta)


# ---------------------------------------------------------------------------
# Reading


def answer(backend: lb.Backend, payload: lh.SealedLatentPayload,
           packet: Packet, cost: lm.CostBreakdown,
           check_leak: bool = True) -> lb.Generation:
    """One fresh reader, one payload, one question.

    ``check_leak`` asserts that no omitted evidence reached the reader through
    a text channel.  Panel A's entire claim rests on that, so it is checked on
    every row rather than sampled.
    """
    if type(payload) is not lh.SealedLatentPayload:
        raise TypeError("reader accepts only a sealed payload")
    schema = RECORD_SCHEMA if payload.channel == "text_record" else (LATENT_SCHEMA if payload.dtype != lh.TEXT_DTYPE and dict(payload.metadata).get("alignment") != "input-embedding-replay" else "")
    material = payload.as_text() if payload.dtype == lh.TEXT_DTYPE else payload.as_array()
    # Audit actual encoder input boundaries separately. A generated text match
    # to omitted evidence is an outcome to record, not a reason to remove a row.
    gen = backend.read_message(material, payload.question, schema)
    cost.reader_prefill_s += gen.usage.prefill_s
    cost.reader_decode_s += gen.usage.decode_s
    return gen


def score_row(raw: str, golds: Sequence[str]) -> dict:
    short = extract_short_answer(raw)
    em, f1 = score_against_golds(short, list(golds))
    return {"raw_answer": raw, "short_answer": short, "em": em, "f1": f1}


# ---------------------------------------------------------------------------
# Panel A


def run_panel_a(backend: lb.Backend, packets: Sequence[Packet], dossiers,
                channels: Sequence[str], band: lm.PositionBand, k: int,
                dtype: str, controls: Sequence[str] = (),
                n_infra_dossiers: int = 4, *, store=None, max_attempts=3) -> tuple[list[dict], list[dict]]:
    """Encode each packet once per channel, then score it on every question."""
    by_id = {d.dossier_id: d for d in dossiers}
    answers: list[dict] = []
    messages: list[dict] = []
    # A donor pool for the shuffled control: a *different* dossier's prefix.
    latent_pool: dict[str, np.ndarray] = {}
    encoded_pool = {}

    for packet in packets:
        dossier = by_id[packet.dossier_id]
        cards = cards_for(dossier)
        if packet.evaluation_ids:
            cards = [c for c in cards if c.fact_id in packet.evaluation_ids]
        questions = [(c.fact_id, c.question, c.golds) for c in cards]

        for channel in channels:
            if store:
                store.check_deadline()
            key = {"text": packet.packet_text,
                   "announced": packet.announced_question, "channel": channel,
                   "band": as_band(band), "max_attempts": max_attempts}
            enc = store.encoding(key, lambda: encode(backend, channel, packet, band, k, max_attempts)) if store else encode(backend, channel, packet, band, k, max_attempts)
            encoded_pool[(packet.packet_id, channel)] = enc
            messages.append({
                "schema_version": SCHEMA_VERSION, "channel": channel,
                **packet.record(), **enc.positions.record(),
                "attempts": enc.attempts, "band_ok": enc.band_ok,
                "alignment": enc.alignment, "alignment_diagnostics": enc.diagnostics,
                "transcript": enc.payload_text, "encoder_cost": enc.cost.record(),
                "panel": packet.panel,
                "state_convention": enc.transcript.state_convention if enc.transcript else None,
                **band.record(enc.positions.added),
                "transcript_realized_length": (
                    enc.transcript.realized_length if enc.transcript else None),
                "transcript_delivered_k": (
                    enc.transcript.delivered_k if enc.transcript else None),
                "transcript_short": (
                    enc.transcript.short if enc.transcript else None),
            })
            if store:
                store.append("messages", messages[-1])
            if enc.vectors is not None:
                latent_pool[packet.packet_id] = enc.vectors

            for fact_id, question, golds in questions:
                cost = lm.CostBreakdown(**vars(enc.cost))
                start = time.perf_counter()
                payload = seal(channel, packet, question, enc, dtype)
                wire = lh.to_wire(payload)
                payload = lh.from_wire(wire)
                cost.serialize_s += time.perf_counter() - start
                invalid = not enc.band_ok
                if invalid:
                    gen = lb.Generation("", lb.Usage(calls=0))
                elif store:
                    gen = store.read(backend, payload, lambda: answer(backend, payload, packet, lm.CostBreakdown()))
                    cost.reader_prefill_s += gen.usage.prefill_s
                    cost.reader_decode_s += gen.usage.decode_s
                else:
                    gen = answer(backend, payload, packet, cost)
                answers.append(_answer_record(
                    packet, channel, fact_id, question, golds, gen, payload,
                    enc, cost, band, arm=channel))
                if store:
                    store.append("answers", answers[-1])

    if controls:
        from latent_controls import run_controls_v2
        answers.extend(run_controls_v2(
            backend, packets, by_id, controls, band, dtype, encoded_pool,
            infrastructure_dossiers(packets, n_infra_dossiers), store))
    return messages, answers


def as_band(band):
    return {"target": band.target, "floor_ratio": band.floor_ratio}


def cards_for(dossier):
    return dossier.cards if hasattr(dossier, "cards") else fqa.evidence_cards(dossier)


def _answer_record(packet: Packet, channel: str, fact_id: str, question: str,
                   golds, gen: lb.Generation, payload: lh.SealedLatentPayload,
                   enc: Encoded, cost: lm.CostBreakdown, band: lm.PositionBand,
                   arm: str) -> dict:
    announced = (fact_id == packet.current_fact_id if packet.current_fact_id else
                 bool(packet.announced_question and packet.announced_question.strip() == question.strip()))
    delivered = fact_id in set(packet.selected_fact_ids)
    return {
        "schema_version": SCHEMA_VERSION, "panel": packet.panel, "arm": arm,
        "channel": channel, "packet_id": packet.packet_id,
        "dossier_id": packet.dossier_id, "rotation_id": packet.rotation_id,
        "mode": packet.mode, "fact_id": fact_id, "question": question,
        "relation": dict(packet.relation_labels).get(fact_id, "current" if announced else "unlabelled"),
        "is_announced": announced, "evidence_delivered": delivered,
        "golds": list(golds),
        **score_row(gen.text, golds),
        **payload.record(),
        **enc.positions.record(),
        **band.record(enc.positions.added),
        **cost.record(),
        "reader_generated_tokens": gen.usage.generated_tokens,
        "reader_usage": gen.usage.record(), "error": enc.error or gen.error,
        "reader_cache_hit": gen.cache_hit,
        "valid_message": enc.band_ok and not enc.error,
        "wire_bytes": len(lh.to_wire(payload)),
        "system_em": score_row(gen.text, golds)["em"] if (arm not in TEXT_CHANNELS+LATENT_CHANNELS or enc.band_ok) and not (enc.error or gen.error) else 0.0,
    }


# ---------------------------------------------------------------------------
# Controls


def infrastructure_dossiers(packets: Sequence[Packet], n: int) -> set[str]:
    """The fixed development dossiers the wrapper checks run on.

    Fixed by sorted id rather than sampled, so the subset does not move between
    runs and cannot be chosen after seeing which dossiers behave well.
    """
    return set(sorted({p.dossier_id for p in packets})[:max(0, n)])


# ---------------------------------------------------------------------------
# Gates


def development_gates(answers: Sequence[dict], cfg: dict) -> dict:
    """The engineering thresholds fixed before the grid runs.

    These are not significance claims.  Failing one triggers a bounded
    implementation investigation -- not a change of dataset, and not a wider
    grid.
    """
    gates = (cfg.get("gates") or {})
    by_arm = defaultdict(list)
    for row in answers:
        by_arm[row["arm"]].append(row)

    def mean_em(arm: str, only_delivered: bool = False) -> float:
        rows = by_arm.get(arm, [])
        if only_delivered:
            rows = [r for r in rows if r["evidence_delivered"]]
        return float(np.mean([r.get("system_em", r["em"]) for r in rows])) if rows else float("nan")

    direct = mean_em("direct_packet", only_delivered=True)
    closed = mean_em("closed_book")
    latent = mean_em("latent_sb", only_delivered=True)
    shuffled = mean_em("shuffled_prefix", only_delivered=True)
    token = mean_em("token_replay", only_delivered=True)
    embed = mean_em("embed_replay", only_delivered=True)

    min_direct = float(gates.get("direct_source_em_min", 0.90))
    max_closed = float(gates.get("closed_book_em_max", 0.05))
    max_drop = float(gates.get("latent_gap_max", 0.10))
    min_sensitivity = float(gates.get("shuffle_sensitivity_min", 0.10))
    max_replay_gap = float(gates.get("replay_gap_max", 0.05))

    results = {
        "direct_packet_em": direct,
        "closed_book_em": closed,
        "latent_em": latent,
        "shuffled_prefix_em": shuffled,
        "token_replay_em": token,
        "embed_replay_em": embed,
        "gate_direct_source": _ge(direct, min_direct),
        "gate_closed_book": _le(closed, max_closed),
        "gate_latent_gap": _le(direct - latent, max_drop),
        "gate_payload_sensitivity": _ge(latent - shuffled, min_sensitivity),
        "gate_replay_answer_agreement": _le(abs(token - embed), max_replay_gap),
        "gate_wrapper_parity": (cfg.get("wrapper_parity") or {}).get("passed"),
        "gate_control_availability": not any(r.get("error") for r in answers if r["arm"] in CONTROLS and not r["arm"].startswith("fixed_fact_")),
        "thresholds": {"direct_source_em_min": min_direct,
                       "closed_book_em_max": max_closed,
                       "latent_gap_max": max_drop,
                       "shuffle_sensitivity_min": min_sensitivity,
                       "replay_gap_max": max_replay_gap},
    }
    results["all_gates_pass"] = all(
        results[key] is True for key in results if key.startswith("gate_"))
    return results


def _ge(value: float, threshold: float):
    return None if not np.isfinite(value) else bool(value >= threshold)


def _le(value: float, threshold: float):
    return None if not np.isfinite(value) else bool(value <= threshold)


# ---------------------------------------------------------------------------
# Dry run


def dry_run_report(packets: Sequence[Packet], channels: Sequence[str],
                   controls: Sequence[str], n_questions: int, cfg: dict,
                   n_infra_dossiers: int = 4) -> dict:
    """Counts only. Writes nothing to data/, runs/ or results/."""
    core = len(packets) * len(channels) * n_questions
    infra = [c for c in controls if c in INFRASTRUCTURE_CONTROLS]
    behavioral = [c for c in controls if c not in INFRASTRUCTURE_CONTROLS]
    infra_ids = infrastructure_dossiers(packets, n_infra_dossiers)
    infra_packets = sum(1 for p in packets if p.dossier_id in infra_ids)
    behavioral_rows = len(packets) * len(behavioral) * n_questions
    infra_rows = infra_packets * len(infra) * n_questions
    return {
        "dry_run": True, "writes": 0,
        "packets": len(packets),
        "generic_packets": sum(1 for p in packets if p.mode == "generic"),
        "conditioned_packets": sum(1 for p in packets if p.mode == "conditioned"),
        "channels": list(channels),
        "behavioral_controls": behavioral,
        "infrastructure_controls": infra,
        "infrastructure_dossiers": sorted(infra_ids),
        "infrastructure_packets": infra_packets,
        "questions_per_packet": n_questions,
        "core_answer_evaluations": core,
        "behavioral_control_evaluations": behavioral_rows,
        "infrastructure_control_evaluations": infra_rows,
        "total_answer_evaluations": core + behavioral_rows + infra_rows,
        "core_messages_to_encode": len(packets) * len(channels),
        "transcript_generation_upper_bound": len(packets) if any(c not in ("direct_packet", "closed_book") for c in controls) or "latent_sb" in channels else 0,
        "backend": (cfg.get("backend") or {}).get("model_id"),
        "note": ("Cache hits reduce physical inference requests, not evaluation "
                 "rows. Requested and actual counts are logged separately."),
    }


# ---------------------------------------------------------------------------
# Entry point


def build_backend(cfg: dict, fake: bool) -> lb.Backend:
    raw = dict(cfg.get("backend") or {})
    raw.pop("fake", None)
    codec = cfg.get("codec") or {}
    raw["alignment"] = codec.get("alignment", lb.ALIGNMENT_VERSION)
    for name in ("regularization", "eigen_floor", "snap_ratio", "vocab_chunk"):
        if name in codec:
            raw["alignment_" + name] = codec[name]
    config = lb.BackendConfig(**raw)
    return lb.FakeBackend(config) if fake else lb.TransformersBackend(config)


def prime_fake(backend: lb.FakeBackend, dossiers) -> None:
    """Give the simulator the shared frozen vocabulary and its reading ability."""
    texts, facts = [], []
    for dossier in dossiers:
        for card in cards_for(dossier):
            texts.extend([card.evidence_text, card.question])
            facts.append(lb.FakeFact(evidence=card.evidence_text,
                                     question=card.question,
                                     answer=card.golds[0] if card.golds else ""))
    backend.learn_vocabulary(texts)
    backend.register_facts(facts)


def main(argv=None):
    from latent_cli import main as execute
    return execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
