"""Experiment 13: resource accounting and utility contrasts.

Three resource axes, kept separate everywhere because a "token saving" claim is
meaningless until it says which one moved:

1. **Receiver positions** -- what the reader's context actually costs, counting
   the payload *plus* every header, schema block and instruction that had to
   travel with it.
2. **Transferred bytes** -- the serialized payload.  Eight latent positions can
   be eighty times the bytes of 128 token ids; a position win is often a byte
   loss, and the report has to say so.
3. **End-to-end cost** -- sender prefill, every decoded token including the ones
   thrown away, alignment, serialization, reader prefill and reader decoding.

The amortization model is ``C(n) = C_encode + n * (C_transfer + C_read)``.  A
message written once and read sixteen times has a different economics from one
read once, and the design asks for n = 1, 4, 16.

Statistics: the sampling unit is the **source**, not the question row.  Four
questions about one dossier are not four independent observations, so every
interval here resamples whole sources with all their questions and arms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Position accounting


@dataclass(frozen=True)
class PositionAccount:
    """Every receiver position one answer costs, itemized.

    ``payload`` is the message.  ``schema`` is the fixed public block a
    non-prose channel needs in order to be interpretable -- a records legend, a
    latent decoding preamble.  Charging it is the difference between an honest
    comparison and one that hides a channel's overhead in the prompt.
    """

    payload: int
    schema: int = 0
    instruction: int = 0
    question: int = 0

    @property
    def added(self) -> int:
        """Positions attributable to the channel: payload plus its schema.

        The instruction and the question are identical across arms, so they are
        logged but excluded from the *added* count that the primary claim uses.
        """
        return self.payload + self.schema

    @property
    def total(self) -> int:
        return self.payload + self.schema + self.instruction + self.question

    def record(self) -> dict:
        return {"payload_positions": self.payload, "schema_positions": self.schema,
                "instruction_positions": self.instruction,
                "question_positions": self.question,
                "added_positions": self.added, "total_positions": self.total}


@dataclass(frozen=True)
class ByteAccount:
    payload: int
    metadata: int = 0

    @property
    def total(self) -> int:
        return self.payload + self.metadata

    def record(self) -> dict:
        return {"payload_bytes": self.payload, "metadata_bytes": self.metadata,
                "total_bytes": self.total}


def vector_payload_bytes(positions: int, width: int, bits: int = 16,
                         metadata_bytes: int = 0) -> ByteAccount:
    """Raw arithmetic for a dense payload. Not a measurement -- label it as such."""
    return ByteAccount(payload=positions * width * bits // 8, metadata=metadata_bytes)


def token_id_payload_bytes(positions: int, bytes_per_id: int = 4,
                           metadata_bytes: int = 0) -> ByteAccount:
    return ByteAccount(payload=positions * bytes_per_id, metadata=metadata_bytes)


def kv_payload_bytes(positions: int, layers: int, kv_heads: int, head_dim: int,
                     bits: int = 16, metadata_bytes: int = 0) -> ByteAccount:
    """A KV comparator must count every transferred layer and source position."""
    per_position = 2 * layers * kv_heads * head_dim * bits // 8
    return ByteAccount(payload=positions * per_position, metadata=metadata_bytes)


# ---------------------------------------------------------------------------
# Runtime and amortization


@dataclass
class CostBreakdown:
    """Seconds, split so that a position win and a compute win stay separable."""

    sender_prefill_s: float = 0.0
    sender_decode_s: float = 0.0
    align_s: float = 0.0
    serialize_s: float = 0.0
    reader_prefill_s: float = 0.0
    reader_decode_s: float = 0.0
    sender_generated_tokens: int = 0
    discarded_tokens: int = 0
    retries: int = 0

    @property
    def encode_s(self) -> float:
        """One-off, per message."""
        return (self.sender_prefill_s + self.sender_decode_s
                + self.align_s + self.serialize_s)

    @property
    def read_s(self) -> float:
        """Repeated, per answered query."""
        return self.reader_prefill_s + self.reader_decode_s

    def amortized(self, n: int, transfer_s: float = 0.0, transfers: int | None = None) -> float:
        """n total reads; a persisted payload transfers once by default."""
        if n < 0:
            raise ValueError("n must be non-negative")
        count = (1 if n else 0) if transfers is None else transfers
        if count < 0:
            raise ValueError("transfers must be nonnegative")
        return self.encode_s + count * transfer_s + n * self.read_s

    def record(self, transfer_s: float = 0.0,
               horizons: Sequence[int] = (1, 4, 16)) -> dict:
        out = {
            "sender_prefill_s": round(self.sender_prefill_s, 6),
            "sender_decode_s": round(self.sender_decode_s, 6),
            "align_s": round(self.align_s, 6),
            "serialize_s": round(self.serialize_s, 6),
            "reader_prefill_s": round(self.reader_prefill_s, 6),
            "reader_decode_s": round(self.reader_decode_s, 6),
            "encode_s": round(self.encode_s, 6),
            "read_s": round(self.read_s, 6),
            "sender_generated_tokens": self.sender_generated_tokens,
            "discarded_tokens": self.discarded_tokens,
            "retries": self.retries,
        }
        for n in horizons:
            out[f"amortized_s_n{n}"] = round(self.amortized(n, transfer_s), 6)
        return out


# ---------------------------------------------------------------------------
# The two-sided position band
#
# `src/budget.py` equalizes *delivered words* for the text-only experiments.
# That metric cannot compare a dense vector with a sentence, so the equivalent
# here is a band on receiver positions.  The lesson from Experiment 10 carries
# over unchanged: a one-sided cap does not equalize a channel.  An arm that
# spends two-thirds of its budget and then loses on future questions has not
# been shown to specialize -- it has been shown to write less.


@dataclass(frozen=True)
class PositionBand:
    target: int
    floor_ratio: float = 0.85

    def __post_init__(self):
        if self.target < 1 or not 0 < self.floor_ratio <= 1:
            raise ValueError("positive position target and floor ratio in (0,1] required")

    @property
    def floor(self) -> int:
        return int(math.ceil(self.target * self.floor_ratio))

    def fits(self, positions: int) -> bool:
        return self.floor <= positions <= self.target

    def verdict(self, positions: int) -> str:
        if positions > self.target:
            return "over"
        if positions < self.floor:
            return "under"
        return "ok"

    def fill_ratio(self, positions: int) -> float:
        return positions / self.target if self.target else float("nan")

    def record(self, positions: int) -> dict:
        return {"band_target": self.target, "band_floor": self.floor,
                "band_floor_ratio": self.floor_ratio,
                "band_verdict": self.verdict(positions),
                "band_fill_ratio": round(self.fill_ratio(positions), 4),
                "band_ok": self.fits(positions)}


def band_summary(positions: Iterable[int], band: PositionBand) -> dict:
    """Realized fill across an arm.  Publish this before any utility comparison.

    If two arms differ in fill, a difference in their utility is confounded with
    how much channel they actually used.
    """
    values = np.asarray(list(positions), dtype=float)
    if values.size == 0:
        return {"n": 0}
    ratios = values / band.target if band.target else values * float("nan")
    return {
        "n": int(values.size),
        "mean_positions": float(values.mean()),
        "sd_positions": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "mean_fill_ratio": float(ratios.mean()),
        "min_fill_ratio": float(ratios.min()),
        "over_budget": int((values > band.target).sum()),
        "under_floor": int((values < band.floor).sum()),
    }


# ---------------------------------------------------------------------------
# Source-clustered inference


def cluster_bootstrap_delta(sources: Sequence[str], treatment: Sequence[float],
                            control: Sequence[float], n_resamples: int = 10_000,
                            ci: float = 0.95, seed: int = 0) -> dict:
    """Paired delta, resampling whole sources with all their rows.

    ``score.paired_bootstrap_delta`` resamples rows, which is right when the row
    *is* the sampling unit.  Here it is not: six questions about one dossier
    share a source, so row resampling would understate the interval.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.shape != c.shape:
        raise ValueError("paired arrays must align")
    keys = np.asarray(list(sources))
    if keys.shape[0] != t.shape[0]:
        raise ValueError("one source label per row")
    if t.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_sources": 0}

    diffs = t - c
    unique = np.unique(keys)
    groups = [np.flatnonzero(keys == s) for s in unique]
    per_source = np.array([diffs[g].mean() for g in groups])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_source.size, size=(n_resamples, per_source.size))
    means = per_source[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    observed = float(per_source.mean())
    centred = means - observed
    return {
        "delta": observed,
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1 - alpha)),
        "p_value": float((np.abs(centred) >= abs(observed)).mean()),
        "n": int(t.size),
        "n_sources": int(per_source.size),
    }


def noninferiority(delta: dict, margin: float = 0.03,
                   one_sided_ci: float = 0.975) -> dict:
    """Is the treatment noninferior to the control at ``-margin``?

    The rule the design fixes in advance: a conservative simultaneous lower
    bound above ``-margin`` for *both* current and future utility.  Using
    one-sided 97.5% bounds for each gives that simultaneity simply.
    """
    lo = delta.get("lo", float("nan"))
    return {
        "margin": -abs(margin),
        "lower_bound": lo,
        "noninferior": bool(lo > -abs(margin)) if np.isfinite(lo) else False,
        "one_sided_ci": one_sided_ci,
    }


def position_saving(treatment_positions: Sequence[float],
                    control_positions: Sequence[float]) -> dict:
    """Realized fractional reduction in added receiver positions.

    Nominal budgets do not decide this; the actual counts do.  A 32-vs-64
    nominal contrast that realizes 34 against 61 is a 44% saving, not 50%.
    """
    t = float(np.mean(treatment_positions)) if len(treatment_positions) else float("nan")
    c = float(np.mean(control_positions)) if len(control_positions) else float("nan")
    reduction = (c - t) / c if c else float("nan")
    return {"treatment_positions": t, "control_positions": c,
            "reduction": reduction}


def efficiency_claim(saving: dict, now: dict, future: dict,
                     min_reduction: float = 0.25, margin: float = 0.03,
                     treatment_floor: dict | None = None,
                     control_floor: dict | None = None,
                     hard_caps_ok: bool = False) -> dict:
    """The full prespecified gate for a context-efficiency claim.

    Both conditions must hold: a real position reduction, *and* noninferior
    utility on the announced task and the orthogonal future task. Narrowing the
    specialization gap by losing current-task accuracy is not a win, so both
    utility changes are always returned alongside the verdict.
    """
    now_ni = noninferiority(now, margin, one_sided_ci=0.9875)
    future_ni = noninferiority(future, margin, one_sided_ci=0.9875)
    reduction_ok = bool(saving.get("reduction", float("nan")) >= min_reduction)
    floors_ok = all(x is not None and x.get("lo", float("nan")) > 0
                    for x in (treatment_floor, control_floor))
    return {
        "min_reduction": min_reduction,
        "realized_reduction": saving.get("reduction"),
        "reduction_ok": reduction_ok,
        "u_now": now, "u_now_noninferior": now_ni,
        "u_future": future, "u_future_noninferior": future_ni,
        "treatment_floor": treatment_floor, "control_floor": control_floor,
        "future_floors_ok": floors_ok, "hard_caps_ok": hard_caps_ok,
        "claim_supported": bool(reduction_ok and now_ni["noninferior"]
                                and future_ni["noninferior"] and floors_ok and hard_caps_ok),
    }


def specialization_gap(u_now: Sequence[float], u_future: Sequence[float]) -> float:
    """U_now - U_future_orthogonal.  Reported only with both components."""
    if not len(u_now) or not len(u_future):
        return float("nan")
    return float(np.mean(u_now) - np.mean(u_future))
