"""The harness action space and policy arms for Experiment 12.

Experiments 10 and 11 gave the writer one lever: how to spend a word budget on
a single irreversible message. A harness has more than that. It can hold a fact
in context, compress it, push it to storage, leave a pointer to it, pull it back
later, or drop it entirely. This module implements those as explicit per-unit
actions and builds the policy arms that combine them.

Six actions. Five are decided at write time and one at read time:

    KEEP        verbatim in the message           costs its own words
    COMPRESS    shortened form in the message     costs fewer words, lossy
    EXTERNALIZE in the store, not the message     costs no context, needs recall
    POINTER     short index line in the message,  costs a few words and makes
                unit in the store                 later recall targeted
    DISCARD     in neither                        free, unrecoverable
    RETRIEVE    (read time) BM25 over the store   costs retrieved words

The distinction that matters is POINTER versus EXTERNALIZE. Both defer the
evidence, but a pointer spends a little context now to say *what was deferred*.
If retrieval turns out to beat proactive preservation, the pointer arm is what
separates "recall is cheap" from "recall is cheap because the writer left a
map".

Selection is deterministic given the units and the prediction. Compression is
the only action needing a model, and a compressed unit depends on nothing but
the unit itself, so it is precomputed once per unit and reused across every
policy, rotation and budget rather than regenerated per arm.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anticipatory_data as ad  # noqa: E402

KEEP = "keep"
COMPRESS = "compress"
EXTERNALIZE = "externalize"
POINTER = "pointer"
DISCARD = "discard"
ACTIONS = (KEEP, COMPRESS, EXTERNALIZE, POINTER, DISCARD)

# A pointer line is deliberately cheap and uninformative about the answer: it
# names the aspect and the first few words of the stored unit, which is what an
# index entry in a real harness looks like. It must never contain the gold.
POINTER_PREVIEW_WORDS = 4


@dataclass
class Allocation:
    """What a policy decided to do with every unit, plus its read-time budget."""

    policy: str
    actions: dict[str, str]
    retrieval_k: int = 0
    p_hat: dict[str, float] | None = None
    notes: dict = field(default_factory=dict)

    def of(self, action: str) -> list[str]:
        return [uid for uid, act in self.actions.items() if act == action]


def normalised_entropy(dist: dict[str, float]) -> float:
    """Shannon entropy over aspects, scaled to [0, 1]."""
    total = 0.0
    for aspect in ad.ASPECTS:
        p = float(dist.get(aspect, 0.0))
        if p > 0.0:
            total -= p * math.log2(p)
    return total / math.log2(len(ad.ASPECTS))


def pointer_line(unit) -> str:
    preview = " ".join(unit.text.split()[:POINTER_PREVIEW_WORDS])
    return f"{unit.aspect}: {preview}..."


# ---------------------------------------------------------------------------
# Greedy budgeted selection


def _fit(units: list, budget_words: int, compressed: dict[str, str],
         actions: dict[str, str]) -> int:
    """Greedily place units, compressing one that does not fit whole.

    Returns the words consumed. A unit that fits verbatim is KEEP; if it does
    not fit but its compressed form does, it is COMPRESS; otherwise it is left
    for the caller to externalize or discard. This is where the COMPRESS action
    earns its place - it is used exactly when a harness would use it, at the
    budget boundary, rather than as a separate arm.
    """
    used = 0
    for unit in units:
        if unit.unit_id in actions:
            continue
        whole = unit.words
        if used + whole <= budget_words:
            actions[unit.unit_id] = KEEP
            used += whole
            continue
        short = compressed.get(unit.unit_id, "")
        short_words = len(short.split())
        if short_words and used + short_words <= budget_words:
            actions[unit.unit_id] = COMPRESS
            used += short_words
    return used


def _current_units(units: list, current_qid: str) -> list:
    return [u for u in units if current_qid in u.answers_qids]


def _aspect_pool(units: list, aspect: str, taken: dict[str, str]) -> list:
    return [u for u in units if u.aspect == aspect and u.unit_id not in taken]


def _fill_by_distribution(units: list, dist: dict[str, float], words: int,
                          compressed: dict[str, str], actions: dict[str, str]) -> int:
    """Spend `words` across aspects in proportion to `dist`.

    Aspects are visited in descending weight so that, when rounding leaves a
    remainder, it goes to the aspect the policy believes in most rather than to
    whichever happens to be first alphabetically.
    """
    used = 0
    order = sorted(ad.ASPECTS, key=lambda a: (-dist.get(a, 0.0), ad.ASPECTS.index(a)))
    for aspect in order:
        share = float(dist.get(aspect, 0.0))
        if share <= 0.0:
            continue
        target = int(round(words * share))
        if target <= 0:
            continue
        used += _fit(_aspect_pool(units, aspect, actions), target, compressed, actions)
    return used


# ---------------------------------------------------------------------------
# Policy arms


def build_allocation(policy: str, units: list, current_qid: str,
                     budget_words: int, compressed: dict[str, str],
                     *, p_hat: dict[str, float] | None = None,
                     future_qid: str | None = None,
                     cfg: dict | None = None) -> Allocation:
    """Construct one policy's action assignment over the units of one dossier."""
    cfg = cfg or {}
    now_fraction = float(cfg.get("now_reserve_fraction", 0.5))
    retrieval_k = int(cfg.get("retrieval_k", 2))
    entropy_gate = float(cfg.get("entropy_gate", 0.75))

    actions: dict[str, str] = {}
    now_words = int(round(budget_words * now_fraction))
    used = _fit(_current_units(units, current_qid), now_words, compressed, actions)
    remaining = max(0, budget_words - used)

    family = policy.split("__")[0]

    if family == "static_conditioned":
        # Experiment 10's `conditioned` analogue: spend everything on the
        # current query and drop the rest. No future provision of any kind.
        used += _fit(_current_units(units, current_qid), remaining, compressed, actions)
        _default(actions, units, DISCARD)
        return Allocation(policy, actions, 0, p_hat)

    if family == "blind":
        # The no-prediction reference for V_anticipation. It provides for the
        # future, but spreads uniformly because it has no belief about where
        # demand will land. This is "preserve broadly".
        _fill_by_distribution(units, ad.uniform(), remaining, compressed, actions)
        _default(actions, units, DISCARD)
        return Allocation(policy, actions, 0, p_hat)

    if family == "oracle_exact":
        # Upper bound: the actual future question is known, not merely its aspect.
        if future_qid:
            target = [u for u in units if future_qid in u.answers_qids]
            _fit(target, remaining, compressed, actions)
        _default(actions, units, DISCARD)
        return Allocation(policy, actions, 0, p_hat)

    if family == "preserve":
        # Strategy 1: predict, then spend the future budget where P-hat says.
        _fill_by_distribution(units, _require(p_hat), remaining, compressed, actions)
        _default(actions, units, DISCARD)
        return Allocation(policy, actions, 0, p_hat)

    if family == "retrieval_only":
        # Strategy 3 in its purest form: keep a small task-focused context and
        # make everything else recallable, with no attempt to anticipate.
        _default(actions, units, EXTERNALIZE)
        return Allocation(policy, actions, retrieval_k, p_hat)

    if family == "pointer":
        # Strategies 1 and 3 combined: pointers for what P-hat expects, plain
        # storage for the rest.
        dist = _require(p_hat)
        ranked = sorted(ad.ASPECTS, key=lambda a: (-dist.get(a, 0.0), ad.ASPECTS.index(a)))
        pointer_budget = remaining
        for aspect in ranked:
            if dist.get(aspect, 0.0) <= 0.0:
                continue
            for unit in _aspect_pool(units, aspect, actions):
                cost = len(pointer_line(unit).split())
                if cost > pointer_budget:
                    break
                actions[unit.unit_id] = POINTER
                pointer_budget -= cost
        _default(actions, units, EXTERNALIZE)
        return Allocation(policy, actions, retrieval_k, p_hat)

    if family == "uncertainty_aware":
        # Strategy 4: let the confidence of the estimate choose the strategy.
        # Confident -> spend context on the predicted aspect. Uncertain ->
        # stop guessing and make everything recallable instead.
        dist = _require(p_hat)
        entropy = normalised_entropy(dist)
        if entropy < entropy_gate:
            _fill_by_distribution(units, dist, remaining, compressed, actions)
            _default(actions, units, DISCARD)
            return Allocation(policy, actions, 0, dist, {"entropy": entropy, "mode": "preserve"})
        _default(actions, units, EXTERNALIZE)
        return Allocation(policy, actions, retrieval_k, dist,
                          {"entropy": entropy, "mode": "retrieve"})

    raise ValueError(f"unknown policy family {family!r}")


def _require(p_hat: dict[str, float] | None) -> dict[str, float]:
    if p_hat is None:
        raise ValueError("this policy family needs a predicted distribution")
    return p_hat


def _default(actions: dict[str, str], units: list, action: str) -> None:
    for unit in units:
        actions.setdefault(unit.unit_id, action)


# ---------------------------------------------------------------------------
# Assembly and cost


def assemble(allocation: Allocation, units: list, compressed: dict[str, str],
             budget_words: int) -> dict:
    """Turn an allocation into the delivered message and the store contents."""
    import budget as bd

    by_id = {u.unit_id: u for u in units}
    pieces: list[str] = []
    for unit in units:                      # source order keeps the prose readable
        action = allocation.actions.get(unit.unit_id, DISCARD)
        if action == KEEP:
            pieces.append(unit.text)
        elif action == COMPRESS:
            pieces.append(compressed.get(unit.unit_id, unit.text))
        elif action == POINTER:
            pieces.append(pointer_line(unit))
    message = " ".join(p.strip() for p in pieces if p.strip())
    delivered, truncated, dropped = bd.truncate_to_words(message, budget_words)

    stored_ids = allocation.of(EXTERNALIZE) + allocation.of(POINTER)
    stored = [by_id[uid] for uid in stored_ids if uid in by_id]
    return {
        "message_text": delivered,
        "stored_units": stored,
        "delivered_words": len(delivered.split()),
        "truncated": truncated,
        "truncated_words_dropped": dropped,
        "n_keep": len(allocation.of(KEEP)),
        "n_compress": len(allocation.of(COMPRESS)),
        "n_pointer": len(allocation.of(POINTER)),
        "n_externalize": len(allocation.of(EXTERNALIZE)),
        "n_discard": len(allocation.of(DISCARD)),
    }
