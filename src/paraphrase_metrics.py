"""Deterministic transition-level measurement for Experiment 5 handoffs.

Measures what happens across one consecutive handoff edge ``M_i -> M_{i+1}``:
how much the wording changed, how much of the wording survived, and whether the
edge is a rewrite or a near-verbatim copy. Everything in this module is pure
string arithmetic -- no API calls, no LLM judge -- so it can run over already
stored ``handoffs.jsonl`` rows for free and is fully offline-testable.

The semantic half of the same question (does the rewritten message still *mean*
the same thing) deliberately lives in ``judge.py`` instead, next to the existing
answer judge, because it is the only part that needs a second model.

Tokenisation reuses ``score.normalize_answer`` so "lexical similarity" here is
measured on exactly the same normalisation the answer metrics use, rather than
introducing a second, silently different notion of a token.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import normalize_answer, token_f1  # noqa: E402


# Thresholds for ``classify_transition``. Defaults are deliberately coarse:
# they exist to sort edges into readable buckets for the plot, never to define
# a metric. Every underlying continuous quantity is stored on the row too, so
# any reader can re-cut the same data with different cut points.
DEFAULT_THRESHOLDS = {
    # At or above this lexical similarity the message was barely reworded at
    # all, so it cannot evidence anything about *rewriting* either way.
    "verbatim_similarity": 0.95,
    # Below this, the message was substantially reworded.
    "rewritten_similarity": 0.70,
    # At or above this the second model called the pair meaning-preserving.
    "semantic_preserved": 0.75,
}


def tokens(text: str) -> list[str]:
    return normalize_answer(text or "").split()


def lexical_similarity(previous: str, current: str) -> float:
    """Bag-of-words token F1 between two messages.

    Reuses the answer-scoring F1 rather than a new formula: it is symmetric
    enough for a similarity, already normalises casing/punctuation/articles,
    and keeps one definition of token overlap across the whole project.
    """
    return token_f1(previous or "", current or "")


def jaccard_similarity(previous: str, current: str) -> float:
    a, b = set(tokens(previous)), set(tokens(current))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novel_token_rate(previous: str, current: str) -> float:
    """Share of the new message's tokens that were absent from the old one.

    High novelty with preserved meaning is the signature of paraphrasing;
    high novelty with degraded meaning is the signature of drift or invention.
    """
    prev, curr = Counter(tokens(previous)), tokens(current)
    if not curr:
        return 0.0
    remaining = Counter(prev)
    carried = 0
    for token in curr:
        if remaining.get(token, 0) > 0:
            remaining[token] -= 1
            carried += 1
    return 1.0 - (carried / len(curr))


def retained_token_rate(previous: str, current: str) -> float:
    """Share of the old message's tokens still present in the new one."""
    prev, curr = tokens(previous), Counter(tokens(current))
    if not prev:
        return 1.0
    remaining = Counter(curr)
    kept = 0
    for token in prev:
        if remaining.get(token, 0) > 0:
            remaining[token] -= 1
            kept += 1
    return kept / len(prev)


def ngram_copy_rate(previous: str, current: str, n: int = 5) -> float:
    """Share of the new message's n-grams copied verbatim from the old one.

    Separates "reworded the same content" from "reproduced the same sentences".
    Token F1 alone cannot: an unordered bag of words scores identically for a
    genuine paraphrase and for a reshuffled copy.
    """
    prev, curr = tokens(previous), tokens(current)
    if len(curr) < n:
        return 1.0 if curr and curr == prev[:len(curr)] else 0.0
    prev_grams = {tuple(prev[i:i + n]) for i in range(max(0, len(prev) - n + 1))}
    curr_grams = [tuple(curr[i:i + n]) for i in range(len(curr) - n + 1)]
    if not curr_grams:
        return 0.0
    return sum(1 for g in curr_grams if g in prev_grams) / len(curr_grams)


def length_ratio(previous: str, current: str) -> float:
    """Characters in the new message per character of the old one."""
    prev_len = len(previous or "")
    if prev_len == 0:
        return float("nan")
    return len(current or "") / prev_len


def transition_measures(previous: str, current: str) -> dict:
    """Every deterministic measure for one ``M_i -> M_{i+1}`` edge."""
    similarity = lexical_similarity(previous, current)
    return {
        "lexical_similarity": round(similarity, 4),
        "lexical_change": round(1.0 - similarity, 4),
        "jaccard_similarity": round(jaccard_similarity(previous, current), 4),
        "novel_token_rate": round(novel_token_rate(previous, current), 4),
        "retained_token_rate": round(retained_token_rate(previous, current), 4),
        "ngram_copy_rate": round(ngram_copy_rate(previous, current), 4),
        "length_ratio": round(length_ratio(previous, current), 4),
        "previous_characters": len(previous or ""),
        "current_characters": len(current or ""),
        "previous_tokens": len(tokens(previous)),
        "current_tokens": len(tokens(current)),
    }


def classify_transition(
    lexical_sim: float,
    semantic_preserved: float | None,
    facts_survived: bool | None,
    answer_correct: float | None,
    thresholds: dict | None = None,
) -> str:
    """Bucket one edge into a readable failure/success mode.

    The four buckets the experiment cares about, in the report's own terms:

    ``benign_paraphrase``
        Wording changed, meaning held, the fact survived, the answer is right.
        Rewriting happened and cost nothing.
    ``information_loss``
        Meaning degraded and the answer is wrong. Rewriting destroyed content.
    ``critical_detail_loss``
        Meaning judged preserved overall, yet the fact or the answer is gone --
        a small, task-critical detail dropped out under an otherwise faithful
        rewrite. This is the case an overall similarity score cannot see.
    ``semantic_drift``
        Meaning degraded but the answer survives anyway, e.g. because the
        answerer reconstructed it. Recorded separately so it is never counted
        as evidence that nothing was lost.

    ``verbatim_copy`` is returned when the edge barely reworded anything, and
    ``unresolved`` when the semantic verdict is unavailable (judge disabled),
    so an absent measurement is never silently scored as a success.
    """
    cuts = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if lexical_sim >= cuts["verbatim_similarity"]:
        return "verbatim_copy"
    if semantic_preserved is None:
        return "unresolved"
    preserved = semantic_preserved >= cuts["semantic_preserved"]
    # An unmeasurable probe (gold too short to string-match) must not be read
    # as a surviving fact, so only an explicit True counts as survival.
    lost_fact = facts_survived is False
    wrong_answer = answer_correct is not None and float(answer_correct) < 1.0
    if preserved and not lost_fact and not wrong_answer:
        return "benign_paraphrase"
    if preserved:
        return "critical_detail_loss"
    if wrong_answer:
        return "information_loss"
    return "semantic_drift"


LABELS = (
    "benign_paraphrase",
    "critical_detail_loss",
    "information_loss",
    "semantic_drift",
    "verbatim_copy",
    "unresolved",
)
