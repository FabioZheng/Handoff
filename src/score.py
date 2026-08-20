"""Scoring: SQuAD-style normalisation, exact match, token-F1, bootstrap CIs.

C7 — no LLM judge anywhere in this file. Answers are short spans and are scored
with deterministic string metrics only. Where EM and F1 disagree materially the
caller is expected to report both; ``em_f1_disagreement`` quantifies it.
"""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation, drop articles, fix whitespace."""
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score_against_golds(pred: str, golds: list[str]) -> tuple[float, float]:
    """Max EM and max F1 over the gold answer and its aliases."""
    golds = [g for g in golds if g and str(g).strip()]
    if not golds:
        return 0.0, 0.0
    em = max(exact_match(pred, g) for g in golds)
    f1 = max(token_f1(pred, g) for g in golds)
    return em, f1


def extract_short_answer(raw: str) -> str:
    """Deterministic post-processing of a model's answer field.

    Models occasionally wrap a short span in scaffolding despite instruction.
    This strips the common wrappers. It is pure string surgery -- no judge, no
    semantic matching, and it never consults the gold answer.
    """
    if not raw:
        return ""
    text = raw.strip()
    # Drop <think>...</think> style blocks if a provider emits them.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # First non-empty line only.
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    # Strip a leading label such as "Answer:" / "**Answer:**".
    text = re.sub(r"^[*_`\s]*answer[*_`\s]*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("*_`").strip()
    # Strip a wrapping pair of quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text.rstrip(".").strip()


# ---------------- bootstrap ----------------

def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 10_000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap over the mean of `values`. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_bootstrap_delta(
    treatment: np.ndarray,
    control: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired bootstrap on (treatment - control), resampling questions, not rows.

    Both arrays must be aligned per question. Returns the point delta, the CI,
    and a two-sided bootstrap p-value for delta != 0.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    assert t.shape == c.shape, "paired arrays must align"
    d = t - c
    if d.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    observed = float(d.mean())
    # Two-sided bootstrap p: fraction of centred resamples at least as extreme.
    centred = means - observed
    p = float((np.abs(centred) >= abs(observed)).mean())
    return {
        "delta": observed,
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1 - alpha)),
        "p_value": p,
        "n": int(d.size),
    }


def em_f1_disagreement(em: np.ndarray, f1: np.ndarray) -> float:
    """Fraction of items scored 0 by EM but >=0.5 by F1 -- the ambiguous band.

    Reported so a materially different EM and F1 picture is visible rather than
    quietly averaged away (C7).
    """
    em = np.asarray(em, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    if em.size == 0:
        return float("nan")
    return float(((em == 0) & (f1 >= 0.5)).mean())
