"""LLM-judge scoring: does a predicted short answer mean the same as the gold?

Deliberately separate from ``score.py``. EM/F1 remain the deterministic primary
metrics and are always kept alongside the judge verdict, so any judge call can
be audited against the string metrics rather than replacing them silently.

The judge runs on a *different model family* from the systems under test
(llama-*), because a model grading its own family's outputs is a known source
of self-preference bias. It is deterministic (temperature 0) and every call is
content-hashed into the same on-disk cache as the rest of the pipeline, so
repeat scoring of an unchanged answer is free.
"""

from __future__ import annotations

import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import LLMClient  # noqa: E402

JUDGE_SYSTEM = (
    "You grade short answers to reading-comprehension questions. "
    "You are given a question, one or more reference answers, and a predicted answer. "
    "Decide whether the prediction conveys the same fact as any reference answer."
)

JUDGE_INSTRUCTION = (
    "A prediction is CORRECT if it conveys the same fact as any reference answer, allowing for "
    "differences in wording, word order, abbreviation, casing, punctuation, added detail, or a "
    "fuller sentence around the same fact.\n"
    "A prediction is INCORRECT if it states a different fact, names a different entity, is empty, "
    "says the information is unavailable, or hedges without committing to an answer.\n"
    "Reply with exactly one word: CORRECT or INCORRECT."
)


def judge_user_prompt(question: str, golds: list[str], prediction: str) -> str:
    references = "\n".join(f"- {g}" for g in golds)
    shown = prediction.strip() if prediction and prediction.strip() else "(no answer given)"
    return (
        f"Question: {question}\n\n"
        f"Reference answer(s):\n{references}\n\n"
        f"Predicted answer: {shown}\n\n"
        f"{JUDGE_INSTRUCTION}"
    )


def parse_verdict(text: str) -> tuple[int, bool]:
    """Return (correct, parse_ok). 'INCORRECT' is checked first: it contains 'CORRECT'."""
    upper = (text or "").strip().upper()
    if "INCORRECT" in upper:
        return 0, True
    if "CORRECT" in upper:
        return 1, True
    return 0, False


def judge_config(cfg: dict) -> dict:
    spec = dict(cfg.get("judge") or {})
    spec.setdefault("enabled", False)
    spec.setdefault("model_id", "openai/gpt-4o-mini")
    spec.setdefault("max_tokens", 8)
    spec.setdefault("cap_usd", 2.0)
    # Judge calls are tiny and independent; they need not inherit the
    # experiment's deliberately conservative generation concurrency.
    spec.setdefault("concurrency", 12)
    return spec


def make_judge_client(cfg: dict, dry_run: bool = False) -> tuple[LLMClient, dict]:
    """A second client pinned to the judge model.

    The cache key already includes the model id, so judge calls can never
    collide with the experiment's own calls despite sharing ``cache_dir``. The
    judge gets its own cost ledger so it cannot consume the experiment's cap
    (several experiment configs run with caps as tight as $0.25).
    """
    spec = judge_config(cfg)
    judge_cfg = copy.deepcopy(cfg)
    judge_cfg["model"]["id"] = spec["model_id"]
    judge_cfg["model"]["reasoning"] = None
    # Model-native prompt controls (for example Qwen3's ``/no_think``) are
    # properties of the system under test, never of the independent judge.
    judge_cfg["model"].pop("system_suffix", None)
    judge_cfg["cost"] = {
        "cap_usd": float(spec["cap_usd"]),
        "warn_at_fraction": cfg.get("cost", {}).get("warn_at_fraction", 0.8),
    }
    return LLMClient(judge_cfg, dry_run=dry_run), spec


def add_judge(rows: list[dict], cfg: dict, dry_run: bool = False, tag: str = "judge") -> LLMClient | None:
    """Add ``judge_correct`` (0/1) to every row that lacks it.

    Each row must carry ``question``, ``pred``, and ``golds`` (or ``gold``).
    Rows already holding a verdict are left untouched, so this is safe to call
    on every rerun. Returns the judge client (for its cost ledger), or None if
    judging is disabled or nothing needed scoring.
    """
    spec = judge_config(cfg)
    if not spec["enabled"] or dry_run:
        return None
    missing = [r for r in rows if "judge_correct" not in r]
    if not missing:
        print(f"[judge] reusing {len(rows)} verdicts")
        return None

    client, spec = make_judge_client(cfg, dry_run=dry_run)

    def score(row: dict) -> dict:
        golds = [str(g) for g in (row.get("golds") or [row.get("gold", "")]) if str(g).strip()]
        result = client.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_user_prompt(
                    str(row.get("question", "")), golds or [""], str(row.get("pred", "")),
                )},
            ],
            temperature=0.0,
            max_tokens=int(spec["max_tokens"]),
            seed=None,
            tag=tag,
        )
        correct, parse_ok = parse_verdict(result.text)
        return {"judge_correct": correct, "judge_parse_ok": parse_ok, "judge_raw": result.text.strip()}

    with ThreadPoolExecutor(max_workers=int(spec["concurrency"])) as pool:
        futures = {pool.submit(score, row): row for row in missing}
        for future, row in futures.items():
            row.update(future.result())

    unparsed = sum(1 for r in missing if not r.get("judge_parse_ok", True))
    print(f"[judge] scored {len(missing)} answers with {spec['model_id']}"
          + (f" ({unparsed} unparsed verdicts scored 0)" if unparsed else ""))
    return client


# ---------- semantic preservation across one handoff edge ----------
#
# The answer judge above grades a short span against a gold answer. This second
# judge grades a *message pair*: did rewriting M_i into M_{i+1} preserve what
# M_i said? It is the semantic counterpart to the deterministic lexical
# measures in ``paraphrase_metrics.py`` -- together they separate "reworded but
# equivalent" from "reworded and degraded". Same model family rule, same
# temperature 0, same content-addressed cache as every other call.

PRESERVATION_SYSTEM = (
    "You compare two research notes, where the second was written by rewriting the first. "
    "You judge only whether the rewrite preserves the information in the original. "
    "You never judge style, length, or writing quality."
)

PRESERVATION_INSTRUCTION = (
    "Reply EQUIVALENT if the rewrite preserves every fact, name, number, date, qualifier, "
    "relationship, and uncertainty stated in the original, even if the wording is completely "
    "different or the order has changed.\n"
    "Reply MINOR_LOSS if the rewrite preserves the main content but drops or blurs a detail, "
    "such as one number, date, name, qualifier, or a single supporting fact.\n"
    "Reply MAJOR_LOSS if the rewrite omits or contradicts a substantial part of the original, "
    "or adds a fact that the original did not state.\n"
    "Reply with exactly one word: EQUIVALENT, MINOR_LOSS, or MAJOR_LOSS."
)

# Ordinal encoding so the verdict can be averaged and plotted next to the
# continuous lexical measures. The raw word is always stored alongside it.
PRESERVATION_SCORES = {"EQUIVALENT": 1.0, "MINOR_LOSS": 0.5, "MAJOR_LOSS": 0.0}


def preservation_user_prompt(previous: str, current: str) -> str:
    original = previous.strip() if previous and previous.strip() else "(empty)"
    rewrite = current.strip() if current and current.strip() else "(empty)"
    return (
        f"Original notes:\n{original}\n\n"
        f"Rewritten notes:\n{rewrite}\n\n"
        f"{PRESERVATION_INSTRUCTION}"
    )


def parse_preservation(text: str) -> tuple[float, str, bool]:
    """Return (score, verdict, parse_ok). Checks the loss verdicts first."""
    upper = (text or "").strip().upper()
    for verdict in ("MAJOR_LOSS", "MINOR_LOSS", "EQUIVALENT"):
        if verdict in upper or verdict.replace("_", " ") in upper:
            return PRESERVATION_SCORES[verdict], verdict, True
    # An unparsed verdict is not evidence of preservation; leave it unscored
    # so ``classify_transition`` reports ``unresolved`` rather than success.
    return float("nan"), "", False


def add_preservation_judge(
    rows: list[dict], cfg: dict, dry_run: bool = False, tag: str = "preservation_judge",
) -> LLMClient | None:
    """Add ``semantic_preserved`` (1.0/0.5/0.0) to every transition row lacking it.

    Each row must carry ``previous_text`` and ``current_text``. Rows already
    holding a verdict are left untouched, so reruns are free. Returns the judge
    client for its cost ledger, or None when disabled or nothing needed scoring.
    """
    spec = judge_config(cfg)
    if not spec["enabled"] or dry_run:
        return None
    missing = [r for r in rows if "semantic_preserved" not in r]
    if not missing:
        print(f"[preservation] reusing {len(rows)} verdicts")
        return None

    client, spec = make_judge_client(cfg, dry_run=dry_run)

    def score(row: dict) -> dict:
        result = client.chat(
            [
                {"role": "system", "content": PRESERVATION_SYSTEM},
                {"role": "user", "content": preservation_user_prompt(
                    str(row.get("previous_text", "")), str(row.get("current_text", "")),
                )},
            ],
            temperature=0.0,
            # Verdict words are longer than the answer judge's CORRECT/INCORRECT.
            max_tokens=max(8, int(spec["max_tokens"])),
            seed=None,
            tag=tag,
        )
        preserved, verdict, parse_ok = parse_preservation(result.text)
        return {"semantic_preserved": preserved, "semantic_verdict": verdict,
                "semantic_parse_ok": parse_ok, "semantic_raw": result.text.strip()}

    with ThreadPoolExecutor(max_workers=int(spec["concurrency"])) as pool:
        futures = {pool.submit(score, row): row for row in missing}
        for future, row in futures.items():
            row.update(future.result())

    unparsed = sum(1 for r in missing if not r.get("semantic_parse_ok", True))
    print(f"[preservation] scored {len(missing)} transitions with {spec['model_id']}"
          + (f" ({unparsed} unparsed verdicts left unscored)" if unparsed else ""))
    return client


def backfill_field(rows: list[dict], field: str, lookup: dict, key_of) -> None:
    """Fill a missing field (e.g. the question text) on already-stored rows."""
    for row in rows:
        if not row.get(field):
            value = lookup.get(key_of(row))
            if value is not None:
                row[field] = value
