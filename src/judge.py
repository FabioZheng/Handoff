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


def backfill_field(rows: list[dict], field: str, lookup: dict, key_of) -> None:
    """Fill a missing field (e.g. the question text) on already-stored rows."""
    for row in rows:
        if not row.get(field):
            value = lookup.get(key_of(row))
            if value is not None:
                row[field] = value
