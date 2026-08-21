"""Conservative LLM screening for candidate distractors.

This is an automatic audit, not a substitute for human relevance judgements.
Only passages classified ``IRRELEVANT`` are eligible to be used as negatives;
``RELEVANT`` and unparseable answers are rejected.  Every decision is cached
by the normal content-addressed LLM cache and written to an auditable CSV by
the calling experiment.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

from llm import LLMClient

SYSTEM = "You are a strict relevance auditor for question-answering evaluation."
INSTRUCTION = """Classify the candidate passage for the target question.

Reply IRRELEVANT only if the passage supplies no answer, partial answer, or
fact that would help a careful reader determine the target answer. A passage
that merely shares words or topic is IRRELEVANT only when it supplies no such
useful fact. Reply RELEVANT if it directly or indirectly supports the answer,
contains a gold-answer variant, or would be useful evidence. When uncertain,
reply RELEVANT. Reply with exactly IRRELEVANT or RELEVANT."""


def verifier_config(cfg: dict) -> dict:
    spec = dict(cfg.get("negative_verification") or {})
    spec.setdefault("enabled", False)
    spec.setdefault("model_id", "openai/gpt-4o-mini")
    spec.setdefault("max_tokens", 8)
    spec.setdefault("cap_usd", 3.0)
    spec.setdefault("concurrency", 12)
    spec.setdefault("candidate_scan_k", 60)
    return spec


def screen(candidates: list[dict], cfg: dict) -> tuple[list[dict], LLMClient | None]:
    """Return audited candidate records with an ``is_irrelevant`` boolean.

    Candidate records must include ``question``, ``golds``, and ``text``.
    Other fields are preserved verbatim for the audit artifact.
    """
    spec = verifier_config(cfg)
    if not spec["enabled"]:
        return [{**row, "verdict": "UNSCREENED", "is_irrelevant": False} for row in candidates], None
    verifier_cfg = copy.deepcopy(cfg)
    verifier_cfg["model"]["id"] = spec["model_id"]
    verifier_cfg["cost"] = {"cap_usd": float(spec["cap_usd"]), "warn_at_fraction": 0.8}
    client = LLMClient(verifier_cfg)

    def one(row: dict) -> dict:
        golds = "\n".join(f"- {value}" for value in row["golds"])
        prompt = (
            f"Question: {row['question']}\n\nReference answers:\n{golds}\n\n"
            f"Candidate passage:\n{row['text']}\n\n{INSTRUCTION}"
        )
        result = client.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=int(spec["max_tokens"]), seed=None,
            tag="negative_relevance_audit",
        )
        verdict = result.text.strip().upper()
        irrelevant = verdict == "IRRELEVANT"
        return {**row, "verdict": verdict, "is_irrelevant": irrelevant,
                "verdict_parse_ok": verdict in {"IRRELEVANT", "RELEVANT"},
                "cached": result.cached}

    with ThreadPoolExecutor(max_workers=int(spec["concurrency"])) as pool:
        audited = list(pool.map(one, candidates))
    return audited, client
