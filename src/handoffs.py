"""The five handoff mechanisms, plus the sealed subagent -> orchestrator boundary.

Stage 2 isolation (spec §5) is enforced structurally, not by prompt instruction:
``orchestrator_answer`` accepts a frozen ``SealedHandoff`` carrying three strings
and nothing else. The ``Question`` object -- and therefore every paragraph -- is
unreachable from that call frame. Passing a ``Question`` raises ``TypeError``.
``run.py`` asserts this at startup so the guarantee is tested, not asserted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from data import Question, split_sentences, _contains
from score import extract_short_answer

# ---------------------------------------------------------------- prompts

# C2: one answering persona, shared verbatim by the A_full single agent and the
# orchestrator, so no wording asymmetry can masquerade as a handoff effect.
ANSWER_SYSTEM = (
    "You answer questions using only the material you are given. "
    "Reply with only the short answer span - no explanation, no full sentence, no preamble. "
    "If the material seems insufficient, still give your single best guess."
)

SUBAGENT_SYSTEM = (
    "You are a research subagent. You have read source material that the orchestrator "
    "cannot see and will never see. Your output is the orchestrator's ONLY information "
    "for answering the question. Anything you leave out is lost."
)

FREEFORM_INSTRUCTION = (
    "Write a summary of what you found that will let the orchestrator answer the question.\n"
    "Write prose. Do not answer the question yourself."
)

STRUCTURED_INSTRUCTION = """Emit your findings as a single JSON object, and nothing else.

Schema:
{
  "claims": [
    {
      "text": "one self-contained factual claim, as a sentence",
      "source_para_id": <integer paragraph id the claim came from>,
      "constraints": ["qualifiers that make the claim precise: dates, numbers, roles, locations"],
      "confidence": "high" | "medium" | "low"
    }
  ],
  "unresolved": ["anything you could not establish from the source material"]
}

Include every claim the orchestrator needs, and no claim the source material does not support.
Do not answer the question yourself. Output only the JSON object."""

EXTRACTIVE_INSTRUCTION = """Select the sentences from the source material that the orchestrator needs.

Copy each selected sentence VERBATIM - character for character from the source. Do not paraphrase,
merge, shorten or rewrite. Write nothing of your own.

Output a single JSON object and nothing else:
{"sentences": [{"source_para_id": <integer paragraph id>, "text": "<verbatim sentence>"}]}"""


# ---------------------------------------------------------------- rendering

def render_paragraphs(q: Question) -> str:
    """The subagent's (and A_full's) view: gold paragraphs plus distractors,
    in the fixed shuffled order set in data.build_question."""
    return "\n\n".join(f"[P{p.pid}] {p.title}\n{p.text}" for p in q.paragraphs)


def render_sentence_list(items: list[dict]) -> str:
    """Shared serialisation for D_extractive and E_oracle so they are comparable."""
    return "\n".join(f"[P{it['source_para_id']}] {it['text']}" for it in items)


def render_structured(obj: dict) -> str:
    """Deterministic text serialisation of a C_structured object.

    Injections mutate the object; this function is the only thing that turns an
    object into orchestrator-visible text, so a corrupted object and a clean one
    are rendered by identical code (C3).
    """
    lines: list[str] = []
    for i, c in enumerate(obj.get("claims") or [], start=1):
        src = c.get("source_para_id")
        src_s = f"P{src}" if src is not None else "unknown source"
        parts = [f"{i}. {str(c.get('text', '')).strip()}"]
        cons = [str(x).strip() for x in (c.get("constraints") or []) if str(x).strip()]
        if cons:
            parts.append(f"   constraints: {'; '.join(cons)}")
        conf = c.get("confidence")
        parts.append(f"   source: {src_s}" + (f" | confidence: {conf}" if conf else ""))
        lines.append("\n".join(parts))
    body = "\n".join(lines) if lines else "(no claims)"
    unresolved = [str(u).strip() for u in (obj.get("unresolved") or []) if str(u).strip()]
    if unresolved:
        body += "\n\nUnresolved:\n" + "\n".join(f"- {u}" for u in unresolved)
    return body


# ---------------------------------------------------------------- handoff type

@dataclass
class Handoff:
    qid: str
    mechanism: str
    seed: int | None
    text: str
    structured: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "qid": self.qid,
            "mechanism": self.mechanism,
            "seed": self.seed,
            "text": self.text,
            "structured": self.structured,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> "Handoff":
        return Handoff(d["qid"], d["mechanism"], d["seed"], d["text"],
                       d.get("structured"), d.get("meta", {}))


# ---------------------------------------------------------------- the seal

@dataclass(frozen=True)
class SealedHandoff:
    """Everything the orchestrator is permitted to know. Three strings."""
    qid: str
    question: str
    handoff_text: str
    mechanism: str


def seal(question_text: str, handoff: Handoff) -> SealedHandoff:
    if not isinstance(question_text, str):
        raise TypeError("seal() takes the question as a plain string")
    if not isinstance(handoff, Handoff):
        raise TypeError("seal() takes a Handoff")
    return SealedHandoff(handoff.qid, question_text, handoff.text, handoff.mechanism)


# ---------------------------------------------------------------- JSON parsing

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(raw: str) -> tuple[dict | None, str]:
    """Best-effort parse of a model's JSON output. Returns (obj, error_reason)."""
    if not raw or not raw.strip():
        return None, "empty"
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return (obj, "") if isinstance(obj, dict) else (None, "not_an_object")
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return (obj, "") if isinstance(obj, dict) else (None, "not_an_object")
        except json.JSONDecodeError as exc:
            return None, f"decode_error:{exc.msg}"
    return None, "no_json_found"


# ---------------------------------------------------------------- mechanisms

def _subagent_call(client, q: Question, cfg: dict, instruction: str, seed: int, tag: str):
    user = (
        f"Source material:\n{render_paragraphs(q)}\n\n"
        f"Question the orchestrator must answer: {q.question}\n\n"
        f"{instruction}"
    )
    return client.chat(
        [{"role": "system", "content": SUBAGENT_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        seed=seed,
        tag=tag,
    )


def make_freeform(client, q: Question, cfg: dict, seed: int) -> Handoff:
    res = _subagent_call(client, q, cfg, FREEFORM_INSTRUCTION, seed, "handoff_B_freeform")
    return Handoff(
        qid=q.qid, mechanism="B_freeform", seed=seed, text=res.text.strip(),
        meta={"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
              "cached": res.cached, "finish_reason": res.finish_reason},
    )


def make_structured(client, q: Question, cfg: dict, seed: int) -> Handoff:
    res = _subagent_call(client, q, cfg, STRUCTURED_INSTRUCTION, seed, "handoff_C_structured")
    obj, err = parse_json_object(res.text)
    valid_pids = {p.pid for p in q.paragraphs}

    if obj is None:
        # Parse failure is a real property of the mechanism, not a bug to hide.
        # Fall back to the raw text and record it so the rate is reportable.
        return Handoff(
            qid=q.qid, mechanism="C_structured", seed=seed, text=res.text.strip(),
            structured=None,
            meta={"parse_ok": False, "parse_error": err,
                  "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                  "cached": res.cached, "finish_reason": res.finish_reason},
        )

    claims = []
    for c in (obj.get("claims") or []):
        if not isinstance(c, dict):
            continue
        pid = c.get("source_para_id")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = None
        if pid not in valid_pids:
            pid = None  # hallucinated provenance -> recorded as unknown, claim kept
        cons = c.get("constraints") or []
        if isinstance(cons, str):
            cons = [cons]
        conf = str(c.get("confidence", "")).strip().lower()
        claims.append({
            "text": str(c.get("text", "")).strip(),
            "source_para_id": pid,
            "constraints": [str(x).strip() for x in cons if str(x).strip()],
            "confidence": conf if conf in ("high", "medium", "low") else "medium",
        })
    unresolved = obj.get("unresolved") or []
    if isinstance(unresolved, str):
        unresolved = [unresolved]
    norm = {"claims": claims, "unresolved": [str(u).strip() for u in unresolved if str(u).strip()]}

    return Handoff(
        qid=q.qid, mechanism="C_structured", seed=seed, text=render_structured(norm),
        structured=norm,
        meta={"parse_ok": True, "n_claims": len(claims),
              "n_claims_bad_pid": sum(1 for c in claims if c["source_para_id"] is None),
              "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
              "cached": res.cached, "finish_reason": res.finish_reason},
    )


def make_extractive(client, q: Question, cfg: dict, seed: int) -> Handoff:
    """Verbatim sentences only. 'Verbatim' is enforced in code, not trusted.

    Each returned sentence is checked against the source paragraph it claims to
    come from; anything that is not actually present there is dropped. The drop
    rate is recorded, because a high one means the model could not comply and
    that is itself a result about D.
    """
    res = _subagent_call(client, q, cfg, EXTRACTIVE_INSTRUCTION, seed, "handoff_D_extractive")
    obj, err = parse_json_object(res.text)
    by_pid = {p.pid: p.text for p in q.paragraphs}

    proposed: list[dict] = []
    if obj is not None:
        for it in (obj.get("sentences") or []):
            if not isinstance(it, dict):
                continue
            try:
                pid = int(it.get("source_para_id"))
            except (TypeError, ValueError):
                continue
            txt = " ".join(str(it.get("text", "")).split())
            if txt:
                proposed.append({"source_para_id": pid, "text": txt})

    kept: list[dict] = []
    for it in proposed:
        src = by_pid.get(it["source_para_id"])
        if src is not None and _contains(src, it["text"]):
            kept.append(it)
        else:
            # Allow a correct sentence attributed to the wrong paragraph through,
            # relabelled, rather than discarding a genuinely verbatim extraction.
            match = next((pid for pid, t in by_pid.items() if _contains(t, it["text"])), None)
            if match is not None:
                kept.append({"source_para_id": match, "text": it["text"]})

    return Handoff(
        qid=q.qid, mechanism="D_extractive", seed=seed, text=render_sentence_list(kept),
        structured={"sentences": kept},
        meta={"parse_ok": obj is not None, "parse_error": err,
              "n_proposed": len(proposed), "n_kept": len(kept),
              "n_dropped_not_verbatim": len(proposed) - len(kept),
              "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
              "cached": res.cached, "finish_reason": res.finish_reason},
    )


def make_oracle(q: Question) -> Handoff:
    """Gold supporting sentences, verbatim, with paragraph ids. No LLM involved.

    MuSiQue annotates supporting *paragraphs*, not sentences, so the sentence set
    is derived deterministically in data.build_question from the gold decomposition.
    Sentences are emitted in paragraph order, not hop order, so the handoff does
    not leak the reasoning chain's ordering.
    """
    seen: set[tuple[int, str]] = set()
    items: list[dict] = []
    for gs in sorted(q.gold_sentences, key=lambda s: (s.pid, s.step)):
        key = (gs.pid, gs.text)
        if key in seen:
            continue
        seen.add(key)
        items.append({"source_para_id": gs.pid, "text": gs.text})
    return Handoff(
        qid=q.qid, mechanism="E_oracle", seed=None, text=render_sentence_list(items),
        structured={"sentences": items},
        meta={"n_sentences": len(items), "n_gold_paras": len(q.gold_pids), "llm_calls": 0},
    )


MECHANISM_BUILDERS = {
    "B_freeform": make_freeform,
    "C_structured": make_structured,
    "D_extractive": make_extractive,
}


# ---------------------------------------------------------------- answering

def full_context_answer(client, q: Question, cfg: dict) -> dict:
    """A_full -- the ceiling. A single agent answers straight from the paragraphs.

    This deliberately does NOT go through orchestrator_answer: the ceiling condition
    is the one place paragraphs are allowed to reach an answering call.
    """
    user = f"Source material:\n{render_paragraphs(q)}\n\nQuestion: {q.question}\nAnswer:"
    res = client.chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["orchestrator_temperature"],
        max_tokens=cfg["decoding"]["answer_max_tokens"],
        seed=None,
        tag="answer_A_full",
    )
    return {
        "qid": q.qid, "mechanism": "A_full", "seed": None,
        "raw": res.text, "pred": extract_short_answer(res.text),
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "cached": res.cached, "cost_usd": res.cost_usd,
    }


def orchestrator_answer(client, sealed: SealedHandoff, cfg: dict, tag: str = "answer") -> dict:
    """The orchestrator. Its entire world is `sealed`: a question and a handoff.

    There is no parameter through which a paragraph could arrive.
    """
    if not isinstance(sealed, SealedHandoff):
        raise TypeError(
            f"orchestrator_answer requires a SealedHandoff, got {type(sealed).__name__}. "
            "The source paragraphs must not be reachable from this call."
        )
    user = (
        f"Research notes from your subagent:\n{sealed.handoff_text}\n\n"
        f"Question: {sealed.question}\nAnswer:"
    )
    res = client.chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["orchestrator_temperature"],
        max_tokens=cfg["decoding"]["answer_max_tokens"],
        seed=None,
        tag=tag,
    )
    return {
        "qid": sealed.qid, "mechanism": sealed.mechanism,
        "raw": res.text, "pred": extract_short_answer(res.text),
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "cached": res.cached, "cost_usd": res.cost_usd,
    }
