"""Experiment 10: communication regret under a hard communication budget.

Question. When an agent writes a bounded handoff for the query it currently
knows, does it buy accuracy on that query at the cost of accuracy on other
queries the same source could later be asked?

Design. Each source context independently answers k questions. Every question
takes a turn as the conditioning query; each resulting message is then answered
against *every* question of that context. That gives a utility matrix per
context, whose diagonal is present-query utility ``U_now`` and whose
off-diagonal is future-query utility ``U_future``. Because the same question
appears on the diagonal for one rotation and off it for the others, question
difficulty cannot drive the contrast, and no fixed A/B split has to be trusted.

Controls that make the contrast mean something:

* **Hard budget.** Every policy is held to the same delivered word *band* -
  floor and cap - by the three-stage procedure in ``src/budget.py``, and the
  realised lengths are
  audited, tested pairwise, and re-tested on a length-matched subsample and on
  a variant trimmed to the cell-minimum length. An equal ``max_tokens`` is not
  a length control and is not relied on here.
* **Rotation.** See above.
* **Ceiling.** The answerer also reads the source directly for every question,
  which is the no-handoff ceiling ``U(D, q)`` and the reference in the regret difference
  ``R(q) = U(D, q) - U(m, q)``.
* **Leakage.** Every question passed a closed-book filter at build time, and
  the closed-book baseline is re-measured here.

Policies differ only in what the sender is told about the information need
(``budget.policy_block``): ``generic`` knows nothing, ``conditioned`` knows the
current question, ``reusable`` knows it and is told further questions may
follow, ``oracle`` sees the whole question set. Two extractive arms replace
abstractive rewriting with verbatim sentence selection, separating
information-selection loss from rewriting loss.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anticipatory_data as ad  # noqa: E402
import budget as bd  # noqa: E402
import elimination as el  # noqa: E402
import handoffs as hm  # noqa: E402
import lm_unit_scores as lms  # noqa: E402
import regret_data as rd  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import LLMClient, estimate_tokens, load_config  # noqa: E402
from score import (bootstrap_ci, extract_short_answer, paired_bootstrap_delta,  # noqa: E402
                   score_against_golds)

SCHEMA_VERSION = "communication-regret-v1"
METRICS = ("em", "f1", "judge_correct")
ENDPOINTS = ("now", "future")

# Message keys for arms that are derived from a generated message rather than
# generated themselves. Trimming happens after generation, so a trimmed arm
# costs answer calls only.
TRIM_SUFFIX = "@trim"


# ---------------------------------------------------------------------------
# Small deterministic helpers


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(*parts) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def cohens_dz(deltas: np.ndarray) -> float:
    """Paired effect size: mean difference over its own standard deviation.

    Reported next to every contrast because a delta on a 0-1 accuracy scale is
    interpretable but says nothing about how consistent it is across contexts.
    """
    d = np.asarray(deltas, dtype=float)
    if d.size < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    return float(d.mean() / sd) if sd > 0 else float("nan")


def make_clients(cfg: dict, dry_run: bool = False) -> tuple[LLMClient, LLMClient]:
    """Sender and answerer under one cache and one cost cap.

    They are the same model here, but the split is kept explicit so a
    second-model robustness check needs a config change and nothing else.
    """
    sender = LLMClient(cfg, dry_run=dry_run)
    spec = cfg.get("answerer") or cfg["model"]
    if spec["id"] == cfg["model"]["id"]:
        return sender, sender
    answer_cfg = copy.deepcopy(cfg)
    answer_cfg["model"] = copy.deepcopy(spec)
    answerer = LLMClient(answer_cfg, dry_run=dry_run)
    # One experiment, one cap. The model id is already inside every cache key.
    answerer.cache = sender.cache
    answerer.ledger = sender.ledger
    return sender, answerer


# ---------------------------------------------------------------------------
# Design


def load_design(cfg: dict, limit: int | None = None):
    name = cfg["dataset"]["active"]
    spec = cfg["dataset"][name]
    manifest, contexts = rd.load_contexts(ROOT / spec["contexts_jsonl"])
    n = int(limit if limit is not None else spec["n_contexts"])
    contexts = tuple(contexts[:n])
    if not contexts:
        raise RuntimeError(f"{spec['contexts_jsonl']} produced zero contexts")
    for context in contexts:
        if context.dataset != name:
            raise ValueError(f"{context.context_id}: dataset {context.dataset!r} != {name!r}")
        if context.dataset == "relation_dossiers":
            rd.validate_relation_context(context)
    rotations = {c.context_id: rd.rotations_for(c) for c in contexts}
    return manifest, contexts, rotations


KNOWN_POLICIES = bd.ALL_POLICIES + el.SELECTION_POLICIES


def resolve_policies(cfg: dict, cli_value: str | None) -> list[str]:
    configured = list(cfg["policies"])
    for policy in configured:
        if policy not in KNOWN_POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
    if cli_value is None:
        return configured
    chosen = [p.strip() for p in cli_value.split(",") if p.strip()]
    unknown = [p for p in chosen if p not in configured]
    if not chosen or unknown:
        raise ValueError(f"--policies must be a non-empty subset of {configured}")
    return list(dict.fromkeys(chosen))


# ---------------------------------------------------------------------------
# Message generation


def rotation_invariant(policy: str) -> bool:
    """Does this policy's message depend on which question is currently known?

    Selection arms answer it the same way the generated arms do: a selector that
    never reads a query produces one message per (context, budget), shared by
    every rotation, because running it k times would produce k identical
    messages and invite a rotation effect that is really bookkeeping.
    """
    if el.is_selection(policy):
        return not el.is_query_aware(policy)
    return policy in bd.ROTATION_INVARIANT


ELIMINATION_DEFAULTS = {
    "lm_scores_path": "data/compression_mechanism/lm_unit_scores.jsonl",
    "nonllm_generic_scorer": "centrality",
    "random_seed_salt": "elimination",
}


def elimination_settings(cfg: dict) -> dict:
    settings = dict(ELIMINATION_DEFAULTS)
    settings.update(cfg.get("elimination") or {})
    return settings


def selection_resources(cfg: dict, contexts, policies: list[str]) -> dict | None:
    """Evidence units, and the LM scores the ``lm_*`` arms select on.

    Built once per run and shared by every spec: unit extraction is
    deterministic but not free, and the LM score table has to be validated
    against the corpus exactly once, not per message.
    """
    selection = [p for p in policies if el.is_selection(p)]
    if not selection:
        return None
    spec = cfg["dataset"][cfg["dataset"]["active"]]
    corpus = ROOT / spec["contexts_jsonl"]
    _, dossiers = ad.load_dossiers(corpus)
    wanted = {c.context_id for c in contexts}
    units = {}
    for dossier in dossiers:
        if dossier["context_id"] not in wanted:
            continue
        rows = ad.extract_units(dossier)
        ad.check_coverage(dossier, rows)
        units[dossier["context_id"]] = rows
    missing = sorted(wanted - set(units))
    if missing:
        raise RuntimeError(f"no evidence units for {len(missing)} contexts "
                           f"(first: {missing[0]})")

    settings = elimination_settings(cfg)
    manifest, scores = {}, {"generic": {}, "conditioned": {}}
    if any(p in el.LM_POLICIES for p in selection):
        manifest, scores = lms.load_scores(
            ROOT / settings["lm_scores_path"], lms.corpus_hash(corpus))
    return {"units": units, "lm_manifest": manifest, "lm_scores": scores,
            "settings": settings,
            "scorer_hash": stable_hash(manifest.get("model_id", ""),
                                       manifest.get("commit_hash", ""),
                                       manifest.get("cond_template", ""),
                                       manifest.get("generic_formula", ""),
                                       manifest.get("conditioned_formula", ""),
                                       settings["nonllm_generic_scorer"],
                                       settings["random_seed_salt"])}


def message_specs(contexts, rotations, policies: list[str], budgets, cfg: dict,
                  resources: dict | None = None) -> list[dict]:
    """One spec per message that has to exist.

    Rotation-invariant policies get exactly one message per (context, budget):
    their prompt does not contain the current question, so generating k copies
    would pay k times for one prompt and, worse, let sampling noise masquerade
    as a rotation effect.
    """
    specs = []
    for context in contexts:
        for b in budgets:
            for policy in policies:
                if rotation_invariant(policy):
                    specs.append(_message_spec(context, None, policy, b, cfg, resources))
                else:
                    for rotation in rotations[context.context_id]:
                        specs.append(_message_spec(context, rotation, policy, b, cfg, resources))
    keys = [s["message_key"] for s in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate message key")
    return specs


def _message_spec(context, rotation, policy: str, b: bd.Budget, cfg: dict,
                  resources: dict | None = None) -> dict:
    current = context.question(rotation.current_qid).question if rotation else None
    rotation_id = rotation.rotation_id if rotation else "-"
    base = {
        "message_key": f"{context.context_id}|{policy}|{b.label}|{rotation_id}",
        "context": context,
        "context_id": context.context_id,
        "rotation_id": rotation_id,
        "current_qid": rotation.current_qid if rotation else "",
        "policy": policy,
        "budget": b,
    }
    if el.is_selection(policy):
        if resources is None:
            raise RuntimeError(f"policy {policy!r} needs selection resources")
        # The selector's whole world, assembled here so the leakage guarantee is
        # visible in one place: the units of this source, this budget, and - only
        # for a query-aware arm - the current question. The hidden questions of
        # the rotation are not in scope and have nowhere to enter.
        query = current if el.is_query_aware(policy) else None
        return {
            **base,
            "kind": "selection",
            "prompt": "",
            "query": query,
            "units": resources["units"][context.context_id],
            "lm_scores": _lm_score_view(policy, resources, rotation),
            "settings": resources["settings"],
            "request_hash": stable_hash(
                SCHEMA_VERSION, "selection", policy, b.words, context.content_hash,
                resources["scorer_hash"], query or "",
            ),
        }
    prompt = bd.sender_prompt(
        policy, context.source, b,
        current_question=current,
        all_questions=context.all_question_texts if policy == "oracle" else None,
    )
    return {
        **base,
        "kind": "generated",
        "prompt": prompt,
        "request_hash": stable_hash(
            SCHEMA_VERSION, cfg["model"]["id"], bd.SENDER_SYSTEM, prompt,
            float(cfg["decoding"]["sender_temperature"]), float(cfg["decoding"]["top_p"]),
            b.as_dict(), int(cfg["budget"]["max_length_corrections"]),
            context.content_hash,
        ),
    }


def _lm_score_view(policy: str, resources: dict, rotation) -> dict[str, float] | None:
    """The LM scores this one message may see - and no others.

    A conditioned selector gets the score table for its OWN conditioning
    question only. Handing it the whole table would leave the leakage guarantee
    resting on a lookup key rather than on what was passed.
    """
    if policy not in el.LM_POLICIES:
        return None
    table = resources["lm_scores"]
    if policy == "lm_generic":
        return dict(table["generic"])
    qid = rotation.current_qid
    return {unit_id: score for (score_qid, unit_id), score in table["conditioned"].items()
            if score_qid == qid}


def run_selection_message(spec: dict) -> dict:
    """Build an elimination message. No model, no network, no sampling.

    The row carries the same length fields as a generated one so every
    downstream length audit, table and plot treats the two mechanisms
    identically, plus the selection diagnostics that let a wrong answer be
    attributed to deleted evidence rather than to the reader.
    """
    b: bd.Budget = spec["budget"]
    context = spec["context"]
    message = el.build_message(
        spec["policy"], spec["units"], b.words,
        source=context.source,
        questions=context.questions,
        query=spec["query"],
        lm_scores=spec["lm_scores"],
        seed_material=f"{spec['settings']['random_seed_salt']}|{context.context_id}|{b.words}",
        generic_scorer=spec["settings"]["nonllm_generic_scorer"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "message_key": spec["message_key"],
        "request_hash": spec["request_hash"],
        "context_id": spec["context_id"],
        "context_hash": context.content_hash,
        "rotation_id": spec["rotation_id"],
        "current_qid": spec["current_qid"],
        "policy": spec["policy"],
        "derived_from": "",
        "handoff_text": message.text,
        "handoff_hash": stable_hash(message.text),
        "mechanism_family": "elimination",
        # Length fields, on the same footing as a generated message. A selector
        # cannot overrun (it packs whole sentences under the cap) and cannot pad,
        # so the corrections are structurally zero rather than incidentally so.
        "budget_words": b.words,
        "floor_words": b.floor_words,
        "budget_tokens": b.tokens,
        "api_max_tokens": 0,
        "attempts": 1,
        "shrink_corrections": 0,
        "expand_corrections": 0,
        "first_draft_words": message.delivered_words,
        "final_draft_words": message.delivered_words,
        "truncated": False,
        "truncated_words_dropped": 0,
        "over_budget_before_truncation": False,
        "under_floor_delivered": message.delivered_words < b.floor_words,
        "repetition_rate": 0.0,
        "delivered_estimated_tokens": estimate_tokens(message.text) if message.text else 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "finish_reasons": ["selection"],
        "cached": True,
        "text": message.text,
        **message.as_dict(),
    }


def run_message_call(client: LLMClient, spec: dict, cfg: dict) -> dict:
    if spec.get("kind") == "selection":
        return run_selection_message(spec)
    b: bd.Budget = spec["budget"]
    message = bd.generate_within_budget(
        client, spec["prompt"], b,
        temperature=float(cfg["decoding"]["sender_temperature"]),
        seed=None,
        tag=f"regret_{spec['policy']}_{b.label}",
        max_corrections=int(cfg["budget"]["max_length_corrections"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "message_key": spec["message_key"],
        "request_hash": spec["request_hash"],
        "context_id": spec["context_id"],
        "context_hash": spec["context"].content_hash,
        "rotation_id": spec["rotation_id"],
        "current_qid": spec["current_qid"],
        "policy": spec["policy"],
        "derived_from": "",
        "handoff_text": message.text,
        "handoff_hash": stable_hash(message.text),
        **message.as_dict(),
    }


def generate_messages(client, specs: list[dict], cfg: dict,
                      stored: list[dict]) -> tuple[list[dict], int]:
    index = {row["message_key"]: row for row in stored
             if row.get("schema_version") == SCHEMA_VERSION}
    current, missing = {}, []
    for spec in specs:
        old = index.get(spec["message_key"])
        if old is not None and old.get("request_hash") == spec["request_hash"]:
            current[spec["message_key"]] = old
        else:
            missing.append(spec)
    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        futures = {pool.submit(run_message_call, client, spec, cfg): spec for spec in missing}
        for future, spec in futures.items():
            current[spec["message_key"]] = future.result()
    return [current[spec["message_key"]] for spec in specs], len(missing)


def trimmed_messages(messages: list[dict], budgets_words: list[int]) -> list[dict]:
    """Strictest length control: cut every policy in a cell to the cell minimum.

    A cell is one (context, budget). Rotation-invariant policies hold a single
    message that participates in every rotation of that context, so the minimum
    is taken over the whole group: trimming the shared message to a different
    length per rotation would mean the generic arm was no longer one message.

    Only the abstractive policies take part. The extractive arms are a
    different mechanism, and letting a very short verbatim selection set the
    floor would trim the comparison of interest down to nothing.

    This costs no generation calls: it is deterministic truncation of text that
    already exists, answered again.
    """
    if not budgets_words:
        return []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in messages:
        if (int(row["budget_words"]) in budgets_words and not row.get("derived_from")
                and row["policy"] in bd.ABSTRACTIVE_POLICIES):
            groups[(row["context_id"], int(row["budget_words"]))].append(row)
    out = []
    for (_context_id, _words), group in sorted(groups.items()):
        floor = min(int(row["delivered_words"]) for row in group)
        if floor < 1:
            continue
        for row in group:
            text, cut, dropped = bd.truncate_to_words(row["handoff_text"], floor)
            new = dict(row)
            new.update({
                "message_key": row["message_key"] + TRIM_SUFFIX,
                "policy": row["policy"] + TRIM_SUFFIX,
                "derived_from": row["message_key"],
                "handoff_text": text,
                "handoff_hash": stable_hash(text),
                "delivered_words": bd.word_count(text),
                "truncated": bool(row["truncated"] or cut),
                "truncated_words_dropped": int(row["truncated_words_dropped"]) + dropped,
                "fill_ratio": bd.word_count(text) / float(row["budget_words"]),
                "trim_floor_words": floor,
            })
            out.append(new)
    return out


# ---------------------------------------------------------------------------
# Answering


def answer_request_hash(model_id: str, question: str, material: str,
                        temperature: float, seed: int | None, cfg: dict) -> str:
    return stable_hash(
        SCHEMA_VERSION, model_id, hm.ANSWER_SYSTEM, question, material,
        float(temperature), float(cfg["decoding"]["top_p"]),
        int(cfg["decoding"]["answer_max_tokens"]), seed,
    )


def answer_specs(contexts, rotations, messages: list[dict], cfg: dict) -> list[dict]:
    """Every answer the analysis needs: baselines first, then message x question.

    Each message is evaluated against every question of its context. For a
    rotation-invariant policy that single set of answers is reused by every
    rotation during analysis rather than re-requested.
    """
    specs = []
    answer_model = (cfg.get("answerer") or cfg["model"])["id"]
    answer_temp = float(cfg["decoding"]["answer_temperature"])
    closed_temp = float(cfg["decoding"]["closed_book_temperature"])
    closed_samples = int(cfg["baselines"]["closed_book_samples"])

    for context in contexts:
        for q in context.questions:
            if cfg["baselines"].get("direct_context", True):
                specs.append({
                    "answer_key": f"baseline|{context.context_id}|{q.qid}|direct_context|0",
                    "message_key": "", "context_id": context.context_id, "policy": "direct_context",
                    "budget_words": 0, "rotation_id": "-", "current_qid": "",
                    "baseline": "direct_context", "sample": 0, "question": q,
                    "material": context.source, "temperature": answer_temp, "seed": None,
                    "technical_failure": False,
                })
            for sample in range(closed_samples):
                specs.append({
                    "answer_key": f"baseline|{context.context_id}|{q.qid}|closed_book|{sample}",
                    "message_key": "", "context_id": context.context_id, "policy": "closed_book",
                    "budget_words": 0, "rotation_id": "-", "current_qid": "",
                    "baseline": "closed_book", "sample": sample, "question": q,
                    "material": "(No research material was supplied.)",
                    "temperature": closed_temp, "seed": 1000 + sample,
                    "technical_failure": False,
                })

    context_map = {c.context_id: c for c in contexts}
    for message in messages:
        context = context_map[message["context_id"]]
        for q in context.questions:
            specs.append({
                "answer_key": f"message|{message['message_key']}|q={q.qid}",
                "message_key": message["message_key"], "context_id": context.context_id,
                "policy": message["policy"], "budget_words": int(message["budget_words"]),
                "rotation_id": message["rotation_id"], "current_qid": message["current_qid"],
                "baseline": "", "sample": 0, "question": q,
                "material": message["handoff_text"], "temperature": answer_temp, "seed": None,
                "technical_failure": not str(message["handoff_text"]).strip(),
            })

    for spec in specs:
        spec["request_hash"] = answer_request_hash(
            answer_model, spec["question"].question, spec["material"],
            spec["temperature"], spec["seed"], cfg)
    keys = [s["answer_key"] for s in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate answer key")
    return specs


def answer_record_base(spec: dict) -> dict:
    q = spec["question"]
    return {
        "schema_version": SCHEMA_VERSION, "answer_key": spec["answer_key"],
        "request_hash": spec["request_hash"], "message_key": spec["message_key"],
        "context_id": spec["context_id"], "policy": spec["policy"],
        "budget_words": spec["budget_words"], "rotation_id": spec["rotation_id"],
        "current_qid": spec["current_qid"], "eval_qid": q.qid,
        "is_diagonal": bool(spec["current_qid"] and spec["current_qid"] == q.qid),
        "baseline": spec["baseline"], "sample": spec["sample"],
        "question": q.question, "golds": list(q.golds),
        "technical_failure": bool(spec["technical_failure"]),
    }


def run_answer_call(client: LLMClient, spec: dict, cfg: dict) -> dict:
    q = spec["question"]
    base = answer_record_base(spec)
    if spec["technical_failure"]:
        # An empty handoff is a real outcome of a budgeted channel, not a bug to
        # be hidden: it scores zero rather than dropping the cell and quietly
        # shrinking one arm's n.
        return {**base, "raw": "", "pred": "", "em": 0.0, "f1": 0.0, "cached": True,
                "prompt_tokens": 0, "completion_tokens": 0, "judge_correct": 0,
                "judge_parse_ok": True, "judge_raw": "EMPTY_HANDOFF"}
    sealed = hm.SealedHandoff(qid=q.qid, question=q.question,
                              handoff_text=spec["material"], mechanism=spec["policy"])
    if not isinstance(sealed, hm.SealedHandoff):  # pragma: no cover - defensive
        raise TypeError("the answerer accepts only a frozen SealedHandoff")
    user = (f"Research material:\n{sealed.handoff_text}\n\n"
            f"Question:\n{sealed.question}\nAnswer:")
    result = client.chat(
        [{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=spec["temperature"], max_tokens=int(cfg["decoding"]["answer_max_tokens"]),
        seed=spec["seed"], tag=f"regret_answer_{spec['policy']}",
    )
    pred = extract_short_answer(result.text)
    em, f1 = score_against_golds(pred, list(q.golds))
    return {**base, "raw": result.text, "pred": pred, "em": em, "f1": f1,
            "cached": result.cached, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens}


def _answer_signature(row: dict) -> tuple:
    return (str(row.get("raw", "")), str(row.get("pred", "")),
            float(row.get("em", 0.0)), float(row.get("f1", 0.0)))


def generate_answers(client, specs: list[dict], cfg: dict,
                     stored: list[dict]) -> tuple[list[dict], int]:
    """Single-flight identical requests.

    Policy, rotation and diagonal/off-diagonal status are evaluator metadata:
    none of them reach the answering prompt. Two logically distinct cells whose
    full request hash matches are therefore the same request, and issuing both
    would let concurrent cache misses manufacture between-arm noise. This
    matters most here, where a generic message is deliberately shared across
    every rotation of its context.
    """
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        by_hash[spec["request_hash"]].append(spec)
    stored_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in stored:
        if row.get("schema_version") == SCHEMA_VERSION and row.get("request_hash"):
            stored_by_hash[row["request_hash"]].append(row)

    templates, missing = {}, []
    for request_hash, group in by_hash.items():
        candidates = stored_by_hash.get(request_hash, [])
        signatures = {_answer_signature(row) for row in candidates}
        if candidates and len(signatures) == 1:
            templates[request_hash] = candidates[0]
        else:
            missing.append((request_hash, group[0]))

    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        futures = {pool.submit(run_answer_call, client, spec, cfg): (h, spec)
                   for h, spec in missing}
        for future, (request_hash, _spec) in futures.items():
            template = future.result()
            match = next((row for row in stored_by_hash.get(request_hash, [])
                          if _answer_signature(row) == _answer_signature(template)), None)
            if match is not None:
                for field in ("judge_correct", "judge_parse_ok", "judge_raw"):
                    if field in match:
                        template[field] = match[field]
            templates[request_hash] = template

    current = {spec["answer_key"]: {**templates[spec["request_hash"]], **answer_record_base(spec)}
               for spec in specs}
    return [current[spec["answer_key"]] for spec in specs], len(missing)


# ---------------------------------------------------------------------------
# Utility matrices and rotation rows


def _utility_index(answers: list[dict]) -> dict[tuple[str, str], dict]:
    """(message_key, eval_qid) -> answer row, for the non-baseline answers."""
    index = {}
    for row in answers:
        if row.get("baseline"):
            continue
        index[(row["message_key"], row["eval_qid"])] = row
    return index


def ceiling_index(answers: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["context_id"], row["eval_qid"]): row for row in answers
            if row.get("baseline") == "direct_context"}


def build_matrix_rows(contexts, rotations, messages: list[dict], answers: list[dict],
                      policies: list[str]) -> list[dict]:
    """The full utility matrix M[a][b] for every (context, policy, budget).

    ``a`` is the conditioning question, ``b`` the evaluated one. For a
    rotation-invariant policy every row of the matrix comes from the same
    message; that is recorded rather than hidden, because it is exactly what
    makes those arms unspecialised by construction.
    """
    util = _utility_index(answers)
    ceil = ceiling_index(answers)
    # (context, policy, rotation) -> messages. Rotation-invariant policies are
    # filed under "-" and looked up once per rotation.
    by_slot: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for message in messages:
        by_slot[(message["context_id"], message["policy"],
                 message["rotation_id"])].append(message)

    rows = []
    for context in contexts:
        for rotation in rotations[context.context_id]:
            for policy in policies:
                shared = rotation_invariant(policy.replace(TRIM_SUFFIX, ""))
                slot = "-" if shared else rotation.rotation_id
                for message in by_slot.get((context.context_id, policy, slot), ()):
                    for q in context.questions:
                        answer = util.get((message["message_key"], q.qid))
                        if answer is None:
                            continue
                        ceiling = ceil.get((context.context_id, q.qid))
                        rows.append({
                            "context_id": context.context_id,
                            "dataset": context.dataset,
                            "policy": policy,
                            "budget_words": int(message["budget_words"]),
                            "rotation_id": rotation.rotation_id,
                            "cond_qid": rotation.current_qid,
                            "eval_qid": q.qid,
                            "is_diagonal": q.qid == rotation.current_qid,
                            "relation": (rd.UNLABELLED if q.qid == rotation.current_qid
                                         else rd.relation_label(context, rotation.current_qid,
                                                                q.qid)),
                            "message_key": message["message_key"],
                            "message_shared_across_rotations": shared,
                            "delivered_words": int(message["delivered_words"]),
                            "fill_ratio": float(message["fill_ratio"]),
                            "truncated": bool(message["truncated"]),
                            "attempts": int(message["attempts"]),
                            **{f"u_{m}": float(answer.get(m, 0.0)) for m in METRICS},
                            **{f"ceiling_{m}": (float(ceiling[m]) if ceiling else float("nan"))
                               for m in METRICS},
                            **{f"regret_{m}": ((float(ceiling[m]) - float(answer.get(m, 0.0)))
                                               if ceiling else float("nan"))
                               for m in METRICS},
                        })
    # A policy that produced messages but no matrix rows has been dropped by a
    # slot lookup, not by the data: that is how an arm silently disappears from
    # every table downstream while the run still reports success.
    produced = {row["policy"] for row in rows}
    dropped = sorted({m["policy"] for m in messages if m["policy"] in set(policies)} - produced)
    if dropped:
        raise AssertionError(
            f"{len(dropped)} policies have messages but no utility rows "
            f"(first: {dropped[0]}); check rotation_invariant() covers them")
    return rows


def build_rotation_rows(matrix_rows: list[dict]) -> list[dict]:
    """Collapse each matrix into one row per (context, rotation, policy, budget).

    ``now`` is the diagonal cell; ``future`` is the mean of the off-diagonal
    cells, so a context with many hidden questions does not outvote one with
    few when the context means are later bootstrapped.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in matrix_rows:
        grouped[(row["context_id"], row["rotation_id"], row["policy"],
                 row["budget_words"])].append(row)
    out = []
    for (context_id, rotation_id, policy, budget_words), cells in sorted(grouped.items()):
        diagonal = [c for c in cells if c["is_diagonal"]]
        off = [c for c in cells if not c["is_diagonal"]]
        if not diagonal or not off:
            continue
        row = {
            "context_id": context_id, "dataset": cells[0]["dataset"], "rotation_id": rotation_id,
            "cond_qid": diagonal[0]["cond_qid"], "policy": policy,
            "budget_words": budget_words, "n_hidden": len(off),
            "message_key": cells[0]["message_key"],
            "delivered_words": cells[0]["delivered_words"],
            "fill_ratio": cells[0]["fill_ratio"],
            "truncated": cells[0]["truncated"], "attempts": cells[0]["attempts"],
            "message_shared_across_rotations": cells[0]["message_shared_across_rotations"],
        }
        for metric in METRICS:
            row[f"now_{metric}"] = float(np.mean([c[f"u_{metric}"] for c in diagonal]))
            row[f"future_{metric}"] = float(np.mean([c[f"u_{metric}"] for c in off]))
            row[f"gap_{metric}"] = row[f"now_{metric}"] - row[f"future_{metric}"]
            ceil_now = [c[f"ceiling_{metric}"] for c in diagonal if finite(c[f"ceiling_{metric}"])]
            ceil_future = [c[f"ceiling_{metric}"] for c in off if finite(c[f"ceiling_{metric}"])]
            row[f"ceiling_now_{metric}"] = float(np.mean(ceil_now)) if ceil_now else float("nan")
            row[f"ceiling_future_{metric}"] = (float(np.mean(ceil_future)) if ceil_future
                                               else float("nan"))
            row[f"regret_now_{metric}"] = (row[f"ceiling_now_{metric}"] - row[f"now_{metric}"]
                                           if ceil_now else float("nan"))
            row[f"regret_future_{metric}"] = (row[f"ceiling_future_{metric}"]
                                              - row[f"future_{metric}"]
                                              if ceil_future else float("nan"))
            # Ceiling-restricted utility: on cells the source itself can answer,
            # U(D, q) is 1 and retained utility is just U(m, q). This is the
            # normalisation that has no near-zero denominator to guard.
            answerable_now = [c[f"u_{metric}"] for c in diagonal
                              if finite(c[f"ceiling_{metric}"]) and c[f"ceiling_{metric}"] > 0]
            answerable_future = [c[f"u_{metric}"] for c in off
                                 if finite(c[f"ceiling_{metric}"]) and c[f"ceiling_{metric}"] > 0]
            row[f"retained_now_{metric}"] = (float(np.mean(answerable_now)) if answerable_now
                                             else float("nan"))
            row[f"retained_future_{metric}"] = (float(np.mean(answerable_future))
                                                if answerable_future else float("nan"))
            row[f"n_ceiling_now_{metric}"] = len(answerable_now)
            row[f"n_ceiling_future_{metric}"] = len(answerable_future)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Aggregation


def context_means(rows: list[dict], policy: str, budget_words: int, field: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        if (row["policy"] == policy and int(row["budget_words"]) == int(budget_words)
                and finite(row.get(field))):
            grouped[row["context_id"]].append(float(row[field]))
    return {cid: float(np.mean(values)) for cid, values in grouped.items()}


MEASURES = (
    ("utility", "now"), ("utility", "future"), ("utility", "gap"),
    ("regret", "now"), ("regret", "future"),
    ("retained", "now"), ("retained", "future"),
)


def measure_field(kind: str, endpoint: str, metric: str) -> str:
    if kind == "utility":
        return f"{endpoint}_{metric}"
    return f"{kind}_{endpoint}_{metric}"


def analyse_metrics(rows: list[dict], policies: list[str], budgets_words: list[int],
                    cfg: dict) -> list[dict]:
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    out = []
    for policy in policies:
        for words in budgets_words:
            for kind, endpoint in MEASURES:
                for metric in METRICS:
                    field = measure_field(kind, endpoint, metric)
                    values = context_means(rows, policy, words, field)
                    if not values:
                        continue
                    mean, lo, hi = bootstrap_ci(
                        np.array(list(values.values()), dtype=float), boot, ci, seed=1010)
                    out.append({
                        "policy": policy, "budget_words": words, "kind": kind,
                        "endpoint": endpoint, "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                        "n_contexts": len(values),
                        "n_rotations": sum(1 for r in rows if r["policy"] == policy
                                           and int(r["budget_words"]) == words),
                    })
            # Channel cost, reported on the same context-clustered footing as
            # utility so the Pareto axes are commensurable.
            for field in ("delivered_words", "fill_ratio"):
                values = context_means(rows, policy, words, field)
                if not values:
                    continue
                mean, lo, hi = bootstrap_ci(
                    np.array(list(values.values()), dtype=float), boot, ci, seed=1011)
                out.append({"policy": policy, "budget_words": words, "kind": "channel",
                            "endpoint": "message", "metric": field, "mean": mean, "lo": lo,
                            "hi": hi, "n_contexts": len(values),
                            "n_rotations": sum(1 for r in rows if r["policy"] == policy
                                               and int(r["budget_words"]) == words)})
    return out


DEFAULT_COMPARISONS = (
    ("conditioned", "generic"),
    ("conditioned", "reusable"),
    ("reusable", "generic"),
    ("oracle", "conditioned"),
    ("oracle", "generic"),
    ("extractive_conditioned", "extractive_generic"),
)


def comparisons(cfg: dict) -> tuple[tuple[str, str], ...]:
    """Which paired contrasts to compute.

    ``analysis.comparisons`` in the config replaces the default list. An
    experiment that adds arms - Experiment 14 adds four elimination arms and a
    paraphrase arm to the same harness - names its own contrasts there rather
    than growing a constant every runner shares.
    """
    configured = (cfg.get("analysis") or {}).get("comparisons")
    if not configured:
        return DEFAULT_COMPARISONS
    out = []
    for pair in configured:
        if len(pair) != 2:
            raise ValueError(f"analysis.comparisons entries must be [treatment, control]: {pair}")
        out.append((str(pair[0]), str(pair[1])))
    return tuple(out)


def analyse_contrasts(rows: list[dict], policies: list[str], budgets_words: list[int],
                      cfg: dict) -> list[dict]:
    """Paired bootstrap over contexts, the unit the design actually repeats on.

    Rotations within a context share a source and are therefore not independent;
    resampling them would understate every interval.
    """
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    wanted = comparisons(cfg)
    pairs = [(a, b) for a, b in wanted if a in policies and b in policies]
    pairs += [(f"{a}{TRIM_SUFFIX}", f"{b}{TRIM_SUFFIX}") for a, b in wanted
              if f"{a}{TRIM_SUFFIX}" in policies and f"{b}{TRIM_SUFFIX}" in policies]
    out = []
    for treatment, control in pairs:
        for words in budgets_words:
            for kind, endpoint in MEASURES:
                for metric in METRICS:
                    field = measure_field(kind, endpoint, metric)
                    t = context_means(rows, treatment, words, field)
                    c = context_means(rows, control, words, field)
                    ids = sorted(set(t) & set(c))
                    if not ids:
                        continue
                    t_arr = np.array([t[i] for i in ids])
                    c_arr = np.array([c[i] for i in ids])
                    result = paired_bootstrap_delta(t_arr, c_arr, boot, ci, seed=1020)
                    if len(ids) < 2:
                        result.update({"lo": float("nan"), "hi": float("nan"),
                                       "p_value": float("nan")})
                    out.append({
                        "comparison": f"{treatment}_minus_{control}",
                        "treatment": treatment, "control": control, "budget_words": words,
                        "kind": kind, "endpoint": endpoint, "metric": metric,
                        **result, "cohens_dz": cohens_dz(t_arr - c_arr),
                    })
            # Difference in differences on the specialisation gap. This is the
            # single number the hypothesis predicts to be positive: conditioning
            # widens the distance between present and future utility.
            for metric in METRICS:
                t = context_means(rows, treatment, words, f"gap_{metric}")
                c = context_means(rows, control, words, f"gap_{metric}")
                ids = sorted(set(t) & set(c))
                if not ids:
                    continue
                d = np.array([t[i] - c[i] for i in ids])
                result = paired_bootstrap_delta(d, np.zeros_like(d), boot, ci, seed=1021)
                if len(ids) < 2:
                    result.update({"lo": float("nan"), "hi": float("nan"),
                                   "p_value": float("nan")})
                out.append({
                    "comparison": f"{treatment}_minus_{control}",
                    "treatment": treatment, "control": control, "budget_words": words,
                    "kind": "specialisation_did", "endpoint": "now_minus_future",
                    "metric": metric, **result, "cohens_dz": cohens_dz(d),
                })
    return out


def analyse_budget_interaction(rows: list[dict], policies: list[str],
                               budgets_words: list[int], cfg: dict) -> list[dict]:
    """Does scarcity sharpen the trade-off?

    Tests ``gap(smallest budget) - gap(largest budget)`` per policy, and the
    same difference for the conditioned-minus-generic contrast, paired within
    context. A positive value is the prediction ``B down => U_now - U_future up``.
    """
    if len(budgets_words) < 2:
        return []
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    low, high = min(budgets_words), max(budgets_words)
    out = []
    for policy in policies:
        for metric in METRICS:
            lo_vals = context_means(rows, policy, low, f"gap_{metric}")
            hi_vals = context_means(rows, policy, high, f"gap_{metric}")
            ids = sorted(set(lo_vals) & set(hi_vals))
            if len(ids) < 2:
                continue
            d = np.array([lo_vals[i] - hi_vals[i] for i in ids])
            result = paired_bootstrap_delta(d, np.zeros_like(d), boot, ci, seed=1030)
            out.append({"quantity": "gap_low_minus_high", "policy": policy, "metric": metric,
                        "low_budget_words": low, "high_budget_words": high,
                        **result, "cohens_dz": cohens_dz(d)})
    if "conditioned" in policies and "generic" in policies:
        for metric in METRICS:
            terms = {}
            for words in (low, high):
                cond = context_means(rows, "conditioned", words, f"gap_{metric}")
                gen = context_means(rows, "generic", words, f"gap_{metric}")
                terms[words] = {i: cond[i] - gen[i] for i in set(cond) & set(gen)}
            ids = sorted(set(terms[low]) & set(terms[high]))
            if len(ids) < 2:
                continue
            d = np.array([terms[low][i] - terms[high][i] for i in ids])
            result = paired_bootstrap_delta(d, np.zeros_like(d), boot, ci, seed=1031)
            out.append({"quantity": "conditioning_gap_low_minus_high",
                        "policy": "conditioned_minus_generic", "metric": metric,
                        "low_budget_words": low, "high_budget_words": high,
                        **result, "cohens_dz": cohens_dz(d)})
    return out


# ---------------------------------------------------------------------------
# Length audit


def analyse_length(rows: list[dict], messages: list[dict], policies: list[str],
                   budgets_words: list[int], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Was the channel actually equal? Reported before any utility claim.

    Returns (per-arm length table, paired length deltas). If a paired delta's
    interval excludes zero, the budget did not equalise that pair and the
    length-matched subsample below is the comparison that counts.
    """
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    table = []
    for policy in policies:
        for words in budgets_words:
            group = [m for m in messages if m["policy"] == policy
                     and int(m["budget_words"]) == words]
            if not group:
                continue
            delivered = np.array([int(m["delivered_words"]) for m in group], dtype=float)
            table.append({
                "policy": policy, "budget_words": words,
                "floor_words": int(group[0].get("floor_words", 0)),
                "n_messages": len(group),
                "delivered_words_mean": float(delivered.mean()),
                "delivered_words_sd": float(delivered.std(ddof=1)) if len(group) > 1 else 0.0,
                "delivered_words_min": float(delivered.min()),
                "delivered_words_max": float(delivered.max()),
                "fill_ratio_mean": float(np.mean([float(m["fill_ratio"]) for m in group])),
                "under_floor_rate": float(np.mean(
                    [bool(m.get("under_floor_delivered", False)) for m in group])),
                "over_budget_rate": float(np.mean(
                    [bool(m.get("over_budget_before_truncation", False)) for m in group])),
                "truncated_rate": float(np.mean([bool(m["truncated"]) for m in group])),
                "correction_rate": float(np.mean([int(m["attempts"]) > 1 for m in group])),
                "shrink_corrections_mean": float(np.mean(
                    [int(m.get("shrink_corrections", 0)) for m in group])),
                "expand_corrections_mean": float(np.mean(
                    [int(m.get("expand_corrections", 0)) for m in group])),
                # Guards the floor against the failure mode Experiment 9 found:
                # an expansion request answered by restating the same content.
                "repetition_rate_mean": float(np.mean(
                    [float(m.get("repetition_rate", 0.0)) for m in group])),
                "budget_tokens": int(group[0]["budget_tokens"]),
                "api_max_tokens": int(group[0]["api_max_tokens"]),
                "completion_tokens_mean": float(np.mean(
                    [int(m.get("completion_tokens", 0)) for m in group])),
                "delivered_estimated_tokens_mean": float(np.mean(
                    [int(m.get("delivered_estimated_tokens", 0)) for m in group])),
                "any_over_budget_delivered": bool(
                    any(int(m["delivered_words"]) > words for m in group)),
            })
    deltas = []
    pairs = [(a, b) for a, b in comparisons(cfg) if a in policies and b in policies]
    for treatment, control in pairs:
        for words in budgets_words:
            t = context_means(rows, treatment, words, "delivered_words")
            c = context_means(rows, control, words, "delivered_words")
            ids = sorted(set(t) & set(c))
            if len(ids) < 2:
                continue
            t_arr = np.array([t[i] for i in ids])
            c_arr = np.array([c[i] for i in ids])
            result = paired_bootstrap_delta(t_arr, c_arr, boot, ci, seed=1040)
            deltas.append({"comparison": f"{treatment}_minus_{control}", "budget_words": words,
                           "metric": "delivered_words", **result,
                           "cohens_dz": cohens_dz(t_arr - c_arr)})
    return table, deltas


def length_matched_contexts(rows: list[dict], policies: list[str], words: int,
                            tolerance: int) -> list[str]:
    """Contexts where every compared arm delivered within ``tolerance`` words."""
    per_policy = {p: context_means(rows, p, words, "delivered_words") for p in policies}
    common = set.intersection(*[set(v) for v in per_policy.values()]) if per_policy else set()
    keep = []
    for context_id in sorted(common):
        lengths = [per_policy[p][context_id] for p in policies]
        if max(lengths) - min(lengths) <= tolerance:
            keep.append(context_id)
    return keep


def analyse_length_matched(rows: list[dict], policies: list[str], budgets_words: list[int],
                           cfg: dict) -> list[dict]:
    """Headline contrasts restricted to cells the budget genuinely equalised."""
    core = [p for p in ("generic", "conditioned", "reusable") if p in policies]
    if len(core) < 2:
        return []
    tolerance = int(cfg["budget"]["match_tolerance_words"])
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    out = []
    for words in budgets_words:
        keep = set(length_matched_contexts(rows, core, words, tolerance))
        subset = [r for r in rows if r["context_id"] in keep]
        for treatment, control in [(a, b) for a, b in comparisons(cfg)
                                   if a in core and b in core]:
            for endpoint in ENDPOINTS:
                for metric in METRICS:
                    t = context_means(subset, treatment, words, f"{endpoint}_{metric}")
                    c = context_means(subset, control, words, f"{endpoint}_{metric}")
                    ids = sorted(set(t) & set(c))
                    if len(ids) < 2:
                        continue
                    t_arr = np.array([t[i] for i in ids])
                    c_arr = np.array([c[i] for i in ids])
                    result = paired_bootstrap_delta(t_arr, c_arr, boot, ci, seed=1050)
                    out.append({
                        "comparison": f"{treatment}_minus_{control}", "budget_words": words,
                        "endpoint": endpoint, "metric": metric,
                        "tolerance_words": tolerance, "n_contexts_matched": len(keep),
                        **result, "cohens_dz": cohens_dz(t_arr - c_arr),
                    })
    return out


# ---------------------------------------------------------------------------
# Pareto frontier


def pareto_points(metrics: list[dict], policies: list[str], budgets_words: list[int],
                  metric: str, epsilon: float) -> list[dict]:
    """One point per (policy, budget); mark the empirically nondominated set.

    Cost is mean delivered words -- what the channel actually carried, not what
    was requested. A configuration is dominated when another is at least as good
    on present utility, at least as good on future utility, and no more
    expensive, with at least one strict improvement.

    Trimmed arms are excluded: their length is set by looking at what the other
    arms wrote, so they are a length control rather than a policy anyone could
    deploy, and letting them onto the frontier would compare a deployable
    configuration against a retrospective one.
    """
    epsilon = float(epsilon)
    if not finite(epsilon) or epsilon < 0:
        raise ValueError("Pareto epsilon must be finite and non-negative")

    # ``metrics.csv`` should contain one aggregate row per key, but analysis
    # should not depend on input order if a concatenated/recovered file contains
    # duplicates. Average all finite duplicates and ignore non-finite values. If
    # every duplicate is invalid, the configuration is incomplete and is
    # omitted below rather than allowing NaN/Inf into dominance comparisons.
    indexed: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(list)
    for row in metrics:
        if not finite(row.get("mean")):
            continue
        try:
            key = (str(row["policy"]), int(row["budget_words"]), str(row["kind"]),
                   str(row["endpoint"]), str(row["metric"]))
        except (KeyError, TypeError, ValueError):
            continue
        indexed[key].append(float(row["mean"]))

    policies = list(dict.fromkeys(
        p for p in policies if isinstance(p, str) and TRIM_SUFFIX not in p))
    budgets_words = list(dict.fromkeys(int(w) for w in budgets_words))
    points = []
    for policy in policies:
        for words in budgets_words:
            def look(kind, endpoint, met=metric):
                values = indexed.get((policy, words, kind, endpoint, met), [])
                return float(np.mean(values)) if values else None
            u_now, u_future = look("utility", "now"), look("utility", "future")
            cost = look("channel", "message", "delivered_words")
            if u_now is None or u_future is None or cost is None:
                continue
            points.append({"policy": policy, "budget_words": words, "metric": metric,
                           "u_now": float(u_now), "u_future": float(u_future),
                           "cost_words": float(cost)})
    for point in points:
        dominators = []
        for other in points:
            if other is point:
                continue
            better_equal = (other["u_now"] >= point["u_now"] - epsilon
                            and other["u_future"] >= point["u_future"] - epsilon
                            and other["cost_words"] <= point["cost_words"] + epsilon)
            strictly = (other["u_now"] > point["u_now"] + epsilon
                        or other["u_future"] > point["u_future"] + epsilon
                        or other["cost_words"] < point["cost_words"] - epsilon)
            if better_equal and strictly:
                dominators.append(f"{other['policy']}@{other['budget_words']}w")
        point["dominated"] = bool(dominators)
        point["dominated_by"] = ";".join(sorted(dominators))

    # Also mark the two-dimensional nondominated *observations* within each
    # requested budget. This is useful for a faceted plot, but it is a discrete
    # set of categorical policies -- not evidence that intermediate utility
    # combinations are achievable.
    for words in budgets_words:
        cell = [p for p in points if p["budget_words"] == words]
        for point in cell:
            beaten = any(
                other is not point
                and other["u_now"] >= point["u_now"] - epsilon
                and other["u_future"] >= point["u_future"] - epsilon
                and (other["u_now"] > point["u_now"] + epsilon
                     or other["u_future"] > point["u_future"] + epsilon)
                for other in cell)
            point["dominated_within_budget"] = bool(beaten)
    return points


def budget_frontier(points: list[dict], words: int) -> list[dict]:
    """Discrete within-budget nondominated observations, ordered left to right.

    The returned policies are categories, not samples from a continuous control
    variable. Callers must not connect them as an achievable interpolation.
    """
    cell = [p for p in points if p["budget_words"] == words
            and not p.get("dominated_within_budget", True)]
    return sorted(cell, key=lambda p: (p["u_now"], -p["u_future"]))


# ---------------------------------------------------------------------------
# Relation distance


def analyse_relations(matrix_rows: list[dict], policies: list[str], budgets_words: list[int],
                      cfg: dict) -> tuple[list[dict], list[dict]]:
    """Regret as a function of designed distance from the conditioning query.

    Only runs on the relation corpus, where the label is part of the
    construction. The monotonicity test is the paired difference between the
    furthest and nearest tier, which is the claim worth stating.
    """
    labelled = [r for r in matrix_rows
                if not r["is_diagonal"] and r["relation"] in rd.RELATION_ORDER]
    if not labelled:
        return [], []
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])

    def per_context(policy, words, relation, field):
        grouped = defaultdict(list)
        for row in labelled:
            if (row["policy"] == policy and int(row["budget_words"]) == words
                    and row["relation"] == relation and finite(row.get(field))):
                grouped[row["context_id"]].append(float(row[field]))
        return {cid: float(np.mean(v)) for cid, v in grouped.items()}

    table, tests = [], []
    for policy in policies:
        for words in budgets_words:
            for relation in rd.RELATION_ORDER:
                for metric in METRICS:
                    for kind, field in (("utility", f"u_{metric}"),
                                        ("regret", f"regret_{metric}")):
                        values = per_context(policy, words, relation, field)
                        if not values:
                            continue
                        mean, lo, hi = bootstrap_ci(
                            np.array(list(values.values())), boot, ci, seed=1060)
                        table.append({"policy": policy, "budget_words": words,
                                      "relation": relation, "kind": kind, "metric": metric,
                                      "mean": mean, "lo": lo, "hi": hi,
                                      "n_contexts": len(values),
                                      "n_cells": sum(1 for r in labelled
                                                     if r["policy"] == policy
                                                     and int(r["budget_words"]) == words
                                                     and r["relation"] == relation)})
            for metric in METRICS:
                far = per_context(policy, words, rd.ORTHOGONAL, f"regret_{metric}")
                near = per_context(policy, words, rd.PARAPHRASE, f"regret_{metric}")
                ids = sorted(set(far) & set(near))
                if len(ids) < 2:
                    continue
                d = np.array([far[i] - near[i] for i in ids])
                result = paired_bootstrap_delta(d, np.zeros_like(d), boot, ci, seed=1061)
                tests.append({"policy": policy, "budget_words": words, "metric": metric,
                              "quantity": "regret_orthogonal_minus_paraphrase",
                              **result, "cohens_dz": cohens_dz(d)})
    return table, tests


def analyse_normalised(rows: list[dict], policies: list[str], budgets_words: list[int],
                       cfg: dict) -> list[dict]:
    """The ratio form of retained utility, with the near-zero denominator guarded.

    Per context: ``mean U(m, q) / mean U(D, q)`` over the relevant cells. A
    context whose own direct-context ceiling falls below
    ``analysis.min_ceiling_for_ratio`` is excluded from the ratio -- dividing by
    a ceiling of 0.05 turns noise into a number -- and the count of exclusions is
    reported so the exclusion is visible rather than silent. The
    ceiling-restricted ``retained_*`` fields in ``metrics.csv`` are the primary
    normalisation; this table is the arithmetic version of the same idea.
    """
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    floor = float(cfg["analysis"]["min_ceiling_for_ratio"])
    out = []
    for policy in policies:
        for words in budgets_words:
            for endpoint in ENDPOINTS:
                for metric in METRICS:
                    util = context_means(rows, policy, words, f"{endpoint}_{metric}")
                    ceil = context_means(rows, policy, words,
                                         f"ceiling_{endpoint}_{metric}")
                    ids = sorted(set(util) & set(ceil))
                    kept = [i for i in ids if ceil[i] >= floor]
                    if not kept:
                        continue
                    ratios = np.array([util[i] / ceil[i] for i in kept], dtype=float)
                    mean, lo, hi = bootstrap_ci(ratios, boot, ci, seed=1080)
                    out.append({
                        "policy": policy, "budget_words": words, "endpoint": endpoint,
                        "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                        "n_contexts": len(kept),
                        "n_excluded_low_ceiling": len(ids) - len(kept),
                        "min_ceiling_for_ratio": floor,
                    })
    return out


def analyse_baselines(answers: list[dict], cfg: dict) -> list[dict]:
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    out = []
    for baseline in ("direct_context", "closed_book"):
        rows = [r for r in answers if r.get("baseline") == baseline]
        if not rows:
            continue
        for metric in METRICS:
            grouped = defaultdict(list)
            for row in rows:
                if finite(row.get(metric)):
                    grouped[row["context_id"]].append(float(row[metric]))
            values = {cid: float(np.mean(v)) for cid, v in grouped.items()}
            if not values:
                continue
            mean, lo, hi = bootstrap_ci(np.array(list(values.values())), boot, ci, seed=1070)
            out.append({"baseline": baseline, "metric": metric, "mean": mean, "lo": lo,
                        "hi": hi, "n_contexts": len(values), "n_answers": len(rows)})
    return out


# ---------------------------------------------------------------------------
# Plots

POLICY_COLOURS = {
    "generic": "#1b9e77",
    "conditioned": "#d95f02",
    "reusable": "#7570b3",
    "oracle": "#e7298a",
    "extractive_generic": "#66a61e",
    "extractive_conditioned": "#a6761d",
    # Experiment 14. Family = hue (summary green/orange above, LM blue,
    # non-LLM purple), conditioning = saturation within the family.
    "paraphrase": "#8c564b",
    "lm_generic": "#6baed6",
    "lm_conditioned": "#08519c",
    "nonllm_generic": "#bcbddc",
    "nonllm_conditioned": "#54278f",
    "random_selection": "#999999",
    "passthrough": "#000000",
}


def _colour(policy: str) -> str:
    return POLICY_COLOURS.get(policy.replace(TRIM_SUFFIX, ""), "#666666")


def _deployable(rows: list[dict]) -> list[dict]:
    """Drop the trimmed sensitivity arms from a figure.

    They exist at one budget only, so a grouped bar or a budget line built from
    them is misaligned rather than informative, and their length is chosen by
    looking at the other arms. They get their own panel in ``budget_control``
    and their own rows in every table.
    """
    return [r for r in rows if TRIM_SUFFIX not in str(r.get("policy", ""))]


def plot_pareto(points: list[dict], output: Path, metric_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points = _deployable(points)
    if not points:
        return
    sizes = sorted({p["budget_words"] for p in points})
    policies = sorted({p["policy"] for p in points})
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")
    policy_marker = {policy: markers[i % len(markers)]
                     for i, policy in enumerate(policies)}
    ncols = min(2, len(sizes))
    nrows = int(math.ceil(len(sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.7 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for ax, words in zip(axes.flat, sizes):
        cell = [p for p in points if p["budget_words"] == words]
        ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":", linewidth=1.0,
                zorder=0)
        for row in sorted(cell, key=lambda p: (p.get("dominated_within_budget", True),
                                                p["policy"])):
            on_frontier = not row.get("dominated_within_budget", True)
            ax.scatter(
                row["u_now"], row["u_future"],
                marker=policy_marker[row["policy"]],
                s=78 if on_frontier else 60,
                facecolor=_colour(row["policy"]) if on_frontier else "#d7d7d7",
                edgecolor="#222222" if on_frontier else "#aaaaaa",
                linewidth=1.45 if on_frontier else .8,
                alpha=1.0 if on_frontier else .60,
                zorder=3 if on_frontier else 2,
            )
        ax.set(title=f"Requested cap: {words} words", xlim=(0, 1), ylim=(0, 1),
               xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.22)

    for ax in list(axes.flat)[len(sizes):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(f"$U_{{now}}$ - present-query {metric_label}")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(f"$U_{{future}}$ - future-query {metric_label}")

    handles = [
        Line2D([], [], marker=policy_marker[p], linestyle="none",
               markerfacecolor=_colour(p), markeredgecolor="#555555",
               markersize=7, label=p)
        for p in policies
    ]
    handles.extend([
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#777777",
               markeredgecolor="#222222", markeredgewidth=1.45, markersize=8,
               label="within-cap nondominated"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#d7d7d7",
               markeredgecolor="#aaaaaa", markersize=7, alpha=.60,
               label="within-cap dominated"),
    ])
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=min(4, len(handles)),
               frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle("Observed present-versus-future utility by requested communication cap\n"
                 "Markers are evaluated policies; no interpolation is implied",
                 fontsize=11)
    fig.tight_layout(rect=(0, .09, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_utility_vs_budget(metrics: list[dict], baselines: list[dict], output: Path,
                           metric: str, metric_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    metrics = _deployable(metrics)
    policies = sorted({r["policy"] for r in metrics if r["kind"] == "utility"})
    if not policies:
        return
    budgets = sorted({r["budget_words"] for r in metrics})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=False)
    panels = [("utility", "now", "Present query $U_{now}$"),
              ("utility", "future", "Future queries $U_{future}$"),
              ("utility", "gap", "Specialisation gap $U_{now}-U_{future}$")]
    for ax, (kind, endpoint, title) in zip(axes, panels):
        for policy in policies:
            rows = sorted([r for r in metrics if r["policy"] == policy and r["kind"] == kind
                           and r["endpoint"] == endpoint and r["metric"] == metric],
                          key=lambda r: r["budget_words"])
            if not rows:
                continue
            x = [r["budget_words"] for r in rows]
            y = [r["mean"] for r in rows]
            err = [[r["mean"] - r["lo"] for r in rows], [r["hi"] - r["mean"] for r in rows]]
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, color=_colour(policy),
                        linewidth=1.6, label=policy)
        ceiling = next((r["mean"] for r in baselines if r["baseline"] == "direct_context"
                        and r["metric"] == metric), None)
        if ceiling is not None and endpoint != "gap":
            ax.axhline(ceiling, color="#333333", linestyle=":", linewidth=1.2,
                       label="direct-context ceiling")
        if endpoint == "gap":
            ax.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
        ax.set(title=title, xlabel="Requested word cap, $B$ (words)")
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.grid(alpha=.25)
    axes[0].set_ylabel(metric_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Utility against communication budget", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_regret(metrics: list[dict], output: Path, metric: str, metric_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in _deployable(metrics)
            if r["kind"] == "regret" and r["metric"] == metric]
    if not rows:
        return
    policies = sorted({r["policy"] for r in rows})
    budgets = sorted({r["budget_words"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    width = 0.8 / max(1, len(policies))
    for ax, endpoint, title in zip(axes, ENDPOINTS,
                                   ("$R_{now}$ on the present query",
                                    "$R_{future}$ on hidden future queries")):
        for i, policy in enumerate(policies):
            sel = sorted([r for r in rows if r["policy"] == policy and r["endpoint"] == endpoint],
                         key=lambda r: r["budget_words"])
            if not sel:
                continue
            x = np.array([budgets.index(r["budget_words"]) for r in sel], dtype=float)
            x = x + (i - (len(policies) - 1) / 2) * width
            y = [r["mean"] for r in sel]
            err = [[r["mean"] - r["lo"] for r in sel], [r["hi"] - r["mean"] for r in sel]]
            ax.bar(x, y, width=width * 0.92, color=_colour(policy), label=policy, alpha=.85)
            ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#222222", capsize=2, linewidth=.9)
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.set(title=title, xticks=range(len(budgets)),
               xlabel="Communication budget (words)")
        ax.set_xticklabels([str(b) for b in budgets])
        ax.grid(alpha=.2, axis="y")
    axes[0].set_ylabel(f"Regret vs direct context ({metric_label})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Communication regret $R(q) = U(D,q) - U(m,q)$", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_budget_control(length_table: list[dict], metrics: list[dict], output: Path,
                        metric: str, metric_label: str) -> None:
    """The evidence behind every utility claim: was the channel actually equal?

    Left: realised length against the contracted band, per policy and budget.
    Right: the trimmed sensitivity arms, where length is identical within a
    context by construction, beside the arms they were derived from.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not length_table:
        return
    budgets = sorted({r["budget_words"] for r in length_table})
    policies = sorted({r["policy"] for r in length_table if TRIM_SUFFIX not in r["policy"]})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    for i, words in enumerate(budgets):
        floor = next((r["floor_words"] for r in length_table if r["budget_words"] == words), 0)
        ax.axhspan(floor / words, 1.0, xmin=i / len(budgets), xmax=(i + 1) / len(budgets),
                   color="#cccccc", alpha=.35, zorder=0)
    width = 0.8 / max(1, len(policies))
    for j, policy in enumerate(policies):
        rows = sorted([r for r in length_table if r["policy"] == policy],
                      key=lambda r: r["budget_words"])
        if not rows:
            continue
        x = np.array([budgets.index(r["budget_words"]) for r in rows], dtype=float)
        x = x + (j - (len(policies) - 1) / 2) * width
        y = [r["delivered_words_mean"] / r["budget_words"] for r in rows]
        err = [r["delivered_words_sd"] / r["budget_words"] for r in rows]
        ax.errorbar(x, y, yerr=err, fmt="o", capsize=3, color=_colour(policy),
                    markersize=5, label=policy)
    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.set(title="Realised fill against the contracted band",
           ylabel="delivered words / cap", xlabel="Communication budget (words)",
           xticks=range(len(budgets)), ylim=(0, 1.12))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.grid(alpha=.2, axis="y")
    ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=.9)

    ax = axes[1]
    trim_rows = [r for r in metrics if TRIM_SUFFIX in r["policy"] and r["kind"] == "utility"
                 and r["metric"] == metric and r["endpoint"] in ENDPOINTS]
    if trim_rows:
        from matplotlib.lines import Line2D
        base_policies = sorted({r["policy"].replace(TRIM_SUFFIX, "") for r in trim_rows})
        words = sorted({r["budget_words"] for r in trim_rows})[0]
        for i, policy in enumerate(base_policies):
            ax.axvspan(i - 0.46, i + 0.46, color=_colour(policy), alpha=.10, zorder=0)
            for offset, endpoint, marker in ((-0.16, "now", "o"), (0.16, "future", "s")):
                for arm, dx, face in (("", -0.06, _colour(policy)), (TRIM_SUFFIX, 0.06, "white")):
                    row = lookup(metrics, policy=policy + arm, budget_words=words,
                                 kind="utility", endpoint=endpoint, metric=metric)
                    if row is None:
                        continue
                    ax.errorbar([i + offset + dx], [row["mean"]],
                                yerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                                fmt=marker, capsize=2, markersize=7, linestyle="none",
                                markerfacecolor=face, markeredgecolor=_colour(policy),
                                ecolor=_colour(policy), markeredgewidth=1.4)
        ax.set(title=f"Trimmed to one length per context ({words}-word budget)",
               ylabel=metric_label, xticks=range(len(base_policies)), ylim=(-0.03, 1.03))
        ax.set_xticklabels(base_policies, fontsize=8, rotation=12)
        ax.grid(alpha=.2, axis="y")
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", color="#444444", label="$U_{now}$"),
            Line2D([], [], marker="s", linestyle="none", color="#444444", label="$U_{future}$"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="#444444",
                   markeredgecolor="#444444", label="as delivered"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor="#444444", label="trimmed"),
        ], fontsize=7, loc="upper right", ncol=2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no trimmed arm in this run", ha="center", va="center",
                fontsize=9, color="#777777")
    fig.suptitle("Was the channel actually equal?", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_relation_distance(table: list[dict], output: Path, metric: str,
                           metric_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in table if r["kind"] == "regret" and r["metric"] == metric]
    if not rows:
        return
    budgets = sorted({r["budget_words"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.6 * len(budgets), 4.0), sharey=True,
                             squeeze=False)
    xs = list(range(len(rd.RELATION_ORDER)))
    for ax, words in zip(axes[0], budgets):
        for policy in policies:
            sel = [next((r for r in rows if r["policy"] == policy
                         and r["budget_words"] == words and r["relation"] == rel), None)
                   for rel in rd.RELATION_ORDER]
            if any(s is None for s in sel):
                continue
            y = [s["mean"] for s in sel]
            err = [[s["mean"] - s["lo"] for s in sel], [s["hi"] - s["mean"] for s in sel]]
            ax.errorbar(xs, y, yerr=err, marker="o", capsize=3, color=_colour(policy),
                        label=policy)
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.set(title=f"{words}-word budget", xticks=xs)
        ax.set_xticklabels([r.replace("_", "\n") for r in rd.RELATION_ORDER], fontsize=8)
        ax.grid(alpha=.25)
    axes[0][0].set_ylabel(f"Future-query regret ({metric_label})")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Regret by designed distance from the conditioning query", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if not finite(value) else f"{float(value):.{digits}f}"


def _ci(row: dict) -> str:
    if row is None:
        return "n/a"
    return f"{_fmt(row.get('delta', row.get('mean')))} [{_fmt(row.get('lo'))}, {_fmt(row.get('hi'))}]"


def lookup(rows: list[dict], **where):
    for row in rows:
        if all(row.get(k) == v for k, v in where.items()):
            return row
    return None


def write_report(path: Path, *, cfg, dataset, contexts, policies, budgets_words, metric,
                 metrics, contrasts, length_table, length_deltas, matched, interaction,
                 pareto, baselines, normalised, relation_table, relation_tests,
                 rotation_rows, source_manifest) -> None:
    metric_label = {"judge_correct": "LLM-judge accuracy", "em": "exact match",
                    "f1": "token F1"}[metric]
    lines: list[str] = []
    out = lines.append

    out("# Experiment 10 - communication regret under a hard communication budget\n")
    out(f"Corpus `{dataset}`: {len(contexts)} contexts, "
        f"{sum(len(c.questions) for c in contexts)} questions, "
        f"{len(rotation_rows)} rotation x policy x budget cells. "
        f"Sender and answerer `{cfg['model']['id']}`; judge "
        f"`{cfg.get('judge', {}).get('model_id')}`. Primary utility: {metric_label}.\n")
    out("`U_now` is the diagonal of each context's utility matrix (the message answered on the "
        "question it was conditioned on); `U_future` is the mean of the off-diagonal (the same "
        "message answered on questions the sender never saw). Every question rotates through "
        "both roles, so question difficulty cannot produce the contrast. All intervals are "
        "95% percentile bootstrap over contexts, the unit the design repeats on.\n")

    out("## 1. Was the channel actually equal?\n")
    out("The hypothesis is only testable if no policy simply wrote more. Delivered length is "
        "capped deterministically, so the question is whether the arms *fill* the budget "
        "equally.\n")
    out("| policy | band (words) | delivered (mean +- sd) | fill | under floor | over cap "
        "pre-truncation | truncated | corrections | repetition |")
    out("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(length_table, key=lambda r: (r["budget_words"], r["policy"])):
        out(f"| `{row['policy']}` | {row['floor_words']}-{row['budget_words']} | "
            f"{_fmt(row['delivered_words_mean'], 1)} +- {_fmt(row['delivered_words_sd'], 1)} | "
            f"{_fmt(row['fill_ratio_mean'], 2)} | {_fmt(row['under_floor_rate'], 2)} | "
            f"{_fmt(row['over_budget_rate'], 2)} | {_fmt(row['truncated_rate'], 2)} | "
            f"{_fmt(row['correction_rate'], 2)} | {_fmt(row['repetition_rate_mean'], 3)} |")
    violations = [r for r in length_table if r["any_over_budget_delivered"]]
    out("")
    out(f"Delivered messages over the cap after truncation: **{len(violations)}** arm(s) "
        "(must be zero; truncation is unconditional).\n")
    out("Paired length differences between arms (context-clustered):\n")
    out("| comparison | budget | delta words [95% CI] | p |")
    out("|---|---|---|---|")
    for row in sorted(length_deltas, key=lambda r: (r["budget_words"], r["comparison"])):
        out(f"| {row['comparison']} | {row['budget_words']} | {_ci(row)} | "
            f"{_fmt(row['p_value'], 4)} |")
    out("")

    out("## 2. Present and future utility\n")
    out("| policy | budget | U_now [CI] | U_future [CI] | gap [CI] |")
    out("|---|---|---|---|---|")
    for policy in policies:
        for words in budgets_words:
            now = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="now", metric=metric)
            fut = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="future", metric=metric)
            gap = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="gap", metric=metric)
            if not now:
                continue
            out(f"| `{policy}` | {words} | {_ci(now)} | {_ci(fut)} | {_ci(gap)} |")
    ceiling = lookup(baselines, baseline="direct_context", metric=metric)
    closed = lookup(baselines, baseline="closed_book", metric=metric)
    out("")
    out(f"Direct-context ceiling `U(D, q)`: {_ci(ceiling)}. "
        f"Closed-book baseline (leakage audit): {_ci(closed)}.\n")

    out("## 3. The hypothesis\n")
    out("A supporting result needs `delta U_now > 0` **and** `delta U_future < 0` for "
        "`conditioned` against a generic or reusable baseline, strengthening as the budget "
        "shrinks.\n")
    out("| comparison | budget | delta U_now [CI] (p, dz) | delta U_future [CI] (p, dz) | "
        "gap DiD [CI] (p) |")
    out("|---|---|---|---|---|")
    for treatment, control in DEFAULT_COMPARISONS:
        if treatment not in policies or control not in policies:
            continue
        name = f"{treatment}_minus_{control}"
        for words in budgets_words:
            now = lookup(contrasts, comparison=name, budget_words=words, kind="utility",
                         endpoint="now", metric=metric)
            fut = lookup(contrasts, comparison=name, budget_words=words, kind="utility",
                         endpoint="future", metric=metric)
            did = lookup(contrasts, comparison=name, budget_words=words,
                         kind="specialisation_did", metric=metric)
            if not now:
                continue
            out(f"| {name} | {words} | {_ci(now)} ({_fmt(now['p_value'], 3)}, "
                f"{_fmt(now['cohens_dz'], 2)}) | {_ci(fut)} ({_fmt(fut['p_value'], 3)}, "
                f"{_fmt(fut['cohens_dz'], 2)}) | {_ci(did)} ({_fmt(did['p_value'], 3)}) |")
    out("")

    out("### Does scarcity sharpen the trade-off?\n")
    out("| quantity | policy | gap(low) - gap(high) [CI] | p | dz |")
    out("|---|---|---|---|---|")
    for row in interaction:
        if row["metric"] != metric:
            continue
        out(f"| {row['quantity']} | `{row['policy']}` | {_ci(row)} | "
            f"{_fmt(row['p_value'], 3)} | {_fmt(row['cohens_dz'], 2)} |")
    out("")

    out("## 4. Communication regret against the source\n")
    out("`R(q) = U(D, q) - U(m, q)`, the accuracy the handoff costs relative to reading the "
        "source directly.\n")
    out("| policy | budget | R_now [CI] | R_future [CI] | retained U_future on answerable "
        "cells [CI] |")
    out("|---|---|---|---|---|")
    for policy in policies:
        for words in budgets_words:
            r_now = lookup(metrics, policy=policy, budget_words=words, kind="regret",
                           endpoint="now", metric=metric)
            r_fut = lookup(metrics, policy=policy, budget_words=words, kind="regret",
                           endpoint="future", metric=metric)
            retained = lookup(metrics, policy=policy, budget_words=words, kind="retained",
                              endpoint="future", metric=metric)
            if not r_now:
                continue
            out(f"| `{policy}` | {words} | {_ci(r_now)} | {_ci(r_fut)} | {_ci(retained)} |")
    out("")
    out("`retained` is the ceiling-restricted normalisation: it averages `U(m, q)` only over "
        "cells the source itself answers, where `U(D, q) = 1` and the ratio has no near-zero "
        "denominator to guard.\n")
    excluded = sum(int(r["n_excluded_low_ceiling"]) for r in normalised
                   if r["metric"] == metric and r["endpoint"] == "future")
    out(f"The arithmetic ratio `mean U(m,q) / mean U(D,q)` is in `normalised_utility.csv`. "
        f"Contexts whose own ceiling falls below "
        f"{cfg['analysis']['min_ceiling_for_ratio']} are excluded from it, because dividing by "
        f"a near-zero ceiling turns noise into a number: {excluded} context-cell(s) were "
        f"excluded on the future endpoint across all arms.\n")
    out("| policy | budget | retained U_now [CI] | retained U_future [CI] | n contexts | "
        "excluded (low ceiling) |")
    out("|---|---|---|---|---|---|")
    for policy in policies:
        for words in budgets_words:
            now = lookup(normalised, policy=policy, budget_words=words, endpoint="now",
                         metric=metric)
            fut = lookup(normalised, policy=policy, budget_words=words, endpoint="future",
                         metric=metric)
            if not now:
                continue
            out(f"| `{policy}` | {words} | {_ci(now)} | {_ci(fut)} | {now['n_contexts']} | "
                f"{now['n_excluded_low_ceiling']} |")
    out("")

    out("### Secondary metrics\n")
    out("EM and token F1 are computed for every answer and reported here in full, never "
        "replaced by the judge. F1 penalises answers that are correct but differently worded, "
        "which is exactly what compression produces, so a disagreement between the three is "
        "information rather than noise.\n")
    for secondary in [m for m in METRICS if m != metric]:
        label = {"em": "Exact match", "f1": "Token F1", "judge_correct": "LLM judge"}[secondary]
        out(f"**{label}** &mdash; `conditioned` minus `generic`:\n")
        out("| endpoint | " + " | ".join(f"{w} w" for w in budgets_words) + " |")
        out("|---|" + "---|" * len(budgets_words))
        for endpoint in ENDPOINTS:
            cells = []
            for words in budgets_words:
                row = lookup(contrasts, comparison="conditioned_minus_generic",
                             budget_words=words, kind="utility", endpoint=endpoint,
                             metric=secondary)
                cells.append("n/a" if row is None else
                             f"{_fmt(row['delta'])} [{_fmt(row['lo'])}, {_fmt(row['hi'])}]")
            out(f"| dU_{endpoint} | " + " | ".join(cells) + " |")
        out("")

    out("## 5. Pareto frontier\n")
    out("Cost is mean delivered words. A configuration is dominated when another is at least "
        "as good on both utilities and no more expensive, with one strict improvement.\n")
    out("| policy | requested cap | U_now | U_future | delivered cost (words) | "
        "3D nondominated | within-cap 2D nondominated | dominated by |")
    out("|---|---|---|---|---|---|---|---|")
    for point in sorted(pareto, key=lambda p: (p["policy"], p["budget_words"])):
        out(f"| `{point['policy']}` | {point['budget_words']} | {_fmt(point['u_now'])} | "
            f"{_fmt(point['u_future'])} | {_fmt(point['cost_words'], 1)} | "
            f"{'' if point['dominated'] else 'yes'} | "
            f"{'' if point.get('dominated_within_budget', True) else 'yes'} | "
            f"{point['dominated_by'] or '-'} |")
    out("")
    frontier = [p for p in sorted(pareto, key=lambda p: (-p["u_now"], -p["u_future"]))
                if not p["dominated"]]
    out(f"**Nondominated set ({len(frontier)} of {len(pareto)} configurations):** "
        + ", ".join(f"`{p['policy']}`@{p['budget_words']}w" for p in frontier) + ".\n")
    out("Cost enters that test, so the set above is not a curve: a configuration can survive "
        "purely by being cheaper than everything that beats it on utility. The figure facets "
        "the observations by requested cap and outlines the two-dimensional nondominated "
        "policies within each facet. Markers are deliberately not joined: policy is "
        "categorical, so a line or staircase would imply unevaluated intermediate choices. "
        "The discrete within-cap sets are listed here from best present-query utility to best "
        "future-query utility:\n")
    for words in budgets_words:
        front = budget_frontier(pareto, words)
        if not front:
            continue
        out(f"* **{words} words** -- "
            + "  ".join(f"`{p['policy']}` ({_fmt(p['u_now'], 2)}, {_fmt(p['u_future'], 2)})"
                        for p in reversed(front)))
    out("")

    if matched:
        out("## 6. Length-matched subsample\n")
        out(f"Contexts where `generic`, `conditioned` and `reusable` delivered within "
            f"{cfg['budget']['match_tolerance_words']} words of each other. This is the "
            "comparison that survives if section 1 shows the arms filled the budget "
            "differently.\n")
        out("| comparison | budget | endpoint | delta [CI] | p | n contexts |")
        out("|---|---|---|---|---|---|")
        for row in sorted(matched, key=lambda r: (r["budget_words"], r["comparison"],
                                                  r["endpoint"])):
            if row["metric"] != metric:
                continue
            out(f"| {row['comparison']} | {row['budget_words']} | {row['endpoint']} | "
                f"{_ci(row)} | {_fmt(row['p_value'], 3)} | {row['n_contexts_matched']} |")
        out("")

    if relation_table:
        out("## 7. Regret by distance from the conditioning query\n")
        out("Distances are read off the corpus design, not estimated from embeddings.\n")
        out("| policy | budget | " + " | ".join(rd.RELATION_ORDER) + " |")
        out("|---|---|" + "---|" * len(rd.RELATION_ORDER))
        for policy in policies:
            for words in budgets_words:
                cells = [lookup(relation_table, policy=policy, budget_words=words,
                                relation=rel, kind="regret", metric=metric)
                         for rel in rd.RELATION_ORDER]
                if any(c is None for c in cells):
                    continue
                out(f"| `{policy}` | {words} | "
                    + " | ".join(_ci(c) for c in cells) + " |")
        out("")
        out("| policy | budget | R(orthogonal) - R(paraphrase) [CI] | p | dz |")
        out("|---|---|---|---|---|")
        for row in relation_tests:
            if row["metric"] != metric:
                continue
            out(f"| `{row['policy']}` | {row['budget_words']} | {_ci(row)} | "
                f"{_fmt(row['p_value'], 3)} | {_fmt(row['cohens_dz'], 2)} |")
        out("")

    out("## Verdict\n")
    out(verdict_text(contrasts, interaction, policies, budgets_words, metric))
    out("")
    out("## Provenance\n")
    out("```json")
    out(json.dumps({"schema_version": SCHEMA_VERSION, "dataset": dataset,
                    "source_manifest": source_manifest,
                    "policies": policies, "budgets_words": budgets_words,
                    "primary_metric": metric}, ensure_ascii=False, indent=2))
    out("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verdict_text(contrasts, interaction, policies, budgets_words, metric) -> str:
    """State what the numbers support, including when that is 'nothing'."""
    if "conditioned" not in policies or "generic" not in policies:
        return "The conditioned-versus-generic contrast was not run in this configuration."
    lines = []
    supported = []
    for words in budgets_words:
        now = lookup(contrasts, comparison="conditioned_minus_generic", budget_words=words,
                     kind="utility", endpoint="now", metric=metric)
        fut = lookup(contrasts, comparison="conditioned_minus_generic", budget_words=words,
                     kind="utility", endpoint="future", metric=metric)
        if not now or not fut:
            continue
        gain = finite(now["lo"]) and now["lo"] > 0
        loss = finite(fut["hi"]) and fut["hi"] < 0
        state = ("both directions" if gain and loss else
                 "present gain only" if gain else
                 "future loss only" if loss else "neither direction")
        if gain and loss:
            supported.append(words)
        lines.append(f"* {words}-word budget: dU_now = {_fmt(now['delta'])} "
                     f"[{_fmt(now['lo'])}, {_fmt(now['hi'])}], dU_future = {_fmt(fut['delta'])} "
                     f"[{_fmt(fut['lo'])}, {_fmt(fut['hi'])}] - {state} significant.")
    header = ("Communication regret from conditioning is **supported** at budget(s) "
              f"{supported} (present gain and future loss both excluded zero)."
              if supported else
              "Communication regret from conditioning is **not supported** at any budget "
              "tested: no budget shows a present-utility gain and a future-utility loss that "
              "both exclude zero. Reported as measured rather than tuned toward.")
    slope = lookup(interaction, quantity="conditioning_gap_low_minus_high", metric=metric)
    tail = ""
    if slope:
        direction = ("increases" if finite(slope["lo"]) and slope["lo"] > 0 else
                     "decreases" if finite(slope["hi"]) and slope["hi"] < 0 else
                     "does not measurably change")
        tail = (f"\n\nThe conditioning-induced specialisation gap {direction} as the budget "
                f"falls from {slope['high_budget_words']} to {slope['low_budget_words']} words "
                f"({_ci(slope)}, p = {_fmt(slope['p_value'], 3)}).")
    return header + "\n\n" + "\n".join(lines) + tail


# ---------------------------------------------------------------------------
# Orchestration


def analyse_and_write(cfg, dataset, contexts, rotations, messages, answers, policies,
                      budgets_words, result_root: Path, source_manifest: dict) -> None:
    metric = cfg["analysis"]["primary_metric"]
    metric_label = {"judge_correct": "LLM-judge accuracy", "em": "exact match",
                    "f1": "token F1"}[metric]
    present_policies = [p for p in policies if any(m["policy"] == p for m in messages)]

    matrix_rows = build_matrix_rows(contexts, rotations, messages, answers, present_policies)
    rotation_rows = build_rotation_rows(matrix_rows)
    metrics = analyse_metrics(rotation_rows, present_policies, budgets_words, cfg)
    contrasts = analyse_contrasts(rotation_rows, present_policies, budgets_words, cfg)
    interaction = analyse_budget_interaction(rotation_rows, present_policies, budgets_words, cfg)
    length_table, length_deltas = analyse_length(
        rotation_rows, messages, present_policies, budgets_words, cfg)
    matched = analyse_length_matched(rotation_rows, present_policies, budgets_words, cfg)
    pareto = pareto_points(metrics, present_policies, budgets_words, metric,
                           float(cfg["analysis"]["pareto_epsilon"]))
    normalised = analyse_normalised(rotation_rows, present_policies, budgets_words, cfg)
    baselines = analyse_baselines(answers, cfg)
    relation_table, relation_tests = analyse_relations(
        matrix_rows, present_policies, budgets_words, cfg)

    result_root.mkdir(parents=True, exist_ok=True)
    write_csv(result_root / "utility_matrix.csv", matrix_rows)
    write_csv(result_root / "rotation_rows.csv", rotation_rows)
    write_csv(result_root / "metrics.csv", metrics)
    write_csv(result_root / "contrasts.csv", contrasts)
    write_csv(result_root / "budget_interaction.csv", interaction)
    write_csv(result_root / "length_audit.csv", length_table)
    write_csv(result_root / "length_deltas.csv", length_deltas)
    write_csv(result_root / "length_matched_contrasts.csv", matched)
    write_csv(result_root / "pareto.csv", pareto)
    write_csv(result_root / "normalised_utility.csv", normalised)
    write_csv(result_root / "baselines.csv", baselines)
    write_csv(result_root / "relation_regret.csv", relation_table)
    write_csv(result_root / "relation_tests.csv", relation_tests)

    plot_pareto(pareto, result_root / "pareto_now_vs_future.png", metric_label)
    plot_budget_control(length_table, metrics, result_root / "budget_control.png",
                        metric, metric_label)
    plot_utility_vs_budget(metrics, baselines, result_root / "utility_vs_budget.png",
                           metric, metric_label)
    plot_regret(metrics, result_root / "communication_regret.png", metric, metric_label)
    plot_relation_distance(relation_table, result_root / "relation_distance.png",
                           metric, metric_label)

    write_report(result_root / "report.md", cfg=cfg, dataset=dataset, contexts=contexts,
                 policies=present_policies, budgets_words=budgets_words, metric=metric,
                 metrics=metrics, contrasts=contrasts, length_table=length_table,
                 length_deltas=length_deltas, matched=matched, interaction=interaction,
                 pareto=pareto, baselines=baselines, normalised=normalised,
                 relation_table=relation_table,
                 relation_tests=relation_tests, rotation_rows=rotation_rows,
                 source_manifest=source_manifest)

    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset": dataset,
        "source_manifest": source_manifest,
        "source_file": cfg["dataset"][dataset]["contexts_jsonl"],
        "sender_model": cfg["model"]["id"],
        "answerer_model": (cfg.get("answerer") or cfg["model"])["id"],
        "judge_model": cfg.get("judge", {}).get("model_id"),
        "policies": present_policies, "budgets_words": budgets_words,
        "primary_metric": metric, "cluster": "context",
        **rd.coverage_table(contexts),
    }
    (result_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def dry_run_report(contexts, rotations, policies, budgets, cfg) -> dict:
    per_context_questions = {c.context_id: len(c.questions) for c in contexts}
    messages = 0
    selection_messages = 0
    answers = 0
    for context in contexts:
        k = per_context_questions[context.context_id]
        r = len(rotations[context.context_id])
        for _b in budgets:
            for policy in policies:
                n = 1 if rotation_invariant(policy) else r
                messages += n
                if el.is_selection(policy):
                    selection_messages += n
                answers += n * k
    trim_budgets = [int(w) for w in cfg["budget"].get("trim_sensitivity_words") or []
                    if int(w) in {b.words for b in budgets}]
    trim_messages = 0
    for context in contexts:
        k = per_context_questions[context.context_id]
        r = len(rotations[context.context_id])
        for _b in trim_budgets:
            for policy in policies:
                if policy not in bd.ABSTRACTIVE_POLICIES:
                    continue
                n = 1 if rotation_invariant(policy) else r
                trim_messages += n
                answers += n * k
    baseline_answers = sum(
        len(c.questions) * (1 + int(cfg["baselines"]["closed_book_samples"])) for c in contexts)
    return {
        "writes": 0,
        "contexts": len(contexts),
        "questions": sum(per_context_questions.values()),
        "rotations": sum(len(v) for v in rotations.values()),
        "budgets_words": [b.words for b in budgets],
        "budgets_tokens": [b.tokens for b in budgets],
        "policies": policies,
        "generated_messages": messages - selection_messages,
        "selection_messages_no_generation": selection_messages,
        "trimmed_messages_no_generation": trim_messages,
        "answer_calls": answers + baseline_answers,
        "judge_calls": answers + baseline_answers,
        "note": ("Message counts assume no shrink retries; each retry adds one sender call. "
                 "Identical requests are single-flighted, so realised call counts are lower."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/communication_regret_config.yaml")
    parser.add_argument("--dataset", choices=["squad_groups", "relation_dossiers"],
                        help="override dataset.active")
    parser.add_argument("--limit", type=int, help="bounded pilot over the first N contexts")
    parser.add_argument("--policies", help="comma-separated subset of the configured policies")
    parser.add_argument("--budgets", help="comma-separated subset of the configured word budgets")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--no-trim", action="store_true",
                        help="skip the trimmed-to-common-length sensitivity arm")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the call plan; write nothing")
    parser.add_argument("--analyse-only", action="store_true",
                        help="re-analyse stored rows; no API calls")
    args = parser.parse_args(argv)
    if args.dry_run and args.analyse_only:
        parser.error("--dry-run and --analyse-only are mutually exclusive")

    cfg = load_config(ROOT / args.config)
    if args.dataset:
        cfg["dataset"]["active"] = args.dataset
    if args.concurrency is not None:
        if args.concurrency < 1:
            parser.error("--concurrency must be >= 1")
        cfg["runtime"]["concurrency"] = args.concurrency
    if args.no_judge:
        cfg["judge"]["enabled"] = False
    if args.no_trim:
        cfg["budget"]["trim_sensitivity_words"] = []

    dataset = cfg["dataset"]["active"]
    policies = resolve_policies(cfg, args.policies)
    budgets = bd.budgets_from_config(cfg)
    if args.budgets:
        wanted = {int(x) for x in args.budgets.split(",") if x.strip()}
        budgets = [b for b in budgets if b.words in wanted]
        if not budgets:
            parser.error("--budgets selected none of the configured word budgets")
    budgets_words = [b.words for b in budgets]

    source_manifest, contexts, rotations = load_design(cfg, args.limit)
    if args.dry_run:
        # Before any client or mkdir: --dry-run is pure read/compute/print and
        # must never touch data/, runs/ or results/.
        print(json.dumps(dry_run_report(contexts, rotations, policies, budgets, cfg), indent=2))
        return 0

    n = len(contexts)
    run_root = ROOT / cfg["outputs"]["run_root"] / dataset / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / dataset / f"n{n}"
    message_path = run_root / "messages.jsonl"
    answer_path = run_root / "answers.jsonl"

    resources = selection_resources(cfg, contexts, policies)
    specs = message_specs(contexts, rotations, policies, budgets, cfg, resources)
    trim_words = [int(w) for w in cfg["budget"].get("trim_sensitivity_words") or []
                  if int(w) in budgets_words]

    if args.analyse_only:
        stored = read_jsonl(message_path)
        wanted = {s["message_key"] for s in specs}
        messages = [row for row in stored if row.get("message_key") in wanted]
        if {row["message_key"] for row in messages} != wanted:
            missing = wanted - {row["message_key"] for row in messages}
            raise RuntimeError(f"analyse-only is missing {len(missing)} message rows")
        messages = messages + trimmed_messages(messages, trim_words)
        a_specs = answer_specs(contexts, rotations, messages, cfg)
        expected = {s["answer_key"] for s in a_specs}
        answers = [row for row in read_jsonl(answer_path) if row.get("answer_key") in expected]
        if {row["answer_key"] for row in answers} != expected:
            missing = expected - {row["answer_key"] for row in answers}
            raise RuntimeError(f"analyse-only is missing {len(missing)} answer rows")
        all_policies = sorted({row["policy"] for row in messages})
        analyse_and_write(cfg, dataset, contexts, rotations, messages, answers,
                          all_policies, budgets_words, result_root, source_manifest)
        print(f"[regret] reanalysed {len(messages)} messages / {len(answers)} answers")
        return 0

    sender, answerer = make_clients(cfg)
    stored_messages = read_jsonl(message_path)
    messages, generated = generate_messages(sender, specs, cfg, stored_messages)
    index = {row.get("message_key"): row for row in stored_messages}
    index.update({row["message_key"]: row for row in messages})
    write_jsonl(message_path, sorted(index.values(), key=lambda r: str(r.get("message_key"))))
    over = sum(1 for m in messages if int(m["delivered_words"]) > int(m["budget_words"]))
    print(f"[regret:send] {len(messages)} messages; {generated} generated this pass; "
          f"{over} over the delivered cap (must be 0); "
          f"{sum(1 for m in messages if int(m['attempts']) > 1)} needed a shrink retry")

    messages = messages + trimmed_messages(messages, trim_words)
    a_specs = answer_specs(contexts, rotations, messages, cfg)
    stored_answers = read_jsonl(answer_path)
    answers, answered = generate_answers(answerer, a_specs, cfg, stored_answers)
    answer_index = {row.get("answer_key"): row for row in stored_answers}
    answer_index.update({row["answer_key"]: row for row in answers})
    write_jsonl(answer_path, sorted(answer_index.values(),
                                    key=lambda r: str(r.get("answer_key"))))
    judge_client = add_judge(answers, cfg, tag="regret_judge")
    answer_index.update({row["answer_key"]: row for row in answers})
    write_jsonl(answer_path, sorted(answer_index.values(),
                                    key=lambda r: str(r.get("answer_key"))))
    print(f"[regret:answer] {len(answers)} rows; {answered} calls this pass")

    all_policies = sorted({row["policy"] for row in messages})
    analyse_and_write(cfg, dataset, contexts, rotations, messages, answers,
                      all_policies, budgets_words, result_root, source_manifest)
    print("[regret:cost] " + json.dumps(sender.ledger.summary()))
    if judge_client is not None:
        print("[regret:judge-cost] " + json.dumps(judge_client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
