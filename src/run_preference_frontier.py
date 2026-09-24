"""Experiment 11: objective- and allocation-sweep bounded communication.

The utility matrix and channel are inherited from Experiment 10.  What is new
is a controlled family of policies indexed by the scalar preference

    max U_now + lambda U_future  subject to C <= B.

``scalar`` gives that objective preference to one free-form writer without
prescribing a quota.  ``split`` separately varies a reusable-evidence word
share alpha, indexed on the convenient grid alpha=lambda/(1+lambda), with
separately generated current-query and reusable-evidence sections whose
*combined delivered text* obeys the same hard word budget.  Objective weights
do not mathematically imply proportional word allocation; the two treatments
are therefore analysed separately.  Every message is sealed and evaluated on
every question.

This runner writes only observed rows.  Distribution weighting, nondominance,
hypervolume, uncertainty and plots live in ``render_frontier_study.py`` so the
same answers can be audited under several future-query distributions without
new model calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import budget as bd  # noqa: E402
import run_communication_regret as cr  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import estimate_tokens, load_config  # noqa: E402
from size_metrics import internal_repetition_rate  # noqa: E402


SCHEMA_VERSION = "bounded-communication-frontier-v2"
FAMILIES = ("scalar", "split")


@dataclass(frozen=True)
class Preference:
    id: str
    value: float
    future_share: float

    @property
    def infinite(self) -> bool:
        return math.isinf(self.value)

    @property
    def display(self) -> str:
        return "infinity" if self.infinite else f"{self.value:g}"

    def as_dict(self) -> dict:
        return {
            "preference_id": self.id,
            "lambda": None if self.infinite else self.value,
            "lambda_infinite": self.infinite,
            "future_share": self.future_share,
        }


def _as_float(value) -> float:
    if isinstance(value, str) and value.strip().lower() in {"inf", ".inf", "infinity"}:
        return float("inf")
    return float(value)


def preferences_from_config(cfg: dict) -> list[Preference]:
    out = []
    for raw in cfg["frontier"]["preferences"]:
        pref = Preference(str(raw["id"]), _as_float(raw["lambda"]),
                          float(raw["future_share"]))
        if not 0.0 <= pref.future_share <= 1.0:
            raise ValueError(f"{pref.id}: future_share must be in [0, 1]")
        expected = 1.0 if pref.infinite else pref.value / (1.0 + pref.value)
        if not math.isclose(pref.future_share, expected, abs_tol=1e-9):
            raise ValueError(
                f"{pref.id}: future_share {pref.future_share} != lambda/(1+lambda) {expected}")
        out.append(pref)
    if len({p.id for p in out}) != len(out):
        raise ValueError("frontier.preferences contains duplicate ids")
    return out


def policy_name(family: str, preference_id: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    return f"{family}__{preference_id}"


def preference_block(pref: Preference, current_question: str) -> str:
    if pref.infinite:
        objective = (
            "Optimization preference: maximize expected utility on an unknown later question; "
            "the currently known question has zero weight. This is the lambda = infinity "
            "endpoint of U_now + lambda U_future."
        )
    else:
        objective = (
            f"Optimization preference: maximize U_now + {pref.value:g} * U_future. "
            "The coefficient is the relative value of a future-query answer, not a "
            "prescribed word quota. Choose whatever evidence allocation best serves this "
            "objective under the shared hard budget."
        )
    return (
        "The next agent first receives this currently known question:\n"
        f"  - {current_question}\n"
        "Afterwards it may receive one different, presently unknown question about the same "
        "source. U_now is accuracy on the known question and U_future is expected accuracy "
        "on that later question.\n"
        f"{objective}\n"
        "Select evidence to implement that preference. Do not discuss the coefficient in the "
        "handoff and do not answer in a conversational voice."
    )


def scalar_prompt(source: str, current_question: str, budget: bd.Budget,
                  pref: Preference) -> str:
    return (
        f"Source material:\n{source}\n\n"
        f"{preference_block(pref, current_question)}\n\n"
        f"{bd.LENGTH_CONTRACT.format(floor=budget.floor_words, words=budget.words)}\n"
        f"{bd.PROSE_FORM}"
    )


def _component_prompt(source: str, current_question: str, component: str,
                      budget: bd.Budget, pref: Preference) -> str:
    if component == "current":
        role = (
            "Write only the CURRENT-EVIDENCE component. Preserve the facts needed for the "
            "currently known question. Omit general background unless it is needed to "
            "interpret those facts."
        )
    elif component == "reusable":
        role = (
            "Write only the REUSABLE-EVIDENCE component. Preserve diverse facts that could "
            "answer other, presently unknown questions about the source. Do not state the "
            "answer to the currently known question and do not repeat facts narrowly tied "
            "only to it."
        )
    else:  # pragma: no cover - guarded by the allocator
        raise ValueError(component)
    return (
        f"Source material:\n{source}\n\n"
        f"Currently known question:\n  - {current_question}\n\n"
        f"{role}\n"
        f"This component implements the lambda={pref.display} structured allocation.\n\n"
        f"{bd.LENGTH_CONTRACT.format(floor=budget.floor_words, words=budget.words)}\n"
        "Write only compact evidence prose. Do not include a heading and do not answer in a "
        "conversational voice."
    )


def split_components(source: str, current_question: str, outer: bd.Budget,
                     pref: Preference, cfg: dict) -> list[dict]:
    """Allocate evidence words exactly; both headings always consume the cap.

    Holding the four-word heading overhead constant matters at the endpoints:
    otherwise lambda=0 and lambda=infinity would silently receive two extra
    evidence words relative to the interior preferences.
    """
    sections = (("current", "Current evidence:"),
                ("reusable", "Reusable evidence:"))
    heading_words = sum(bd.word_count(label) for _, label in sections)
    available = outer.words - heading_words
    if available < 2:
        raise ValueError(f"{outer.words}-word budget is too small for structured headings")
    reusable = int(round(available * pref.future_share))
    if 0.0 < pref.future_share < 1.0:
        reusable = min(available - 1, max(1, reusable))
    counts = {"reusable": reusable, "current": available - reusable}

    spec = cfg["budget"]
    out = []
    for component, label in sections:
        words = counts[component]
        b = (bd.make_budget(
            words, float(spec["tokens_per_word"]),
            float(spec["api_max_tokens_headroom"]), int(spec["api_max_tokens_floor"]),
            float(spec["fill_floor_ratio"]),
        ) if words else None)
        out.append({
            "component": component,
            "label": label,
            "budget": b,
            "planned_words": words,
            "prompt": (_component_prompt(source, current_question, component, b, pref)
                       if b else ""),
        })
    if sum(x["planned_words"] + bd.word_count(x["label"]) for x in out) != outer.words:
        raise AssertionError("structured allocation does not sum to the outer word cap")
    return out


def message_specs(contexts, rotations, budgets: list[bd.Budget], preferences,
                  families: list[str], cfg: dict, replication: str) -> list[dict]:
    specs = []
    for context in contexts:
        for rotation in rotations[context.context_id]:
            question = context.question(rotation.current_qid).question
            for budget in budgets:
                for pref in preferences:
                    for family in families:
                        policy = policy_name(family, pref.id)
                        if family == "scalar":
                            prompt = scalar_prompt(context.source, question, budget, pref)
                            components = []
                            hash_material = [prompt]
                        else:
                            prompt = ""
                            components = split_components(
                                context.source, question, budget, pref, cfg)
                            hash_material = [
                                (x["component"], x["label"], x["planned_words"],
                                 x["budget"].as_dict() if x["budget"] else None, x["prompt"])
                                for x in components
                            ]
                        key = (f"{replication}|{context.context_id}|{policy}|{budget.label}|"
                               f"{rotation.rotation_id}")
                        specs.append({
                            "message_key": key,
                            "replication": replication,
                            "context": context,
                            "context_id": context.context_id,
                            "rotation_id": rotation.rotation_id,
                            "current_qid": rotation.current_qid,
                            "policy": policy,
                            "family": family,
                            "preference": pref,
                            "sender_seed": cfg.get("_frontier_sender_seed"),
                            "budget": budget,
                            "prompt": prompt,
                            "components": components,
                            "request_hash": cr.stable_hash(
                                SCHEMA_VERSION, cfg["model"]["id"], bd.SENDER_SYSTEM,
                                hash_material, budget.as_dict(), pref.as_dict(), family,
                                float(cfg["decoding"]["sender_temperature"]),
                                float(cfg["decoding"]["top_p"]),
                                cfg.get("_frontier_sender_seed"),
                                int(cfg["budget"]["max_length_corrections"]),
                                context.content_hash,
                            ),
                        })
    keys = [s["message_key"] for s in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate message key")
    return specs


def _call(client, prompt: str, budget: bd.Budget, tag: str, cfg: dict):
    return bd.generate_within_budget(
        client, prompt, budget,
        temperature=float(cfg["decoding"]["sender_temperature"]),
        seed=cfg.get("_frontier_sender_seed"), tag=tag,
        max_corrections=int(cfg["budget"]["max_length_corrections"]),
    )


def run_message_call(client, spec: dict, cfg: dict) -> dict:
    outer: bd.Budget = spec["budget"]
    if spec["family"] == "scalar":
        result = _call(client, spec["prompt"], outer,
                       f"frontier_{spec['policy']}_{outer.label}", cfg)
        text = result.text
        records = [{"component": "scalar", "label": "", **result.as_dict()}]
        results = [result]
        extra_cut = False
        extra_dropped = 0
    else:
        results, records, pieces = [], [], []
        for component in spec["components"]:
            if component["budget"] is None:
                records.append({
                    "component": component["component"], "label": component["label"],
                    "planned_words": 0, "delivered_words": 0, "text": "",
                })
                pieces.append(component["label"])
                continue
            result = _call(
                client, component["prompt"], component["budget"],
                f"frontier_{spec['policy']}_{outer.label}_{component['component']}", cfg)
            results.append(result)
            records.append({
                "component": component["component"], "label": component["label"],
                "planned_words": component["planned_words"], "text": result.text,
                **result.as_dict(),
            })
            pieces.append(f"{component['label']} {result.text}".strip())
        assembled = " ".join(pieces)
        text, extra_cut, extra_dropped = bd.truncate_to_words(assembled, outer.words)

    delivered = bd.word_count(text)
    pref: Preference = spec["preference"]
    finish_reasons = [reason for r in results for reason in r.finish_reasons]
    row = {
        "schema_version": SCHEMA_VERSION,
        "message_key": spec["message_key"],
        "request_hash": spec["request_hash"],
        "replication": spec["replication"],
        "context_id": spec["context_id"],
        "context_hash": spec["context"].content_hash,
        "rotation_id": spec["rotation_id"],
        "current_qid": spec["current_qid"],
        "policy": spec["policy"],
        "family": spec["family"],
        "sender_seed": spec.get("sender_seed"),
        **pref.as_dict(),
        "derived_from": "",
        "handoff_text": text,
        "handoff_hash": cr.stable_hash(text),
        "budget_words": outer.words,
        "floor_words": outer.floor_words,
        "budget_tokens": outer.tokens,
        "api_max_tokens": sum(r.api_max_tokens for r in results),
        "attempts": sum(r.attempts for r in results),
        "shrink_corrections": sum(r.shrink_corrections for r in results),
        "expand_corrections": sum(r.expand_corrections for r in results),
        "first_draft_words": (sum(r.first_draft_words for r in results)
                              + (4 if spec["family"] == "split" else 0)),
        "final_draft_words": (sum(r.final_draft_words for r in results)
                              + (4 if spec["family"] == "split" else 0)),
        "delivered_words": delivered,
        "truncated": bool(extra_cut or any(r.truncated for r in results)),
        "truncated_words_dropped": extra_dropped + sum(r.truncated_words_dropped for r in results),
        "over_budget_before_truncation": bool(
            extra_cut or any(r.over_budget_before_truncation for r in results)),
        "under_floor_delivered": delivered < outer.floor_words,
        "fill_ratio": delivered / float(outer.words),
        "repetition_rate": internal_repetition_rate(text),
        "delivered_estimated_tokens": estimate_tokens(text) if text else 0,
        "completion_tokens": sum(r.completion_tokens for r in results),
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "finish_reasons": finish_reasons,
        "cached": all(r.cached for r in results),
        "component_records": records,
    }
    by_component = {r["component"]: r for r in records}
    for component in ("current", "reusable"):
        record = by_component.get(component, {})
        row[f"{component}_planned_words"] = int(record.get("planned_words", 0))
        row[f"{component}_delivered_words"] = int(record.get("delivered_words", 0))
    content_words = (row["current_delivered_words"] + row["reusable_delivered_words"])
    row["realized_future_share"] = (
        row["reusable_delivered_words"] / content_words if content_words else None)
    row["allocation_absolute_error"] = (
        abs(row["realized_future_share"] - pref.future_share)
        if row["realized_future_share"] is not None else None)
    current_text = str(by_component.get("current", {}).get("text", ""))
    reusable_text = str(by_component.get("reusable", {}).get("text", ""))
    tokenize = lambda value: set(re.findall(r"[a-z0-9]+", value.lower()))
    current_terms, reusable_terms = tokenize(current_text), tokenize(reusable_text)
    union = current_terms | reusable_terms
    row["component_overlap_jaccard"] = (
        len(current_terms & reusable_terms) / len(union) if union else 0.0)
    if delivered > outer.words:
        raise AssertionError("hard delivered word cap was violated")
    return row


def generate_messages(client, specs: list[dict], cfg: dict,
                      stored: list[dict]) -> tuple[list[dict], int]:
    index = {r.get("message_key"): r for r in stored
             if r.get("schema_version") == SCHEMA_VERSION}
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
    return [current[s["message_key"]] for s in specs], len(missing)


def configure_replication(cfg: dict, name: str, dataset: str | None,
                          limit: int | None) -> tuple[dict, dict, str, int]:
    reps = cfg["frontier"]["replications"]
    if name not in reps:
        raise ValueError(f"unknown replication {name!r}; choose from {sorted(reps)}")
    rep = copy.deepcopy(reps[name])
    datasets = list(rep["datasets"])
    selected = dataset or (datasets[0] if len(datasets) == 1 else cfg["dataset"]["active"])
    if selected not in datasets:
        raise ValueError(f"replication {name!r} does not configure dataset {selected!r}")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["id"] = rep["sender_model"]
    cfg["answerer"]["id"] = rep["answerer_model"]
    if "sender_temperature" in rep:
        cfg["decoding"]["sender_temperature"] = float(rep["sender_temperature"])
    cfg["_frontier_sender_seed"] = rep.get("sender_seed")
    cfg["dataset"]["active"] = selected
    n = int(limit if limit is not None else rep["limits"][selected])
    return cfg, rep, selected, n


def replication_budget_words(rep: dict, dataset: str) -> list[int]:
    """Resolve an optional dataset-specific subset of a replication's grid."""
    default = [int(value) for value in rep["budgets_words"]]
    raw = (rep.get("budgets_by_dataset", {}) or {}).get(dataset, default)
    selected = list(dict.fromkeys(int(value) for value in raw))
    invalid = [value for value in selected if value not in set(default)]
    if not selected or invalid:
        raise ValueError(
            f"configured budgets for {dataset!r} must be a non-empty subset of "
            f"{default}; invalid={invalid}"
        )
    return selected


def _select_csv(values: str | None, allowed: list[str], what: str) -> list[str]:
    if values is None:
        return list(allowed)
    chosen = [x.strip() for x in values.split(",") if x.strip()]
    bad = [x for x in chosen if x not in allowed]
    if not chosen or bad:
        raise ValueError(f"{what} must be a non-empty subset of {allowed}; invalid={bad}")
    return list(dict.fromkeys(chosen))


def dry_run_report(contexts, rotations, budgets, preferences, families, cfg: dict) -> dict:
    messages = sum(len(rotations[c.context_id]) for c in contexts) * len(budgets) \
        * len(preferences) * len(families)
    scalar = sum(len(rotations[c.context_id]) for c in contexts) * len(budgets) \
        * len(preferences) * int("scalar" in families)
    split_components_count = 0
    if "split" in families:
        for context in contexts:
            for _rotation in rotations[context.context_id]:
                for budget in budgets:
                    for pref in preferences:
                        split_components_count += len(split_components(
                            context.source, context.question(_rotation.current_qid).question,
                            budget, pref, cfg)) - sum(
                                1 for part in split_components(
                                    context.source,
                                    context.question(_rotation.current_qid).question,
                                    budget, pref, cfg)
                                if part["budget"] is None)
    answer_rows = sum(len(rotations[c.context_id]) * len(c.questions) for c in contexts) \
        * len(budgets) * len(preferences) * len(families)
    baselines = sum(len(c.questions) * (
        int(bool(cfg["baselines"].get("direct_context", True)))
        + int(cfg["baselines"]["closed_book_samples"])) for c in contexts)
    return {
        "writes": 0,
        "contexts": len(contexts),
        "rotations": sum(len(rotations[c.context_id]) for c in contexts),
        "policies": [policy_name(f, p.id) for f in families for p in preferences],
        "budgets_words": [b.words for b in budgets],
        "messages": messages,
        "sender_calls_before_length_corrections": scalar + split_components_count,
        "answer_rows": answer_rows + baselines,
        "judge_rows": answer_rows + baselines,
        "note": "Identical requests hit the shared content cache; length corrections add calls.",
    }


def analyse_and_write(cfg: dict, replication: str, rep: dict, dataset: str, contexts,
                      rotations, messages: list[dict], answers: list[dict], budgets,
                      preferences, families, result_root: Path, source_manifest: dict) -> None:
    policies = [policy_name(f, p.id) for f in families for p in preferences]
    matrix = cr.build_matrix_rows(contexts, rotations, messages, answers, policies)
    meta = {m["message_key"]: m for m in messages}
    for row in matrix:
        m = meta[row["message_key"]]
        row.update({k: m[k] for k in ("replication", "family", "preference_id", "lambda",
                                      "lambda_infinite", "future_share", "sender_seed")})
    rotation_rows = cr.build_rotation_rows(matrix)
    pmeta = {policy_name(f, p.id): (f, p) for f in families for p in preferences}
    for row in rotation_rows:
        family, pref = pmeta[row["policy"]]
        row.update({"replication": replication, "family": family, **pref.as_dict()})

    budgets_words = [b.words for b in budgets]
    metrics = cr.analyse_metrics(rotation_rows, policies, budgets_words, cfg)
    for row in metrics:
        family, pref = pmeta[row["policy"]]
        row.update({"replication": replication, "family": family, **pref.as_dict()})
    length, _ = cr.analyse_length(rotation_rows, messages, policies, budgets_words, cfg)
    for row in length:
        family, pref = pmeta[row["policy"]]
        row.update({"replication": replication, "family": family, **pref.as_dict()})

    result_root.mkdir(parents=True, exist_ok=True)
    cr.write_csv(result_root / "utility_matrix.csv", matrix)
    cr.write_csv(result_root / "rotation_rows.csv", rotation_rows)
    cr.write_csv(result_root / "metrics.csv", metrics)
    cr.write_csv(result_root / "length_audit.csv", length)
    cr.write_csv(result_root / "baselines.csv", cr.analyse_baselines(answers, cfg))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "replication": replication,
        "replication_label": rep["label"],
        "dataset": dataset,
        "source_manifest": source_manifest,
        "sender_model": cfg["model"]["id"],
        "answerer_model": cfg["answerer"]["id"],
        "judge_model": cfg.get("judge", {}).get("model_id"),
        "sender_temperature": float(cfg["decoding"]["sender_temperature"]),
        "sender_seed": cfg.get("_frontier_sender_seed"),
        "families": families,
        "preferences": [p.as_dict() for p in preferences],
        "budgets_words": budgets_words,
        "primary_metric": cfg["analysis"]["primary_metric"],
        "cluster": "context",
        **cr.rd.coverage_table(contexts),
    }
    (result_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bounded_communication_frontier_config.yaml")
    parser.add_argument("--replication", default="primary")
    parser.add_argument("--dataset", choices=["squad_groups", "relation_dossiers"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budgets", help="comma-separated configured word budgets")
    parser.add_argument("--families", help="comma-separated subset of scalar,split")
    parser.add_argument("--preferences", help="comma-separated preference ids")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.analyse_only:
        parser.error("--dry-run and --analyse-only are mutually exclusive")

    cfg = load_config(ROOT / args.config)
    try:
        cfg, rep, dataset, n = configure_replication(
            cfg, args.replication, args.dataset, args.limit)
        if args.concurrency is not None:
            if args.concurrency < 1:
                parser.error("--concurrency must be >= 1")
            cfg["runtime"]["concurrency"] = args.concurrency
        if args.no_judge:
            cfg["judge"]["enabled"] = False

        allowed_families = list(rep.get("families") or cfg["frontier"]["families"])
        families = _select_csv(args.families, allowed_families, "--families")
        all_preferences = {p.id: p for p in preferences_from_config(cfg)}
        allowed_pref_ids = list(rep["preference_ids"])
        pref_ids = _select_csv(args.preferences, allowed_pref_ids, "--preferences")
        preferences = [all_preferences[p] for p in pref_ids]
        allowed_budget_words = replication_budget_words(rep, dataset)
    except ValueError as exc:
        parser.error(str(exc))

    budgets = [b for b in bd.budgets_from_config(cfg)
               if b.words in set(allowed_budget_words)]
    if args.budgets:
        wanted = {int(x) for x in args.budgets.split(",") if x.strip()}
        budgets = [b for b in budgets if b.words in wanted]
    if not budgets:
        parser.error("no configured budgets selected")

    source_manifest, contexts, rotations = cr.load_design(cfg, n)
    specs = message_specs(contexts, rotations, budgets, preferences, families, cfg,
                          args.replication)
    if args.dry_run:
        print(json.dumps(dry_run_report(contexts, rotations, budgets, preferences, families, cfg),
                         indent=2))
        return 0

    run_root = (ROOT / cfg["outputs"]["run_root"] / args.replication / dataset / f"n{n}")
    result_root = (ROOT / cfg["outputs"]["result_root"] / args.replication / dataset / f"n{n}")
    message_path = run_root / "messages.jsonl"
    answer_path = run_root / "answers.jsonl"
    wanted_messages = {s["message_key"] for s in specs}

    if args.analyse_only:
        spec_by_key = {s["message_key"]: s for s in specs}
        messages = [m for m in cr.read_jsonl(message_path)
                    if m.get("message_key") in wanted_messages
                    and m.get("request_hash") == spec_by_key[m["message_key"]]["request_hash"]]
        missing = wanted_messages - {m["message_key"] for m in messages}
        if missing:
            raise RuntimeError(f"analyse-only is missing {len(missing)} messages")
        answer_specs = cr.answer_specs(contexts, rotations, messages, cfg)
        wanted_answers = {s["answer_key"] for s in answer_specs}
        answer_spec_by_key = {s["answer_key"]: s for s in answer_specs}
        answers = [a for a in cr.read_jsonl(answer_path)
                   if a.get("answer_key") in wanted_answers
                   and a.get("request_hash") == answer_spec_by_key[a["answer_key"]]["request_hash"]]
        missing = wanted_answers - {a["answer_key"] for a in answers}
        if missing:
            raise RuntimeError(f"analyse-only is missing {len(missing)} answers")
        analyse_and_write(cfg, args.replication, rep, dataset, contexts, rotations,
                          messages, answers, budgets, preferences, families, result_root,
                          source_manifest)
        print(f"[frontier] reanalysed {len(messages)} messages / {len(answers)} answers")
        return 0

    sender, answerer = cr.make_clients(cfg)
    stored_messages = cr.read_jsonl(message_path)
    messages, generated = generate_messages(sender, specs, cfg, stored_messages)
    message_index = {m.get("message_key"): m for m in stored_messages}
    message_index.update({m["message_key"]: m for m in messages})
    cr.write_jsonl(message_path, sorted(message_index.values(),
                                        key=lambda r: str(r.get("message_key"))))
    print(f"[frontier:send] {len(messages)} messages; {generated} generated; "
          f"{sum(m['delivered_words'] > m['budget_words'] for m in messages)} over cap")

    answer_specs = cr.answer_specs(contexts, rotations, messages, cfg)
    stored_answers = cr.read_jsonl(answer_path)
    answers, answered = cr.generate_answers(answerer, answer_specs, cfg, stored_answers)
    answer_index = {a.get("answer_key"): a for a in stored_answers}
    answer_index.update({a["answer_key"]: a for a in answers})
    cr.write_jsonl(answer_path, sorted(answer_index.values(),
                                       key=lambda r: str(r.get("answer_key"))))
    judge_client = add_judge(answers, cfg, tag="frontier_judge")
    answer_index.update({a["answer_key"]: a for a in answers})
    cr.write_jsonl(answer_path, sorted(answer_index.values(),
                                       key=lambda r: str(r.get("answer_key"))))
    print(f"[frontier:answer] {len(answers)} rows; {answered} model calls this pass")

    analyse_and_write(cfg, args.replication, rep, dataset, contexts, rotations,
                      messages, answers, budgets, preferences, families, result_root,
                      source_manifest)
    print("[frontier:cost] " + json.dumps(sender.ledger.summary()))
    if judge_client is not None:
        print("[frontier:judge-cost] " + json.dumps(judge_client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
