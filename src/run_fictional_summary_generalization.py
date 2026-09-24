"""Experiment 5b: fixed-capacity summary reuse on fictional QA dossiers.

The source corpus is shared with Experiment 9.  Each fictional dossier carries
six independently answerable facts.  We rotate every fact through the visible
Question-A role, keep the other questions hidden, and compare two selectors:

* ``conditioned`` sees the one currently announced question;
* ``generic`` sees no question and is generated once per dossier/capacity.

Both selectors return exactly K evidence-card ids.  The ids are validated and
rendered deterministically as K source-grounded packet slots, so an arm cannot
buy future-query accuracy by simply writing a longer free-form summary.  The
final answerer receives only a frozen ``SealedHandoff`` and one evaluation
question.  It never receives the source dossier or the unselected cards.
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

import fictional_qa as fqa  # noqa: E402
import handoffs as hm  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import LLMClient, estimate_tokens, load_config  # noqa: E402
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402


SCHEMA_VERSION = "fictional-summary-generalization-v1"
SELECTOR_SYSTEM = (
    "You select source-grounded evidence for a sealed handoff to another agent. "
    "Obey the requested JSON schema exactly. Select only supplied evidence-card ids; "
    "never write, merge, paraphrase, or invent evidence."
)


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


def make_clients(cfg: dict, dry_run: bool = False) -> tuple[LLMClient, LLMClient]:
    """Build selector and fixed-answerer clients under one cache and cost cap."""
    selector = LLMClient(cfg, dry_run=dry_run)
    answerer_spec = cfg.get("answerer") or cfg["model"]
    if answerer_spec["id"] == cfg["model"]["id"] and all(
            answerer_spec.get(k) == cfg["model"].get(k)
            for k in ("reasoning", "system_suffix")):
        return selector, selector
    answer_cfg = copy.deepcopy(cfg)
    answer_cfg["model"] = copy.deepcopy(answerer_spec)
    answerer = LLMClient(answer_cfg, dry_run=dry_run)
    # One experiment, one cap.  The model id is already in every cache key.
    answerer.cache = selector.cache
    answerer.ledger = selector.ledger
    return selector, answerer


def load_design(cfg: dict, limit: int | None = None):
    manifest, corpus = fqa.load_fictional_corpus(ROOT / cfg["dataset"]["items_jsonl"])
    n = int(limit if limit is not None else cfg["dataset"]["n_items"])
    dossiers = tuple(corpus[:n])
    if not dossiers:
        raise RuntimeError("fictional corpus produced zero dossiers")
    rotations_per = int(cfg["dataset"]["rotations_per_dossier"])
    card_seed = int(cfg["dataset"]["card_seed"])
    rotation_seed = int(cfg["dataset"]["rotation_seed"])
    cards = {d.dossier_id: fqa.evidence_cards(d, seed=card_seed) for d in dossiers}
    rotations = {
        d.dossier_id: fqa.role_rotations(d, count=rotations_per, seed=rotation_seed)
        for d in dossiers
    }
    for dossier in dossiers:
        if len(cards[dossier.dossier_id]) < max(int(x) for x in cfg["selection"]["slots"]):
            raise ValueError(f"{dossier.dossier_id}: fewer cards than the largest K")
        if len(rotations[dossier.dossier_id]) != rotations_per:
            raise ValueError(f"{dossier.dossier_id}: expected {rotations_per} rotations")
    return manifest, dossiers, cards, rotations


def parse_slots(cfg: dict, cli_value: str | None) -> list[int]:
    configured = [int(x) for x in cfg["selection"]["slots"]]
    if cli_value is None:
        return configured
    chosen = [int(x.strip()) for x in cli_value.split(",") if x.strip()]
    if not chosen or any(x not in configured for x in chosen):
        raise ValueError(f"--slots must be a non-empty subset of {configured}")
    return list(dict.fromkeys(chosen))


# ---------------------------------------------------------------------------
# Exact-K selection


def selection_key(dossier, rotation, mode: str, slots: int, seed: int) -> str:
    material_key = fqa.selection_material_key(
        dossier, rotation if mode == "conditioned" else None, mode, slots)
    return f"{material_key}|seed={seed}"


def selection_prompt(dossier, cards, rotation, mode: str, slots: int) -> str:
    if mode == "generic":
        return fqa.generic_selection_prompt(dossier, cards, slots)
    if mode == "conditioned":
        return fqa.conditioned_selection_prompt(dossier, rotation, cards, slots)
    raise ValueError(f"unknown selection mode {mode!r}")


def selection_specs(dossiers, cards_by_dossier, rotations_by_dossier,
                    slots_list: list[int], cfg: dict) -> list[dict]:
    """Logical calls, with one generic call per dossier/K (never per rotation)."""
    seed = int(cfg["selection"]["seed"])
    specs = []
    for dossier in dossiers:
        cards = cards_by_dossier[dossier.dossier_id]
        for slots in slots_list:
            prompt = selection_prompt(dossier, cards, None, "generic", slots)
            specs.append({
                "selection_key": selection_key(dossier, None, "generic", slots, seed),
                "dossier": dossier, "cards": cards, "rotation": None,
                "mode": "generic", "slots": slots, "seed": seed, "prompt": prompt,
            })
            for rotation in rotations_by_dossier[dossier.dossier_id]:
                prompt = selection_prompt(dossier, cards, rotation, "conditioned", slots)
                specs.append({
                    "selection_key": selection_key(
                        dossier, rotation, "conditioned", slots, seed),
                    "dossier": dossier, "cards": cards, "rotation": rotation,
                    "mode": "conditioned", "slots": slots, "seed": seed,
                    "prompt": prompt,
                })
    keys = [spec["selection_key"] for spec in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError("selection keys are not unique; generic may have been duplicated")
    return specs


def selection_request_hash(spec: dict, cfg: dict) -> str:
    return stable_hash(
        SCHEMA_VERSION, cfg["model"]["id"], SELECTOR_SYSTEM, spec["prompt"],
        float(cfg["decoding"]["selector_temperature"]),
        float(cfg["decoding"]["top_p"]), int(cfg["decoding"]["selection_max_tokens"]),
        int(spec["seed"]), {"type": "json_object"}, spec["dossier"].content_hash,
    )


def run_selection_call(client: LLMClient, spec: dict, cfg: dict) -> dict:
    valid_ids = [card.card_id for card in spec["cards"]]
    max_attempts = int(cfg["selection"]["max_parse_attempts"])
    attempts = []
    parsed = None
    for attempt in range(max_attempts):
        correction = ""
        if parsed is not None:
            correction = (
                "\n\nFORMAT CORRECTION: the previous response was invalid because "
                f"{parsed.error}. Return exactly one JSON object with exactly "
                f"{spec['slots']} distinct valid ids and no other key."
            )
        result = client.chat(
            [{"role": "system", "content": SELECTOR_SYSTEM},
             {"role": "user", "content": spec["prompt"] + correction}],
            temperature=float(cfg["decoding"]["selector_temperature"]),
            max_tokens=int(cfg["decoding"]["selection_max_tokens"]),
            seed=int(spec["seed"]) + attempt,
            response_format={"type": "json_object"},
            tag=f"fictional_exp5b_{spec['mode']}_k{spec['slots']}_attempt{attempt + 1}",
        )
        parsed = fqa.parse_fixed_k_selection(result.text, valid_ids, spec["slots"])
        attempts.append({
            "attempt": attempt + 1, "raw": result.text, "cached": result.cached,
            "finish_reason": result.finish_reason, "parse_valid": parsed.valid,
            "parse_error": parsed.error, "recognized_card_ids": list(parsed.recognized_fact_ids),
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        })
        if parsed.valid:
            break
    assert parsed is not None
    selected_cards = list(parsed.selected_fact_ids)
    selected = (list(fqa.card_ids_to_fact_ids(spec["cards"], selected_cards))
                if parsed.valid else [])
    recognized_cards = list(parsed.recognized_fact_ids)
    recognized = list(fqa.card_ids_to_fact_ids(spec["cards"], recognized_cards))
    packet = fqa.render_packet_slots(spec["cards"], selected, spec["slots"]) if parsed.valid else ""
    rotation = spec["rotation"]
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_key": spec["selection_key"],
        "request_hash": selection_request_hash(spec, cfg),
        "dossier_id": spec["dossier"].dossier_id,
        "dossier_hash": spec["dossier"].content_hash,
        "rotation_id": rotation.rotation_id if rotation is not None else None,
        "announced_fact_id": rotation.target_fact_id if rotation is not None else None,
        "mode": spec["mode"], "slots": int(spec["slots"]), "seed": int(spec["seed"]),
        "selection_valid": bool(parsed.valid), "selection_error": parsed.error,
        "selected_card_ids": selected_cards, "selected_fact_ids": selected,
        "recognized_card_ids": recognized_cards, "recognized_fact_ids": recognized,
        "selected_count": len(selected), "attempt_count": len(attempts), "attempts": attempts,
        "handoff_text": packet,
        "handoff_hash": stable_hash(packet),
        "handoff_characters": len(packet),
        "handoff_words": len(packet.split()),
        "handoff_estimated_tokens": estimate_tokens(packet) if packet else 0,
        "prompt_tokens": sum(int(a["prompt_tokens"]) for a in attempts),
        "completion_tokens": sum(int(a["completion_tokens"]) for a in attempts),
    }


def generate_selections(client, specs: list[dict], cfg: dict,
                        stored: list[dict]) -> tuple[list[dict], int]:
    stored_index = {row["selection_key"]: row for row in stored
                    if row.get("schema_version") == SCHEMA_VERSION}
    current: dict[str, dict] = {}
    missing = []
    for spec in specs:
        old = stored_index.get(spec["selection_key"])
        if old is not None and old.get("request_hash") == selection_request_hash(spec, cfg):
            current[spec["selection_key"]] = old
        else:
            missing.append(spec)
    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        futures = {pool.submit(run_selection_call, client, spec, cfg): spec for spec in missing}
        for future, spec in futures.items():
            current[spec["selection_key"]] = future.result()
    return [current[spec["selection_key"]] for spec in specs], len(missing)


# ---------------------------------------------------------------------------
# Sealed answering and baselines


def answer_request_hash(model_id: str, question: str, material: str,
                        temperature: float, seed: int | None, cfg: dict) -> str:
    return stable_hash(
        SCHEMA_VERSION, model_id, hm.ANSWER_SYSTEM, question, material,
        float(temperature), float(cfg["decoding"]["top_p"]),
        int(cfg["decoding"]["answer_max_tokens"]), seed,
    )


def answer_sealed(client: LLMClient, sealed: hm.SealedHandoff, cfg: dict,
                  temperature: float, seed: int | None, tag: str) -> dict:
    if not isinstance(sealed, hm.SealedHandoff):
        raise TypeError("answer_sealed accepts only a frozen SealedHandoff")
    user = (f"Research material:\n{sealed.handoff_text}\n\n"
            f"Question:\n{sealed.question}\nAnswer:")
    result = client.chat(
        [{"role": "system", "content": hm.ANSWER_SYSTEM},
         {"role": "user", "content": user}],
        temperature=temperature, max_tokens=int(cfg["decoding"]["answer_max_tokens"]),
        seed=seed, tag=tag,
    )
    return {"raw": result.text, "pred": extract_short_answer(result.text),
            "cached": result.cached, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens}


def answer_specs(dossiers, cards_by_dossier, selections: list[dict], cfg: dict) -> list[dict]:
    """Unique answer calls. Generic answers are not duplicated across rotations."""
    specs = []
    answer_model = (cfg.get("answerer") or cfg["model"])["id"]
    answer_temp = float(cfg["decoding"]["answer_temperature"])
    closed_temp = float(cfg["decoding"]["closed_book_temperature"])
    closed_samples = int(cfg["baselines"]["closed_book_samples"])
    for dossier in dossiers:
        cards = cards_by_dossier[dossier.dossier_id]
        baselines = {
            "direct_source": dossier.document,
            "direct_cards": fqa.render_source_cards(cards),
        }
        for fact in dossier.facts:
            for baseline, material in baselines.items():
                key = f"baseline|{dossier.dossier_id}|{fact.fact_id}|{baseline}|sample=0"
                specs.append({
                    "answer_key": key, "selection_key": None, "dossier_id": dossier.dossier_id,
                    "announced_fact_id": None, "eval_fact": fact, "mode": baseline,
                    "slots": None, "baseline": baseline, "sample": 0, "material": material,
                    "temperature": answer_temp, "seed": None, "technical_failure": False,
                })
            for sample in range(closed_samples):
                key = f"baseline|{dossier.dossier_id}|{fact.fact_id}|closed_book|sample={sample}"
                specs.append({
                    "answer_key": key, "selection_key": None, "dossier_id": dossier.dossier_id,
                    "announced_fact_id": None, "eval_fact": fact, "mode": "closed_book",
                    "slots": None, "baseline": "closed_book", "sample": sample,
                    "material": "(No research material was supplied.)",
                    "temperature": closed_temp, "seed": 1000 + sample,
                    "technical_failure": False,
                })
    dossier_map = {d.dossier_id: d for d in dossiers}
    for selection in selections:
        dossier = dossier_map[selection["dossier_id"]]
        for fact in dossier.facts:
            key = f"selection|{selection['selection_key']}|fact={fact.fact_id}"
            specs.append({
                "answer_key": key, "selection_key": selection["selection_key"],
                "dossier_id": dossier.dossier_id,
                "announced_fact_id": selection.get("announced_fact_id"),
                "eval_fact": fact, "mode": selection["mode"], "slots": selection["slots"],
                "baseline": None, "sample": 0, "material": selection["handoff_text"],
                "temperature": answer_temp, "seed": None,
                "technical_failure": not bool(selection["selection_valid"]),
            })
    for spec in specs:
        fact = spec["eval_fact"]
        spec["request_hash"] = answer_request_hash(
            answer_model, fact.question, spec["material"], spec["temperature"],
            spec["seed"], cfg)
    keys = [spec["answer_key"] for spec in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate answer key; generic answers may have been expanded by rotation")
    return specs


def answer_record_base(spec: dict) -> dict:
    fact = spec["eval_fact"]
    return {
        "schema_version": SCHEMA_VERSION, "answer_key": spec["answer_key"],
        "request_hash": spec["request_hash"], "selection_key": spec["selection_key"],
        "dossier_id": spec["dossier_id"],
        "announced_fact_id": spec["announced_fact_id"], "eval_fact_id": fact.fact_id,
        "query_type": ("announced" if spec["announced_fact_id"] == fact.fact_id else
                       "hidden" if spec["announced_fact_id"] else "unconditioned"),
        "mode": spec["mode"], "slots": spec["slots"], "baseline": spec["baseline"],
        "sample": spec["sample"], "question": fact.question, "golds": list(fact.golds),
        "technical_failure": bool(spec["technical_failure"]),
    }


def run_answer_call(client: LLMClient, spec: dict, cfg: dict) -> dict:
    fact = spec["eval_fact"]
    base = answer_record_base(spec)
    if spec["technical_failure"]:
        return {**base, "raw": "", "pred": "", "em": 0.0, "f1": 0.0,
                "cached": True, "prompt_tokens": 0, "completion_tokens": 0,
                "judge_correct": 0, "judge_parse_ok": True,
                "judge_raw": "TECHNICAL_FAILURE"}
    sealed = hm.SealedHandoff(
        qid=fact.fact_id, question=fact.question, handoff_text=spec["material"],
        mechanism=spec["mode"])
    result = answer_sealed(
        client, sealed, cfg, spec["temperature"], spec["seed"],
        tag=f"fictional_exp5b_answer_{spec['mode']}")
    em, f1 = score_against_golds(result["pred"], list(fact.golds))
    return {**base, **result, "em": em, "f1": f1}


def clone_answer_for_spec(template: dict, spec: dict) -> dict:
    """Attach one response to another logically distinct but identical request.

    Selection keys and announced/hidden roles are evaluator metadata. They do
    not enter the model-facing answer prompt. Reusing one response whenever the
    full request hash matches prevents concurrent cache misses for an identical
    prompt from manufacturing between-arm noise.
    """

    return {**template, **answer_record_base(spec)}


def _answer_signature(row: dict) -> tuple:
    return (str(row.get("raw", "")), str(row.get("pred", "")),
            float(row.get("em", 0.0)), float(row.get("f1", 0.0)))


def generate_answers(client, specs: list[dict], cfg: dict,
                     stored: list[dict]) -> tuple[list[dict], int]:
    specs_by_hash: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        specs_by_hash[spec["request_hash"]].append(spec)

    stored_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in stored:
        if row.get("schema_version") == SCHEMA_VERSION and row.get("request_hash"):
            stored_by_hash[row["request_hash"]].append(row)

    templates: dict[str, dict] = {}
    missing: list[tuple[str, dict]] = []
    for request_hash, group in specs_by_hash.items():
        candidates = stored_by_hash.get(request_hash, [])
        # Earlier versions submitted identical prompts concurrently. If those
        # cache misses raced, their stored responses can disagree even at
        # temperature zero. A conflict is reconciled by reading the now-stable
        # content-addressed cache exactly once.
        signatures = {_answer_signature(row) for row in candidates}
        if candidates and len(signatures) == 1:
            templates[request_hash] = candidates[0]
        else:
            missing.append((request_hash, group[0]))

    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        futures = {
            pool.submit(run_answer_call, client, spec, cfg): (request_hash, spec)
            for request_hash, spec in missing
        }
        for future, (request_hash, _spec) in futures.items():
            template = future.result()
            # Preserve an already-paid judge verdict when the canonical cache
            # response matches one of the formerly racing stored responses.
            match = next((row for row in stored_by_hash.get(request_hash, [])
                          if _answer_signature(row) == _answer_signature(template)), None)
            if match is not None:
                for field in ("judge_correct", "judge_parse_ok", "judge_raw"):
                    if field in match:
                        template[field] = match[field]
            templates[request_hash] = template

    current = {
        spec["answer_key"]: clone_answer_for_spec(templates[spec["request_hash"]], spec)
        for spec in specs
    }
    return [current[spec["answer_key"]] for spec in specs], len(missing)


# ---------------------------------------------------------------------------
# Dossier-clustered analysis


def build_rotation_rows(dossiers, rotations_by_dossier, selections: list[dict],
                        answers: list[dict], slots_list: list[int]) -> list[dict]:
    selection_index = {
        (row["dossier_id"], row["mode"], int(row["slots"]), row.get("announced_fact_id")): row
        for row in selections
    }
    answer_index = {(row.get("selection_key"), row["eval_fact_id"]): row
                    for row in answers if row.get("selection_key")}
    rows = []
    for dossier in dossiers:
        for rotation in rotations_by_dossier[dossier.dossier_id]:
            for slots in slots_list:
                for mode in ("generic", "conditioned"):
                    announced_key = None if mode == "generic" else rotation.target_fact_id
                    selection = selection_index[(dossier.dossier_id, mode, slots, announced_key)]
                    target = answer_index[(selection["selection_key"], rotation.target_fact_id)]
                    hidden = [answer_index[(selection["selection_key"], fact_id)]
                              for fact_id in rotation.hidden_fact_ids]
                    selected = set(selection["selected_fact_ids"])
                    row = {
                        "dossier_id": dossier.dossier_id, "rotation_id": rotation.rotation_id,
                        "announced_fact_id": rotation.target_fact_id, "mode": mode, "slots": slots,
                        "selection_valid": float(bool(selection["selection_valid"])),
                        "selected_count": len(selected),
                        "announced_card_selected": float(rotation.target_fact_id in selected),
                        "hidden_card_selection_rate": (
                            sum(fid in selected for fid in rotation.hidden_fact_ids)
                            / len(rotation.hidden_fact_ids)),
                        "handoff_characters": selection["handoff_characters"],
                        "handoff_words": selection["handoff_words"],
                        "handoff_estimated_tokens": selection["handoff_estimated_tokens"],
                    }
                    for metric in ("em", "f1", "judge_correct"):
                        if metric in target:
                            row[f"immediate_{metric}"] = float(target[metric])
                        values = [float(answer[metric]) for answer in hidden if metric in answer]
                        if values:
                            row[f"reusable_{metric}"] = float(np.mean(values))
                    rows.append(row)
    return rows


OUTCOME_FIELDS = (
    "immediate_em", "immediate_f1", "immediate_judge_correct",
    "reusable_em", "reusable_f1", "reusable_judge_correct",
    "announced_card_selected", "hidden_card_selection_rate", "selection_valid",
    "handoff_characters", "handoff_words", "handoff_estimated_tokens",
)


def field_parts(field: str) -> tuple[str, str]:
    for prefix in ("immediate_", "reusable_"):
        if field.startswith(prefix):
            return prefix[:-1], field[len(prefix):]
    if field in ("announced_card_selected", "hidden_card_selection_rate"):
        return "support", field
    return "channel", field


def dossier_means(rows: list[dict], mode: str, slots: int, field: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        if row["mode"] == mode and int(row["slots"]) == int(slots) and finite(row.get(field)):
            grouped[row["dossier_id"]].append(float(row[field]))
    return {dossier_id: float(np.mean(values)) for dossier_id, values in grouped.items()}


def analyse_rotation_rows(rows: list[dict], slots_list: list[int], cfg: dict):
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    metrics, contrasts = [], []
    for slots in slots_list:
        for field in OUTCOME_FIELDS:
            endpoint, metric = field_parts(field)
            available = {}
            for mode in ("generic", "conditioned"):
                values = dossier_means(rows, mode, slots, field)
                available[mode] = values
                if not values:
                    continue
                mean, lo, hi = bootstrap_ci(
                    np.array(list(values.values()), dtype=float), boot, ci, seed=505)
                metrics.append({
                    "mode": mode, "slots": slots, "endpoint": endpoint, "metric": metric,
                    "n_dossiers": len(values),
                    "n_rotations": sum(1 for row in rows if row["mode"] == mode
                                       and int(row["slots"]) == slots),
                    "mean": mean, "lo": lo, "hi": hi,
                })
            ids = sorted(set(available.get("conditioned", {}))
                         & set(available.get("generic", {})))
            if ids:
                result = paired_bootstrap_delta(
                    np.array([available["conditioned"][i] for i in ids]),
                    np.array([available["generic"][i] for i in ids]),
                    boot, ci, seed=507)
                if len(ids) < 2:
                    result.update({"lo": float("nan"), "hi": float("nan"),
                                   "p_value": float("nan")})
                contrasts.append({
                    "comparison": "conditioned_minus_generic", "slots": slots,
                    "endpoint": endpoint, "metric": metric, **result,
                })
        # Difference in differences.  It is secondary: the paper's trade-off
        # requires immediate gain > 0 AND reusable loss < 0 separately.
        for metric in ("em", "f1", "judge_correct"):
            values = {}
            for mode in ("generic", "conditioned"):
                immediate = dossier_means(rows, mode, slots, f"immediate_{metric}")
                reusable = dossier_means(rows, mode, slots, f"reusable_{metric}")
                values[mode] = {i: immediate[i] - reusable[i]
                                for i in set(immediate) & set(reusable)}
            ids = sorted(set(values["conditioned"]) & set(values["generic"]))
            if ids:
                interaction = np.array([
                    values["conditioned"][i] - values["generic"][i] for i in ids])
                result = paired_bootstrap_delta(
                    interaction, np.zeros_like(interaction), boot, ci, seed=509)
                if len(ids) < 2:
                    result.update({"lo": float("nan"), "hi": float("nan"),
                                   "p_value": float("nan")})
                contrasts.append({
                    "comparison": "specialization_interaction", "slots": slots,
                    "endpoint": "immediate_minus_reusable", "metric": metric, **result,
                })
    return metrics, contrasts


def paired_compliant_contrasts(rows: list[dict], slots_list: list[int], cfg: dict) -> list[dict]:
    """Paired sensitivity: require both arms valid for the same rotation.

    Filtering each arm independently would compare different rotations after a
    schema failure.  Differences are therefore formed within rotation first,
    averaged within dossier second, and only dossiers are bootstrapped.
    """

    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    index = {(r["dossier_id"], r["rotation_id"], int(r["slots"]), r["mode"]): r
             for r in rows}
    out = []
    for slots in slots_list:
        for field in OUTCOME_FIELDS:
            if field in {"selection_valid", "selected_count"}:
                continue
            grouped = defaultdict(list)
            n_pairs = 0
            keys = sorted({(r["dossier_id"], r["rotation_id"])
                           for r in rows if int(r["slots"]) == slots})
            for dossier_id, rotation_id in keys:
                conditioned = index.get((dossier_id, rotation_id, slots, "conditioned"))
                generic = index.get((dossier_id, rotation_id, slots, "generic"))
                if not conditioned or not generic:
                    continue
                if not (conditioned["selection_valid"] and generic["selection_valid"]):
                    continue
                if not (finite(conditioned.get(field)) and finite(generic.get(field))):
                    continue
                grouped[dossier_id].append(float(conditioned[field]) - float(generic[field]))
                n_pairs += 1
            values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
            if not len(values):
                continue
            result = paired_bootstrap_delta(values, np.zeros_like(values), boot, ci, seed=513)
            endpoint, metric = field_parts(field)
            out.append({
                "scope": "paired_compliant", "comparison": "conditioned_minus_generic",
                "slots": slots, "endpoint": endpoint, "metric": metric,
                "n_dossiers": len(values), "n_paired_rotations": n_pairs, **result,
            })
    return out


def analyse_baselines(answers: list[dict], cfg: dict) -> list[dict]:
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    rows = [row for row in answers if row.get("baseline")]
    results = []
    for baseline in ("direct_source", "direct_cards", "closed_book"):
        subset = [row for row in rows if row["baseline"] == baseline]
        for metric in ("em", "f1", "judge_correct"):
            # First average repeated closed-book samples within each fact, then
            # facts within each dossier; only dossiers enter the bootstrap.
            per_fact = defaultdict(list)
            for row in subset:
                if metric in row and finite(row[metric]):
                    per_fact[(row["dossier_id"], row["eval_fact_id"])].append(float(row[metric]))
            per_dossier = defaultdict(list)
            for (dossier_id, _), values in per_fact.items():
                per_dossier[dossier_id].append(float(np.mean(values)))
            values = np.array([np.mean(v) for v in per_dossier.values()], dtype=float)
            if not values.size:
                continue
            mean, lo, hi = bootstrap_ci(values, boot, ci, seed=511)
            results.append({
                "baseline": baseline, "metric": metric, "n_dossiers": len(values),
                "n_fact_sample_rows": len(subset), "mean": mean, "lo": lo, "hi": hi,
            })
    return results


def make_plot(metrics: list[dict], baseline_metrics: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available_judge = any(row["metric"] == "judge_correct" for row in metrics)
    metric = "judge_correct" if available_judge else "f1"
    label = "LLM-judge accuracy" if available_judge else "Token F1"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"generic": "#1b9e77", "conditioned": "#d95f02"}
    for axis, endpoint, title in zip(
            axes, ("immediate", "reusable"),
            ("Announced Question A", "Mean unannounced future questions")):
        for mode in ("generic", "conditioned"):
            rows = sorted([row for row in metrics if row["mode"] == mode
                           and row["endpoint"] == endpoint and row["metric"] == metric],
                          key=lambda row: row["slots"])
            if not rows:
                continue
            x = [row["slots"] for row in rows]
            y = [row["mean"] for row in rows]
            lo = [row["mean"] - row["lo"] for row in rows]
            hi = [row["hi"] - row["mean"] for row in rows]
            axis.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3,
                          color=colors[mode], label=mode)
        direct = next((row for row in baseline_metrics
                       if row["baseline"] == "direct_cards" and row["metric"] == metric), None)
        if direct:
            axis.axhline(direct["mean"], color="#7570b3", linestyle=":",
                         linewidth=1.4, label="all cards")
        axis.set(title=title, xlabel="Evidence-card slots (exact K)", ylim=(-0.03, 1.03))
        axis.grid(alpha=.25)
    axes[0].set_ylabel(label)
    axes[1].legend(fontsize=8)
    fig.suptitle("Fixed-capacity task specialization versus handoff reusability")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def lookup_metric(metrics, mode, slots, endpoint, metric):
    return next((row for row in metrics if row["mode"] == mode and row["slots"] == slots
                 and row["endpoint"] == endpoint and row["metric"] == metric), None)


def lookup_contrast(contrasts, comparison, slots, endpoint, metric):
    return next((row for row in contrasts if row["comparison"] == comparison
                 and row["slots"] == slots and row["endpoint"] == endpoint
                 and row["metric"] == metric), None)


def write_report(path: Path, metrics: list[dict], contrasts: list[dict],
                  baseline_metrics: list[dict], rotation_rows: list[dict], cfg: dict,
                  n_dossiers: int, slots_list: list[int],
                  compliant_contrasts: list[dict]) -> None:
    primary = int(cfg["selection"]["primary_slots"])
    if primary not in slots_list:
        primary = slots_list[0]
    metric = "judge_correct" if any(row["metric"] == "judge_correct" for row in metrics) else "f1"
    metric_label = "judge accuracy" if metric == "judge_correct" else "token F1"
    lines = [
        "# Experiment 5b: fictional fixed-capacity handoffs", "",
        f"- **Design:** {n_dossiers} fictional dossiers; each dossier is the independent unit.",
        f"- **Manipulation:** generic versus one-question-conditioned selection into exactly "
        f"K evidence slots (`K={slots_list}`).",
        "- **Isolation:** hidden questions are never shown to either selector; the fixed answerer "
        "receives only one sealed handoff and one evaluation query.",
        "- **Interpretation:** this is a controlled closed-world evidence-allocation test. The "
        "published natural-prose Experiment 5 supplies the complementary naturalistic result.",
        "", f"## Primary capacity: K={primary}", "",
        f"| Arm | Immediate {metric_label} | Reusable {metric_label} |",
        "|---|---:|---:|",
    ]
    if n_dossiers < 10:
        lines[8:8] = ["> **Pipeline pilot only.** With fewer than ten dossiers, effect sizes are "
                      "diagnostic and inferential intervals/p-values are not interpreted.", ""]
    for mode in ("generic", "conditioned"):
        immediate = lookup_metric(metrics, mode, primary, "immediate", metric)
        reusable = lookup_metric(metrics, mode, primary, "reusable", metric)
        if immediate and reusable:
            lines.append(f"| {mode} | {immediate['mean']:.3f} "
                         f"[{immediate['lo']:.3f}, {immediate['hi']:.3f}] | "
                         f"{reusable['mean']:.3f} [{reusable['lo']:.3f}, {reusable['hi']:.3f}] |")
    lines += ["", "### Conditioned minus generic", "",
              "| Endpoint | Delta | 95% CI | p |", "|---|---:|---:|---:|"]
    for endpoint in ("immediate", "reusable"):
        row = lookup_contrast(contrasts, "conditioned_minus_generic", primary, endpoint, metric)
        if row:
            p_text = "<0.001" if float(row["p_value"]) < .001 else f"{row['p_value']:.3f}"
            lines.append(f"| {endpoint} | {row['delta']:+.3f} | "
                         f"[{row['lo']:+.3f}, {row['hi']:+.3f}] | {p_text} |")
    interaction = lookup_contrast(
        contrasts, "specialization_interaction", primary,
        "immediate_minus_reusable", metric)
    if interaction:
        lines += ["", f"Specialization interaction: **{interaction['delta']:+.3f}** "
                  f"[{interaction['lo']:+.3f}, {interaction['hi']:+.3f}]. "
                  "This is secondary: a trade-off requires a positive immediate delta and a "
                  "negative reusable delta separately."]
    lines += ["", "## Baseline checks", "",
              "| Material | Metric | Mean | 95% CI |", "|---|---|---:|---:|"]
    for row in baseline_metrics:
        if row["metric"] in (metric, "f1"):
            lines.append(f"| {row['baseline']} | {row['metric']} | {row['mean']:.3f} | "
                         f"[{row['lo']:.3f}, {row['hi']:.3f}] |")
    valid = [row["selection_valid"] for row in rotation_rows]
    exact = [float(row["selected_count"] == row["slots"]) for row in rotation_rows]
    lines += ["", "## Integrity checks", "",
              f"- Valid selector outputs: {np.mean(valid):.3f}.",
              f"- Rotation rows with exactly K selected cards: {np.mean(exact):.3f}.",
              "- Confidence intervals bootstrap dossier means, not question rotations.",
              "- Invalid selections remain in the intention-to-treat result as zero-answer "
              "technical failures; inspect `selections.jsonl` for parse diagnostics.",
              "", "## Artifacts", "",
              "- `metrics.csv`: dossier-clustered arm estimates.",
              "- `contrasts.csv`: paired conditioned-minus-generic effects.",
              "- `rotation_rows.csv`: auditable per-rotation outcomes and exact selections.",
              "- `baseline_metrics.csv`: source, full-card, and closed-book controls.",
              "- `fixed_capacity_generalization.png`: immediate/reusable capacity curves.", ""]
    ceiling = next((r for r in compliant_contrasts
                    if r["slots"] == max(slots_list) and r["endpoint"] == "reusable"
                    and r["metric"] == metric), None)
    if ceiling:
        lines[lines.index("## Artifacts") : lines.index("## Artifacts")] = [
            "## Paired-compliant no-selection control", "",
            f"At K={max(slots_list)}, valid conditioned and generic packets both contain all "
            f"six cards. Their reusable {metric_label} delta is **{ceiling['delta']:+.3f}** "
            f"[{ceiling['lo']:+.3f}, {ceiling['hi']:+.3f}] over "
            f"{ceiling['n_paired_rotations']} paired rotations.", "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def dry_run_report(dossiers, rotations_by_dossier, slots_list: list[int], cfg: dict) -> dict:
    n_facts = sum(len(d.facts) for d in dossiers)
    n_rotations = sum(len(rotations_by_dossier[d.dossier_id]) for d in dossiers)
    selection_calls = len(slots_list) * (len(dossiers) + n_rotations)
    selection_worst = selection_calls * int(cfg["selection"]["max_parse_attempts"])
    handoff_answer_calls = len(slots_list) * sum(
        len(d.facts) * (1 + len(rotations_by_dossier[d.dossier_id])) for d in dossiers)
    baseline_answer_calls = n_facts * (2 + int(cfg["baselines"]["closed_book_samples"]))
    answer_calls = handoff_answer_calls + baseline_answer_calls
    judge_calls = answer_calls if cfg.get("judge", {}).get("enabled", False) else 0
    return {
        "dry_run": True, "writes": 0, "dossiers": len(dossiers), "facts": n_facts,
        "rotations": n_rotations, "slots": slots_list,
        "selection_calls_expected": selection_calls,
        "selection_calls_worst_case_with_parse_retries": selection_worst,
        "handoff_answer_calls": handoff_answer_calls,
        "baseline_answer_calls": baseline_answer_calls,
        "judge_calls": judge_calls,
        "api_calls_expected": selection_calls + answer_calls + judge_calls,
        "api_calls_worst_case": selection_worst + answer_calls + judge_calls,
        "cost_cap_usd": float(cfg["cost"]["cap_usd"]),
        "effective_n": len(dossiers),
    }


def analyse_and_write(dossiers, cards, rotations, selections, answers,
                      slots_list, cfg, result_root: Path, source_manifest: dict) -> None:
    rotation_rows = build_rotation_rows(
        dossiers, rotations, selections, answers, slots_list)
    metrics, contrasts = analyse_rotation_rows(rotation_rows, slots_list, cfg)
    compliant_contrasts = paired_compliant_contrasts(rotation_rows, slots_list, cfg)
    baseline_metrics = analyse_baselines(answers, cfg)
    result_root.mkdir(parents=True, exist_ok=True)
    write_csv(result_root / "rotation_rows.csv", rotation_rows)
    write_csv(result_root / "metrics.csv", metrics)
    write_csv(result_root / "contrasts.csv", contrasts)
    write_csv(result_root / "contrasts_paired_compliant.csv", compliant_contrasts)
    write_csv(result_root / "baseline_metrics.csv", baseline_metrics)
    make_plot(metrics, baseline_metrics, result_root / "fixed_capacity_generalization.png")
    write_report(result_root / "report.md", metrics, contrasts, baseline_metrics,
                 rotation_rows, cfg, len(dossiers), slots_list, compliant_contrasts)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": source_manifest,
        "source_file": cfg["dataset"]["items_jsonl"],
        "source_file_hash": stable_hash(
            (ROOT / cfg["dataset"]["items_jsonl"]).read_text(encoding="utf-8")),
        "selector_model": cfg["model"]["id"],
        "answerer_model": (cfg.get("answerer") or cfg["model"])["id"],
        "judge_model": cfg.get("judge", {}).get("model_id"),
        "n_dossiers": len(dossiers), "slots": slots_list,
        "rotations_per_dossier": int(cfg["dataset"]["rotations_per_dossier"]),
        "cluster": "dossier",
    }
    (result_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Command line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fictional_summary_generalization_config.yaml")
    parser.add_argument("--limit", type=int, help="bounded dossier pilot")
    parser.add_argument("--slots", help="comma-separated subset of configured K values")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true", help="print call plan; write nothing")
    parser.add_argument("--analyse-only", action="store_true", help="reanalyse stored rows; no API calls")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.analyse_only:
        parser.error("--dry-run and --analyse-only are mutually exclusive")
    cfg = load_config(ROOT / args.config)
    if args.concurrency is not None:
        if args.concurrency < 1:
            parser.error("--concurrency must be >= 1")
        cfg["runtime"]["concurrency"] = args.concurrency
    if args.no_judge:
        cfg["judge"]["enabled"] = False
    slots_list = parse_slots(cfg, args.slots)
    source_manifest, dossiers, cards, rotations = load_design(cfg, args.limit)
    specs = selection_specs(dossiers, cards, rotations, slots_list, cfg)
    if args.dry_run:
        # Deliberately before constructing an LLMClient or output Path.mkdir:
        # dry-run is a pure read/compute/print operation.
        print(json.dumps(dry_run_report(dossiers, rotations, slots_list, cfg), indent=2))
        return 0

    n = len(dossiers)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    selection_path = run_root / "selections.jsonl"
    answer_path = run_root / "answers.jsonl"

    if args.analyse_only:
        stored_selections = read_jsonl(selection_path)
        expected_keys = {spec["selection_key"] for spec in specs}
        selections = [row for row in stored_selections if row.get("selection_key") in expected_keys]
        if {row["selection_key"] for row in selections} != expected_keys:
            missing = expected_keys - {row["selection_key"] for row in selections}
            raise RuntimeError(f"analyse-only is missing {len(missing)} selection rows")
        answer_specs_now = answer_specs(dossiers, cards, selections, cfg)
        expected_answers = {spec["answer_key"] for spec in answer_specs_now}
        answers = [row for row in read_jsonl(answer_path)
                   if row.get("answer_key") in expected_answers]
        if {row["answer_key"] for row in answers} != expected_answers:
            missing = expected_answers - {row["answer_key"] for row in answers}
            raise RuntimeError(f"analyse-only is missing {len(missing)} answer rows")
        analyse_and_write(dossiers, cards, rotations, selections, answers,
                          slots_list, cfg, result_root, source_manifest)
        print(f"[fictional-exp5b] reanalysed {len(selections)} selections / {len(answers)} answers")
        return 0

    selector, answerer = make_clients(cfg)
    stored_selections = read_jsonl(selection_path)
    selections, generated = generate_selections(selector, specs, cfg, stored_selections)
    # Preserve rows for other configured K subsets while replacing current keys.
    selection_index = {row.get("selection_key"): row for row in stored_selections}
    selection_index.update({row["selection_key"]: row for row in selections})
    write_jsonl(selection_path, sorted(selection_index.values(),
                                       key=lambda row: str(row.get("selection_key"))))
    invalid = sum(not row["selection_valid"] for row in selections)
    print(f"[fictional-exp5b:selection] {len(selections)} rows; {generated} calls this pass; "
          f"{invalid} invalid after retries")

    a_specs = answer_specs(dossiers, cards, selections, cfg)
    stored_answers = read_jsonl(answer_path)
    answers, answered = generate_answers(answerer, a_specs, cfg, stored_answers)
    answer_index = {row.get("answer_key"): row for row in stored_answers}
    answer_index.update({row["answer_key"]: row for row in answers})
    write_jsonl(answer_path, sorted(answer_index.values(),
                                    key=lambda row: str(row.get("answer_key"))))
    judge_client = add_judge(answers, cfg, tag="fictional_exp5b_judge")
    answer_index.update({row["answer_key"]: row for row in answers})
    write_jsonl(answer_path, sorted(answer_index.values(),
                                    key=lambda row: str(row.get("answer_key"))))
    print(f"[fictional-exp5b:answer] {len(answers)} rows; {answered} calls this pass")

    analyse_and_write(dossiers, cards, rotations, selections, answers,
                      slots_list, cfg, result_root, source_manifest)
    print("[fictional-exp5b:cost] " + json.dumps(selector.ledger.summary()))
    if judge_client is not None:
        print("[fictional-exp5b:judge-cost] " + json.dumps(judge_client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
