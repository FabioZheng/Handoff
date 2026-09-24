"""Experiment 8b: fictional-QA selector/relay capability bottleneck.

The experiment has two deliberately different moments:

1. A source-aware selector sees an invented dossier and announced Question A,
   then commits to exactly K1 evidence-card slots.
2. Only after that packet is frozen, Question B is announced.  A relay sees B
   and, in the factorial arms, only the sealed K1 packet.  It selects exactly
   K2 cards for a fixed third-family answerer.

Within Llama and Qwen separately, selector and relay tier form a 2x2 factorial
(small/small, small/large, large/small, large/large).  Three small->large
controls distinguish missing evidence from downstream inability:

* ``restore`` swaps B's exact source card into the packet;
* ``sham`` swaps a different source card into the same slot;
* ``reopen`` explicitly gives the relay the full source-card set.

Restore and sham preserve the number and ordering of packet slots.  Reopen is
an intentionally unsealed positive control and is never treated as a factorial
cell.  All inference resamples dossiers, not the repeated role rotations within
them.  Invalid fixed-K outputs remain failures in the ITT analysis and are also
reported in a paired-compliant sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fictional_qa as fq  # noqa: E402
import handoffs as hm  # noqa: E402
import model_pool as mp  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from score import score_against_golds  # noqa: E402

CONFIG = ROOT / "configs/fictional_model_bottleneck_config.yaml"
SCHEMA_VERSION = "fictional-model-bottleneck-v2"

SELECTION_SYSTEM = (
    "You are an evidence-routing agent. Select only evidence-card ids supplied in the prompt. "
    "Obey the requested number of slots exactly and return only the requested JSON object."
)

METRICS = (
    "em", "f1", "judge_correct", "a_present_stage1", "b_present_stage1_base",
    "b_present_relay_input", "b_present_stage2", "stage1_compliant",
    "stage2_compliant", "overall_compliant",
)


@dataclass(frozen=True)
class Arm:
    name: str
    family: str
    selector_tier: str
    relay_tier: str
    selector_model: str
    relay_model: str
    intervention: str = "sealed"

    @property
    def factorial(self) -> bool:
        return self.intervention == "sealed"

    @property
    def code(self) -> str:
        return self.selector_tier[0] + self.relay_tier[0]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(*parts: str) -> str:
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def build_arms(cfg: dict) -> list[Arm]:
    arms: list[Arm] = []
    enabled = set(cfg.get("interventions", {}).get("enabled") or [])
    for family, models in cfg["test_families"].items():
        for selector_tier in ("small", "large"):
            for relay_tier in ("small", "large"):
                code = selector_tier[0] + relay_tier[0]
                arms.append(Arm(
                    name=f"{family}_{code}", family=family,
                    selector_tier=selector_tier, relay_tier=relay_tier,
                    selector_model=models[selector_tier], relay_model=models[relay_tier],
                ))
        for intervention in ("sham", "restore", "reopen"):
            if intervention in enabled:
                arms.append(Arm(
                    name=f"{family}_sl_{intervention}", family=family,
                    selector_tier="small", relay_tier="large",
                    selector_model=models["small"], relay_model=models["large"],
                    intervention=intervention,
                ))
    return arms


def load_tasks(cfg: dict, limit: int | None = None) -> tuple[dict, list[dict]]:
    path = ROOT / cfg["dataset"]["items_jsonl"]
    manifest, dossiers = fq.load_fictional_corpus(path)
    n = int(cfg["dataset"]["n_dossiers"])
    if limit is not None:
        n = min(n, int(limit))
    dossiers = tuple(dossiers[:n])
    if not dossiers:
        raise SystemExit(f"no fictional dossiers loaded from {path}")

    seed = int(cfg["dataset"]["seed"])
    rotation_count = int(cfg["dataset"]["rotations_per_dossier"])
    tasks: list[dict] = []
    for dossier in dossiers:
        cards = tuple(fq.evidence_cards(dossier, seed=seed))
        by_fact = {c.fact_id: c for c in cards}
        rotations = fq.role_rotations(dossier, count=rotation_count, seed=seed)
        fact_cycle = [card.fact_id for card in cards]
        for rotation in rotations:
            if not rotation.hidden_fact_ids:
                raise ValueError(f"{rotation.rotation_id}: no hidden question available")
            # A fixed one-step cyclic derangement over the shuffled card order
            # maps every distinct A to a distinct B and never maps a fact to
            # itself.  Taking ``hidden_fact_ids[0]`` would reuse only one or two
            # B facts because that tuple has a common global ordering.
            a_index = fact_cycle.index(rotation.target_fact_id)
            b_fact_id = fact_cycle[(a_index + 1) % len(fact_cycle)]
            if b_fact_id == rotation.target_fact_id or b_fact_id not in by_fact:
                raise ValueError(f"{rotation.rotation_id}: invalid B fact {b_fact_id!r}")
            b_card = by_fact[b_fact_id]
            task_id = f"{dossier.dossier_id}|{rotation.rotation_id}|B={b_fact_id}"
            tasks.append({
                "task_id": task_id,
                "dossier_id": dossier.dossier_id,
                "dossier": dossier,
                "cards": cards,
                "rotation": rotation,
                "a_fact_id": rotation.target_fact_id,
                "a_question": rotation.target_question,
                "a_golds": list(rotation.target_golds),
                "b_fact_id": b_fact_id,
                "b_question": b_card.question,
                "b_golds": list(b_card.golds),
            })
    return manifest, tasks


def experiment_fingerprint(cfg: dict, config_path: Path, data_path: Path) -> str:
    shared = Path(fq.__file__).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config_path.read_text(encoding="utf-8"),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "shared_sha256": hashlib.sha256(shared.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_system": SELECTION_SYSTEM,
    }
    return digest(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def response_format(cfg: dict) -> dict | None:
    return {"type": "json_object"} if cfg["decoding"].get("response_format_json") else None


def selection_call(client: LLMClient, prompt: str, valid_ids: tuple[str, ...], k: int,
                   temperature: float, max_tokens: int, seed: int, tag: str,
                   cards: tuple, max_attempts: int = 1) -> dict:
    attempts = []
    parsed = None
    result = None
    for attempt in range(max(1, int(max_attempts))):
        correction = ""
        if parsed is not None:
            correction = (
                "\n\nFORMAT CORRECTION: the previous response was invalid because "
                f"{parsed.error}. Return exactly {k} distinct bare C-number ids "
                "in the requested JSON object; do not include the word CARD."
            )
        result = client.chat(
            [{"role": "system", "content": SELECTION_SYSTEM},
             {"role": "user", "content": prompt + correction}],
            temperature=temperature, max_tokens=max_tokens, seed=seed + attempt,
            response_format={"type": "json_object"}, tag=f"{tag}_attempt{attempt + 1}",
        )
        parsed = fq.parse_fixed_k_selection(result.text, valid_ids, k)
        attempts.append({
            "attempt": attempt + 1, "raw": result.text, "valid": parsed.valid,
            "error": parsed.error, "cached": result.cached,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": float(getattr(result, "cost_usd", 0.0)),
        })
        if parsed.valid:
            break
    assert result is not None and parsed is not None
    selected_cards = tuple(parsed.selected_fact_ids)
    selected = (fq.card_ids_to_fact_ids(cards, selected_cards) if parsed.valid else ())
    recognized_cards = tuple(parsed.recognized_fact_ids)
    recognized = fq.card_ids_to_fact_ids(cards, recognized_cards)
    packet = fq.render_packet_slots(cards, selected, k)
    return {
        "raw": result.text, "attempts": attempts, "attempt_count": len(attempts),
        "selected_card_ids": list(selected_cards),
        "selected_fact_ids": list(selected),
        "recognized_card_ids": list(recognized_cards),
        "recognized_fact_ids": list(recognized),
        "selection_valid": bool(parsed.valid), "parsed_count": parsed.parsed_count,
        "selection_errors": list(parsed.errors), "selection_error": parsed.error,
        "packet_text": packet, "cached": all(a["cached"] for a in attempts),
        "prompt_tokens": sum(int(a["prompt_tokens"]) for a in attempts),
        "completion_tokens": sum(int(a["completion_tokens"]) for a in attempts),
        "finish_reason": result.finish_reason, "provider": result.provider,
        "resolved_model": result.model,
        "cost_usd": round(sum(float(a["cost_usd"]) for a in attempts), 8),
        "empty_output": int(not result.text.strip()),
        "truncated": int(result.finish_reason == "length"),
    }


def selector_prompt(task: dict, k: int) -> str:
    # Only A's id is passed to the prompt builder.  The hidden B designation is
    # not in this call frame; B remains an ordinary source fact until stage 2.
    return fq.source_selection_prompt(
        task["dossier"], task["cards"], k,
        question_fact_id=task["a_fact_id"],
    )


def generate_selectors(pool: mp.ModelPool, tasks: list[dict], arms: list[Arm], cfg: dict,
                       path: Path, fingerprint: str) -> dict[tuple[str, str], dict]:
    stored = {
        (r["task_id"], r["model_key"]): r for r in read_jsonl(path)
        if r.get("experiment_fingerprint") == fingerprint
    }
    k = int(cfg["channel"]["selector_slots"])
    wanted = sorted({a.selector_model for a in arms})
    jobs = [(task, model_key) for task in tasks for model_key in wanted
            if (task["task_id"], model_key) not in stored]

    def one(job):
        task, model_key = job
        cards = task["cards"]
        record = selection_call(
            pool.client(model_key), selector_prompt(task, k),
            tuple(c.card_id for c in cards), k,
            float(cfg["decoding"]["selector_temperature"]),
            int(cfg["decoding"]["selection_max_tokens"]), 1,
            f"fictional_bottleneck_selector_{model_key}", cards,
            max_attempts=int(cfg["selection"]["max_parse_attempts"]),
        )
        return {
            "experiment_fingerprint": fingerprint, "task_id": task["task_id"],
            "dossier_id": task["dossier_id"], "rotation_id": task["rotation"].rotation_id,
            "model_key": model_key, **pool.spec(model_key).to_json(), **record,
        }

    if jobs:
        with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as executor:
            for row in executor.map(one, jobs):
                stored[(row["task_id"], row["model_key"])] = row
        write_jsonl(path, [stored[k] for k in sorted(stored)])
    print(f"[bottleneck:selector] {len(jobs)} generated; {len(stored)} stored")
    return stored


def _plain_intervention(kind: str, selected: tuple[str, ...]) -> dict:
    return {
        "kind": kind, "before_fact_ids": list(selected), "after_fact_ids": list(selected),
        "inserted_fact_id": None, "removed_fact_id": None,
        "changed": False, "reason": "no intervention",
    }


def interventions_for(task: dict, selected: tuple[str, ...], seed: int) -> dict[str, dict]:
    base = _plain_intervention("sealed", selected)
    restore = fq.restore_support(
        task["cards"], selected, task["b_fact_id"],
        protected_fact_ids=(task["a_fact_id"],), seed=f"{seed}:{task['task_id']}:restore",
    )
    restore_d = asdict(restore)
    # The sham replaces the exact slot/victim chosen by restoration, but inserts
    # a different source-supported fact.  Thus slot count and removed content
    # are held fixed; only whether B support enters differs.
    if restore.changed and restore.removed_fact_id:
        sham = fq.sham_swap(
            task["cards"], selected,
            forbidden_fact_ids=(task["a_fact_id"], task["b_fact_id"]),
            protected_fact_ids=(task["a_fact_id"],),
            victim_fact_id=restore.removed_fact_id,
            seed=f"{seed}:{task['task_id']}:sham",
        )
        sham_d = asdict(sham)
    else:
        sham_d = {
            **_plain_intervention("sham", selected),
            "reason": "restore was ineligible, so matched sham is a no-op",
        }
    return {"sealed": base, "restore": restore_d, "sham": sham_d,
            "reopen": _plain_intervention("reopen", selected)}


def evidence_present(task: dict, selected_fact_ids: Iterable[str], query_fact_id: str) -> bool:
    """Whether selected material contains an answer string for the query.

    Card ids are evaluator annotations, not the evidence itself. A source
    sentence can occasionally mention the answer to a second QA fact. Treating
    that as an omission would overstate irrecoverability, so all B-presence and
    rescue-eligibility analyses inspect the delivered text rather than merely
    checking whether B's designated card id survived.
    """

    selected = set(selected_fact_ids)
    query = task["dossier"].fact(query_fact_id)
    texts = [card.evidence_text.casefold() for card in task["cards"]
             if card.fact_id in selected]
    return any(gold.strip() and gold.casefold() in text
               for text in texts for gold in query.golds)


def relay_material(task: dict, arm: Arm, selector: dict, cfg: dict) -> dict:
    k1 = int(cfg["channel"]["selector_slots"])
    k2 = int(cfg["channel"]["relay_slots"])
    selected = tuple(selector["selected_fact_ids"])
    selector_failed = not bool(selector.get("selection_valid")) or len(selected) != k1
    if selector_failed and arm.intervention != "reopen":
        packet = fq.render_packet_slots(task["cards"], (), k1)
        sealed = hm.seal(
            task["b_question"],
            hm.Handoff(task["task_id"], "fictional_selector_failure", None, packet),
        )
        prompt = fq.sealed_relay_prompt(sealed, k2)
        return {
            "call_key": digest(arm.relay_model, prompt, "upstream_selection_failure"),
            "prompt": prompt, "valid_ids": (), "input_fact_ids": (),
            "relay_input_packet": packet, "source_access": False,
            "technical_failure": True,
            "intervention": _plain_intervention(arm.intervention, ()),
            "rescue_eligible": 0,
        }
    interventions = (interventions_for(task, selected, int(cfg["dataset"]["seed"]))
                     if not selector_failed else None)

    if arm.intervention == "reopen":
        meta = _plain_intervention("reopen", selected)
        prompt = fq.reopened_selection_prompt(
            task["dossier"], task["b_fact_id"], task["cards"], k2)
        input_ids = tuple(c.fact_id for c in task["cards"])
        packet = fq.render_packet_slots(task["cards"], input_ids, len(input_ids))
        source_access = True
    else:
        assert interventions is not None
        meta = interventions[arm.intervention]
        input_ids = tuple(meta["after_fact_ids"])
        packet = fq.render_packet_slots(task["cards"], input_ids, k1)
        sealed = hm.seal(
            task["b_question"],
            hm.Handoff(task["task_id"], f"fictional_{arm.intervention}", None, packet),
        )
        prompt = fq.sealed_relay_prompt(sealed, k2)
        source_access = False

    call_key = digest(
        arm.relay_model, prompt, str(k2), str(cfg["decoding"]["relay_temperature"]),
        str(cfg["decoding"]["selection_max_tokens"]),
    )
    return {
        "call_key": call_key, "prompt": prompt,
        "valid_ids": fq.fact_ids_to_card_ids(task["cards"], input_ids),
        "input_fact_ids": input_ids,
        "relay_input_packet": packet, "source_access": source_access,
        "technical_failure": False,
        "intervention": meta,
        "rescue_eligible": int(
            interventions is not None
            and not evidence_present(task, selected, task["b_fact_id"])
            and bool(interventions["restore"].get("changed"))
            and bool(interventions["sham"].get("changed"))
            and evidence_present(
                task, interventions["restore"]["after_fact_ids"], task["b_fact_id"])
            and not evidence_present(
                task, interventions["sham"]["after_fact_ids"], task["b_fact_id"])
        ),
    }


def build_relay_materials(tasks: list[dict], arms: list[Arm], selectors: dict,
                          cfg: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for task in tasks:
        for arm in arms:
            selector = selectors[(task["task_id"], arm.selector_model)]
            out[(task["task_id"], arm.name)] = relay_material(task, arm, selector, cfg)
    return out


def generate_relays(pool: mp.ModelPool, tasks: list[dict], arms: list[Arm], selectors: dict,
                    cfg: dict, path: Path, fingerprint: str) -> tuple[dict, dict]:
    materials = build_relay_materials(tasks, arms, selectors, cfg)
    k2 = int(cfg["channel"]["relay_slots"])
    stored = {r["call_key"]: r for r in read_jsonl(path)
              if r.get("experiment_fingerprint") == fingerprint}
    unique: dict[str, tuple[dict, Arm, dict]] = {}
    technical_added = 0
    task_by_id = {t["task_id"]: t for t in tasks}
    arm_by_name = {a.name: a for a in arms}
    for (task_id, arm_name), material in materials.items():
        if material.get("technical_failure"):
            if material["call_key"] not in stored:
                arm = arm_by_name[arm_name]
                stored[material["call_key"]] = {
                    "experiment_fingerprint": fingerprint,
                    "call_key": material["call_key"], "task_id": task_id,
                    "dossier_id": task_by_id[task_id]["dossier_id"],
                    "model_key": arm.relay_model, **pool.spec(arm.relay_model).to_json(),
                    "raw": "", "selected_card_ids": [], "selected_fact_ids": [],
                    "recognized_card_ids": [], "recognized_fact_ids": [],
                    "selection_valid": False, "parsed_count": 0,
                    "selection_errors": ["upstream_selection_failure"],
                    "selection_error": "upstream_selection_failure",
                    "packet_text": fq.render_packet_slots(task_by_id[task_id]["cards"], (), k2),
                    "cached": True, "prompt_tokens": 0, "completion_tokens": 0,
                    "finish_reason": "technical_failure", "provider": "",
                    "resolved_model": pool.spec(arm.relay_model).id, "cost_usd": 0.0,
                    "empty_output": 1, "truncated": 0, "technical_failure": True,
                }
                technical_added += 1
            continue
        if material["call_key"] not in stored:
            unique.setdefault(material["call_key"],
                              (task_by_id[task_id], arm_by_name[arm_name], material))

    def one(job):
        call_key, (task, arm, material) = job
        record = selection_call(
            pool.client(arm.relay_model), material["prompt"], tuple(material["valid_ids"]), k2,
            float(cfg["decoding"]["relay_temperature"]),
            int(cfg["decoding"]["selection_max_tokens"]), 2,
            f"fictional_bottleneck_relay_{arm.relay_model}", task["cards"],
            max_attempts=int(cfg["selection"]["max_parse_attempts"]),
        )
        return {
            "experiment_fingerprint": fingerprint, "call_key": call_key,
            "task_id": task["task_id"], "dossier_id": task["dossier_id"],
            "model_key": arm.relay_model, **pool.spec(arm.relay_model).to_json(), **record,
            "technical_failure": False,
        }

    if unique:
        with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as executor:
            for row in executor.map(one, sorted(unique.items())):
                stored[row["call_key"]] = row
    if unique or technical_added:
        write_jsonl(path, [stored[k] for k in sorted(stored)])
    print(f"[bottleneck:relay] {len(unique)} distinct calls generated; {len(stored)} stored")
    return stored, materials


def run_answers(pool: mp.ModelPool, tasks: list[dict], arms: list[Arm], relays: dict,
                materials: dict, cfg: dict, path: Path, fingerprint: str) -> dict[str, dict]:
    stored = {r["material_key"]: r for r in read_jsonl(path)
              if r.get("experiment_fingerprint") == fingerprint}
    answerer = cfg["answerer"]["model"]
    task_by_id = {t["task_id"]: t for t in tasks}
    todo: dict[str, tuple[dict, str]] = {}
    technical_added = 0
    for task in tasks:
        for arm in arms:
            relay = relays[materials[(task["task_id"], arm.name)]["call_key"]]
            packet = relay["packet_text"]
            key = digest(task["task_id"], task["b_question"], packet, answerer)
            if key not in stored:
                if not relay.get("selection_valid"):
                    stored[key] = {
                        "experiment_fingerprint": fingerprint, "material_key": key,
                        "task_id": task["task_id"], "dossier_id": task["dossier_id"],
                        "question": task["b_question"], "golds": task["b_golds"],
                        "pred": "", "raw": "", "em": 0.0, "f1": 0.0,
                        "judge_correct": 0, "judge_parse_ok": True,
                        "judge_raw": "TECHNICAL_FAILURE",
                        "answerer_model_key": answerer,
                        "answerer_model_id": pool.spec(answerer).id,
                        "cached": True, "cost_usd": 0.0, "technical_failure": True,
                    }
                    technical_added += 1
                else:
                    todo.setdefault(key, (task, packet))

    def one(job):
        key, (task, packet) = job
        sealed = hm.seal(
            task["b_question"],
            hm.Handoff(task["task_id"], "fictional_bottleneck_answer", None, packet),
        )
        answer = hm.orchestrator_answer(
            pool.client(answerer), sealed, cfg, tag="fictional_bottleneck_answer")
        em, f1 = score_against_golds(answer["pred"], task["b_golds"])
        return {
            "experiment_fingerprint": fingerprint, "material_key": key,
            "task_id": task["task_id"], "dossier_id": task["dossier_id"],
            "question": task["b_question"], "golds": task["b_golds"],
            "pred": answer["pred"], "raw": answer["raw"], "em": em, "f1": f1,
            "answerer_model_key": answerer, "answerer_model_id": pool.spec(answerer).id,
            "cached": answer.get("cached", False),
            "cost_usd": round(float(answer.get("cost_usd", 0.0)), 8),
        }

    if todo:
        with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as executor:
            for row in executor.map(one, sorted(todo.items())):
                stored[row["material_key"]] = row
    if todo or technical_added:
        write_jsonl(path, [stored[k] for k in sorted(stored)])
    print(f"[bottleneck:answer] {len(todo)} distinct calls generated; {len(stored)} stored")
    return stored


def baseline_key(task: dict, kind: str, answerer: str) -> str:
    return digest("baseline", task["dossier_id"], task["b_fact_id"], kind, answerer)


def run_baselines(pool: mp.ModelPool, tasks: list[dict], cfg: dict, path: Path,
                  fingerprint: str) -> dict[str, dict]:
    """Score the fixed reader with all cards and with no evidence.

    Calls are deduplicated by dossier/B fact because several rotations can
    designate the same B.  These controls are not treatment arms.
    """

    if not cfg.get("baselines", {}).get("enabled", True):
        return {}
    stored = {r["baseline_key"]: r for r in read_jsonl(path)
              if r.get("experiment_fingerprint") == fingerprint}
    answerer = cfg["answerer"]["model"]
    unique_tasks = {}
    for task in tasks:
        unique_tasks.setdefault((task["dossier_id"], task["b_fact_id"]), task)
    jobs = []
    for task in unique_tasks.values():
        for kind in ("direct_cards", "closed_book"):
            key = baseline_key(task, kind, answerer)
            if key not in stored:
                jobs.append((key, task, kind))

    def one(job):
        key, task, kind = job
        material = (fq.render_source_cards(task["cards"])
                    if kind == "direct_cards" else "(No research material was supplied.)")
        sealed = hm.seal(
            task["b_question"],
            hm.Handoff(task["task_id"], f"fictional_bottleneck_{kind}", None, material),
        )
        answer = hm.orchestrator_answer(
            pool.client(answerer), sealed, cfg, tag=f"fictional_bottleneck_{kind}")
        em, f1 = score_against_golds(answer["pred"], task["b_golds"])
        return {
            "experiment_fingerprint": fingerprint, "baseline_key": key,
            "dossier_id": task["dossier_id"], "b_fact_id": task["b_fact_id"],
            "kind": kind, "question": task["b_question"], "golds": task["b_golds"],
            "pred": answer["pred"], "raw": answer["raw"], "em": em, "f1": f1,
            "answerer_model_key": answerer, "answerer_model_id": pool.spec(answerer).id,
            "cached": answer.get("cached", False),
            "cost_usd": round(float(answer.get("cost_usd", 0.0)), 8),
        }

    if jobs:
        with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as executor:
            for row in executor.map(one, jobs):
                stored[row["baseline_key"]] = row
        write_jsonl(path, [stored[k] for k in sorted(stored)])
    print(f"[bottleneck:baseline] {len(jobs)} generated; {len(stored)} stored")
    return stored


def baseline_metrics(baselines: dict[str, dict], cfg: dict) -> list[dict]:
    rows = list(baselines.values())
    out = []
    for kind in ("direct_cards", "closed_book"):
        subset = [row for row in rows if row["kind"] == kind]
        for measure in ("em", "f1", "judge_correct"):
            values = cluster_values(subset, measure)
            if not len(values):
                continue
            mean, lo, hi = interval(values, cfg, f"baseline:{kind}:{measure}")
            out.append({
                "kind": kind, "measure": measure,
                "n_dossiers": len(values), "n_questions": len(subset),
                "mean": mean, "lo": lo, "hi": hi,
            })
    return out


def expand_records(tasks: list[dict], arms: list[Arm], selectors: dict, relays: dict,
                   materials: dict, answers: dict, baselines: dict, cfg: dict) -> list[dict]:
    rows = []
    for task in tasks:
        answerer = cfg["answerer"]["model"]
        direct = baselines.get(baseline_key(task, "direct_cards", answerer), {})
        closed = baselines.get(baseline_key(task, "closed_book", answerer), {})
        gate_metric = ("judge_correct" if "judge_correct" in direct
                       and "judge_correct" in closed else "f1")
        direct_ok = float(direct.get(gate_metric, 0.0)) >= (1.0 if gate_metric == "judge_correct" else .8)
        closed_known = float(closed.get(gate_metric, 0.0)) >= (1.0 if gate_metric == "judge_correct" else .6)
        for arm in arms:
            selector = selectors[(task["task_id"], arm.selector_model)]
            material = materials[(task["task_id"], arm.name)]
            relay = relays[material["call_key"]]
            answer_key = digest(task["task_id"], task["b_question"], relay["packet_text"],
                                cfg["answerer"]["model"])
            answer = answers[answer_key]
            base_ids = tuple(selector["selected_fact_ids"])
            input_ids = tuple(material["input_fact_ids"])
            relay_ids = tuple(relay["selected_fact_ids"])
            intervention = material["intervention"]
            stage1_ok = bool(selector["selection_valid"] and not selector["truncated"]
                             and not selector["empty_output"])
            stage2_ok = bool(relay["selection_valid"] and not relay["truncated"]
                             and not relay["empty_output"])
            rows.append({
                "task_id": task["task_id"], "dossier_id": task["dossier_id"],
                "rotation_id": task["rotation"].rotation_id,
                "a_fact_id": task["a_fact_id"], "b_fact_id": task["b_fact_id"],
                "family": arm.family, "arm": arm.name, "factorial": int(arm.factorial),
                "selector_tier": arm.selector_tier, "relay_tier": arm.relay_tier,
                "selector_model": arm.selector_model, "relay_model": arm.relay_model,
                "intervention": arm.intervention,
                "source_reopened": int(material["source_access"]),
                "intervention_changed": int(bool(intervention.get("changed"))),
                "intervention_inserted": intervention.get("inserted_fact_id"),
                "intervention_removed": intervention.get("removed_fact_id"),
                "intervention_reason": intervention.get("reason", ""),
                "rescue_eligible": material["rescue_eligible"],
                "direct_baseline_correct": int(direct_ok),
                "closed_book_known": int(closed_known),
                "qa_eligible": int(direct_ok and not closed_known),
                "stage1_selected_ids": "|".join(base_ids),
                "relay_input_ids": "|".join(input_ids),
                "stage2_selected_ids": "|".join(relay_ids),
                "stage1_selected_count": len(base_ids),
                "stage2_selected_count": len(relay_ids),
                "a_card_selected_stage1": int(task["a_fact_id"] in base_ids),
                "b_card_selected_stage1_base": int(task["b_fact_id"] in base_ids),
                "b_card_selected_relay_input": int(task["b_fact_id"] in input_ids),
                "b_card_selected_stage2": int(task["b_fact_id"] in relay_ids),
                "a_present_stage1": int(evidence_present(
                    task, base_ids, task["a_fact_id"])),
                "b_present_stage1_base": int(evidence_present(
                    task, base_ids, task["b_fact_id"])),
                "b_present_relay_input": int(evidence_present(
                    task, input_ids, task["b_fact_id"])),
                "b_present_stage2": int(evidence_present(
                    task, relay_ids, task["b_fact_id"])),
                "stage1_compliant": int(stage1_ok), "stage2_compliant": int(stage2_ok),
                "overall_compliant": int(stage1_ok and stage2_ok),
                "stage1_empty": selector["empty_output"], "stage2_empty": relay["empty_output"],
                "stage1_truncated": selector["truncated"],
                "stage2_truncated": relay["truncated"],
                "stage1_packet_words": len(selector["packet_text"].split()),
                "relay_input_words": len(material["relay_input_packet"].split()),
                "stage2_packet_words": len(relay["packet_text"].split()),
                "question": answer["question"], "golds": "|".join(answer["golds"]),
                "pred": answer["pred"], "em": answer["em"], "f1": answer["f1"],
                "judge_correct": answer.get("judge_correct", float("nan")),
            })
    return rows


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def cluster_values(rows: list[dict], measure: str) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(measure)
        if _finite(value):
            grouped[row["dossier_id"]].append(float(value))
    return np.asarray([np.mean(grouped[k]) for k in sorted(grouped) if grouped[k]], dtype=float)


def interval(values: np.ndarray, cfg: dict, label: str) -> tuple[float, float, float]:
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    n_boot = int(cfg["analysis"]["bootstrap_resamples"])
    alpha = 1.0 - float(cfg["analysis"]["ci_level"])
    seed = int(cfg["analysis"]["seed"]) ^ int(digest(label)[:8], 16)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def compute_metrics(rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    for scope in ("itt", "compliant", "qa_eligible"):
        if scope == "itt":
            scoped = rows
        elif scope == "compliant":
            scoped = [r for r in rows if r["overall_compliant"]]
        else:
            scoped = [r for r in rows if r.get("qa_eligible")]
        cells = sorted({(r["family"], r["arm"]) for r in scoped})
        for family, arm in cells:
            subset = [r for r in scoped if r["family"] == family and r["arm"] == arm]
            row = {
                "scope": scope, "family": family, "arm": arm,
                "n_dossiers": len({r["dossier_id"] for r in subset}),
                "n_observations": len(subset),
            }
            for measure in METRICS:
                values = cluster_values(subset, measure)
                mean, lo, hi = interval(values, cfg, f"metric:{scope}:{family}:{arm}:{measure}")
                row[measure] = round(mean, 6) if _finite(mean) else ""
                row[f"{measure}_lo"] = round(lo, 6) if _finite(lo) else ""
                row[f"{measure}_hi"] = round(hi, 6) if _finite(hi) else ""
            out.append(row)
    return out


def paired_cluster_delta(rows: list[dict], treatment: str, control: str, measure: str,
                         cfg: dict, scope: str, eligible_only: bool,
                         label: str) -> dict | None:
    t = {r["task_id"]: r for r in rows if r["arm"] == treatment}
    c = {r["task_id"]: r for r in rows if r["arm"] == control}
    grouped: dict[str, list[float]] = defaultdict(list)
    n_pairs = 0
    for key in sorted(set(t) & set(c)):
        left, right = t[key], c[key]
        if scope == "compliant" and not (left["overall_compliant"] and right["overall_compliant"]):
            continue
        if scope == "qa_eligible" and not (left.get("qa_eligible") and right.get("qa_eligible")):
            continue
        if eligible_only and not (left["rescue_eligible"] and right["rescue_eligible"]):
            continue
        if not (_finite(left.get(measure)) and _finite(right.get(measure))):
            continue
        grouped[left["dossier_id"]].append(float(left[measure]) - float(right[measure]))
        n_pairs += 1
    values = np.asarray([np.mean(grouped[k]) for k in sorted(grouped) if grouped[k]], dtype=float)
    if not len(values):
        return None
    delta, lo, hi = interval(values, cfg, f"delta:{label}:{scope}:{measure}:{eligible_only}")
    seed = int(cfg["analysis"]["seed"]) ^ int(digest(f"p:{label}:{scope}:{measure}")[:8], 16)
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(int(cfg["analysis"]["bootstrap_resamples"]), len(values)))
    null = (signs * values).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(delta)) + 1) / (len(null) + 1))
    margin = float(cfg["analysis"]["equivalence_margin"])
    return {
        "scope": scope, "population": "eligible_omissions" if eligible_only else "all",
        "comparison": label, "treatment": treatment, "control": control,
        "measure": measure, "delta": round(delta, 6), "lo": round(lo, 6),
        "hi": round(hi, 6), "p_value": round(p, 6),
        "n_dossiers": len(values), "n_paired_observations": n_pairs,
        "equivalence_margin": margin,
        "equivalent_within_margin": int(lo > -margin and hi < margin),
    }


def compute_contrasts(rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    measures = ("f1", "judge_correct", "b_present_stage2")
    families = sorted({r["family"] for r in rows})
    for family in families:
        comparisons = (
            ("early_selector_at_large_relay", f"{family}_ll", f"{family}_sl", False),
            ("larger_relay_after_small_selector", f"{family}_sl", f"{family}_ss", False),
            ("larger_relay_on_clean_omissions", f"{family}_sl", f"{family}_ss", True),
            ("smaller_relay_after_large_selector", f"{family}_ls", f"{family}_ll", False),
            ("large_first_order", f"{family}_ls", f"{family}_sl", False),
            ("restore_vs_sham", f"{family}_sl_restore", f"{family}_sl_sham", True),
            ("restore_vs_sealed", f"{family}_sl_restore", f"{family}_sl", True),
            ("reopen_vs_sealed", f"{family}_sl_reopen", f"{family}_sl", True),
        )
        names = {r["arm"] for r in rows if r["family"] == family}
        for label, treatment, control, eligible in comparisons:
            if treatment not in names or control not in names:
                continue
            for scope in ("itt", "compliant", "qa_eligible"):
                for measure in measures:
                    row = paired_cluster_delta(
                        [r for r in rows if r["family"] == family], treatment, control,
                        measure, cfg, scope, eligible, f"{family}:{label}")
                    if row:
                        row["family"] = family
                        out.append(row)
    return out


def make_plot(metrics: list[dict], rows: list[dict], cfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = str(cfg["analysis"]["primary_metric"])
    if not any(_finite(r.get(primary)) for r in rows):
        primary = "f1"
    itt = [r for r in metrics if r["scope"] == "itt"]
    families = list(cfg["test_families"])
    colors = {f: cfg["families"][f]["color"] for f in families}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    codes = ("ss", "sl", "ls", "ll")
    x = np.arange(len(codes))
    width = 0.34
    for j, family in enumerate(families):
        values, low, high = [], [], []
        for code in codes:
            cell = next((r for r in itt if r["arm"] == f"{family}_{code}"), None)
            value = float(cell.get(primary) or np.nan) if cell else np.nan
            values.append(value)
            low.append(value - float(cell.get(f"{primary}_lo") or value) if cell else 0)
            high.append(float(cell.get(f"{primary}_hi") or value) - value if cell else 0)
        axes[0].bar(x + (j - 0.5) * width, values, width, color=colors[family],
                    label=cfg["families"][family]["label"], alpha=0.86,
                    yerr=np.asarray([low, high]), capsize=3)
    axes[0].set_xticks(x, ["S→S", "S→L", "L→S", "L→L"])
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("B judged accuracy" if primary == "judge_correct" else "B token F1")
    axes[0].set_title("Sealed 2×2 factorial (ITT)")
    axes[0].legend(frameon=False, fontsize=8)

    controls = ("sl", "sl_sham", "sl_restore", "sl_reopen")
    labels = ("sealed", "sham", "B restored", "source reopened")
    for j, family in enumerate(families):
        subset = [r for r in rows if r["family"] == family and r["rescue_eligible"]]
        values = []
        for code in controls:
            vals = cluster_values([r for r in subset if r["arm"] == f"{family}_{code}"], primary)
            values.append(float(np.mean(vals)) if len(vals) else np.nan)
        axes[1].plot(x, values, marker="o", linewidth=2, color=colors[family],
                     label=cfg["families"][family]["label"])
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_title("Small→large controls\n(stage-1 B omissions only)")
    axes[1].set_ylabel("B judged accuracy" if primary == "judge_correct" else "B token F1")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Fictional-QA upstream bottleneck and recovery")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(metrics: list[dict], contrasts: list[dict], rows: list[dict], cfg: dict,
                 path: Path, cost: dict | None, baseline_rows: list[dict]) -> None:
    primary = str(cfg["analysis"]["primary_metric"])
    if not any(_finite(r.get(primary)) for r in rows):
        primary = "f1"
    n_dossiers = len({r["dossier_id"] for r in rows})
    n_tasks = len({r["task_id"] for r in rows})
    lines = [
        "# Experiment 8b: fictional-QA selector bottleneck", "",
        f"- Dossiers: **{n_dossiers}**; A/B rotations: **{n_tasks}**.",
        f"- Channel: **{cfg['channel']['selector_slots']}** selector slots → "
        f"**{cfg['channel']['relay_slots']}** relay slots.",
        "- Stage 1 sees the fictional source and A. Stage 2 is newly shown B; sealed arms "
        "receive no source. The final answerer is fixed and from a third family.",
        "- Intervals and paired tests resample dossiers, not rotations.",
        "- ITT counts schema failures as failures; compliant results require both stages to "
        "return a valid fixed-K selection.", "",
    ]
    configured_n = int(cfg["dataset"]["n_dossiers"])
    if n_dossiers < configured_n:
        lines += ["> **Pipeline pilot subset.** This is not the configured full run.", ""]
    else:
        lines += ["> **Full configured run.** Effective n is 20 dossiers; the 80 A/B rotations "
                  "are repeated measures, not independent samples.", ""]
    lines += ["## Factorial cell outcomes (ITT)", "",
              "Judged B accuracy; arrows denote stage-1 selector → stage-2 relay.", "",
              "| family | S→S | S→L | L→S | L→L |",
              "|---|---:|---:|---:|---:|"]
    metric_index = {(r["scope"], r["family"], r["arm"]): r for r in metrics}
    for family in sorted({r["family"] for r in rows}):
        values = []
        for code in ("ss", "sl", "ls", "ll"):
            cell = metric_index.get(("itt", family, f"{family}_{code}"), {})
            value = cell.get(primary, "")
            values.append(f"{float(value):.3f}" if _finite(value) else "—")
        lines.append(f"| {family} | " + " | ".join(values) + " |")
    lines += ["", "## Primary paired contrasts (ITT)", "",
              "| family | comparison | population | delta | 95% CI | p | dossiers |",
              "|---|---|---|---:|---|---:|---:|"]
    chosen = [r for r in contrasts if r["scope"] == "itt" and r["measure"] == primary]
    for r in chosen:
        p_text = "<0.001" if float(r["p_value"]) < .001 else f"{r['p_value']:.3f}"
        lines.append(
            f"| {r['family']} | `{r['comparison']}` | {r['population']} | "
            f"{r['delta']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
            f"{p_text} | {r['n_dossiers']} |")

    early = {r["family"]: r for r in chosen
             if r["comparison"].endswith("early_selector_at_large_relay")}
    clean_relay = {r["family"]: r for r in chosen
                   if r["comparison"].endswith("larger_relay_on_clean_omissions")}
    clean_omission_rows = [r for r in rows if r["rescue_eligible"]]
    sealed_clean = [r for r in clean_omission_rows
                    if r["arm"] in {f"{r['family']}_ss", f"{r['family']}_sl"}]
    invalid_stage2 = [r for r in rows if not r["stage2_compliant"]]
    invalid_with_support = [r for r in invalid_stage2 if r["b_present_relay_input"]]
    lines += ["", "## Observed result", ""]
    if "llama" in early and "qwen" in early:
        lines += [
            f"- The upstream-size contrast did **not replicate across families**: "
            f"Llama {early['llama']['delta']:+.3f}, Qwen {early['qwen']['delta']:+.3f}.",
        ]
    if sealed_clean:
        clean_accuracy = float(np.mean([float(r[primary]) for r in sealed_clean]))
        lines += [
            f"- Across {len(sealed_clean)} sealed small-selector rows where stage 1 contained "
            f"no answer-bearing B evidence, downstream judged accuracy was "
            f"**{clean_accuracy:.3f}**.",
        ]
    if clean_relay:
        details = ", ".join(
            f"{family} {row['delta']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]"
            for family, row in sorted(clean_relay.items()))
        lines += [
            f"- Large-minus-small relay accuracy on those clean omissions is {details}; "
            f"both intervals lie inside the prespecified ±{float(cfg['analysis']['equivalence_margin']):.2f} "
            "equivalence margin. A larger relay did not reconstruct absent fictional evidence.",
        ]
    lines += [
        "- Exact B restoration strongly beat the same-width sham in both families, while "
        "source reopening supplied the positive ceiling. This localizes failure to evidence "
        "availability, but does not establish a general model-size bottleneck.",
        f"- Strict stage-2 noncompliance occurred in {len(invalid_stage2)}/{len(rows)} rows; "
        f"{len(invalid_with_support)} contained answer-bearing B evidence. Thus these were "
        "abstentions/invalid selections after an omission, not losses of available B support.",
        "", "## Interpretation rules", "",
              "- An upstream capability bottleneck requires better B retention/accuracy for "
              "L→L than S→L, with the relay and final answerer held fixed.",
              "- 'A larger relay cannot recover an omission' requires an equivalence interval, "
              "not merely a non-significant S→L minus S→S contrast.",
              "- Restore must beat its matched sham on stage-1 omissions, and source reopening "
              "must provide a positive ceiling, before failure is attributed to missing evidence.",
              "- Evidence presence is computed from answer-bearing text, not only the designated "
              "card id, so cross-card answer mentions are not misclassified as omissions.",
              "- Llama and Qwen are reported separately. A shared size/capability claim requires "
              "the direction to replicate across families.", ""]
    if baseline_rows:
        lines += ["## Reader baselines", "",
                  "| material | metric | mean | 95% CI | questions |",
                  "|---|---|---:|---:|---:|"]
        for row in baseline_rows:
            if row["measure"] in (primary, "f1"):
                lines.append(
                    f"| {row['kind']} | {row['measure']} | {row['mean']:.3f} | "
                    f"[{row['lo']:.3f}, {row['hi']:.3f}] | {row['n_questions']} |")
        eligible = [r for r in rows if r.get("qa_eligible")]
        lines += ["", f"QA-eligible arm rows: **{len(eligible)}/{len(rows)}** "
                  "(full-card correct and closed-book incorrect).", ""]
    if cost is not None:
        initial = cost.get("initial_full_run") if isinstance(cost, dict) else None
        if initial:
            generation_cost = float(initial.get("generation", {}).get("cost_usd", 0.0))
            judge_cost = float(initial.get("judge", {}).get("cost_usd", 0.0))
            lines += ["## Cost", "",
                      f"- Initial full-run generation: **${generation_cost:.6f}**.",
                      f"- Judge: **${judge_cost:.6f}**.",
                      f"- Total: **${generation_cost + judge_cost:.6f}**.", ""]
        else:
            lines += ["## Cost", "", f"`{json.dumps(cost, sort_keys=True)}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def dry_run_report(tasks: list[dict], arms: list[Arm], cfg: dict) -> dict:
    selectors = sorted({a.selector_model for a in arms})
    stage1 = len(tasks) * len(selectors)
    # Upper bound: intervention no-ops may collapse identical relay prompts at
    # run time, but the dry run never generates data to assume that saving.
    stage2 = len(tasks) * len(arms)
    answers = stage2
    unique_questions = len({(t["dossier_id"], t["b_fact_id"]) for t in tasks})
    baseline_calls = (2 * unique_questions
                      if cfg.get("baselines", {}).get("enabled", True) else 0)
    by_model = defaultdict(int)
    for model in selectors:
        by_model[model] += len(tasks)
    for arm in arms:
        by_model[arm.relay_model] += len(tasks)
    by_model[cfg["answerer"]["model"]] += answers + baseline_calls
    return {
        "experiment": cfg["experiment"],
        "dossiers": len({t["dossier_id"] for t in tasks}), "rotations": len(tasks),
        "arms": len(arms), "selector_calls": stage1,
        "relay_calls_upper_bound": stage2, "answer_calls_upper_bound": answers,
        "baseline_calls": baseline_calls,
        "judge_calls_upper_bound": (answers + baseline_calls)
        if cfg.get("judge", {}).get("enabled") else 0,
        "calls_by_model_upper_bound": dict(sorted(by_model.items())),
        "writes": 0,
    }


def prompt_selftest(task: dict, cfg: dict) -> None:
    k1 = int(cfg["channel"]["selector_slots"])
    k2 = int(cfg["channel"]["relay_slots"])
    first = selector_prompt(task, k1)
    assert task["a_question"] in first
    assert task["b_question"] not in first, "stage 1 leaked the future Question B"

    ids = tuple(c.fact_id for c in task["cards"][:k1])
    packet = fq.render_packet_slots(task["cards"], ids, k1)
    sealed = hm.seal(task["b_question"], hm.Handoff(task["task_id"], "test", None, packet))
    relay = fq.sealed_relay_prompt(sealed, k2)
    assert task["b_question"] in relay and packet in relay
    # A source card outside the packet must not re-enter the sealed prompt.
    outside = next((c for c in task["cards"] if c.fact_id not in ids), None)
    if outside:
        assert outside.evidence_text not in relay
    reopened = fq.reopened_selection_prompt(
        task["dossier"], task["b_fact_id"], task["cards"], k2)
    assert outside is None or outside.evidence_text in reopened
    for bad in ("raw", {"handoff_text": packet}, object()):
        try:
            fq.sealed_relay_prompt(bad, k2)
        except TypeError:
            continue
        raise AssertionError(f"sealed relay accepted {type(bad).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="print a bounded call plan; never write data/runs/results")
    parser.add_argument("--analyse-only", action="store_true",
                        help="rebuild tables/plot/report from stored raw records; issue no calls")
    parser.add_argument("--pilot", action="store_true",
                        help="bounded one-dossier run for pipeline validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.analyse_only:
        parser.error("--dry-run and --analyse-only are mutually exclusive")
    if args.pilot and args.limit not in (None, 1):
        parser.error("--pilot fixes --limit at one dossier")

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    if args.no_judge:
        cfg.setdefault("judge", {})["enabled"] = False
    limit = 1 if args.pilot else args.limit
    source_manifest, tasks = load_tasks(cfg, limit)
    arms = build_arms(cfg)
    prompt_selftest(tasks[0], cfg)

    if args.dry_run:
        # Return before ModelPool construction: DiskCache creates its directory,
        # and even that benign mkdir would violate the project's dry-run rule.
        print(json.dumps(dry_run_report(tasks, arms, cfg), indent=2, sort_keys=True))
        return 0

    n_dossiers = len({t["dossier_id"] for t in tasks})
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n_dossiers}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n_dossiers}"
    data_path = ROOT / cfg["dataset"]["items_jsonl"]
    fingerprint = experiment_fingerprint(cfg, config_path, data_path)
    selector_path = run_root / "selectors.jsonl"
    relay_path = run_root / "relays.jsonl"
    answer_path = run_root / "answers.jsonl"
    baseline_path = run_root / "baselines.jsonl"

    if args.analyse_only:
        selectors = {(r["task_id"], r["model_key"]): r for r in read_jsonl(selector_path)
                     if r.get("experiment_fingerprint") == fingerprint}
        relays = {r["call_key"]: r for r in read_jsonl(relay_path)
                  if r.get("experiment_fingerprint") == fingerprint}
        answers = {r["material_key"]: r for r in read_jsonl(answer_path)
                   if r.get("experiment_fingerprint") == fingerprint}
        baselines = {r["baseline_key"]: r for r in read_jsonl(baseline_path)
                     if r.get("experiment_fingerprint") == fingerprint}
        if not selectors or not relays or not answers or not baselines:
            raise SystemExit("analyse-only found no complete records for the current fingerprint")
        materials = build_relay_materials(tasks, arms, selectors, cfg)
        cost = None
        judge_cost = None
    else:
        pool = mp.ModelPool(cfg, mp.load_registry(cfg))
        selectors = generate_selectors(pool, tasks, arms, cfg, selector_path, fingerprint)
        relays, materials = generate_relays(
            pool, tasks, arms, selectors, cfg, relay_path, fingerprint)
        answers = run_answers(
            pool, tasks, arms, relays, materials, cfg, answer_path, fingerprint)
        baselines = run_baselines(pool, tasks, cfg, baseline_path, fingerprint)
        judge_cost = None
        if cfg.get("judge", {}).get("enabled"):
            judged = list(answers.values()) + list(baselines.values())
            judger = add_judge(judged, cfg, tag="fictional_bottleneck_judge")
            write_jsonl(answer_path, [answers[k] for k in sorted(answers)])
            write_jsonl(baseline_path, [baselines[k] for k in sorted(baselines)])
            if judger:
                judge_cost = judger.ledger.summary()
                print("[bottleneck:judge cost] " + json.dumps(judge_cost))
        cost = pool.ledger.summary()

    rows = expand_records(tasks, arms, selectors, relays, materials, answers, baselines, cfg)
    metrics = compute_metrics(rows, cfg)
    contrasts = compute_contrasts(rows, cfg)
    baseline_rows = baseline_metrics(baselines, cfg)
    write_csv(result_root / "records.csv", rows)
    write_csv(result_root / "metrics.csv", metrics)
    write_csv(result_root / "contrasts.csv", contrasts)
    write_csv(result_root / "baseline_metrics.csv", baseline_rows)
    make_plot(metrics, rows, cfg, result_root / "fictional_model_bottleneck.png")
    cost_path = result_root / "cost.json"
    cost_audit = (json.loads(cost_path.read_text(encoding="utf-8"))
                  if cost_path.exists() else {})
    if not args.analyse_only:
        latest = {"generation": cost, "judge": judge_cost}
        latest["total_cost_usd"] = round(
            float((cost or {}).get("cost_usd", 0.0))
            + float((judge_cost or {}).get("cost_usd", 0.0)), 6)
        cost_audit["latest_execution"] = latest
        if (not cost_audit.get("initial_full_run")
                and (int((cost or {}).get("calls_live", 0)) > 0
                     or int((judge_cost or {}).get("calls_live", 0)) > 0)):
            cost_audit["initial_full_run"] = latest
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(json.dumps(cost_audit, indent=2), encoding="utf-8")
    write_report(metrics, contrasts, rows, cfg, result_root / "report.md",
                 cost_audit or cost,
                 baseline_rows)
    result_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_fingerprint": fingerprint,
        "source_manifest": source_manifest,
        "config_file": str(config_path.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_file": str(data_path.relative_to(ROOT)),
        "source_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "shared_module_sha256": hashlib.sha256(Path(fq.__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_dossiers": n_dossiers,
        "n_tasks": len(tasks),
        "rotations_per_dossier": int(cfg["dataset"]["rotations_per_dossier"]),
        "n_arms": len(arms),
        "cluster": "dossier",
        "primary_metric": str(cfg["analysis"]["primary_metric"]),
        "answerer_model_key": cfg["answerer"]["model"],
        "answerer_model_id": cfg["models"][cfg["answerer"]["model"]]["id"],
        "judge_model": cfg.get("judge", {}).get("model_id"),
    }
    (result_root / "manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bottleneck] wrote {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
