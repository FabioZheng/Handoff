"""Offline checks for Experiment 8b's selector/relay bottleneck design."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import fictional_qa as fq  # noqa: E402
import run_fictional_model_bottleneck as exp  # noqa: E402
from llm import load_config  # noqa: E402


failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


cfg = load_config(ROOT / "configs/fictional_model_bottleneck_config.yaml")
_manifest, tasks = exp.load_tasks(cfg, limit=2)
arms = exp.build_arms(cfg)
task = tasks[0]

print("=== schedule and prompt isolation ===")
check("four rotations per dossier", len(tasks) == 8)
check("each dossier assigns four distinct future B facts",
      all(len({t["b_fact_id"] for t in tasks if t["dossier_id"] == dossier_id}) == 4
          for dossier_id in {t["dossier_id"] for t in tasks}))
check("two within-family 2x2 factorials",
      sum(arm.factorial for arm in arms) == 8)
check("restore, sham and reopen exist per family", len(arms) == 14)
try:
    exp.prompt_selftest(task, cfg)
    check("A/B timing and seal selftest", True)
except Exception as exc:
    check("A/B timing and seal selftest", False, str(exc))

stage1_prompt = exp.selector_prompt(task, int(cfg["channel"]["selector_slots"]))
check("stage 1 sees A", task["a_question"] in stage1_prompt)
check("stage 1 does not see B", task["b_question"] not in stage1_prompt)
check("stage 1 exposes no evaluator role ids",
      all(card.fact_id not in stage1_prompt for card in task["cards"]))


class StubClient:
    def __init__(self, text: str):
        self.text = text

    def chat(self, *args, **kwargs):
        return SimpleNamespace(
            text=self.text, cached=True, finish_reason="stop",
            prompt_tokens=10, completion_tokens=4, provider="stub",
            model="stub-model", cost_usd=0.0,
        )


print("\n=== opaque-id selection and fixed capacity ===")
k1 = int(cfg["channel"]["selector_slots"])
chosen_cards = tuple(card.card_id for card in task["cards"][:k1])
selected = exp.selection_call(
    StubClient(json.dumps({fq.SELECTION_KEY: list(chosen_cards)})),
    stage1_prompt, chosen_cards, k1, 0.0, 64, 1, "stub", task["cards"])
expected_facts = fq.card_ids_to_fact_ids(task["cards"], chosen_cards)
check("selection maps opaque ids to facts", tuple(selected["selected_fact_ids"]) == expected_facts)
check("packet has exactly selector K", selected["packet_text"].count("[SLOT ") == k1)
check("packet does not expose evaluator ids",
      all(fact_id not in selected["packet_text"] for fact_id in expected_facts))

print("\n=== restoration, sham and source reopening ===")
# Force an eligible omission: include A, exclude B, and fill remaining slots.
pool = [card.fact_id for card in task["cards"]]
base = [task["a_fact_id"]] + [fid for fid in pool
                              if fid not in {task["a_fact_id"], task["b_fact_id"]}][:k1 - 1]
ints = exp.interventions_for(task, tuple(base), int(cfg["dataset"]["seed"]))
restore, sham = ints["restore"], ints["sham"]
check("restoration inserts B without widening",
      restore["changed"] and task["b_fact_id"] in restore["after_fact_ids"]
      and len(restore["after_fact_ids"]) == k1)
check("sham changes the same victim without B",
      sham["changed"] and sham["removed_fact_id"] == restore["removed_fact_id"]
      and task["b_fact_id"] not in sham["after_fact_ids"]
      and len(sham["after_fact_ids"]) == k1)
check("semantic omission check uses delivered evidence text",
      not exp.evidence_present(task, base, task["b_fact_id"])
      and exp.evidence_present(task, restore["after_fact_ids"], task["b_fact_id"])
      and not exp.evidence_present(task, sham["after_fact_ids"], task["b_fact_id"]))

selector_row = {
    **selected, "selected_fact_ids": base, "selection_valid": True,
    "truncated": 0, "empty_output": 0,
}
llama_restore = next(a for a in arms if a.name == "llama_sl_restore")
llama_sham = next(a for a in arms if a.name == "llama_sl_sham")
llama_reopen = next(a for a in arms if a.name == "llama_sl_reopen")
restore_material = exp.relay_material(task, llama_restore, selector_row, cfg)
sham_material = exp.relay_material(task, llama_sham, selector_row, cfg)
reopen_material = exp.relay_material(task, llama_reopen, selector_row, cfg)
check("sealed restore carries B", task["b_fact_id"] in restore_material["input_fact_ids"])
check("sealed sham omits B", task["b_fact_id"] not in sham_material["input_fact_ids"])
check("restore and sham keep equal slot count",
      len(restore_material["input_fact_ids"]) == len(sham_material["input_fact_ids"]) == k1)
check("rescue population requires a clean text-level B omission",
      restore_material["rescue_eligible"] == sham_material["rescue_eligible"] == 1)
check("normal relay remains sealed",
      not restore_material["source_access"] and task["b_question"] in restore_material["prompt"])
check("reopen is explicit source access",
      reopen_material["source_access"]
      and len(reopen_material["input_fact_ids"]) == len(task["cards"]))

print("\n=== malformed-output ITT path ===")
bad_selector = {
    **selector_row, "selected_fact_ids": [], "selection_valid": False,
    "truncated": 0, "empty_output": 1,
}
sealed_arm = next(a for a in arms if a.name == "llama_sl")
failed = exp.relay_material(task, sealed_arm, bad_selector, cfg)
check("bad upstream selection becomes a recorded technical failure",
      failed["technical_failure"] and failed["valid_ids"] == ())
check("source-reopen remains a usable positive control after upstream failure",
      not exp.relay_material(task, llama_reopen, bad_selector, cfg)["technical_failure"])

print("\n=== dossier-clustered contrasts ===")
rows = []
for d_index in range(2):
    dossier_id = tasks[d_index * 4]["dossier_id"]
    for rotation in range(2):
        task_id = f"{dossier_id}:r{rotation}"
        for arm, value in (("llama_ss", .25 + .1 * d_index),
                           ("llama_sl", .50 + .1 * d_index),
                           ("llama_ll", .75 + .1 * d_index)):
            rows.append({
                "task_id": task_id, "dossier_id": dossier_id, "family": "llama",
                "arm": arm, "f1": value, "judge_correct": value,
                "b_present_stage2": value, "overall_compliant": 1,
                "rescue_eligible": 0,
            })
delta = exp.paired_cluster_delta(
    rows, "llama_sl", "llama_ss", "f1", cfg, "itt", False, "fixture")
check("paired delta uses two dossier clusters",
      delta is not None and delta["n_dossiers"] == 2)
check("paired delta is computed within task before clustering",
      delta is not None and abs(delta["delta"] - .25) < 1e-9)
equiv_rows = []
for row in rows:
    if row["arm"] in {"llama_ss", "llama_sl"}:
        equiv_rows.append({**row, "judge_correct": 0.0, "rescue_eligible": 1})
equiv = exp.paired_cluster_delta(
    equiv_rows, "llama_sl", "llama_ss", "judge_correct", cfg, "itt", True,
    "clean-omission-fixture")
check("zero recovery satisfies the prespecified equivalence margin",
      equiv is not None and equiv["delta"] == 0.0
      and equiv["lo"] == 0.0 and equiv["hi"] == 0.0
      and equiv["equivalent_within_margin"] == 1)

plan = exp.dry_run_report(tasks, arms, cfg)
check("dry-run declares zero writes", plan["writes"] == 0)
cfg_no_judge = {**cfg, "judge": {**cfg["judge"], "enabled": False}}
check("no-judge dry-run has zero judge calls",
      exp.dry_run_report(tasks, arms, cfg_no_judge)["judge_calls_upper_bound"] == 0)

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed: {', '.join(failures)}")
print("\nALL CHECKS PASSED")
