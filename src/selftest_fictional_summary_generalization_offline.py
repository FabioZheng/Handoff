"""Offline checks for Experiment 5b's fictional fixed-capacity runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fictional_qa as fq  # noqa: E402
import handoffs as hm  # noqa: E402
import run_fictional_summary_generalization as exp  # noqa: E402
from llm import load_config  # noqa: E402


failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


cfg = load_config(ROOT / "fictional_summary_generalization_config.yaml")
manifest, dossiers, cards_by_id, rotations_by_id = exp.load_design(cfg, limit=2)
dossier = dossiers[0]
cards = cards_by_id[dossier.dossier_id]
rotations = rotations_by_id[dossier.dossier_id]

print("=== design and prompt matching ===")
check("two-dossier bounded loader", len(dossiers) == 2)
check("six target rotations per dossier", len(rotations) == 6)
check("opaque ids hide evaluator roles",
      all("target" not in c.card_id and "side" not in c.card_id for c in cards))
generic = fq.generic_selection_prompt(dossier, cards, 4)
conditioned = fq.conditioned_selection_prompt(dossier, rotations[0], cards, 4)
marker = fq.question_block(rotations[0])
check("conditioned minus A block equals generic", conditioned.replace(marker, "") == generic)
check("generic prompt leaks no QA question",
      all(fact.question not in generic for fact in dossier.facts))
check("generic prompt leaks no evaluator fact id",
      all(fact.fact_id not in generic for fact in dossier.facts))

specs = exp.selection_specs(dossiers, cards_by_id, rotations_by_id, [4], cfg)
generic_specs = [s for s in specs if s["mode"] == "generic"]
conditioned_specs = [s for s in specs if s["mode"] == "conditioned"]
check("one generic call per dossier", len(generic_specs) == len(dossiers))
check("one conditioned call per rotation", len(conditioned_specs) == 12)
check("all logical selection keys are unique",
      len({s["selection_key"] for s in specs}) == len(specs))


class StubClient:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            text=self.text, cached=True, finish_reason="stop",
            prompt_tokens=10, completion_tokens=5,
        )


print("\n=== model-facing ids map back to evaluator facts ===")
spec = conditioned_specs[0]
chosen_cards = [card.card_id for card in spec["cards"][:4]]
record = exp.run_selection_call(
    StubClient(json.dumps({fq.SELECTION_KEY: chosen_cards})), spec, cfg)
expected_facts = list(fq.card_ids_to_fact_ids(spec["cards"], chosen_cards))
check("strict selector output is valid", record["selection_valid"])
check("stored card ids are opaque", record["selected_card_ids"] == chosen_cards)
check("stored fact ids are evaluator-side", record["selected_fact_ids"] == expected_facts)
check("sealed packet exposes no evaluator ids",
      all(fact_id not in record["handoff_text"] for fact_id in expected_facts))
check("packet has exactly K slots", record["handoff_text"].count("[SLOT ") == 4)

bad = exp.run_selection_call(
    StubClient(json.dumps({fq.SELECTION_KEY: chosen_cards[:2]})), spec,
    {**cfg, "selection": {**cfg["selection"], "max_parse_attempts": 1}})
check("invalid output is retained as an ITT failure",
      not bad["selection_valid"] and bad["selected_fact_ids"] == []
      and bad["handoff_text"] == "")

print("\n=== sealed answering boundary and clustered inference ===")
try:
    exp.answer_sealed(StubClient("x"), "raw material", cfg, 0.0, None, "bad")
    check("answerer rejects an unsealed string", False)
except TypeError:
    check("answerer rejects an unsealed string", True)

# Two rotations per dossier are deliberately correlated.  The analysis must
# average them before resampling, leaving exactly two independent clusters.
rows = []
for d_index, d in enumerate(dossiers):
    for r_index in range(2):
        for mode, value in (("generic", 0.25 + .1 * d_index),
                            ("conditioned", 0.75 + .1 * d_index)):
            rows.append({
                "dossier_id": d.dossier_id, "rotation_id": f"r{r_index}",
                "mode": mode, "slots": 4, "immediate_f1": value,
                "reusable_f1": 1.0 - value, "selection_valid": 1.0,
                "announced_card_selected": value,
                "hidden_card_selection_rate": 1.0 - value,
                "handoff_characters": 100.0, "handoff_words": 20.0,
                "handoff_estimated_tokens": 25.0,
            })
metrics, contrasts = exp.analyse_rotation_rows(rows, [4], cfg)
f1_metric = next(r for r in metrics if r["mode"] == "conditioned"
                 and r["endpoint"] == "immediate" and r["metric"] == "f1")
delta = next(r for r in contrasts if r["comparison"] == "conditioned_minus_generic"
             and r["endpoint"] == "immediate" and r["metric"] == "f1")
check("bootstrap effective n is dossiers", f1_metric["n_dossiers"] == 2)
check("paired immediate delta is computed after dossier averaging",
      abs(delta["delta"] - 0.5) < 1e-9)

print("\n=== identical answer-request single-flight ===")
fact = dossier.facts[0]
answer_model = (cfg.get("answerer") or cfg["model"])["id"]
shared_material = fq.render_source_cards(cards)
answer_specs = []
for index, mode in enumerate(("generic", "conditioned")):
    answer_spec = {
        "answer_key": f"dedup-{index}", "selection_key": f"selection-{index}",
        "dossier_id": dossier.dossier_id,
        "announced_fact_id": None if mode == "generic" else fact.fact_id,
        "eval_fact": fact, "mode": mode, "slots": 6, "baseline": None,
        "sample": 0, "material": shared_material,
        "temperature": float(cfg["decoding"]["answer_temperature"]),
        "seed": None, "technical_failure": False,
    }
    answer_spec["request_hash"] = exp.answer_request_hash(
        answer_model, fact.question, shared_material, answer_spec["temperature"],
        None, cfg)
    answer_specs.append(answer_spec)
stored_conflict = [
    exp.run_answer_call(StubClient("wrong answer"), answer_specs[0], cfg),
    exp.run_answer_call(StubClient(fact.answer), answer_specs[1], cfg),
]
canonical_client = StubClient(fact.answer)
deduped, reconciled = exp.generate_answers(
    canonical_client, answer_specs, cfg, stored_conflict)
check("one canonical call reconciles conflicting duplicate requests",
      reconciled == 1 and canonical_client.calls == 1)
check("identical prompts receive one identical response",
      len({row["pred"] for row in deduped}) == 1)
check("logical evaluator metadata remains distinct after reuse",
      {row["mode"] for row in deduped} == {"generic", "conditioned"})

plan = exp.dry_run_report(dossiers, rotations_by_id, [4], cfg)
check("dry-run declares zero writes", plan["writes"] == 0)
check("dry-run counts generic reuse",
      plan["selection_calls_expected"] == len(dossiers) + sum(map(len, rotations_by_id.values())))

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed: {', '.join(failures)}")
print("\nALL CHECKS PASSED")
