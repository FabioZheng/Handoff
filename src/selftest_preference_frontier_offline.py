"""Offline regression checks for Experiment 11's preference runner."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import budget as bd  # noqa: E402
import regret_data as rd  # noqa: E402
import run_communication_regret as cr  # noqa: E402
import run_preference_frontier as rf  # noqa: E402
from llm import LLMResult, load_config  # noqa: E402


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAIL  {label}")
    print(f"PASS  {label}")


class FakeClient:
    """Returns long unique-enough prose so deterministic word caps bind."""

    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, *, temperature, max_tokens, seed, tag, **_kwargs):
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "seed": seed, "tag": tag})
        words = [f"fact{i}" for i in range(max(200, max_tokens))]
        return LLMResult(" ".join(words), 10, len(words), 0.0, 0.0, True,
                         "fake/model", finish_reason="stop", tag=tag)


cfg = load_config(ROOT / "bounded_communication_frontier_config.yaml")
prefs = rf.preferences_from_config(cfg)
budgets = bd.budgets_from_config(cfg)

print("=== preference parameterization ===")
check("five ordered preference levels", [p.future_share for p in prefs]
      == [0.0, 0.25, 0.5, 0.75, 1.0])
check("lambda endpoints are represented", prefs[0].value == 0 and prefs[-1].infinite)
check("policy ids are stable", rf.policy_name("split", "l1") == "split__l1")
primary_rep = cfg["frontier"]["replications"]["primary"]
check("primary SQuAD resolves its 40/80 budget subset",
      rf.replication_budget_words(primary_rep, "squad_groups") == [40, 80])
check("primary relation dossiers retain 40/80/160",
      rf.replication_budget_words(primary_rep, "relation_dossiers") == [40, 80, 160])

print("\n=== structured allocation ===")
for outer in budgets:
    heading_totals = set()
    for pref in prefs:
        parts = rf.split_components("source", "question", outer, pref, cfg)
        heading_totals.add(sum(bd.word_count(part["label"]) for part in parts))
        planned = sum(part["planned_words"] + bd.word_count(part["label"])
                      for part in parts)
        check(f"{outer.words}w/{pref.id} allocation sums to cap", planned == outer.words)
    check(f"{outer.words}w heading overhead is preference-invariant", heading_totals == {4})

outer = next(b for b in budgets if b.words == 40)
client = FakeClient()
pref = next(p for p in prefs if p.id == "l1")
context = rd.RegretContext(
    context_id="ctx", dataset="squad_groups", title="t", source="source words here",
    questions=(
        rd.RegretQuestion("q1", "Question one?", ("one",)),
        rd.RegretQuestion("q2", "Question two?", ("two",)),
    ), content_hash="hash", meta={},
)
rotations = {context.context_id: rd.rotations_for(context)}
spec = rf.message_specs((context,), rotations, [outer], [pref], ["split"], cfg,
                        "primary")[0]
message = rf.run_message_call(client, spec, cfg)
check("assembled structured message obeys hard cap", message["delivered_words"] <= 40)
check("both headings reach the sealed reader", "Current evidence:" in message["handoff_text"]
      and "Reusable evidence:" in message["handoff_text"])
check("planned split is recorded", message["current_planned_words"] == 18
      and message["reusable_planned_words"] == 18)
check("component delivery is auditable", message["current_delivered_words"] <= 18
      and message["reusable_delivered_words"] <= 18)

print("\n=== prompt and hash isolation ===")
scalar = rf.message_specs((context,), rotations, [outer], prefs[:2], ["scalar"], cfg,
                          "probe")
check("numeric preference changes the prompt", scalar[0]["prompt"] != scalar[1]["prompt"])
check("scalar objective prompt does not prescribe a word-share equivalence",
      "normalized weights" not in scalar[1]["prompt"]
      and "not a prescribed word quota" in scalar[1]["prompt"])
check("numeric preference changes request hash",
      scalar[0]["request_hash"] != scalar[1]["request_hash"])
cfg17, _rep, _ds, _n = rf.configure_replication(cfg, "seed17", None, 1)
cfg29, _rep, _ds, _n = rf.configure_replication(cfg, "seed29", None, 1)
s17 = rf.message_specs((context,), rotations, [outer], [pref], ["split"], cfg17,
                       "seed17")[0]
s29 = rf.message_specs((context,), rotations, [outer], [pref], ["split"], cfg29,
                       "seed29")[0]
check("seed sensitivity cannot collide in cache", s17["request_hash"] != s29["request_hash"])
check("single-dataset replication selects its configured dataset", _ds == "squad_groups")

print("\n=== Experiment-10 matrix compatibility ===")
messages = []
answers = []
for rotation in rotations[context.context_id]:
    s = next(x for x in rf.message_specs(
        (context,), {context.context_id: (rotation,)}, [outer], [pref], ["split"], cfg,
        "compat") if x["rotation_id"] == rotation.rotation_id)
    m = rf.run_message_call(FakeClient(), s, cfg)
    messages.append(m)
    for question in context.questions:
        answers.append({
            "message_key": m["message_key"], "eval_qid": question.qid,
            "em": 1.0, "f1": 1.0, "judge_correct": 1.0,
        })
matrix = cr.build_matrix_rows((context,), rotations, messages, answers, ["split__l1"])
rotation_rows = cr.build_rotation_rows(matrix)
check("every message-question cell is retained", len(matrix) == 4)
check("utility matrix collapses to both rotations", len(rotation_rows) == 2)
check("sealed source never enters an answer record", all("source" not in row for row in matrix))

print("\n=== dry-run purity ===")
before_runs = set((ROOT / "runs" / "bounded_communication_frontier").rglob("*")) \
    if (ROOT / "runs" / "bounded_communication_frontier").exists() else set()
before_results = set((ROOT / "results" / "bounded_communication_frontier").rglob("*")) \
    if (ROOT / "results" / "bounded_communication_frontier").exists() else set()
with contextlib.redirect_stdout(io.StringIO()):
    code = rf.main(["--dry-run", "--replication", "primary", "--dataset",
                    "relation_dossiers", "--limit", "1", "--budgets", "40"])
after_runs = set((ROOT / "runs" / "bounded_communication_frontier").rglob("*")) \
    if (ROOT / "runs" / "bounded_communication_frontier").exists() else set()
after_results = set((ROOT / "results" / "bounded_communication_frontier").rglob("*")) \
    if (ROOT / "results" / "bounded_communication_frontier").exists() else set()
check("dry-run succeeds", code == 0)
check("dry-run writes no run artifacts", before_runs == after_runs)
check("dry-run writes no result artifacts", before_results == after_results)

print("\nALL PREFERENCE FRONTIER CHECKS PASSED")
