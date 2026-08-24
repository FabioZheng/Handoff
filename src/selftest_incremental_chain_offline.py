"""Offline regression checks for the incremental-evidence chain."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm
import incremental_chain_data as inc_data
import run_chain
import run_incremental_chain as incremental
from data import GoldSentence, Paragraph, Question
from llm import load_config


failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def fixture_question() -> Question:
    paragraphs = [
        Paragraph(1, 0, "Alpha", "Alpha was founded by Beta.", True),
        Paragraph(2, 1, "Noise", "Unrelated material.", False),
        Paragraph(3, 2, "Beta", "Beta was born in Gamma City.", True),
        Paragraph(4, 3, "Gamma City", "Gamma City had a population of 7531 in 1900.", True),
    ]
    decomposition = [
        {"step": 0, "question": "Alpha >> founded by", "answer": "Beta", "support_pid": 1},
        {"step": 1, "question": "#1 >> place of birth", "answer": "Gamma City", "support_pid": 3},
        {"step": 2, "question": "What was the population of #2 in 1900?", "answer": "7531", "support_pid": 4},
    ]
    return Question(
        qid="fixture", question="What was the 1900 population of the founder's birthplace?",
        answer="7531", aliases=["7,531"], paragraphs=paragraphs,
        decomposition=decomposition, gold_pids=[1, 3, 4],
        gold_sentences=[
            GoldSentence(0, 1, paragraphs[0].text, "Beta", False),
            GoldSentence(1, 3, paragraphs[2].text, "Gamma City", False),
            GoldSentence(2, 4, paragraphs[3].text, "7531", True),
        ], n_hops=3,
    )


class FakeResult:
    def __init__(self, text="notes"):
        self.text = text
        self.prompt_tokens = 10
        self.completion_tokens = 4
        self.cached = False
        self.finish_reason = "stop"
        self.cost_usd = 0.0


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return FakeResult()


q = fixture_question()
packets = inc_data.build_packets(q)

print("=== packet construction ===")
check("one packet per supporting paragraph", len(packets) == 3)
check("decomposition order is retained", [p.support_pid for p in packets] == [1, 3, 4])
check("placeholder #1 resolves", "Beta" in packets[1].probes[0].question)
check("placeholder #2 resolves", "Gamma City" in packets[2].probes[0].question)
check("final probe retains main aliases", set(packets[2].probes[0].golds) == {"7531", "7,531"})
rendered = inc_data.render_packet(packets[0], 1)
check("packet rendering contains evidence", "Alpha was founded by Beta" in rendered)
check("packet rendering hides probe question", packets[0].probes[0].question not in rendered)

print("\n=== deterministic order and schedule ===")
order_a, packets_a = inc_data.order_packets(q, "counterbalanced", 99)
order_b, packets_b = inc_data.order_packets(q, "counterbalanced", 99)
check("counterbalanced assignment is deterministic",
      order_a == order_b and [p.packet_id for p in packets_a] == [p.packet_id for p in packets_b])
assigned = [incremental.assigned_order_mode("counterbalanced", index, 99) for index in range(4)]
check("selected examples are exactly counterbalanced", assigned.count("forward") == assigned.count("reverse") == 2)
schedule0 = inc_data.build_schedule(packets, 0)
schedule5 = inc_data.build_schedule(packets, 5)
check("r0 contains specialists only", len(schedule0) == 3 and all(s.agent_type == "specialist" for s in schedule0))
check("r5 inserts relays only between specialists", len(schedule5) == 13)
check("specialist evidence order is invariant to relay depth",
      [s.packet_id for s in schedule0 if s.agent_type == "specialist"] ==
      [s.packet_id for s in schedule5 if s.agent_type == "specialist"])
intro0 = inc_data.introduction_stages(schedule0)
intro5 = inc_data.introduction_stages(schedule5)
check("r0 final ages are 2,1,0",
      [inc_data.handoff_age(3, intro0[p.packet_id]) for p in packets] == [2, 1, 0])
check("r5 final ages are 12,6,0",
      [inc_data.handoff_age(13, intro5[p.packet_id]) for p in packets] == [12, 6, 0])

print("\n=== prompt and isolation controls ===")
incremental.prompt_equivalence_selftest(packets[0])
check("specialist conditions differ only by question block", True)
signature = inspect.signature(run_chain.recompress)
check("relay interface has no evidence-packet parameter", "packet" not in signature.parameters)
cfg = load_config(ROOT / "config.yaml")
client = FakeClient()
base_handoff = hm.Handoff("fixture", "incremental_fixture_r1_d1", 1, "old notes")
sealed = hm.seal(q.question, base_handoff)
relay = run_chain.recompress(client, sealed, cfg, 2, 2, True, decoding_seed=4242,
                             tag="incremental_test_relay")
relay_prompt = client.calls[-1]["messages"][1]["content"]
check("relay receives previous message", "old notes" in relay_prompt)
check("relay receives no new evidence", "New evidence packet" not in relay_prompt and "Alpha was" not in relay_prompt)
check("incremental override controls relay seed", client.calls[-1]["seed"] == 4242)
client_specialist = FakeClient()
specialist = incremental.specialist_update(
    client_specialist, sealed, packets[1], 2, q.qid, q.question, cfg, 2, 3,
    True, "incremental_fixture_r1_d3", 200000,
)
specialist_prompt = client_specialist.calls[-1]["messages"][1]["content"]
check("specialist receives sealed notes and exactly its new packet",
      "old notes" in specialist_prompt and packets[1].text in specialist_prompt
      and packets[0].text not in specialist_prompt and packets[2].text not in specialist_prompt)
check("specialist and relay share the configured output budget",
      client_specialist.calls[-1]["max_tokens"] == client.calls[-1]["max_tokens"]
      == cfg["decoding"]["handoff_max_tokens"])
check("event seed cannot depend on physical stage",
      "stage" not in inspect.signature(incremental.event_seed).parameters)
check("specialist output remains Handoff-compatible",
      hm.Handoff.from_json(specialist.to_json()).qid == q.qid)
client2 = FakeClient()
legacy = run_chain.recompress(client2, sealed, cfg, 2, 3, True)
check("legacy relay seed remains unchanged", client2.calls[-1]["seed"] == 2003)
check("legacy handoff remains round-trip compatible",
      hm.Handoff.from_json(legacy.to_json()).text == legacy.text)

print("\n=== fact age records ===")
args = SimpleNamespace(conditions=["conditioned"], relay_depths=[1], seeds=[1])
schedule1 = inc_data.build_schedule(packets, 1)
handoff_rows = {}
for step in schedule1:
    text = "Beta Gamma City 7531" if step.stage == len(schedule1) else "Beta Gamma City"
    handoff_rows[("conditioned", 1, 1, q.qid, step.stage)] = {
        "text": text, "agent_type": step.agent_type,
    }
survival = incremental.fact_survival_rows([q], {q.qid: ("forward", packets)}, args, handoff_rows)
check("no fact is measured before introduction", all(r["handoff_age"] >= 0 for r in survival))
check("introduced facts start at age zero", all(any(r["packet_id"] == p.packet_id and r["handoff_age"] == 0
                                                     for r in survival) for p in packets))

print("\n=== future-query regret ===")
probe_rows = [
    {"source": "original_evidence", "qid": "q", "probe_id": "p", "em": 1.0, "f1": 1.0,
     "judge_correct": 1},
    {"source": "handoff", "qid": "q", "probe_id": "p", "em": 0.0, "f1": 0.25,
     "judge_correct": 0, "final_handoff": True},
]
incremental.annotate_future_query_regret(probe_rows)
check("future-query regret uses original minus handoff F1",
      probe_rows[1]["future_query_regret_f1"] == 0.75)
check("future-query regret uses existing judge metric",
      probe_rows[1]["future_query_regret_judge_correct"] == 1.0)

print("\n=== compatible analysis outputs ===")
analysis_args = SimpleNamespace(conditions=["conditioned"], relay_depths=[0, 1], seeds=[1])
main_rows = []
for qid, score in (("q1", 1.0), ("q2", 0.5)):
    main_rows.append({"source": "original_evidence", "qid": qid, "em": score, "f1": score,
                      "judge_correct": int(score == 1.0)})
    for depth in (0, 1):
        main_rows.append({"source": "final_handoff", "condition": "conditioned",
                          "relay_depth": depth, "qid": qid, "em": score - .1 * depth,
                          "f1": score - .1 * depth, "judge_correct": int(score == 1.0),
                          "packet_count": 3, "stage": 3 + 2 * depth,
                          "final_handoff_characters": 100,
                          "chain_prompt_tokens": 20, "chain_completion_tokens": 10})
probe_rows = []
survival_rows = []
for qid in ("q1", "q2"):
    probe_rows.append({"source": "original_evidence", "qid": qid, "probe_id": f"{qid}:p",
                       "em": 1.0, "f1": 1.0, "judge_correct": 1})
    for depth in (0, 1):
        for age in (0, 1):
            probe_rows.append({"source": "handoff", "condition": "conditioned",
                               "relay_depth": depth, "qid": qid, "probe_id": f"{qid}:p",
                               "handoff_age": age, "em": 1.0 - .1 * age,
                               "f1": 1.0 - .1 * age, "judge_correct": 1,
                               "final_handoff": age == 1})
            survival_rows.append({"condition": "conditioned", "relay_depth": depth,
                                  "qid": qid, "handoff_age": age,
                                  "answer_string_survival": 1.0 - .5 * age,
                                  "source_topic_survival": 1.0})
incremental.annotate_future_query_regret(probe_rows)
analysis_cfg = {"analysis": {"bootstrap_resamples": 100, "ci_level": .95},
                "evidence_order": {"mode": "counterbalanced"}}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    incremental.analyse(main_rows, probe_rows, survival_rows, analysis_cfg, analysis_args, root,
                        {"cost_usd": 0, "calls_live": 0, "calls_cached": 0,
                         "prompt_tokens": 0, "completion_tokens": 0})
    expected = ["stage_metrics.csv", "relay_depth_deltas.csv", "probe_metrics.csv",
                "future_query_regret.csv", "survival_by_age.csv", "survival.jsonl",
                "incremental_chain.png", "report.md"]
    check("analysis emits all compatible artifacts", all((root / name).exists() for name in expected))
    check("combined plot is non-empty", (root / "incremental_chain.png").stat().st_size > 1000)

print("\n=== persisted-run manifest ===")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    manifest = {"experiment": "fixture", "model": "m1"}
    incremental.validate_run_manifest(root, manifest, force=False, dry_run=False)
    check("run manifest is written", (root / "manifest.json").exists())
    rejected = False
    try:
        incremental.validate_run_manifest(root, {**manifest, "model": "m2"}, force=False, dry_run=False)
    except RuntimeError:
        rejected = True
    check("stale persisted rows are rejected", rejected)

print("\n" + "=" * 60)
print("ALL INCREMENTAL CHAIN CHECKS PASSED" if not failures else "FAILURES: " + ", ".join(failures))
print("=" * 60)
raise SystemExit(1 if failures else 0)
