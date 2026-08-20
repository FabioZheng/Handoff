"""Offline regression checks for repeated handoffs and dataset/context adapters."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_data
import run_chain
from data import GoldSentence, Paragraph, Question
from llm import load_config


failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def fixture_question():
    paragraphs = [
        Paragraph(1, 0, "Gold A", "Alpha links to beta.", True),
        Paragraph(2, 1, "Distractor A", "Unrelated one.", False),
        Paragraph(3, 2, "Gold B", "Beta gives the answer gamma.", True),
        Paragraph(4, 3, "Distractor B", "Unrelated two.", False),
        Paragraph(5, 4, "Distractor C", "Unrelated three.", False),
        Paragraph(6, 5, "Distractor D", "Unrelated four.", False),
    ]
    return Question(
        qid="fixture", question="What is the answer?", answer="gamma", aliases=[],
        paragraphs=paragraphs, decomposition=[], gold_pids=[1, 3],
        gold_sentences=[
            GoldSentence(0, 1, paragraphs[0].text, "beta", False),
            GoldSentence(1, 3, paragraphs[2].text, "gamma", True),
        ], n_hops=2,
    )


print("=== evidence-length variants ===")
cfg = load_config(ROOT / "chain_config.yaml")
q = fixture_question()
full = chain_data.select_paragraphs(q, "full", cfg["contexts"])
check("full has every document", len(full) == 6)
check("full retains every gold document", {1, 3}.issubset({p.pid for p in full}))

print("\n=== chain isolation ===")
try:
    run_chain.isolation_selftest()
    check("later compressors reject unsealed inputs", True)
except AssertionError as exc:
    check("later compressors reject unsealed inputs", False, str(exc))

print("\n=== question persistence manifest ===")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "questions.jsonl"
    manifest = {"dataset": "fixture", "model": "m"}
    chain_data.write_questions([q], path, manifest)
    loaded = chain_data.read_questions(path, manifest)
    check("matching manifest loads questions", loaded is not None and loaded[0].qid == q.qid)
    check("mismatched manifest refuses stale data",
          chain_data.read_questions(path, {"dataset": "other", "model": "m"}) is None)

print("\n=== generated Wikipedia adapter ===")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    dataset_path = root / "questions.jsonl"
    question = fixture_question()
    dataset_path.write_text(json.dumps({"_manifest": {}}) + "\n" + json.dumps(question.to_json()) + "\n", encoding="utf-8")
    questions = chain_data.sample_candidates(
        "wikipedia_random", {"source": "generated_wikipedia", "local_jsonl": "questions.jsonl"},
        root, cfg["sampling"]["seed"], 5,
    )
    check("generated Wikipedia questions load", len(questions) == 1 and questions[0].qid == "fixture")
    check("generated Wikipedia source is preserved", questions[0].paragraphs[0].text == "Alpha links to beta.")

print("\n=== plot generation ===")
fake_metrics = []
for dataset in ("wikipedia_random",):
    for variant in ("full",):
        for depth in (0, 1, 2, 3, 5):
            fake_metrics.append({
                "dataset": dataset, "context_variant": variant, "depth": depth,
                "f1": 0.8 - 0.03 * depth, "f1_lo": 0.7 - 0.03 * depth,
                "f1_hi": 0.9 - 0.03 * depth,
            })
with tempfile.TemporaryDirectory() as tmp:
    destination = Path(tmp) / "plot.png"
    class Args:
        datasets = ["wikipedia_random"]
        contexts = ["full"]
        depths = [0, 1, 2, 3, 5]
    run_chain.make_plot(fake_metrics, Args(), destination)
    check("degradation plot is produced", destination.exists() and destination.stat().st_size > 1000)

print("\n" + "=" * 60)
print("ALL CHAIN CHECKS PASSED" if not failures else "FAILURES: " + ", ".join(failures))
print("=" * 60)
raise SystemExit(1 if failures else 0)
