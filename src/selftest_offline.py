"""Offline checks that need no API key: data shaping, gold-sentence derivation,
scoring metrics, the oracle handoff, and the Stage-2 isolation guarantee."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod
import handoffs as hm
from llm import load_config, cache_key
from score import (normalize_answer, exact_match, token_f1, score_against_golds,
                   extract_short_answer, paired_bootstrap_delta)
import numpy as np

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=== scoring ===")
check("normalisation strips articles/punct/case",
      normalize_answer("The  Beatles, ") == "beatles", normalize_answer("The  Beatles, "))
check("EM is normalised", exact_match("the Beatles.", "Beatles") == 1.0)
check("EM rejects a different span", exact_match("Rolling Stones", "Beatles") == 0.0)
check("F1 partial credit", 0.4 < token_f1("Steve Hillage band", "Steve Hillage") < 1.0,
      f"{token_f1('Steve Hillage band', 'Steve Hillage'):.3f}")
check("aliases are used", score_against_golds("NYC", ["New York City", "NYC"]) == (1.0, 1.0))
check("answer extraction strips label",
      extract_short_answer("**Answer:** Miquette Giraudy.") == "Miquette Giraudy",
      extract_short_answer("**Answer:** Miquette Giraudy."))
check("answer extraction takes first line",
      extract_short_answer("Paris\n\nBecause the text says so.") == "Paris")
check("answer extraction strips quotes", extract_short_answer('"Green"') == "Green")

print("\n=== bootstrap ===")
rng = np.random.default_rng(0)
a = rng.random(200)
d0 = paired_bootstrap_delta(a, a, 2000, 0.95, seed=1)
check("identical arrays give zero delta", abs(d0["delta"]) < 1e-12 and d0["p_value"] >= 0.99,
      f"delta={d0['delta']:.2e} p={d0['p_value']:.3f}")
d1 = paired_bootstrap_delta(a + 0.2, a, 2000, 0.95, seed=1)
check("a real shift is detected", d1["lo"] > 0 and d1["p_value"] < 0.05,
      f"delta={d1['delta']:.3f} CI=[{d1['lo']:.3f},{d1['hi']:.3f}] p={d1['p_value']:.4f}")

print("\n=== cache key ===")
m = [{"role": "user", "content": "hi"}]
p = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8, "seed": None, "response_format": None}
check("cache key is stable", cache_key("m", m, p) == cache_key("m", m, p))
check("cache key tracks temperature",
      cache_key("m", m, p) != cache_key("m", m, {**p, "temperature": 0.7}))
check("cache key tracks seed", cache_key("m", m, p) != cache_key("m", m, {**p, "seed": 1}))

print("\n=== sentence splitting ===")
s = data_mod.split_sentences(
    "Green is the fourth studio album by Steve Hillage. It was written in 1977. "
    "Dr. Smith produced it. The album sold well.")
check("splits into 4 sentences", len(s) == 4, f"{len(s)}: {s}")
check("abbreviation not split", any(x.startswith("Dr. Smith") for x in s), str(s))

print("\n=== dataset shaping ===")
cfg = load_config(ROOT / "config.yaml")
qs = data_mod.sample_candidates(cfg, ROOT)[:40]
check("candidates built", len(qs) == 40)

q = qs[0]
check("paragraph ids are 1..n and contiguous",
      [p.pid for p in q.paragraphs] == list(range(1, len(q.paragraphs) + 1)))
check("gold pids match supporting flags",
      set(q.gold_pids) == {p.pid for p in q.paragraphs if p.is_supporting})
check("has distractors", len(q.paragraphs) > len(q.gold_pids),
      f"{len(q.paragraphs)} paras, {len(q.gold_pids)} gold")

# Shuffle must be deterministic across independent calls.
import pandas as pd
df = pd.read_parquet(ROOT / cfg["dataset"]["local_parquet"])
row = df[df["id"] == q.qid].iloc[0]
q2 = data_mod.build_question(row, cfg["sampling"]["seed"])
check("paragraph order is deterministic",
      [p.orig_idx for p in q.paragraphs] == [p.orig_idx for p in q2.paragraphs])

print("\n=== gold sentence derivation (all 40 candidates) ===")
n_no_sent, n_final_missing, verbatim_bad, total_sents = 0, 0, 0, 0
covered_gold_paras = 0
for qq in qs:
    if not qq.gold_sentences:
        n_no_sent += 1
    if not any(g.is_final_hop for g in qq.gold_sentences):
        n_final_missing += 1
    pid_text = {p.pid: p.text for p in qq.paragraphs}
    for g in qq.gold_sentences:
        total_sents += 1
        if g.text not in pid_text[g.pid]:
            verbatim_bad += 1
    if {g.pid for g in qq.gold_sentences} == set(qq.gold_pids):
        covered_gold_paras += 1

check("every question yields gold sentences", n_no_sent == 0, f"{n_no_sent} without")
check("every question has a final-hop sentence", n_final_missing == 0, f"{n_final_missing} without")
check("all gold sentences are verbatim substrings of their paragraph",
      verbatim_bad == 0, f"{verbatim_bad}/{total_sents} bad")
check("gold sentences span every gold paragraph",
      covered_gold_paras == len(qs), f"{covered_gold_paras}/{len(qs)}")

# Does the derived oracle actually contain the final answer? If not, E_oracle is broken.
contains_answer = sum(
    1 for qq in qs
    if any(data_mod._contains(g.text, qq.answer) for g in qq.gold_sentences)
)
check("oracle sentences contain the gold answer string",
      contains_answer >= int(0.9 * len(qs)),
      f"{contains_answer}/{len(qs)} ({100*contains_answer/len(qs):.0f}%)")

print("\n=== oracle handoff ===")
h = hm.make_oracle(q)
check("oracle uses zero LLM calls", h.meta["llm_calls"] == 0)
check("oracle text is non-empty", len(h.text) > 0)
check("oracle is much shorter than full context",
      len(h.text) < 0.5 * len(hm.render_paragraphs(q)),
      f"{len(h.text)} vs {len(hm.render_paragraphs(q))} chars")
check("oracle cites paragraph ids", h.text.startswith("[P"))

print("\n=== structured rendering + injection substrate ===")
obj = {"claims": [
    {"text": "Green is an album by Steve Hillage.", "source_para_id": 3,
     "constraints": ["released 1977"], "confidence": "high"},
    {"text": "Miquette Giraudy is Hillage's partner.", "source_para_id": 7,
     "constraints": [], "confidence": "medium"}],
    "unresolved": ["exact wedding date"]}
txt = hm.render_structured(obj)
check("render cites sources", "P3" in txt and "P7" in txt)
check("render includes constraints", "released 1977" in txt)
check("render includes unresolved", "exact wedding date" in txt)
stripped = hm.render_structured({"claims": [{**c, "source_para_id": None} for c in obj["claims"]],
                                 "unresolved": obj["unresolved"]})
check("provenance can be nulled without losing claim text",
      "unknown source" in stripped and "Steve Hillage" in stripped)

print("\n=== stage-2 isolation ===")
import run as run_mod
try:
    run_mod.selftest_orchestrator_isolation()
    check("orchestrator rejects non-sealed inputs", True)
except AssertionError as exc:
    check("orchestrator rejects non-sealed inputs", False, str(exc))

sealed = hm.seal(q.question, h)
leaked = [p for p in q.paragraphs if p.text and p.text in sealed.handoff_text]
check("no full paragraph reaches the orchestrator via the oracle handoff",
      len(leaked) == 0, f"{len(leaked)} paragraphs appeared verbatim")

print("\n=== example question ===")
print(f"qid       : {q.qid}  ({q.n_hops} hops)")
print(f"question  : {q.question}")
print(f"answer    : {q.answer}")
print(f"gold paras: {q.gold_pids} of {len(q.paragraphs)}")
print(f"full ctx  : {len(hm.render_paragraphs(q))} chars")
print(f"oracle    : {len(h.text)} chars")
print("oracle handoff:")
for line in h.text.splitlines():
    print("   " + line[:150])

print("\n" + "=" * 60)
print(f"{'ALL OFFLINE CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
print("=" * 60)
sys.exit(1 if fails else 0)
