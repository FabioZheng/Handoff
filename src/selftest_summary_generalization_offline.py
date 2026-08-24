"""Offline regression checks for Experiment 5's arms and transition metrics.

Runs with no API key and no network: every check is either pure string
arithmetic or exercises a code path with a stub client. Covers the two
additions (per-edge paraphrase measurement, paraphrase-only and pass-through
arms) plus the backward-compatibility guarantees the original two arms rely on.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paraphrase_metrics as pmx  # noqa: E402
import run_summary_generalization as rsg  # noqa: E402
from judge import parse_preservation, preservation_user_prompt  # noqa: E402
from llm import load_config  # noqa: E402


failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def fixture_pair() -> dict:
    return {
        "pair_id": "fixture",
        "question_A": "Who signed the treaty in 1802?",
        "question_B": "Which city hosted the conference?",
        "golds_A": ["Marshal Vance"],
        "golds_B": ["Bergenholm"],
        "passages": [{
            "role": "gold_AB",
            "text": ("Marshal Vance signed the treaty in 1802. The negotiating conference "
                     "had been hosted by Bergenholm the previous winter."),
        }],
        "gold_position": 1,
        "context_chars": 140,
    }


def fixture_cfg(modes) -> dict:
    return {
        "modes": list(modes),
        "max_depth": 3,
        "depths": [0, 1, 2, 3],
        "analysis": {"bootstrap_resamples": 200, "ci_level": 0.95},
    }


print("=== deterministic transition measures ===")
identical = pmx.transition_measures("alpha beta gamma delta", "alpha beta gamma delta")
check("identical text scores lexical similarity 1.0", identical["lexical_similarity"] == 1.0)
check("identical text has zero novel tokens", identical["novel_token_rate"] == 0.0)
check("identical text is fully copied verbatim", identical["ngram_copy_rate"] == 1.0)

reworded = pmx.transition_measures(
    "The marshal signed the treaty in 1802.",
    "In 1802, a treaty was put to signature by the marshal.")
check("a reworded sentence keeps most tokens", reworded["retained_token_rate"] >= 0.7,
      f"retained={reworded['retained_token_rate']}")
check("a reworded sentence breaks verbatim n-grams", reworded["ngram_copy_rate"] < 0.5,
      f"copy={reworded['ngram_copy_rate']}")

disjoint = pmx.transition_measures("alpha beta gamma", "delta epsilon zeta")
check("disjoint text scores lexical similarity 0.0", disjoint["lexical_similarity"] == 0.0)
check("disjoint text is entirely novel", disjoint["novel_token_rate"] == 1.0)

shorter = pmx.transition_measures("a" * 100, "a" * 25)
check("length ratio tracks shrinkage", abs(shorter["length_ratio"] - 0.25) < 1e-9)
check("empty previous text yields NaN length ratio",
      pmx.transition_measures("", "abc")["length_ratio"] != pmx.transition_measures("", "abc")["length_ratio"])

print("\n=== edge classification ===")
check("reworded + preserved + fact kept + right answer is benign",
      pmx.classify_transition(0.5, 1.0, True, 1.0) == "benign_paraphrase")
check("reworded + preserved + fact lost is a critical-detail loss",
      pmx.classify_transition(0.5, 1.0, False, 1.0) == "critical_detail_loss")
check("reworded + preserved + wrong answer is a critical-detail loss",
      pmx.classify_transition(0.5, 1.0, True, 0.0) == "critical_detail_loss")
check("degraded + wrong answer is information loss",
      pmx.classify_transition(0.5, 0.0, False, 0.0) == "information_loss")
check("degraded + right answer is recorded as drift, not success",
      pmx.classify_transition(0.5, 0.0, True, 1.0) == "semantic_drift")
check("an unchanged message is a verbatim copy",
      pmx.classify_transition(0.99, None, True, 1.0) == "verbatim_copy")
check("a missing semantic verdict never scores as success",
      pmx.classify_transition(0.5, None, True, 1.0) == "unresolved")
check("an unmeasurable probe is not counted as a lost fact",
      pmx.classify_transition(0.5, 1.0, None, 1.0) == "benign_paraphrase")

print("\n=== preservation judge parsing ===")
for verdict, expected in (("EQUIVALENT", 1.0), ("MINOR_LOSS", 0.5), ("MAJOR_LOSS", 0.0),
                          ("  major_loss  ", 0.0), ("MINOR LOSS", 0.5)):
    score, _, ok = parse_preservation(verdict)
    check(f"parses {verdict.strip()!r}", ok and score == expected, f"got {score}")
score, _, ok = parse_preservation("I am not sure")
check("an unparsed verdict is left unscored", not ok and score != score)
prompt = preservation_user_prompt("first notes", "second notes")
check("preservation prompt carries both messages",
      "first notes" in prompt and "second notes" in prompt)

print("\n=== arm prompts stay matched ===")
pair = fixture_pair()
cfg = fixture_cfg(["conditioned", "generic", "paraphrase", "passthrough"])
try:
    rsg.prompt_difference_selftest(pair, cfg)
    check("all arms differ only where intended", True)
except AssertionError as exc:
    check("all arms differ only where intended", False, str(exc))

for stage in (1, 2):
    blind = rsg.compression_user_prompt(pair, "previous notes", "paraphrase", stage, cfg)
    check(f"paraphrase arm stage {stage} sees no question",
          pair["question_A"] not in blind and pair["question_B"] not in blind)
    check(f"paraphrase arm stage {stage} forbids compression",
          "not a summary" in blind and "do not condense" in blind)
conditioned = rsg.compression_user_prompt(pair, "previous notes", "conditioned", 2, cfg)
check("conditioned arm still carries question A", pair["question_A"] in conditioned)
check("conditioned arm keeps the original instruction",
      rsg.RECOMPRESS_INSTRUCTION in conditioned)

print("\n=== length-target guard ===")
try:
    rsg.validate_modes({"modes": ["paraphrase"], "length_target_words": 120})
    check("a word budget cannot be combined with the paraphrase arm", False)
except ValueError:
    check("a word budget cannot be combined with the paraphrase arm", True)
try:
    rsg.validate_modes({"modes": ["conditioned", "generic"], "length_target_words": 120})
    check("a word budget is still allowed for the original arms", True)
except ValueError as exc:
    check("a word budget is still allowed for the original arms", False, str(exc))


class RefusingClient:
    """Any model call at all is a failure for the pass-through arm."""

    def chat(self, *a, **k):
        raise AssertionError("passthrough must not issue a model call")


print("\n=== pass-through arm ===")
stage1 = rsg.compress(RefusingClient(), pair, None, "passthrough", 1, cfg)
check("passthrough stage 1 forwards the raw context", stage1["text"] == rsg.render_context(pair))
check("passthrough reports no tokens spent",
      stage1["prompt_tokens"] == 0 and stage1["completion_tokens"] == 0)
check("passthrough is marked distinctly, not as a truncation",
      stage1["finish_reason"] == "passthrough")
stage2 = rsg.compress(RefusingClient(), pair, "carried message", "passthrough", 2, cfg)
check("passthrough stage 2 forwards its input unchanged", stage2["text"] == "carried message")

print("\n=== transition assembly ===")
handoffs = {}
texts = {
    1: "Marshal Vance signed the treaty in 1802. Bergenholm hosted the conference.",
    2: "In 1802 the treaty was signed by Marshal Vance; the conference sat at Bergenholm.",
    3: "A treaty was signed in 1802 by Marshal Vance.",  # drops fact B here
}
for stage, text in texts.items():
    handoffs[(pair["pair_id"], "generic", stage)] = {
        "pair_id": pair["pair_id"], "mode": "generic", "stage": stage, "text": text}
small_cfg = fixture_cfg(["generic"])
transitions = rsg.build_transition_rows([pair], handoffs, small_cfg)
check("one row per consecutive edge", len(transitions) == 3, f"got {len(transitions)}")
check("edge 1 is labelled source compression", transitions[0]["edge_type"] == "source_compression")
check("later edges are labelled message rewrites",
      all(r["edge_type"] == "message_rewrite" for r in transitions[1:]))
check("fact B survives the first two edges",
      transitions[0]["fact_B_survived"] == 1.0 and transitions[1]["fact_B_survived"] == 1.0)
check("fact B is gone after edge 3", transitions[2]["fact_B_survived"] == 0.0)
check("the loss is attributed to edge 3 only",
      transitions[2]["fact_B_lost_here"] == 1.0
      and transitions[1]["fact_B_lost_here"] == 0.0)
check("fact A survives every edge",
      all(r["fact_A_survived"] == 1.0 for r in transitions))

short_gold = dict(pair, golds_B=["a"])
unmeasurable = rsg.build_transition_rows([short_gold], handoffs, small_cfg)
check("a too-short gold is reported unmeasurable, not lost",
      unmeasurable[0]["fact_B_measurable"] is False)

print("\n=== answer join, labelling, and caching ===")
answer_rows = [
    {"mode": "generic", "query_type": "target", "depth": 1, "pair_id": "fixture",
     "em": 1.0, "f1": 1.0, "judge_correct": 1},
    {"mode": "generic", "query_type": "heldout", "depth": 3, "pair_id": "fixture",
     "em": 0.0, "f1": 0.0, "judge_correct": 0},
]
rsg.attach_answer_outcomes(transitions, answer_rows)
check("target answer joins onto its own edge", transitions[0]["target_judge_correct"] == 1)
check("heldout answer joins onto its own edge", transitions[2]["heldout_judge_correct"] == 0)
check("edges without an answer stay unannotated", "target_judge_correct" not in transitions[1])

rsg.label_transitions(transitions, small_cfg)
check("labels are assigned without a semantic verdict",
      all(r["transition_label_target"] == "unresolved" for r in transitions[1:]),
      str([r["transition_label_target"] for r in transitions]))

for row in transitions:
    row["semantic_preserved"] = 1.0
    row["semantic_verdict"] = "EQUIVALENT"
rsg.label_transitions(transitions, small_cfg)
check("a preserved rewrite that drops fact B is a critical-detail loss",
      transitions[2]["transition_label_heldout"] == "critical_detail_loss",
      transitions[2]["transition_label_heldout"])

stored = [rsg.storable_transition(r) for r in transitions]
check("stored rows drop the full message texts",
      all("previous_text" not in r and "current_text" not in r for r in stored))
check("stored rows keep a hash identifying the pair",
      all(len(r["text_pair_hash"]) == 16 for r in stored))

rebuilt = rsg.build_transition_rows([pair], handoffs, small_cfg)
rsg.merge_cached_transitions(rebuilt, stored)
check("cached verdicts are reused for unchanged messages",
      all(r.get("semantic_preserved") == 1.0 for r in rebuilt))

handoffs[(pair["pair_id"], "generic", 2)]["text"] = "Completely different wording now."
changed = rsg.build_transition_rows([pair], handoffs, small_cfg)
rsg.merge_cached_transitions(changed, stored)
check("a changed message discards its stale verdict",
      "semantic_preserved" not in changed[1])
handoffs[(pair["pair_id"], "generic", 2)]["text"] = texts[2]

print("\n=== annotation writes back to the per-stage rows ===")
rsg.annotate_handoff_rows(handoffs, transitions)
row = handoffs[(pair["pair_id"], "generic", 3)]
check("handoff rows keep their original fields",
      row["pair_id"] == "fixture" and row["mode"] == "generic" and row["stage"] == 3
      and row["text"] == texts[3])
check("handoff rows gain the transition annotations",
      all(f in row for f in ("lexical_similarity", "fact_B_survived", "semantic_verdict")))

print("\n=== analysis and plots ===")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    rsg.analyse_transitions(transitions, small_cfg, root)
    check("transition_metrics.csv is written", (root / "transition_metrics.csv").exists())
    check("paraphrase_transitions.png is written", (root / "paraphrase_transitions.png").exists())
    header = (root / "transition_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    for column in ("lexical_similarity", "semantic_preserved", "fact_B_survived",
                   "label_target_benign_paraphrase"):
        check(f"transition_metrics.csv carries {column}", column in header)

print("\n=== backward compatibility ===")
check("original arms produce the original contrast name",
      ("conditioned", "generic") == rsg.CONTRAST_PAIRS[0])
present = [f"{t}_minus_{c}" for t, c in rsg.CONTRAST_PAIRS
           if t in ("conditioned", "generic") and c in ("conditioned", "generic")]
check("a two-arm config yields exactly one contrast, unchanged",
      present == ["conditioned_minus_generic"], str(present))
for name in ("summary_generalization_squad_pairs_config.yaml",
             "summary_generalization_squad_pairs_goldonly_config.yaml",
             "summary_generalization_paraphrase_config.yaml"):
    path = ROOT / name
    if not path.exists():
        check(f"{name} exists", False)
        continue
    loaded = load_config(path)
    try:
        rsg.validate_modes(loaded)
        check(f"{name} passes the mode guard", True)
    except ValueError as exc:
        check(f"{name} passes the mode guard", False, str(exc))
paraphrase_cfg = load_config(ROOT / "summary_generalization_paraphrase_config.yaml")
check("paraphrase config runs all four arms",
      paraphrase_cfg["modes"] == ["conditioned", "generic", "paraphrase", "passthrough"])
goldonly_cfg = load_config(ROOT / "summary_generalization_squad_pairs_goldonly_config.yaml")
check("the published gold-only config is untouched",
      goldonly_cfg["modes"] == ["conditioned", "generic"])
check("the paraphrase run writes to its own output roots",
      paraphrase_cfg["outputs"]["run_root"] != goldonly_cfg["outputs"]["run_root"]
      and paraphrase_cfg["outputs"]["result_root"] != goldonly_cfg["outputs"]["result_root"])
for key in ("model", "decoding", "dataset"):
    matched = paraphrase_cfg[key] == goldonly_cfg[key]
    check(f"paraphrase config matches gold-only on {key}", matched,
          "" if matched else f"{paraphrase_cfg[key]} != {goldonly_cfg[key]}")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
raise SystemExit(1 if failures else 0)
