"""Offline checks for Experiment 10 (communication regret under a hard budget).

No API key, no network, no writes into data/, runs/ or results/. Everything the
experiment claims to control is checked here on a synthetic corpus: that the
policies differ only in their conditioning block, that the delivered budget is
never exceeded, that the rotation puts every question on both the diagonal and
the off-diagonal, that a rotation-invariant policy is unspecialised by
construction, that the answerer can never see the source, and that the Pareto
and regret arithmetic is what the report says it is.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import budget as bd  # noqa: E402
import build_regret_data as build  # noqa: E402
import regret_data as rd  # noqa: E402
import run_communication_regret as exp  # noqa: E402
from llm import load_config  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------------------
# A synthetic corpus, so the checks never depend on a built dataset.

SOURCE_SQUAD = (
    "The Verrick Institute was founded in 1874 by Aldous Penhale on a strip of "
    "reclaimed marsh east of Carrow. Its first building, Marlow Hall, held two "
    "lecture rooms and a small observatory whose refractor measured 9 inches. "
    "The Institute charged a yearly subscription of 12 guineas until 1901, when "
    "the Trethowan bequest allowed it to be abolished. Every autumn since 1888 "
    "the Institute has held a public lecture known as the Carrow Address."
)

SQUAD_FIXTURE = {
    "context_id": "sq:1", "dataset": "squad_groups", "title": "Verrick_Institute",
    "source": SOURCE_SQUAD,
    "questions": [
        {"qid": "q1", "question": "Who founded the Verrick Institute?", "golds": ["Aldous Penhale"]},
        {"qid": "q2", "question": "What was the Institute's first building called?",
         "golds": ["Marlow Hall"]},
        {"qid": "q3", "question": "How large was the refractor?", "golds": ["9 inches"]},
        {"qid": "q4", "question": "What is the autumn public lecture called?",
         "golds": ["the Carrow Address"]},
    ],
    "meta": {},
}


def relation_fixture(index: int) -> dict:
    text = (
        f"The Kelder Union {index} was chartered in 18{index}4 by Ovina Reyst, who served as "
        "its first warden for eleven years. The charter itself, the Vessom Roll, was "
        "sealed at Brant Quay. Its premises stand on Lowther Street, a hall of four "
        "bays; the adjoining store, Corrin Shed, holds two presses. The Union levies a "
        "quarterly due of nine marks, and holds an endowment named the Wray Fund worth "
        "four hundred marks; the collector of the due is Tam Oswine. Each spring the "
        "Union keeps a custom called the Grey Walk, first held in 1902, and issues a "
        "pamphlet, the Kelder Sheet, printed in a run of three hundred."
    )
    facts = {
        "founding": [("anchor", "Who chartered the Union?", "Ovina Reyst"),
                     ("paraphrase", "By whom was the Union brought into being?", "Ovina Reyst"),
                     ("same_entity", "For how long did the first warden serve?", "eleven years"),
                     ("same_topic", "Where was the Vessom Roll sealed?", "Brant Quay")],
        "facilities": [("anchor", "On what street do the premises stand?", "Lowther Street"),
                       ("paraphrase", "Where are the Union's premises located?", "Lowther Street"),
                       ("same_entity", "How many bays does the hall have?", "four bays"),
                       ("same_topic", "How many presses does Corrin Shed hold?", "two presses")],
        "finance": [("anchor", "What is the quarterly due?", "nine marks"),
                    ("paraphrase", "How much is levied each quarter?", "nine marks"),
                    ("same_entity", "What is the endowment worth?", "four hundred marks"),
                    ("same_topic", "Who collects the due?", "Tam Oswine")],
        "custom": [("anchor", "What is the spring custom called?", "the Grey Walk"),
                   ("paraphrase", "By what name is the spring observance known?", "the Grey Walk"),
                   ("same_entity", "When was the custom first held?", "1902"),
                   ("same_topic", "What is the print run of the Kelder Sheet?",
                    "three hundred")],
    }
    questions = [{"qid": f"rel:{index}:{aspect}:{role}", "question": question,
                  "golds": [answer], "aspect": aspect, "role": role}
                 for aspect, entries in facts.items() for role, question, answer in entries]
    return {"context_id": f"rel:{index}", "dataset": "relation_dossiers",
            "title": f"Kelder Union {index}", "source": text, "questions": questions, "meta": {}}


tmp = Path(tempfile.mkdtemp(prefix="regret-selftest-"))
squad_path = tmp / "squad_groups.jsonl"
relation_path = tmp / "relation_dossiers.jsonl"
squad_path.write_text(
    json.dumps({"_manifest": {"dataset": "squad_groups"}}) + "\n"
    + json.dumps(SQUAD_FIXTURE) + "\n"
    + json.dumps({**SQUAD_FIXTURE, "context_id": "sq:2"}) + "\n", encoding="utf-8")
relation_path.write_text(
    json.dumps({"_manifest": {"dataset": "relation_dossiers"}}) + "\n"
    + "\n".join(json.dumps(relation_fixture(i)) for i in (1, 2)) + "\n", encoding="utf-8")

cfg = load_config(ROOT / "configs/communication_regret_config.yaml")
cfg["dataset"]["squad_groups"]["contexts_jsonl"] = str(squad_path)
cfg["dataset"]["relation_dossiers"]["contexts_jsonl"] = str(relation_path)
cfg["analysis"]["bootstrap_resamples"] = 200


print("=== configuration ===")
budgets = bd.budgets_from_config(cfg)
check("budgets are sorted and distinct",
      [b.words for b in budgets] == sorted({b.words for b in budgets}))
check("api_max_tokens is non-binding against the word cap",
      all(b.api_max_tokens > b.tokens for b in budgets),
      f"{[(b.words, b.tokens, b.api_max_tokens) for b in budgets]}")
check("every budget is a two-sided band",
      all(1 <= b.floor_words <= b.words for b in budgets),
      f"{[(b.floor_words, b.words) for b in budgets]}")
check("a fill floor outside (0, 1] is refused",
      raises(lambda: bd.make_budget(10, 1.35, 2.0, 96, 0.0), ValueError)
      and raises(lambda: bd.make_budget(10, 1.35, 2.0, 96, 1.5), ValueError))
check("every configured policy is known",
      all(p in bd.ALL_POLICIES for p in cfg["policies"]))
check("cost cap and api call cap are set",
      cfg["cost"]["cap_usd"] > 0 and cfg["cost"]["api_call_cap"] > 0)


print("\n=== the length contract ===")
small = bd.make_budget(8, 1.35, 2.0, 96, 0.85)
text = "Alpha beta gamma delta. Epsilon zeta eta theta iota. Kappa lambda."
cut, was_cut, dropped = bd.truncate_to_words(text, 4)
check("truncation respects the cap", bd.word_count(cut) <= 4)
check("truncation prefers a sentence boundary", cut == "Alpha beta gamma delta.")
check("truncation reports what it dropped", was_cut and dropped == 7, f"{dropped}")
long_first = "one two three four five six seven eight nine ten"
hard, _, _ = bd.truncate_to_words(long_first, 3)
check("a first sentence that does not fit is hard-cut", hard == "one two three")
check("text under the cap is untouched", bd.truncate_to_words("a b c", 5)[0] == "a b c")
check("clean_message strips a lead-in",
      bd.clean_message("Here is the handoff: alpha beta") == "alpha beta")
check("clean_message strips a code fence",
      bd.clean_message("```\nalpha beta\n```") == "alpha beta")
sizes = [bd.word_count(bd.truncate_to_words(SOURCE_SQUAD, b.words)[0]) <= b.words
         for b in budgets]
check("every budget truncates the fixture source within cap", all(sizes))


print("\n=== policies differ only in the conditioning block ===")
b = budgets[1]
questions = tuple(q["question"] for q in SQUAD_FIXTURE["questions"])
prompts = {p: bd.sender_prompt(p, SOURCE_SQUAD, b, current_question=questions[0],
                               all_questions=questions)
           for p in bd.ABSTRACTIVE_POLICIES}
blocks = {p: bd.policy_block(p, questions[0], questions) for p in bd.ABSTRACTIVE_POLICIES}
skeletons = {p: prompts[p].replace(blocks[p], "<BLOCK>") for p in bd.ABSTRACTIVE_POLICIES}
check("all four abstractive prompts share one skeleton",
      len(set(skeletons.values())) == 1)
check("the generic prompt contains no question text",
      not any(q in prompts["generic"] for q in questions))
check("the generic prompt does not mention future questions",
      "further questions" not in prompts["generic"]
      and "later" not in prompts["generic"].lower())
check("the conditioned prompt contains exactly the current question",
      questions[0] in prompts["conditioned"]
      and not any(q in prompts["conditioned"] for q in questions[1:]))
check("the reusable prompt adds future-awareness to the conditioned one",
      questions[0] in prompts["reusable"] and "further questions" in prompts["reusable"])
check("the oracle prompt contains every question",
      all(q in prompts["oracle"] for q in questions))
check("the length contract is identical across policies",
      all(bd.LENGTH_CONTRACT.format(floor=b.floor_words, words=b.words) in p
          for p in prompts.values()))
check("the contract states both sides of the band",
      str(b.floor_words) in prompts["generic"] and str(b.words) in prompts["generic"])
extractive = bd.sender_prompt("extractive_conditioned", SOURCE_SQUAD, b,
                              current_question=questions[0])
check("the extractive arm swaps only the output form",
      bd.EXTRACTIVE_FORM in extractive and bd.PROSE_FORM not in extractive)
check("an oracle prompt without the question set is refused",
      raises(lambda: bd.policy_block("oracle", questions[0], None), ValueError))
check("a conditioned prompt without a current question is refused",
      raises(lambda: bd.policy_block("conditioned", None, questions), ValueError))
check("an unknown policy is refused",
      raises(lambda: bd.policy_block("mystery", questions[0], questions), ValueError))


print("\n=== budgeted generation ===")


class StubClient:
    """Records what it was asked, returns a scripted sequence of completions."""

    def __init__(self, texts):
        self.texts = list(texts) if isinstance(texts, (list, tuple)) else [texts]
        self.calls = 0
        self.seen: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.seen.append(messages)
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SimpleNamespace(text=text, cached=False, finish_reason="stop",
                               prompt_tokens=17, completion_tokens=11)


overlong = " ".join(f"w{i}" for i in range(60))
in_band = " ".join(f"s{i}" for i in range(small.floor_words))
underfilled = "too short"
client = StubClient([overlong, overlong, in_band])
message = bd.generate_within_budget(client, "PROMPT", small, temperature=0.0, seed=None,
                                    tag="t", max_corrections=2)
check("an over-long draft triggers shrink corrections", client.calls == 3, f"{client.calls}")
check("the shrink correction is the shared string",
      bd.SHRINK_CORRECTION.format(actual=60, floor=small.floor_words, words=small.words)
      in client.seen[-1][-1]["content"])
check("the delivered message obeys the cap", message.delivered_words <= small.words)
check("shrink and expand corrections are counted separately",
      message.shrink_corrections == 2 and message.expand_corrections == 0)
check("the draft history is recorded",
      message.first_draft_words == 60 and message.final_draft_words == small.floor_words)

under = StubClient([underfilled, underfilled, in_band])
grown = bd.generate_within_budget(under, "PROMPT", small, temperature=0.0, seed=None,
                                  tag="t", max_corrections=2)
check("an under-filled draft triggers expand corrections",
      under.calls == 3 and grown.expand_corrections == 2)
check("the expand correction is the shared string",
      bd.EXPAND_CORRECTION.format(actual=2, floor=small.floor_words, words=small.words)
      in under.seen[-1][-1]["content"])
check("expansion reaches the floor", not grown.under_floor_delivered)

stubborn = StubClient([overlong])
hard = bd.generate_within_budget(stubborn, "PROMPT", small, temperature=0.0, seed=None,
                                 tag="t", max_corrections=1)
check("a stubborn sender is truncated unconditionally",
      hard.delivered_words <= small.words and hard.truncated
      and hard.over_budget_before_truncation)
never = StubClient([underfilled])
short = bd.generate_within_budget(never, "PROMPT", small, temperature=0.0, seed=None,
                                  tag="t", max_corrections=1)
check("a floor miss is recorded rather than faked", short.under_floor_delivered)
# A correction that overshoots must not be delivered in place of a compliant draft.
regress = StubClient([in_band, overlong])
best = bd.generate_within_budget(regress, "PROMPT", small, temperature=0.0, seed=None,
                                 tag="t", max_corrections=1)
check("a compliant draft is kept when it exists",
      best.delivered_words == small.floor_words and not best.truncated)
wide = bd.make_budget(40, 1.35, 2.0, 96, 0.85)
repeating = StubClient([" ".join(["alpha beta gamma delta epsilon zeta"] * 6)])
padded = bd.generate_within_budget(repeating, "PROMPT", wide, temperature=0.0, seed=None,
                                   tag="t", max_corrections=0)
check("padding by repetition is measured", padded.repetition_rate > 0.3,
      f"{padded.repetition_rate}")
varied_text = StubClient([" ".join(f"w{i}" for i in range(36))])
clean = bd.generate_within_budget(varied_text, "PROMPT", wide, temperature=0.0, seed=None,
                                  tag="t", max_corrections=0)
check("non-repeating text scores zero repetition", clean.repetition_rate == 0.0)


print("\n=== design and rotation ===")
manifest, contexts, rotations = exp.load_design(cfg)
context = contexts[0]
rots = rotations[context.context_id]
check("two fixture contexts load", len(contexts) == 2)
check("one rotation per question", len(rots) == len(context.questions))
current_roles = [r.current_qid for r in rots]
check("every question conditions exactly once",
      sorted(current_roles) == sorted(q.qid for q in context.questions))
hidden_counts = {q.qid: sum(q.qid in r.hidden_qids for r in rots) for q in context.questions}
check("every question is hidden exactly k-1 times",
      set(hidden_counts.values()) == {len(context.questions) - 1}, f"{hidden_counts}")
check("no rotation hides its own conditioning question",
      all(r.current_qid not in r.hidden_qids for r in rots))

policies = ["generic", "conditioned", "reusable", "oracle"]
specs = exp.message_specs(contexts, rotations, policies, budgets, cfg)
per_context_budget = len(budgets) * len(contexts)
check("rotation-invariant policies generate one message per context and budget",
      sum(1 for s in specs if s["policy"] == "generic") == per_context_budget)
check("conditioned policies generate one message per rotation",
      sum(1 for s in specs if s["policy"] == "conditioned")
      == per_context_budget * len(context.questions))
check("message keys are unique", len({s["message_key"] for s in specs}) == len(specs))
check("the request hash tracks the budget",
      len({s["request_hash"] for s in specs
           if s["policy"] == "generic" and s["context_id"] == context.context_id})
      == len(budgets))


print("\n=== the answerer never sees the source ===")
fake_messages = []
for spec in specs:
    fake_messages.append({
        "message_key": spec["message_key"], "context_id": spec["context_id"],
        "rotation_id": spec["rotation_id"], "current_qid": spec["current_qid"],
        "policy": spec["policy"], "derived_from": "",
        "handoff_text": f"NOTES for {spec['message_key']}",
        "budget_words": spec["budget"].words, "budget_tokens": spec["budget"].tokens,
        "api_max_tokens": spec["budget"].api_max_tokens, "attempts": 1,
        "delivered_words": 3, "truncated": False, "truncated_words_dropped": 0,
        "over_budget_before_truncation": False, "fill_ratio": 3 / spec["budget"].words,
        "delivered_estimated_tokens": 4, "completion_tokens": 5, "prompt_tokens": 5,
    })
a_specs = exp.answer_specs(contexts, rotations, fake_messages, cfg)
recorder = StubClient(["Aldous Penhale"])
handoff_spec = next(s for s in a_specs if not s["baseline"])
exp.run_answer_call(recorder, handoff_spec, cfg)
sent = "\n".join(m["content"] for m in recorder.seen[0])
check("a handoff answer prompt excludes the source paragraph", SOURCE_SQUAD not in sent)
check("a handoff answer prompt carries the handoff and the question",
      handoff_spec["material"] in sent and handoff_spec["question"].question in sent)
direct_spec = next(s for s in a_specs if s["baseline"] == "direct_context")
check("only the direct-context ceiling is given the source",
      direct_spec["material"] == context.source)
check("answer keys are unique", len({s['answer_key'] for s in a_specs}) == len(a_specs))
check("an empty handoff is scored zero rather than dropped",
      exp.run_answer_call(recorder, {**handoff_spec, "technical_failure": True},
                          cfg)["judge_correct"] == 0)


print("\n=== single-flighting identical requests ===")
shared = [s for s in a_specs
          if not s["baseline"] and s["policy"] == "generic"
          and s["context_id"] == context.context_id
          and s["budget_words"] == budgets[0].words]
counting = StubClient(["Aldous Penhale"])
rows, issued = exp.generate_answers(counting, shared, cfg, [])
check("distinct questions against one shared message stay distinct requests",
      issued == len(shared) and counting.calls == len(shared))
duplicated = shared + [{**shared[0], "answer_key": shared[0]["answer_key"] + "|dup"}]
counting2 = StubClient(["Aldous Penhale"])
rows2, issued2 = exp.generate_answers(counting2, duplicated, cfg, [])
check("an identical request is issued once and cloned",
      issued2 == len(shared) and len(rows2) == len(duplicated))
check("clones keep their own evaluator metadata",
      rows2[-1]["answer_key"].endswith("|dup"))


print("\n=== utility matrix arithmetic ===")


def synthetic_answers(messages, contexts, rule):
    """Score each (message, question) cell by an injected rule, plus a ceiling."""
    out = []
    cmap = {c.context_id: c for c in contexts}
    for m in messages:
        for q in cmap[m["context_id"]].questions:
            value = float(rule(m, q))
            out.append({"schema_version": exp.SCHEMA_VERSION,
                        "answer_key": f"message|{m['message_key']}|q={q.qid}",
                        "message_key": m["message_key"], "context_id": m["context_id"],
                        "policy": m["policy"], "budget_words": m["budget_words"],
                        "rotation_id": m["rotation_id"], "current_qid": m["current_qid"],
                        "eval_qid": q.qid, "baseline": "", "sample": 0,
                        "question": q.question, "golds": list(q.golds),
                        "em": value, "f1": value, "judge_correct": value})
    for c in contexts:
        for q in c.questions:
            out.append({"schema_version": exp.SCHEMA_VERSION,
                        "answer_key": f"baseline|{c.context_id}|{q.qid}|direct_context|0",
                        "message_key": "", "context_id": c.context_id,
                        "policy": "direct_context", "budget_words": 0, "rotation_id": "-",
                        "current_qid": "", "eval_qid": q.qid, "baseline": "direct_context",
                        "sample": 0, "question": q.question, "golds": list(q.golds),
                        "em": 1.0, "f1": 1.0, "judge_correct": 1.0})
    return out


# A perfectly specialised conditioned policy: right on the diagonal, wrong off it.
def rule(message, q):
    if message["policy"] == "conditioned":
        return 1.0 if message["current_qid"] == q.qid else 0.0
    return 0.5


answers = synthetic_answers(fake_messages, contexts, rule)
matrix = exp.build_matrix_rows(contexts, rotations, fake_messages, answers, policies)
rot_rows = exp.build_rotation_rows(matrix)
cond = [r for r in rot_rows if r["policy"] == "conditioned"]
gen = [r for r in rot_rows if r["policy"] == "generic"]
check("a perfectly specialised policy scores 1 on the diagonal",
      all(r["now_judge_correct"] == 1.0 for r in cond))
check("a perfectly specialised policy scores 0 off the diagonal",
      all(r["future_judge_correct"] == 0.0 for r in cond))
check("regret is measured against the direct-context ceiling",
      all(r["regret_future_judge_correct"] == 1.0 for r in cond))
check("a rotation-invariant policy is unspecialised by construction",
      all(abs(r["now_judge_correct"] - r["future_judge_correct"]) < 1e-9 for r in gen))
check("one generic message serves every rotation",
      len({r["message_key"] for r in gen if r["budget_words"] == budgets[0].words
           and r["context_id"] == context.context_id}) == 1)
check("the matrix has k^2 cells per context, policy and budget",
      len([r for r in matrix if r["policy"] == "conditioned"
           and r["context_id"] == context.context_id
           and r["budget_words"] == budgets[0].words]) == len(context.questions) ** 2)
check("every question appears on the diagonal exactly once per policy and budget",
      sorted(r["eval_qid"] for r in matrix
             if r["is_diagonal"] and r["policy"] == "conditioned"
             and r["context_id"] == context.context_id
             and r["budget_words"] == budgets[0].words)
      == sorted(q.qid for q in context.questions))
check("retained utility is restricted to answerable cells",
      all(r["n_ceiling_future_judge_correct"] == len(context.questions) - 1 for r in cond))

metrics = exp.analyse_metrics(rot_rows, policies, [b.words for b in budgets], cfg)
contrasts = exp.analyse_contrasts(rot_rows, policies, [b.words for b in budgets], cfg)
did = exp.lookup(contrasts, comparison="conditioned_minus_generic",
                 budget_words=budgets[0].words, kind="specialisation_did",
                 metric="judge_correct")
check("the specialisation DiD recovers the injected effect",
      did is not None and abs(did["delta"] - 1.0) < 1e-9, f"{did and did['delta']}")
now = exp.lookup(contrasts, comparison="conditioned_minus_generic",
                 budget_words=budgets[0].words, kind="utility", endpoint="now",
                 metric="judge_correct")
future = exp.lookup(contrasts, comparison="conditioned_minus_generic",
                    budget_words=budgets[0].words, kind="utility", endpoint="future",
                    metric="judge_correct")
check("the injected present gain is recovered", abs(now["delta"] - 0.5) < 1e-9)
check("the injected future loss is recovered", abs(future["delta"] + 0.5) < 1e-9)


print("\n=== trimming to a common length ===")
varied = []
for i, m in enumerate(fake_messages):
    words = 5 + (i % 4) * 3
    varied.append({**m, "handoff_text": " ".join(["tok"] * words), "delivered_words": words})
# One very short extractive message, to confirm a different mechanism cannot set
# the floor for the abstractive comparison.
varied.append({**varied[0], "message_key": varied[0]["message_key"] + "|x",
               "policy": "extractive_generic", "handoff_text": "tok",
               "delivered_words": 1})
trimmed = exp.trimmed_messages(varied, [budgets[0].words])
by_cell = {}
for row in trimmed:
    by_cell.setdefault((row["context_id"], row["budget_words"]), set()).add(
        row["delivered_words"])
check("every trimmed arm in a cell has the same length",
      bool(by_cell) and all(len(v) == 1 for v in by_cell.values()), f"{by_cell}")
check("trimming never lengthens a message",
      all(r["delivered_words"] <= next(v["delivered_words"] for v in varied
                                       if v["message_key"] == r["derived_from"])
          for r in trimmed))
check("trimmed arms are labelled and traceable",
      all(r["policy"].endswith(exp.TRIM_SUFFIX) and r["derived_from"] for r in trimmed))
check("only abstractive arms are trimmed",
      all(r["policy"].replace(exp.TRIM_SUFFIX, "") in bd.ABSTRACTIVE_POLICIES
          for r in trimmed))


print("\n=== Pareto domination ===")
pts = exp.pareto_points(
    [{"policy": "a", "budget_words": 10, "kind": "utility", "endpoint": "now",
      "metric": "judge_correct", "mean": 0.5},
     {"policy": "a", "budget_words": 10, "kind": "utility", "endpoint": "future",
      "metric": "judge_correct", "mean": 0.5},
     {"policy": "a", "budget_words": 10, "kind": "channel", "endpoint": "message",
      "metric": "delivered_words", "mean": 10.0},
     {"policy": "b", "budget_words": 10, "kind": "utility", "endpoint": "now",
      "metric": "judge_correct", "mean": 0.6},
     {"policy": "b", "budget_words": 10, "kind": "utility", "endpoint": "future",
      "metric": "judge_correct", "mean": 0.6},
     {"policy": "b", "budget_words": 10, "kind": "channel", "endpoint": "message",
      "metric": "delivered_words", "mean": 10.0}],
    ["a", "b"], [10], "judge_correct", 0.0)
check("a strictly worse configuration is dominated",
      next(p for p in pts if p["policy"] == "a")["dominated"])
check("a strictly better configuration is not dominated",
      not next(p for p in pts if p["policy"] == "b")["dominated"])
check("trimmed arms are kept off the frontier",
      not exp.pareto_points([], [f"x{exp.TRIM_SUFFIX}"], [10], "judge_correct", 0.0))

# Recovered/concatenated metric tables can contain duplicate aggregates. The
# result must be independent of row order, must not duplicate configurations,
# and must never admit NaN/Inf into a dominance comparison.
duplicate_rows = [
    {"policy": "dup", "budget_words": 10, "kind": "utility", "endpoint": "now",
     "metric": "judge_correct", "mean": 0.4},
    {"policy": "dup", "budget_words": 10, "kind": "utility", "endpoint": "now",
     "metric": "judge_correct", "mean": 0.6},
    {"policy": "dup", "budget_words": 10, "kind": "utility", "endpoint": "now",
     "metric": "judge_correct", "mean": float("nan")},
    {"policy": "dup", "budget_words": 10, "kind": "utility", "endpoint": "future",
     "metric": "judge_correct", "mean": 0.5},
    {"policy": "dup", "budget_words": 10, "kind": "channel", "endpoint": "message",
     "metric": "delivered_words", "mean": 10.0},
    {"policy": "invalid", "budget_words": 10, "kind": "utility", "endpoint": "now",
     "metric": "judge_correct", "mean": 0.8},
    {"policy": "invalid", "budget_words": 10, "kind": "utility", "endpoint": "future",
     "metric": "judge_correct", "mean": float("inf")},
    {"policy": "invalid", "budget_words": 10, "kind": "channel", "endpoint": "message",
     "metric": "delivered_words", "mean": 10.0},
]
duplicate_pts = exp.pareto_points(
    list(reversed(duplicate_rows)), ["dup", "dup", "invalid"], [10, 10],
    "judge_correct", 0.0)
check("duplicate metric rows are averaged deterministically",
      len(duplicate_pts) == 1 and abs(duplicate_pts[0]["u_now"] - 0.5) < 1e-12)
check("non-finite or incomplete configurations are omitted",
      all(np.isfinite([p["u_now"], p["u_future"], p["cost_words"]]).all()
          for p in duplicate_pts)
      and not any(p["policy"] == "invalid" for p in duplicate_pts))

tie_rows = []
for policy, now, future in (("tie_a", .7, .6), ("tie_b", .7, .6), ("worse", .6, .5)):
    tie_rows.extend([
        {"policy": policy, "budget_words": 10, "kind": "utility", "endpoint": "now",
         "metric": "judge_correct", "mean": now},
        {"policy": policy, "budget_words": 10, "kind": "utility", "endpoint": "future",
         "metric": "judge_correct", "mean": future},
        {"policy": policy, "budget_words": 10, "kind": "channel", "endpoint": "message",
         "metric": "delivered_words", "mean": 10.0},
    ])
tie_pts = exp.pareto_points(tie_rows, ["tie_a", "tie_b", "worse"], [10],
                            "judge_correct", 0.0)
check("exactly tied configurations do not dominate each other",
      all(not next(p for p in tie_pts if p["policy"] == policy)["dominated"]
          for policy in ("tie_a", "tie_b")))
check("both tied optima may dominate a strictly worse point",
      next(p for p in tie_pts if p["policy"] == "worse")["dominated_by"]
      == "tie_a@10w;tie_b@10w")
check("invalid Pareto tolerances are rejected",
      raises(lambda: exp.pareto_points([], [], [], "judge_correct", float("nan")),
             ValueError)
      and raises(lambda: exp.pareto_points([], [], [], "judge_correct", -0.1),
                 ValueError))

with tempfile.TemporaryDirectory() as td:
    pareto_png = Path(td) / "pareto.png"
    exp.plot_pareto(tie_pts, pareto_png, "accuracy")
    check("faceted Pareto plot handles ties without interpolation",
          pareto_png.exists() and pareto_png.stat().st_size > 0)

check("cohens_dz is the mean over the sd of the paired difference",
      abs(exp.cohens_dz(np.array([1.0, 1.0, 1.0, 3.0]))
          - (1.5 / np.std([1.0, 1.0, 1.0, 3.0], ddof=1))) < 1e-9)


print("\n=== length audit ===")
table, deltas = exp.analyse_length(rot_rows, fake_messages, policies,
                                   [b.words for b in budgets], cfg)
check("the audit reports one row per policy and budget",
      len(table) == len(policies) * len(budgets))
check("no delivered message exceeds its cap",
      not any(r["any_over_budget_delivered"] for r in table))
check("requested and actual token counts are both recorded",
      all({"budget_tokens", "api_max_tokens", "completion_tokens_mean",
           "delivered_estimated_tokens_mean"} <= set(r) for r in table))
check("both sides of the contract are audited",
      all({"floor_words", "under_floor_rate", "repetition_rate_mean",
           "shrink_corrections_mean", "expand_corrections_mean"} <= set(r) for r in table))
check("paired length deltas are computed for the headline comparisons",
      any(r["comparison"] == "conditioned_minus_generic" for r in deltas))
matched_ids = exp.length_matched_contexts(rot_rows, ["generic", "conditioned"],
                                          budgets[0].words, 0)
check("identical lengths match at zero tolerance", len(matched_ids) == len(contexts))


print("\n=== relation labelling ===")
cfg["dataset"]["active"] = "relation_dossiers"
_m, rel_contexts, rel_rotations = exp.load_design(cfg)
rel = rel_contexts[0]
rel_rots = rel_rotations[rel.context_id]
check("only anchors condition a relation rotation",
      len(rel_rots) == 4 and all(rel.question(r.current_qid).role == rd.ANCHOR
                                 for r in rel_rots))
anchor = rel_rots[0]
labels = {rd.relation_label(rel, anchor.current_qid, qid) for qid in anchor.hidden_qids}
check("all four designed distances are present", labels == set(rd.RELATION_ORDER))
counts = {label: sum(1 for qid in anchor.hidden_qids
                     if rd.relation_label(rel, anchor.current_qid, qid) == label)
          for label in rd.RELATION_ORDER}
check("each within-aspect distance appears once",
      counts[rd.PARAPHRASE] == counts[rd.SAME_ENTITY] == counts[rd.SAME_TOPIC] == 1,
      f"{counts}")
check("orthogonal covers every other aspect's questions", counts[rd.ORTHOGONAL] == 12)
check("a paraphrase shares the anchor's gold answer",
      rel.question(f"{rel.context_id}:founding:paraphrase").golds
      == rel.question(f"{rel.context_id}:founding:anchor").golds)
bad = dict(relation_fixture(9))
bad["questions"] = [q for q in bad["questions"] if q["role"] != rd.SAME_TOPIC]
try:
    rd.validate_relation_context(rd._context(bad))
    ok = False
except ValueError:
    ok = True
check("a corpus missing a designed role is rejected", ok)
check("a matrix with no messages is empty rather than an error",
      exp.build_matrix_rows(rel_contexts, rel_rotations, [], [], policies) == [])
check("relation_label refuses a non-anchor conditioning question",
      raises(lambda: rd.relation_label(
          rel, f"{rel.context_id}:founding:paraphrase",
          f"{rel.context_id}:finance:anchor"), ValueError))
check("SQuAD groups carry no relation label",
      rd.relation_label(context, "q1", "q2") == rd.UNLABELLED)


print("\n=== dry run writes nothing ===")
cfg["dataset"]["active"] = "squad_groups"
plan = exp.dry_run_report(contexts, rotations, policies, budgets, cfg)
check("dry-run declares zero writes", plan["writes"] == 0)
check("dry-run counts rotation-invariant reuse",
      plan["generated_messages"]
      == len(budgets) * len(contexts) * (2 + 2 * len(context.questions)))
check("dry-run reports both budget units",
      plan["budgets_words"] == [b.words for b in budgets]
      and plan["budgets_tokens"] == [b.tokens for b in budgets])
before = {p for p in (ROOT / "runs").glob("communication_regret*")}
check("no run directory was created by the offline checks",
      before == {p for p in (ROOT / "runs").glob("communication_regret*")})


print("\n=== SQuAD builder against the real parquet schema ===")
parquet = ROOT / cfg["build"]["squad"]["local_parquet"]
if parquet.exists():
    import pandas as pd
    df = pd.read_parquet(parquet).head(4000)
    groups = build.structural_groups(
        df, {**cfg["build"]["squad"], "max_contexts_scanned": 25}, 7)
    k = int(cfg["build"]["squad"]["questions_per_context"])
    check("the builder reads the live SQuAD schema", len(groups) > 0, f"{len(groups)} groups")
    check("each group keeps at least k questions",
          all(len(g["questions"]) >= k for g in groups))
    check("no group repeats a normalised answer",
          all(len({build.normalize_answer(q["golds"][0]) for q in g["questions"]})
              == len(g["questions"]) for g in groups))
    check("context length stays inside the configured band",
          all(cfg["build"]["squad"]["context_chars_min"] <= len(g["context"])
              <= cfg["build"]["squad"]["context_chars_max"] for g in groups))
    check("group audit prompt names every question",
          all(q["question"] in build.AUDIT_TEMPLATE.format(
              context=groups[0]["context"],
              questions="\n".join(f"{i + 1}. {x['question']}  [reference answer: {x['golds'][0]}]"
                                  for i, x in enumerate(groups[0]["questions"])))
              for q in groups[0]["questions"]))
else:
    print(f"SKIP  SQuAD parquet not present at {parquet}")

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed: {', '.join(failures)}")
print("\nALL CHECKS PASSED")
