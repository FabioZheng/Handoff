"""Offline regression checks for Experiment 14 (compression mechanism).

Runs against the real relation-dossier corpus and the real LM score artefact.
No API key, no network, no model.

Four things are worth failing the build over, and they are the four this file
spends most of its lines on:

1. **No hidden question reaches a selector.** The whole experiment is a claim
   about what a compressor knew when it chose. A conditioned selector that could
   see the future questions would not be measuring anticipation, it would be
   measuring an oracle.
2. **The delivered text is genuinely verbatim source.** Experiment 10's
   ``extractive_*`` arms were an LLM *asked* to copy sentences and complied 0-29%
   of the time. If elimination silently degrades into rewriting, the mechanism
   contrast this experiment exists to draw is gone and the numbers still look
   fine.
3. **The hard budget binds.** Selection cannot pad, so a cap violation would be
   a packing bug, and a length difference between arms is the one confound that
   would explain any utility result on its own.
4. **Experiment 10's rows are still exactly reusable.** 640 messages and 11,264
   answers are inherited rather than regenerated. That is only legitimate while
   this config reproduces their request hashes byte for byte - which is also the
   proof that adding these arms did not perturb the published experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import anticipatory_data as ad  # noqa: E402
import budget as bd  # noqa: E402
import elimination as el  # noqa: E402
import lm_unit_scores as lms  # noqa: E402
import regret_data as rd  # noqa: E402
import run_communication_regret as rcr  # noqa: E402
from llm import load_config  # noqa: E402

CORPUS = ROOT / "data" / "communication_regret" / "relation_dossiers.jsonl"
CONFIG = ROOT / "configs/compression_mechanism_config.yaml"
EXP10_MESSAGES = ROOT / "runs" / "communication_regret" / "relation_dossiers" / "n16" / "messages.jsonl"

CHECKS = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    if not condition:
        raise AssertionError(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))
    CHECKS += 1
    print(f"PASS  {label}" + (f" -- {detail}" if detail else ""))


def raises(label: str, fn, *args, **kwargs) -> None:
    global CHECKS
    try:
        fn(*args, **kwargs)
    except (ValueError, TypeError, AssertionError, FileNotFoundError):
        CHECKS += 1
        print(f"PASS  {label}")
        return
    raise AssertionError(f"FAIL  {label} -- no error raised")


# ---------------------------------------------------------------------------
# Fixtures

cfg = load_config(CONFIG)
manifest, contexts = rd.load_contexts(CORPUS)
_, dossiers = ad.load_dossiers(CORPUS)
by_id = {c.context_id: c for c in contexts}
units_by_context = {d["context_id"]: ad.extract_units(d) for d in dossiers}
rotations = {c.context_id: rd.rotations_for(c) for c in contexts}
budgets = bd.budgets_from_config(cfg)
policies = list(cfg["policies"])

print("# registry and design")

check("the config's arms are all known to the runner",
      all(p in rcr.KNOWN_POLICIES for p in policies), f"{len(policies)} arms")
check("the five selection arms are present",
      set(el.SELECTION_POLICIES) <= set(policies), str(el.SELECTION_POLICIES))
check("selection arms are disjoint from the generated arms",
      not (set(el.SELECTION_POLICIES) & set(bd.ALL_POLICIES)))
check("paraphrase is an abstractive, rotation-invariant arm",
      "paraphrase" in bd.ABSTRACTIVE_POLICIES and "paraphrase" in bd.ROTATION_INVARIANT)
check("only the two conditioned selectors are query-aware",
      set(el.QUERY_AWARE) == {"lm_conditioned", "nonllm_conditioned"})
check("query-agnostic selectors are rotation-invariant, query-aware ones are not",
      all(rcr.rotation_invariant(p) for p in el.QUERY_AGNOSTIC)
      and not any(rcr.rotation_invariant(p) for p in el.QUERY_AWARE))

resources = rcr.selection_resources(cfg, contexts, policies)
specs = rcr.message_specs(contexts, rotations, policies, budgets, cfg, resources)
selection_specs = [s for s in specs if s.get("kind") == "selection"]
check("the message plan has one row per (context, budget) for a query-agnostic arm",
      len([s for s in specs if s["policy"] == "nonllm_generic"]) == 16 * 4)
check("the message plan has one row per rotation for a query-aware arm",
      len([s for s in specs if s["policy"] == "nonllm_conditioned"]) == 16 * 4 * 4)
check("every selection message is free of generation",
      len(selection_specs) == 768, f"{len(selection_specs)} selection messages")

print("\n# leakage: the future questions never reach a selector")

# The structural guarantee. A selector's inputs are (units, one query, scores).
for spec in selection_specs:
    context = spec["context"]
    if spec["query"] is None:
        continue
    current = context.question(spec["current_qid"])
    hidden = [q for q in context.questions if q.qid != spec["current_qid"]]
    assert spec["query"] == current.question, spec["message_key"]
    for q in hidden:
        assert q.question not in spec["query"], spec["message_key"]
check("a conditioned selector is handed exactly its own conditioning question",
      True, f"{len([s for s in selection_specs if s['query']])} query-aware specs")

for spec in selection_specs:
    if spec["policy"] != "lm_conditioned":
        continue
    table = resources["lm_scores"]["conditioned"]
    own = {unit_id for (qid, unit_id) in table if qid == spec["current_qid"]}
    assert set(spec["lm_scores"]) == own, spec["message_key"]
check("an lm_conditioned message sees only its own question's score table", True,
      "other rotations' scores are not in scope")

check("query-agnostic selection specs carry no query",
      all(s["query"] is None for s in selection_specs
          if s["policy"] in el.QUERY_AGNOSTIC))
raises("a query-agnostic selector refuses a query",
       el.score_units, "nonllm_generic", units_by_context["rel:3"],
       query="anything")
raises("a query-aware selector refuses to run without one",
       el.score_units, "nonllm_conditioned", units_by_context["rel:3"])

lm_rows = [json.loads(line) for line in
           (ROOT / cfg["elimination"]["lm_scores_path"]).read_text(encoding="utf-8").splitlines()]
scored_qids = {r["qid"] for r in lm_rows if r.get("qid")}
anchor_qids = {q.qid for c in contexts for q in c.questions if q.role == rd.ANCHOR}
check("the LM score artefact scored anchors only",
      scored_qids == anchor_qids, f"{len(scored_qids)} scored / {len(anchor_qids)} anchors")
non_anchor = [{"context_id": "rel:3", "qid": "rel:3:founding:paraphrase", "kind": "conditioned"}]
raises("the scorer's own leakage gate fires on a non-anchor question",
       lms.check_leakage, non_anchor, contexts)

print("\n# the hard budget binds, and unused capacity is recorded")

built = {}
for context in contexts:
    units = units_by_context[context.context_id]
    for b in budgets:
        for policy in el.SELECTION_POLICIES:
            queries = ([context.question(r.current_qid).question
                        for r in rotations[context.context_id]]
                       if el.is_query_aware(policy) else [None])
            for query in queries:
                scores = (resources["lm_scores"]["generic"] if policy == "lm_generic" else
                          {unit_id: s for (qid, unit_id), s in
                           resources["lm_scores"]["conditioned"].items()
                           if policy == "lm_conditioned"
                           and qid == next(r.current_qid for r in rotations[context.context_id]
                                           if context.question(r.current_qid).question == query)}
                          if policy == "lm_conditioned" else None)
                message = el.build_message(
                    policy, units, b.words, source=context.source,
                    questions=context.questions, query=query, lm_scores=scores,
                    seed_material=f"exp14-mechanism|{context.context_id}|{b.words}",
                    generic_scorer=cfg["elimination"]["nonllm_generic_scorer"])
                built[(context.context_id, policy, b.words, query or "-")] = message

over = [k for k, m in built.items() if m.delivered_words > m.budget_words]
check("no selection message ever exceeds its cap", not over,
      f"{len(built)} messages built, 0 over cap")
check("a selection message is never truncated mid-sentence",
      all(m.text == el.render(units_by_context[k[0]],
                              tuple(i for i, u in enumerate(units_by_context[k[0]])
                                    if u.unit_id in m.selected_unit_ids))
          for k, m in built.items()))
empties = {k: m for k, m in built.items() if m.empty_message}
check("an empty message happens only when no single sentence fits the cap",
      all(m.shortest_unit_words > m.budget_words for m in empties.values()),
      f"{len(empties)} empty messages, all at caps below the shortest sentence")
check("empty messages occur at the 20-word rung only",
      {k[2] for k in empties} == {20},
      "sentence granularity, not a packing failure")
fills = {b.words: sum(m.fill_ratio for k, m in built.items() if k[2] == b.words)
         / max(1, len([k for k in built if k[2] == b.words])) for b in budgets}
check("unused capacity is real and recorded rather than padded",
      all(0.0 <= v <= 1.0 for v in fills.values()),
      " ".join(f"w{w}={v:.2f}" for w, v in sorted(fills.items())))

# Adversarial packing: a cap that fits nothing, a cap that fits everything,
# and a unit ordering that would break a naive "stop at the first misfit".
tiny = units_by_context["rel:3"]
check("packing takes a later unit when an earlier one does not fit",
      len(el.pack(tiny, [10.0] + [1.0] * (len(tiny) - 1), 40)) >= 1)
raises("a zero-word cap is rejected rather than silently emptied",
       el.pack, tiny, [1.0] * len(tiny), 0)
raises("a score vector of the wrong length is rejected",
       el.pack, tiny, [1.0], 40)
check("packing is a pure function of (units, scores, cap)",
      el.pack(tiny, [1.0] * len(tiny), 60) == el.pack(tiny, [1.0] * len(tiny), 60))

print("\n# the delivered text is verbatim source, and provably so")

for (context_id, policy, words, _q), message in built.items():
    units = units_by_context[context_id]
    text = message.text
    for unit_id in message.selected_unit_ids:
        unit = next(u for u in units if u.unit_id == unit_id)
        assert unit.text.strip() in text, (context_id, policy, words, unit_id)
    positions = [text.index(next(u for u in units if u.unit_id == uid).text.strip())
                 for uid in message.selected_unit_ids]
    assert positions == sorted(positions), (context_id, policy, words)
check("every delivered unit appears verbatim, in source order", True,
      f"{sum(m.selected_unit_count for m in built.values())} delivered units")
check("the delivered word count is the sum of its units' word counts",
      all(m.delivered_words == sum(
          el.word_count(next(u for u in units_by_context[k[0]] if u.unit_id == uid).text)
          for uid in m.selected_unit_ids) for k, m in built.items()))
raises("verification rejects a message that was rewritten after selection",
       el.verify_verbatim, "The company was founded, allegedly, in 1827.",
       tiny, (0,))
raises("verification rejects a message with a unit deleted from the text",
       el.verify_verbatim, "", tiny, (0,))

print("\n# diagnostics separate deleted evidence from a failed reader")

sample = built[("rel:3", "nonllm_conditioned", 160,
                by_id["rel:3"].question(rotations["rel:3"][0].current_qid).question)]
check("answer-bearing survival implies the gold string survived",
      set(sample.answer_bearing_survived_qids) <= set(sample.gold_span_survived_qids),
      f"{len(sample.answer_bearing_survived_qids)} bearing / "
      f"{len(sample.gold_span_survived_qids)} spans")
check("aspect retention is a fraction per aspect",
      all(0.0 <= v <= 1.0 for m in built.values() for v in m.aspect_retention.values()))
check("retention fraction compares delivered words to source words",
      all(0.0 <= m.retention_fraction <= 1.0 for m in built.values()),
      f"max={max(m.retention_fraction for m in built.values()):.3f}")
conditioned_keep = sum(
    1 for c in contexts for r in rotations[c.context_id]
    if r.current_qid in built[(c.context_id, "nonllm_conditioned", 160,
                               c.question(r.current_qid).question)].answer_bearing_survived_qids)
generic_keep = sum(
    1 for c in contexts for r in rotations[c.context_id]
    if r.current_qid in built[(c.context_id, "nonllm_generic", 160, "-")]
    .answer_bearing_survived_qids)
check("a query-aware selector keeps the current question's evidence more often",
      conditioned_keep > generic_keep,
      f"nonllm_conditioned {conditioned_keep}/64 vs nonllm_generic {generic_keep}/64")

print("\n# determinism and artefact integrity")

again = el.build_message(
    "nonllm_generic", units_by_context["rel:3"], 80, source=by_id["rel:3"].source,
    questions=by_id["rel:3"].questions,
    seed_material="exp14-mechanism|rel:3|80",
    generic_scorer=cfg["elimination"]["nonllm_generic_scorer"])
check("rebuilding a message reproduces it exactly",
      again.text == built[("rel:3", "nonllm_generic", 80, "-")].text)
shuffled = el.build_message(
    "random_selection", units_by_context["rel:3"], 80, source=by_id["rel:3"].source,
    questions=by_id["rel:3"].questions, seed_material="exp14-mechanism|rel:3|80")
check("the random control is seeded by content, not by clock",
      shuffled.text == built[("rel:3", "random_selection", 80, "-")].text)
check("the LM score artefact is pinned to a resolved checkpoint",
      bool(resources["lm_manifest"].get("commit_hash")),
      f"{resources['lm_manifest'].get('model_id')}@"
      f"{str(resources['lm_manifest'].get('commit_hash'))[:12]}")
raises("LM scores computed against a different corpus are refused",
       lms.load_scores, ROOT / cfg["elimination"]["lm_scores_path"], "deadbeefdeadbeef")
raises("a missing LM score table is a clear error, not a silent fallback",
       lms.load_scores, ROOT / "data" / "compression_mechanism" / "does_not_exist.jsonl")
raises("an lm_* arm refuses to run without its scores",
       el.score_units, "lm_generic", units_by_context["rel:3"])
raises("an lm_* arm refuses a score table missing a unit",
       el.score_units, "lm_generic", units_by_context["rel:3"],
       lm_scores={units_by_context["rel:3"][0].unit_id: 1.0})

print("\n# Experiment 10's rows are still exactly reusable")

stored = [json.loads(line) for line in
          EXP10_MESSAGES.read_text(encoding="utf-8").splitlines() if line.strip()]
by_key = {row["message_key"]: row for row in stored}
reused = [s for s in specs if s["message_key"] in by_key]
mismatched = [s["message_key"] for s in reused
              if by_key[s["message_key"]]["request_hash"] != s["request_hash"]]
check("every stored Experiment 10 message is addressed by this run",
      len(reused) == 640, f"{len(reused)} of {len(stored)} stored messages")
check("and every one of their request hashes still matches",
      not mismatched, "adding the new arms perturbed no published prompt")
check("the only generated arm this run adds is paraphrase",
      {s["policy"] for s in specs if s.get("kind") == "generated"} - set(by_key and
      {row["policy"] for row in stored}) == {"paraphrase"},
      f"{len([s for s in specs if s['policy'] == 'paraphrase'])} new sender calls")

print(f"\nALL COMPRESSION-MECHANISM CHECKS PASSED ({CHECKS} checks)")
