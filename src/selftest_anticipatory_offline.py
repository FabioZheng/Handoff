"""Offline regression checks for Experiment 12 evidence units and demand models.

Runs against the real relation-dossier corpus and needs no API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import anticipatory_data as ad  # noqa: E402

CORPUS = ROOT / "data" / "communication_regret" / "relation_dossiers.jsonl"


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))
    print(f"PASS  {label}" + (f" -- {detail}" if detail else ""))


def raises(label: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except (ValueError, TypeError):
        print(f"PASS  {label}")
        return
    raise AssertionError(f"FAIL  {label} -- no error raised")


# ---------------------------------------------------------------------------
# The corpus

manifest, dossiers = ad.load_dossiers(CORPUS)
check("manifest line is separated from the dossiers", "source" not in manifest and bool(manifest),
      f"manifest keys={list(manifest)}")
check("all 16 relation dossiers load", len(dossiers) == 16, f"n={len(dossiers)}")

all_units = {}
for dossier in dossiers:
    units = ad.extract_units(dossier)
    ad.check_coverage(dossier, units)
    all_units[dossier["context_id"]] = units

total = sum(len(u) for u in all_units.values())
claimed = sum(1 for u in all_units.values() for unit in u if unit.aspect != ad.UNCLAIMED)
check("every dossier passes the coverage gate", len(all_units) == 16)
check("unit count matches the measured corpus", total == 264, f"units={total}")
check("claimed-unit count matches the measured corpus", claimed == 191, f"claimed={claimed}")
check("an unclaimed background pool exists", total - claimed == 73,
      f"unclaimed={total - claimed}")

cells = [(cid, aspect, sum(1 for u in units if u.aspect == aspect))
         for cid, units in all_units.items() for aspect in ad.ASPECTS]
check("every (dossier, aspect) cell owns at least two units",
      all(n >= 2 for _, _, n in cells), f"min={min(n for _, _, n in cells)}")
every_qid = {q["qid"] for d in dossiers for q in d["questions"]}
answered = {qid for units in all_units.values() for u in units for qid in u.answers_qids}
check("every question has an answer-bearing unit", every_qid <= answered,
      f"questions={len(every_qid)}")

# ---------------------------------------------------------------------------
# Leaky and incomplete corpora are rejected, not repaired

leaky = {
    "context_id": "synthetic:leak",
    "source": "The company was founded in 1820 and its mill burned in 1820.",
    "questions": [
        {"qid": "q1", "question": "founded?", "golds": ["1820"], "aspect": "founding",
         "role": "anchor"},
        {"qid": "q2", "question": "burned?", "golds": ["1820"], "aspect": "facilities",
         "role": "anchor"},
    ],
}
raises("a sentence claimed by two aspects is rejected", ad.extract_units, leaky)

thin = {
    "context_id": "synthetic:thin",
    "source": "The company was founded in 1820. It had a mill.",
    "questions": [
        {"qid": "q1", "question": "founded?", "golds": ["1820"], "aspect": "founding",
         "role": "anchor"},
        {"qid": "q2", "question": "revenue?", "golds": ["never stated"], "aspect": "finance",
         "role": "anchor"},
    ],
}
raises("a question with no answer-bearing unit is rejected",
       ad.check_coverage, thin, ad.extract_units(thin))

# ---------------------------------------------------------------------------
# Demand distributions

P = {"founding": 0.5, "facilities": 0.3, "finance": 0.2, "custom": 0.0}

check("dilute(tau=0) is the truth", ad.dilute(P, 0.0) == P)
check("dilute(tau=1) is uniform", ad.dilute(P, 1.0) == ad.uniform())
check("displace(sigma=0) is the truth", ad.displace(P, 0.0) == P)
check("displace(sigma=1) concentrates on the least likely aspect",
      ad.displace(P, 1.0) == ad.point_mass("custom"))

tv_dilute = [ad.total_variation(P, ad.dilute(P, t)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
tv_displace = [ad.total_variation(P, ad.displace(P, s)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
check("total variation increases monotonically with tau",
      all(a < b for a, b in zip(tv_dilute, tv_dilute[1:])),
      " ".join(f"{v:.3f}" for v in tv_dilute))
check("total variation increases monotonically with sigma",
      all(a < b for a, b in zip(tv_displace, tv_displace[1:])),
      " ".join(f"{v:.3f}" for v in tv_displace))

# Both noise models are linear interpolations, so TV is exactly linear in the
# knob. This matters: it means any curvature or threshold later observed in
# regret-vs-mismatch is a property of the policy, not an artefact of how the
# noise was parameterised.
full_dilute = ad.total_variation(P, ad.uniform())
check("total variation is exactly linear in tau",
      all(abs(tv - t * full_dilute) < 1e-12
          for t, tv in zip((0.0, 0.25, 0.5, 0.75, 1.0), tv_dilute)),
      f"TV(P,U)={full_dilute:.3f}")
full_displace = ad.total_variation(P, ad.point_mass("custom"))
check("total variation is exactly linear in sigma",
      all(abs(tv - s * full_displace) < 1e-12
          for s, tv in zip((0.0, 0.25, 0.5, 0.75, 1.0), tv_displace)),
      f"TV(P,wrong)={full_displace:.3f}")

# Displacement is the more damaging error at equal knob setting, which is the
# whole reason the two knobs are kept separate.
check("displacement moves more mass than dilution at equal knob",
      full_displace > full_dilute, f"{full_displace:.3f} > {full_dilute:.3f}")

check("total variation of a distribution with itself is zero",
      ad.total_variation(P, P) == 0.0)
check("total variation is symmetric and bounded",
      abs(ad.total_variation(P, ad.uniform()) - ad.total_variation(ad.uniform(), P)) < 1e-12
      and 0.0 <= full_displace <= 1.0)
check("Jensen-Shannon is zero for identical distributions",
      abs(ad.jensen_shannon(P, P)) < 1e-12)
check("Jensen-Shannon is symmetric and bounded in [0, 1]",
      abs(ad.jensen_shannon(P, ad.uniform()) - ad.jensen_shannon(ad.uniform(), P)) < 1e-12
      and 0.0 <= ad.jensen_shannon(P, ad.point_mass("custom")) <= 1.0)

raises("a distribution that does not sum to one is rejected",
       ad.dilute, {"founding": 0.5, "facilities": 0.2, "finance": 0.2, "custom": 0.0}, 0.5)
raises("a negative weight is rejected",
       ad.dilute, {"founding": 1.2, "facilities": -0.2, "finance": 0.0, "custom": 0.0}, 0.5)
raises("an unknown aspect is rejected",
       ad.dilute, {"founding": 0.5, "facilities": 0.5, "finance": 0.0, "custom": 0.0,
                   "weather": 0.0}, 0.5)
raises("tau outside [0, 1] is rejected", ad.dilute, P, 1.5)
raises("sigma outside [0, 1] is rejected", ad.displace, P, -0.1)
raises("an unknown aspect cannot take a point mass", ad.point_mass, "weather")

# ---------------------------------------------------------------------------
# Expected future utility

utilities = {"founding": 1.0, "facilities": 0.5, "finance": 0.0, "custom": 0.25}
check("expected utility weights aspect means, not raw cells",
      abs(ad.expected_future_utility(utilities, P) - (0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.0)) < 1e-12)
check("a zero-weight aspect cannot influence the expectation",
      abs(ad.expected_future_utility(utilities, ad.point_mass("founding")) - 1.0) < 1e-12)
raises("a missing utility for a positive-weight aspect is rejected",
       ad.expected_future_utility, {"founding": 1.0}, P)

# ---------------------------------------------------------------------------
# Action space, store, and the isolation guarantee

import anticipatory_policies as ap  # noqa: E402
import anticipatory_store as ast_  # noqa: E402

dossier = dossiers[0]
units = all_units[dossier["context_id"]]
anchor = next(q for q in dossier["questions"] if q["role"] == "anchor")
target_aspect = ad.ASPECTS[(ad.ASPECTS.index(anchor["aspect"]) + 1) % 4]
future_q = next(q for q in dossier["questions"]
                if q["aspect"] == target_aspect and q["role"] == "anchor")
TRUE = {a: (0.7 if a == target_aspect else 0.1) for a in ad.ASPECTS}
# Offline stand-in for the model's compression: no API call needed to test the
# allocator, only a shorter string than the original.
COMPRESSED = {u.unit_id: " ".join(u.text.split()[: max(4, u.words // 2)]) for u in units}
SETTINGS = {"now_reserve_fraction": 0.5, "retrieval_k": 2, "entropy_gate": 0.75}

FAMILIES = ["static_conditioned", "blind", "oracle_exact", "preserve__tau0",
            "preserve__sigma1", "pointer__tau0", "retrieval_only",
            "uncertainty_aware__tau0", "uncertainty_aware__tau1"]

for budget_words in (40, 80):
    for policy in FAMILIES:
        p_hat = None
        if "__" in policy:
            knob = policy.split("__")[1]
            p_hat = (ad.dilute(TRUE, {"tau0": 0.0, "tau1": 1.0}.get(knob, 0.0))
                     if knob.startswith("tau") else ad.displace(TRUE, 1.0))
        alloc = ap.build_allocation(policy, units, anchor["qid"], budget_words,
                                    COMPRESSED, p_hat=p_hat, future_qid=future_q["qid"],
                                    cfg=SETTINGS)
        built = ap.assemble(alloc, units, COMPRESSED, budget_words)
        assert set(alloc.actions) == {u.unit_id for u in units}, policy
        assert built["delivered_words"] <= budget_words, (policy, built["delivered_words"])
check("every policy assigns exactly one action to every unit", True)
check("no policy ever exceeds its word budget", True)

alloc_static = ap.build_allocation("static_conditioned", units, anchor["qid"], 80,
                                   COMPRESSED, cfg=SETTINGS)
check("a static policy stores nothing and cannot retrieve",
      alloc_static.retrieval_k == 0
      and not ap.assemble(alloc_static, units, COMPRESSED, 80)["stored_units"])

alloc_ret = ap.build_allocation("retrieval_only", units, anchor["qid"], 80,
                                COMPRESSED, cfg=SETTINGS)
built_ret = ap.assemble(alloc_ret, units, COMPRESSED, 80)
check("a retrieval policy stores what it did not deliver",
      alloc_ret.retrieval_k == 2 and len(built_ret["stored_units"]) > 0,
      f"stored={len(built_ret['stored_units'])}")

store = ast_.EvidenceStore(built_ret["stored_units"])
hits = store.retrieve(future_q["question"], 2)
check("BM25 retrieval returns k units and is deterministic",
      len(hits) == 2 and [u.unit_id for u in hits]
      == [u.unit_id for u in store.retrieve(future_q["question"], 2)])
check("an empty store retrieves nothing",
      ast_.EvidenceStore([]).retrieve("anything", 2) == [])

pointer_alloc = ap.build_allocation("pointer__tau0", units, anchor["qid"], 80,
                                    COMPRESSED, p_hat=TRUE, cfg=SETTINGS)
pointer_units = [u for u in units if pointer_alloc.actions[u.unit_id] == ap.POINTER]
leaked = [u for u in pointer_units
          for q in dossier["questions"]
          for g in q["golds"]
          if g.lower() in ap.pointer_line(u).lower()]
check("a pointer line never leaks a gold answer", not leaked,
      f"pointers={len(pointer_units)}")

# The load-bearing check: no single answer context may approach the whole source.
raises("delivering the whole source with retrieval on top is rejected",
       ast_.assert_bounded_recall,
       " ".join(u.text for u in units), 2, units, "cheating")
raises("a tiny message with unbounded k is rejected",
       ast_.assert_bounded_recall, "", len(units), units, "cheating_k")
ast_.assert_bounded_recall(built_ret["message_text"], alloc_ret.retrieval_k,
                           units, "retrieval_only")
delivered_units = sum(1 for u in units if u.text in built_ret["message_text"])
check("a real retrieval policy stays a bounded recall channel", True,
      f"{delivered_units} delivered + k=2 of {len(units)} units")

sealed = ast_.SealedContext(qid="q", question="Q?", message_text="M",
                            retrieved_text=("R1", "R2"), policy="p")
check("a sealed context renders message and recalled evidence",
      "M" in sealed.render() and "R1" in sealed.render())
raises("the answerer rejects anything that is not a SealedContext",
       ast_.answer_prompt, {"question": "Q?", "message_text": "M"})

entropy_low = ap.normalised_entropy(ad.point_mass("finance"))
entropy_high = ap.normalised_entropy(ad.uniform())
check("normalised entropy spans [0, 1] from point mass to uniform",
      abs(entropy_low) < 1e-12 and abs(entropy_high - 1.0) < 1e-12)
mode_confident = ap.build_allocation("uncertainty_aware__tau0", units, anchor["qid"], 80,
                                     COMPRESSED, p_hat=TRUE, cfg=SETTINGS).notes["mode"]
mode_uncertain = ap.build_allocation("uncertainty_aware__tau1", units, anchor["qid"], 80,
                                     COMPRESSED, p_hat=ad.uniform(), cfg=SETTINGS).notes["mode"]
check("the uncertainty-aware policy switches strategy on entropy",
      mode_confident == "preserve" and mode_uncertain == "retrieve",
      f"{mode_confident} -> {mode_uncertain}")

raises("an unknown policy family is rejected",
       ap.build_allocation, "telepathy", units, anchor["qid"], 80, COMPRESSED)
raises("a predicting policy without an estimate is rejected",
       ap.build_allocation, "preserve__tau0", units, anchor["qid"], 80, COMPRESSED)

print("\nALL ANTICIPATORY CHECKS PASSED")
