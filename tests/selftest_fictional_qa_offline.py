"""Offline checks for the shared fictional-QA adapter. No API key or network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import fictional_qa as fq  # noqa: E402
import handoffs as hm  # noqa: E402


failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
    if not condition:
        failures.append(label)


print("=== corpus loading and evidence projection ===")
manifest, dossiers = fq.load_fictional_corpus()
check("manifest is loaded separately", manifest.get("dataset") == "size_adaptation_fictional")
check("all 20 existing fictional dossiers load", len(dossiers) == 20, str(len(dossiers)))
check("every dossier exposes six QA facts", all(len(d.facts) == 6 for d in dossiers))
all_fact_ids = [fact.fact_id for dossier in dossiers for fact in dossier.facts]
check("fact ids are globally unique", len(all_fact_ids) == len(set(all_fact_ids)))
check("content hashes are stable SHA-256 values",
      all(len(d.content_hash) == 64 for d in dossiers))

for dossier in dossiers:
    for fact in dossier.facts:
        assert any(fq._contains(fact.evidence_text, gold) for gold in fact.golds), fact.fact_id
        if fact.role == "target":
            assert not fact.evidence_text.casefold().startswith("the answer to the target question")
            assert fq._normal(fact.evidence_text) in fq._normal(dossier.document)
check("all evidence units contain their golds", True)
check("generic target statements are replaced by source sentences", True)


print("\n=== deterministic cards and rotations ===")
dossier = dossiers[0]
cards_a = fq.evidence_cards(dossier)
cards_b = fq.evidence_cards(dossier)
cards_other = fq.evidence_cards(dossier, seed="different-order")
check("card order is reproducible", cards_a == cards_b)
check("order seed affects the six-card order",
      [c.fact_id for c in cards_a] != [c.fact_id for c in cards_other])
check("card ids are opaque and role-neutral",
      all(c.card_id != c.fact_id and "target" not in c.card_id and "side" not in c.card_id
          for c in cards_a))
check("opaque card ids round-trip to evaluator fact ids",
      fq.card_ids_to_fact_ids(cards_a, [c.card_id for c in cards_a])
      == tuple(c.fact_id for c in cards_a))
rendered_source = fq.render_source_cards(cards_a)
check("source cards expose every id and evidence unit",
      all(c.card_id in rendered_source and c.evidence_text in rendered_source for c in cards_a))
check("source cards do not expose evaluator ids",
      all(c.fact_id not in rendered_source for c in cards_a))
check("source cards expose no fact questions",
      all(c.question not in rendered_source for c in cards_a))

rot4_a = fq.role_rotations(dossier, 4)
rot4_b = fq.role_rotations(dossier, 4)
rot6 = fq.role_rotations(dossier, 6)
check("four-role rotations are deterministic", rot4_a == rot4_b)
check("requested rotation counts are exact", len(rot4_a) == 4 and len(rot6) == 6)
original_target = next(f.fact_id for f in dossier.facts if f.role == "target")
check("four-role design retains the original target",
      original_target in {r.target_fact_id for r in rot4_a})
check("six-role design announces every fact exactly once",
      {r.target_fact_id for r in rot6} == set(dossier.fact_map))
check("every rotation hides all other five facts",
      all(len(r.hidden_fact_ids) == 5
          and set(r.hidden_fact_ids) == set(dossier.fact_map) - {r.target_fact_id}
          for r in (*rot4_a, *rot6)))


print("\n=== prompt isolation and generic reuse ===")
k = 3
generic = fq.generic_selection_prompt(dossier, cards_a, k)
keys = {fq.selection_material_key(dossier, rotation, "generic", k) for rotation in rot4_a}
check("generic material has one reusable key per dossier", len(keys) == 1)
check("generic source prompt contains no QA question",
      all(fact.question not in generic for fact in dossier.facts))
for rotation in rot4_a:
    conditioned = fq.conditioned_selection_prompt(dossier, rotation, cards_a, k)
    block = fq.question_block(rotation)
    check(f"conditioned prompt differs only by A block ({rotation.target_fact_id})",
          conditioned.replace(block, "") == generic)
    check(f"conditioned prompt includes only A question ({rotation.target_fact_id})",
          rotation.target_question in conditioned
          and all(question not in conditioned for question in rotation.hidden_questions))
conditioned_keys = {
    fq.selection_material_key(dossier, rotation, "conditioned", k) for rotation in rot4_a
}
check("conditioned material keys distinguish target rotations", len(conditioned_keys) == 4)

hidden_id = rot4_a[0].hidden_fact_ids[0]
reopened = fq.reopened_selection_prompt(dossier, hidden_id, cards_a, k)
check("reopened prompt shows B and no other question",
      dossier.fact(hidden_id).question in reopened
      and all(f.question not in reopened for f in dossier.facts if f.fact_id != hidden_id))


print("\n=== strict fixed-K selection parsing ===")
valid_ids = [card.card_id for card in cards_a]
chosen_cards = valid_ids[:k]
chosen = list(fq.card_ids_to_fact_ids(cards_a, chosen_cards))
valid_raw = json.dumps({fq.SELECTION_KEY: chosen_cards})
parsed = fq.parse_fixed_k_selection(valid_raw, valid_ids, k)
check("exact schema and K parse", parsed.valid and parsed.selected_fact_ids == tuple(chosen_cards))
check("valid parse has no diagnostics", parsed.error == "" and parsed.errors == ())

wrong_count = fq.parse_fixed_k_selection(
    json.dumps({fq.SELECTION_KEY: chosen_cards[:2]}), valid_ids, k)
check("wrong K is invalid without operative repair",
      not wrong_count.valid and wrong_count.selected_fact_ids == ()
      and len(wrong_count.recognized_fact_ids) == 2
      and "wrong_count" in wrong_count.error)
duplicate = fq.parse_fixed_k_selection(
    json.dumps({fq.SELECTION_KEY: [chosen_cards[0], chosen_cards[0], chosen_cards[1]]}), valid_ids, k)
check("duplicate ids are diagnosed", not duplicate.valid and "duplicate_id" in duplicate.error)
unknown = fq.parse_fixed_k_selection(
    json.dumps({fq.SELECTION_KEY: [chosen_cards[0], chosen_cards[1], "not-a-card"]}), valid_ids, k)
check("unknown ids are diagnosed", not unknown.valid and "unknown_id" in unknown.error)
extra = fq.parse_fixed_k_selection(
    json.dumps({fq.SELECTION_KEY: chosen_cards, "explanation": "x"}), valid_ids, k)
check("extra JSON keys are rejected", not extra.valid and "keys_must_equal" in extra.error)
duplicate_key = fq.parse_fixed_k_selection(
    '{"selected_fact_ids": [], "selected_fact_ids": []}', valid_ids, k)
check("duplicate JSON keys are rejected",
      not duplicate_key.valid and "duplicate_json_key" in duplicate_key.error)
fenced = fq.parse_fixed_k_selection(f"```json\n{valid_raw}\n```", valid_ids, k)
check("markdown-fenced JSON is rejected by the strict parser",
      not fenced.valid and fenced.error == "invalid_json")


print("\n=== deterministic fixed-capacity packet rendering ===")
partial = fq.render_packet_slots(cards_a, chosen[:2], k)
check("packet renderer emits exactly K slots", partial.count("[SLOT ") == k)
check("underfilled selection receives explicit EMPTY padding", "[SLOT 3] EMPTY" in partial)
check("packet order follows source-card order, not selection order",
      fq.render_packet_slots(cards_a, tuple(reversed(chosen)), k)
      == fq.render_packet_slots(cards_a, chosen, k))
check("packets expose ids but no questions",
      all(question not in partial for question in (f.question for f in dossier.facts)))


print("\n=== sealed relay boundary ===")
handoff = hm.Handoff(dossier.item_id, "fictional_packet", None,
                     fq.render_packet_slots(cards_a, chosen, k))
sealed = hm.seal(rot4_a[0].target_question, handoff)
relay = fq.sealed_relay_prompt(sealed, k=2)
check("relay sees sealed packets and its one question",
      sealed.handoff_text in relay and sealed.question in relay)
check("relay cannot recover omitted source cards",
      all(card.evidence_text not in relay for card in cards_a if card.fact_id not in chosen))
for bad in (dossier, handoff, "notes", {"handoff_text": "notes"}):
    try:
        fq.sealed_relay_prompt(bad, 2)
        check(f"relay rejects {type(bad).__name__}", False)
    except TypeError:
        check(f"relay rejects {type(bad).__name__}", True)


print("\n=== capacity-preserving support interventions ===")
target_id = rot4_a[0].target_fact_id
support_id = next(fact_id for fact_id in rot4_a[0].hidden_fact_ids
                  if fact_id not in chosen)
before = tuple(chosen)
# Ensure the protected A card participates in the fixture.
if target_id not in before:
    before = (target_id,) + tuple(fid for fid in before if fid != target_id)[:k - 1]
restoration = fq.restore_support(
    cards_a, before, support_id, protected_fact_ids=(target_id,), seed="fixture")
check("restoration inserts B and keeps K",
      restoration.changed and support_id in restoration.after_fact_ids
      and len(restoration.after_fact_ids) == len(before))
check("restoration protects announced A",
      target_id in restoration.after_fact_ids and restoration.removed_fact_id != target_id)
sham = fq.sham_swap(
    cards_a,
    before,
    forbidden_fact_ids=(target_id, support_id),
    protected_fact_ids=(target_id,),
    victim_fact_id=restoration.removed_fact_id,
    seed="fixture",
)
check("sham changes the same slot without adding B",
      sham.changed and sham.removed_fact_id == restoration.removed_fact_id
      and support_id not in sham.after_fact_ids
      and len(sham.after_fact_ids) == len(before))
paired_restore, paired_sham = fq.restoration_and_sham(
    cards_a, before, support_id, protected_fact_ids=(target_id,), seed="fixture-paired")
check("paired helper preserves capacity in both arms",
      paired_restore.changed and paired_sham.changed
      and len(paired_restore.after_fact_ids) == len(before)
      and len(paired_sham.after_fact_ids) == len(before)
      and paired_restore.removed_fact_id == paired_sham.removed_fact_id)
check("intervened packets still render to exactly K slots",
      fq.render_packet_slots(cards_a, restoration.after_fact_ids, k).count("[SLOT ") == k
      and fq.render_packet_slots(cards_a, sham.after_fact_ids, k).count("[SLOT ") == k)


print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
raise SystemExit(1 if failures else 0)
