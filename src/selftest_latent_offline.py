"""Offline regression checks for Experiment 13. No API key, no GPU, no weights.

What this gates, in order of how badly a silent failure would corrupt results:

1. **Isolation.** ``SealedHandoff`` is unchanged, the new sealed payload holds
   immutable bytes rather than a live tensor, evaluator ids cannot ride in the
   metadata, and a payload survives a wire round-trip with nothing added.
2. **Accounting.** Positions, bytes and amortized cost are computed the way the
   design says, including the arithmetic that makes a position win a byte loss.
3. **The band.** A two-sided position band behaves like Experiment 10's word
   band, so an arm cannot lose future utility merely by sending less.
4. **Controls.** On the deterministic backend, the payload-dependence controls
   move in the right direction: zeroing or swapping the payload destroys the
   answer, and token ids agree with their own embeddings.
5. **Statistics.** Source-clustered resampling is actually clustered, and the
   efficiency gate refuses to fire on a reduction bought with lost accuracy.

The fake backend is a plumbing simulator. Nothing it produces is a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fictional_qa as fqa  # noqa: E402
import latent_backend as lb  # noqa: E402
import latent_handoff as lh  # noqa: E402
import latent_metrics as lm  # noqa: E402
import run_latent_reusability as run  # noqa: E402


PASSED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))
    PASSED += 1
    print(f"PASS  {label}" + (f" -- {detail}" if detail else ""))


def raises(label: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except (ValueError, TypeError, lh.PayloadError):
        global PASSED
        PASSED += 1
        print(f"PASS  {label}")
        return
    raise AssertionError(f"FAIL  {label} -- no error raised")


# ===========================================================================
# 1. Isolation
# ===========================================================================

print("\n-- isolation ---------------------------------------------------------")

lh.assert_sealed_handoff_untouched()
check("SealedHandoff is still four strings", True)

vectors = np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / 7.0
payload = lh.seal_vectors("px-demo", "Where is it?", "latent_sb", vectors)
check("a vector payload seals to bytes", type(payload.payload) is bytes,
      f"{payload.nbytes} bytes for {payload.shape}")
check("one delivered vector is one receiver position", payload.positions == 8)

raises("a live tensor cannot be sealed as a payload",
       lh.SealedLatentPayload, qid="q", question="?", channel="c",
       dtype="float32", shape=(1, 2), positions=1,
       payload=bytearray(b"\0" * 8), payload_sha256="x")

raises("a mismatched payload hash is rejected",
       lh.SealedLatentPayload, qid="q", question="?", channel="c",
       dtype="float32", shape=(1, 2), positions=1, payload=b"\0" * 8,
       payload_sha256="deadbeef")

raises("a byte count inconsistent with dtype/shape is rejected",
       lh.SealedLatentPayload, qid="q", question="?", channel="c",
       dtype="float32", shape=(4, 4), positions=4, payload=b"\0" * 8,
       payload_sha256=lh.sha256_bytes(b"\0" * 8))

raises("positions must match the delivered row count",
       lh.seal_vectors, "q", "?", "c", vectors, "float32", {"positions": 99})

raises("an unknown metadata key is rejected",
       lh.seal_text, "q", "?", "text_prose", "hello", 3, {"source_path": "data/x"})

raises("an evaluator card id cannot ride in the metadata",
       lh.seal_text, "q", "?", "text_prose", "hello", 3, {"codec": "cards C01 C02"})

raises("a fictional fact id cannot ride in the metadata",
       lh.seal_text, "q", "?", "text_prose", "hello", 3, {"codec": "fic:10"})

wire = lh.to_wire(payload)
restored = lh.from_wire(wire)
check("a payload survives a wire round-trip exactly",
      restored.payload_sha256 == payload.payload_sha256
      and restored.shape == payload.shape
      and restored.question == payload.question)
check("the restored array matches the original within bf16 precision",
      np.allclose(restored.as_array(), payload.as_array(), atol=0.01))
check("as_array returns a fresh copy, not a view into the sealed bytes",
      restored.as_array().flags.owndata)

text_payload = lh.seal_text("px-t", "Where is it?", "text_prose",
                            "The Cabinet is on Harthane Quay.", positions=9)
raises("a text payload refuses as_array", text_payload.as_array)
raises("a vector payload refuses as_text", payload.as_text)
check("a reader prompt for a latent payload carries no message block",
      "MESSAGE FROM THE OTHER AGENT" not in lh.reader_prompt(payload))
check("a reader prompt for a text payload carries the message",
      "Harthane Quay" in lh.reader_prompt(text_payload))

raises("omitted evidence in reader material is caught",
       lh.assert_no_source_leak,
       "MESSAGE: the vault is under the pier\nQUESTION: x",
       ["the vault is under the pier"])
lh.assert_no_source_leak("MESSAGE: nothing relevant", ["the vault is under the pier"])
check("clean reader material passes the leak check", True)

check("opaque ids do not contain the corpus id",
      "fic:10" not in lh.opaque_id("fic:10", "latent_sb", "q"))
check("opaque ids separate a control from its donor",
      lh.opaque_id("p", "latent_sb", "q") != lh.opaque_id("p", "latent_sb", "q", "shuffled"))

# bfloat16 conversion, since numpy has no native dtype for it
probe = np.array([[1.0, -2.5, 0.125, 65504.0]], dtype=np.float32)
check("bf16 round-trip preserves value to bf16 precision",
      np.allclose(lh.bf16_bytes_to_f32(lh.f32_to_bf16_bytes(probe), (1, 4)),
                  probe, rtol=0.01))
raises("a non-finite payload is rejected", lh.seal_vectors, "q", "?", "c",
       np.array([[np.nan, 1.0]], dtype=np.float32))


# ===========================================================================
# 2. Accounting
# ===========================================================================

print("\n-- accounting --------------------------------------------------------")

acct = lm.PositionAccount(payload=32, schema=11, instruction=40, question=12)
check("added positions charge the schema block, not just the payload",
      acct.added == 43, f"added={acct.added}")
check("total positions include the shared instruction and question",
      acct.total == 95)

# The accounting trap, as arithmetic. Qwen3-4B: hidden 2560, 36 layers,
# 8 KV heads, head_dim 128.
vec = lm.vector_payload_bytes(positions=8, width=2560, bits=16)
ids = lm.token_id_payload_bytes(positions=128, bytes_per_id=4)
kv = lm.kv_payload_bytes(positions=8, layers=36, kv_heads=8, head_dim=128, bits=16)
check("8 bf16 hidden vectors are 40 KiB", vec.payload == 40 * 1024)
check("128 int32 token ids are 512 bytes", ids.payload == 512)
check("8 full KV positions are 1.125 MiB", kv.payload == int(1.125 * 1024 * 1024))
check("16x fewer positions can be 80x more bytes",
      vec.payload == 80 * ids.payload,
      f"{vec.payload} vs {ids.payload}")

cost = lm.CostBreakdown(sender_prefill_s=0.4, sender_decode_s=1.6,
                        align_s=0.05, serialize_s=0.05,
                        reader_prefill_s=0.2, reader_decode_s=0.3)
check("encode cost is one-off", abs(cost.encode_s - 2.1) < 1e-9)
check("read cost is per query", abs(cost.read_s - 0.5) < 1e-9)
check("C(1) = encode + read", abs(cost.amortized(1) - 2.6) < 1e-9)
check("C(16) amortizes the encode over reads",
      abs(cost.amortized(16) - (2.1 + 16 * 0.5)) < 1e-9)
check("a persisted payload transfers once for four total reads",
      abs(cost.amortized(4, transfer_s=0.1) - (2.1 + 0.1 + 4 * 0.5)) < 1e-9)
check("actual repeated transfers are charged explicitly",
      abs(cost.amortized(4, transfer_s=0.1, transfers=4) - (2.1 + 4 * 0.6)) < 1e-9)
raises("a negative horizon is rejected", cost.amortized, -1)

record = cost.record(horizons=(1, 4, 16))
check("the cost record itemizes rather than totalling",
      {"sender_decode_s", "reader_prefill_s", "amortized_s_n16"} <= set(record))


# ===========================================================================
# 3. The two-sided band
# ===========================================================================

print("\n-- position band -----------------------------------------------------")

band = lm.PositionBand(target=32, floor_ratio=0.85)
check("the floor is the ceiling of target*ratio", band.floor == 28)
check("a full message fits", band.fits(31))
check("an over-budget message does not fit", not band.fits(33))
check("an underfilled message does not fit either -- a cap is not a band",
      not band.fits(20))
check("verdicts name the direction",
      (band.verdict(20), band.verdict(30), band.verdict(40)) == ("under", "ok", "over"))

summary = lm.band_summary([28, 30, 31, 32], band)
check("band summaries expose realized fill, not just the target",
      abs(summary["mean_fill_ratio"] - (121 / 4) / 32) < 1e-9,
      f"fill={summary['mean_fill_ratio']:.3f}")
check("under-floor messages are counted, not dropped",
      lm.band_summary([10, 30], band)["under_floor"] == 1)


# ===========================================================================
# 4. The backend and its controls
# ===========================================================================

print("\n-- deterministic backend --------------------------------------------")

dossiers = fqa.load_fictional_dossiers()
check("the fictional corpus loads", len(dossiers) == 20, f"n={len(dossiers)}")

backend = lb.FakeBackend(lb.BackendConfig(model_id="fake/deterministic",
                                          device="cpu", dtype="float32"))
run.prime_fake(backend, dossiers)

dossier = dossiers[0]
cards = fqa.evidence_cards(dossier)
check("each dossier has six evidence cards", len(cards) == 6)

target = cards[0]
material = (f"MESSAGE FROM THE OTHER AGENT:\n{target.evidence_text}\n\n"
            f"QUESTION: {target.question}\nANSWER:")
answered = backend.generate(material).text
check("the simulated reader answers from delivered evidence",
      answered == target.golds[0], f"{answered!r}")

blank = backend.generate(f"MESSAGE FROM THE OTHER AGENT:\n\n\n"
                         f"QUESTION: {target.question}\nANSWER:").text
check("it cannot answer with no evidence delivered", blank == lb.NOT_IN_MESSAGE)

other = backend.generate(
    f"MESSAGE FROM THE OTHER AGENT:\n{cards[3].evidence_text}\n\n"
    f"QUESTION: {target.question}\nANSWER:").text
check("it cannot answer from a different card's evidence",
      other != target.golds[0], f"{other!r}")

check("generation is deterministic",
      backend.generate(material).text == backend.generate(material).text)

# -- the continuous path
embeds = backend.embed_text(target.evidence_text)
check("embedding rows match token count",
      embeds.shape == (backend.count_positions(target.evidence_text),
                       backend.hidden_width))

suffix = f"\n\nQUESTION: {target.question}\nANSWER:"
from_embeds = backend.generate_from_prefix(embeds, suffix).text
check("token ids and their own embeddings agree -- the wrapper is sound",
      from_embeds == answered, f"{from_embeds!r} vs {answered!r}")

zeroed = backend.generate_from_prefix(np.zeros_like(embeds), suffix).text
check("a zeroed prefix cannot answer", zeroed == lb.NOT_IN_MESSAGE)

donor = backend.embed_text(dossiers[1].fact(
    fqa.evidence_cards(dossiers[1])[0].fact_id).evidence_text)
shuffled = backend.generate_from_prefix(donor, suffix).text
check("a cross-dossier prefix cannot answer this question",
      shuffled != answered, f"{shuffled!r}")

# -- source conditioning
prompt = f"{run.TRANSCRIPT_INSTRUCTION}\n\nEVIDENCE:\n{target.evidence_text}\n"
with_source = backend.transcript_states(prompt, k=8)
replay = backend.transcript_states(
    f"{run.TRANSCRIPT_INSTRUCTION}\n\nEVIDENCE:\n{with_source.text}\n", k=8)
check("H's realized length is recorded, not just the k transmitted rows",
      with_source.realized_length >= with_source.delivered_k,
      f"realized={with_source.realized_length} delivered={with_source.delivered_k}")
check("source-conditioned states differ from a source-free re-encoding",
      not np.allclose(with_source.states, replay.states),
      "so the source_free control can detect the difference")
check("a short transcript is flagged rather than padded",
      backend.transcript_states("EVIDENCE:\nx\n", k=64).short)

# -- alignment
raw = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32) * 50
aligned = lb.align_states(raw, target_rms=0.02)
check("alignment rescales to the target RMS",
      abs(float(np.sqrt(np.mean(aligned ** 2))) - 0.02) < 1e-6)
check("alignment preserves direction",
      float(np.corrcoef(raw.ravel(), aligned.ravel())[0, 1]) > 0.999)
check("a degenerate state block aligns to zeros rather than NaN",
      np.all(lb.align_states(np.zeros((4, 8), np.float32), 0.02) == 0))


# ===========================================================================
# 5. Packets and the runner
# ===========================================================================

print("\n-- packets and runner ------------------------------------------------")

selections = run.read_jsonl(
    ROOT / "runs/fictional_summary_generalization/n20/selections.jsonl")
packets = run.build_packets(dossiers, selections, k=4)
check("the K=4 panel has 140 packets", len(packets) == 140, f"n={len(packets)}")
check("20 generic and 120 conditioned",
      (sum(1 for p in packets if p.mode == "generic"),
       sum(1 for p in packets if p.mode == "conditioned")) == (20, 120))
check("every packet carries exactly four selected facts",
      all(len(p.selected_fact_ids) == 4 for p in packets))
check("every packet records the two omitted facts",
      all(len(p.omitted_fact_ids) == 2 for p in packets))
check("packet text never carries an omitted card's evidence",
      all(all(o not in p.packet_text for o in p.omitted_evidence) for p in packets))
check("packet ids are opaque",
      all("fic:" not in p.packet_id for p in packets))

sample = next(p for p in packets if p.mode == "conditioned")
wp = run.writer_prompt("text_prose", sample, band)
check("a conditioned writer prompt shows the announced question only",
      sample.announced_question in wp)
hidden = [c.question for c in fqa.evidence_cards(dossiers[0])
          if c.question != sample.announced_question]
generic = next(p for p in packets if p.mode == "generic")
check("a generic writer prompt shows no question at all",
      "current question" not in run.writer_prompt("text_prose", generic, band))
check("no writer prompt carries a gold answer",
      not any(g in wp for c in fqa.evidence_cards(dossiers[0])
              for g in c.golds if len(g) > 6 and c.question not in wp))

check("the record channel is told its schema, and pays for it",
      run.RECORD_SCHEMA and "field: value" in run.RECORD_SCHEMA)

# -- a small end-to-end pass on the deterministic backend
small = [p for p in packets if p.dossier_id == packets[0].dossier_id][:2]
messages, answers = run.run_panel_a(
    backend, small, dossiers, run.CHANNELS, band, k=4, dtype="bfloat16",
    controls=("direct_packet", "closed_book", "zero_prefix", "shuffled_prefix",
              "token_replay", "embed_replay"))
check("every packet-channel pair produced one message",
      len(messages) == len(small) * len(run.CHANNELS),
      f"messages={len(messages)}")
check("every message is scored on all six questions",
      len(answers) == len(small) * (len(run.CHANNELS) + 6) * 6,
      f"answers={len(answers)}")
check("answer rows carry both position and byte accounting",
      {"added_positions", "payload_bytes", "band_fill_ratio", "encode_s"}
      <= set(answers[0]))
check("answer rows mark whether the question's evidence was delivered",
      any(r["evidence_delivered"] for r in answers)
      and any(not r["evidence_delivered"] for r in answers))
check("the announced question is flagged on conditioned packets",
      any(r["is_announced"] for r in answers if r["mode"] == "conditioned"))
check("no answer row leaks the source document",
      all(dossiers[0].document[:60] not in json.dumps(r) for r in answers))

by_arm = {}
for row in answers:
    by_arm.setdefault(row["arm"], []).append(row)
delivered = lambda arm: [r for r in by_arm.get(arm, []) if r["evidence_delivered"]]

direct_em = float(np.mean([r["em"] for r in delivered("direct_packet")]))
closed_em = float(np.mean([r["em"] for r in by_arm["closed_book"]]))
zero_em = float(np.mean([r["em"] for r in delivered("zero_prefix")]))
shuf_em = float(np.mean([r["em"] for r in delivered("shuffled_prefix")]))
tok_em = float(np.mean([r["em"] for r in delivered("token_replay")]))
emb_em = float(np.mean([r["em"] for r in delivered("embed_replay")]))

check("the ceiling control answers from the selected packet",
      direct_em == 1.0, f"direct EM={direct_em}")
check("the closed-book floor is zero on fictional material",
      closed_em == 0.0, f"closed-book EM={closed_em}")
check("a zeroed payload answers nothing", zero_em == 0.0)
check("a shuffled payload answers nothing", shuf_em == 0.0)
check("token ids and their embeddings agree end-to-end",
      abs(tok_em - emb_em) < 1e-9, f"{tok_em} vs {emb_em}")
check("controls never collide with the real arm in a cache",
      len({r["qid"] for r in answers}) == len(answers))

# Infrastructure controls are a yes/no plumbing question and must not be paid
# for on all 140 packets; behavioural controls must be.
two_dossiers = ([p for p in packets if p.dossier_id == "fic:1"][:2]
                + [p for p in packets if p.dossier_id == "fic:10"][:2])
check("two dossiers are available for the split check",
      len({p.dossier_id for p in two_dossiers}) == 2)
_, split_answers = run.run_panel_a(
    backend, two_dossiers, dossiers, ("text_prose",), band, k=4,
    dtype="bfloat16", controls=("direct_packet", "token_replay"),
    n_infra_dossiers=1)
infra_dossiers = {r["dossier_id"] for r in split_answers
                  if r["arm"] == "token_replay"}
behav_dossiers = {r["dossier_id"] for r in split_answers
                  if r["arm"] == "direct_packet"}
check("infrastructure controls run on the fixed subset only",
      len(infra_dossiers) == 1, f"dossiers={sorted(infra_dossiers)}")
check("behavioural controls run on every packet",
      len(behav_dossiers) == 2, f"dossiers={sorted(behav_dossiers)}")
check("the infrastructure subset is fixed by sorted id, not sampled",
      run.infrastructure_dossiers(packets, 4)
      == run.infrastructure_dossiers(list(reversed(packets)), 4))

dry = run.dry_run_report(packets, run.CHANNELS, run.CONTROLS, 6, {}, 4)
check("the dry run separates the two control classes",
      dry["infrastructure_control_evaluations"]
      < dry["behavioral_control_evaluations"],
      f"infra={dry['infrastructure_control_evaluations']} "
      f"behavioural={dry['behavioral_control_evaluations']}")

gates = run.development_gates(answers, {"gates": {}})
check("the gate block reports every threshold it applied",
      set(gates["thresholds"]) == {"direct_source_em_min", "closed_book_em_max",
                                   "latent_gap_max", "shuffle_sensitivity_min",
                                   "replay_gap_max"})
check("gates are evaluated, not assumed",
      gates["gate_direct_source"] is True and gates["gate_closed_book"] is True)

# -- dry run writes nothing
before = {p: p.stat().st_mtime for p in (ROOT / "runs").rglob("*") if p.is_file()}
report = run.dry_run_report(packets, run.CHANNELS, run.CONTROLS, 6, {})
after = {p: p.stat().st_mtime for p in (ROOT / "runs").rglob("*") if p.is_file()}
check("--dry-run touches no file under runs/", before == after)
check("the dry run reports the design's 2,520 core evaluations",
      report["core_answer_evaluations"] == 2520,
      f"core={report['core_answer_evaluations']}")
check("the dry run declares zero writes", report["writes"] == 0)


# ===========================================================================
# 6. Statistics
# ===========================================================================

print("\n-- statistics --------------------------------------------------------")

rng = np.random.default_rng(7)
sources = [f"d{i // 6}" for i in range(120)]
base = rng.random(120)
clustered = lm.cluster_bootstrap_delta(sources, base + 0.05, base)
check("a constant paired shift is recovered",
      abs(clustered["delta"] - 0.05) < 1e-9, f"delta={clustered['delta']:.4f}")
check("the sampling unit is the source, not the row",
      clustered["n_sources"] == 20 and clustered["n"] == 120)

# Correlated within source: row resampling would understate the interval.
source_effect = np.repeat(rng.standard_normal(20), 6)
t = source_effect + rng.standard_normal(120) * 0.05
c = np.zeros(120)
wide = lm.cluster_bootstrap_delta(sources, t, c)
narrow_sources = [f"d{i}" for i in range(120)]
narrow = lm.cluster_bootstrap_delta(narrow_sources, t, c)
check("clustering widens the interval when rows share a source",
      (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"]),
      f"clustered={wide['hi'] - wide['lo']:.3f} vs unclustered="
      f"{narrow['hi'] - narrow['lo']:.3f}")
raises("misaligned pairs are rejected",
       lm.cluster_bootstrap_delta, sources, base, base[:10])

ni = lm.noninferiority({"lo": -0.01}, margin=0.03)
check("a lower bound inside the margin is noninferior", ni["noninferior"])
check("a lower bound outside the margin is not",
      not lm.noninferiority({"lo": -0.06}, margin=0.03)["noninferior"])

saving = lm.position_saving([34] * 10, [61] * 10)
check("position saving uses realized counts, not nominal budgets",
      abs(saving["reduction"] - (61 - 34) / 61) < 1e-9,
      f"reduction={saving['reduction']:.3f}")

good = lm.efficiency_claim(saving, {"lo": -0.01}, {"lo": -0.02},
                           treatment_floor={"lo": 0.3}, control_floor={"lo": 0.3}, hard_caps_ok=True)
check("the efficiency claim fires only when all three conditions hold",
      good["claim_supported"])
bought = lm.efficiency_claim(saving, {"lo": -0.20}, {"lo": 0.05})
check("a reduction bought with lost current accuracy is not a win",
      not bought["claim_supported"])
small_cut = lm.efficiency_claim(lm.position_saving([58] * 10, [61] * 10),
                                {"lo": -0.01}, {"lo": -0.01})
check("a position reduction under 25% does not support the claim",
      not small_cut["claim_supported"])
check("both utility changes are always reported alongside the verdict",
      "u_now" in bought and "u_future" in bought)
check("noninferiority against an uninformative baseline cannot pass",
      not lm.efficiency_claim(saving, {"lo": 0}, {"lo": 0},
          treatment_floor={"lo": -0.5}, control_floor={"lo": -0.5},
          hard_caps_ok=True)["claim_supported"])

print(f"\nAll {PASSED} offline checks passed.")
