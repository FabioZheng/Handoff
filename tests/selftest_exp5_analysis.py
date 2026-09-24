"""Offline proof that analyse_exp5_capped separates total from direct effects.

Two synthetic worlds, each with a perfect reader (F1 = 1 exactly when the gold
answer is in the message), so the right answer is known in advance:

World A - length is the only mechanism. Whether an answer is in a message
    depends only on how many words the message has, and the conditioned writer
    is simply shorter. The total effect must be negative; the controlled direct
    effect (regression and matched length) must be near zero.

World B - allocation is the mechanism. Both arms deliver the same lengths, but
    the conditioned writer drops far evidence. The total effect and the direct
    effect must both be negative, and far-evidence survival must fall while
    current-question survival rises.

Also checks the report renders, Holm is monotone, and the allocation step table
has the right shape.

    python tests/selftest_exp5_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import analyse_exp5_capped as an  # noqa: E402

CAPS = (64, 128, 256)
N_DOCS = 40


def make_docs() -> list[dict]:
    docs = []
    for d in range(N_DOCS):
        paragraphs = [{"id": f"p{i}", "text": f"filler paragraph {i} of doc {d}."} for i in range(12)]
        # q0 anchors on p0; q1 shares p0 (shared); q2 on p2 (near); q3 on p10 (far).
        evidence = {"q0": "p0", "q1": "p0", "q2": "p2", "q3": "p10"}
        questions = []
        for q, para in evidence.items():
            gold = f"zeta{d}x{q} omega{d}x{q}"
            idx = int(para[1:])
            paragraphs[idx]["text"] += f" The answer is {gold}."
            questions.append({"qid": f"{d}-{q}", "question": f"question {q}", "golds": [gold],
                              "answer_type": "extractive", "evidence_sets": [[para]]})
        source = " ".join(p["text"] for p in paragraphs)
        docs.append({"id": f"doc{d}", "corpus": "qasper", "source": source,
                     "paragraphs": paragraphs, "questions": questions})
    return docs


def message(doc, policy, cap, anchor, words, include) -> dict:
    filler = [f"w{i}" for i in range(max(0, words - 2 * len(include)))]
    text = " ".join(filler + [g for q in doc["questions"] if q["qid"] in include for g in q["golds"]])
    return {"message_id": f"{doc['id']}|{policy}|{cap}|{anchor}", "document_id": doc["id"],
            "corpus": doc["corpus"], "policy": policy, "anchor": anchor, "cap": cap, "block": "core",
            "valid": True, "text": text, "delivered_words": len(text.split()),
            "fill_ratio": len(text.split()) / cap, "realized_tokens": len(text.split()),
            "truncated_words": 0, "over_cap": False}


def answers_for(doc, m) -> list[dict]:
    rows = []
    for q in doc["questions"]:
        hit = float(an.answer_present(m["text"], q["golds"]))
        rows.append({"message_id": m["message_id"], "document_id": doc["id"], "corpus": doc["corpus"],
                     "qid": q["qid"], "policy": m["policy"], "anchor": m["anchor"], "cap": m["cap"],
                     "block": "core", "f1": hit, "em": hit, "judge_correct": hit,
                     "delivered_words": m["delivered_words"]})
    return rows


def build(world: str):
    docs, messages, answers = make_docs(), [], []
    for i, doc in enumerate(make_docs()):
        doc = docs[i]
        jitter = 1 + 0.08 * ((i % 5) - 2) / 2
        for cap in CAPS:
            for family, (cond, gen) in an.FAMILIES.items():
                if world == "A":
                    # Length is everything: an answer is present iff the message
                    # is long enough for it; thresholds differ by question.
                    thresholds = {q["qid"]: 60 + 50 * k for k, q in enumerate(doc["questions"])}
                    g_words = int(0.95 * cap * jitter)
                    g_inc = [q for q, t in thresholds.items() if g_words >= t]
                    m = message(doc, gen, cap, None, g_words, g_inc)
                    messages.append(m); answers += answers_for(doc, m)
                    for q in doc["questions"]:
                        c_words = int(0.55 * cap * jitter)
                        c_inc = [x for x, t in thresholds.items() if c_words >= t]
                        m = message(doc, cond, cap, q["qid"], c_words, c_inc)
                        messages.append(m); answers += answers_for(doc, m)
                else:
                    # Same lengths; the conditioned writer keeps q_now and shared
                    # evidence but drops everything else, whatever the cap.
                    words = int(0.9 * cap * jitter)
                    g_inc = [q["qid"] for q in doc["questions"]][: 1 + CAPS.index(cap) + 1]
                    m = message(doc, gen, cap, None, words, g_inc)
                    messages.append(m); answers += answers_for(doc, m)
                    for q in doc["questions"]:
                        keep = [x["qid"] for x in doc["questions"]
                                if x["qid"] == q["qid"] or x["evidence_sets"] == q["evidence_sets"]]
                        m = message(doc, cond, cap, q["qid"], words, keep)
                        messages.append(m); answers += answers_for(doc, m)
    return docs, messages, answers


def row(rows, **match):
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    assert len(hits) == 1, (match, len(hits))
    return hits[0]


def main() -> int:
    # ---------------------------------------------------------------- world A
    docs, messages, answers = build("A")
    ix = an.Index(docs, messages, answers)
    prim = an.primary(ix, ["qasper"])
    total = row(prim, family="generation", metric="f1", cap="mean")
    assert total["delta_future"] < -0.1 and total["delta_future_hi"] < 0, total
    sec = row(an.secondary(ix, ["qasper"]), family="generation")
    print("A. length-only world:",
          f"total {total['delta_future']:+.3f} [{total['delta_future_lo']:+.3f}, {total['delta_future_hi']:+.3f}]",
          f"| bins {sec['bins_visible']:+.3f} [{sec['bins_lo']:+.3f}, {sec['bins_hi']:+.3f}]",
          f"| loglin {sec['loglin_visible']:+.3f} vif={sec['loglin_vif']:.1f} identified={sec['loglin_identified']}",
          f"| matched {sec['matched_delta']:+.3f} [{sec['matched_lo']:+.3f}, {sec['matched_hi']:+.3f}] rate={sec['matched_rate']:.2f}",
          f"| bins support={sec['bins_common_support']} identified={sec['bins_identified']}")
    # Length is the whole mechanism here, so the reported direct effect must be
    # a small fraction of the total, and any check that cannot see common
    # support must say so rather than report a number.
    assert abs(sec["matched_delta"]) < 0.25 * abs(total["delta_future"]), sec
    assert not sec["loglin_identified"], "log-linear should be flagged: length is collinear here"
    assert (not sec["bins_identified"]) or abs(sec["bins_visible"]) < 0.25 * abs(total["delta_future"]), sec

    # ---------------------------------------------------------------- world B
    docs, messages, answers = build("B")
    ix = an.Index(docs, messages, answers)
    prim = an.primary(ix, ["qasper"])
    total = row(prim, family="generation", metric="f1", cap="mean")
    assert total["delta_future"] < -0.1 and total["delta_future_hi"] < 0, total
    sec = row(an.secondary(ix, ["qasper"]), family="generation")
    print("B. allocation world secondary:",
          f"bins {sec['bins_visible']:+.3f} [{sec['bins_lo']:+.3f}, {sec['bins_hi']:+.3f}] vif={sec['bins_vif']:.2f}",
          f"| loglin {sec['loglin_visible']:+.3f} vif={sec['loglin_vif']:.1f}",
          f"| matched {sec['matched_delta']:+.3f}")
    # Allocation is the whole mechanism here: equal lengths, so every estimator
    # sees the direct effect and all must agree it is negative.
    assert sec["matched_delta"] < -0.3 and sec["matched_hi"] < 0, sec
    assert sec["bins_identified"] and sec["bins_visible"] < -0.3, sec
    surv = an.survival(ix, ["qasper"])
    far = row(surv, family="generation", cap=256, tier="far", measure="answer_token_recall")
    now = row(surv, family="generation", cap=64, tier="now", measure="answer_token_recall")
    shared = row(surv, family="generation", cap=64, tier="shared", measure="answer_token_recall")
    assert far["conditioned_minus_generic"] < -0.5, far
    assert now["conditioned_minus_generic"] >= 0, now
    assert shared["conditioned_minus_generic"] >= 0, shared
    print(f"B. allocation world: total {total['delta_future']:+.3f}, direct {sec['bins_visible']:+.3f}, "
          f"far survival {far['conditioned_minus_generic']:+.2f} at 256w, "
          f"now {now['conditioned_minus_generic']:+.2f}, shared {shared['conditioned_minus_generic']:+.2f}")

    # --------------------------------------------------------------- plumbing
    holmed = [r["delta_future_p_holm"] for r in prim
              if r["family"] == "generation" and r["metric"] == "f1" and r["cap"] != "mean"]
    raw = [r["delta_future_p"] for r in prim
           if r["family"] == "generation" and r["metric"] == "f1" and r["cap"] != "mean"]
    assert all(h >= p for h, p in zip(holmed, raw)), (holmed, raw)
    import analyse_exp5_followup as fu  # the ratio-of-totals allocation the reports use
    alloc = fu.allocation(ix, ["qasper"])
    assert {r["step"] for r in alloc} == {"64->128", "128->256"}
    assert len(alloc) == 2 * 2 * 2  # 2 families x 2 arms x 2 steps
    inter = an.interaction(ix, ["qasper"])
    audit = an.length_audit(ix, ["qasper"])
    meta = {"writer": "test", "run": "synthetic", "split": "dev", "caps": list(CAPS), "headline": ["qasper"]}
    text = an.report(meta, prim, inter, surv, an.secondary(ix, ["qasper"]), audit)
    for heading in ("## Primary", "## Confirmatory", "## Mechanism", "## Secondary", "## Channel audit"):
        assert heading in text, heading
    print("C. Holm monotone, allocation shape, report sections present")
    print("\nALL ANALYSIS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
