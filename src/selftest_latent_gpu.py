"""GPU-side checks for Experiment 13. Needs weights and an accelerator.

Run this *before* any batch. It answers three engineering questions, in the
order in which a failure would invalidate everything downstream:

1. **Does the continuous-input wrapper work?** Feed the model a sequence as
   token ids, then feed it the input embeddings of those same ids. The
   next-token distributions must agree to numerical tolerance. If they do not,
   the wrapper is broken -- masks, positions, template or slicing -- and every
   latent result is uninterpretable. This is not a test of latent
   communication; it is a test of the plumbing beneath it.

2. **Can this reader read at all, and does it already know the answers?**
   Direct-source accuracy and a closed-book probe on the fictional corpus. The
   old Llama closed-book results do not transfer to a new reader.

3. **Does the reader actually use a delivered payload?** A real prefix versus a
   zeroed one versus another dossier's, under distinct opaque ids so a cache
   cannot restore the original's answer.

A failure here triggers a bounded implementation investigation. It does not
license switching datasets or widening the grid.

    python src/selftest_latent_gpu.py --items 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fictional_qa as fqa  # noqa: E402
import latent_backend as lb  # noqa: E402
import latent_handoff as lh  # noqa: E402
import run_latent_reusability as run  # noqa: E402
from llm import load_config  # noqa: E402
from score import extract_short_answer, score_against_golds  # noqa: E402


def report(label: str, ok: bool | None, detail: str = "") -> bool:
    mark = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"{mark}  {label}" + (f" -- {detail}" if detail else ""))
    return ok is not False


def check_wrapper_parity(backend, text, atol):
    result = backend.wrapper_parity(text, "What value is in the evidence?", atol)
    return report("production continuous-input wrapper", result["passed"], json.dumps(result))


def main(argv=None):
    from latent_manifest import code_identity, environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="latent_reusability_config.yaml")
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--out", default="runs/latent_gpu_smoke/gates.json")
    args = parser.parse_args(argv)
    if args.items < 2:
        parser.error("at least two dossiers are required for cross-dossier controls")
    cfg = load_config(ROOT / args.config)
    backend = run.build_backend(cfg, False)
    dossiers = fqa.load_fictional_dossiers(ROOT / cfg["data"]["corpus"])[:args.items]
    output = {"backend":backend.config.record(), "code_identity":code_identity(),
              "environment":environment(), "simulated":False, "passed":False}
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    # Save even a failed/partial gate; it cannot authorize a batch.
    try:
        probe = fqa.evidence_cards(dossiers[0])[0].evidence_text
        parity = backend.wrapper_parity(probe, "What value is in the evidence?", args.atol)
        output["wrapper_parity"] = parity
        if not parity["passed"]:
            print(json.dumps(parity, indent=2))
            return 1
        direct, closed, own_scores, zero, shuffled = [], [], [], [], []
        prefixes = {}
        for d in dossiers:
            cards = fqa.evidence_cards(d)
            source = "\n".join(c.evidence_text for c in cards)
            h = backend.transcript_states(f"{run.TRANSCRIPT_INSTRUCTION}\n\nEVIDENCE:\n{source}",
                                          k=128, system=run.WRITER_SYSTEM)
            prefixes[d.dossier_id] = backend.align(h.states, h.reference_embeddings)
        common_k = min(len(v) for v in prefixes.values())
        prefixes = {key:value[-common_k:] for key,value in prefixes.items()} if common_k else prefixes
        output["actual_common_payload_positions"] = common_k
        if not common_k:
            output["error"] = "no valid generated message states"
            return 1
        for i,d in enumerate(dossiers):
            cards = fqa.evidence_cards(d)
            source = "\n".join(c.evidence_text for c in cards)
            own = prefixes[d.dossier_id]
            candidate = prefixes[dossiers[(i+1)%len(dossiers)].dossier_id]
            # Keep equal physical lengths; short donors make the gate unavailable.
            if len(candidate) < len(own):
                output["error"] = "shuffled donor shorter than own high-capacity prefix"
                return 1
            for card in cards:
                def score(material, schema=""):
                    got = backend.read_message(material, card.question, schema).text
                    return score_against_golds(extract_short_answer(got), list(card.golds))[0]
                direct.append(score(source))
                closed.append(score(""))
                for material,bucket in ((own,own_scores),(np.zeros_like(own),zero),(candidate[:len(own)],shuffled)):
                    wire = lh.to_wire(lh.seal_vectors(lh.opaque_id(d.dossier_id,card.fact_id),
                           card.question,"latent_sb",material,dtype=cfg["codec"]["dtype"]))
                    bucket.append(score(lh.from_wire(wire).as_array(),run.LATENT_SCHEMA))
        metrics = {k:float(np.mean(v)) for k,v in {"direct":direct,"closed":closed,"latent":own_scores,"zero":zero,"shuffled":shuffled}.items()}
        g = cfg["gates"]
        checks = {"direct":metrics["direct"] >= g["direct_source_em_min"],
                  "closed":metrics["closed"] <= g["closed_book_em_max"],
                  "latent_gap":metrics["direct"]-metrics["latent"] <= g["latent_gap_max"],
                  "sensitivity":metrics["latent"]-max(metrics["zero"],metrics["shuffled"]) >= g["shuffle_sensitivity_min"]}
        output.update(metrics=metrics, checks=checks, passed=all(checks.values()), n_sources=len(dossiers),
                      high_capacity_payload_positions=128)
        print(json.dumps(output,indent=2))
        return 0 if output["passed"] else 1
    finally:
        target.write_text(json.dumps(output,indent=2),encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
