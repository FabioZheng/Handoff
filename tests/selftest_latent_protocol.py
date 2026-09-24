"""Regression tests for the v2 scientific protocol and durable runner."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import latent_backend as lb
import latent_handoff as lh
import latent_metrics as lm
import run_latent_reusability as run
from latent_cli import main
from latent_data import full_source_panel
from latent_store import RunStore
from latent_interventions import prepare_pair


def test():
    checks=[]
    def check(label,ok):
        assert ok,label
        checks.append(label);print("PASS",label)
    dossiers=run.fqa.load_fictional_dossiers()
    packets=run.build_packets(dossiers,run.read_jsonl(run.ROOT/"data/latent_reusability/fictional_summary_generalization_selections_n20.jsonl"),4)
    chosen=[next(p for p in packets if p.dossier_id==d.dossier_id) for d in dossiers[:2]]
    backend=lb.FakeBackend();run.prime_fake(backend,dossiers)
    band=lm.PositionBand(32)
    encoded=run.encode(backend,"latent_sb",chosen[0],band,k=4)
    check("vector capacity comes from total cap, never evidence-card k",encoded.positions.added<=32 and len(encoded.vectors)>4)
    check("schema is separately charged",encoded.positions.schema==backend.count_positions(run.LATENT_SCHEMA+"\n"))
    controlset=("direct_packet","closed_book","source_free","zero_prefix","shuffled_prefix","token_replay","embed_replay")
    with tempfile.TemporaryDirectory(prefix="latent_protocol_") as temp:
        manifest={"test":"protocol-v2"}
        store=RunStore(temp,manifest)
        backend.call_log.clear()
        messages,answers=run.run_panel_a(backend,chosen,dossiers,run.CHANNELS,band,4,"bfloat16",controlset,store=store)
        check("one H generated per packet, reused across controls",backend.call_log.count("transcript")==len(chosen))
        srcfree=[r for r in answers if r["arm"]=="source_free"]
        check("source-free control uses exact teacher-forced H",all(r["control_diagnostics"]["fixed_transcript"] for r in srcfree) and backend.call_log.count("teacher_force")==len(chosen))
        shuf=[r for r in answers if r["arm"]=="shuffled_prefix"]
        check("shuffled donors are from different dossiers",all(r["control_diagnostics"].get("donor_dossier_id")!=r["dossier_id"] and not r["error"] for r in shuf))
        check("invalid messages retained as zero-scored system failures",all(r["system_em"]==0 for r in answers if r["arm"] in run.CHANNELS and not r["band_ok"]))
        check("exact wire byte counts retained",all(r["wire_bytes"]>r["payload_bytes"] for r in answers))
        store.export()
        count=len((Path(temp)/"answers.jsonl").read_text().splitlines())
        again=RunStore(temp,manifest)
        run.run_panel_a(backend,chosen,dossiers,run.CHANNELS,band,4,"bfloat16",controlset,store=again)
        again.export()
        check("resume does not duplicate evaluation records",len((Path(temp)/"answers.jsonl").read_text().splitlines())==count)
        check("resume reuses core encodings and reader requests",again.counts["encoding_calls"]==0 and again.counts["reader_calls"]==0)
        try:
            RunStore(temp,{"different":True})
        except ValueError:
            check("mismatched manifests cannot overwrite a run",True)
        else:
            raise AssertionError("manifest mismatch accepted")
        gate=run.development_gates(answers,{})
        check("answer agreement cannot substitute for numerical wrapper parity",gate["gate_wrapper_parity"] is None and not gate["all_gates_pass"])
    ds,ps,_=full_source_panel(run.ROOT/"data/communication_regret/relation_dossiers.jsonl")
    check("Panel B has four rotations and two modes per source",len(ps)==16*4*2)
    check("Panel B sends full identical evidence under both policies",all(p.packet_text==next(d.document for d in ds if d.dossier_id==p.dossier_id) for p in ps))
    check("Panel B evaluates current, paraphrase, nearby and orthogonal",all(set(dict(p.relation_labels).values())=={"current","paraphrase","nearby","orthogonal"} for p in ps))
    check("generic writers cannot see evaluation-current query",all(not p.announced_question for p in ps if p.mode=="generic"))
    with contextlib.redirect_stdout(io.StringIO()), patch.object(Path,"write_text",side_effect=AssertionError("dry run wrote")), patch.object(Path,"mkdir",side_effect=AssertionError("dry run made directory")), patch.object(run,"build_backend",side_effect=AssertionError("dry run constructed backend")):
        main(["--dry-run"]);main(["--panel","B","--dry-run","--sweep"])
    check("actual CLI dry runs construct no backend and write nothing",True)
    wire=lh.to_wire(lh.seal_vectors("opaque","question","latent_sb",np.ones((3,8))))
    try:
        lh.from_wire(wire[:-1]+bytes([wire[-1]^1]))
    except lh.PayloadError:
        check("wire tampering fails checksum validation",True)
    else:
        raise AssertionError("tampered payload accepted")
    base=lb.BackendConfig()
    from dataclasses import replace
    check("alignment parameters enter cache identity",base.fingerprint()!=replace(base,alignment_snap_ratio=0.4).fingerprint())
    h=replace(encoded.transcript,text="fixed visible notes",token_ids=())
    pair=prepare_pair(backend,chosen[0],run.cards_for(dossiers[0]),h)
    check("factual swap/sham can be selected without changing prompt length",pair["available"] and pair["original_prompt_positions"]==pair["swap_prompt_positions"]==pair["sham_prompt_positions"])
    check("intervention changes only source material, holds H fixed",pair["changed_gold"] not in h.text)
    print(f"All {len(checks)} protocol checks passed.")


if __name__=="__main__":
    test()
