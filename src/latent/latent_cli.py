"""Gated execution, exact manifests and restart-safe output for Experiment 13."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import argparse
import json
from pathlib import Path
import time
from dataclasses import replace
import latent_backend as lb
import latent_handoff as lh
import latent_metrics as lm
from latent_manifest import code_identity, environment, verify_gate
from latent_store import RunStore


def main(argv=None):
    import run_latent_reusability as run
    from llm import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/latent_reusability_config.yaml")
    parser.add_argument("--panel", choices=("A","B"), default="A")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="development packet limit")
    parser.add_argument("--channels", default="")
    parser.add_argument("--controls", default=None, help="comma-separated; 'none' disables")
    parser.add_argument("--positions", type=int)
    parser.add_argument("--sweep", action="store_true", help="gated development caps from config")
    parser.add_argument("--gpu-gates", default="runs/latent_gpu_smoke/gates.json")
    parser.add_argument("--panel-a-gates")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--benchmark", action="store_true", help="disable result-cache reuse for timing")
    args = parser.parse_args(argv)
    cfg = load_config(run.ROOT / args.config)
    data = cfg["data"]
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.panel == "A":
        dossiers = run.fqa.load_fictional_dossiers(run.ROOT / data["corpus"])
        packets = run.build_packets(dossiers, run.read_jsonl(run.ROOT/data["selections"]), data["k"])
        questions = 6
    else:
        from latent_data import full_source_panel
        dossiers, packets, _ = full_source_panel(run.ROOT / data["relation_corpus"])
        questions = 4
    if args.limit:
        packets = packets[:args.limit]
    if not packets:
        parser.error("no valid source packets")
    channels = args.channels.split(",") if args.channels else cfg["channels"]
    controls = cfg["controls"] if args.controls is None else ([] if args.controls == "none" else args.controls.split(","))
    if not set(channels) <= set(run.TEXT_CHANNELS+run.LATENT_CHANNELS) or not set(controls) <= set(run.CONTROLS):
        parser.error("unknown channel or control")
    caps = cfg["budget"]["sweep"] if args.sweep else [args.positions or cfg["budget"]["positions"]]
    if args.positions is not None and args.positions <= 0:
        parser.error("position cap must be positive")
    n_infra = data.get("infrastructure_dossiers",4)
    if args.dry_run:
        result = run.dry_run_report(packets, channels, controls, questions, cfg, n_infra)
        result.update(panel=args.panel, caps=caps, simulated=args.fake,
                      total_evaluations_all_caps=result["total_answer_evaluations"]*len(caps),
                      weights_downloaded=False, stage="development")
        print(json.dumps(result,indent=2))
        return 0
    # Validate all gate identities before weights are loaded or an output is opened.
    probe = run.build_backend(cfg, True)
    backend_record = probe.config.record()
    parity = {"passed":True,"simulated":True} if args.fake else None
    if not args.fake:
        gate = verify_gate(run.ROOT/args.gpu_gates, backend_record)
        parity = gate["wrapper_parity"]
        if args.panel == "B" or args.sweep:
            if not args.panel_a_gates:
                parser.error("Panel B and sweeps require --panel-a-gates")
            verify_gate(run.ROOT/args.panel_a_gates, backend_record, "panel_a")
    seconds = args.max_seconds or cfg["compute"]["diagnostic_gpu_hours_max"]*3600
    if seconds <= 0 or seconds > cfg["compute"]["diagnostic_gpu_hours_max"]*3600:
        parser.error("--max-seconds must be within the configured diagnostic envelope")
    backend = probe if args.fake else run.build_backend(cfg, False)
    if args.fake:
        run.prime_fake(backend,dossiers)
    deadline = time.monotonic()+seconds
    backend.deadline = deadline
    base = run.ROOT / (args.out or f"runs/latent_reusability/{'fake' if args.fake else 'real'}/panel_{args.panel.lower()}")
    exit_code = 0
    for cap in caps:
        band = lm.PositionBand(cap, cfg["budget"]["fill_floor_ratio"])
        manifest = {"schema_version":run.SCHEMA_VERSION, "panel":args.panel,
            "stage":"development", "simulated":args.fake, "backend":backend_record,
            "code_identity":code_identity(), "environment":environment(),
            "source_hash":lh.canonical_hash([(p.record(),p.packet_text,p.announced_question) for p in packets]),
            "gold_hash":lh.canonical_hash([(d.dossier_id,[(c.fact_id,c.question,c.golds) for c in run.cards_for(d)]) for d in dossiers]),
            "band":run.as_band(band), "channels":channels, "controls":controls,
            "codec":cfg["codec"], "max_attempts":cfg["budget"]["max_attempts"],
            "infrastructure_dossiers":n_infra,"benchmark":args.benchmark}
        store = RunStore(base / f"cap_{cap}", manifest,
                         max(0,deadline-time.monotonic()),args.benchmark)
        try:
            _, answers = run.run_panel_a(backend,packets,dossiers,channels,band,data["k"],
                cfg["codec"]["dtype"],controls,n_infra,store=store,
                max_attempts=cfg["budget"]["max_attempts"])
            gates = run.development_gates(answers,{**cfg,"wrapper_parity":parity})
            status = {"backend":backend_record,"code_identity":code_identity(),
                      "environment":environment(),"simulated":args.fake,"panel":args.panel,
                      "n_sources":len({p.dossier_id for p in packets}), "complete":True,
                      "passed":gates["all_gates_pass"] and not args.fake,"gates":gates}
            store._json(store.path/"gates.json",status)
            print(json.dumps({"out":str(store.path),"simulated":args.fake,"gates":gates},indent=2))
            if not args.fake and not gates["all_gates_pass"]:
                exit_code = 1
        except TimeoutError as error:
            store._json(store.path/"status.json",{"complete":False,"reason":str(error)})
            print(str(error))
            exit_code = 2
        finally:
            store.export()
        if exit_code:
            break
    return exit_code
