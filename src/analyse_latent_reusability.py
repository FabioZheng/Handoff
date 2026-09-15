"""Source-level utility, separate resource plots and prespecified comparisons.

Reads saved answers only. Never calls a model, and never treats simulator
outputs or development contrasts as confirmatory evidence.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import latent_metrics as lm


def load(path):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (path/"answers.jsonl").read_text(encoding="utf-8").splitlines() if s]
    return manifest, rows


def source_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dossier_id"]].append(row[field])
    return float(np.mean([np.mean(x) for x in groups.values()])) if groups else None


def summarize(manifest, rows):
    groups = defaultdict(list)
    for r in rows:
        endpoint = r.get("relation", "unlabelled")
        if manifest["panel"] == "A":
            endpoint = "current" if r["is_announced"] else "other_questions"
        groups[(r["arm"],r["mode"],endpoint)].append(r)
    result=[]
    for (arm,mode,endpoint), group in sorted(groups.items()):
        # Encoding cost is per unique source-message; each row's sum is a
        # one-read scenario, not the physical total of this cached experiment.
        for row in group:
            row["one_read_s"] = row["encode_s"] + row["read_s"]
            row["failed"] = bool(row.get("error")) or (arm in manifest["channels"] and not row["band_ok"])
        result.append({"arm":arm,"mode":mode,"endpoint":endpoint,
            "n_rows":len(group),"n_sources":len({r["dossier_id"] for r in group}),
            "em":source_mean(group,"system_em"),"f1":source_mean(group,"f1"),
            "added_positions":source_mean(group,"added_positions"),
            "wire_bytes":source_mean(group,"wire_bytes"),
            "one_read_s":source_mean(group,"one_read_s"),
            "failure_fraction":source_mean(group,"failed")})
    return result


def paired_comparison(treatment, control, treatment_arm, control_arm, n_resamples=10000):
    """Strict source/rotation/question pairing; no inner-join disappearance."""
    tm,tr = treatment; cm,cr = control
    if tm["panel"] != "B" or cm["panel"] != "B":
        raise ValueError("current/orthogonal comparison requires Panel B labels")
    if tm["source_hash"] != cm["source_hash"] or tm["gold_hash"] != cm["gold_hash"]:
        raise ValueError("comparison sources and golds differ")
    if tm["backend"] != cm["backend"] or tm["environment"] != cm["environment"]:
        raise ValueError("comparison model/runtime differs")
    def index(rows,arm,endpoint):
        selected = [r for r in rows if r["arm"] == arm and r["mode"] == "conditioned" and r["relation"] == endpoint]
        out = {(r["dossier_id"],r["rotation_id"],r["fact_id"]):r for r in selected}
        if len(out) != len(selected):
            raise ValueError("duplicate evaluation keys")
        return out
    contrasts = {}
    endpoints = {}
    for endpoint in ("current","orthogonal"):
        a,b = index(tr,treatment_arm,endpoint),index(cr,control_arm,endpoint)
        if not a or set(a) != set(b):
            raise ValueError("missing paired rows; do not drop failed or absent questions")
        keys = sorted(a)
        src = [k[0] for k in keys]
        av,bv = [a[k]["system_em"] for k in keys],[b[k]["system_em"] for k in keys]
        contrasts[endpoint] = lm.cluster_bootstrap_delta(src,av,bv,n_resamples,ci=0.975)
        endpoints[endpoint] = (keys,a,b)
    keys,a,b = endpoints["orthogonal"]
    direct_t,direct_c = index(tr,"direct_packet","orthogonal"),index(cr,"direct_packet","orthogonal")
    if not set(keys) <= set(direct_t) or not set(keys) <= set(direct_c):
        raise ValueError("direct-source rows are required for useful-future-utility floors")
    floors=[]
    for candidate,direct in ((a,direct_t),(b,direct_c)):
        floors.append(lm.cluster_bootstrap_delta([k[0] for k in keys],
            [candidate[k]["system_em"] for k in keys],
            [0.5*direct[k]["system_em"] for k in keys],n_resamples,ci=0.975))
    ar=[r for r in tr if r["arm"]==treatment_arm and r["mode"]=="conditioned"]
    br=[r for r in cr if r["arm"]==control_arm and r["mode"]=="conditioned"]
    saving=lm.position_saving([r["added_positions"] for r in ar],[r["added_positions"] for r in br])
    caps_ok=all(r["added_positions"] <= r["band_target"] for r in ar+br)
    result=lm.efficiency_claim(saving,contrasts["current"],contrasts["orthogonal"],
                             treatment_floor=floors[0],control_floor=floors[1],hard_caps_ok=caps_ok)
    result["numerical_gate_passed"]=result["claim_supported"]
    result["stage"]=tm["stage"]
    result["simulated"]=tm["simulated"] or cm["simulated"]
    # This implementation runs development panels only. Confirmation requires
    # an independently frozen protocol/test manifest and sample-size decision.
    result["claim_supported"]=False
    result["interpretation"]="Exploratory development contrast; no confirmatory claim."
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--compare")
    p.add_argument("--treatment-arm",default="latent_sb")
    p.add_argument("--control-arm",default="text_record")
    args=p.parse_args()
    root=Path(args.run)
    manifest,rows=load(root)
    summary=summarize(manifest,rows)
    if not summary:
        raise ValueError("no answer records")
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,field,label in zip(axes,("added_positions","wire_bytes","one_read_s"),
                              ("Added receiver positions","Serialized wire bytes","One-read inference seconds")):
        for row in summary:
            if row["arm"] not in manifest["channels"] or row["mode"]!="conditioned":
                continue
            ax.scatter(row[field],row["em"],label=f"{row['arm']} / {row['endpoint']}")
        ax.set_xlabel(label);ax.set_ylim(-0.02,1.02);ax.grid(alpha=.2)
    axes[0].set_ylabel("Source-averaged exact match")
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=7)
    fig.suptitle(f"Panel {manifest['panel']} - {'SIMULATOR: NOT RESEARCH RESULTS' if manifest['simulated'] else 'development results'}")
    fig.tight_layout(rect=(0,.18,1,.94)); fig.savefig(root/"resources.png",dpi=180);plt.close(fig)
    if args.compare:
        result=paired_comparison((manifest,rows),load(args.compare),args.treatment_arm,args.control_arm)
        (root/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(str(root/"summary.csv"))


if __name__=="__main__":
    main()
