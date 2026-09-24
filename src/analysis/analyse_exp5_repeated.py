"""Analysis of the exp5_cap_only repeated handoff (run_exp5_repeated.py).

For each depth - depth 1 being the frozen main run's messages for the same
documents and caps - and each cap:

- conditioned minus generic future and current F1, paired on document, as in
  analyse_exp5_capped.py, with Holm correction across depths within a cap;
- normalised future retention of each arm and its paired difference, using the
  closed-book and full-source reference answers from the main run (they do not
  depend on the message, so one set serves every depth);
- highlighted-evidence survival by distance tier (analyse_exp5_followup.py);
- delivered length and the share of chains still carrying a valid message.

    python src/analysis/analyse_exp5_repeated.py --config configs/exp5_cap_only_frozen_config.yaml --split main
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import analyse_exp5_capped as base  # noqa: E402
import analyse_exp5_followup as fu  # noqa: E402
from exp5_runs import parent_signature  # noqa: E402
from reuse_common import config, read_rows  # noqa: E402

POLICIES = ("summary_generic", "summary_conditioned")


def depth_data(parent: Path, root: Path, depth: int, writer: str, doc_ids: set, caps: set):
    """Messages and answers for one depth, restricted to the chain's documents and caps."""
    source = parent if depth == 1 else root / f"depth{depth}"
    keep = (lambda r: r["document_id"] in doc_ids and r["block"] == "core"
            and r["policy"] in POLICIES and r["cap"] in caps)
    messages = [m for m in base.load_stage(source, "write", writer, "messages") if keep(m)]
    answers = [a for a in base.load_stage(source, "read", writer, "answers") if keep(a)]
    verdicts = base.load_stage(source, "judge", writer, "verdicts") if (source / "judge").is_dir() else []
    base.attach_verdicts(answers, verdicts)
    return messages, answers


def analyse(ix, refs, corpus: str, depth: int, caps: list[int]) -> tuple[list, list]:
    contrasts, levels = [], []
    for cap in caps:
        for metric in ("f1", "judge_correct"):
            pr = base.paired_rows(ix, corpus, "generation", cap, metric)
            fut, now = base.bootstrap(list(pr["future"].values())), base.bootstrap(list(pr["now"].values()))
            units = {p: {d: u for d, doc in ix.docs.items() if doc["corpus"] == corpus
                         if (u := fu.retention_units(ix, refs, d, p, cap, metric)) is not None} for p in POLICIES}
            ret = fu.paired_ratio_bootstrap(units["summary_conditioned"], units["summary_generic"])
            contrasts.append({"corpus": corpus, "depth": depth, "cap": cap, "metric": metric,
                              "delta_future": fut["estimate"], "delta_future_lo": fut["lo"],
                              "delta_future_hi": fut["hi"], "delta_future_p": fut["p"],
                              "delta_now": now["estimate"], "delta_now_lo": now["lo"], "delta_now_hi": now["hi"],
                              "retention_delta": ret["estimate"], "retention_delta_lo": ret["lo"],
                              "retention_delta_hi": ret["hi"], "documents": fut["n"]})
            for p in POLICIES:
                b = fu.ratio_bootstrap({d: (u[0] - u[1], u[2] - u[1]) for d, u in units[p].items()})
                levels.append({"corpus": corpus, "depth": depth, "cap": cap, "metric": metric, "policy": p,
                               "U_future": base.level(ix, corpus, p, cap, metric, True),
                               "U_now": base.level(ix, corpus, p, cap, metric, False) if p.endswith("conditioned") else None,
                               "retention": b["estimate"], "retention_lo": b["lo"], "retention_hi": b["hi"]})
    return contrasts, levels


def lengths(messages: list[dict], expected: dict, depth: int) -> list[dict]:
    rows = []
    for (policy, cap), n in sorted(expected.items()):
        ms = [m for m in messages if m["policy"] == policy and m["cap"] == cap]
        valid = [m for m in ms if m["valid"] and m["text"]]
        rows.append({"depth": depth, "policy": policy, "cap": cap, "chains": n,
                     "valid_share": len(valid) / n if n else None,
                     "words_median": st.median(m["delivered_words"] for m in valid) if valid else None,
                     "truncated": sum(m.get("truncated_words", 0) > 0 for m in valid)})
    return rows


def figure(out: Path, corpus: str, contrasts, levels, surv, lens, caps) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(len(caps), 4, figsize=(15, 3.4 * len(caps)), facecolor=fu.SURFACE, squeeze=False)
    for i, cap in enumerate(caps):
        lv = [r for r in levels if r["cap"] == cap and r["metric"] == "f1" and r["corpus"] == corpus]
        for p in POLICIES:
            rows = sorted((r for r in lv if r["policy"] == p), key=lambda r: r["depth"])
            generic = p.endswith("generic")
            fu._line(axes[i][0], [r["depth"] for r in rows], [r["retention"] for r in rows],
                     [r["retention_lo"] for r in rows], [r["retention_hi"] for r in rows],
                     "generation", generic, "generic" if generic else "conditioned")
        ct = sorted((r for r in contrasts if r["cap"] == cap and r["metric"] == "f1" and r["corpus"] == corpus),
                    key=lambda r: r["depth"])
        fu._line(axes[i][1], [r["depth"] for r in ct], [r["delta_future"] for r in ct],
                 [r["delta_future_lo"] for r in ct], [r["delta_future_hi"] for r in ct], "generation", False,
                 "future F1, conditioned - generic")
        axes[i][1].axhline(0, color=fu.INK_2, lw=0.8)
        for tier, generic_style in (("far", None),):
            sv = sorted((r for r in surv if r["cap"] == cap and r["tier"] == tier and r["family"] == "generation"),
                        key=lambda r: r["depth"])
            fu._line(axes[i][2], [r["depth"] for r in sv], [r["recall_generic"] for r in sv], [None] * len(sv),
                     [None] * len(sv), "generation", True, "generic")
            fu._line(axes[i][2], [r["depth"] for r in sv], [r["recall_conditioned"] for r in sv], [None] * len(sv),
                     [None] * len(sv), "generation", False, "conditioned")
        for p in POLICIES:
            rows = sorted((r for r in lens if r["cap"] == cap and r["policy"] == p), key=lambda r: r["depth"])
            fu._line(axes[i][3], [r["depth"] for r in rows], [r["words_median"] for r in rows], [None] * len(rows),
                     [None] * len(rows), "generation", p.endswith("generic"),
                     "generic" if p.endswith("generic") else "conditioned")
        titles = (f"cap {cap}: normalised future retention", f"cap {cap}: future F1, cond - gen",
                  f"cap {cap}: far highlighted evidence kept", f"cap {cap}: delivered words (median)")
        for ax, title in zip(axes[i], titles):
            fu._style(ax, title, "handoff depth")
            ax.set_xticks(sorted({r["depth"] for r in lv}))
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "repeated.png", dpi=150, facecolor=fu.SURFACE)
    plt.close(fig)
    return "repeated.png"


def fmt(x, nd=3, signed=True):
    return "—" if x is None else (f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}")


def report(meta, contrasts, levels, surv, lens, fig) -> str:
    lines = [f"# exp5_cap_only repeated handoff — {meta['writer']}", "",
             f"Run `{meta['run']}` (parent `{meta['parent_run']}`), {len(meta['document_ids'])} documents, caps "
             f"{meta['caps']}, depths {meta['depths']}. Depth 1 is the main run's message; each later depth "
             "rewrites the previous message under the same cap, visibility and conditioning question. "
             "Intervals are 95% document-bootstrap intervals; p values are Holm-corrected across depths within a cap.", "",
             "## Conditioned minus generic, by depth (F1)", "",
             "| cap | depth | Δ future [95% CI] | p (Holm) | Δ now | Δ retention [95% CI] | docs |",
             "|---:|---:|---|---:|---:|---|---:|"]
    for r in contrasts:
        if r["metric"] != "f1":
            continue
        lines.append(f"| {r['cap']} | {r['depth']} | {fmt(r['delta_future'])} [{fmt(r['delta_future_lo'])}, "
                     f"{fmt(r['delta_future_hi'])}] | {fmt(r.get('delta_future_p_holm'), 4, False)} | "
                     f"{fmt(r['delta_now'])} | {fmt(r['retention_delta'])} [{fmt(r['retention_delta_lo'])}, "
                     f"{fmt(r['retention_delta_hi'])}] | {r['documents']} |")
    lines += ["", "## Levels by depth (F1)", "",
              "| cap | depth | U_future generic | U_future conditioned | U_now conditioned | retention generic | "
              "retention conditioned |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for cap in meta["caps"]:
        for depth in meta["depths"]:
            lv = {r["policy"]: r for r in levels if r["cap"] == cap and r["depth"] == depth and r["metric"] == "f1"}
            if not lv:
                continue
            g, c = lv.get("summary_generic", {}), lv.get("summary_conditioned", {})
            lines.append(f"| {cap} | {depth} | {fmt(g.get('U_future'), 3, False)} | {fmt(c.get('U_future'), 3, False)} | "
                         f"{fmt(c.get('U_now'), 3, False)} | {fmt(g.get('retention'), 3, False)} | "
                         f"{fmt(c.get('retention'), 3, False)} |")
    lines += ["", "## Highlighted evidence kept, by depth", "",
              "| cap | depth | tier | generic | conditioned | cond − gen [95% CI] |", "|---:|---:|---|---:|---:|---|"]
    for r in surv:
        if r["family"] == "generation":
            lines.append(f"| {r['cap']} | {r['depth']} | {r['tier']} | {fmt(r['recall_generic'], 3, False)} | "
                         f"{fmt(r['recall_conditioned'], 3, False)} | {fmt(r['conditioned_minus_generic'])} "
                         f"[{fmt(r['lo'])}, {fmt(r['hi'])}] |")
    lines += ["", "## Chain health", "", "| depth | policy | cap | chains | valid | words (median) | truncated |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for r in lens:
        lines.append(f"| {r['depth']} | {r['policy']} | {r['cap']} | {r['chains']} | {fmt(r['valid_share'], 2, False)} | "
                     f"{fmt(r['words_median'], 0, False)} | {r['truncated']} |")
    if fig:
        lines += ["", f"![{fig}]({fig})"]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp5_cap_only_frozen_config.yaml")
    parser.add_argument("--split", default="main")
    parser.add_argument("--writer", default="mistral24")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--run-dir", default="",
                        help="parent main run; defaults to the one the current code's protocol hash names")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = config(str(ROOT / args.config))
    if args.fake:  # mirror run_exp5_capped.py: both roots enter the protocol hash
        cfg["run_root"] = str(Path(cfg["run_root"]) / "fake")
        cfg["cache_root"] = str(Path(cfg["cache_root"]) / "fake")
    parent = (Path(args.run_dir) if args.run_dir
              else ROOT / cfg["run_root"] / args.split / parent_signature(cfg)[:12])
    root = parent.with_name(f"{parent.name}-repeated")
    meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    doc_ids, caps = set(meta["document_ids"]), sorted(meta["caps"])
    docs = [d for d in read_rows(ROOT / cfg["data_root"] / f"{args.split}.jsonl") if d["id"] in doc_ids]
    corpora = sorted({d["corpus"] for d in docs})
    refs = fu.references(base.load_stage(parent, "read", args.writer, "answers"))
    depth1, _ = depth_data(parent, root, 1, args.writer, doc_ids, set(caps))
    expected = {}
    for m in depth1:
        expected[(m["policy"], m["cap"])] = expected.get((m["policy"], m["cap"]), 0) + 1
    contrasts, levels, surv, lens = [], [], [], []
    for depth in meta["depths"]:
        messages, answers = depth_data(parent, root, depth, args.writer, doc_ids, set(caps))
        ix = base.Index(docs, messages, answers)
        for corpus in corpora:
            c, lv = analyse(ix, refs, corpus, depth, caps)
            contrasts += c
            levels += lv
        surv += [{**r, "depth": depth} for r in fu.evidence_survival(ix, corpora) if r["family"] == "generation"]
        lens += lengths(messages, expected, depth)
    for key in {(r["corpus"], r["cap"], r["metric"]) for r in contrasts}:
        block = sorted((r for r in contrasts if (r["corpus"], r["cap"], r["metric"]) == key), key=lambda r: r["depth"])
        for row, adj in zip(block, base.holm([r["delta_future_p"] for r in block])):
            row["delta_future_p_holm"] = adj
    out = Path(args.out) if args.out else ROOT / cfg["results_root"] / args.split / root.name
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("depth_contrasts", contrasts), ("depth_levels", levels),
                       ("depth_evidence", surv), ("chain_health", lens)):
        base.write_csv(out / f"{name}.csv", rows)
    fig = figure(out, corpora[0], contrasts, levels, surv, lens, caps) if corpora else None
    meta_out = {**{k: meta[k] for k in ("parent_run", "caps", "depths", "document_ids", "writer")}, "run": root.name}
    (out / "report.md").write_text(report(meta_out, contrasts, levels, surv, lens, fig), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"repeated-handoff analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
