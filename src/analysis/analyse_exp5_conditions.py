"""When does the reuse loss appear? Conditions for the exp5_cap_only main run.

Boundary corpus (SQuAD)
    SQuAD was dropped from the headline study at calibration because even generic
    messages kept almost nothing for held-out questions. This part describes why,
    from the calibration decision and the data alone: document length, question
    count, answer length, and how far apart the questions' evidence sits.

Moderators (QASPER)
    For every query-aware message and every held-out question q != anchor, the
    difference in the reader's score on q between the query-aware message and the
    generic message at the same cap. Rows are grouped by
      - document length (tertiles of source words),
      - evidence spread (tertiles of distinct paragraphs cited as evidence by any
        of the document's questions),
      - questions per document,
      - the held-out question's answer type,
      - the held-out question's distance from the anchor (shared, near, far, as in
        analyse_exp5_capped.py).
    Each document gets equal weight within a group: rows are averaged per document
    first, then documents are resampled (95% bootstrap). The contrast between the
    first and last group of each moderator is computed on the same resamples.

These are descriptive splits of one experiment, not randomised comparisons: the
moderators are properties of the documents and questions, and several of them
travel together.

    python src/analysis/analyse_exp5_conditions.py --config configs/exp5_cap_only_frozen_config.yaml --split main
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import analyse_exp5_capped as base  # noqa: E402
import analyse_exp5_followup as fu  # noqa: E402
from reuse_common import config, read_rows  # noqa: E402

BOOTSTRAP = base.BOOTSTRAP
TIER_ORDER = ("shared", "near", "far")


def as_list(value):
    """Some corpora store list fields as their repr."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return value if value is not None else []


def answer_type(q: dict) -> str:
    kinds = as_list(q.get("answer_type")) or ["none"]
    return kinds[0] if isinstance(kinds, list) else str(kinds)


def evidence_paragraphs(doc: dict) -> int:
    return len({p for q in doc["questions"] for es in as_list(q.get("evidence_sets")) for p in as_list(es)})


def normalise(doc: dict) -> dict:
    qs = [{**q, "evidence_sets": [as_list(es) for es in as_list(q.get("evidence_sets"))]} for q in doc["questions"]]
    return {**doc, "questions": qs}


# ------------------------------------------------------------------ boundary

def boundary(docs: list[dict], calibration: dict) -> list[dict]:
    rows = []
    for corpus in sorted({d["corpus"] for d in docs}):
        ds = [normalise(d) for d in docs if d["corpus"] == corpus]
        tiers = defaultdict(int)
        for d in ds:
            for a in d["questions"]:
                for q in d["questions"]:
                    if q["qid"] != a["qid"]:
                        tiers[base.distance(d, a, q) or "unknown"] += 1
        pairs = sum(tiers.values())
        golds = [len(str(g).split()) for d in ds for q in d["questions"] for g in as_list(q.get("golds"))[:1]]
        table = calibration["table"].get(corpus, {})
        region = calibration["decision"]["regions"].get(corpus, {})
        best = max((r["R"] for r in table.get("rows", [])), default=None)
        rows.append({"corpus": corpus, "documents": len(ds),
                     "source_words_median": st.median(len(d["source"].split()) for d in ds),
                     "paragraphs_median": st.median(len(d["paragraphs"]) for d in ds),
                     "questions_per_doc_median": st.median(len(d["questions"]) for d in ds),
                     "answer_words_median": st.median(golds) if golds else None,
                     **{f"pairs_{t}": tiers.get(t, 0) / pairs if pairs else None for t in TIER_ORDER + ("unknown",)},
                     "U_closed_book": table.get("U_closed_book"), "U_full_source": table.get("U_full_source"),
                     "natural_generic_words": table.get("natural_generic_median"),
                     "natural_conditioned_words": table.get("natural_conditioned_median"),
                     "best_generic_retention": best, "useful_region": region.get("has_region")})
    return rows


# ---------------------------------------------------------------- moderators

def pair_rows(ix: base.Index, corpus: str, family: str, metric: str = "f1") -> list[dict]:
    """One row per (query-aware message, held-out question): cond - generic score on that question."""
    cond, gen = base.FAMILIES[family]
    rows = []
    for doc_id, doc in ix.docs.items():
        if doc["corpus"] != corpus:
            continue
        questions = ix.questions[doc_id]
        for cap in ix.caps():
            g_scores = ix.score.get((doc_id, gen, cap, None), {})
            for m in ix.by_cell.get((doc_id, cond, cap), []):
                a = m["anchor"]
                c_scores = ix.score.get((doc_id, cond, cap, a), {})
                for qid, q in questions.items():
                    if qid == a or qid not in c_scores or qid not in g_scores:
                        continue
                    c, g = c_scores[qid].get(metric), g_scores[qid].get(metric)
                    if c is None or g is None:
                        continue
                    rows.append({"doc": doc_id, "cap": cap, "anchor": a, "qid": qid, "delta": c - g,
                                 "tier": base.distance(doc, questions[a], q) if a in questions else None,
                                 "answer_type": answer_type(q)})
    return rows


def tertiles(values: dict) -> dict:
    """doc -> value  =>  doc -> label of its tertile, labelled by the value range."""
    cuts = np.quantile(list(values.values()), [1 / 3, 2 / 3])
    groups = {d: int(np.searchsorted(cuts, v, side="right")) for d, v in values.items()}
    labels = {}
    for g in range(3):
        vs = [values[d] for d in values if groups[d] == g]
        labels[g] = f"{g + 1}: {min(vs):.0f}-{max(vs):.0f}" if vs else f"{g + 1}"
    return {d: labels[g] for d, g in groups.items()}


def grouped(rows: list[dict], key, order: list[str] | None, seed: int = 0) -> list[dict]:
    """Per-document means within each group, a document bootstrap, and a last-minus-first contrast."""
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = key(r)
        if k is not None:
            cell[k][r["doc"]].append(r["delta"])
    groups = order or sorted(cell)
    groups = [g for g in groups if g in cell]
    docs = sorted({d for g in groups for d in cell[g]})
    pos = {d: i for i, d in enumerate(docs)}
    means = np.full((len(groups), len(docs)), np.nan)
    for gi, g in enumerate(groups):
        for d, vals in cell[g].items():
            means[gi, pos[d]] = np.mean(vals)
    idx = np.random.default_rng(seed).integers(0, len(docs), (BOOTSTRAP, len(docs)))
    boot = np.nanmean(means[:, idx], axis=2)  # groups x resamples
    out = []
    for gi, g in enumerate(groups):
        out.append({"group": str(g), "documents": int(np.sum(~np.isnan(means[gi]))),
                    "pairs": sum(len(v) for v in cell[g].values()),
                    "estimate": float(np.nanmean(means[gi])),
                    "lo": float(np.nanquantile(boot[gi], 0.025)), "hi": float(np.nanquantile(boot[gi], 0.975))})
    if len(groups) >= 2:
        diff = boot[-1] - boot[0]
        diff = diff[~np.isnan(diff)]
        est = float(np.nanmean(means[-1]) - np.nanmean(means[0]))
        p = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))) if diff.size else None
        out.append({"group": f"{groups[-1]} minus {groups[0]}", "documents": len(docs), "pairs": None,
                    "estimate": est, "lo": float(np.quantile(diff, 0.025)) if diff.size else None,
                    "hi": float(np.quantile(diff, 0.975)) if diff.size else None, "p": p, "contrast": True})
    return out


def moderators(ix: base.Index, corpus: str) -> list[dict]:
    docs = {d: ix.docs[d] for d in ix.docs if ix.docs[d]["corpus"] == corpus}
    length = tertiles({d: len(doc["source"].split()) for d, doc in docs.items()})
    spread = tertiles({d: evidence_paragraphs(doc) for d, doc in docs.items()})
    nq = {d: (str(len(doc["questions"])) if len(doc["questions"]) < 5 else "5+") for d, doc in docs.items()}
    specs = (("document length (source words)", lambda r: length.get(r["doc"]), None),
             ("evidence spread (distinct evidence paragraphs)", lambda r: spread.get(r["doc"]), None),
             ("questions per document", lambda r: nq.get(r["doc"]), None),
             ("held-out answer type", lambda r: r["answer_type"], ["extractive", "free_form", "yes_no", "none"]),
             ("distance from the anchor question", lambda r: r["tier"], list(TIER_ORDER)))
    out = []
    for family in base.FAMILIES:
        rows = pair_rows(ix, corpus, family)
        for name, key, order in specs:
            for r in grouped(rows, key, order):
                out.append({"corpus": corpus, "family": family, "moderator": name, **r})
    return out


# -------------------------------------------------------------------- output

def figure(out: Path, rows: list[dict], corpus: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = list(dict.fromkeys(r["moderator"] for r in rows))
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.8), facecolor=fu.SURFACE, sharex=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        groups = list(dict.fromkeys(r["group"] for r in rows if r["moderator"] == name and not r.get("contrast")))
        for off, family in ((-0.12, "generation"), (0.12, "selection")):
            sel = {r["group"]: r for r in rows if r["moderator"] == name and r["family"] == family
                   and not r.get("contrast")}
            ys = [i + off for i, g in enumerate(groups) if g in sel]
            est = [sel[g]["estimate"] for g in groups if g in sel]
            lo = [sel[g]["estimate"] - sel[g]["lo"] for g in groups if g in sel]
            hi = [sel[g]["hi"] - sel[g]["estimate"] for g in groups if g in sel]
            ax.errorbar(est, ys, xerr=[lo, hi], fmt="o", color=fu.COLORS[family], ms=5, lw=1.4, capsize=0,
                        label=family)
        ax.axvline(0, color=fu.INK_2, lw=0.8)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=8)
        ax.invert_yaxis()
        fu._style(ax, name, "query-aware minus generic, held-out F1")
        ax.set_title(name, fontsize=9, color=fu.INK, loc="left")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle(f"{corpus}: where the held-out-question loss appears", fontsize=10, color=fu.INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "conditions.png", dpi=150, facecolor=fu.SURFACE)
    plt.close(fig)
    return "conditions.png"


def fmt(x, nd=3, signed=False):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def report(run: str, bound: list[dict], mods: list[dict], fig: str | None) -> str:
    lines = [f"# exp5_cap_only: when does the held-out-question loss appear? (run `{run}`)", "",
             "## Boundary corpus", "",
             "From the calibration decision and the data. `pairs` is the share of (anchor, held-out question) "
             "pairs at each evidence distance; `best retention` is the highest normalised future retention any "
             "calibration budget gave generic messages.", "",
             "| corpus | docs | source words | paragraphs | questions/doc | answer words | pairs shared / near / far "
             "/ unknown | closed book | full source | natural words gen / cond | best retention | useful region |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---|"]
    for r in bound:
        pairs = " / ".join(fmt(r[f"pairs_{t}"], 2) for t in TIER_ORDER + ("unknown",))
        lines.append(f"| {r['corpus']} | {r['documents']} | {r['source_words_median']:.0f} | "
                     f"{r['paragraphs_median']:.0f} | {r['questions_per_doc_median']:.0f} | "
                     f"{fmt(r['answer_words_median'], 0)} | {pairs} | {fmt(r['U_closed_book'])} | "
                     f"{fmt(r['U_full_source'])} | {fmt(r['natural_generic_words'], 0)} / "
                     f"{fmt(r['natural_conditioned_words'], 0)} | {fmt(r['best_generic_retention'], 2)} | "
                     f"{'yes' if r['useful_region'] else 'no'} |")
    lines += ["", "## Moderators", "",
              "Query-aware minus generic score on held-out questions (F1), pooled over the six caps. Documents "
              "weigh equally within a group; 95% document-bootstrap intervals. The last row of each block is the "
              "last group minus the first, on the same resamples. Descriptive splits: the moderators are not "
              "randomised and some travel together.", ""]
    for corpus in sorted({r["corpus"] for r in mods}):
        for name in dict.fromkeys(r["moderator"] for r in mods if r["corpus"] == corpus):
            lines += [f"### {corpus}: {name}", "",
                      "| group | generation [95% CI] | selection [95% CI] | docs | pairs (generation) |",
                      "|---|---|---|---:|---:|"]
            block = [r for r in mods if r["corpus"] == corpus and r["moderator"] == name]
            for group in dict.fromkeys(r["group"] for r in block):
                cells = {r["family"]: r for r in block if r["group"] == group}
                g, s = cells.get("generation"), cells.get("selection")
                ci = (lambda r: "—" if r is None else
                      f"{fmt(r['estimate'], signed=True)} [{fmt(r['lo'], signed=True)}, {fmt(r['hi'], signed=True)}]")
                bold = "**" if g and g.get("contrast") else ""
                lines.append(f"| {bold}{group}{bold} | {ci(g)} | {ci(s)} | {g['documents'] if g else '—'} | "
                             f"{g['pairs'] if g and g['pairs'] is not None else ''} |")
            lines.append("")
    if fig:
        lines += [f"![{fig}]({fig})", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp5_cap_only_frozen_config.yaml")
    parser.add_argument("--split", default="main")
    parser.add_argument("--writer", default="mistral24")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = config(str(ROOT / args.config))
    run_dir = Path(args.run_dir) if args.run_dir else base.run_directory(cfg, args.split, args.tag)
    all_docs = read_rows(ROOT / cfg["data_root"] / f"{args.split}.jsonl")
    messages = base.load_stage(run_dir, "write", args.writer, "messages")
    answers = base.load_stage(run_dir, "read", args.writer, "answers")
    seen = {m["document_id"] for m in messages}
    docs = [d for d in all_docs if d["id"] in seen]
    ix = base.Index(docs, messages, answers)
    decision_file = (cfg["budgets"].get("calibrated_from") or {}).get("decision_file")
    calibration = json.loads((ROOT / decision_file).read_text(encoding="utf-8"))
    bound = boundary(all_docs, calibration)
    mods = [r for corpus in sorted({d["corpus"] for d in docs}) for r in moderators(ix, corpus)]
    out = Path(args.out) if args.out else ROOT / cfg["results_root"] / args.split / run_dir.name / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    base.write_csv(out / "boundary.csv", bound)
    base.write_csv(out / "moderators.csv", mods)
    fig = figure(out, mods, sorted({d["corpus"] for d in docs})[0]) if mods else None
    (out / "report.md").write_text(report(run_dir.name, bound, mods, fig), encoding="utf-8")
    print(f"conditions analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
