"""Follow-up analyses for the exp5_cap_only main run.

Reads the same stored write/read/judge outputs as analyse_exp5_capped.py and adds
three layers the design lists but the main analysis does not report:

Normalised future retention
    R = (U_future - U_closed_book) / (U_full_source - U_closed_book): the share of
    the full-source advantage over closed book that a message keeps for the
    document's other questions. The reference answers are scored on exactly the
    questions each message is scored on, R is a ratio of document sums, and the
    interval comes from a document bootstrap. Conditioned minus generic is paired
    on document.

Evidence survival from annotations
    For each question, the best share over annotators of the highlighted
    evidence's content words present in the message, split by distance from the
    conditioning question (now / shared / near / far, defined as in
    analyse_exp5_capped.py). Both arms' levels and their paired difference. Unlike
    the answer-token table, yes/no questions count here: their evidence is text.

Marginal allocation as a ratio of totals
    For each step between consecutive caps, the total change in each quantity over
    the total words added, per 100 words, with a document bootstrap. Evidence is
    measured on the highlighted evidence. Averaging per-message ratios, as the main
    analysis does, lets a message that grows by one word dominate a step; a ratio
    of totals weights each message by the words it actually added.

    python src/analysis/analyse_exp5_followup.py --config configs/exp5_cap_only_frozen_config.yaml --split main
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import analyse_exp5_capped as base  # noqa: E402
from reuse_common import config, read_rows  # noqa: E402

REFERENCES = ("closed_book", "full_source", "annotated_evidence")
POLICIES = [p for pair in base.FAMILIES.values() for p in pair]
FAMILY_OF = {p: f for f, pair in base.FAMILIES.items() for p in pair}
TIERS = ("now", "shared", "near", "far")
ALLOCATION_KEYS = ("now", "shared", "near", "far", "repeated_5grams", "unsupported_terms")
BOOTSTRAP = base.BOOTSTRAP

# Reference palette (validated): categorical slots 1 and 2, light surface, ink.
COLORS = {"generation": "#2a78d6", "selection": "#eb6834"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ------------------------------------------------------------------ helpers

def mean(values):
    values = list(values)
    return st.mean(values) if values else None


def references(answers: list[dict]) -> dict:
    """(doc, reference policy) -> {qid: answer row}."""
    out = defaultdict(dict)
    for a in answers:
        if a["block"] == "reference" and a["policy"] in REFERENCES:
            out[(a["document_id"], a["policy"])][a["qid"]] = a
    return out


def ratio_bootstrap(per_doc: dict, seed: int = 0) -> dict:
    """per_doc: doc -> (numerator, denominator); ratio of sums with a document bootstrap."""
    if not per_doc:
        return {"estimate": None, "lo": None, "hi": None, "n": 0}
    arr = np.asarray([per_doc[d] for d in sorted(per_doc)], dtype=float)
    num, den = arr[:, 0], arr[:, 1]
    est = float(num.sum() / den.sum()) if den.sum() else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (BOOTSTRAP, len(arr)))
    dens = den[idx].sum(axis=1)
    ok = dens != 0
    boots = num[idx].sum(axis=1)[ok] / dens[ok]
    lo, hi = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if boots.size else (None, None)
    return {"estimate": est, "lo": lo, "hi": hi, "n": len(arr)}


def paired_ratio_bootstrap(cond: dict, gen: dict, seed: int = 0) -> dict:
    """cond/gen: doc -> (U, C, F). Difference of retention ratios, documents resampled jointly."""
    docs = sorted(set(cond) & set(gen))
    if not docs:
        return {"estimate": None, "lo": None, "hi": None, "p": None, "n": 0}
    c = np.asarray([cond[d] for d in docs], dtype=float)
    g = np.asarray([gen[d] for d in docs], dtype=float)

    def delta(ci, gi):
        rc = (ci[..., 0].sum(-1) - ci[..., 1].sum(-1)) / (ci[..., 2].sum(-1) - ci[..., 1].sum(-1))
        rg = (gi[..., 0].sum(-1) - gi[..., 1].sum(-1)) / (gi[..., 2].sum(-1) - gi[..., 1].sum(-1))
        return rc - rg

    est = float(delta(c, g))
    idx = np.random.default_rng(seed).integers(0, len(docs), (BOOTSTRAP, len(docs)))
    boots = delta(c[idx], g[idx])
    boots = boots[np.isfinite(boots)]
    if not np.isfinite(est) or not boots.size:  # full source no better than closed book
        return {"estimate": None, "lo": None, "hi": None, "p": None, "n": len(docs)}
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"estimate": est, "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "p": float(min(1.0, p)), "n": len(docs)}


@lru_cache(maxsize=None)
def _evidence_sets(doc_id: str, qid: str, highlighted: tuple) -> tuple:
    return tuple(frozenset(base.content_tokens(text)) for text in highlighted if base.content_tokens(text))


def evidence_sets(doc_id: str, q: dict) -> tuple:
    """One content-word set per annotator with highlighted evidence."""
    texts = tuple(" ".join(a.get("highlighted_evidence") or []) for a in q.get("original_answers", []))
    return _evidence_sets(doc_id, q["qid"], texts)


def evidence_recall(message_tokens: set, sets: tuple) -> float | None:
    if not sets:
        return None
    return max(len(s & message_tokens) / len(s) for s in sets)


def tier_of(doc: dict, anchor: dict, q: dict) -> str | None:
    return "now" if q["qid"] == anchor["qid"] else base.distance(doc, anchor, q)


# ----------------------------------------------------------------- retention

def retention_units(ix, refs, doc_id: str, policy: str, cap: int, metric: str):
    """Per document: mean over messages of (U, closed, full) on each message's future questions."""
    closed, full = refs.get((doc_id, "closed_book"), {}), refs.get((doc_id, "full_source"), {})
    triples = []
    for m in ix.by_cell.get((doc_id, policy, cap), []):
        scores = ix.score.get((doc_id, policy, cap, m["anchor"]), {})
        qs = [q for q in scores if q != m["anchor"]
              and scores[q].get(metric) is not None
              and closed.get(q, {}).get(metric) is not None
              and full.get(q, {}).get(metric) is not None]
        if qs:
            triples.append((mean(scores[q][metric] for q in qs), mean(closed[q][metric] for q in qs),
                            mean(full[q][metric] for q in qs)))
    if not triples:
        return None
    return tuple(mean(t[i] for t in triples) for i in range(3))


def retention(ix, refs, corpora: list[str], metrics=("f1", "judge_correct")) -> tuple[list, list]:
    levels, deltas = [], []
    for corpus in corpora:
        doc_ids = [d for d, doc in ix.docs.items() if doc["corpus"] == corpus]
        for metric in metrics:
            for cap in ix.caps():
                units = {}
                for policy in POLICIES:
                    units[policy] = {d: u for d in doc_ids
                                     if (u := retention_units(ix, refs, d, policy, cap, metric)) is not None}
                    per_doc = {d: (u[0] - u[1], u[2] - u[1]) for d, u in units[policy].items()}
                    b = ratio_bootstrap(per_doc)
                    us = list(units[policy].values())
                    levels.append({"corpus": corpus, "metric": metric, "cap": cap, "policy": policy,
                                   "family": FAMILY_OF[policy], "retention": b["estimate"],
                                   "lo": b["lo"], "hi": b["hi"],
                                   "U_future": mean(u[0] for u in us), "U_closed_book": mean(u[1] for u in us),
                                   "U_full_source": mean(u[2] for u in us), "documents": b["n"]})
                for family, (cond, gen) in base.FAMILIES.items():
                    b = paired_ratio_bootstrap(units[cond], units[gen])
                    deltas.append({"corpus": corpus, "metric": metric, "cap": cap, "family": family,
                                   "retention_conditioned_minus_generic": b["estimate"],
                                   "lo": b["lo"], "hi": b["hi"], "p": b["p"], "documents": b["n"]})
    for metric_block in {(r["corpus"], r["metric"], r["family"]) for r in deltas}:
        block = [r for r in deltas if (r["corpus"], r["metric"], r["family"]) == metric_block]
        for row, adj in zip(block, base.holm([r["p"] for r in block])):
            row["p_holm"] = adj
    return levels, deltas


# ---------------------------------------------------------- evidence survival

def evidence_survival(ix, corpora: list[str]) -> list[dict]:
    rows = []
    for corpus in corpora:
        for family, (cond, gen) in base.FAMILIES.items():
            for cap in ix.caps():
                lv_c, lv_g, diff = ({t: defaultdict(list) for t in TIERS} for _ in range(3))
                for doc_id, doc in ix.docs.items():
                    if doc["corpus"] != corpus:
                        continue
                    generic = ix.by_cell.get((doc_id, gen, cap), [])
                    if not generic:
                        continue
                    g_tokens = set(base.content_tokens(generic[0]["text"]))
                    qs = ix.questions[doc_id]
                    for m in ix.by_cell.get((doc_id, cond, cap), []):
                        anchor = qs.get(m["anchor"])
                        if anchor is None:
                            continue
                        c_tokens = set(base.content_tokens(m["text"]))
                        for q in qs.values():
                            tier = tier_of(doc, anchor, q)
                            sets = evidence_sets(doc_id, q)
                            if tier is None or not sets:
                                continue
                            rc, rg = evidence_recall(c_tokens, sets), evidence_recall(g_tokens, sets)
                            lv_c[tier][doc_id].append(rc)
                            lv_g[tier][doc_id].append(rg)
                            diff[tier][doc_id].append(rc - rg)
                for tier in TIERS:
                    b = base.bootstrap([st.mean(v) for v in diff[tier].values() if v])
                    rows.append({"corpus": corpus, "family": family, "cap": cap, "tier": tier,
                                 "recall_conditioned": mean(st.mean(v) for v in lv_c[tier].values() if v),
                                 "recall_generic": mean(st.mean(v) for v in lv_g[tier].values() if v),
                                 "conditioned_minus_generic": b["estimate"], "lo": b["lo"], "hi": b["hi"],
                                 "p": b["p"], "documents": b["n"]})
    return rows


# ---------------------------------------------------------------- allocation

def evidence_profile(doc: dict, text: str, anchor: dict | None) -> dict:
    """Highlighted-evidence recall summed by tier, plus repetition and unsupported terms."""
    tokens = set(base.content_tokens(text))
    out = {"words": len(text.split()), "now": 0.0, "shared": 0.0, "near": 0.0, "far": 0.0}
    if anchor is not None:
        for q in doc["questions"]:
            sets = evidence_sets(doc["id"], q)
            tier = tier_of(doc, anchor, q)
            if sets and tier:
                out[tier] += evidence_recall(tokens, sets)
    grams = base.sx.ngrams(text, 5)
    out["repeated_5grams"] = len(grams) - len(set(grams))
    out["unsupported_terms"] = base.sx.unsupported_scan(text, doc["source"], base.UNSUPPORTED_SPEC)["unsupported_count"]
    return out


def allocation(ix, corpora: list[str]) -> list[dict]:
    caps, rows, cache = ix.caps(), [], {}

    def prof(doc, m, anchor):
        key = (m["message_id"], anchor["qid"] if anchor else None)
        if key not in cache:
            cache[key] = evidence_profile(doc, m["text"], anchor)
        return cache[key]

    for corpus in corpora:
        for family, pair in base.FAMILIES.items():
            for policy in pair:
                for lo, hi in zip(caps, caps[1:]):
                    per_doc = defaultdict(lambda: defaultdict(float))
                    added_all = []
                    for doc_id, doc in ix.docs.items():
                        if doc["corpus"] != corpus:
                            continue
                        for anchor_id in [m["anchor"] for m in ix.by_cell.get((doc_id, pair[0], hi), [])]:
                            anchor = ix.questions[doc_id].get(anchor_id)
                            pick = (lambda cap: [m for m in ix.by_cell.get((doc_id, policy, cap), [])
                                                 if m["anchor"] in (anchor_id, None)])
                            before, after = pick(lo), pick(hi)
                            if not before or not after:
                                continue
                            a, b = prof(doc, before[0], anchor), prof(doc, after[0], anchor)
                            added = b["words"] - a["words"]
                            added_all.append(added)
                            per_doc[doc_id]["added"] += added
                            for k in ALLOCATION_KEYS:
                                per_doc[doc_id][k] += b[k] - a[k]
                    row = {"corpus": corpus, "family": family, "policy": policy, "step": f"{lo}->{hi}",
                           "cap_from": lo, "cap_to": hi, "pairs": len(added_all),
                           "mean_added_words": mean(added_all),
                           "share_pairs_not_longer": (sum(x <= 0 for x in added_all) / len(added_all)
                                                      if added_all else None)}
                    for k in ALLOCATION_KEYS:
                        b = ratio_bootstrap({d: (100 * v[k], v["added"]) for d, v in per_doc.items()})
                        row[f"{k}_per_100"], row[f"{k}_lo"], row[f"{k}_hi"] = b["estimate"], b["lo"], b["hi"]
                    rows.append(row)
    return rows


# ------------------------------------------------------------------- figures

def _style(ax, title, xlabel="nominal cap (words)"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _cap_axis(ax, caps):
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(caps)
    ax.set_xticklabels([str(c) for c in caps])


def _line(ax, xs, ys, lo, hi, family, generic, label):
    xs, ys = np.asarray(xs, float), np.asarray([np.nan if y is None else y for y in ys], float)
    lo = np.asarray([np.nan if v is None else v for v in lo], float)
    hi = np.asarray([np.nan if v is None else v for v in hi], float)
    ax.plot(xs, ys, "--" if generic else "-", marker=None if generic else "o", ms=5, lw=2,
            color=COLORS[family], label=label)
    ax.fill_between(xs, lo, hi, color=COLORS[family], alpha=0.12, lw=0)


def figures(out: Path, corpus: str, levels, deltas, surv, alloc) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    names = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    for policy in POLICIES:
        rows = [r for r in levels if r["corpus"] == corpus and r["metric"] == "f1" and r["policy"] == policy]
        generic = policy.endswith("generic")
        _line(axes[0], [r["cap"] for r in rows], [r["retention"] for r in rows], [r["lo"] for r in rows],
              [r["hi"] for r in rows], FAMILY_OF[policy], generic,
              f"{FAMILY_OF[policy]} {'generic' if generic else 'conditioned'}")
    for family in base.FAMILIES:
        rows = [r for r in deltas if r["corpus"] == corpus and r["metric"] == "f1" and r["family"] == family]
        _line(axes[1], [r["cap"] for r in rows], [r["retention_conditioned_minus_generic"] for r in rows],
              [r["lo"] for r in rows], [r["hi"] for r in rows], family, False, family)
    axes[1].axhline(0, color=INK_2, lw=0.8)
    _style(axes[0], f"{corpus}: normalised future retention (F1)")
    _style(axes[1], f"{corpus}: retention, conditioned - generic")
    for ax in axes:
        _cap_axis(ax, sorted({r["cap"] for r in levels}))
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "retention.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    names.append("retention.png")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=SURFACE, sharey=True)
    for ax, tier in zip(axes, TIERS):
        for family in base.FAMILIES:
            rows = [r for r in surv if r["corpus"] == corpus and r["family"] == family and r["tier"] == tier]
            caps = [r["cap"] for r in rows]
            _line(ax, caps, [r["recall_generic"] for r in rows], [None] * len(rows), [None] * len(rows),
                  family, True, f"{family} generic")
            _line(ax, caps, [r["recall_conditioned"] for r in rows], [None] * len(rows), [None] * len(rows),
                  family, False, f"{family} conditioned")
        _style(ax, f"{corpus}: highlighted evidence kept, {tier}")
        _cap_axis(ax, sorted({r["cap"] for r in surv}))
    axes[0].set_ylabel("share of evidence content words", fontsize=9, color=INK_2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "evidence_survival.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    names.append("evidence_survival.png")

    panels = (("now", "current-question evidence"), ("far", "far evidence"),
              ("repeated_5grams", "repeated 5-grams"), ("unsupported_terms", "unsupported terms"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        for policy in POLICIES:
            rows = [r for r in alloc if r["corpus"] == corpus and r["policy"] == policy]
            xs = list(range(len(rows)))
            generic = policy.endswith("generic")
            _line(ax, xs, [r[f"{key}_per_100"] for r in rows], [r[f"{key}_lo"] for r in rows],
                  [r[f"{key}_hi"] for r in rows], FAMILY_OF[policy], generic,
                  f"{FAMILY_OF[policy]} {'generic' if generic else 'conditioned'}")
        ax.axhline(0, color=INK_2, lw=0.8)
        _style(ax, f"{title} per 100 added words", "cap step (words)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['cap_from']}→{r['cap_to']}" for r in rows], fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "allocation.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    names.append("allocation.png")
    return names


# -------------------------------------------------------------------- report

def fmt(x, nd=3, signed=False):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def report(meta, levels, deltas, surv, alloc, figs) -> str:
    lines = [f"# exp5_cap_only follow-up analyses — {meta['writer']}", "",
             f"Run `{meta['run']}`, split `{meta['split']}`, caps {meta['caps']}. "
             "Intervals are 95% document-bootstrap intervals.", ""]
    lines += ["## Normalised future retention", "",
              "R = (U_future − U_closed_book) / (U_full_source − U_closed_book), F1, with the reference answers "
              "scored on each message's own future questions.", "",
              "| corpus | cap | summary gen | summary cond | Δ summary [95% CI] | selection gen | selection cond "
              "| Δ selection [95% CI] |", "|---|---:|---:|---:|---|---:|---:|---|"]
    for corpus in meta["corpora"]:
        for cap in meta["caps"]:
            lv = {r["policy"]: r for r in levels if r["corpus"] == corpus and r["metric"] == "f1" and r["cap"] == cap}
            dl = {r["family"]: r for r in deltas if r["corpus"] == corpus and r["metric"] == "f1" and r["cap"] == cap}

            def d(f):
                r = dl.get(f, {})
                return (f"{fmt(r.get('retention_conditioned_minus_generic'), signed=True)} "
                        f"[{fmt(r.get('lo'), signed=True)}, {fmt(r.get('hi'), signed=True)}]")
            lines.append(f"| {corpus} | {cap} | {fmt(lv['summary_generic']['retention'])} | "
                         f"{fmt(lv['summary_conditioned']['retention'])} | {d('generation')} | "
                         f"{fmt(lv['selection_generic']['retention'])} | {fmt(lv['selection_conditioned']['retention'])} | "
                         f"{d('selection')} |")
    refs = [r for r in levels if r["metric"] == "f1" and r["policy"] == "summary_generic"]
    if refs:
        lines += ["", f"Reference levels on the generic arm's questions: closed book "
                      f"{fmt(mean(r['U_closed_book'] for r in refs))}, full source {fmt(mean(r['U_full_source'] for r in refs))} F1."]
    lines += ["", "## Evidence survival from highlighted evidence", "",
              "Share of each question's highlighted-evidence content words present in the message (best annotator).", "",
              "| corpus | family | cap | tier | generic | conditioned | cond − gen [95% CI] |",
              "|---|---|---:|---|---:|---:|---|"]
    for r in surv:
        lines.append(f"| {r['corpus']} | {r['family']} | {r['cap']} | {r['tier']} | {fmt(r['recall_generic'])} | "
                     f"{fmt(r['recall_conditioned'])} | {fmt(r['conditioned_minus_generic'], signed=True)} "
                     f"[{fmt(r['lo'], signed=True)}, {fmt(r['hi'], signed=True)}] |")
    lines += ["", "## Marginal allocation (ratio of totals)", "",
              "Per 100 added words: change in summed evidence recall by tier (question-equivalents), repeated "
              "5-grams and unsupported terms. `not longer` is the share of message pairs that did not grow.", "",
              "| corpus | policy | step | added words | not longer | now | near | far | repeated 5-grams | unsupported |",
              "|---|---|---|---:|---:|---|---|---|---|---|"]
    for r in alloc:
        cell = (lambda k: f"{fmt(r[f'{k}_per_100'], 2, True)} [{fmt(r[f'{k}_lo'], 2, True)}, {fmt(r[f'{k}_hi'], 2, True)}]")
        lines.append(f"| {r['corpus']} | {r['policy']} | {r['step']} | {fmt(r['mean_added_words'], 1)} | "
                     f"{fmt(r['share_pairs_not_longer'], 2)} | {cell('now')} | {cell('near')} | {cell('far')} | "
                     f"{cell('repeated_5grams')} | {cell('unsupported_terms')} |")
    lines += [""] + [f"![{f}]({f})" for f in figs] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------- main

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
    docs = read_rows(ROOT / cfg["data_root"] / f"{args.split}.jsonl")
    messages = base.load_stage(run_dir, "write", args.writer, "messages")
    answers = base.load_stage(run_dir, "read", args.writer, "answers")
    verdicts = base.load_stage(run_dir, "judge", args.writer, "verdicts") if (run_dir / "judge").is_dir() else []
    base.attach_verdicts(answers, verdicts)
    seen = {m["document_id"] for m in messages}
    docs = [d for d in docs if d["id"] in seen]
    corpora = sorted({d["corpus"] for d in docs})
    ix = base.Index(docs, messages, answers)
    refs = references(answers)
    levels, deltas = retention(ix, refs, corpora)
    surv = evidence_survival(ix, corpora)
    alloc = allocation(ix, corpora)
    out = Path(args.out) if args.out else ROOT / cfg["results_root"] / args.split / run_dir.name / "followup"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("retention", levels), ("retention_delta", deltas), ("evidence_survival", surv),
                       ("allocation_ratio", alloc)):
        base.write_csv(out / f"{name}.csv", rows)
    figs = []
    for corpus in corpora:
        figs += figures(out, corpus, levels, deltas, surv, alloc)
    meta = {"writer": args.writer, "run": run_dir.name, "split": args.split, "caps": ix.caps(), "corpora": corpora}
    (out / "report.md").write_text(report(meta, levels, deltas, surv, alloc, figs), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"follow-up analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
