"""Forced-fill diagnostic on the legacy fill-corrected run (results family forced_fill_legacy).

The legacy writer prompt asked for 85-100% of the cap, and the runner
re-prompted any message that came back shorter. Each stored message keeps its
first attempt's length and request key, and the response cache keeps that first
attempt's text, so the extra words produced under correction can be compared
with what the writer wrote on its own - same writer, prompt, document, question
and cap.

For summaries whose first attempt fell below the fill floor, per 100 words added
by correction (ratio of totals, document bootstrap):

- highlighted-evidence recall gained, summed over questions: all questions, and
  for conditioned summaries by distance from the conditioning question;
- repeated 5-grams and unsupported terms (Experiment 9's deterministic scan);
- new content-word types, split into those found in the source and those not.

A secondary table compares the legacy forced-fill summaries at 256 words with the
exp5_cap_only summaries at the same cap. The two prompts differ in more than the
fill instruction (see results/exp5_prompt_family/JOB_AUDIT.md), so that table
describes the two protocols side by side and attributes nothing to fill alone.
These outputs are never pooled with the exp5_cap_only curves.

The corrected message is a fresh generation, not an extension of the first
attempt, so every quantity is a difference between two complete messages divided
by the net words added. Rewording therefore counts: new word types can exceed the
net words added when the length barely changed. The figure shows per-100-word
values only where correction added at least 25 words per message on
average across at least 30 messages; the tables report every budget.

    python src/analysis/analyse_forced_fill_legacy.py --run-dir runs/reuse/main/<run> \
        --cache-dir cache/reuse [--exp5-run-dir runs/exp5_cap_only/main/<run>]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import analyse_exp5_capped as base  # noqa: E402
import analyse_exp5_followup as fu  # noqa: E402
from reuse_common import read_rows  # noqa: E402

FILL_FLOOR = 0.85  # the legacy prompt's "aim for ceil(0.85 * cap) to cap words"
POLICIES = ("summary_conditioned", "summary_generic")
TIERS = ("now", "shared", "near", "far")
KEYS = ("all_questions",) + TIERS + ("repeated_5grams", "unsupported_terms",
                                     "new_supported_types", "new_unsupported_types")
BUDGET_ORDER = ("w64", "w128", "w256", "w512", "w1024", "w2048", "r0.25", "r0.5", "r0.75", "r1")
KEY_PATTERN = re.compile(r'"key":\s*"([0-9a-f]+)"')
MIN_NET_WORDS, MIN_PAIRS = 25, 30  # figure only; see the module docstring


def budget_label(message: dict) -> str:
    labels = message.get("labels") or []
    return labels[0] if labels else f"w{message['cap']}"


def fill_class(message: dict) -> str:
    first = message["attempts"][0]["raw_words"]
    if first > message["cap"]:
        return "over_cap"
    return "short" if first < math.ceil(FILL_FLOOR * message["cap"]) else "in_band"


def load_messages(run_dir: Path, writer: str, doc_ids: set) -> list[dict]:
    out = []
    for path in sorted((run_dir / "write" / writer).glob("*.json")):
        if path.stem not in doc_ids:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for m in payload["messages"]:
            if m.get("block") == "core" and m["policy"] in POLICIES and m.get("valid") and m.get("attempts"):
                out.append({**m, "document_id": payload["document_id"]})  # stored once per file
    return out


def cached_texts(cache_dir: Path, keys: set) -> dict:
    """First-attempt texts by request key, read from the append-only result cache."""
    found = {}
    for path in sorted(cache_dir.glob("*/write-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = KEY_PATTERN.search(line[:120])
                if match and match.group(1) in keys and match.group(1) not in found:
                    try:
                        found[match.group(1)] = json.loads(line)["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue
    return found


def content_profile(doc: dict, text: str, anchor: dict | None) -> dict:
    tokens = set(base.content_tokens(text))
    out = {k: 0.0 for k in ("all_questions",) + TIERS}
    for q in doc["questions"]:
        sets = fu.evidence_sets(doc["id"], q)
        if not sets:
            continue
        recall = fu.evidence_recall(tokens, sets)
        out["all_questions"] += recall
        if anchor is not None:
            tier = fu.tier_of(doc, anchor, q)
            if tier:
                out[tier] += recall
    grams = base.sx.ngrams(text, 5)
    out["repeated_5grams"] = len(grams) - len(set(grams))
    out["unsupported_terms"] = base.sx.unsupported_scan(text, doc["source"], base.UNSUPPORTED_SPEC)["unsupported_count"]
    out["words"] = len(text.split())
    out["types"] = tokens
    return out


def expansion(docs: dict, messages: list[dict], first_texts: dict, source_types: dict) -> tuple[list, list]:
    classes, rows = [], []
    groups = defaultdict(list)
    for m in messages:
        groups[(m["policy"], budget_label(m))].append(m)
    for (policy, label), ms in sorted(groups.items(), key=lambda kv: (kv[0][0], BUDGET_ORDER.index(kv[0][1])
                                                                        if kv[0][1] in BUDGET_ORDER else 99)):
        counts = Counter(fill_class(m) for m in ms)
        classes.append({"policy": policy, "budget": label, "messages": len(ms),
                        **{f"share_{k}": counts[k] / len(ms) for k in ("short", "in_band", "over_cap")},
                        "first_words_median": st.median(m["attempts"][0]["raw_words"] for m in ms),
                        "delivered_words_median": st.median(m["delivered_words"] for m in ms)})
        per_doc = defaultdict(lambda: defaultdict(float))
        pairs = missing = 0
        for m in ms:
            if fill_class(m) != "short":
                continue
            first = first_texts.get(m["attempts"][0]["request_key"])
            if first is None:
                missing += 1
                continue
            doc = docs[m["document_id"]]
            anchor = next((q for q in doc["questions"] if q["qid"] == m.get("anchor")), None)
            a, b = content_profile(doc, first, anchor), content_profile(doc, m["text"], anchor)
            new_types = b["types"] - a["types"]
            src = source_types[doc["id"]]
            delta = {k: b[k] - a[k] for k in ("all_questions",) + TIERS + ("repeated_5grams", "unsupported_terms")}
            delta["new_supported_types"] = len(new_types & src)
            delta["new_unsupported_types"] = len(new_types - src)
            per_doc[doc["id"]]["added"] += b["words"] - a["words"]
            for k in KEYS:
                per_doc[doc["id"]][k] += delta[k]
            pairs += 1
        added_total = sum(v["added"] for v in per_doc.values())
        row = {"policy": policy, "budget": label, "pairs": pairs, "first_attempt_text_missing": missing,
               "added_words_total": added_total, "mean_net_added_words": added_total / pairs if pairs else None}
        for k in KEYS:
            if policy == "summary_generic" and k in TIERS:
                continue
            b = fu.ratio_bootstrap({d: (100 * v[k], v["added"]) for d, v in per_doc.items()})
            row[f"{k}_per_100"], row[f"{k}_lo"], row[f"{k}_hi"] = b["estimate"], b["lo"], b["hi"]
        rows.append(row)
    return classes, rows


def protocol_levels(docs: dict, messages: list[dict], protocol: str) -> list[dict]:
    """Per-message levels: mean evidence recall per tier, repetition and unsupported per 100 words."""
    per_policy = defaultdict(lambda: defaultdict(list))
    for m in messages:
        doc = docs[m["document_id"]]
        anchor = next((q for q in doc["questions"] if q["qid"] == m.get("anchor")), None)
        tokens = set(base.content_tokens(m["text"]))
        words = max(1, len(m["text"].split()))
        tiers = defaultdict(list)
        for q in doc["questions"]:
            sets = fu.evidence_sets(doc["id"], q)
            if not sets:
                continue
            recall = fu.evidence_recall(tokens, sets)
            tiers["all_questions"].append(recall)
            if anchor is not None and (t := fu.tier_of(doc, anchor, q)):
                tiers[t].append(recall)
        rec = per_policy[m["policy"]]
        rec["words"].append(words)
        for k, v in tiers.items():
            rec[k].append(st.mean(v))
        grams = base.sx.ngrams(m["text"], 5)
        rec["repeated_5grams_per_100"].append(100 * (len(grams) - len(set(grams))) / words)
        rec["unsupported_per_100"].append(
            100 * base.sx.unsupported_scan(m["text"], doc["source"], base.UNSUPPORTED_SPEC)["unsupported_count"] / words)
    rows = []
    for policy, rec in sorted(per_policy.items()):
        row = {"protocol": protocol, "policy": policy, "messages": len(rec["words"])}
        for k, v in rec.items():
            row[k] = st.mean(v) if v else None
        rows.append(row)
    return rows


def figure(out: Path, classes: list, rows: list) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    budgets = [b for b in BUDGET_ORDER if any(r["budget"] == b for r in rows)]
    xs = list(range(len(budgets)))
    panels = (("share_short", "first attempt below the fill floor", classes, None),
              ("all_questions_per_100", "evidence gained, all questions", rows, "all_questions"),
              ("far_per_100", "far evidence gained (conditioned)", rows, "far"),
              ("new_supported_types_per_100", "new source-supported word types", rows, "new_supported_types"),
              ("repeated_5grams_per_100", "repeated 5-grams", rows, "repeated_5grams"),
              ("unsupported_terms_per_100", "unsupported terms", rows, "unsupported_terms"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor=fu.SURFACE)
    for ax, (key, title, table, stem) in zip(axes.flat, panels):
        for policy in POLICIES:
            by = {r["budget"]: r for r in table if r["policy"] == policy}
            shown = {b for b, r in by.items() if key == "share_short"
                     or (r.get("pairs", 0) >= MIN_PAIRS and (r.get("mean_net_added_words") or 0) >= MIN_NET_WORDS)}
            ys = [by.get(b, {}).get(key) if b in shown else None for b in budgets]
            if all(y is None for y in ys):
                continue
            lo = [by.get(b, {}).get(f"{stem}_lo") if b in shown else None for b in budgets] if stem else [None] * len(budgets)
            hi = [by.get(b, {}).get(f"{stem}_hi") if b in shown else None for b in budgets] if stem else [None] * len(budgets)
            # Both arms are generation: blue, with generic dashed as in the exp5 figures.
            fu._line(ax, xs, ys, lo, hi, "generation", policy == "summary_generic",
                     policy.replace("summary_", "summary, "))
        ax.axhline(0, color=fu.INK_2, lw=0.8)
        fu._style(ax, title + ("" if key == "share_short" else " per 100 added words"), "legacy budget")
        ax.set_xticks(xs)
        ax.set_xticklabels(budgets, fontsize=8)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "forced_fill.png", dpi=150, facecolor=fu.SURFACE)
    plt.close(fig)
    return "forced_fill.png"


def fmt(x, nd=2, signed=True):
    return "—" if x is None else (f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}")


def report(meta, classes, rows, protocol, fig) -> str:
    lines = [f"# Forced-fill diagnostic — legacy fill-corrected run `{meta['run']}`, {meta['writer']}, {meta['corpus']}", "",
             "Legacy prompt: aim for ceil(0.85 × cap) to cap words; shorter messages were re-prompted. "
             "Per 100 words added by correction, first attempt versus delivered message, same prompt and "
             "question. Ratio of totals with 95% document-bootstrap intervals.", "",
             "## How often the writer stopped short on its own", "",
             "| policy | budget | messages | first attempt short | in band | over cap | first words (median) "
             "| delivered words (median) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in classes:
        lines.append(f"| {r['policy']} | {r['budget']} | {r['messages']} | {r['share_short']:.2f} | "
                     f"{r['share_in_band']:.2f} | {r['share_over_cap']:.2f} | {r['first_words_median']:.0f} | "
                     f"{r['delivered_words_median']:.0f} |")
    lines += ["", "## What the words added under correction contain", "",
              "Evidence is highlighted-evidence recall summed over questions (question-equivalents). The corrected "
              "message is a fresh generation, so each value compares two complete messages per 100 net added words; "
              "where the net gain is small (see `net added`), rewording dominates and the values are unstable.", "",
              "| policy | budget | pairs | net added (mean words) | evidence, all questions | now | far | "
              "new supported types | new unsupported types | repeated 5-grams | unsupported terms |",
              "|---|---|---:|---:|---|---|---|---|---|---|---|"]
    for r in rows:
        cell = (lambda k: "—" if f"{k}_per_100" not in r else
                f"{fmt(r[f'{k}_per_100'])} [{fmt(r[f'{k}_lo'])}, {fmt(r[f'{k}_hi'])}]")
        lines.append(f"| {r['policy']} | {r['budget']} | {r['pairs']} | {fmt(r['mean_net_added_words'], 0, False)} | "
                     f"{cell('all_questions')} | {cell('now')} | "
                     f"{cell('far')} | {cell('new_supported_types')} | {cell('new_unsupported_types')} | "
                     f"{cell('repeated_5grams')} | {cell('unsupported_terms')} |")
    missing = sum(r["first_attempt_text_missing"] for r in rows)
    lines += ["", f"First attempts whose text was not in the cache: {missing}."]
    if protocol:
        lines += ["", "## Legacy forced fill and exp5_cap_only side by side at 256 words", "",
                  "The prompts differ beyond the fill instruction, so this compares protocols, not fill alone. "
                  "Evidence columns are mean highlighted-evidence recall per question.", "",
                  "| protocol | policy | messages | words | all questions | now | near | far | "
                  "repeated 5-grams /100w | unsupported /100w |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in protocol:
            g = (lambda k: fmt(r.get(k), 3, False))
            lines.append(f"| {r['protocol']} | {r['policy']} | {r['messages']} | {r['words']:.0f} | "
                         f"{g('all_questions')} | {g('now')} | {g('near')} | {g('far')} | "
                         f"{fmt(r['repeated_5grams_per_100'], 2, False)} | {fmt(r['unsupported_per_100'], 2, False)} |")
    if fig:
        lines += ["", f"![{fig}]({fig})"]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="legacy fill-corrected run root, e.g. runs/reuse/main/<run>")
    parser.add_argument("--cache-dir", required=True, help="result cache holding the first attempts")
    parser.add_argument("--data", default="data/reuse_v2/main.jsonl")
    parser.add_argument("--writer", default="mistral24")
    parser.add_argument("--corpus", default="qasper")
    parser.add_argument("--exp5-run-dir", default="", help="exp5_cap_only run root for the 256-word comparison")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_dir = Path(args.run_dir)
    docs = {d["id"]: d for d in read_rows(ROOT / args.data) if d["corpus"] == args.corpus}
    source_types = {d_id: set(base.content_tokens(d["source"])) for d_id, d in docs.items()}
    messages = load_messages(run_dir, args.writer, set(docs))
    keys = {m["attempts"][0]["request_key"] for m in messages if fill_class(m) == "short"}
    first_texts = cached_texts(Path(args.cache_dir), keys)
    print(json.dumps({"event": "loaded", "messages": len(messages), "short_first_attempts": len(keys),
                      "texts_found": len(first_texts)}), flush=True)
    classes, rows = expansion(docs, messages, first_texts, source_types)
    protocol = []
    if args.exp5_run_dir:
        legacy_256 = [m for m in messages if budget_label(m) == "w256"]
        exp5 = [m for m in base.load_stage(Path(args.exp5_run_dir), "write", args.writer, "messages")
                if m["block"] == "core" and m["valid"] and m["cap"] == 256 and m["policy"] in POLICIES
                and m["document_id"] in docs]
        protocol = protocol_levels(docs, legacy_256, "legacy forced fill") + protocol_levels(docs, exp5, "exp5_cap_only")
    out = Path(args.out) if args.out else ROOT / "results" / "forced_fill_legacy" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    base.write_csv(out / "fill_classes.csv", classes)
    base.write_csv(out / "expansion.csv", rows)
    if protocol:
        base.write_csv(out / "protocol_w256.csv", protocol)
    fig = figure(out, classes, rows)
    meta = {"run": run_dir.name, "writer": args.writer, "corpus": args.corpus, "family": "forced_fill_legacy",
            "exp5_run": Path(args.exp5_run_dir).name if args.exp5_run_dir else None}
    (out / "report.md").write_text(report(meta, classes, rows, protocol, fig), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"forced-fill diagnostic written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
