"""Apply the pre-registered freeze rule to the exp5_cap_only calibration run.

Reads two development runs and nothing else:

  runs/exp5_uncapped/development/<writer>/messages.jsonl
      how long each arm writes when nothing constrains it
  runs/exp5_cap_only/development/<hash>-calib/{write,read}/<writer>/
      the capped generic summaries at every candidate budget, and the reader's
      answers to them, plus the full-source and closed-book references

and applies ``budgets.freeze_rule`` from configs/exp5_cap_only_config.yaml, which was
written before this run returned.

What it deliberately does not do: compute any conditioned-minus-generic
utility. Conditioned answers are discarded at load time. Conditioned messages
are read only for their lengths, which item 9 of the design asks for and which
cannot reveal the effect under study. The grid is chosen from the generic
control and the uncapped length distribution alone.

    python src/analysis/analyse_exp5_calibration.py --writer mistral24
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

from reuse_common import CLOSED_BOOK, config, read_rows, source_view  # noqa: E402

GENERIC = "summary_generic"
CONDITIONED = "summary_conditioned"
KEEP_ANSWERS = {GENERIC, "full_source", CLOSED_BOOK, "annotated_evidence"}
CORPORA = ("qasper", "squad")


# --------------------------------------------------------------------- load

def find_calibration_run(writer: str) -> Path:
    matches = sorted(glob.glob(str(ROOT / "runs/exp5_cap_only/development/*-calib")))
    matches = [Path(m) for m in matches if (Path(m) / "write" / writer).is_dir()]
    if len(matches) != 1:
        raise SystemExit(f"expected one calibration run for {writer}, found {len(matches)}: "
                         f"{[m.name for m in matches]}")
    return matches[0]


def load_uncapped(writer: str, path: Path | None = None) -> list[dict]:
    path = path or ROOT / "runs/exp5_uncapped/development" / writer / "messages.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_capped(run_dir: Path, writer: str) -> tuple[list[dict], list[dict]]:
    messages, answers = [], []
    for path in sorted((run_dir / "write" / writer).glob("*.json")):
        messages.extend(m for m in json.loads(path.read_text(encoding="utf-8"))["messages"]
                        if m["block"] == "core")
    for path in sorted((run_dir / "read" / writer).glob("*.json")):
        for answer in json.loads(path.read_text(encoding="utf-8"))["answers"]:
            # Conditioned answers are dropped here, before anything is computed.
            if answer["policy"] in KEEP_ANSWERS:
                answers.append(answer)
    assert not any(a["policy"] == CONDITIONED for a in answers)
    return messages, answers


# --------------------------------------------------------------- statistics

def doc_means(answers: list[dict], policy: str, cap: int | None = None) -> dict[str, float]:
    """Per-document mean F1: documents are the resampling unit."""
    by_doc: dict[str, list[float]] = {}
    for a in answers:
        if a["policy"] != policy or (cap is not None and a["cap"] != cap):
            continue
        by_doc.setdefault(a["document_id"], []).append(a["f1"])
    return {d: st.mean(v) for d, v in by_doc.items()}


def retention(gen: dict, empty: dict, full: dict) -> float | None:
    docs = sorted(set(gen) & set(empty) & set(full))
    if not docs:
        return None
    u_gen = st.mean(gen[d] for d in docs)
    u_empty = st.mean(empty[d] for d in docs)
    u_full = st.mean(full[d] for d in docs)
    return None if u_full <= u_empty else (u_gen - u_empty) / (u_full - u_empty)


def retention_ci(gen, empty, full, samples=2000, seed=0) -> tuple[float, float]:
    docs = sorted(set(gen) & set(empty) & set(full))
    rng, values = random.Random(seed), []
    for _ in range(samples):
        pick = [docs[rng.randrange(len(docs))] for _ in docs]
        u_gen = st.mean(gen[d] for d in pick)
        u_empty = st.mean(empty[d] for d in pick)
        u_full = st.mean(full[d] for d in pick)
        if u_full > u_empty:
            values.append((u_gen - u_empty) / (u_full - u_empty))
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def natural_lengths(uncapped: list[dict], corpus: str, arm: str) -> list[float]:
    """A message that hit the token ceiling was longer than any candidate: +inf."""
    out = []
    for r in uncapped:
        if r["corpus"] != corpus or r["arm"] != arm:
            continue
        out.append(math.inf if r["finish_reason"] == "length" else r["realized_words"])
    return out


def median_sentence_words(corpus: str) -> float:
    docs = [d for d in read_rows(ROOT / "data/reuse_v2/development.jsonl")
            if d["corpus"] == corpus]
    return st.median(u.words for d in docs for u in source_view(d["id"], d["source"]).units)


# ---------------------------------------------------------------------- rule

def geometric_grid(low: int, high: int, count: int, step: int) -> list[int]:
    ratio = (high / low) ** (1 / (count - 1))
    raw = [low * ratio ** i for i in range(count)]
    return sorted({max(step, int(round(x / step)) * step) for x in raw})


def apply_rule(table: dict, rule: dict, sentence: dict) -> dict:
    regions = {}
    for corpus, rows in table.items():
        binding = [r["budget"] for r in rows if r["binding_share"] >= rule["binding_share"]]
        useful = [r["budget"] for r in rows
                  if r["R"] is not None and r["R"] >= rule["retention_floor"]]
        floor_ok = rule["min_sentence_multiple"] * sentence[corpus]
        b_max = max(binding) if binding else None
        b_min = min((b for b in useful if b >= floor_ok), default=None)
        has_region = b_min is not None and b_max is not None and b_min < b_max
        ceiling = None
        if b_max is not None:
            top = next(r for r in rows if r["budget"] == b_max)
            ceiling = top["R"] is not None and top["R"] > rule["retention_ceiling"]
        regions[corpus] = {"B_min": b_min, "B_max": b_max, "has_region": has_region,
                           "granularity_floor": floor_ok, "ceiling_warning": ceiling}
    headline = [c for c, r in regions.items() if r["has_region"]]
    decision = {"regions": regions, "headline": headline, "grid": None, "flag": None}
    if not headline:
        decision["flag"] = "no corpus has a region where the cap binds and R >= floor"
        return decision
    low = max(regions[c]["B_min"] for c in headline)
    high = min(regions[c]["B_max"] for c in headline)
    if len(headline) > 1 and low >= high:
        decision["flag"] = ("both corpora have regions but they do not overlap; "
                            "choose the headline corpus by hand")
        return decision
    grid = geometric_grid(low, high, rule["count"], rule["round_to"])
    if len(grid) < rule["count"]:
        decision["flag"] = (f"region [{low}, {high}] is too narrow for {rule['count']} "
                            f"distinct budgets at a step of {rule['round_to']}")
    decision["grid"] = grid
    return decision


# -------------------------------------------------------------------- report

def build_table(uncapped, messages, answers, candidates) -> dict:
    table = {}
    for corpus in CORPORA:
        ans = [a for a in answers if a["corpus"] == corpus]
        empty = doc_means(ans, CLOSED_BOOK)
        full = doc_means(ans, "full_source")
        evid = doc_means(ans, "annotated_evidence")
        nat_gen = natural_lengths(uncapped, corpus, "generic")
        nat_cond = natural_lengths(uncapped, corpus, "conditioned")
        rows = []
        for cap in candidates:
            gen = doc_means(ans, GENERIC, cap)
            r = retention(gen, empty, full)
            lo, hi = retention_ci(gen, empty, full) if r is not None else (None, None)
            gm = [m for m in messages if m["corpus"] == corpus and m["cap"] == cap
                  and m["policy"] == GENERIC and m["valid"]]
            cm = [m for m in messages if m["corpus"] == corpus and m["cap"] == cap
                  and m["policy"] == CONDITIONED and m["valid"]]
            rows.append({
                "budget": cap,
                "binding_share": sum(n > cap for n in nat_gen) / len(nat_gen),
                "U_generic": st.mean(gen.values()) if gen else None,
                "R": r, "R_lo": lo, "R_hi": hi,
                "generic_words_median": st.median(m["delivered_words"] for m in gm) if gm else None,
                "generic_fill_median": st.median(m["fill_ratio"] for m in gm) if gm else None,
                "generic_truncated": sum(m["truncated_words"] > 0 for m in gm),
                "conditioned_words_median": st.median(m["delivered_words"] for m in cm) if cm else None,
                "conditioned_fill_median": st.median(m["fill_ratio"] for m in cm) if cm else None,
                "conditioned_truncated": sum(m["truncated_words"] > 0 for m in cm),
            })
        table[corpus] = {
            "rows": rows,
            "U_closed_book": st.mean(empty.values()) if empty else None,
            "U_full_source": st.mean(full.values()) if full else None,
            "U_annotated_evidence": st.mean(evid.values()) if evid else None,
            "natural_generic_median": st.median(nat_gen) if nat_gen else None,
            "natural_conditioned_median": st.median(nat_cond) if nat_cond else None,
        }
    return table


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render(table, decision, sentence, rule, writer, run_dir) -> str:
    out = [f"# exp5_cap_only budget calibration — {writer}", "",
           f"Calibration run `{run_dir.name}`. Freeze rule from `configs/exp5_cap_only_config.yaml`, "
           "written before this run returned. Conditioned answers are discarded at load time, "
           "so no conditioned-minus-generic utility is computed here.", ""]
    for corpus in CORPORA:
        t = table[corpus]
        out += [f"## {corpus}", "",
                f"Closed book {fmt(t['U_closed_book'])} · full source {fmt(t['U_full_source'])} · "
                f"annotated evidence {fmt(t['U_annotated_evidence'])} F1. "
                f"Natural length, uncapped: generic median {fmt(t['natural_generic_median'], 0)} words, "
                f"conditioned {fmt(t['natural_conditioned_median'], 0)}. "
                f"Median source sentence {sentence[corpus]:.0f} words.", "",
                "| budget | binding share | U generic | R [95% CI] | generic words / fill | "
                "conditioned words / fill | truncated g / c |",
                "|---:|---:|---:|---|---|---|---:|"]
        for r in t["rows"]:
            ci = f"[{fmt(r['R_lo'], 2)}, {fmt(r['R_hi'], 2)}]" if r["R_lo"] is not None else ""
            out.append(
                f"| {r['budget']} | {r['binding_share']:.2f} | {fmt(r['U_generic'])} | "
                f"{fmt(r['R'], 2)} {ci} | {fmt(r['generic_words_median'], 0)} / "
                f"{fmt(r['generic_fill_median'], 2)} | {fmt(r['conditioned_words_median'], 0)} / "
                f"{fmt(r['conditioned_fill_median'], 2)} | "
                f"{r['generic_truncated']} / {r['conditioned_truncated']} |")
        reg = decision["regions"][corpus]
        out += ["", f"Region: B_min = {reg['B_min']}, B_max = {reg['B_max']}, "
                f"granularity floor {reg['granularity_floor']:.0f} words → "
                f"**{'useful region' if reg['has_region'] else 'no useful region'}**"
                + (" (ceiling warning at B_max)" if reg["ceiling_warning"] else ""), ""]
    out += ["## Decision", "",
            f"Headline corpora: {', '.join(decision['headline']) or 'none'}.",
            f"Proposed grid: {decision['grid']}." if decision["grid"] else "No grid proposed.",
            f"Flag: {decision['flag']}." if decision["flag"] else "No flags.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer", default="mistral24")
    parser.add_argument("--config", default="configs/exp5_cap_only_config.yaml")
    parser.add_argument("--run-dir", default="", help="explicit calibration run directory")
    parser.add_argument("--uncapped", default="", help="explicit uncapped messages.jsonl")
    parser.add_argument("--out", default="", help="explicit output directory")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    cfg = config(str(ROOT / args.config))
    rule = cfg["budgets"]["freeze_rule"]
    run_dir = Path(args.run_dir) if args.run_dir else find_calibration_run(args.writer)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = manifest["budgets"]
    uncapped = load_uncapped(args.writer, Path(args.uncapped) if args.uncapped else None)
    messages, answers = load_capped(run_dir, args.writer)
    sentence = {c: median_sentence_words(c) for c in CORPORA}
    table = build_table(uncapped, messages, answers, candidates)
    decision = apply_rule({c: t["rows"] for c, t in table.items()}, rule, sentence)
    out = Path(args.out) if args.out else ROOT / "results/exp5_cap_only/calibration" / args.writer
    out.mkdir(parents=True, exist_ok=True)
    report = render(table, decision, sentence, rule, args.writer, run_dir)
    (out / "calibration.md").write_text(report, encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps(
        {"run_dir": run_dir.name, "candidates": candidates, "rule": rule,
         "rule_in_run_manifest": manifest["config"]["budgets"].get("freeze_rule") == rule,
         "sentence_median_words": sentence, "table": table, "decision": decision},
        indent=2, default=lambda x: None if isinstance(x, float) and math.isinf(x) else x),
        encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
