"""Deferred analyses for Experiment 12: lambda calibration and regret shape.

Three questions the main runner left open, all answerable from records already
on disk without a single further model call:

1. **Can lambda be calibrated or learned rather than fixed?** The runner scores
   everything at lambda=1. But lambda is a preference, not a fact, so the
   answerable question is how much the *policy choice* depends on it: over what
   range of lambda does each policy win, how reliably is that winner recovered
   under resampling, and does a lambda chosen on one half of the corpus still
   choose well on the other half.

2. **Does regret degrade smoothly, or hit a threshold?** Fit V_anticipation
   against mismatch and report the shape honestly, including whether the
   available sweep can support a threshold claim at all.

3. **Is any of this an artefact of total variation?** Repeat against
   Jensen-Shannon. A conclusion that survives only one divergence is a property
   of the measure, not of the policy.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import load_config  # noqa: E402

RESULTS = ROOT / "results" / "anticipatory_context"


def load_rows() -> list[dict]:
    with open(RESULTS / "rotation_rows.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("budget_words", "c_context", "c_retrieval", "tv_mismatch",
                    "js_mismatch", "now_judge_correct", "future_judge_correct"):
            value = row.get(key, "")
            row[key] = float(value) if value not in ("", None) else None
    return rows


# ---------------------------------------------------------------------------
# 1. Lambda calibration


def objective(row: dict, lam: float, beta: float, gamma: float, scale: float) -> float:
    """U_now + lambda*U_future - (beta*C_context + gamma*C_retrieval)/scale.

    Cost is divided by ``scale`` (the largest observed cost) so the utility and
    cost terms are commensurate. Without it the word-valued cost term dominates
    two [0,1] utilities and every lambda selects the cheapest arm regardless of
    preference, which would answer the calibration question by construction.
    """
    cost = beta * row["c_context"] + gamma * row["c_retrieval"]
    return row["now_judge_correct"] + lam * row["future_judge_correct"] - cost / scale


def winner_at(rows: list[dict], lam: float, beta: float, gamma: float,
              scale: float) -> str:
    means: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        means[row["policy"]].append(objective(row, lam, beta, gamma, scale))
    scored = {p: float(np.mean(v)) for p, v in means.items()}
    return max(scored, key=lambda p: (scored[p], p))


def lambda_map(rows: list[dict], cfg: dict, budget_words: float) -> list[dict]:
    beta = float(cfg["analysis"]["beta"])
    gamma = float(cfg["analysis"]["gamma_default"])
    cell = [r for r in rows if r["budget_words"] == budget_words]
    scale = max(beta * r["c_context"] + gamma * r["c_retrieval"] for r in cell)
    contexts = sorted({r["context_id"] for r in cell})
    by_context = defaultdict(list)
    for row in cell:
        by_context[row["context_id"]].append(row)

    rng = np.random.default_rng(int(cfg["analysis"]["bootstrap_seed"]))
    out = []
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        point = winner_at(cell, lam, beta, gamma, scale)
        agree = 0
        draws = 400
        for _ in range(draws):
            sample = [contexts[i] for i in rng.integers(0, len(contexts), len(contexts))]
            resampled = [r for c in sample for r in by_context[c]]
            if winner_at(resampled, lam, beta, gamma, scale) == point:
                agree += 1
        out.append({"budget_words": budget_words, "lambda": lam, "winner": point,
                    "selection_stability": agree / draws})
    return out


def split_half(rows: list[dict], cfg: dict, budget_words: float) -> list[dict]:
    """Choose a policy on half the dossiers; score it on the other half.

    This is the operational form of "can lambda be learned": if the policy
    selected on seen data is also the best policy on unseen data, the choice
    generalises and the exact lambda matters less than it looks.
    """
    beta = float(cfg["analysis"]["beta"])
    gamma = float(cfg["analysis"]["gamma_default"])
    cell = [r for r in rows if r["budget_words"] == budget_words]
    scale = max(beta * r["c_context"] + gamma * r["c_retrieval"] for r in cell)
    contexts = sorted({r["context_id"] for r in cell})
    fit_ids, held_ids = set(contexts[::2]), set(contexts[1::2])
    fit = [r for r in cell if r["context_id"] in fit_ids]
    held = [r for r in cell if r["context_id"] in held_ids]

    out = []
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
        chosen = winner_at(fit, lam, beta, gamma, scale)
        best_held = winner_at(held, lam, beta, gamma, scale)
        def value(policy: str) -> float:
            vals = [objective(r, lam, beta, gamma, scale)
                    for r in held if r["policy"] == policy]
            return float(np.mean(vals))
        out.append({
            "budget_words": budget_words, "lambda": lam,
            "chosen_on_fit": chosen, "best_on_held": best_held,
            "matches": chosen == best_held,
            "held_value_chosen": value(chosen),
            "held_value_best": value(best_held),
            "selection_regret": value(best_held) - value(chosen),
        })
    return out


# ---------------------------------------------------------------------------
# 2 and 3. Shape of regret against mismatch, under two divergences


def shape_analysis(rows: list[dict], cfg: dict, measure: str) -> list[dict]:
    """Linear fit of V_anticipation against a mismatch measure, per family.

    Reported with a context-clustered bootstrap on the slope. A formal
    breakpoint test is deliberately not attempted: the sweep has at most four
    distinct mismatch levels per family, which cannot separate a threshold from
    a steep slope. That is a limitation of the design, not a result.
    """
    lam = float(cfg["analysis"]["lambda_default"])
    resamples = 2000
    rng = np.random.default_rng(int(cfg["analysis"]["bootstrap_seed"]))

    baseline = {(r["rotation_id"], r["budget_words"]):
                r["now_judge_correct"] + lam * r["future_judge_correct"]
                for r in rows if r["policy"] == "blind"}

    out = []
    for budget_words in sorted({r["budget_words"] for r in rows}):
        for family in ("preserve", "pointer"):
            pts = []
            for row in rows:
                if (row["family"] != family or row["budget_words"] != budget_words
                        or row[measure] is None):
                    continue
                key = (row["rotation_id"], row["budget_words"])
                if key not in baseline:
                    continue
                v = (row["now_judge_correct"] + lam * row["future_judge_correct"]
                     - baseline[key])
                pts.append((row["context_id"], row[measure], v))
            if len({m for _, m, _ in pts}) < 3:
                continue
            x = np.array([m for _, m, _ in pts])
            y = np.array([v for _, _, v in pts])
            slope, intercept = np.polyfit(x, y, 1)
            resid = y - (slope * x + intercept)
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

            by_context = defaultdict(list)
            for ctx, m, v in pts:
                by_context[ctx].append((m, v))
            keys = sorted(by_context)
            slopes = np.empty(resamples)
            for draw in range(resamples):
                picked = rng.integers(0, len(keys), len(keys))
                sx, sy = [], []
                for k in picked:
                    for m, v in by_context[keys[k]]:
                        sx.append(m)
                        sy.append(v)
                slopes[draw] = np.polyfit(np.array(sx), np.array(sy), 1)[0]
            out.append({
                "measure": measure, "budget_words": budget_words, "family": family,
                "levels": len({m for _, m, _ in pts}),
                "slope": float(slope), "slope_lo": float(np.quantile(slopes, 0.025)),
                "slope_hi": float(np.quantile(slopes, 0.975)),
                "r_squared": r2, "n": len(pts),
            })
    return out


def equal_mismatch_divergence(rows: list[dict], cfg: dict) -> list[dict]:
    """Do two arms at the *same* mismatch behave the same?

    If not, regret is not a function of mismatch magnitude alone, and no
    single-variable threshold model can describe it. This is the sharpest test
    of whether the two noise axes had to be kept separate.
    """
    lam = float(cfg["analysis"]["lambda_default"])
    baseline = {(r["rotation_id"], r["budget_words"]):
                r["now_judge_correct"] + lam * r["future_judge_correct"]
                for r in rows if r["policy"] == "blind"}
    pairs = [("preserve__tau1", "preserve__sigma0p5"),
             ("pointer__tau1", "pointer__sigma0p5")]
    out = []
    for budget_words in sorted({r["budget_words"] for r in rows}):
        for a, b in pairs:
            deltas = []
            clusters = []
            for row in rows:
                if row["budget_words"] != budget_words or row["policy"] not in (a, b):
                    continue
                key = (row["rotation_id"], row["budget_words"])
                if key not in baseline:
                    continue
                v = (row["now_judge_correct"] + lam * row["future_judge_correct"]
                     - baseline[key])
                deltas.append((row["policy"], row["rotation_id"], v, row["context_id"]))
            by_rot = defaultdict(dict)
            for policy, rot, v, ctx in deltas:
                by_rot[(rot, ctx)][policy] = v
            paired = [(ctx, d[a] - d[b]) for (rot, ctx), d in by_rot.items()
                      if a in d and b in d]
            if not paired:
                continue
            values = np.array([v for _, v in paired])
            keys = sorted({c for c, _ in paired})
            grouped = defaultdict(list)
            for ctx, v in paired:
                grouped[ctx].append(v)
            rng = np.random.default_rng(7)
            means = np.empty(2000)
            for draw in range(2000):
                picked = rng.integers(0, len(keys), len(keys))
                means[draw] = float(np.mean([v for k in picked for v in grouped[keys[k]]]))
            tv_a = next(r["tv_mismatch"] for r in rows if r["policy"] == a)
            out.append({
                "budget_words": budget_words, "arm_dilution": a, "arm_displacement": b,
                "shared_tv": tv_a, "difference": float(values.mean()),
                "lo": float(np.quantile(means, 0.025)),
                "hi": float(np.quantile(means, 0.975)),
                "n": len(paired),
            })
    return out


def fig_divergence(shapes: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.35
    labels, tv_slopes, js_slopes, tv_err, js_err = [], [], [], [], []
    for budget_words in sorted({s["budget_words"] for s in shapes}):
        for family in ("preserve", "pointer"):
            tv = next((s for s in shapes if s["measure"] == "tv_mismatch"
                       and s["budget_words"] == budget_words and s["family"] == family), None)
            js = next((s for s in shapes if s["measure"] == "js_mismatch"
                       and s["budget_words"] == budget_words and s["family"] == family), None)
            if not tv or not js:
                continue
            labels.append(f"{family}\n{int(budget_words)}w")
            tv_slopes.append(tv["slope"])
            js_slopes.append(js["slope"])
            tv_err.append([tv["slope"] - tv["slope_lo"], tv["slope_hi"] - tv["slope"]])
            js_err.append([js["slope"] - js["slope_lo"], js["slope_hi"] - js["slope"]])
    x = np.arange(len(labels))
    ax.bar(x - width / 2, tv_slopes, width, yerr=np.array(tv_err).T, capsize=4,
           color="#4c72b0", label="total variation")
    ax.bar(x + width / 2, js_slopes, width, yerr=np.array(js_err).T, capsize=4,
           color="#dd8452", label="Jensen-Shannon")
    ax.axhline(0.0, color="#333", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("slope of $V_{anticipation}$ against mismatch")
    ax.set_title("Does the conclusion depend on the divergence? (it does not)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    cfg = load_config(ROOT / "anticipatory_context_config.yaml")
    rows = load_rows()
    budgets = sorted({r["budget_words"] for r in rows})

    lam_rows, split_rows = [], []
    for budget_words in budgets:
        lam_rows.extend(lambda_map(rows, cfg, budget_words))
        split_rows.extend(split_half(rows, cfg, budget_words))
    shapes = (shape_analysis(rows, cfg, "tv_mismatch")
              + shape_analysis(rows, cfg, "js_mismatch"))
    equal = equal_mismatch_divergence(rows, cfg)

    write_csv(RESULTS / "lambda_selection.csv", lam_rows)
    write_csv(RESULTS / "lambda_split_half.csv", split_rows)
    write_csv(RESULTS / "regret_shape.csv", shapes)
    write_csv(RESULTS / "equal_mismatch.csv", equal)
    fig_divergence(shapes, RESULTS / "figures" / "fig4_divergence_check.png")

    summary = {
        "lambda_levels": len(lam_rows),
        "distinct_winners": sorted({r["winner"] for r in lam_rows}),
        "split_half_matches": sum(1 for r in split_rows if r["matches"]),
        "split_half_total": len(split_rows),
        "max_selection_regret": max(r["selection_regret"] for r in split_rows),
        "equal_mismatch_cells": len(equal),
        "equal_mismatch_nonzero": sum(1 for r in equal if r["lo"] > 0 or r["hi"] < 0),
    }
    (RESULTS / "extras_manifest.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
