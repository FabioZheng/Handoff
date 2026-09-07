"""Report and figures for Experiment 12.

Answers four questions in the order they were pre-registered:

1. What is the value of anticipation, and where does it cross zero?
2. Does regret degrade smoothly with prediction error, or hit a threshold?
3. Do retrieval-enabled policies move the frontier outward, or only along it?
4. Which of the four candidate strategies does the evidence support?
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

import pareto_frontier as pf  # noqa: E402
from llm import load_config  # noqa: E402

RESULTS = ROOT / "results" / "anticipatory_context"
RETRIEVAL_FAMILIES = {"pointer", "retrieval_only"}


def uses_retrieval(row: dict) -> bool:
    """Classify by behaviour, not by policy name.

    ``uncertainty_aware`` is the reason this matters: it *chooses* whether to
    retrieve based on the entropy of its estimate, so at high entropy it is a
    retrieval arm despite its family name. Splitting on the name put it in the
    static baseline, which inflated the static hypervolume and made the
    incremental gain from retrieval look like nothing.
    """
    for key in ("retrieval_k", "c_retrieval", "retrieval_rate"):
        if key in row and row.get(key) not in ("", None):
            try:
                if float(row[key]) > 0:
                    return True
            except ValueError:
                pass
    return False


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str, default=float("nan")) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Frontier: does retrievability change what is achievable?


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated (u_future, -cost) pairs; higher utility, lower cost is better."""
    front = []
    for u, c in points:
        if not any((u2 >= u and c2 <= c) and (u2 > u or c2 < c) for u2, c2 in points):
            front.append((u, c))
    return sorted(set(front), key=lambda p: p[1])


def hypervolume(points: list[tuple[float, float]], cost_max: float) -> float:
    """Area dominated in (utility, 1 - cost/cost_max) space, reference at origin."""
    coords = [{"u_now": u, "u_future": max(0.0, 1.0 - c / cost_max)} for u, c in points]
    if not coords:
        return 0.0
    for row in coords:
        row.update({"dataset": "d", "model": "m", "future_distribution": "f",
                    "metric": "judge_correct", "policy": "p", "budget_words": 0})
    return pf.hypervolume_2d(coords)


def frontier_shift(rows: list[dict], cfg: dict, budget_words: int,
                   gamma: float, resamples: int, seed: int) -> dict:
    """Incremental hypervolume from adding retrieval-enabled policies.

    The comparison that matters is *outward shift* versus *moving along*: a
    policy set that merely trades U_now for U_future picks a different point on
    the same curve and adds no hypervolume. Bootstrapped over contexts, because
    contexts are the independent unit.
    """
    beta = float(cfg["analysis"]["beta"])
    cell = [r for r in rows if int(num(r, "budget_words")) == budget_words]
    if not cell:
        return {}
    contexts = sorted({r["context_id"] for r in cell})
    by_context = defaultdict(list)
    for row in cell:
        by_context[row["context_id"]].append(row)

    def summarise(sample: list[str]) -> tuple[float, float]:
        acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
        retrieving: set[str] = set()
        for context in sample:
            for row in by_context[context]:
                cost = beta * num(row, "c_context") + gamma * num(row, "c_retrieval")
                acc[row["policy"]].append((num(row, "future_judge_correct"), cost))
                if uses_retrieval(row):
                    retrieving.add(row["policy"])
        static, everything = [], []
        for policy, values in acc.items():
            point = (float(np.mean([v for v, _ in values])),
                     float(np.mean([c for _, c in values])))
            everything.append(point)
            if policy not in retrieving:
                static.append(point)
        cost_max = max([c for _, c in everything] + [1.0])
        return hypervolume(static, cost_max), hypervolume(everything, cost_max)

    base_static, base_all = summarise(contexts)
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples)
    for draw in range(resamples):
        sample = [contexts[i] for i in rng.integers(0, len(contexts), len(contexts))]
        s, a = summarise(sample)
        deltas[draw] = a - s
    return {
        "budget_words": budget_words, "gamma": gamma,
        "hv_static": base_static, "hv_all": base_all,
        "delta_hv": base_all - base_static,
        "lo": float(np.quantile(deltas, 0.025)),
        "hi": float(np.quantile(deltas, 0.975)),
    }


# ---------------------------------------------------------------------------
# Figures


def fig_anticipation(value: list[dict], path: Path) -> None:
    """One panel per budget.

    Budgets must not share an axis here: each mismatch level appears once per
    budget, so pooling them draws a line between two different experiments and
    invents a slope that no arm has.
    """
    families = {"preserve": ("#c44e52", "o", "preserve (predict, no recall)"),
                "pointer": ("#4c72b0", "s", "pointer (predict + recall)"),
                "uncertainty_aware": ("#55a868", "^", "uncertainty-aware (adaptive)")}
    budgets = sorted({int(num(r, "budget_words")) for r in value})
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.4 * len(budgets), 5.0),
                             squeeze=False, sharey=True)
    for ax, budget_words in zip(axes[0], budgets):
        for family, (colour, marker, label) in families.items():
            pts = sorted(((num(r, "tv_mismatch"), num(r, "v_anticipation"),
                           num(r, "lo"), num(r, "hi"))
                          for r in value
                          if r["family"] == family
                          and int(num(r, "budget_words")) == budget_words
                          and r.get("tv_mismatch") not in ("", None)),
                         key=lambda t: t[0])
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
            ax.errorbar(x, y, yerr=err, color=colour, marker=marker, capsize=3,
                        lw=1.8, label=label)
        ax.axhline(0.0, color="#333", lw=1.2)
        ax.set_xlabel("mismatch   TV(P̂, P)")
        ax.set_title(f"{budget_words}-word budget")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("$V_{anticipation}$   (utility vs no prediction)")
    axes[0][-1].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("The value of anticipation as prediction degrades — "
                 "below zero, anticipating is worse than not anticipating", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_frontier(points: list[dict], cfg: dict, path: Path) -> None:
    gamma = float(cfg["analysis"]["gamma_default"])
    beta = float(cfg["analysis"]["beta"])
    budgets = sorted({int(num(p, "budget_words")) for p in points})
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.0 * len(budgets), 5.0), squeeze=False)
    for ax, budget_words in zip(axes[0], budgets):
        cell = [p for p in points if int(num(p, "budget_words")) == budget_words]
        static = [p for p in cell if not uses_retrieval(p)]
        recall = [p for p in cell if uses_retrieval(p)]
        for group, colour, label in ((static, "#c44e52", "static compression"),
                                     (recall, "#4c72b0", "retrieval-enabled")):
            xs = [beta * num(p, "c_context") + gamma * num(p, "c_retrieval") for p in group]
            ys = [num(p, "u_future") for p in group]
            ax.scatter(xs, ys, color=colour, s=46, label=label, zorder=3, alpha=0.85)
            front = pareto_front(list(zip(ys, xs)))
            if front:
                ax.plot([c for _, c in front], [u for u, _ in front], color=colour,
                        lw=1.6, ls="--", alpha=0.8)
        ax.set_xlabel(f"cost   C = {beta:g}·context + {gamma:g}·retrieval  (words)")
        ax.set_ylabel("$U_{future}$ (true-P weighted)")
        ax.set_title(f"{budget_words}-word budget")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Does making evidence retrievable move the frontier outward?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_tradeoff(points: list[dict], path: Path) -> None:
    budgets = sorted({int(num(p, "budget_words")) for p in points})
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.2 * len(budgets), 5.0), squeeze=False)
    for ax, budget_words in zip(axes[0], budgets):
        cell = sorted((p for p in points if int(num(p, "budget_words")) == budget_words),
                      key=lambda p: num(p, "u_future"))
        names = [p["policy"] for p in cell]
        y = np.arange(len(cell))
        ax.barh(y - 0.2, [num(p, "u_now") for p in cell], height=0.38,
                color="#937860", label="$U_{now}$")
        ax.barh(y + 0.2, [num(p, "u_future") for p in cell], height=0.38,
                color="#4c72b0", label="$U_{future}$")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("judged accuracy")
        ax.set_title(f"{budget_words}-word budget")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Present and future utility by policy", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> int:
    cfg = load_config(ROOT / "anticipatory_context_config.yaml")
    points = read_csv("policy_points.csv")
    value = read_csv("anticipation_value.csv")
    rows = read_csv("rotation_rows.csv")

    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig_anticipation(value, figures / "fig1_value_of_anticipation.png")
    fig_frontier(points, cfg, figures / "fig2_retrieval_frontier.png")
    fig_tradeoff(points, figures / "fig3_policy_tradeoff.png")

    resamples = int(cfg["analysis"]["bootstrap_resamples"])
    seed = int(cfg["analysis"]["bootstrap_seed"])
    shifts = []
    for budget_words in sorted({int(num(p, "budget_words")) for p in points}):
        for gamma in cfg["analysis"]["gamma_grid"]:
            shift = frontier_shift(rows, cfg, budget_words, float(gamma),
                                   min(resamples, 2000), seed)
            if shift:
                shifts.append(shift)
    with open(RESULTS / "frontier_shift.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shifts[0]))
        writer.writeheader()
        writer.writerows(shifts)

    summary = {
        "figures": ["fig1_value_of_anticipation", "fig2_retrieval_frontier",
                    "fig3_policy_tradeoff"],
        "frontier_shift_cells": len(shifts),
        "positive_shift_cells": sum(1 for s in shifts if s["lo"] > 0),
        "harmful_anticipation_arms": sum(1 for r in value if r["harmful"] == "True"),
        "helpful_anticipation_arms": sum(1 for r in value if r["helpful"] == "True"),
    }
    (RESULTS / "render_manifest.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
