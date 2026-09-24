"""Render the two manuscript-specific figures for the bounded-handoff paper.

The source data are the audited Experiment 14 CSV artifacts.  This module is
deliberately small: it reads rather than recomputes the estimates, so the
paper figures cannot silently differ from the published report tables.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "compression_mechanism" / "relation_dossiers" / "n16"
OUT = ROOT / "results" / "paper_frame_a" / "figures"

COLORS = {
    "summary": "#c25102",
    "lm": "#08519c",
    "lexical": "#54278f",
}
LABELS = {
    "summary": "Abstractive summary",
    "lm": "LM-scored elimination",
    "lexical": "BM25 elimination",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selected(rows: list[dict[str, str]], **where: str) -> dict[str, str]:
    hits = [row for row in rows if all(row.get(key) == str(value) for key, value in where.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {where}; found {len(hits)}")
    return hits[0]


def errorbar(ax, xs, rows, *, color: str, label: str, field: str = "mean") -> None:
    ys = [float(row[field]) for row in rows]
    lows = [y - float(row["lo"]) for y, row in zip(ys, rows)]
    highs = [float(row["hi"]) - y for y, row in zip(ys, rows)]
    ax.errorbar(xs, ys, yerr=[lows, highs], color=color, marker="o", linewidth=2,
                capsize=3, label=label)


def render_summary_budget_utility(metrics: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [20, 40, 80, 160]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), sharex=True, sharey=True)
    for endpoint, ax, title in zip(
        ("now", "future"), axes,
        ("Present-query utility $U_{now}$", "Future-query utility $U_{future}$"),
    ):
        for policy, color, label in (
            ("generic", "#237f6e", "Query-agnostic summary"),
            ("conditioned", COLORS["summary"], "Query-aware summary"),
        ):
            rows = [selected(metrics, policy=policy, budget_words=budget,
                             kind="utility", endpoint=endpoint, metric="judge_correct")
                    for budget in budgets]
            errorbar(ax, budgets, rows, color=color, label=label)
        ax.set_title(title)
        ax.set_xticks(budgets)
        ax.set_xlabel("Delivered-word cap")
        ax.set_ylim(-0.05, 1.07)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("LLM-judged accuracy")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Task conditioning reallocates a bounded abstractive message", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "summary_budget_utility.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "summary_budget_utility.pdf", bbox_inches="tight")
    plt.close(fig)


def contrast_row(rows: list[dict[str, str]], family: str, budget: int, endpoint: str) -> dict[str, str]:
    comparison = {
        "summary": "conditioned_minus_generic",
        "lm": "lm_conditioned_minus_lm_generic",
        "lexical": "nonllm_conditioned_minus_nonllm_generic",
    }[family]
    return selected(rows, comparison=comparison, budget_words=budget, kind="utility",
                    endpoint=endpoint, metric="judge_correct")


def render_conditioning_deltas(contrasts: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [20, 40, 80, 160]
    fig, axes = plt.subplots(1, 2, figsize=(11.25, 4.4), sharex=True, sharey=False)
    for endpoint, ax, title in zip(
        ("now", "future"), axes,
        (r"$\Delta U_{now}$: conditioned minus matched control",
         r"$\Delta U_{future}$: conditioned minus matched control"),
    ):
        ax.axhline(0, color="#4a4a4a", linewidth=1)
        for family in ("summary", "lm", "lexical"):
            rows = [contrast_row(contrasts, family, budget, endpoint) for budget in budgets]
            errorbar(ax, budgets, rows, color=COLORS[family], label=LABELS[family], field="delta")
        ax.set_title(title)
        ax.set_xticks(budgets)
        ax.set_xlabel("Delivered-word cap")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Paired change in LLM-judged accuracy")
    axes[0].set_ylim(-0.48, 0.92)
    axes[1].set_ylim(-0.48, 0.92)
    axes[1].legend(loc="lower right", frameon=False)
    fig.text(0.5, -0.02,
             "Points and bars: 16-dossier paired estimates with 95% percentile bootstrap intervals. "
             "The 20-word selection points are shown but are a sentence-granularity stress test.",
             ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "conditioning_deltas.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "conditioning_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(DATA / "metrics.csv")
    contrasts = read_csv(DATA / "contrasts.csv")
    render_summary_budget_utility(metrics)
    render_conditioning_deltas(contrasts)


if __name__ == "__main__":
    main()
