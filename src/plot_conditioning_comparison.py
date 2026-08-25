"""Plot matched serial-handoff runs with and without question conditioning.

This is analysis-only: it reads the completed stage-level summaries produced by
``run_chain.py`` and makes no API calls.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_chain import context_display_label


ROOT = Path(__file__).resolve().parent.parent
COLORS = {"short": "#1b9e77", "medium": "#d95f02", "full": "#7570b3"}
DATASETS = ["musique", "hotpotqa"]
VARIANTS = ["short", "medium", "full"]
CONDITIONS = [
    ("question-conditioned", "Question-conditioned"),
    ("question-omitted", "Question omitted\n(matched instructions)"),
]


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Finish the generic chain and its scoring pass before plotting."
        )
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def make_plot(conditioned: list[dict], generic: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_condition = {"question-conditioned": conditioned, "question-omitted": generic}
    metrics = [("f1", "Token F1"), ("judge_correct", "LLM-judge accuracy")]
    # Same size encoding as run_chain.py's make_plot(): marker area proportional
    # to the mean size (characters) of whatever text fed the answerer at that
    # point, one scale for the whole figure so area is comparable across both
    # conditions and both datasets.
    all_chars = [as_float(r, "handoff_characters_mean") for rows in by_condition.values() for r in rows
                if "handoff_characters_mean" in r]
    max_chars = max(all_chars) if all_chars else 1.0
    max_area, min_area = 900.0, 15.0
    size_scale = max_area / max_chars

    def marker_area(row: dict) -> float:
        chars = row.get("handoff_characters_mean")
        return max_area if chars is None else max(min_area, size_scale * as_float(row, "handoff_characters_mean"))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey="row", squeeze=False)

    for row_idx, (metric, label) in enumerate(metrics):
        for dataset_idx, dataset in enumerate(DATASETS):
            for condition_idx, (condition_key, condition_label) in enumerate(CONDITIONS):
                axis = axes[row_idx][dataset_idx * 2 + condition_idx]
                rows = by_condition[condition_key]
                for variant in VARIANTS:
                    subset = sorted(
                        [r for r in rows if r["dataset"] == dataset
                         and r["context_variant"] == variant and metric in r],
                        key=lambda r: int(r["depth"]),
                    )
                    if not subset:
                        continue
                    x = [int(r["depth"]) for r in subset]
                    y = [as_float(r, metric) for r in subset]
                    lower = [v - as_float(r, f"{metric}_lo") for r, v in zip(subset, y)]
                    upper = [as_float(r, f"{metric}_hi") - v for r, v in zip(subset, y)]
                    axis.errorbar(
                        x, y, yerr=[lower, upper], marker="", linewidth=2,
                        capsize=3, label=context_display_label(rows, dataset, variant),
                        color=COLORS[variant], zorder=2,
                    )
                    axis.scatter(x, y, s=[marker_area(r) for r in subset], color=COLORS[variant],
                                edgecolors="white", linewidths=0.6, zorder=3)
                if row_idx == 0:
                    axis.set_title(f"{dataset}: {condition_label}")
                if row_idx == 1:
                    axis.set_xlabel("Number of compression handoffs")
                if dataset_idx == 0 and condition_idx == 0:
                    axis.set_ylabel(label)
                if row_idx == 0 and condition_idx == 0:
                    axis.legend(title="Evidence composition", fontsize=7, title_fontsize=8, loc="best")
                axis.set_xticks(range(0, 11))
                axis.grid(alpha=0.25)

    fig.suptitle("Effect of question conditioning on repeated handoff degradation", y=.99)
    fig.tight_layout(rect=(0, 0.02, 1, .96))
    fig.text(0.5, 0.005,
              f"Marker area ∝ mean characters in the answerer's input at that point "
              f"(full context at depth 0, else the handoff text) — smallest marker "
              f"{min_area:.0f}pt² floor, largest ≈{max_chars:,.0f} characters.",
              ha="center", fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare conditioned and generic handoff chains")
    parser.add_argument("--conditioned", type=Path, default=ROOT / "results/chain/stage_metrics.csv")
    parser.add_argument("--generic", type=Path,
                        default=ROOT / "results/chain_generic/stage_metrics.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/chain_generic/conditioning_comparison.png")
    args = parser.parse_args()
    make_plot(read_metrics(args.conditioned), read_metrics(args.generic), args.output)
    print(f"[conditioning-comparison] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
