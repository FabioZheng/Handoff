"""Plot matched serial-handoff runs with and without question conditioning.

This is analysis-only: it reads the completed stage-level summaries produced by
``run_chain.py`` and makes no API calls.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
    metrics = [("f1", "Token F1"), ("bertscore_f1", "BERTScore F1")]
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
                        x, y, yerr=[lower, upper], marker="o", linewidth=2,
                        capsize=3, label=variant, color=COLORS[variant],
                    )
                if row_idx == 0:
                    axis.set_title(f"{dataset}: {condition_label}")
                if row_idx == 1:
                    axis.set_xlabel("Number of compression handoffs")
                if dataset_idx == 0 and condition_idx == 0:
                    axis.set_ylabel(label)
                axis.set_xticks(range(0, 11))
                axis.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Evidence length", loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Effect of question conditioning on repeated handoff degradation", y=1.04)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare conditioned and generic handoff chains")
    parser.add_argument("--conditioned", type=Path, default=ROOT / "results/chain/stage_metrics.csv")
    parser.add_argument("--generic", type=Path,
                        default=ROOT / "results/chain_question_omitted/stage_metrics.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/chain_question_omitted/conditioning_comparison.png")
    args = parser.parse_args()
    make_plot(read_metrics(args.conditioned), read_metrics(args.generic), args.output)
    print(f"[conditioning-comparison] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
