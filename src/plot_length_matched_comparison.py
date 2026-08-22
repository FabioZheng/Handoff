"""Plot the natural-length vs length-matched same-passage conditioning runs together.

Analysis-only: reads the two completed result sets and makes no API calls.
Both runs share the identical 10 SQuAD same-passage pairs and prompts, differing
only in whether both arms are given the same explicit word-budget directive
(see length_directive() in run_summary_generalization.py). Without that
control, conditioned and generic self-select very different summary lengths on
the same context, so a held-out accuracy gap cannot be separated from "one arm
simply wrote more" -- this figure puts both runs on the same axes so that
separation is visible directly.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"conditioned": "#d95f02", "generic": "#1b9e77", "direct": "#7570b3"}
STYLES = {"natural": "-", "matched": "--"}
RUN_LABELS = {"natural": "natural length", "matched": "length-matched (~400 words)"}


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run both configs before plotting.")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def make_plot(natural: list[dict], matched: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_run = {"natural": natural, "matched": matched}
    plot_metrics = [("f1", "Token F1"), ("judge_correct", "LLM-judge accuracy")]
    titles = ("Conditioning question A", "Unrelated held-out question B")

    # Same marker-area size encoding as run_summary_generalization.py's own
    # plot: proportional to mean characters actually fed to the answerer.
    all_chars = [as_float(r, "handoff_characters_mean") for rows in by_run.values() for r in rows
                if r.get("handoff_characters_mean")]
    max_chars = max(all_chars) if all_chars else 1.0
    max_area, min_area = 500.0, 12.0
    size_scale = max_area / max_chars

    def marker_area(row: dict) -> float:
        chars = row.get("handoff_characters_mean")
        return max_area if not chars else max(min_area, size_scale * as_float(row, "handoff_characters_mean"))

    fig, axes = plt.subplots(len(plot_metrics), 2, figsize=(11, 4.5 * len(plot_metrics)),
                             sharey="row", squeeze=False)

    for row_idx, (metric, label) in enumerate(plot_metrics):
        for col_idx, (query_type, title) in enumerate(zip(("target", "heldout"), titles)):
            axis = axes[row_idx][col_idx]
            direct = next(r for r in natural if r["mode"] == "direct" and r["query_type"] == query_type)
            axis.scatter([0], [as_float(direct, metric)], s=marker_area(direct), color=COLORS["direct"],
                        edgecolors="white", linewidths=0.6, label="direct context", zorder=3)
            for run_key, rows in by_run.items():
                for mode in ("conditioned", "generic"):
                    subset = sorted([r for r in rows if r["mode"] == mode and r["query_type"] == query_type],
                                    key=lambda r: int(r["depth"]))
                    if not subset:
                        continue
                    x = [int(r["depth"]) for r in subset]
                    y = [as_float(r, metric) for r in subset]
                    axis.plot(x, y, marker="", linewidth=2, linestyle=STYLES[run_key],
                             color=COLORS[mode], label=f"{mode}, {RUN_LABELS[run_key]}", zorder=2)
                    axis.scatter(x, y, s=[marker_area(r) for r in subset], color=COLORS[mode],
                                edgecolors="white", linewidths=0.6, zorder=3)
            if row_idx == 0:
                axis.set_title(title)
            if row_idx == len(plot_metrics) - 1:
                axis.set_xlabel("Compression handoffs")
            axis.set_xticks(range(0, 11))
            axis.grid(alpha=0.25)
        axes[row_idx][0].set_ylabel(label)

    fig.suptitle("Natural-length vs length-matched summaries: does length explain the held-out effect?", y=1.02)
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985), fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.text(0.5, 0.005,
              f"Solid = natural length (arms self-select summary length). Dashed = both arms told to aim for "
              f"~400 words. Dot area ∝ mean characters fed to the answerer at that point "
              f"(largest ≈{max_chars:,.0f} characters, {min_area:.0f}pt² floor). "
              f"Same 10 SQuAD same-passage pairs and prompts in both runs.",
              ha="center", fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    natural = read_metrics(ROOT / "results/squad_same_passage/n10/metrics.csv")
    matched = read_metrics(ROOT / "results/squad_same_passage_matched/n10/metrics.csv")
    output = ROOT / "results/squad_same_passage/n10/length_matched_comparison.png"
    make_plot(natural, matched, output)
    print(f"[plot] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
