"""Render a concise, plain-language brief for Experiment 10.

The full cross-corpus report is intentionally exhaustive.  This renderer makes
the reader-facing layer above it: three simplified figures and a short Markdown
report with no raw result tables.  Every quoted value and plotted point is read
from the current result CSVs so the brief cannot silently drift from the audit
report.

Run from the repository root with::

    python src/analysis/render_regret_brief.py
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "communication_regret"
ASSET_ROOT = RESULT_ROOT / "plain_language"
REPORT_PATH = RESULT_ROOT / "PLAIN_LANGUAGE_REPORT.md"

METRIC = "judge_correct"
BUDGETS = (20, 40, 80, 160)

# Okabe-Ito-derived colours remain distinguishable for common colour-vision
# deficiencies.  Endpoint colours are held fixed across corpora and figures.
CURRENT_COLOUR = "#D55E00"
FUTURE_COLOUR = "#0072B2"
GENERIC_COLOUR = "#0072B2"
ORACLE_COLOUR = "#009E73"
NATURAL_COLOUR = "#0072B2"
DESIGNED_COLOUR = "#009E73"
GRID_COLOUR = "#D0D0D0"
TEXT_COLOUR = "#222222"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "_manifest" not in row:
            rows.append(row)
    return rows


def latest_run(dataset: str) -> Path:
    """Return the highest numeric n* result directory for one corpus."""

    candidates: list[tuple[int, Path]] = []
    for manifest in (RESULT_ROOT / dataset).glob("n*/manifest.json"):
        name = manifest.parent.name
        try:
            n = int(name.removeprefix("n"))
        except ValueError:
            continue
        candidates.append((n, manifest.parent))
    if not candidates:
        raise FileNotFoundError(f"No result run found for {dataset}")
    return max(candidates, key=lambda item: item[0])[1]


def one(rows: list[dict[str, str]], **where: object) -> dict[str, str]:
    matches = [
        row for row in rows
        if all(str(row.get(key)) == str(value) for key, value in where.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {where}, found {len(matches)}")
    return matches[0]


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def percentage(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}%"


def signed_points(value: float, digits: int = 0) -> str:
    quantum = Decimal("1").scaleb(-digits)
    rounded = Decimal(str(100 * value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:+.{digits}f}"


def mean_words(length_rows: list[dict[str, str]], policy: str, budget: int) -> float:
    row = one(length_rows, policy=policy, budget_words=budget)
    return number(row, "delivered_words_mean")


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOUR)
    ax.title.set_color(TEXT_COLOUR)
    ax.xaxis.label.set_color(TEXT_COLOUR)
    ax.yaxis.label.set_color(TEXT_COLOUR)


def asymmetric_error(row: dict[str, str], *, key: str) -> list[list[float]]:
    centre = number(row, key)
    return [[centre - number(row, "lo")], [number(row, "hi") - centre]]


def plot_tradeoff(corpora: list[dict[str, object]], output: Path) -> None:
    """Plot focused-minus-question-blind accuracy for both corpora."""

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
    endpoint_specs = (
        ("now", "Current known question", CURRENT_COLOUR, "o"),
        ("future", "Other possible questions", FUTURE_COLOUR, "s"),
    )
    x = list(range(len(BUDGETS)))

    for ax, corpus in zip(axes, corpora):
        rows = corpus["contrasts"]
        assert isinstance(rows, list)
        for endpoint, label, colour, marker in endpoint_specs:
            selected = [
                one(
                    rows,
                    comparison="conditioned_minus_generic",
                    budget_words=budget,
                    kind="utility",
                    endpoint=endpoint,
                    metric=METRIC,
                )
                for budget in BUDGETS
            ]
            means = [100 * number(row, "delta") for row in selected]
            lower = [100 * (number(row, "delta") - number(row, "lo")) for row in selected]
            upper = [100 * (number(row, "hi") - number(row, "delta")) for row in selected]
            ax.errorbar(
                x,
                means,
                yerr=[lower, upper],
                color=colour,
                marker=marker,
                markersize=6.5,
                linewidth=2.0,
                capsize=3,
                label=label,
                zorder=3,
            )
            for xpos, value in zip(x, means):
                ax.annotate(
                    signed_points(value / 100),
                    (xpos, value),
                    xytext=(0, -11),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color=colour,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4,
                          "alpha": 0.82},
                    zorder=5,
                )
        ax.axhline(0, color="#555555", linewidth=1.1)
        ax.set_xticks(x, [str(budget) for budget in BUDGETS])
        ax.set_xlabel("Handoff limit (words)")
        ax.set_title(str(corpus["label"]), fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(-52, 102)
        style_axis(ax)

    axes[0].set_ylabel("Change vs question-blind handoff\n(percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Focusing helps the current question, but hurts later questions",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOUR,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Whiskers show 95% context-level bootstrap intervals; labels are percentage-point changes.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.84), w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distance(relation_rows: list[dict[str, str]], output: Path) -> None:
    """Show what a focused handoff retains as later questions move away."""

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    relations = ("paraphrase", "same_entity", "same_topic", "orthogonal")
    relation_labels = (
        "Same answer,\nreworded",
        "Same\nsubject",
        "Related\nsubject",
        "Different\ntopic",
    )
    x = list(range(len(relations)))

    focused = [
        one(
            relation_rows,
            policy="conditioned",
            budget_words=160,
            relation=relation,
            kind="utility",
            metric=METRIC,
        )
        for relation in relations
    ]
    focused_means = [100 * number(row, "mean") for row in focused]
    focused_lower = [100 * (number(row, "mean") - number(row, "lo")) for row in focused]
    focused_upper = [100 * (number(row, "hi") - number(row, "mean")) for row in focused]
    axes[0].errorbar(
        x,
        focused_means,
        yerr=[focused_lower, focused_upper],
        fmt="o",
        color=CURRENT_COLOUR,
        markersize=8,
        capsize=4,
        linewidth=1.8,
        zorder=3,
    )
    for xpos, value in zip(x, focused_means):
        axes[0].annotate(
            f"{value:.0f}%",
            (xpos, value),
            xytext=(0, 9 if value < 90 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if value < 90 else "top",
            fontsize=9,
            color=CURRENT_COLOUR,
            fontweight="bold",
        )
    axes[0].set_xticks(x, relation_labels)
    axes[0].set_xlabel("How the later question relates to the known one")
    axes[0].set_title("Focused handoff, 160 words", fontsize=12, fontweight="bold", pad=10)

    policy_specs = (
        ("conditioned", "Focused on current question", CURRENT_COLOUR, "o", "-"),
        ("generic", "Question-blind", GENERIC_COLOUR, "s", "--"),
        ("oracle", "Shown all questions", ORACLE_COLOUR, "^", "-."),
    )
    budget_x = list(range(len(BUDGETS)))
    label_offsets = {"conditioned": -1.5, "generic": 0.0, "oracle": 0.0}
    for policy, label, colour, marker, linestyle in policy_specs:
        selected = [
            one(
                relation_rows,
                policy=policy,
                budget_words=budget,
                relation="orthogonal",
                kind="utility",
                metric=METRIC,
            )
            for budget in BUDGETS
        ]
        means = [100 * number(row, "mean") for row in selected]
        lower = [100 * (number(row, "mean") - number(row, "lo")) for row in selected]
        upper = [100 * (number(row, "hi") - number(row, "mean")) for row in selected]
        axes[1].errorbar(
            budget_x,
            means,
            yerr=[lower, upper],
            color=colour,
            marker=marker,
            linestyle=linestyle,
            markersize=6.5,
            linewidth=2.0,
            capsize=3,
            zorder=3,
        )
        final = means[-1]
        axes[1].annotate(
            f"{label}: {final:.0f}%",
            (budget_x[-1], final),
            xytext=(12, label_offsets[policy]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=colour,
            fontweight="bold",
            clip_on=False,
        )
    axes[1].set_xticks(budget_x, [str(budget) for budget in BUDGETS])
    axes[1].set_xlim(-0.15, 4.35)
    axes[1].set_xlabel("Handoff limit (words)")
    axes[1].set_title("Accuracy on a different-topic question", fontsize=12,
                      fontweight="bold", pad=10)

    for ax in axes:
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        style_axis(ax)
    axes[0].set_ylabel("Later questions answered correctly")
    fig.suptitle(
        "A focused handoff remembers nearby answers, not broad coverage",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOUR,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Designed fictional dossiers; whiskers show 95% context-level bootstrap intervals.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.91), w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_length_control(corpora: list[dict[str, object]], output: Path) -> None:
    """Plot focused-minus-question-blind delivered-word differences."""

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    x = list(range(len(BUDGETS)))
    specs = (
        (corpora[0], "Natural passages", NATURAL_COLOUR, "o", -0.06),
        (corpora[1], "Designed dossiers", DESIGNED_COLOUR, "s", 0.06),
    )
    for corpus, label, colour, marker, shift in specs:
        rows = corpus["length_deltas"]
        assert isinstance(rows, list)
        selected = [
            one(
                rows,
                comparison="conditioned_minus_generic",
                budget_words=budget,
                metric="delivered_words",
            )
            for budget in BUDGETS
        ]
        means = [number(row, "delta") for row in selected]
        lower = [number(row, "delta") - number(row, "lo") for row in selected]
        upper = [number(row, "hi") - number(row, "delta") for row in selected]
        shifted_x = [value + shift for value in x]
        ax.errorbar(
            shifted_x,
            means,
            yerr=[lower, upper],
            color=colour,
            marker=marker,
            markersize=6.5,
            linewidth=1.8,
            capsize=3,
            label=label,
            zorder=3,
        )
        for xpos, value in zip(shifted_x, means):
            ax.annotate(
                f"{value:+.1f}",
                (xpos, value),
                xytext=(0, 7 if value >= 0 else -9),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.5,
                color=colour,
            )
    ax.axhline(0, color="#555555", linewidth=1.1)
    ax.set_xticks(x, [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Handoff limit (words)")
    ax.set_ylabel("Focused minus question-blind\ndelivered words")
    ax.set_ylim(-4.8, 5.0)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax)
    fig.suptitle(
        "The compared handoffs used essentially the same number of words",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOUR,
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Zero means equal average length; whiskers show 95% context-level bootstrap intervals.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.98, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_report(
    corpora: list[dict[str, object]],
    relation_rows: list[dict[str, str]],
    natural_example: dict[str, object],
    designed_example: dict[str, object],
) -> str:
    natural, designed = corpora
    natural_manifest = natural["manifest"]
    designed_manifest = designed["manifest"]
    natural_root = natural["root"]
    designed_root = designed["root"]
    assert isinstance(natural_manifest, dict)
    assert isinstance(designed_manifest, dict)
    assert isinstance(natural_root, Path)
    assert isinstance(designed_root, Path)

    natural_rel = natural_root.relative_to(RESULT_ROOT).as_posix()
    designed_rel = designed_root.relative_to(RESULT_ROOT).as_posix()

    natural_contrasts = natural["contrasts"]
    designed_contrasts = designed["contrasts"]
    natural_metrics = natural["metrics"]
    natural_lengths = natural["length_audit"]
    interaction = natural["budget_interaction"]
    assert isinstance(natural_contrasts, list)
    assert isinstance(designed_contrasts, list)
    assert isinstance(natural_metrics, list)
    assert isinstance(natural_lengths, list)
    assert isinstance(interaction, list)

    def contrast(rows: list[dict[str, str]], budget: int, endpoint: str) -> dict[str, str]:
        return one(
            rows,
            comparison="conditioned_minus_generic",
            budget_words=budget,
            kind="utility",
            endpoint=endpoint,
            metric=METRIC,
        )

    natural_now = [number(contrast(natural_contrasts, budget, "now"), "delta")
                   for budget in BUDGETS]
    natural_future = [number(contrast(natural_contrasts, budget, "future"), "delta")
                      for budget in BUDGETS]
    designed_now_160 = number(contrast(designed_contrasts, 160, "now"), "delta")
    designed_future_160 = number(contrast(designed_contrasts, 160, "future"), "delta")

    gap_interaction = one(
        interaction,
        quantity="conditioning_gap_low_minus_high",
        policy="conditioned_minus_generic",
        metric=METRIC,
        low_budget_words=20,
        high_budget_words=160,
    )

    def utility(policy: str, endpoint: str, budget: int = 40) -> float:
        row = one(
            natural_metrics,
            policy=policy,
            budget_words=budget,
            kind="utility",
            endpoint=endpoint,
            metric=METRIC,
        )
        return number(row, "mean")

    focused_orthogonal = number(
        one(
            relation_rows,
            policy="conditioned",
            budget_words=160,
            relation="orthogonal",
            kind="utility",
            metric=METRIC,
        ),
        "mean",
    )
    focused_paraphrase = number(
        one(
            relation_rows,
            policy="conditioned",
            budget_words=160,
            relation="paraphrase",
            kind="utility",
            metric=METRIC,
        ),
        "mean",
    )
    generic_orthogonal = number(
        one(
            relation_rows,
            policy="generic",
            budget_words=160,
            relation="orthogonal",
            kind="utility",
            metric=METRIC,
        ),
        "mean",
    )
    oracle_orthogonal = number(
        one(
            relation_rows,
            policy="oracle",
            budget_words=160,
            relation="orthogonal",
            kind="utility",
            metric=METRIC,
        ),
        "mean",
    )

    natural_questions = natural_example["questions"]
    designed_questions = designed_example["questions"]
    assert isinstance(natural_questions, list)
    assert isinstance(designed_questions, list)

    def example_question(
        questions: list[dict[str, object]], *, aspect: str, role: str
    ) -> dict[str, object]:
        matches = [
            question for question in questions
            if question.get("aspect") == aspect and question.get("role") == role
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one example question for aspect={aspect!r}, role={role!r}; "
                f"found {len(matches)}"
            )
        return matches[0]

    def qa_text(question: dict[str, object]) -> tuple[str, str]:
        golds = question["golds"]
        assert isinstance(golds, list) and golds
        return str(question["question"]), str(golds[0])

    natural_qa = [qa_text(question) for question in natural_questions]
    founding_anchor = qa_text(
        example_question(designed_questions, aspect="founding", role="anchor")
    )
    founding_paraphrase = qa_text(
        example_question(designed_questions, aspect="founding", role="paraphrase")
    )
    founding_same_entity = qa_text(
        example_question(designed_questions, aspect="founding", role="same_entity")
    )
    founding_same_topic = qa_text(
        example_question(designed_questions, aspect="founding", role="same_topic")
    )
    finance_anchor = qa_text(
        example_question(designed_questions, aspect="finance", role="anchor")
    )

    gain_list = ", ".join(signed_points(value) for value in natural_now[:3])
    loss_list = ", ".join(signed_points(value) for value in natural_future[:3])
    gap_delta = number(gap_interaction, "delta")
    gap_lo = number(gap_interaction, "lo")
    gap_hi = number(gap_interaction, "hi")

    lines = [
        "# What a Question-Specific Handoff Forgets",
        "",
        "**A plain-language brief of Experiment 10: communication regret**",
        "",
        "> **Bottom line.** When an agent writes a fixed-length handoff for one known "
        "question, it becomes much better for that question and less useful for questions "
        "that arrive later. The missing information would often have fit; the sender simply "
        "chose not to preserve it.",
        "",
        "Here, **communication regret** means accuracy lost because evidence in the source "
        "did not survive the handoff.",
        "",
        "## What was tested",
        "",
        f"A sender compressed each source into a 20-, 40-, 80-, or 160-word handoff. In "
        f"{natural_manifest['contexts']} natural SQuAD passages with "
        f"{natural_manifest['questions']} questions, every question took a turn as the known "
        "question; each resulting handoff was then tested on that question and on the other "
        "questions from the same passage. The test was repeated on "
        f"{designed_manifest['contexts']} fictional dossiers with "
        f"{designed_manifest['questions']} questions, where the distance between questions "
        "was designed in advance.",
        "",
        "Every handoff policy received the same source and the same word band. The policies "
        "differed only in which questions the sender could see and, for the extractive controls, "
        "whether it could rewrite the source.",
        "",
        "## What “natural passages” and “designed dossiers” mean",
        "",
        "### Natural passages",
        "",
        "These are pre-existing Wikipedia-derived passages and human-written questions from "
        "the SQuAD reading-comprehension dataset. They were not written for this experiment "
        "and do not impose a designed relationship among their questions. That makes them the "
        "more realistic test, although it also means question distance cannot be controlled "
        "precisely.",
        "",
        f"**Example — {str(natural_example['title']).replace('_', ' ')}.** Its "
        f"{natural_example['meta']['source_words']}-word passage surveys the university’s "
        "history. Four naturally co-occurring facts become separate questions:",
        "",
        f"- *{natural_qa[0][0]}* → **{natural_qa[0][1]}**",
        "",
        f"- *{natural_qa[1][0]}* → **{natural_qa[1][1]}**",
        "",
        f"- *{natural_qa[2][0]}* → **{natural_qa[2][1]}**",
        "",
        f"- *{natural_qa[3][0]}* → **{natural_qa[3][1]}**",
        "",
        "A focused handoff written for any one of these questions may omit the facts needed "
        "for the other three.",
        "",
        "### Designed dossiers",
        "",
        "These are fictional documents created specifically for the experiment, using invented "
        "names and facts that the answering model could not already know. Each dossier has four "
        "aspects—founding, facilities, finance, and a document-specific custom—with four "
        "questions per aspect. An anchor question supplies the known question, while the other "
        "questions create controlled near and far comparisons. This lets the experiment control "
        "how far a later question is from the known question.",
        "",
        f"**Example — {designed_example['title']}.** Relative to its founding question:",
        "",
        f"- **Known question:** *{founding_anchor[0]}* → **{founding_anchor[1]}**",
        "",
        f"- **Same answer, reworded:** *{founding_paraphrase[0]}* → "
        f"**{founding_paraphrase[1]}**",
        "",
        f"- **Same subject:** *{founding_same_entity[0]}* → "
        f"**{founding_same_entity[1]}**",
        "",
        f"- **Related subject:** *{founding_same_topic[0]}* → "
        f"**{founding_same_topic[1]}**",
        "",
        f"- **Different topic:** *{finance_anchor[0]}* → **{finance_anchor[1]}**",
        "",
        "The labels are built into the document design rather than estimated afterward: a "
        "rewording asks for the same answer, the next two ask for other facts within the same "
        "aspect, and a different-topic question comes from another aspect.",
        "",
        "## Policy and plot-label glossary",
        "",
        "These code labels appear in the original diagnostic plots:",
        "",
        "- **`generic` — question-blind.** The sender reads the source and knows that the next "
        "agent will answer a question, but it is not told which question. It writes a general "
        "prose handoff and is not warned that further questions may follow.",
        "",
        "- **`conditioned` — focused on the current question.** The sender reads the source and "
        "one specific question, then preserves the information needed to answer that question. "
        "The plain-language plots call this the **focused handoff**.",
        "",
        "- **`reusable` — focused, with a reuse warning.** The sender sees the same current "
        "question as `conditioned`, but is also told that unknown questions may follow and is "
        "asked to retain other useful evidence within the same word limit. This tests whether a "
        "simple instruction can prevent over-specialisation.",
        "",
        "- **`oracle` — shown all evaluated questions.** The sender sees the complete set of "
        "questions used to evaluate that source before "
        "writing one handoff. This is a diagnostic capacity benchmark, not a perfect-answer "
        "oracle or a deployable strategy when future questions are genuinely unknown. The plain-language "
        "plots call it **shown all questions**.",
        "",
        "- **`extractive_generic` — question-blind sentence selection.** It receives the same "
        "information as `generic`, but may only copy source sentences verbatim; it cannot "
        "paraphrase, merge, shorten, or add text.",
        "",
        "- **`extractive_conditioned` — focused sentence selection.** It sees the current "
        "question like `conditioned`, but must build the handoff only from verbatim source "
        "sentences. Comparing the two extractive policies isolates evidence selection from "
        "abstractive rewriting.",
        "",
        "The extractive controls were included in the natural-passage run, not the designed-"
        "dossier replication.",
        "",
        "Other labels describe controls or scores rather than new sender policies:",
        "",
        "- **`@trim` — shared-truncation-cap sensitivity control.** At the 40-word setting, "
        "each prose handoff within one natural-passage context was re-truncated using the same "
        "context-specific cap: the shortest original delivered length. Because truncation "
        "prefers a complete sentence, final messages can end below that cap and need not be "
        "exactly equal in length. This is a post-hoc check, not a deployable policy.",
        "",
        "- **Direct-context ceiling.** The answerer reads the complete source instead of a "
        "handoff. Despite the plot label, this is an empirical no-handoff reference rather than "
        "a guaranteed mathematical maximum; sampling can occasionally put a handoff above it. "
        "It appears as a dotted reference line.",
        "",
        "- **`U_now`, `U_future`, and specialisation gap.** `U_now` is accuracy on the question "
        "known during writing; `U_future` is average accuracy on the other questions. Their "
        "difference is the specialisation gap: zero is balanced, while a large positive value "
        "means the handoff strongly favours the known question.",
        "",
        "- **Communication regret.** This is direct-context accuracy minus handoff accuracy. "
        "Unlike the utility plots, lower is better: zero means the handoff lost no measured "
        "accuracy relative to reading the full source.",
        "",
        "## Focus helps now and hurts later",
        "",
        "![Accuracy change from focusing the handoff](plain_language/focus_tradeoff.png)",
        "",
        "**How to read Plot 1.** The horizontal axis is the word limit. The vertical axis is "
        "focused minus question-blind accuracy, in percentage points: above zero means focusing "
        "helped, while below zero means it hurt. Orange circles score the known question, blue "
        "squares score the other questions, and whiskers are 95% context-level intervals.",
        "",
        "**What it shows.** On the natural passages, focused handoffs improved the known question by "
        f"{gain_list} percentage points at 20, 40, and 80 words, while changing later-question "
        f"accuracy by {loss_list} points. The 95% intervals excluded zero in both directions. "
        f"At 160 words, the present gain had shrunk to {signed_points(natural_now[3])} points "
        f"and was uncertain, but the later-question loss remained "
        f"{signed_points(natural_future[3])} points.",
        "",
        f"Scarcity sharpened the trade-off: the focus-induced gap between current and later "
        f"accuracy was {100 * gap_delta:.0f} points wider at 20 than at 160 words "
        f"(95% interval {100 * gap_lo:.0f} to {100 * gap_hi:.0f}). The designed dossiers "
        f"replicated the pattern; even at 160 words, focusing gained "
        f"{signed_points(designed_now_160)} points on the known question and lost "
        f"{abs(100 * designed_future_160):.0f} points on later ones.",
        "",
        "## The loss comes from evidence selection",
        "",
        "![Later-question accuracy by distance and budget](plain_language/question_distance.png)",
        "",
        "**How to read Plot 2.** The left panel holds a focused handoff at 160 words and moves "
        "from a rewording of the known question to a different topic. The right panel follows "
        "only different-topic accuracy as the budget grows. Higher is better; whiskers are 95% "
        "context-level intervals.",
        "",
        f"**What it shows.** At the widest 160-word budget, a focused dossier handoff answered "
        f"{percentage(focused_paraphrase)} of rewordings of the known question, but only "
        f"{percentage(focused_orthogonal)} of questions on a different topic. A question-blind "
        f"handoff answered {percentage(generic_orthogonal)} of those different-topic questions, "
        f"and the all-questions benchmark answered {percentage(oracle_orthogonal)}.",
        "",
        f"The clearest capacity check came at 40 words in the natural corpus. Focused and "
        f"all-questions handoffs both scored {percentage(utility('conditioned', 'now'), 1)} on "
        f"the current question, but later-question accuracy was "
        f"{percentage(utility('conditioned', 'future'), 1)} for the focused handoff versus "
        f"{percentage(utility('oracle', 'future'), 1)} for the all-questions handoff. Their "
        f"average lengths were {mean_words(natural_lengths, 'conditioned', 40):.1f} and "
        f"{mean_words(natural_lengths, 'oracle', 40):.1f} words. The channel could carry the "
        "answers; the focused sender did not select them.",
        "",
        "Verbatim, sentence-selection handoffs reproduced the same trade-off, so abstractive "
        "rewriting is not the main cause. Telling the focused sender to keep the handoff "
        "reusable recovered some nearby facts at larger budgets, but future-question accuracy "
        "still stayed below the question-blind baseline at every budget in both corpora.",
        "",
        "## The comparison really used the same channel",
        "",
        "![Delivered-word difference between focused and question-blind handoffs]"
        "(plain_language/word_budget_check.png)",
        "",
        "**How to read Plot 3.** Each point is focused minus question-blind delivered words. "
        "Zero means equal average length, positive means the focused handoff was longer, and "
        "negative means it was shorter. Colours separate the corpora; whiskers are 95% "
        "context-level intervals.",
        "",
        "**What it shows.** Across both corpora and all four limits, the mean length difference between focused "
        "and question-blind handoffs stayed within two words, and every interval included "
        "zero. No delivered message exceeded its cap. A separate 40-word check applied the same "
        "stricter, context-specific truncation cap to every prose policy and preserved the "
        "qualitative result.",
        "",
        "## Pareto frontier",
        "",
        "**How to read Plots 4 and 5.** Each panel is one requested word cap, and each marker "
        "is one evaluated handoff policy. Moving right means better accuracy on the current "
        "question; moving up means better accuracy on later questions. Saturated markers "
        "with black outlines are nondominated within that cap; dominated observations are "
        "muted. Markers are not joined because the policies are categories, not samples from "
        "an achievable continuous path. The dotted diagonal means equal current and later "
        "performance.",
        "",
        "### Plot 4 — Natural-passage Pareto frontier",
        "",
        f"![Pareto frontier for natural passages]({natural_rel}/pareto_now_vs_future.png)",
        "",
        "**What it shows.** Focused handoffs move right toward high current-question accuracy "
        "but remain low on later-question accuracy. Question-blind and all-questions handoffs "
        "sit nearer the diagonal; the all-questions condition reaches the strongest balanced "
        "accuracy as its budget grows.",
        "",
        "### Plot 5 — Designed-dossier Pareto frontier",
        "",
        f"![Pareto frontier for designed dossiers]({designed_rel}/pareto_now_vs_future.png)",
        "",
        "**What it shows.** The same trade-off is sharper: focused and reusable handoffs cluster "
        "near perfect current accuracy but weak later accuracy, while larger all-questions "
        "handoffs move toward the upper-right. No point is universally best because current "
        "accuracy, future reuse, and message length compete.",
        "",
        "## What this means",
        "",
        "- For a one-off known question, a focused handoff is effective.",
        "",
        "- If the handoff may be reused, keep the source available or tell the sender which "
        "downstream questions matter. A vague request to “stay reusable” is weak protection.",
        "",
        "- More space does not automatically create broader coverage; a focused sender may "
        "spend it deepening the same topic.",
        "",
        "## Limits",
        "",
        "This experiment used one sender/answerer model stack and one judge model. The "
        f"effective sample sizes were {natural_manifest['contexts']} natural contexts and "
        f"{designed_manifest['contexts']} fictional contexts, not the much larger number of "
        "question evaluations. The fictional corpus supports a clear near-versus-far result, "
        "but not a strict ranking of its two middle distance categories.",
        "",
        "## Complete figure set",
        "",
        "The three simplified figures and both Pareto plots appear above. The remaining "
        "machine-generated diagnostics are collected here so every plot from both final runs "
        "is visible without adding the raw tables. Original code labels are retained: "
        "`generic` means question-blind, `conditioned` means focused, and `oracle` means shown "
        "all questions; `extractive` variants copy source sentences.",
        "",
        "### Natural passages",
        "",
        "#### Plot 6 — Word-budget and shared-truncation-cap controls",
        "",
        f"![Natural-passage word-budget controls]({natural_rel}/budget_control.png)",
        "",
        "**How to read it.** In the left panel, each dot is mean delivered words divided by the "
        "cap; the grey band is the required 85–100% fill range and whiskers show one standard "
        "deviation. In the right panel, circles are current accuracy, squares are later accuracy, "
        "filled marks are the original 40-word messages, and hollow marks are messages after a "
        "shared context-specific truncation cap; those whiskers are 95% intervals. The plot’s "
        "“one length” title names the shared cap—sentence-boundary truncation can make delivered "
        "lengths shorter and unequal.",
        "",
        "**What it shows.** All policies used similar amounts of space. Applying the common, "
        "stricter cap preserves the large current-versus-later gap for focused and reusable "
        "handoffs, supporting the conclusion from the direct word-count comparison.",
        "",
        "#### Plot 7 — Accuracy across word budgets",
        "",
        f"![Natural-passage accuracy across budgets]({natural_rel}/utility_vs_budget.png)",
        "",
        "**How to read it.** The first panel shows current-question accuracy, the second shows "
        "later-question accuracy, and the third subtracts later from current accuracy. Higher "
        "is better in the first two panels; zero in the third means no specialisation. Lines are "
        "policies, whiskers are 95% intervals, and the dotted horizontal line is the accuracy "
        "available from the full source (or zero in the gap panel).",
        "",
        "**What it shows.** Focused handoffs remain strong on the known question but weak on "
        "later questions. Broader policies gain future accuracy as the budget grows, and the "
        "focused specialisation gap narrows but does not disappear.",
        "",
        "#### Plot 8 — Accuracy lost through the handoff",
        "",
        f"![Natural-passage communication regret]({natural_rel}/communication_regret.png)",
        "",
        "**How to read it.** Each bar is full-source accuracy minus handoff accuracy, so lower is "
        "better and zero means no measured loss. The left panel scores the current question, the "
        "right scores later questions, colours identify policies, and whiskers are 95% "
        "context-level intervals.",
        "",
        "**What it shows.** Focused handoffs lose almost nothing on the current question but much "
        "more on later questions. All-questions regret falls toward zero as the budget grows, "
        "showing that the channel can preserve broad evidence when the sender knows it is needed.",
        "",
        "### Designed dossiers",
        "",
        "#### Plot 9 — Word-budget control",
        "",
        f"![Designed-dossier word-budget controls]({designed_rel}/budget_control.png)",
        "",
        "**How to read it.** The left panel plots mean delivered words divided by the cap; the "
        "grey band is the required 85–100% range and whiskers show one standard deviation. The "
        "right panel is intentionally blank because this run did not create a separately "
        "re-trimmed sensitivity arm.",
        "",
        "**What it shows.** All four policies filled the same contracted bands closely at every "
        "budget. Their accuracy differences therefore reflect what they selected, not one policy "
        "being allowed a longer message.",
        "",
        "#### Plot 10 — Accuracy across word budgets",
        "",
        f"![Designed-dossier accuracy across budgets]({designed_rel}/utility_vs_budget.png)",
        "",
        "**How to read it.** The panels show current accuracy, later accuracy, and their "
        "difference from left to right. Higher is better in the first two; a larger value in the "
        "third means stronger specialisation. Lines identify policies, whiskers are 95% "
        "intervals, and the dotted line marks the empirical full-dossier reference or zero gap.",
        "",
        "**What it shows.** Focused and reusable handoffs stay almost perfect on the known "
        "question across all budgets, yet remain poor on unseen questions. Question-blind and "
        "all-questions handoffs convert extra words into much broader coverage.",
        "",
        "#### Plot 11 — Accuracy lost through the handoff",
        "",
        f"![Designed-dossier communication regret]({designed_rel}/communication_regret.png)",
        "",
        "**How to read it.** Bars measure full-dossier accuracy minus handoff accuracy: zero is "
        "best and taller bars mean more information was lost. The left panel covers the current "
        "question, the right covers later questions, and whiskers are 95% intervals.",
        "",
        "**What it shows.** Focused and reusable handoffs have almost no current-question regret "
        "but retain high future regret even as the budget expands. All-questions regret drops "
        "steeply with more words because that sender spreads evidence across possible questions.",
        "",
        "#### Plot 12 — Loss by distance from the known question",
        "",
        f"![Designed-dossier regret by question distance]({designed_rel}/relation_distance.png)",
        "",
        "**How to read it.** Each panel fixes one word budget. The horizontal categories move "
        "from a paraphrase of the known question to a different dossier aspect; the vertical "
        "axis is later-question regret, so lower is better. Lines identify policies and whiskers "
        "are 95% intervals. The two middle categories are related types, not a guaranteed strict "
        "ordering.",
        "",
        "**What it shows.** Focused handoffs have almost no loss on a rewording but near-total "
        "loss on a different topic, even at 160 words. The distance gradient is much weaker for "
        "question-blind and all-questions handoffs, confirming that it is created by focus.",
        "",
        "For full methods, uncertainty intervals, robustness checks, and machine-readable "
        "results, see the [complete communication-regret report](REPORT.md).",
        "",
    ]
    report = "\n".join(lines)
    if any(line.startswith("|") for line in lines):
        raise AssertionError("The plain-language report must not contain Markdown tables")
    return report


def assert_all_plots_linked(report: str, corpora: list[dict[str, object]]) -> None:
    """Fail if any generated plot is absent from the visual report."""

    plot_roots = [ASSET_ROOT]
    for corpus in corpora:
        root = corpus["root"]
        assert isinstance(root, Path)
        plot_roots.append(root)
    missing = []
    for plot_root in plot_roots:
        for path in plot_root.glob("*.png"):
            relative = path.relative_to(RESULT_ROOT).as_posix()
            if f"]({relative})" not in report:
                missing.append(relative)
    if missing:
        raise AssertionError("Plots missing from plain-language report: " + ", ".join(missing))


def load_corpus(dataset: str, label: str) -> dict[str, object]:
    root = latest_run(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("primary_metric") != METRIC:
        raise ValueError(
            f"Expected primary metric {METRIC!r}, got {manifest.get('primary_metric')!r} "
            f"in {root}"
        )
    return {
        "dataset": dataset,
        "label": label,
        "root": root,
        "manifest": manifest,
        "metrics": read_csv(root / "metrics.csv"),
        "contrasts": read_csv(root / "contrasts.csv"),
        "length_audit": read_csv(root / "length_audit.csv"),
        "length_deltas": read_csv(root / "length_deltas.csv"),
        "budget_interaction": read_csv(root / "budget_interaction.csv"),
    }


def load_example(manifest: dict[str, object], context_id: str) -> dict[str, object]:
    source_file = ROOT / str(manifest["source_file"])
    matches = [row for row in read_jsonl(source_file) if row.get("context_id") == context_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one example {context_id!r} in {source_file}, found {len(matches)}")
    return matches[0]


def main() -> None:
    corpora = [
        load_corpus("squad_groups", "Natural passages"),
        load_corpus("relation_dossiers", "Designed dossiers"),
    ]
    designed_root = corpora[1]["root"]
    assert isinstance(designed_root, Path)
    relation_rows = read_csv(designed_root / "relation_regret.csv")

    natural_manifest = corpora[0]["manifest"]
    designed_manifest = corpora[1]["manifest"]
    assert isinstance(natural_manifest, dict)
    assert isinstance(designed_manifest, dict)
    natural_example = load_example(natural_manifest, "sq:15")
    designed_example = load_example(designed_manifest, "rel:3")

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(corpora, ASSET_ROOT / "focus_tradeoff.png")
    plot_distance(relation_rows, ASSET_ROOT / "question_distance.png")
    plot_length_control(corpora, ASSET_ROOT / "word_budget_check.png")

    report = render_report(corpora, relation_rows, natural_example, designed_example)
    assert_all_plots_linked(report, corpora)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    for path in sorted(ASSET_ROOT.glob("*.png")):
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
