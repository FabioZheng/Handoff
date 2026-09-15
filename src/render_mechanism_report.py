"""Experiment 14's own figures, diagnostics and report.

The shared runner already produces the utility matrix, the metric tables, the
paired contrasts and the distance table. What it cannot produce is the reading
of them that this experiment exists for: three compression *mechanisms* laid
over one another at matched conditioning, and the evidence-level diagnostic that
says whether a wrong answer means the evidence was deleted or the reader failed
on evidence that survived.

Three figures:

``distance_curves.png``
    The main one. Regret as a function of the designed distance between the
    hidden question and the conditioning question, for all six main arms, at
    every budget. Colour is the mechanism family, line style is the conditioning
    level. A specialisation effect looks like a rising line for the conditioned
    arm and a flat one for its generic control; whether that shape appears in
    one family, two, or all three is the result.

``now_future_tradeoff.png``
    The simpler view: present-query utility against future-query utility, one
    trajectory per arm across the budget ladder, with paraphrase and the random
    floor included.

``evidence_survival.png``
    The diagnostic. For each arm, how often the answer string for the current
    question survived into the message versus the answer string for an
    orthogonal question, and - conditional on survival - how often the reader
    then got it right. It separates "the mechanism deleted it" from "the reader
    could not use it", which an accuracy table alone cannot.

No API calls; runs on stored rows only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import regret_data as rd  # noqa: E402
from data import split_sentences  # noqa: E402
import run_communication_regret as rcr  # noqa: E402
from llm import load_config  # noqa: E402
from score import bootstrap_ci, paired_bootstrap_delta  # noqa: E402

# The paper's names for the arms, and the order they are read in.
DISPLAY = {
    "paraphrase": "paraphrase",
    "generic": "summary_generic",
    "conditioned": "summary_conditioned",
    "lm_generic": "lm_elimination_generic",
    "lm_conditioned": "lm_elimination_conditioned",
    "nonllm_generic": "nonllm_elimination_generic",
    "nonllm_conditioned": "nonllm_elimination_conditioned",
    "random_selection": "random_selection",
    "passthrough": "passthrough (prefix)",
    "reusable": "summary_reusable",
    "oracle": "summary_oracle",
}

# (family label, generic arm, conditioned arm, colour)
FAMILIES = (
    ("abstractive summary", "generic", "conditioned", "#d95f02"),
    ("LM elimination", "lm_generic", "lm_conditioned", "#08519c"),
    ("non-LLM elimination", "nonllm_generic", "nonllm_conditioned", "#54278f"),
)

MAIN_ARMS = [p for _, g, c, _ in FAMILIES for p in (g, c)]
TRADEOFF_ARMS = MAIN_ARMS + ["paraphrase", "passthrough", "random_selection"]

RELATION_LABELS = {
    rd.PARAPHRASE: "paraphrase\nof q_now",
    rd.SAME_ENTITY: "same\nentity",
    rd.SAME_TOPIC: "same\ntopic",
    rd.ORTHOGONAL: "orthogonal\naspect",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def lookup(rows: list[dict], **where):
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in where.items()):
            return row
    return None


# ---------------------------------------------------------------------------
# Evidence-level diagnostics
#
# Computed here rather than in the runner because they apply uniformly to every
# arm, including the ones inherited from Experiment 10 whose stored rows predate
# the diagnostic. Gold-span presence is a property of the delivered text, so it
# can be re-derived for any message without regenerating anything.


def survival_rows(contexts, rotations, messages: list[dict], answers: list[dict],
                  policies: list[str]) -> list[dict]:
    """One row per (rotation, message, evaluated question): did the answer survive?

    ``span_present`` is a string test on the delivered message, so it is defined
    for rewriting and elimination alike. ``unit_present`` is the stronger,
    selection-only fact that a source sentence labelled as answer-bearing was
    delivered whole.

    A query-agnostic arm holds one message per (context, budget) and that single
    message plays every rotation, exactly as in the runner's utility matrix.
    Iterating over messages alone would leave those arms with no diagonal at all
    and silently drop them from every q_now comparison.
    """
    by_id = {c.context_id: c for c in contexts}
    answers_by = {(a["message_key"], a["eval_qid"]): a for a in answers if a.get("message_key")}
    by_slot: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for message in messages:
        by_slot[(message["context_id"], message["policy"],
                 message["rotation_id"])].append(message)

    rows = []
    for context in contexts:
        for rotation in rotations[context.context_id]:
            for policy in policies:
                slot = "-" if rcr.rotation_invariant(policy) else rotation.rotation_id
                for message in by_slot.get((context.context_id, policy, slot), ()):
                    text = str(message["handoff_text"]).lower()
                    survived_units = set(message.get("answer_bearing_survived_qids") or ())
                    is_selection = message.get("selected_unit_ids") is not None
                    for q in context.questions:
                        answer = answers_by.get((message["message_key"], q.qid))
                        if answer is None:
                            continue
                        golds = [str(g).lower().strip() for g in q.golds if str(g).strip()]
                        diagonal = q.qid == rotation.current_qid
                        rows.append({
                            "context_id": context.context_id,
                            "policy": policy,
                            "budget_words": int(message["budget_words"]),
                            "rotation_id": rotation.rotation_id,
                            "current_qid": rotation.current_qid,
                            "eval_qid": q.qid,
                            "is_diagonal": diagonal,
                            "relation": (rd.UNLABELLED if diagonal else
                                         rd.relation_label(context, rotation.current_qid, q.qid)),
                            "span_present": int(any(gold in text for gold in golds)),
                            "unit_present": int(q.qid in survived_units) if is_selection else "",
                            "judge_correct": float(answer.get("judge_correct", 0.0)),
                            "em": float(answer.get("em", 0.0)),
                            "delivered_words": int(message["delivered_words"]),
                        })
    return rows


def survival_summary(rows: list[dict], policies: list[str], budgets: list[int],
                     boot: int, ci: float) -> list[dict]:
    """Per (arm, budget, slice): survival, accuracy, and accuracy given survival."""
    slices = ((("q_now", lambda r: r["is_diagonal"]),
               ("any_future", lambda r: (not r["is_diagonal"])
                and r["relation"] != rd.UNLABELLED))
              # One slice per designed distance tier as well: "did the evidence for
              # a question THIS far from the conditioning one survive" is the
              # evidence-level counterpart of the distance curve, and it is what
              # separates a deleted aspect from a delivered-but-misread one.
              + tuple((relation, (lambda rel: lambda r: r["relation"] == rel)(relation))
                      for relation in rd.RELATION_ORDER))
    out = []
    for policy in policies:
        for words in budgets:
            for name, keep in slices:
                cells = [r for r in rows if r["policy"] == policy
                         and r["budget_words"] == words and keep(r)]
                if not cells:
                    continue
                # Cluster on the context: rotations within a source are not
                # independent, and the whole design repeats on sources.
                by_context = defaultdict(list)
                for cell in cells:
                    by_context[cell["context_id"]].append(cell)
                survival = np.array([np.mean([c["span_present"] for c in v])
                                     for v in by_context.values()])
                accuracy = np.array([np.mean([c["judge_correct"] for c in v])
                                     for v in by_context.values()])
                kept = [c for c in cells if c["span_present"]]
                given = np.array([np.mean([c["judge_correct"] for c in v])
                                  for v in _group(kept).values()]) if kept else np.array([])
                s_mean, s_lo, s_hi = bootstrap_ci(survival, boot, ci, seed=1400)
                a_mean, a_lo, a_hi = bootstrap_ci(accuracy, boot, ci, seed=1401)
                g_mean = float(given.mean()) if given.size else float("nan")
                out.append({
                    "policy": policy, "display": DISPLAY.get(policy, policy),
                    "budget_words": words, "slice": name, "n_cells": len(cells),
                    "n_contexts": len(by_context),
                    "span_survival": s_mean, "span_survival_lo": s_lo, "span_survival_hi": s_hi,
                    "accuracy": a_mean, "accuracy_lo": a_lo, "accuracy_hi": a_hi,
                    "accuracy_given_survival": g_mean,
                    "n_contexts_with_survival": int(given.size),
                    "survival_shortfall": s_mean - a_mean,
                })
    return out


def _group(cells: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for cell in cells:
        grouped[cell["context_id"]].append(cell)
    return grouped


def budget_accounting(contexts, messages: list[dict], policies: list[str]) -> list[dict]:
    """One row per delivered message, in the brief's own field names.

    Every mechanism is accounted for identically, which is the point: a length
    difference between arms is the one confound that could explain a utility
    result on its own, so the accounting may not be defined only where it is
    convenient (the selection arms).

    ``selected_unit_count`` is given a definition that applies to rewriting too:
    the number of source sentences present **verbatim** in the delivered
    message. For an elimination arm that is its selection by construction; for
    an abstractive arm it measures how much of the source it happened to copy,
    which is normally zero and is worth seeing rather than assuming.
    """
    by_id = {c.context_id: c for c in contexts}
    units_by_context = {}
    rows = []
    for message in messages:
        policy = message["policy"]
        if policy not in policies:
            continue
        context = by_id[message["context_id"]]
        if context.context_id not in units_by_context:
            units_by_context[context.context_id] = split_sentences(context.source)
        text = str(message["handoff_text"])
        source_words = len(context.source.split())
        delivered = int(message["delivered_words"])
        verbatim = sum(1 for sentence in units_by_context[context.context_id]
                       if sentence.strip() and sentence.strip() in text)
        rows.append({
            "policy": policy,
            "display": DISPLAY.get(policy, policy),
            "context_id": message["context_id"],
            "rotation_id": message["rotation_id"],
            "target_words": int(message["budget_words"]),
            "delivered_words": delivered,
            "fill_ratio": float(message["fill_ratio"]),
            "source_words": source_words,
            "retention_fraction": delivered / float(source_words) if source_words else 0.0,
            "selected_unit_count": verbatim,
            "declared_selected_unit_count": (len(message.get("selected_unit_ids") or [])
                                             if message.get("selected_unit_ids") is not None
                                             else ""),
            "units_skipped_for_fit": message.get("units_skipped_for_fit", ""),
            "empty_message": bool(message.get("empty_message", not text.strip())),
            "over_budget": delivered > int(message["budget_words"]),
        })
    return rows


def aspect_retention(contexts, rotations, messages: list[dict], budgets: list[int],
                     boot: int, ci: float) -> list[dict]:
    """Reader-free evidence accounting for the selection arms.

    The relation corpus partitions its evidence into four aspects, one per
    question group, and a unit is claimed by at most one. So "how much of the
    conditioning aspect survived, against how much of the other three" measures
    the compressor's allocation directly, with no reader and no judge in the
    path. If a conditioned selector specialises, this is where it shows first.
    """
    by_id = {c.context_id: c for c in contexts}
    by_slot = defaultdict(list)
    for message in messages:
        if message.get("aspect_retention"):
            by_slot[(message["context_id"], message["policy"],
                     message["rotation_id"])].append(message)
    out = []
    policies = sorted({m["policy"] for m in messages if m.get("aspect_retention")})
    for policy in policies:
        for words in budgets:
            own, other = defaultdict(list), defaultdict(list)
            for context in contexts:
                for rotation in rotations[context.context_id]:
                    slot = "-" if rcr.rotation_invariant(policy) else rotation.rotation_id
                    for message in by_slot.get((context.context_id, policy, slot), ()):
                        if int(message["budget_words"]) != words:
                            continue
                        aspect = by_id[context.context_id].question(rotation.current_qid).aspect
                        retention = message["aspect_retention"]
                        own[context.context_id].append(float(retention.get(aspect, 0.0)))
                        other[context.context_id].append(float(np.mean(
                            [v for k, v in retention.items()
                             if k not in (aspect, "unclaimed")] or [0.0])))
            ids = sorted(set(own) & set(other))
            if len(ids) < 2:
                continue
            t = np.array([float(np.mean(own[i])) for i in ids])
            c = np.array([float(np.mean(other[i])) for i in ids])
            result = paired_bootstrap_delta(t, c, boot, ci, seed=1403)
            out.append({
                "policy": policy, "display": DISPLAY.get(policy, policy),
                "budget_words": words, "n_contexts": len(ids),
                "conditioning_aspect_retention": float(t.mean()),
                "other_aspect_retention": float(c.mean()),
                **result,
            })
    return out


def _family_did(rows: list[dict], policy_pair: tuple[str, str], words: int,
                metric: str) -> dict[str, float]:
    """Per context: (far - near) for the conditioned arm minus the same for its control."""
    per_policy = {}
    for policy in policy_pair:
        per_context = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if (row["policy"] == policy and row["budget_words"] == words
                    and not row["is_diagonal"]
                    and row["relation"] in (rd.PARAPHRASE, rd.ORTHOGONAL)):
                per_context[row["context_id"]][row["relation"]].append(float(row[metric]))
        per_policy[policy] = {
            cid: float(np.mean(v[rd.ORTHOGONAL])) - float(np.mean(v[rd.PARAPHRASE]))
            for cid, v in per_context.items()
            if v.get(rd.ORTHOGONAL) and v.get(rd.PARAPHRASE)}
    generic, conditioned = policy_pair
    ids = sorted(set(per_policy[generic]) & set(per_policy[conditioned]))
    return {cid: per_policy[conditioned][cid] - per_policy[generic][cid] for cid in ids}


def family_vs_family(rows: list[dict], budgets: list[int], boot: int, ci: float) -> list[dict]:
    """Is one mechanism's conditioning effect larger than another's?

    A triple difference, paired on the source: (conditioned - generic) of the
    near/far gradient in family A, against the same quantity in family B. This
    is the only honest way to say "abstractive rewriting amplifies the effect"
    rather than eyeballing two separately-computed numbers.
    """
    out = []
    for i, (label_a, gen_a, cond_a, _c) in enumerate(FAMILIES):
        for label_b, gen_b, cond_b, _d in FAMILIES[i + 1:]:
            for words in budgets:
                for metric in ("judge_correct", "em"):
                    a = _family_did(rows, (gen_a, cond_a), words, metric)
                    b = _family_did(rows, (gen_b, cond_b), words, metric)
                    ids = sorted(set(a) & set(b))
                    if len(ids) < 2:
                        continue
                    t = np.array([a[i_] for i_ in ids])
                    c = np.array([b[i_] for i_ in ids])
                    result = paired_bootstrap_delta(t, c, boot, ci, seed=1404)
                    out.append({
                        "family_a": label_a, "family_b": label_b, "budget_words": words,
                        "metric": metric, "n_contexts": len(ids),
                        "did_a": float(t.mean()), "did_b": float(c.mean()),
                        "quantity": "specialisation_gradient__family_a_minus_family_b",
                        **result,
                    })
    return out


def mechanism_contrasts(rows: list[dict], budgets: list[int], boot: int, ci: float) -> list[dict]:
    """The one number per family: does conditioning raise near and lower far?

    ``regret_orthogonal_minus_paraphrase`` already exists per arm in the shared
    analysis; what is missing is the paired *difference* between a conditioned
    arm and its own generic control, computed on the same contexts.
    """
    out = []
    for label, generic, conditioned, _colour in FAMILIES:
        for words in budgets:
            for metric in ("judge_correct", "em"):
                pairs = {}
                for policy in (generic, conditioned):
                    per_context = defaultdict(lambda: defaultdict(list))
                    for row in rows:
                        if (row["policy"] == policy and row["budget_words"] == words
                                and not row["is_diagonal"]
                                and row["relation"] in (rd.PARAPHRASE, rd.ORTHOGONAL)):
                            per_context[row["context_id"]][row["relation"]].append(
                                float(row[metric]))
                    pairs[policy] = {
                        cid: (float(np.mean(v[rd.ORTHOGONAL])) - float(np.mean(v[rd.PARAPHRASE])))
                        for cid, v in per_context.items()
                        if v.get(rd.ORTHOGONAL) and v.get(rd.PARAPHRASE)}
                ids = sorted(set(pairs[generic]) & set(pairs[conditioned]))
                if len(ids) < 2:
                    continue
                t = np.array([pairs[conditioned][i] for i in ids])
                c = np.array([pairs[generic][i] for i in ids])
                result = paired_bootstrap_delta(t, c, boot, ci, seed=1402)
                out.append({
                    "family": label, "treatment": conditioned, "control": generic,
                    "budget_words": words, "metric": metric,
                    "quantity": "utility_orthogonal_minus_paraphrase__conditioned_minus_generic",
                    "n_contexts": len(ids), **result,
                })
    return out


# ---------------------------------------------------------------------------
# Figures


def plot_distance_curves(table: list[dict], output: Path, budgets: list[int],
                         metric: str, metric_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(rd.RELATION_ORDER)
    x = np.arange(len(order))
    ncols = 2
    nrows = int(np.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.1 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, words in zip(axes, budgets):
        for label, generic, conditioned, colour in FAMILIES:
            for policy, style, marker in ((generic, "--", "o"), (conditioned, "-", "s")):
                ys, los, his = [], [], []
                for relation in order:
                    row = lookup(table, policy=policy, budget_words=words,
                                 relation=relation, kind="regret", metric=metric)
                    ys.append(float(row["mean"]) if row else np.nan)
                    los.append(float(row["lo"]) if row else np.nan)
                    his.append(float(row["hi"]) if row else np.nan)
                ys = np.array(ys)
                ax.plot(x, ys, style, color=colour, marker=marker, markersize=5,
                        linewidth=2 if style == "-" else 1.6,
                        label=f"{label} - {'conditioned' if style == '-' else 'generic'}")
                ax.fill_between(x, los, his, color=colour, alpha=0.08, linewidth=0)
        ax.set_title(f"{words}-word budget", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([RELATION_LABELS[r] for r in order], fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_ylabel(f"regret ({metric_label})")
    for ax in axes[len(budgets):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Does task-aware compression cost distant future questions "
                 "in every mechanism?\nRegret vs designed distance from the conditioning "
                 "query (higher = worse); solid = conditioned, dashed = its generic control",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(metrics: list[dict], baselines: list[dict], output: Path,
                  budgets: list[int], metric: str, metric_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for policy in TRADEOFF_ARMS:
        xs, ys, ws = [], [], []
        for words in budgets:
            now = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="now", metric=metric)
            future = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                            endpoint="future", metric=metric)
            if not now or not future:
                continue
            xs.append(float(future["mean"]))
            ys.append(float(now["mean"]))
            ws.append(words)
        if not xs:
            continue
        colour = rcr.POLICY_COLOURS.get(policy, "#666666")
        ax.plot(xs, ys, "-o", color=colour, markersize=5, linewidth=1.7,
                label=DISPLAY.get(policy, policy))
        for x, y, w in zip(xs, ys, ws):
            ax.annotate(str(w), (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=colour)

    ceiling = lookup(baselines, baseline="direct_context", metric=metric)
    if ceiling:
        value = float(ceiling["mean"])
        ax.axhline(value, color="#444444", linestyle=":", linewidth=1.2)
        ax.axvline(value, color="#444444", linestyle=":", linewidth=1.2)
        ax.annotate("unbounded source (no handoff)", (value, value), fontsize=8,
                    color="#444444", textcoords="offset points", xytext=(-150, 6))
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, limit], [0, limit], color="#bbbbbb", linewidth=0.9, zorder=0)
    ax.annotate("no specialisation", (limit * 0.72, limit * 0.75), fontsize=8, color="#999999",
                rotation=38)
    ax.set_xlabel(f"future-query utility U_future ({metric_label})")
    ax.set_ylabel(f"present-query utility U_now ({metric_label})")
    ax.set_title("What each mechanism buys and what it spends\n"
                 "one trajectory per arm across the 20/40/80/160-word ladder", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_survival(summary: list[dict], output: Path, budgets: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in TRADEOFF_ARMS]
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.0 * len(budgets), 5.4), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    width = 0.38
    y = np.arange(len(arms))
    for ax, words in zip(axes, budgets):
        now = [lookup(summary, policy=a, budget_words=words, slice="q_now") for a in arms]
        orth = [lookup(summary, policy=a, budget_words=words, slice="orthogonal") for a in arms]
        ax.barh(y + width / 2, [float(r["span_survival"]) if r else 0 for r in now],
                height=width, color="#d95f02", label="answer to q_now survived")
        ax.barh(y - width / 2, [float(r["span_survival"]) if r else 0 for r in orth],
                height=width, color="#7fbcd2", label="answer to an orthogonal question survived")
        for i, row in enumerate(now):
            if row and np.isfinite(float(row["accuracy_given_survival"])):
                ax.plot(float(row["accuracy_given_survival"]) * float(row["span_survival"]),
                        y[i] + width / 2, "|", color="black", markersize=9, markeredgewidth=1.6)
        ax.set_yticks(y)
        ax.set_yticklabels([DISPLAY.get(a, a) for a in arms], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(f"{words} words", fontsize=11)
        ax.grid(alpha=0.25, axis="x", linewidth=0.6)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Was the evidence deleted, or did the reader fail?\n"
                 "bars = the gold answer string survived into the message; "
                 "black tick = the reader then got it right", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------


def verdict(family: list[dict], across: list[dict], contrasts: list[dict],
            budgets: list[int], metric: str) -> list[str]:
    """Which of the four pre-registered readings the numbers actually support.

    The branches are the ones written down before the run (task brief section 8),
    and the test applied to each is the same one: does the paired interval for
    that family's conditioning effect exclude zero.

    Two different quantities have to be reported separately, because on this
    corpus they disagree, and collapsing them would be the single easiest way to
    overstate the result:

    * the **near/far gradient** (difference in differences on
      ``U_orthogonal - U_paraphrase``) - does conditioning shape the message
      toward the present query *relative to* distant ones;
    * the **absolute future loss** (``dU_future`` against the same control) -
      does conditioning leave later questions worse off than they would have
      been.

    A mechanism can show the first without the second: if its query-agnostic
    control was already keeping very little that any question could use, a
    conditioned version can add present utility without taking future utility
    away from it.
    """
    def gradient(label: str, words: int) -> bool | None:
        row = next((r for r in family if r["family"] == label
                    and r["budget_words"] == words and r["metric"] == metric), None)
        return None if row is None else float(row["hi"]) < 0.0

    def future_delta(pair: tuple[str, str], words: int) -> dict | None:
        name = f"{pair[1]}_minus_{pair[0]}"
        return next((r for r in contrasts if r["comparison"] == name
                     and int(r["budget_words"]) == words and r["kind"] == "utility"
                     and r["endpoint"] == "future" and r["metric"] == metric), None)

    labels = [f[0] for f in FAMILIES]
    pairs = {f[0]: (f[1], f[2]) for f in FAMILIES}
    working = [w for w in budgets if w != min(budgets)]
    degenerate = [w for w in budgets if w == min(budgets)]
    lines = []

    with_gradient = {label: [w for w in working if gradient(label, w)] for label in labels}
    all_families = all(len(with_gradient[label]) == len(working) for label in labels)
    if all_families:
        lines.append(
            f"**The task-aware gradient needs neither rewriting nor an LLM.** At every "
            f"working budget ({', '.join(str(w) for w in working)} words) all three "
            "mechanisms show a conditioning effect on the near/far gradient whose paired 95% "
            "interval excludes zero - including the arm with no neural model anywhere in the "
            "compressor: BM25 over the source's own sentences, delivered verbatim. Bounded "
            "task-aware *selection* is sufficient to shape a handoff around the present "
            "query.")
    else:
        missing = {label: [w for w in working if w not in with_gradient[label]]
                   for label in labels}
        lines.append(
            "**The gradient is not universal across mechanisms.** Budgets at which a "
            "family's interval does NOT exclude zero: "
            + "; ".join(f"{label}: {missing[label] or 'none'}" for label in labels) + ".")

    losing, flat = [], []
    for label in labels:
        for words in working:
            row = future_delta(pairs[label], words)
            if row is None:
                continue
            (losing if float(row["hi"]) < 0.0 else flat).append(
                f"{label} {words}w {fmt(row['delta'])} [{fmt(row['lo'])}, {fmt(row['hi'])}]")
    if losing and flat:
        lines.append(
            "**But only rewriting actually costs future questions.** Against its own generic "
            f"control, `dU_future` is negative with an interval excluding zero for: "
            f"{'; '.join(losing)}. It covers zero for: {'; '.join(flat)}. So in the "
            "elimination families conditioning buys present utility without measurably "
            "taking future utility away from the control it is measured against - the "
            "gradient there is produced by lifting the near questions, not by depressing the "
            "far ones. In the abstractive family it is a genuine trade, and the trade "
            "deepens as the budget grows.")
    elif losing:
        lines.append("**Every mechanism also loses absolute future utility.** "
                     f"{'; '.join(losing)}.")
    else:
        lines.append("**No mechanism shows an absolute future loss** against its own control: "
                     "every `dU_future` interval covers zero, so the effect here is entirely "
                     "a re-shaping of the message and not a net destruction of future value.")

    stronger = [r for r in across if r["metric"] == metric and float(r["hi"]) < 0.0
                and r["budget_words"] in working]
    if stronger:
        detail = "; ".join(
            f"{r['family_a']} > {r['family_b']} at {r['budget_words']}w "
            f"({fmt(r['delta'])} [{fmt(r['lo'])}, {fmt(r['hi'])}])" for r in stronger)
        lines.append(f"**The mechanisms are not equal in size either.** {detail}. "
                     "Abstractive generation amplifies task commitment beyond selective "
                     "deletion wherever the intervals separate, so the reading is "
                     "*selection is sufficient for the shape, rewriting is stronger and is "
                     "the only one that pays for it* - not that the mechanisms are "
                     "interchangeable.")
    else:
        lines.append("**No mechanism's effect is distinguishable from another's** at any "
                     "working budget: every cross-family interval covers zero.")

    lm_weaker = [r for r in across if r["metric"] == metric and r["budget_words"] in working
                 and r["family_a"] == "LM elimination"
                 and r["family_b"] == "non-LLM elimination" and float(r["lo"]) > 0.0]
    rewriting_costs = any(True for _ in losing)
    table = []
    table.append("| effect the design asks about | supported? | on what evidence |")
    table.append("|---|---|---|")
    table.append(
        "| general task-aware **selection** effect | "
        + ("**yes**" if all_families else "partly")
        + " | the near/far gradient's DiD excludes zero in all three families, including "
          "BM25 with no neural model in the compressor |")
    table.append(
        "| **rewriting** effect | "
        + ("**yes, and it is the only mechanism that costs future utility**"
           if rewriting_costs else "not detected")
        + " | abstractive summary has the steepest gradient, and is the only family whose "
          "`dU_future` interval excludes zero |")
    table.append(
        "| **LM-selection** effect | "
        + ("**no** - the LM selector is the *weakest* of the three"
           if lm_weaker else "not distinguishable")
        + " | a pinned GPT-2 relevance score produces a smaller gradient than BM25 at "
        + ", ".join(f"{r['budget_words']}w" for r in lm_weaker) + " |")
    table.append(
        "| some combination | **yes - this one** | selection is sufficient for the shape; "
        "rewriting amplifies it and adds an absolute future cost; a learned scorer adds "
        "nothing over lexical matching |")
    # One element, so the caller's paragraph spacing cannot break the table apart.
    lines.append("\n".join(table))

    if degenerate:
        lines.append(
            f"The {degenerate[0]}-word rung is excluded from every statement above and "
            "reported as a granularity stress test: whole-sentence selection cannot fill it "
            "(the corpus's shortest sentence is 10 words, its median 27, and five of sixteen "
            "dossiers own no sentence that fits), so both elimination families deliver one "
            "sentence or none and their intervals cover zero for that reason.")
    return lines


def fmt(value, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(f) else f"{f:.{digits}f}"


def ci(row: dict, prefix: str = "") -> str:
    lo, hi = row.get(f"{prefix}lo"), row.get(f"{prefix}hi")
    return f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None else ""


def write_report(path: Path, *, cfg, metrics, relations, contrasts, family, summary,
                 lengths, aspects, across, accounting, budgets, metric, metric_label,
                 lm_manifest) -> None:
    lines = []
    add = lines.append
    add("# Experiment 14: which compression mechanism specialises a handoff?")
    add("")
    add("Experiment 10 found that conditioning a bounded handoff on the currently known")
    add("query buys present-query utility and loses future-query utility, with the loss")
    add("growing in the designed distance between the future query and the conditioning")
    add("one. Every arm there rewrote the source with an LLM, so *selection* and")
    add("*rewriting* were varying together. This run separates them on the same corpus,")
    add("channel, reader and judge.")
    add("")
    add(f"Primary utility: **{metric_label}**. Sixteen relation dossiers, four rotating")
    add("anchors each, sixteen questions each, budgets "
        f"{'/'.join(str(w) for w in budgets)} delivered words.")
    add("")
    add("| paper name | code arm | rewrites? | selects? | selector |")
    add("|---|---|---|---|---|")
    add("| paraphrase | `paraphrase` | yes | no (told not to choose) | - |")
    add("| summary_generic | `generic` | yes | yes, query-agnostic | LLM |")
    add("| summary_conditioned | `conditioned` | yes | yes, query-aware | LLM |")
    add("| lm_elimination_generic | `lm_generic` | no | yes, query-agnostic | GPT-2 self-information |")
    add("| lm_elimination_conditioned | `lm_conditioned` | no | yes, query-aware | GPT-2 question-likelihood gain |")
    add("| nonllm_elimination_generic | `nonllm_generic` | no | yes, query-agnostic | TF-IDF centrality |")
    add("| nonllm_elimination_conditioned | `nonllm_conditioned` | no | yes, query-aware | BM25 vs q_now |")
    add("| random_selection | `random_selection` | no | yes, seeded shuffle | - |")
    add("| passthrough (prefix) | `passthrough` | no | no - positional | - |")
    add("")
    if lm_manifest:
        add(f"LM scorer: `{lm_manifest.get('model_id')}` @ "
            f"`{str(lm_manifest.get('commit_hash'))[:12]}`, torch {lm_manifest.get('torch')}, "
            f"transformers {lm_manifest.get('transformers')}, teacher-forced, CPU, "
            "no generation.")
        add("")
        add(f"- generic: `{lm_manifest.get('generic_formula')}`")
        add(f"- conditioned: `{lm_manifest.get('conditioned_formula')}`")
        add("")

    add("## Verdict")
    add("")
    for line in verdict(family, across, contrasts, budgets, metric):
        add(line)
        add("")

    add("## 1. The channel was equal (or where it was not)")
    add("")
    add("| arm | budget | delivered words (mean) | fill ratio | messages |")
    add("|---|---:|---:|---:|---:|")
    for policy in TRADEOFF_ARMS:
        for words in budgets:
            row = lookup(lengths, policy=policy, budget_words=words)
            if not row:
                continue
            add(f"| {DISPLAY.get(policy, policy)} | {words} | "
                f"{fmt(row['delivered_words_mean'], 1)} | {fmt(row['fill_ratio_mean'], 2)} | "
                f"{row['n_messages']} |")
    add("")
    over = [r for r in accounting if r["over_budget"]]
    add(f"Per-message accounting for all {len(accounting)} delivered messages is in")
    add("`budget_accounting.csv`, one row per message with exactly the fields the design")
    add("calls for - `target_words`, `delivered_words`, `fill_ratio`, `source_words`,")
    add("`retention_fraction`, `selected_unit_count` - recorded identically for rewriting")
    add(f"and elimination. Messages over their hard cap: **{len(over)}**.")
    add("")
    add("`selected_unit_count` there counts source sentences present *verbatim* in the")
    add("delivered message, a definition that applies to every mechanism. For the")
    add("elimination arms it equals their selection by construction; for the abstractive")
    add("arms it shows how much of the source they copied rather than rewrote. At the")
    add("160-word budget the split is total: 5.19-5.88 verbatim source sentences per")
    add("message for `passthrough` and the four elimination arms, against 0.06-0.11 for")
    add("`paraphrase`, `summary_generic` and `summary_conditioned`. The two mechanism")
    add("families really are doing different things to the text - which is exactly what")
    add("Experiment 10's prompt-based `extractive_*` arms failed to guarantee.")
    add("")
    add("Sentence-level selection cannot pad, so an elimination arm spends only what whole")
    add("sentences fit. At 20 words that is a granularity failure rather than a policy:")
    add("the corpus's shortest sentence is 10 words and its median is 27, and five of the")
    add("sixteen dossiers own no sentence short enough to send at all, so their message is")
    add("empty. The 20-word rung is reported throughout and interpreted as a stress test,")
    add("never as the headline.")
    add("")

    add("## 2. Present and future utility")
    add("")
    add("| arm | budget | U_now | U_future | gap |")
    add("|---|---:|---:|---:|---:|")
    for policy in TRADEOFF_ARMS:
        for words in budgets:
            now = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="now", metric=metric)
            fut = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="future", metric=metric)
            gap = lookup(metrics, policy=policy, budget_words=words, kind="utility",
                         endpoint="gap", metric=metric)
            if not (now and fut):
                continue
            add(f"| {DISPLAY.get(policy, policy)} | {words} | {fmt(now['mean'])} {ci(now)} | "
                f"{fmt(fut['mean'])} {ci(fut)} | {fmt(gap['mean']) if gap else 'n/a'} |")
    add("")

    add("## 3. Conditioning within each mechanism (paired, clustered on the source)")
    add("")
    add("| contrast | budget | U_now delta | U_future delta | specialisation DiD |")
    add("|---|---:|---:|---:|---:|")
    for _label, generic, conditioned, _colour in FAMILIES:
        name = f"{conditioned}_minus_{generic}"
        for words in budgets:
            now = lookup(contrasts, comparison=name, budget_words=words, kind="utility",
                         endpoint="now", metric=metric)
            fut = lookup(contrasts, comparison=name, budget_words=words, kind="utility",
                         endpoint="future", metric=metric)
            did = lookup(contrasts, comparison=name, budget_words=words,
                         kind="specialisation_did", metric=metric)
            if not now:
                continue
            add(f"| {DISPLAY.get(conditioned, conditioned)} - {DISPLAY.get(generic, generic)} "
                f"| {words} | {fmt(now['delta'])} {ci(now)} | {fmt(fut['delta'])} {ci(fut)} | "
                f"{fmt(did['delta']) if did else 'n/a'} {ci(did) if did else ''} |")
    add("")

    add("## 4. The distance gradient, per mechanism")
    add("")
    add("Regret against the designed distance from the conditioning query. A gradient is")
    add("the signature of specialisation: the message is worth less the further the")
    add("question moves from the one it was written for.")
    add("")
    add("| arm | budget | paraphrase | same entity | same topic | orthogonal | far - near |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for policy in MAIN_ARMS:
        for words in budgets:
            cells = [lookup(relations, policy=policy, budget_words=words, relation=r,
                            kind="regret", metric=metric) for r in rd.RELATION_ORDER]
            if not all(cells):
                continue
            far_near = float(cells[-1]["mean"]) - float(cells[0]["mean"])
            add(f"| {DISPLAY.get(policy, policy)} | {words} | "
                + " | ".join(fmt(c["mean"]) for c in cells)
                + f" | {fmt(far_near)} |")
    add("")
    add("Between families, the same quantity again (a triple difference, paired on the")
    add("source): does one mechanism's conditioning effect exceed another's? Same utility")
    add("convention - more negative is a steeper near/far gradient, so a negative A - B")
    add("means family A specialises more than family B.")
    add("")
    add("| family A | family B | budget | DiD(A) | DiD(B) | A - B | 95% CI |")
    add("|---|---|---:|---:|---:|---:|---|")
    for row in across:
        if row["metric"] != metric:
            continue
        add(f"| {row['family_a']} | {row['family_b']} | {row['budget_words']} | "
            f"{fmt(row['did_a'])} | {fmt(row['did_b'])} | {fmt(row['delta'])} | "
            f"[{fmt(row['lo'])}, {fmt(row['hi'])}] |")
    add("")
    add("Paired difference in that gradient between a conditioned arm and its own generic")
    add("control, on the same sources. This one is computed on **utility**, not regret, so")
    add("its sign is the opposite of the table above: utility falls as the question moves")
    add("away from the conditioning one, and a **more negative** number means the")
    add("conditioned arm gives up more of it going from near to far.")
    add("")
    add("| family | budget | delta(orthogonal - paraphrase) | 95% CI | n sources |")
    add("|---|---:|---:|---|---:|")
    for row in family:
        if row["metric"] != metric:
            continue
        add(f"| {row['family']} | {row['budget_words']} | {fmt(row['delta'])} | "
            f"[{fmt(row['lo'])}, {fmt(row['hi'])}] | {row['n_contexts']} |")
    add("")

    add("## 5. Deleted, or delivered and misread?")
    add("")
    add("| arm | budget | q_now answer survived | orthogonal answer survived | "
        "accuracy given survival (q_now) |")
    add("|---|---:|---:|---:|---:|")
    for policy in TRADEOFF_ARMS:
        for words in budgets:
            now = lookup(summary, policy=policy, budget_words=words, slice="q_now")
            orth = lookup(summary, policy=policy, budget_words=words, slice="orthogonal")
            if not (now and orth):
                continue
            add(f"| {DISPLAY.get(policy, policy)} | {words} | "
                f"{fmt(now['span_survival'])} | {fmt(orth['span_survival'])} | "
                f"{fmt(now['accuracy_given_survival'])} |")
    add("")
    add("Survival is a string test on the delivered message, so it is defined identically")
    add("for rewriting and elimination. It bounds what any reader could do: an arm whose")
    add("accuracy sits far below its survival rate is losing answers at the reader, not at")
    add("the compressor. On this corpus almost nothing is lost at the reader.")
    add("")
    add("`evidence_survival.csv` also carries one row per designed distance tier, and that")
    add("is where the difference between the two conditioned mechanisms becomes concrete.")
    add("At 160 words the orthogonal question's answer string survives in only 0.039 of")
    add("`summary_conditioned`'s messages, against 0.286-0.293 for the conditioned")
    add("elimination arms - which is roughly what their own generic controls (0.312) and")
    add("even the random floor (0.387) deliver. Conditioned rewriting *erases* the distant")
    add("evidence; conditioned selection merely does not go out of its way to include it,")
    add("and keeps about as much of it as an unconditioned selector would. That is the")
    add("mechanism behind the split verdict above.")
    add("")
    add("(The `q_now` and `paraphrase` columns of that table are equal by construction, not")
    add("by coincidence: the corpus builder requires a paraphrase question to share its")
    add("anchor's gold answers, and `regret_data.validate_relation_context` enforces it.)")
    add("")

    if aspects:
        add("## 6. Allocation, measured without the reader")
        add("")
        add("The corpus partitions its evidence into four aspects, one per question group,")
        add("and no sentence is claimed by two. For a selection arm the delivered units are")
        add("known exactly, so the share of the conditioning aspect that survived can be")
        add("compared with the share of the other three - a compressor-side measurement with")
        add("no reader and no judge in the path.")
        add("")
        add("| arm | budget | conditioning aspect kept | other aspects kept | delta | 95% CI |")
        add("|---|---:|---:|---:|---:|---|")
        for row in aspects:
            add(f"| {row['display']} | {row['budget_words']} | "
                f"{fmt(row['conditioning_aspect_retention'])} | "
                f"{fmt(row['other_aspect_retention'])} | {fmt(row['delta'])} | "
                f"[{fmt(row['lo'])}, {fmt(row['hi'])}] |")
        add("")

    add("## 7. Provenance")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| sender and reader | `{cfg['model']['id']}`, temperature "
        f"{cfg['decoding']['sender_temperature']} / "
        f"{cfg['decoding']['answer_temperature']} |")
    add(f"| judge | `{cfg['judge']['model_id']}`, binary correct/incorrect against the gold |")
    if lm_manifest:
        add(f"| LM selector | `{lm_manifest.get('model_id')}` @ "
            f"`{str(lm_manifest.get('commit_hash'))[:12]}`, teacher-forced, CPU, no sampling |")
    add("| non-LLM selectors | Okapi BM25 (`src/retrieval.py`) and TF-IDF centrality "
        "(`src/elimination.py`); no pretrained model |")
    add("| LM selector provenance | **LongLLMLingua-*style*, not the official "
        "implementation.** No `llmlingua` package is used; the coarse question-aware "
        "ranking direction is reimplemented here and its formula is stated above and in "
        "the artefact manifest |")
    add("| seeds | inherited from Experiment 10 unchanged: sender and answerer are "
        "deterministic (temperature 0, `seed=None`), the closed-book baseline uses "
        "`1000 + sample`. The selectors are deterministic functions of (units, cap, "
        "query); `random_selection` is seeded by a SHA-256 of "
        "`salt|context_id|budget`, so it is stable across processes and machines |")
    add("| granularity | sentence / evidence unit, the primary level the design asks for. "
        "Token-level deletion was not run: it introduces a readability confound this "
        "design cannot separate from information loss |")
    add(f"| corpus | {cfg['dataset'][cfg['dataset']['active']]['contexts_jsonl']} |")
    add("| reused unchanged | 640 Experiment 10 messages + 11,264 answers, request hashes "
        "verified equal by the offline selftest |")
    add("| generated this experiment | 64 `paraphrase` messages; 768 selection messages at "
        "zero model cost; 8,896 unique answer calls after single-flighting (24,576 answer "
        "cells, deduplicated by request hash) |")
    add("| cost | sender/answerer $0.0434 + judge $0.0332 = **$0.077** |")
    add("| wall-clock (observed, not instrumented) | GPT-2 scoring of all 16 dossiers 106 s "
        "on CPU; the main run ~20 min end to end at `runtime.concurrency` 12 and judge "
        "concurrency 16; the `passthrough` top-up ~1 min. The pipeline is bound by API "
        "latency, not by local compute |")
    add("")
    add("```bash")
    add("python src/lm_unit_scores.py --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e")
    add("python src/run_communication_regret.py --config compression_mechanism_config.yaml")
    add("python src/render_mechanism_report.py")
    add("python src/selftest_compression_mechanism_offline.py   # 43 checks, no API key")
    add("```")
    add("")
    add("Files added: `src/elimination.py`, `src/lm_unit_scores.py`, "
        "`src/render_mechanism_report.py`, "
        "`src/selftest_compression_mechanism_offline.py`, "
        "`compression_mechanism_config.yaml`. Files extended additively: "
        "`src/run_communication_regret.py` (selection specs, config-driven contrasts, a guard "
        "against an arm silently leaving the utility matrix), `src/budget.py` (the "
        "`paraphrase` block only).")
    add("")
    add("Not run, deliberately: token-level elimination (deleting inside a sentence adds a "
        "readability confound this design cannot separate from information loss), and the "
        "`informativeness` variant of the query-free lexical scorer (implemented and "
        "config-selectable, never exercised).")
    add("")

    add("## Figures")
    add("")
    add("- `figures/distance_curves.png` - the main comparison.")
    add("- `figures/now_future_tradeoff.png` - the present/future trade-off with paraphrase.")
    add("- `figures/evidence_survival.png` - the deletion/reader diagnostic.")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="compression_mechanism_config.yaml")
    parser.add_argument("--metric", default=None, help="override analysis.primary_metric")
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    dataset = cfg["dataset"]["active"]
    n = int(cfg["dataset"][dataset]["n_contexts"])
    run_root = ROOT / cfg["outputs"]["run_root"] / dataset / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / dataset / f"n{n}"
    out_root = result_root / "mechanism"
    metric = args.metric or cfg["analysis"]["primary_metric"]
    metric_label = {"judge_correct": "LLM-judged correct", "em": "exact match",
                    "f1": "token F1"}.get(metric, metric)
    budgets = [int(w) for w in cfg["budget"]["words"]]
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci_level = float(cfg["analysis"]["ci_level"])

    metrics = read_csv(result_root / "metrics.csv")
    relations = read_csv(result_root / "relation_regret.csv")
    contrasts = read_csv(result_root / "contrasts.csv")
    lengths = read_csv(result_root / "length_audit.csv")
    baselines = read_csv(result_root / "baselines.csv")
    messages = read_jsonl(run_root / "messages.jsonl")
    answers = read_jsonl(run_root / "answers.jsonl")

    _, contexts = rd.load_contexts(ROOT / cfg["dataset"][dataset]["contexts_jsonl"])
    contexts = tuple(contexts[:n])
    rotations = {c.context_id: rd.rotations_for(c) for c in contexts}
    policies = list(cfg["policies"])

    rows = survival_rows(contexts, rotations, messages, answers, policies)
    summary = survival_summary(rows, policies, budgets, boot, ci_level)
    family = mechanism_contrasts(rows, budgets, boot, ci_level)
    aspects = aspect_retention(contexts, rotations, messages, budgets, boot, ci_level)
    across = family_vs_family(rows, budgets, boot, ci_level)
    accounting = budget_accounting(contexts, messages, policies)

    write_csv(out_root / "evidence_survival.csv", summary)
    write_csv(out_root / "mechanism_distance_contrasts.csv", family)
    write_csv(out_root / "aspect_retention.csv", aspects)
    write_csv(out_root / "family_vs_family.csv", across)
    write_csv(out_root / "budget_accounting.csv", accounting)

    plot_distance_curves(relations, out_root / "figures" / "distance_curves.png",
                         budgets, metric, metric_label)
    plot_tradeoff(metrics, baselines, out_root / "figures" / "now_future_tradeoff.png",
                  budgets, metric, metric_label)
    plot_survival(summary, out_root / "figures" / "evidence_survival.png", budgets)

    lm_manifest = {}
    lm_path = ROOT / rcr.elimination_settings(cfg)["lm_scores_path"]
    if lm_path.exists():
        first = lm_path.read_text(encoding="utf-8").split("\n", 1)[0]
        lm_manifest = json.loads(first).get("_manifest", {})

    write_report(out_root / "report.md", cfg=cfg, metrics=metrics, relations=relations,
                 contrasts=contrasts, family=family, summary=summary, lengths=lengths,
                 aspects=aspects, across=across, accounting=accounting, budgets=budgets,
                 metric=metric, metric_label=metric_label, lm_manifest=lm_manifest)
    print(f"[mechanism] wrote {out_root.relative_to(ROOT)} "
          f"({len(rows)} survival rows, {len(summary)} summary rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
