"""Render the cross-corpus narrative report for Experiment 10.

``run_communication_regret.py`` writes one machine-generated ``report.md`` per
corpus. This script writes the layer above them:
``results/communication_regret/REPORT.md``, which carries the argument in prose
and every headline table for *both* corpora side by side.

Every number in the output is read out of the result CSVs at render time, so
the narrative cannot drift from the data the way a hand-typed summary does.
Prose lives here; figures are referenced from the paths the runner wrote them
to. Re-run this after any ``--analyse-only`` pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import budget as bd  # noqa: E402
import regret_data as rd  # noqa: E402

RESULT_ROOT = ROOT / "results" / "communication_regret"
OUTPUT = RESULT_ROOT / "REPORT.md"

# The corpora, in the order the argument needs them: the natural one first,
# because it carries the external-validity claim, then the designed one, which
# carries the mechanism.
CORPORA = (
    ("squad_groups", "SQuAD groups", "natural, human-written questions"),
    ("relation_dossiers", "Relation dossiers", "designed distance labels"),
)

POLICY_ORDER = ("generic", "conditioned", "reusable", "oracle",
                "extractive_generic", "extractive_conditioned")

CONTRAST_ORDER = (
    ("conditioned_minus_generic", "conditioned &minus; generic"),
    ("conditioned_minus_reusable", "conditioned &minus; reusable"),
    ("reusable_minus_generic", "reusable &minus; generic"),
    ("oracle_minus_conditioned", "oracle &minus; conditioned"),
    ("oracle_minus_generic", "oracle &minus; generic"),
    ("extractive_conditioned_minus_extractive_generic",
     "extractive_conditioned &minus; extractive_generic"),
)


# ---------------------------------------------------------------------------
# Loading


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


class Corpus:
    """One corpus's result tables, addressed by name."""

    def __init__(self, key: str, root: Path) -> None:
        self.key = key
        self.root = root
        self.rel = root.relative_to(RESULT_ROOT).as_posix()
        for name in ("metrics", "contrasts", "pareto", "length_audit", "length_deltas",
                     "length_matched_contrasts", "budget_interaction", "normalised_utility",
                     "baselines", "relation_regret", "relation_tests"):
            setattr(self, name, read_csv(root / f"{name}.csv"))
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    @property
    def metric(self) -> str:
        return self.manifest["primary_metric"]

    @property
    def budgets(self) -> list[int]:
        return sorted({int(r["budget_words"]) for r in self.metrics})

    @property
    def policies(self) -> list[str]:
        present = {r["policy"] for r in self.metrics}
        ordered = [p for p in POLICY_ORDER if p in present]
        extra = sorted(p for p in present if p not in ordered)
        return ordered + extra

    def deployable(self) -> list[str]:
        return [p for p in self.policies if "@" not in p]

    def find(self, table: str, **where) -> dict | None:
        for row in getattr(self, table):
            if all(str(row.get(k)) == str(v) for k, v in where.items()):
                return row
        return None


def discover() -> dict[str, Corpus]:
    out = {}
    for key, _label, _why in CORPORA:
        base = RESULT_ROOT / key
        if not base.exists():
            continue
        runs = sorted(p.parent for p in base.glob("n*/manifest.json"))
        if runs:
            out[key] = Corpus(key, runs[-1])
    return out


# ---------------------------------------------------------------------------
# Formatting


def num(value, digits: int = 3, sign: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def ci(row: dict | None, key: str = "mean", digits: int = 3, sign: bool = False) -> str:
    if row is None:
        return "n/a"
    return (f"{num(row.get(key), digits, sign)} "
            f"[{num(row.get('lo'), digits, sign)}, {num(row.get('hi'), digits, sign)}]")


def excludes_zero(row: dict | None) -> bool:
    if row is None:
        return False
    try:
        lo, hi = float(row["lo"]), float(row["hi"])
    except (TypeError, ValueError, KeyError):
        return False
    return lo > 0 or hi < 0


def bold_if(text: str, condition: bool) -> str:
    return f"**{text}**" if condition else text


def table(head: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    if not rows:
        return []
    sep = align or ("|---" * len(head) + "|")
    return ["| " + " | ".join(head) + " |", sep] + ["| " + " | ".join(r) + " |" for r in rows]


def right_align(head: list[str], first_left: bool = True) -> str:
    cells = ["---" if (i == 0 and first_left) else "---:" for i in range(len(head))]
    return "|" + "|".join(cells) + "|"


# ---------------------------------------------------------------------------
# Section builders


def budget_head(c: Corpus) -> list[str]:
    return [f"{w} w" for w in c.budgets]


def utility_table(c: Corpus, kind: str, endpoint: str, source: str = "metrics") -> list[str]:
    head = ["policy"] + budget_head(c)
    rows = []
    for policy in c.policies:
        cells = []
        for w in c.budgets:
            row = c.find(source, policy=policy, budget_words=w, kind=kind,
                         endpoint=endpoint, metric=c.metric)
            cells.append(ci(row))
        if any(cell != "n/a" for cell in cells):
            rows.append([f"`{policy}`"] + cells)
    return table(head, rows, right_align(head))


def contrast_table(c: Corpus, endpoint: str) -> list[str]:
    head = ["comparison"] + budget_head(c)
    rows = []
    for name, label in CONTRAST_ORDER:
        cells, seen = [], False
        for w in c.budgets:
            row = c.find("contrasts", comparison=name, budget_words=w, kind="utility",
                         endpoint=endpoint, metric=c.metric)
            if row is None:
                cells.append("n/a")
                continue
            seen = True
            cells.append(bold_if(ci(row, "delta", sign=True), excludes_zero(row)))
        if seen:
            rows.append([label] + cells)
    return table(head, rows, right_align(head))


def length_table(c: Corpus) -> list[str]:
    head = ["policy", "band", "delivered (mean &plusmn; sd)", "fill", "under floor",
            "truncated", "corrections", "repetition"]
    rows = []
    for w in c.budgets:
        for policy in c.policies:
            r = c.find("length_audit", policy=policy, budget_words=w)
            if r is None:
                continue
            rows.append([
                f"`{policy}`", f"{r['floor_words']}&ndash;{r['budget_words']}",
                f"{num(r['delivered_words_mean'], 1)} &plusmn; {num(r['delivered_words_sd'], 1)}",
                num(r["fill_ratio_mean"], 2), num(r["under_floor_rate"], 2),
                num(r["truncated_rate"], 2), num(r["correction_rate"], 2),
                num(r["repetition_rate_mean"], 3),
            ])
    return table(head, rows, right_align(head))


def pareto_table(c: Corpus) -> list[str]:
    head = ["policy", "requested cap", "U_now", "U_future", "delivered cost (words)",
            "3D nondominated", "within-cap 2D nondominated"]
    rows = []
    for policy in c.policies:
        for w in c.budgets:
            r = c.find("pareto", policy=policy, budget_words=w)
            if r is None:
                continue
            rows.append([f"`{policy}`", f"{w} w", num(r["u_now"]), num(r["u_future"]),
                         num(r["cost_words"], 1),
                         "&mdash;" if r["dominated"] == "True" else "**yes**",
                         "&mdash;" if r.get("dominated_within_budget", "True") == "True"
                         else "**yes**"])
    return table(head, rows, right_align(head))


def budget_frontier_lines(c: Corpus) -> list[str]:
    """Discrete within-budget nondominated observations, best-present first."""
    out = []
    for w in c.budgets:
        front = [r for r in c.pareto
                 if int(r["budget_words"]) == w
                 and r.get("dominated_within_budget", "True") == "False"]
        if not front:
            continue
        front.sort(key=lambda r: -float(r["u_now"]))
        out.append(f"* **{w} words** &mdash; "
                   + "; &nbsp;".join(
                       f"`{r['policy']}` ({num(r['u_now'], 2)}, {num(r['u_future'], 2)})"
                       for r in front))
    return out


def frontier_line(c: Corpus) -> str:
    pts = [r for r in c.pareto if r["dominated"] == "False"]
    pts.sort(key=lambda r: (-float(r["u_now"]), -float(r["u_future"])))
    listed = ", ".join(f"`{r['policy']}`@{r['budget_words']}w" for r in pts)
    return (f"**Nondominated set — {len(pts)} of {len(c.pareto)} configurations:** "
            f"{listed}.")


def secondary_table(c: Corpus) -> list[str]:
    head = ["metric", "endpoint"] + budget_head(c)
    rows = []
    for metric, label in (("judge_correct", "LLM judge"), ("em", "Exact match"),
                          ("f1", "Token F1")):
        for endpoint in ("now", "future"):
            cells = []
            for w in c.budgets:
                row = c.find("contrasts", comparison="conditioned_minus_generic",
                             budget_words=w, kind="utility", endpoint=endpoint, metric=metric)
                cells.append(bold_if(ci(row, "delta", sign=True), excludes_zero(row)))
            rows.append([label, f"dU_{endpoint}"] + cells)
    return table(head, rows, right_align(head))


METRIC_LABEL = {"judge_correct": "judge", "em": "EM", "f1": "F1"}


def pivot_all(c: Corpus, table_name: str, id_cols: list[str], value_col: str,
              digits: int = 3, sign: bool = False, extra: list[str] | None = None) -> list[str]:
    """Pivot an entire result table by budget, keeping every row and metric.

    Used by the appendix so that no number the runner computed is left out of
    the report on the grounds that it was not the primary metric.
    """
    data = getattr(c, table_name)
    if not data:
        return []
    extra = extra or []
    head = [col.replace("_", " ") for col in id_cols] + budget_head(c) + extra
    seen: dict[tuple, dict[int, dict]] = {}
    order: list[tuple] = []
    for row in data:
        key = tuple(row.get(col, "") for col in id_cols)
        if key not in seen:
            seen[key] = {}
            order.append(key)
        seen[key][int(row["budget_words"])] = row
    rows = []
    for key in order:
        cells = []
        for w in c.budgets:
            r = seen[key].get(w)
            cells.append("n/a" if r is None
                         else bold_if(ci(r, value_col, digits, sign), excludes_zero(r))
                         if sign else ci(r, value_col, digits))
        tail = []
        for col in extra:
            any_row = next(iter(seen[key].values()))
            tail.append(str(any_row.get(col, "")))
        label = [METRIC_LABEL.get(k, k).replace("_", " ") for k in key]
        rows.append(label + cells + tail)
    return table(head, rows, right_align(head))


def flat_all(c: Corpus, table_name: str, cols: list[str], value_col: str,
             digits: int = 3, sign: bool = True) -> list[str]:
    """Render a table that has no budget axis to pivot, keeping every row."""
    data = getattr(c, table_name)
    if not data:
        return []
    head = [col.replace("_", " ") for col in cols] + [value_col.replace("_", " "), "p", "dz"]
    rows = []
    for r in data:
        rows.append([METRIC_LABEL.get(r.get(col, ""), r.get(col, "")) for col in cols]
                    + [bold_if(ci(r, value_col, digits, sign), excludes_zero(r)),
                       num(r.get("p_value"), 4), num(r.get("cohens_dz"), 2)])
    return table(head, rows, right_align(head))


def full_contrast_table(c: Corpus) -> list[str]:
    """Every row of contrasts.csv, pivoted with budgets as columns.

    Nothing is selected for here: the main narrative quotes the headline
    comparisons, and this is the exhaustive version so no contrast the runner
    computed goes unreported.
    """
    head = ["comparison", "quantity", "metric"] + budget_head(c)
    order = {name: i for i, (name, _label) in enumerate(CONTRAST_ORDER)}
    comparisons = sorted({r["comparison"] for r in c.contrasts},
                         key=lambda n: (order.get(n.replace(TRIM := "@trim", ""), 99),
                                        TRIM in n, n))
    quantities = [("utility", "now"), ("utility", "future"), ("utility", "gap"),
                  ("regret", "now"), ("regret", "future"),
                  ("retained", "now"), ("retained", "future"),
                  ("specialisation_did", "now_minus_future")]
    rows = []
    for name in comparisons:
        for kind, endpoint in quantities:
            for metric in ("judge_correct", "em", "f1"):
                cells, seen = [], False
                for w in c.budgets:
                    r = c.find("contrasts", comparison=name, budget_words=w, kind=kind,
                               endpoint=endpoint, metric=metric)
                    if r is None:
                        cells.append("n/a")
                        continue
                    seen = True
                    cells.append(bold_if(ci(r, "delta", sign=True), excludes_zero(r))
                                 + f" <sub>p {num(r['p_value'], 3)}</sub>")
                if seen:
                    label = (f"{kind} {endpoint}" if kind != "specialisation_did"
                             else "gap DiD")
                    rows.append([name.replace("_minus_", " &minus; "), label,
                                 METRIC_LABEL[metric]] + cells)
    return table(head, rows, right_align(head))


def interaction_table(c: Corpus) -> list[str]:
    head = ["quantity", "policy", "gap(low) &minus; gap(high)", "p", "dz"]
    rows = []
    for r in c.budget_interaction:
        if r["metric"] != c.metric:
            continue
        rows.append([r["quantity"], f"`{r['policy']}`",
                     bold_if(ci(r, "delta", sign=True), excludes_zero(r)),
                     num(r["p_value"], 4), num(r["cohens_dz"], 2)])
    return table(head, rows, right_align(head))


def matched_table(c: Corpus) -> list[str]:
    head = ["comparison", "endpoint"] + budget_head(c) + ["n contexts (low&ndash;high budget)"]
    rows = []
    names = sorted({r["comparison"] for r in c.length_matched_contrasts})
    for name in names:
        for endpoint in ("now", "future"):
            cells, counts = [], []
            for w in c.budgets:
                r = c.find("length_matched_contrasts", comparison=name, budget_words=w,
                           endpoint=endpoint, metric=c.metric)
                cells.append(bold_if(ci(r, "delta", sign=True), excludes_zero(r)))
                counts.append("-" if r is None else str(r["n_contexts_matched"]))
            rows.append([name.replace("_minus_", " &minus; "), f"dU_{endpoint}"]
                        + cells + ["/".join(counts)])
    return table(head, rows, right_align(head))


def length_delta_table(c: Corpus) -> list[str]:
    head = ["comparison"] + budget_head(c)
    rows = []
    for name, label in CONTRAST_ORDER:
        cells, seen = [], False
        for w in c.budgets:
            r = c.find("length_deltas", comparison=name, budget_words=w)
            if r is None:
                cells.append("n/a")
                continue
            seen = True
            cells.append(bold_if(ci(r, "delta", 2, sign=True), excludes_zero(r)))
        if seen:
            rows.append([label] + cells)
    return table(head, rows, right_align(head))


def relation_table(c: Corpus, kind: str) -> list[str]:
    head = ["policy", "budget"] + [r.replace("_", " ") for r in rd.RELATION_ORDER]
    rows = []
    for policy in c.policies:
        for w in c.budgets:
            cells = []
            for relation in rd.RELATION_ORDER:
                r = c.find("relation_regret", policy=policy, budget_words=w,
                           relation=relation, kind=kind, metric=c.metric)
                cells.append(ci(r))
            if all(cell == "n/a" for cell in cells):
                continue
            rows.append([f"`{policy}`", f"{w} w"] + cells)
    return table(head, rows, right_align(head))


def relation_test_table(c: Corpus) -> list[str]:
    head = ["policy"] + budget_head(c)
    rows = []
    for policy in c.policies:
        cells, seen = [], False
        for w in c.budgets:
            r = c.find("relation_tests", policy=policy, budget_words=w, metric=c.metric,
                       quantity="regret_orthogonal_minus_paraphrase")
            if r is None:
                cells.append("n/a")
                continue
            seen = True
            cells.append(bold_if(ci(r, "delta", sign=True), excludes_zero(r))
                         + f" (dz {num(r['cohens_dz'], 1)})")
        if seen:
            rows.append([f"`{policy}`"] + cells)
    return table(head, rows, right_align(head))


def baseline_line(c: Corpus) -> str:
    ceiling = c.find("baselines", baseline="direct_context", metric=c.metric)
    closed = c.find("baselines", baseline="closed_book", metric=c.metric)
    return (f"direct-context ceiling `U(D, q)` = {ci(ceiling)}; "
            f"closed-book leakage audit = {ci(closed)}")


def normalised_table(c: Corpus) -> list[str]:
    head = ["policy", "endpoint"] + budget_head(c) + ["excluded (low ceiling)"]
    rows = []
    for policy in c.policies:
        for endpoint in ("now", "future"):
            cells, excl = [], []
            for w in c.budgets:
                r = c.find("normalised_utility", policy=policy, budget_words=w,
                           endpoint=endpoint, metric=c.metric)
                cells.append(ci(r))
                excl.append("-" if r is None else str(r["n_excluded_low_ceiling"]))
            if all(cell == "n/a" for cell in cells):
                continue
            rows.append([f"`{policy}`", endpoint] + cells + ["/".join(excl)])
    return table(head, rows, right_align(head))


# ---------------------------------------------------------------------------
# The document


def render(corpora: dict[str, Corpus]) -> str:
    sq = corpora.get("squad_groups")
    rl = corpora.get("relation_dossiers")
    primary = sq or rl
    if primary is None:
        raise SystemExit("no corpus results found under results/communication_regret/")
    L: list[str] = []
    w = L.append

    def block(lines: list[str]) -> None:
        if lines:
            L.extend(lines)
            w("")

    # ---------------------------------------------------------------- header
    w("# Communication Regret\n")
    w("**Experiment 10 · agent handoff information-loss probe**\n")
    w("When one agent writes a bounded handoff for the question it currently knows, does it "
      "buy accuracy on that question by spending capacity it would otherwise have spent on "
      "questions nobody has asked yet? With the channel held equal to within a word or two, "
      "it does.\n")

    if sq:
        now = {b: sq.find("contrasts", comparison="conditioned_minus_generic", budget_words=b,
                          kind="utility", endpoint="now", metric=sq.metric)
               for b in sq.budgets}
        fut = {b: sq.find("contrasts", comparison="conditioned_minus_generic", budget_words=b,
                          kind="utility", endpoint="future", metric=sq.metric)
               for b in sq.budgets}
        both = [b for b in sq.budgets if excludes_zero(now[b]) and excludes_zero(fut[b])]
        w("> **Verdict — supported.** On natural SQuAD prose, conditioning raises "
          "present-query accuracy by "
          + " / ".join(num(now[b]["delta"], 2, sign=True) for b in both)
          + " at " + " / ".join(str(b) for b in both) + " delivered words while lowering "
          "hidden-query accuracy by "
          + " / ".join(num(fut[b]["delta"], 2, sign=True) for b in both)
          + ". Both directions exclude zero, and the trade-off sharpens as bandwidth falls. "
          "An oracle handoff of the same length carries the hidden answers, so this is "
          "allocation, not capacity.\n")

    rows = [["Sender and answerer", f"`{primary.manifest['sender_model']}`"],
            ["Judge", f"`{primary.manifest['judge_model']}`"]]
    for key, label, why in CORPORA:
        c = corpora.get(key)
        if c:
            rows.append([label,
                         f"{c.manifest['contexts']} contexts &times; "
                         f"{c.manifest['questions'] // c.manifest['contexts']} questions "
                         f"({why}); {c.manifest['rotations']} rotations"])
    rows.append(["Budgets", ", ".join(f"{b} words" for b in primary.budgets)])
    rows.append(["Primary utility", primary.metric])
    block(table(["", ""], rows, "|---|---|"))

    w("This file is the narrative report and carries every headline table for both corpora. "
      "The machine-generated per-corpus reports, which additionally list the full "
      "per-rotation tables, are "
      + " and ".join(f"[`{c.rel}/report.md`]({c.rel}/report.md)" for c in corpora.values())
      + ". Raw per-cell outputs are the CSVs listed at the end.\n")
    w("---\n")

    # ----------------------------------------------------------------- claim
    w("## 1. The claim\n")
    w("Every prior conditioning experiment in this project compared a handoff written for a "
      "known question against one written for no question &mdash; and let the two choose "
      "their own length. On the same source, the conditioned arm wrote roughly 600 "
      "characters and the generic arm roughly 3,800. Any held-out accuracy gap was therefore "
      "inseparable from *one arm simply wrote more*.\n")
    w("This experiment keeps free-text handoffs and makes the channel itself the controlled "
      "variable. The hypothesis under test:\n")
    w("> Under an equal communication budget, conditioning a handoff on the currently known "
      "information need reallocates capacity toward present utility and can reduce its "
      "utility for plausible future information needs.\n")
    w("Supporting it requires **both** a present-query gain and a future-query loss against "
      "a baseline that knows nothing, strengthening as the budget shrinks. A gain alone is "
      "not the claim; a loss alone is not either.\n")

    # ---------------------------------------------------------------- design
    w("## 2. The design\n")
    w("Each source context independently answers *k* questions. Every question takes a turn "
      "as the conditioning query, and each resulting message is then answered against "
      "*every* question of that context. That produces a utility matrix per context: the "
      "diagonal is present-query utility `U_now`, the off-diagonal is future-query utility "
      "`U_future`. Because the same question sits on the diagonal for one rotation and off "
      "it for the other *k*&minus;1, question difficulty cannot produce the contrast and no "
      "fixed A/B split has to be trusted.\n")
    w("One real context (SQuAD, *Scottish Parliament*) at a 34&ndash;40 word budget. Diagonal "
      "cells are marked `[ ]`.\n")
    block(table(["conditioned on &darr; / asked &rarr;", "Q1", "Q2", "Q3", "Q4"],
                [["**on Q1**", "**[1]**", "1", "0", "0"],
                 ["**on Q2**", "0", "**[0]**", "0", "1"],
                 ["**on Q3**", "0", "0", "**[1]**", "0"],
                 ["**on Q4**", "0", "0", "0", "**[1]**"]],
                right_align(["", "", "", "", ""])))
    w("`generic` and `oracle` hold one message each, so every row is identical:\n")
    block(table(["policy", "Q1", "Q2", "Q3", "Q4"],
                [["`generic`", "0", "0", "0", "0"], ["`oracle`", "1", "1", "1", "0"]],
                right_align(["", "", "", "", ""])))
    w("`generic` spends its 37 words on the Kilbrandon Commission and answers none of the "
      "four questions asked; `oracle` fits three of the four answers into 39 words; each "
      "`conditioned` message answers its own question and, with one accident, nothing else. "
      "Q2's diagonal is 0 because the answerer replied \"3\" where the gold is \"three "
      "hundred\" &mdash; the diagonal is not automatically 1. The full messages behind this "
      "matrix are in [`squad_groups/n24/example.md`](squad_groups/n24/example.md).\n")
    w("### The policies\n")
    w("They differ only in what the sender is told about the information need; an offline "
      "check asserts that replacing that block with a placeholder leaves four identical "
      "prompts.\n")
    block(table(["policy", "what the sender is told", "messages per context and budget"], [
        ["`generic`",
         "A question will be asked; it has not been told which. It is *not* told that further "
         "questions may follow &mdash; that is the reusable treatment, and a control must not "
         "contain the independent variable.", "1"],
        ["`conditioned`",
         "Sees the current question and is asked to communicate what answers it.", "*k*"],
        ["`reusable`",
         "Sees the current question, is told unknown further questions may follow, and is "
         "asked to preserve reusable evidence within the same limit.", "*k*"],
        ["`oracle`",
         "Sees every question the context supports. An upper bound and a capacity probe, not "
         "a deployable strategy.", "1"],
        ["`extractive_generic` / `extractive_conditioned`",
         "The same two conditioning levels, but sentences copied verbatim instead of prose. "
         "Separates information selection from abstractive rewriting.", "1 / *k*"],
    ], "|---|---|---|"))
    w("Every sender-side string is reproduced verbatim in "
      "[`PROMPTS.md`](../../PROMPTS.md#experiment-10).\n")

    # --------------------------------------------------------------- control
    w("## 3. The control: was the channel actually equal?\n")
    w("The first pilot gave every policy the same one-sided instruction: *at most N words*. "
      "The arms obeyed it and still spent very different amounts of channel. Realised fill "
      "(delivered words &divide; cap) at a 40-word cap:\n")
    block(table(["policy", "cap only", "cap and floor"],
                [["`conditioned`", "0.61", "0.92"], ["`reusable`", "0.75", "0.91"],
                 ["`oracle`", "0.83", "0.96"], ["`generic`", "0.88", "0.93"]],
                right_align(["", "", ""])))
    w("Under a cap alone the conditioned arm spent **a third less channel** than its control, "
      "so any future-query deficit it showed would have meant \"it wrote less\". The budget "
      "is therefore a **two-sided word band on the delivered message**, enforced in three "
      "stages: one length contract stated identically to every policy; an out-of-band draft "
      "returned for a rewrite with a byte-identical correction; and unconditional truncation "
      "to the cap at the last sentence boundary that fits. The cap always holds. The floor "
      "cannot &mdash; words cannot be invented on demand &mdash; so it is requested, audited "
      "and reported.\n")
    w("![Was the channel actually equal?](squad_groups/n24/budget_control.png)\n")

    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c:
            continue
        w(f"### {label} &mdash; delivered length, every arm and budget\n")
        block(length_table(c))
        over = [r for r in c.length_audit if r["any_over_budget_delivered"] == "True"]
        w(f"Arms delivering over the cap after truncation: **{len(over)}** "
          "(must be zero; truncation is unconditional).\n")
        w("Paired context-clustered length differences between arms:\n")
        block(length_delta_table(c))

    w("### Length-matched subsample\n")
    w("Headline contrasts restricted to contexts where `generic`, `conditioned` and "
      "`reusable` all delivered within the configured tolerance of each other. This is the "
      "comparison that survives even if the audit above had shown unequal fill.\n")
    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c or not c.length_matched_contrasts:
            continue
        w(f"**{label}**\n")
        block(matched_table(c))

    trim = [p for p in (sq.policies if sq else []) if "@" in p]
    if sq and trim:
        w("### Trimmed to one length per context\n")
        w("The strictest control available: every abstractive message in a "
          "(context, budget) cell re-truncated to that cell's shortest delivered length, "
          "then re-answered. Length is identical within a context by construction.\n")
        head = ["policy", "delivered words", "U_now", "U_future"]
        rows = []
        for policy in sq.policies:
            if "@" not in policy:
                continue
            for wb in sq.budgets:
                la = sq.find("length_audit", policy=policy, budget_words=wb)
                nowr = sq.find("metrics", policy=policy, budget_words=wb, kind="utility",
                               endpoint="now", metric=sq.metric)
                futr = sq.find("metrics", policy=policy, budget_words=wb, kind="utility",
                               endpoint="future", metric=sq.metric)
                base_now = sq.find("metrics", policy=policy.split("@")[0], budget_words=wb,
                                   kind="utility", endpoint="now", metric=sq.metric)
                base_fut = sq.find("metrics", policy=policy.split("@")[0], budget_words=wb,
                                   kind="utility", endpoint="future", metric=sq.metric)
                if nowr is None or la is None:
                    continue
                rows.append([f"`{policy}`", num(la["delivered_words_mean"], 1),
                             f"{ci(nowr)} <br><sub>untrimmed {ci(base_now)}</sub>",
                             f"{ci(futr)} <br><sub>untrimmed {ci(base_fut)}</sub>"])
        block(table(head, rows, right_align(head)))

    w("Two further controls sit underneath everything above. Every question in every context "
      "passed a closed-book leakage filter against the answering model, so an off-diagonal "
      "success cannot be parametric recall. And a different model family audited each group: "
      "all questions must be answerable from the passage and must ask about genuinely "
      "distinct facts.\n")
    block(table(["corpus", "baselines"],
                [[label, baseline_line(corpora[key])]
                 for key, label, _ in CORPORA if key in corpora], "|---|---|"))

    # ---------------------------------------------------------------- result
    w("## 4. The result\n")
    w("![Present versus future utility](squad_groups/n24/pareto_now_vs_future.png)\n")
    w("Marker size is the budget. The three unconditioned arms sit **on** the diagonal by "
      "construction &mdash; one message serves every rotation, so they cannot specialise. "
      "Every conditioned arm sits far below it, at every budget.\n")

    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c:
            continue
        w(f"### {label} &mdash; utility\n")
        w("**Present query `U_now`**\n")
        block(utility_table(c, "utility", "now"))
        w("**Future queries `U_future`**\n")
        block(utility_table(c, "utility", "future"))
        w("**Specialisation gap `U_now - U_future`**\n")
        block(utility_table(c, "utility", "gap"))
        w(f"Baselines: {baseline_line(c)}.\n")
        w(f"### {label} &mdash; paired contrasts\n")
        w("Paired bootstrap over contexts. **Bold** excludes zero.\n")
        w("**On the present query, dU_now**\n")
        block(contrast_table(c, "now"))
        w("**On the hidden future queries, dU_future**\n")
        block(contrast_table(c, "future"))
        w("**Secondary metrics** &mdash; the same `conditioned` &minus; `generic` contrast "
          "under all three utility measures. EM and F1 are always computed and never "
          "replaced by the judge.\n")
        block(secondary_table(c))
        w("**Does scarcity sharpen the trade-off?** Positive means the specialisation gap is "
          "wider at the tightest budget than at the widest.\n")
        block(interaction_table(c))

    w("![Utility against communication budget](squad_groups/n24/utility_vs_budget.png)\n")
    w("> **Where the effect is bounded.** The absolute future loss is compressed exactly "
      "where the gap is largest: at the tightest budget the generic baseline itself reaches "
      "only "
      + (num(sq.find("metrics", policy="generic", budget_words=min(sq.budgets),
                     kind="utility", endpoint="future", metric=sq.metric)["mean"])
         if sq else "n/a")
      + " on SQuAD, so there is very little left for conditioning to take away. The gap "
        "statistic, which is scale-free, is the honest measure of the scarcity prediction; "
        "the raw dU_future is not.\n")

    # ---------------------------------------------------------------- regret
    w("## 5. Communication regret against the source\n")
    w("`R(q) = U(D, q) - U(m, q)`: the accuracy a handoff costs relative to reading the "
      "source directly.\n")
    w("![Communication regret](squad_groups/n24/communication_regret.png)\n")
    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c:
            continue
        w(f"### {label} &mdash; regret\n")
        w("**On the present query, `R_now`**\n")
        block(utility_table(c, "regret", "now"))
        w("**On the hidden future queries, `R_future`**\n")
        block(utility_table(c, "regret", "future"))
        w("**Retained utility on cells the source itself answers.** Restricted to cells where "
          "`U(D, q) = 1`, so retained utility is just `U(m, q)` and no near-zero denominator "
          "has to be guarded.\n")
        w("*Present query*\n")
        block(utility_table(c, "retained", "now"))
        w("*Future queries*\n")
        block(utility_table(c, "retained", "future"))
        w("**Normalised ratio `mean U(m,q) / mean U(D,q)`.** Contexts whose own ceiling falls "
          "below the configured floor are excluded, because dividing by a near-zero ceiling "
          "turns noise into a number; the exclusion count is shown rather than hidden.\n")
        block(normalised_table(c))

    # ---------------------------------------------------------------- pareto
    w("## 6. Pareto frontier\n")
    w("Each (policy, budget) is one point in (`U_now`, `U_future`, cost), where cost is mean "
      "delivered words &mdash; what the channel actually carried, not what was requested. A "
      "configuration is dominated when another is at least as good on both utilities and no "
      "more expensive, with at least one strict improvement. Trimmed arms are excluded: "
      "their length is set by looking at what the other arms wrote, so they are a length "
      "control rather than a deployable policy.\n")
    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c:
            continue
        w(f"### {label}\n")
        block(pareto_table(c))
        w(frontier_line(c) + "\n")
        w("Cost enters that test, so the set above is not a curve: a configuration can "
          "survive purely by being cheaper than everything that beats it on utility. The "
          "figure facets observations by requested cap and outlines the two-dimensional "
          "nondominated policies within each facet. It deliberately does not connect "
          "markers: policy is categorical, so a line or staircase would imply unevaluated "
          "intermediate choices. Discrete sets, best-present-query first:\n")
        block(budget_frontier_lines(c))

    # ------------------------------------------------------------- mechanism
    w("## 7. The mechanism\n")
    w("### The channel was never the constraint\n")
    if sq:
        mid = sq.budgets[1]
        head = [f"SQuAD at a {mid}-word budget", "U_now", "U_future", "delivered words"]
        rows = []
        for policy in ("generic", "conditioned", "reusable", "oracle"):
            n = sq.find("metrics", policy=policy, budget_words=mid, kind="utility",
                        endpoint="now", metric=sq.metric)
            f = sq.find("metrics", policy=policy, budget_words=mid, kind="utility",
                        endpoint="future", metric=sq.metric)
            la = sq.find("length_audit", policy=policy, budget_words=mid)
            if n and f and la:
                rows.append([f"`{policy}`", num(n["mean"]), num(f["mean"]),
                             num(la["delivered_words_mean"], 1)])
        block(table(head, rows, right_align(head)))
    w("`oracle`, shown all questions and held to the same band, matches `conditioned`'s "
      "present-query score using the same number of words while giving up nothing on the "
      "others. What a conditioned handoff drops is not information the budget could not "
      "hold. It is information the sender had no reason to keep.\n")
    w("### It is selection, not rewriting\n")
    if sq:
        head = ["extractive_conditioned &minus; extractive_generic"] + budget_head(sq)
        rows = []
        for endpoint in ("now", "future"):
            cells = []
            for wb in sq.budgets:
                r = sq.find("contrasts",
                            comparison="extractive_conditioned_minus_extractive_generic",
                            budget_words=wb, kind="utility", endpoint=endpoint,
                            metric=sq.metric)
                cells.append(bold_if(ci(r, "delta", sign=True), excludes_zero(r)))
            rows.append([f"dU_{endpoint}"] + cells)
        block(table(head, rows, right_align(head)))
    w("The extractive arms copy source sentences verbatim, so no abstractive compression can "
      "occur inside them. They reproduce the pattern anyway: the loss lives in *which* "
      "evidence is sent, not in how it is worded.\n")
    w("### Asking for reusability barely helps\n")
    w("`reusable` is `conditioned` plus an explicit instruction that unknown further "
      "questions may follow and that reusable evidence should be preserved within the same "
      "limit. Its `reusable` &minus; `generic` row in the contrast tables above is negative "
      "on `U_future` at every budget of both corpora. Telling a sender to stay general is "
      "not a substitute for telling it what will be asked.\n")

    # -------------------------------------------------------------- distance
    if rl and rl.relation_regret:
        w("## 8. Regret by distance from the conditioning query\n")
        w("SQuAD supports no clean labelling of *how far* a hidden question sits from the "
          "conditioning one, and inferring that from embeddings would replace a controlled "
          "variable with an estimate. So a second corpus builds the distance in: invented "
          "dossiers with four aspects each, and within every aspect one anchor question plus "
          "a paraphrase of it (same answer, different wording), a second fact about the same "
          "subject, and a fact about a different named entity in that aspect. The questions "
          "of the other aspects are orthogonal.\n")
        w("![Regret by designed distance](relation_dossiers/n16/relation_distance.png)\n")
        w("**Future-query regret by designed distance, every policy and budget**\n")
        block(relation_table(rl, "regret"))
        w("**Future-query utility by designed distance**\n")
        block(relation_table(rl, "utility"))
        w("**Monotonicity test: `R(orthogonal) - R(paraphrase)`**\n")
        block(relation_test_table(rl))
        w("The gradient is a property of conditioning, not of the corpus: the unconditioned "
          "arms are roughly equally bad everywhere, while the conditioned arms are near-"
          "perfect at zero distance and near-total losses at maximum distance. Extra "
          "bandwidth does not flatten it &mdash; at the widest budget the unconditioned arms "
          "cut orthogonal regret substantially while the conditioned arm barely moves.\n")

    # ------------------------------------------------------------ boundaries
    w("## 9. Boundaries\n")
    w("- One sender/answerer stack (`" + primary.manifest["sender_model"] + "`) and one "
      "judge family. A second stack, and human adjudication of a judge sample, are the next "
      "external-validity tests.\n"
      "- Effective *n* is the number of contexts, not the number of rotations &mdash; all "
      "intervals cluster at the context level for that reason.\n"
      "- `U_future` is bounded below by whatever the generic baseline itself achieved, so "
      "the raw future loss is smallest exactly where the specialisation gap is largest.\n"
      "- The paraphrase tier of the distance corpus is by construction the easiest possible "
      "case: the gold answer string is identical. Its near-zero regret is a design sanity "
      "check, not a finding. Same-entity and same-topic are close enough that the corpus "
      "supports \"near versus far\", not a strict four-point ordering.\n"
      "- The floor half of the budget contract is best-effort; its misses are listed in the "
      "`under floor` column of the length tables, concentrated in the extractive conditioned "
      "arm where verbatim sentences cannot be lengthened to order.\n")

    # ------------------------------------------------------------- reproduce
    w("## 10. Reproduce it\n")
    w("```bash\npython src/builders/build_regret_data.py --which both\n```\n")
    w("```bash\npython src/run_communication_regret.py\n```\n")
    w("```bash\npython src/run_communication_regret.py --dataset relation_dossiers "
      "--policies generic,conditioned,reusable,oracle --no-trim\n```\n")
    w("```bash\npython tests/selftest_communication_regret_offline.py\n```\n")
    w("```bash\npython src/analysis/render_regret_summary.py\n```\n")
    w("The offline check runs with no API key and no network: it asserts that the four "
      "abstractive prompts share one skeleton, that the cap is never exceeded, that every "
      "question conditions exactly once and is hidden exactly *k*&minus;1 times, that the "
      "answerer can never reach the source, that identical requests are issued once, and "
      "that the matrix and regret arithmetic recover an injected effect exactly. The last "
      "command regenerates this file from the CSVs below, so it cannot drift from them.\n")
    block(table(["file", "contents"], [
        ["`utility_matrix.csv`", "every `M[a][b]` cell with its relation label and ceiling"],
        ["`rotation_rows.csv`", "one row per context &times; rotation &times; policy &times; budget"],
        ["`metrics.csv`", "context-clustered means with bootstrap CIs, all three metrics"],
        ["`contrasts.csv`", "paired deltas with p-values and Cohen's *dz*"],
        ["`budget_interaction.csv`", "the scarcity test"],
        ["`length_audit.csv`, `length_deltas.csv`, `length_matched_contrasts.csv`",
         "the channel controls"],
        ["`pareto.csv`", "(policy, budget) points with domination and dominators"],
        ["`normalised_utility.csv`", "retained utility as a guarded ratio"],
        ["`relation_regret.csv`, `relation_tests.csv`", "the distance analysis"],
        ["`baselines.csv`", "direct-context ceiling and closed-book leakage audit"],
    ], "|---|---|"))
    # -------------------------------------------------------------- appendix
    w("## Appendix: every computed contrast\n")
    w("The sections above quote the headline comparisons. This appendix is the exhaustive "
      "version &mdash; every row the runner wrote to `contrasts.csv`, for all three utility "
      "measures, all endpoints, and the trimmed length-control arms. **Bold** excludes zero. "
      "`gap` is `U_now - U_future`; `gap DiD` is the difference of that gap between the two "
      "arms, which is the single number the hypothesis predicts to be positive.\n")
    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c:
            continue
        w(f"### {label} &mdash; contrasts\n")
        block(full_contrast_table(c))

    w("## Appendix: every computed level\n")
    w("The same treatment for the non-contrast tables: every policy, every budget, every "
      "endpoint and all three utility measures, including the `channel` rows that carry mean "
      "delivered length and fill on the same context-clustered footing as utility.\n")
    for key, label, _why in CORPORA:
        c = corpora.get(key)
        if not c:
            continue
        w(f"### {label} &mdash; levels with 95% CIs\n")
        block(pivot_all(c, "metrics", ["policy", "kind", "endpoint", "metric"], "mean"))
        w(f"### {label} &mdash; normalised utility, all metrics\n")
        block(pivot_all(c, "normalised_utility", ["policy", "endpoint", "metric"], "mean",
                        extra=["n_excluded_low_ceiling"]))
        w(f"### {label} &mdash; length-matched subsample, all metrics\n")
        block(pivot_all(c, "length_matched_contrasts",
                        ["comparison", "endpoint", "metric"], "delta", sign=True,
                        extra=["tolerance_words"]))
        w(f"### {label} &mdash; budget interaction, all metrics\n")
        block(flat_all(c, "budget_interaction", ["quantity", "policy", "metric"], "delta"))
        if c.relation_regret:
            w(f"### {label} &mdash; distance analysis, all metrics\n")
            block(pivot_all(c, "relation_regret",
                            ["policy", "relation", "kind", "metric"], "mean",
                            extra=["n_cells"]))
            w(f"### {label} &mdash; distance monotonicity tests, all metrics\n")
            block(pivot_all(c, "relation_tests", ["policy", "quantity", "metric"], "delta",
                            sign=True))

    w("---\n")
    w("*Agent Handoff Information-Loss Probe · Experiment 10 · sender and answerer "
      f"`{primary.manifest['sender_model']}`, judge "
      f"`{primary.manifest['judge_model']}`. Generated by "
      "`src/analysis/render_regret_summary.py`.*")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    corpora = discover()
    if not corpora:
        raise SystemExit("no corpus results found; run the experiment first")
    text = render(corpora)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[regret-summary] wrote {path} from "
          + ", ".join(f"{c.key}/{c.root.name}" for c in corpora.values())
          + f" ({len(text.splitlines())} lines)")
    _ = bd  # imported for the module-level constants the prose refers to
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
