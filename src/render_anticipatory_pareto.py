"""Experiment 12, deferred figure: the Pareto plane under preference and
estimation noise.

Two results were reported as tables only -- the lambda sweep ("one breakpoint,
then constant to lambda=8") and the regret-vs-mismatch slopes ("recall absorbs
prediction error, prediction alone does not"). Both are geometric statements
about the same plane, so this draws them.

Two conventions make the drawing exact rather than illustrative.

1. **The x axis is ``U_now`` net of normalised cost.** The scalarised objective
   is ``U_now + lambda*U_future - (beta*C_context + gamma*C_retrieval)/scale``.
   Plotting raw ``U_now`` would make each policy's iso-preference line carry its
   own cost offset, so no single tangent line could select a winner and the
   figure would silently disagree with the reported table. Writing
   ``x = U_now - cost/scale`` turns the objective into ``x + lambda*y``, whose
   level sets are straight lines of slope ``-1/lambda``. The tangent point is
   then genuinely the argmax, and the drawn winners reproduce
   ``lambda_selection.csv`` exactly.

2. **Dilution and displacement are drawn as separate trajectories.** They are
   different failure modes of the same estimate -- uncertain versus wrong -- and
   the design keeps them as separate knobs for that reason. Pooling them into
   one "noise" line by their shared TV would draw a slope between two arms that
   were never on the same sweep.

The objective, the cost scale and the winner selection are imported from
``analyse_anticipatory_extras`` rather than reimplemented, so this figure cannot
drift away from the table it illustrates.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_anticipatory_extras as extras  # noqa: E402
from llm import load_config  # noqa: E402

RESULTS = ROOT / "results" / "anticipatory_context"

LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Which arms form a noise sweep, and which knob each rung belongs to. ``tau0``
# is the clean estimate and is therefore the shared origin of both sweeps.
DILUTION = ("__tau0", "__tau0p5", "__tau1")
DISPLACEMENT = ("__tau0", "__sigma0p5", "__sigma1")

FAMILY_STYLE = {
    "preserve": ("#c44e52", "o", "guess & keep it (no lookup)"),
    "pointer": ("#4c72b0", "s", "guess & leave a link"),
    "uncertainty_aware": ("#55a868", "^", "looks things up when unsure"),
}
# Arms with no P-hat at all: no mismatch is defined for them, so they are
# reference points on the plane rather than rungs of a sweep.
REFERENCE_STYLE = {
    "blind": ("#8c8c8c", "v"),
    "oracle_exact": ("#000000", "*"),
    "retrieval_only": ("#4c72b0", "D"),
    "static_conditioned": ("#937860", "X"),
}

# Plain-language labels for the drawing only. The policy ids stay untouched in
# ``lambda_noise_pareto_points.csv`` and in the manifest, so the figure can be
# read without the vocabulary while every number remains traceable to the arm
# that produced it.
PLAIN_NAME = {
    "blind": "no guess at all",
    "oracle_exact": "told the real next question",
    "retrieval_only": "always looks it up",
    "static_conditioned": "writes for now only",
    "preserve": "guess & keep it",
    "pointer": "guess & leave a link",
    "uncertainty_aware": "looks it up when unsure",
}


def plain(policy: str) -> str:
    """Readable label for a policy id, with its noise rung dropped."""
    if policy in PLAIN_NAME:
        return PLAIN_NAME[policy]
    base = policy.split("__")[0]
    return PLAIN_NAME.get(base, policy)


def cell_points(rows: list[dict], cfg: dict, budget_words: float) -> tuple[dict, float]:
    """Per-policy mean coordinates on the objective plane for one budget.

    The mean of a linear objective is the objective of the means, so a policy's
    mean score at any lambda is recoverable from ``(x, y)`` alone -- which is
    what lets a straight tangent line pick the same winner the table reports.
    """
    beta = float(cfg["analysis"]["beta"])
    gamma = float(cfg["analysis"]["gamma_default"])
    cell = [r for r in rows if r["budget_words"] == budget_words]
    scale = max(beta * r["c_context"] + gamma * r["c_retrieval"] for r in cell)

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    family: dict[str, str] = {}
    mismatch: dict[str, float | None] = {}
    for row in cell:
        policy = row["policy"]
        family[policy] = row["family"]
        mismatch[policy] = row["tv_mismatch"]
        cost = (beta * row["c_context"] + gamma * row["c_retrieval"]) / scale
        acc[policy]["u_now"].append(row["now_judge_correct"])
        acc[policy]["u_future"].append(row["future_judge_correct"])
        acc[policy]["cost"].append(cost)

    points = {}
    for policy, series in acc.items():
        u_now = float(np.mean(series["u_now"]))
        cost = float(np.mean(series["cost"]))
        points[policy] = {
            "policy": policy,
            "family": family[policy],
            "budget_words": budget_words,
            "u_now": u_now,
            "u_future": float(np.mean(series["u_future"])),
            "cost_norm": cost,
            "x": u_now - cost,
            "tv_mismatch": mismatch[policy],
        }
    return points, scale


def pareto_front(points: dict) -> list[dict]:
    """Non-dominated in (x, u_future); both are maximised."""
    front = []
    for p in points.values():
        dominated = any(
            q["x"] >= p["x"] and q["u_future"] >= p["u_future"]
            and (q["x"] > p["x"] or q["u_future"] > p["u_future"])
            for q in points.values()
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["x"])


def series_for(points: dict, family: str, suffixes: tuple[str, ...]) -> list[dict]:
    out = []
    for suffix in suffixes:
        name = f"{family}{suffix}"
        if name in points:
            out.append(points[name])
    return out if len(out) > 1 else []


# Four line roles on the top panel, kept separable on three axes at once --
# colour, dash pattern and width -- so no two are told apart by dash alone.
# The two sweeps share a family colour and differ by dash; the front and the
# cone edges use colours no family uses and are the thinnest and the only
# dash-dot lines respectively.
SWEEP_WIDTH = 2.4
VAGUER_STYLE = "-"                  # solid: the estimate goes vague
WRONG_STYLE = (0, (7, 3))           # long dash: the estimate goes wrong
FRONT_COLOUR = "#111111"
FRONT_STYLE = (0, (1, 2.2))         # tight dot, black, widest
FRONT_WIDTH = 3.0
CONE_COLOUR = "#7b3294"             # purple, used by no policy family
CONE_STYLE = (0, (6, 2, 1, 2))      # dash-dot, used by no other line
CONE_WIDTH = 1.7

# Nudges that stop reference labels landing on top of each other. The plane is
# crowded near the bottom-right corner, where three near-identical arms sit.
LABEL_OFFSET = {
    "blind": (0, -16, "center"),
    "oracle_exact": (-12, 13, "right"),
    "retrieval_only": (9, -3, "left"),
    "static_conditioned": (-6, -14, "right"),
}


FAN_RADIUS_PT = 9.0

# How a sweep rung is named inside a cluster callout. Naming the rung is what
# turns "four markers on one spot" from a drawing defect into the result it
# actually is: the estimate degraded and the policy did not move.
RUNG_NAME = {
    "__tau0": "perfect guess",
    "__tau0p5": "half-vague",
    "__tau1": "vaguest",
    "__sigma0p5": "half-wrong",
    "__sigma1": "most wrong",
}

# Anchors for callout text, in axes fractions, chosen to be clear of data in
# both panels. Text sits here and a leader line runs to the marker, so no label
# has to fit in the gap between two touching points.
CALLOUT_ANCHORS = [(0.22, 0.66), (0.04, 0.20)]


def offset_points(ax, dx: float, dy: float):
    """A transData variant shifted by (dx, dy) in typographic points."""
    if dx == 0.0 and dy == 0.0:
        return ax.transData
    fig = ax.get_figure()
    return ax.transData + ScaledTranslation(dx / 72.0, dy / 72.0,
                                            fig.dpi_scale_trans)


def find_clusters(points: dict, ax) -> list[list[str]]:
    """Group policies whose plotted positions are indistinguishable.

    Tolerance is a fraction of the visible range rather than an absolute
    number, because the two budgets have different spreads and a fixed epsilon
    would over-merge one panel and under-merge the other.
    """
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    tol_x, tol_y = 0.025 * (xhi - xlo), 0.025 * (yhi - ylo)
    remaining = sorted(points)
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        for other in list(remaining):
            if (abs(points[other]["x"] - points[seed]["x"]) < tol_x
                    and abs(points[other]["u_future"] - points[seed]["u_future"]) < tol_y):
                group.append(other)
                remaining.remove(other)
        clusters.append(group)
    return clusters


def fan_offsets(clusters: list[list[str]]) -> dict[str, tuple[float, float]]:
    """Push coincident markers onto a small ring, in display points."""
    out: dict[str, tuple[float, float]] = {}
    for group in clusters:
        if len(group) == 1:
            out[group[0]] = (0.0, 0.0)
            continue
        for i, policy in enumerate(sorted(group)):
            angle = 2 * math.pi * i / len(group) + math.pi / 2
            out[policy] = (FAN_RADIUS_PT * math.cos(angle),
                           FAN_RADIUS_PT * math.sin(angle))
    return out


def member_name(policy: str) -> str:
    base, _, suffix = policy.partition("__")
    if not suffix:
        return plain(policy)
    return f"{plain(base)} ({RUNG_NAME.get('__' + suffix, suffix)})"


def draw_plane(ax, points: dict, winners: dict[float, str], cmap, norm) -> None:
    named = set(winners.values())
    front = pareto_front(points)
    ax.plot([p["x"] for p in front], [p["u_future"] for p in front],
            color=FRONT_COLOUR, lw=FRONT_WIDTH, ls=FRONT_STYLE, alpha=0.9,
            zorder=1, solid_capstyle="round")

    # Noise trajectories. These are the only lines that touch data points, they
    # are the widest, and each ends in an arrow head pointing at the worst rung
    # of its sweep -- three cues, so the reader never has to resolve them by
    # dash pattern alone.
    for family, (colour, marker, _label) in FAMILY_STYLE.items():
        for suffixes, style in ((DILUTION, VAGUER_STYLE), (DISPLACEMENT, WRONG_STYLE)):
            series = series_for(points, family, suffixes)
            if not series:
                continue
            ax.plot([p["x"] for p in series], [p["u_future"] for p in series],
                    color=colour, lw=SWEEP_WIDTH, ls=style, alpha=0.95, zorder=2,
                    solid_capstyle="round")
            tail, head = series[-2], series[-1]
            if (tail["x"], tail["u_future"]) != (head["x"], head["u_future"]):
                ax.annotate("", xy=(head["x"], head["u_future"]),
                            xytext=(tail["x"], tail["u_future"]),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.1,
                                            mutation_scale=17, shrinkA=0, shrinkB=0),
                            zorder=3)
            # The line is named through the legend, not on the plot. In-plot
            # labels were tried and removed: the `pointer` sweep is only a few
            # points wide, so its name ran off the left edge, and the other two
            # landed on the callout boxes. The legend instead draws each
            # family's marker *on* a segment of its own line, which answers
            # "which policy is this coloured line" without costing panel space.
            # No end-of-sweep text. The worst rung of a sweep lands on top of a
            # reference arm often enough that the labels collided in both
            # panels -- and that coincidence is itself a result, not a layout
            # problem to write over.

    # Markers are fanned around coincident locations. Four arms share one spot
    # in every panel, so no label nudging can separate them -- the overlap is
    # the data, not the layout. Each cluster gets a faint ring at its true
    # position and its members are pushed onto that ring in *display* points,
    # so the fan never implies a coordinate difference that does not exist.
    clusters = find_clusters(points, ax)
    offsets = fan_offsets(clusters)
    for members in clusters:
        if len(members) < 2:
            continue
        cx = float(np.mean([points[m]["x"] for m in members]))
        cy = float(np.mean([points[m]["u_future"] for m in members]))
        ax.scatter(cx, cy, s=(FAN_RADIUS_PT * 2.6) ** 2, facecolors="none",
                   edgecolors="#9a9a9a", linewidths=1.0, zorder=3.5)

    for policy, p in points.items():
        family = p["family"]
        trans = offset_points(ax, *offsets[policy])
        if family in FAMILY_STYLE:
            _colour, marker, _ = FAMILY_STYLE[family]
            ax.scatter(p["x"], p["u_future"], c=[p["tv_mismatch"]], cmap=cmap,
                       norm=norm, marker=marker, s=90, zorder=4,
                       edgecolors="white", linewidths=1.3, transform=trans)
        else:
            colour, marker = REFERENCE_STYLE.get(family, ("#8c8c8c", "."))
            ax.scatter(p["x"], p["u_future"], color=colour, marker=marker, s=115,
                       zorder=4, edgecolors="white", linewidths=1.3, transform=trans)

    # One preference cone per distinct winner, not one line per lambda: seven
    # lines for two winners implies six decisions that were never made. The
    # cone's edges are the extreme lambdas that policy wins over, so its width
    # *is* the reported claim that the choice stops moving above the breakpoint.
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    won_at = defaultdict(list)
    for lam in sorted(LAMBDA_GRID):
        won_at[winners[lam]].append(lam)
    callouts: list = []

    # Cone edges are drawn in one neutral colour that no policy family uses, in
    # a dash-dot pattern used nowhere else, and as short segments through the
    # winning point rather than lines spanning the panel. Family-coloured
    # full-width lines were indistinguishable from the sweeps they crossed.
    half_x = 0.20 * (xhi - xlo)
    half_y = 0.20 * (yhi - ylo)
    for winner, lams in won_at.items():
        w = points[winner]
        colour = FAMILY_STYLE.get(w["family"],
                                  REFERENCE_STYLE.get(w["family"], ("#333", "o")))[0]
        for lam in (min(lams), max(lams)):
            if lam == 0.0:
                ax.plot([w["x"], w["x"]],
                        [w["u_future"] - half_y, w["u_future"] + half_y],
                        color=CONE_COLOUR, lw=CONE_WIDTH, ls=CONE_STYLE,
                        alpha=0.95, zorder=3)
            else:
                xs = np.array([w["x"] - half_x, w["x"] + half_x])
                ax.plot(xs, w["u_future"] + (w["x"] - xs) / lam, color=CONE_COLOUR,
                        lw=CONE_WIDTH, ls=CONE_STYLE, alpha=0.95, zorder=3)
        span = (f"a future question is worth\nless than {max(lams):g}× the current one"
                if min(lams) == 0.0
                else f"a future question is worth\nat least {min(lams):g}× the current one")
        # Keep the line breaks inside `span`. Flattening it to one line made a
        # box wider than the panel, which pushed the text outside the axes.
        callouts.append(((w["x"], w["u_future"]),
                         f"best choice when\n{span}\n→ {plain(winner)}", colour))

    # Only the two winners get boxed text. Naming every member of every cluster
    # inside the axes produced five multi-line boxes per panel that collided
    # with each other -- worse than the marker overlap they were fixing. The
    # ring plus a count says "these share one spot"; which arms they are is
    # readable from the fanned markers themselves and stated in the report.
    for anchor, (target, text, colour) in zip(CALLOUT_ANCHORS, callouts):
        ax.annotate(text, xy=target, xycoords="data",
                    xytext=anchor, textcoords="axes fraction",
                    fontsize=7.5, color=colour, zorder=7, fontweight="bold",
                    ha="left", va="center",
                    # Leader lines are grey and hairline, never the callout's
                    # own colour: a long brown or green curve crossing the
                    # panel reads as another data series, which is the exact
                    # confusion this figure is trying to remove.
                    arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                    alpha=0.65, shrinkA=2, shrinkB=11,
                                    connectionstyle="arc3,rad=0.12"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour,
                              lw=0.9, alpha=0.93))

    for group in clusters:
        if len(group) < 2:
            continue
        cx = float(np.mean([points[m]["x"] for m in group]))
        cy = float(np.mean([points[m]["u_future"] for m in group]))
        ax.annotate(f"×{len(group)}", (cx, cy), fontsize=7.5, color="#666666",
                    textcoords="offset points", xytext=(FAN_RADIUS_PT + 8, -3),
                    ha="left", va="center", zorder=7)

    # Every reference arm keeps its name, including the ones inside a ring.
    # Skipping clustered arms left the blue diamond and the grey triangle
    # unlabelled in one panel each, which is worse than a crowded label: the
    # count tells you how many arms are there but not which.
    for policy in sorted(points):
        p = points[policy]
        if p["family"] in FAMILY_STYLE or policy in named:
            continue
        fdx, fdy = offsets[policy]
        if fdx or fdy:
            # Push the label radially outward from the ring it sits on, so it
            # leaves the cluster rather than landing on a sibling marker.
            scale = (FAN_RADIUS_PT + 11.0) / math.hypot(fdx, fdy)
            dx, dy = fdx * scale, fdy * scale
            # A ring sitting against the top of the panel has no room above it;
            # pushing the label outward there put it over the panel title.
            height = (p["u_future"] - ylo) / (yhi - ylo)
            if height > 0.88:
                dy = -abs(dy) - 4
            elif height < 0.12:
                dy = abs(dy) + 4
            ha = "left" if dx >= 0 else "right"
        else:
            dx, dy, ha = LABEL_OFFSET.get(policy, (6, -9, "left"))
        ax.annotate(plain(policy), (p["x"], p["u_future"]), fontsize=7.5,
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    va="center", color="#333", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                              alpha=0.8))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)


def draw_envelope(ax, points: dict, winners: dict[float, str]) -> None:
    """Mean objective against lambda, with the upper envelope highlighted.

    The breakpoint is a property of which line is on top, not of any single
    line, so the envelope is what has to be drawn.
    """
    lams = np.linspace(0.0, 8.0, 200)
    winning = {winners[lam] for lam in LAMBDA_GRID}
    for policy, p in points.items():
        values = p["x"] + lams * p["u_future"]
        if policy in winning:
            continue
        ax.plot(lams, values, color="#c8c8c8", lw=0.9, zorder=1)
    for policy in sorted(winning):
        p = points[policy]
        colour = FAMILY_STYLE.get(p["family"], REFERENCE_STYLE.get(p["family"],
                                                                  ("#333", "o")))[0]
        ax.plot(lams, p["x"] + lams * p["u_future"], color=colour, lw=2.2,
                zorder=3)
        # Labelled on the line rather than in a corner key: with only two
        # coloured lines per panel and a large vertical gap between them at the
        # right edge, the name fits exactly where the line ends.
        end = 7.85
        ax.annotate(plain(policy), (end, p["x"] + end * p["u_future"]),
                    fontsize=8.5, color=colour, fontweight="bold",
                    ha="right", va="bottom", zorder=6,
                    textcoords="offset points", xytext=(0, 5),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour,
                              lw=0.8, alpha=0.9))

    # The breakpoint band: between the last lambda won by the first policy and
    # the first lambda won by the second.
    order = sorted(LAMBDA_GRID)
    for a, b in zip(order, order[1:]):
        if winners[a] != winners[b]:
            ax.axvspan(a, b, color="#7b3294", alpha=0.15, zorder=0)
            ax.annotate("the best choice\nflips here", ((a + b) / 2, 0.62),
                        xycoords=("data", "axes fraction"), fontsize=8,
                        ha="left", va="center", color="#7b3294",
                        textcoords="offset points", xytext=(8, 0))
    ax.set_xlabel("how much a future question is worth,\n"
                  "compared with the one being answered now")
    ax.grid(alpha=0.25)


def main() -> int:
    cfg = load_config(ROOT / "anticipatory_context_config.yaml")
    rows = extras.load_rows()
    beta = float(cfg["analysis"]["beta"])
    gamma = float(cfg["analysis"]["gamma_default"])

    budgets = sorted({r["budget_words"] for r in rows})
    cells = {}
    for budget_words in budgets:
        points, scale = cell_points(rows, cfg, budget_words)
        cell_rows = [r for r in rows if r["budget_words"] == budget_words]
        winners = {lam: extras.winner_at(cell_rows, lam, beta, gamma, scale)
                   for lam in LAMBDA_GRID}
        cells[budget_words] = (points, winners)

    mismatches = [p["tv_mismatch"] for points, _ in cells.values()
                  for p in points.values() if p["tv_mismatch"] is not None]
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=min(mismatches), vmax=max(mismatches))

    fig, axes = plt.subplots(2, len(budgets), figsize=(7.2 * len(budgets), 12.8),
                             squeeze=False)
    for col, budget_words in enumerate(budgets):
        points, winners = cells[budget_words]
        top = axes[0][col]
        draw_plane(top, points, winners, cmap, norm)
        top.set_title(f"{int(budget_words)}-word budget", fontsize=12)
        top.set_xlabel("how well it answers the question it was written for\n"
                       "(after subtracting what the message and lookups cost)")
        top.grid(alpha=0.25)
        if col == 0:
            top.set_ylabel("how well it answers questions\nit was never told about")

        bottom = axes[1][col]
        draw_envelope(bottom, points, winners)
        if col == 0:
            bottom.set_ylabel("overall score\n(now + future, minus cost)")

    # Two legends, one per row. A single shared key forced the reader to work
    # out which entries applied to points and which to lines; the rows are
    # different objects (a policy is a point above and a line below) and each
    # needs its own vocabulary.
    handles = [Line2D([], [], color=c, marker=m, ls=VAGUER_STYLE, lw=SWEEP_WIDTH,
                      markersize=9, markeredgecolor="white", markeredgewidth=1.2,
                      label=lab)
               for _f, (c, m, lab) in FAMILY_STYLE.items()]
    # Each line entry names its own appearance. If two styles still look alike
    # at print size, the label itself resolves them without a second lookup.
    handles += [Line2D([], [], color="#555", ls=VAGUER_STYLE, lw=SWEEP_WIDTH,
                       label="coloured solid, with arrow:\nthe guess gets vaguer"),
                Line2D([], [], color="#555", ls=WRONG_STYLE, lw=SWEEP_WIDTH,
                       label="coloured dashed, with arrow:\nthe guess gets wrong"),
                Line2D([], [], color=FRONT_COLOUR, ls=FRONT_STYLE, lw=FRONT_WIDTH,
                       label="black dotted:\nbest trade-offs available"),
                Line2D([], [], color=CONE_COLOUR, ls=CONE_STYLE, lw=CONE_WIDTH,
                       label="purple dash-dot:\neverything on it scores the same")]
    # Figure-level, between the rows: an in-axes legend covers the 80-word
    # panel's own points, which are the ones the noise sweep is about.
    # handlelength has to be long enough to show a whole dash period, otherwise
    # every dashed style renders as one solid stub in the key and the legend
    # actively misleads about which line is which.
    top_legend = fig.legend(
        handles=handles, frameon=False, fontsize=9, ncol=4,
        handlelength=4.4, handletextpad=0.9, columnspacing=2.4,
        loc="upper center", bbox_to_anchor=(0.5, 0.545),
        title="TOP ROW — each policy is a point, and its sweep line carries the same colour; arrows follow a worsening guess")
    top_legend.get_title().set_fontsize(10)
    top_legend.get_title().set_fontweight("bold")

    bottom_handles = [
        Line2D([], [], color="#555", lw=2.2,
               label="thick coloured: best choice somewhere\n"
                     "in the range (named in each panel)"),
        Line2D([], [], color="#c8c8c8", lw=1.4,
               label="thin grey: one of the other policies,\n"
                     "never the best choice anywhere"),
        Patch(facecolor="#7b3294", alpha=0.15,
              label="shaded band: where the topmost\nline changes"),
    ]
    bottom_legend = fig.legend(
        handles=bottom_handles, frameon=False, fontsize=9, ncol=3,
        handlelength=3.0, handletextpad=0.9, columnspacing=2.4,
        loc="upper center", bbox_to_anchor=(0.5, 0.058),
        title="BOTTOM ROW — each policy is a line: its height at 0 is how well "
              "it answers now, its slope is how well it answers later")
    bottom_legend.get_title().set_fontsize(10)
    bottom_legend.get_title().set_fontweight("bold")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes[0].tolist(), fraction=0.03, pad=0.04)
    cbar.set_label("how wrong the guess about future questions was\n(0 = exactly right)")

    fig.subplots_adjust(hspace=0.62, top=0.94, bottom=0.13)
    fig.suptitle("Guessing what will be asked next degrades badly; being able to "
                 "look it up later does not", fontsize=13)
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "fig6_lambda_noise_pareto.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    audit = []
    for budget_words in budgets:
        points, winners = cells[budget_words]
        won_at = defaultdict(list)
        for lam, policy in winners.items():
            won_at[policy].append(lam)
        front = {p["policy"] for p in pareto_front(points)}
        for policy, p in sorted(points.items()):
            audit.append({
                "budget_words": int(budget_words),
                "policy": policy,
                "family": p["family"],
                "tv_mismatch": "" if p["tv_mismatch"] is None else p["tv_mismatch"],
                "u_now": round(p["u_now"], 6),
                "u_future": round(p["u_future"], 6),
                "cost_norm": round(p["cost_norm"], 6),
                "x_net_of_cost": round(p["x"], 6),
                "on_pareto_front": policy in front,
                "wins_at_lambda": ";".join(str(l) for l in sorted(won_at[policy])),
            })
    with open(RESULTS / "lambda_noise_pareto_points.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    manifest = {
        "figure": "fig6_lambda_noise_pareto",
        "lambda_grid": LAMBDA_GRID,
        "budgets": [int(b) for b in budgets],
        "winners": {str(int(b)): cells[b][1] for b in budgets},
        "beta": beta,
        "gamma": gamma,
        "n_points_plotted": len(audit),
    }
    with open(RESULTS / "pareto_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"wrote {path}")
    for budget_words in budgets:
        _points, winners = cells[budget_words]
        summary = ", ".join(f"lambda={lam:g} -> {winners[lam]}" for lam in LAMBDA_GRID)
        print(f"  {int(budget_words):>3}w  {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
