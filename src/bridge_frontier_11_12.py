"""Put Experiments 11 and 12 on one frontier, validly.

The report declines to pool the two experiments as they stand, and that is
correct: Experiment 11 weights its off-diagonal by *relation* to the
conditioning query, Experiment 12 by *aspect* under a declared true `P`. Two
different objectives, so a shared frontier would rank policies that were never
competing.

But the obstacle is the weighting, not the data. Both experiments answer every
question of the same 16 relation dossiers, under the same models, on the same
64 rotations, with byte-identical rotation ids. Rescoring Experiment 12's
sealed answers under Experiment 11's relation distributions therefore makes the
comparison legitimate without generating anything new: same corpus, same
rotations, same weighting, same cost definition.

That matters because it is the strongest available form of the retrieval claim.
"Retrieval beats the static policies I happened to build" is weak. "Retrieval
beats the tuned preference-allocation frontier from the previous experiment,
scored on that experiment's own objective" is the claim worth making - or
worth failing to make, if it does not hold.
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

import render_anticipatory_report as rr  # noqa: E402
from llm import load_config  # noqa: E402

E11 = ROOT / "results" / "bounded_communication_frontier" / "primary" / "relation_dossiers" / "n16"
E12_RUN = ROOT / "runs" / "anticipatory_context"
OUT = ROOT / "results" / "anticipatory_context"
RELATIONS = ("paraphrase", "same_entity", "same_topic", "orthogonal")


def weighted_future(by_relation: dict[str, list[float]], weights: dict[str, float]):
    """Average within relation, then weight. Returns None if a needed one is absent.

    The order matters and matches both source experiments: weighting raw cells
    would let the twelve orthogonal cells outvote the single paraphrase cell and
    silently turn a balanced target into an orthogonal one.
    """
    total = 0.0
    for relation, weight in weights.items():
        if weight <= 0:
            continue
        values = by_relation.get(relation)
        if not values:
            return None
        total += weight * float(np.mean(values))
    return total


# ---------------------------------------------------------------------------


def experiment11_cells() -> dict:
    """(policy, budget, rotation) -> {'now': [...], 'rel': {relation: [...]}, 'cost': w}."""
    cells: dict = defaultdict(lambda: {"now": [], "rel": defaultdict(list), "cost": []})
    with open(E11 / "utility_matrix.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["policy"], int(row["budget_words"]), row["rotation_id"])
            value = float(row["u_judge_correct"])
            entry = cells[key]
            entry["cost"].append(float(row["delivered_words"]))
            if row["is_diagonal"].lower() == "true":
                entry["now"].append(value)
            else:
                entry["rel"][row["relation"]].append(value)
    return cells


def experiment12_cells() -> dict:
    """Same shape, with relation derived from the design rather than stored."""
    cond_aspect, meta = {}, {}
    with open(E12_RUN / "messages.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            msg = json.loads(line)
            cond_aspect[msg["message_key"]] = msg["conditioning_aspect"]
            meta[msg["message_key"]] = (msg["policy"], int(msg["budget_words"]),
                                        msg["rotation_id"], float(msg["delivered_words"]))

    cells: dict = defaultdict(lambda: {"now": [], "rel": defaultdict(list),
                                       "cost": [], "retrieval": []})
    with open(E12_RUN / "answers.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ans = json.loads(line)
            mk = ans["message_key"]
            if mk not in meta:
                continue
            policy, budget_words, rotation_id, delivered = meta[mk]
            entry = cells[(policy, budget_words, rotation_id)]
            entry["cost"].append(delivered)
            entry["retrieval"].append(float(ans.get("retrieved_words", 0) or 0))
            value = float(ans.get("judge_correct", 0) or 0)
            if ans["is_diagonal"]:
                entry["now"].append(value)
            else:
                # The conditioning question is always an anchor, so a hidden
                # question in the same aspect is labelled by its own role and
                # anything in another aspect is orthogonal - exactly
                # regret_data.relation_label, which cannot be imported here
                # because Experiment 12 stores aspects rather than contexts.
                relation = (ans["eval_role"] if ans["eval_aspect"] == cond_aspect[mk]
                            else "orthogonal")
                entry["rel"][relation].append(value)
    return cells


def points_for(cells: dict, weights: dict[str, float], budget_words: int,
               gamma: float, source: str) -> list[dict]:
    grouped = defaultdict(list)
    for (policy, budget, rotation), entry in cells.items():
        if budget != budget_words or not entry["now"]:
            continue
        future = weighted_future(entry["rel"], weights)
        if future is None:
            continue
        retrieval = float(np.mean(entry.get("retrieval", [0.0]) or [0.0]))
        grouped[policy].append((
            rotation,
            float(np.mean(entry["now"])),
            future,
            float(np.mean(entry["cost"])),
            retrieval,
        ))
    out = []
    for policy, rows in sorted(grouped.items()):
        out.append({
            "source": source, "policy": policy, "budget_words": budget_words,
            "n_rotations": len(rows),
            "u_now": float(np.mean([r[1] for r in rows])),
            "u_future": float(np.mean([r[2] for r in rows])),
            "c_context": float(np.mean([r[3] for r in rows])),
            "c_retrieval": float(np.mean([r[4] for r in rows])),
            "cost": float(np.mean([r[3] + gamma * r[4] for r in rows])),
            # Per-rotation values are retained so the bootstrap can recompute
            # each policy's mean from a resampled set of contexts. Keeping only
            # the pooled mean makes resampling a no-op and produces a
            # zero-width interval that looks like certainty.
            "per_rotation": [(r[0], r[2], r[3] + gamma * r[4]) for r in rows],
        })
    return out


def delta_hv(points: list[dict], seed: int, resamples: int = 2000) -> dict:
    """Incremental hypervolume from adding Experiment 12's retrieval arms."""
    static = [p for p in points if p["source"] == "exp11"]
    recall = [p for p in points if p["source"] == "exp12" and p["c_retrieval"] > 0]
    if not static or not recall:
        return {}
    cost_max = max(p["cost"] for p in points)

    def hv(group: list[dict]) -> float:
        return rr.hypervolume([(p["u_future"], p["cost"]) for p in group], cost_max)

    base = hv(static + recall) - hv(static)
    rng = np.random.default_rng(seed)
    by_context: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for index, point in enumerate(points):
        for rotation, future, cost in point["per_rotation"]:
            by_context[rotation.split("|")[0]][index].append((future, cost))
    contexts = sorted(by_context)

    def resampled(sample: list[str], indices: list[int]) -> list[tuple[float, float]]:
        out = []
        for index in indices:
            vals = [v for ctx in sample for v in by_context[ctx].get(index, ())]
            if vals:
                out.append((float(np.mean([f for f, _ in vals])),
                            float(np.mean([c for _, c in vals]))))
        return out

    static_idx = [i for i, p in enumerate(points) if p["source"] == "exp11"]
    recall_idx = [i for i, p in enumerate(points)
                  if p["source"] == "exp12" and p["c_retrieval"] > 0]
    deltas = np.empty(resamples)
    for draw in range(resamples):
        sample = [contexts[i] for i in rng.integers(0, len(contexts), len(contexts))]
        both = resampled(sample, static_idx + recall_idx)
        only = resampled(sample, static_idx)
        deltas[draw] = (rr.hypervolume(both, cost_max)
                        - rr.hypervolume(only, cost_max))
    return {"delta_hv": base, "lo": float(np.quantile(deltas, 0.025)),
            "hi": float(np.quantile(deltas, 0.975)),
            "hv_exp11_only": hv(static), "hv_combined": hv(static + recall)}


def main() -> int:
    cfg11 = load_config(ROOT / "bounded_communication_frontier_config.yaml")
    cfg12 = load_config(ROOT / "anticipatory_context_config.yaml")
    dists = cfg11["frontier"]["future_distributions"]
    gamma = float(cfg12["analysis"]["gamma_default"])
    seed = int(cfg12["analysis"]["bootstrap_seed"])

    c11, c12 = experiment11_cells(), experiment12_cells()
    rows, shifts = [], []
    for name in ("empirical_cells", "relation_balanced", "far"):
        weights = {r: float(dists[name].get(r, 0.0)) for r in RELATIONS}
        for budget_words in (40, 80):
            pts = (points_for(c11, weights, budget_words, gamma, "exp11")
                   + points_for(c12, weights, budget_words, gamma, "exp12"))
            if not pts:
                continue
            for p in pts:
                rows.append({k: v for k, v in p.items() if k != "per_rotation"}
                            | {"future_distribution": name})
            shift = delta_hv(pts, seed)
            if shift:
                shifts.append({"future_distribution": name,
                               "budget_words": budget_words, "gamma": gamma, **shift})

    for path, data in ((OUT / "bridge_points.csv", rows),
                       (OUT / "bridge_frontier_shift.csv", shifts)):
        keys: list[str] = []
        for row in data:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    # One figure: the shared frontier under the empirical relation distribution.
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), squeeze=False)
    for ax, budget_words in zip(axes[0], (40, 80)):
        cell = [r for r in rows if r["future_distribution"] == "empirical_cells"
                and r["budget_words"] == budget_words]
        for source, colour, label in (("exp11", "#c44e52", "Exp 11 preference allocation"),
                                      ("exp12", "#4c72b0", "Exp 12 anticipatory/retrieval")):
            grp = [r for r in cell if r["source"] == source]
            ax.scatter([r["cost"] for r in grp], [r["u_future"] for r in grp],
                       color=colour, s=52, label=label, alpha=0.85, zorder=3)
            front = rr.pareto_front([(r["u_future"], r["cost"]) for r in grp])
            if front:
                ax.plot([c for _, c in front], [u for u, _ in front], color=colour,
                        ls="--", lw=1.6, alpha=0.8)
        ax.set_xlabel(f"cost   context + {gamma:g}·retrieval  (words)")
        ax.set_ylabel("$U_{future}$ under Exp 11 relation weighting")
        ax.set_title(f"{budget_words}-word budget")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Experiments 11 and 12 on one frontier, scored on Experiment 11's objective",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig5_bridge_frontier.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"points": len(rows), "shift_cells": len(shifts),
                      "positive": sum(1 for s in shifts if s["lo"] > 0)}, indent=2))
    for s in shifts:
        print(f"  {s['future_distribution']:<18} b={s['budget_words']:<3} "
              f"dHV={s['delta_hv']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}]  "
              f"(exp11 {s['hv_exp11_only']:.3f} -> {s['hv_combined']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
