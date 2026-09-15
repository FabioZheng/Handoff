"""Offline regression checks for the bounded-communication Pareto pipeline."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pareto_frontier as pf  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))
    print(f"PASS  {label}" + (f" -- {detail}" if detail else ""))


def point(policy, budget, now, future, cost, *, dataset="d", model="m", dist="uniform"):
    return {
        "dataset": dataset,
        "model": model,
        "future_distribution": dist,
        "metric": "judge_correct",
        "policy": policy,
        "budget_words": budget,
        "u_now": now,
        "u_future": future,
        "cost": cost,
    }


REGIMES = ("dataset", "model", "future_distribution", "metric")
CONFIGS = ("policy", "budget_words")


print("=== validation and equivalence ===")
raw = [point("a", 20, 0.5, 0.5, 19), point("b", 20, 0.5, 0.5, 19)]
san = pf.sanitize_points(raw, regime_keys=REGIMES, config_keys=CONFIGS)
check("distinct configurations at duplicate coordinates are retained", len(san.points) == 2)
check("duplicate coordinates share an equivalence group",
      san.points[0]["equivalence_id"] == san.points[1]["equivalence_id"]
      and san.points[0]["equivalence_size"] == 2)

try:
    pf.sanitize_points([raw[0], dict(raw[0])], regime_keys=REGIMES, config_keys=CONFIGS)
except pf.ParetoValidationError:
    pass
else:
    raise AssertionError("FAIL  duplicate configuration rows are rejected")
print("PASS  duplicate configuration rows are rejected")

invalid = [point("ok", 20, 0.5, 0.5, 19), point("nan", 20, math.nan, 0.5, 19),
           point("inf", 20, 0.5, 0.5, math.inf)]
dropped = pf.sanitize_points(invalid, regime_keys=REGIMES, config_keys=CONFIGS,
                             on_invalid="drop")
check("NaN and infinity are excluded rather than marked nondominated",
      len(dropped.points) == 1 and len(dropped.excluded) == 2)

try:
    pf.sanitize_points([{"dataset": "d"}], regime_keys=REGIMES, config_keys=CONFIGS)
except pf.ParetoValidationError:
    pass
else:
    raise AssertionError("FAIL  missing values fail strict validation")
print("PASS  missing values fail strict validation")


print("\n=== exact nondominance and regime isolation ===")
raw = [
    point("worse", 20, 0.4, 0.4, 20),
    point("better", 20, 0.6, 0.6, 20),
    point("tradeoff", 20, 0.8, 0.2, 20),
    point("same_better", 20, 0.6, 0.6, 20, dataset="other"),
    point("other_peak", 20, 0.9, 0.9, 20, dataset="other"),
]
san = pf.sanitize_points(raw, regime_keys=REGIMES, config_keys=CONFIGS)
marked = pf.mark_nondominated(san.points)
by_key = {(row["dataset"], row["policy"]): row for row in marked}
check("strictly worse observation is dominated", by_key[("d", "worse")]["dominated"])
check("an incomparable trade-off remains nondominated", by_key[("d", "tradeoff")]["nondominated"])
check("dominance never crosses regimes",
      by_key[("d", "better")]["nondominated"]
      and by_key[("other", "same_better")]["dominated"])

dupes = pf.mark_nondominated(pf.sanitize_points(
    [point("a", 20, .5, .5, 20), point("b", 20, .5, .5, 20)],
    regime_keys=REGIMES, config_keys=CONFIGS).points)
check("exact objective ties do not dominate one another", all(row["nondominated"] for row in dupes))

tied = pf.mark_nondominated(pf.sanitize_points(
    [point("a", 20, .5, .5, 20), point("b", 20, .5, .6, 20)],
    regime_keys=REGIMES, config_keys=CONFIGS).points)
check("a tie in one objective plus a strict gain dominates",
      next(row for row in tied if row["policy"] == "a")["dominated"])

tiny = pf.sanitize_points(
    [point("a", 20, .5, .5, 20), point("b", 20, .50001, .5, 20)],
    regime_keys=REGIMES, config_keys=CONFIGS).points
check("epsilon suppresses a sub-tolerance strict difference",
      all(row["nondominated"] for row in pf.mark_nondominated(tiny, {"u_now": 0.001})))

forward = pf.mark_nondominated(san.points)
reverse = pf.mark_nondominated(list(reversed(san.points)))
signature = lambda rows: sorted((row["regime_id"], row["config_id"], row["dominated"])
                                for row in rows)
check("nondominance is permutation invariant", signature(forward) == signature(reverse))


print("\n=== constrained feasible frontiers ===")
raw = [
    point("low", 20, .7, .7, 18),
    point("high_but_worse", 40, .6, .6, 37),
    point("high_tradeoff", 40, .9, .4, 37),
    point("nonmonotonic", 80, .5, .5, 70),
]
points = pf.sanitize_points(raw, regime_keys=REGIMES, config_keys=CONFIGS).points
fronts = pf.constrained_frontiers(points, [20, 40, 80])
at40 = {row["policy"]: row for row in fronts if row["constraint_budget"] == 40}
check("C <= B includes a lower generation rung", at40["low"]["feasible"])
check("a feasible lower rung can dominate a higher rung",
      at40["high_but_worse"]["dominated_at_budget"])
check("an over-budget point is explicitly marked infeasible",
      not at40["nonmonotonic"]["feasible"]
      and at40["nonmonotonic"]["nondominated_at_budget"] is None)
at80 = {row["policy"]: row for row in fronts if row["constraint_budget"] == 80}
check("nonmonotonic budget effects are retained as observations",
      at80["nonmonotonic"]["feasible"] and at80["nonmonotonic"]["dominated_at_budget"])

strict = [dict(points[0], assigned_cap=40.0), dict(points[1], assigned_cap=20.0)]
strict_front = pf.constrained_frontiers(strict, [20], feasibility_key="assigned_cap")
strict_by_policy = {row["policy"]: row for row in strict_front}
check("hard feasibility can be separated from mean realised cost",
      not strict_by_policy["low"]["feasible"]
      and strict_by_policy["high_but_worse"]["feasible"])


print("\n=== exact hypervolume ===")
coords = [
    {"u_now": .2, "u_future": .8, "cost": 10, "_regime_key": ("d",)},
    {"u_now": .5, "u_future": .5, "cost": 10, "_regime_key": ("d",)},
    {"u_now": .8, "u_future": .2, "cost": 10, "_regime_key": ("d",)},
]
check("2D hypervolume matches an analytic staircase union",
      abs(pf.hypervolume_2d(coords) - .37) < 1e-12)
check("duplicate points do not inflate 2D hypervolume",
      abs(pf.hypervolume_2d(coords + [dict(coords[0])]) - .37) < 1e-12)

one = [{"u_now": .5, "u_future": .4, "cost": 50, "_regime_key": ("d",)}]
check("normalised 3D hypervolume matches one analytic box",
      abs(pf.normalized_hypervolume_3d(one, cost_reference=100) - .1) < 1e-12)
two = [
    {"u_now": .2, "u_future": .8, "cost": 20, "_regime_key": ("d",)},
    {"u_now": .8, "u_future": .2, "cost": 80, "_regime_key": ("d",)},
]
check("normalised 3D hypervolume matches exact slicing",
      abs(pf.normalized_hypervolume_3d(two, cost_reference=100) - .152) < 1e-12)
check("duplicates do not inflate 3D hypervolume",
      abs(pf.normalized_hypervolume_3d(two + [dict(two[0])], cost_reference=100) - .152)
      < 1e-12)

try:
    pf.hypervolume_2d(coords + [{"u_now": .3, "u_future": .3, "cost": 3,
                                 "_regime_key": ("other",)}])
except pf.ParetoValidationError:
    pass
else:
    raise AssertionError("FAIL  hypervolume refuses incomparable pooled regimes")
print("PASS  hypervolume refuses incomparable pooled regimes")

try:
    pf.normalized_hypervolume_3d(
        [{"u_now": 1.1, "u_future": .4, "cost": 10, "_regime_key": ("d",)}],
        cost_reference=100)
except pf.ParetoValidationError:
    pass
else:
    raise AssertionError("FAIL  3D hypervolume rejects undeclared bounds")
print("PASS  3D hypervolume rejects undeclared bounds")


print("\n=== scalar preference sweep ===")
tradeoffs = pf.sanitize_points([
    point("present", 20, .9, .1, 20),
    point("future", 20, .5, .7, 20),
    point("future_duplicate", 20, .5, .7, 20),
    point("too_expensive", 80, 1.0, 1.0, 80),
], regime_keys=REGIMES, config_keys=CONFIGS).points
winners = pf.lambda_argmax(tradeoffs, [20], [0, 2 / 3, 1, math.inf], tie_tolerance=1e-10)
by_lambda = {row["lambda"]: row for row in winners}
check("lambda zero selects present utility", by_lambda[0]["winner_config_ids"] == (
    'policy="present",budget_words=20',))
check("the analytic crossing preserves every tie", by_lambda[2 / 3]["n_winners"] == 3)
check("larger lambda selects both equivalent future policies", by_lambda[1]["n_winners"] == 2)
check("infinite lambda is the future-only endpoint", by_lambda[math.inf]["n_winners"] == 2)
check("the cost constraint excludes a higher-scoring arm",
      all("too_expensive" not in winner for winner in by_lambda[1]["winner_config_ids"]))


print("\n=== joint context-cluster bootstrap ===")
observations = []
for dataset in ("d", "other"):
    for context in ("c1", "c2", "c3"):
        observations.append({**point("a", 20, .8, .8, 10, dataset=dataset),
                             "context_id": context})
        observations.append({**point("b", 20, .5, .5, 20, dataset=dataset),
                             "context_id": context})

boot = pf.joint_context_bootstrap(
    observations,
    regime_keys=REGIMES,
    config_keys=CONFIGS,
    context_key="context_id",
    budgets=[20],
    n_resamples=200,
    seed=77,
    cost_reference=100,
)
membership = {(row["dataset"], row["policy"]): row["frontier_probability"]
              for row in boot.global_membership}
check("stable dominator has bootstrap membership one", membership[("d", "a")] == 1.0)
check("stable dominated point has bootstrap membership zero", membership[("d", "b")] == 0.0)
check("bootstrap dominance remains isolated by regime",
      membership[("other", "a")] == 1.0 and membership[("other", "b")] == 0.0)
hv2 = next(row for row in boot.hypervolume_2d if row["dataset"] == "d")
check("constant-panel 2D HV and interval are exact",
      abs(hv2["point_estimate"] - .64) < 1e-12
      and abs(hv2["lo"] - .64) < 1e-12 and abs(hv2["hi"] - .64) < 1e-12)
hv3 = next(row for row in boot.hypervolume_3d if row["dataset"] == "d")
check("constant-panel normalised 3D HV is exact",
      abs(hv3["point_estimate"] - .576) < 1e-12
      and abs(hv3["lo"] - .576) < 1e-12 and abs(hv3["hi"] - .576) < 1e-12)

boot_again = pf.joint_context_bootstrap(
    observations,
    regime_keys=REGIMES,
    config_keys=CONFIGS,
    context_key="context_id",
    budgets=[20],
    n_resamples=200,
    seed=77,
    cost_reference=100,
)
check("fixed-seed bootstrap summary is a reproducible snapshot",
      boot.hypervolume_2d == boot_again.hypervolume_2d
      and boot.global_membership == boot_again.global_membership)

hard_observations = []
for context, assigned in (("c1", 20), ("c2", 20), ("c3", 40)):
    hard_observations.append({
        **point("hard", 40, .7, .7, 18),
        "context_id": context,
        "assigned_cap": assigned,
    })
hard_boot = pf.joint_context_bootstrap(
    hard_observations,
    regime_keys=REGIMES,
    config_keys=CONFIGS,
    context_key="context_id",
    budgets=[20, 40],
    n_resamples=200,
    seed=91,
    cost_reference=100,
    feasibility_cost_key="assigned_cap",
)
hard_at20 = next(row for row in hard_boot.constrained_membership
                 if row["constraint_budget"] == 20)
check("bootstrap hard feasibility uses the maximum sampled cap",
      not hard_at20["point_estimate_feasible"]
      and 0.0 < hard_at20["feasible_probability"] < 1.0)

unbalanced = observations[:-1]
try:
    pf.joint_context_bootstrap(
        unbalanced,
        regime_keys=REGIMES,
        config_keys=CONFIGS,
        context_key="context_id",
        budgets=[20],
        n_resamples=10,
        cost_reference=100,
    )
except pf.ParetoValidationError:
    pass
else:
    raise AssertionError("FAIL  an unbalanced context panel is rejected")
print("PASS  an unbalanced context panel is rejected")

print("\nALL PARETO FRONTIER CHECKS PASSED")
