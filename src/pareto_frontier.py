"""Robust empirical Pareto analysis for bounded communication experiments.

The functions in this module deliberately separate three related objects:

* the raw three-objective frontier, maximising ``U_now`` and ``U_future``
  while minimising realised communication cost ``C``;
* the two-objective feasible frontier at a cap ``B``, which compares *all*
  observed configurations satisfying ``C <= B``; and
* the maximisers of ``U_now + lambda * U_future`` inside that feasible set.

No interpolation between observed configurations is assumed.  Distinct
configurations with identical objective coordinates remain traceable as an
equivalence group, but contribute only once to geometric summaries.

All public analysis functions operate independently within an explicit regime
(for example dataset x model x future-query distribution x metric).  This
prevents an accidental pooled call from allowing incomparable observations to
dominate one another.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


OBJECTIVES = ("u_now", "u_future", "cost")


class ParetoValidationError(ValueError):
    """Raised when frontier inputs are incomplete, ambiguous, or non-finite."""


@dataclass
class SanitizationResult:
    """Validated point rows plus an auditable list of excluded input rows."""

    points: list[dict]
    excluded: list[dict]
    equivalence_groups: list[dict]


@dataclass
class BootstrapResult:
    """Joint context-cluster bootstrap summaries.

    ``point_estimates`` contains the globally marked three-objective points.
    The other fields are tidy tables suitable for CSV output.
    """

    point_estimates: list[dict]
    global_membership: list[dict]
    constrained_membership: list[dict]
    hypervolume_2d: list[dict]
    hypervolume_3d: list[dict]


def _stable_text(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _row_id(keys: Sequence[str], values: Sequence[object], fallback: str) -> str:
    if not keys:
        return fallback
    return ",".join(f"{key}={_stable_text(value)}" for key, value in zip(keys, values))


def _sort_key(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(_stable_text(value) for value in values)


def _finite_float(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParetoValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ParetoValidationError(f"{label} is not finite: {value!r}")
    return number


def _epsilon_tuple(epsilon: float | Sequence[float] | Mapping[str, float],
                   *, include_cost: bool = True) -> tuple[float, float, float]:
    """Return non-negative tolerances in each objective's own units.

    A scalar is accepted for convenience, but a mapping is preferable because
    utility and word cost have different units.  Epsilon suppresses a
    numerically trivial strict improvement; it never permits deterioration in
    another objective.
    """
    if isinstance(epsilon, Mapping):
        values = tuple(float(epsilon.get(key, 0.0)) for key in OBJECTIVES)
    elif isinstance(epsilon, (tuple, list, np.ndarray)):
        raw = tuple(float(value) for value in epsilon)
        if len(raw) == 2 and not include_cost:
            values = (raw[0], raw[1], 0.0)
        elif len(raw) == 3:
            values = raw
        else:
            raise ParetoValidationError("epsilon must have two utility values or three objectives")
    else:
        value = float(epsilon)
        values = (value, value, value if include_cost else 0.0)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ParetoValidationError(f"epsilon values must be finite and non-negative: {values}")
    return values


def sanitize_points(
    rows: Iterable[Mapping],
    *,
    regime_keys: Sequence[str],
    config_keys: Sequence[str],
    u_now_key: str = "u_now",
    u_future_key: str = "u_future",
    cost_key: str = "cost",
    on_invalid: str = "raise",
) -> SanitizationResult:
    """Validate and standardise one aggregate row per configuration.

    ``regime_keys`` name dimensions that must never be compared across one
    another. ``config_keys`` must uniquely identify a configuration inside a
    regime.  Distinct configurations at exactly the same three objective
    coordinates are retained and assigned a shared ``equivalence_id``.

    ``on_invalid='drop'`` is useful for a report that must continue, but every
    exclusion is returned.  The default is intentionally strict.
    """
    if on_invalid not in {"raise", "drop"}:
        raise ParetoValidationError("on_invalid must be 'raise' or 'drop'")
    if not config_keys:
        raise ParetoValidationError("config_keys may not be empty")

    valid: list[dict] = []
    excluded: list[dict] = []
    seen: dict[tuple[tuple, tuple], int] = {}

    for index, source in enumerate(rows):
        row = dict(source)
        try:
            missing = [key for key in (*regime_keys, *config_keys) if key not in row]
            if missing:
                raise ParetoValidationError(f"missing identity fields: {', '.join(missing)}")
            regime = tuple(row[key] for key in regime_keys)
            config = tuple(row[key] for key in config_keys)
            if any(value is None for value in (*regime, *config)):
                raise ParetoValidationError("regime and configuration fields may not be null")
            u_now = _finite_float(row.get(u_now_key), u_now_key)
            u_future = _finite_float(row.get(u_future_key), u_future_key)
            cost = _finite_float(row.get(cost_key), cost_key)
            if cost < 0:
                raise ParetoValidationError(f"cost must be non-negative: {cost}")
            identity = (regime, config)
            if identity in seen:
                raise ParetoValidationError(
                    "duplicate configuration row for "
                    f"{_row_id(regime_keys, regime, 'all')} / "
                    f"{_row_id(config_keys, config, 'configuration')} "
                    f"(first seen at input row {seen[identity]})"
                )
            seen[identity] = index
            row.update({
                "u_now": u_now,
                "u_future": u_future,
                "cost": cost,
                "regime_id": _row_id(regime_keys, regime, "all"),
                "config_id": _row_id(config_keys, config, "configuration"),
                "_regime_key": regime,
                "_config_key": config,
            })
            valid.append(row)
        except ParetoValidationError as exc:
            excluded.append({"input_index": index, "reason": str(exc), "row": row})

    if excluded and on_invalid == "raise":
        preview = "; ".join(
            f"row {entry['input_index']}: {entry['reason']}" for entry in excluded[:5]
        )
        more = "" if len(excluded) <= 5 else f"; plus {len(excluded) - 5} more"
        raise ParetoValidationError(f"invalid Pareto input ({len(excluded)} row(s)): {preview}{more}")

    grouped: dict[
        tuple[tuple, tuple[float, float, float]], list[dict]
    ] = defaultdict(list)
    for row in valid:
        grouped[(row["_regime_key"], (row["u_now"], row["u_future"], row["cost"]))].append(row)

    equivalence_groups: list[dict] = []
    by_regime: dict[tuple, list[tuple[tuple[float, float, float], list[dict]]]] = defaultdict(list)
    for (regime, coordinates), members in grouped.items():
        by_regime[regime].append((coordinates, members))

    for regime in sorted(by_regime, key=_sort_key):
        ordered = sorted(by_regime[regime], key=lambda item: item[0])
        for ordinal, (coordinates, members) in enumerate(ordered, start=1):
            equivalence_id = f"{_row_id(regime_keys, regime, 'all')}|eq={ordinal:04d}"
            member_ids = tuple(sorted(row["config_id"] for row in members))
            for row in members:
                row["equivalence_id"] = equivalence_id
                row["equivalent_config_ids"] = member_ids
                row["equivalence_size"] = len(member_ids)
            equivalence_groups.append({
                "regime_id": _row_id(regime_keys, regime, "all"),
                "_regime_key": regime,
                "equivalence_id": equivalence_id,
                "u_now": coordinates[0],
                "u_future": coordinates[1],
                "cost": coordinates[2],
                "config_ids": member_ids,
                "n_configurations": len(member_ids),
            })

    return SanitizationResult(valid, excluded, equivalence_groups)


def dominates(a: Mapping, b: Mapping,
              epsilon: float | Sequence[float] | Mapping[str, float] = 0.0,
              *, include_cost: bool = True) -> bool:
    """Whether ``a`` epsilon-dominates ``b`` under the declared objectives.

    Utilities are maximised and cost is minimised.  The relation requires no
    deterioration at all and an improvement larger than epsilon in at least
    one objective.  At epsilon zero this is ordinary Pareto dominance.
    """
    eps_now, eps_future, eps_cost = _epsilon_tuple(epsilon, include_cost=include_cost)
    a_now = _finite_float(a.get("u_now"), "u_now")
    a_future = _finite_float(a.get("u_future"), "u_future")
    b_now = _finite_float(b.get("u_now"), "u_now")
    b_future = _finite_float(b.get("u_future"), "u_future")
    no_worse = a_now >= b_now and a_future >= b_future
    strict = a_now > b_now + eps_now or a_future > b_future + eps_future
    if include_cost:
        a_cost = _finite_float(a.get("cost"), "cost")
        b_cost = _finite_float(b.get("cost"), "cost")
        no_worse = no_worse and a_cost <= b_cost
        strict = strict or a_cost < b_cost - eps_cost
    return bool(no_worse and strict)


def _require_sanitized(points: Iterable[Mapping]) -> list[dict]:
    rows = [dict(point) for point in points]
    for index, row in enumerate(rows):
        missing = [key for key in ("u_now", "u_future", "cost", "config_id",
                                   "_regime_key") if key not in row]
        if missing:
            raise ParetoValidationError(
                f"point {index} is not sanitized; missing {', '.join(missing)}"
            )
        for key in OBJECTIVES:
            _finite_float(row[key], key)
    return rows


def mark_nondominated(
    points: Iterable[Mapping],
    epsilon: float | Sequence[float] | Mapping[str, float] = 0.0,
) -> list[dict]:
    """Mark the raw three-objective nondominated set within every regime."""
    rows = _require_sanitized(points)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["_regime_key"]].append(row)
    for cell in grouped.values():
        for point in cell:
            dominators = sorted(
                other["config_id"] for other in cell
                if other["config_id"] != point["config_id"]
                and dominates(other, point, epsilon, include_cost=True)
            )
            point["dominated"] = bool(dominators)
            point["nondominated"] = not dominators
            point["dominated_by"] = tuple(dominators)
    return rows


def _normalise_budgets(budgets: Iterable[float]) -> tuple[float, ...]:
    out = []
    for value in budgets:
        budget = _finite_float(value, "budget")
        if budget < 0:
            raise ParetoValidationError(f"budget must be non-negative: {budget}")
        out.append(budget)
    return tuple(sorted(set(out)))


def constrained_frontiers(
    points: Iterable[Mapping],
    budgets: Iterable[float],
    epsilon: float | Sequence[float] | Mapping[str, float] = 0.0,
    *,
    feasibility_epsilon: float = 0.0,
    feasibility_key: str = "cost",
) -> list[dict]:
    """Mark the two-utility frontier among every point satisfying ``C <= B``.

    Lower-cost configurations from smaller generation rungs remain eligible at
    a larger cap.  This is therefore the feasible set in the scalarised
    constrained objective, unlike a comparison restricted to points that share
    the same requested budget label.  ``feasibility_key`` may name a stricter
    channel quantity (for example, the maximum delivered length or assigned
    hard cap) while ``cost`` remains the mean realised third objective.
    """
    rows = _require_sanitized(points)
    caps = _normalise_budgets(budgets)
    tolerance = _finite_float(feasibility_epsilon, "feasibility_epsilon")
    if tolerance < 0:
        raise ParetoValidationError("feasibility_epsilon must be non-negative")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["_regime_key"]].append(row)

    out: list[dict] = []
    for regime in sorted(grouped, key=_sort_key):
        cell = grouped[regime]
        for cap in caps:
            feasible = [
                row for row in cell
                if _finite_float(row.get(feasibility_key), feasibility_key)
                <= cap + tolerance
            ]
            for point in cell:
                result = dict(point)
                result["constraint_budget"] = cap
                result["feasible"] = point in feasible
                if point not in feasible:
                    result["dominated_at_budget"] = None
                    result["nondominated_at_budget"] = None
                    result["dominated_by_at_budget"] = tuple()
                else:
                    dominators = sorted(
                        other["config_id"] for other in feasible
                        if other["config_id"] != point["config_id"]
                        and dominates(other, point, epsilon, include_cost=False)
                    )
                    result["dominated_at_budget"] = bool(dominators)
                    result["nondominated_at_budget"] = not dominators
                    result["dominated_by_at_budget"] = tuple(dominators)
                out.append(result)
    return out


def _assert_single_regime(points: Sequence[Mapping]) -> None:
    regimes = {tuple(point.get("_regime_key", ())) for point in points}
    if len(regimes) > 1:
        raise ParetoValidationError(
            "hypervolume is undefined across incomparable regimes; group the points first"
        )


def _hypervolume_2d_coordinates(coordinates: Iterable[tuple[float, float]],
                                reference: tuple[float, float]) -> float:
    ref_x, ref_y = reference
    usable = sorted({(float(x), float(y)) for x, y in coordinates
                     if x > ref_x and y > ref_y})
    if not usable:
        return 0.0
    xs = sorted({x for x, _ in usable})
    area = 0.0
    previous_x = ref_x
    for x in xs:
        height = max(y for point_x, y in usable if point_x >= x) - ref_y
        area += (x - previous_x) * height
        previous_x = x
    return float(area)


def hypervolume_2d(points: Iterable[Mapping],
                   reference: tuple[float, float] = (0.0, 0.0)) -> float:
    """Exact union area dominated by observed ``(U_now, U_future)`` points."""
    rows = list(points)
    _assert_single_regime(rows)
    ref = (_finite_float(reference[0], "reference U_now"),
           _finite_float(reference[1], "reference U_future"))
    coordinates = []
    for point in rows:
        coordinates.append((_finite_float(point.get("u_now"), "u_now"),
                            _finite_float(point.get("u_future"), "u_future")))
    return _hypervolume_2d_coordinates(coordinates, ref)


def _normalised_3d_coordinates(
    points: Iterable[Mapping],
    *,
    utility_reference: tuple[float, float],
    utility_ideal: tuple[float, float],
    cost_ideal: float,
    cost_reference: float,
) -> list[tuple[float, float, float]]:
    ref_now, ref_future = map(float, utility_reference)
    ideal_now, ideal_future = map(float, utility_ideal)
    cost_ideal = float(cost_ideal)
    cost_reference = float(cost_reference)
    values = (*utility_reference, *utility_ideal, cost_ideal, cost_reference)
    if any(not math.isfinite(float(value)) for value in values):
        raise ParetoValidationError("hypervolume bounds must be finite")
    if ideal_now <= ref_now or ideal_future <= ref_future:
        raise ParetoValidationError("utility ideal must exceed the utility reference")
    if cost_reference <= cost_ideal:
        raise ParetoValidationError("cost reference must exceed the ideal cost")

    tolerance = 1e-12
    coordinates = []
    for point in points:
        now = _finite_float(point.get("u_now"), "u_now")
        future = _finite_float(point.get("u_future"), "u_future")
        cost = _finite_float(point.get("cost"), "cost")
        if not (ref_now - tolerance <= now <= ideal_now + tolerance):
            raise ParetoValidationError(f"U_now={now} lies outside the declared HV bounds")
        if not (ref_future - tolerance <= future <= ideal_future + tolerance):
            raise ParetoValidationError(f"U_future={future} lies outside the declared HV bounds")
        if not (cost_ideal - tolerance <= cost <= cost_reference + tolerance):
            raise ParetoValidationError(f"cost={cost} lies outside the declared HV bounds")
        x = min(1.0, max(0.0, (now - ref_now) / (ideal_now - ref_now)))
        y = min(1.0, max(0.0, (future - ref_future) / (ideal_future - ref_future)))
        z = min(1.0, max(0.0, (cost_reference - cost)
                             / (cost_reference - cost_ideal)))
        if x > 0 and y > 0 and z > 0:
            coordinates.append((x, y, z))
    return coordinates


def _hypervolume_3d_origin(coordinates: Iterable[tuple[float, float, float]]) -> float:
    unique = sorted(set(coordinates))
    if not unique:
        return 0.0
    levels = sorted({z for _, _, z in unique if z > 0})
    volume = 0.0
    previous_z = 0.0
    for z in levels:
        cross_section = [(x, y) for x, y, point_z in unique if point_z >= z]
        volume += (z - previous_z) * _hypervolume_2d_coordinates(cross_section, (0.0, 0.0))
        previous_z = z
    return float(volume)


def normalized_hypervolume_3d(
    points: Iterable[Mapping],
    *,
    cost_reference: float,
    utility_reference: tuple[float, float] = (0.0, 0.0),
    utility_ideal: tuple[float, float] = (1.0, 1.0),
    cost_ideal: float = 0.0,
) -> float:
    """Exact dominated volume after mapping all objectives to the unit cube.

    Cost is converted to ``(cost_reference - C) / (cost_reference -
    cost_ideal)`` so all three transformed objectives are maximised.  With the
    defaults, the reference point is ``(0, 0, 0)`` and the result lies in
    ``[0, 1]``.  Bounds are strict by design so cross-regime values cannot be
    silently clipped into an apparently comparable statistic.
    """
    rows = list(points)
    _assert_single_regime(rows)
    coordinates = _normalised_3d_coordinates(
        rows,
        utility_reference=utility_reference,
        utility_ideal=utility_ideal,
        cost_ideal=cost_ideal,
        cost_reference=cost_reference,
    )
    return _hypervolume_3d_origin(coordinates)


def lambda_argmax(
    points: Iterable[Mapping],
    budgets: Iterable[float],
    lambdas: Iterable[float],
    *,
    tie_tolerance: float = 1e-12,
    feasibility_epsilon: float = 0.0,
) -> list[dict]:
    """Return all maximisers of ``U_now + lambda U_future`` under ``C <= B``.

    ``lambda=inf`` is accepted as the future-only endpoint.  Ties are preserved
    rather than resolved by input order or by an undeclared cost preference.
    """
    rows = _require_sanitized(points)
    caps = _normalise_budgets(budgets)
    tolerance = _finite_float(tie_tolerance, "tie_tolerance")
    cost_tolerance = _finite_float(feasibility_epsilon, "feasibility_epsilon")
    if tolerance < 0 or cost_tolerance < 0:
        raise ParetoValidationError("tie and feasibility tolerances must be non-negative")
    weights = []
    for value in lambdas:
        weight = float(value)
        if math.isnan(weight) or weight < 0:
            raise ParetoValidationError(f"lambda must be non-negative or infinity: {value!r}")
        weights.append(weight)
    weights = sorted(set(weights), key=lambda value: (math.isinf(value), value))

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["_regime_key"]].append(row)
    out = []
    for regime in sorted(grouped, key=_sort_key):
        cell = grouped[regime]
        regime_id = cell[0]["regime_id"]
        for cap in caps:
            feasible = [row for row in cell if row["cost"] <= cap + cost_tolerance]
            for weight in weights:
                if math.isinf(weight):
                    scores = {row["config_id"]: row["u_future"] for row in feasible}
                    objective = "U_future"
                else:
                    scores = {row["config_id"]: row["u_now"] + weight * row["u_future"]
                              for row in feasible}
                    objective = "U_now + lambda * U_future"
                if not scores:
                    winners: tuple[str, ...] = tuple()
                    equivalence_ids: tuple[str, ...] = tuple()
                    best = float("nan")
                else:
                    best = max(scores.values())
                    winners = tuple(sorted(config_id for config_id, score in scores.items()
                                           if abs(score - best) <= tolerance))
                    by_id = {row["config_id"]: row for row in feasible}
                    equivalence_ids = tuple(sorted({by_id[w]["equivalence_id"] for w in winners}))
                out.append({
                    "regime_id": regime_id,
                    "_regime_key": regime,
                    "constraint_budget": cap,
                    "lambda": weight,
                    "objective": objective,
                    "best_score": best,
                    "winner_config_ids": winners,
                    "winner_equivalence_ids": equivalence_ids,
                    "n_winners": len(winners),
                    "n_feasible": len(feasible),
                })
    return out


def _dominance_flags(now: np.ndarray, future: np.ndarray, cost: np.ndarray,
                     epsilon: tuple[float, float, float], *, include_cost: bool,
                     feasible: np.ndarray | None = None) -> np.ndarray:
    """Vectorised flags with rows as possible dominators and columns as targets."""
    n = len(now)
    eligible = np.ones(n, dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
    no_worse = now[:, None] >= now[None, :]
    no_worse &= future[:, None] >= future[None, :]
    strict = now[:, None] > now[None, :] + epsilon[0]
    strict |= future[:, None] > future[None, :] + epsilon[1]
    if include_cost:
        no_worse &= cost[:, None] <= cost[None, :]
        strict |= cost[:, None] < cost[None, :] - epsilon[2]
    relation = no_worse & strict
    relation &= eligible[:, None] & eligible[None, :]
    np.fill_diagonal(relation, False)
    dominated = relation.any(axis=0)
    dominated[~eligible] = False
    return dominated


def _percentile(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def joint_context_bootstrap(
    observations: Iterable[Mapping],
    *,
    regime_keys: Sequence[str],
    config_keys: Sequence[str],
    context_key: str,
    budgets: Iterable[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    epsilon: float | Sequence[float] | Mapping[str, float] = 0.0,
    feasibility_epsilon: float = 0.0,
    cost_reference: float | None = None,
    utility_reference: tuple[float, float] = (0.0, 0.0),
    utility_ideal: tuple[float, float] = (1.0, 1.0),
    cost_ideal: float = 0.0,
    u_now_key: str = "u_now",
    u_future_key: str = "u_future",
    cost_key: str = "cost",
    feasibility_cost_key: str | None = None,
) -> BootstrapResult:
    """Jointly bootstrap frontier membership and hypervolume by context.

    Multiple observation rows inside a context/configuration cell are averaged
    first (for example, question rotations).  Every configuration in a regime
    must cover the same context IDs.  A bootstrap replicate samples those IDs
    once and applies the identical draw to every configuration, preserving the
    paired covariance that determines dominance.  When
    ``feasibility_cost_key`` is supplied, constrained membership uses the
    maximum of that quantity over the sampled context rows.  This supports a
    true per-message hard cap while retaining mean delivered words as the raw
    three-objective cost.
    """
    if int(n_resamples) != n_resamples or int(n_resamples) <= 0:
        raise ParetoValidationError("n_resamples must be a positive integer")
    n_resamples = int(n_resamples)
    ci = _finite_float(ci, "ci")
    if not 0 < ci < 1:
        raise ParetoValidationError("ci must lie strictly between zero and one")
    caps = _normalise_budgets(budgets)
    if not caps:
        raise ParetoValidationError("at least one constraint budget is required")
    if not config_keys:
        raise ParetoValidationError("config_keys may not be empty")
    if cost_reference is None:
        cost_reference = max(caps)
    cost_reference = _finite_float(cost_reference, "cost_reference")
    eps = _epsilon_tuple(epsilon)
    feasibility_epsilon = _finite_float(feasibility_epsilon, "feasibility_epsilon")
    if feasibility_epsilon < 0:
        raise ParetoValidationError("feasibility_epsilon must be non-negative")

    buckets: dict[tuple[tuple, tuple, object], list[tuple[float, float, float, float]]] = defaultdict(list)
    config_meta: dict[tuple[tuple, tuple], dict] = {}
    invalid = []
    for index, source in enumerate(observations):
        row = dict(source)
        try:
            identity_keys = (*regime_keys, *config_keys, context_key)
            missing = [key for key in identity_keys if key not in row]
            if missing:
                raise ParetoValidationError(f"missing fields: {', '.join(missing)}")
            regime = tuple(row[key] for key in regime_keys)
            config = tuple(row[key] for key in config_keys)
            context = row[context_key]
            if any(value is None for value in (*regime, *config, context)):
                raise ParetoValidationError("regime, configuration, and context may not be null")
            now = _finite_float(row.get(u_now_key), u_now_key)
            future = _finite_float(row.get(u_future_key), u_future_key)
            cost = _finite_float(row.get(cost_key), cost_key)
            if cost < 0:
                raise ParetoValidationError("cost must be non-negative")
            feasibility_cost = (
                _finite_float(row.get(feasibility_cost_key), feasibility_cost_key)
                if feasibility_cost_key is not None else cost
            )
            if feasibility_cost < 0:
                raise ParetoValidationError("feasibility cost must be non-negative")
            buckets[(regime, config, context)].append(
                (now, future, cost, feasibility_cost)
            )
            config_meta.setdefault((regime, config), row)
        except ParetoValidationError as exc:
            invalid.append(f"row {index}: {exc}")
    if invalid:
        raise ParetoValidationError("invalid bootstrap observations: " + "; ".join(invalid[:5]))
    if not buckets:
        raise ParetoValidationError("bootstrap observations are empty")

    # Collapse rotations or other repeated measurements within a context first.
    collapsed: dict[tuple[tuple, tuple, object], tuple[float, float, float, float]] = {}
    for key, values in buckets.items():
        array = np.asarray(values, dtype=float)
        averaged = np.mean(array[:, :3], axis=0)
        collapsed[key] = (
            float(averaged[0]), float(averaged[1]), float(averaged[2]),
            float(np.max(array[:, 3]) if feasibility_cost_key is not None
                  else np.mean(array[:, 3])),
        )

    regimes = sorted({key[0] for key in collapsed}, key=_sort_key)
    aggregate_rows = []
    panels = {}
    for regime in regimes:
        configs = sorted({key[1] for key in collapsed if key[0] == regime}, key=_sort_key)
        contexts_by_config = {
            config: {key[2] for key in collapsed if key[0] == regime and key[1] == config}
            for config in configs
        }
        reference_contexts = contexts_by_config[configs[0]]
        mismatched = [config for config in configs
                      if contexts_by_config[config] != reference_contexts]
        if mismatched:
            detail = "; ".join(
                f"{_row_id(config_keys, config, 'configuration')}: "
                f"{len(contexts_by_config[config])} contexts"
                for config in mismatched[:5]
            )
            raise ParetoValidationError(
                "joint bootstrap requires a balanced context panel within "
                f"{_row_id(regime_keys, regime, 'all')}; {detail}"
            )
        contexts = sorted(reference_contexts, key=lambda value: _stable_text(value))
        values = np.empty((len(configs), len(contexts), 4), dtype=float)
        for i, config in enumerate(configs):
            for j, context in enumerate(contexts):
                values[i, j, :] = collapsed[(regime, config, context)]
            meta = dict(config_meta[(regime, config)])
            meta.update({
                **{key: value for key, value in zip(regime_keys, regime)},
                **{key: value for key, value in zip(config_keys, config)},
                "u_now": float(values[i, :, 0].mean()),
                "u_future": float(values[i, :, 1].mean()),
                "cost": float(values[i, :, 2].mean()),
                "_feasibility_cost": float(
                    values[i, :, 3].max() if feasibility_cost_key is not None
                    else values[i, :, 3].mean()
                ),
            })
            aggregate_rows.append(meta)
        panels[regime] = (configs, contexts, values)

    sanitized = sanitize_points(
        aggregate_rows,
        regime_keys=regime_keys,
        config_keys=config_keys,
        u_now_key="u_now",
        u_future_key="u_future",
        cost_key="cost",
    )
    point_estimates = mark_nondominated(sanitized.points, eps)
    estimate_by_key = {(row["_regime_key"], row["_config_key"]): row
                       for row in point_estimates}
    constrained_estimates = constrained_frontiers(
        point_estimates, caps, eps, feasibility_epsilon=feasibility_epsilon,
        feasibility_key="_feasibility_cost",
    )
    constrained_estimate_by_key = {
        (row["_regime_key"], row["_config_key"], row["constraint_budget"]): row
        for row in constrained_estimates
    }

    rng = np.random.default_rng(seed)
    global_membership = []
    constrained_membership = []
    hypervolume_2d_rows = []
    hypervolume_3d_rows = []

    for regime in regimes:
        configs, contexts, values = panels[regime]
        n_configs, n_contexts, _ = values.shape
        global_counts = np.zeros(n_configs, dtype=int)
        feasible_counts = {cap: np.zeros(n_configs, dtype=int) for cap in caps}
        frontier_counts = {cap: np.zeros(n_configs, dtype=int) for cap in caps}
        hv2_samples = {cap: np.zeros(n_resamples, dtype=float) for cap in caps}
        hv3_samples = np.zeros(n_resamples, dtype=float)

        for replicate in range(n_resamples):
            draw = rng.integers(0, n_contexts, size=n_contexts)
            sampled = values[:, draw, :]
            means = sampled[:, :, :3].mean(axis=1)
            now, future, cost = means[:, 0], means[:, 1], means[:, 2]
            feasibility_cost = (
                sampled[:, :, 3].max(axis=1) if feasibility_cost_key is not None
                else sampled[:, :, 3].mean(axis=1)
            )
            dominated_global = _dominance_flags(now, future, cost, eps, include_cost=True)
            global_counts += (~dominated_global).astype(int)
            for cap in caps:
                feasible = feasibility_cost <= cap + feasibility_epsilon
                dominated = _dominance_flags(
                    now, future, cost, eps, include_cost=False, feasible=feasible
                )
                feasible_counts[cap] += feasible.astype(int)
                frontier_counts[cap] += (feasible & ~dominated).astype(int)
                hv2_samples[cap][replicate] = _hypervolume_2d_coordinates(
                    [(now[i], future[i]) for i in range(n_configs) if feasible[i]],
                    utility_reference,
                )
            coords = _normalised_3d_coordinates(
                ({"u_now": now[i], "u_future": future[i], "cost": cost[i]}
                 for i in range(n_configs)),
                utility_reference=utility_reference,
                utility_ideal=utility_ideal,
                cost_ideal=cost_ideal,
                cost_reference=cost_reference,
            )
            hv3_samples[replicate] = _hypervolume_3d_origin(coords)

        regime_id = _row_id(regime_keys, regime, "all")
        estimate_points = [estimate_by_key[(regime, config)] for config in configs]
        for i, config in enumerate(configs):
            estimate = estimate_by_key[(regime, config)]
            identity = {
                **{key: value for key, value in zip(regime_keys, regime)},
                **{key: value for key, value in zip(config_keys, config)},
                "regime_id": regime_id,
                "config_id": estimate["config_id"],
                "equivalence_id": estimate["equivalence_id"],
            }
            global_membership.append({
                **identity,
                "point_estimate_nondominated": estimate["nondominated"],
                "frontier_probability": float(global_counts[i] / n_resamples),
                "n_resamples": n_resamples,
                "n_contexts": n_contexts,
            })
            for cap in caps:
                constrained = constrained_estimate_by_key[(regime, config, cap)]
                feasible_probability = float(feasible_counts[cap][i] / n_resamples)
                frontier_probability = float(frontier_counts[cap][i] / n_resamples)
                conditional = (float(frontier_counts[cap][i] / feasible_counts[cap][i])
                               if feasible_counts[cap][i] else float("nan"))
                constrained_membership.append({
                    **identity,
                    "constraint_budget": cap,
                    "point_estimate_feasible": constrained["feasible"],
                    "point_estimate_nondominated": constrained["nondominated_at_budget"],
                    "feasible_probability": feasible_probability,
                    "frontier_probability": frontier_probability,
                    "frontier_probability_given_feasible": conditional,
                    "n_resamples": n_resamples,
                    "n_contexts": n_contexts,
                })

        for cap in caps:
            estimate_feasible = [
                point for point in estimate_points
                if point["_feasibility_cost"] <= cap + feasibility_epsilon
            ]
            point_hv = hypervolume_2d(estimate_feasible, utility_reference)
            lo, hi = _percentile(hv2_samples[cap], ci)
            hypervolume_2d_rows.append({
                **{key: value for key, value in zip(regime_keys, regime)},
                "regime_id": regime_id,
                "constraint_budget": cap,
                "reference_u_now": utility_reference[0],
                "reference_u_future": utility_reference[1],
                "point_estimate": point_hv,
                "bootstrap_mean": float(hv2_samples[cap].mean()),
                "lo": lo,
                "hi": hi,
                "ci_level": ci,
                "n_resamples": n_resamples,
                "n_contexts": n_contexts,
            })

        point_hv3 = normalized_hypervolume_3d(
            estimate_points,
            cost_reference=cost_reference,
            utility_reference=utility_reference,
            utility_ideal=utility_ideal,
            cost_ideal=cost_ideal,
        )
        lo, hi = _percentile(hv3_samples, ci)
        hypervolume_3d_rows.append({
            **{key: value for key, value in zip(regime_keys, regime)},
            "regime_id": regime_id,
            "cost_reference": cost_reference,
            "cost_ideal": cost_ideal,
            "reference_u_now": utility_reference[0],
            "reference_u_future": utility_reference[1],
            "ideal_u_now": utility_ideal[0],
            "ideal_u_future": utility_ideal[1],
            "point_estimate": point_hv3,
            "bootstrap_mean": float(hv3_samples.mean()),
            "lo": lo,
            "hi": hi,
            "ci_level": ci,
            "n_resamples": n_resamples,
            "n_contexts": n_contexts,
        })

    return BootstrapResult(
        point_estimates=point_estimates,
        global_membership=global_membership,
        constrained_membership=constrained_membership,
        hypervolume_2d=hypervolume_2d_rows,
        hypervolume_3d=hypervolume_3d_rows,
    )
