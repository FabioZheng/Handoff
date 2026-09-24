"""Render the paper-oriented bounded-communication frontier study.

This is an analysis-only stage.  It combines the completed Experiment 10
utility matrices with any completed Experiment 11 preference-sweep matrices,
reweights the relation-dossier off-diagonal cells to prespecified deployment
distributions, and computes Pareto objects without comparing incomparable
regimes.

The default deliberately excludes smoke tests and partially written
Experiment 11 directories.  ``--include-partial`` exists for development, but
the resulting report is stamped PROVISIONAL.  Re-running the command is enough
to discover newly completed replications.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import pareto_frontier as pf  # noqa: E402


REPORT_SCHEMA = "bounded-communication-frontier-report-v1"
REGIME_KEYS = ("stack_id", "dataset", "future_distribution", "metric", "scope")
CONFIG_KEYS = ("configuration_id",)
METRIC_COLUMNS = {
    "judge_correct": "u_judge_correct",
    "em": "u_em",
    "f1": "u_f1",
}
RELATIONS = ("paraphrase", "same_entity", "same_topic", "orthogonal")
PRIMARY_METRIC = "judge_correct"
STANDARD_CAPS = (20, 40, 80, 160)

FAMILY_COLORS = {
    "legacy": "#4472C4",
    "extractive": "#70AD47",
    "scalar": "#ED7D31",
    "split": "#C00000",
    "oracle": "#7030A0",
}
FAMILY_MARKERS = {
    "legacy": "o",
    "extractive": "s",
    "scalar": "D",
    "split": "^",
    "oracle": "*",
}
POLICY_LABELS = {
    "generic": "Generic",
    "conditioned": "Conditioned",
    "reusable": "Reusable",
    "oracle": "Query-aware oracle",
    "extractive_generic": "Extractive generic",
    "extractive_conditioned": "Extractive conditioned",
}
DIST_LABELS = {
    "empirical_questions": "Natural questions",
    "empirical_cells": "Empirical cells",
    "relation_balanced": "Relation-balanced",
    "novel_near": "Nearby new questions",
    "rho25": "25% orthogonal",
    "rho50": "50% orthogonal",
    "rho75": "75% orthogonal",
    "far": "Orthogonal only",
    "paraphrase_sanity": "Paraphrase sanity",
}

FAMILY_DISPLAY = {
    "legacy": "Legacy",
    "extractive": "Extractive",
    "scalar": "Free-form objective λ",
    "split": "Structured allocation α",
    "oracle": "Oracle",
}


@dataclass(frozen=True)
class Artifact:
    kind: str
    result_dir: Path
    manifest: dict
    complete: bool
    completeness_note: str
    selected: bool


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _jsonable(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})


def _as_float(value, default: float | None = None) -> float | None:
    if value in (None, "", "None", "null", "nan", "NaN"):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**32)


def _short_model(model: str) -> str:
    aliases = {
        "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
        "mistralai/ministral-8b": "Ministral-8B",
        "mistralai/ministral-8b-2512": "Ministral-3-8B-2512",
    }
    return aliases.get(model, model.rsplit("/", 1)[-1])


def _stack_metadata(manifest: Mapping) -> tuple[str, str]:
    sender = str(manifest.get("sender_model") or manifest.get("model") or "unknown")
    reader = str(manifest.get("answerer_model") or sender)
    temperature = float(manifest.get("sender_temperature", 0.0) or 0.0)
    seed = manifest.get("sender_seed")
    seed_text = "none" if seed in (None, "") else str(seed)
    stack_id = f"sender={sender}|reader={reader}|temperature={temperature:g}|seed={seed_text}"
    label = f"{_short_model(sender)} writer / {_short_model(reader)} reader"
    if temperature:
        label += f", T={temperature:g}, seed={seed_text}"
    return stack_id, label


def _largest_by_context(paths: Iterable[Path]) -> Path | None:
    candidates = []
    for path in paths:
        try:
            manifest = _read_json(path)
            contexts = int(manifest.get("contexts", -1))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((contexts, path.stat().st_mtime_ns, path))
    return max(candidates, default=(None, None, None))[2]


def _replication_budgets(replication: Mapping, dataset: str) -> set[int]:
    """Return the prespecified budgets for one replication/dataset cell.

    ``budgets_words`` remains the backwards-compatible default and union used
    by runners.  ``budgets_by_dataset`` may narrow that grid for a particular
    corpus when a common replication spans datasets with different useful
    budget ranges.
    """
    by_dataset = replication.get("budgets_by_dataset", {}) or {}
    configured = by_dataset.get(dataset, replication.get("budgets_words", []))
    return {int(value) for value in configured}


def _control_completeness(manifest: Mapping, result_dir: Path,
                          expected: Mapping | None) -> tuple[bool, str]:
    """Fail closed for declared Experiment 10 control panels."""
    matrix = result_dir / "utility_matrix.csv"
    missing: list[str] = []
    if not matrix.is_file() or matrix.stat().st_size == 0:
        missing.append("utility matrix")
    if int(manifest.get("contexts", 0)) <= 0:
        missing.append("contexts")
    if expected:
        required_n = int(expected.get("contexts",
                                      expected.get("limits", {}).get(
                                          str(manifest.get("dataset")), 0)))
        got_n = int(manifest.get("contexts", 0))
        if got_n < required_n:
            missing.append(f"contexts {got_n}/{required_n}")
        required_policies = set(map(str, expected.get("policies", [])))
        got_policies = set(map(str, manifest.get("policies", [])))
        if not required_policies.issubset(got_policies):
            missing.append("policies " + ",".join(sorted(required_policies-got_policies)))
        required_budgets = {int(value) for value in expected.get("budgets_words", [])}
        got_budgets = {int(value) for value in manifest.get("budgets_words", [])}
        if not required_budgets.issubset(got_budgets):
            missing.append("budgets " + ",".join(
                map(str, sorted(required_budgets-got_budgets))))
    return (not missing, "completed source experiment" if not missing
            else "incomplete: " + "; ".join(missing))


def discover_artifacts(root: Path, cfg: Mapping, *, include_partial: bool = False
                       ) -> tuple[list[Artifact], list[dict]]:
    """Find one best result directory per experiment/replication/dataset.

    Experiment 11 completeness is checked against the prespecified config,
    not merely against the directory name.  This catches a full-n run that
    was analysed with only one budget or preference after a chunked execution.
    """
    artifacts: list[Artifact] = []
    inventory: list[dict] = []

    legacy = root / "results" / "communication_regret"
    legacy_paths = defaultdict(list)
    for path in legacy.glob("*/n*/manifest.json"):
        try:
            dataset = str(_read_json(path).get("dataset"))
        except Exception:
            continue
        legacy_paths[dataset].append(path)
    for dataset, paths in sorted(legacy_paths.items()):
        chosen = _largest_by_context(paths)
        if chosen is None:
            continue
        manifest = _read_json(chosen)
        expected = (cfg.get("frontier", {})
                    .get("source_experiment10_requirements", {}).get(dataset))
        complete, note = _control_completeness(manifest, chosen.parent, expected)
        artifact = Artifact("experiment10", chosen.parent, manifest, complete,
                            note, complete or include_partial)
        if artifact.selected:
            artifacts.append(artifact)
        inventory.append(_inventory_row(artifact))

    frontier_root = root / str(cfg["outputs"]["result_root"])
    grouped = defaultdict(list)
    embedded_controls = defaultdict(list)
    for path in frontier_root.glob("*/*/n*/manifest.json"):
        try:
            manifest = _read_json(path)
            replication = str(manifest.get("replication") or path.parents[2].name)
            dataset = str(manifest.get("dataset") or path.parents[1].name)
            key = (replication, dataset)
            if manifest.get("schema_version") == "communication-regret-v1":
                embedded_controls[key].append(path)
            else:
                grouped[key].append(path)
        except Exception:
            continue

    # Model-matched Experiment 10 controls can live beside the preference
    # replications.  Keep them as Experiment 10 artifacts, but record their
    # directory replication id so the inventory remains unambiguous.  Their
    # sender/reader/temperature/seed fields produce the same stack_id as a
    # matching Experiment 11 run; no cross-model Pareto comparison is created.
    for (replication, _dataset), paths in sorted(embedded_controls.items()):
        chosen = _largest_by_context(paths)
        if chosen is None:
            continue
        manifest = _read_json(chosen)
        manifest.setdefault("replication", replication)
        expected = (cfg.get("frontier", {}).get("control_replications", {})
                    .get(replication))
        complete, note = _control_completeness(manifest, chosen.parent, expected)
        artifact = Artifact("experiment10", chosen.parent, manifest, complete, note,
                            complete or include_partial)
        if artifact.selected:
            artifacts.append(artifact)
        inventory.append(_inventory_row(artifact))

    rep_cfg = cfg.get("frontier", {}).get("replications", {})
    for (replication, dataset), paths in sorted(grouped.items()):
        chosen = _largest_by_context(paths)
        if chosen is None:
            continue
        manifest = _read_json(chosen)
        if (manifest.get("schema_version") != "bounded-communication-frontier-v2"
                or replication not in rep_cfg):
            artifact = Artifact(
                "unrecognized", chosen.parent, manifest, False,
                "excluded: unrecognized schema or replication id", False,
            )
            inventory.append(_inventory_row(artifact))
            continue
        expected = rep_cfg.get(replication, {})
        required_n = int(expected.get("limits", {}).get(dataset, manifest.get("contexts", 0)))
        required_budgets = _replication_budgets(expected, dataset)
        required_prefs = set(map(str, expected.get("preference_ids", [])))
        required_families = set(map(str, expected.get("families", [])))
        got_n = int(manifest.get("contexts", 0))
        got_budgets = {int(x) for x in manifest.get("budgets_words", [])}
        got_prefs = {str(item.get("preference_id"))
                     for item in manifest.get("preferences", [])}
        got_families = set(map(str, manifest.get("families", [])))
        missing = []
        if got_n < required_n:
            missing.append(f"contexts {got_n}/{required_n}")
        if not required_budgets.issubset(got_budgets):
            missing.append("budgets " + ",".join(map(str, sorted(required_budgets-got_budgets))))
        if not required_prefs.issubset(got_prefs):
            missing.append("preferences " + ",".join(sorted(required_prefs-got_prefs)))
        if not required_families.issubset(got_families):
            missing.append("families " + ",".join(sorted(required_families-got_families)))
        complete = not missing
        note = "complete" if complete else "incomplete: " + "; ".join(missing)
        artifact = Artifact("experiment11", chosen.parent, manifest, complete, note,
                            complete or include_partial)
        if artifact.selected:
            artifacts.append(artifact)
        inventory.append(_inventory_row(artifact))

    # Prespecified-but-not-yet-created runs remain visible in the report.
    existing = {(row["replication"], row["dataset"]) for row in inventory
                if row["kind"] == "experiment11"}
    for replication, rep in rep_cfg.items():
        for dataset in rep.get("datasets", []):
            if (replication, dataset) in existing:
                continue
            inventory.append({
                "kind": "experiment11",
                "replication": replication,
                "dataset": dataset,
                "contexts": 0,
                "selected": False,
                "complete": False,
                "status": "not found",
                "result_dir": "",
            })
    return artifacts, sorted(inventory, key=lambda row: (
        row["kind"], row["replication"], row["dataset"]))


def _inventory_row(artifact: Artifact) -> dict:
    return {
        "kind": artifact.kind,
        "replication": artifact.manifest.get("replication", "legacy"),
        "dataset": artifact.manifest.get("dataset", "unknown"),
        "contexts": artifact.manifest.get("contexts", 0),
        "selected": artifact.selected,
        "complete": artifact.complete,
        "status": artifact.completeness_note,
        "result_dir": str(artifact.result_dir.relative_to(ROOT)
                          if artifact.result_dir.is_relative_to(ROOT)
                          else artifact.result_dir),
    }


def validate_distributions(cfg: Mapping) -> dict[str, dict[str, float]]:
    distributions = {}
    for name, source in cfg["frontier"]["future_distributions"].items():
        weights = {relation: float(source.get(relation, 0.0)) for relation in RELATIONS}
        if any(value < 0 or not math.isfinite(value) for value in weights.values()):
            raise ValueError(f"future distribution {name!r} has invalid weights")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError(f"future distribution {name!r} does not sum to one")
        distributions[str(name)] = weights
    return distributions


def _policy_family(row: Mapping) -> str:
    explicit = str(row.get("family") or "").strip()
    if explicit in {"scalar", "split"}:
        return explicit
    policy = str(row.get("policy") or "")
    if policy.startswith("oracle"):
        return "oracle"
    if policy.startswith("extractive"):
        return "extractive"
    return "legacy"


def _lambda_label(value: float | None, infinite: bool = False) -> str:
    if infinite:
        return "∞"
    return "?" if value is None else f"{value:g}"


def _policy_label(policy: str, family: str, preference_id: str,
                  future_share: float | None, lambda_value: float | None = None,
                  lambda_infinite: bool = False) -> str:
    if family == "scalar":
        return f"Free-form objective, λ={_lambda_label(lambda_value, lambda_infinite)}"
    if family == "split":
        share = "?" if future_share is None else f"{future_share:g}"
        return f"Structured allocation, α={share}"
    clean = policy.replace("@trim", " (trim sensitivity)")
    return POLICY_LABELS.get(clean, clean.replace("_", " ").title())


def load_matrix_rows(artifacts: Sequence[Artifact]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    excluded: list[dict] = []
    for artifact in artifacts:
        matrix_path = artifact.result_dir / "utility_matrix.csv"
        if not matrix_path.exists():
            excluded.append({"stage": "load", "reason": "missing utility_matrix.csv",
                             "path": str(matrix_path)})
            continue
        stack_id, stack_label = _stack_metadata(artifact.manifest)
        for index, row in enumerate(_read_csv(matrix_path), start=2):
            policy = str(row.get("policy") or "")
            budget = _as_float(row.get("budget_words"))
            delivered = _as_float(row.get("delivered_words"))
            if not policy or budget is None or delivered is None:
                excluded.append({"stage": "load", "reason": "missing policy/budget/cost",
                                 "path": str(matrix_path), "csv_row": index})
                continue
            if delivered > budget + 1e-9:
                excluded.append({"stage": "load", "reason": "hard cap violation",
                                 "path": str(matrix_path), "csv_row": index,
                                 "delivered_words": delivered,
                                 "budget_words": budget})
                continue
            family = _policy_family(row)
            pref = str(row.get("preference_id") or "")
            share = _as_float(row.get("future_share"))
            configuration_id = f"{policy}|requested_B={int(budget)}"
            lambda_value = _as_float(row.get("lambda"))
            lambda_infinite = _as_bool(row.get("lambda_infinite"))
            row.update({
                "artifact_kind": artifact.kind,
                "artifact_dir": str(artifact.result_dir),
                "stack_id": stack_id,
                "stack_label": stack_label,
                "policy": policy,
                "policy_family": family,
                "policy_label": _policy_label(policy, family, pref, share,
                                              lambda_value, lambda_infinite),
                "preference_id": pref,
                "future_share": share,
                "lambda": lambda_value,
                "lambda_infinite": lambda_infinite,
                "requested_budget": int(budget),
                "cost": float(delivered),
                "configuration_id": configuration_id,
                "is_diagonal": _as_bool(row.get("is_diagonal")),
                "deployable": family != "oracle",
                "analysis_eligible": "@trim" not in policy,
            })
            rows.append(row)
    return rows, excluded


def load_baseline_rows(artifacts: Sequence[Artifact]) -> list[dict]:
    """Load and deduplicate model/corpus control baselines for the narrative."""
    by_key: dict[tuple, dict] = {}
    for artifact in artifacts:
        path = artifact.result_dir / "baselines.csv"
        if not path.is_file():
            continue
        reader = str(artifact.manifest.get("answerer_model")
                     or artifact.manifest.get("sender_model")
                     or artifact.manifest.get("model") or "unknown")
        stack_id = f"reader={reader}"
        stack_label = f"{_short_model(reader)} reader"
        dataset = str(artifact.manifest.get("dataset", "unknown"))
        for row in _read_csv(path):
            key = (stack_id, dataset, str(row.get("baseline")), str(row.get("metric")))
            candidate = {
                **row, "stack_id": stack_id, "stack_label": stack_label,
                "dataset": dataset,
            }
            previous = by_key.get(key)
            if previous is not None:
                left = _as_float(previous.get("mean"))
                right = _as_float(candidate.get("mean"))
                if left is not None and right is not None and abs(left-right) > 1e-12:
                    raise ValueError(f"inconsistent duplicate baseline for {key}")
            else:
                by_key[key] = candidate
    return [by_key[key] for key in sorted(by_key, key=lambda item: tuple(map(str, item)))]


def build_rotation_observations(matrix_rows: Sequence[Mapping], cfg: Mapping
                                ) -> tuple[list[dict], list[dict]]:
    """Construct context/rotation utilities under each query distribution.

    Relation weights apply to category means and then to rotations.  Thus an
    orthogonal category with twelve cells does not receive twelve times the
    weight of a one-cell category unless the explicitly named
    ``empirical_cells`` distribution says so.
    """
    distributions = validate_distributions(cfg)
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    identity = ("stack_id", "stack_label", "dataset", "context_id", "rotation_id",
                "policy", "policy_family", "policy_label", "preference_id",
                "future_share", "lambda", "lambda_infinite", "requested_budget",
                "configuration_id", "deployable", "analysis_eligible")
    for row in matrix_rows:
        key = tuple(row.get(field) for field in identity)
        grouped[key].append(row)

    observations: list[dict] = []
    excluded: list[dict] = []
    for key, cells in grouped.items():
        meta = dict(zip(identity, key))
        dataset = str(meta["dataset"])
        diagonal = [row for row in cells if row["is_diagonal"]]
        off = [row for row in cells if not row["is_diagonal"]]
        if not diagonal or not off:
            excluded.append({"stage": "rotation", "reason": "missing diagonal/off-diagonal",
                             "configuration_id": meta["configuration_id"],
                             "context_id": meta["context_id"],
                             "rotation_id": meta["rotation_id"]})
            continue
        costs = [float(row["cost"]) for row in cells]
        if max(costs) - min(costs) > 1e-9:
            excluded.append({"stage": "rotation", "reason": "inconsistent delivered cost",
                             "configuration_id": meta["configuration_id"],
                             "context_id": meta["context_id"],
                             "rotation_id": meta["rotation_id"]})
            continue
        active_distributions: dict[str, dict[str, float] | None]
        if dataset == "relation_dossiers":
            active_distributions = distributions
        else:
            active_distributions = {"empirical_questions": None}

        for metric, column in METRIC_COLUMNS.items():
            now_values = [_as_float(row.get(column)) for row in diagonal]
            if any(value is None for value in now_values):
                excluded.append({"stage": "rotation", "reason": f"missing {column}",
                                 "configuration_id": meta["configuration_id"],
                                 "context_id": meta["context_id"],
                                 "rotation_id": meta["rotation_id"]})
                continue
            u_now = float(np.mean(now_values))
            relation_values: dict[str, list[float]] = defaultdict(list)
            missing_metric = False
            for row in off:
                value = _as_float(row.get(column))
                if value is None:
                    missing_metric = True
                    break
                relation_values[str(row.get("relation") or "unlabelled")].append(value)
            if missing_metric:
                excluded.append({"stage": "rotation", "reason": f"missing off-diagonal {column}",
                                 "configuration_id": meta["configuration_id"],
                                 "context_id": meta["context_id"],
                                 "rotation_id": meta["rotation_id"]})
                continue

            for distribution, weights in active_distributions.items():
                if weights is None:
                    u_future = float(np.mean([value for values in relation_values.values()
                                              for value in values]))
                else:
                    absent = [relation for relation, weight in weights.items()
                              if weight > 0 and not relation_values.get(relation)]
                    if absent:
                        excluded.append({
                            "stage": "distribution",
                            "reason": "positive-weight relation absent: " + ",".join(absent),
                            "future_distribution": distribution,
                            "configuration_id": meta["configuration_id"],
                            "context_id": meta["context_id"],
                            "rotation_id": meta["rotation_id"],
                        })
                        continue
                    means = {relation: float(np.mean(values))
                             for relation, values in relation_values.items() if values}
                    u_future = sum(weight * means[relation]
                                   for relation, weight in weights.items() if weight > 0)
                observations.append({
                    **meta,
                    "future_distribution": distribution,
                    "metric": metric,
                    "u_now": u_now,
                    "u_future": float(u_future),
                    "cost": costs[0],
                    "feasibility_cost": costs[0],
                })
    return observations, excluded


def collapse_contexts(rotation_rows: Sequence[Mapping]) -> list[dict]:
    keys = ("stack_id", "stack_label", "dataset", "future_distribution", "metric",
            "configuration_id", "policy", "policy_family", "policy_label",
            "preference_id", "future_share", "lambda", "lambda_infinite",
            "requested_budget", "deployable", "analysis_eligible", "context_id")
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rotation_rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, rows in groups.items():
        meta = dict(zip(keys, key))
        out.append({
            **meta,
            "u_now": float(np.mean([float(row["u_now"]) for row in rows])),
            "u_future": float(np.mean([float(row["u_future"]) for row in rows])),
            "cost": float(np.mean([float(row["cost"]) for row in rows])),
            "feasibility_cost": float(max(float(row["feasibility_cost"])
                                           for row in rows)),
            "n_rotations": len(rows),
        })
    return out


def _analysis_cells(context_rows: Sequence[Mapping]) -> dict[tuple, list[dict]]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in context_rows:
        if not row["analysis_eligible"]:
            continue
        key = (row["stack_id"], row["dataset"], row["future_distribution"], row["metric"])
        cells[key].append(dict(row))
    return cells


def _balanced_panel(rows: Sequence[Mapping], scope: str) -> tuple[list[dict], set[str], list[str]]:
    selected = [dict(row) for row in rows
                if scope == "all" or bool(row["deployable"])]
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_config[str(row["configuration_id"])].append(row)
    if not by_config:
        return [], set(), []
    contexts = {config: {str(row["context_id"]) for row in values}
                for config, values in by_config.items()}
    common = set.intersection(*contexts.values())
    filtered = [row for row in selected if str(row["context_id"]) in common]
    incomplete = sorted(config for config, available in contexts.items() if available != common)
    return filtered, common, incomplete


def _quantile(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _point_intervals(rows: Sequence[Mapping], n_resamples: int, ci: float,
                     seed: int) -> dict[str, dict]:
    by_config: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_config[str(row["configuration_id"])].append(row)
    out = {}
    for config, values in by_config.items():
        array = np.asarray([[float(row["u_now"]), float(row["u_future"]),
                             float(row["cost"])] for row in values], dtype=float)
        rng = np.random.default_rng(_stable_seed(seed, config, len(values)))
        draw = rng.integers(0, len(values), size=(n_resamples, len(values)))
        means = array[draw, :].mean(axis=1)
        now_lo, now_hi = _quantile(means[:, 0], ci)
        future_lo, future_hi = _quantile(means[:, 1], ci)
        cost_lo, cost_hi = _quantile(means[:, 2], ci)
        out[config] = {
            "u_now_lo": now_lo, "u_now_hi": now_hi,
            "u_future_lo": future_lo, "u_future_hi": future_hi,
            "cost_lo": cost_lo, "cost_hi": cost_hi,
        }
    return out


def analyse_frontiers(context_rows: Sequence[Mapping], cfg: Mapping,
                      n_resamples: int, ci: float, seed: int
                      ) -> dict[str, list[dict]]:
    points_out: list[dict] = []
    raw_out: list[dict] = []
    constrained_out: list[dict] = []
    membership_out: list[dict] = []
    hv_boot_out: list[dict] = []
    panel_audit: list[dict] = []
    configured_caps = tuple(sorted({int(x) for x in cfg["budget"]["words"]}))
    epsilon = cfg["analysis"].get("dominance_epsilon", 0.0)
    cost_reference = float(cfg["analysis"].get("hypervolume_cost_max_words",
                                                max(configured_caps)))

    for cell_key, cell_rows in sorted(_analysis_cells(context_rows).items()):
        stack_id, dataset, distribution, metric = cell_key
        for scope in ("deployable", "all"):
            panel, common, incomplete = _balanced_panel(cell_rows, scope)
            if not panel or not common:
                continue
            max_requested = max(int(row["requested_budget"]) for row in panel)
            caps = [cap for cap in configured_caps if cap <= max(max_requested, min(configured_caps))]
            meta_by_config = {}
            aggregates = []
            by_config = defaultdict(list)
            for row in panel:
                by_config[str(row["configuration_id"])].append(row)
                meta_by_config[str(row["configuration_id"])] = row
            for config, values in sorted(by_config.items()):
                meta = meta_by_config[config]
                aggregates.append({
                    **{key: meta[key] for key in meta
                       if key not in {"u_now", "u_future", "cost", "feasibility_cost"}},
                    "scope": scope,
                    "u_now": float(np.mean([float(row["u_now"]) for row in values])),
                    "u_future": float(np.mean([float(row["u_future"]) for row in values])),
                    "cost": float(np.mean([float(row["cost"]) for row in values])),
                    "feasibility_cost": float(max(float(row["feasibility_cost"])
                                                  for row in values)),
                    "n_contexts": len(common),
                    "n_rotations": int(sum(int(row["n_rotations"]) for row in values)),
                })
            intervals = _point_intervals(panel, n_resamples, ci,
                                         _stable_seed(seed, *cell_key, scope))
            sanitized = pf.sanitize_points(
                aggregates,
                regime_keys=REGIME_KEYS,
                config_keys=CONFIG_KEYS,
            )
            marked = pf.mark_nondominated(sanitized.points, epsilon)
            for row in marked:
                row.update(intervals[row["configuration_id"]])
                points_out.append(row)
                raw_out.append(row)
            constrained = pf.constrained_frontiers(
                marked, caps, epsilon, feasibility_key="feasibility_cost")
            constrained_out.extend(constrained)

            boot_observations = [{**row, "scope": scope} for row in panel]
            boot = pf.joint_context_bootstrap(
                boot_observations,
                regime_keys=REGIME_KEYS,
                config_keys=CONFIG_KEYS,
                context_key="context_id",
                budgets=caps,
                n_resamples=n_resamples,
                ci=ci,
                seed=_stable_seed(seed, *cell_key, scope, "membership"),
                epsilon=epsilon,
                cost_reference=cost_reference,
                feasibility_cost_key="feasibility_cost",
            )
            for row in boot.global_membership:
                row.update({"scope": scope, "membership_type": "raw_3d"})
                membership_out.append(row)
            for row in boot.constrained_membership:
                row["scope"] = scope
                membership_out.append({**row, "membership_type": "constrained"})
            for row in boot.hypervolume_2d:
                row.update({"scope": scope, "hypervolume_dimension": "2d"})
                hv_boot_out.append(row)
            for row in boot.hypervolume_3d:
                row.update({"scope": scope, "hypervolume_dimension": "3d"})
                hv_boot_out.append(row)
            panel_audit.append({
                "stack_id": stack_id,
                "dataset": dataset,
                "future_distribution": distribution,
                "metric": metric,
                "scope": scope,
                "n_configurations": len(by_config),
                "n_common_contexts": len(common),
                "configs_with_extra_contexts": len(incomplete),
                "common_context_ids": json.dumps(sorted(common)),
            })
    return {
        "points": points_out,
        "raw_frontier": raw_out,
        "constrained": constrained_out,
        "membership": membership_out,
        "bootstrap_hv": hv_boot_out,
        "panel_audit": panel_audit,
    }


def _sets_for_configs(meta_by_config: Mapping[str, Mapping]) -> dict[str, set[str]]:
    configs = set(meta_by_config)
    legacy_core = {config for config, row in meta_by_config.items()
                   if row["policy"] in {"generic", "conditioned", "reusable"}}
    extractive = {config for config, row in meta_by_config.items()
                  if row["policy_family"] == "extractive"}
    scalar = {config for config, row in meta_by_config.items()
              if row["policy_family"] == "scalar"}
    split = {config for config, row in meta_by_config.items()
             if row["policy_family"] == "split"}
    deployable = {config for config, row in meta_by_config.items() if row["deployable"]}
    oracle = configs - deployable
    legacy_deployable = legacy_core | extractive
    candidates: list[tuple[str, set[str]]] = []
    if legacy_core:
        if extractive:
            candidates.append(("legacy_core", legacy_core))
        candidates.append(("legacy_deployable", legacy_deployable))
        if scalar:
            candidates.append(("legacy_plus_scalar", legacy_deployable | scalar))
        if split:
            candidates.append(("legacy_plus_split", legacy_deployable | split))
    if deployable:
        candidates.append(("all_deployable", deployable))
    if oracle:
        candidates.append(("all_including_oracle", configs))

    # Keep scientifically meaningful labels, but avoid plotting identical sets.
    out: dict[str, set[str]] = {}
    seen: set[frozenset[str]] = set()
    for name, members in candidates:
        frozen = frozenset(members)
        if not frozen or frozen in seen:
            continue
        seen.add(frozen)
        out[name] = members
    return out


def _hv2_values(now: np.ndarray, future: np.ndarray, indices: Iterable[int]) -> float:
    coords = [(float(now[i]), float(future[i])) for i in indices]
    # The implementation is tested in pareto_frontier; using the coordinate
    # kernel avoids manufacturing fake regime fields in every bootstrap draw.
    return pf._hypervolume_2d_coordinates(coords, (0.0, 0.0))


def _hv3_values(now: np.ndarray, future: np.ndarray, cost: np.ndarray,
                indices: Iterable[int], cost_reference: float) -> float:
    coords = []
    for i in indices:
        x, y, c = float(now[i]), float(future[i]), float(cost[i])
        z = max(0.0, min(1.0, (cost_reference - c) / cost_reference))
        if x > 0 and y > 0 and z > 0:
            coords.append((x, y, z))
    return pf._hypervolume_3d_origin(coords)


def analyse_policy_sets(context_rows: Sequence[Mapping], cfg: Mapping,
                        n_resamples: int, ci: float, seed: int
                        ) -> tuple[list[dict], list[dict]]:
    """Bootstrap nested policy-set hypervolume on one paired context panel."""
    rows_out: list[dict] = []
    effects_out: list[dict] = []
    caps = tuple(sorted({int(x) for x in cfg["budget"]["words"]}))
    cost_reference = float(cfg["analysis"].get("hypervolume_cost_max_words", max(caps)))
    zero_tolerance = float(cfg["analysis"].get("hypervolume_zero_tolerance", 1e-12))
    if zero_tolerance < 0 or not math.isfinite(zero_tolerance):
        raise ValueError("hypervolume_zero_tolerance must be finite and non-negative")

    for cell_key, cell_rows in sorted(_analysis_cells(context_rows).items()):
        stack_id, dataset, distribution, metric = cell_key
        panel, common, _ = _balanced_panel(cell_rows, "all")
        if not panel or not common:
            continue
        by_config: dict[str, dict[str, Mapping]] = defaultdict(dict)
        meta_by_config: dict[str, Mapping] = {}
        for row in panel:
            config = str(row["configuration_id"])
            by_config[config][str(row["context_id"])] = row
            meta_by_config[config] = row
        configs = sorted(by_config)
        contexts = sorted(common)
        values = np.asarray([
            [[float(by_config[config][context][objective])
              for objective in ("u_now", "u_future", "cost")]
             for context in contexts]
            for config in configs
        ], dtype=float)
        feasibility_values = np.asarray([
            [float(by_config[config][context].get("feasibility_cost",
                                                  by_config[config][context]["cost"]))
             for context in contexts]
            for config in configs
        ], dtype=float)
        index = {config: i for i, config in enumerate(configs)}
        sets = _sets_for_configs(meta_by_config)
        if not sets:
            continue
        rng = np.random.default_rng(_stable_seed(seed, *cell_key, "policy_sets"))
        draws = rng.integers(0, len(contexts), size=(n_resamples, len(contexts)))
        means = values[:, draws, :].mean(axis=2).transpose(1, 0, 2)
        feasibility_samples = feasibility_values[:, draws].max(axis=2).T
        point = values.mean(axis=1)
        point_feasibility = feasibility_values.max(axis=1)

        hv2_samples: dict[tuple[str, int], np.ndarray] = {}
        hv3_samples: dict[str, np.ndarray] = {}
        for set_name, members in sets.items():
            member_indices = [index[config] for config in sorted(members)]
            hv3 = np.asarray([
                _hv3_values(sample[:, 0], sample[:, 1], sample[:, 2],
                            member_indices, cost_reference)
                for sample in means
            ])
            hv3_samples[set_name] = hv3
            point_hv3 = _hv3_values(point[:, 0], point[:, 1], point[:, 2],
                                    member_indices, cost_reference)
            lo, hi = _quantile(hv3, ci)
            rows_out.append({
                "stack_id": stack_id, "dataset": dataset,
                "future_distribution": distribution, "metric": metric,
                "policy_set": set_name, "constraint_budget": "",
                "dimension": "3d", "point_estimate": point_hv3,
                "bootstrap_mean": float(hv3.mean()), "lo": lo, "hi": hi,
                "ci_level": ci, "n_resamples": n_resamples,
                "n_contexts": len(contexts), "n_configurations": len(member_indices),
                "cost_reference": cost_reference,
            })
            for cap in caps:
                samples = np.asarray([
                    _hv2_values(sample[:, 0], sample[:, 1],
                                [i for i in member_indices
                                 if feasibility_samples[draw_index, i] <= cap])
                    for draw_index, sample in enumerate(means)
                ])
                hv2_samples[(set_name, cap)] = samples
                feasible = [i for i in member_indices if point_feasibility[i] <= cap]
                point_hv = _hv2_values(point[:, 0], point[:, 1], feasible)
                lo, hi = _quantile(samples, ci)
                rows_out.append({
                    "stack_id": stack_id, "dataset": dataset,
                    "future_distribution": distribution, "metric": metric,
                    "policy_set": set_name, "constraint_budget": cap,
                    "dimension": "2d", "point_estimate": point_hv,
                    "bootstrap_mean": float(samples.mean()), "lo": lo, "hi": hi,
                    "ci_level": ci, "n_resamples": n_resamples,
                    "n_contexts": len(contexts), "n_configurations": len(member_indices),
                    "cost_reference": "",
                })

        baseline = "legacy_deployable" if "legacy_deployable" in sets else None
        if baseline:
            baseline_members = [index[c] for c in sorted(sets[baseline])]
            for set_name, members in sets.items():
                if set_name == baseline or not sets[baseline].issubset(members):
                    continue
                for cap in caps:
                    base_samples = hv2_samples[(baseline, cap)]
                    aug_samples = hv2_samples[(set_name, cap)]
                    delta = aug_samples - base_samples
                    if float(np.min(delta)) < -zero_tolerance:
                        raise AssertionError(
                            "nested policy-set hypervolume decreased beyond numerical tolerance")
                    delta[np.abs(delta) <= zero_tolerance] = 0.0
                    remaining = 1.0 - base_samples
                    reduction = np.divide(delta, remaining,
                                          out=np.zeros_like(delta), where=remaining > 1e-12)
                    point_base = next(row["point_estimate"] for row in rows_out
                                      if row["stack_id"] == stack_id
                                      and row["dataset"] == dataset
                                      and row["future_distribution"] == distribution
                                      and row["metric"] == metric
                                      and row["policy_set"] == baseline
                                      and row["dimension"] == "2d"
                                      and row["constraint_budget"] == cap)
                    point_aug = next(row["point_estimate"] for row in rows_out
                                     if row["stack_id"] == stack_id
                                     and row["dataset"] == dataset
                                     and row["future_distribution"] == distribution
                                     and row["metric"] == metric
                                     and row["policy_set"] == set_name
                                     and row["dimension"] == "2d"
                                     and row["constraint_budget"] == cap)
                    delta_lo, delta_hi = _quantile(delta, ci)
                    reduction_lo, reduction_hi = _quantile(reduction, ci)
                    point_delta = float(point_aug - point_base)
                    if point_delta < -zero_tolerance:
                        raise AssertionError(
                            "nested point-estimate hypervolume decreased beyond numerical tolerance")
                    if abs(point_delta) <= zero_tolerance:
                        point_delta = 0.0
                    point_reduction = (point_delta / (1.0 - point_base)
                                       if point_base < 1.0 - 1e-12 else 0.0)
                    effects_out.append({
                        "stack_id": stack_id, "dataset": dataset,
                        "future_distribution": distribution, "metric": metric,
                        "baseline_set": baseline, "augmented_set": set_name,
                        "constraint_budget": cap,
                        "hypervolume_delta": point_delta,
                        "hypervolume_delta_lo": delta_lo,
                        "hypervolume_delta_hi": delta_hi,
                        "dominated_area_reduction": point_reduction,
                        "dominated_area_reduction_lo": reduction_lo,
                        "dominated_area_reduction_hi": reduction_hi,
                        "robust_improvement": delta_lo > zero_tolerance,
                        "zero_tolerance": zero_tolerance,
                        "n_contexts": len(contexts), "n_resamples": n_resamples,
                    })
    return rows_out, effects_out


def analyse_coverage(constrained_rows: Sequence[Mapping], membership_rows: Sequence[Mapping]
                     ) -> list[dict]:
    membership = {}
    for row in membership_rows:
        if row.get("membership_type") != "constrained":
            continue
        key = (row.get("stack_id"), row.get("dataset"), row.get("future_distribution"),
               row.get("metric"), row.get("scope"), row.get("config_id"),
               float(row.get("constraint_budget")))
        membership[key] = row.get("frontier_probability")
    keys = ("stack_id", "dataset", "future_distribution", "metric", "scope",
            "constraint_budget")
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in constrained_rows:
        if row.get("feasible"):
            groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        frontier = [row for row in rows if row.get("nondominated_at_budget")]
        by_eq: dict[str, list[Mapping]] = defaultdict(list)
        for row in frontier:
            by_eq[str(row["equivalence_id"])].append(row)
        families = sorted({str(row["policy_family"]) for row in rows})
        for family in families:
            configs = [row for row in frontier if row["policy_family"] == family]
            groups_any = {row["equivalence_id"] for row in configs}
            exclusive = {
                equivalence for equivalence, members in by_eq.items()
                if {member["policy_family"] for member in members} == {family}
            }
            probs = []
            for row in rows:
                if row["policy_family"] != family:
                    continue
                mkey = (*key[:5], row["config_id"], float(key[5]))
                if mkey in membership:
                    probs.append(float(membership[mkey]))
            out.append({
                **dict(zip(keys, key)),
                "policy_family": family,
                "n_feasible_configurations": sum(row["policy_family"] == family for row in rows),
                "n_frontier_configurations": len(configs),
                "n_frontier_equivalence_points": len(groups_any),
                "n_exclusive_frontier_points": len(exclusive),
                "frontier_coordinate_coverage": (len(groups_any) / len(by_eq) if by_eq else 0.0),
                "mean_bootstrap_frontier_probability": float(np.mean(probs)) if probs else "",
                "total_frontier_equivalence_points": len(by_eq),
            })
    return out


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 2 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    denominator = float(np.sum((a - a.mean()) ** 2))
    return 0.0 if denominator <= 1e-15 else float(np.sum((a-a.mean()) * (b-b.mean())) / denominator)


def analyse_preference_trends(context_rows: Sequence[Mapping], n_resamples: int,
                              ci: float, seed: int) -> list[dict]:
    keys = ("stack_id", "stack_label", "dataset", "future_distribution", "metric",
            "policy_family", "requested_budget")
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in context_rows:
        if row["analysis_eligible"] and row["policy_family"] in {"scalar", "split"}:
            groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        by_share: dict[float, dict[str, Mapping]] = defaultdict(dict)
        pref_by_share = {}
        for row in rows:
            share = _as_float(row.get("future_share"))
            if share is None:
                continue
            by_share[share][str(row["context_id"])] = row
            pref_by_share[share] = row.get("preference_id")
        shares = sorted(by_share)
        if len(shares) < 3:
            continue
        common = set.intersection(*({*by_share[share]} for share in shares))
        if not common:
            continue
        contexts = sorted(common)
        values = np.asarray([
            [[float(by_share[share][context]["u_now"]),
              float(by_share[share][context]["u_future"])] for context in contexts]
            for share in shares
        ], dtype=float)
        point = values.mean(axis=1)
        now = point[:, 0]
        future = point[:, 1]
        rng = np.random.default_rng(_stable_seed(seed, *key, "trend"))
        draws = rng.integers(0, len(contexts), size=(n_resamples, len(contexts)))
        sampled = values[:, draws, :].mean(axis=2).transpose(1, 0, 2)
        slope_now_samples = np.asarray([_slope(shares, sample[:, 0]) for sample in sampled])
        slope_future_samples = np.asarray([_slope(shares, sample[:, 1]) for sample in sampled])
        delta_now_samples = sampled[:, -1, 0] - sampled[:, 0, 0]
        delta_future_samples = sampled[:, -1, 1] - sampled[:, 0, 1]
        now_lo, now_hi = _quantile(slope_now_samples, ci)
        future_lo, future_hi = _quantile(slope_future_samples, ci)
        dn_lo, dn_hi = _quantile(delta_now_samples, ci)
        df_lo, df_hi = _quantile(delta_future_samples, ci)
        joint_sign_probability = float(np.mean(
            (slope_now_samples < 0) & (slope_future_samples > 0)
            & (delta_now_samples < 0) & (delta_future_samples > 0)
        ))
        nonincrease = float(np.mean(np.diff(now) <= 1e-12))
        nondecrease = float(np.mean(np.diff(future) >= -1e-12))
        out.append({
            **dict(zip(keys, key)),
            "n_preferences": len(shares), "n_contexts": len(contexts),
            "future_shares": json.dumps(shares),
            "preference_ids": json.dumps([pref_by_share[share] for share in shares]),
            "u_now_values": json.dumps(list(map(float, now))),
            "u_future_values": json.dumps(list(map(float, future))),
            "slope_u_now": _slope(shares, now),
            "slope_u_now_lo": now_lo, "slope_u_now_hi": now_hi,
            "slope_u_future": _slope(shares, future),
            "slope_u_future_lo": future_lo, "slope_u_future_hi": future_hi,
            "spearman_u_now": _correlation(_rankdata(shares), _rankdata(now)),
            "spearman_u_future": _correlation(_rankdata(shares), _rankdata(future)),
            "endpoint_delta_u_now": float(now[-1] - now[0]),
            "endpoint_delta_u_now_lo": dn_lo, "endpoint_delta_u_now_hi": dn_hi,
            "endpoint_delta_u_future": float(future[-1] - future[0]),
            "endpoint_delta_u_future_lo": df_lo, "endpoint_delta_u_future_hi": df_hi,
            "adjacent_nonincrease_u_now": nonincrease,
            "adjacent_nondecrease_u_future": nondecrease,
            "point_direction_correct": bool(now[-1] <= now[0] and future[-1] >= future[0]),
            "joint_sign_probability": joint_sign_probability,
            # This simultaneous bootstrap event uses slopes through every
            # preference level as well as the endpoints. It is intentionally
            # not described as a joint confidence interval.
            "joint_direction_robust": bool(joint_sign_probability >= ci),
            "ci_level": ci, "n_resamples": n_resamples,
        })
    return out


def _configured_objective_grid(cfg: Mapping, points: Sequence[Mapping]
                               ) -> list[dict]:
    """Return the paired objective-λ/allocation-α treatment grid.

    The pairing is an experimental design choice.  In particular, this helper
    never treats ``alpha=lambda/(1+lambda)`` as an optimization identity: λ
    weights two measured utilities, whereas α reserves a fraction of words.
    """

    raw_grid = list(cfg.get("frontier", {}).get("preferences", []))
    if not raw_grid:
        seen = {}
        for row in points:
            if row.get("policy_family") != "split":
                continue
            infinite = bool(row.get("lambda_infinite"))
            weight = math.inf if infinite else _as_float(row.get("lambda"))
            alpha = _as_float(row.get("future_share"))
            if weight is None or alpha is None:
                continue
            seen[(weight, alpha)] = {
                "preference_id": str(row.get("preference_id") or ""),
                "objective_lambda": weight,
                "allocation_alpha": alpha,
            }
        return sorted(seen.values(), key=lambda row: (
            math.isinf(float(row["objective_lambda"])),
            float(row["objective_lambda"]), float(row["allocation_alpha"])))

    grid = []
    for raw in raw_grid:
        raw_weight = raw.get("lambda")
        if (isinstance(raw_weight, str)
                and raw_weight.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}):
            weight = math.inf
        else:
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid objective lambda in preference grid: {raw_weight!r}") from exc
        alpha = _as_float(raw.get("future_share"))
        if math.isnan(weight) or weight < 0 or alpha is None or not 0 <= alpha <= 1:
            raise ValueError(f"invalid objective/allocation grid entry: {raw!r}")
        grid.append({
            "preference_id": str(raw.get("id") or ""),
            "objective_lambda": weight,
            "allocation_alpha": alpha,
        })
    keys = [(row["objective_lambda"], row["allocation_alpha"]) for row in grid]
    if len(keys) != len(set(keys)):
        raise ValueError("objective/allocation treatment grid contains duplicate pairs")
    return sorted(grid, key=lambda row: (
        math.isinf(float(row["objective_lambda"])),
        float(row["objective_lambda"]), float(row["allocation_alpha"])))


def _row_regime_key(row: Mapping) -> tuple:
    """Recover a regime tuple from internal or serialized bootstrap rows."""
    internal = row.get("_regime_key")
    if internal not in (None, (), []):
        return tuple(internal)
    return tuple(row.get(key) for key in REGIME_KEYS)


def analyse_scalar_selection(points: Sequence[Mapping], constrained_rows: Sequence[Mapping],
                             membership_rows: Sequence[Mapping], cfg: Mapping,
                             *, stability_threshold: float = 0.8) -> list[dict]:
    """Evaluate objective-λ support over the observed structured α arms.

    For every comparable regime and feasible cap, ``pareto_frontier.lambda_argmax``
    selects all structured-allocation configurations maximizing
    ``U_now + lambda*U_future`` (or ``U_future`` at λ=∞).  Separately, the
    function scores the same-budget α arm paired with that λ in the treatment
    grid.  Their difference is the *intended-alpha scalar regret*.

    This analysis deliberately does not mix the free-form scalar family into
    the choice set and does not infer that an allocation share is an objective
    weight.
    """

    if not 0 <= stability_threshold <= 1:
        raise ValueError("stability_threshold must lie in [0,1]")
    structured = [dict(row) for row in points
                  if row.get("scope") == "deployable"
                  and row.get("analysis_eligible")
                  and row.get("policy_family") == "split"]
    grid = _configured_objective_grid(cfg, structured)
    if not structured or not grid:
        return []

    permitted: set[tuple[tuple, float]] = set()
    frontier_state: dict[tuple[tuple, float, str], bool] = {}
    for row in constrained_rows:
        if row.get("scope") != "deployable" or row.get("policy_family") != "split":
            continue
        regime = _row_regime_key(row)
        cap = float(row["constraint_budget"])
        config_id = str(row["config_id"])
        if row.get("feasible"):
            permitted.add((regime, cap))
        frontier_state[(regime, cap, config_id)] = bool(
            row.get("feasible") and row.get("nondominated_at_budget"))

    probability: dict[tuple[tuple, float, str], float] = {}
    for row in membership_rows:
        if (row.get("membership_type") != "constrained"
                or row.get("scope") != "deployable"):
            continue
        value = _as_float(row.get("frontier_probability_given_feasible"))
        if value is None:
            value = _as_float(row.get("frontier_probability"))
        if value is None:
            continue
        probability[(_row_regime_key(row),
                     float(row["constraint_budget"]), str(row["config_id"]))] = value

    caps = sorted({cap for _, cap in permitted})
    if not caps:
        return []
    weights = [float(row["objective_lambda"]) for row in grid]
    tie_tolerance = float(cfg.get("analysis", {}).get("scalar_tie_tolerance", 1e-12))
    selection_points = [
        {**row, "cost": float(row.get("feasibility_cost", row["cost"]))}
        for row in structured
    ]
    selections = pf.lambda_argmax(selection_points, caps, weights,
                                  tie_tolerance=tie_tolerance)
    point_lookup = {(_row_regime_key(row), str(row["config_id"])): row
                    for row in structured}
    grid_lookup = {float(row["objective_lambda"]): row for row in grid}
    out = []
    for selected in selections:
        regime = tuple(selected["_regime_key"])
        cap = float(selected["constraint_budget"])
        if (regime, cap) not in permitted or int(selected["n_feasible"]) == 0:
            continue
        weight = float(selected["lambda"])
        treatment = grid_lookup[weight]
        alpha = float(treatment["allocation_alpha"])
        feasible = [row for row in structured
                    if _row_regime_key(row) == regime
                    and float(row.get("feasibility_cost", row["cost"])) <= cap]
        winners = [point_lookup[(regime, config_id)]
                   for config_id in selected["winner_config_ids"]]
        intended = [row for row in feasible
                    if int(row["requested_budget"]) == int(cap)
                    and _as_float(row.get("future_share")) is not None
                    and abs(float(row["future_share"]) - alpha) <= 1e-12]

        def score(row: Mapping) -> float:
            if math.isinf(weight):
                return float(row["u_future"])
            return float(row["u_now"]) + weight * float(row["u_future"])

        intended_score = max((score(row) for row in intended), default=None)
        intended_best = ([] if intended_score is None else
                         [row for row in intended
                          if abs(score(row) - intended_score) <= tie_tolerance])
        winner_ids = {str(row["config_id"]) for row in winners}
        intended_best_ids = {str(row["config_id"]) for row in intended_best}
        intended_supported = bool(winner_ids & intended_best_ids)
        intended_nondominated = any(
            frontier_state.get((regime, cap, config_id), False)
            for config_id in intended_best_ids)
        intended_probabilities = [
            probability[(regime, cap, config_id)] for config_id in intended_best_ids
            if (regime, cap, config_id) in probability]
        intended_probability = (max(intended_probabilities)
                                if intended_probabilities else None)
        intended_stable = bool(
            intended_supported and intended_nondominated
            and intended_probability is not None
            and intended_probability >= stability_threshold)
        if intended_score is None:
            regret = None
            interpretation = "intended_alpha_arm_unavailable"
        else:
            regret = max(0.0, float(selected["best_score"]) - intended_score)
            if intended_stable:
                interpretation = "objective_supported_and_stable_frontier_level"
            elif intended_supported and intended_nondominated:
                interpretation = "objective_supported_but_frontier_uncertain"
            elif intended_nondominated:
                interpretation = "nondominated_but_not_objective_supported"
            else:
                interpretation = "tradeoff_level_not_on_frontier"

        winner_flags = [frontier_state.get((regime, cap, str(row["config_id"])), False)
                        for row in winners]
        winner_probabilities = [probability.get((regime, cap, str(row["config_id"])))
                                for row in winners]
        meta = winners[0] if winners else feasible[0]
        out.append({
            "stack_id": meta["stack_id"],
            "stack_label": meta.get("stack_label", ""),
            "dataset": meta["dataset"],
            "future_distribution": meta["future_distribution"],
            "metric": meta["metric"],
            "scope": "deployable",
            "constraint_budget": cap,
            "objective_lambda": None if math.isinf(weight) else weight,
            "objective_lambda_infinite": math.isinf(weight),
            "objective_lambda_label": _lambda_label(
                None if math.isinf(weight) else weight, math.isinf(weight)),
            "intended_preference_id": treatment["preference_id"],
            "intended_allocation_alpha": alpha,
            "n_feasible_structured_arms": int(selected["n_feasible"]),
            "best_observed_scalar_score": float(selected["best_score"]),
            "winner_config_ids": json.dumps([str(row["configuration_id"]) for row in winners]),
            "winner_policy_ids": json.dumps([str(row["policy"]) for row in winners]),
            "winner_allocation_alphas": json.dumps([
                _as_float(row.get("future_share")) for row in winners]),
            "winner_nondominated_flags": json.dumps(winner_flags),
            "winner_frontier_probabilities": json.dumps(winner_probabilities),
            "n_winner_ties": len(winners),
            "intended_configuration_ids": json.dumps([
                str(row["configuration_id"]) for row in intended]),
            "intended_best_configuration_ids": json.dumps([
                str(row["configuration_id"]) for row in intended_best]),
            "intended_alpha_scalar_score": intended_score,
            "intended_alpha_scalar_regret": regret,
            "intended_alpha_is_objective_winner": intended_supported,
            "intended_alpha_is_nondominated": intended_nondominated,
            "intended_alpha_frontier_probability": intended_probability,
            "intended_alpha_is_stably_supported": intended_stable,
            "frontier_stability_threshold": stability_threshold,
            "interpretation": interpretation,
        })
    return out


def _stack_components(stack_id: str) -> dict[str, str]:
    """Parse the stable stack id emitted by :func:`_stack_metadata`."""
    out: dict[str, str] = {}
    for component in str(stack_id).split("|"):
        key, separator, value = component.partition("=")
        if separator:
            out[key] = value
    return out


def _semantic_configuration_id(row: Mapping) -> str:
    """Identify the same treatment without using a model/seed-specific id."""
    family = str(row.get("policy_family") or "")
    budget = int(float(row["requested_budget"]))
    if family in {"scalar", "split"}:
        preference = str(row.get("preference_id") or "")
        share = _as_float(row.get("future_share"))
        share_text = "missing" if share is None else f"{share:.12g}"
        treatment = f"preference={preference}|future_share={share_text}"
    else:
        policy = str(row.get("policy") or "").replace("@trim", "")
        treatment = f"policy={policy}"
    return f"family={family}|{treatment}|requested_B={budget}"


def _frontier_member_ids(points: Sequence[Mapping], epsilon) -> set[str]:
    """Recompute a two-utility frontier inside exactly one supplied regime."""
    members: set[str] = set()
    for point in points:
        if not any(
            other["semantic_configuration_id"] != point["semantic_configuration_id"]
            and pf.dominates(other, point, epsilon, include_cost=False)
            for other in points
        ):
            members.add(str(point["semantic_configuration_id"]))
    return members


def analyse_replication_robustness(
    context_rows: Sequence[Mapping], cfg: Mapping, n_resamples: int,
    ci: float, seed: int, *, budget: int = 80,
) -> list[dict]:
    """Summarize the structured endpoints and matched cross-stack fronts.

    Endpoint contrasts pair the present-only and future-only structured
    messages within each context and bootstrap whole contexts.  Cross-stack
    Jaccard is deliberately stricter than comparing already-computed frontier
    labels: for each pair of stacks, we first restrict both to the same
    semantic configurations and common contexts, then recompute nondominance
    *separately* in each stack.  Thus an unshared policy cannot remove a shared
    policy from only one side of the comparison.
    """
    target_distribution = {
        "relation_dossiers": "far",
        "squad_groups": "empirical_questions",
    }
    eligible = [dict(row) for row in context_rows
                if bool(row.get("analysis_eligible"))
                and str(row.get("metric")) == PRIMARY_METRIC
                and str(row.get("dataset")) in target_distribution
                and str(row.get("future_distribution"))
                    == target_distribution[str(row.get("dataset"))]]

    # One paired structured endpoint contrast per stack and corpus.
    endpoint_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in eligible:
        if (row.get("policy_family") == "split"
                and int(float(row.get("requested_budget", -1))) == int(budget)):
            key = (str(row["stack_id"]), str(row.get("stack_label", row["stack_id"])),
                   str(row["dataset"]), str(row["future_distribution"]))
            endpoint_groups[key].append(row)

    endpoint_rows: list[dict] = []
    for key, rows in sorted(endpoint_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        by_share: dict[float, dict[str, list[Mapping]]] = defaultdict(
            lambda: defaultdict(list))
        preference_by_share: dict[float, set[str]] = defaultdict(set)
        for row in rows:
            share = _as_float(row.get("future_share"))
            if share is None:
                continue
            by_share[share][str(row["context_id"])].append(row)
            preference_by_share[share].add(str(row.get("preference_id") or ""))
        shares = sorted(by_share)
        if len(shares) < 2:
            continue
        low_share, high_share = shares[0], shares[-1]
        common = sorted(set(by_share[low_share]) & set(by_share[high_share]))
        if not common:
            continue

        # context x endpoint x objective
        values = np.asarray([
            [[float(np.mean([float(row[objective])
                              for row in by_share[share][context]]))
              for objective in ("u_now", "u_future")]
             for share in (low_share, high_share)]
            for context in common
        ], dtype=float)
        differences = values[:, 1, :] - values[:, 0, :]
        rng = np.random.default_rng(_stable_seed(seed, "replication_endpoint", *key, budget))
        draws = rng.integers(0, len(common), size=(n_resamples, len(common)))
        sampled = differences[draws, :].mean(axis=1)
        now_lo, now_hi = _quantile(sampled[:, 0], ci)
        future_lo, future_hi = _quantile(sampled[:, 1], ci)
        delta = differences.mean(axis=0)
        components = _stack_components(key[0])
        temperature = _as_float(components.get("temperature"), 0.0) or 0.0
        sender_seed = components.get("seed", "none")
        endpoint_rows.append({
            "analysis": "structured_endpoint",
            "stack_id": key[0], "stack_label": key[1],
            "sender_model": components.get("sender", "unknown"),
            "reader_model": components.get("reader", "unknown"),
            "sender_temperature": temperature, "sender_seed": sender_seed,
            "dataset": key[2], "future_distribution": key[3],
            "metric": PRIMARY_METRIC, "policy_family": "split",
            "constraint_budget": int(budget), "n_contexts": len(common),
            "low_preference_id": ",".join(sorted(preference_by_share[low_share])),
            "high_preference_id": ",".join(sorted(preference_by_share[high_share])),
            "low_future_share": low_share, "high_future_share": high_share,
            "u_now_low": float(values[:, 0, 0].mean()),
            "u_now_high": float(values[:, 1, 0].mean()),
            "u_future_low": float(values[:, 0, 1].mean()),
            "u_future_high": float(values[:, 1, 1].mean()),
            "delta_u_now": float(delta[0]),
            "delta_u_now_lo": now_lo, "delta_u_now_hi": now_hi,
            "delta_u_future": float(delta[1]),
            "delta_u_future_lo": future_lo, "delta_u_future_hi": future_hi,
            "point_direction_correct": bool(delta[0] < 0 and delta[1] > 0),
            "ci_direction_supported": bool(now_hi < 0 and future_lo > 0),
            "ci_level": ci, "n_resamples": n_resamples,
        })

    out: list[dict] = list(endpoint_rows)

    # The three temperature-.2 runs are a repeated-generation control, not
    # three additional datasets.  Preserve individuals above and add one
    # transparent range/sign summary rather than pooling their observations.
    seed_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in endpoint_rows:
        if (abs(float(row["sender_temperature"]) - 0.2) <= 1e-12
                and str(row["sender_seed"]) not in {"", "none"}):
            key = (row["sender_model"], row["reader_model"], row["dataset"],
                   row["future_distribution"], row["metric"])
            seed_groups[key].append(row)
    for key, rows in sorted(seed_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        seed_ids = sorted({str(row["sender_seed"]) for row in rows},
                          key=lambda value: (not value.isdigit(), value))
        now = [float(row["delta_u_now"]) for row in rows]
        future = [float(row["delta_u_future"]) for row in rows]
        out.append({
            "analysis": "temperature_seed_summary",
            "stack_id": f"sender={key[0]}|reader={key[1]}|temperature=0.2|seed=summary",
            "stack_label": f"{_short_model(str(key[0]))}, T=0.2 seeds",
            "sender_model": key[0], "reader_model": key[1],
            "sender_temperature": 0.2, "sender_seed": "summary",
            "dataset": key[2], "future_distribution": key[3], "metric": key[4],
            "policy_family": "split", "constraint_budget": int(budget),
            "n_stacks": len(rows), "expected_seed_count": 3,
            "complete_seed_triplet": len(seed_ids) == 3,
            "seed_ids": json.dumps(seed_ids),
            "all_sign_agreement": all(row["point_direction_correct"] for row in rows),
            "all_ci_direction_supported": all(row["ci_direction_supported"] for row in rows),
            "delta_u_now_min": min(now), "delta_u_now_max": max(now),
            "delta_u_future_min": min(future), "delta_u_future_max": max(future),
        })

    # Matched frontier-set agreement.  Stack candidates are those for which a
    # structured endpoint was estimable in the corpus.  All deployable policy
    # families may enter, but only their shared semantic treatments survive.
    epsilon = cfg.get("analysis", {}).get("dominance_epsilon", 0.0)
    endpoint_stacks: dict[str, set[str]] = defaultdict(set)
    for row in endpoint_rows:
        endpoint_stacks[str(row["dataset"])].add(str(row["stack_id"]))
    for dataset, stacks in sorted(endpoint_stacks.items()):
        dist = target_distribution[dataset]
        cell_rows = [row for row in eligible
                     if row["dataset"] == dataset and row["future_distribution"] == dist
                     and bool(row.get("deployable"))]
        by_stack: dict[str, dict[str, dict[str, list[Mapping]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        stack_labels: dict[str, str] = {}
        for row in cell_rows:
            stack = str(row["stack_id"])
            if stack not in stacks:
                continue
            semantic = _semantic_configuration_id(row)
            by_stack[stack][semantic][str(row["context_id"])].append(row)
            stack_labels[stack] = str(row.get("stack_label", stack))

        ordered_stacks = sorted(stacks)
        for left_index, left in enumerate(ordered_stacks):
            for right in ordered_stacks[left_index + 1:]:
                shared = sorted(set(by_stack[left]) & set(by_stack[right]))
                if not shared:
                    continue
                context_sets = [set(by_stack[stack][semantic])
                                for stack in (left, right) for semantic in shared]
                common_contexts = sorted(set.intersection(*context_sets)) if context_sets else []
                if not common_contexts:
                    continue

                aggregates: dict[str, dict[str, dict]] = defaultdict(dict)
                for stack in (left, right):
                    for semantic in shared:
                        source_rows = [row
                                       for context in common_contexts
                                       for row in by_stack[stack][semantic][context]]
                        aggregates[stack][semantic] = {
                            "semantic_configuration_id": semantic,
                            "u_now": float(np.mean([float(row["u_now"])
                                                     for row in source_rows])),
                            "u_future": float(np.mean([float(row["u_future"])
                                                        for row in source_rows])),
                            "cost": float(np.mean([float(row["cost"])
                                                   for row in source_rows])),
                            "feasibility_cost": float(max(
                                float(row.get("feasibility_cost", row["cost"]))
                                for row in source_rows)),
                        }
                feasible_shared = [semantic for semantic in shared
                                   if all(aggregates[stack][semantic]["feasibility_cost"] <= budget
                                          for stack in (left, right))]
                if not feasible_shared:
                    continue
                left_points = [aggregates[left][semantic] for semantic in feasible_shared]
                right_points = [aggregates[right][semantic] for semantic in feasible_shared]
                left_front = _frontier_member_ids(left_points, epsilon)
                right_front = _frontier_member_ids(right_points, epsilon)
                out.append({
                    "analysis": "matched_frontier_jaccard",
                    "stack_id": left, "stack_label": stack_labels.get(left, left),
                    "comparison_stack_id": right,
                    "comparison_stack_label": stack_labels.get(right, right),
                    "dataset": dataset, "future_distribution": dist,
                    "metric": PRIMARY_METRIC, "policy_family": "all_deployable",
                    "constraint_budget": int(budget),
                    "estimate": _jaccard(left_front, right_front),
                    "n_shared_configurations": len(feasible_shared),
                    "n_common_contexts": len(common_contexts),
                    "shared_semantic_configurations": json.dumps(feasible_shared),
                    "left_frontier": json.dumps(sorted(left_front)),
                    "right_frontier": json.dumps(sorted(right_front)),
                    "left_frontier_size": len(left_front),
                    "right_frontier_size": len(right_front),
                    "detail": ("restricted to shared semantic configurations and common contexts; "
                               "nondominance recomputed separately within each stack"),
                })
    return out


def analyse_policy_effects(constrained_rows: Sequence[Mapping]) -> list[dict]:
    keys = ("stack_id", "dataset", "future_distribution", "metric", "scope",
            "constraint_budget")
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in constrained_rows:
        if row.get("scope") == "deployable" and row.get("feasible"):
            groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        legacy = [row for row in rows
                  if row.get("policy_family") in {"legacy", "extractive"}]
        if not legacy:
            continue
        legacy_hv = pf.hypervolume_2d(legacy)
        for row in rows:
            if row.get("policy_family") not in {"scalar", "split"}:
                continue
            dominated_by = [base["configuration_id"] for base in legacy
                            if pf.dominates(base, row, include_cost=False)]
            dominates = [base["configuration_id"] for base in legacy
                         if pf.dominates(row, base, include_cost=False)]
            augmented_hv = pf.hypervolume_2d([*legacy, row])
            gain = augmented_hv - legacy_hv
            if dominated_by:
                classification = "dominated_by_legacy"
            elif gain > 1e-12 and dominates:
                classification = "frontier_shift_and_dominance"
            elif gain > 1e-12:
                classification = "frontier_expansion"
            elif row.get("nondominated_at_budget"):
                classification = "lies_on_existing_point_estimate_frontier"
            else:
                classification = "dominated_by_other_deployable"
            out.append({
                **dict(zip(keys, key)),
                "configuration_id": row["configuration_id"],
                "policy": row["policy"], "policy_family": row["policy_family"],
                "preference_id": row.get("preference_id"),
                "future_share": row.get("future_share"),
                "requested_budget": row["requested_budget"],
                "u_now": row["u_now"], "u_future": row["u_future"], "cost": row["cost"],
                "classification": classification,
                "dominated_by_legacy": json.dumps(dominated_by),
                "dominates_legacy": json.dumps(dominates),
                "incremental_hypervolume": gain,
            })
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def analyse_robustness(constrained_rows: Sequence[Mapping], trends: Sequence[Mapping]
                       ) -> list[dict]:
    out = []
    for row in trends:
        out.append({
            "analysis": "preference_direction",
            "stack_id": row["stack_id"], "dataset": row["dataset"],
            "future_distribution": row["future_distribution"], "metric": row["metric"],
            "policy_family": row["policy_family"],
            "constraint_budget": row["requested_budget"],
            "comparison": "future-share endpoint",
            "estimate": int(bool(row["point_direction_correct"])),
            "detail": ("joint 95% direction supported" if row["joint_direction_robust"]
                       else "point direction only or mixed"),
        })

    groups: dict[tuple, set[str]] = defaultdict(set)
    for row in constrained_rows:
        if (row.get("scope") == "deployable" and row.get("feasible")
                and row.get("nondominated_at_budget")):
            key = (row["stack_id"], row["dataset"], row["future_distribution"],
                   row["metric"], float(row["constraint_budget"]))
            groups[key].add(str(row["configuration_id"]))

    bases = defaultdict(dict)
    for (stack, dataset, dist, metric, cap), members in groups.items():
        bases[(stack, dataset, dist, cap)][metric] = members
    for key, metrics in sorted(bases.items(), key=lambda item: tuple(map(str, item[0]))):
        if PRIMARY_METRIC not in metrics:
            continue
        for metric in ("em", "f1"):
            if metric not in metrics:
                continue
            out.append({
                "analysis": "metric_frontier_jaccard",
                "stack_id": key[0], "dataset": key[1], "future_distribution": key[2],
                "metric": metric, "policy_family": "all_deployable",
                "constraint_budget": key[3],
                "comparison": f"{PRIMARY_METRIC} vs {metric}",
                "estimate": _jaccard(metrics[PRIMARY_METRIC], metrics[metric]),
                "detail": (f"{len(metrics[PRIMARY_METRIC])} vs {len(metrics[metric])} "
                           "nondominated configurations"),
            })

    dist_groups = defaultdict(dict)
    for (stack, dataset, dist, metric, cap), members in groups.items():
        if dataset == "relation_dossiers" and metric == PRIMARY_METRIC:
            dist_groups[(stack, cap)][dist] = members
    for (stack, cap), distributions in sorted(dist_groups.items()):
        if "relation_balanced" not in distributions:
            continue
        for dist, members in sorted(distributions.items()):
            if dist == "relation_balanced":
                continue
            out.append({
                "analysis": "distribution_frontier_jaccard",
                "stack_id": stack, "dataset": "relation_dossiers",
                "future_distribution": dist, "metric": PRIMARY_METRIC,
                "policy_family": "all_deployable", "constraint_budget": cap,
                "comparison": f"relation_balanced vs {dist}",
                "estimate": _jaccard(distributions["relation_balanced"], members),
                "detail": "configuration-set Jaccard; regimes were not pooled",
            })
    return out


def _display_policy(row: Mapping) -> str:
    family = str(row.get("policy_family", ""))
    if family == "scalar":
        return "λ=" + _lambda_label(_as_float(row.get("lambda")),
                                        _as_bool(row.get("lambda_infinite")))
    if family == "split":
        share = _as_float(row.get("future_share"))
        return f"α={share:g}" if share is not None else "α=?"
    labels = {
        "generic": "G", "conditioned": "C", "reusable": "R",
        "oracle": "O", "extractive_generic": "EG",
        "extractive_conditioned": "EC",
    }
    return labels.get(str(row.get("policy")), str(row.get("policy", "?"))[:5])


def _pick_primary_stack(points: Sequence[Mapping]) -> str | None:
    counts: dict[str, tuple[int, int]] = {}
    for stack in {str(row["stack_id"]) for row in points}:
        rows = [row for row in points if row["stack_id"] == stack]
        has_legacy = int(any(row["policy_family"] == "legacy" for row in rows))
        counts[stack] = (has_legacy, len(rows))
    return max(counts, key=lambda item: counts[item]) if counts else None


def _figure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
    })


def _save_figure(fig, output_stem: Path) -> dict:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {"png": png.relative_to(output_stem.parents[1]).as_posix(),
            "pdf": pdf.relative_to(output_stem.parents[1]).as_posix()}


def _frontier_step(ax, rows: Sequence[Mapping], *, color="#222222") -> None:
    unique = sorted({(float(row["u_now"]), float(row["u_future"])) for row in rows})
    if len(unique) < 2:
        return
    xs, ys = zip(*unique)
    # 'pre' draws the exact boundary of the union of attained dominated
    # rectangles.  It is a visual envelope, not an interpolated policy curve.
    ax.step(xs, ys, where="pre", color=color, linewidth=1.15,
            linestyle=(0, (3, 2)), alpha=0.85, zorder=2)


def _plot_points(ax, rows: Sequence[Mapping], *, annotate: bool = True,
                 show_intervals: bool = True) -> None:
    for row in rows:
        family = str(row["policy_family"])
        nondominated = bool(row.get("nondominated_at_budget",
                                    row.get("nondominated", False)))
        color = FAMILY_COLORS.get(family, "#555555")
        marker = FAMILY_MARKERS.get(family, "o")
        if nondominated:
            size = 100 if family == "oracle" else 46
            alpha, face = 0.95, color
        else:
            size = 52 if family == "oracle" else 27
            alpha, face = 0.35, "none"
        ax.scatter(float(row["u_now"]), float(row["u_future"]), s=size,
                   marker=marker, facecolor=face, edgecolor=color,
                   linewidth=1.0, alpha=alpha, zorder=4 if nondominated else 1)
        if show_intervals and nondominated and row.get("u_now_lo") is not None:
            x = float(row["u_now"])
            y = float(row["u_future"])
            ax.errorbar(x, y,
                        xerr=[[x-float(row["u_now_lo"])],
                              [float(row["u_now_hi"])-x]],
                        yerr=[[y-float(row["u_future_lo"])],
                              [float(row["u_future_hi"])-y]],
                        color=color, linewidth=0.65, capsize=1.5, alpha=0.55,
                        zorder=2)
        if annotate and nondominated:
            ax.annotate(_display_policy(row), (float(row["u_now"]), float(row["u_future"])),
                        xytext=(3, 3), textcoords="offset points", fontsize=6.5,
                        color="#222222")


def _fixed_axes(ax) -> None:
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(-0.035, 1.035)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_aspect("equal", adjustable="box")


def _frontier_rows_at(constrained: Sequence[Mapping], stack: str, dataset: str,
                      distribution: str, cap: int, scope: str = "all") -> list[dict]:
    return [dict(row) for row in constrained
            if row["stack_id"] == stack and row["dataset"] == dataset
            and row["future_distribution"] == distribution
            and row["metric"] == PRIMARY_METRIC and row["scope"] == scope
            and float(row["constraint_budget"]) == float(cap) and row.get("feasible")]


def _with_deployable_flags(all_rows: Sequence[Mapping], deployable_rows: Sequence[Mapping]
                           ) -> list[dict]:
    deployable_front = {str(row["configuration_id"]) for row in deployable_rows
                        if row.get("nondominated_at_budget")}
    out = []
    for source in all_rows:
        row = dict(source)
        if row.get("deployable"):
            row["nondominated_at_budget"] = str(row["configuration_id"]) in deployable_front
        out.append(row)
    return out


def _replication_plot_label(row: Mapping) -> str:
    components = _stack_components(str(row.get("stack_id", "")))
    sender = _short_model(components.get("sender", str(row.get("sender_model", "unknown"))))
    reader = _short_model(components.get("reader", str(row.get("reader_model", "unknown"))))
    model = sender if sender == reader else f"{sender} → {reader}"
    temperature = _as_float(components.get("temperature"), 0.0) or 0.0
    sender_seed = components.get("seed", "none")
    if temperature:
        return f"{model}, T={temperature:g}, seed={sender_seed}"
    return f"{model}, deterministic"


def render_figures(analysis: Mapping[str, Sequence[Mapping]], hypervolume: Sequence[Mapping],
                   trends: Sequence[Mapping], output_dir: Path,
                   replication_robustness: Sequence[Mapping] = ()) -> tuple[list[dict], dict]:
    _figure_style()
    figures: list[dict] = []
    semantic: dict = {
        "schema_version": REPORT_SCHEMA,
        "axis_contract": {
            "x": {"name": "U_now", "range": [0, 1], "direction": "higher_is_better"},
            "y": {"name": "U_future", "range": [0, 1], "direction": "higher_is_better"},
            "cost": {"name": "mean_delivered_words", "units": "words",
                     "direction": "lower_is_better"},
        },
        "categorical_interpolation": False,
        "frontier_envelope": "right-angle boundary of dominated rectangles; not interpolation",
        "figures": [],
    }
    points = list(analysis["points"])
    constrained = list(analysis["constrained"])
    stack = _pick_primary_stack(points)
    if stack is None:
        return figures, semantic

    # Figure 1: future-distribution sensitivity at a fixed feasible cap.
    available_dists = {row["future_distribution"] for row in constrained
                       if row["stack_id"] == stack and row["dataset"] == "relation_dossiers"}
    wanted = [name for name in ("relation_balanced", "novel_near", "far", "empirical_cells")
              if name in available_dists]
    if wanted:
        cap = 80 if any(float(row["constraint_budget"]) == 80 for row in constrained) \
            else int(max(float(row["constraint_budget"]) for row in constrained))
        fig, axes = plt.subplots(1, len(wanted), figsize=(3.25*len(wanted), 3.35),
                                 sharex=True, sharey=True, squeeze=False)
        panel_specs = []
        for ax, dist in zip(axes[0], wanted):
            rows = _frontier_rows_at(constrained, stack, "relation_dossiers", dist, cap, "all")
            deploy = _frontier_rows_at(constrained, stack, "relation_dossiers", dist, cap,
                                       "deployable")
            deploy_front = [row for row in deploy if row.get("nondominated_at_budget")]
            _frontier_step(ax, deploy_front)
            _plot_points(ax, _with_deployable_flags(rows, deploy))
            _fixed_axes(ax)
            ax.set_title(DIST_LABELS.get(dist, dist))
            panel_specs.append({"future_distribution": dist, "constraint_budget": cap,
                                "n_points": len(rows),
                                "n_deployable_frontier": len(deploy_front)})
        axes[0, 0].set_ylabel("Future-query utility  $U_{future}$")
        for ax in axes[0]:
            ax.set_xlabel("Present-query utility  $U_{now}$")
        fig.suptitle(f"Empirical frontiers under alternative future-query distributions (C ≤ {cap} words)",
                     y=1.02, fontsize=11)
        present_families = sorted({row["policy_family"] for dist in wanted
                                   for row in _frontier_rows_at(
                                       constrained, stack, "relation_dossiers", dist, cap, "all")})
        handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(family, "o"),
                          color="none", markerfacecolor=FAMILY_COLORS.get(family, "#555"),
                          markeredgecolor=FAMILY_COLORS.get(family, "#555"), markersize=6,
                          label=FAMILY_DISPLAY.get(family, family.title()))
                   for family in present_families]
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                              markeredgecolor="#777", alpha=.45, label="Dominated"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.05),
                   ncol=min(6, len(handles)), frameon=False)
        files = _save_figure(fig, output_dir / "fig1_distribution_frontiers")
        spec = {"id": "fig1_distribution_frontiers", "panels": panel_specs, **files}
        figures.append(spec)
        semantic["figures"].append(spec)

    # Figure 2: no cross-budget connections; each cap is its own feasible set.
    dataset_specs = []
    for dataset, dist in (("squad_groups", "empirical_questions"),
                          ("relation_dossiers", "relation_balanced")):
        if any(row["stack_id"] == stack and row["dataset"] == dataset
               and row["future_distribution"] == dist for row in constrained):
            dataset_specs.append((dataset, dist))
    caps = [cap for cap in (40, 80, 160)
            if any(float(row["constraint_budget"]) == cap for row in constrained)]
    if dataset_specs and caps:
        fig, axes = plt.subplots(len(dataset_specs), len(caps),
                                 figsize=(3.15*len(caps), 3.0*len(dataset_specs)),
                                 sharex=True, sharey=True, squeeze=False)
        panel_specs = []
        for i, (dataset, dist) in enumerate(dataset_specs):
            for j, cap in enumerate(caps):
                ax = axes[i, j]
                rows = _frontier_rows_at(constrained, stack, dataset, dist, cap, "all")
                deploy = _frontier_rows_at(constrained, stack, dataset, dist, cap, "deployable")
                frontier = [row for row in deploy if row.get("nondominated_at_budget")]
                _frontier_step(ax, frontier)
                _plot_points(ax, _with_deployable_flags(rows, deploy),
                             annotate=(len(rows) <= 25), show_intervals=False)
                _fixed_axes(ax)
                ax.set_title(f"C ≤ {cap} words")
                if j == 0:
                    label = "Natural passages" if dataset == "squad_groups" else "Designed dossiers"
                    ax.set_ylabel(label + "\n$U_{future}$")
                if i == len(dataset_specs)-1:
                    ax.set_xlabel("$U_{now}$")
                panel_specs.append({"dataset": dataset, "future_distribution": dist,
                                    "constraint_budget": cap, "n_points": len(rows),
                                    "n_deployable_frontier": len(frontier)})
        fig.suptitle("Constrained frontiers by dataset and communication cap", y=1.01, fontsize=11)
        present_families = sorted({row["policy_family"] for dataset, dist in dataset_specs
                                   for cap in caps for row in _frontier_rows_at(
                                       constrained, stack, dataset, dist, cap, "all")})
        handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(family, "o"),
                          color="none", markerfacecolor=FAMILY_COLORS.get(family, "#555"),
                          markeredgecolor=FAMILY_COLORS.get(family, "#555"), markersize=6,
                          label=FAMILY_DISPLAY.get(family, family.title()))
                   for family in present_families]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.015),
                   ncol=min(6, len(handles)), frameon=False)
        files = _save_figure(fig, output_dir / "fig2_budget_frontiers")
        spec = {"id": "fig2_budget_frontiers", "panels": panel_specs, **files}
        figures.append(spec)
        semantic["figures"].append(spec)

    # Figure 3: raw three-objective projection; cost is encoded continuously.
    raw_panels = []
    for dataset, dist in dataset_specs:
        rows = [dict(row) for row in points if row["stack_id"] == stack
                and row["dataset"] == dataset and row["future_distribution"] == dist
                and row["metric"] == PRIMARY_METRIC and row["scope"] == "deployable"]
        if rows:
            raw_panels.append((dataset, dist, rows))
    if raw_panels:
        fig, axes = plt.subplots(1, len(raw_panels), figsize=(4.0*len(raw_panels), 3.5),
                                 sharex=True, sharey=True, squeeze=False)
        panel_specs = []
        norm = matplotlib.colors.Normalize(vmin=0, vmax=160)
        cmap = plt.get_cmap("viridis")
        for ax, (dataset, dist, rows) in zip(axes[0], raw_panels):
            for row in rows:
                color = cmap(norm(float(row["cost"])))
                ax.scatter(row["u_now"], row["u_future"],
                           s=52 if row["nondominated"] else 25,
                           marker=FAMILY_MARKERS.get(row["policy_family"], "o"),
                           facecolor=color if row["nondominated"] else "none",
                           edgecolor=color, linewidth=1.2 if row["nondominated"] else .7,
                           alpha=.95 if row["nondominated"] else .38)
            _fixed_axes(ax)
            ax.set_xlabel("Present-query utility  $U_{now}$")
            ax.set_title("Natural passages" if dataset == "squad_groups" else "Designed dossiers")
            panel_specs.append({"dataset": dataset, "future_distribution": dist,
                                "n_points": len(rows),
                                "n_raw_3d_frontier": sum(bool(r["nondominated"]) for r in rows)})
        axes[0, 0].set_ylabel("Future-query utility  $U_{future}$")
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=list(axes[0]), label="Mean delivered words (cost C)",
                     fraction=.035, pad=.03)
        fig.suptitle("Raw three-objective observations (outlined = dominated; filled = nondominated)",
                     y=1.02, fontsize=11)
        files = _save_figure(fig, output_dir / "fig3_raw_three_objective")
        spec = {"id": "fig3_raw_three_objective", "panels": panel_specs, **files}
        figures.append(spec)
        semantic["figures"].append(spec)

    # Figure 4: ordered within-family treatment paths.  The two x-axes have
    # distinct meanings: objective weight lambda for scalar, allocation alpha
    # for split.  Lines aid treatment-order reading, not Pareto interpolation.
    relation_trends = [row for row in trends if row["stack_id"] == stack
                       and row["metric"] == PRIMARY_METRIC
                       and row["dataset"] == "relation_dossiers"
                       and row["future_distribution"] == "far"]
    trend_cells = relation_trends or [row for row in trends if row["stack_id"] == stack
                                      and row["metric"] == PRIMARY_METRIC
                                      and row["future_distribution"] == "empirical_questions"]
    if trend_cells:
        cells = sorted(trend_cells, key=lambda row: (
            row["dataset"], row["policy_family"], int(row["requested_budget"])))
        fig, axes = plt.subplots(1, len(cells), figsize=(3.05*len(cells), 3.1),
                                 sharex=True, sharey=True, squeeze=False)
        panel_specs = []
        for ax, row in zip(axes[0], cells):
            shares = json.loads(row["future_shares"])
            now = json.loads(row["u_now_values"])
            future = json.loads(row["u_future_values"])
            ax.plot(shares, now, color="#C00000", marker="o", linewidth=1.4,
                    label="$U_{now}$")
            ax.plot(shares, future, color="#4472C4", marker="s", linewidth=1.4,
                    linestyle="--", label="$U_{future}$")
            ax.set_xlim(-.03, 1.03); ax.set_ylim(-.035, 1.035)
            ax.set_xticks([0, .25, .5, .75, 1])
            if row["policy_family"] == "scalar":
                ax.set_xticklabels(["0", "1/3", "1", "3", "∞"])
                ax.set_xlabel("Stated objective weight λ")
                title = "Free-form objective"
                treatment_axis = "objective_lambda"
            else:
                ax.set_xlabel("Reusable-evidence allocation α")
                title = "Structured allocation"
                treatment_axis = "allocation_alpha"
            ax.set_title(f"{title}, B={int(row['requested_budget'])}")
            panel_specs.append({"dataset": row["dataset"],
                                "future_distribution": row["future_distribution"],
                                "family": row["policy_family"],
                                "treatment_axis": treatment_axis,
                                "requested_budget": row["requested_budget"],
                                "n_preferences": row["n_preferences"]})
        axes[0, 0].set_ylabel("Utility")
        axes[0, 0].legend(frameon=False, loc="best")
        fig.suptitle("Treatment response on designed dossiers (far future queries)", y=1.02,
                     fontsize=11)
        files = _save_figure(fig, output_dir / "fig4_preference_response")
        spec = {"id": "fig4_preference_response", "panels": panel_specs, **files}
        figures.append(spec)
        semantic["figures"].append(spec)

    # Figure 5: set-valued hypervolume with uncertainty, faceted by regime.
    hv_cells = [row for row in hypervolume if row["stack_id"] == stack
                and row["metric"] == PRIMARY_METRIC and row["dimension"] == "2d"
                and row["future_distribution"] in {"empirical_questions", "relation_balanced", "far"}]
    hv_panels = sorted({(row["dataset"], row["future_distribution"]) for row in hv_cells})
    if hv_panels:
        fig, axes = plt.subplots(1, len(hv_panels), figsize=(3.6*len(hv_panels), 3.25),
                                 sharex=True, sharey=True, squeeze=False)
        panel_specs = []
        colors = {"legacy_core": "#9EADCA", "legacy_deployable": "#4472C4",
                  "legacy_plus_scalar": "#ED7D31", "legacy_plus_split": "#C00000",
                  "all_deployable": "#222222", "all_including_oracle": "#7030A0"}
        for ax, panel in zip(axes[0], hv_panels):
            rows = [row for row in hv_cells
                    if (row["dataset"], row["future_distribution"]) == panel]
            for set_name in sorted({row["policy_set"] for row in rows}):
                values = sorted([row for row in rows if row["policy_set"] == set_name],
                                key=lambda row: float(row["constraint_budget"]))
                x = np.asarray([float(row["constraint_budget"]) for row in values])
                y = np.asarray([float(row["point_estimate"]) for row in values])
                lo = np.asarray([float(row["lo"]) for row in values])
                hi = np.asarray([float(row["hi"]) for row in values])
                color = colors.get(set_name, "#777777")
                ax.errorbar(
                    x, y, yerr=np.vstack((np.maximum(0.0, y-lo),
                                          np.maximum(0.0, hi-y))),
                    fmt="o", linestyle="none", color=color, ecolor=color,
                    markersize=4.8, elinewidth=1.0, capsize=2.0,
                    label=set_name.replace("_", " "),
                )
            ax.set_ylim(0, 1.02); ax.set_xlim(15, 165)
            ax.set_xticks([20, 40, 80, 160])
            ax.set_xlabel("Feasible cap B (words)")
            ax.set_title(DIST_LABELS.get(panel[1], panel[1]))
            panel_specs.append({"dataset": panel[0], "future_distribution": panel[1],
                                "policy_sets": sorted({r["policy_set"] for r in rows}),
                                "interpolation": False})
        axes[0, 0].set_ylabel("2D hypervolume (reference 0,0)")
        axes[0, -1].legend(frameon=False, fontsize=6.7, loc="lower right")
        fig.suptitle("Attained utility area and context-bootstrap uncertainty", y=1.02,
                     fontsize=11)
        files = _save_figure(fig, output_dir / "fig5_hypervolume")
        spec = {"id": "fig5_hypervolume", "panels": panel_specs, **files}
        figures.append(spec)
        semantic["figures"].append(spec)

    # Figure 6: paired endpoint effects are shown as a forest plot.  Each
    # corpus is a separate panel and each stack remains a separate estimate;
    # there is no pooled model effect and no line connecting regimes.
    endpoints = [dict(row) for row in replication_robustness
                 if row.get("analysis") == "structured_endpoint"
                 and row.get("metric") == PRIMARY_METRIC
                 and int(float(row.get("constraint_budget", -1))) == 80]
    endpoint_panels = [(dataset, distribution) for dataset, distribution in (
        ("squad_groups", "empirical_questions"), ("relation_dossiers", "far"))
        if any(row.get("dataset") == dataset
               and row.get("future_distribution") == distribution for row in endpoints)]
    if endpoint_panels:
        max_rows = max(sum(row.get("dataset") == dataset
                           and row.get("future_distribution") == distribution
                           for row in endpoints)
                       for dataset, distribution in endpoint_panels)
        fig, axes = plt.subplots(1, len(endpoint_panels),
                                 figsize=(5.0*len(endpoint_panels), max(3.0, .55*max_rows+1.7)),
                                 sharex=True, squeeze=False)
        panel_specs = []
        for ax, (dataset, distribution) in zip(axes[0], endpoint_panels):
            rows = sorted([row for row in endpoints
                           if row.get("dataset") == dataset
                           and row.get("future_distribution") == distribution],
                          key=lambda row: (_replication_plot_label(row), row["stack_id"]))
            positions = np.arange(len(rows), dtype=float)
            for position, row in zip(positions, rows):
                for objective, color, marker, offset in (
                    ("now", "#C00000", "o", -.11),
                    ("future", "#4472C4", "s", .11),
                ):
                    estimate = float(row[f"delta_u_{objective}"])
                    lo = float(row[f"delta_u_{objective}_lo"])
                    hi = float(row[f"delta_u_{objective}_hi"])
                    ax.errorbar(
                        estimate, position + offset,
                        xerr=[[max(0.0, estimate-lo)], [max(0.0, hi-estimate)]],
                        fmt=marker, linestyle="none", color=color, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=.45,
                        markersize=5.2, elinewidth=1.05, capsize=2.0, zorder=3,
                    )
            ax.axvline(0, color="#555555", linewidth=.8, linestyle=(0, (3, 2)), zorder=1)
            ax.set_xlim(-1.03, 1.03)
            ax.set_xticks([-.8, -.4, 0, .4, .8])
            ax.set_yticks(positions)
            ax.set_yticklabels([_replication_plot_label(row) for row in rows])
            ax.invert_yaxis()
            ax.set_xlabel("Endpoint change (future-only − present-only)")
            corpus = "Natural passages" if dataset == "squad_groups" else "Designed dossiers"
            ax.set_title(f"{corpus}\n{DIST_LABELS.get(distribution, distribution)}, B=80")
            panel_specs.append({
                "dataset": dataset, "future_distribution": distribution,
                "metric": PRIMARY_METRIC, "constraint_budget": 80,
                "n_stacks": len(rows),
                "stack_ids": [row["stack_id"] for row in rows],
                "series": ["delta_u_now", "delta_u_future"],
                "interpolation": False, "cross_regime_domination": False,
            })
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", color="#C00000",
                   markerfacecolor="#C00000", label="Δ present utility"),
            Line2D([0], [0], marker="s", linestyle="none", color="#4472C4",
                   markerfacecolor="#4472C4", label="Δ future utility"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.03),
                   ncol=2, frameon=False)
        fig.suptitle("Structured-allocation endpoint effects across model and seed replications",
                     y=1.02, fontsize=11)
        files = _save_figure(fig, output_dir / "fig6_replication_endpoint_deltas")
        spec = {"id": "fig6_replication_endpoint_deltas", "panels": panel_specs,
                "categorical_interpolation": False, "cross_regime_domination": False,
                **files}
        figures.append(spec)
        semantic["figures"].append(spec)

    return figures, semantic


def _fmt(value, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _primary_summary(points: Sequence[Mapping], stack: str | None) -> list[dict]:
    if stack is None:
        return []
    out = []
    for dataset, distribution in (("squad_groups", "empirical_questions"),
                                  ("relation_dossiers", "empirical_cells"),
                                  ("relation_dossiers", "relation_balanced"),
                                  ("relation_dossiers", "far")):
        for budget in STANDARD_CAPS:
            cell = {row["policy"]: row for row in points
                    if row["stack_id"] == stack and row["dataset"] == dataset
                    and row["future_distribution"] == distribution
                    and row["metric"] == PRIMARY_METRIC and row["scope"] == "deployable"
                    and int(row["requested_budget"]) == budget
                    and row["policy"] in {"conditioned", "generic"}}
            if set(cell) != {"conditioned", "generic"}:
                continue
            out.append({
                "dataset": dataset, "future_distribution": distribution,
                "budget": budget,
                "delta_now": cell["conditioned"]["u_now"] - cell["generic"]["u_now"],
                "delta_future": cell["conditioned"]["u_future"] - cell["generic"]["u_future"],
            })
    return out


def _frontier_summary_lines(stack: str | None, hypervolume: Sequence[Mapping],
                            coverage: Sequence[Mapping]) -> list[str]:
    if stack is None:
        return ["- No valid frontier regimes were available."]
    lines = []
    for dataset, distribution in (("squad_groups", "empirical_questions"),
                                  ("relation_dossiers", "empirical_cells"),
                                  ("relation_dossiers", "relation_balanced"),
                                  ("relation_dossiers", "far")):
        cell = [row for row in hypervolume
                if row["stack_id"] == stack and row["dataset"] == dataset
                and row["future_distribution"] == distribution
                and row["metric"] == PRIMARY_METRIC and row["dimension"] == "2d"
                and float(row["constraint_budget"]) == 80]
        if not cell:
            continue
        deploy = next((row for row in cell if row["policy_set"] == "all_deployable"), None)
        if deploy is None:
            deploy = next((row for row in cell if row["policy_set"] == "legacy_deployable"), None)
        oracle = next((row for row in cell if row["policy_set"] == "all_including_oracle"), None)
        coverage_cell = [row for row in coverage
                         if row["stack_id"] == stack and row["dataset"] == dataset
                         and row["future_distribution"] == distribution
                         and row["metric"] == PRIMARY_METRIC and row["scope"] == "deployable"
                         and float(row["constraint_budget"]) == 80]
        n_frontier = max((int(row["total_frontier_equivalence_points"])
                          for row in coverage_cell), default=0)
        family_bits = [f"{row['policy_family']} {float(row['frontier_coordinate_coverage']):.0%}"
                       for row in coverage_cell if int(row["n_frontier_equivalence_points"]) > 0]
        if deploy is None:
            continue
        corpus = "natural passages" if dataset == "squad_groups" else "designed dossiers"
        oracle_text = (f"; oracle-inclusive HV={_fmt(oracle['point_estimate'])}"
                       if oracle is not None else "")
        lines.append(
            f"- {corpus}, {DIST_LABELS.get(distribution, distribution)}, C≤80: deployable "
            f"HV={_fmt(deploy['point_estimate'])} [{_fmt(deploy['lo'])}, {_fmt(deploy['hi'])}]"
            f"{oracle_text}; {n_frontier} distinct deployable frontier coordinates"
            + (f" ({', '.join(family_bits)})" if family_bits else "") + "."
        )
    return lines or ["- No matched C≤80 hypervolume summaries were available."]


def _primary_distribution(cfg: Mapping, dataset: str) -> str:
    configured = cfg.get("analysis", {}).get("primary_future_distributions", {})
    fallback = "empirical_questions" if dataset == "squad_groups" else "far"
    return str(configured.get(dataset, fallback))


def _is_primary_distribution(row: Mapping, cfg: Mapping) -> bool:
    return str(row.get("future_distribution")) == _primary_distribution(
        cfg, str(row.get("dataset")))


def _report_verdict(inventory: Sequence[Mapping], trends: Sequence[Mapping],
                    effects: Sequence[Mapping], scalar_selection: Sequence[Mapping],
                    cfg: Mapping) -> tuple[str, list[str]]:
    complete_new = [row for row in inventory
                    if row["kind"] == "experiment11" and row["complete"] and row["selected"]]
    primary_trends = [row for row in trends if row["metric"] == PRIMARY_METRIC
                      and row["policy_family"] == "split"
                      and _is_primary_distribution(row, cfg)]
    robust_trends = [row for row in primary_trends if row["joint_direction_robust"]]
    scalar_trends = [row for row in trends if row["metric"] == PRIMARY_METRIC
                     and row["policy_family"] == "scalar"
                     and _is_primary_distribution(row, cfg)]
    robust_scalar_trends = [row for row in scalar_trends if row["joint_direction_robust"]]
    structured_effects = [row for row in effects
                          if row["augmented_set"] == "legacy_plus_split"
                          and row["metric"] == PRIMARY_METRIC
                          and _is_primary_distribution(row, cfg)
                          and row["constraint_budget"] in {40, 80, 160}]
    robust_effects = [row for row in structured_effects if row["robust_improvement"]]
    sensitivity_eligible = [
        row for row in robust_effects
        if cfg.get("analysis", {}).get("sensitivity_future_distributions", {})
        .get(row["dataset"])
    ]
    sensitivity_qualified = []
    for primary in robust_effects:
        same = [row for row in effects
                if row["augmented_set"] == "legacy_plus_split"
                and row["stack_id"] == primary["stack_id"]
                and row["dataset"] == primary["dataset"]
                and row["constraint_budget"] == primary["constraint_budget"]
                and row["robust_improvement"]]
        designated_metrics = tuple(cfg.get("analysis", {}).get(
            "sensitivity_metrics", ["em", "f1"]))
        designated_distributions = tuple(cfg.get("analysis", {}).get(
            "sensitivity_future_distributions", {}).get(primary["dataset"], []))
        metric_support = all(any(
            row["future_distribution"] == primary["future_distribution"]
            and row["metric"] == metric for row in same)
            for metric in designated_metrics)
        distribution_support = bool(designated_distributions) and all(any(
            row["metric"] == PRIMARY_METRIC
            and row["future_distribution"] == distribution for row in same)
            for distribution in designated_distributions)
        if metric_support and distribution_support:
            sensitivity_qualified.append(primary)
    facts = [
        f"{len(complete_new)} completed Experiment 11 dataset/replication cells were available.",
        f"{len(robust_trends)}/{len(primary_trends)} primary structured allocation-α trends pass "
        "the simultaneous bootstrap sign-stability threshold (present slope down, future slope "
        "up, with matching endpoints).",
        f"{len(robust_scalar_trends)}/{len(scalar_trends)} free-form objective-λ trends pass the "
        "same directional threshold; this is reported separately from allocation control.",
        f"{len(robust_effects)}/{len(structured_effects)} declared-primary structured policy-set "
        "additions have a paired ΔHV lower bound above numerical zero.",
        f"{len(sensitivity_qualified)}/{len(sensitivity_eligible)} primary improvements with a "
        "declared distribution-sensitivity grid also survive every designated metric and "
        "query-distribution cell.",
    ]
    selection_rows = [row for row in scalar_selection
                      if row["metric"] == PRIMARY_METRIC
                      and _is_primary_distribution(row, cfg)
                      and row.get("intended_alpha_scalar_regret") is not None]
    stable_selection = [row for row in selection_rows
                        if row.get("intended_alpha_is_stably_supported")]
    objective_winners = [row for row in selection_rows
                         if row.get("intended_alpha_is_objective_winner")]
    facts.append(
        f"Across all completed stacks, the design-mapped α arm is an empirical λ-objective winner in "
        f"{len(objective_winners)}/{len(selection_rows)} matched selections; "
        f"{len(stable_selection)}/{len(selection_rows)} are also nondominated with bootstrap "
        "frontier probability at least 0.80."
    )
    if not complete_new:
        verdict = (
            "The existing Experiment 10 data support a specialization cost, but the new causal "
            "claims about objective-weight and allocation control remain untested in "
            "the confirmatory report because no complete Experiment 11 matrix is yet available."
        )
    elif primary_trends and len(robust_trends) == len(primary_trends):
        if sensitivity_qualified:
            verdict = (
                "The structured allocation-α sweep traces the predicted present–future trade-off, "
                "and the allocation family expands the attained utility area in at least one "
                "declared primary regime with metric and query-distribution sensitivity support. "
                "This supports a setting-specific outward frontier shift; only the mapped levels "
                "identified as both nondominated and bootstrap-stable are treated as stable "
                "frontier levels. The result is distribution-conditional rather than a universal "
                "frontier."
            )
        elif robust_effects:
            verdict = (
                "The structured allocation-α sweep traces the predicted present–future trade-off, "
                "and its policy set increases attained hypervolume in at least one designated "
                "primary cell. That outward shift is not consistent across every prespecified "
                "metric and query-distribution sensitivity, so it is reported as setting-specific "
                "rather than robust policy dominance."
            )
        else:
            verdict = (
                "The structured allocation-α sweep traces the predicted specialization–reusability "
                "trade-off, but no structured frontier expansion has a positive lower confidence "
                "bound. Allocation control is supported; genuine policy dominance is not."
            )
    else:
        verdict = (
            "The structured allocation-α sweep does not move both utilities consistently across "
            "the prespecified primary regimes. Allocation share therefore does not provide robust "
            "trade-off control, even where individual endpoint contrasts point the right way; this "
            "does not by itself falsify the distinct objective-λ formulation."
        )
    return verdict, facts


def _inventory_markdown(inventory: Sequence[Mapping]) -> str:
    lines = ["| Study cell | Dataset | Contexts | Status |",
             "|---|---:|---:|---|"]
    for row in inventory:
        if row["kind"] == "experiment10":
            replication = str(row.get("replication", "legacy"))
            name = ("Experiment 10" if replication == "legacy"
                    else f"{replication} (Experiment 10 controls)")
        else:
            name = str(row["replication"])
        status = str(row["status"])
        if row["selected"] and row["complete"]:
            status = "included"
        elif row["selected"]:
            status = "included as provisional — " + status
        lines.append(f"| {name} | {row['dataset']} | {row['contexts']} | {status} |")
    return "\n".join(lines)


def _replication_report_lines(rows: Sequence[Mapping]) -> tuple[list[str], list[str]]:
    """Return paper-readable individual endpoint and aggregate robustness lines."""
    endpoints = sorted(
        [row for row in rows if row.get("analysis") == "structured_endpoint"],
        key=lambda row: (str(row.get("dataset")), _replication_plot_label(row)),
    )
    individual = []
    for row in endpoints:
        corpus = ("Natural passages" if row["dataset"] == "squad_groups"
                  else "Designed dossiers")
        individual.append(
            f"- {corpus}, {_replication_plot_label(row)}: future-only minus present-only "
            f"ΔU_now={_fmt(row['delta_u_now'])} "
            f"[{_fmt(row['delta_u_now_lo'])}, {_fmt(row['delta_u_now_hi'])}], "
            f"ΔU_future={_fmt(row['delta_u_future'])} "
            f"[{_fmt(row['delta_u_future_lo'])}, {_fmt(row['delta_u_future_hi'])}] "
            f"(B=80; n={int(row['n_contexts'])} contexts)."
        )
    if not individual:
        individual = ["- No complete structured endpoint replication is available at B=80."]

    aggregate = []
    for row in rows:
        if row.get("analysis") != "temperature_seed_summary":
            continue
        aggregate.append(
            f"- Temperature-0.2 seed sweep ({int(row['n_stacks'])}/"
            f"{int(row['expected_seed_count'])} runs): expected-sign agreement="
            f"{'yes' if row['all_sign_agreement'] else 'no'}; ΔU_now range "
            f"[{_fmt(row['delta_u_now_min'])}, {_fmt(row['delta_u_now_max'])}] and "
            f"ΔU_future range [{_fmt(row['delta_u_future_min'])}, "
            f"{_fmt(row['delta_u_future_max'])}]. All individual paired intervals support both "
            f"directions={'yes' if row['all_ci_direction_supported'] else 'no'}."
        )

    jaccard_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("analysis") == "matched_frontier_jaccard":
            estimate = _as_float(row.get("estimate"))
            if estimate is not None:
                jaccard_by_dataset[str(row["dataset"])].append(estimate)
    for dataset, values in sorted(jaccard_by_dataset.items()):
        corpus = "natural passages" if dataset == "squad_groups" else "designed dossiers"
        aggregate.append(
            f"- Matched cross-stack frontier overlap on {corpus}: median Jaccard "
            f"{_fmt(float(np.median(values)))} (range {_fmt(min(values))}–{_fmt(max(values))}; "
            f"{len(values)} stack pair{'s' if len(values) != 1 else ''}). Each pair was first "
            "restricted to shared semantic configurations and common contexts, then nondominance "
            "was recomputed separately in each stack."
        )
    if not aggregate:
        aggregate = ["- Seed-range and matched-frontier summaries require at least two complete stacks."]
    return individual, aggregate


def _baseline_report_lines(rows: Sequence[Mapping]) -> list[str]:
    selected = [row for row in rows if row.get("metric") == PRIMARY_METRIC
                and row.get("baseline") in {"direct_context", "closed_book"}]
    by_cell: dict[tuple[str, str], dict[str, Mapping]] = defaultdict(dict)
    for row in selected:
        by_cell[(str(row["stack_id"]), str(row["dataset"]))][str(row["baseline"])] = row
    lines = []
    for (_, dataset), baselines in sorted(by_cell.items(),
                                          key=lambda item: tuple(map(str, item[0]))):
        if not {"direct_context", "closed_book"}.issubset(baselines):
            continue
        direct = baselines["direct_context"]
        closed = baselines["closed_book"]
        corpus = "natural passages" if dataset == "squad_groups" else "designed dossiers"
        lines.append(
            f"- {direct['stack_label']}, {corpus}: direct-context accuracy "
            f"{_fmt(direct.get('mean'))} [{_fmt(direct.get('lo'))}, "
            f"{_fmt(direct.get('hi'))}]; closed-book accuracy "
            f"{_fmt(closed.get('mean'))} [{_fmt(closed.get('lo'))}, "
            f"{_fmt(closed.get('hi'))}]."
        )
    return lines or ["- No complete direct-context/closed-book baseline pair was available."]


def write_report(output_root: Path, inventory: Sequence[Mapping],
                 analysis: Mapping[str, Sequence[Mapping]], hypervolume: Sequence[Mapping],
                 effects: Sequence[Mapping], coverage: Sequence[Mapping],
                 trends: Sequence[Mapping], scalar_selection: Sequence[Mapping],
                 robustness: Sequence[Mapping],
                 replication_robustness: Sequence[Mapping], baseline_rows: Sequence[Mapping],
                 figures: Sequence[Mapping],
                 cfg: Mapping, n_resamples: int,
                 ci: float, include_partial: bool) -> Path:
    points = list(analysis["points"])
    membership = list(analysis["membership"])
    stack = _pick_primary_stack(points)
    primary = _primary_summary(points, stack)
    frontier_lines = _frontier_summary_lines(stack, hypervolume, coverage)
    verdict, verdict_facts = _report_verdict(
        inventory, trends, effects, scalar_selection, cfg)
    title_stamp = " — PROVISIONAL (partial runs included)" if include_partial else ""
    replication_lines, replication_summary_lines = _replication_report_lines(
        replication_robustness)
    baseline_lines = _baseline_report_lines(baseline_rows)
    primary_structured = [row for row in trends
                          if row["metric"] == PRIMARY_METRIC
                          and row["policy_family"] == "split"
                          and _is_primary_distribution(row, cfg)]
    primary_scalar = [row for row in trends
                      if row["metric"] == PRIMARY_METRIC
                      and row["policy_family"] == "scalar"
                      and _is_primary_distribution(row, cfg)]
    if (primary_structured
            and all(row["joint_direction_robust"] for row in primary_structured)
            and primary_scalar
            and not any(row["joint_direction_robust"] for row in primary_scalar)):
        strongest_claim = (
            "The strongest supported claim is that an explicit evidence-allocation quota α "
            "reproducibly controls the present–future trade-off in the declared future-query "
            "regimes, whereas a numerical λ instruction did not do so in its dedicated probe. "
            "Any outward frontier shift remains "
            "setting-specific and is claimed only where paired incremental hypervolume excludes "
            "the numerical zero tolerance."
        )
    else:
        strongest_claim = (
            "The strongest claim licensed by the current data is deliberately narrower than "
            "‘structured is better’: bounded communication exhibits a present–future trade-off, "
            "and outward frontier shifts require a positive paired incremental-hypervolume bound."
        )

    tradeoff_lines = []
    for row in primary:
        if row["budget"] not in {20, 80, 160}:
            continue
        corpus = "natural" if row["dataset"] == "squad_groups" else "designed"
        dist = DIST_LABELS.get(row["future_distribution"], row["future_distribution"])
        tradeoff_lines.append(
            f"- {corpus.capitalize()} corpus, {dist}, B={row['budget']}: conditioning changes "
            f"present utility by {row['delta_now']:+.3f} and future utility by "
            f"{row['delta_future']:+.3f} versus generic."
        )
    if not tradeoff_lines:
        tradeoff_lines = ["- No matched conditioned-versus-generic cells were available."]

    structured_trend_lines = []
    for row in trends:
        if row["metric"] != PRIMARY_METRIC or row["policy_family"] != "split":
            continue
        if stack is not None and row["stack_id"] != stack:
            continue
        if row["future_distribution"] not in {"relation_balanced", "empirical_questions", "far"}:
            continue
        structured_trend_lines.append(
            f"- {row['dataset']} / {DIST_LABELS.get(row['future_distribution'], row['future_distribution'])} "
            f"at B={int(row['requested_budget'])}: α=1 minus α=0 gives ΔU_now="
            f"{_fmt(row['endpoint_delta_u_now'])} [{_fmt(row['endpoint_delta_u_now_lo'])}, "
            f"{_fmt(row['endpoint_delta_u_now_hi'])}], ΔU_future="
            f"{_fmt(row['endpoint_delta_u_future'])} [{_fmt(row['endpoint_delta_u_future_lo'])}, "
            f"{_fmt(row['endpoint_delta_u_future_hi'])}]; "
            f"simultaneous sign stability={_fmt(row['joint_sign_probability'])}; "
            f"{'direction stable' if row['joint_direction_robust'] else 'direction not stable'}."
        )
    if not structured_trend_lines:
        structured_trend_lines = [
            "- Structured allocation-α endpoint estimates are pending a complete matrix."]

    scalar_trend_lines = []
    for row in trends:
        if row["metric"] != PRIMARY_METRIC or row["policy_family"] != "scalar":
            continue
        if stack is not None and row["stack_id"] != stack:
            continue
        if row["future_distribution"] not in {"relation_balanced", "empirical_questions", "far"}:
            continue
        scalar_trend_lines.append(
            f"- {row['dataset']} / "
            f"{DIST_LABELS.get(row['future_distribution'], row['future_distribution'])} "
            f"at B={int(row['requested_budget'])}: stated λ=∞ minus λ=0 gives ΔU_now="
            f"{_fmt(row['endpoint_delta_u_now'])} [{_fmt(row['endpoint_delta_u_now_lo'])}, "
            f"{_fmt(row['endpoint_delta_u_now_hi'])}], ΔU_future="
            f"{_fmt(row['endpoint_delta_u_future'])} [{_fmt(row['endpoint_delta_u_future_lo'])}, "
            f"{_fmt(row['endpoint_delta_u_future_hi'])}]; "
            f"simultaneous sign stability={_fmt(row['joint_sign_probability'])}; "
            f"{'direction stable' if row['joint_direction_robust'] else 'direction not stable'}."
        )
    if not scalar_trend_lines:
        scalar_trend_lines = [
            "- No complete free-form objective-λ sweep is available in the primary stack."]

    selected_scalar = [row for row in scalar_selection
                       if row["metric"] == PRIMARY_METRIC
                       and (stack is None or row["stack_id"] == stack)
                       and _is_primary_distribution(row, cfg)
                       and row.get("intended_alpha_scalar_regret") is not None]
    if selected_scalar:
        winner_count = sum(bool(row["intended_alpha_is_objective_winner"])
                           for row in selected_scalar)
        stable_count = sum(bool(row["intended_alpha_is_stably_supported"])
                           for row in selected_scalar)
        regrets = [float(row["intended_alpha_scalar_regret"])
                   for row in selected_scalar]
        selection_lines = [
            f"- Across {len(selected_scalar)} primary metric × distribution × cap × λ "
            f"selections, the design-mapped α arm is a scalar-objective winner in "
            f"{winner_count}; median intended-α scalar regret is "
            f"{_fmt(float(np.median(regrets)))} (maximum {_fmt(max(regrets))}).",
            f"- {stable_count}/{len(selected_scalar)} mapped levels are simultaneously "
            "λ-objective winners, point-estimate nondominated, and bootstrap-stable "
            "(frontier probability ≥0.80). Other α levels are described only as tracing "
            "a trade-off, not as lying on a stable frontier.",
            "- Objective maximization is computed over every feasible structured arm with C≤B, "
            "including cheaper budget rungs; all exact winner ties are retained.",
        ]
    else:
        selection_lines = [
            "- Objective-λ selection over structured α arms is pending a complete structured "
            "matrix."]

    effect_lines = []
    primary_effect_rows = [
        row for row in effects
        if row["metric"] == PRIMARY_METRIC
        and row["augmented_set"] in {"legacy_plus_split", "legacy_plus_scalar"}
        and _is_primary_distribution(row, cfg)
        and float(row["hypervolume_delta"]) > 0
    ]
    stress_rows = [
        row for row in effects
        if row["metric"] == PRIMARY_METRIC
        and row["augmented_set"] == "legacy_plus_split"
        and row["dataset"] == "relation_dossiers"
        and row["future_distribution"] == "relation_balanced"
        and int(row["constraint_budget"]) == 80
    ]
    relevant_effects = sorted(
        [*primary_effect_rows, *stress_rows],
        key=lambda row: (0 if _is_primary_distribution(row, cfg) else 1,
                         str(row["dataset"]), str(row["stack_id"]),
                         str(row["augmented_set"]), int(row["constraint_budget"])),
    )
    for row in relevant_effects[:12]:
        stack_bits = _stack_components(str(row["stack_id"]))
        model_label = _short_model(stack_bits.get("sender", "unknown"))
        status = ("robust in this cell" if row["robust_improvement"]
                  else "not robust in this cell")
        estimand = ("primary" if _is_primary_distribution(row, cfg)
                    else "prompt–estimand stress test")
        effect_lines.append(
            f"- {model_label}, {row['augmented_set'].replace('_', ' ')}, {row['dataset']} / "
            f"{DIST_LABELS.get(row['future_distribution'], row['future_distribution'])}, "
            f"C≤{int(row['constraint_budget'])}: ΔHV={_fmt(row['hypervolume_delta'])} "
            f"[{_fmt(row['hypervolume_delta_lo'])}, {_fmt(row['hypervolume_delta_hi'])}], "
            f"dominated-area reduction={_fmt(row['dominated_area_reduction'])} "
            f"({estimand}; {status})."
        )
    if not effect_lines:
        effect_lines = ["- No completed new family is available for a paired hypervolume comparison."]

    constrained_membership = [row for row in membership
                              if row.get("membership_type") == "constrained"
                              and row.get("scope") == "deployable"
                              and row.get("metric") == PRIMARY_METRIC
                              and row.get("point_estimate_nondominated")]
    fragile = [row for row in constrained_membership
               if _as_float(row.get("frontier_probability"), 0.0) < 0.8]
    metric_jaccard = [_as_float(row.get("estimate")) for row in robustness
                      if row.get("analysis") == "metric_frontier_jaccard"]
    metric_jaccard = [value for value in metric_jaccard if value is not None]
    dist_jaccard = [_as_float(row.get("estimate")) for row in robustness
                    if row.get("analysis") == "distribution_frontier_jaccard"]
    dist_jaccard = [value for value in dist_jaccard if value is not None]

    fig_sections = []
    explanations = {
        "fig1_distribution_frontiers": (
            "Each panel changes only the assumed distribution of later questions. Filled colored "
            "markers are nondominated deployable observations; pale/open markers are dominated; "
            "the purple star is the non-deployable query-aware oracle. Error bars are marginal "
            "context-bootstrap intervals (paired resampling is used for differences and "
            "frontier membership). The right-angle envelope is the boundary of the area "
            "dominated by attained points, not an interpolated policy."
        ),
        "fig2_budget_frontiers": (
            "Every panel recomputes the feasible set C≤B, so a cheaper observation remains eligible "
            "at a larger cap. No line joins categorical policies or crosses budget panels. This "
            "reveals nonmonotonic budget effects instead of assuming that more words always help."
        ),
        "fig3_raw_three_objective": (
            "This is the direct projection of max(U_now,U_future,−C). Color is mean delivered words; "
            "filled markers are nondominated in all three objectives and outlined markers are "
            "dominated. A cheap point can remain on this raw frontier even if it is below a high-cap "
            "two-utility frontier."
        ),
        "fig4_preference_response": (
            "This figure contains two distinct within-family interventions. Free-form panels use "
            "the stated utility-objective weight λ; structured panels use the reserved reusable-"
            "evidence share α. Lines only guide the eye through each ordered treatment grid and "
            "are not Pareto interpolation. The paired grid α∈{0,.25,.5,.75,1} was chosen using "
            "α=λ/(1+λ) for coverage, but an allocation share is not mathematically equivalent "
            "to an objective weight."
        ),
        "fig5_hypervolume": (
            "Hypervolume is the exact union area dominated by observed feasible utility pairs, using "
            "reference (0,0). Caps are discrete markers with context-bootstrap intervals; no line "
            "interpolates between budgets. A higher marker for a nested policy set means additional "
            "attained area. Interval overlap is not a test of the paired "
            "difference; robustness is decided from the paired ΔHV interval reported separately."
        ),
        "fig6_replication_endpoint_deltas": (
            "Each marker is the within-context endpoint change from the present-only structured "
            "allocation to the future-only allocation at B=80; whiskers are paired context-bootstrap "
            "intervals. The predicted trade-off is a negative red ΔU_now and positive blue "
            "ΔU_future. Corpora are faceted, stacks and seeds are not pooled, and no line or "
            "cross-regime dominance comparison is drawn."
        ),
    }
    for figure in figures:
        fig_id = figure["id"]
        fig_sections.append(
            f"### {fig_id.replace('_', ' ').title()}\n\n"
            f"![{fig_id}](figures/{Path(figure['png']).name})\n\n"
            f"{explanations.get(fig_id, '')} "
            f"[Vector PDF](figures/{Path(figure['pdf']).name})."
        )
    if not fig_sections:
        fig_sections = ["No valid analysis panels were available to plot."]

    report = fr"""# Bounded Agent Communication: Empirical Frontier Study{title_stamp}

## Paper claim

{verdict}

""" + "\n".join(f"- {fact}" for fact in verdict_facts) + fr"""

{strongest_claim}

## Formal target

We preserve the three-objective formulation

\[
\max_\pi \left(U_{{\mathrm{{now}}}}, U_{{\mathrm{{future}}}}, -C\right)
\]

and its scalarized constrained form

\[
\max_\pi \left[U_{{\mathrm{{now}}}} + \lambda U_{{\mathrm{{future}}}}\right]
\quad\text{{subject to}}\quad C\le B.
\]

Utility is answer correctness in [0,1] (primary: independent binary judge; EM and token F1 are
robustness metrics). Cost C is the **mean number of delivered words**, not the requested cap. Hard
feasibility additionally requires the maximum delivered message length to be at most B. Higher
utility is better; lower cost is better in every figure and computation.

## Design in plain language

- **Natural passages:** 24 real SQuAD paragraphs, each paired with four human-written questions.
  Example: the Super Bowl passage separately asks who received Newton’s 45-yard pass, who missed a
  field goal, who intercepted a later pass, and who recovered a fumble.
- **Designed dossiers:** 16 fictional, leakage-screened, four-aspect dossiers with 16 questions each.
  Example: the Vellunar canal dossier asks an incorporation-date question, its paraphrase, a new
  question about the same company, a related-topic question, and questions about orthogonal aspects
  such as facilities or finance. Their fictional facts make closed-book leakage measurable rather
  than assumed; the exact model-specific baselines are reported below.
- **Budgets:** the source experiment uses 20/40/80/160 delivered-word caps; structured allocation is
  prespecified at 40/80/160 because a 20-word message cannot support two meaningful labeled sections.
- **Two distinct treatment sweeps:** the free-form writer receives objective weight
  λ∈{{0, 1/3, 1, 3, ∞}}. The structured writer instead receives a reusable-evidence allocation
  share α∈{{0,.25,.5,.75,1}}. We paired the grids by the design convention α=λ/(1+λ) to span
  comparable endpoints and interior levels; α is a word-allocation control and is **not
  mathematically equivalent** to the utility weight λ.
- **Future distributions:** relation categories are averaged first and then weighted. This makes a
  relation-balanced target genuinely 25% per category rather than letting the 12 orthogonal cells
  silently outweigh each one-cell category. Near, graded-orthogonality, far, empirical-cell, and
  paraphrase-sanity regimes are never allowed to dominate one another.
- **Primary dossier estimand:** the structured reusable section was instructed not to repeat the
  current answer. A paraphrase asks for that same answer, so the far/orthogonal distribution is the
  primary allocation test. This interpretation was declared during internal design audit after the
  mismatch was identified; the far distribution itself was already in the analysis grid, and every
  alternative remains visible. Relation-balanced results are a prompt–estimand stress test, not
  evidence for a distribution-invariant frontier.

## Run status

{_inventory_markdown(inventory)}

Incomplete/smoke result directories are audited but excluded by default. Run this renderer again
after any matrix completes; discovery and all summaries update without editing the report code.
Bootstrap: {n_resamples:,} paired context resamples, {int(ci*100)}% percentile intervals.

### Capacity and leakage controls

""" + "\n".join(baseline_lines) + """

## Main numerical results

### Existing specialization contrast

""" + "\n".join(tradeoff_lines) + """

### Structured allocation-share response

""" + "\n".join(structured_trend_lines) + """

### Objective-λ selection over structured allocation arms

""" + "\n".join(selection_lines) + """

### Free-form scalar-objective response

""" + "\n".join(scalar_trend_lines) + """

### Model and sampling-seed replication at B=80

""" + "\n".join(replication_lines) + """

""" + "\n".join(replication_summary_lines) + """

### Does a new policy shift the frontier?

""" + "\n".join(effect_lines) + fr"""

### Quantitative frontier summary

""" + "\n".join(frontier_lines) + fr"""

## How the frontier is computed

1. For each context, policy, requested budget, metric, and future distribution, rotations are averaged
   **within context**. Contexts then receive equal weight.
2. Invalid or missing utilities and incomplete relation support are excluded with an auditable reason;
   positive distribution weights are never silently renormalized.
3. Exact duplicate objective coordinates are one geometric equivalence point but retain every policy
   identity. Ties do not dominate each other.
4. In the raw frontier, point a dominates b only if it is no worse in both utilities and no more costly,
   with at least one strict improvement. In a constrained frontier at cap B, a configuration is
   eligible only when its maximum delivered message length is ≤B; eligible points are then compared
   on the two utilities. Mean delivered words remains the third objective. A cheaper budget rung
   remains eligible at larger B.
5. Nondominance is computed separately for every dataset × writer/reader/seed stack × future-query
   distribution × metric. Pooling incomparable regimes is rejected by construction.
6. Hypervolume is exact union area in [0,1]² from reference (0,0). Raw 3D hypervolume maps cost to
   1−C/160 and uses reference (0,0,0); it is reported only inside one regime. Frontier-membership
   probabilities and HV intervals use the same paired context resample across all policies.
7. For each λ and cap, the empirical supported set contains **all** feasible structured α arms
   tying for the maximum of U_now+λU_future (future utility alone at λ=∞). Intended-α scalar
   regret is that best observed score minus the score of the same-cap α arm paired with λ in the
   treatment grid. This evaluates the pairing; it does not identify α with λ.

Frontier coverage counts distinct nondominated coordinate groups, so several policies tied at exactly
the same point do not inflate geometric coverage. “Dominated-area reduction” is ΔHV divided by the
unit-square area not already covered by the full legacy deployable set (including extractive controls
where they exist).

## Figures

""" + "\n\n".join(fig_sections) + f"""

## Robustness and non-robust improvements

- {len(fragile)} point-estimate constrained-frontier memberships have bootstrap probability below
  0.80. These are visually plausible frontier points but are not stable enough to support a dominance
  claim.
- Frontier membership agreement between the binary judge and EM/F1 has median Jaccard
  {_fmt(float(np.median(metric_jaccard)) if metric_jaccard else None)} across available matched cells.
- Changing the relation-query distribution gives median frontier Jaccard
  {_fmt(float(np.median(dist_jaccard)) if dist_jaccard else None)}. Low agreement is substantive:
  a policy useful for paraphrases need not be useful for orthogonal later questions.
- Seed and model replications remain separate regimes. Agreement is summarized through endpoint
  direction and matched frontier-set overlap. For overlap, both stacks are restricted to shared
  semantic configurations and common contexts before their frontiers are recomputed separately;
  one model’s point is never allowed to dominate another model’s point.
- Requested budget is not treated as achieved cost. Maximum delivered length enforces hard
  feasibility; mean delivered words is the plotted cost. Nonmonotonic outcomes remain in the tables
  and can be dominated by a cheaper rung.
- Intervals are prespecified descriptive bootstrap intervals, not family-wise-error-adjusted tests.
  The report therefore qualifies an outward-shift claim only when the primary ΔHV lower bound is
  above the numerical zero tolerance. The stronger robustness label additionally requires every
  designated metric and future-distribution sensitivity cell; favorable panels are not selected
  after the fact.

The free-form control hypothesis would be falsified if increasing the stated objective weight λ does
not reproducibly trade present for future utility. Separately, the allocation-control hypothesis would
be falsified if increasing α does not produce that directional response. Failure of the latter does
not by itself falsify the mathematical scalarized objective, because α and λ are different
quantities. A structured method **does not shift the frontier** merely because one plotted point looks
higher: its paired ΔHV interval must exclude the numerical zero tolerance. A robustness claim further
requires every explicitly designated metric and future-distribution sensitivity cell; model and seed
replications are reported separately rather than used as interchangeable significance filters.

## Glossary

- **Generic:** a message written without seeing the current question; a reusable baseline.
- **Conditioned:** a message optimized for the one question currently known.
- **Reusable:** sees the current question and is warned that unknown later questions will follow.
- **Oracle:** sees all evaluated questions. It is a non-deployable channel-capacity control, not a
  proposed policy and not direct access to the source at answer time.
- **Objective weight λ / scalar free-form:** λ multiplies measured future utility in
  U_now+λU_future; the writer receives this priority but chooses its own internal allocation.
- **Allocation share α / structured split:** α is the fraction of the evidence-word allowance
  reserved for reusable evidence; 1−α is reserved for current-task evidence. It is an intervention,
  not an objective coefficient.
- **Nondominated / Pareto point:** no comparable observation is at least as good in every objective
  and strictly better in one. **Dominated** means such an observation exists.
- **Reusable evidence:** facts likely to answer later questions beyond the one currently known.
- **Communication budget B:** hard maximum per delivered message. **Cost C** is mean observed
  delivered words for the policy point; both are audited rather than conflated.
- **Frontier membership probability:** fraction of paired bootstrap resamples in which a configuration
  is nondominated; it measures sampling stability, not posterior probability of a theory.

## Reproducible artifacts

- [Capacity and leakage baselines](analysis/baselines.csv)
- [Point estimates and intervals](analysis/points.csv)
- [Raw three-objective frontier](analysis/raw_frontier.csv)
- [Constrained feasible frontiers](analysis/constrained_frontiers.csv)
- [Bootstrap frontier membership](analysis/frontier_membership.csv)
- [Hypervolume summaries](analysis/hypervolume.csv)
- [Hypervolume gains / dominated-area reduction](analysis/hypervolume_effects.csv)
- [Frontier coverage](analysis/frontier_coverage.csv)
- [Preference trends](analysis/preference_trends.csv)
- [Objective-λ winners and intended-α scalar regret](analysis/scalar_selection.csv)
- [Model/seed endpoint and matched-frontier robustness](analysis/replication_robustness.csv)
- [Robustness checks](analysis/robustness.csv)
- [Input and panel audit](analysis/panel_audit.csv)
- [Semantic plot snapshot](plot_spec.json)

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by
`python src/analysis/render_frontier_study.py`. The CSV files are audit artifacts; raw tables are intentionally
kept out of this concise narrative.
"""
    path = output_root / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def _clean_rows(rows: Sequence[Mapping]) -> list[dict]:
    cleaned = []
    for row in rows:
        cleaned.append({key: _jsonable(value) for key, value in row.items()
                        if not key.startswith("_")})
    return cleaned


def run_report(root: Path, config_path: Path, output_root: Path, *,
               n_resamples: int, ci: float, seed: int,
               include_partial: bool = False) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts, inventory = discover_artifacts(root, cfg, include_partial=include_partial)
    matrix_rows, load_excluded = load_matrix_rows(artifacts)
    baseline_rows = load_baseline_rows(artifacts)
    rotation_rows, rotation_excluded = build_rotation_observations(matrix_rows, cfg)
    context_rows = collapse_contexts(rotation_rows)
    analysis = analyse_frontiers(context_rows, cfg, n_resamples, ci, seed)
    hypervolume, hv_effects = analyse_policy_sets(context_rows, cfg, n_resamples, ci, seed)
    coverage = analyse_coverage(analysis["constrained"], analysis["membership"])
    trends = analyse_preference_trends(context_rows, n_resamples, ci, seed)
    scalar_selection = analyse_scalar_selection(
        analysis["points"], analysis["constrained"], analysis["membership"], cfg)
    policy_effects = analyse_policy_effects(analysis["constrained"])
    robustness = analyse_robustness(analysis["constrained"], trends)
    replication_robustness = analyse_replication_robustness(
        context_rows, cfg, n_resamples, ci, seed)

    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_root / "analysis"
    figures_dir = output_root / "figures"
    _write_csv(analysis_dir / "artifact_inventory.csv", inventory)
    _write_csv(analysis_dir / "baselines.csv", baseline_rows)
    _write_csv(analysis_dir / "excluded_rows.csv", [*load_excluded, *rotation_excluded])
    _write_csv(analysis_dir / "context_utilities.csv", context_rows)
    _write_csv(analysis_dir / "points.csv", _clean_rows(analysis["points"]))
    _write_csv(analysis_dir / "raw_frontier.csv", _clean_rows(analysis["raw_frontier"]))
    _write_csv(analysis_dir / "constrained_frontiers.csv",
               _clean_rows(analysis["constrained"]))
    _write_csv(analysis_dir / "frontier_membership.csv",
               _clean_rows(analysis["membership"]))
    _write_csv(analysis_dir / "hypervolume.csv", hypervolume)
    _write_csv(analysis_dir / "hypervolume_effects.csv", hv_effects)
    _write_csv(analysis_dir / "frontier_coverage.csv", coverage)
    _write_csv(analysis_dir / "preference_trends.csv", trends)
    _write_csv(analysis_dir / "scalar_selection.csv", scalar_selection)
    _write_csv(analysis_dir / "policy_effects.csv", policy_effects)
    _write_csv(analysis_dir / "replication_robustness.csv", replication_robustness)
    _write_csv(analysis_dir / "robustness.csv", robustness)
    _write_csv(analysis_dir / "panel_audit.csv", analysis["panel_audit"])
    figures, plot_spec = render_figures(
        analysis, hypervolume, trends, figures_dir, replication_robustness)
    plot_spec.update({
        "bootstrap_resamples": n_resamples,
        "ci_level": ci,
        "bootstrap_seed": seed,
        "include_partial": include_partial,
        "inventory": inventory,
    })
    (output_root / "plot_spec.json").write_text(
        json.dumps(plot_spec, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_report(output_root, inventory, analysis, hypervolume, hv_effects,
                          coverage, trends, scalar_selection, robustness,
                          replication_robustness, baseline_rows, figures, cfg, n_resamples, ci,
                          include_partial)
    manifest = {
        "schema_version": REPORT_SCHEMA,
        "report": str(report.relative_to(ROOT) if report.is_relative_to(ROOT) else report),
        "artifacts_selected": len(artifacts),
        "matrix_rows": len(matrix_rows),
        "baseline_rows": len(baseline_rows),
        "rotation_distribution_rows": len(rotation_rows),
        "context_rows": len(context_rows),
        "point_rows": len(analysis["points"]),
        "constrained_rows": len(analysis["constrained"]),
        "scalar_selection_rows": len(scalar_selection),
        "replication_robustness_rows": len(replication_robustness),
        "figures": [figure["id"] for figure in figures],
        "bootstrap_resamples": n_resamples,
        "ci_level": ci,
        "include_partial": include_partial,
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bounded_communication_frontier_config.yaml")
    parser.add_argument("--output-dir", default="results/bounded_communication_frontier")
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--ci", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args(argv)
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    n_resamples = (args.bootstrap_resamples if args.bootstrap_resamples is not None
                   else int(cfg["analysis"]["bootstrap_resamples"]))
    ci = args.ci if args.ci is not None else float(cfg["analysis"]["ci_level"])
    seed = args.seed if args.seed is not None else int(cfg["analysis"]["bootstrap_seed"])
    if n_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")
    if not 0 < ci < 1:
        parser.error("--ci must lie between zero and one")
    manifest = run_report(ROOT, config_path, ROOT / args.output_dir,
                          n_resamples=n_resamples, ci=ci, seed=seed,
                          include_partial=args.include_partial)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
