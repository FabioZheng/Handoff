"""Offline semantic/snapshot checks for the frontier report renderer."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import render_frontier_study as report  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))
    print(f"PASS  {label}" + (f" -- {detail}" if detail else ""))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_matrix() -> list[dict]:
    rows = []
    policies = {
        "generic": {20: (.45, {"paraphrase": .50, "same_entity": .55,
                                "same_topic": .45, "orthogonal": .40}),
                    40: (.55, {"paraphrase": .65, "same_entity": .65,
                                "same_topic": .60, "orthogonal": .55})},
        "conditioned": {20: (.80, {"paraphrase": .75, "same_entity": .40,
                                    "same_topic": .30, "orthogonal": .10}),
                        40: (.90, {"paraphrase": .85, "same_entity": .45,
                                    "same_topic": .35, "orthogonal": .15})},
        "oracle": {20: (.75, {"paraphrase": .75, "same_entity": .75,
                               "same_topic": .75, "orthogonal": .70}),
                   40: (.95, {"paraphrase": .95, "same_entity": .90,
                               "same_topic": .90, "orthogonal": .85})},
    }
    for context_number in range(3):
        context = f"c{context_number}"
        offset = (context_number - 1) * .02
        for policy, budgets in policies.items():
            for budget, (now, futures) in budgets.items():
                common = {
                    "context_id": context,
                    "dataset": "relation_dossiers",
                    "policy": policy,
                    "budget_words": budget,
                    "rotation_id": f"{context}|r0",
                    "cond_qid": f"{context}:anchor",
                    "message_key": f"{context}|{policy}|{budget}",
                    "message_shared_across_rotations": "False",
                    "delivered_words": budget - 1,
                    "fill_ratio": (budget-1)/budget,
                    "truncated": "False",
                    "attempts": 1,
                    "ceiling_em": 1.0,
                    "ceiling_f1": 1.0,
                    "ceiling_judge_correct": 1.0,
                    "regret_em": 0.0,
                    "regret_f1": 0.0,
                    "regret_judge_correct": 0.0,
                }
                rows.append({**common, "eval_qid": f"{context}:anchor",
                             "is_diagonal": "True", "relation": "unlabelled",
                             "u_em": now+offset, "u_f1": now+offset,
                             "u_judge_correct": now+offset})
                for relation, value in futures.items():
                    # Two orthogonal cells ensure the report averages within a
                    # category before applying category weights.
                    repeats = 2 if relation == "orthogonal" else 1
                    for repeat in range(repeats):
                        rows.append({**common,
                                     "eval_qid": f"{context}:{relation}:{repeat}",
                                     "is_diagonal": "False", "relation": relation,
                                     "u_em": value+offset, "u_f1": value+offset,
                                     "u_judge_correct": value+offset})
    return rows


def synthetic_replication_contexts() -> list[dict]:
    """Small paired panels for endpoint and matched-frontier semantics."""
    rows: list[dict] = []

    def add(stack: str, label: str, dataset: str, context: str,
            preference: str, share: float, now: float, future: float) -> None:
        rows.append({
            "stack_id": stack, "stack_label": label, "dataset": dataset,
            "future_distribution": ("far" if dataset == "relation_dossiers"
                                    else "empirical_questions"),
            "metric": "judge_correct", "context_id": context,
            "policy": f"split__{preference}", "policy_family": "split",
            "policy_label": preference, "preference_id": preference,
            "future_share": share, "lambda": share,
            "lambda_infinite": preference == "linf",
            "requested_budget": 80,
            "configuration_id": f"split__{preference}|requested_B=80",
            "deployable": True, "analysis_eligible": True,
            "u_now": now, "u_future": future, "cost": 76.0,
            "n_rotations": 1,
        })

    stack_a = "sender=model/a|reader=model/a|temperature=0|seed=none"
    stack_b = "sender=model/b|reader=model/b|temperature=0|seed=none"
    for context, offset in (("c0", -.01), ("c1", .01)):
        # The unshared midpoint dominates both endpoints in A's unrestricted
        # set. It must disappear before A's frontier is recomputed against B.
        add(stack_a, "A", "relation_dossiers", context, "l0", 0, .80+offset, .20+offset)
        add(stack_a, "A", "relation_dossiers", context, "l1", .5, .95+offset, .95+offset)
        add(stack_a, "A", "relation_dossiers", context, "linf", 1, .20+offset, .80+offset)
        add(stack_b, "B", "relation_dossiers", context, "l0", 0, .78+offset, .22+offset)
        add(stack_b, "B", "relation_dossiers", context, "linf", 1, .22+offset, .78+offset)

    for seed_index, sender_seed in enumerate((17, 29, 43)):
        stack = ("sender=model/a|reader=model/a|temperature=0.2|"
                 f"seed={sender_seed}")
        for context, offset in (("s0", -.01), ("s1", .01)):
            add(stack, f"seed {sender_seed}", "squad_groups", context,
                "l0", 0, .82+offset, .28+offset)
            add(stack, f"seed {sender_seed}", "squad_groups", context,
                "linf", 1, .42-.02*seed_index+offset,
                .56+.02*seed_index+offset)
    return rows


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    cfg = {
        "budget": {"words": [20, 40]},
        "analysis": {
            "bootstrap_resamples": 40,
            "ci_level": .95,
            "bootstrap_seed": 9,
            "dominance_epsilon": 0.0,
            "hypervolume_cost_max_words": 160,
        },
        "outputs": {"result_root": "results/bounded_communication_frontier"},
        "frontier": {
            "future_distributions": {
                "empirical_cells": {"paraphrase": .1, "same_entity": .1,
                                    "same_topic": .1, "orthogonal": .7},
                "relation_balanced": {"paraphrase": .25, "same_entity": .25,
                                      "same_topic": .25, "orthogonal": .25},
                "novel_near": {"paraphrase": 0, "same_entity": .5,
                               "same_topic": .5, "orthogonal": 0},
                "far": {"paraphrase": 0, "same_entity": 0,
                        "same_topic": 0, "orthogonal": 1},
            },
            "replications": {
                "primary": {
                    "datasets": ["relation_dossiers"],
                    "limits": {"relation_dossiers": 3},
                    "budgets_words": [40],
                    "preference_ids": ["l0", "l1", "linf"],
                    "families": ["split"],
                }
            },
        },
    }
    config_path = tmp / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    legacy_dir = tmp / "results/communication_regret/relation_dossiers/n3"
    legacy_dir.mkdir(parents=True)
    legacy_manifest = {
        "schema_version": "communication-regret-v1",
        "dataset": "relation_dossiers",
        "sender_model": "meta-llama/llama-3.1-8b-instruct",
        "answerer_model": "meta-llama/llama-3.1-8b-instruct",
        "policies": ["generic", "conditioned", "oracle"],
        "budgets_words": [20, 40],
        "contexts": 3,
    }
    (legacy_dir / "manifest.json").write_text(json.dumps(legacy_manifest), encoding="utf-8")
    write_csv(legacy_dir / "utility_matrix.csv", synthetic_matrix())

    # A smoke output exists but must not enter confirmatory analysis.
    partial_dir = tmp / "results/bounded_communication_frontier/primary/relation_dossiers/n1"
    partial_dir.mkdir(parents=True)
    partial_manifest = {
        "schema_version": "bounded-communication-frontier-v2",
        "replication": "primary", "dataset": "relation_dossiers", "contexts": 1,
        "sender_model": "meta-llama/llama-3.1-8b-instruct",
        "answerer_model": "meta-llama/llama-3.1-8b-instruct",
        "families": ["split"], "budgets_words": [40],
        "preferences": [{"preference_id": "l0"}],
    }
    (partial_dir / "manifest.json").write_text(json.dumps(partial_manifest), encoding="utf-8")
    write_csv(partial_dir / "utility_matrix.csv", synthetic_matrix()[:2])

    output = tmp / "paper"
    manifest = report.run_report(tmp, config_path, output, n_resamples=40,
                                 ci=.95, seed=9, include_partial=False)
    check("partial Experiment 11 output is excluded",
          manifest["artifacts_selected"] == 1)
    check("renderer emits both constrained and raw Pareto rows",
          manifest["point_rows"] > 0 and manifest["constrained_rows"] > 0)

    context_rows = list(csv.DictReader(
        (output / "analysis/context_utilities.csv").open(encoding="utf-8")))
    generic = next(row for row in context_rows
                   if row["policy"] == "generic" and row["metric"] == "judge_correct"
                   and row["future_distribution"] == "relation_balanced"
                   and row["requested_budget"] == "20" and row["context_id"] == "c1")
    # (.50 + .55 + .45 + mean(.40,.40)) / 4 = .475. A cell-weighted
    # calculation would be .46, so this checks context/category weighting.
    check("relation distribution weights category means, not raw cells",
          abs(float(generic["u_future"]) - .475) < 1e-12,
          generic["u_future"])

    spec = json.loads((output / "plot_spec.json").read_text(encoding="utf-8"))
    snapshot = {
        "schema_version": spec["schema_version"],
        "axis_contract": spec["axis_contract"],
        "categorical_interpolation": spec["categorical_interpolation"],
        "figure_ids": [figure["id"] for figure in spec["figures"]],
        "distribution_panels": [panel["future_distribution"]
                                for panel in spec["figures"][0]["panels"]],
    }
    expected = {
        "schema_version": "bounded-communication-frontier-report-v1",
        "axis_contract": {
            "x": {"name": "U_now", "range": [0, 1], "direction": "higher_is_better"},
            "y": {"name": "U_future", "range": [0, 1], "direction": "higher_is_better"},
            "cost": {"name": "mean_delivered_words", "units": "words",
                     "direction": "lower_is_better"},
        },
        "categorical_interpolation": False,
        "figure_ids": ["fig1_distribution_frontiers", "fig2_budget_frontiers",
                       "fig3_raw_three_objective", "fig5_hypervolume"],
        "distribution_panels": ["relation_balanced", "novel_near", "far",
                                "empirical_cells"],
    }
    check("semantic plot specification matches snapshot", snapshot == expected,
          json.dumps(snapshot, sort_keys=True))
    fig5 = next(figure for figure in spec["figures"]
                if figure["id"] == "fig5_hypervolume")
    check("hypervolume caps are discrete and never interpolated",
          all(panel.get("interpolation") is False for panel in fig5["panels"]))
    for figure in spec["figures"]:
        check(f"{figure['id']} has 300-dpi PNG and vector PDF",
              (output / figure["png"]).is_file()
              and (output / figure["pdf"]).is_file())

    report_text = (output / "REPORT.md").read_text(encoding="utf-8")
    check("report preserves TeX commands without escape corruption",
          r"\right)" in report_text and r"\text{subject to}" in report_text
          and "\r" not in report_text)
    check("report defines natural passages and designed dossiers",
          "**Natural passages:**" in report_text and "**Designed dossiers:**" in report_text)
    check("report links the replication robustness audit",
          "analysis/replication_robustness.csv" in report_text
          and (output / "analysis/replication_robustness.csv").is_file())
    normalized_report = " ".join(report_text.split())
    check("report keeps objective lambda distinct from allocation alpha",
          "not mathematically equivalent" in normalized_report
          and "equivalently future shares" not in report_text
          and "implied by λ/(1+λ)" not in report_text)
    check("renderer exports the scalar-selection audit artifact",
          (output / "analysis/scalar_selection.csv").is_file())

    print("\n=== objective-lambda selection over structured alpha arms ===")
    scalar_cfg = json.loads(json.dumps(cfg))
    scalar_cfg["frontier"]["preferences"] = [
        {"id": "l0", "lambda": 0.0, "future_share": 0.0},
        {"id": "l1", "lambda": 1.0, "future_share": 0.5},
        {"id": "linf", "lambda": "inf", "future_share": 1.0},
    ]
    scalar_sources = []
    for policy, alpha, now, future in (
        ("split_l0", 0.0, .90, .10),
        ("split_l1", .5, .65, .45),
        ("split_linf", 1.0, .50, .70),
        ("split_linf_tie", 1.0, .50, .70),
    ):
        scalar_sources.append({
            "stack_id": "stack", "stack_label": "Synthetic stack",
            "dataset": "relation_dossiers", "future_distribution": "far",
            "metric": "judge_correct", "scope": "deployable",
            "configuration_id": f"{policy}|requested_B=40",
            "policy": policy, "policy_family": "split",
            "preference_id": policy.removeprefix("split_"),
            "future_share": alpha, "requested_budget": 40,
            "deployable": True, "analysis_eligible": True,
            "u_now": now, "u_future": future, "cost": 39.0,
        })
    sanitized = report.pf.sanitize_points(
        scalar_sources, regime_keys=report.REGIME_KEYS,
        config_keys=report.CONFIG_KEYS).points
    marked = report.pf.mark_nondominated(sanitized)
    constrained = report.pf.constrained_frontiers(marked, [40])
    membership = [{
        **{key: value for key, value in row.items() if not key.startswith("_")},
        "membership_type": "constrained",
        "frontier_probability": 1.0,
        "frontier_probability_given_feasible": 1.0,
    } for row in constrained if row["feasible"]]
    scalar_selection = report.analyse_scalar_selection(
        marked, constrained, membership, scalar_cfg)
    selection_by_lambda = {
        row["objective_lambda_label"]: row for row in scalar_selection}
    check("lambda zero selects the present-utility endpoint",
          json.loads(selection_by_lambda["0"]["winner_policy_ids"]) == ["split_l0"]
          and selection_by_lambda["0"]["intended_allocation_alpha"] == 0.0
          and selection_by_lambda["0"]["intended_alpha_is_stably_supported"])
    check("lambda infinity preserves tied future-only winners",
          selection_by_lambda["∞"]["n_winner_ties"] == 2
          and set(json.loads(selection_by_lambda["∞"]["winner_policy_ids"]))
          == {"split_linf", "split_linf_tie"}
          and selection_by_lambda["∞"]["intended_allocation_alpha"] == 1.0
          and selection_by_lambda["∞"]["intended_alpha_is_stably_supported"])
    check("intended-alpha scalar regret exposes a non-optimal grid pairing",
          abs(float(selection_by_lambda["1"]["intended_alpha_scalar_regret"]) - .10)
          < 1e-12
          and not selection_by_lambda["1"]["intended_alpha_is_objective_winner"]
          and selection_by_lambda["1"]["interpretation"]
          == "nondominated_but_not_objective_supported")

    zero_cfg = json.loads(json.dumps(cfg))
    zero_cfg.setdefault("analysis", {})["hypervolume_zero_tolerance"] = 1e-12
    zero_rows = []
    for context in ("z1", "z2"):
        for policy, family in (("generic", "legacy"), ("split__l0", "split")):
            zero_rows.append({
                "stack_id": "zero-stack", "stack_label": "Zero stack",
                "dataset": "squad_groups", "future_distribution": "empirical_questions",
                "metric": "judge_correct", "context_id": context,
                "configuration_id": f"{policy}|requested_B=20",
                "policy": policy, "policy_family": family,
                "policy_label": policy, "preference_id": "l0" if family == "split" else "",
                "future_share": 0.0 if family == "split" else None,
                "lambda": 0.0 if family == "split" else None,
                "lambda_infinite": False, "requested_budget": 20,
                "deployable": True, "analysis_eligible": True,
                "u_now": .6, "u_future": .4, "cost": 19.0,
                "feasibility_cost": 20.0, "n_rotations": 1,
            })
    _, zero_effects = report.analyse_policy_sets(
        zero_rows, zero_cfg, n_resamples=30, ci=.95, seed=4)
    zero_split = next(row for row in zero_effects
                      if row["augmented_set"] == "legacy_plus_split"
                      and row["constraint_budget"] == 20)
    check("zero hypervolume gain is not a floating-point robust improvement",
          zero_split["hypervolume_delta"] == 0.0
          and zero_split["hypervolume_delta_lo"] == 0.0
          and not zero_split["robust_improvement"])

    print("\n=== per-dataset completeness and embedded controls ===")
    audit_cfg = json.loads(json.dumps(cfg))
    audit_cfg["frontier"]["replications"]["multi"] = {
        "datasets": ["squad_groups", "relation_dossiers"],
        "limits": {"squad_groups": 3, "relation_dossiers": 3},
        "budgets_words": [40, 80, 160],
        "budgets_by_dataset": {
            "squad_groups": [40, 80],
            "relation_dossiers": [40, 80, 160],
        },
        "preference_ids": ["l0"],
        "families": ["split"],
    }
    common_new_manifest = {
        "schema_version": "bounded-communication-frontier-v2",
        "contexts": 3,
        "sender_model": "meta-llama/llama-3.1-8b-instruct",
        "answerer_model": "meta-llama/llama-3.1-8b-instruct",
        "families": ["split"],
        "preferences": [{"preference_id": "l0"}],
    }
    for dataset in ("squad_groups", "relation_dossiers"):
        cell_dir = tmp / f"results/bounded_communication_frontier/multi/{dataset}/n3"
        cell_dir.mkdir(parents=True)
        cell_manifest = {
            **common_new_manifest,
            "replication": "multi",
            "dataset": dataset,
            "budgets_words": [40, 80],
        }
        (cell_dir / "manifest.json").write_text(
            json.dumps(cell_manifest), encoding="utf-8")

    _artifacts, audit_inventory = report.discover_artifacts(tmp, audit_cfg)
    squad_cell = next(row for row in audit_inventory
                      if row["replication"] == "multi"
                      and row["dataset"] == "squad_groups")
    relation_cell = next(row for row in audit_inventory
                         if row["replication"] == "multi"
                         and row["dataset"] == "relation_dossiers")
    check("per-dataset budgets accept the planned SQuAD 40/80 grid",
          squad_cell["complete"] and squad_cell["selected"], squad_cell["status"])
    check("per-dataset budgets still reject a missing relation 160-word cell",
          not relation_cell["complete"] and "budgets 160" in relation_cell["status"],
          relation_cell["status"])

    mistral_model = "mistralai/ministral-8b-2512"
    audit_cfg["frontier"]["replications"]["mistral_stack"] = {
        "datasets": ["relation_dossiers"],
        "limits": {"relation_dossiers": 3},
        "budgets_words": [40],
        "preference_ids": ["l0"],
        "families": ["split"],
    }
    control_dir = (tmp / "results/bounded_communication_frontier/mistral_controls"
                   / "relation_dossiers/n3")
    control_dir.mkdir(parents=True)
    control_manifest = {
        "schema_version": "communication-regret-v1",
        "dataset": "relation_dossiers",
        "contexts": 3,
        "sender_model": mistral_model,
        "answerer_model": mistral_model,
        "policies": ["generic", "conditioned", "oracle"],
        "budgets_words": [20, 40],
    }
    (control_dir / "manifest.json").write_text(
        json.dumps(control_manifest), encoding="utf-8")
    write_csv(control_dir / "utility_matrix.csv", synthetic_matrix())

    split_dir = (tmp / "results/bounded_communication_frontier/mistral_stack"
                 / "relation_dossiers/n3")
    split_dir.mkdir(parents=True)
    split_manifest = {
        "schema_version": "bounded-communication-frontier-v2",
        "replication": "mistral_stack",
        "dataset": "relation_dossiers",
        "contexts": 3,
        "sender_model": mistral_model,
        "answerer_model": mistral_model,
        "sender_temperature": 0.0,
        "sender_seed": None,
        "families": ["split"],
        "preferences": [{"preference_id": "l0"}],
        "budgets_words": [40],
    }
    (split_dir / "manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    split_rows = []
    for row in synthetic_matrix():
        if row["policy"] != "generic" or row["budget_words"] != 40:
            continue
        split_rows.append({
            **row,
            "policy": "split_l0",
            "family": "split",
            "preference_id": "l0",
            "future_share": 0.0,
            "lambda": 0.0,
            "lambda_infinite": "False",
        })
    write_csv(split_dir / "utility_matrix.csv", split_rows)

    matched_artifacts, matched_inventory = report.discover_artifacts(tmp, audit_cfg)
    control_inventory = next(row for row in matched_inventory
                             if row["replication"] == "mistral_controls")
    check("embedded Experiment 10 Mistral controls are selected",
          control_inventory["kind"] == "experiment10"
          and control_inventory["complete"] and control_inventory["selected"],
          json.dumps(control_inventory, sort_keys=True))
    strict_control_cfg = json.loads(json.dumps(audit_cfg))
    strict_control_cfg["frontier"]["control_replications"] = {
        "mistral_controls": {
            "datasets": ["relation_dossiers"], "contexts": 3,
            "budgets_words": [20, 40],
            "policies": ["generic", "conditioned", "reusable", "oracle"],
        }
    }
    _, strict_inventory = report.discover_artifacts(tmp, strict_control_cfg)
    strict_control = next(row for row in strict_inventory
                          if row["replication"] == "mistral_controls")
    check("declared partial control matrices fail closed",
          not strict_control["complete"] and not strict_control["selected"]
          and "policies reusable" in strict_control["status"],
          json.dumps(strict_control, sort_keys=True))
    loaded, load_exclusions = report.load_matrix_rows(matched_artifacts)
    mistral_exclusions = [row for row in load_exclusions
                          if str(control_dir) in str(row.get("path", ""))
                          or str(split_dir) in str(row.get("path", ""))]
    check("embedded Mistral control and split matrices load without exclusions",
          not mistral_exclusions, json.dumps(mistral_exclusions, sort_keys=True))
    control_stacks = {row["stack_id"] for row in loaded
                      if Path(row["artifact_dir"]) == control_dir}
    split_stacks = {row["stack_id"] for row in loaded
                    if Path(row["artifact_dir"]) == split_dir}
    check("Mistral controls and split replication share one analysis stack",
          len(control_stacks) == 1 and control_stacks == split_stacks,
          f"controls={control_stacks}, split={split_stacks}")

    print("\n=== model/seed replication semantics ===")
    replication_rows = report.analyse_replication_robustness(
        synthetic_replication_contexts(), cfg, n_resamples=80, ci=.95, seed=9)
    endpoints = [row for row in replication_rows
                 if row["analysis"] == "structured_endpoint"]
    check("B=80 endpoint effects retain every model/seed stack",
          len(endpoints) == 5, str(len(endpoints)))
    seed_summary = next(row for row in replication_rows
                        if row["analysis"] == "temperature_seed_summary")
    check("three sampling seeds have an explicit all-sign/range summary",
          seed_summary["complete_seed_triplet"]
          and seed_summary["n_stacks"] == 3
          and seed_summary["all_sign_agreement"]
          and seed_summary["delta_u_now_min"] <= seed_summary["delta_u_now_max"]
          and seed_summary["delta_u_future_min"] <= seed_summary["delta_u_future_max"],
          json.dumps(seed_summary, sort_keys=True))
    relation_jaccard = next(
        row for row in replication_rows
        if row["analysis"] == "matched_frontier_jaccard"
        and row["dataset"] == "relation_dossiers")
    check("cross-stack Jaccard restricts then recomputes matched frontiers",
          relation_jaccard["estimate"] == 1.0
          and relation_jaccard["n_shared_configurations"] == 2
          and relation_jaccard["left_frontier_size"] == 2
          and relation_jaccard["right_frontier_size"] == 2,
          json.dumps(relation_jaccard, sort_keys=True))

    minimal_analysis = {
        "points": [{"stack_id": "plot", "policy_family": "split"}],
        "constrained": [],
    }
    replication_figures, replication_spec = report.render_figures(
        minimal_analysis, [], [], output / "replication_figures", replication_rows)
    replication_snapshot = {
        "ids": [figure["id"] for figure in replication_figures],
        "panels": [{key: panel[key] for key in
                    ("dataset", "future_distribution", "constraint_budget",
                     "n_stacks", "interpolation", "cross_regime_domination")}
                   for panel in replication_spec["figures"][0]["panels"]],
        "categorical_interpolation": replication_spec["figures"][0][
            "categorical_interpolation"],
        "cross_regime_domination": replication_spec["figures"][0][
            "cross_regime_domination"],
    }
    expected_replication_snapshot = {
        "ids": ["fig6_replication_endpoint_deltas"],
        "panels": [
            {"dataset": "squad_groups", "future_distribution": "empirical_questions",
             "constraint_budget": 80, "n_stacks": 3,
             "interpolation": False, "cross_regime_domination": False},
            {"dataset": "relation_dossiers", "future_distribution": "far",
             "constraint_budget": 80, "n_stacks": 2,
             "interpolation": False, "cross_regime_domination": False},
        ],
        "categorical_interpolation": False,
        "cross_regime_domination": False,
    }
    check("replication forest/facet semantic snapshot is stable",
          replication_snapshot == expected_replication_snapshot,
          json.dumps(replication_snapshot, sort_keys=True))
    replication_figure = replication_figures[0]
    check("replication forest has 300-dpi PNG and vector PDF",
          (output / replication_figure["png"]).is_file()
          and (output / replication_figure["pdf"]).is_file())

print("\nALL FRONTIER REPORT CHECKS PASSED")
