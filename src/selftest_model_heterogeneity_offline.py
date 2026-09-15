"""Offline checks for Experiment 8. No API key, no network, no spend.

What is worth asserting here is everything that would silently make the
experiment measure the wrong thing:

* the compression prompts never mention a model, a family or a size, so the
  only thing that differs between arms is the decoder;
* stage 2+ is sealed, so source passages cannot re-enter a chain;
* the sealed recompression prompt is byte-identical to Experiment 5's
  published ``conditioned`` prompt, so this experiment inherits that baseline
  rather than quietly forking it;
* the schedules do what the design claims -- cross-family arms cross family at
  every handoff, "same family" arms never do, start families are balanced, and
  the start-matched control shares stage 1 with its cross-family treatment;
* the up/down/alt trajectories share a tier multiset at max depth, so the
  direction contrast is an ordering contrast;
* prefix memoisation actually collapses the shared chain prefixes it claims to;
* marker area is proportional to parameter count, not to radius;
* an undisclosed parameter count survives the whole pipeline as unknown rather
  than being filled in;
* both figures render from synthetic metrics.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_pool as mp  # noqa: E402
import run_model_heterogeneity as hx  # noqa: E402
from llm import load_config  # noqa: E402
from run_summary_generalization import compression_user_prompt, load_prebuilt_pairs  # noqa: E402

CONFIG = ROOT / "model_heterogeneity_config.yaml"


def load(n: int = 8) -> tuple[dict, list[dict], dict, list, list, dict]:
    cfg = load_config(CONFIG)
    pairs = load_prebuilt_pairs(cfg, n)
    registry = mp.load_registry(cfg)
    arms, derived = mp.load_arms(cfg)
    schedules = hx.build_schedules(pairs, arms, derived, cfg, registry)
    return cfg, pairs, registry, arms, derived, schedules


def test_prompts(cfg, pairs) -> None:
    hx.prompt_selftest(pairs[0], cfg)
    pair = pairs[0]
    notes = "previous notes"
    import handoffs as hm

    sealed = hm.seal(pair["question_A"], hm.Handoff(pair["pair_id"], "chain", None, notes))
    rebuilt = hx.recompress_user_prompt(sealed)
    assert rebuilt == compression_user_prompt(pair, notes, hx.MODE, 4, cfg)
    # The seal carries no passage text, so no distractor can reappear later.
    for passage in pair["passages"]:
        assert passage["text"][:80] not in rebuilt
    print("[selftest] prompts match Experiment 5, are model-blind, and stage 2+ is sealed")


def test_schedules(cfg, pairs, registry, arms, derived, schedules) -> None:
    hx.schedule_selftest(pairs, arms, derived, schedules, cfg, registry)
    by_name = {a.name: a for a in list(arms) + list(derived)}
    max_depth = int(cfg["max_depth"])

    # A start-matched control must be one of the four homogeneous arms, chosen
    # by the pair's own start family -- that is what makes it a control.
    for position, pair in enumerate(pairs):
        family = mp.start_family(position, cfg["family_cycle"])
        assert (schedules[("homog_small_matched", pair["pair_id"])]
                == schedules[(f"homog_small_{family}", pair["pair_id"])])
        assert (schedules[("homog_large_matched", pair["pair_id"])]
                == schedules[(f"homog_large_{family}", pair["pair_id"])])

    # Reverse and forward cycles must visit different sequences after stage 1.
    for pair in pairs:
        forward = schedules[("xfam_small_fwd", pair["pair_id"])]
        reverse = schedules[("xfam_small_rev", pair["pair_id"])]
        assert forward[0] == reverse[0] and forward[1] != reverse[1]

    # Tiers must follow the configured trajectory exactly.
    for name, pattern in (("size_up_samefam", "up"), ("size_down_samefam", "down"),
                          ("size_alt_samefam", "alt")):
        wanted = cfg["size_patterns"][pattern][:max_depth]
        for pair in pairs:
            tiers = [registry[k].tier for k in schedules[(name, pair["pair_id"])]]
            assert tiers == wanted, f"{name} tiers {tiers} != {wanted}"

    # Every family transition count is what the arm advertises.
    for name in ("homog_small_llama", "xfam_small_fwd", "size_up_samefam", "size_up_xfam"):
        summary = mp.chain_summary(schedules[(name, pairs[0]["pair_id"])], registry)
        expected = 0 if by_name[name].family_mode in ("fixed_named", "start") else max_depth - 1
        assert summary["n_family_transitions"] == expected, (name, summary)
    print(f"[selftest] {len(by_name)} arm schedules satisfy the design's structural claims")


def test_prefix_sharing(cfg, pairs, arms, schedules) -> None:
    """Prefix memoisation must actually collapse the shared prefixes."""
    max_depth = int(cfg["max_depth"])
    naive = len(arms) * len(pairs) * max_depth
    distinct = set()
    for arm in arms:
        for pair in pairs:
            keys = schedules[(arm.name, pair["pair_id"])]
            for stage in range(1, max_depth + 1):
                distinct.add((pair["pair_id"], hx.prefix_id(keys, stage)))
    assert len(distinct) < naive, "no prefix sharing at all -- the memo is pointless"
    # Stage 1 is the strongest case: it can only ever be one call per (pair,
    # first model), and there are at most len(registry) first models.
    stage_one = {slot for slot in distinct if "|" not in slot[1]}
    assert len(stage_one) <= len(pairs) * len(cfg["models"])
    print(f"[selftest] prefix memo: {len(distinct)} distinct messages vs {naive} arm-stages "
          f"({100 * (1 - len(distinct) / naive):.0f}% collapsed); {len(stage_one)} stage-1 calls")


def test_transition_labels(registry) -> None:
    small = registry["llama_small"]
    large = registry["llama_large"]
    other = registry["qwen_small"]
    assert mp.transition_type(None, small)["transition_type"] == "source_to_first_agent"
    assert mp.transition_type(small, large)["transition_type"] == "same_family_size_larger"
    assert mp.transition_type(large, small)["transition_type"] == "same_family_size_smaller"
    assert mp.transition_type(small, other)["transition_type"] == "cross_family_size_same"
    assert mp.transition_type(small, large)["size_ratio"] > 1
    assert mp.transition_type(large, small)["log_size_change"] < 0
    # 8.0B -> 8.2B is inside the tolerance band and must not read as a change.
    assert mp.transition_type(registry["mistral_small"], registry["qwen_small"])["size_direction"] == "same"
    print("[selftest] transition typing separates family change from size change")


def test_unknown_size_stays_unknown(cfg) -> None:
    """An undisclosed parameter count must never be filled in downstream."""
    patched = copy.deepcopy(cfg)
    patched["models"]["gemma_small"]["parameters_b"] = None
    patched["models"]["gemma_small"]["parameters_basis"] = "undisclosed by the vendor"
    registry = mp.load_registry(patched)
    spec = registry["gemma_small"]
    assert spec.parameters_b is None and not spec.size_known
    assert json.loads(json.dumps(spec.to_json()))["parameters_b"] is None
    edge = mp.transition_type(registry["llama_small"], spec)
    assert edge["size_direction"] == "unknown" and edge["size_ratio"] is None
    assert edge["transition_type"] == "cross_family_size_unknown"
    summary = mp.chain_summary(["llama_small", "gemma_small"], registry)
    assert summary["final_parameters_b"] is None
    # It still gets a visible marker rather than vanishing from the figure.
    assert mp.marker_area(None, 70.6, cfg["analysis"]["size_scale"]) == \
        cfg["analysis"]["size_scale"]["min_area"]
    print("[selftest] an undisclosed size stays unknown through typing, summaries and plotting")


def test_marker_area(cfg, registry) -> None:
    scale = cfg["analysis"]["size_scale"]
    largest = max(s.parameters_b for s in registry.values() if s.size_known)
    smallest = min(s.parameters_b for s in registry.values() if s.size_known)
    big = mp.marker_area(largest, largest, scale)
    small = mp.marker_area(smallest, largest, scale)
    assert abs(big / small - largest / smallest) < 0.05 * (largest / smallest), (
        "dot AREA must be proportional to parameter count, not dot radius")
    assert small >= scale["min_area"], "the smallest model must stay above the visibility floor"
    # sqrt mode must compress the range rather than invert or flatten it.
    compressed = {"mode": "sqrt_area", "min_area": 1.0, "max_area": scale["max_area"]}
    ratio = mp.marker_area(largest, largest, compressed) / mp.marker_area(smallest, largest, compressed)
    assert 1.0 < ratio < largest / smallest
    print(f"[selftest] marker area is proportional to parameters ({smallest:g}B -> {small:.0f} pt2, "
          f"{largest:g}B -> {big:.0f} pt2); sqrt mode compresses the range to {ratio:.1f}x")


def synthetic_metrics(cfg, arms, derived) -> list[dict]:
    """Plausible metrics for every cell, so the figures can be rendered."""
    rows = []
    for index, arm in enumerate(list(arms) + list(derived)):
        for query in hx.QUERY_TYPES:
            for role in (a["role"] for a in cfg["answerers"]):
                for depth in cfg["depths"]:
                    base = 0.9 if query == "target" else 0.85
                    value = max(0.05, base - 0.03 * depth - 0.01 * (index % 5))
                    row = {
                        "arm": arm.name, "block": arm.block, "derived": arm.is_derived,
                        "size_pattern": arm.size_pattern, "family_mode": arm.family_mode,
                        "query_type": query, "answerer_role": role, "depth": depth, "n": 20,
                        "n_family_transitions": 0 if depth == 0 else max(0, depth - 1),
                        "n_tier_transitions": 0, "mean_parameters_b": 8.0,
                        "chain_short_codes": "L8-Q8", "material_characters_mean": 900.0,
                    }
                    for metric in ("em", "f1", "judge_correct"):
                        row[metric] = round(value, 4)
                        row[f"{metric}_lo"] = round(value - 0.08, 4)
                        row[f"{metric}_hi"] = round(value + 0.08, 4)
                    rows.append(row)
    return rows


def synthetic_transition_metrics() -> list[dict]:
    rows = []
    for family_changed in (False, True):
        for direction in ("same", "larger", "smaller"):
            prefix = "cross_family" if family_changed else "same_family"
            row = {"scope": "by_type", "transition_type": f"{prefix}_size_{direction}",
                   "to_stage": "all", "family_changed": family_changed,
                   "size_direction": direction, "n_edges": 100, "n_pairs": 20}
            for field, value in (("semantic_preserved", 0.8), ("fact_B_survived", 0.4),
                                 ("lexical_similarity", 0.75), ("length_ratio", 0.9)):
                row[field] = value
                row[f"{field}_lo"] = value - 0.07
                row[f"{field}_hi"] = value + 0.07
            rows.append(row)
    return rows


def synthetic_transition_deltas() -> list[dict]:
    rows = []
    for contrast, measure, delta, lo, hi in (
        ("cross_vs_same_family_at_matched_size", "semantic_preserved", -0.05, -0.09, -0.01),
        ("cross_vs_same_family_at_matched_size", "fact_B_survived", -0.10, -0.18, -0.02),
        ("cross_vs_same_family_at_matched_size", "lexical_similarity", -0.16, -0.20, -0.13),
        ("cross_vs_same_family_at_matched_size", "length_ratio", 0.26, 0.14, 0.39),
        ("cross_vs_same_family_when_growing", "semantic_preserved", -0.00, -0.03, 0.03),
        ("cross_vs_same_family_when_growing", "fact_B_survived", 0.01, -0.03, 0.07),
        ("cross_vs_same_family_when_growing", "lexical_similarity", -0.12, -0.17, -0.08),
        ("cross_vs_same_family_when_growing", "length_ratio", 0.19, 0.07, 0.32),
        ("cross_vs_same_family_when_shrinking", "semantic_preserved", 0.03, -0.01, 0.08),
        ("cross_vs_same_family_when_shrinking", "fact_B_survived", 0.00, -0.06, 0.08),
        ("cross_vs_same_family_when_shrinking", "lexical_similarity", -0.07, -0.11, -0.02),
        ("cross_vs_same_family_when_shrinking", "length_ratio", -0.08, -0.23, 0.04),
    ):
        rows.append({"comparison": contrast, "measure": measure, "delta": delta, "lo": lo, "hi": hi,
                     "p_value": 0.01 if lo * hi > 0 else 0.5, "n": 20})
    return rows


def test_transition_contrasts(cfg, pairs) -> None:
    """Paired edge contrasts must pair within a pair and recover a known delta."""
    edges = []
    for index, pair in enumerate(pairs):
        for stage in range(2, int(cfg["max_depth"]) + 1):
            for kind, similarity in (("same_family_size_same", 0.90),
                                     ("cross_family_size_same", 0.70)):
                edges.append({
                    "pair_id": pair["pair_id"], "to_stage": stage, "transition_type": kind,
                    "family_changed": kind.startswith("cross"), "size_direction": "same",
                    # A per-pair offset makes a between-pair comparison noisy and
                    # a correctly paired one exact, so the test can tell them apart.
                    "lexical_similarity": similarity + 0.05 * index,
                    "novel_token_rate": 0.2, "retained_token_rate": 0.8, "ngram_copy_rate": 0.5,
                    "length_ratio": 1.0, "current_characters": 400,
                    "semantic_preserved": 1.0, "fact_A_survived": 1, "fact_B_survived": 0,
                })
    deltas = hx.compute_transition_contrasts(edges, cfg)
    assert deltas, "no per-edge contrasts computed"
    row = next(d for d in deltas if d["comparison"] == "cross_vs_same_family_at_matched_size"
               and d["measure"] == "lexical_similarity")
    assert abs(row["delta"] + 0.20) < 1e-9, f"paired delta {row['delta']} should be -0.20"
    assert row["n"] == len(pairs), "every pair holds both edge types, so all should be paired"
    # Every pair's difference is the same, so a correctly *paired* bootstrap
    # collapses to a point. An unpaired one would inherit the 0.05-per-pair
    # offset and produce a visibly wide interval instead.
    assert row["hi"] - row["lo"] < 1e-9, (
        f"paired interval [{row['lo']}, {row['hi']}] should be a point; a wide one means the "
        "per-pair pairing was lost")
    # A contrast whose types never co-occur must be skipped, not silently empty.
    missing = copy.deepcopy(cfg)
    missing["transition_contrasts"] = [{"name": "impossible", "treatment": "cross_family_size_larger",
                                        "control": "same_family_size_larger"}]
    assert hx.compute_transition_contrasts(edges, missing) == []
    print(f"[selftest] paired edge contrasts recover an exact -0.200 delta over {row['n']} pairs")


def test_plots(cfg, pairs, registry, arms, derived, schedules) -> None:
    metrics = synthetic_metrics(cfg, arms, derived)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hx.make_primary_plot(metrics, cfg, "target", root / "primary.png")
        hx.make_primary_plot(metrics, cfg, "heldout", root / "heldout.png")
        hx.make_comparison_plot(metrics, cfg, root / "comparison.png")
        hx.make_trajectory_plot(metrics, cfg, schedules, pairs, registry, arms, derived,
                                root / "trajectories.png")
        hx.make_transition_plot(synthetic_transition_metrics(), synthetic_transition_deltas(), cfg,
                                root / "transitions.png")
        for name in ("primary.png", "heldout.png", "comparison.png", "trajectories.png", "transitions.png"):
            size = (root / name).stat().st_size
            assert size > 20_000, f"{name} rendered at only {size} bytes"
            print(f"[selftest] {name} rendered ({size // 1024} KiB)")


def test_analysis(cfg, pairs, arms, derived, registry, schedules) -> None:
    """Metric and contrast machinery on deterministic synthetic answers."""
    answered = {}
    for role in (a["role"] for a in cfg["answerers"]):
        for position, pair in enumerate(pairs):
            materials = {"direct"}
            for arm in arms:
                keys = schedules[(arm.name, pair["pair_id"])]
                materials.update(hx.prefix_id(keys, d) for d in cfg["depths"] if d)
            for material in materials:
                for query in hx.QUERY_TYPES:
                    depth = 0 if material == "direct" else material.count("|") + 1
                    score = max(0.0, 1.0 - 0.1 * depth - 0.02 * position)
                    answered[(role, pair["pair_id"], material, query)] = {
                        "answerer_role": role, "answerer_model_key": "llama_small",
                        "answerer_model_id": "x", "pair_id": pair["pair_id"],
                        "material_key": material, "query_type": query,
                        "material_characters": 900, "question": "q", "golds": ["g"],
                        "pred": "g", "raw": "g", "em": round(score, 4), "f1": round(score, 4),
                        "judge_correct": 1 if score > 0.5 else 0, "cached": True,
                        "prompt_tokens": 10, "completion_tokens": 2,
                    }
    rows = hx.expand_answers(answered, pairs, arms, derived, schedules, cfg, registry)
    assert rows, "expansion produced no rows"
    cells = {hx.cell_key(r) for r in rows}
    expected = (len(arms) + len(derived)) * len(hx.QUERY_TYPES) * len(cfg["answerers"]) * len(cfg["depths"])
    assert len(cells) == expected, f"{len(cells)} cells, expected {expected}"

    metrics = hx.compute_metrics(rows, cfg)
    assert all(r["n"] == len(pairs) for r in metrics), "a cell is missing pairs"
    # A derived control must equal, per pair, the homogeneous arm it selects.
    for position, pair in enumerate(pairs):
        family = mp.start_family(position, cfg["family_cycle"])
        left = [r for r in rows if r["arm"] == "homog_small_matched"
                and r["pair_id"] == pair["pair_id"]]
        right = [r for r in rows if r["arm"] == f"homog_small_{family}"
                 and r["pair_id"] == pair["pair_id"]]
        assert [r["f1"] for r in left] == [r["f1"] for r in right]

    deltas = hx.compute_deltas(rows, cfg)
    assert deltas, "no contrasts computed"
    names = {d["comparison"] for d in deltas}
    assert {c["name"] for c in cfg["contrasts"]} <= names, "a configured contrast was dropped"
    did = [d for d in deltas if d["kind"] == "did"]
    assert did and all(int(d["depth"]) > int(d["from_depth"]) for d in did)
    # Identical synthetic scores across arms make every contrast exactly zero;
    # that is the right sanity check for a paired-difference implementation.
    assert all(abs(d["delta"]) < 1e-9 for d in deltas), "paired contrast is not zero on identical inputs"
    print(f"[selftest] {len(metrics)} metric cells and {len(deltas)} contrast rows; "
          f"paired deltas are exactly zero on identical arms")


def test_family_transition_table(cfg, pairs, arms, derived, registry, schedules) -> None:
    rows = []
    for arm in list(arms) + list(derived):
        for position, pair in enumerate(pairs):
            keys = schedules[(arm.name, pair["pair_id"])]
            for depth in cfg["depths"]:
                if depth == 0:
                    continue
                summary = mp.chain_summary(keys[:depth], registry)
                rows.append({
                    "arm": arm.name, "block": arm.block, "derived": arm.is_derived,
                    "size_pattern": arm.size_pattern, "family_mode": arm.family_mode,
                    "pair_id": pair["pair_id"], "depth": depth, "query_type": "target",
                    "answerer_role": cfg["analysis"]["primary_answerer"],
                    "f1": 0.5, "judge_correct": 1, "material_characters": 900, **summary,
                })
    table = hx.compute_family_transition_metrics(rows, cfg)
    assert table, "no family-transition rows"
    at_max = [r for r in table if int(r["depth"]) == int(cfg["max_depth"])]
    counts = sorted({int(r["n_family_transitions"]) for r in at_max})
    assert 0 in counts and int(cfg["max_depth"]) - 1 in counts, (
        f"switch counts {counts} should span homogeneous to fully cross-family")
    print(f"[selftest] family-switch table spans counts {counts} at depth {cfg['max_depth']}")


def main() -> int:
    cfg, pairs, registry, arms, derived, schedules = load()
    test_prompts(cfg, pairs)
    test_schedules(cfg, pairs, registry, arms, derived, schedules)
    test_prefix_sharing(cfg, pairs, arms, schedules)
    test_transition_labels(registry)
    test_unknown_size_stays_unknown(cfg)
    test_marker_area(cfg, registry)
    test_analysis(cfg, pairs, arms, derived, registry, schedules)
    test_family_transition_table(cfg, pairs, arms, derived, registry, schedules)
    test_transition_contrasts(cfg, pairs)
    test_plots(cfg, pairs, registry, arms, derived, schedules)
    print("[selftest] model-heterogeneity offline checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
