"""Experiment 8: model heterogeneity across sequential agent handoffs.

Every earlier chain experiment holds the model fixed and varies something about
the message. This one holds the message pipeline fixed and varies *who writes
each stage*:

  block 1  matched size, small tier   -- four homogeneous families vs chains
                                         that cross family at every handoff
  block 2  matched size, large tier   -- the same comparison one tier up
  block 3  size trajectory x family   -- small->large, large->small and
                                         alternating sizes, each with family
                                         held constant and with family rotating

Everything else is inherited unchanged from Experiment 5's ``conditioned`` arm
on the same SQuAD same-passage pairs: dataset, context width, system prompt,
handoff instructions, question conditioning, temperature, token budgets, seed
policy, answer extraction, EM/F1, and the different-family LLM judge. The
handoff prompts never mention which model is writing, so the only thing that
changes between arms is the decoder.

Two controls make the comparison fair rather than merely different:

* **The answerer is constant.** Every arm and every depth is answered by the
  same model at temperature 0. If the last chain model also answered, a
  "large models late" trajectory would win because a large model answered, and
  nothing about the handoff would have been measured. A second answerer from
  another family runs alongside it to test the one confound a single fixed
  answerer cannot: whether a Llama answerer simply reads Llama-written notes
  more easily.
* **Start models are stratified and matched.** Each pair gets a start family by
  its index, so every family starts an equal share of pairs. A cross-family arm
  is compared against ``homog_*_matched``, which for each pair is the
  homogeneous chain built from *that pair's* start family. The two therefore
  share a byte-identical stage 1 and diverge only from stage 2 onward.

Stages 2+ receive a frozen ``SealedHandoff`` and nothing else, so the source
passages cannot re-enter a chain -- the project's load-bearing isolation
guarantee, asserted at startup here as it is in ``run.py`` and ``run_chain.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
import model_pool as mp  # noqa: E402
import paraphrase_metrics as pmx  # noqa: E402
from judge import add_judge, add_preservation_judge  # noqa: E402
from llm import CostCapExceeded, load_config  # noqa: E402
from run_chain import CHAIN_SYSTEM, INITIAL_INSTRUCTION, RECOMPRESS_INSTRUCTION  # noqa: E402
from run_slack_facts import mentions_answer  # noqa: E402
from run_summary_generalization import (  # noqa: E402
    answer as answer_from_material,
    compression_user_prompt,
    load_prebuilt_pairs,
    render_context,
)
from score import bootstrap_ci, paired_bootstrap_delta  # noqa: E402

# This experiment runs exactly one handoff mode. Naming it once here keeps the
# reused Experiment 5 prompt builder from being called with a mode that would
# silently change what is being measured.
MODE = "conditioned"
METRICS = ("em", "f1", "judge_correct")
QUERY_TYPES = ("target", "heldout")
# Kept out of transitions.jsonl: they are re-derived for free from
# handoffs.jsonl on every run, and storing every stage-1 source context would
# multiply the file by the size of the corpus.
TRANSITION_TEXT_FIELDS = ("previous_text", "current_text")


# ---------------------------------------------------------------- io helpers

def read_jsonl(path: Path) -> list[dict]:
    return [] if not path.exists() else [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(*parts: str) -> str:
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()


def _number(value, default=float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(result) else result


# ---------------------------------------------------------------- prompts

def recompress_user_prompt(sealed: hm.SealedHandoff) -> str:
    """Stage >= 2 prompt, built from the seal and nothing else.

    ``compression_user_prompt`` (Experiment 5) needs the whole pair dict, which
    would put ``render_context(pair)`` back inside a later stage's call frame.
    This rebuilds the identical string from the sealed fields instead;
    ``prompt_selftest`` asserts the two are byte-for-byte equal, so isolation is
    gained without the arms drifting from the published Experiment 5 prompt.
    """
    if not isinstance(sealed, hm.SealedHandoff):
        raise TypeError(
            f"recompress_user_prompt requires a SealedHandoff, got {type(sealed).__name__}. "
            "Source passages must not be reachable from a stage-2+ compression call."
        )
    return (
        f"Previous agent's notes:\n{sealed.handoff_text}\n\n"
        f"Question the final agent must answer: {sealed.question}\n\n"
        f"{RECOMPRESS_INSTRUCTION}"
    )


def prompt_selftest(pair: dict, cfg: dict) -> None:
    """Prove the model is invisible to the prompt and the seal is honoured."""
    notes = "identical previous notes"
    sealed = hm.seal(pair["question_A"], hm.Handoff(pair["pair_id"], "chain", None, notes))
    for stage in (2, 3, 6):
        assert recompress_user_prompt(sealed) == compression_user_prompt(pair, notes, MODE, stage, cfg), (
            "sealed recompression prompt drifted from Experiment 5's published prompt")
    stage1 = compression_user_prompt(pair, None, MODE, 1, cfg)
    assert INITIAL_INSTRUCTION in stage1 and pair["question_A"] in stage1
    # Nothing in either prompt names a model, a family, or a size: the only
    # thing that differs between arms is which decoder receives the string.
    for text in (stage1, recompress_user_prompt(sealed)):
        lowered = text.lower()
        for token in ("llama", "qwen", "mistral", "ministral", "gemma", "parameter count"):
            assert token not in lowered, f"handoff prompt leaks model identity: {token!r}"
    for bad in ("notes", {"handoff_text": "notes"}, object()):
        try:
            recompress_user_prompt(bad)
        except TypeError:
            continue
        raise AssertionError(f"isolation broken: recompression accepted {type(bad).__name__}")
    print("[hetero:selftest] prompts are model-blind, match Experiment 5, and stage 2+ is sealed")


def compress(pool: mp.ModelPool, model_key: str, pair: dict, previous_text: str | None,
             stage: int, cfg: dict) -> dict:
    """One handoff message, written by ``model_key``."""
    if stage == 1:
        user = compression_user_prompt(pair, None, MODE, stage, cfg)
    else:
        sealed = hm.seal(pair["question_A"],
                         hm.Handoff(pair["pair_id"], "chain", None, previous_text or ""))
        user = recompress_user_prompt(sealed)
    result = pool.client(model_key).chat(
        [{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        # Same seed policy as Experiments 5 and 6: deterministic decoding with
        # the seed keyed to the stage. Model identity already separates cache
        # entries, so no per-model seed offset is needed.
        seed=stage,
        tag=f"hetero_compress_{model_key}_stage{stage}",
    )
    text = result.text.strip()
    return {
        "text": text,
        "cached": result.cached,
        "handoff_prompt_tokens": result.prompt_tokens,
        "handoff_completion_tokens": result.completion_tokens,
        "handoff_finish_reason": result.finish_reason,
        "handoff_cost_usd": round(result.cost_usd, 8),
        "handoff_provider": result.provider,
        "handoff_resolved_model": result.model,
        "summary_characters": len(text),
        "summary_words": len(text.split()),
    }


# ---------------------------------------------------------------- schedules

def build_schedules(pairs: list[dict], arms: list[mp.ArmSpec], derived: list[mp.ArmSpec],
                    cfg: dict, registry: dict[str, mp.ModelSpec]) -> dict[tuple[str, str], list[str]]:
    index = mp.by_family_tier(registry)
    schedules: dict[tuple[str, str], list[str]] = {}
    for arm in arms:
        for position, pair in enumerate(pairs):
            schedules[(arm.name, pair["pair_id"])] = mp.schedule_for(arm, position, cfg, index)
    for arm in derived:
        for position, pair in enumerate(pairs):
            source = mp.resolve_derived(arm, position, cfg)
            key = (source, pair["pair_id"])
            if key not in schedules:
                raise ValueError(
                    f"derived arm {arm.name} resolves to {source!r}, which is not a generated arm")
            schedules[(arm.name, pair["pair_id"])] = list(schedules[key])
    return schedules


def prefix_id(model_keys: list[str], stage: int) -> str:
    return "|".join(model_keys[:stage])


def schedule_selftest(pairs: list[dict], arms: list[mp.ArmSpec], derived: list[mp.ArmSpec],
                      schedules: dict, cfg: dict, registry: dict[str, mp.ModelSpec]) -> None:
    """Check the design's structural claims before any money is spent."""
    cycle = cfg["family_cycle"]
    max_depth = int(cfg["max_depth"])
    by_name = {a.name: a for a in list(arms) + list(derived)}
    for pair in pairs:
        for name in by_name:
            keys = schedules[(name, pair["pair_id"])]
            assert len(keys) == max_depth, f"{name}: schedule has {len(keys)} stages, needs {max_depth}"
    # Start-matched controls must share stage 1 with the cross-family arms.
    for control, treatments in (("homog_small_matched", ("xfam_small_fwd", "xfam_small_rev")),
                                ("homog_large_matched", ("xfam_large_fwd",))):
        if control not in by_name:
            continue
        for treatment in treatments:
            if treatment not in by_name:
                continue
            for pair in pairs:
                assert (schedules[(treatment, pair["pair_id"])][0]
                        == schedules[(control, pair["pair_id"])][0]), (
                    f"{treatment} and {control} must start from the same model on {pair['pair_id']}")
    # Cross-family arms must actually cross family at every handoff; arms that
    # claim to hold family constant must never cross it.
    for name, arm in by_name.items():
        for pair in pairs:
            families = [registry[k].family for k in schedules[(name, pair["pair_id"])]]
            if arm.family_mode in ("cycle_forward", "cycle_reverse"):
                assert all(a != b for a, b in zip(families, families[1:])), \
                    f"{name} is a cross-family arm but repeats a family"
            else:
                assert len(set(families)) == 1, f"{name} is supposed to hold family constant"
    # Every family must start an equal share of pairs, or the start-matched
    # control is unbalanced and the cross-family contrast is not clean.
    starts = [mp.start_family(i, cycle) for i in range(len(pairs))]
    counts = {f: starts.count(f) for f in cycle}
    if len(set(counts.values())) != 1:
        print(f"[hetero:selftest] WARNING start families are unbalanced ({counts}); "
              f"n should be a multiple of {len(cycle)}")
    # up/down/alt must share a tier multiset at max_depth, so comparing them at
    # that depth is an ordering comparison and nothing else.
    patterns = cfg["size_patterns"]
    multisets = {name: tuple(sorted(patterns[name][:max_depth]))
                 for name in ("up", "down", "alt") if name in patterns}
    if len(set(multisets.values())) != 1:
        print(f"[hetero:selftest] WARNING size trajectories differ in composition at max_depth "
              f"({multisets}); the direction-of-change contrast is then confounded")
    print(f"[hetero:selftest] {len(by_name)} arms x {len(pairs)} pairs x {max_depth} stages; "
          f"start families {counts}")


# ---------------------------------------------------------------- generation

def generate_handoffs(pool: mp.ModelPool, pairs: list[dict], arms: list[mp.ArmSpec],
                      schedules: dict, cfg: dict, registry: dict[str, mp.ModelSpec],
                      run_root: Path, fingerprint: str,
                      dry_run: bool = False) -> dict[tuple[str, str], dict]:
    """Generate every stage, memoised on the *model prefix* rather than the arm.

    Two arms that agree on the first ``s`` models produce byte-identical stage-s
    messages, because decoding is deterministic and nothing in the prompt names
    the arm. Keying generation on the prefix means each such message is written
    once. That is not only cheaper: it makes the shared prefix exact rather than
    merely probable, which is what the start-matched contrast depends on.
    """
    path = run_root / "handoffs.jsonl"
    generated: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(path):
        if row.get("experiment_fingerprint") == fingerprint:
            generated.setdefault((row["pair_id"], row["chain_prefix"]), row)
    max_depth = int(cfg["max_depth"])

    for stage in range(1, max_depth + 1):
        todo: dict[tuple[str, str], tuple[dict, str, str | None]] = {}
        for arm in arms:
            for pair in pairs:
                keys = schedules[(arm.name, pair["pair_id"])]
                slot = (pair["pair_id"], prefix_id(keys, stage))
                if slot in generated or slot in todo:
                    continue
                previous = None
                if stage > 1:
                    previous = generated[(pair["pair_id"], prefix_id(keys, stage - 1))]["text"]
                todo[slot] = (pair, keys[stage - 1], previous)
        if todo:
            with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as executor:
                futures = {
                    executor.submit(compress, pool, model_key, pair, previous, stage, cfg): slot
                    for slot, (pair, model_key, previous) in todo.items()
                }
                for future, slot in futures.items():
                    _pair, model_key, _previous = todo[slot]
                    generated[slot] = {
                        "pair_id": slot[0], "chain_prefix": slot[1], "stage": stage,
                        "experiment_fingerprint": fingerprint,
                        **registry[model_key].to_json(), **future.result(),
                    }
        current = [r for r in generated.values() if int(r["stage"]) == stage]
        truncated = sum(1 for r in current if r.get("handoff_finish_reason") == "length")
        print(f"[hetero:handoff] stage {stage}: {len(todo)} distinct messages generated, "
              f"{len(current)} stored; {truncated} hit the token guard")
        # A dry run must not touch runs/: an estimate that overwrote the real
        # records with empty placeholders would poison the next real run.
        if not dry_run:
            write_jsonl(path, [generated[k] for k in sorted(generated)])
    return generated


# ---------------------------------------------------------------- answering

def run_answers(pool: mp.ModelPool, pairs: list[dict], arms: list[mp.ArmSpec], schedules: dict,
                generated: dict, cfg: dict, registry: dict[str, mp.ModelSpec], run_root: Path,
                fingerprint: str, dry_run: bool = False) -> dict[tuple, dict]:
    """Answer every distinct (answerer, pair, material, query) once, then judge.

    The answer depends only on those four things -- never on which arm routed
    the material here -- so an arm that shares a chain prefix with another arm
    shares its answers exactly rather than approximately.
    """
    path = run_root / "answers.jsonl"
    answered: dict[tuple, dict] = {
        (r["answerer_role"], r["pair_id"], r["material_key"], r["query_type"]): r
        for r in read_jsonl(path) if r.get("experiment_fingerprint") == fingerprint
    }
    answerers = {a["role"]: a["model"] for a in cfg["answerers"]}
    depths = [int(d) for d in cfg["depths"]]

    todo: dict[tuple, tuple[dict, str]] = {}
    for pair in pairs:
        materials = {"direct": render_context(pair)} if 0 in depths else {}
        for arm in arms:
            keys = schedules[(arm.name, pair["pair_id"])]
            for depth in depths:
                if depth == 0:
                    continue
                key = prefix_id(keys, depth)
                materials.setdefault(key, generated[(pair["pair_id"], key)]["text"])
        for role in answerers:
            for query in QUERY_TYPES:
                for material_key, material in materials.items():
                    slot = (role, pair["pair_id"], material_key, query)
                    if slot not in answered:
                        todo[slot] = (pair, material)

    if todo:
        with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as executor:
            futures = {
                executor.submit(
                    answer_from_material, pool.client(answerers[slot[0]]), pair, slot[3], material,
                    cfg, f"hetero_answer_{slot[0]}_{slot[3]}",
                ): slot
                for slot, (pair, material) in todo.items()
            }
            for future, slot in futures.items():
                answered[slot] = {
                    "answerer_role": slot[0], "answerer_model_key": answerers[slot[0]],
                    "answerer_model_id": registry[answerers[slot[0]]].id,
                    "pair_id": slot[1], "material_key": slot[2], "query_type": slot[3],
                    "material_characters": len(todo[slot][1]),
                    "experiment_fingerprint": fingerprint, **future.result(),
                }
    print(f"[hetero:answers] {len(todo)} distinct answers generated, {len(answered)} stored")
    if not dry_run:
        write_jsonl(path, [answered[k] for k in sorted(answered)])
    return answered


def degenerate_prefixes(generated: dict) -> set[tuple[str, str]]:
    """Slots whose model returned no text at all.

    A provider that answers with an empty body still reports a finish reason and
    a completion-token count, so an empty handoff is invisible unless it is
    looked for. Every later stage of that chain then compresses nothing, which
    would otherwise be scored as if the chain had simply lost information.
    """
    return {slot for slot, row in generated.items() if not str(row.get("text", "")).strip()}


def expand_answers(answered: dict, pairs: list[dict], arms: list[mp.ArmSpec],
                   derived: list[mp.ArmSpec], schedules: dict, cfg: dict,
                   registry: dict[str, mp.ModelSpec], degenerate: set | None = None) -> list[dict]:
    """Attach each distinct answer to every (arm, depth) that produced it."""
    depths = [int(d) for d in cfg["depths"]]
    roles = [a["role"] for a in cfg["answerers"]]
    rows: list[dict] = []
    for arm in list(arms) + list(derived):
        for position, pair in enumerate(pairs):
            keys = schedules[(arm.name, pair["pair_id"])]
            for depth in depths:
                material_key = "direct" if depth == 0 else prefix_id(keys, depth)
                summary = mp.chain_summary(keys[:depth], registry) if depth else {}
                degraded = any(
                    (pair["pair_id"], prefix_id(keys, stage)) in (degenerate or set())
                    for stage in range(1, depth + 1)
                )
                for role in roles:
                    for query in QUERY_TYPES:
                        base = answered.get((role, pair["pair_id"], material_key, query))
                        if base is None:
                            continue
                        rows.append({
                            "arm": arm.name, "block": arm.block, "derived": arm.is_derived,
                            "size_pattern": arm.size_pattern, "family_mode": arm.family_mode,
                            "pair_id": pair["pair_id"], "pair_index": position,
                            "start_family": mp.start_family(position, cfg["family_cycle"]),
                            "depth": depth, "query_type": query, "answerer_role": role,
                            "chain_id": f"{arm.name}::{pair['pair_id']}",
                            "chain_has_empty_handoff": int(degraded),
                            "material_key": material_key, **summary,
                            **{k: v for k, v in base.items() if k != "experiment_fingerprint"},
                        })
    return rows


# ---------------------------------------------------------------- transitions

def build_transitions(pairs: list[dict], arms: list[mp.ArmSpec], schedules: dict, generated: dict,
                      cfg: dict, registry: dict[str, mp.ModelSpec], run_root: Path,
                      fingerprint: str) -> list[dict]:
    """One row per distinct handoff edge: what changed, and what survived it.

    Rebuilt from scratch on every run -- every deterministic measure is free --
    and only the judged verdict is carried over, and only when the exact message
    pair was judged before. That is Experiment 5a's rule, kept identical.
    """
    stored = {
        (r["pair_id"], r["chain_prefix"]): r
        for r in read_jsonl(run_root / "transitions.jsonl")
        if r.get("experiment_fingerprint") == fingerprint
    }
    max_depth = int(cfg["max_depth"])
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for arm in arms:
        for position, pair in enumerate(pairs):
            keys = schedules[(arm.name, pair["pair_id"])]
            for stage in range(1, max_depth + 1):
                slot = (pair["pair_id"], prefix_id(keys, stage))
                if slot in seen:
                    continue
                seen.add(slot)
                record = generated[slot]
                current = record["text"]
                if stage == 1:
                    previous, from_spec = render_context(pair), None
                else:
                    previous = generated[(pair["pair_id"], prefix_id(keys, stage - 1))]["text"]
                    from_spec = registry[keys[stage - 2]]
                to_spec = registry[keys[stage - 1]]
                row = {
                    "pair_id": pair["pair_id"], "pair_index": position,
                    "chain_prefix": slot[1], "to_stage": stage,
                    "experiment_fingerprint": fingerprint,
                    "from_model_key": None if from_spec is None else from_spec.key,
                    "from_family": None if from_spec is None else from_spec.family,
                    "from_tier": None if from_spec is None else from_spec.tier,
                    "from_parameters_b": None if from_spec is None else from_spec.parameters_b,
                    "to_model_key": to_spec.key, "to_model_id": to_spec.id,
                    "to_family": to_spec.family, "to_tier": to_spec.tier,
                    "to_parameters_b": to_spec.parameters_b,
                    **mp.transition_type(from_spec, to_spec),
                    **pmx.transition_measures(previous, current),
                    "fact_A_survived": int(mentions_answer(current, pair["golds_A"])),
                    "fact_B_survived": int(mentions_answer(current, pair["golds_B"])),
                    "handoff_prompt_tokens": record.get("handoff_prompt_tokens"),
                    "handoff_completion_tokens": record.get("handoff_completion_tokens"),
                    "handoff_cost_usd": record.get("handoff_cost_usd"),
                    "handoff_provider": record.get("handoff_provider"),
                    "handoff_finish_reason": record.get("handoff_finish_reason"),
                    "message_pair_hash": digest(previous, current),
                    "previous_text": previous, "current_text": current,
                }
                reuse = stored.get(slot)
                if reuse is not None and reuse.get("message_pair_hash") == row["message_pair_hash"]:
                    for field in ("semantic_preserved", "semantic_verdict", "semantic_parse_ok",
                                  "semantic_raw"):
                        if field in reuse:
                            row[field] = reuse[field]
                rows.append(row)
    print(f"[hetero:transitions] {len(rows)} distinct handoff edges")
    return rows


def attach_arms(transitions: list[dict], pairs: list[dict], arms: list[mp.ArmSpec],
                derived: list[mp.ArmSpec], schedules: dict, cfg: dict) -> list[dict]:
    """Expand distinct edges back out to (arm, pair, stage) rows for analysis."""
    by_slot = {(r["pair_id"], r["chain_prefix"]): r for r in transitions}
    max_depth = int(cfg["max_depth"])
    expanded: list[dict] = []
    for arm in list(arms) + list(derived):
        for pair in pairs:
            keys = schedules[(arm.name, pair["pair_id"])]
            for stage in range(1, max_depth + 1):
                edge = by_slot[(pair["pair_id"], prefix_id(keys, stage))]
                expanded.append({
                    "arm": arm.name, "block": arm.block, "derived": arm.is_derived,
                    "chain_id": f"{arm.name}::{pair['pair_id']}",
                    **{k: v for k, v in edge.items()
                       if k not in TRANSITION_TEXT_FIELDS + ("experiment_fingerprint",)},
                })
    return expanded


# ---------------------------------------------------------------- judging

def judge_deduplicated(rows: list[dict], key_fields: tuple, cfg: dict, judger, tag: str):
    """Run a judge over de-duplicated rows, then broadcast the verdict back.

    Distinct arms routinely reach the same answer for the same pair, and the
    same message pair at the same edge. Judging every copy would fire concurrent
    identical live calls before any of them reached the cache, so duplicates are
    collapsed first and the verdict copied onto every twin.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(digest(str(row.get(f, ""))) for f in key_fields), []).append(row)
    client = judger([group[0] for group in groups.values()], cfg, tag=tag)
    for group in groups.values():
        head = group[0]
        verdict = {k: v for k, v in head.items() if k.startswith(("judge_", "semantic_"))}
        for twin in group[1:]:
            twin.update(verdict)
    return client


# ---------------------------------------------------------------- analysis

def cell_key(row: dict) -> tuple:
    return (row["arm"], row["query_type"], row["answerer_role"], int(row["depth"]))


def paired_values(rows: list[dict], arm: str, query: str, role: str, depth: int,
                  metric: str) -> dict[str, float]:
    return {
        r["pair_id"]: float(r[metric]) for r in rows
        if r["arm"] == arm and r["query_type"] == query and r["answerer_role"] == role
        and int(r["depth"]) == depth and r.get(metric) is not None
    }


def chain_label(subset: list[dict]) -> str:
    """One model sequence if the arm has one, otherwise how many rotations."""
    sequences = {r.get("chain_short_codes", "") for r in subset}
    if len(sequences) == 1:
        return next(iter(sequences))
    return f"rotates ({len(sequences)} start models)"


def compute_metrics(rows: list[dict], cfg: dict) -> list[dict]:
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    metrics: list[dict] = []
    for key in sorted({cell_key(r) for r in rows}):
        arm, query, role, depth = key
        subset = sorted([r for r in rows if cell_key(r) == key], key=lambda r: r["pair_id"])
        head = subset[0]
        record = {
            "arm": arm, "block": head["block"], "derived": head["derived"],
            "size_pattern": head["size_pattern"], "family_mode": head["family_mode"],
            "query_type": query, "answerer_role": role, "depth": depth, "n": len(subset),
            "n_family_transitions": int(head.get("n_family_transitions") or 0) if depth else 0,
            "n_tier_transitions": int(head.get("n_tier_transitions") or 0) if depth else 0,
            "mean_parameters_b": head.get("mean_parameters_b") if depth else "",
            # An arm whose family rotates per pair has no single model sequence.
            # Reporting the first pair's would read as if the whole arm used it.
            "chain_short_codes": chain_label(subset) if depth else "direct",
            "material_characters_mean": round(
                float(np.mean([r["material_characters"] for r in subset])), 1),
            # Share of this cell's chains that ran through an empty handoff. A
            # non-zero value means the cell measures a generation failure as
            # well as a compression effect, and must be read with deltas_sensitivity.csv.
            "empty_handoff_chain_fraction": round(
                float(np.mean([int(r.get("chain_has_empty_handoff", 0)) for r in subset])), 4),
        }
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in subset if r.get(metric) is not None])
            if values.size == 0:
                continue
            mean, lo, hi = bootstrap_ci(values, boot, ci, seed=101)
            record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4),
                           f"{metric}_hi": round(hi, 4)})
        metrics.append(record)
    return metrics


def compute_deltas(rows: list[dict], cfg: dict, drop_degenerate: bool = False) -> list[dict]:
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    depths = [int(d) for d in cfg["depths"] if int(d) > 0]
    roles = [a["role"] for a in cfg["answerers"]]
    available = {r["arm"] for r in rows}
    deltas: list[dict] = []
    for contrast in cfg["contrasts"]:
        treatment, control = contrast["treatment"], contrast["control"]
        if treatment not in available or control not in available:
            print(f"[hetero:deltas] skipping {contrast['name']}: arm not present in this run")
            continue
        kind = contrast.get("kind", "depth")
        base_depth = int(contrast.get("from_depth", 0))
        for query in QUERY_TYPES:
            for role in roles:
                for metric in METRICS:
                    for depth in depths:
                        if kind == "did" and depth <= base_depth:
                            continue
                        left = paired_values(rows, treatment, query, role, depth, metric)
                        right = paired_values(rows, control, query, role, depth, metric)
                        ids = sorted(set(left) & set(right))
                        if drop_degenerate:
                            tainted = degenerate_pairs(rows, (treatment, control), depth)
                            ids = [i for i in ids if i not in tainted]
                        if kind == "did":
                            left_base = paired_values(rows, treatment, query, role, base_depth, metric)
                            right_base = paired_values(rows, control, query, role, base_depth, metric)
                            ids = sorted(set(ids) & set(left_base) & set(right_base))
                            if not ids:
                                continue
                            treated = np.array([left[i] - left_base[i] for i in ids])
                            controlled = np.array([right[i] - right_base[i] for i in ids])
                        else:
                            if not ids:
                                continue
                            treated = np.array([left[i] for i in ids])
                            controlled = np.array([right[i] for i in ids])
                        result = paired_bootstrap_delta(treated, controlled, boot, ci, seed=103)
                        deltas.append({
                            "comparison": contrast["name"], "kind": kind,
                            "treatment": treatment, "control": control,
                            "from_depth": base_depth if kind == "did" else "",
                            "query_type": query, "answerer_role": role, "depth": depth,
                            "metric": metric, **result,
                        })
    return deltas


TRANSITION_FIELDS = ("lexical_similarity", "novel_token_rate", "retained_token_rate",
                     "ngram_copy_rate", "length_ratio", "current_characters",
                     "semantic_preserved", "fact_A_survived", "fact_B_survived")


def compute_transition_metrics(edges: list[dict], cfg: dict) -> list[dict]:
    """Stage-level loss, grouped by what changed at the edge.

    The bootstrap resamples *pairs*, not edges, averaging a pair's edges first.
    That is the project's convention everywhere else and it respects the fact
    that edges from one pair are not independent observations.
    """
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    out: list[dict] = []
    groupings = (("by_type", lambda r: (r["transition_type"], "all")),
                 ("by_type_stage", lambda r: (r["transition_type"], int(r["to_stage"]))))
    for scope, keyer in groupings:
        for group_key in sorted({keyer(r) for r in edges}, key=lambda k: (k[0], str(k[1]))):
            subset = [r for r in edges if keyer(r) == group_key]
            record = {
                "scope": scope, "transition_type": group_key[0], "to_stage": group_key[1],
                "family_changed": subset[0]["family_changed"],
                "size_direction": subset[0]["size_direction"],
                "n_edges": len(subset), "n_pairs": len({r["pair_id"] for r in subset}),
            }
            for field in TRANSITION_FIELDS:
                by_pair: dict[str, list[float]] = {}
                for row in subset:
                    value = _number(row.get(field))
                    if np.isnan(value):
                        continue
                    by_pair.setdefault(row["pair_id"], []).append(value)
                if not by_pair:
                    continue
                values = np.array([float(np.mean(v)) for v in by_pair.values()])
                mean, lo, hi = bootstrap_ci(values, boot, ci, seed=107)
                record.update({field: round(mean, 4), f"{field}_lo": round(lo, 4),
                               f"{field}_hi": round(hi, 4)})
            out.append(record)
    return out


def degenerate_pairs(rows: list[dict], arms: tuple[str, ...], depth: int) -> set[str]:
    """Pairs that either arm of a contrast routed through an empty handoff.

    Scoped to the two arms and the depth actually being compared. Excluding
    every pair that *any* arm degraded would drop 13 of 20 here purely because
    one homogeneous arm is affected, needlessly shrinking contrasts that arm
    never enters.
    """
    return {
        r["pair_id"] for r in rows
        if r["arm"] in arms and int(r["depth"]) <= depth
        and int(r.get("chain_has_empty_handoff", 0))
    }


def compute_transition_contrasts(edges: list[dict], cfg: dict) -> list[dict]:
    """Paired per-pair contrasts between two kinds of handoff edge.

    ``compute_transition_metrics`` reports each edge type on its own, so a
    difference between two of them could only be read off two independent
    intervals. This pairs them the way every other contrast in the project is
    paired: for each pair, average that pair's edges of each type, then
    bootstrap the per-pair difference. Only pairs contributing at least one edge
    of both types are used, so the comparison is within-pair.
    """
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    by_type_pair: dict[tuple[str, str], list[dict]] = {}
    for edge in edges:
        by_type_pair.setdefault((edge["transition_type"], edge["pair_id"]), []).append(edge)
    out: list[dict] = []
    for contrast in cfg.get("transition_contrasts") or []:
        treatment, control = contrast["treatment"], contrast["control"]
        shared = sorted({p for t, p in by_type_pair if t == treatment}
                        & {p for t, p in by_type_pair if t == control})
        if not shared:
            print(f"[hetero:transition-deltas] skipping {contrast['name']}: no pair holds both types")
            continue
        for field in TRANSITION_FIELDS:
            left, right = [], []
            for pair_id in shared:
                a = [v for v in (_number(e.get(field)) for e in by_type_pair[(treatment, pair_id)])
                     if not np.isnan(v)]
                b = [v for v in (_number(e.get(field)) for e in by_type_pair[(control, pair_id)])
                     if not np.isnan(v)]
                if not a or not b:
                    continue
                left.append(float(np.mean(a)))
                right.append(float(np.mean(b)))
            if not left:
                continue
            result = paired_bootstrap_delta(np.array(left), np.array(right), boot, ci, seed=113)
            out.append({"comparison": contrast["name"], "treatment": treatment, "control": control,
                        "measure": field, "treatment_mean": round(float(np.mean(left)), 4),
                        "control_mean": round(float(np.mean(right)), 4), **result})
    return out


def compute_family_transition_metrics(rows: list[dict], cfg: dict) -> list[dict]:
    """Accuracy against the number of family switches, always *within* a depth.

    Depth and switch count are collinear inside one arm (a fully cross-family
    chain of depth d has d-1 switches), so a pooled regression over depths would
    be reporting depth twice. Reporting within a depth is the honest cut; the
    size-trajectory arms are what supply intermediate switch counts there.
    """
    boot = int(cfg["analysis"]["bootstrap_resamples"])
    ci = float(cfg["analysis"]["ci_level"])
    role = cfg["analysis"]["primary_answerer"]
    out: list[dict] = []
    for query in QUERY_TYPES:
        for depth in sorted({int(r["depth"]) for r in rows if int(r["depth"]) > 0}):
            subset = [r for r in rows if int(r["depth"]) == depth and r["query_type"] == query
                      and r["answerer_role"] == role and not r["derived"]]
            for switches in sorted({int(r["n_family_transitions"] or 0) for r in subset}):
                cell = [r for r in subset if int(r["n_family_transitions"] or 0) == switches]
                record = {"query_type": query, "depth": depth, "n_family_transitions": switches,
                          "n_rows": len(cell), "n_arms": len({r["arm"] for r in cell}),
                          "arms": ",".join(sorted({r["arm"] for r in cell}))}
                for metric in ("f1", "judge_correct"):
                    by_pair: dict[str, list[float]] = {}
                    for row in cell:
                        if row.get(metric) is None:
                            continue
                        by_pair.setdefault(row["pair_id"], []).append(float(row[metric]))
                    if not by_pair:
                        continue
                    values = np.array([float(np.mean(v)) for v in by_pair.values()])
                    mean, lo, hi = bootstrap_ci(values, boot, ci, seed=109)
                    record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4),
                                   f"{metric}_hi": round(hi, 4)})
                out.append(record)
    return out


def label_transitions(edges: list[dict], rows: list[dict], cfg: dict) -> None:
    """Attach Experiment 5a's edge buckets, once per evaluated question."""
    role = cfg["analysis"]["primary_answerer"]
    lookup = {
        (r["arm"], r["pair_id"], int(r["depth"]), r["query_type"]): r for r in rows
        if r["answerer_role"] == role and int(r["depth"]) > 0
    }
    for edge in edges:
        for query, suffix in (("target", "A"), ("heldout", "B")):
            answer_row = lookup.get((edge["arm"], edge["pair_id"], int(edge["to_stage"]), query))
            correct = None if answer_row is None else answer_row.get("judge_correct")
            preserved = _number(edge.get("semantic_preserved"), default=float("nan"))
            edge[f"transition_label_{query}"] = pmx.classify_transition(
                float(edge["lexical_similarity"]),
                None if np.isnan(preserved) else preserved,
                bool(edge[f"fact_{suffix}_survived"]), correct,
            )


def build_stage_records(edges: list[dict], rows: list[dict], cfg: dict) -> list[dict]:
    """The per-stage audit log required of this experiment.

    One row per (chain, stage): who wrote it, how big that model is and on what
    authority, what the edge did to the message, and how the chain scored at
    that depth on both the conditioning target and the held-out question.
    """
    role = cfg["analysis"]["primary_answerer"]
    scores: dict[tuple, dict] = {}
    for row in rows:
        if row["answerer_role"] != role or int(row["depth"]) == 0:
            continue
        scores.setdefault((row["arm"], row["pair_id"], int(row["depth"])), {})[row["query_type"]] = row
    out: list[dict] = []
    for edge in edges:
        cell = scores.get((edge["arm"], edge["pair_id"], int(edge["to_stage"])), {})
        target, heldout = cell.get("target", {}), cell.get("heldout", {})
        out.append({
            "chain_id": edge["chain_id"], "arm": edge["arm"], "block": edge["block"],
            "derived": edge["derived"], "pair_id": edge["pair_id"],
            "stage": edge["to_stage"], "depth": edge["to_stage"],
            "model_id": edge["to_model_id"], "model_key": edge["to_model_key"],
            "family": edge["to_family"], "tier": edge["to_tier"],
            "parameters_b": edge["to_parameters_b"],
            "from_model_key": edge["from_model_key"], "from_parameters_b": edge["from_parameters_b"],
            "transition_type": edge["transition_type"], "family_changed": edge["family_changed"],
            "size_direction": edge["size_direction"], "size_ratio": edge["size_ratio"],
            "log_size_change": edge["log_size_change"],
            "handoff_prompt_tokens": edge.get("handoff_prompt_tokens"),
            "handoff_completion_tokens": edge.get("handoff_completion_tokens"),
            "handoff_cost_usd": edge.get("handoff_cost_usd"),
            "handoff_provider": edge.get("handoff_provider"),
            "handoff_finish_reason": edge.get("handoff_finish_reason"),
            "previous_characters": edge["previous_characters"],
            "handoff_characters": edge["current_characters"],
            "handoff_tokens": edge["current_tokens"],
            "length_ratio": edge["length_ratio"],
            "lexical_similarity": edge["lexical_similarity"],
            "novel_token_rate": edge["novel_token_rate"],
            "ngram_copy_rate": edge["ngram_copy_rate"],
            "semantic_preserved": edge.get("semantic_preserved"),
            "semantic_verdict": edge.get("semantic_verdict"),
            "fact_A_survived": edge["fact_A_survived"], "fact_B_survived": edge["fact_B_survived"],
            "transition_label_target": edge.get("transition_label_target"),
            "transition_label_heldout": edge.get("transition_label_heldout"),
            "answerer_role": role, "answerer_model_id": target.get("answerer_model_id"),
            "target_pred": target.get("pred"), "target_em": target.get("em"),
            "target_f1": target.get("f1"), "target_judge_correct": target.get("judge_correct"),
            "heldout_pred": heldout.get("pred"), "heldout_f1": heldout.get("f1"),
            "heldout_judge_correct": heldout.get("judge_correct"),
            "answer_prompt_tokens": target.get("prompt_tokens"),
            "answer_completion_tokens": target.get("completion_tokens"),
        })
    return out


# ---------------------------------------------------------------- plots

# Plain-language names for metric keys, wherever a metric name is drawn
# straight onto a figure rather than used as an internal dict key.
METRIC_LABELS = {"f1": "Token F1", "judge_correct": "LLM-judge accuracy", "em": "Exact match"}


def _series(metrics: list[dict], arm: str, query: str, role: str, metric: str):
    subset = sorted([r for r in metrics if r["arm"] == arm and r["query_type"] == query
                     and r["answerer_role"] == role and int(r["depth"]) > 0 and metric in r],
                    key=lambda r: int(r["depth"]))
    x = [int(r["depth"]) for r in subset]
    y = [float(r[metric]) for r in subset]
    lower = [float(r[metric]) - float(r[f"{metric}_lo"]) for r in subset]
    upper = [float(r[f"{metric}_hi"]) - float(r[metric]) for r in subset]
    return x, y, [lower, upper]


def primary_panels(cfg: dict) -> list[tuple]:
    """(title, thin reference curves, heavy contrast curves) for each column."""
    colours = {k: v["color"] for k, v in cfg["families"].items()}
    return [
        ("Matched size — small tier",
         [(f"homog_small_{f}", colours[f], "-", 1.3, 0.7, f"{f}, same model the whole chain")
          for f in cfg["family_cycle"]],
         [("homog_small_matched", "#111111", "-", 2.6, "same family the whole chain"),
          ("xfam_small_fwd", "#d7191c", "--", 2.6, "different family at every step"),
          ("xfam_small_rev", "#d7191c", ":", 2.2, "different family at every step (reverse order)")]),
        ("Matched size — large tier",
         [(f"homog_large_{f}", colours[f], "-", 1.3, 0.7, f"{f}, same model the whole chain")
          for f in cfg["family_cycle"]],
         [("homog_large_matched", "#111111", "-", 2.6, "same family the whole chain"),
          ("xfam_large_fwd", "#d7191c", "--", 2.6, "different family at every step")]),
        ("Size trajectory — family fixed vs. family also changing",
         [("homog_small_matched", "#777777", "-", 1.6, 0.9, "always small"),
          ("homog_large_matched", "#777777", "--", 1.6, 0.9, "always large"),
          ("size_up_xfam", "#2c7fb8", ":", 1.7, 0.85, "small→large, family also changes"),
          ("size_down_xfam", "#e6550d", ":", 1.7, 0.85, "large→small, family also changes"),
          ("size_alt_xfam", "#31a354", ":", 1.7, 0.85, "alternating, family also changes")],
         [("size_up_samefam", "#2c7fb8", "-", 2.6, "small→large, one family"),
          ("size_down_samefam", "#e6550d", "-", 2.6, "large→small, one family"),
          ("size_alt_samefam", "#31a354", "-", 2.6, "alternating, one family")]),
    ]


def make_primary_plot(metrics: list[dict], cfg: dict, query: str, output: Path) -> None:
    """Accuracy against handoff depth for the major conditions.

    Homogeneous chains are thin lines in their family colour, the start-matched
    homogeneous control is a heavy black line, and every heterogeneous chain is
    a heavy coloured line, so "does crossing families cost anything" is a
    comparison between two line weights rather than a hunt through a legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    role = cfg["analysis"]["primary_answerer"]
    depths = [int(d) for d in cfg["depths"] if int(d) > 0]
    panels = primary_panels(cfg)
    plot_metrics = [("f1", METRIC_LABELS["f1"]), ("judge_correct", METRIC_LABELS["judge_correct"])]
    fig, axes = plt.subplots(len(plot_metrics), len(panels), figsize=(16.5, 9.4),
                             sharex=True, sharey="row", squeeze=False)
    direct = {}
    for metric, _ in plot_metrics:
        direct[metric] = next((r for r in metrics if r["query_type"] == query
                               and r["answerer_role"] == role and int(r["depth"]) == 0
                               and metric in r), None)
    sample = next((r["n"] for r in metrics if r["query_type"] == query), "?")

    for row_index, (metric, ylabel) in enumerate(plot_metrics):
        for col_index, (title, thin, thick) in enumerate(panels):
            axis = axes[row_index][col_index]
            baseline = direct.get(metric)
            if baseline is not None:
                axis.axhline(float(baseline[metric]), color="#999999", linestyle=(0, (1, 2)),
                             linewidth=1.5, zorder=1)
            for arm, colour, style, width, alpha, _label in thin:
                x, y, _ = _series(metrics, arm, query, role, metric)
                if x:
                    axis.plot(x, y, color=colour, linestyle=style, linewidth=width, alpha=alpha,
                              marker="o", markersize=3, zorder=2)
            for arm, colour, style, width, _label in thick:
                x, y, err = _series(metrics, arm, query, role, metric)
                if x:
                    axis.errorbar(x, y, yerr=err, color=colour, linestyle=style, linewidth=width,
                                  marker="o", markersize=5, capsize=3, elinewidth=1.1, zorder=4)
            axis.set_xticks(depths)
            axis.grid(alpha=0.25)
            if col_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(plot_metrics) - 1:
                axis.set_xlabel("Handoff depth (number of compression stages)")
            if row_index == 0:
                axis.set_title(title, fontsize=11)
                handles = [Line2D([], [], color=c, linestyle=s, linewidth=w, label=label)
                           for _a, c, s, w, label in thick]
                handles += [Line2D([], [], color=c, linestyle=s, linewidth=w, alpha=a, label=label)
                            for _a, c, s, w, a, label in thin]
                handles.append(Line2D([], [], color="#999999", linestyle=(0, (1, 2)), linewidth=1.5,
                                      label="direct context, no handoff"))
                axis.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.92)

    label = "conditioning target A (the final task)" if query == "target" else \
        "held-out question B (information the task never asked for)"
    fig.suptitle(f"Experiment 8: model heterogeneity across sequential handoffs — {label}",
                 fontsize=14, y=0.977)
    # Any arm that ran through an empty handoff is named on the figure itself: a
    # curve depressed by a provider returning no text reads as a weak model
    # unless the figure says otherwise. Only the badly affected arms are listed
    # by name; the rest are counted, so the note stays inside the canvas.
    deepest = max(int(x["depth"]) for x in metrics)
    degraded = {r["arm"]: float(r.get("empty_handoff_chain_fraction") or 0) for r in metrics
                if int(r["depth"]) == deepest and float(r.get("empty_handoff_chain_fraction") or 0) > 0}
    notes = [
        f"n={sample} SQuAD question pairs, one shared passage plus nine unrelated ones per pair. Every "
        f"line is answered by the same fixed model ({role}), so the lines only differ in who wrote the "
        f"handoffs, not who answered.",
        "Thick lines are the comparisons that matter; thin lines just show each family on its own, for "
        "reference, and skip their error bars to stay readable. Error bars show the range this result "
        "could fall in by chance alone (95% confidence, computed over the 20 pairs).",
    ]
    if degraded:
        severe = sorted((a for a, f in degraded.items() if f >= 0.5), key=lambda a: -degraded[a])
        mild = len(degraded) - len(severe)
        named = ", ".join(f"{a} ({100 * degraded[a]:.0f}% of chains)" for a in severe) or "none severely"
        notes.append(
            f"Some chains ran through an EMPTY handoff (the provider returned no text), so those curves "
            f"are not compression results: {named}"
            + (f", plus {mild} further arms at 15% or less" if mild else "")
            + ". Every contrast is repeated without those pairs in deltas_sensitivity.csv.")
    fig.tight_layout(rect=(0, 0.028 + 0.021 * len(notes), 1, 0.955))
    for offset, text in enumerate(notes):
        fig.text(0.5, 0.021 * (len(notes) - offset) - 0.007, text,
                 ha="center", fontsize=8.5, color="#555555")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


# The two questions asked of every handoff, side by side. `make_primary_plot`
# renders target and held-out as two entirely separate figures (one per query),
# each broken out into the three experimental blocks; this instead puts target
# and held-out in adjacent columns for a *single* headline contrast, matching
# how Experiments 5 and 6 lay out their target-vs-held-out figures. It answers
# a different question than the two block figures do: not "how does each block
# behave", but "does the same contrast cost more on the question the chain
# never saw" -- which is where Experiment 8's one surviving heterogeneity
# effect actually shows up.
COMPARISON_SERIES = (
    ("homog_small_matched", "#1b9e77", "-", 2.4, "small models, same family"),
    ("xfam_small_fwd", "#1b9e77", "--", 2.4, "small models, family changes"),
    ("homog_large_matched", "#7570b3", "-", 2.4, "large models, same family"),
    ("xfam_large_fwd", "#7570b3", "--", 2.4, "large models, family changes"),
)


def make_comparison_plot(metrics: list[dict], cfg: dict, output: Path) -> None:
    """Target A and held-out B side by side, for the headline family contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    role = cfg["analysis"]["primary_answerer"]
    depths = [int(d) for d in cfg["depths"] if int(d) > 0]
    plot_metrics = [("f1", METRIC_LABELS["f1"]), ("judge_correct", METRIC_LABELS["judge_correct"])]
    columns = [("target", "Question A (the one the chain was told about)"),
              ("heldout", "Question B (never shown to any model in the chain)")]
    fig, axes = plt.subplots(len(plot_metrics), len(columns), figsize=(11.5, 9.0),
                             sharex=True, sharey="row", squeeze=False)

    for row_index, (metric, ylabel) in enumerate(plot_metrics):
        for col_index, (query, title) in enumerate(columns):
            axis = axes[row_index][col_index]
            baseline = next((r for r in metrics if r["arm"] == "homog_small_matched"
                             and r["query_type"] == query and r["answerer_role"] == role
                             and int(r["depth"]) == 0 and metric in r), None)
            if baseline is not None:
                axis.axhline(float(baseline[metric]), color="#999999", linestyle=(0, (1, 2)),
                             linewidth=1.5, zorder=1)
            for arm, colour, style, width, _label in COMPARISON_SERIES:
                x, y, err = _series(metrics, arm, query, role, metric)
                if x:
                    axis.errorbar(x, y, yerr=err, color=colour, linestyle=style, linewidth=width,
                                  marker="o", markersize=5, capsize=3, elinewidth=1.1, zorder=3)
            axis.set_xticks(depths)
            axis.grid(alpha=0.25)
            if col_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(plot_metrics) - 1:
                axis.set_xlabel("Handoff depth")
            if row_index == 0:
                axis.set_title(title, fontsize=11)
            if row_index == 0 and col_index == 0:
                handles = [Line2D([], [], color=c, linestyle=s, linewidth=w, label=label)
                           for _a, c, s, w, label in COMPARISON_SERIES]
                handles.append(Line2D([], [], color="#999999", linestyle=(0, (1, 2)), linewidth=1.5,
                                      label="direct context, no handoff"))
                axis.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.92)

    sample = next((r["n"] for r in metrics), "?")
    fig.suptitle("Experiment 8: does changing model family cost more on the question nobody asked?",
                 fontsize=13, y=0.975)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.5, 0.028,
             f"n={sample} pairs, same fixed model answering every line. Solid = same family the whole "
             f"chain; dashed = a different family writes every step.",
             ha="center", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.008,
             "Small-tier lines here include the pairs where Gemma's first handoff came back empty "
             "(see model_heterogeneity.png); deltas_sensitivity.csv redoes this comparison with those "
             "pairs left out.",
             ha="center", fontsize=8.5, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_trajectory_plot(metrics: list[dict], cfg: dict, schedules: dict, pairs: list[dict],
                         registry: dict[str, mp.ModelSpec], arms, derived, output: Path) -> None:
    """Which model handled every stage of every chain, beside how it scored.

    Dot *area* is proportional to parameter count. Matplotlib's ``s`` is an area
    in points squared, so feeding a size-derived number straight into it is
    already the area encoding -- the one that reads correctly, rather than a
    radius that would exaggerate the large models roughly threefold. Colour is
    family. The right-hand panel carries the same row's result at maximum depth,
    which is the point of the figure: chain architecture and chain outcome share
    one y axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    role = cfg["analysis"]["primary_answerer"]
    metric = cfg["analysis"]["primary_metric"]
    query = cfg["analysis"]["primary_query"]
    max_depth = int(cfg["max_depth"])
    colours = {k: v["color"] for k, v in cfg["families"].items()}
    scale = cfg["analysis"]["size_scale"]
    reference_family = cfg["analysis"]["trajectory_reference_family"]
    reference_index = cfg["family_cycle"].index(reference_family)
    if reference_index >= len(pairs):
        reference_index = 0
    reference_pair = pairs[reference_index]["pair_id"]
    known = [s.parameters_b for s in registry.values() if s.parameters_b is not None]
    max_parameters = max(known) if known else 1.0

    ordered = list(arms) + list(derived)
    blocks = list(dict.fromkeys(a.block for a in ordered))
    grouped = [a for block in blocks for a in ordered if a.block == block]
    rows = grouped[::-1]  # first configured arm ends up at the top
    boundaries = []
    seen_blocks = []
    for index, arm in enumerate(rows):
        if arm.block not in seen_blocks:
            seen_blocks.append(arm.block)
            if index:
                boundaries.append(index - 0.5)

    # Reserve a fixed band in *inches* below the axes for the two legends, the
    # short-code key and the rotation note. A fraction would shrink that band as
    # rows are added, which is exactly when it is needed most.
    legend_inches = 2.55
    height = 1.5 + legend_inches + 0.50 * len(rows)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.0, height), sharey=True,
        gridspec_kw={"width_ratios": [3.0, 1.2], "wspace": 0.03})

    for y, arm in enumerate(rows):
        keys = schedules[(arm.name, reference_pair)]
        left.plot(range(1, max_depth + 1), [y] * max_depth, color="#c4c4c4", linewidth=1.3,
                  zorder=1, solid_capstyle="round")
        for stage, key in enumerate(keys, start=1):
            spec = registry[key]
            area = mp.marker_area(spec.parameters_b, max_parameters, scale)
            left.scatter([stage], [y], s=area, color=colours[spec.family],
                         edgecolors="white" if spec.size_known else "#111111",
                         linewidths=0.9 if spec.size_known else 1.6, zorder=3)
            if area >= 320:
                left.annotate(spec.short_code, (stage, y), ha="center", va="center",
                              fontsize=6.4, color="white", fontweight="bold", zorder=4)
            else:
                left.annotate(spec.short_code, (stage + 0.19, y), ha="left", va="center",
                              fontsize=6.0, color="#333333", zorder=4)

    left.set_yticks(range(len(rows)))
    left.set_yticklabels([a.name + ("  (comparison baseline)" if a.is_derived else "") for a in rows],
                         fontsize=8)
    left.set_xticks(range(1, max_depth + 1))
    left.set_xlim(0.45, max_depth + 0.7)
    left.set_ylim(-0.8, len(rows) - 0.2)
    left.set_xlabel("Handoff stage")
    left.set_title("Which model handled each stage", fontsize=11)
    left.grid(axis="x", alpha=0.2)
    for boundary in boundaries:
        left.axhline(boundary, color="#dddddd", linewidth=1.2)

    lookup = {r["arm"]: r for r in metrics if r["query_type"] == query
              and r["answerer_role"] == role and int(r["depth"]) == max_depth and metric in r}
    baseline = next((r for r in metrics if r["query_type"] == query and r["answerer_role"] == role
                     and int(r["depth"]) == 0 and metric in r), None)
    for y, arm in enumerate(rows):
        record = lookup.get(arm.name)
        if record is None:
            continue
        value = float(record[metric])
        low, high = float(record[f"{metric}_lo"]), float(record[f"{metric}_hi"])
        right.errorbar([value], [y], xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                       fmt="o", color="#222222", markersize=5, capsize=3, elinewidth=1.1, zorder=3)
    if baseline is not None:
        right.axvline(float(baseline[metric]), color="#999999", linestyle=(0, (1, 2)),
                      linewidth=1.5, label="direct context, no handoff")
        right.legend(fontsize=7, loc="lower left")
    for boundary in boundaries:
        right.axhline(boundary, color="#dddddd", linewidth=1.2)
    query_label = "question A" if query == "target" else "question B (not shown to the chain)"
    right.set_xlabel(f"{METRIC_LABELS.get(metric, metric)}\n({query_label}, stage {max_depth})", fontsize=9)
    right.set_title("How that chain scored", fontsize=11)
    right.grid(axis="x", alpha=0.25)

    family_handles = [Line2D([], [], marker="o", linestyle="", markersize=9,
                             markerfacecolor=colours[f], markeredgecolor="white",
                             label=cfg["families"][f]["label"]) for f in cfg["family_cycle"]]
    reference_sizes = sorted({s.parameters_b for s in registry.values() if s.size_known})
    picks = list(dict.fromkeys([reference_sizes[0],
                                reference_sizes[len(reference_sizes) // 2],
                                reference_sizes[-1]]))
    size_handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="#8c8c8c",
                           markeredgecolor="white",
                           markersize=mp.marker_area(p, max_parameters, scale) ** 0.5,
                           label=f"{p:g}B") for p in picks]
    fig.suptitle("Experiment 8: model trajectory of every handoff chain", fontsize=14, y=0.99)

    # Explicit margins rather than tight_layout: the two figure-level legends
    # below the axes are not part of the layout engine's box, so letting it
    # guess both warns and overlaps the legend band.
    def band(inches_from_bottom: float) -> float:
        return inches_from_bottom / height

    fig.subplots_adjust(left=0.175, right=0.985, top=1 - band(0.62),
                        bottom=band(legend_inches), wspace=0.03)
    fig.legend(handles=family_handles, title="Model family (dot colour)", fontsize=8.5,
               title_fontsize=8.5, loc="lower left", bbox_to_anchor=(0.05, band(0.10)),
               ncol=2, frameon=False)
    fig.legend(handles=size_handles, title="Model size (bigger dot = more parameters)",
               fontsize=8.5, title_fontsize=8.5, loc="lower right",
               bbox_to_anchor=(0.965, band(0.10)), ncol=3, frameon=False,
               labelspacing=1.6, borderpad=0.9, handletextpad=1.4, columnspacing=2.6)

    # One key line per tier: a single line of eight model ids does not fit the
    # canvas, and silently clipping the key would defeat its purpose.
    for offset, tier in enumerate(sorted({s.tier for s in registry.values()}, reverse=True)):
        codes = "    ".join(
            f"{s.short_code} = {s.id}"
            f" ({'undisclosed' if s.parameters_b is None else f'{s.parameters_b:g}B'})"
            for s in sorted(registry.values(), key=lambda s: s.family) if s.tier == tier)
        fig.text(0.5, band(legend_inches - 0.64 - 0.19 * offset), codes,
                 ha="center", fontsize=7.4, color="#333333")
    for offset, text in enumerate((
        f"Rows where the family changes each step are shown for the pairs that start on "
        f"{reference_family}; other pairs run the same pattern starting from a different family, so "
        f"every family gets an equal turn going first.",
        "Model sizes are the numbers each company has published; if a company never published one, "
        "that dot is drawn as small as possible with a dark outline, instead of guessing a number.",
    )):
        fig.text(0.5, band(legend_inches - 1.26 - 0.19 * offset), text,
                 ha="center", fontsize=8.2, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


# Size direction on the x-axis, same/cross-family as adjacent bar pairs within
# each group. Grouping this way -- rather than by family first, size second --
# puts exactly the two bars the paired contrast compares next to each other,
# so "does crossing family cost something here" is a same-group comparison
# instead of a look-up across the whole axis. Order matches how the report's
# own prose introduces them: matched size first, then the two size changes.
COMPARISON_GROUPS = (
    ("Same size", "same", "cross_vs_same_family_at_matched_size"),
    ("Growing", "larger", "cross_vs_same_family_when_growing"),
    ("Shrinking", "smaller", "cross_vs_same_family_when_shrinking"),
)


def make_transition_plot(transition_metrics: list[dict], transition_deltas: list[dict],
                         cfg: dict, output: Path) -> None:
    """Per-edge information loss: same-family vs cross-family, at each size direction.

    Unpaired per-category bootstrap intervals can visually overlap even when the
    *paired* difference between two adjacent bars is reliably nonzero -- pairing
    cancels the pair-to-pair variance that makes each bar's own interval wide.
    Experiment 8's own data shows this: held-out fact survival at matched size
    looks like it might overlap from the bars alone (0.15-0.38 vs 0.03-0.31) but
    the paired contrast excludes zero (p=0.012). So every group gets its actual
    paired delta and p-value from transition_deltas.csv printed above it, rather
    than leaving the reader to eyeball two independent error bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    by_type = {r["transition_type"]: r for r in transition_metrics if r["scope"] == "by_type"}
    by_contrast = {(r["comparison"], r["measure"]): r for r in transition_deltas}
    if not by_type:
        return
    panels = [
        ("semantic_preserved", "Did the meaning stay the same?"),
        ("fact_B_survived", "Does it still contain the answer to question B?"),
        ("lexical_similarity", "How much of the wording carried over"),
        ("length_ratio", "Message length: after ÷ before"),
    ]
    centers = [0.0, 1.15, 2.3]
    half_gap, bar_w = 0.24, 0.42
    same_colour, cross_colour = "#4575b4", "#d73027"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), squeeze=False)
    for index, (field, title) in enumerate(panels):
        axis = axes[index // 2][index % 2]
        tops: list[float] = [0.0]
        bars: list[tuple] = []  # (x, value, lower, upper, colour)
        # (x_left, x_right, x_center, group_label, delta_label, significant)
        brackets: list[tuple] = []
        for center, (group_label, direction, contrast_name) in zip(centers, COMPARISON_GROUPS):
            same_row = by_type.get(f"same_family_size_{direction}")
            cross_row = by_type.get(f"cross_family_size_{direction}")
            if not same_row or not cross_row or field not in same_row or field not in cross_row:
                continue
            # Shown under the group label so it's visible on the chart itself,
            # not only in a caption: each bar averages this many handoffs, not one.
            group_label = (f"{group_label}\n(from {same_row['n_edges']} + {cross_row['n_edges']} "
                           f"handoffs)")
            x_same, x_cross = center - half_gap, center + half_gap
            for x, row, colour in ((x_same, same_row, same_colour), (x_cross, cross_row, cross_colour)):
                value = float(row[field])
                lower = value - float(row[f"{field}_lo"])
                upper = float(row[f"{field}_hi"]) - value
                bars.append((x, value, lower, upper, colour))
                tops.append(value + upper)
            delta_row = by_contrast.get((contrast_name, field))
            if delta_row:
                significant = float(delta_row["lo"]) * float(delta_row["hi"]) > 0
                label = f"{float(delta_row['delta']):+.2f}" + (" *" if significant else " n.s.")
                brackets.append((x_same, x_cross, center, group_label, label, significant))
        for x, value, lower, upper, colour in bars:
            axis.bar(x, value, width=bar_w, yerr=[[lower], [upper]], color=colour,
                     capsize=3, edgecolor="white", linewidth=0.8, zorder=2)
        # Tick positions come from the groups actually drawn, not a slice of
        # `centers` -- a missing middle group would otherwise mislabel ticks.
        axis.set_xticks([b[2] for b in brackets])
        axis.set_xticklabels([b[3] for b in brackets], fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        # Headroom is one unit, sized off the tallest bar top, then every
        # bracket element (tick, gap, label) is a fraction of that one unit --
        # keeps the empty space above the bars proportionate instead of the
        # fixed multiplier occasionally leaving a third of the panel blank.
        pad = 0.15 * max(max(tops), 0.1)
        bracket_y = max(tops) + 0.30 * pad
        axis.set_ylim(bottom=0, top=max(tops) + pad)
        for x_left, x_right, _xc, _label, delta_label, significant in brackets:
            axis.plot([x_left, x_left, x_right, x_right],
                      [bracket_y - 0.22 * pad, bracket_y, bracket_y, bracket_y - 0.22 * pad],
                      color="#333333", linewidth=1.0, zorder=3)
            axis.text((x_left + x_right) / 2, bracket_y + 0.08 * pad, delta_label,
                      ha="center", va="bottom", fontsize=8.5,
                      fontweight="bold" if significant else "normal",
                      color="#111111" if significant else "#777777")

    fig.suptitle("Experiment 8: what one handoff step costs, by size change and family change",
                 fontsize=13, y=0.995)
    # loc="upper center" anchors the legend's TOP edge, so its size (not
    # tight_layout, which does not know about figure-level legends or
    # suptitles at all) is what determines how far it extends downward --
    # predictable, unlike anchoring the bottom edge and guessing its height.
    fig.legend(handles=[Patch(color=same_colour, label="next step: same family"),
                        Patch(color=cross_colour, label="next step: different family")],
               loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.165, 1, 0.86))
    # Five short lines rather than long ones: at this figure width a line over
    # ~160 characters runs past both edges and is clipped on save.
    for offset, text in enumerate((
        "Each bar is the average across every handoff of that kind (the count is printed under each "
        "pair of bars) -- not a single handoff.",
        "Semantic preservation: 1 = kept the same meaning, 0.5 = lost a small detail, "
        "0 = changed the meaning.",
        "The number above each pair of bars is how much they actually differ (different-family minus "
        "same-family), measured directly -- not just the visual gap between the two bar heights.",
        "'*' means that difference is unlikely to be chance. 'n.s.' means it could be. The very first "
        "handoff (source text to the first model) isn't included: there's no earlier model to compare it to.",
        "The black error bars on the bars themselves are a separate, looser measure -- two bars can look "
        "like they overlap while the number above them still shows a real, reliable difference.",
    )):
        fig.text(0.5, 0.128 - 0.026 * offset, text, ha="center", fontsize=8.1, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- report

def write_report(metrics, deltas, transition_metrics, transition_deltas, family_metrics, cfg,
                 registry, catalogue, reference_chains, ledger, judge_ledgers, path: Path) -> None:
    role = cfg["analysis"]["primary_answerer"]
    reference_family = cfg["analysis"]["trajectory_reference_family"]
    max_depth = int(cfg["max_depth"])
    nan = float("nan")
    lines = [
        "# Experiment 8: model heterogeneity across sequential handoffs", "",
        f"Depths: {', '.join(str(d) for d in cfg['depths'])}  ",
        "Answerers (held constant across every arm): "
        + "; ".join(f"{a['role']} = `{registry[a['model']].id}`" for a in cfg["answerers"]) + "  ",
        f"Judge: `{cfg['judge']['model_id']}` at temperature 0  ",
        f"Handoff mode: {MODE} (question-conditioned), temperature "
        f"{cfg['decoding']['subagent_temperature']}, shared budget "
        f"{cfg['decoding']['handoff_max_tokens']} tokens", "",
        "## Model pool", "",
        "| key | model id | family | tier | parameters | basis | released | $/Mtok in | $/Mtok out |",
        "|---|---|---|---|---:|---|---|---:|---:|",
    ]
    for spec in sorted(registry.values(), key=lambda s: (s.tier, s.family)):
        live = (catalogue.get("models") or {}).get(spec.id, {})
        size = "undisclosed" if spec.parameters_b is None else f"{spec.parameters_b:g}B"
        lines.append(
            f"| `{spec.key}` | `{spec.id}` | {spec.family} | {spec.tier} | {size} | "
            f"{spec.parameters_basis} | {spec.released} | "
            f"{live.get('prompt_usd_per_mtok', '')} | {live.get('completion_usd_per_mtok', '')} |")

    at_depth = {(r["arm"], r["query_type"]): r for r in metrics
                if r["answerer_role"] == role and int(r["depth"]) == max_depth}
    lines += ["", f"## Accuracy at depth {max_depth} ({role} answerer)", "",
              f"Chains are shown for the pairs that start on {reference_family}; an arm whose family "
              f"rotates runs the same pattern from each of the other start families too.", "",
              "| arm | block | chain | family switches | target F1 | "
              "target judge | held-out F1 | held-out judge |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for arm in sorted({a for a, _ in at_depth}):
        target, heldout = at_depth.get((arm, "target")), at_depth.get((arm, "heldout"), {})
        if target is None:
            continue
        lines.append(
            f"| `{arm}` | {target['block']} | `{reference_chains.get(arm, '?')}` | "
            f"{target['n_family_transitions']} | {target.get('f1', nan):.3f} | "
            f"{target.get('judge_correct', nan):.3f} | {heldout.get('f1', nan):.3f} | "
            f"{heldout.get('judge_correct', nan):.3f} |")

    lines += ["", f"## Contrasts at depth {max_depth} ({role} answerer, judged accuracy)", "",
              "| comparison | query | delta | 95% CI | p |", "|---|---|---:|---|---:|"]
    for row in deltas:
        if (row["answerer_role"] != role or row["metric"] != "judge_correct"
                or int(row["depth"]) != max_depth):
            continue
        lines.append(f"| {row['comparison']} | {row['query_type']} | {row['delta']:+.3f} | "
                     f"[{row['lo']:+.3f}, {row['hi']:+.3f}] | {row['p_value']:.3f} |")

    lines += ["", "## Per-edge information loss by transition type", "",
              "| transition | edges | pairs | semantic preserved | held-out fact kept | "
              "target fact kept | lexical similarity | length ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in transition_metrics:
        if row["scope"] != "by_type" or row["transition_type"] == "source_to_first_agent":
            continue
        lines.append(
            f"| {row['transition_type']} | {row['n_edges']} | {row['n_pairs']} | "
            f"{row.get('semantic_preserved', nan):.3f} | {row.get('fact_B_survived', nan):.3f} | "
            f"{row.get('fact_A_survived', nan):.3f} | {row.get('lexical_similarity', nan):.3f} | "
            f"{row.get('length_ratio', nan):.3f} |")

    lines += ["", "## Paired per-edge contrasts (same pairs, differing only in the edge type)", "",
              "| comparison | measure | treatment | control | delta | 95% CI | p | pairs |",
              "|---|---|---:|---:|---:|---|---:|---:|"]
    for row in transition_deltas:
        if row["measure"] not in ("lexical_similarity", "length_ratio", "semantic_preserved",
                                  "fact_A_survived", "fact_B_survived"):
            continue
        lines.append(
            f"| {row['comparison']} | {row['measure']} | {row['treatment_mean']:.3f} | "
            f"{row['control_mean']:.3f} | {row['delta']:+.3f} | [{row['lo']:+.3f}, {row['hi']:+.3f}] | "
            f"{row['p_value']:.3f} | {row['n']} |")

    lines += ["", f"## Judged accuracy by number of family switches, depth {max_depth}", "",
              "| query | switches | arms pooled | judge | 95% CI |", "|---|---:|---:|---:|---|"]
    for row in family_metrics:
        if int(row["depth"]) != max_depth or "judge_correct" not in row:
            continue
        lines.append(f"| {row['query_type']} | {row['n_family_transitions']} | {row['n_arms']} | "
                     f"{row['judge_correct']:.3f} | [{row['judge_correct_lo']:.3f}, "
                     f"{row['judge_correct_hi']:.3f}] |")
    lines += ["", "Switch count and depth are collinear inside a single arm, so this table is only "
              "ever read within one depth; the size-trajectory arms are what supply the intermediate "
              "switch counts.", ""]

    lines += ["## Cost", "",
              f"- chain models: ${ledger['cost_usd']:.4f} over {ledger['calls_live']} live and "
              f"{ledger['calls_cached']} cached calls"]
    for name, summary in judge_ledgers.items():
        lines.append(f"- {name}: ${summary['cost_usd']:.4f} over {summary['calls_live']} live calls")
    lines += ["", "Model metadata was read from OpenRouter's live `/api/v1/models` at run time and "
              "stored beside the raw records in `model_catalogue.json`. Parameter counts are "
              "vendor-published and are never inferred from price, tier, or benchmark score; an "
              "undisclosed count is recorded as unknown and excluded from size contrasts.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main

def experiment_fingerprint(cfg: dict, pairs: list[dict]) -> str:
    payload = {
        "models": cfg["models"], "answerers": cfg["answerers"], "arms": cfg["arms"],
        "derived_arms": cfg.get("derived_arms"), "size_patterns": cfg["size_patterns"],
        "family_cycle": cfg["family_cycle"], "depths": cfg["depths"], "max_depth": cfg["max_depth"],
        "decoding": cfg["decoding"], "mode": MODE,
        "experiment_context": cfg["dataset"].get("experiment_context"),
        "chain_system": CHAIN_SYSTEM, "initial": INITIAL_INSTRUCTION,
        "recompress": RECOMPRESS_INSTRUCTION,
        "pairs": [p["pair_id"] for p in pairs],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dry_run_report(pool: mp.ModelPool) -> dict:
    """Aggregate the per-client dry-run plans into one experiment estimate."""
    total = {"uncached_calls": 0, "prompt_tokens_est": 0, "completion_tokens_est_max": 0,
             "cost_usd_est_low": 0.0, "cost_usd_est_high": 0.0, "by_model": {}}
    for key, client in pool.clients.items():
        report = client.dry_run_report()
        if not report["uncached_calls"]:
            continue
        total["by_model"][key] = report
        total["uncached_calls"] += report["uncached_calls"]
        total["prompt_tokens_est"] += report["prompt_tokens_est"]
        total["completion_tokens_est_max"] += report["completion_tokens_est_max"]
        total["cost_usd_est_low"] += report["cost_usd_est_low"]
        total["cost_usd_est_high"] += report["cost_usd_est_high"]
    total["cost_usd_est_low"] = round(total["cost_usd_est_low"], 4)
    total["cost_usd_est_high"] = round(total["cost_usd_est_high"], 4)
    total["cached_calls"] = pool.ledger.summary()["calls_cached"]
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Model heterogeneity across sequential handoffs")
    parser.add_argument("--config", default=str(ROOT / "model_heterogeneity_config.yaml"))
    parser.add_argument("--n", type=int, help="number of A/B pairs (default: dataset.n_pairs)")
    parser.add_argument("--arms", help="comma-separated subset of arm names")
    parser.add_argument("--depths", help="comma-separated subset of depths")
    parser.add_argument("--dry-run", action="store_true", help="estimate calls and cost, spend nothing")
    parser.add_argument("--analyse-only", action="store_true",
                        help="re-derive metrics, plots and report from stored records")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = args.n or int(cfg["dataset"]["n_pairs"])
    if args.depths:
        cfg["depths"] = sorted({int(x) for x in args.depths.split(",") if x.strip()})
        cfg["max_depth"] = max(cfg["depths"])
    pairs = load_prebuilt_pairs(cfg, n)
    registry = mp.load_registry(cfg)
    arms, derived = mp.load_arms(cfg)
    if args.arms:
        wanted = {x.strip() for x in args.arms.split(",") if x.strip()}
        arms = [a for a in arms if a.name in wanted]
        derived = [a for a in derived if a.name in wanted]
        if not arms:
            parser.error("--arms selected no generated arm")

    prompt_selftest(pairs[0], cfg)
    schedules = build_schedules(pairs, arms, derived, cfg, registry)
    schedule_selftest(pairs, arms, derived, schedules, cfg, registry)

    fingerprint = experiment_fingerprint(cfg, pairs)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    print(f"[hetero] fingerprint {fingerprint[:12]} | run_root {run_root} | result_root {result_root}")
    # Live provider metadata, taken once and persisted with the run. It is
    # reproducibility material, not an input: nothing in the analysis reads it.
    catalogue = mp.catalogue_snapshot([spec.id for spec in registry.values()])

    pool = mp.ModelPool(cfg, registry, dry_run=args.dry_run)
    judge_ledgers: dict[str, dict] = {}
    try:
        if args.analyse_only:
            generated = {(r["pair_id"], r["chain_prefix"]): r
                         for r in read_jsonl(run_root / "handoffs.jsonl")
                         if r.get("experiment_fingerprint") == fingerprint}
            if not generated:
                parser.error(f"--analyse-only found no stored handoffs for {fingerprint[:12]}")
        else:
            generated = generate_handoffs(pool, pairs, arms, schedules, cfg, registry,
                                          run_root, fingerprint, args.dry_run)
        answered = run_answers(pool, pairs, arms, schedules, generated, cfg, registry,
                               run_root, fingerprint, args.dry_run)
        if args.dry_run:
            print(json.dumps(dry_run_report(pool), indent=2))
            return 0

        answer_client = judge_deduplicated(list(answered.values()), ("question", "golds", "pred"),
                                           cfg, add_judge, "hetero_answer_judge")
        write_jsonl(run_root / "answers.jsonl", [answered[k] for k in sorted(answered)])
        if answer_client is not None:
            judge_ledgers["answer judge"] = answer_client.ledger.summary()

        edges = build_transitions(pairs, arms, schedules, generated, cfg, registry,
                                  run_root, fingerprint)
        preservation_client = judge_deduplicated(edges, TRANSITION_TEXT_FIELDS, cfg,
                                                 add_preservation_judge, "hetero_preservation_judge")
        write_jsonl(run_root / "transitions.jsonl",
                    [{k: v for k, v in row.items() if k not in TRANSITION_TEXT_FIELDS}
                     for row in edges])
        if preservation_client is not None:
            judge_ledgers["preservation judge"] = preservation_client.ledger.summary()

        degenerate = degenerate_prefixes(generated)
        if degenerate:
            print(f"[hetero:WARNING] {len(degenerate)} handoffs came back empty from their provider: "
                  + ", ".join(sorted({generated[slot]["model_key"] + f"@stage{generated[slot]['stage']}"
                                      for slot in degenerate})))
        rows = expand_answers(answered, pairs, arms, derived, schedules, cfg, registry, degenerate)
        expanded = attach_arms(edges, pairs, arms, derived, schedules, cfg)
        label_transitions(expanded, rows, cfg)

        metrics = compute_metrics(rows, cfg)
        deltas = compute_deltas(rows, cfg)
        # An edge whose *input* was empty is not a rewrite of anything: its
        # lexical similarity is 0 and its length ratio undefined by construction,
        # not because a model discarded content. Measuring it as a handoff would
        # charge a provider failure to whichever transition type followed it.
        distinct_edges = [e for e in expanded if not e["derived"]]
        measurable = [e for e in distinct_edges if int(e.get("previous_characters", 0)) > 0]
        if len(measurable) < len(distinct_edges):
            print(f"[hetero:transitions] excluding {len(distinct_edges) - len(measurable)} edges "
                  f"whose input message was empty; {len(measurable)} measurable edges remain")
        transition_metrics = compute_transition_metrics(measurable, cfg)
        transition_deltas = compute_transition_contrasts(measurable, cfg)
        family_metrics = compute_family_transition_metrics(rows, cfg)
        stage_records = build_stage_records(expanded, rows, cfg)

        result_root.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "model_catalogue.json").write_text(
            json.dumps(catalogue, indent=2, sort_keys=True), encoding="utf-8")
        write_csv(result_root / "model_registry.csv",
                  [{**spec.to_json(), **(catalogue.get("models") or {}).get(spec.id, {})}
                   for spec in sorted(registry.values(), key=lambda s: (s.tier, s.family))])
        write_csv(result_root / "chain_index.csv", [
            {"arm": arm.name, "block": arm.block, "derived": arm.is_derived,
             "pair_id": pair["pair_id"], "pair_index": position,
             "start_family": mp.start_family(position, cfg["family_cycle"]),
             **mp.chain_summary(schedules[(arm.name, pair["pair_id"])], registry)}
            for arm in list(arms) + list(derived) for position, pair in enumerate(pairs)])
        write_csv(result_root / "metrics.csv", metrics)
        write_csv(result_root / "deltas.csv", deltas)
        # Same contrasts on the pairs no arm routed through an empty handoff.
        # If a headline conclusion only survives in one of the two files, it is
        # a statement about a provider failure, not about model composition.
        sensitivity = compute_deltas(rows, cfg, drop_degenerate=True)
        affected = {r["arm"] for r in rows if int(r.get("chain_has_empty_handoff", 0))}
        print(f"[hetero:sensitivity] recomputed every contrast excluding, per contrast, the pairs "
              f"its own arms degraded; arms affected at all: {', '.join(sorted(affected)) or 'none'}")
        write_csv(result_root / "deltas_sensitivity.csv", sensitivity)
        write_csv(result_root / "transition_metrics.csv", transition_metrics)
        write_csv(result_root / "transition_deltas.csv", transition_deltas)
        write_csv(result_root / "family_transition_metrics.csv", family_metrics)
        write_csv(result_root / "stage_records.csv", stage_records)

        make_primary_plot(metrics, cfg, "target", result_root / "model_heterogeneity.png")
        make_primary_plot(metrics, cfg, "heldout", result_root / "model_heterogeneity_heldout.png")
        make_comparison_plot(metrics, cfg, result_root / "model_heterogeneity_target_vs_heldout.png")
        make_trajectory_plot(metrics, cfg, schedules, pairs, registry, arms, derived,
                             result_root / "model_trajectories.png")
        make_transition_plot(transition_metrics, transition_deltas, cfg,
                             result_root / "transition_diagnostics.png")
        # The figure and the report must describe the same rotation, or a reader
        # comparing them is comparing two different chains under one arm name.
        reference_index = cfg["family_cycle"].index(cfg["analysis"]["trajectory_reference_family"])
        reference_pair = pairs[reference_index % len(pairs)]["pair_id"]
        reference_chains = {
            arm.name: mp.chain_summary(schedules[(arm.name, reference_pair)], registry)["chain_short_codes"]
            for arm in list(arms) + list(derived)
        }
        write_report(metrics, deltas, transition_metrics, transition_deltas, family_metrics, cfg,
                     registry, catalogue, reference_chains, pool.ledger.summary(), judge_ledgers,
                     result_root / "report.md")
        print(f"[hetero:analysis] wrote {result_root}")
        print("[hetero:model cost] " + json.dumps(pool.ledger.summary()))
        for name, summary in judge_ledgers.items():
            print(f"[hetero:{name} cost] " + json.dumps(summary))
    except CostCapExceeded as exc:
        print(f"[hetero:ABORT] {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
