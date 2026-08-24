"""Question conditioning on two SQuAD questions from one shared passage.

Each prebuilt example contains one gold passage that answers both A and B plus
nine passages screened irrelevant to both questions. The conditioned chain sees
only A; the generic chain sees neither question. Both are evaluated at depths
0 through 10. The obsolete constructor that paired separate passages is not
supported by this runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
import paraphrase_metrics as pmx  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from judge import add_judge, add_preservation_judge  # noqa: E402
from run_chain import (  # noqa: E402
    CHAIN_SYSTEM,
    INITIAL_INSTRUCTION,
    RECOMPRESS_INSTRUCTION,
)
from run_slack_facts import mentions_answer  # noqa: E402
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402


# --- Handoff conditions -----------------------------------------------------
#
# ``conditioned`` and ``generic`` are the original pair and are untouched: they
# share Experiment 2's question-conditioned prompt set and differ only by the
# Question A block. Two conditions are added to separate *rewriting* from
# *compression*, which the original pair cannot do because both arms summarise.
#
# ``paraphrase``  rewrites the whole message with no compression or relevance
#                 filtering requested. If repeated rewriting is itself lossy,
#                 this arm degrades; if the loss comes from compression and
#                 task-relevance selection, it does not.
# ``passthrough`` copies the message forward with no model call at all. It is
#                 the zero-rewriting floor: any movement across depth in this
#                 arm is answerer noise, not handoff loss, so it calibrates how
#                 large a "flat" curve can look by chance.
#
# Both new arms are question-blind, exactly like ``generic``: a paraphraser that
# saw Question A would be selecting task-relevant content, which is the variable
# this arm exists to hold fixed.
MODES_WITH_QUESTION = ("conditioned",)
MODES_WITHOUT_MODEL_CALL = ("passthrough",)

# Existing arms keep their existing colours so previously published figures
# stay comparable; the two new arms extend the same Dark2 palette.
MODE_COLORS = {
    "direct": "#7570b3", "conditioned": "#d95f02", "generic": "#1b9e77",
    "paraphrase": "#e7298a", "passthrough": "#777777",
}

# Contrasts are emitted in this order for whichever arms a config actually
# runs. ``conditioned_minus_generic`` keeps its exact historical name and
# position so existing deltas.csv files stay byte-identical.
CONTRAST_PAIRS = (
    ("conditioned", "generic"),
    ("paraphrase", "generic"),
    ("paraphrase", "passthrough"),
    ("conditioned", "paraphrase"),
    ("generic", "passthrough"),
)

PARAPHRASE_INITIAL_INSTRUCTION = (
    "Rewrite the source material in your own words for another agent. Restate every fact, "
    "qualifier, date, number, relationship, uncertainty, and source id it contains. "
    "This is a rewrite, not a summary: do not condense, shorten, omit, prioritise, or keep "
    "only what seems relevant, and do not add any fact that is not already present."
)
PARAPHRASE_RECOMPRESS_INSTRUCTION = (
    "Rewrite the previous agent's notes in your own words for another agent. Restate every fact, "
    "qualifier, date, number, relationship, uncertainty, and source id they contain. "
    "This is a rewrite, not a summary: do not condense, shorten, omit, prioritise, or keep "
    "only what seems relevant, and do not add any fact that is not already present. "
    "Use only the previous notes."
)


def mode_instruction(mode: str, stage: int) -> str:
    """The one condition-specific string. Everything else stays matched."""
    if mode == "paraphrase":
        return PARAPHRASE_INITIAL_INSTRUCTION if stage == 1 else PARAPHRASE_RECOMPRESS_INSTRUCTION
    return INITIAL_INSTRUCTION if stage == 1 else RECOMPRESS_INSTRUCTION


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
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_prebuilt_pairs(cfg: dict, n: int) -> list[dict]:
    """Load pairs built and validated by a separate builder script.

    Construction that needs LLM calls (independence/salience audit, C1 leakage
    filtering) belongs in the builder, not in the experiment runner, so the
    experiment never silently re-derives its own dataset. See
    src/build_squad_same_passage.py.
    """
    path = ROOT / cfg["dataset"]["pairs_jsonl"]
    if not path.exists():
        raise FileNotFoundError(
            f"Prebuilt pair file missing: {path}. Build it with "
            "python src/build_squad_same_passage.py")
    pairs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    if len(pairs) < n:
        raise RuntimeError(f"{path} holds {len(pairs)} pairs, need {n}")
    pairs = pairs[:n]
    width = cfg["dataset"]["context_passages"]
    for pair in pairs:
        assert len(pair["passages"]) == width, f"{pair['pair_id']} has {len(pair['passages'])} passages"
        assert sum(p["role"] == "gold_AB" for p in pair["passages"]) == 1
    positions = sorted(p["gold_position"] for p in pairs)
    chars = [p["context_chars"] for p in pairs]
    print(f"[generalization:pairs] {len(pairs)} prebuilt pairs from {path.name}; "
          f"gold positions {min(positions)}-{max(positions)} over "
          f"{len(set(positions))} distinct slots; context "
          f"{min(chars)}-{max(chars)} chars")
    return pairs


def render_context(pair: dict) -> str:
    return "\n\n".join(f"[P{i + 1}] {p['text']}" for i, p in enumerate(pair["passages"]))


def gold_only_pairs(pairs: list[dict], cfg: dict) -> list[dict]:
    """Project reusable 1+N packs down to the one shared gold passage.

    Shared with Experiment 6 (run_multilingual_handoffs.py imports this,
    the design this projection was written for); Experiment 5's own
    gold-only config (experiment_context: gold_only) reuses it unchanged so
    both experiments strip distractors identically rather than maintaining
    two copies of the same projection.
    """
    import copy
    mode = cfg["dataset"].get("experiment_context")
    if mode != "gold_only":
        raise ValueError("gold_only_pairs requires dataset.experiment_context: gold_only")
    projected: list[dict] = []
    for source in pairs:
        pair = copy.deepcopy(source)
        gold = [p for p in pair["passages"] if p.get("role") == "gold_AB"]
        if len(gold) != 1:
            raise ValueError(f"{pair['pair_id']} has {len(gold)} shared gold passages; expected one")
        pair["passages"] = gold
        pair["experiment_context"] = "gold_only"
        pair["source_context_passages"] = len(source["passages"])
        pair["context_chars"] = len(gold[0]["text"])
        pair["gold_position"] = 1
        projected.append(pair)
    lengths = [len(pair["passages"][0]["text"]) for pair in projected]
    print(f"[generalization:context] gold-only: {len(projected)} pairs, one shared gold passage, "
          f"0 distractors; {min(lengths)}-{max(lengths)} characters")
    return projected


def length_directive(cfg: dict) -> str:
    """Optional word-budget sentence appended IDENTICALLY to both arms.

    Without it the two arms self-select very different lengths (conditioned
    ~600 characters, generic ~3,800 on the same context), so a held-out
    accuracy gap cannot be separated from "one arm simply wrote more". Adding
    the same target to both makes summary length a controlled variable rather
    than an uncontrolled mediator, while leaving the question block as the
    only arm-level difference -- prompt_difference_selftest() still holds.
    """
    target = cfg.get("length_target_words")
    if not target:
        return ""
    return (f" Aim for about {int(target)} words: keep it close to that length, "
            "neither much shorter nor much longer.")


def validate_modes(cfg: dict) -> None:
    """Reject configurations whose arms would contradict each other.

    A word budget is a compression instruction. Appending it to the paraphrase
    arm would tell that arm to shorten while its own instruction tells it not
    to, which destroys the one thing the arm exists to isolate. Refuse rather
    than silently run an arm that no longer means what its name says.
    """
    modes = list(cfg.get("modes") or [])
    if "paraphrase" in modes and cfg.get("length_target_words"):
        raise ValueError(
            "length_target_words cannot be combined with the paraphrase arm: a word budget "
            "is a compression instruction and the paraphrase arm is defined by the absence "
            "of one. Drop the key, or drop 'paraphrase' from modes.")


def compression_user_prompt(pair: dict, notes: str | None, mode: str, stage: int,
                            cfg: dict | None = None) -> str:
    """Build prompts that differ only by the question block and the instruction.

    Material block, ordering, separators, and the optional length directive are
    byte-identical across every arm; ``mode_instruction`` supplies the single
    condition-specific sentence and only ``conditioned`` gets a question.
    """
    if stage == 1:
        material = f"Source material:\n{render_context(pair)}"
    else:
        material = f"Previous agent's notes:\n{notes}"
    instruction = mode_instruction(mode, stage) + length_directive(cfg or {})
    if mode in MODES_WITH_QUESTION:
        user = f"{material}\n\nQuestion the final agent must answer: {pair['question_A']}\n\n{instruction}"
    else:
        user = f"{material}\n\n{instruction}"
        assert pair["question_A"] not in user and pair["question_B"] not in user
    return user


def prompt_difference_selftest(pair: dict, cfg: dict | None = None) -> None:
    """Prove each arm differs from the others only where it is supposed to.

    1. Deleting the question block makes conditioned and generic byte-identical.
    2. Every question-blind arm (generic, paraphrase) leaks neither question.
    3. paraphrase differs from generic *only* in the instruction sentence: the
       material block and all surrounding structure stay byte-identical, so a
       paraphrase-vs-generic contrast measures the instruction and nothing else.
    """
    marker = f"\n\nQuestion the final agent must answer: {pair['question_A']}"
    modes = list((cfg or {}).get("modes") or ["conditioned", "generic"])
    for stage, notes in ((1, None), (2, "identical previous notes")):
        conditioned = compression_user_prompt(pair, notes, "conditioned", stage, cfg)
        generic = compression_user_prompt(pair, notes, "generic", stage, cfg)
        assert conditioned.replace(marker, "") == generic
        for mode in modes:
            if mode in MODES_WITH_QUESTION or mode in MODES_WITHOUT_MODEL_CALL:
                continue
            blind = compression_user_prompt(pair, notes, mode, stage, cfg)
            assert pair["question_A"] not in blind and pair["question_B"] not in blind
            # Swapping this arm's instruction back to the generic one must
            # reproduce the generic prompt exactly.
            own = mode_instruction(mode, stage) + length_directive(cfg or {})
            shared = mode_instruction("generic", stage) + length_directive(cfg or {})
            assert blind.replace(own, shared) == generic, f"{mode} differs from generic beyond its instruction"


def compress(client, pair: dict, notes: str | None, mode: str, stage: int, cfg: dict) -> dict:
    """Produce one handoff message.

    ``passthrough`` is the no-rewrite control and therefore issues no model call
    at all: it forwards its input unchanged (the raw context at stage 1). Doing
    it here rather than as a separate code path keeps every arm on one
    execution loop, so nothing else about the run can drift between arms.
    """
    if mode in MODES_WITHOUT_MODEL_CALL:
        text = render_context(pair) if stage == 1 else (notes or "")
        return {"text": text, "cached": True, "prompt_tokens": 0, "completion_tokens": 0,
                "finish_reason": "passthrough"}
    user = compression_user_prompt(pair, notes, mode, stage, cfg)
    result = client.chat([{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["subagent_temperature"],
                         max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                         tag=f"summary_generalization_{mode}_stage{stage}")
    return {"text": result.text.strip(), "cached": result.cached,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason}


def write_example(pair: dict, handoffs: dict, root: Path, modes: list[str] | None = None) -> None:
    modes = list(modes or ("conditioned", "generic"))
    lines = [
        "# Example: question-only conditioning difference",
        "",
        f"**Question A passed only in the conditioned arm:** {pair['question_A']}",
        "",
        f"**Unrelated held-out Question B:** {pair['question_B']}",
        "",
        pair.get("context_note",
                  "Both gold passages and the same eight distractors were present in the shared depth-0 context."),
    ]
    for stage in (1, 5):
        for mode in modes:
            row = handoffs.get((pair["pair_id"], mode, stage))
            if row is None:
                continue
            lines += ["", f"## {mode}, stage {stage}", "", row["text"]]
    root.mkdir(parents=True, exist_ok=True)
    (root / "example.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def answer(client, pair: dict, query_type: str, material: str, cfg: dict, tag: str) -> dict:
    suffix = "A" if query_type == "target" else "B"
    question, golds = pair[f"question_{suffix}"], pair[f"golds_{suffix}"]
    user = f"Research material:\n{material}\n\nQuestion:\n{question}\nAnswer:"
    result = client.chat([{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["orchestrator_temperature"],
                         max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    pred = extract_short_answer(result.text)
    em, f1 = score_against_golds(pred, golds)
    return {"question": question, "golds": golds, "pred": pred, "raw": result.text,
            "em": em, "f1": f1, "cached": result.cached}


# --- Transition-level measurement (M_i -> M_{i+1}) ---------------------------

TRANSITION_KEY = ("pair_id", "mode", "to_stage")
# Fields copied back onto the per-stage handoff rows. Deliberately a compact
# subset: handoffs.jsonl stays a handoff record with transition annotations,
# not a second copy of the transition table.
HANDOFF_ANNOTATION_FIELDS = (
    "lexical_similarity", "lexical_change", "novel_token_rate", "ngram_copy_rate",
    "length_ratio", "fact_A_survived", "fact_B_survived",
    "semantic_preserved", "semantic_verdict",
    "transition_label_target", "transition_label_heldout",
)


def probe_survival(text: str, golds: list[str]) -> tuple[float, bool]:
    """Exact answer-string survival in a handoff, reusing the shared helper.

    Returns (survived, measurable). ``mentions_answer`` ignores golds under
    three characters, so a pair whose gold is that short cannot be probed this
    way; that is reported rather than scored as a loss.
    """
    usable = [g for g in golds if g and len(str(g).strip()) >= 3]
    if not usable:
        return float("nan"), False
    return float(mentions_answer(text, [str(g) for g in usable])), True


def build_transition_rows(pairs: list[dict], handoffs: dict, cfg: dict) -> list[dict]:
    """One row per consecutive handoff edge, with every deterministic measure.

    Stage 1 is included as the ``from_stage: 0`` edge out of the source context.
    It is the compression step; stages 2+ are the rewriting steps. Keeping both
    in one table is what lets the analysis separate the two.
    """
    rows: list[dict] = []
    for pair in pairs:
        context = render_context(pair)
        for mode in cfg["modes"]:
            for stage in range(1, cfg["max_depth"] + 1):
                current = handoffs[(pair["pair_id"], mode, stage)]["text"]
                previous = context if stage == 1 else handoffs[(pair["pair_id"], mode, stage - 1)]["text"]
                survived_a, measurable_a = probe_survival(current, pair["golds_A"])
                survived_b, measurable_b = probe_survival(current, pair["golds_B"])
                prev_a, _ = probe_survival(previous, pair["golds_A"])
                prev_b, _ = probe_survival(previous, pair["golds_B"])
                rows.append({
                    "pair_id": pair["pair_id"], "mode": mode,
                    "from_stage": stage - 1, "to_stage": stage, "depth": stage,
                    "edge_type": "source_compression" if stage == 1 else "message_rewrite",
                    **pmx.transition_measures(previous, current),
                    "fact_A_survived": survived_a, "fact_A_measurable": measurable_a,
                    "fact_B_survived": survived_b, "fact_B_measurable": measurable_b,
                    # A fact present before this edge and absent after it was
                    # lost *at* this edge, which is what per-edge attribution
                    # needs; a fact already gone cannot be lost again.
                    "fact_A_lost_here": float(prev_a == 1.0 and survived_a == 0.0) if measurable_a else float("nan"),
                    "fact_B_lost_here": float(prev_b == 1.0 and survived_b == 0.0) if measurable_b else float("nan"),
                    "previous_text": previous, "current_text": current,
                })
    return rows


def attach_answer_outcomes(transitions: list[dict], answer_rows: list[dict]) -> None:
    """Join the downstream answer produced from each message onto its edge."""
    lookup = {(r["mode"], r["query_type"], int(r["depth"]), r["pair_id"]): r for r in answer_rows}
    for row in transitions:
        for query_type, suffix in (("target", "target"), ("heldout", "heldout")):
            answer = lookup.get((row["mode"], query_type, int(row["to_stage"]), row["pair_id"]))
            if answer is None:
                continue
            for metric in ("em", "f1", "judge_correct"):
                if metric in answer:
                    row[f"{suffix}_{metric}"] = answer[metric]


def label_transitions(transitions: list[dict], cfg: dict) -> None:
    thresholds = (cfg.get("analysis") or {}).get("transition_thresholds") or {}
    for row in transitions:
        semantic = row.get("semantic_preserved")
        if semantic is None or (isinstance(semantic, float) and semantic != semantic):
            semantic = None
        for suffix, survived_key in (("target", "fact_A_survived"), ("heldout", "fact_B_survived")):
            survived = row.get(survived_key)
            measurable = row.get(f"fact_{'A' if suffix == 'target' else 'B'}_measurable")
            facts = None if not measurable else bool(survived == 1.0)
            row[f"transition_label_{suffix}"] = pmx.classify_transition(
                float(row["lexical_similarity"]), semantic, facts,
                row.get(f"{suffix}_judge_correct"), thresholds)


def annotate_handoff_rows(handoffs: dict, transitions: list[dict]) -> None:
    """Copy the compact annotation subset onto the per-stage handoff records."""
    for row in transitions:
        key = (row["pair_id"], row["mode"], int(row["to_stage"]))
        target = handoffs.get(key)
        if target is None:
            continue
        for field in HANDOFF_ANNOTATION_FIELDS:
            if field in row:
                target[field] = row[field]


def merge_cached_transitions(fresh: list[dict], stored: list[dict]) -> None:
    """Carry stored semantic verdicts onto freshly rebuilt rows.

    Transition rows are rebuilt from ``handoffs.jsonl`` every run because that
    is free and keeps them consistent with the messages actually on disk. Only
    the judged fields cost money, so only those are reused -- and only when the
    message pair they were produced from is byte-identical, which the stored
    text hashes prove.
    """
    index = {tuple(r[k] for k in TRANSITION_KEY): r for r in stored}
    for row in fresh:
        previous = index.get(tuple(row[k] for k in TRANSITION_KEY))
        if previous is None or "semantic_preserved" not in previous:
            continue
        if previous.get("text_pair_hash") != transition_hash(row):
            continue
        for field in ("semantic_preserved", "semantic_verdict", "semantic_parse_ok", "semantic_raw"):
            if field in previous:
                row[field] = previous[field]


def transition_hash(row: dict) -> str:
    import hashlib
    digest = hashlib.sha256()
    digest.update(str(row.get("previous_text", "")).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(row.get("current_text", "")).encode("utf-8"))
    return digest.hexdigest()[:16]


def storable_transition(row: dict) -> dict:
    """Drop the full message texts, keeping a hash that identifies them."""
    out = {k: v for k, v in row.items() if k not in ("previous_text", "current_text")}
    out["text_pair_hash"] = transition_hash(row)
    return out


def analyse_transitions(transitions: list[dict], cfg: dict, root: Path) -> None:
    """Aggregate per-edge measures by arm and depth, then plot them."""
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    continuous = ["lexical_similarity", "lexical_change", "novel_token_rate", "ngram_copy_rate",
                  "retained_token_rate", "length_ratio", "semantic_preserved",
                  "fact_A_survived", "fact_B_survived", "fact_A_lost_here", "fact_B_lost_here",
                  "target_f1", "heldout_f1", "target_judge_correct", "heldout_judge_correct"]
    records = []
    for mode in cfg["modes"]:
        for stage in sorted({int(r["to_stage"]) for r in transitions}):
            subset = [r for r in transitions if r["mode"] == mode and int(r["to_stage"]) == stage]
            if not subset:
                continue
            record = {"mode": mode, "from_stage": stage - 1, "to_stage": stage, "depth": stage,
                      "edge_type": subset[0]["edge_type"], "n": len(subset)}
            for metric in continuous:
                values = np.array([float(r[metric]) for r in subset if metric in r
                                   and float(r[metric]) == float(r[metric])], dtype=float)
                if values.size == 0:
                    record[metric] = float("nan")
                    continue
                mean, lo, hi = bootstrap_ci(values, boot, ci, seed=83)
                record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4),
                               f"{metric}_hi": round(hi, 4)})
            for label in pmx.LABELS:
                record[f"label_target_{label}"] = round(
                    sum(1 for r in subset if r.get("transition_label_target") == label) / len(subset), 4)
            records.append(record)
    write_csv(root / "transition_metrics.csv", records)
    plot_transitions(records, cfg["modes"], root)


def plot_transitions(records: list[dict], modes: list[str], root: Path) -> None:
    """Four aligned views of the same edges: wording, meaning, facts, answers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("lexical_similarity", "Lexical similarity to previous message", (-.02, 1.02)),
        ("semantic_preserved", "Judged semantic preservation", (-.02, 1.02)),
        ("fact_B_survived", "Held-out fact B present in message", (-.02, 1.02)),
        ("target_judge_correct", "Answer accuracy on target question A", (-.02, 1.02)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for index, (metric, title, ylim) in enumerate(panels):
        axis = axes[index // 2][index % 2]
        for mode in modes:
            subset = sorted([r for r in records if r["mode"] == mode], key=lambda r: r["to_stage"])
            points = [(r["to_stage"], r[metric]) for r in subset
                      if metric in r and float(r[metric]) == float(r[metric])]
            if not points:
                continue
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                      markersize=4, linewidth=1.8, color=MODE_COLORS.get(mode), label=mode)
        axis.set(title=title, xlabel="Handoff edge (stage)", ylim=ylim)
        axis.grid(alpha=.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Per-edge paraphrasing, meaning preservation, fact survival, and answer accuracy")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005,
             "Edge 1 compresses the source context; edges 2+ rewrite the previous message. "
             "'passthrough' copies its input forward with no model call.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(root / "paraphrase_transitions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyse(rows: list[dict], cfg: dict, root: Path, pairs: list[dict], handoffs: dict) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    pair_lookup = {p["pair_id"]: p for p in pairs}

    def mean_chars(mode: str, depth: int, pair_ids: list[str]) -> float:
        # "direct" (depth 0) has no generated summary -- the answerer sees the
        # raw context instead, so size that marker by context length. Every
        # other point is sized by the actual generated handoff/summary text,
        # so shrinkage across repeated compression is visible on the accuracy
        # curve itself, not just in a separate token-count table.
        if mode == "direct":
            lengths = [len(render_context(pair_lookup[pid])) for pid in pair_ids]
        else:
            lengths = [len(handoffs[(pid, mode, depth)]["text"]) for pid in pair_ids]
        return sum(lengths) / len(lengths)

    metrics, vectors = [], {}
    keys = sorted(set((r["mode"], r["query_type"], int(r["depth"])) for r in rows))
    for mode, query_type, depth in keys:
        subset = sorted([r for r in rows if (r["mode"], r["query_type"], int(r["depth"])) == (mode, query_type, depth)], key=lambda x: x["pair_id"])
        vectors[(mode, query_type, depth)] = {m: {r["pair_id"]: r[m] for r in subset} for m in ("em", "f1", "judge_correct")}
        record = {"mode": mode, "query_type": query_type, "depth": depth, "n": len(subset),
                  "handoff_characters_mean": round(mean_chars(mode, depth, [r["pair_id"] for r in subset]), 1)}
        for metric in ("em", "f1", "judge_correct"):
            mean, lo, hi = bootstrap_ci(np.array([r[metric] for r in subset]), boot, ci, seed=71)
            record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
        metrics.append(record)
    deltas = []
    for query_type in ("target", "heldout"):
        base = vectors[("direct", query_type, 0)]
        for mode in cfg["modes"]:
            for depth in [d for d in cfg["depths"] if d > 0]:
                current = vectors[(mode, query_type, depth)]
                for metric in ("em", "f1", "judge_correct"):
                    ids = sorted(set(base[metric]) & set(current[metric]))
                    result = paired_bootstrap_delta(np.array([current[metric][i] for i in ids]), np.array([base[metric][i] for i in ids]), boot, ci, seed=73)
                    deltas.append({"comparison": "depth_minus_direct", "mode": mode, "query_type": query_type, "depth": depth, "metric": metric, **result})
        # Every arm pair the config actually ran, in a fixed order. A config
        # with only the original two arms produces exactly the rows it always
        # produced, under the same comparison name.
        for treatment, control in CONTRAST_PAIRS:
            if treatment not in cfg["modes"] or control not in cfg["modes"]:
                continue
            name = f"{treatment}_minus_{control}"
            for depth in [d for d in cfg["depths"] if d > 0]:
                left, right = vectors[(treatment, query_type, depth)], vectors[(control, query_type, depth)]
                for metric in ("em", "f1", "judge_correct"):
                    ids = sorted(set(left[metric]) & set(right[metric]))
                    result = paired_bootstrap_delta(np.array([left[metric][i] for i in ids]), np.array([right[metric][i] for i in ids]), boot, ci, seed=79)
                    deltas.append({"comparison": name, "mode": name, "query_type": query_type, "depth": depth, "metric": metric, **result})
    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)
    plot_summary_generalization(metrics, cfg["modes"], cfg["depths"], root)


def plot_summary_generalization(metrics: list[dict], modes: list[str], depths: list[int], root: Path) -> None:
    """Draw the four-panel F1/judge plot from an already-built metrics list.

    Deliberately takes only ``metrics`` (not raw answer rows, pairs, or
    handoffs) so a plot can be regenerated -- e.g. after a marker-size tuning
    change -- straight from a saved metrics.csv, without needing the original
    run's cached handoffs/answers or its (possibly since-removed) pair
    constructor to still be present.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_metrics = [("f1", "Token F1"), ("judge_correct", "LLM-judge accuracy")]
    fig, axes = plt.subplots(len(plot_metrics), 2, figsize=(11, 4.5 * len(plot_metrics)),
                             sharey="row", squeeze=False)
    colors = MODE_COLORS
    titles = ("Conditioning question A", "Unrelated held-out question B")
    # Marker area proportional to the mean size (characters) of whatever text
    # actually fed the answerer at that point -- the raw context at depth 0,
    # the generated summary at depth >=1 -- one scale for the whole figure so
    # area is comparable across both panels, both arms, and both metric rows.
    # Kept deliberately small (max_area well under matplotlib's own default
    # marker area of ~36pt^2 x a few): the first version of this encoding
    # (max_area=900) produced markers large enough to visually dominate the
    # line they sit on and crowd into neighbouring points.
    all_chars = [r["handoff_characters_mean"] for r in metrics if "handoff_characters_mean" in r]
    max_chars = max(all_chars) if all_chars else 1.0
    max_area, min_area = 220.0, 6.0
    size_scale = max_area / max_chars

    def marker_area(record: dict) -> float:
        chars = record.get("handoff_characters_mean")
        return max_area if chars is None else max(min_area, size_scale * chars)

    for row_idx, (metric, label) in enumerate(plot_metrics):
        for col_idx, (query_type, title) in enumerate(zip(("target", "heldout"), titles)):
            axis = axes[row_idx][col_idx]
            direct = next(r for r in metrics if r["mode"] == "direct" and r["query_type"] == query_type)
            axis.scatter([0], [direct[metric]], s=marker_area(direct), color=colors["direct"],
                        edgecolors="white", linewidths=0.6, label="direct context", zorder=3)
            for mode in modes:
                subset = sorted([r for r in metrics if r["mode"] == mode and r["query_type"] == query_type], key=lambda r: r["depth"])
                x = [r["depth"] for r in subset]
                axis.plot(x, [r[metric] for r in subset], marker="", linewidth=2, color=colors.get(mode), label=mode, zorder=2)
                axis.scatter(x, [r[metric] for r in subset], s=[marker_area(r) for r in subset],
                            color=colors.get(mode), edgecolors="white", linewidths=0.6, zorder=3)
            if row_idx == 0:
                axis.set_title(title)
            if row_idx == len(plot_metrics) - 1:
                axis.set_xlabel("Compression handoffs")
            axis.set_xticks(depths)
            axis.grid(alpha=.25)
        axes[row_idx][0].set_ylabel(label)
    axes[0][1].legend()
    fig.suptitle("Question-only conditioning and summary generalizability")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.5, 0.005,
              f"Marker area ∝ mean characters in the answerer's input at that point "
              f"(raw context at depth 0, else the generated summary) — smallest marker "
              f"{min_area:.0f}pt² floor, largest ≈{max_chars:,.0f} characters.",
              ha="center", fontsize=8, color="#555555")
    fig.savefig(root / "summary_generalization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "summary_generalization_squad_pairs_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # Request-rate override. The OpenRouter key is shared across every runner in
    # this project, so a second experiment started while a long job is running
    # can push the account into 429s and exhaust that job's retries. Throttling
    # is a per-invocation decision, not a property of the experiment, so it
    # belongs on the command line rather than baked into the config.
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override runtime.concurrency for this run only")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.concurrency is not None:
        if args.concurrency < 1:
            parser.error("--concurrency must be at least 1")
        cfg["runtime"]["concurrency"] = args.concurrency
        print(f"[generalization:runtime] concurrency overridden to {args.concurrency}")
    validate_modes(cfg)
    n = args.n or cfg["dataset"]["n_pairs"]
    pairs = load_prebuilt_pairs(cfg, n)
    if cfg["dataset"].get("experiment_context") == "gold_only":
        pairs = gold_only_pairs(pairs, cfg)
    prompt_difference_selftest(pairs[0], cfg)
    print("[generalization:selftest] conditioned/generic prompts differ only by the Question A block; "
          f"question-blind arms leak no question ({', '.join(cfg['modes'])})")
    if args.construct_only:
        return 0
    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    result_root.mkdir(parents=True, exist_ok=True)
    handoff_path = run_root / "handoffs.jsonl"
    handoffs = {(r["pair_id"], r["mode"], int(r["stage"])): r for r in read_jsonl(handoff_path)}
    for mode in cfg["modes"]:
        for stage in range(1, cfg["max_depth"] + 1):
            missing = [p for p in pairs if (p["pair_id"], mode, stage) not in handoffs]
            def build(pair):
                previous = None if stage == 1 else handoffs[(pair["pair_id"], mode, stage - 1)]["text"]
                return compress(client, pair, previous, mode, stage, cfg)
            with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
                futures = {pool.submit(build, p): p for p in missing}
                for future, pair in futures.items():
                    handoffs[(pair["pair_id"], mode, stage)] = {"pair_id": pair["pair_id"], "mode": mode, "stage": stage, **future.result()}
            if not args.dry_run:
                write_jsonl(handoff_path, [r for _, r in sorted(handoffs.items())])
            stage_rows = [handoffs[(pair["pair_id"], mode, stage)] for pair in pairs]
            # A handoff cut off at the cap is not a summary the model chose to end --
            # it is a truncated one, and the lost tail is silently absent from every
            # later stage. Surface it loudly; a binding cap invalidates the arm.
            truncated = [r for r in stage_rows if r.get("finish_reason") == "length"]
            note = ""
            if truncated:
                note = (f"  WARNING: {len(truncated)}/{len(stage_rows)} hit the "
                        f"{cfg['decoding']['handoff_max_tokens']}-token cap (truncated mid-summary)")
            print(f"[generalization:handoff] {mode}/stage{stage}: {len(missing)} generated{note}")
    answer_path = run_root / "answers.jsonl"
    existing = {(r["mode"], r["query_type"], int(r["depth"]), r["pair_id"]): r for r in read_jsonl(answer_path)}
    jobs = []
    for pair in pairs:
        for query_type in ("target", "heldout"):
            key = ("direct", query_type, 0, pair["pair_id"])
            if key not in existing:
                jobs.append((key, pair, render_context(pair)))
            for mode in cfg["modes"]:
                for depth in [d for d in cfg["depths"] if d > 0]:
                    key = (mode, query_type, depth, pair["pair_id"])
                    if key not in existing:
                        jobs.append((key, pair, handoffs[(pair["pair_id"], mode, depth)]["text"]))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, pair, key[1], material, cfg, f"summary_generalization_{key[0]}_{key[1]}_d{key[2]}"): key for key, pair, material in jobs}
        for future, key in futures.items():
            existing[key] = {"mode": key[0], "query_type": key[1], "depth": key[2], "pair_id": key[3], **future.result()}
    rows = sorted(existing.values(), key=lambda r: (r["mode"], r["query_type"], int(r["depth"]), r["pair_id"]))
    if not args.dry_run:
        write_jsonl(answer_path, rows)
        judge_client = add_judge(rows, cfg, tag="summary_generalization_judge")
        write_jsonl(answer_path, rows)

        # Transition-level pass. Rebuilt from the messages on disk every run
        # (free, deterministic), with only the judged fields carried over from
        # the previous run's file.
        transition_path = run_root / "transitions.jsonl"
        transitions = build_transition_rows(pairs, handoffs, cfg)
        merge_cached_transitions(transitions, read_jsonl(transition_path))
        attach_answer_outcomes(transitions, rows)
        preservation_client = add_preservation_judge(
            transitions, cfg, tag="summary_generalization_preservation")
        label_transitions(transitions, cfg)
        write_jsonl(transition_path, [storable_transition(r) for r in transitions])
        annotate_handoff_rows(handoffs, transitions)
        write_jsonl(handoff_path, [r for _, r in sorted(handoffs.items())])
        analyse_transitions(transitions, cfg, result_root)

        analyse(rows, cfg, result_root, pairs, handoffs)
        write_example(pairs[0], handoffs, result_root, cfg["modes"])
        rewrites = [r for r in transitions if r["edge_type"] == "message_rewrite"]
        if rewrites:
            mean_sim = sum(float(r["lexical_similarity"]) for r in rewrites) / len(rewrites)
            print(f"[generalization:transitions] {len(transitions)} edges "
                  f"({len(rewrites)} rewrites); mean rewrite lexical similarity {mean_sim:.3f}")
        if judge_client is not None:
            print("[generalization:judge cost] " + json.dumps(judge_client.ledger.summary()))
        if preservation_client is not None:
            print("[generalization:preservation cost] " + json.dumps(preservation_client.ledger.summary()))
    print(f"[generalization:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
