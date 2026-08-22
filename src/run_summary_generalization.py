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
from llm import LLMClient, load_config  # noqa: E402
from judge import add_judge  # noqa: E402
from run_chain import (  # noqa: E402
    CHAIN_SYSTEM,
    INITIAL_INSTRUCTION,
    RECOMPRESS_INSTRUCTION,
)
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402

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


def compression_user_prompt(pair: dict, notes: str | None, mode: str, stage: int,
                            cfg: dict | None = None) -> str:
    """Build prompts whose only arm-level difference is the question block."""
    if stage == 1:
        material = f"Source material:\n{render_context(pair)}"
        instruction = INITIAL_INSTRUCTION
    else:
        material = f"Previous agent's notes:\n{notes}"
        instruction = RECOMPRESS_INSTRUCTION
    instruction = instruction + length_directive(cfg or {})
    if mode == "conditioned":
        user = f"{material}\n\nQuestion the final agent must answer: {pair['question_A']}\n\n{instruction}"
    else:
        user = f"{material}\n\n{instruction}"
        assert pair["question_A"] not in user and pair["question_B"] not in user
    return user


def prompt_difference_selftest(pair: dict, cfg: dict | None = None) -> None:
    """Prove that deleting the question block makes the prompts byte-identical."""
    marker = f"\n\nQuestion the final agent must answer: {pair['question_A']}"
    for stage, notes in ((1, None), (2, "identical previous notes")):
        conditioned = compression_user_prompt(pair, notes, "conditioned", stage, cfg)
        generic = compression_user_prompt(pair, notes, "generic", stage, cfg)
        assert conditioned.replace(marker, "") == generic


def compress(client, pair: dict, notes: str | None, mode: str, stage: int, cfg: dict) -> dict:
    user = compression_user_prompt(pair, notes, mode, stage, cfg)
    result = client.chat([{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["subagent_temperature"],
                         max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                         tag=f"summary_generalization_{mode}_stage{stage}")
    return {"text": result.text.strip(), "cached": result.cached,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason}


def write_example(pair: dict, handoffs: dict, root: Path) -> None:
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
        for mode in ("conditioned", "generic"):
            lines += ["", f"## {mode}, stage {stage}", "", handoffs[(pair["pair_id"], mode, stage)]["text"]]
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
        for depth in [d for d in cfg["depths"] if d > 0]:
            conditioned, generic = vectors[("conditioned", query_type, depth)], vectors[("generic", query_type, depth)]
            for metric in ("em", "f1", "judge_correct"):
                ids = sorted(set(conditioned[metric]) & set(generic[metric]))
                result = paired_bootstrap_delta(np.array([conditioned[metric][i] for i in ids]), np.array([generic[metric][i] for i in ids]), boot, ci, seed=79)
                deltas.append({"comparison": "conditioned_minus_generic", "mode": "conditioned_minus_generic", "query_type": query_type, "depth": depth, "metric": metric, **result})
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
    colors = {"direct": "#7570b3", "conditioned": "#d95f02", "generic": "#1b9e77"}
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
                axis.plot(x, [r[metric] for r in subset], marker="", linewidth=2, color=colors[mode], label=mode, zorder=2)
                axis.scatter(x, [r[metric] for r in subset], s=[marker_area(r) for r in subset],
                            color=colors[mode], edgecolors="white", linewidths=0.6, zorder=3)
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_pairs"]
    pairs = load_prebuilt_pairs(cfg, n)
    if cfg["dataset"].get("experiment_context") == "gold_only":
        pairs = gold_only_pairs(pairs, cfg)
    prompt_difference_selftest(pairs[0], cfg)
    print("[generalization:selftest] conditioned/generic prompts differ only by the Question A block")
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
        analyse(rows, cfg, result_root, pairs, handoffs)
        write_example(pairs[0], handoffs, result_root)
        if judge_client is not None:
            print("[generalization:judge cost] " + json.dumps(judge_client.ledger.summary()))
    print(f"[generalization:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
