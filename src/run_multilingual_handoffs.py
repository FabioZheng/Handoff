"""Experiment 6: repeated handoffs with fixed versus switching languages.

The design is a 2x2 factorial comparison on the corrected SQuAD same-passage
dataset. Every experimental input contains only the single shared gold passage;
the nine retrieval distractors stored in the reusable source dataset are
removed before prompting:

* question-conditioned versus generic compression;
* one fixed language versus a different supported language at every handoff.

Each example receives a stratified starting language. Its fixed and switching
arms therefore have byte-identical stage-1 prompts; only later stages can
differ because the switching arm changes output language. The final answerer
always receives the original English question. EM/F1 and a different-family
LLM judge score the answer, while a separate language audit checks compliance.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from langdetect import DetectorFactory, detect_langs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
from judge import add_judge, make_judge_client  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from run_chain import CHAIN_SYSTEM, INITIAL_INSTRUCTION, RECOMPRESS_INSTRUCTION  # noqa: E402
from run_summary_generalization import gold_only_pairs, load_prebuilt_pairs, render_context  # noqa: E402
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402

LANGUAGE_DIRECTIVE = (
    "OUTPUT LANGUAGE (mandatory): {language}. Write the entire replacement handoff in "
    "{language}. Proper names, identifiers, numbers, and short source quotations may remain "
    "unchanged when translation would alter them. Do not mix in another language for the prose. "
    "Summarize; do not translate or rewrite the source passage by passage. Do not use headings or "
    "one section per passage."
)
LANGUAGE_JUDGE_SYSTEM = (
    "You are a strict language-identification auditor. Reply with exactly MATCH or MISMATCH."
)
DetectorFactory.seed = 0


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


def arm_name(conditioning: str, schedule: str) -> str:
    return f"{conditioning}_{schedule}"


def experiment_fingerprint(cfg: dict, pairs: list[dict]) -> str:
    payload = {
        "model": cfg["model"], "decoding": cfg["decoding"],
        "conditioning": cfg["conditioning"], "schedules": cfg["schedules"],
        "depths": cfg["depths"], "languages": cfg["languages"],
        "experiment_context": cfg["dataset"].get("experiment_context"),
        "language_directive": LANGUAGE_DIRECTIVE,
        "chain_system": CHAIN_SYSTEM, "initial_instruction": INITIAL_INSTRUCTION,
        "recompress_instruction": RECOMPRESS_INSTRUCTION,
        "pairs": pairs,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def language_for(pair_index: int, schedule: str, stage: int, languages: list[dict]) -> dict:
    start = pair_index % len(languages)
    offset = 0 if schedule == "fixed" else stage - 1
    return languages[(start + offset) % len(languages)]


def compression_prompt(pair: dict, notes: str | None, conditioning: str,
                       language: dict, stage: int) -> str:
    if stage == 1:
        material = f"Source material:\n{render_context(pair)}"
        instruction = INITIAL_INSTRUCTION
    else:
        material = f"Previous agent's notes:\n{notes}"
        instruction = RECOMPRESS_INSTRUCTION
    question_block = (
        f"\n\nQuestion the final agent must answer: {pair['question_A']}"
        if conditioning == "conditioned" else ""
    )
    directive = LANGUAGE_DIRECTIVE.format(language=language["name"])
    user = f"{material}{question_block}\n\n{instruction}\n\n{directive}"
    if conditioning == "generic":
        assert pair["question_A"] not in user and pair["question_B"] not in user
    return user


def prompt_selftest(pair: dict, languages: list[dict]) -> None:
    """Verify question visibility and the matched stage-1 schedule control."""
    marker = f"\n\nQuestion the final agent must answer: {pair['question_A']}"
    for stage, notes in ((1, None), (2, "identical prior notes")):
        conditioned = compression_prompt(pair, notes, "conditioned", languages[0], stage)
        generic = compression_prompt(pair, notes, "generic", languages[0], stage)
        assert conditioned.replace(marker, "") == generic
    fixed = compression_prompt(pair, None, "conditioned", languages[0], 1)
    switching = compression_prompt(pair, None, "conditioned", languages[0], 1)
    assert fixed == switching


def compress(client: LLMClient, pair: dict, notes: str | None, conditioning: str,
             schedule: str, language: dict, stage: int, cfg: dict) -> dict:
    prompt = compression_prompt(pair, notes, conditioning, language, stage)
    result = client.chat(
        [{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        seed=stage,
        tag=f"multilingual_{conditioning}_{schedule}_{language['code']}_d{stage}",
    )
    text = result.text.strip()
    return {
        "text": text, "cached": result.cached,
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason, "summary_characters": len(text),
        "summary_words": len(text.split()),
        "requested_language": language["name"], "language_code": language["code"],
    }


def language_audit(handoffs: list[dict], cfg: dict) -> LLMClient | None:
    spec = dict(cfg.get("language_judge") or {})
    if not spec.get("enabled", False):
        return None
    for row in handoffs:
        # Migrate records created before deterministic language identification
        # became the primary compliance diagnostic.
        if "language_judge_match" not in row and "language_match" in row:
            row["language_judge_match"] = row.pop("language_match")
    missing = [r for r in handoffs if "language_judge_match" not in r]
    if not missing:
        print(f"[multilingual:language-judge] reusing {len(handoffs)} verdicts")
        return None
    judge_cfg = copy.deepcopy(cfg)
    judge_cfg["judge"] = {
        "enabled": True, "model_id": spec["model_id"],
        "max_tokens": spec.get("max_tokens", 8), "cap_usd": spec.get("cap_usd", 1.0),
        "concurrency": spec.get("concurrency", 20),
    }
    client, judge_spec = make_judge_client(judge_cfg)

    def one(row: dict) -> dict:
        prompt = (
            f"Expected language: {row['requested_language']}\n\nText:\n{row['text']}\n\n"
            "Reply MATCH if the prose is predominantly in the expected language. Allow proper names, "
            "numbers, identifiers, and short untranslated quotations. Otherwise reply MISMATCH."
        )
        result = client.chat(
            [{"role": "system", "content": LANGUAGE_JUDGE_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=int(judge_spec["max_tokens"]), seed=None,
            tag="multilingual_language_compliance",
        )
        verdict = result.text.strip().upper()
        return {"language_judge_match": int(verdict == "MATCH"),
                "language_judge_parse_ok": verdict in {"MATCH", "MISMATCH"},
                "language_judge_raw": result.text.strip()}

    with ThreadPoolExecutor(max_workers=int(judge_spec["concurrency"])) as pool:
        futures = {pool.submit(one, row): row for row in missing}
        for future, row in futures.items():
            row.update(future.result())
    print(f"[multilingual:language-judge] scored {len(missing)} handoffs")
    return client


def apply_language_detection(handoffs: list[dict]) -> None:
    """Add deterministic predominant-language compliance fields."""
    for row in handoffs:
        try:
            ranked = detect_langs(row["text"])
            detected = ranked[0].lang if ranked else "unknown"
            probability = float(ranked[0].prob) if ranked else 0.0
        except Exception:
            detected, probability = "unknown", 0.0
        row["detected_language_code"] = detected
        row["detected_language_probability"] = round(probability, 4)
        row["language_match"] = int(detected == row["language_code"])


def answer(client: LLMClient, pair: dict, query_type: str, material: str,
           cfg: dict, tag: str) -> dict:
    suffix = "A" if query_type == "target" else "B"
    question, golds = pair[f"question_{suffix}"], pair[f"golds_{suffix}"]
    user = f"Research material:\n{material}\n\nQuestion:\n{question}\nAnswer:"
    result = client.chat(
        [{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["orchestrator_temperature"],
        max_tokens=cfg["decoding"]["answer_max_tokens"],
        seed=None, tag=tag,
    )
    pred = extract_short_answer(result.text)
    em, f1 = score_against_golds(pred, golds)
    return {"question": question, "golds": golds, "pred": pred, "raw": result.text,
            "em": em, "f1": f1, "cached": result.cached}


def paired_values(rows: list[dict], arm: str, query_type: str, depth: int,
                  metric: str) -> dict[str, float]:
    return {r["pair_id"]: float(r[metric]) for r in rows
            if r["arm"] == arm and r["query_type"] == query_type and int(r["depth"]) == depth}


def analyse(rows: list[dict], handoffs: list[dict], pairs: list[dict], cfg: dict,
            root: Path) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    handoff_lookup = {(r["pair_id"], r["arm"], int(r["stage"])): r for r in handoffs}
    pair_lookup = {p["pair_id"]: p for p in pairs}
    metrics: list[dict] = []
    keys = sorted(set((r["arm"], r["query_type"], int(r["depth"])) for r in rows))
    for arm, query_type, depth in keys:
        subset = sorted([r for r in rows if (r["arm"], r["query_type"], int(r["depth"]))
                         == (arm, query_type, depth)], key=lambda r: r["pair_id"])
        if arm == "direct":
            chars = [len(render_context(pair_lookup[r["pair_id"]])) for r in subset]
            words = [len(render_context(pair_lookup[r["pair_id"]]).split()) for r in subset]
            compliance = None
            conditioning = schedule = "direct"
        else:
            hs = [handoff_lookup[(r["pair_id"], arm, depth)] for r in subset]
            chars = [len(h["text"]) for h in hs]
            words = [len(h["text"].split()) for h in hs]
            compliance = sum(h["language_match"] for h in hs) / len(hs)
            conditioning, schedule = arm.rsplit("_", 1)
        record = {
            "arm": arm, "conditioning": conditioning, "schedule": schedule,
            "query_type": query_type, "depth": depth, "n": len(subset),
            "summary_characters_mean": round(sum(chars) / len(chars), 1),
            "summary_words_mean": round(sum(words) / len(words), 1),
            "language_match_rate": "" if compliance is None else round(compliance, 4),
        }
        for metric in ("em", "f1", "judge_correct"):
            mean, lo, hi = bootstrap_ci(np.array([r[metric] for r in subset]), boot, ci, seed=91)
            record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4),
                           f"{metric}_hi": round(hi, 4)})
        metrics.append(record)

    deltas: list[dict] = []
    depths = [int(d) for d in cfg["depths"] if int(d) > 0]
    for query_type in ("target", "heldout"):
        for depth in depths:
            for metric in ("em", "f1", "judge_correct"):
                for schedule in cfg["schedules"]:
                    left = paired_values(rows, arm_name("conditioned", schedule), query_type, depth, metric)
                    right = paired_values(rows, arm_name("generic", schedule), query_type, depth, metric)
                    ids = sorted(set(left) & set(right))
                    result = paired_bootstrap_delta(
                        np.array([left[i] for i in ids]), np.array([right[i] for i in ids]),
                        boot, ci, seed=93)
                    deltas.append({"comparison": "conditioned_minus_generic", "stratum": schedule,
                                   "query_type": query_type, "depth": depth, "metric": metric, **result})
                for conditioning in cfg["conditioning"]:
                    left = paired_values(rows, arm_name(conditioning, "switching"), query_type, depth, metric)
                    right = paired_values(rows, arm_name(conditioning, "fixed"), query_type, depth, metric)
                    ids = sorted(set(left) & set(right))
                    result = paired_bootstrap_delta(
                        np.array([left[i] for i in ids]), np.array([right[i] for i in ids]),
                        boot, ci, seed=95)
                    deltas.append({"comparison": "switching_minus_fixed", "stratum": conditioning,
                                   "query_type": query_type, "depth": depth, "metric": metric, **result})
                cs = paired_values(rows, arm_name("conditioned", "switching"), query_type, depth, metric)
                gs = paired_values(rows, arm_name("generic", "switching"), query_type, depth, metric)
                cf = paired_values(rows, arm_name("conditioned", "fixed"), query_type, depth, metric)
                gf = paired_values(rows, arm_name("generic", "fixed"), query_type, depth, metric)
                ids = sorted(set(cs) & set(gs) & set(cf) & set(gf))
                switch_effect = np.array([cs[i] - gs[i] for i in ids])
                fixed_effect = np.array([cf[i] - gf[i] for i in ids])
                result = paired_bootstrap_delta(switch_effect, fixed_effect, boot, ci, seed=97)
                deltas.append({"comparison": "conditioning_x_switching_interaction", "stratum": "factorial",
                               "query_type": query_type, "depth": depth, "metric": metric, **result})
    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)

    diagnostics: list[dict] = []
    for arm in sorted({r["arm"] for r in handoffs}):
        subset = [r for r in handoffs if r["arm"] == arm]
        diagnostics.append({
            "arm": arm, "handoffs": len(subset),
            "truncations": sum(r.get("finish_reason") == "length" for r in subset),
            "language_matches": sum(r.get("language_match", 0) for r in subset),
            "language_match_rate": round(sum(r.get("language_match", 0) for r in subset) / len(subset), 4),
            "summary_characters_mean": round(sum(len(r["text"]) for r in subset) / len(subset), 1),
            "summary_words_mean": round(sum(len(r["text"].split()) for r in subset) / len(subset), 1),
        })
    write_csv(root / "diagnostics.csv", diagnostics)
    make_plot(metrics, cfg, root / "multilingual_handoffs.png")
    make_plot(metrics, cfg, root / "multilingual_handoffs_compact.png", compact_markers=True)
    # Present each language schedule separately in the same two-series layout
    # used by Experiment 5.  Overlaying all four arms makes the conditioning
    # comparison difficult to read and is not visually comparable to that plot.
    for schedule in ("fixed", "switching"):
        make_plot(metrics, cfg, root / f"multilingual_handoffs_{schedule}.png",
                  schedule=schedule)
        make_plot(metrics, cfg, root / f"multilingual_handoffs_{schedule}_compact.png",
                  schedule=schedule, compact_markers=True)


def make_plot(metrics: list[dict], cfg: dict, output: Path, *, schedule: str | None = None,
              compact_markers: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_metrics = [("f1", "Token F1"), ("judge_correct", "LLM-judge accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row", squeeze=False)
    if schedule is None:
        colors = {
            "conditioned_fixed": "#d95f02", "generic_fixed": "#1b9e77",
            "conditioned_switching": "#7570b3", "generic_switching": "#e7298a",
        }
        linestyles = {"conditioned_fixed": "-", "generic_fixed": "-",
                      "conditioned_switching": "--", "generic_switching": "--"}
        labels = {
            "conditioned_fixed": "conditioned · fixed language",
            "generic_fixed": "generic · fixed language",
            "conditioned_switching": "conditioned · switching languages",
            "generic_switching": "generic · switching languages",
        }
    else:
        colors = {f"conditioned_{schedule}": "#d95f02", f"generic_{schedule}": "#1b9e77"}
        linestyles = {arm: "-" for arm in colors}
        labels = {f"conditioned_{schedule}": "conditioned",
                  f"generic_{schedule}": "generic"}
    all_sizes = [float(r["summary_characters_mean"]) for r in metrics]
    max_chars = max(all_sizes)
    # The compact figure keeps the same area∝input-length encoding as the main
    # figure; it merely uses a smaller absolute scale so overlapping tracks are
    # easier to follow.  Do not make its markers fixed-size: that would discard
    # the size signal the figure is meant to retain.
    # Matches Experiment 5's marker scale after its own too-large-markers fix;
    # compact halves it again for schedule-overlay figures with more tracks.
    min_area, max_area = (4.0, 110.0) if compact_markers else (6.0, 220.0)
    scale = max_area / max_chars

    def area(row: dict) -> float:
        return max(min_area, scale * float(row["summary_characters_mean"]))

    for row_index, (metric, ylabel) in enumerate(plot_metrics):
        for col_index, (query_type, title) in enumerate(
                (("target", "Conditioning target A"), ("heldout", "Held-out question B"))):
            ax = axes[row_index][col_index]
            direct = next(r for r in metrics if r["arm"] == "direct" and r["query_type"] == query_type)
            ax.scatter([0], [float(direct[metric])], s=area(direct), color="#7570b3",
                       edgecolors="white", linewidths=.6, label="direct context", zorder=4)
            for arm in colors:
                subset = sorted([r for r in metrics if r["arm"] == arm and r["query_type"] == query_type],
                                key=lambda r: int(r["depth"]))
                x = [int(r["depth"]) for r in subset]
                y = [float(r[metric]) for r in subset]
                ax.plot(x, y, color=colors[arm], linestyle=linestyles[arm], linewidth=2,
                        label=labels[arm], zorder=2)
                ax.scatter(x, y, s=[area(r) for r in subset], color=colors[arm],
                           edgecolors="white", linewidths=.6, zorder=3)
            if row_index == 0:
                ax.set_title(title)
            if row_index == 1:
                ax.set_xlabel("Compression handoffs")
            if col_index == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(cfg["depths"])
            ax.grid(alpha=.25)
    axes[0][1].legend(fontsize=8, loc="best")
    names = " → ".join(language["name"] for language in cfg["languages"])
    if schedule is None:
        title = "Experiment 6 (gold-only): fixed vs switching handoff languages"
        schedule_note = f"switching cycle: {names}"
    elif schedule == "fixed":
        title = "Experiment 6 (gold-only): question conditioning — fixed language"
        schedule_note = "one assigned language retained across handoffs"
    else:
        title = "Experiment 6 (gold-only): question conditioning — switching languages"
        schedule_note = f"switching cycle: {names}"
    if compact_markers:
        title += " — compact markers"
    fig.suptitle(title, fontsize=14, y=.965)
    fig.subplots_adjust(left=.085, right=.985, bottom=.115, top=.875,
                        hspace=.14, wspace=.04)
    fig.text(.5, .025, f"One shared gold passage, 0 distractors · {schedule_note}",
             ha="center", fontsize=8, color="#555555")
    size_note = ("Small markers; area ∝ mean answer-input size; shared scale maximum ≈ "
                 f"{max_chars:,.0f} characters."
                 if compact_markers else
                 f"Dot area ∝ mean answer-input size; shared scale maximum ≈ {max_chars:,.0f} characters.")
    fig.text(.5, .008, size_note, ha="center", fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "multilingual_handoff_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_pairs"]
    pairs = gold_only_pairs(load_prebuilt_pairs(cfg, n), cfg)
    languages = cfg["languages"]
    if cfg["max_depth"] != len(languages):
        raise ValueError("max_depth must equal the number of languages so switching uses each once")
    prompt_selftest(pairs[0], languages)
    print("[multilingual:selftest] question-block isolation and matched stage-1 prompts passed")
    fingerprint = experiment_fingerprint(cfg, pairs)

    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    handoff_path = run_root / "handoffs.jsonl"
    stored = [r for r in read_jsonl(handoff_path) if r.get("experiment_fingerprint") == fingerprint]
    handoffs = {(r["pair_id"], r["arm"], int(r["stage"])): r for r in stored}

    pair_indices = {pair["pair_id"]: index for index, pair in enumerate(pairs)}
    for stage in range(1, cfg["max_depth"] + 1):
        jobs = []
        for pair in pairs:
            pair_index = pair_indices[pair["pair_id"]]
            for conditioning in cfg["conditioning"]:
                for schedule in cfg["schedules"]:
                    arm = arm_name(conditioning, schedule)
                    key = (pair["pair_id"], arm, stage)
                    if key in handoffs:
                        continue
                    # At stage 1 the schedules have not diverged. Generate the
                    # fixed record once, then copy it byte-for-byte below.
                    if stage == 1 and schedule == "switching":
                        continue
                    previous = None if stage == 1 else handoffs[(pair["pair_id"], arm, stage - 1)]["text"]
                    language = language_for(pair_index, schedule, stage, languages)
                    jobs.append((key, pair, previous, conditioning, schedule, language))
        with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
            futures = {
                pool.submit(compress, client, pair, previous, conditioning, schedule,
                            language, stage, cfg): (key, conditioning, schedule)
                for key, pair, previous, conditioning, schedule, language in jobs
            }
            for future, (key, conditioning, schedule) in futures.items():
                handoffs[key] = {
                    "pair_id": key[0], "arm": key[1], "conditioning": conditioning,
                    "schedule": schedule, "stage": stage,
                    "experiment_fingerprint": fingerprint, **future.result(),
                }
        if stage == 1:
            for pair in pairs:
                for conditioning in cfg["conditioning"]:
                    fixed_key = (pair["pair_id"], arm_name(conditioning, "fixed"), 1)
                    switching_key = (pair["pair_id"], arm_name(conditioning, "switching"), 1)
                    if switching_key not in handoffs:
                        fixed = handoffs[fixed_key]
                        handoffs[switching_key] = {
                            **fixed, "arm": switching_key[1], "schedule": "switching",
                            "cached": True, "copied_from_fixed_stage1": True,
                        }
        if not args.dry_run:
            write_jsonl(handoff_path, [r for _, r in sorted(handoffs.items())])
        current = [r for r in handoffs.values() if int(r["stage"]) == stage]
        trunc = sum(r.get("finish_reason") == "length" for r in current)
        print(f"[multilingual:handoff] stage {stage}: {len(jobs)} generated; "
              f"{trunc}/{len(current)} truncated")

    handoff_rows = [r for _, r in sorted(handoffs.items())]
    if not args.dry_run:
        language_client = language_audit(handoff_rows, cfg)
        # Stage-1 fixed/switching records contain identical text and requested
        # language. Reuse one compliance verdict exactly; concurrent duplicate
        # judge calls can otherwise differ despite deterministic decoding.
        lookup = {(r["pair_id"], r["arm"], int(r["stage"])): r for r in handoff_rows}
        for pair in pairs:
            for conditioning in cfg["conditioning"]:
                fixed = lookup[(pair["pair_id"], arm_name(conditioning, "fixed"), 1)]
                switching = lookup[(pair["pair_id"], arm_name(conditioning, "switching"), 1)]
                for field in ("language_judge_match", "language_judge_parse_ok", "language_judge_raw"):
                    switching[field] = fixed[field]
                switching["language_verdict_copied_from_fixed_stage1"] = True
        apply_language_detection(handoff_rows)
        write_jsonl(handoff_path, handoff_rows)
    else:
        language_client = None

    answer_path = run_root / "answers.jsonl"
    existing = {(r["arm"], r["query_type"], int(r["depth"]), r["pair_id"]): r
                for r in read_jsonl(answer_path) if r.get("experiment_fingerprint") == fingerprint}
    # Discard independently generated switching answers at depth 1. Their
    # inputs are identical to the fixed arm, so they must reuse the same answer
    # record for the switching contrast to be exactly zero before any switch.
    for key in list(existing):
        if key[0].endswith("_switching") and key[2] == 1:
            del existing[key]
    jobs = []
    for pair in pairs:
        for query_type in ("target", "heldout"):
            direct_key = ("direct", query_type, 0, pair["pair_id"])
            if direct_key not in existing:
                jobs.append((direct_key, pair, render_context(pair), "", ""))
            for conditioning in cfg["conditioning"]:
                for schedule in cfg["schedules"]:
                    arm = arm_name(conditioning, schedule)
                    for depth in [int(d) for d in cfg["depths"] if int(d) > 0]:
                        key = (arm, query_type, depth, pair["pair_id"])
                        if schedule == "switching" and depth == 1:
                            continue
                        if key in existing:
                            continue
                        handoff = handoffs[(pair["pair_id"], arm, depth)]
                        jobs.append((key, pair, handoff["text"], handoff["requested_language"],
                                     handoff["language_code"]))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {
            pool.submit(answer, client, pair, key[1], material, cfg,
                        f"multilingual_answer_{key[0]}_{key[1]}_d{key[2]}"):
            (key, language, code) for key, pair, material, language, code in jobs
        }
        for future, (key, language, code) in futures.items():
            conditioning, schedule = ("direct", "direct") if key[0] == "direct" else key[0].rsplit("_", 1)
            existing[key] = {
                "arm": key[0], "conditioning": conditioning, "schedule": schedule,
                "query_type": key[1], "depth": key[2], "pair_id": key[3],
                "experiment_fingerprint": fingerprint,
                "requested_language": language, "language_code": code, **future.result(),
            }
    for pair in pairs:
        for query_type in ("target", "heldout"):
            for conditioning in cfg["conditioning"]:
                fixed_key = (arm_name(conditioning, "fixed"), query_type, 1, pair["pair_id"])
                switching_key = (arm_name(conditioning, "switching"), query_type, 1, pair["pair_id"])
                fixed = existing[fixed_key]
                existing[switching_key] = {
                    **fixed, "arm": switching_key[0], "schedule": "switching",
                    "cached": True, "answer_copied_from_fixed_stage1": True,
                }
    rows = [r for _, r in sorted(existing.items())]
    if not args.dry_run:
        write_jsonl(answer_path, rows)
        judge_client = add_judge(rows, cfg, tag="multilingual_answer_judge")
        write_jsonl(answer_path, rows)
        analyse(rows, handoff_rows, pairs, cfg, result_root)
        if language_client is not None:
            print("[multilingual:language-judge cost] " + json.dumps(language_client.ledger.summary()))
        if judge_client is not None:
            print("[multilingual:answer-judge cost] " + json.dumps(judge_client.ledger.summary()))
    print(f"[multilingual:answers] {len(rows)} rows; {len(jobs)} generated this pass")
    print("[multilingual:model cost] " + json.dumps(client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
