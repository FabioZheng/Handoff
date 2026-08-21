"""Repeated-compression experiment across datasets and evidence lengths.

Depth 0 answers from selected source documents. Depth N answers from a handoff
that has been compressed N consecutive times. Only the first compressor sees
source documents; every later compressor accepts a frozen SealedHandoff.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_data  # noqa: E402
import data as data_mod  # noqa: E402
import handoffs as hm  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import CostCapExceeded, LLMClient, load_config  # noqa: E402
from score import bootstrap_ci, em_f1_disagreement, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402


QUESTION_CONDITIONED_SYSTEM = (
    "You are a research handoff agent. Preserve every fact needed to answer the question. "
    "Your output will replace your entire input for the next agent, so omitted information is lost."
)
GENERIC_SYSTEM = (
    "You are a research handoff agent. Preserve the important factual content, including "
    "qualifiers, dates, numbers, relationships, uncertainties, and source ids. Your output "
    "will replace your entire input for the next agent, so omitted information is lost."
)
QUESTION_CONDITIONED_INITIAL_INSTRUCTION = (
    "Write concise prose research notes that preserve all evidence needed to answer the question. "
    "Do not answer the question directly and do not add unsupported facts."
)
GENERIC_INITIAL_INSTRUCTION = (
    "Write concise, general-purpose prose research notes. Preserve important factual content "
    "without adding unsupported facts."
)
QUESTION_CONDITIONED_RECOMPRESS_INSTRUCTION = (
    "Rewrite the previous agent's notes into concise prose research notes for another agent. "
    "Preserve every answer-relevant fact, qualifier, date, number, relationship, uncertainty, and source id. "
    "Use only the previous notes. Do not answer the question directly."
)
GENERIC_RECOMPRESS_INSTRUCTION = (
    "Rewrite the previous agent's notes into concise, general-purpose prose research notes for "
    "another agent. Preserve important factual content, qualifiers, dates, numbers, relationships, "
    "uncertainty, and source ids. Use only the previous notes."
)

# Backward-compatible names for the already-run paired-question experiment.
# That experiment intentionally keeps a shared system/instruction prompt and
# varies only whether the Question A block is present in the user message.
CHAIN_SYSTEM = QUESTION_CONDITIONED_SYSTEM
INITIAL_INSTRUCTION = QUESTION_CONDITIONED_INITIAL_INSTRUCTION
RECOMPRESS_INSTRUCTION = QUESTION_CONDITIONED_RECOMPRESS_INSTRUCTION


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def initial_compress(client, question, context_text: str, dataset: str, variant: str, cfg: dict, seed: int,
                     question_conditioned: bool):
    # The matched question-omission condition changes only whether the actual
    # question block is visible. System and handoff instructions stay byte-for-
    # byte identical to the conditioned chain.
    system = QUESTION_CONDITIONED_SYSTEM
    instruction = QUESTION_CONDITIONED_INITIAL_INSTRUCTION
    user = (
        f"Dataset: {dataset}\nEvidence length: {variant}\n\n"
        f"Source material:\n{context_text}\n\n"
        + (f"Question the final agent must answer: {question.question}\n\n" if question_conditioned else "")
        + instruction
    )
    result = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        seed=seed * 1000 + 1,
        tag="chain_compress_stage1",
    )
    return hm.Handoff(
        qid=question.qid,
        mechanism=f"chain_{dataset}_{variant}_d1",
        seed=seed,
        text=result.text.strip(),
        meta={
            "stage": 1, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens, "cached": result.cached,
        },
    )


def recompress(client, sealed: hm.SealedHandoff, cfg: dict, seed: int, stage: int,
               question_conditioned: bool) -> hm.Handoff:
    if not isinstance(sealed, hm.SealedHandoff):
        raise TypeError("recompress requires a SealedHandoff; source documents are forbidden")
    system = QUESTION_CONDITIONED_SYSTEM
    instruction = QUESTION_CONDITIONED_RECOMPRESS_INSTRUCTION
    user = (
        f"Previous agent's notes:\n{sealed.handoff_text}\n\n"
        + (f"Question the final agent must answer: {sealed.question}\n\n" if question_conditioned else "")
        + instruction
    )
    result = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        seed=seed * 1000 + stage,
        tag=f"chain_compress_stage{stage}",
    )
    return hm.Handoff(
        qid=sealed.qid,
        mechanism=sealed.mechanism.rsplit("_d", 1)[0] + f"_d{stage}",
        seed=seed,
        text=result.text.strip(),
        meta={
            "stage": stage, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens, "cached": result.cached,
        },
    )


def answer_context(client, question, context_text: str, dataset: str, variant: str, cfg: dict) -> dict:
    user = (
        f"Dataset: {dataset}\nEvidence length: {variant}\n\nSource material:\n{context_text}\n\n"
        f"Question: {question.question}\nAnswer:"
    )
    result = client.chat(
        [{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["orchestrator_temperature"],
        max_tokens=cfg["decoding"]["answer_max_tokens"],
        seed=None,
        tag="chain_answer_depth0",
    )
    return {
        "raw": result.text, "pred": extract_short_answer(result.text),
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        "cached": result.cached,
    }


def isolation_selftest() -> None:
    for bad in ("notes", {"handoff_text": "notes"}, object()):
        try:
            recompress(None, bad, {}, 1, 2, False)
        except TypeError:
            continue
        raise AssertionError(f"chain isolation broken: accepted {type(bad).__name__}")
    print("[chain:selftest] later compressors accept SealedHandoff only")


def prepare_questions(client, base_cfg: dict, chain_cfg: dict, args, data_root: Path) -> dict[str, list]:
    result: dict[str, list] = {}
    for dataset in args.datasets:
        manifest = {
            "dataset": dataset,
            "model": base_cfg["model"]["id"],
            "seed": chain_cfg["sampling"]["seed"],
            "candidates": args.candidates,
            "target": args.n,
            "c1_samples": base_cfg["leakage_filter"]["samples"],
            "c1_temperature": base_cfg["leakage_filter"]["temperature"],
            "c1_f1_threshold": base_cfg["leakage_filter"]["f1_known_threshold"],
        }
        path = data_root / dataset / "filtered.jsonl"
        questions = None if args.force else chain_data.read_questions(path, manifest)
        if questions is None:
            candidates = chain_data.sample_candidates(
                dataset, chain_cfg["datasets"][dataset], ROOT,
                chain_cfg["sampling"]["seed"], args.candidates,
            )
            filter_cfg = chain_data.filter_config(base_cfg, args.n)
            questions, report = data_mod.apply_c1(
                client, candidates, filter_cfg, base_cfg["runtime"]["concurrency"],
                dry_run=args.dry_run,
            )
            if len(questions) < args.n:
                print(f"[chain:stage0] WARNING {dataset}: only {len(questions)}/{args.n} survived C1")
            if not args.dry_run:
                chain_data.write_questions(questions, path, manifest)
                report_path = data_root / dataset / "c1_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(f"[chain:stage0] {dataset}: reusing {len(questions)} filtered questions")
        result[dataset] = questions[: args.n]
    return result


def generate_chains(client, base_cfg: dict, chain_cfg: dict, args, questions_by_dataset, run_root: Path):
    handoffs: dict[tuple, hm.Handoff] = {}
    max_depth = max(args.depths)
    if max_depth == 0:
        return handoffs
    for dataset, questions in questions_by_dataset.items():
        for variant in args.contexts:
            for seed in args.seeds:
                path = run_root / "handoffs" / f"{dataset}_{variant}_seed{seed}.jsonl"
                existing_rows = [] if args.force else read_jsonl(path)
                existing = {(r["qid"], int(r["stage"])): r for r in existing_rows}
                for stage in range(1, max_depth + 1):
                    todo = [q for q in questions if (q.qid, stage) not in existing]
                    if todo:
                        def build(q):
                            if stage == 1:
                                context = chain_data.render_context(q, variant, chain_cfg["contexts"])
                                return initial_compress(
                                    client, q, context, dataset, variant, base_cfg, seed,
                                    bool(chain_cfg.get("question_conditioned", True)),
                                )
                            previous = hm.Handoff.from_json(existing[(q.qid, stage - 1)])
                            sealed = hm.seal(q.question, previous)
                            return recompress(
                                client, sealed, base_cfg, seed, stage,
                                bool(chain_cfg.get("question_conditioned", True)),
                            )

                        with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
                            futures = {pool.submit(build, q): q for q in todo}
                            for future, q in futures.items():
                                handoff = future.result()
                                row = handoff.to_json()
                                row.update({"dataset": dataset, "context_variant": variant, "stage": stage})
                                existing[(q.qid, stage)] = row
                    ordered = [existing[(q.qid, s)] for q in questions for s in range(1, stage + 1)
                               if (q.qid, s) in existing]
                    if not args.dry_run:
                        write_jsonl(path, ordered)
                    print(f"[chain:handoff] {dataset}/{variant}/seed{seed}/stage{stage}: "
                          f"{len(questions) - len(todo)} reused, {len(todo)} generated")
                for key, row in existing.items():
                    handoffs[(dataset, variant, seed, key[0], key[1])] = hm.Handoff.from_json(row)
    return handoffs


def run_answers(client, base_cfg: dict, chain_cfg: dict, args, questions_by_dataset, handoffs, run_root: Path):
    path = run_root / "answers.jsonl"
    existing_rows = [] if args.force else read_jsonl(path)
    existing = {
        (r["dataset"], r["context_variant"], int(r["depth"]), r.get("seed"), r["qid"]): r
        for r in existing_rows
    }
    jobs = []
    wanted_keys = set()
    qmap = {}
    for dataset, questions in questions_by_dataset.items():
        for q in questions:
            qmap[(dataset, q.qid)] = q
            for variant in args.contexts:
                stats = chain_data.context_stats(q, variant, chain_cfg["contexts"])
                for depth in args.depths:
                    seeds = [None] if depth == 0 else args.seeds
                    for seed in seeds:
                        key = (dataset, variant, depth, seed, q.qid)
                        wanted_keys.add(key)
                        if key in existing:
                            continue
                        if depth == 0:
                            context = chain_data.render_context(q, variant, chain_cfg["contexts"])
                            call = lambda q=q, c=context, d=dataset, v=variant: answer_context(
                                client, q, c, d, v, base_cfg
                            )
                        else:
                            h = handoffs[(dataset, variant, seed, q.qid, depth)]
                            sealed = hm.seal(q.question, h)
                            call = lambda s=sealed, d=depth: hm.orchestrator_answer(
                                client, s, base_cfg, f"chain_answer_depth{d}"
                            )
                        jobs.append((key, stats, call))

    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(call): (key, stats) for key, stats, call in jobs}
        for future, (key, stats) in futures.items():
            dataset, variant, depth, seed, qid = key
            q = qmap[(dataset, qid)]
            answer = future.result()
            em, f1 = score_against_golds(answer["pred"], q.golds)
            chain_prompt = chain_completion = 0
            final_chars = stats["context_characters"]
            if depth > 0:
                for stage in range(1, depth + 1):
                    h = handoffs[(dataset, variant, seed, qid, stage)]
                    chain_prompt += int(h.meta.get("prompt_tokens", 0) or 0)
                    chain_completion += int(h.meta.get("completion_tokens", 0) or 0)
                final_chars = len(handoffs[(dataset, variant, seed, qid, depth)].text)
            existing[key] = {
                "dataset": dataset, "context_variant": variant, "depth": depth,
                "seed": seed, "qid": qid, "n_hops": q.n_hops,
                "pred": answer["pred"], "raw": answer["raw"], "gold": q.answer,
                "golds": list(q.golds),
                "em": em, "f1": f1, **stats,
                "final_handoff_characters": final_chars,
                "chain_prompt_tokens": chain_prompt,
                "chain_completion_tokens": chain_completion,
                "answer_prompt_tokens": answer["prompt_tokens"],
                "answer_completion_tokens": answer["completion_tokens"],
            }
    all_rows = sorted(existing.values(), key=lambda r: (
        r["dataset"], r["context_variant"], int(r["depth"]), r.get("seed") or -1, r["qid"]
    ))
    if not args.dry_run:
        write_jsonl(path, all_rows)
    rows = [row for row in all_rows if (
        row["dataset"], row["context_variant"], int(row["depth"]), row.get("seed"), row["qid"]
    ) in wanted_keys]
    print(f"[chain:answers] {len(rows)} selected answers ({len(jobs)} generated)")
    return rows


def run_oracle_baseline(client, base_cfg: dict, questions_by_dataset: dict, args, run_root: Path) -> list[dict]:
    """E_oracle equivalent: gold-sentence handoff, no compression, on the same
    C1-filtered questions the chain itself uses -- so it is directly comparable
    to depth 0..N on this run, not a separately-sampled pilot."""
    path = run_root / "oracle_answers.jsonl"
    existing_rows = [] if args.force else read_jsonl(path)
    existing = {(r["dataset"], r["qid"]): r for r in existing_rows}
    jobs = []
    for dataset, questions in questions_by_dataset.items():
        for q in questions:
            key = (dataset, q.qid)
            if key in existing:
                continue

            def call(q=q):
                sealed = hm.seal(q.question, hm.make_oracle(q))
                return hm.orchestrator_answer(client, sealed, base_cfg, "chain_oracle_answer")

            jobs.append((key, q, call))

    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(call): (key, q) for key, q, call in jobs}
        for future, (key, q) in futures.items():
            dataset, qid = key
            answer = future.result()
            em, f1 = score_against_golds(answer["pred"], q.golds)
            existing[key] = {
                "dataset": dataset, "qid": qid, "question": q.question,
                "pred": answer["pred"], "raw": answer["raw"],
                "gold": q.answer, "golds": list(q.golds), "em": em, "f1": f1,
            }
    rows = sorted(existing.values(), key=lambda r: (r["dataset"], r["qid"]))
    if not args.dry_run:
        write_jsonl(path, rows)
    print(f"[chain:oracle] {len(jobs)} generated, {len(rows)} total")
    return rows


def compute_oracle_metrics(rows: list[dict], boot_n: int, ci: float) -> dict[str, dict]:
    result = {}
    for dataset in sorted(set(r["dataset"] for r in rows)):
        subset = [r for r in rows if r["dataset"] == dataset]
        em = np.array([r["em"] for r in subset])
        f1 = np.array([r["f1"] for r in subset])
        em_m, em_lo, em_hi = bootstrap_ci(em, boot_n, ci, seed=17)
        f1_m, f1_lo, f1_hi = bootstrap_ci(f1, boot_n, ci, seed=17)
        entry = {
            "dataset": dataset, "n": len(subset),
            "em": round(em_m, 4), "em_lo": round(em_lo, 4), "em_hi": round(em_hi, 4),
            "f1": round(f1_m, 4), "f1_lo": round(f1_lo, 4), "f1_hi": round(f1_hi, 4),
        }
        if all("judge_correct" in r for r in subset):
            bert = np.array([r["judge_correct"] for r in subset])
            b_m, b_lo, b_hi = bootstrap_ci(bert, boot_n, ci, seed=17)
            entry.update({
                "judge_correct": round(b_m, 4), "judge_correct_lo": round(b_lo, 4),
                "judge_correct_hi": round(b_hi, 4),
            })
        result[dataset] = entry
    return result


def add_bertscore(rows: list[dict], chain_cfg: dict) -> None:
    """Add best-over-alias BERTScore precision, recall, and F1 in one batch."""
    cfg = chain_cfg.get("analysis", {}).get("bertscore", {})
    if not cfg.get("enabled", False):
        return
    missing = [row for row in rows if "bertscore_f1" not in row]
    if not missing:
        print(f"[chain:bertscore] reusing {len(rows)} semantic scores")
        return
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore is enabled but bert-score is not installed; run pip install -r requirements.txt"
        ) from exc

    candidates = [str(row.get("pred", "")) for row in missing]
    references = []
    for row in missing:
        golds = row.get("golds") or [row.get("gold", "")]
        references.append([str(g) for g in golds if str(g).strip()] or [""])
    precision, recall, f1 = bert_score(
        candidates,
        references,
        model_type=cfg.get("model_type", "roberta-large"),
        num_layers=cfg.get("num_layers", 17),
        batch_size=int(cfg.get("batch_size", 64)),
        lang="en",
        rescale_with_baseline=bool(cfg.get("rescale_with_baseline", True)),
        verbose=True,
    )
    for row, p, r, f in zip(missing, precision.tolist(), recall.tolist(), f1.tolist()):
        row["bertscore_precision"] = round(float(p), 6)
        row["bertscore_recall"] = round(float(r), 6)
        row["bertscore_f1"] = round(float(f), 6)
    print(f"[chain:bertscore] computed {len(missing)} semantic scores")


def backfill_golds(rows: list[dict], data_root: Path) -> None:
    """Restore answer aliases and question text from the filtered-question manifests.

    The question text is what the LLM judge needs; answer rows written before
    the judge existed do not carry it.
    """
    by_key = {}
    questions_by_key = {}
    for dataset_dir in data_root.iterdir() if data_root.exists() else []:
        path = dataset_dir / "filtered.jsonl"
        for question in read_jsonl(path):
            if "qid" in question:
                by_key[(dataset_dir.name, question["qid"])] = (
                    question.get("golds") or [question.get("answer", "")]
                )
                questions_by_key[(dataset_dir.name, question["qid"])] = question.get("question", "")
    for row in rows:
        key = (row["dataset"], row["qid"])
        row["golds"] = list(by_key.get(key, row.get("golds") or [row.get("gold", "")]))
        if not row.get("question"):
            row["question"] = questions_by_key.get(key, "")


def analyse(rows: list[dict], chain_cfg: dict, args, output_root: Path, ledger: dict,
            oracle_metrics: dict[str, dict] | None = None) -> None:
    grouped: dict[tuple, dict[str, list[dict]]] = {}
    for row in rows:
        if row["dataset"] not in args.datasets or row["context_variant"] not in args.contexts:
            continue
        if int(row["depth"]) not in args.depths:
            continue
        grouped.setdefault((row["dataset"], row["context_variant"], int(row["depth"])), {}) \
            .setdefault(row["qid"], []).append(row)

    boot_n = chain_cfg["analysis"]["bootstrap_resamples"]
    ci = chain_cfg["analysis"]["ci_level"]
    metrics = []
    vectors = {}
    for key, by_qid in sorted(grouped.items()):
        dataset, variant, depth = key
        qids = sorted(by_qid)
        em = np.array([np.mean([x["em"] for x in by_qid[qid]]) for qid in qids])
        f1 = np.array([np.mean([x["f1"] for x in by_qid[qid]]) for qid in qids])
        vectors[key] = {"qids": qids, "em": dict(zip(qids, em)), "f1": dict(zip(qids, f1))}
        bert_f1 = None
        if all("judge_correct" in x for values in by_qid.values() for x in values):
            bert_f1 = np.array([
                np.mean([x["judge_correct"] for x in by_qid[qid]]) for qid in qids
            ])
            vectors[key]["judge_correct"] = dict(zip(qids, bert_f1))
        em_m, em_lo, em_hi = bootstrap_ci(em, boot_n, ci, seed=17)
        f1_m, f1_lo, f1_hi = bootstrap_ci(f1, boot_n, ci, seed=17)
        mean = lambda field: float(np.mean([
            np.mean([x[field] for x in by_qid[qid]]) for qid in qids
        ]))
        metric_row = {
            "dataset": dataset, "context_variant": variant, "depth": depth, "n": len(qids),
            "em": round(em_m, 4), "em_lo": round(em_lo, 4), "em_hi": round(em_hi, 4),
            "f1": round(f1_m, 4), "f1_lo": round(f1_lo, 4), "f1_hi": round(f1_hi, 4),
            "em_f1_disagree": round(em_f1_disagreement(em, f1), 4),
            "documents_mean": round(mean("context_documents"), 2),
            "context_characters_mean": round(mean("context_characters"), 1),
            "handoff_characters_mean": round(mean("final_handoff_characters"), 1),
            "chain_tokens_mean": round(mean("chain_prompt_tokens") + mean("chain_completion_tokens"), 1),
            "answer_tokens_mean": round(mean("answer_prompt_tokens") + mean("answer_completion_tokens"), 1),
        }
        if bert_f1 is not None:
            b_m, b_lo, b_hi = bootstrap_ci(bert_f1, boot_n, ci, seed=17)
            metric_row.update({
                "judge_correct": round(b_m, 4),
                "judge_correct_lo": round(b_lo, 4),
                "judge_correct_hi": round(b_hi, 4),
            })
        metrics.append(metric_row)

    deltas = []
    for row in metrics:
        if row["depth"] == 0:
            row["f1_delta_from_depth0"] = 0.0
            row["em_delta_from_depth0"] = 0.0
            if "judge_correct" in row:
                row["judge_correct_delta_from_depth0"] = 0.0
            continue
        key = (row["dataset"], row["context_variant"], row["depth"])
        base_key = (row["dataset"], row["context_variant"], 0)
        if base_key not in vectors:
            continue
        shared = sorted(set(vectors[key]["qids"]) & set(vectors[base_key]["qids"]))
        comparison_metrics = ["em", "f1"]
        if "judge_correct" in vectors[key] and "judge_correct" in vectors[base_key]:
            comparison_metrics.append("judge_correct")
        for metric in comparison_metrics:
            treatment = np.array([vectors[key][metric][qid] for qid in shared])
            control = np.array([vectors[base_key][metric][qid] for qid in shared])
            delta = paired_bootstrap_delta(treatment, control, boot_n, ci, seed=23)
            deltas.append({
                "dataset": row["dataset"], "context_variant": row["context_variant"],
                "depth": row["depth"], "metric": metric, **delta,
            })
            row[f"{metric}_delta_from_depth0"] = round(delta["delta"], 4)

    oracle_metrics = oracle_metrics or {}
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "stage_metrics.csv", metrics)
    write_csv(output_root / "depth0_deltas.csv", deltas)
    if oracle_metrics:
        write_csv(output_root / "oracle_metrics.csv", sorted(oracle_metrics.values(), key=lambda r: r["dataset"]))
    make_plot(metrics, args, output_root / "degradation.png", oracle_metrics)
    write_report(metrics, deltas, args, ledger, chain_cfg, output_root / "report.md", oracle_metrics)
    print(f"[chain:analysis] wrote {output_root / 'report.md'} and degradation.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(metrics: list[dict], args, path: Path, oracle_metrics: dict[str, dict] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in args.datasets if any(r["dataset"] == d for r in metrics)]
    plot_metrics = [("f1", "Token F1")]
    if any("judge_correct" in row for row in metrics):
        plot_metrics.append(("judge_correct", "LLM-judge accuracy"))
    fig, axes = plt.subplots(
        len(plot_metrics), len(datasets),
        figsize=(6.4 * len(datasets), 4.5 * len(plot_metrics)),
        sharex=True, squeeze=False,
    )
    colors = {"short": "#1b9e77", "medium": "#d95f02", "full": "#7570b3"}
    # Marker area proportional to the mean size (characters) of whatever text
    # actually fed the answerer at that point -- the full context at depth 0,
    # the final handoff's text at depth >=1 -- so shrinkage across repeated
    # compression is visible directly on the accuracy curve, not just in a
    # separate table. One scale factor for the whole figure (not per subplot)
    # so marker area is comparable across datasets/variants/metric rows.
    all_chars = [r["handoff_characters_mean"] for r in metrics if "handoff_characters_mean" in r]
    max_chars = max(all_chars) if all_chars else 1.0
    max_area, min_area = 900.0, 15.0
    size_scale = max_area / max_chars

    def marker_area(record):
        chars = record.get("handoff_characters_mean")
        return max_area if chars is None else max(min_area, size_scale * chars)

    for row_idx, (metric, label) in enumerate(plot_metrics):
        for col_idx, dataset in enumerate(datasets):
            axis = axes[row_idx][col_idx]
            for variant in args.contexts:
                subset = sorted(
                    [r for r in metrics if r["dataset"] == dataset
                     and r["context_variant"] == variant and metric in r],
                    key=lambda r: r["depth"],
                )
                if not subset:
                    continue
                x = [r["depth"] for r in subset]
                y = [r[metric] for r in subset]
                lower = [r[metric] - r[f"{metric}_lo"] for r in subset]
                upper = [r[f"{metric}_hi"] - r[metric] for r in subset]
                axis.errorbar(x, y, yerr=[lower, upper], marker="", linewidth=2,
                              capsize=3, label=variant, color=colors.get(variant), zorder=2)
                axis.scatter(x, [r[metric] for r in subset], s=[marker_area(r) for r in subset],
                             color=colors.get(variant), edgecolors="white", linewidths=0.6, zorder=3)
            # A_full equivalent: this run's own depth-0/full-context point (no
            # handoff), drawn flat so it is comparable to every depth, not just x=0.
            full_depth0 = next(
                (r for r in metrics if r["dataset"] == dataset and r["context_variant"] == "full"
                 and r["depth"] == 0 and metric in r), None,
            )
            def n_suffix(record):
                return f" (n={record['n']})" if "n" in record else ""

            if full_depth0 is not None:
                axis.axhline(full_depth0[metric], color="#999999", linestyle="--", linewidth=1.3,
                              label=f"A_full equivalent{n_suffix(full_depth0)}")
            oracle = (oracle_metrics or {}).get(dataset)
            if oracle is not None and metric in oracle:
                axis.axhline(oracle[metric], color="#e7298a", linestyle="--", linewidth=1.3,
                              label=f"E_oracle equivalent{n_suffix(oracle)}")
            if full_depth0 is not None or oracle is not None:
                axis.legend(fontsize=8, loc="best")
            if row_idx == 0:
                axis.set_title(dataset)
            if row_idx == len(plot_metrics) - 1:
                axis.set_xlabel("Number of compression handoffs")
            axis.set_xticks(args.depths)
            if col_idx == 0:
                axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0][-1].legend(title="Evidence length")
    fig.suptitle("Answer accuracy across repeated handoffs")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.5, 0.005,
              f"Marker area ∝ mean characters in the answerer's input at that point "
              f"(full context at depth 0, else the handoff text) — smallest marker "
              f"{min_area:.0f}pt² floor, largest ≈{max_chars:,.0f} characters.",
              ha="center", fontsize=8, color="#555555")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics, deltas, args, ledger, chain_cfg: dict, path: Path,
                  oracle_metrics: dict[str, dict] | None = None) -> None:
    judge_cfg = chain_cfg.get("judge", {})
    lines = [
        "# Repeated-handoff degradation results", "",
        f"Datasets: {', '.join(args.datasets)}  ",
        f"Evidence variants: {', '.join(args.contexts)}  ",
        f"Compression depths: {', '.join(map(str, args.depths))}", "",
        "Summaries are question-conditioned." if chain_cfg.get("question_conditioned", True)
        else "Summaries are general-purpose: compressors never receive the final question; only the fresh answerer does.", "",
        "All evidence variants retain every gold document. Short uses gold documents only; "
        "medium adds distractors up to five documents; full uses the complete dataset context.", "",
        f"LLM-judge accuracy is the fraction of answers a judge model "
        f"({judge_cfg.get('model_id', 'disabled')}) rules equivalent to a gold answer, at "
        f"temperature 0. EM and token F1 remain the deterministic primary metrics.", "",
        "| dataset | evidence | depth | n | EM | Token F1 | Judge acc | Token F1 change | Judge change | docs | context chars | handoff chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics:
        lines.append(
            f"| {r['dataset']} | {r['context_variant']} | {r['depth']} | {r['n']} | "
            f"{r['em']:.3f} | {r['f1']:.3f} | "
            f"{r.get('judge_correct', float('nan')):.3f} | "
            f"{r.get('f1_delta_from_depth0', 0):+.3f} | "
            f"{r.get('judge_correct_delta_from_depth0', 0):+.3f} | {r['documents_mean']:.1f} | "
            f"{r['context_characters_mean']:.0f} | {r['handoff_characters_mean']:.0f} |"
        )
    if oracle_metrics:
        full_depth0 = {r["dataset"]: r for r in metrics if r["context_variant"] == "full" and r["depth"] == 0}
        lines += ["", "## Baselines (same questions as this run, no compression)", "",
                   "| dataset | baseline | n | EM | Token F1 | Judge acc |",
                   "|---|---|---:|---:|---:|---:|"]
        nan = float("nan")
        for dataset in sorted(oracle_metrics):
            if dataset in full_depth0:
                r = full_depth0[dataset]
                lines.append(f"| {dataset} | A_full equivalent | {r['n']} | {r['em']:.3f} | "
                             f"{r['f1']:.3f} | {r.get('judge_correct', nan):.3f} |")
            o = oracle_metrics[dataset]
            lines.append(f"| {dataset} | E_oracle equivalent | {o['n']} | {o['em']:.3f} | "
                         f"{o['f1']:.3f} | {o.get('judge_correct', nan):.3f} |")
        lines.append("")
    lines += ["", "## Cost", "",
              f"- live spend: ${ledger['cost_usd']:.4f}",
              f"- calls: {ledger['calls_live']} live, {ledger['calls_cached']} cached",
              f"- prompt/completion tokens: {ledger['prompt_tokens']} / {ledger['completion_tokens']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv_arg(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeated handoff degradation experiment")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--chain-config", default=str(ROOT / "chain_config.yaml"))
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--contexts", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    chain_cfg = load_config(args.chain_config)
    args.datasets = parse_csv_arg(args.datasets) if args.datasets else list(chain_cfg["datasets"])
    args.contexts = parse_csv_arg(args.contexts) if args.contexts else list(chain_cfg["contexts"])
    args.depths = parse_csv_arg(args.depths, int) if args.depths else list(chain_cfg["depths"])
    args.seeds = parse_csv_arg(args.seeds, int) if args.seeds else list(chain_cfg["seeds"])
    args.n = args.n or int(chain_cfg["sampling"]["n_target_per_dataset"])
    args.candidates = args.candidates or int(chain_cfg["sampling"]["n_candidates_per_dataset"])
    unknown_datasets = set(args.datasets) - set(chain_cfg["datasets"])
    unknown_contexts = set(args.contexts) - set(chain_cfg["contexts"])
    if unknown_datasets or unknown_contexts or min(args.depths) < 0:
        parser.error(f"invalid datasets={unknown_datasets}, contexts={unknown_contexts}, or depths")

    # The judge lives in the base config (it needs model/runtime/cost); expose it
    # on chain_cfg so the report can name the judge model it used.
    chain_cfg["judge"] = base_cfg.get("judge", {})

    output_root = ROOT / chain_cfg["outputs"]["root"]
    run_root = ROOT / chain_cfg["outputs"]["run_root"]
    data_root = ROOT / chain_cfg["outputs"]["data_root"]
    if args.plot_only:
        rows = read_jsonl(run_root / "answers.jsonl")
        backfill_golds(rows, data_root)
        judge_client = add_judge(rows, base_cfg, tag="chain_judge")
        write_jsonl(run_root / "answers.jsonl", rows)
        oracle_metrics = {}
        oracle_rows = read_jsonl(run_root / "oracle_answers.jsonl")
        if oracle_rows:
            backfill_golds(oracle_rows, data_root)
            add_judge(oracle_rows, base_cfg, tag="chain_judge")
            write_jsonl(run_root / "oracle_answers.jsonl", oracle_rows)
            oracle_metrics = compute_oracle_metrics(
                oracle_rows, chain_cfg["analysis"]["bootstrap_resamples"], chain_cfg["analysis"]["ci_level"]
            )
        analyse(rows, chain_cfg, args, output_root, {
            "cost_usd": 0.0, "calls_live": 0, "calls_cached": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
        }, oracle_metrics)
        if judge_client is not None:
            print("[chain:judge cost] " + json.dumps(judge_client.ledger.summary()))
        return 0

    isolation_selftest()
    client = LLMClient(base_cfg, dry_run=args.dry_run)
    print(f"[chain] datasets={args.datasets} contexts={args.contexts} depths={args.depths} "
          f"n={args.n} candidates={args.candidates} "
          f"question_conditioned={chain_cfg.get('question_conditioned', True)}")
    try:
        questions = prepare_questions(client, base_cfg, chain_cfg, args, data_root)
        handoffs = generate_chains(client, base_cfg, chain_cfg, args, questions, run_root)
        answers = run_answers(client, base_cfg, chain_cfg, args, questions, handoffs, run_root)
        oracle_rows = run_oracle_baseline(client, base_cfg, questions, args, run_root)
        oracle_metrics = {}
        if not args.dry_run:
            backfill_golds(answers, data_root)
            judge_client = add_judge(answers, base_cfg, tag="chain_judge")
            add_judge(oracle_rows, base_cfg, tag="chain_judge")
            write_jsonl(run_root / "oracle_answers.jsonl", oracle_rows)
            oracle_metrics = compute_oracle_metrics(
                oracle_rows, chain_cfg["analysis"]["bootstrap_resamples"], chain_cfg["analysis"]["ci_level"]
            )
            write_jsonl(run_root / "answers.jsonl", answers)
            analyse(answers, chain_cfg, args, output_root, client.ledger.summary(), oracle_metrics)
            if judge_client is not None:
                print("[chain:judge cost] " + json.dumps(judge_client.ledger.summary()))
        else:
            print(json.dumps(client.dry_run_report(), indent=2))
    except CostCapExceeded as exc:
        print(f"[chain:ABORT] {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
