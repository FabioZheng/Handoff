"""Legacy intermediate-fact-retention experiment.

The targeted arm gives the task question to every compressor.  The generic arm
never gives a compressor the question: it receives documents at stage 1 and
only the previous notes thereafter.  Both arms give a frozen depth-5 summary
and the requested question to the final answerer. MuSiQue decomposition steps
are related intermediate questions, not unrelated held-out questions; therefore
this script must not be used to claim cross-question summary generalizability.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_data  # noqa: E402
import handoffs as hm  # noqa: E402
from data import Question  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402
from run_chain import add_bertscore  # noqa: E402


COMPRESS_SYSTEM = (
    "You are a research handoff agent. Your output replaces your entire input for the next "
    "agent, so omitted information is lost. Preserve factual claims, qualifiers, dates, "
    "numbers, relationships, uncertainty, and source identifiers."
)
TARGETED_INITIAL = "Write concise research notes preserving everything needed to answer the question. Do not answer it directly."
GENERIC_INITIAL = "Write concise reusable research notes preserving important verifiable facts. Do not answer any question directly."
TARGETED_REWRITE = "Rewrite these notes for another agent while preserving everything needed to answer the question. Do not answer it directly."
GENERIC_REWRITE = "Rewrite these notes into concise reusable research notes. Preserve important verifiable facts, qualifiers, dates, numbers, relationships, uncertainty, and source identifiers. Do not answer any question directly."


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def load_questions(path: Path) -> list[Question]:
    return [Question.from_json(row) for row in read_jsonl(path) if "qid" in row]


def queries_for(question: Question) -> list[dict]:
    queries = [{"query_id": "target", "query_type": "target", "text": question.question,
                "golds": question.golds}]
    previous = []
    for idx, step in enumerate(question.decomposition):
        text = str(step["question"])
        text = re.sub(r"#(\d+)", lambda m: previous[int(m.group(1)) - 1]
                      if 0 < int(m.group(1)) <= len(previous) else m.group(0), text)
        queries.append({
            "query_id": f"step{idx}", "query_type": "alternate",
            "text": "Answer this evidence-grounded reasoning step. In `A >> relation`, give A's relation: " + text,
            "golds": [str(step["answer"])],
        })
        previous.append(str(step["answer"]))
    return queries


def initial_compress(client, question: Question, context: str, mode: str, cfg: dict, seed: int) -> dict:
    if mode == "targeted":
        user = f"Source material:\n{context}\n\nQuestion:\n{question.question}\n\n{TARGETED_INITIAL}"
    else:
        user = f"Source material:\n{context}\n\n{GENERIC_INITIAL}"
        assert question.question not in user, "generic initial prompt leaked the question"
    res = client.chat([{"role": "system", "content": COMPRESS_SYSTEM}, {"role": "user", "content": user}],
                      temperature=cfg["decoding"]["subagent_temperature"],
                      max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=seed * 1000 + 1,
                      tag=f"question_context_{mode}_stage1")
    return {"text": res.text.strip(), "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "cached": res.cached}


def recompress(client, notes: str, question_text: str, mode: str, cfg: dict, seed: int, stage: int) -> dict:
    if mode == "targeted":
        user = f"Previous notes:\n{notes}\n\nQuestion:\n{question_text}\n\n{TARGETED_REWRITE}"
    else:
        user = f"Previous notes:\n{notes}\n\n{GENERIC_REWRITE}"
        assert question_text not in user, "generic relay prompt leaked the question"
    res = client.chat([{"role": "system", "content": COMPRESS_SYSTEM}, {"role": "user", "content": user}],
                      temperature=cfg["decoding"]["subagent_temperature"],
                      max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=seed * 1000 + stage,
                      tag=f"question_context_{mode}_stage{stage}")
    return {"text": res.text.strip(), "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "cached": res.cached}


def answer(client, query: dict, material: str, cfg: dict, tag: str) -> dict:
    user = f"Research material:\n{material}\n\nQuestion:\n{query['text']}\nAnswer:"
    res = client.chat([{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
                      temperature=cfg["decoding"]["orchestrator_temperature"],
                      max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    pred = extract_short_answer(res.text)
    em, f1 = score_against_golds(pred, query["golds"])
    return {"pred": pred, "raw": res.text, "em": em, "f1": f1,
            "answer_prompt_tokens": res.prompt_tokens, "answer_completion_tokens": res.completion_tokens,
            "cached": res.cached}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyse(rows: list[dict], cfg: dict, output_root: Path) -> None:
    boot_n, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["query_type"], int(row["depth"])), {}).setdefault(
            (row["qid"], row["query_id"]), []).append(row)
    metrics, vectors = [], {}
    for key, by_query in sorted(grouped.items()):
        mode, query_type, depth = key
        ids = sorted(by_query)
        values = {}
        for metric in ("em", "f1", "bertscore_f1"):
            if all(metric in row for items in by_query.values() for row in items):
                values[metric] = np.array([np.mean([r[metric] for r in by_query[i]]) for i in ids])
        vectors[key] = {metric: dict(zip(ids, value)) for metric, value in values.items()}
        record = {"mode": mode, "query_type": query_type, "depth": depth, "n": len(ids)}
        for metric, value in values.items():
            mean, lo, hi = bootstrap_ci(value, boot_n, ci, seed=31)
            record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
        metrics.append(record)

    deltas = []
    for row in metrics:
        if row["depth"] == 0:
            continue
        key = (row["mode"], row["query_type"], row["depth"])
        base = ("direct", row["query_type"], 0)
        if base not in vectors:
            continue
        for metric in ("em", "f1", "bertscore_f1"):
            if metric not in vectors[key] or metric not in vectors[base]:
                continue
            shared = sorted(set(vectors[key][metric]) & set(vectors[base][metric]))
            result = paired_bootstrap_delta(
                np.array([vectors[key][metric][i] for i in shared]),
                np.array([vectors[base][metric][i] for i in shared]), boot_n, ci, seed=37,
            )
            row[f"{metric}_delta_from_direct"] = round(result["delta"], 4)
            deltas.append({"mode": row["mode"], "query_type": row["query_type"],
                           "depth": row["depth"], "metric": metric, **result})
    write_csv(output_root / "metrics.csv", metrics)
    write_csv(output_root / "deltas.csv", deltas)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"targeted": "#d95f02", "generic": "#1b9e77", "direct": "#7570b3"}
    for axis, query_type in zip(axes, ("target", "alternate")):
        for mode in ("direct", "targeted", "generic"):
            subset = sorted([r for r in metrics if r["query_type"] == query_type and r["mode"] == mode],
                            key=lambda r: r["depth"])
            if not subset:
                continue
            x = [r["depth"] for r in subset]
            y = [r["f1"] for r in subset]
            axis.errorbar(x, y, yerr=[[r["f1"] - r["f1_lo"] for r in subset],
                                      [r["f1_hi"] - r["f1"] for r in subset]],
                          marker="o", capsize=3, linewidth=2, label=mode, color=colors[mode])
        axis.set_title(f"{query_type} questions")
        axis.set_xlabel("Compression handoffs")
        axis.set_xticks([0, cfg["compression_depth"]])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Token F1")
    axes[1].legend(title="summary mode")
    fig.suptitle("Question-aware versus generic repeated handoffs")
    fig.tight_layout()
    fig.savefig(output_root / "question_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--experiment-config", default=str(ROOT / "question_context_config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base_cfg, cfg = load_config(args.config), load_config(args.experiment_config)
    questions = load_questions(ROOT / cfg["source_questions"])
    output_root, run_root = ROOT / cfg["outputs"]["root"], ROOT / cfg["outputs"]["run_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    client = LLMClient(base_cfg, dry_run=args.dry_run)
    print(f"[question-context] questions={len(questions)} modes={cfg['modes']} depth={cfg['compression_depth']}")

    handoff_path = run_root / "handoffs.jsonl"
    handoff_rows = read_jsonl(handoff_path)
    handoffs = {(r["qid"], r["mode"], int(r["seed"]), int(r["stage"])): r for r in handoff_rows}
    for mode in cfg["modes"]:
        for seed in cfg["seeds"]:
            for stage in range(1, cfg["compression_depth"] + 1):
                missing = [q for q in questions if (q.qid, mode, seed, stage) not in handoffs]
                if missing:
                    def build(q):
                        if stage == 1:
                            context = chain_data.render_context(q, "full", {"full": {"max_documents": "all"}})
                            return initial_compress(client, q, context, mode, base_cfg, seed)
                        previous = handoffs[(q.qid, mode, seed, stage - 1)]
                        return recompress(client, previous["text"], q.question, mode, base_cfg, seed, stage)
                    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
                        futures = {pool.submit(build, q): q for q in missing}
                        for future, q in futures.items():
                            result = future.result()
                            handoffs[(q.qid, mode, seed, stage)] = {
                                "qid": q.qid, "mode": mode, "seed": seed, "stage": stage, **result,
                            }
                ordered = [handoffs[(q.qid, m, s, st)] for m in cfg["modes"] for s in cfg["seeds"]
                           for q in questions for st in range(1, cfg["compression_depth"] + 1)
                           if (q.qid, m, s, st) in handoffs]
                if not args.dry_run:
                    write_jsonl(handoff_path, ordered)
                print(f"[question-context:handoff] {mode}/seed{seed}/stage{stage}: {len(missing)} generated")

    answer_path = run_root / "answers.jsonl"
    existing_rows = read_jsonl(answer_path)
    existing = {(r["mode"], int(r["depth"]), r.get("seed"), r["qid"], r["query_id"]): r for r in existing_rows}
    jobs = []
    for q in questions:
        context = chain_data.render_context(q, "full", {"full": {"max_documents": "all"}})
        for query in queries_for(q):
            direct_key = ("direct", 0, None, q.qid, query["query_id"])
            if direct_key not in existing:
                jobs.append((direct_key, q, query, context, None))
            for mode in cfg["modes"]:
                for seed in cfg["seeds"]:
                    key = (mode, cfg["compression_depth"], seed, q.qid, query["query_id"])
                    if key not in existing:
                        notes = handoffs[(q.qid, mode, seed, cfg["compression_depth"])]["text"]
                        jobs.append((key, q, query, notes, seed))
    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, query, material, base_cfg,
                               f"question_context_{key[0]}_d{key[1]}"): (key, q, query, seed)
                   for key, q, query, material, seed in jobs}
        for future, (key, q, query, seed) in futures.items():
            result = future.result()
            mode, depth, _, _, _ = key
            existing[key] = {"mode": mode, "depth": depth, "seed": seed, "qid": q.qid,
                             "query_id": query["query_id"], "query_type": query["query_type"],
                             "golds": query["golds"], **result}
    rows = sorted(existing.values(), key=lambda r: (r["mode"], int(r["depth"]), r.get("seed") or -1,
                                                     r["qid"], r["query_id"]))
    if not args.dry_run:
        write_jsonl(answer_path, rows)
        add_bertscore(rows, cfg)
        write_jsonl(answer_path, rows)
        analyse(rows, cfg, output_root)
    print(f"[question-context:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
