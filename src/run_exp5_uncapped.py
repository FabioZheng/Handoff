"""Natural-length calibration: Experiment 5's prompt, no length clause at all.

This answers one question and only one: does showing the writer the current
question change how much it writes when nothing constrains the channel? It is
also what sets the budget grid for the main run, because a cap is only
informative while it still binds.

Write stage only. No reader, no judge, no accuracy. That keeps it cheap enough
to run on a development subset and leaves the expensive compute for the capped
study.

    python src/run_exp5_uncapped.py --model mistral24 --split development
    python src/run_exp5_uncapped.py --model mistral24 --fake      # plumbing only

Outputs go to ``runs/exp5_uncapped/`` and ``results/exp5_uncapped/``; nothing
here reads or writes the earlier reuse run roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp5_prompts as ep  # noqa: E402
from vllm_backend import FakeBackend, VLLMBackend  # noqa: E402
from reuse_common import config, digest  # noqa: E402

FAMILY = ep.FAMILY_UNCAPPED
# Generous ceiling: the point is where the writer stops on its own, so any
# message that ends because it ran out of tokens is recorded and excluded from
# the natural-length statistics rather than silently averaged in.
MAX_OUTPUT_TOKENS = 4096


def words(text: str) -> int:
    return len(str(text or "").split())


def load_cohort(cfg: dict, split: str) -> list[dict]:
    path = ROOT / cfg["data_root"] / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def anchors_for(doc: dict, per_doc: int, seed: int) -> list[dict]:
    """A seeded, stable subset of this document's questions."""
    questions = list(doc["questions"])
    if per_doc >= len(questions):
        return questions
    rng = random.Random(f"{seed}|{doc['id']}")
    return sorted(rng.sample(questions, per_doc), key=lambda q: q["qid"])


def build_tasks(docs: list[dict], per_doc: int, seed: int) -> list[dict]:
    tasks = []
    for doc in docs:
        tasks.append({"document_id": doc["id"], "corpus": doc["corpus"], "arm": "generic",
                      "qid": None, "question": None, "source": doc["source"]})
        for question in anchors_for(doc, per_doc, seed):
            tasks.append({"document_id": doc["id"], "corpus": doc["corpus"],
                          "arm": "conditioned", "qid": question["qid"],
                          "question": question["question"], "source": doc["source"]})
    return tasks


def run(cfg: dict, args) -> Path:
    docs = load_cohort(cfg, args.split)
    if args.limit:
        by_corpus: dict[str, list[dict]] = {}
        for doc in docs:
            by_corpus.setdefault(doc["corpus"], []).append(doc)
        docs = [d for group in by_corpus.values() for d in group[: args.limit]]
    tasks = build_tasks(docs, args.anchors, cfg["seed"])

    spec = cfg["models"][args.model]
    backend = (FakeBackend() if args.fake
               else VLLMBackend(cfg, spec, f"{FAMILY}-{args.model}"))

    requests = []
    for task in tasks:
        # assert_minimal_pair runs here, on the real source, for every
        # conditioned message this job writes.
        conversation = ep.conversation(task["source"], task["question"], None)
        requests.append({"messages": conversation, "max_tokens": MAX_OUTPUT_TOKENS})
    results = backend.generate(requests)

    rows = []
    for task, request, result in zip(tasks, requests, results):
        text = (result or {}).get("text", "")
        rendered = request["messages"][1]["content"]
        rows.append({
            "family": FAMILY,
            "split": args.split,
            "writer": args.model,
            "corpus": task["corpus"],
            "document_id": task["document_id"],
            "qid": task["qid"],
            "arm": task["arm"],
            "question_visible": task["arm"] == "conditioned",
            "budget_words": None,
            "realized_words": words(text),
            "realized_tokens": (result or {}).get("output_tokens", 0),
            "input_tokens": (result or {}).get("input_tokens", 0),
            "fill_ratio": None,
            "finish_reason": (result or {}).get("finish_reason", "missing"),
            "source_words": words(task["source"]),
            "base_prompt_hash": ep.base_prompt_hash(),
            "template_hash": ep.template_hash(None),
            "rendered_prompt_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "rendered_prompt": rendered if args.store_prompts else None,
            "system_prompt": ep.system_prompt(),
            "seed": cfg["seed"],
            "text": text,
        })

    # Synthetic outputs never share a path with a real run.
    out_dir = ROOT / "runs" / FAMILY / ("fake" if args.fake else "") / args.split / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = out_dir / "messages.jsonl"
    if messages.exists() and not args.overwrite:
        raise SystemExit(f"{messages} exists; pass --overwrite only if you mean it")
    messages.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({
        "family": FAMILY,
        "split": args.split,
        "writer": args.model,
        "model": spec,
        "documents": len(docs),
        "messages": len(rows),
        "anchors_per_document": args.anchors,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "length_clause": None,
        "base_prompt_hash": ep.base_prompt_hash(),
        "template_hash": ep.template_hash(None),
        "config_hash": digest(cfg),
        "cohort_file": f"{cfg['data_root']}/{args.split}.jsonl",
        "backend": getattr(backend, "identity", {}),
        "slurm_job_id": args.job_id,
        "fake": bool(args.fake),
    }, indent=2), encoding="utf-8")
    return messages


def summarise(messages: Path) -> str:
    rows = [json.loads(line) for line in messages.read_text(encoding="utf-8").splitlines() if line]
    natural = [r for r in rows if r["finish_reason"] == "stop"]
    lines = ["| corpus | arm | n | mean words | median | p10 | p90 | mean tokens | hit ceiling |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    stats: dict[tuple[str, str], list[int]] = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        for arm in ("generic", "conditioned"):
            group = [r for r in natural if r["corpus"] == corpus and r["arm"] == arm]
            capped = [r for r in rows if r["corpus"] == corpus and r["arm"] == arm
                      and r["finish_reason"] == "length"]
            if not group:
                continue
            lengths = sorted(r["realized_words"] for r in group)
            stats[(corpus, arm)] = lengths
            quantile = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]  # noqa: E731
            lines.append(
                f"| {corpus} | {arm} | {len(lengths)} | {statistics.mean(lengths):.0f} | "
                f"{statistics.median(lengths):.0f} | {quantile(0.1)} | {quantile(0.9)} | "
                f"{statistics.mean([r['realized_tokens'] for r in group]):.0f} | {len(capped)} |")
    lines.append("")
    lines.append("| corpus | conditioned/generic median | conditioned/generic mean |")
    lines.append("|---|---:|---:|")
    for corpus in sorted({c for c, _ in stats}):
        gen, cond = stats.get((corpus, "generic")), stats.get((corpus, "conditioned"))
        if not gen or not cond:
            continue
        lines.append(f"| {corpus} | {statistics.median(cond) / statistics.median(gen):.2f} | "
                     f"{statistics.mean(cond) / statistics.mean(gen):.2f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reuse_config.yaml")
    parser.add_argument("--split", default="development")
    parser.add_argument("--model", default="mistral24")
    parser.add_argument("--anchors", type=int, default=2,
                        help="conditioned messages per document")
    parser.add_argument("--limit", type=int, default=0, help="documents per corpus")
    parser.add_argument("--store-prompts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    cfg = config(str(ROOT / args.config))
    messages = run(cfg, args)
    report = summarise(messages)
    out = ROOT / "results" / FAMILY / ("fake" if args.fake else "") / args.split / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "length_calibration.md").write_text(
        f"# {FAMILY} natural-length calibration\n\n"
        f"Writer `{args.model}`, split `{args.split}`, Experiment 5 prompt with no "
        f"length clause.\nOnly messages that stopped on their own are summarised; "
        f"any that hit the {MAX_OUTPUT_TOKENS}-token ceiling are counted separately.\n\n"
        f"{report}\n", encoding="utf-8")
    print(report)
    print(f"\nmessages: {messages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
