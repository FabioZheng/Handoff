"""Repeated handoff in the exp5_cap_only prompt family.

Depth 1 is the frozen main run's message, written from the document. At depth d > 1
the same writer receives only the depth d-1 message as its source material and
writes a new message under the same cap, with the same question visibility and
the same conditioning question throughout the chain. Every writer stage uses the
unchanged exp5_cap_only prompt - Experiment 5's writer prompt plus the identical
maximum-length clause - so depth is the only thing that varies.

Summaries only: a selector re-choosing sentences from an already selected text
under the same cap would mostly return its input.

Outputs go to a sibling run root, <main run>-repeated, with one main-run layout per
depth (depth<d>/write|read|judge/<writer>), so the read and judge stages reuse
run_exp5_capped.py's functions unchanged.

    python src/run_exp5_repeated.py --stage write
    python src/run_exp5_repeated.py --stage read
    python src/run_exp5_repeated.py --stage judge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_exp5_capped as rc  # noqa: E402
from exp5_runs import parent_signature  # noqa: E402
from vllm_backend import FakeBackend, VLLMBackend  # noqa: E402
from reuse_common import atomic_json, digest, read_rows, source_view, write_once  # noqa: E402

FAMILY = "exp5_cap_only_repeated"
POLICIES = ("summary_generic", "summary_conditioned")
DEFAULT_CAPS = (96, 256)
DEFAULT_DEPTHS = (2, 3, 4, 5)
DEFAULT_DOCUMENTS = 50


def roots(cfg: dict, args) -> tuple[Path, Path, str]:
    if args.fake:
        cfg["run_root"] = str(Path(cfg["run_root"]) / "fake")
        cfg["cache_root"] = str(Path(cfg["cache_root"]) / "fake")
    signature = parent_signature(cfg)
    parent = ROOT / cfg["run_root"] / args.split / signature[:12]
    manifest = parent / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"parent run not found: {parent}")
    if json.loads(manifest.read_text(encoding="utf-8"))["protocol_hash"] != signature:
        raise SystemExit("parent run protocol differs from the active code and config")
    return parent, parent.with_name(f"{parent.name}-repeated"), signature


def chain_documents(cfg: dict, split: str, count: int) -> tuple[list[dict], list[dict]]:
    """The first `count` headline-corpus documents in cohort order."""
    all_docs = read_rows(ROOT / cfg["data_root"] / f"{split}.jsonl")
    corpora = cfg.get("cohort", {}).get("corpora")
    eligible = [d for d in all_docs if not corpora or d["corpus"] in corpora]
    return all_docs, eligible[:count]


def previous_messages(parent: Path, root: Path, depth: int, writer: str, doc_id: str,
                      caps: tuple[int, ...]) -> list[dict]:
    source = parent if depth == 2 else root / f"depth{depth - 1}"
    messages = rc.load_messages(source, writer, doc_id)
    return [m for m in messages if m["block"] == "core" and m["policy"] in POLICIES and m["cap"] in caps]


def chain_jobs(doc: dict, previous: list[dict], cfg: dict, writer: str, depth: int) -> list:
    questions = {q["qid"]: q for q in doc["questions"]}
    inference, jobs = cfg["inference"], []
    for prev in previous:
        if not prev["valid"] or not prev["text"]:
            continue  # a chain that broke stays broken; the read stage sees no message
        relay = {**doc, "source": prev["text"]}
        question = questions[prev["anchor"]]["question"] if prev["anchor"] else None
        job = rc.Job(relay, source_view(doc["id"], prev["text"]), prev["policy"], prev["cap"],
                     question, prev["anchor"], writer)
        job.allowance = max(inference["writer_min_tokens"],
                            math.ceil(job.cap * inference["writer_tokens_per_word"]) + 128)
        job.depth, job.parent = depth, prev
        jobs.append(job)
    return jobs


def finish(job) -> dict:
    record = dict(job.record)
    root_id = job.parent.get("chain_root_message_id") or job.parent["message_id"]
    record.update({
        "family": FAMILY, "depth": job.depth, "labels": [f"w{job.cap}", f"d{job.depth}"],
        "message_id": digest([job.doc["content_hash"], job.writer, job.policy, job.cap, job.qid,
                              "depth", job.depth]),
        "parent_message_id": job.parent["message_id"], "chain_root_message_id": root_id,
        "source_words": rc.words(job.parent["text"])})
    return record


def write_depth(group: list[dict], parent: Path, root: Path, depth: int, cfg: dict, writer: str,
                backend, caps: tuple[int, ...]) -> dict:
    jobs = [job for doc in group
            for job in chain_jobs(doc, previous_messages(parent, root, depth, writer, doc["id"], caps),
                                  cfg, writer, depth)]
    for round_index in range(rc.MAX_REGENERATIONS + 1):
        pending = [j for j in jobs if j.record is None]
        if not pending:
            break
        results = backend.generate([{"messages": j.conversation, "max_tokens": j.allowance} for j in pending])
        for job, result in zip(pending, results):
            rc.absorb(job, result, round_index == rc.MAX_REGENERATIONS)
    outputs = {doc["id"]: {"messages": []} for doc in group}
    for job in jobs:
        outputs[job.doc["id"]]["messages"].append(finish(job))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp5_cap_only_frozen_config.yaml")
    parser.add_argument("--stage", choices=["write", "read", "judge"], required=True)
    parser.add_argument("--split", choices=["development", "main"], default="main")
    parser.add_argument("--writer", choices=list(rc.WRITERS), default="mistral24")
    parser.add_argument("--caps", default=",".join(map(str, DEFAULT_CAPS)))
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    cfg = rc.config(str(ROOT / args.config))
    caps = tuple(int(c) for c in args.caps.split(","))
    depths = tuple(sorted(int(d) for d in args.depths.split(",")))
    if depths != tuple(range(2, depths[-1] + 1)):
        parser.error("depths must run consecutively from 2, e.g. 2,3,4,5")
    parent, root, signature = roots(cfg, args)
    missing = set(caps) - set(rc.budgets(cfg))
    if missing:
        parser.error(f"caps {sorted(missing)} are not in the frozen grid")
    all_docs, docs = chain_documents(cfg, args.split, args.documents)
    code = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    write_once(root / "manifest.json", {
        "family": FAMILY, "parent_run": parent.name, "parent_protocol_hash": signature,
        "writer": args.writer, "reader": args.writer, "judge": cfg["judge"], "policies": list(POLICIES),
        "caps": list(caps), "depths": [1, *depths], "document_ids": [d["id"] for d in docs],
        "cohort_hash": digest(all_docs), "fake": args.fake, "repeated_code_sha256": code})
    spec = cfg["judge"] if args.stage == "judge" else cfg["models"][args.writer]
    name = "-".join([FAMILY, args.stage, args.split, args.writer, os.environ.get("SLURM_JOB_ID", str(os.getpid()))])
    backend = None
    started, size = time.monotonic(), cfg["runtime"]["docs_per_group"]
    for depth in depths:
        depth_root = root / f"depth{depth}"
        subpath = Path(args.stage) / args.writer
        todo = [d for d in docs if not (depth_root / subpath / f"{d['id']}.json").exists()]
        print(json.dumps({"event": "start", "family": FAMILY, "stage": args.stage, "depth": depth,
                          "run_dir": str(depth_root), "documents": len(docs), "todo": len(todo)}), flush=True)
        if todo and backend is None:
            backend = FakeBackend() if args.fake else VLLMBackend(cfg, spec, name)
        for start in range(0, len(todo), size):
            group = todo[start:start + size]
            if args.stage == "write":
                outputs = write_depth(group, parent, root, depth, cfg, args.writer, backend, caps)
            elif args.stage == "read":
                outputs = rc.read_group(group, cfg, args.writer, backend, depth_root)
            else:
                outputs = rc.judge_group(group, cfg, backend, depth_root, args.writer)
            for doc_id, payload in outputs.items():
                atomic_json(depth_root / subpath / f"{doc_id}.json", payload)
            print(json.dumps({"event": "group", "depth": depth, "done": start + len(group), "of": len(todo),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
    print(json.dumps({"event": "done", "run_dir": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
