"""Cross-reader robustness for exp5_cap_only: a second reader scores stored messages.

In the main run the writer also reads its own messages. Here a reader from another
model family (Qwen3-32B by default) answers every question from the stored
mistral24 messages - the closed-book, full-source and annotated-evidence
references included - and the study's judge scores those answers. No message is
regenerated, so any change in the contrasts comes from the reader.

Outputs go to a sibling run root, <parent>-x<reader>, laid out like a main run:
write/<writer> links to the parent's messages (read-only), and read/<writer> and
judge/<writer> hold the new reader's answers and verdicts. analyse_exp5_capped.py
and analyse_exp5_followup.py run on it unchanged via --run-dir.

    python src/run_exp5_cross_reader.py --stage read  --reader qwen32 --writer mistral24
    python src/run_exp5_cross_reader.py --stage judge --reader qwen32 --writer mistral24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_exp5_capped as rc  # noqa: E402
from exp5_runs import parent_signature  # noqa: E402
from vllm_backend import FakeBackend, VLLMBackend  # noqa: E402
from reuse_common import atomic_json, digest, read_rows, reader_prompt, write_once  # noqa: E402
from handoffs import SealedHandoff  # noqa: E402
from score import score_against_golds  # noqa: E402

FAMILY = "exp5_cap_only_cross_reader"


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
    return parent, parent.with_name(f"{parent.name}-x{args.reader}"), signature


def link_messages(parent: Path, cross: Path, writer: str) -> None:
    """write/<writer> in the cross root points at the parent's messages."""
    target, link = parent / "write" / writer, cross / "write" / writer
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target.resolve(), link, target_is_directory=True)
    except OSError:
        shutil.copytree(target, link)  # e.g. Windows without symlink rights; fake runs only


def cross_read_group(group: list[dict], cfg: dict, reader: str, writer: str, backend, parent: Path) -> dict:
    """rc.read_group with the reader and the writer decoupled; prompts and scoring unchanged."""
    plan, requests = [], []
    for doc in group:
        for message in rc.load_messages(parent, writer, doc["id"]):
            if not message["valid"]:
                continue
            for q in doc["questions"]:
                if message.get("only_qid") and q["qid"] != message["only_qid"]:
                    continue
                sealed = SealedHandoff(q["qid"], q["question"], message["text"], message["policy"])
                requests.append({"messages": reader_prompt(sealed),
                                 "max_tokens": cfg["inference"]["answer_tokens"]})
                plan.append((doc, message, q))
    outputs = {doc["id"]: {"answers": []} for doc in group}
    for (doc, message, q), response in zip(plan, backend.generate(requests)):
        prediction = response["text"].strip()
        em, f1 = score_against_golds(prediction, q["golds"])
        outputs[doc["id"]]["answers"].append({
            "message_id": message["message_id"], "request_key": response["key"],
            "document_id": doc["id"], "corpus": doc["corpus"], "qid": q["qid"],
            "reader": reader, "writer": message["writer"], "policy": message["policy"],
            "mechanism": message["mechanism"], "question_visible": message["question_visible"],
            "anchor": message["anchor"], "cap": message["cap"], "labels": message["labels"],
            "block": message["block"], "prediction": prediction, "em": em, "f1": f1,
            "is_now": bool(message["anchor"]) and message["anchor"] == q["qid"],
            "delivered_words": message["delivered_words"],
            "fill_ratio": message.get("fill_ratio"),
            "truncated_words": message.get("truncated_words", 0),
            "finish_reason": response["finish_reason"]})
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp5_cap_only_frozen_config.yaml")
    parser.add_argument("--stage", choices=["read", "judge"], required=True)
    parser.add_argument("--split", choices=["development", "main"], default="main")
    parser.add_argument("--writer", choices=list(rc.WRITERS), default="mistral24")
    parser.add_argument("--reader", choices=list(rc.WRITERS), default="qwen32")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--per-corpus", type=int, default=0)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    if args.reader == args.writer:
        parser.error("a cross reader must differ from the writer")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("Invalid shard")
    cfg = rc.config(str(ROOT / args.config))
    parent, cross, signature = roots(cfg, args)
    all_docs = read_rows(ROOT / cfg["data_root"] / f"{args.split}.jsonl")
    corpora = cfg.get("cohort", {}).get("corpora")
    docs = rc.select_documents([d for d in all_docs if not corpora or d["corpus"] in corpora], args)
    code = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    write_once(cross / "manifest.json", {
        "family": FAMILY, "parent_run": parent.name, "parent_protocol_hash": signature,
        "writer": args.writer, "reader": args.reader, "reader_model": cfg["models"][args.reader],
        "judge": cfg["judge"], "cohort_hash": digest(all_docs), "fake": args.fake,
        "cross_reader_code_sha256": code})
    link_messages(parent, cross, args.writer)
    subpath = Path(args.stage) / args.writer
    todo = [d for d in docs if not (cross / subpath / f"{d['id']}.json").exists()]
    print(json.dumps({"event": "start", "family": FAMILY, "stage": args.stage, "reader": args.reader,
                      "writer": args.writer, "run_dir": str(cross), "documents": len(docs),
                      "todo": len(todo)}), flush=True)
    if todo:
        spec = cfg["judge"] if args.stage == "judge" else cfg["models"][args.reader]
        name = "-".join([FAMILY, args.stage, args.split, args.reader, args.writer,
                         f"s{args.shard}of{args.shards}", os.environ.get("SLURM_JOB_ID", str(os.getpid()))])
        backend = FakeBackend() if args.fake else VLLMBackend(cfg, spec, name)
        started, size = time.monotonic(), cfg["runtime"]["docs_per_group"]
        for start in range(0, len(todo), size):
            group = todo[start:start + size]
            if args.stage == "read":
                outputs = cross_read_group(group, cfg, args.reader, args.writer, backend, parent)
            else:
                outputs = rc.judge_group(group, cfg, backend, cross, args.writer)
            for doc_id, payload in outputs.items():
                atomic_json(cross / subpath / f"{doc_id}.json", payload)
            print(json.dumps({"event": "group", "done": start + len(group), "of": len(todo),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
    print(json.dumps({"event": "done", "run_dir": str(cross)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
