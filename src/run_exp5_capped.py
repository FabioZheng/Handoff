"""Write/read/judge stages for the ``exp5_cap_only`` family.

Separate from the earlier reuse pipeline on purpose. That pipeline's protocol hash
covers its own source, so editing it in place would move the legacy runs'
identity and strand outputs that are still wanted as a forced-fill diagnostic.
Nothing here imports its write path; only the pure helpers in ``reuse_common``
are shared, and no existing run root is read or written.

The contract this implements, and the one the older pipeline does not:

    L_actual <= B, and nothing else.

There is no floor, no target fill, no expansion retry and no regeneration
because a message came out short. A writer that stops well below the cap is the
behaviour under measurement. Regeneration happens only for a genuine protocol
violation - over the cap, unparseable, empty, or a context overflow - and after
the last attempt an over-cap message is truncated deterministically so the
channel claim still holds.

Every message records its nominal budget, realized words, realized tokens and
fill ratio, plus the prompt provenance item 20 of the design calls for.

  write/<writer>/<doc>.json   the four arms at every budget, plus references
  read/<reader>/<doc>.json    answers to every question of the document
  judge/<reader>/<doc>.json   binary verdicts from the judge model

    python src/run_exp5_capped.py --stage write --model mistral24 --split development
    python src/run_exp5_capped.py --stage read  --model mistral24 --split development
    python src/run_exp5_capped.py --stage judge --model mistral24 --split development
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp5_prompts as ep  # noqa: E402
from vllm_backend import FakeBackend, VLLMBackend  # noqa: E402
from reuse_common import (CLOSED_BOOK, atomic_json, config, digest, judge_prompt,  # noqa: E402
                         parse_selection, parse_verdict, read_rows, reader_prompt,
                         source_view, truncate_words, words, write_once)
from handoffs import SealedHandoff  # noqa: E402
from score import score_against_golds  # noqa: E402

FAMILY = ep.FAMILY_CAP_ONLY
WRITERS = ("qwen32", "mistral24")
GENERATION = ("summary_generic", "summary_conditioned")
SELECTION = ("selection_generic", "selection_conditioned")
POLICIES = GENERATION + SELECTION
CONDITIONED = ("summary_conditioned", "selection_conditioned")
REFERENCES = ("full_source", CLOSED_BOOK, "annotated_evidence")
ABSTENTIONS = {"insufficient evidence", "i dont know", "i do not know"}
# Genuine protocol violations only. Nothing here fires on a short message.
MAX_REGENERATIONS = 2
PROTOCOL_CODE = ("exp5_prompts.py", "run_exp5_capped.py", "reuse_common.py",
                 "vllm_backend.py", "score.py", "handoffs.py", "data.py")


# --------------------------------------------------------------------- setup

def protocol_hash(cfg: dict) -> str:
    here = Path(__file__).parent
    return digest({"family": FAMILY,
                   "config": {k: v for k, v in cfg.items() if k != "runtime"},
                   "code": {n: (here / n).read_text(encoding="utf-8") for n in PROTOCOL_CODE}})


def budgets(cfg: dict, allow_unfrozen: bool = False,
            override: list[int] | None = None) -> list[int]:
    """The six budgets, refused until calibration has frozen them.

    The guard exists because an arbitrary grid is the failure this protocol was
    rebuilt to avoid: a cap that no longer binds stops testing compression, and
    a cap below the generic control's floor leaves nothing for conditioning to
    remove. ``--allow-unfrozen`` exists for calibration and plumbing runs and
    stamps the manifest so such a run can never be mistaken for a reportable
    one. Only an unfrozen run may pass a candidate grid of another size, since
    choosing six values needs more than six to choose from.
    """
    frozen = bool(cfg["budgets"].get("frozen"))
    if not frozen and not allow_unfrozen:
        raise ValueError(
            "budgets.frozen is not set. Calibrate the six budgets on development "
            "data and freeze them, or pass --allow-unfrozen for a calibration run.")
    if override:
        if frozen:
            raise ValueError("--budgets may not override a frozen grid")
        grid = sorted(set(int(b) for b in override))
    else:
        grid = list(cfg["budgets"]["absolute"])
    if frozen and len(grid) != 6:
        raise ValueError(f"a frozen grid must have six budgets, found {len(grid)}")
    if any(b < 1 for b in grid):
        raise ValueError("budgets must be positive")
    return grid


def mechanism_of(policy: str) -> str:
    return "generation" if policy in GENERATION else "selection"


# --------------------------------------------------------------------- write

class Job:
    """One message under construction: prompt, attempts, and the chosen draft."""

    def __init__(self, doc, view, policy, cap, question, qid, writer):
        self.doc, self.view, self.policy, self.cap = doc, view, policy, cap
        self.question, self.qid, self.writer = question, qid, writer
        self.mechanism = mechanism_of(policy)
        self.source = doc["source"] if self.mechanism == "generation" else ep.render_units(view)
        self.attempts: list[dict] = []
        self.record: dict | None = None
        self.conversation = ep.conversation(self.source, question, cap, self.mechanism)
        self.allowance = 0

    @property
    def rendered(self) -> str:
        return self.conversation[1]["content"]


def _record(job: Job, text: str, ids: list[int], *, valid: bool, truncated: int = 0,
            mode: str | None = None, error: str = "") -> dict:
    delivered = words(text) if valid else 0
    return {
        "family": FAMILY, "policy": job.policy, "mechanism": job.mechanism,
        "question_visible": job.policy in CONDITIONED,
        "anchor": job.qid, "cap": job.cap, "labels": [f"w{job.cap}"], "block": "core",
        "writer": job.writer, "document_id": job.doc["id"], "corpus": job.doc["corpus"],
        "message_id": digest([job.doc["content_hash"], job.writer, job.policy, job.cap, job.qid]),
        "text": text if valid else "", "selected_ids": ids if valid else [],
        "valid": valid, "delivered_words": delivered,
        "fill_ratio": round(delivered / job.cap, 4) if job.cap else None,
        "over_cap": delivered > job.cap, "truncated_words": truncated,
        "realized_tokens": job.attempts[-1]["output_tokens"] if job.attempts else 0,
        "parse_mode": mode, "attempts": job.attempts, "error": error,
        "base_prompt_hash": ep.base_prompt_hash(),
        "template_hash": ep.template_hash(job.cap, job.mechanism),
        "rendered_prompt_hash": digest(job.rendered),
        "source_words": words(job.doc["source"]),
    }


def absorb(job: Job, result: dict, final: bool) -> None:
    """Accept anything within the cap. Regenerate only on a real violation."""
    raw = (result["text"] or "").strip()
    finish, error, text, ids, mode = result["finish_reason"], "", "", [], None
    if finish == "context_overflow":
        error = "context_overflow"
    else:
        try:
            if job.mechanism == "selection":
                text, ids, mode = parse_selection(raw, job.view, job.cap)
            else:
                text, mode = raw, "text"
            count = words(text)
            if not text:
                error = "Message is empty."
            elif count > job.cap:
                error = f"Message has {count} words; it may contain at most {job.cap}."
            # A short message is not an error and is never sent back.
        except (ValueError, TypeError) as exc:
            error = str(exc)
    job.attempts.append({"request_key": result["key"], "finish_reason": finish, "error": error,
                         "raw_words": words(raw), "parse_mode": mode,
                         "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"]})
    if not error:
        job.record = _record(job, text, ids, valid=True, mode=mode)
        return
    if final or finish == "context_overflow":
        if text:
            cut, dropped = truncate_words(text, job.cap)
            job.record = _record(job, cut, ids, valid=True, truncated=dropped, mode=mode,
                                 error=error)
        else:
            job.record = _record(job, "", [], valid=False, error=error)
        return
    # Regenerate from the same permitted source plus the violation notice. The
    # notice is byte-identical across arms and never mentions a floor.
    job.conversation = ep.conversation(job.source, job.question, job.cap, job.mechanism)
    job.conversation[-1]["content"] += (
        "\nPrevious attempt feedback: " + error + " Regenerate the response.")


def anchors_for(doc: dict, limit: int, seed: int) -> list[dict]:
    """A seeded, stable subset of this document's questions, for calibration runs."""
    questions = list(doc["questions"])
    if not limit or limit >= len(questions):
        return questions
    rng = random.Random(f"{seed}|{doc['id']}")
    return sorted(rng.sample(questions, limit), key=lambda q: q["qid"])


def write_jobs(doc: dict, cfg: dict, writer: str, grid: list[int],
               policies: tuple[str, ...], max_anchors: int = 0) -> list[Job]:
    view = source_view(doc["id"], doc["source"])
    jobs = []
    for cap in grid:
        for policy in policies:
            if policy in CONDITIONED:
                for question in anchors_for(doc, max_anchors, cfg["seed"]):
                    jobs.append(Job(doc, view, policy, cap, question["question"],
                                    question["qid"], writer))
            else:
                jobs.append(Job(doc, view, policy, cap, None, None, writer))
    inference = cfg["inference"]
    for job in jobs:
        if job.mechanism == "selection":
            job.allowance = max(inference["writer_min_tokens"], len(view.units) * 7 + 64)
        else:
            job.allowance = max(inference["writer_min_tokens"],
                                math.ceil(job.cap * inference["writer_tokens_per_word"]) + 128)
    return jobs


def reference_messages(doc: dict) -> list[dict]:
    rows = []
    for q in doc["questions"]:
        evidence_ids = set(q["evidence_sets"][0])
        evidence = "\n\n".join(p["text"] for p in doc["paragraphs"] if p["id"] in evidence_ids)
        for policy, text in (("full_source", doc["source"]), (CLOSED_BOOK, ""),
                             ("annotated_evidence", evidence)):
            rows.append({"family": FAMILY, "policy": policy, "mechanism": "reference",
                         "question_visible": False, "anchor": None, "cap": words(text),
                         "labels": [policy], "block": "reference", "writer": "reference",
                         "document_id": doc["id"], "corpus": doc["corpus"],
                         "message_id": digest([doc["content_hash"], policy, q["qid"]]),
                         "text": text, "selected_ids": [], "valid": True,
                         "delivered_words": words(text), "fill_ratio": None, "over_cap": False,
                         "truncated_words": 0, "realized_tokens": 0, "only_qid": q["qid"]})
    return rows


def write_group(group: list[dict], cfg: dict, writer: str, backend, grid: list[int],
                policies: tuple[str, ...], max_anchors: int = 0) -> dict:
    jobs = [job for doc in group
            for job in write_jobs(doc, cfg, writer, grid, policies, max_anchors)]
    for round_index in range(MAX_REGENERATIONS + 1):
        pending = [j for j in jobs if j.record is None]
        if not pending:
            break
        requests = [{"messages": j.conversation, "max_tokens": j.allowance} for j in pending]
        results = backend.generate(requests)
        final = round_index == MAX_REGENERATIONS
        for job, result in zip(pending, results):
            absorb(job, result, final)
    outputs = {doc["id"]: {"messages": []} for doc in group}
    for job in jobs:
        outputs[job.doc["id"]]["messages"].append(job.record)
    for doc in group:
        outputs[doc["id"]]["messages"].extend(reference_messages(doc))
    return outputs


# ---------------------------------------------------------------------- read

def load_messages(run_dir: Path, writer: str, doc_id: str) -> list[dict]:
    path = run_dir / "write" / writer / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Required writer stage missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["messages"]


def read_group(group: list[dict], cfg: dict, reader: str, backend, run_dir: Path) -> dict:
    plan, requests = [], []
    for doc in group:
        for message in load_messages(run_dir, reader, doc["id"]):
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


# --------------------------------------------------------------------- judge

def _abstained(prediction: str) -> bool:
    return re.sub(r"[^a-z ]", "", prediction.lower()).strip() in ABSTENTIONS


def judge_group(group: list[dict], cfg: dict, backend, run_dir: Path, reader: str) -> dict:
    plan, requests, outputs = [], [], {}
    for doc in group:
        path = run_dir / "read" / reader / f"{doc['id']}.json"
        if not path.exists():
            raise FileNotFoundError(f"Required answer set missing: {path}")
        answers = json.loads(path.read_text(encoding="utf-8"))["answers"]
        questions = {q["qid"]: q for q in doc["questions"]}
        outputs[doc["id"]] = {"verdicts": []}
        for answer in answers:
            q = questions[answer["qid"]]
            if _abstained(answer["prediction"]):
                outputs[doc["id"]]["verdicts"].append(
                    {**{k: answer[k] for k in ("message_id", "qid", "policy", "cap")},
                     "judge_correct": 0, "abstained": True, "parsed": True})
                continue
            evidence_ids = set(q["evidence_sets"][0])
            evidence = "\n\n".join(p["text"] for p in doc["paragraphs"] if p["id"] in evidence_ids)
            requests.append({"messages": judge_prompt(q["question"], q["golds"], evidence,
                                                      answer["prediction"]),
                             "max_tokens": cfg["inference"]["judge_tokens"]})
            plan.append((doc, answer))
    for (doc, answer), response in zip(plan, backend.generate(requests)):
        verdict = parse_verdict(response["text"])
        outputs[doc["id"]]["verdicts"].append(
            {**{k: answer[k] for k in ("message_id", "qid", "policy", "cap")},
             "judge_correct": verdict, "abstained": False, "parsed": verdict is not None})
    return outputs


# ----------------------------------------------------------------- execution

def select_documents(docs: list[dict], args) -> list[dict]:
    chosen = [d for i, d in enumerate(docs) if i % args.shards == args.shard]
    if args.per_corpus:
        chosen = [d for corpus in ("squad", "qasper")
                  for d in [x for x in chosen if x["corpus"] == corpus][: args.per_corpus]]
    return chosen


def run(args) -> Path | None:
    cfg = config(str(ROOT / args.config))
    if args.fake:
        cfg["run_root"] = str(Path(cfg["run_root"]) / "fake")
        cfg["cache_root"] = str(Path(cfg["cache_root"]) / "fake")
    all_docs = read_rows(ROOT / cfg["data_root"] / f"{args.split}.jsonl")
    # The frozen config names the headline corpora; a boundary-condition corpus
    # is documented by the calibration run and not rerun at full scale. Sharding
    # happens after this filter so shards stay balanced.
    corpora = cfg.get("cohort", {}).get("corpora")
    eligible = [d for d in all_docs if not corpora or d["corpus"] in corpora]
    docs = select_documents(eligible, args)
    override = [int(x) for x in args.budgets.split(",")] if args.budgets else None
    grid = budgets(cfg, args.allow_unfrozen, override)
    policies = tuple(args.policies.split(",")) if args.policies else POLICIES
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies {sorted(unknown)}")
    signature = protocol_hash(cfg)
    run_dir = ROOT / cfg["run_root"] / args.split / (signature[:12] + (f"-{args.tag}" if args.tag else ""))
    write_once(run_dir / "manifest.json", {
        "family": FAMILY, "protocol_hash": signature, "fake": args.fake,
        "cohort_hash": digest(all_docs), "budgets": grid,
        "budgets_frozen": bool(cfg["budgets"].get("frozen")),
        "base_prompt_hash": ep.base_prompt_hash(),
        "template_hashes": {m: {str(c): ep.template_hash(c, m) for c in grid}
                            for m in ep.MECHANISMS},
        "max_regenerations": MAX_REGENERATIONS, "policies": list(policies),
        "max_anchors": args.max_anchors,
        "config": {k: v for k, v in cfg.items() if k != "runtime"}})
    subpath = Path(args.stage) / args.model
    todo = [d for d in docs if not (run_dir / subpath / f"{d['id']}.json").exists()]
    print(json.dumps({"event": "start", "family": FAMILY, "stage": args.stage,
                      "model": args.model, "split": args.split, "run_dir": str(run_dir),
                      "documents": len(docs), "todo": len(todo), "budgets": grid,
                      "budgets_frozen": bool(cfg["budgets"].get("frozen"))}), flush=True)
    if not todo:
        return run_dir
    spec = cfg["judge"] if args.stage == "judge" else cfg["models"][args.model]
    name = "-".join([FAMILY, args.stage, args.split, args.model,
                     f"s{args.shard}of{args.shards}",
                     os.environ.get("SLURM_JOB_ID", str(os.getpid()))])
    backend = FakeBackend() if args.fake else VLLMBackend(cfg, spec, name)
    started, size = time.monotonic(), cfg["runtime"]["docs_per_group"]
    for start in range(0, len(todo), size):
        group = todo[start:start + size]
        if args.stage == "write":
            outputs = write_group(group, cfg, args.model, backend, grid, policies,
                                  args.max_anchors)
        elif args.stage == "read":
            outputs = read_group(group, cfg, args.model, backend, run_dir)
        else:
            outputs = judge_group(group, cfg, backend, run_dir, args.model)
        for doc_id, payload in outputs.items():
            atomic_json(run_dir / subpath / f"{doc_id}.json", payload)
        print(json.dumps({"event": "group", "done": start + len(group), "of": len(todo),
                          "seconds": round(time.monotonic() - started, 1)}), flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="exp5_cap_only_config.yaml")
    parser.add_argument("--stage", choices=["write", "read", "judge"], default="write")
    parser.add_argument("--model", choices=list(WRITERS), default="mistral24")
    parser.add_argument("--split", choices=["development", "main"], default="development")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--per-corpus", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--policies", default="",
                        help="comma-separated subset, e.g. summary_generic for a floor check")
    parser.add_argument("--max-anchors", type=int, default=0,
                        help="cap conditioned messages per document; 0 means every question")
    parser.add_argument("--budgets", default="",
                        help="candidate grid for an unfrozen calibration run, e.g. 24,40,64")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="plumbing tests only; stamps the manifest as unfrozen")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("Invalid shard")
    run_dir = run(args)
    print(json.dumps({"event": "done", "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
