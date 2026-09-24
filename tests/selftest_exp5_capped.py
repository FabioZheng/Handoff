"""Offline proof that exp5_cap_only enforces a maximum and nothing else.

The fake backend exercises the plumbing but returns trivial text, so it never
reaches the branches that matter. This drives ``absorb`` directly with
synthetic backend results and checks every one:

 1. a short message is accepted as written, never sent back;
 2. a message exactly at the cap is accepted;
 3. an over-cap message is regenerated while attempts remain;
 4. an over-cap message on the last attempt is truncated, not discarded;
 5. an empty message is regenerated, and an empty final message is invalid;
 6. an unparseable selection is regenerated;
 7. a context overflow finalizes immediately;
 8. regeneration is bounded by MAX_REGENERATIONS;
 9. the violation notice is byte-identical across arms;
10. no prompt or notice anywhere contains floor, target or fill wording;
11. fill ratio, realized words and realized tokens are recorded;
12. a shared run manifest survives distinct Slurm array task identities.

    python tests/selftest_exp5_capped.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import exp5_prompts as ep  # noqa: E402
import run_exp5_capped as rc  # noqa: E402
from reuse_common import source_view, words  # noqa: E402

SOURCE = ("Ada Lovelace wrote the first published algorithm intended for a machine. "
          "She collaborated with Charles Babbage on the Analytical Engine. "
          "Her notes were published in 1843 and are longer than the paper they annotate. "
          "She described how the engine could act on symbols other than numbers.")
QUESTION = "In which year were Ada Lovelace's notes published?"
FORBIDDEN = ("at least", "minimum", "floor", "aim for", "approximately",
             "use the space", "use the available", "fill", "longer", "expand",
             "too short", "as close")


def make_doc() -> dict:
    return {"id": "test-doc", "corpus": "squad", "content_hash": "deadbeef",
            "source": SOURCE, "paragraphs": [{"id": "p1", "text": SOURCE}],
            "questions": [{"qid": "q1", "question": QUESTION, "golds": ["1843"],
                           "evidence_sets": [["p1"]]}]}


def make_job(policy: str, cap: int, question: str | None = QUESTION) -> rc.Job:
    doc = make_doc()
    view = source_view(doc["id"], doc["source"])
    if policy not in rc.CONDITIONED:
        question = None
    return rc.Job(doc, view, policy, cap, question, "q1" if question else None, "mistral24")


def result(text: str, finish: str = "stop", out_tokens: int = 42) -> dict:
    return {"key": "k", "text": text, "finish_reason": finish,
            "input_tokens": 100, "output_tokens": out_tokens}


def check_short_accepted() -> None:
    job = make_job("summary_conditioned", 128)
    before = job.conversation[1]["content"]
    rc.absorb(job, result("Notes published in 1843."), final=False)
    assert job.record is not None, "a short message must not be sent back"
    assert job.record["valid"] and job.record["delivered_words"] == 4
    assert job.record["truncated_words"] == 0 and not job.record["over_cap"]
    assert abs(job.record["fill_ratio"] - 4 / 128) < 1e-4  # stored rounded to 4dp
    assert job.record["realized_tokens"] == 42
    assert len(job.attempts) == 1 and job.attempts[0]["error"] == ""
    assert job.conversation[1]["content"] == before, "prompt must be untouched"
    print("1,11. short message accepted as written; length fields recorded")


def check_exact_cap_accepted() -> None:
    cap = 10
    job = make_job("summary_generic", cap)
    rc.absorb(job, result(" ".join(["word"] * cap)), final=False)
    assert job.record["valid"] and job.record["delivered_words"] == cap
    assert job.record["fill_ratio"] == 1.0 and not job.record["over_cap"]
    print("2. message exactly at the cap accepted")


def check_over_cap_regenerates_then_truncates() -> None:
    cap = 10
    job = make_job("summary_conditioned", cap)
    rc.absorb(job, result(" ".join(["word"] * 25)), final=False)
    assert job.record is None, "an over-cap message must be regenerated"
    notice = job.conversation[1]["content"]
    assert "at most 10" in notice and "Regenerate" in notice
    job2 = make_job("summary_conditioned", cap)
    rc.absorb(job2, result(" ".join(["word"] * 25)), final=True)
    assert job2.record["valid"], "a final over-cap message is truncated, not discarded"
    assert job2.record["delivered_words"] == cap
    assert job2.record["truncated_words"] > 0
    print("3,4. over-cap regenerates while attempts remain, then truncates")


def check_empty_and_malformed() -> None:
    job = make_job("summary_generic", 64)
    rc.absorb(job, result("   "), final=False)
    assert job.record is None and "empty" in job.attempts[-1]["error"].lower()
    job = make_job("summary_generic", 64)
    rc.absorb(job, result("   "), final=True)
    assert job.record is not None and not job.record["valid"]
    job = make_job("selection_conditioned", 64)
    rc.absorb(job, result("not a json array at all"), final=False)
    assert job.record is None, "an unparseable selection must be regenerated"
    print("5,6. empty and unparseable outputs regenerate; empty final is invalid")


def check_context_overflow() -> None:
    job = make_job("summary_generic", 64)
    rc.absorb(job, result("", finish="context_overflow"), final=False)
    assert job.record is not None and not job.record["valid"]
    assert job.record["error"] == "context_overflow"
    print("7. context overflow finalizes immediately")


def check_regeneration_bound() -> None:
    job = make_job("summary_conditioned", 10)
    rounds = 0
    for index in range(rc.MAX_REGENERATIONS + 1):
        rc.absorb(job, result(" ".join(["word"] * 40)), final=index == rc.MAX_REGENERATIONS)
        rounds += 1
        if job.record is not None:
            break
    assert rounds == rc.MAX_REGENERATIONS + 1
    assert len(job.attempts) == rc.MAX_REGENERATIONS + 1
    assert job.record["valid"] and job.record["delivered_words"] == 10
    print(f"8. regeneration bounded at {rc.MAX_REGENERATIONS} retries")


def check_notice_identical_across_arms() -> None:
    for pair in (("summary_conditioned", "summary_generic"),
                 ("selection_conditioned", "selection_generic")):
        cond, gen = (make_job(p, 10) for p in pair)
        payload = " ".join(["word"] * 40) if cond.mechanism == "generation" else "[0, 1, 2, 3]"
        rc.absorb(cond, result(payload), final=False)
        rc.absorb(gen, result(payload), final=False)
        if cond.record is not None or gen.record is not None:
            continue
        stripped = cond.conversation[1]["content"].replace(ep.question_marker(QUESTION), "")
        assert stripped == gen.conversation[1]["content"], (
            f"{pair}: regeneration prompts differ by more than the question line")
    print("9. violation notice byte-identical across arms")


def check_no_fill_wording() -> None:
    """Scan the template surface only; the source document is not ours to police."""
    texts = [ep.system_prompt(), ep.QUESTION_PREFIX, ep.SELECTION_TASK]
    for cap in (32, 1024):
        texts.append(ep.cap_clause(cap))
        texts.append(ep.INITIAL_INSTRUCTION + ep.cap_clause(cap))
        texts.append(ep.SELECTION_TASK + ep.cap_clause(cap))
    # Rendered prompts too, with the source block removed so a forbidden word
    # inside the document under test cannot mask or fake a failure.
    for mechanism in ep.MECHANISMS:
        for question in (QUESTION, None):
            for cap in (32, 1024):
                rendered = ep.user_prompt(SOURCE, question, cap, mechanism)
                texts.append(rendered.replace(SOURCE, "").replace(
                    ep.render_units(source_view("t", SOURCE)), ""))
    job = make_job("summary_conditioned", 10)
    rc.absorb(job, result(" ".join(["word"] * 40)), final=False)
    texts.append(job.conversation[1]["content"].split("Previous attempt feedback:")[-1])
    for text in texts:
        lowered = text.lower()
        for bad in FORBIDDEN:
            if bad in lowered:
                raise AssertionError(f"{bad!r} appears in a prompt or notice: {text[:120]!r}")
    print("10. no floor, target or fill wording in any prompt or notice")


def check_record_schema() -> None:
    job = make_job("summary_conditioned", 128)
    rc.absorb(job, result("Notes published in 1843."), final=False)
    required = {"family", "policy", "mechanism", "question_visible", "cap",
                "delivered_words", "fill_ratio", "realized_tokens", "over_cap",
                "truncated_words", "base_prompt_hash", "template_hash",
                "rendered_prompt_hash", "source_words", "message_id"}
    missing = required - set(job.record)
    assert not missing, f"record is missing {sorted(missing)}"
    assert job.record["family"] == "exp5_cap_only"
    assert job.record["question_visible"] is True
    assert job.record["base_prompt_hash"] == ep.base_prompt_hash()
    print("   record carries family, visibility, length and provenance fields")


def check_array_manifest_is_scheduler_independent() -> None:
    """Array tasks share protocol provenance, never a particular Slurm job."""
    frozen_cfg = {
        "run_root": "runs/exp5_cap_only",
        "cache_root": "cache/reuse",
        "data_root": "data",
        "budgets": {"absolute": [96, 120, 144, 176, 208, 256], "frozen": True},
    }
    args = argparse.Namespace(
        config="ignored.yaml", stage="write", model="mistral24", split="main",
        shard=0, shards=2, per_corpus=0, tag="", policies="", max_anchors=0,
        budgets="", allow_unfrozen=False, fake=True,
    )
    original_root, original_config, original_rows = rc.ROOT, rc.config, rc.read_rows
    scheduler_keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
    previous_env = {key: os.environ.get(key) for key in scheduler_keys}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rc.ROOT = Path(tmp)
            rc.config = lambda _path: copy.deepcopy(frozen_cfg)
            rc.read_rows = lambda _path: []
            manifests = []
            for shard, job_id, array_id in ((0, "9001", "9000"),
                                             (1, "9002", "9000"),
                                             (0, "9003", "9003")):
                args.shard = shard
                os.environ.update({"SLURM_JOB_ID": job_id,
                                   "SLURM_ARRAY_JOB_ID": array_id,
                                   "SLURM_ARRAY_TASK_ID": str(shard)})
                run_dir = rc.run(args)
                manifests.append((run_dir / "manifest.json").read_bytes())
            assert manifests[0] == manifests[1] == manifests[2]
            manifest = json.loads(manifests[0])
            assert not any(key.lower().startswith("slurm") for key in manifest)
    finally:
        rc.ROOT, rc.config, rc.read_rows = original_root, original_config, original_rows
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("12. shared manifest ignores per-array Slurm identities")


def main() -> int:
    check_short_accepted()
    check_exact_cap_accepted()
    check_over_cap_regenerates_then_truncates()
    check_empty_and_malformed()
    check_context_overflow()
    check_regeneration_bound()
    check_notice_identical_across_arms()
    check_no_fill_wording()
    check_record_schema()
    check_array_manifest_is_scheduler_independent()
    print("\nALL CAP-ONLY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
