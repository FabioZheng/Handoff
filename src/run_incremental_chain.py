"""Incremental-evidence MuSiQue chain with controlled relay-only depth.

Specialists receive the sealed previous handoff plus exactly one new supporting
evidence packet.  A configurable number of relay-only agents, which accept only
``SealedHandoff``, are inserted between specialists.  The final answerer and
hidden-probe answerers use the same sealed answer path and scoring utilities as
the original repeated-handoff experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
import incremental_chain_data as inc_data  # noqa: E402
import run_chain  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import CostCapExceeded, LLMClient, load_config  # noqa: E402
from run_slack_facts import mentions_answer, mentions_topic  # noqa: E402
from score import bootstrap_ci, em_f1_disagreement, paired_bootstrap_delta, score_against_golds  # noqa: E402


SPECIALIST_INSTRUCTION = (
    "Update the research notes for the next agent by integrating the new evidence packet with "
    "the previous notes. Preserve every answer-relevant fact, qualifier, date, number, "
    "relationship, uncertainty, and source id. Use only the supplied notes and packet. "
    "Do not answer the question directly and do not add unsupported facts."
)


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str


def condition_question_visible(experiment_cfg: dict, condition: str) -> bool:
    return bool(experiment_cfg["conditions"][condition]["question_visible"])


def assigned_order_mode(mode: str, sample_index: int, seed: int) -> str:
    if mode != "counterbalanced":
        return mode
    return "forward" if (int(sample_index) + int(seed)) % 2 == 0 else "reverse"


def specialist_user_prompt(
    previous: hm.SealedHandoff | None,
    packet: inc_data.EvidencePacket,
    packet_index: int,
    question_text: str,
    question_visible: bool,
) -> str:
    if previous is not None and not isinstance(previous, hm.SealedHandoff):
        raise TypeError("specialist previous state must be a SealedHandoff or None")
    parts = []
    if previous is not None:
        parts.append(f"Previous agent's notes:\n{previous.handoff_text}")
    parts.append(f"New evidence packet:\n{inc_data.render_packet(packet, packet_index)}")
    if question_visible:
        parts.append(f"Question the final agent must answer: {question_text}")
    parts.append(SPECIALIST_INSTRUCTION)
    return "\n\n".join(parts)


def specialist_update(
    client,
    previous: hm.SealedHandoff | None,
    packet: inc_data.EvidencePacket,
    packet_index: int,
    qid: str,
    question_text: str,
    cfg: dict,
    seed: int,
    stage: int,
    question_visible: bool,
    mechanism: str,
    decoding_seed: int,
) -> hm.Handoff:
    """One specialist update; hidden probes are never rendered into the prompt."""

    if previous is not None:
        if not isinstance(previous, hm.SealedHandoff):
            raise TypeError("specialist_update accepts previous state only as SealedHandoff")
        if previous.qid != qid:
            raise ValueError("specialist received a handoff for a different question")
    user = specialist_user_prompt(previous, packet, packet_index, question_text, question_visible)
    result = client.chat(
        [
            {"role": "system", "content": run_chain.QUESTION_CONDITIONED_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        seed=decoding_seed,
        tag=f"incremental_specialist_packet{packet_index}",
    )
    return hm.Handoff(
        qid=qid,
        mechanism=mechanism,
        seed=seed,
        text=result.text.strip(),
        meta={
            "stage": stage,
            "agent_type": "specialist",
            "packet_index": packet_index,
            "packet_id": packet.packet_id,
            "source_pid": packet.support_pid,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cached": result.cached,
            "decoding_seed": decoding_seed,
        },
    )


def event_seed(seed: int, logical_packet_index: int, agent_type: str, relay_index: int | None) -> int:
    """Tie decoding randomness to logical events, not depth-dependent stage ids."""

    base = int(seed) * 100_000 + int(logical_packet_index) * 1_000
    return base if agent_type == "specialist" else base + int(relay_index or 0)


def prompt_equivalence_selftest(packet: inc_data.EvidencePacket) -> None:
    marker = "\n\nQuestion the final agent must answer: fixture question"
    conditioned = specialist_user_prompt(None, packet, 1, "fixture question", True)
    generic = specialist_user_prompt(None, packet, 1, "fixture question", False)
    if marker not in conditioned or conditioned.replace(marker, "") != generic:
        raise AssertionError("specialist conditions differ by more than the question block")
    hidden = {probe.question for probe in packet.probes}
    if any(question in conditioned for question in hidden):
        raise AssertionError("a hidden probe leaked into the specialist prompt")


def prepare_examples(client, base_cfg: dict, experiment_cfg: dict, args):
    args.datasets = ["musique"]
    data_root = ROOT / experiment_cfg["outputs"]["data_root"]
    questions_by_dataset = run_chain.prepare_questions(
        client, base_cfg, experiment_cfg, args, data_root,
    )
    minimum = int(experiment_cfg.get("packets", {}).get("minimum_count", 2))
    mode = experiment_cfg.get("evidence_order", {}).get("mode", "counterbalanced")
    order_seed = int(experiment_cfg.get("evidence_order", {}).get(
        "seed", experiment_cfg["sampling"]["seed"],
    ))
    questions = []
    packet_map = {}
    manifests = []
    for sample_index, question in enumerate(questions_by_dataset["musique"]):
        # Exact alternation gives the selected sample a genuinely balanced
        # forward/reverse order assignment (difference at most one when n is odd).
        assigned_mode = assigned_order_mode(mode, sample_index, order_seed)
        order_id, packets = inc_data.order_packets(question, assigned_mode, order_seed)
        if len(packets) < minimum:
            print(f"[incremental:data] skip {question.qid}: {len(packets)} < {minimum} packets")
            continue
        questions.append(question)
        packet_map[question.qid] = (order_id, packets)
        manifests.append(inc_data.packet_manifest_row(question, order_id, packets))
    questions = questions[: args.n]
    packet_map = {question.qid: packet_map[question.qid] for question in questions}
    manifests = [row for row in manifests if row["qid"] in packet_map]
    if not questions:
        raise RuntimeError("no multi-packet MuSiQue examples survived preparation")
    if len(questions) < args.n:
        print(f"[incremental:data] WARNING only {len(questions)}/{args.n} examples available")
    if not args.dry_run:
        run_chain.write_jsonl(data_root / "packets.jsonl", manifests)
    prompt_equivalence_selftest(packet_map[questions[0].qid][1][0])
    return questions, packet_map


def handoff_key(row: dict) -> tuple:
    return (
        row["condition"], int(row["relay_depth"]), int(row["seed"]),
        row["qid"], int(row["stage"]),
    )


def generate_handoffs(
    client,
    base_cfg: dict,
    experiment_cfg: dict,
    args,
    questions,
    packet_map,
    run_root: Path,
) -> tuple[dict[tuple, hm.Handoff], dict[tuple, dict]]:
    path = run_root / "handoffs.jsonl"
    existing_rows = [] if args.force else run_chain.read_jsonl(path)
    existing = {handoff_key(row): row for row in existing_rows}
    schedules = {
        (question.qid, relay_depth): inc_data.build_schedule(packet_map[question.qid][1], relay_depth)
        for question in questions for relay_depth in args.relay_depths
    }

    for condition in args.conditions:
        question_visible = condition_question_visible(experiment_cfg, condition)
        for relay_depth in args.relay_depths:
            for seed in args.seeds:
                max_stage = max(len(schedules[(question.qid, relay_depth)]) for question in questions)
                for stage in range(1, max_stage + 1):
                    todo = [
                        question for question in questions
                        if stage <= len(schedules[(question.qid, relay_depth)])
                        and (condition, relay_depth, seed, question.qid, stage) not in existing
                    ]

                    def build(question, stage=stage):
                        order_id, packets = packet_map[question.qid]
                        schedule = schedules[(question.qid, relay_depth)]
                        step = schedule[stage - 1]
                        logical_packet_index = sum(
                            1 for candidate in schedule[:stage] if candidate.agent_type == "specialist"
                        )
                        previous = None
                        if stage > 1:
                            previous_row = existing[(condition, relay_depth, seed, question.qid, stage - 1)]
                            previous_handoff = hm.Handoff.from_json(previous_row)
                            previous = hm.seal(question.question, previous_handoff)
                        decoding_seed = event_seed(
                            seed, logical_packet_index, step.agent_type, step.relay_index,
                        )
                        mechanism = f"incremental_{condition}_r{relay_depth}_d{stage}"
                        if step.agent_type == "specialist":
                            packet = packets[int(step.packet_index) - 1]
                            handoff = specialist_update(
                                client, previous, packet, int(step.packet_index), question.qid,
                                question.question, base_cfg, seed, stage, question_visible,
                                mechanism, decoding_seed,
                            )
                        else:
                            if previous is None:
                                raise AssertionError("relay cannot be the first chain step")
                            handoff = run_chain.recompress(
                                client, previous, base_cfg, seed, stage, question_visible,
                                decoding_seed=decoding_seed,
                                tag=f"incremental_relay_after_packet{logical_packet_index}_r{step.relay_index}",
                            )
                            handoff.mechanism = mechanism
                            handoff.meta.update({
                                "agent_type": "relay",
                                "packet_index": None,
                                "packet_id": None,
                                "relay_index": step.relay_index,
                            })
                        row = handoff.to_json()
                        row.update({
                            "dataset": "musique",
                            "condition": condition,
                            "question_visible": question_visible,
                            "relay_depth": relay_depth,
                            "evidence_order": order_id,
                            "stage": stage,
                            "agent_type": step.agent_type,
                            "packet_index": step.packet_index,
                            "packet_id": step.packet_id,
                            "relay_index": step.relay_index,
                            "logical_packet_index": logical_packet_index,
                        })
                        return row

                    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
                        futures = {pool.submit(build, question): question for question in todo}
                        for future, question in futures.items():
                            row = future.result()
                            existing[(condition, relay_depth, seed, question.qid, stage)] = row
                    if not args.dry_run:
                        run_chain.write_jsonl(path, [row for _, row in sorted(existing.items())])
                    if todo:
                        print(
                            f"[incremental:handoff] {condition}/r{relay_depth}/seed{seed}/"
                            f"stage{stage}: {len(todo)} generated",
                            flush=True,
                        )

    selected_keys = {
        (condition, relay_depth, seed, question.qid, step.stage)
        for condition in args.conditions for relay_depth in args.relay_depths
        for seed in args.seeds for question in questions
        for step in schedules[(question.qid, relay_depth)]
    }
    rows = {key: row for key, row in existing.items() if key in selected_keys}
    handoffs = {key: hm.Handoff.from_json(row) for key, row in rows.items()}
    return handoffs, rows


def main_answer_key(row: dict) -> tuple:
    if row.get("source") == "original_evidence":
        return ("original_evidence", row["qid"])
    return (
        "final_handoff", row["condition"], int(row["relay_depth"]),
        int(row["seed"]), row["qid"],
    )


def answer_and_score(client, sealed: hm.SealedHandoff, cfg: dict, golds: list[str], tag: str) -> dict:
    answer = hm.orchestrator_answer(client, sealed, cfg, tag)
    em, f1 = score_against_golds(answer["pred"], golds)
    return {**answer, "em": em, "f1": f1}


def run_main_answers(
    client,
    base_cfg: dict,
    args,
    questions,
    packet_map,
    handoffs,
    handoff_rows,
    run_root: Path,
) -> list[dict]:
    path = run_root / "answers.jsonl"
    existing_rows = [] if args.force else run_chain.read_jsonl(path)
    existing = {main_answer_key(row): row for row in existing_rows}
    jobs = []

    for question in questions:
        order_id, packets = packet_map[question.qid]
        original = inc_data.render_original_evidence(packets)
        baseline_key = ("original_evidence", question.qid)
        if baseline_key not in existing:
            def baseline_call(question=question, original=original):
                answer = run_chain.answer_context(
                    client, question, original, "musique", "incremental_original", base_cfg,
                )
                em, f1 = score_against_golds(answer["pred"], question.golds)
                return {**answer, "em": em, "f1": f1}
            jobs.append((baseline_key, question, None, None, baseline_call))

        for condition in args.conditions:
            for relay_depth in args.relay_depths:
                for seed in args.seeds:
                    key = ("final_handoff", condition, relay_depth, seed, question.qid)
                    if key in existing:
                        continue
                    stages = [
                        item[4] for item in handoffs
                        if item[:4] == (condition, relay_depth, seed, question.qid)
                    ]
                    final_stage = max(stages)
                    final_handoff = handoffs[(condition, relay_depth, seed, question.qid, final_stage)]
                    sealed = hm.seal(question.question, final_handoff)
                    call = lambda sealed=sealed, question=question, condition=condition, relay_depth=relay_depth: answer_and_score(
                        client, sealed, base_cfg, list(question.golds),
                        f"incremental_main_{condition}_r{relay_depth}",
                    )
                    jobs.append((key, question, condition, (relay_depth, seed, final_stage), call))

    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(call): (key, question, condition, chain_info)
                   for key, question, condition, chain_info, call in jobs}
        for future, (key, question, condition, chain_info) in futures.items():
            order_id, packets = packet_map[question.qid]
            original = inc_data.render_original_evidence(packets)
            answer = future.result()
            if key[0] == "original_evidence":
                existing[key] = {
                    "dataset": "musique", "context_variant": "original_evidence",
                    "condition": "original_evidence", "source": "original_evidence",
                    "depth": 0, "relay_depth": None, "seed": None, "qid": question.qid,
                    "question": question.question, "n_hops": question.n_hops,
                    "packet_count": len(packets), "evidence_order": order_id,
                    "pred": answer["pred"], "raw": answer["raw"], "gold": question.answer,
                    "golds": list(question.golds), "em": answer["em"], "f1": answer["f1"],
                    "context_documents": len(packets), "gold_documents": len(packets),
                    "context_characters": len(original), "final_handoff_characters": len(original),
                    "chain_prompt_tokens": 0, "chain_completion_tokens": 0,
                    "answer_prompt_tokens": answer["prompt_tokens"],
                    "answer_completion_tokens": answer["completion_tokens"],
                }
            else:
                relay_depth, seed, final_stage = chain_info
                chain_rows = [
                    handoff_rows[(condition, relay_depth, seed, question.qid, stage)]
                    for stage in range(1, final_stage + 1)
                ]
                existing[key] = {
                    "dataset": "musique", "context_variant": condition,
                    "condition": condition, "source": "final_handoff",
                    "depth": relay_depth, "relay_depth": relay_depth, "seed": seed,
                    "qid": question.qid, "question": question.question,
                    "n_hops": question.n_hops, "packet_count": len(packets),
                    "evidence_order": order_id, "stage": final_stage,
                    "pred": answer["pred"], "raw": answer["raw"], "gold": question.answer,
                    "golds": list(question.golds), "em": answer["em"], "f1": answer["f1"],
                    "context_documents": len(packets), "gold_documents": len(packets),
                    "context_characters": len(original),
                    "final_handoff_characters": len(handoffs[(condition, relay_depth, seed, question.qid, final_stage)].text),
                    "chain_prompt_tokens": sum(int(row["meta"].get("prompt_tokens", 0) or 0) for row in chain_rows),
                    "chain_completion_tokens": sum(int(row["meta"].get("completion_tokens", 0) or 0) for row in chain_rows),
                    "answer_prompt_tokens": answer["prompt_tokens"],
                    "answer_completion_tokens": answer["completion_tokens"],
                }

    rows = sorted(existing.values(), key=lambda row: (
        row.get("source", ""), row.get("condition", ""),
        -1 if row.get("relay_depth") is None else int(row["relay_depth"]),
        -1 if row.get("seed") is None else int(row["seed"]), row["qid"],
    ))
    if not args.dry_run:
        run_chain.write_jsonl(path, rows)
    print(f"[incremental:main answers] {len(rows)} rows; {len(jobs)} generated")
    return rows


def probe_answer_key(row: dict) -> tuple:
    if row.get("source") == "original_evidence":
        return ("original_evidence", "", -1, -1, row["qid"], 0, row["probe_id"])
    relay_depth = -1 if row.get("relay_depth") is None else int(row["relay_depth"])
    seed = -1 if row.get("seed") is None else int(row["seed"])
    return (
        row["source"], row.get("condition", ""), relay_depth, seed,
        row["qid"], int(row.get("stage", 0)), row["probe_id"],
    )


def probe_catalog(questions, packet_map) -> dict[tuple[str, str], dict]:
    result = {}
    for question in questions:
        order_id, packets = packet_map[question.qid]
        for packet_index, packet in enumerate(packets, start=1):
            for probe in packet.probes:
                result[(question.qid, probe.probe_id)] = {
                    "question": question,
                    "order_id": order_id,
                    "packets": packets,
                    "packet": packet,
                    "packet_index": packet_index,
                    "probe": probe,
                }
    return result


def run_probe_answers(
    client,
    base_cfg: dict,
    args,
    questions,
    packet_map,
    handoffs,
    handoff_rows,
    run_root: Path,
) -> list[dict]:
    path = run_root / "probe_answers.jsonl"
    existing_rows = [] if args.force else run_chain.read_jsonl(path)
    existing = {probe_answer_key(row): row for row in existing_rows}
    catalog = probe_catalog(questions, packet_map)
    jobs = []

    for (qid, probe_id), item in catalog.items():
        probe = item["probe"]
        original = inc_data.render_original_evidence(item["packets"])
        key = ("original_evidence", "", -1, -1, qid, 0, probe_id)
        if key not in existing:
            eval_question = EvalQuestion(probe_id, probe.question)

            def baseline_call(eval_question=eval_question, original=original, probe=probe):
                answer = run_chain.answer_context(
                    client, eval_question, original, "musique", "incremental_probe_original", base_cfg,
                )
                em, f1 = score_against_golds(answer["pred"], list(probe.golds))
                return {**answer, "em": em, "f1": f1}
            jobs.append((key, item, None, baseline_call))

    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            for seed in args.seeds:
                for question in questions:
                    schedule = inc_data.build_schedule(packet_map[question.qid][1], relay_depth)
                    intro = inc_data.introduction_stages(schedule)
                    for step in schedule:
                        handoff = handoffs[(condition, relay_depth, seed, question.qid, step.stage)]
                        handoff_row = handoff_rows[(condition, relay_depth, seed, question.qid, step.stage)]
                        for (qid, probe_id), item in catalog.items():
                            if qid != question.qid:
                                continue
                            introduced_at = intro[item["packet"].packet_id]
                            if step.stage < introduced_at:
                                continue
                            key = (
                                "handoff", condition, relay_depth, seed,
                                question.qid, step.stage, probe_id,
                            )
                            if key in existing:
                                continue
                            sealed = hm.seal(item["probe"].question, handoff)
                            call = lambda sealed=sealed, item=item, condition=condition, relay_depth=relay_depth: answer_and_score(
                                client, sealed, base_cfg, list(item["probe"].golds),
                                f"incremental_probe_{condition}_r{relay_depth}",
                            )
                            metadata = {
                                "introduced_at": introduced_at,
                                "handoff_row": handoff_row,
                                "final_stage": len(schedule),
                            }
                            jobs.append((key, item, metadata, call))

    with ThreadPoolExecutor(max_workers=base_cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(call): (key, item, metadata)
                   for key, item, metadata, call in jobs}
        for future, (key, item, metadata) in futures.items():
            probe = item["probe"]
            answer = future.result()
            common = {
                "dataset": "musique", "evaluation_type": "probe",
                "qid": item["question"].qid, "probe_id": probe.probe_id,
                "probe_step": probe.step, "question": probe.question,
                "question_template": probe.question_template,
                "packet_id": item["packet"].packet_id,
                "packet_index": item["packet_index"], "support_pid": probe.support_pid,
                "evidence_order": item["order_id"],
                "pred": answer["pred"], "raw": answer["raw"],
                "gold": probe.answer, "golds": list(probe.golds),
                "em": answer["em"], "f1": answer["f1"],
                "answer_prompt_tokens": answer["prompt_tokens"],
                "answer_completion_tokens": answer["completion_tokens"],
            }
            if key[0] == "original_evidence":
                existing[key] = {
                    **common, "source": "original_evidence", "condition": "original_evidence",
                    "context_variant": "original_evidence", "relay_depth": None,
                    "depth": 0, "seed": None, "stage": 0,
                    "introduction_stage": None, "handoff_age": None,
                    "agent_type": "source", "final_handoff": False,
                }
            else:
                handoff_row = metadata["handoff_row"]
                existing[key] = {
                    **common, "source": "handoff", "condition": key[1],
                    "context_variant": key[1], "relay_depth": key[2],
                    "depth": key[2], "seed": key[3], "stage": key[5],
                    "introduction_stage": metadata["introduced_at"],
                    "handoff_age": inc_data.handoff_age(key[5], metadata["introduced_at"]),
                    "agent_type": handoff_row["agent_type"],
                    "final_handoff": key[5] == metadata["final_stage"],
                    "final_handoff_characters": len(handoff_row["text"]),
                }

    rows = sorted(existing.values(), key=lambda row: (
        row["source"], row.get("condition", ""),
        -1 if row.get("relay_depth") is None else int(row["relay_depth"]),
        -1 if row.get("seed") is None else int(row["seed"]),
        row["qid"], int(row.get("stage", 0)), row["probe_id"],
    ))
    if not args.dry_run:
        run_chain.write_jsonl(path, rows)
    print(f"[incremental:probe answers] {len(rows)} rows; {len(jobs)} generated")
    return rows


def annotate_future_query_regret(rows: list[dict]) -> None:
    baselines = {
        (row["qid"], row["probe_id"]): row
        for row in rows if row["source"] == "original_evidence"
    }
    for row in rows:
        if row["source"] != "handoff" or not row.get("final_handoff"):
            continue
        baseline = baselines[(row["qid"], row["probe_id"])]
        for metric in ("em", "f1", "judge_correct"):
            if metric in baseline and metric in row:
                row[f"original_evidence_{metric}"] = baseline[metric]
                row[f"future_query_regret_{metric}"] = float(baseline[metric]) - float(row[metric])


def fact_survival_rows(questions, packet_map, args, handoff_rows) -> list[dict]:
    rows = []
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            for seed in args.seeds:
                for question in questions:
                    _, packets = packet_map[question.qid]
                    schedule = inc_data.build_schedule(packets, relay_depth)
                    intro = inc_data.introduction_stages(schedule)
                    packet_indices = {packet.packet_id: index for index, packet in enumerate(packets, start=1)}
                    for step in schedule:
                        handoff_row = handoff_rows[(condition, relay_depth, seed, question.qid, step.stage)]
                        text = handoff_row["text"]
                        for packet in packets:
                            introduced_at = intro[packet.packet_id]
                            if step.stage < introduced_at:
                                continue
                            for probe in packet.probes:
                                rows.append({
                                    "dataset": "musique", "condition": condition,
                                    "relay_depth": relay_depth, "depth": relay_depth,
                                    "seed": seed, "qid": question.qid,
                                    "stage": step.stage, "agent_type": step.agent_type,
                                    "probe_id": probe.probe_id, "probe_step": probe.step,
                                    "packet_id": packet.packet_id,
                                    "packet_index": packet_indices[packet.packet_id],
                                    "support_pid": packet.support_pid,
                                    "introduction_stage": introduced_at,
                                    "handoff_age": inc_data.handoff_age(step.stage, introduced_at),
                                    "answer_string_survival": float(mentions_answer(text, list(probe.golds))),
                                    "source_topic_survival": float(mentions_topic(text, packet.title)),
                                    "handoff_characters": len(text),
                                })
    return rows


def question_level_vector(rows: list[dict], metric: str) -> tuple[list[str], np.ndarray]:
    by_qid: dict[str, list[float]] = {}
    for row in rows:
        by_qid.setdefault(row["qid"], []).append(float(row[metric]))
    qids = sorted(by_qid)
    return qids, np.array([np.mean(by_qid[qid]) for qid in qids], dtype=float)


def metric_interval(values: np.ndarray, boot_n: int, ci: float, seed: int) -> tuple[float, float, float]:
    return bootstrap_ci(values, boot_n, ci, seed=seed)


def analyse(
    main_rows: list[dict],
    probe_rows: list[dict],
    survival_rows: list[dict],
    experiment_cfg: dict,
    args,
    output_root: Path,
    ledger: dict,
) -> None:
    boot_n = int(experiment_cfg["analysis"]["bootstrap_resamples"])
    ci = float(experiment_cfg["analysis"]["ci_level"])
    output_root.mkdir(parents=True, exist_ok=True)

    original_main = [row for row in main_rows if row["source"] == "original_evidence"]
    _, original_f1 = question_level_vector(original_main, "f1")
    _, original_em = question_level_vector(original_main, "em")
    baseline = {"dataset": "musique", "condition": "original_evidence", "n": len(original_main)}
    for metric, vector in (("em", original_em), ("f1", original_f1)):
        mean, lo, hi = metric_interval(vector, boot_n, ci, 201)
        baseline.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
    if original_main and all("judge_correct" in row for row in original_main):
        _, vector = question_level_vector(original_main, "judge_correct")
        mean, lo, hi = metric_interval(vector, boot_n, ci, 201)
        baseline.update({"judge_correct": round(mean, 4), "judge_correct_lo": round(lo, 4),
                         "judge_correct_hi": round(hi, 4)})

    stage_metrics, vectors = [], {}
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            subset = [row for row in main_rows if row["source"] == "final_handoff"
                      and row["condition"] == condition and int(row["relay_depth"]) == relay_depth]
            record = {
                "dataset": "musique", "context_variant": condition,
                "condition": condition, "depth": relay_depth,
                "relay_depth": relay_depth,
            }
            qids, em = question_level_vector(subset, "em")
            _, f1 = question_level_vector(subset, "f1")
            record["n"] = len(qids)
            vectors[(condition, relay_depth)] = {
                "qids": qids,
                "em": dict(zip(qids, em)), "f1": dict(zip(qids, f1)),
            }
            for metric, vector in (("em", em), ("f1", f1)):
                mean, lo, hi = metric_interval(vector, boot_n, ci, 203)
                record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4),
                               f"{metric}_hi": round(hi, 4)})
            if subset and all("judge_correct" in row for row in subset):
                _, vector = question_level_vector(subset, "judge_correct")
                vectors[(condition, relay_depth)]["judge_correct"] = dict(zip(qids, vector))
                mean, lo, hi = metric_interval(vector, boot_n, ci, 203)
                record.update({"judge_correct": round(mean, 4),
                               "judge_correct_lo": round(lo, 4), "judge_correct_hi": round(hi, 4)})
            record["em_f1_disagree"] = round(em_f1_disagreement(em, f1), 4)
            record["packet_count_mean"] = round(float(np.mean([row["packet_count"] for row in subset])), 2)
            record["stages_mean"] = round(float(np.mean([row["stage"] for row in subset])), 2)
            record["handoff_characters_mean"] = round(float(np.mean([
                row["final_handoff_characters"] for row in subset
            ])), 1)
            record["chain_tokens_mean"] = round(float(np.mean([
                row["chain_prompt_tokens"] + row["chain_completion_tokens"] for row in subset
            ])), 1)
            stage_metrics.append(record)

    relay_deltas = []
    for condition in args.conditions:
        base = vectors[(condition, 0)]
        for relay_depth in [depth for depth in args.relay_depths if depth > 0]:
            current = vectors[(condition, relay_depth)]
            ids = sorted(set(base["qids"]) & set(current["qids"]))
            metrics = ["em", "f1"]
            if "judge_correct" in base and "judge_correct" in current:
                metrics.append("judge_correct")
            for metric in metrics:
                result = paired_bootstrap_delta(
                    np.array([current[metric][qid] for qid in ids]),
                    np.array([base[metric][qid] for qid in ids]),
                    boot_n, ci, seed=211,
                )
                relay_deltas.append({
                    "comparison": "relay_depth_minus_r0", "condition": condition,
                    "relay_depth": relay_depth, "depth": relay_depth, "metric": metric, **result,
                })

    probe_metrics = []
    ages = sorted({int(row["handoff_age"]) for row in probe_rows if row["source"] == "handoff"})
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            for age in ages:
                subset = [row for row in probe_rows if row["source"] == "handoff"
                          and row["condition"] == condition and int(row["relay_depth"]) == relay_depth
                          and int(row["handoff_age"]) == age]
                if not subset:
                    continue
                record = {"condition": condition, "relay_depth": relay_depth,
                          "depth": relay_depth, "handoff_age": age}
                for metric in ("em", "f1", "judge_correct"):
                    if not all(metric in row for row in subset):
                        continue
                    qids, vector = question_level_vector(subset, metric)
                    mean, lo, hi = metric_interval(vector, boot_n, ci, 223)
                    record.update({"n": len(qids), metric: round(mean, 4),
                                   f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
                probe_metrics.append(record)

    regret_metrics = []
    final_probes = [row for row in probe_rows if row["source"] == "handoff" and row.get("final_handoff")]
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            subset = [row for row in final_probes if row["condition"] == condition
                      and int(row["relay_depth"]) == relay_depth]
            record = {"condition": condition, "relay_depth": relay_depth, "depth": relay_depth}
            for metric in ("em", "f1", "judge_correct"):
                field = f"future_query_regret_{metric}"
                if not all(field in row for row in subset):
                    continue
                qids, vector = question_level_vector(subset, field)
                mean, lo, hi = metric_interval(vector, boot_n, ci, 227)
                record.update({"n": len(qids), field: round(mean, 4),
                               f"{field}_lo": round(lo, 4), f"{field}_hi": round(hi, 4)})
            regret_metrics.append(record)

    survival_metrics = []
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            for age in sorted({int(row["handoff_age"]) for row in survival_rows}):
                subset = [row for row in survival_rows if row["condition"] == condition
                          and int(row["relay_depth"]) == relay_depth and int(row["handoff_age"]) == age]
                if not subset:
                    continue
                record = {"condition": condition, "relay_depth": relay_depth,
                          "depth": relay_depth, "handoff_age": age}
                for metric in ("answer_string_survival", "source_topic_survival"):
                    qids, vector = question_level_vector(subset, metric)
                    mean, lo, hi = metric_interval(vector, boot_n, ci, 229)
                    record.update({"n": len(qids), metric: round(mean, 4),
                                   f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
                survival_metrics.append(record)

    run_chain.write_csv(output_root / "baseline_metrics.csv", [baseline])
    run_chain.write_csv(output_root / "stage_metrics.csv", stage_metrics)
    run_chain.write_csv(output_root / "relay_depth_deltas.csv", relay_deltas)
    run_chain.write_csv(output_root / "probe_metrics.csv", probe_metrics)
    run_chain.write_csv(output_root / "future_query_regret.csv", regret_metrics)
    run_chain.write_csv(output_root / "survival_by_age.csv", survival_metrics)
    run_chain.write_jsonl(output_root / "survival.jsonl", survival_rows)
    make_plot(stage_metrics, probe_metrics, regret_metrics, survival_metrics, baseline, args,
              output_root / "incremental_chain.png")
    write_report(stage_metrics, regret_metrics, baseline, args, experiment_cfg, ledger,
                 output_root / "report.md")


def make_plot(stage_metrics, probe_metrics, regret_metrics, survival_metrics, baseline, args, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {depth: color for depth, color in zip(args.relay_depths, plt.cm.viridis(np.linspace(0, .85, len(args.relay_depths))))}
    linestyles = {condition: style for condition, style in zip(args.conditions, ("-", "--", ":", "-."))}

    ax = axes[0, 0]
    for condition in args.conditions:
        subset = sorted([row for row in stage_metrics if row["condition"] == condition], key=lambda row: row["relay_depth"])
        ax.errorbar([row["relay_depth"] for row in subset], [row["f1"] for row in subset],
                    yerr=[[row["f1"] - row["f1_lo"] for row in subset],
                          [row["f1_hi"] - row["f1"] for row in subset]],
                    marker="o", capsize=3, label=condition)
    ax.axhline(baseline["f1"], color="#777777", ls=":", label="original evidence")
    ax.set(title="Final multi-hop QA", xlabel="Relay-only agents between specialists", ylabel="Token F1")
    ax.set_xticks(args.relay_depths)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for condition in args.conditions:
        subset = sorted([row for row in regret_metrics if row["condition"] == condition], key=lambda row: row["relay_depth"])
        if not subset or "future_query_regret_f1" not in subset[0]:
            continue
        ax.errorbar([row["relay_depth"] for row in subset], [row["future_query_regret_f1"] for row in subset],
                    yerr=[[row["future_query_regret_f1"] - row["future_query_regret_f1_lo"] for row in subset],
                          [row["future_query_regret_f1_hi"] - row["future_query_regret_f1"] for row in subset]],
                    marker="o", capsize=3, label=condition)
    ax.axhline(0, color="#777777", ls=":")
    ax.set(title="Future-query regret at final handoff", xlabel="Relay-only agents between specialists",
           ylabel="Original-evidence F1 − handoff F1")
    ax.set_xticks(args.relay_depths)
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            subset = sorted([row for row in probe_metrics if row["condition"] == condition
                             and row["relay_depth"] == relay_depth], key=lambda row: row["handoff_age"])
            if not subset:
                continue
            ax.plot([row["handoff_age"] for row in subset], [row["f1"] for row in subset],
                    color=colors[relay_depth], ls=linestyles[condition], marker=".",
                    label=f"{condition}, r={relay_depth}")
    ax.set(title="Hidden-probe accuracy by fact age", xlabel="Transformations since introduction",
           ylabel="Probe token F1", ylim=(-.02, 1.02))
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    for condition in args.conditions:
        for relay_depth in args.relay_depths:
            subset = sorted([row for row in survival_metrics if row["condition"] == condition
                             and row["relay_depth"] == relay_depth], key=lambda row: row["handoff_age"])
            if not subset:
                continue
            ax.plot([row["handoff_age"] for row in subset], [row["answer_string_survival"] for row in subset],
                    color=colors[relay_depth], ls=linestyles[condition], marker=".",
                    label=f"{condition}, r={relay_depth}")
    ax.set(title="Answer-fact survival in message", xlabel="Transformations since introduction",
           ylabel="Exact answer-string survival", ylim=(-.02, 1.02))
    ax.grid(alpha=.25)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    fig.suptitle("Incremental evidence: acquisition separated from relay-only transformation")
    fig.tight_layout(rect=(0, .07, 1, .97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(stage_metrics, regret_metrics, baseline, args, cfg, ledger, path: Path) -> None:
    lines = [
        "# Incremental-evidence handoff results", "",
        "MuSiQue supporting paragraphs arrive one packet at a time. Relay-only agents receive no new evidence.", "",
        f"Conditions: {', '.join(args.conditions)}  ",
        f"Relay-only agents between specialists: {', '.join(map(str, args.relay_depths))}  ",
        f"Seeds: {', '.join(map(str, args.seeds))}  ",
        f"Evidence order: {cfg.get('evidence_order', {}).get('mode', 'counterbalanced')}", "",
        "## Final multi-hop answer", "",
        f"Original-evidence baseline: EM {baseline['em']:.3f}, F1 {baseline['f1']:.3f}", "",
        "| condition | relays between specialists | n | stages | EM | F1 | judge | handoff chars | chain tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_metrics:
        lines.append(
            f"| {row['condition']} | {row['relay_depth']} | {row['n']} | {row['stages_mean']:.1f} | "
            f"{row['em']:.3f} | {row['f1']:.3f} | {row.get('judge_correct', float('nan')):.3f} | "
            f"{row['handoff_characters_mean']:.0f} | {row['chain_tokens_mean']:.0f} |"
        )
    lines += ["", "## Future-query regret", "",
              "Positive values mean the final handoff answers hidden packet probes worse than the original evidence.", "",
              "| condition | relays between specialists | n | EM regret | F1 regret | judge regret |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in regret_metrics:
        lines.append(
            f"| {row['condition']} | {row['relay_depth']} | {row.get('n', 0)} | "
            f"{row.get('future_query_regret_em', float('nan')):.3f} | "
            f"{row.get('future_query_regret_f1', float('nan')):.3f} | "
            f"{row.get('future_query_regret_judge_correct', float('nan')):.3f} |"
        )
    lines += ["", "## Outputs", "",
              "- `stage_metrics.csv`: existing-compatible final-answer metrics keyed by relay depth.",
              "- `relay_depth_deltas.csv`: paired relay-depth minus no-relay contrasts.",
              "- `probe_metrics.csv`: hidden-probe accuracy by handoff age.",
              "- `future_query_regret.csv`: original-evidence minus final-handoff probe scores.",
              "- `survival_by_age.csv` and `survival.jsonl`: direct fact survival by age.",
              "- `incremental_chain.png`: combined QA, regret, probe, and fact-survival plot.",
              "", "## Cost", "",
              f"- live spend: ${ledger.get('cost_usd', 0.0):.4f}",
              f"- calls: {ledger.get('calls_live', 0)} live, {ledger.get('calls_cached', 0)} cached",
              f"- prompt/completion tokens: {ledger.get('prompt_tokens', 0)} / {ledger.get('completion_tokens', 0)}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_args(parser, args, cfg) -> None:
    unknown_conditions = set(args.conditions) - set(cfg["conditions"])
    if unknown_conditions:
        parser.error(f"unknown conditions: {sorted(unknown_conditions)}")
    if not args.relay_depths or min(args.relay_depths) < 0 or 0 not in args.relay_depths:
        parser.error("relay depths must be non-negative and include 0 as the no-relay control")
    if not args.seeds:
        parser.error("at least one seed is required")
    output_paths = {cfg["outputs"][name] for name in ("root", "run_root", "data_root")}
    if len(output_paths) != 3:
        parser.error("result, run, and data roots must be distinct")


def run_manifest(base_cfg: dict, experiment_cfg: dict) -> dict:
    """Parameters that make persisted run rows safe to reuse."""

    return {
        "experiment": "incremental_evidence_chain_v1",
        "model": base_cfg["model"],
        "decoding": base_cfg["decoding"],
        "dataset": experiment_cfg["datasets"]["musique"],
        "sampling_seed": experiment_cfg["sampling"]["seed"],
        "packets": experiment_cfg.get("packets", {}),
        "evidence_order": experiment_cfg.get("evidence_order", {}),
        "conditions": experiment_cfg["conditions"],
        "configured_relay_depths": experiment_cfg["relay_depths"],
        "specialist_system": run_chain.QUESTION_CONDITIONED_SYSTEM,
        "specialist_instruction": SPECIALIST_INSTRUCTION,
        "relay_instruction": run_chain.QUESTION_CONDITIONED_RECOMPRESS_INSTRUCTION,
    }


def validate_run_manifest(run_root: Path, manifest: dict, force: bool, dry_run: bool) -> None:
    path = run_root / "manifest.json"
    legacy_rows = run_root.exists() and any(run_root.glob("*.jsonl"))
    if legacy_rows and not path.exists() and not force:
        raise RuntimeError(
            f"incremental run rows exist without a manifest in {run_root}. Use --force to "
            "regenerate only this experiment's rows."
        )
    if path.exists() and not force:
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != manifest:
            raise RuntimeError(
                f"incremental run manifest changed: {path}. Use --force to regenerate only "
                "this experiment's rows, or choose new output roots."
            )
    if not dry_run and (force or not path.exists()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental-evidence MuSiQue chain with relay-only depth controls",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"),
                        help="shared model, decoding, cache, cost, and judge config")
    parser.add_argument("--experiment-config", default=str(ROOT / "incremental_chain_config.yaml"))
    parser.add_argument("--relay-depths", help="comma-separated relay counts, e.g. 0,1,3,5")
    parser.add_argument("--conditions", help="comma-separated configured question-visibility conditions")
    parser.add_argument("--seeds", help="comma-separated handoff seeds")
    parser.add_argument("--n", type=int, help="number of C1-filtered MuSiQue examples")
    parser.add_argument("--candidates", type=int, help="candidate count before C1 filtering")
    parser.add_argument("--dry-run", action="store_true", help="estimate calls/cost without writing outputs")
    parser.add_argument("--force", action="store_true", help="regenerate this experiment's cached run records")
    parser.add_argument("--analyse-only", action="store_true", help="rescore/analyse existing run records only")
    args = parser.parse_args()
    cfg = load_config(args.experiment_config)
    args.relay_depths = (run_chain.parse_csv_arg(args.relay_depths, int) if args.relay_depths
                         else list(cfg["relay_depths"]))
    args.conditions = (run_chain.parse_csv_arg(args.conditions) if args.conditions
                       else list(cfg["conditions"]))
    args.seeds = run_chain.parse_csv_arg(args.seeds, int) if args.seeds else list(cfg["seeds"])
    args.n = args.n or int(cfg["sampling"]["n_target_per_dataset"])
    args.candidates = args.candidates or int(cfg["sampling"]["n_candidates_per_dataset"])
    validate_args(parser, args, cfg)
    return args, cfg


def main() -> int:
    args, experiment_cfg = parse_args()
    base_cfg = load_config(args.config)
    output_root = ROOT / experiment_cfg["outputs"]["root"]
    run_root = ROOT / experiment_cfg["outputs"]["run_root"]
    validate_run_manifest(run_root, run_manifest(base_cfg, experiment_cfg), args.force, args.dry_run)

    if args.analyse_only:
        main_rows = run_chain.read_jsonl(run_root / "answers.jsonl")
        probe_rows = run_chain.read_jsonl(run_root / "probe_answers.jsonl")
        survival = run_chain.read_jsonl(run_root / "survival.jsonl")
        if not main_rows or not probe_rows or not survival:
            raise RuntimeError("analyse-only requires answers.jsonl, probe_answers.jsonl, and survival.jsonl")
        judge_client_main = add_judge(main_rows, base_cfg, tag="incremental_main_judge")
        judge_client_probe = add_judge(probe_rows, base_cfg, tag="incremental_probe_judge")
        annotate_future_query_regret(probe_rows)
        run_chain.write_jsonl(run_root / "answers.jsonl", main_rows)
        run_chain.write_jsonl(run_root / "probe_answers.jsonl", probe_rows)
        analyse(main_rows, probe_rows, survival, experiment_cfg, args, output_root, {
            "cost_usd": 0.0, "calls_live": 0, "calls_cached": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
        })
        for client in (judge_client_main, judge_client_probe):
            if client is not None:
                print("[incremental:judge cost] " + json.dumps(client.ledger.summary()))
        return 0

    client = LLMClient(base_cfg, dry_run=args.dry_run)
    print(
        f"[incremental] relay_depths={args.relay_depths} conditions={args.conditions} "
        f"seeds={args.seeds} n={args.n} candidates={args.candidates}",
    )
    try:
        questions, packet_map = prepare_examples(client, base_cfg, experiment_cfg, args)
        handoffs, handoff_rows = generate_handoffs(
            client, base_cfg, experiment_cfg, args, questions, packet_map, run_root,
        )
        main_rows = run_main_answers(
            client, base_cfg, args, questions, packet_map, handoffs, handoff_rows, run_root,
        )
        probe_rows = run_probe_answers(
            client, base_cfg, args, questions, packet_map, handoffs, handoff_rows, run_root,
        )
        survival = fact_survival_rows(questions, packet_map, args, handoff_rows)
        if args.dry_run:
            print(json.dumps(client.dry_run_report(), indent=2))
            return 0
        run_chain.write_jsonl(run_root / "survival.jsonl", survival)
        judge_client_main = add_judge(main_rows, base_cfg, tag="incremental_main_judge")
        judge_client_probe = add_judge(probe_rows, base_cfg, tag="incremental_probe_judge")
        annotate_future_query_regret(probe_rows)
        run_chain.write_jsonl(run_root / "answers.jsonl", main_rows)
        run_chain.write_jsonl(run_root / "probe_answers.jsonl", probe_rows)
        analyse(main_rows, probe_rows, survival, experiment_cfg, args, output_root,
                client.ledger.summary())
        for judge_client in (judge_client_main, judge_client_probe):
            if judge_client is not None:
                print("[incremental:judge cost] " + json.dumps(judge_client.ledger.summary()))
    except CostCapExceeded as exc:
        print(f"[incremental:ABORT] {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
