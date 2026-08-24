"""MuSiQue evidence packets and schedules for incremental handoff chains.

The existing :class:`data.Question` remains the source-of-truth record.  This
module projects its decomposition into immutable packet/probe objects without
changing the shared dataset schema used by the original chain experiments.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass

from data import Question


@dataclass(frozen=True)
class Probe:
    probe_id: str
    step: int
    support_pid: int
    question_template: str
    question: str
    answer: str
    golds: tuple[str, ...]

    def to_json(self) -> dict:
        row = asdict(self)
        row["golds"] = list(self.golds)
        return row


@dataclass(frozen=True)
class EvidencePacket:
    packet_id: str
    support_pid: int
    title: str
    text: str
    decomposition_position: int
    probes: tuple[Probe, ...]

    def to_json(self) -> dict:
        row = asdict(self)
        row["probes"] = [probe.to_json() for probe in self.probes]
        return row


@dataclass(frozen=True)
class ChainStep:
    stage: int
    agent_type: str
    packet_index: int | None
    packet_id: str | None
    relay_index: int | None


_REFERENCE = re.compile(r"#(\d+)")


def resolve_probe_question(template: str, decomposition: list[dict]) -> str:
    """Resolve MuSiQue ``#1`` references into standalone hidden probes."""

    answers = {int(item["step"]) + 1: str(item["answer"]) for item in decomposition}

    def replace(match: re.Match) -> str:
        index = int(match.group(1))
        return answers.get(index, match.group(0))

    resolved = " ".join(_REFERENCE.sub(replace, str(template)).split())
    if ">>" in resolved:
        subject, relation = (part.strip(" ,") for part in resolved.split(">>", 1))
        return f'For "{subject}", what is the value of the relation "{relation}"?'
    return resolved if resolved.endswith("?") else resolved.rstrip(" .") + "?"


def build_packets(question: Question) -> list[EvidencePacket]:
    """Create one packet per distinct supporting paragraph in hop order.

    Multiple decomposition steps may use the same paragraph; those steps become
    multiple probes attached to one packet, and the paragraph is introduced
    exactly once.
    """

    if not isinstance(question, Question):
        raise TypeError(f"build_packets requires Question, got {type(question).__name__}")
    if not question.decomposition:
        raise ValueError(f"{question.qid} has no MuSiQue decomposition")

    paragraphs = {paragraph.pid: paragraph for paragraph in question.paragraphs}
    pid_order: list[int] = []
    probes_by_pid: dict[int, list[Probe]] = {}
    for item in sorted(question.decomposition, key=lambda row: int(row["step"])):
        pid = int(item["support_pid"])
        if pid not in paragraphs:
            raise ValueError(f"{question.qid} decomposition references missing P{pid}")
        if pid not in probes_by_pid:
            pid_order.append(pid)
            probes_by_pid[pid] = []
        step = int(item["step"])
        answer = str(item["answer"]).strip()
        golds = tuple(question.golds) if step == len(question.decomposition) - 1 else (answer,)
        probes_by_pid[pid].append(Probe(
            probe_id=f"{question.qid}:step{step + 1}",
            step=step,
            support_pid=pid,
            question_template=str(item["question"]),
            question=resolve_probe_question(str(item["question"]), question.decomposition),
            answer=answer,
            golds=golds,
        ))

    packets = []
    for position, pid in enumerate(pid_order, start=1):
        paragraph = paragraphs[pid]
        packets.append(EvidencePacket(
            packet_id=f"{question.qid}:P{pid}",
            support_pid=pid,
            title=paragraph.title,
            text=paragraph.text,
            decomposition_position=position,
            probes=tuple(probes_by_pid[pid]),
        ))
    return packets


def order_packets(question: Question, mode: str, seed: int) -> tuple[str, list[EvidencePacket]]:
    """Return a deterministic packet order shared by every relay condition."""

    packets = build_packets(question)
    if mode == "forward":
        return "forward", packets
    if mode == "reverse":
        return "reverse", list(reversed(packets))
    if mode == "randomized":
        ordered = list(packets)
        random.Random(f"{seed}:{question.qid}:packet-order").shuffle(ordered)
        return "randomized", ordered
    if mode == "counterbalanced":
        digest = hashlib.sha256(f"{seed}:{question.qid}:packet-order".encode("utf-8")).digest()
        if digest[0] % 2:
            return "reverse", list(reversed(packets))
        return "forward", packets
    raise ValueError(f"unknown evidence-order mode: {mode}")


def build_schedule(packets: list[EvidencePacket], relay_depth: int) -> list[ChainStep]:
    """Insert exactly ``relay_depth`` no-evidence relays between specialists."""

    if relay_depth < 0:
        raise ValueError("relay_depth must be non-negative")
    if not packets:
        raise ValueError("at least one evidence packet is required")
    result: list[ChainStep] = []
    stage = 0
    for packet_index, packet in enumerate(packets, start=1):
        stage += 1
        result.append(ChainStep(stage, "specialist", packet_index, packet.packet_id, None))
        if packet_index < len(packets):
            for relay_index in range(1, relay_depth + 1):
                stage += 1
                result.append(ChainStep(stage, "relay", None, None, relay_index))
    return result


def introduction_stages(schedule: list[ChainStep]) -> dict[str, int]:
    return {
        step.packet_id: step.stage
        for step in schedule
        if step.agent_type == "specialist" and step.packet_id is not None
    }


def handoff_age(stage: int, introduced_at: int) -> int:
    """Transformations since introduction; the introducing output has age 0."""

    age = int(stage) - int(introduced_at)
    if age < 0:
        raise ValueError("a fact cannot be evaluated before it is introduced")
    return age


def render_packet(packet: EvidencePacket, packet_index: int) -> str:
    """Render only source evidence; hidden probes never enter model prompts."""

    return f"[E{packet_index}; source P{packet.support_pid}] {packet.title}\n{packet.text}"


def render_original_evidence(packets: list[EvidencePacket]) -> str:
    return "\n\n".join(render_packet(packet, index) for index, packet in enumerate(packets, start=1))


def packet_manifest_row(question: Question, order_id: str, packets: list[EvidencePacket]) -> dict:
    return {
        "qid": question.qid,
        "question": question.question,
        "answer": question.answer,
        "golds": list(question.golds),
        "evidence_order": order_id,
        "packet_count": len(packets),
        "packets": [packet.to_json() for packet in packets],
    }
