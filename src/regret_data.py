"""Multi-query source contexts and the question rotation for Experiment 10.

One *context* is a source document that independently answers several questions.
Every question takes a turn as the currently known conditioning query while the
rest stay hidden from the sender, which is what removes question difficulty as
a confound: the same question contributes to both the diagonal (present-query)
and the off-diagonal (future-query) cells of the utility matrix.

Two corpora share this loader:

``squad_groups``
    Human-written SQuAD questions grouped by their shared paragraph.  Every
    question in a group has passed the project's closed-book leakage filter
    against the answering model, so an off-diagonal success cannot come from
    parametric knowledge.

``relation_dossiers``
    Purpose-built fictional dossiers whose hidden questions are labelled by
    their *relation* to the conditioning question - paraphrase, same entity,
    same topic, orthogonal aspect.  SQuAD supports no such clean labelling, so
    the relation analysis runs only here rather than forcing noisy categories
    onto natural data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Roles a relation-dossier question can hold within its aspect. ``anchor`` is
# the only role that ever rotates into the conditioning slot; the others exist
# purely to sit at a known distance from it.
ANCHOR = "anchor"
PARAPHRASE = "paraphrase"
SAME_ENTITY = "same_entity"
SAME_TOPIC = "same_topic"
ORTHOGONAL = "orthogonal"
UNLABELLED = "unlabelled"

RELATION_ORDER = (PARAPHRASE, SAME_ENTITY, SAME_TOPIC, ORTHOGONAL)


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RegretQuestion:
    qid: str
    question: str
    golds: tuple[str, ...]
    aspect: str = ""
    role: str = ""

    def as_dict(self) -> dict:
        return {"qid": self.qid, "question": self.question, "golds": list(self.golds),
                "aspect": self.aspect, "role": self.role}


@dataclass(frozen=True)
class RegretContext:
    context_id: str
    dataset: str
    title: str
    source: str
    questions: tuple[RegretQuestion, ...]
    content_hash: str
    meta: dict

    @property
    def question_map(self) -> dict[str, RegretQuestion]:
        return {q.qid: q for q in self.questions}

    def question(self, qid: str) -> RegretQuestion:
        try:
            return self.question_map[qid]
        except KeyError as exc:
            raise KeyError(f"{qid!r} is not a question of context {self.context_id!r}") from exc

    @property
    def all_question_texts(self) -> tuple[str, ...]:
        return tuple(q.question for q in self.questions)

    @property
    def source_words(self) -> int:
        return len(self.source.split())


@dataclass(frozen=True)
class Rotation:
    """One assignment of the conditioning role.

    ``hidden_qids`` are the questions the sender was never shown; they are the
    off-diagonal of this context's utility matrix.
    """

    rotation_id: str
    context_id: str
    current_qid: str
    hidden_qids: tuple[str, ...]

    @property
    def evaluated_qids(self) -> tuple[str, ...]:
        return (self.current_qid,) + self.hidden_qids


# ---------------------------------------------------------------------------
# Loading


def _question(raw: dict) -> RegretQuestion:
    golds = tuple(str(g) for g in (raw.get("golds") or []) if str(g).strip())
    if not golds:
        raise ValueError(f"question {raw.get('qid')!r} has no gold answers")
    return RegretQuestion(
        qid=str(raw["qid"]),
        question=str(raw["question"]).strip(),
        golds=golds,
        aspect=str(raw.get("aspect") or ""),
        role=str(raw.get("role") or ""),
    )


def _context(raw: dict) -> RegretContext:
    questions = tuple(_question(q) for q in raw["questions"])
    if len({q.qid for q in questions}) != len(questions):
        raise ValueError(f"context {raw['context_id']!r} has duplicate question ids")
    source = str(raw["source"]).strip()
    if not source:
        raise ValueError(f"context {raw['context_id']!r} has an empty source")
    return RegretContext(
        context_id=str(raw["context_id"]),
        dataset=str(raw.get("dataset") or "unknown"),
        title=str(raw.get("title") or ""),
        source=source,
        questions=questions,
        content_hash=_hash({"source": source, "questions": [q.as_dict() for q in questions]}),
        meta=dict(raw.get("meta") or {}),
    )


def load_contexts(path: str | Path) -> tuple[dict, tuple[RegretContext, ...]]:
    """Read a corpus file. The first line is the build manifest, as elsewhere."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it first with src/builders/build_regret_data.py.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    manifest = rows[0].get("_manifest")
    if manifest is None:
        raise ValueError(f"{path} is missing its leading _manifest line")
    return manifest, tuple(_context(row) for row in rows[1:])


# ---------------------------------------------------------------------------
# Rotation


def rotations_for(context: RegretContext) -> tuple[Rotation, ...]:
    """Every question that may hold the conditioning role, in file order.

    For SQuAD groups that is every question, so each one appears once on the
    diagonal and k-1 times off it.  For relation dossiers only the per-aspect
    anchors rotate; the paraphrase/same-entity/same-topic questions exist to
    populate known distances from an anchor and never condition anything.
    """
    if context.dataset == "relation_dossiers":
        conditioning = [q for q in context.questions if q.role == ANCHOR]
    else:
        conditioning = list(context.questions)
    if len(conditioning) < 2:
        raise ValueError(
            f"context {context.context_id!r} has fewer than two conditioning questions; "
            "a rotation needs at least one hidden query")
    out = []
    for q in conditioning:
        hidden = tuple(other.qid for other in context.questions if other.qid != q.qid)
        out.append(Rotation(
            rotation_id=f"{context.context_id}|cond={q.qid}",
            context_id=context.context_id,
            current_qid=q.qid,
            hidden_qids=hidden,
        ))
    return tuple(out)


def relation_label(context: RegretContext, current_qid: str, hidden_qid: str) -> str:
    """How far the hidden information need sits from the conditioning one.

    Labels are read off the build-time design, never inferred from embeddings:
    the point of the synthetic corpus is that the distance is a controlled
    variable rather than an estimate.
    """
    if context.dataset != "relation_dossiers":
        return UNLABELLED
    current = context.question(current_qid)
    hidden = context.question(hidden_qid)
    if current.role != ANCHOR:
        raise ValueError(f"{current_qid!r} is not an anchor and cannot condition a rotation")
    if hidden.aspect == current.aspect:
        if hidden.role in (PARAPHRASE, SAME_ENTITY, SAME_TOPIC):
            return hidden.role
        raise ValueError(
            f"{hidden_qid!r} shares aspect {current.aspect!r} with the anchor but has "
            f"role {hidden.role!r}; the corpus design allows only one anchor per aspect")
    return ORTHOGONAL


# ---------------------------------------------------------------------------
# Structural validation, shared by the builder and the runner


def validate_relation_context(context: RegretContext) -> None:
    """Assert the designed relation structure actually holds.

    Every aspect must carry exactly one anchor plus one question at each of the
    three within-aspect distances, and there must be at least two aspects so an
    orthogonal distance exists at all.  A corpus that quietly lost a role would
    otherwise produce a distance curve estimated from unequal cells.
    """
    by_aspect: dict[str, list[RegretQuestion]] = {}
    for q in context.questions:
        if not q.aspect:
            raise ValueError(f"{context.context_id}: question {q.qid!r} has no aspect")
        by_aspect.setdefault(q.aspect, []).append(q)
    if len(by_aspect) < 2:
        raise ValueError(f"{context.context_id}: needs at least two aspects")
    expected = {ANCHOR, PARAPHRASE, SAME_ENTITY, SAME_TOPIC}
    for aspect, group in sorted(by_aspect.items()):
        roles = sorted(q.role for q in group)
        if sorted(expected) != roles:
            raise ValueError(
                f"{context.context_id}/{aspect}: roles {roles} != {sorted(expected)}")
    anchors = [q for q in context.questions if q.role == ANCHOR]
    for anchor in anchors:
        para = next(q for q in by_aspect[anchor.aspect] if q.role == PARAPHRASE)
        if set(map(str.lower, para.golds)) != set(map(str.lower, anchor.golds)):
            raise ValueError(
                f"{context.context_id}/{anchor.aspect}: the paraphrase must share the "
                "anchor's gold answers, otherwise it is not a paraphrase")


def coverage_table(contexts: Sequence[RegretContext]) -> dict[str, int]:
    """Rotation and evaluation counts, used by --dry-run and the manifest."""
    rotations = sum(len(rotations_for(c)) for c in contexts)
    evaluations = sum(len(r.evaluated_qids) for c in contexts for r in rotations_for(c))
    return {
        "contexts": len(contexts),
        "questions": sum(len(c.questions) for c in contexts),
        "rotations": rotations,
        "rotation_evaluations": evaluations,
    }


def iter_rotations(contexts: Iterable[RegretContext]):
    for context in contexts:
        for rotation in rotations_for(context):
            yield context, rotation
