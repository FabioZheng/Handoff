"""Offline adapters for multi-question QA over Experiment 9's fictional corpus.

The source file was already generated and closed-book checked by Experiment 9.
This module never calls an API and never writes data.  It projects each existing
item into six question/evidence cards, supplies deterministic role rotations,
and implements the fixed-capacity selection interface shared by Experiments 5b
and 8b.

Questions are metadata, not source material.  ``render_source_cards`` exposes
only card ids and evidence text; prompt builders add at most the one announced
question.  Later relays accept only :class:`handoffs.SealedHandoff`, preserving
the repository's source-isolation boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import handoffs as hm
from data import split_sentences


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "size_adaptation" / "fictional_items.jsonl"
DEFAULT_CARD_ORDER_SEED = "fictional-qa-card-order-v1"
DEFAULT_ROTATION_SEED = "fictional-qa-role-rotation-v1"
SELECTION_KEY = "selected_fact_ids"

SELECTION_SYSTEM = (
    "You select evidence for another agent. Follow the requested JSON schema exactly and "
    "select only ids that occur in the supplied evidence cards."
)

SELECTION_INSTRUCTION = """Select exactly {k} evidence cards for a fixed-capacity handoff.
If a question is provided, prioritize evidence useful for that question. If no question is
provided, preserve broadly reusable evidence. Do not infer or add facts.

Return one JSON object and nothing else, with exactly this key:
{{"selected_fact_ids": [exactly {k} distinct bare ids such as "C03"]}}
Copy only the C-number inside each CARD label; never include the word CARD in a value."""

RELAY_INSTRUCTION = """Select exactly {k} evidence cards from the sealed handoff for the next
agent. Use only cards present in the sealed handoff; omitted source evidence is unavailable.
Return one JSON object and nothing else, with exactly this key:
{{"selected_fact_ids": [exactly {k} distinct bare ids such as "C03"]}}
Copy only the C-number inside each CARD label; never include the word CARD in a value."""


def _digest(*parts: object) -> str:
    return hashlib.sha256("\N{UNIT SEPARATOR}".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normal(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _contains(text: str, needle: str) -> bool:
    return bool(needle.strip()) and _normal(needle) in _normal(text)


@dataclass(frozen=True)
class QAFact:
    fact_id: str
    role: str
    statement: str
    question: str
    answer: str
    aliases: tuple[str, ...]
    golds: tuple[str, ...]
    probes: tuple[str, ...]
    evidence_text: str


@dataclass(frozen=True)
class EvidenceCard:
    """One question-independent evidence unit.

    ``question`` and ``role`` are retained for evaluation only.  Neither is
    emitted by either rendering function.
    """

    card_id: str
    fact_id: str
    evidence_text: str
    question: str
    golds: tuple[str, ...]
    role: str

    @property
    def text(self) -> str:
        return self.evidence_text


@dataclass(frozen=True)
class FictionalDossier:
    item_id: str
    title: str
    document: str
    facts: tuple[QAFact, ...]
    content_hash: str
    source_manifest: dict[str, Any]

    @property
    def dossier_id(self) -> str:
        return self.item_id

    @property
    def fact_map(self) -> dict[str, QAFact]:
        return {fact.fact_id: fact for fact in self.facts}

    def fact(self, fact_id: str) -> QAFact:
        try:
            return self.fact_map[fact_id]
        except KeyError as exc:
            raise KeyError(f"{fact_id!r} is not a fact in dossier {self.item_id!r}") from exc


@dataclass(frozen=True)
class RoleRotation:
    rotation_id: str
    dossier_id: str
    target_fact_id: str
    target_question: str
    target_golds: tuple[str, ...]
    hidden_fact_ids: tuple[str, ...]
    hidden_questions: tuple[str, ...]

    # Terminology aliases used in the paper design.
    @property
    def announced_fact_id(self) -> str:
        return self.target_fact_id

    @property
    def announced_question(self) -> str:
        return self.target_question

    @property
    def announced_golds(self) -> tuple[str, ...]:
        return self.target_golds


@dataclass(frozen=True)
class SelectionParse:
    """Strict parse plus non-operative diagnostics for malformed outputs.

    ``selected_fact_ids`` is populated only when ``valid`` is true.  Recognised
    ids from an invalid response remain in ``recognized_fact_ids`` for auditing,
    but callers must opt in explicitly if they want to salvage them.
    """

    valid: bool
    selected_fact_ids: tuple[str, ...]
    recognized_fact_ids: tuple[str, ...]
    requested_k: int
    parsed_count: int | None
    errors: tuple[str, ...]
    error: str


@dataclass(frozen=True)
class PacketIntervention:
    kind: str
    before_fact_ids: tuple[str, ...]
    after_fact_ids: tuple[str, ...]
    inserted_fact_id: str | None
    removed_fact_id: str | None
    changed: bool
    reason: str


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _DuplicateJSONKey(key)
        obj[key] = value
    return obj


def _source_sentence(document: str, golds: Sequence[str], fact_id: str) -> str:
    """Return the first source sentence containing a gold answer.

    The Experiment 9 target statement is deliberately generic (``The answer to
    the target question is ...``), so it cannot serve as source evidence.  The
    actual source sentence is deterministic and necessarily contains a gold.
    """

    candidates = []
    for index, sentence in enumerate(split_sentences(document)):
        matched = next((gold for gold in golds if _contains(sentence, gold)), None)
        if matched is not None:
            # Prefer the primary answer, then earlier document order.
            candidates.append((0 if _normal(matched) == _normal(golds[0]) else 1, index, sentence))
    if not candidates:
        raise ValueError(f"{fact_id}: no source sentence contains any gold answer")
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def _qa_fact(raw: dict[str, Any], document: str) -> QAFact:
    required = ("fact_id", "role", "statement", "question", "answer")
    missing = [key for key in required if not str(raw.get(key, "")).strip()]
    if missing:
        raise ValueError(f"fact is missing required fields: {', '.join(missing)}")
    fact_id = str(raw["fact_id"])
    answer = str(raw["answer"]).strip()
    aliases = tuple(dict.fromkeys(str(x).strip() for x in raw.get("aliases", []) if str(x).strip()))
    supplied_golds = tuple(str(x).strip() for x in raw.get("golds", []) if str(x).strip())
    golds = tuple(dict.fromkeys((answer,) + aliases + supplied_golds))
    probes = tuple(dict.fromkeys(str(x).strip() for x in raw.get("probes", []) if str(x).strip()))
    statement = str(raw["statement"]).strip()
    generic_target = _normal(statement).startswith("the answer to the target question is ")
    evidence = _source_sentence(document, golds, fact_id) if generic_target else statement
    if not any(_contains(evidence, gold) for gold in golds):
        raise ValueError(f"{fact_id}: evidence unit contains none of its gold answers")
    return QAFact(
        fact_id=fact_id,
        role=str(raw["role"]).strip(),
        statement=statement,
        question=str(raw["question"]).strip(),
        answer=answer,
        aliases=aliases,
        golds=golds,
        probes=probes or golds,
        evidence_text=evidence,
    )


def _dossier(raw: dict[str, Any], manifest: dict[str, Any]) -> FictionalDossier:
    item_id = str(raw.get("item_id", "")).strip()
    document = str(raw.get("document", "")).strip()
    if not item_id or not document:
        raise ValueError("fictional item requires non-empty item_id and document")
    raw_facts = raw.get("facts")
    if not isinstance(raw_facts, list) or len(raw_facts) != 6:
        raise ValueError(f"{item_id}: expected exactly six QA facts, got "
                         f"{len(raw_facts) if isinstance(raw_facts, list) else 'non-list'}")
    facts = tuple(_qa_fact(fact, document) for fact in raw_facts)
    ids = [fact.fact_id for fact in facts]
    if len(set(ids)) != 6:
        raise ValueError(f"{item_id}: fact ids are not unique")
    if sum(fact.role == "target" for fact in facts) != 1:
        raise ValueError(f"{item_id}: expected exactly one original target fact")
    if any(fact.role not in {"target", "side"} for fact in facts):
        raise ValueError(f"{item_id}: unsupported fact role")
    content_hash = _canonical_hash({
        "item_id": item_id,
        "title": raw.get("title", ""),
        "document": document,
        "facts": raw_facts,
    })
    return FictionalDossier(
        item_id=item_id,
        title=str(raw.get("title", "")).strip(),
        document=document,
        facts=facts,
        content_hash=content_hash,
        source_manifest=dict(manifest),
    )


def load_fictional_corpus(
    path: str | Path = DEFAULT_PATH,
) -> tuple[dict[str, Any], tuple[FictionalDossier, ...]]:
    """Load and strictly validate the manifest and body of the existing corpus."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"fictional QA source is missing: {source}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or set(rows[0]) != {"_manifest"} or not isinstance(rows[0]["_manifest"], dict):
        raise ValueError(f"{source}: first non-empty row must contain only _manifest")
    if any("_manifest" in row for row in rows[1:]):
        raise ValueError(f"{source}: manifest may occur only on the first row")
    manifest = dict(rows[0]["_manifest"])
    dossiers = tuple(_dossier(row, manifest) for row in rows[1:])
    if not dossiers:
        raise ValueError(f"{source}: no fictional dossiers")
    ids = [dossier.item_id for dossier in dossiers]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate dossier ids")
    all_facts = [fact.fact_id for dossier in dossiers for fact in dossier.facts]
    if len(all_facts) != len(set(all_facts)):
        raise ValueError(f"{source}: fact ids are not globally unique")
    return manifest, dossiers


def load_fictional_dossiers(
    path: str | Path = DEFAULT_PATH,
) -> tuple[FictionalDossier, ...]:
    """Convenience body-only loader."""

    return load_fictional_corpus(path)[1]


def evidence_cards(
    dossier: FictionalDossier,
    seed: str | int = DEFAULT_CARD_ORDER_SEED,
) -> tuple[EvidenceCard, ...]:
    """Return all six facts in a content-hashed, reproducible source order."""

    if not isinstance(dossier, FictionalDossier):
        raise TypeError("evidence_cards requires a FictionalDossier")
    # Model-facing ids must be opaque.  The source annotations use ids such as
    # ``fic:1:target`` and ``fic:1:side2``; exposing those labels would tell a
    # question-blind selector which fact the corpus originally treated as the
    # target.  Assign neutral ids only *after* the deterministic shuffle while
    # retaining ``fact_id`` for evaluator-side joins and interventions.
    ordered = sorted(dossier.facts, key=lambda fact: _digest(
        seed, dossier.content_hash, "card-order", fact.fact_id))
    return tuple(
        EvidenceCard(
            card_id=f"C{index:02d}",
            fact_id=fact.fact_id,
            evidence_text=fact.evidence_text,
            question=fact.question,
            golds=fact.golds,
            role=fact.role,
        )
        for index, fact in enumerate(ordered, start=1)
    )


def role_rotations(
    dossier: FictionalDossier,
    count: int = 6,
    seed: str | int = DEFAULT_ROTATION_SEED,
) -> tuple[RoleRotation, ...]:
    """Make four or six A/hidden-role rotations.

    Four-rotation runs always retain the corpus's original target plus three
    hash-selected side facts.  In every rotation *all five* non-target facts are
    hidden, including facts not used as announced targets in the four-arm run.
    """

    if not isinstance(dossier, FictionalDossier):
        raise TypeError("role_rotations requires a FictionalDossier")
    if count not in (4, 6):
        raise ValueError("role rotation count must be 4 or 6")
    ordered = sorted(dossier.facts, key=lambda fact: _digest(
        seed, dossier.content_hash, "rotation-order", fact.fact_id))
    if count == 4:
        original = next(fact for fact in dossier.facts if fact.role == "target")
        selected_ids = {original.fact_id}
        selected_ids.update(fact.fact_id for fact in ordered
                            if fact.role != "target" and len(selected_ids) < 4)
        announced = [fact for fact in ordered if fact.fact_id in selected_ids]
    else:
        announced = ordered
    fact_order = {fact.fact_id: index for index, fact in enumerate(ordered)}
    rotations = []
    for fact in announced:
        hidden = tuple(sorted(
            (other for other in dossier.facts if other.fact_id != fact.fact_id),
            key=lambda other: fact_order[other.fact_id],
        ))
        suffix = _digest(dossier.content_hash, seed, count, fact.fact_id)[:12]
        rotations.append(RoleRotation(
            rotation_id=f"{dossier.item_id}:A={fact.fact_id.rsplit(':', 1)[-1]}:{suffix}",
            dossier_id=dossier.item_id,
            target_fact_id=fact.fact_id,
            target_question=fact.question,
            target_golds=fact.golds,
            hidden_fact_ids=tuple(other.fact_id for other in hidden),
            hidden_questions=tuple(other.question for other in hidden),
        ))
    return tuple(rotations)


def render_source_cards(cards: Sequence[EvidenceCard]) -> str:
    """Render card ids and evidence only; questions and role labels stay hidden."""

    _validate_cards(cards)
    return "\n\n".join(f"[CARD {card.card_id}]\n{card.evidence_text}" for card in cards)


def question_block(question: RoleRotation | str) -> str:
    text = question.target_question if isinstance(question, RoleRotation) else question
    if not isinstance(text, str) or not text.strip():
        raise TypeError("question_block requires a RoleRotation or non-empty question string")
    return f"\n\n<ANNOUNCED_QUESTION>\n{text.strip()}\n</ANNOUNCED_QUESTION>"


def _validate_k(k: int, n_cards: int | None = None) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("K must be a positive integer")
    if n_cards is not None and k > n_cards:
        raise ValueError(f"K={k} exceeds the {n_cards} available cards")


def _validate_cards(cards: Sequence[EvidenceCard]) -> None:
    if not all(isinstance(card, EvidenceCard) for card in cards):
        raise TypeError("all cards must be EvidenceCard instances")
    ids = [card.card_id for card in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("card ids must be unique")
    fact_ids = [card.fact_id for card in cards]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("fact ids must be unique")
    if any(re.search(r"target|side", card.card_id, flags=re.I) for card in cards):
        raise ValueError("model-facing card ids must not expose evaluator roles")


def card_ids_to_fact_ids(
    cards: Sequence[EvidenceCard], card_ids: Iterable[str]
) -> tuple[str, ...]:
    """Translate model-facing opaque ids to evaluator-side fact ids."""

    _validate_cards(cards)
    by_card = {card.card_id: card.fact_id for card in cards}
    values = tuple(card_ids)
    if len(values) != len(set(values)):
        raise ValueError("card ids contain duplicates")
    unknown = [card_id for card_id in values if card_id not in by_card]
    if unknown:
        raise ValueError(f"unknown card ids: {unknown}")
    return tuple(by_card[card_id] for card_id in values)


def fact_ids_to_card_ids(
    cards: Sequence[EvidenceCard], fact_ids: Iterable[str]
) -> tuple[str, ...]:
    """Translate evaluator-side fact ids to model-facing opaque ids."""

    _validate_cards(cards)
    by_fact = {card.fact_id: card.card_id for card in cards}
    values = tuple(fact_ids)
    if len(values) != len(set(values)):
        raise ValueError("fact ids contain duplicates")
    unknown = [fact_id for fact_id in values if fact_id not in by_fact]
    if unknown:
        raise ValueError(f"unknown fact ids: {unknown}")
    return tuple(by_fact[fact_id] for fact_id in values)


def source_selection_prompt(
    dossier: FictionalDossier,
    cards: Sequence[EvidenceCard],
    k: int,
    question_fact_id: str | None = None,
) -> str:
    """Build a generic or one-question-conditioned source selection prompt."""

    if not isinstance(dossier, FictionalDossier):
        raise TypeError("source_selection_prompt requires a FictionalDossier")
    _validate_cards(cards)
    _validate_k(k, len(cards))
    if {card.fact_id for card in cards} != set(dossier.fact_map):
        raise ValueError("source cards must contain exactly the dossier's six facts")
    prompt = f"Source evidence cards:\n{render_source_cards(cards)}"
    if question_fact_id is not None:
        prompt += question_block(dossier.fact(question_fact_id).question)
    prompt += "\n\n" + SELECTION_INSTRUCTION.format(k=k)
    return prompt


def generic_selection_prompt(
    dossier: FictionalDossier,
    cards: Sequence[EvidenceCard],
    k: int,
) -> str:
    return source_selection_prompt(dossier, cards, k, question_fact_id=None)


def conditioned_selection_prompt(
    dossier: FictionalDossier,
    rotation: RoleRotation,
    cards: Sequence[EvidenceCard],
    k: int,
) -> str:
    if not isinstance(rotation, RoleRotation) or rotation.dossier_id != dossier.item_id:
        raise ValueError("rotation must belong to the supplied dossier")
    return source_selection_prompt(dossier, cards, k, rotation.target_fact_id)


def reopened_selection_prompt(
    dossier: FictionalDossier,
    question_fact_id: str,
    cards: Sequence[EvidenceCard],
    k: int,
) -> str:
    """B-aware source-reopening control for Experiment 8b."""

    return source_selection_prompt(dossier, cards, k, question_fact_id)


def selection_material_key(
    dossier: FictionalDossier,
    rotation: RoleRotation | None,
    mode: str,
    k: int,
    *,
    card_order_seed: str | int = DEFAULT_CARD_ORDER_SEED,
) -> str:
    """Stable material id; generic keys deliberately ignore the rotation."""

    if mode not in {"generic", "conditioned"}:
        raise ValueError("mode must be 'generic' or 'conditioned'")
    _validate_k(k, len(dossier.facts))
    if mode == "conditioned":
        if not isinstance(rotation, RoleRotation) or rotation.dossier_id != dossier.item_id:
            raise ValueError("conditioned material requires a rotation from this dossier")
        target = rotation.target_fact_id
    else:
        target = "question-blind"
    suffix = _digest("fictional-selection-v1", dossier.content_hash, card_order_seed, mode, k, target)[:20]
    return f"{dossier.item_id}:{mode}:k{k}:{suffix}"


def parse_fixed_k_selection(
    raw: str,
    valid_fact_ids: Iterable[str],
    k: int,
) -> SelectionParse:
    """Strictly parse ``{"selected_fact_ids": [...]}`` without repair.

    Markdown fences, extra keys, wrong counts, non-string ids, duplicates and
    unknown ids are invalid.  Known unique ids are retained only as diagnostics;
    the operative ``selected_fact_ids`` is empty for every invalid response.
    """

    valid_ids = tuple(valid_fact_ids)
    if len(valid_ids) != len(set(valid_ids)):
        raise ValueError("valid_fact_ids contains duplicates")
    _validate_k(k, len(valid_ids))
    errors: list[str] = []
    parsed_count: int | None = None
    raw_ids: list[Any] = []
    if not isinstance(raw, str) or not raw.strip():
        errors.append("empty")
        obj = None
    else:
        try:
            obj = json.loads(raw.strip(), object_pairs_hook=_reject_duplicate_json_keys)
        except _DuplicateJSONKey as exc:
            obj = None
            errors.append(f"duplicate_json_key:{exc.args[0]}")
        except json.JSONDecodeError:
            obj = None
            errors.append("invalid_json")
    if obj is not None:
        if not isinstance(obj, dict):
            errors.append("top_level_not_object")
        else:
            keys = set(obj)
            if keys != {SELECTION_KEY}:
                errors.append("keys_must_equal_selected_fact_ids")
            value = obj.get(SELECTION_KEY)
            if not isinstance(value, list):
                errors.append("selected_fact_ids_not_list")
            else:
                raw_ids = value
                parsed_count = len(raw_ids)
                if len(raw_ids) != k:
                    errors.append(f"wrong_count:{len(raw_ids)}!=K{k}")
                seen: set[str] = set()
                valid_set = set(valid_ids)
                for index, value_id in enumerate(raw_ids):
                    if not isinstance(value_id, str) or not value_id:
                        errors.append(f"non_string_or_empty_id:{index}")
                        continue
                    if value_id in seen:
                        errors.append(f"duplicate_id:{value_id}")
                    seen.add(value_id)
                    if value_id not in valid_set:
                        errors.append(f"unknown_id:{value_id}")
    valid_set = set(valid_ids)
    recognized: list[str] = []
    for value_id in raw_ids:
        if isinstance(value_id, str) and value_id in valid_set and value_id not in recognized:
            recognized.append(value_id)
    valid = not errors
    selected = tuple(raw_ids) if valid else ()
    joined = ";".join(errors)
    return SelectionParse(
        valid=valid,
        selected_fact_ids=selected,
        recognized_fact_ids=tuple(recognized),
        requested_k=k,
        parsed_count=parsed_count,
        errors=tuple(errors),
        error=joined,
    )


def render_packet_slots(
    cards: Sequence[EvidenceCard],
    selected_fact_ids: Iterable[str],
    k: int,
) -> str:
    """Render exactly K slots in deterministic source-card order.

    Fewer than K selected cards are padded explicitly with ``EMPTY``.  Invalid
    ids, duplicates, or more than K selections raise rather than being silently
    repaired.
    """

    _validate_cards(cards)
    _validate_k(k, len(cards))
    selected = tuple(selected_fact_ids)
    if len(selected) > k:
        raise ValueError(f"received {len(selected)} selections for K={k}")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_fact_ids contains duplicates")
    by_id = {card.fact_id: card for card in cards}
    unknown = [fact_id for fact_id in selected if fact_id not in by_id]
    if unknown:
        raise ValueError(f"unknown selected fact ids: {unknown}")
    chosen = [card for card in cards if card.fact_id in set(selected)]
    lines = [
        f"[SLOT {index} | CARD {card.card_id}]\n{card.evidence_text}"
        for index, card in enumerate(chosen, start=1)
    ]
    lines.extend(f"[SLOT {index}] EMPTY" for index in range(len(lines) + 1, k + 1))
    return "\n\n".join(lines)


def sealed_relay_prompt(sealed: hm.SealedHandoff, k: int) -> str:
    """Build a relay-selection prompt from a sealed handoff and nothing else."""

    if not isinstance(sealed, hm.SealedHandoff):
        raise TypeError(
            f"sealed_relay_prompt requires SealedHandoff, got {type(sealed).__name__}; "
            "source dossiers and evidence cards are forbidden"
        )
    _validate_k(k)
    return (
        f"Previous sealed handoff:\n{sealed.handoff_text}"
        f"{question_block(sealed.question)}\n\n"
        f"{RELAY_INSTRUCTION.format(k=k)}"
    )


def _card_map(cards: Sequence[EvidenceCard]) -> dict[str, EvidenceCard]:
    _validate_cards(cards)
    return {card.fact_id: card for card in cards}


def _validate_selected(by_id: dict[str, EvidenceCard], selected: Iterable[str]) -> tuple[str, ...]:
    result = tuple(selected)
    if not result:
        raise ValueError("selected_fact_ids must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("selected_fact_ids contains duplicates")
    unknown = [fact_id for fact_id in result if fact_id not in by_id]
    if unknown:
        raise ValueError(f"unknown selected fact ids: {unknown}")
    return result


def _word_count(card: EvidenceCard) -> int:
    return len(re.findall(r"\S+", card.evidence_text))


def restore_support(
    cards: Sequence[EvidenceCard],
    selected_fact_ids: Iterable[str],
    support_fact_id: str,
    *,
    protected_fact_ids: Iterable[str] = (),
    seed: str | int = "restore-support-v1",
) -> PacketIntervention:
    """Insert B support by replacing one length-nearest unprotected card."""

    by_id = _card_map(cards)
    before = _validate_selected(by_id, selected_fact_ids)
    if support_fact_id not in by_id:
        raise ValueError(f"unknown support fact id: {support_fact_id}")
    if support_fact_id in before:
        return PacketIntervention("restore_support", before, before, None, None, False,
                                  "support_already_present")
    protected = set(protected_fact_ids)
    removable = [fact_id for fact_id in before if fact_id not in protected]
    if not removable:
        return PacketIntervention("restore_support", before, before, None, None, False,
                                  "no_unprotected_selected_card")
    support_words = _word_count(by_id[support_fact_id])
    victim = min(removable, key=lambda fact_id: (
        abs(_word_count(by_id[fact_id]) - support_words),
        _digest(seed, support_fact_id, fact_id),
    ))
    after = list(before)
    after[after.index(victim)] = support_fact_id
    assert len(after) == len(before) and len(set(after)) == len(after)
    return PacketIntervention("restore_support", before, tuple(after), support_fact_id,
                              victim, True, "replaced_length_nearest_card")


def sham_swap(
    cards: Sequence[EvidenceCard],
    selected_fact_ids: Iterable[str],
    *,
    forbidden_fact_ids: Iterable[str] = (),
    protected_fact_ids: Iterable[str] = (),
    victim_fact_id: str | None = None,
    seed: str | int = "sham-swap-v1",
) -> PacketIntervention:
    """Swap one selected non-protected card for an unselected non-forbidden card.

    Pass ``victim_fact_id=restoration.removed_fact_id`` to make a sham occupy
    the exact slot changed by a support restoration.  The inserted sham is the
    eligible unselected card closest in word length to that victim.
    """

    by_id = _card_map(cards)
    before = _validate_selected(by_id, selected_fact_ids)
    protected = set(protected_fact_ids)
    forbidden = set(forbidden_fact_ids)
    removable = [fact_id for fact_id in before if fact_id not in protected]
    if victim_fact_id is not None:
        if victim_fact_id not in before:
            raise ValueError("victim_fact_id is not selected")
        if victim_fact_id in protected:
            raise ValueError("victim_fact_id is protected")
        victim = victim_fact_id
    elif removable:
        victim = min(removable, key=lambda fact_id: _digest(seed, "victim", fact_id))
    else:
        return PacketIntervention("sham_swap", before, before, None, None, False,
                                  "no_unprotected_selected_card")
    eligible = [fact_id for fact_id in by_id
                if fact_id not in before and fact_id not in forbidden and fact_id not in protected]
    if not eligible:
        return PacketIntervention("sham_swap", before, before, None, None, False,
                                  "no_eligible_unselected_card")
    victim_words = _word_count(by_id[victim])
    inserted = min(eligible, key=lambda fact_id: (
        abs(_word_count(by_id[fact_id]) - victim_words),
        _digest(seed, victim, fact_id),
    ))
    after = list(before)
    after[after.index(victim)] = inserted
    assert len(after) == len(before) and len(set(after)) == len(after)
    return PacketIntervention("sham_swap", before, tuple(after), inserted, victim,
                              True, "replaced_with_length_nearest_non_support_card")


def restoration_and_sham(
    cards: Sequence[EvidenceCard],
    selected_fact_ids: Iterable[str],
    support_fact_id: str,
    *,
    protected_fact_ids: Iterable[str] = (),
    seed: str | int = "paired-restoration-v1",
) -> tuple[PacketIntervention, PacketIntervention]:
    """Construct matched support-restoration and same-slot sham interventions."""

    protected = tuple(protected_fact_ids)
    restoration = restore_support(
        cards, selected_fact_ids, support_fact_id,
        protected_fact_ids=protected, seed=f"{seed}:restore",
    )
    if not restoration.changed:
        sham = PacketIntervention("sham_swap", restoration.before_fact_ids,
                                  restoration.before_fact_ids, None, None, False,
                                  "restoration_not_applicable")
    else:
        sham = sham_swap(
            cards,
            selected_fact_ids,
            forbidden_fact_ids=set(protected) | {support_fact_id},
            protected_fact_ids=protected,
            victim_fact_id=restoration.removed_fact_id,
            seed=f"{seed}:sham",
        )
    return restoration, sham
