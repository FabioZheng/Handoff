"""Experiment 14: compression by *elimination* rather than by rewriting.

Experiment 10 showed that conditioning a bounded handoff on the currently known
query buys present-query utility and loses future-query utility, and that the
loss grows with the designed distance between the future query and the
conditioning one. Every arm in that experiment was an **abstractive** one: an
LLM read the source and wrote new prose. Two mechanisms are therefore confounded
in its result -- the sender *selected* what to keep, and it *rewrote* what it
kept -- and either could be what commits the message to the present task.

This module supplies the missing arms: selectors that keep original evidence
verbatim and delete the rest. Nothing here generates text.

    paraphrase            rewriting, no deliberate selection
    summary               rewriting + selection          (Experiment 10, reused)
    lm_elimination        selection, no rewriting, LM scorer
    nonllm_elimination    selection, no rewriting, no neural model at all

The two elimination families answer the two remaining questions: is an LLM
needed to specialise a message toward the present task, and is a *learned*
relevance signal needed, or does classical lexical matching do it too?

Three properties are load-bearing and are enforced, not assumed:

* **Verbatim.** A delivered message is a concatenation of whole source
  sentences, in source order, joined by one space. ``verify_verbatim`` re-checks
  that against the units it claims to contain, and the caller raises if it
  fails. Experiment 10's ``extractive_*`` arms are only an LLM *asked* to copy
  sentences, and on the SQuAD run 0-29% of their messages were actually
  verbatim; a prompt is not a mechanism.
* **The hard budget is never exceeded.** Selection cannot pad, so the cap binds
  by construction: a unit that does not fit is skipped, never truncated.
  Unused capacity is real and is recorded (``fill_ratio``), because sentence
  granularity is coarse -- at a 20-word budget five of the sixteen relation
  dossiers own no sentence short enough to send at all, and the honest report of
  that is an empty message, not a chopped one.
* **The future queries never reach a selector.** A conditioned selector is
  handed exactly one query string, the current one. There is no parameter on
  ``score_units`` through which the hidden questions could arrive, and the
  offline selftest checks that no hidden question's text or gold answer can be
  recovered from a selector's inputs.

Scores come from three sources, all deterministic:

``lm_*``
    Precomputed by ``src/lm_unit_scores.py`` with a pinned local GPT-2, read
    here from a manifest-carrying artefact. Kept out of this module so the
    experiment's runtime path needs no torch, and so the scores are auditable
    on their own.
``nonllm_*``
    Computed here in plain Python: Okapi BM25 against the current query for the
    conditioned arm (``src/retrieval.py``), and TF-IDF graph centrality or
    IDF informativeness for the query-agnostic arm.
``random_selection``
    A seeded permutation. The sanity floor: any effect an ordering-free
    selector reproduces is not evidence about relevance.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anticipatory_data import EvidenceUnit  # noqa: E402
from retrieval import BM25Index, tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# The arms

LM_POLICIES = ("lm_generic", "lm_conditioned")
NONLLM_POLICIES = ("nonllm_generic", "nonllm_conditioned")
RANDOM_POLICY = "random_selection"

# The null mechanism: neither rewriting nor scored selection, just the front of
# the source. Under a hard budget a true pass-through is impossible on this
# corpus - the dossiers run 389-517 words against a 160-word cap - so the only
# honest budgeted form of "send the source" is prefix truncation, which is
# positional elimination at the same sentence granularity as the scored arms.
# It is the control that says whether a selector beats simply reading from the
# top. The UNBOUNDED pass-through reference is the separate `direct_context`
# baseline, where the reader sees the whole source.
PASSTHROUGH_POLICY = "passthrough"

SELECTION_POLICIES = LM_POLICIES + NONLLM_POLICIES + (RANDOM_POLICY, PASSTHROUGH_POLICY)

# Selectors that read the current query. Everything else must be generated once
# per (context, budget) and shared by every rotation: a query-agnostic selector
# that were run per rotation would produce k identical messages and let bookkeeping
# noise look like a rotation effect.
QUERY_AWARE = ("lm_conditioned", "nonllm_conditioned")
QUERY_AGNOSTIC = ("lm_generic", "nonllm_generic", RANDOM_POLICY, PASSTHROUGH_POLICY)

# Generic non-LLM scorers. Centrality is the default because it is the classical
# unsupervised extractive summariser (LexRank's degree centrality) and therefore
# the fairest query-independent counterpart to BM25.
NONLLM_GENERIC_SCORERS = ("centrality", "informativeness")


def is_selection(policy: str) -> bool:
    return policy in SELECTION_POLICIES


def is_query_aware(policy: str) -> bool:
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"{policy!r} is not a selection policy")
    return policy in QUERY_AWARE


# ---------------------------------------------------------------------------
# Query-agnostic lexical scores


def _tf_idf_vectors(units: list[EvidenceUnit]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """L2-normalised TF-IDF vectors over the units of one source.

    The document collection is the source's own sentences, which is what makes
    this query-independent *and* source-relative: a term carried by every
    sentence of this dossier gets idf 0 and cannot make a sentence look
    informative.
    """
    tokenised = [tokenize(u.text) for u in units]
    n_docs = len(tokenised)
    df: Counter = Counter()
    for toks in tokenised:
        df.update(set(toks))
    idf = {term: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()}
    vectors: list[dict[str, float]] = []
    for toks in tokenised:
        counts = Counter(toks)
        vec = {term: (1.0 + math.log(tf)) * idf[term] for term, tf in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({term: v / norm for term, v in vec.items()})
    return vectors, idf


def centrality_scores(units: list[EvidenceUnit]) -> list[float]:
    """Mean TF-IDF cosine similarity of each unit to the other units.

    LexRank's degree centrality without the eigenvector step: a sentence scores
    high when it shares vocabulary with the rest of the source, which is the
    classical query-free notion of "this is what the document is about".
    """
    vectors, _ = _tf_idf_vectors(units)
    n = len(vectors)
    if n < 2:
        return [0.0] * n
    scores = []
    for i, vi in enumerate(vectors):
        total = 0.0
        for j, vj in enumerate(vectors):
            if i == j:
                continue
            short, long_ = (vi, vj) if len(vi) <= len(vj) else (vj, vi)
            total += sum(weight * long_.get(term, 0.0) for term, weight in short.items())
        scores.append(total / (n - 1))
    return scores


def informativeness_scores(units: list[EvidenceUnit]) -> list[float]:
    """Mean IDF of a unit's distinct terms: rare vocabulary per word.

    The opposite bet to centrality -- it rewards the unusual sentence rather
    than the typical one -- and it is offered as a robustness variant precisely
    because "query-independent importance" has no single classical definition.
    """
    vectors, idf = _tf_idf_vectors(units)
    scores = []
    for unit in units:
        terms = set(tokenize(unit.text))
        scores.append(sum(idf[t] for t in terms) / len(terms) if terms else 0.0)
    return scores


def bm25_scores(units: list[EvidenceUnit], query: str) -> list[float]:
    """Okapi BM25 of each unit against the current query.

    The units of one source are the collection, so the idf weighting is
    relative to this dossier rather than to a corpus the sender has not read.
    """
    if not query:
        raise ValueError("bm25_scores requires the current query")
    index = BM25Index([(u.unit_id, u.text) for u in units])
    ranked = dict(index.top_k(query, len(units)))
    return [float(ranked.get(u.unit_id, 0.0)) for u in units]


def positional_scores(units: list[EvidenceUnit]) -> list[float]:
    """Earlier sentence, higher score: the front of the source, nothing more.

    Expressed as a score so it goes through the identical packing rule as every
    other arm - the only thing that differs between arms is the score vector,
    which is what makes the mechanism the independent variable.
    """
    return [float(len(units) - index) for index in range(len(units))]


def random_scores(units: list[EvidenceUnit], seed_material: str) -> list[float]:
    """A seeded permutation, expressed as scores so it packs like the others."""
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    order = list(range(len(units)))
    rng.shuffle(order)
    scores = [0.0] * len(units)
    for rank, index in enumerate(order):
        scores[index] = float(len(units) - rank)
    return scores


# ---------------------------------------------------------------------------
# Scoring entry point


def score_units(policy: str, units: list[EvidenceUnit], *,
                query: str | None = None,
                lm_scores: dict[str, float] | None = None,
                seed_material: str = "",
                generic_scorer: str = "centrality") -> list[float]:
    """Score every unit of one source under one selector.

    The signature is the leakage guarantee: a selector receives the units, at
    most ONE query string, and nothing else. There is no context object, no
    question list and no gold answer anywhere in scope, so a hidden question
    cannot influence a score even by accident.
    """
    if not units:
        raise ValueError("score_units needs at least one unit")
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"unknown selection policy {policy!r}")
    if is_query_aware(policy):
        if not query:
            raise ValueError(f"policy {policy!r} requires the current query")
    elif query is not None:
        raise ValueError(f"policy {policy!r} must not be given a query")

    if policy in LM_POLICIES:
        if not lm_scores:
            raise ValueError(f"policy {policy!r} requires precomputed LM scores "
                             "(run src/lm_unit_scores.py)")
        missing = [u.unit_id for u in units if u.unit_id not in lm_scores]
        if missing:
            raise ValueError(f"LM scores missing for {len(missing)} units "
                             f"(first: {missing[0]})")
        return [float(lm_scores[u.unit_id]) for u in units]
    if policy == "nonllm_conditioned":
        return bm25_scores(units, query or "")
    if policy == "nonllm_generic":
        if generic_scorer not in NONLLM_GENERIC_SCORERS:
            raise ValueError(f"unknown generic scorer {generic_scorer!r}")
        return (centrality_scores(units) if generic_scorer == "centrality"
                else informativeness_scores(units))
    if policy == PASSTHROUGH_POLICY:
        return positional_scores(units)
    return random_scores(units, seed_material or "random")


# ---------------------------------------------------------------------------
# Packing the channel


def word_count(text: str) -> int:
    """Same rule as ``budget.word_count``: no tokenizer, no locale, no model."""
    return len(str(text or "").split())


def pack(units: list[EvidenceUnit], scores: list[float], cap_words: int) -> tuple[int, ...]:
    """Take units in score order while they fit; return their indices in SOURCE order.

    Skipping a unit that does not fit -- rather than stopping at the first one --
    is what lets the smallest budgets spend the channel they have. It is also
    why the count of skipped-for-fit units is recorded: at a 20-word cap the
    skip is the mechanism's failure mode, not a detail.

    Ties break toward the earlier sentence, so the packing is a pure function of
    (units, scores, cap) with no dependence on sort stability or dict order.
    """
    if len(units) != len(scores):
        raise ValueError("one score per unit")
    if cap_words < 1:
        raise ValueError("cap_words must be >= 1")
    order = sorted(range(len(units)), key=lambda i: (-float(scores[i]), i))
    chosen: list[int] = []
    used = 0
    for index in order:
        cost = word_count(units[index].text)
        if used + cost <= cap_words:
            chosen.append(index)
            used += cost
    return tuple(sorted(chosen))


def render(units: list[EvidenceUnit], chosen: tuple[int, ...]) -> str:
    """The delivered message: chosen sentences, source order, one space between."""
    return " ".join(units[i].text.strip() for i in chosen)


def verify_verbatim(text: str, units: list[EvidenceUnit], chosen: tuple[int, ...]) -> None:
    """Re-derive the message from the source units and require an exact match.

    Cheap, and it is the invariant that separates this mechanism from an LLM
    asked politely to copy: if rendering ever gains a join character, a
    normalisation step or a truncation, this raises instead of quietly shipping
    a paraphrase under the name of an extraction.
    """
    expected = render(units, chosen)
    if text != expected:
        raise ValueError("delivered text is not the verbatim rendering of its units")
    for i in chosen:
        if units[i].text.strip() not in text:
            raise ValueError(f"unit {units[i].unit_id} is not present verbatim in the message")


# ---------------------------------------------------------------------------
# One message, with its diagnostics


@dataclass(frozen=True)
class SelectionMessage:
    """A delivered elimination message and everything needed to audit it.

    The diagnostics exist to separate two very different failures that look
    identical in an accuracy table: the evidence was deleted, or the evidence
    survived and the reader still missed it.
    """

    text: str
    policy: str
    budget_words: int
    delivered_words: int
    fill_ratio: float
    source_words: int
    retention_fraction: float
    selected_unit_count: int
    selected_unit_ids: tuple[str, ...]
    total_unit_count: int
    units_skipped_for_fit: int
    shortest_unit_words: int
    empty_message: bool
    scorer: str
    aspect_retention: dict[str, float] = field(default_factory=dict)
    answer_bearing_survived_qids: tuple[str, ...] = ()
    gold_span_survived_qids: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "handoff_text": self.text,
            "policy": self.policy,
            "budget_words": self.budget_words,
            "delivered_words": self.delivered_words,
            "fill_ratio": self.fill_ratio,
            "source_words": self.source_words,
            "retention_fraction": self.retention_fraction,
            "selected_unit_count": self.selected_unit_count,
            "selected_unit_ids": list(self.selected_unit_ids),
            "total_unit_count": self.total_unit_count,
            "units_skipped_for_fit": self.units_skipped_for_fit,
            "shortest_unit_words": self.shortest_unit_words,
            "empty_message": self.empty_message,
            "scorer": self.scorer,
            "aspect_retention": dict(self.aspect_retention),
            "answer_bearing_survived_qids": list(self.answer_bearing_survived_qids),
            "gold_span_survived_qids": list(self.gold_span_survived_qids),
        }


def gold_span_survivors(text: str, questions) -> tuple[str, ...]:
    """Which questions have one of their gold strings present in the message.

    Works for any mechanism, including the abstractive ones, so "the answer
    string was still in the channel" is comparable across rewriting and
    elimination. It is a presence test, not a correctness test: the reader can
    still fail on a message that contains the span.
    """
    low = str(text or "").lower()
    out = []
    for q in questions:
        golds = [str(g).lower().strip() for g in q.golds if str(g).strip()]
        if any(gold in low for gold in golds):
            out.append(q.qid)
    return tuple(out)


def build_message(policy: str, units: list[EvidenceUnit], cap_words: int, *,
                  source: str, questions, query: str | None = None,
                  lm_scores: dict[str, float] | None = None,
                  seed_material: str = "",
                  generic_scorer: str = "centrality",
                  scorer_label: str = "") -> SelectionMessage:
    """Score, pack, render, verify, and record what survived."""
    scores = score_units(policy, units, query=query, lm_scores=lm_scores,
                         seed_material=seed_material, generic_scorer=generic_scorer)
    chosen = pack(units, scores, cap_words)
    text = render(units, chosen)
    verify_verbatim(text, units, chosen)
    delivered = word_count(text)
    if delivered > cap_words:
        raise AssertionError(f"selection delivered {delivered} words over a {cap_words} cap")

    chosen_set = set(chosen)
    unit_words = [word_count(u.text) for u in units]
    remaining = cap_words - delivered
    skipped = sum(1 for i, cost in enumerate(unit_words)
                  if i not in chosen_set and cost > remaining)
    source_words = word_count(source)

    aspects = sorted({u.aspect for u in units})
    aspect_retention = {}
    for aspect in aspects:
        owned = [i for i, u in enumerate(units) if u.aspect == aspect]
        aspect_retention[aspect] = (
            sum(1 for i in owned if i in chosen_set) / float(len(owned)) if owned else 0.0)

    survived_qids = sorted({qid for i in chosen for qid in units[i].answers_qids})
    return SelectionMessage(
        text=text,
        policy=policy,
        budget_words=int(cap_words),
        delivered_words=delivered,
        fill_ratio=delivered / float(cap_words),
        source_words=source_words,
        retention_fraction=delivered / float(source_words) if source_words else 0.0,
        selected_unit_count=len(chosen),
        selected_unit_ids=tuple(units[i].unit_id for i in chosen),
        total_unit_count=len(units),
        units_skipped_for_fit=skipped,
        shortest_unit_words=min(unit_words) if unit_words else 0,
        empty_message=not text.strip(),
        scorer=scorer_label or policy,
        aspect_retention=aspect_retention,
        answer_bearing_survived_qids=tuple(survived_qids),
        gold_span_survived_qids=gold_span_survivors(text, questions),
    )
