"""External evidence store, retrieval, and the sealed context for Experiment 12.

Experiments 1-11 hand the reader exactly one thing: a frozen ``SealedHandoff``
of four strings. ``handoffs.orchestrator_answer`` rejects anything else with a
``TypeError`` whose message says "the source paragraphs must not be reachable
from this call", and AGENTS.md calls that the load-bearing isolation guarantee.

Experiment 12 needs the reader to be able to *retrieve* evidence the writer
chose not to put in the message. That genuinely gives the reader source text,
so the invariant has to be restated rather than quietly relaxed. Two rules:

1. ``SealedContext`` is a new frozen type. ``SealedHandoff`` is untouched and
   its type check is not loosened, so every earlier experiment keeps exactly
   the guarantee it was verified under.
2. **The store is part of the policy, not a background resource.** It holds
   only the units a policy chose to externalize. This is the subtle failure
   mode of the whole design: if the store held the entire source, then
   "externalize everything and retrieve without limit" would be direct context
   under another name and the experiment would measure nothing. ``k`` and the
   retrieval cost are what keep retrieval from being free, and
   ``assert_not_reconstituted`` is the regression check that a policy has not
   accidentally handed the reader the whole document by two routes at once.

Retrieval is real lexical IR - the project's own dependency-free Okapi BM25
from ``retrieval.py``, already used by Experiments 3 and 4 - not a simulated
oracle that returns the right passage by construction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import BM25Index, tokenize  # noqa: E402


@dataclass(frozen=True)
class SealedContext:
    """Everything the reader is permitted to know, for a retrieval-enabled arm.

    ``message_text`` is what the writer delivered inside the word budget.
    ``retrieved_text`` is what lexical retrieval pulled from the policy's own
    store at answer time. Both are plain strings; there is no parameter through
    which the full source, the gold answers, or the aspect labels can arrive.
    """

    qid: str
    question: str
    message_text: str
    retrieved_text: tuple[str, ...]
    policy: str

    def render(self) -> str:
        """The single material block the reader sees."""
        if not self.retrieved_text:
            return self.message_text
        recalled = "\n".join(f"- {piece}" for piece in self.retrieved_text)
        return f"{self.message_text}\n\nRecalled from storage:\n{recalled}"


class EvidenceStore:
    """A policy's externalized units, queryable by BM25.

    Built per (context, rotation, policy): two policies never share a store,
    because what was externalized is itself a policy decision.
    """

    def __init__(self, units: list) -> None:
        self.units = list(units)
        self._by_id = {unit.unit_id: unit for unit in self.units}
        self._index = (BM25Index([(u.unit_id, u.text) for u in self.units])
                       if self.units else None)

    def __len__(self) -> int:
        return len(self.units)

    def retrieve(self, query: str, k: int) -> list:
        """Top-k units for a query. Deterministic, and empty when the store is.

        The query is the live question, which is what a harness would actually
        have. That makes retrieval strong, so a degraded-query control is
        needed before concluding that retrieval beats preservation.
        """
        if self._index is None or k <= 0 or not tokenize(query):
            return []
        return [self._by_id[doc_id] for doc_id, _ in self._index.top_k(query, k)]


def assert_bounded_recall(message_text: str, retrieval_k: int, all_units: list,
                          policy: str, max_fraction: float = 0.75) -> None:
    """Fail if any single answer's context could contain most of the source.

    The first version of this guard checked the store, and it was wrong. A
    policy that externalizes everything it did not deliver puts 100% of the
    source in its store, which looks alarming but is not what the reader sees:
    the reader gets the message plus at most ``k`` retrieved units *for one
    question*, and it has to earn those through BM25 against a query. Storage
    is not exposure.

    What actually has to be bounded is the worst-case context assembled for a
    single answer - message units plus ``k``. That is the quantity which, if it
    approached the whole document, would turn a retrieval arm into direct
    context wearing a different label and make the comparison meaningless.
    """
    if not all_units:
        return
    in_message = sum(1 for u in all_units if u.text in message_text)
    worst_case = min(len(all_units), in_message + max(0, int(retrieval_k)))
    fraction = worst_case / len(all_units)
    if fraction > max_fraction:
        raise ValueError(
            f"policy {policy!r} can assemble {fraction:.0%} of the source units into a "
            f"single answer context ({in_message} delivered + k={retrieval_k}, limit "
            f"{max_fraction:.0%}); that is a full-context baseline, not a bounded channel"
        )


def answer_prompt(sealed: SealedContext) -> str:
    if not isinstance(sealed, SealedContext):
        raise TypeError(
            f"the Experiment 12 answerer requires a SealedContext, got "
            f"{type(sealed).__name__}. The source must not be reachable from this call."
        )
    return (f"Research material:\n{sealed.render()}\n\n"
            f"Question:\n{sealed.question}\nAnswer:")
