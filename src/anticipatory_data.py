"""Evidence units and future-demand distributions for Experiment 12.

Experiments 10 and 11 treat a handoff as one irreversible message. This module
supplies the two pieces that a *harness* needs instead: a decomposition of the
source into addressable **evidence units**, and an explicit representation of
what the harness believes about future demand.

The central modelling choice is that future demand is expressed over **aspects**
rather than over natural-language questions. The relation dossiers are built on
a fully crossed 4 aspect x 4 role grid, and the aspect axis partitions the
source cleanly: measured over all 16 dossiers, every one of the 256 questions
has its gold answer located verbatim in some source sentence, 191 of 264
sentences are claimed by exactly one aspect, and **no sentence is claimed by
two** (mean pairwise aspect Jaccard 0.000). Predicting "the next question will
be about finance" is therefore a statement about a real, disjoint subset of the
evidence, not a soft topic label. That is what makes `P(Q_future)` estimable by
a harness at all, and it is why this experiment does not require predicting
exact future question strings.

The 73 unclaimed sentences are not a defect. They are background prose that
answers nothing, and a policy that spends budget on them should be penalised
for it - they are the corpus's own distractors.

Nothing here calls a model. Unit extraction is deterministic given the source
text, so the labelling is reproducible and auditable without an API key.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import split_sentences  # noqa: E402

# The four aspects of a relation dossier, in a fixed order. Every distribution
# in this module is a vector over ASPECTS in exactly this order, so that a
# caller can never silently mis-align a probability with an aspect.
ASPECTS: tuple[str, ...] = ("founding", "facilities", "finance", "custom")

UNCLAIMED = "unclaimed"


# ---------------------------------------------------------------------------
# Evidence units


@dataclass(frozen=True)
class EvidenceUnit:
    """One addressable piece of the source.

    ``aspect`` is the aspect whose questions this unit answers, or ``UNCLAIMED``
    for background prose that answers none of them. ``answers_qids`` records
    exactly which questions the unit is answer-bearing for, so a later
    retrieval or scoring step never has to re-derive it from string matching.
    """

    unit_id: str
    context_id: str
    index: int
    text: str
    aspect: str
    answers_qids: tuple[str, ...]

    @property
    def words(self) -> int:
        return len(self.text.split())

    def as_dict(self) -> dict:
        out = asdict(self)
        out["answers_qids"] = list(self.answers_qids)
        out["words"] = self.words
        return out


def extract_units(dossier: dict) -> list[EvidenceUnit]:
    """Split one dossier source into aspect-labelled evidence units.

    Labelling is by gold-answer containment, which is exact rather than
    heuristic: a unit is answer-bearing for a question iff one of that
    question's gold strings occurs in it. A unit claimed by questions from more
    than one aspect would make the aspect partition leaky, so it is an error
    rather than a silently-resolved tie - on the current corpus this never
    fires, and if a future corpus makes it fire the design needs revisiting
    rather than patching.
    """
    source = dossier["source"]
    context_id = dossier["context_id"]
    sentences = split_sentences(source)
    questions = dossier["questions"]

    units: list[EvidenceUnit] = []
    for index, sentence in enumerate(sentences):
        low = sentence.lower()
        qids: list[str] = []
        aspects: set[str] = set()
        for question in questions:
            golds = [g.lower().strip() for g in question["golds"] if g.strip()]
            if any(gold in low for gold in golds):
                qids.append(question["qid"])
                aspects.add(question["aspect"])
        if len(aspects) > 1:
            raise ValueError(
                f"{context_id} sentence {index} is claimed by aspects {sorted(aspects)}; "
                "the aspect partition of the evidence is not disjoint, which invalidates "
                "an aspect-level future-demand distribution"
            )
        aspect = aspects.pop() if aspects else UNCLAIMED
        units.append(EvidenceUnit(
            unit_id=f"{context_id}:u{index}",
            context_id=context_id,
            index=index,
            text=sentence,
            aspect=aspect,
            answers_qids=tuple(qids),
        ))
    return units


def check_coverage(dossier: dict, units: list[EvidenceUnit], min_units_per_aspect: int = 2) -> None:
    """Reject a dossier whose evidence cannot support the design.

    Two gates, both fatal. Every question's gold must be recoverable from at
    least one unit, otherwise a policy could be scored on evidence that the
    action space cannot preserve at any budget. And every aspect must own at
    least ``min_units_per_aspect`` units, otherwise an aspect-level allocation
    has nothing to allocate and a "predict finance" policy degenerates to
    "keep one sentence".
    """
    answerable = {qid for unit in units for qid in unit.answers_qids}
    missing = [q["qid"] for q in dossier["questions"] if q["qid"] not in answerable]
    if missing:
        raise ValueError(
            f"{dossier['context_id']}: {len(missing)} questions have no answer-bearing unit "
            f"(first: {missing[0]})"
        )
    for aspect in ASPECTS:
        owned = sum(1 for unit in units if unit.aspect == aspect)
        if owned < min_units_per_aspect:
            raise ValueError(
                f"{dossier['context_id']}: aspect {aspect!r} owns {owned} units, "
                f"needs at least {min_units_per_aspect}"
            )


def load_dossiers(path: str | Path) -> tuple[dict, list[dict]]:
    """Read the relation-dossier corpus, separating its manifest line.

    The corpus carries a manifest as its first line, matching the convention in
    ``data/filtered_questions.jsonl``. Callers that iterate the file naively
    will treat that manifest as a dossier and fail on a missing ``source``.
    """
    import json

    manifest: dict = {}
    dossiers: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "source" in row:
                dossiers.append(row)
            else:
                manifest = row
    return manifest, dossiers


# ---------------------------------------------------------------------------
# Future-demand distributions
#
# Every distribution is a dict keyed by ASPECTS. They are validated on the way
# in rather than trusted, because a distribution that quietly fails to sum to
# one turns every downstream expectation and divergence into nonsense.


def _validate(dist: dict[str, float], what: str) -> dict[str, float]:
    out = {aspect: float(dist.get(aspect, 0.0)) for aspect in ASPECTS}
    unknown = set(dist) - set(ASPECTS)
    if unknown:
        raise ValueError(f"{what} has unknown aspects {sorted(unknown)}")
    if any(value < 0 or not math.isfinite(value) for value in out.values()):
        raise ValueError(f"{what} has a negative or non-finite weight")
    total = sum(out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{what} sums to {total}, not 1")
    return out


def uniform() -> dict[str, float]:
    return {aspect: 1.0 / len(ASPECTS) for aspect in ASPECTS}


def point_mass(aspect: str) -> dict[str, float]:
    if aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}")
    return {a: (1.0 if a == aspect else 0.0) for a in ASPECTS}


def dilute(true: dict[str, float], tau: float) -> dict[str, float]:
    """Mix the truth toward uniform: ``P_hat = (1-tau)P + tau*U``.

    This raises the entropy of the estimate without biasing it - the harness
    becomes less certain but not wrong. tau=0 is the oracle and tau=1 is the
    maximum-entropy hedge. Keeping this separate from ``displace`` is the whole
    point: "uncertain" and "confidently wrong" are different failure modes and
    the brief treats them as different regimes.
    """
    true = _validate(true, "true distribution")
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    unit = uniform()
    return _validate(
        {a: (1.0 - tau) * true[a] + tau * unit[a] for a in ASPECTS}, "diluted estimate")


def displace(true: dict[str, float], sigma: float,
             toward: str | None = None) -> dict[str, float]:
    """Mix the truth toward the *least* likely aspect: systematic error.

    Unlike ``dilute`` this introduces bias at roughly constant entropy, which is
    the distribution-shift regime. ``toward`` defaults to the aspect with the
    least true mass - the most wrong a single-aspect prediction can be - with
    ties broken by ASPECTS order so the choice is deterministic and auditable.
    """
    true = _validate(true, "true distribution")
    if not 0.0 <= float(sigma) <= 1.0:
        raise ValueError("sigma must be in [0, 1]")
    if toward is None:
        toward = min(ASPECTS, key=lambda a: (true[a], ASPECTS.index(a)))
    wrong = point_mass(toward)
    return _validate(
        {a: (1.0 - sigma) * true[a] + sigma * wrong[a] for a in ASPECTS},
        "displaced estimate")


# ---------------------------------------------------------------------------
# Mismatch between P and P-hat


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    """Primary mismatch measure, in [0, 1].

    Reads directly as "probability mass in the wrong place", which is what makes
    a threshold in it interpretable rather than merely detectable.
    """
    p, q = _validate(p, "p"), _validate(q, "q")
    return 0.5 * sum(abs(p[a] - q[a]) for a in ASPECTS)


def jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    """Secondary mismatch measure, base 2, in [0, 1].

    Reported alongside total variation so that any threshold in the regret
    curve can be checked against a second geometry. A break that appears under
    one divergence and not the other is a property of the measure, not of the
    policy.
    """
    p, q = _validate(p, "p"), _validate(q, "q")

    def kl(x: dict[str, float], m: dict[str, float]) -> float:
        total = 0.0
        for aspect in ASPECTS:
            if x[aspect] > 0.0:
                total += x[aspect] * math.log2(x[aspect] / m[aspect])
        return total

    mid = {a: 0.5 * (p[a] + q[a]) for a in ASPECTS}
    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def expected_future_utility(utility_by_aspect: dict[str, float],
                            dist: dict[str, float]) -> float:
    """``E_{Q~dist}[U_future]`` given per-aspect utilities.

    Aspect utilities are averaged within aspect before weighting, matching the
    aggregation order used for the relation-weighted future utilities in
    Experiment 11: weighting raw cells would let an aspect with more questions
    silently outvote one with fewer.
    """
    dist = _validate(dist, "distribution")
    missing = [a for a in ASPECTS if dist[a] > 0 and a not in utility_by_aspect]
    if missing:
        raise ValueError(f"no utility for positive-weight aspects {missing}")
    return sum(dist[a] * float(utility_by_aspect[a]) for a in ASPECTS if dist[a] > 0)
