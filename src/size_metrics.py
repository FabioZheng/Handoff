"""Measurements specific to Experiment 9: what changed when a handoff resized.

The project already has lexical transition measures (``paraphrase_metrics``)
and semantic ones (``judge``). Neither answers the two questions this
experiment adds:

* when a handoff got *longer*, did it gain content or repeat itself?
* when material appears in a handoff that its source document does not
  contain, where did that material come from?

Both are answered deterministically here and cross-checked by an LLM judge in
the runner. The deterministic side is primary, as everywhere else in this
project, because "this proper noun is not in the document" is a fact about two
strings and needs no model's opinion.

The unsupported-term scan is deliberately built for *precision, not recall*: it
only looks at single capitalised tokens and numeric literals, and it clears a
term the moment its normalised form occurs anywhere in the normalised document.
A rewrite that reorders or rephrases supported material therefore cannot
register as invention. Recall is the support judge's job.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# Capitalised word tokens, allowing internal apostrophes and hyphens so
# "O'Brien" and "Saint-Denis" survive as one term.
_CAP_TOKEN = re.compile(r"\b[A-ZÀ-Þ][\w'’-]*", re.UNICODE)
# Numeric literals: years, counts, decimals, and comma-grouped thousands.
# Percentages and currency keep their digits only; the symbol is not the claim.
_NUMBER = re.compile(r"\d[\d,.]*")
_NGRAM_TOKEN = re.compile(r"[\w']+", re.UNICODE)

# Capitalised tokens that are function words rather than names. Only used as a
# second line of defence: a common word is almost always present in the source
# document too, and the document test alone already clears it.
_FUNCTION_WORDS = frozenset("""
about above after again against along among around because before behind below
beneath beside between beyond during except from into like near over since
than that their theirs them then there these this those through toward under
until upon what when where which while whose with within without
also although and both but come does each every for had has have here its
more most much none note only other same some such their they this were what
""".split())


def normalize(text: str) -> str:
    """Casefold, strip accents, and collapse whitespace and punctuation.

    Accent stripping matters: a chain that rewrites "Gobernacion" for
    "Gobernacion" with the accent dropped has not invented anything, and a scan
    that called that a new entity would report noise as fabrication.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stripped.casefold()).strip()


def _padded(text: str) -> str:
    return f" {normalize(text)} "


def in_document(term: str, document_padded: str) -> bool:
    """Is this term present in the (already normalised, space-padded) document?

    Word-boundary aware via the padding, so "Ana" does not match "Antioquia"
    while "Vos" still matches "Taren Vos".
    """
    needle = normalize(term)
    return bool(needle) and f" {needle} " in document_padded


def candidate_terms(text: str, min_characters: int) -> tuple[list[str], list[str]]:
    """Return (capitalised terms, numeric literals) worth checking.

    Duplicates are collapsed on the normalised form but the first surface form
    is kept, so the audit trail shows what the model actually wrote.
    """
    seen: set[str] = set()
    caps: list[str] = []
    for match in _CAP_TOKEN.finditer(text or ""):
        term = match.group(0).strip("'’-")
        key = normalize(term)
        if len(key) < min_characters or key in seen or key in _FUNCTION_WORDS:
            continue
        seen.add(key)
        caps.append(term)
    seen.clear()
    numbers: list[str] = []
    for match in _NUMBER.finditer(text or ""):
        term = match.group(0).strip(".,")
        key = normalize(term)
        if not key or key in seen:
            continue
        seen.add(key)
        numbers.append(term)
    return caps, numbers


def nearest_term(term: str, pool: list[str]) -> tuple[str, float]:
    """The document term this one most resembles, and how closely.

    Reported alongside every unsupported term so the corruption-vs-invention
    split stays a *reporting* threshold applied to a recorded number, rather
    than a hidden judgement baked into the counts.
    """
    needle = normalize(term)
    best, best_score = "", 0.0
    for candidate in pool:
        score = difflib.SequenceMatcher(None, needle, normalize(candidate)).ratio()
        if score > best_score:
            best, best_score = candidate, score
    return best, round(best_score, 4)


def variant_scan(text: str, tracked_terms: list[str], document: str, spec: dict) -> list[dict]:
    """Near-miss restatements of a tracked value: "Taren Vos" -> "Leran Vos".

    Matching is done at PHRASE length, sliding a window of the tracked value's
    own token count across the text. Comparing single tokens instead would ask
    whether "Leran" resembles "Taren" -- two five-letter strings that share
    little character alignment and score far below any usable threshold -- when
    the thing a reader recognises as a corruption is the whole name.

    A window is only reported when it does NOT occur in the source document, so
    a document that genuinely contains two similar names never yields one as a
    corruption of the other.
    """
    threshold = float(spec.get("corruption_similarity", 0.6))
    padded_doc = _padded(document)
    words = re.findall(r"\S+", text or "")
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for value in tracked_terms:
        value = str(value).strip()
        target = normalize(value)
        if not target:
            continue
        span = max(1, len(target.split()))
        for i in range(len(words) - span + 1):
            window = " ".join(words[i:i + span]).strip(".,;:()[]\"'")
            candidate = normalize(window)
            if not candidate or candidate == target:
                continue
            score = difflib.SequenceMatcher(None, target, candidate).ratio()
            if score < threshold or in_document(window, padded_doc):
                continue
            key = (target, candidate)
            if key in seen:
                continue
            seen.add(key)
            found.append({"tracked_term": value, "surface": window,
                          "similarity": round(score, 4)})
    return found


def unsupported_scan(text: str, document: str, spec: dict,
                     tracked_terms: list[str] | None = None) -> dict:
    """Terms asserted by ``text`` that ``document`` does not contain.

    Support is decided against the whole document, token by token. The
    corruption/invention split is then decided by ``variant_scan``: a token is
    ``corrupted`` when it sits inside a phrase that is a near-miss restatement
    of one of the item's tracked values, and ``invented`` otherwise. Doing the
    split at phrase level rather than by nearest-token similarity is what makes
    it meaningful -- see ``variant_scan``.
    """
    min_characters = int(spec.get("min_term_characters", 4))
    padded = _padded(document)
    tracked = [str(t) for t in (tracked_terms or []) if str(t).strip()]
    variants = variant_scan(text, tracked, document, spec) if tracked else []
    # Tokens belonging to a detected near-miss phrase, so "Leran" inside
    # "Leran Vos" is attributed to the corruption rather than counted twice.
    variant_tokens = {normalize(w) for v in variants for w in re.findall(r"\S+", v["surface"])}

    detail: list[dict] = []
    caps, numbers = candidate_terms(text, min_characters)
    for kind, terms in (("term", caps), ("number", numbers)):
        for term in terms:
            if in_document(term, padded):
                continue
            corrupted = normalize(term) in variant_tokens
            near, score = nearest_term(term, tracked) if tracked else ("", 0.0)
            detail.append({
                "kind": kind, "surface": term,
                "nearest_tracked_term": near, "nearest_similarity": score,
                "classification": "corrupted" if corrupted else "invented",
            })
    corrupted_count = sum(1 for d in detail if d["classification"] == "corrupted")
    checked = len(caps) + len(numbers)
    return {
        "unsupported_terms": detail,
        "tracked_variants": variants,
        "unsupported_count": len(detail),
        "unsupported_term_count": sum(1 for d in detail if d["kind"] == "term"),
        "unsupported_number_count": sum(1 for d in detail if d["kind"] == "number"),
        "unsupported_corrupted_count": corrupted_count,
        "unsupported_invented_count": len(detail) - corrupted_count,
        "tracked_variant_count": len(variants),
        # Denominator, so a long handoff with three unsupported terms is not
        # read as noisier than a short one with three.
        "checked_term_count": checked,
        "unsupported_rate": (len(detail) / checked) if checked else 0.0,
    }


def ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    words = [w.casefold() for w in _NGRAM_TOKEN.findall(text or "")]
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def internal_repetition_rate(text: str, n: int = 5) -> float:
    """Share of a handoff's n-grams that already occurred earlier within it.

    0.0 for text that never repeats a five-word run; rises towards 1.0 as a
    message pads itself out by restating what it has already said. Reported
    against length, this is what separates "expanded by adding detail" from
    "expanded by saying the same thing twice".
    """
    grams = ngrams(text, n)
    if not grams:
        return 0.0
    return round(1.0 - (len(set(grams)) / len(grams)), 6)


def fact_present(text: str, probes: list[str]) -> bool:
    """Does the text contain any of the probe strings for one tracked fact?

    Matching is accent- and case-insensitive on word boundaries, so a fact
    counts as surviving when the chain rewords the sentence around it but keeps
    the value itself.
    """
    padded = _padded(text)
    return any(in_document(p, padded) for p in probes if p and len(str(p).strip()) >= 3)


def presence_trajectory(present_by_stage: dict[int, bool], stages: list[int]) -> dict:
    """Classify one fact's life across a chain.

    ``reappeared`` is the event this experiment exists to catch: the fact was
    in an earlier handoff, absent from a later one, and back again after that.
    Because stage >= 2 sees a sealed handoff and nothing else, the text it
    reappeared from provably did not contain it -- so it was reconstructed, not
    copied. Whether that reconstruction is right is scored separately.
    """
    seq = [bool(present_by_stage.get(s, False)) for s in stages]
    ever = any(seq)
    lost_at, reappeared_at = None, None
    for i in range(1, len(seq)):
        if lost_at is None and seq[i - 1] and not seq[i]:
            lost_at = stages[i]
        elif lost_at is not None and reappeared_at is None and seq[i] and not seq[i - 1]:
            reappeared_at = stages[i]
    return {
        "present_by_stage": {str(s): seq[i] for i, s in enumerate(stages)},
        "ever_present": ever,
        "present_at_first": seq[0] if seq else False,
        "present_at_last": seq[-1] if seq else False,
        "lost_at_stage": lost_at,
        "reappeared_at_stage": reappeared_at,
        "was_lost": lost_at is not None,
        "reappeared": reappeared_at is not None,
        "survived_throughout": all(seq) if seq else False,
    }


def answer_origin(document_correct: bool, parametric_correct: bool,
                  has_parametric: bool) -> str:
    """Where an answer came from, given both gold comparisons.

    ``both`` is kept as its own category rather than folded into ``document``:
    an answer naming the document's value *and* the memorised one is a
    different failure from one that simply reverted, and collapsing them would
    hide exactly the hedging behaviour expansion is expected to produce.
    """
    if not has_parametric:
        return "document" if document_correct else "unsupported"
    if document_correct and parametric_correct:
        return "both"
    if document_correct:
        return "document"
    if parametric_correct:
        return "parametric"
    return "other"
