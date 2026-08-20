"""Shared BM25 retrieval and content-fingerprint utilities.

Used by any experiment that needs real lexical retrieval to define "hard
negative" distractors (BM25-ranked highly for a query, but not actually the
relevant/gold evidence), instead of a uniformly random unrelated passage.
No external dependency: standard Okapi BM25 over an inverted index built in
plain Python, consistent with this project's preference for small,
auditable, dependency-free implementations over e.g. a sentence splitter or
NLP library.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over (doc_id, text) pairs, built once and queried many times."""

    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids = [doc_id for doc_id, _ in docs]
        doc_tokens = [tokenize(text) for _, text in docs]
        self.doc_len = [len(toks) for toks in doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if doc_tokens else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter = Counter()
        for doc_idx, toks in enumerate(doc_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((doc_idx, tf))
                df[term] += 1
        n_docs = len(docs)
        self.idf = {term: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def top_k(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (doc_id, score) pairs, highest BM25 score first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query_text)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.doc_ids[doc_idx], score) for doc_idx, score in ranked]


def content_fingerprint(texts: list[str]) -> str:
    """Identify a set of passages by their actual content, order-independent.

    Keying an on-disk handoff/answer cache on a question/pack id alone lets a
    stale cached handoff for an OLD passage set silently answer a NEW,
    unrelated passage set that happens to reuse the same id (e.g. after a
    construction method changes, or a bug fix reruns with different
    distractors). Include this fingerprint in every cache key derived from a
    pack's passages so changed content is always a cache miss, not a silent
    collision.
    """
    payload = "|".join(sorted(texts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
