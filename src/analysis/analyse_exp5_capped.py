"""Main analysis for the exp5_cap_only family.

Reads one frozen main run (write, read and judge stages) and the cohort, and
writes the four layers of results the design fixes in advance:

Primary (total effect)
    Paired on document, conditioned minus generic future utility at the same
    nominal cap, within each mechanism. For a conditioned message written for
    question a, future utility is its mean score over the document's other
    questions; the generic message of the same document and cap is scored on
    exactly the same questions. Reported per cap with Holm correction across
    caps, plus one pre-specified summary: the mean over all frozen caps. No
    single cap is the headline.

Confirmatory (no reader in the path)
    Evidence survival by question distance. For each future question, the share
    of its gold answer's content words present in the message, split by how the
    question's evidence relates to the conditioning question's: shared, near
    (disjoint but within three paragraphs, a positional stand-in for "same
    section", since paragraphs carry no section labels), or far.

Mechanistic
    Marginal allocation: for each step between consecutive caps, what the extra
    words bought, per 100 added words - current-question evidence, shared,
    near and far evidence, repeated five-word runs, and unsupported terms
    (Experiment 9's deterministic scan, so the forced-expansion study and this
    one use the same instrument).

Secondary (controlled direct effect)
    Realized length is a mediator, so these estimate what remains once the
    length pathway is closed, not the total effect: a matched realized-length
    comparison, and a within-document regression of future utility on question
    visibility, log realized words and log nominal cap. Neither replaces the
    primary test.

    python src/analysis/analyse_exp5_capped.py --config configs/exp5_cap_only_frozen_config.yaml --split main
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import size_metrics as sx  # noqa: E402
from reuse_common import config, evidence_relationship, read_rows  # noqa: E402
from exp5_runs import parent_signature  # noqa: E402
from score import normalize_answer  # noqa: E402

FAMILIES = {"generation": ("summary_conditioned", "summary_generic"),
            "selection": ("selection_conditioned", "selection_generic")}
METRICS = ("f1", "em", "judge_correct")
TIERS = ("shared", "near", "far")
NEAR_GAP = 3
MATCH_TOLERANCE = 0.15
BOOTSTRAP = 2000
STOPWORDS = set("""a an the of in on at to for from by with and or but is are was were be been being
it its this that these those as into than then there their they them he she his her we our you your
i not no yes do does did has have had which who whom what when where why how also can could would
should may might will shall""".split())
UNSUPPORTED_SPEC = {"min_term_characters": 4}


# ---------------------------------------------------------------------- load

def run_directory(cfg: dict, split: str, tag: str) -> Path:
    name = parent_signature(cfg)[:12] + (f"-{tag}" if tag else "")
    path = ROOT / cfg["run_root"] / split / name
    if not path.is_dir():
        raise SystemExit(f"run directory not found: {path}")
    return path


def load_stage(run_dir: Path, stage: str, writer: str, key: str) -> list[dict]:
    rows = []
    for path in sorted((run_dir / stage / writer).glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))[key])
    return rows


def attach_verdicts(answers: list[dict], verdicts: list[dict]) -> None:
    index = {(v["message_id"], v["qid"]): v.get("judge_correct") for v in verdicts}
    for a in answers:
        a["judge_correct"] = index.get((a["message_id"], a["qid"]))


# ------------------------------------------------------------------ helpers

def content_tokens(text: str) -> list[str]:
    return [t for t in normalize_answer(text).split() if t not in STOPWORDS]


def answer_recall(message: str, golds: list[str]) -> float | None:
    """Best share of a gold answer's content words present in the message."""
    have = set(content_tokens(message))
    best = None
    for gold in golds:
        tokens = content_tokens(gold)
        if tokens:
            value = sum(t in have for t in tokens) / len(tokens)
            best = value if best is None else max(best, value)
    return best


def answer_present(message: str, golds: list[str]) -> bool:
    text = f" {normalize_answer(message)} "
    return any(f" {normalize_answer(g)} " in text for g in golds if normalize_answer(g))


def is_yes_no(question: dict) -> bool:
    kind = question.get("answer_type")
    kinds = kind if isinstance(kind, list) else [kind]
    return "yes_no" in kinds


def distance(doc: dict, a: dict, b: dict) -> str | None:
    rel = evidence_relationship(a, b)["label"]
    if rel == "unknown":
        return None
    if rel in ("shared", "ambiguous"):
        return "shared"
    order = {p["id"]: i for i, p in enumerate(doc["paragraphs"])}
    pa = [order[p] for es in a["evidence_sets"] for p in es if p in order]
    pb = [order[p] for es in b["evidence_sets"] for p in es if p in order]
    if not pa or not pb:
        return None
    gap = min(abs(x - y) for x in pa for y in pb)
    return "near" if gap <= NEAR_GAP else "far"


def bootstrap(values: list[float], seed: int = 0) -> dict:
    if not values:
        return {"estimate": None, "lo": None, "hi": None, "p": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), (BOOTSTRAP, len(arr)))].mean(axis=1)
    est = float(arr.mean())
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"estimate": est, "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "p": float(min(1.0, p)), "n": len(arr)}


def holm(pvalues: list[float | None]) -> list[float | None]:
    indexed = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out: list[float | None] = [None] * len(pvalues)
    running, m = 0.0, len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


# ------------------------------------------------------------------ tables

class Index:
    """Scores and messages keyed the way every analysis needs them."""

    def __init__(self, docs, messages, answers):
        self.docs = {d["id"]: d for d in docs}
        self.questions = {d["id"]: {q["qid"]: q for q in d["questions"]} for d in docs}
        self.message = {m["message_id"]: m for m in messages if m["block"] == "core"}
        self.score = defaultdict(dict)   # (doc, policy, cap, anchor) -> {qid: row}
        for a in answers:
            if a["block"] != "core":
                continue
            self.score[(a["document_id"], a["policy"], a["cap"], a["anchor"])][a["qid"]] = a
        self.by_cell = defaultdict(list)  # (doc, policy, cap) -> [message]
        for m in self.message.values():
            if m["valid"]:
                self.by_cell[(m["document_id"], m["policy"], m["cap"])].append(m)

    def caps(self) -> list[int]:
        return sorted({m["cap"] for m in self.message.values()})


def paired_rows(ix: Index, corpus: str, family: str, cap: int, metric: str) -> dict:
    """Per-document mean over anchors of (conditioned - generic) on q != anchor."""
    cond, gen = FAMILIES[family]
    out = {"future": {}, "now": {}}
    for doc_id, doc in ix.docs.items():
        if doc["corpus"] != corpus:
            continue
        g_scores = ix.score.get((doc_id, gen, cap, None), {})
        d_future, d_now = [], []
        for m in ix.by_cell.get((doc_id, cond, cap), []):
            a = m["anchor"]
            c_scores = ix.score.get((doc_id, cond, cap, a), {})
            others = [q for q in ix.questions[doc_id] if q != a
                      and q in c_scores and q in g_scores
                      and c_scores[q].get(metric) is not None and g_scores[q].get(metric) is not None]
            if others:
                d_future.append(st.mean(c_scores[q][metric] for q in others)
                                - st.mean(g_scores[q][metric] for q in others))
            if a in c_scores and a in g_scores and None not in (c_scores[a].get(metric), g_scores[a].get(metric)):
                d_now.append(c_scores[a][metric] - g_scores[a][metric])
        if d_future:
            out["future"][doc_id] = st.mean(d_future)
        if d_now:
            out["now"][doc_id] = st.mean(d_now)
    return out


def level(ix: Index, corpus: str, policy: str, cap: int, metric: str, future: bool) -> float | None:
    values = []
    for doc_id, doc in ix.docs.items():
        if doc["corpus"] != corpus:
            continue
        per_doc = []
        for m in ix.by_cell.get((doc_id, policy, cap), []):
            scores = ix.score.get((doc_id, policy, cap, m["anchor"]), {})
            if m["anchor"] is None:
                qs = list(scores)
            elif future:
                qs = [q for q in scores if q != m["anchor"]]
            else:
                qs = [m["anchor"]] if m["anchor"] in scores else []
            vals = [scores[q][metric] for q in qs if scores[q].get(metric) is not None]
            if vals:
                per_doc.append(st.mean(vals))
        if per_doc:
            values.append(st.mean(per_doc))
    return st.mean(values) if values else None


def primary(ix: Index, corpora: list[str]) -> list[dict]:
    rows = []
    for corpus in corpora:
        for family in FAMILIES:
            for metric in METRICS:
                block = []
                for cap in ix.caps():
                    pr = paired_rows(ix, corpus, family, cap, metric)
                    fut, now = bootstrap(list(pr["future"].values())), bootstrap(list(pr["now"].values()))
                    cond, gen = FAMILIES[family]
                    block.append({"corpus": corpus, "family": family, "metric": metric, "cap": cap,
                                  "U_future_conditioned": level(ix, corpus, cond, cap, metric, True),
                                  "U_future_generic": level(ix, corpus, gen, cap, metric, True),
                                  "U_now_conditioned": level(ix, corpus, cond, cap, metric, False),
                                  "delta_future": fut["estimate"], "delta_future_lo": fut["lo"],
                                  "delta_future_hi": fut["hi"], "delta_future_p": fut["p"],
                                  "delta_now": now["estimate"], "documents": fut["n"],
                                  "_per_doc": pr["future"]})
                for row, adj in zip(block, holm([r["delta_future_p"] for r in block])):
                    row["delta_future_p_holm"] = adj
                # Pre-specified summary: mean over all frozen caps, per document.
                docs = set.intersection(*[set(r["_per_doc"]) for r in block]) if block else set()
                summary = bootstrap([st.mean(r["_per_doc"][d] for r in block) for d in sorted(docs)])
                for row in block:
                    row.pop("_per_doc")
                rows.extend(block)
                rows.append({"corpus": corpus, "family": family, "metric": metric, "cap": "mean",
                             "delta_future": summary["estimate"], "delta_future_lo": summary["lo"],
                             "delta_future_hi": summary["hi"], "delta_future_p": summary["p"],
                             "documents": summary["n"]})
    return rows


def interaction(ix: Index, corpora: list[str], metric: str = "f1") -> list[dict]:
    rows = []
    for corpus in corpora:
        for cap in ix.caps():
            g = paired_rows(ix, corpus, "generation", cap, metric)["future"]
            s = paired_rows(ix, corpus, "selection", cap, metric)["future"]
            common = sorted(set(g) & set(s))
            b = bootstrap([g[d] - s[d] for d in common])
            rows.append({"corpus": corpus, "cap": cap, "metric": metric,
                         "generation_minus_selection": b["estimate"], "lo": b["lo"], "hi": b["hi"],
                         "p": b["p"], "documents": b["n"]})
    return rows


# ------------------------------------------------------------- confirmatory

def survival(ix: Index, corpora: list[str]) -> list[dict]:
    """Conditioned minus generic evidence survival, by distance tier, paired on document."""
    rows = []
    for corpus in corpora:
        for family, (cond, gen) in FAMILIES.items():
            for cap in ix.caps():
                per_tier = {t: defaultdict(list) for t in TIERS + ("now",)}
                exact_far = defaultdict(list)
                for doc_id, doc in ix.docs.items():
                    if doc["corpus"] != corpus:
                        continue
                    generic = ix.by_cell.get((doc_id, gen, cap), [])
                    if not generic:
                        continue
                    g_text = generic[0]["text"]
                    qs = ix.questions[doc_id]
                    for m in ix.by_cell.get((doc_id, cond, cap), []):
                        a = qs.get(m["anchor"])
                        if a is None:
                            continue
                        for qid, q in qs.items():
                            if is_yes_no(q):
                                continue
                            tier = "now" if qid == a["qid"] else distance(doc, a, q)
                            if tier is None:
                                continue
                            rc, rg = answer_recall(m["text"], q["golds"]), answer_recall(g_text, q["golds"])
                            if rc is None or rg is None:
                                continue
                            per_tier[tier][doc_id].append(rc - rg)
                            if tier == "far" and max(len(g.split()) for g in q["golds"]) <= 5:
                                exact_far[doc_id].append(int(answer_present(m["text"], q["golds"]))
                                                         - int(answer_present(g_text, q["golds"])))
                for tier, by_doc in per_tier.items():
                    b = bootstrap([st.mean(v) for v in by_doc.values() if v])
                    rows.append({"corpus": corpus, "family": family, "cap": cap, "tier": tier,
                                 "measure": "answer_token_recall", "conditioned_minus_generic": b["estimate"],
                                 "lo": b["lo"], "hi": b["hi"], "documents": b["n"]})
                b = bootstrap([st.mean(v) for v in exact_far.values() if v])
                rows.append({"corpus": corpus, "family": family, "cap": cap, "tier": "far",
                             "measure": "exact_short_answer", "conditioned_minus_generic": b["estimate"],
                             "lo": b["lo"], "hi": b["hi"], "documents": b["n"]})
    return rows


# --------------------------------------------------------------- mechanism

def profile(doc: dict, text: str, anchor: dict | None) -> dict:
    """What one message carries: evidence by tier, repetition, unsupported terms."""
    qs = [q for q in doc["questions"] if not is_yes_no(q)]
    out = {"words": len(text.split()), "now": 0.0, "shared": 0.0, "near": 0.0, "far": 0.0}
    for q in qs:
        r = answer_recall(text, q["golds"]) or 0.0
        if anchor is not None and q["qid"] == anchor["qid"]:
            out["now"] += r
        elif anchor is not None:
            tier = distance(doc, anchor, q)
            if tier:
                out[tier] += r
    grams = sx.ngrams(text, 5)
    out["repeated_5grams"] = len(grams) - len(set(grams))
    out["unsupported_terms"] = sx.unsupported_scan(text, doc["source"], UNSUPPORTED_SPEC)["unsupported_count"]
    return out


def allocation(ix: Index, corpora: list[str]) -> list[dict]:
    """Per step between consecutive caps: what each 100 added words bought."""
    keys = ("now", "shared", "near", "far", "repeated_5grams", "unsupported_terms")
    caps = ix.caps()
    rows = []
    for corpus in corpora:
        for family, pair in FAMILIES.items():
            for policy in pair:
                for lo, hi in zip(caps, caps[1:]):
                    steps = defaultdict(list)
                    for doc_id, doc in ix.docs.items():
                        if doc["corpus"] != corpus:
                            continue
                        anchors = [m["anchor"] for m in ix.by_cell.get((doc_id, pair[0], hi), [])]
                        for anchor_id in anchors:
                            anchor = ix.questions[doc_id].get(anchor_id)
                            pick = (lambda cap: [m for m in ix.by_cell.get((doc_id, policy, cap), [])
                                                 if m["anchor"] in (anchor_id, None)])
                            before, after = pick(lo), pick(hi)
                            if not before or not after:
                                continue
                            a, b = profile(doc, before[0]["text"], anchor), profile(doc, after[0]["text"], anchor)
                            added = b["words"] - a["words"]
                            steps["added_words"].append(added)
                            if added > 0:
                                for k in keys:
                                    steps[k].append(100 * (b[k] - a[k]) / added)
                    row = {"corpus": corpus, "family": family, "policy": policy, "step": f"{lo}->{hi}",
                           "pairs": len(steps["added_words"]),
                           "mean_added_words": st.mean(steps["added_words"]) if steps["added_words"] else None}
                    for k in keys:
                        row[f"{k}_per_100_words"] = st.mean(steps[k]) if steps[k] else None
                    rows.append(row)
    return rows


# --------------------------------------------------------------- secondary

def message_level(ix: Index, corpus: str, family: str, metric: str = "f1") -> list[dict]:
    """One row per (doc, cap, anchor, arm): future utility on q != anchor."""
    cond, gen = FAMILIES[family]
    rows = []
    for doc_id, doc in ix.docs.items():
        if doc["corpus"] != corpus:
            continue
        for cap in ix.caps():
            g_msgs = ix.by_cell.get((doc_id, gen, cap), [])
            g_scores = ix.score.get((doc_id, gen, cap, None), {})
            for m in ix.by_cell.get((doc_id, cond, cap), []):
                a = m["anchor"]
                c_scores = ix.score.get((doc_id, cond, cap, a), {})
                others = [q for q in ix.questions[doc_id] if q != a and q in c_scores and q in g_scores]
                if not others or not g_msgs:
                    continue
                rows.append({"doc": doc_id, "cap": cap, "anchor": a, "visible": 1,
                             "words": m["delivered_words"],
                             "y": st.mean(c_scores[q][metric] for q in others)})
                rows.append({"doc": doc_id, "cap": cap, "anchor": a, "visible": 0,
                             "words": g_msgs[0]["delivered_words"],
                             "y": st.mean(g_scores[q][metric] for q in others)})
    return rows


# Length bins for the flexible specification: log2 width 0.2, so realized
# lengths inside one bin differ by at most about 15%, the same tolerance the
# matched comparison uses.
LENGTH_BIN = 0.2
VIF_WARNING = 10.0
# Below this share of rows in bins shared by both arms, the two arms barely
# overlap in realized length and the bin comparison is not read.
MIN_COMMON_SUPPORT = 0.3


def _design(group: list[dict], spec: str) -> np.ndarray:
    if spec == "log_linear":
        return np.array([[r["visible"], math.log(r["words"] + 1), math.log(r["cap"])] for r in group])
    raise ValueError(spec)


def _fit(sample: list[dict], spec: str) -> tuple[np.ndarray, float]:
    """Within-document OLS; returns coefficients and the VIF of `visible`."""
    by_doc = defaultdict(list)
    for r in sample:
        by_doc[r["_doc"]].append(r)
    if spec == "length_bins":
        bins = sorted({r["_bin"] for r in sample})
        column = {b: i for i, b in enumerate(bins[1:])}   # first bin is the reference

        def design(group):
            out = np.zeros((len(group), 1 + len(column)))
            for i, r in enumerate(group):
                out[i, 0] = r["visible"]
                if r["_bin"] in column:
                    out[i, 1 + column[r["_bin"]]] = 1.0
            return out
    else:
        def design(group):
            return _design(group, spec)
    X, y = [], []
    for group in by_doc.values():
        feats, target = design(group), np.array([r["y"] for r in group])
        X.append(feats - feats.mean(axis=0))
        y.append(target - target.mean())
    X, y = np.vstack(X), np.concatenate(y)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    # VIF of `visible`: how well the other regressors predict it.
    others = X[:, 1:]
    if others.shape[1] == 0 or not np.any(X[:, 0]):
        vif = 1.0
    else:
        fitted = others @ np.linalg.lstsq(others, X[:, 0], rcond=None)[0]
        ss_res = float(((X[:, 0] - fitted) ** 2).sum())
        ss_tot = float((X[:, 0] ** 2).sum())
        vif = math.inf if ss_res <= 1e-12 * max(ss_tot, 1e-12) else ss_tot / ss_res
    return coef, vif


def regression(rows: list[dict], spec: str = "length_bins", seed: int = 0) -> dict:
    """Controlled direct effect of visibility, doc-cluster bootstrap.

    ``length_bins``: y ~ visible + realized-length-bin fixed effects, within
    document. Compares arms at comparable realized length and makes no
    functional-form assumption about how length helps. This is the reported
    secondary specification.

    ``log_linear``: y ~ visible + log(words) + log(cap), within document. Kept
    as a check. Under cap-only, realized length is largely determined by
    visibility and cap together, so this specification can be close to
    collinear; the VIF of `visible` is reported and a value above
    ``VIF_WARNING`` means the coefficient is not identified and must not be read.
    """
    empty = {"spec": spec, "visible": None, "lo": None, "hi": None, "vif": None,
             "identified": False, "common_support": None, "n_rows": len(rows), "documents": 0}
    if len(rows) < 8:
        return empty
    for r in rows:
        r["_bin"] = int(math.floor(math.log2(max(r["words"], 1)) / LENGTH_BIN))
    support = 1.0
    if spec == "length_bins":
        # Only bins where both arms occur can say anything about the arms at
        # comparable length; everything else is extrapolation across bins.
        arms = defaultdict(set)
        for r in rows:
            arms[r["_bin"]].add(r["visible"])
        kept = [r for r in rows if len(arms[r["_bin"]]) == 2]
        support = len(kept) / len(rows)
        rows = kept
        empty["common_support"] = support
        if len(rows) < 8:
            return empty
    try:
        coef, vif = _fit([{**r, "_doc": r["doc"]} for r in rows], spec)
    except np.linalg.LinAlgError:
        return empty
    docs = sorted({r["doc"] for r in rows})
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc"]].append(r)
    rng, draws = random.Random(seed), []
    for _ in range(BOOTSTRAP // 4):
        sample = []
        for k, d in enumerate(rng.choice(docs) for _ in docs):
            sample.extend({**r, "_doc": f"{d}#{k}"} for r in by_doc[d])
        try:
            draws.append(_fit(sample, spec)[0][0])
        except np.linalg.LinAlgError:
            continue
    draws.sort()
    identified = vif <= VIF_WARNING and support >= MIN_COMMON_SUPPORT
    out = {"spec": spec, "visible": float(coef[0]),
           "lo": float(draws[int(0.025 * len(draws))]) if draws else None,
           "hi": float(draws[int(0.975 * len(draws)) - 1]) if draws else None,
           "vif": vif, "identified": identified, "common_support": support,
           "n_rows": len(rows), "documents": len(docs)}
    if spec == "log_linear":
        out["log_words"], out["log_cap"] = float(coef[1]), float(coef[2])
    return out


def matched_length(rows: list[dict]) -> dict:
    """Conditioned message against the generic message of the same document, any
    cap, whose realized length is closest and within the tolerance."""
    by_doc = defaultdict(lambda: {"c": [], "g": []})
    for r in rows:
        by_doc[r["doc"]]["c" if r["visible"] else "g"].append(r)
    per_doc, matched, total = [], 0, 0
    for group in by_doc.values():
        deltas = []
        for c in group["c"]:
            total += 1
            pool = [g for g in group["g"] if g["anchor"] == c["anchor"] and c["words"]
                    and abs(g["words"] - c["words"]) / c["words"] <= MATCH_TOLERANCE]
            if pool:
                g = min(pool, key=lambda g: abs(g["words"] - c["words"]))
                deltas.append(c["y"] - g["y"])
                matched += 1
        if deltas:
            per_doc.append(st.mean(deltas))
    b = bootstrap(per_doc)
    return {"delta": b["estimate"], "lo": b["lo"], "hi": b["hi"], "documents": b["n"],
            "matched": matched, "conditioned_messages": total,
            "match_rate": matched / total if total else None}


def secondary(ix: Index, corpora: list[str]) -> list[dict]:
    rows = []
    for corpus in corpora:
        for family in FAMILIES:
            data = message_level(ix, corpus, family)
            bins, loglin = regression(data, "length_bins"), regression(data, "log_linear")
            mat = matched_length(data)
            rows.append({"corpus": corpus, "family": family, "estimand": "controlled_direct_effect",
                         "bins_visible": bins["visible"], "bins_lo": bins["lo"], "bins_hi": bins["hi"],
                         "bins_vif": bins["vif"], "bins_identified": bins["identified"],
                         "bins_common_support": bins["common_support"],
                         "loglin_visible": loglin["visible"], "loglin_lo": loglin["lo"],
                         "loglin_hi": loglin["hi"], "loglin_log_words": loglin.get("log_words"),
                         "loglin_vif": loglin["vif"], "loglin_identified": loglin["identified"],
                         "matched_delta": mat["delta"], "matched_lo": mat["lo"], "matched_hi": mat["hi"],
                         "matched_rate": mat["match_rate"], "documents": bins.get("documents")})
    return rows


def length_audit(ix: Index, corpora: list[str]) -> list[dict]:
    rows = []
    for corpus in corpora:
        for policy in sorted({m["policy"] for m in ix.message.values()}):
            for cap in ix.caps():
                ms = [m for m in ix.message.values() if m["corpus"] == corpus and m["policy"] == policy
                      and m["cap"] == cap]
                valid = [m for m in ms if m["valid"]]
                if not ms:
                    continue
                rows.append({"corpus": corpus, "policy": policy, "cap": cap, "messages": len(ms),
                             "valid": len(valid) / len(ms),
                             "words_median": st.median(m["delivered_words"] for m in valid) if valid else None,
                             "fill_median": st.median(m["fill_ratio"] for m in valid) if valid else None,
                             "tokens_median": st.median(m["realized_tokens"] for m in valid) if valid else None,
                             "truncated": sum(m["truncated_words"] > 0 for m in valid),
                             "over_cap": sum(m["over_cap"] for m in valid)})
    return rows


# ------------------------------------------------------------------ output

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(out: Path, prim: list[dict], surv: list[dict], corpora: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"generation": "#2a78d6", "selection": "#eb6834"}
    fig, axes = plt.subplots(len(corpora), 3, figsize=(13, 3.6 * len(corpora)), squeeze=False)
    for i, corpus in enumerate(corpora):
        for family in FAMILIES:
            rows = [r for r in prim if r["corpus"] == corpus and r["family"] == family
                    and r["metric"] == "f1" and r["cap"] != "mean"]
            caps = [r["cap"] for r in rows]
            axes[i][0].plot(caps, [r["U_future_generic"] for r in rows], "--", color=colors[family],
                            label=f"{family} generic")
            axes[i][0].plot(caps, [r["U_future_conditioned"] for r in rows], "-o", color=colors[family],
                            label=f"{family} conditioned")
            # A cell with no paired documents has no interval; leave it out of the error bars.
            ci = [r for r in rows if r["delta_future_lo"] is not None]
            est = [r["delta_future"] for r in ci]
            axes[i][1].errorbar([r["cap"] for r in ci], est,
                                yerr=[[e - r["delta_future_lo"] for e, r in zip(est, ci)],
                                      [r["delta_future_hi"] - e for e, r in zip(est, ci)]],
                                fmt="-o", color=colors[family], capsize=3, label=family)
            far = [r for r in surv if r["corpus"] == corpus and r["family"] == family and r["tier"] == "far"
                   and r["measure"] == "answer_token_recall"]
            axes[i][2].plot([r["cap"] for r in far], [r["conditioned_minus_generic"] for r in far], "-o",
                            color=colors[family], label=family)
        for ax, title in zip(axes[i], ("future utility (F1)", "conditioned - generic, future",
                                       "far-evidence survival, cond - gen")):
            ax.set_xscale("log")
            ax.set_title(f"{corpus}: {title}", fontsize=10)
            ax.set_xlabel("nominal cap (words)")
            if "-" in title:
                ax.axhline(0, color="#898781", lw=0.8)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "exp5_cap_only_main.png", dpi=150)
    plt.close(fig)


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def report(meta, prim, inter, surv, alloc, sec, audit) -> str:
    lines = [f"# exp5_cap_only main analysis — {meta['writer']}", "",
             f"Run `{meta['run']}`, split `{meta['split']}`, frozen caps {meta['caps']}. "
             f"Headline corpora from calibration: {', '.join(meta['headline']) or 'not recorded'}.", "",
             "## Primary: total effect of question visibility on future utility (F1)", "",
             "Conditioned minus generic at the same nominal cap, paired on document. "
             "Holm-corrected across caps; the `mean` row is the pre-specified summary over all caps.", "",
             "| corpus | family | cap | U_future cond / gen | delta future [95% CI] | p (Holm) | delta now | docs |",
             "|---|---|---|---|---|---|---|---:|"]
    for r in prim:
        if r["metric"] != "f1":
            continue
        u = (f"{r['U_future_conditioned']:.3f} / {r['U_future_generic']:.3f}"
             if r.get("U_future_conditioned") is not None and r.get("U_future_generic") is not None else "")
        ci = (f"{fmt(r['delta_future'])} [{fmt(r['delta_future_lo'])}, {fmt(r['delta_future_hi'])}]"
              if r["delta_future"] is not None else "—")
        p = r.get("delta_future_p_holm", r.get("delta_future_p"))
        lines.append(f"| {r['corpus']} | {r['family']} | {r['cap']} | {u} | {ci} | "
                     f"{'' if p is None else f'{p:.4f}'} | {fmt(r.get('delta_now'))} | {r['documents']} |")
    lines += ["", "## Generation minus selection (interaction), F1", "",
              "| corpus | cap | generation − selection [95% CI] | docs |", "|---|---:|---|---:|"]
    lines += [f"| {r['corpus']} | {r['cap']} | {fmt(r['generation_minus_selection'])} "
              f"[{fmt(r['lo'])}, {fmt(r['hi'])}] | {r['documents']} |" for r in inter]
    lines += ["", "## Confirmatory: evidence survival by distance (conditioned − generic)", "",
              "Answer-token recall; yes/no questions excluded. `near` is disjoint evidence within "
              f"{NEAR_GAP} paragraphs, a positional stand-in for same section.", "",
              "| corpus | family | cap | now | shared | near | far |", "|---|---|---:|---:|---:|---:|---:|"]
    cells = defaultdict(dict)
    for r in surv:
        if r["measure"] == "answer_token_recall":
            cells[(r["corpus"], r["family"], r["cap"])][r["tier"]] = r["conditioned_minus_generic"]
    for (corpus, family, cap), t in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        lines.append(f"| {corpus} | {family} | {cap} | " + " | ".join(fmt(t.get(k), 2) for k in
                                                                    ("now",) + TIERS) + " |")
    lines += ["", "## Mechanism: what each 100 added words bought", "",
              "| corpus | policy | step | now | shared | near | far | repeated 5-grams | unsupported |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in alloc:
        lines.append(f"| {r['corpus']} | {r['policy']} | {r['step']} | " + " | ".join(
            fmt(r.get(f"{k}_per_100_words"), 2) for k in
            ("now", "shared", "near", "far", "repeated_5grams", "unsupported_terms")) + " |")
    lines += ["", "## Secondary: controlled direct effect (length pathway closed)", "",
              "Realized length is a mediator; these do not estimate the total effect, and the direct "
              "effect is identified only where the two arms' realized lengths overlap. "
              "**Matched length** is the reported estimate: each conditioned message against the generic "
              f"message of the same document, any cap, within {MATCH_TOLERANCE:.0%} realized length; the match "
              "rate says how much overlap there is. The two regressions are checks and are marked "
              f"**not identified** when common support is below {MIN_COMMON_SUPPORT:.0%} or the VIF of "
              f"visibility exceeds {VIF_WARNING:.0f}.", "",
              "| corpus | family | matched length [95% CI] | match rate | length bins [95% CI] (support) | "
              "log-linear [95% CI] (VIF) |", "|---|---|---|---:|---|---|"]

    def check(r, prefix, extra):
        flag = "" if r[f"{prefix}_identified"] else " **not identified**"
        return (f"{fmt(r[f'{prefix}_visible'])} [{fmt(r[f'{prefix}_lo'])}, {fmt(r[f'{prefix}_hi'])}] "
                f"({extra}){flag}")

    for r in sec:
        rate = "" if r["matched_rate"] is None else f"{r['matched_rate']:.2f}"
        vif = r["loglin_vif"]
        vif_text = "—" if vif is None else ("inf" if math.isinf(vif) else f"{vif:.1f}")
        support = r["bins_common_support"]
        support_text = "—" if support is None else f"{support:.0%}"
        lines.append(f"| {r['corpus']} | {r['family']} | "
                     f"{fmt(r['matched_delta'])} [{fmt(r['matched_lo'])}, {fmt(r['matched_hi'])}] | {rate} | "
                     f"{check(r, 'bins', support_text)} | {check(r, 'loglin', 'VIF ' + vif_text)} |")
    lines += ["", "## Channel audit", "",
              "| corpus | policy | cap | valid | median words | median fill | median tokens | truncated | over cap |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in audit:
        lines.append(f"| {r['corpus']} | {r['policy']} | {r['cap']} | {r['valid']:.3f} | {r['words_median']} | "
                     f"{r['fill_median']} | {r['tokens_median']} | {r['truncated']} | {r['over_cap']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp5_cap_only_frozen_config.yaml")
    parser.add_argument("--split", default="main")
    parser.add_argument("--writer", default="mistral24")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = config(str(ROOT / args.config))
    run_dir = Path(args.run_dir) if args.run_dir else run_directory(cfg, args.split, args.tag)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = read_rows(ROOT / cfg["data_root"] / f"{args.split}.jsonl")
    messages = load_stage(run_dir, "write", args.writer, "messages")
    answers = load_stage(run_dir, "read", args.writer, "answers")
    verdicts = load_stage(run_dir, "judge", args.writer, "verdicts") if (run_dir / "judge").is_dir() else []
    attach_verdicts(answers, verdicts)
    seen = {m["document_id"] for m in messages}
    docs = [d for d in docs if d["id"] in seen]
    corpora = sorted({d["corpus"] for d in docs})
    ix = Index(docs, messages, answers)
    calib = cfg["budgets"].get("calibrated_from") or {}
    meta = {"writer": args.writer, "run": run_dir.name, "split": args.split, "caps": ix.caps(),
            "headline": calib.get("headline", []), "budgets_frozen": manifest.get("budgets_frozen")}
    prim, inter = primary(ix, corpora), interaction(ix, corpora)
    surv, alloc = survival(ix, corpora), allocation(ix, corpora)
    sec, audit = secondary(ix, corpora), length_audit(ix, corpora)
    out = Path(args.out) if args.out else ROOT / cfg["results_root"] / args.split / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("primary", prim), ("interaction", inter), ("survival", surv),
                       ("allocation", alloc), ("secondary", sec), ("length_audit", audit)):
        write_csv(out / f"{name}.csv", rows)
    (out / "report.md").write_text(report(meta, prim, inter, surv, alloc, sec, audit), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot(out, prim, surv, corpora)
    print(f"analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
