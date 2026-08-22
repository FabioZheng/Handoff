"""Build same-passage A/B question pairs from SQuAD for the conditioning experiment.

Replaces the generated-Wikipedia variant of Experiment 5, which was confounded:
its questions were deliberately written to target "details that are not obvious",
its passages were whole Wikipedia articles ~10x longer than the summary budget,
and its gold passage position was unstratified, so the generic arm was truncated
before ever reaching the evidence and scored a degenerate 0.

Every fairness control that build lacked is enforced here:

* One SQuAD passage answers BOTH questions (same-source pairing), so conditioning
  cannot simply discard a separate gold document.
* Questions are human-written SQuAD questions about that paragraph's main
  content -- never model-authored, never selected for obscurity.
* Passage length is banded and total context length matched across examples.
* Gold position is stratified uniformly across the ten slots.
* A/B independence and salience are audited by an LLM in a different family from
  the system under test, and both questions must pass the project's own
  closed-book C1 leakage filter.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
from data import Paragraph, Question  # noqa: E402
from judge import make_judge_client  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from negative_verifier import screen, verifier_config  # noqa: E402

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
             "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what",
             "when", "where", "which", "who", "why", "with", "many", "much", "name", "whose"}

VALIDATOR_SYSTEM = (
    "You audit question pairs for a reading-comprehension experiment. You are strict and "
    "reply only with the requested JSON."
)

VALIDATOR_TEMPLATE = """Passage:
{context}

Question A: {qa}
Answer A: {aa}

Question B: {qb}
Answer B: {ab}

Judge three things independently. Do not assume the answer is yes.

1. a_salient: is Question A about a reasonably PROMINENT fact of this passage -- something a
   competent one-paragraph summary written for a general reader would be expected to retain?
   Answer false if it targets an incidental aside: a detail mentioned only in passing, a minor
   parenthetical, or an isolated date/number/name that is not part of the passage's main point.
2. b_salient: the same test, applied to Question B.
3. independent: would knowing the answer to one question fail to give away, strongly imply, or
   make obvious the answer to the other? Answer false if both really probe the same underlying
   fact, even when worded differently.

In "fact_a" and "fact_b", state in a few words which fact each question targets, so the
salience calls can be checked.

Reply with JSON only, using this schema:
{{"fact_a": "<few words>", "fact_b": "<few words>", "a_salient": <true|false>,
  "b_salient": <true|false>, "independent": <true|false>}}"""


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def content_tokens(text: str) -> set[str]:
    return {t for t in norm(text).split() if t not in STOPWORDS and len(t) > 1}


def jaccard(a: str, b: str) -> float:
    x, y = content_tokens(a), content_tokens(b)
    return len(x & y) / len(x | y) if (x | y) else 0.0


def answers_of(value) -> list[str]:
    texts = value.get("text", []) if isinstance(value, dict) else value["text"]
    return list(dict.fromkeys(str(t).strip() for t in texts if str(t).strip()))


def candidate_pairs(cfg: dict, rng: random.Random) -> list[dict]:
    """Every (context, qA, qB) satisfying the deterministic structural filters."""
    ds = cfg["dataset"]
    frame = pd.read_parquet(ROOT / ds["local_parquet"])
    lo, hi = ds["passage_chars_min"], ds["passage_chars_max"]
    max_ans_tokens = ds["max_answer_tokens"]
    max_j = ds["max_question_jaccard"]

    by_context: dict[str, list[dict]] = collections.defaultdict(list)
    for row in frame.itertuples(index=False):
        context = " ".join(str(row.context).split())
        if not (lo <= len(context) <= hi):
            continue
        golds = answers_of(row.answers)
        if not golds:
            continue
        # The answer must be a verbatim span and short enough to be a crisp target.
        if norm(golds[0]) not in norm(context) or len(norm(golds[0]).split()) > max_ans_tokens:
            continue
        by_context[context].append(
            {"qid": str(row.id), "title": str(row.title), "question": str(row.question), "golds": golds})

    out: list[dict] = []
    for context, items in by_context.items():
        if len(items) < 2:
            continue
        ctx_tokens = content_tokens(context)
        items = sorted(items, key=lambda r: r["qid"])
        best = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                # Distinct facts: no shared answer tokens, neither answer inside the other.
                if content_tokens(" ".join(a["golds"])) & content_tokens(" ".join(b["golds"])):
                    continue
                na, nb = norm(a["golds"][0]), norm(b["golds"][0])
                if na in nb or nb in na:
                    continue
                qj = jaccard(a["question"], b["question"])
                if qj > max_j:
                    continue
                # Salience proxy: each question must be anchored in the passage's own
                # wording rather than being a peripheral aside.
                if len(content_tokens(a["question"]) & ctx_tokens) < 2:
                    continue
                if len(content_tokens(b["question"]) & ctx_tokens) < 2:
                    continue
                if best is None or qj < best[0]:
                    best = (qj, a, b)
        if best is not None:
            qj, a, b = best
            out.append({"context": context, "title": a["title"],
                        "question_jaccard": round(qj, 4), "A": a, "B": b})
    rng.shuffle(out)
    return out


def validate_pair(client, cand: dict) -> tuple[bool, str]:
    """LLM audit of the two requirements no regex can express: independence and salience."""
    prompt = VALIDATOR_TEMPLATE.format(
        context=cand["context"], qa=cand["A"]["question"], aa=cand["A"]["golds"][0],
        qb=cand["B"]["question"], ab=cand["B"]["golds"][0])
    result = client.chat(
        [{"role": "system", "content": VALIDATOR_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=200, seed=None, tag="squad_pair_validate_v2")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.text.strip(), flags=re.I)
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return False, "unparsable validator response"
    ok = (bool(obj.get("independent")) and bool(obj.get("a_salient"))
          and bool(obj.get("b_salient")))
    detail = f"A={obj.get('fact_a', '?')} | B={obj.get('fact_b', '?')}"
    return ok, detail[:120]


def as_question(item: dict, context: str, title: str) -> Question:
    """Wrap one side of a pair so the existing C1 leakage filter can score it."""
    return Question(qid=item["qid"], question=item["question"], answer=item["golds"][0],
                    aliases=item["golds"][1:], paragraphs=[Paragraph(1, 0, title, context, True)],
                    decomposition=[], gold_pids=[1], gold_sentences=[], n_hops=1)


def verify_distractors(cfg: dict, clean: list[dict], source: list[dict],
                       rng: random.Random) -> tuple[dict[str, list[dict]], list[dict], LLMClient | None]:
    """Require every candidate distractor to be irrelevant to both A and B."""
    scan_k = int(verifier_config(cfg)["candidate_scan_k"])
    candidates_by_pair: dict[str, list[dict]] = {}
    audit_jobs: list[dict] = []
    for cand in clean:
        pair_id = f"{cand['A']['qid']}__{cand['B']['qid']}"
        pool = [d for d in source if d["title"] != cand["title"]]
        sampled = rng.sample(pool, min(scan_k, len(pool)))
        candidates_by_pair[pair_id] = sampled
        for distractor in sampled:
            candidate_id = hashlib.sha256(distractor["context"].encode("utf-8")).hexdigest()[:16]
            for side in ("A", "B"):
                audit_jobs.append({
                    "pair_id": pair_id, "query_side": side,
                    "target_qid": cand[side]["qid"],
                    "question": cand[side]["question"], "golds": cand[side]["golds"],
                    "candidate_id": candidate_id, "candidate_title": distractor["title"],
                    "text": distractor["context"],
                })
    audited, client = screen(audit_jobs, cfg)
    verdicts = {(r["pair_id"], r["candidate_id"], r["query_side"]): r["is_irrelevant"]
                for r in audited}
    eligible: dict[str, list[dict]] = {}
    for pair_id, candidates in candidates_by_pair.items():
        eligible[pair_id] = []
        for distractor in candidates:
            candidate_id = hashlib.sha256(distractor["context"].encode("utf-8")).hexdigest()[:16]
            if (verdicts.get((pair_id, candidate_id, "A"), False)
                    and verdicts.get((pair_id, candidate_id, "B"), False)):
                eligible[pair_id].append(distractor)
    return eligible, audited, client


def build(cfg: dict, n: int) -> tuple[list[dict], dict, list[dict], LLMClient | None]:
    ds = cfg["dataset"]
    rng = random.Random(ds["sample_seed"])
    cands = candidate_pairs(cfg, rng)
    print(f"[squad_pairs] {len(cands)} contexts pass the structural filters")

    validator, _ = make_judge_client(cfg)
    accepted: list[dict] = []
    n_rejected = 0
    for cand in cands[: ds["max_candidates"]]:
        if len(accepted) >= ds["validated_target"]:
            break
        ok, reason = validate_pair(validator, cand)
        if ok:
            accepted.append(cand)
        else:
            n_rejected += 1
    print(f"[squad_pairs] independence/salience audit: {len(accepted)} kept, {n_rejected} rejected")

    # C1 closed-book leakage, run with the model actually under test, on BOTH sides.
    test_client = LLMClient(cfg)
    probes = []
    for cand in accepted:
        probes.append(as_question(cand["A"], cand["context"], cand["title"]))
        probes.append(as_question(cand["B"], cand["context"], cand["title"]))
    c1_cfg = dict(cfg)
    c1_cfg["sampling"] = {"n_target": len(probes)}
    _, c1 = data_mod.apply_c1(test_client, probes, c1_cfg, cfg["runtime"]["concurrency"])
    leaked = set(c1["leaked_qids"])
    clean = [c for c in accepted if c["A"]["qid"] not in leaked and c["B"]["qid"] not in leaked]
    print(f"[squad_pairs] C1 leakage ({c1['known_method']}, model={cfg['model']['id']}): "
          f"{c1['leaked_excluded']}/{len(probes)} questions leaked -> "
          f"{len(clean)}/{len(accepted)} pairs fully clean")
    if len(clean) < n:
        raise SystemExit(f"only {len(clean)} clean pairs, need {n}; raise dataset.validated_target")
    clean = clean[:n]

    used_contexts = {c["context"] for c in clean}
    distract_source = [c for c in cands if c["context"] not in used_contexts]
    width = ds["context_passages"]
    eligible, negative_audit, negative_client = verify_distractors(
        cfg, clean, distract_source, rng)
    minimum = min(len(rows) for rows in eligible.values())
    if minimum < width - 1:
        raise SystemExit(
            f"only {minimum} jointly irrelevant distractors for at least one pair; "
            "raise negative_verification.candidate_scan_k")
    print(f"[squad_pairs] negative verification: {len(negative_audit)} A/B verdicts; "
          f"minimum {minimum} jointly irrelevant candidates per pair")

    pairs = []
    for index, cand in enumerate(clean):
        pair_id = f"{cand['A']['qid']}__{cand['B']['qid']}"
        distractors = eligible[pair_id][:width - 1]
        # Stratified gold position: each of the ten slots is used equally often.
        gold_slot = index % width
        passages = [{"text": d["context"], "role": "distractor", "source_title": d["title"]}
                    for d in distractors]
        passages.insert(gold_slot, {"text": cand["context"], "role": "gold_AB",
                                    "source_title": cand["title"]})
        pairs.append({
            "pair_id": pair_id,
            "question_A": cand["A"]["question"], "golds_A": cand["A"]["golds"], "qid_A": cand["A"]["qid"],
            "question_B": cand["B"]["question"], "golds_B": cand["B"]["golds"], "qid_B": cand["B"]["qid"],
            "title_A": cand["title"], "title_B": cand["title"],
            "question_jaccard": cand["question_jaccard"],
            "gold_position": gold_slot + 1,
            "context_chars": sum(len(p["text"]) for p in passages),
            "passages": passages,
            "context_note": (f"One shared SQuAD passage ({cand['title']}) answers both questions, "
                             f"placed at P{gold_slot + 1} among {width - 1} length-matched "
                             "SQuAD passages LLM-screened irrelevant to both A and B."),
        })
        assert len(passages) == width
        assert sum(p["role"] == "gold_AB" for p in passages) == 1
    return pairs, c1, negative_audit, negative_client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "summary_generalization_squad_pairs_config.yaml"))
    ap.add_argument("--n", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_pairs"]
    out = ROOT / cfg["outputs"]["data_root"] / f"pairs_n{n}.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to rebuild")

    pairs, c1, negative_audit, negative_client = build(cfg, n)
    chars = [p["context_chars"] for p in pairs]
    gold_chars = [len(next(x for x in p["passages"] if x["role"] == "gold_AB")["text"]) for p in pairs]
    report = {
        "pairs": len(pairs), "passages_per_pair": cfg["dataset"]["context_passages"],
        "same_passage_answers_both": len(pairs),
        "questions_model_authored": 0,
        "gold_positions_used": len(set(p["gold_position"] for p in pairs)),
        "gold_position_counts": json.dumps(dict(sorted(
            collections.Counter(p["gold_position"] for p in pairs).items()))),
        "gold_passage_chars_min": min(gold_chars), "gold_passage_chars_max": max(gold_chars),
        "context_chars_min": min(chars), "context_chars_max": max(chars),
        "context_chars_spread_pct": round(100 * (max(chars) - min(chars)) / min(chars), 2),
        "mean_question_jaccard": round(sum(p["question_jaccard"] for p in pairs) / len(pairs), 4),
        "max_question_jaccard": max(p["question_jaccard"] for p in pairs),
        "c1_questions_probed": c1["candidates_run"], "c1_questions_leaked": c1["leaked_excluded"],
        "c1_method": c1["known_method"],
        "negative_verifier_model": verifier_config(cfg)["model_id"],
        "negative_verdicts": len(negative_audit),
        "negative_candidate_passages": len(negative_audit) // 2,
        "selected_distractor_placements": n * (cfg["dataset"]["context_passages"] - 1),
        "selected_distractors_screened_against_both": n * (cfg["dataset"]["context_passages"] - 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    rpath = out.parent / f"construction_n{n}.csv"
    with open(rpath, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report))
        w.writeheader(); w.writerow(report)
    apath = out.parent / f"negative_verification_n{n}.csv"
    with open(apath, "w", encoding="utf-8", newline="") as fh:
        fields = list(dict.fromkeys(key for row in negative_audit for key in row))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(negative_audit)
    print("[squad_pairs:construction] " + json.dumps(report))
    if negative_client is not None:
        print("[squad_pairs:negative-verification cost] "
              + json.dumps(negative_client.ledger.summary()))
    print(f"[squad_pairs] wrote {len(pairs)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
