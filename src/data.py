"""Stage 0: load MuSiQue-Answerable, sample with a fixed seed, apply the C1 filter.

C1 is the load-bearing step. If the model can already answer a question from
pretraining, corrupting the handoff cannot hurt it and that question measures
nothing. Every candidate is run closed-book (no evidence) three times at
temperature 0.7 and kept only if all three attempts fail.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

from score import score_against_golds, extract_short_answer

CLOSED_BOOK_SYSTEM = (
    "You are answering a factual question from memory. "
    "Reply with only the short answer span - no explanation, no full sentence."
)


@dataclass
class Paragraph:
    pid: int            # presentation id, 1-based, in the fixed shuffled order
    orig_idx: int       # original MuSiQue paragraph index
    title: str
    text: str
    is_supporting: bool


@dataclass
class GoldSentence:
    step: int           # decomposition step index, 0-based
    pid: int            # presentation id of the paragraph it came from
    text: str
    hop_answer: str
    is_final_hop: bool


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    aliases: list[str]
    paragraphs: list[Paragraph]
    decomposition: list[dict]
    gold_pids: list[int]
    gold_sentences: list[GoldSentence]
    n_hops: int

    @property
    def golds(self) -> list[str]:
        return [self.answer] + list(self.aliases)

    def to_json(self) -> dict:
        d = asdict(self)
        d["golds"] = self.golds
        return d

    @staticmethod
    def from_json(d: dict) -> "Question":
        return Question(
            qid=d["qid"],
            question=d["question"],
            answer=d["answer"],
            aliases=d["aliases"],
            paragraphs=[Paragraph(**p) for p in d["paragraphs"]],
            decomposition=d["decomposition"],
            gold_pids=d["gold_pids"],
            gold_sentences=[GoldSentence(**s) for s in d["gold_sentences"]],
            n_hops=d["n_hops"],
        )


# ---------------- sentence splitting ----------------

_SENT_SPLIT = re.compile(
    r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])"
)
_ABBREV_FIX = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|No|Inc|Ltd|Co|Mt|Rev|Prof|Gen|Sen|Rep)\.$")


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter. No extra dependency; good enough for encyclopedic prose.

    Fragments ending in a known abbreviation are re-joined with the next fragment.
    """
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        if out and (_ABBREV_FIX.search(out[-1]) or len(out[-1]) < 12):
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]


def _norm_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def _contains(haystack: str, needle: str) -> bool:
    h, n = _norm_for_match(haystack), _norm_for_match(needle)
    h, n = " ".join(h.split()), " ".join(n.split())
    return bool(n) and n in h


# ---------------- loading ----------------

def ensure_parquet(cfg: dict, repo_root: Path) -> Path:
    dst = repo_root / cfg["dataset"]["local_parquet"]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["dataset"]["parquet_url"]
    print(f"[data] downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dst)
    print(f"[data] saved {dst} ({dst.stat().st_size} bytes)")
    return dst


def build_question(row, rng_seed: int) -> Question:
    """Convert one MuSiQue row into a Question with a fixed shuffled paragraph order.

    Presentation ids (P1..Pn) are assigned *after* shuffling, so provenance ids the
    model emits are stable and the gold/distractor split is not guessable from the id.
    """
    raw_paras = list(row["paragraphs"])
    order = list(range(len(raw_paras)))
    # Deterministic per-question shuffle: same order on every run, every machine.
    random.Random(f"{rng_seed}:{row['id']}").shuffle(order)

    paragraphs: list[Paragraph] = []
    orig_to_pid: dict[int, int] = {}
    for pid, oi in enumerate(order, start=1):
        p = raw_paras[oi]
        paragraphs.append(
            Paragraph(
                pid=pid,
                orig_idx=int(p["idx"]),
                title=str(p["title"]),
                text=" ".join(str(p["paragraph_text"]).split()),
                is_supporting=bool(p["is_supporting"]),
            )
        )
        orig_to_pid[int(p["idx"])] = pid

    decomposition = [
        {
            "step": i,
            "question": str(d["question"]),
            "answer": str(d["answer"]),
            "support_pid": orig_to_pid[int(d["paragraph_support_idx"])],
        }
        for i, d in enumerate(row["question_decomposition"])
    ]

    # MuSiQue annotates supporting *paragraphs*, not sentences. Gold sentences are
    # derived deterministically from the decomposition: within each hop's supporting
    # paragraph, keep the sentences that contain that hop's answer string.
    n_steps = len(decomposition)
    gold_sentences: list[GoldSentence] = []
    for d in decomposition:
        para = paragraphs[d["support_pid"] - 1]
        sents = split_sentences(para.text)
        hit = [s for s in sents if _contains(s, d["answer"])]
        if not hit:
            # Fall back to the sentence bearing the hop's subject, else the lead sentence.
            subject = d["question"].split(">>")[0].strip()
            hit = [s for s in sents if subject and _contains(s, subject)]
        if not hit:
            hit = sents[:1]
        for s in hit:
            gold_sentences.append(
                GoldSentence(
                    step=d["step"],
                    pid=para.pid,
                    text=s,
                    hop_answer=d["answer"],
                    is_final_hop=(d["step"] == n_steps - 1),
                )
            )

    return Question(
        qid=str(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        aliases=[str(a) for a in list(row["answer_aliases"])],
        paragraphs=paragraphs,
        decomposition=decomposition,
        gold_pids=sorted(p.pid for p in paragraphs if p.is_supporting),
        gold_sentences=gold_sentences,
        n_hops=n_steps,
    )


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage0_manifest(cfg: dict, repo_root: Path) -> dict:
    """Everything that determines the C1-filtered survivor set.

    Written as the first line of ``filtered_questions.jsonl`` (same convention
    as ``chain_data.write_questions`` and the generated Wikipedia dataset
    files) so a cached filter output can be validated against the *current*
    config and source data, not just trusted because it exists on disk. The
    source file is content-hashed rather than just path-compared: swapping in
    a rebuilt dataset at the same path (as happened when the Wikipedia
    dataset was regenerated) must invalidate the cache even though every
    config value is unchanged.
    """
    ds = cfg["dataset"]
    if ds.get("source") == "generated_wikipedia":
        source_path = ds["local_jsonl"]
    else:
        source_path = ds.get("local_parquet")
    judge = cfg.get("judge") or {}
    return {
        "dataset_source": ds.get("source", "musique"),
        "dataset_name": ds.get("name"),
        "source_path": source_path,
        "source_hash": _hash_file(repo_root / source_path) if source_path else None,
        "model_id": cfg["model"]["id"],
        "sampling_seed": cfg["sampling"]["seed"],
        "n_candidates": cfg["sampling"]["n_candidates"],
        "n_target": cfg["sampling"]["n_target"],
        "leakage_filter": cfg["leakage_filter"],
        "judge_enabled": bool(judge.get("enabled")),
        "judge_model_id": judge.get("model_id") if judge.get("enabled") else None,
    }


def sample_candidates(cfg: dict, repo_root: Path) -> list[Question]:
    import pandas as pd

    if cfg["dataset"].get("source") == "generated_wikipedia":
        path = repo_root / cfg["dataset"]["local_jsonl"]
        if not path.exists():
            raise FileNotFoundError(
                f"Generated dataset is missing: {path}. Build it first with "
                "python src/build_wikipedia_dataset.py"
            )
        with open(path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        loaded = [Question.from_json(row) for row in rows if "qid" in row]
        n_cand = min(cfg["sampling"]["n_candidates"], len(loaded))
        print(f"[data] loaded {n_cand}/{len(loaded)} generated Wikipedia questions")
        return loaded[:n_cand]

    path = ensure_parquet(cfg, repo_root)
    df = pd.read_parquet(path)
    if cfg["dataset"]["answerable_only"]:
        df = df[df["answerable"] == True]  # noqa: E712
    df = df.reset_index(drop=True)

    seed = cfg["sampling"]["seed"]
    n_cand = min(cfg["sampling"]["n_candidates"], len(df))
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_cand]

    qs = [build_question(df.iloc[i], seed) for i in idx]
    print(f"[data] sampled {len(qs)} candidates from {len(df)} answerable questions (seed={seed})")
    return qs


# ---------------- C1: parametric leakage filter ----------------

def closed_book_attempts(client, q: Question, cfg: dict) -> list[dict]:
    lf = cfg["leakage_filter"]
    messages = [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"Question: {q.question}\nAnswer:"},
    ]
    out = []
    for k in range(lf["samples"]):
        res = client.chat(
            messages,
            temperature=lf["temperature"],
            max_tokens=lf["max_tokens"],
            seed=1000 + k,
            tag="c1_closed_book",
        )
        pred = extract_short_answer(res.text)
        em, f1 = score_against_golds(pred, q.golds)
        # "known" is filled in by apply_c1 once every attempt has been generated:
        # by LLM judge when enabled, else by the em/f1 fallback below.
        out.append(
            {
                "sample": k,
                "pred": pred,
                "em": em,
                "f1": f1,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "cached": res.cached,
            }
        )
    return out


def apply_c1(
    client, candidates: list[Question], cfg: dict, concurrency: int, dry_run: bool = False,
) -> tuple[list[Question], dict]:
    """Run the closed-book filter and return (survivors, filter report).

    A closed-book attempt counts as "the model already knows this" via an LLM
    judge verdict (semantic match against the gold, same judge used for the
    main experiment's answers) when ``judge.enabled`` in config, since token-F1
    is a poor leakage signal for short entity-name answers: generic phrase
    overlap (e.g. "... University School of Medicine") clears a fixed F1
    threshold even when the model named the wrong entity. Falls back to the
    em/f1 threshold only if the judge is disabled.
    """
    import judge as judge_mod
    from concurrent.futures import ThreadPoolExecutor

    n_target = cfg["sampling"]["n_target"]
    lf = cfg["leakage_filter"]
    records: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(closed_book_attempts, client, q, cfg): q for q in candidates}
        for fut, q in futs.items():
            records[q.qid] = fut.result()

    qmap = {q.qid: q for q in candidates}
    use_judge = judge_mod.judge_config(cfg)["enabled"] and not dry_run

    if use_judge:
        rows = [
            {"qid": qid, "sample": a["sample"], "question": qmap[qid].question,
             "pred": a["pred"], "golds": qmap[qid].golds}
            for qid, attempts in records.items() for a in attempts
        ]
        judge_mod.add_judge(rows, cfg, dry_run=dry_run, tag="c1_leakage_judge")
        verdicts = {(r["qid"], r["sample"]): bool(r["judge_correct"]) for r in rows}
        for qid, attempts in records.items():
            for a in attempts:
                a["judge_correct"] = verdicts[(qid, a["sample"])]
                a["known"] = a["judge_correct"]
        known_method = "llm_judge"
    else:
        for attempts in records.values():
            for a in attempts:
                a["known"] = bool(a["em"] == 1.0 or a["f1"] >= lf["f1_known_threshold"])
        known_method = "f1_threshold"

    survivors: list[Question] = []
    leaked: list[str] = []
    for q in candidates:  # preserve the seeded sample order -> deterministic survivor set
        att = records[q.qid]
        if any(a["known"] for a in att):
            leaked.append(q.qid)
        else:
            survivors.append(q)

    kept = survivors[:n_target]
    report = {
        "known_method": known_method,
        "candidates_run": len(candidates),
        "leaked_excluded": len(leaked),
        "leaked_qids": leaked,
        "survived": len(survivors),
        "survival_rate": len(survivors) / len(candidates) if candidates else 0.0,
        "leak_rate": len(leaked) / len(candidates) if candidates else 0.0,
        "n_target": n_target,
        "kept": len(kept),
        "target_met": len(kept) >= n_target,
        "closed_book_mean_em": sum(a["em"] for r in records.values() for a in r)
        / max(1, sum(len(r) for r in records.values())),
        "closed_book_mean_f1": sum(a["f1"] for r in records.values() for a in r)
        / max(1, sum(len(r) for r in records.values())),
        "attempts": records,
    }
    return kept, report


def write_filtered(questions: list[Question], path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_manifest": manifest}, ensure_ascii=False) + "\n")
        for q in questions:
            fh.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"[data] wrote {len(questions)} questions -> {path}")


def read_filtered(path: Path, expected_manifest: dict) -> list[Question] | None:
    """Return survivors from ``path``, or None if missing/stale.

    Stale covers both a manifest that no longer matches ``expected_manifest``
    (dataset source, model, sampling, or filter config changed) and a legacy
    file with no manifest line at all -- the caller should treat None exactly
    like a cache miss and regenerate.
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
        if not first_line:
            return None
        first = json.loads(first_line)
        if "_manifest" not in first:
            return None  # legacy file predating manifest validation
        if first["_manifest"] != expected_manifest:
            return None
        out = []
        for line in fh:
            line = line.strip()
            if line:
                out.append(Question.from_json(json.loads(line)))
        return out
