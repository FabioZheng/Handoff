"""Stage 0: load MuSiQue-Answerable, sample with a fixed seed, apply the C1 filter.

C1 is the load-bearing step. If the model can already answer a question from
pretraining, corrupting the handoff cannot hurt it and that question measures
nothing. Every candidate is run closed-book (no evidence) three times at
temperature 0.7 and kept only if all three attempts fail.
"""

from __future__ import annotations

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


def sample_candidates(cfg: dict, repo_root: Path) -> list[Question]:
    import pandas as pd

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
        out.append(
            {
                "sample": k,
                "pred": pred,
                "em": em,
                "f1": f1,
                "known": bool(em == 1.0 or f1 >= lf["f1_known_threshold"]),
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "cached": res.cached,
            }
        )
    return out


def apply_c1(client, candidates: list[Question], cfg: dict, concurrency: int) -> tuple[list[Question], dict]:
    """Run the closed-book filter and return (survivors, filter report)."""
    from concurrent.futures import ThreadPoolExecutor

    n_target = cfg["sampling"]["n_target"]
    records: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(closed_book_attempts, client, q, cfg): q for q in candidates}
        for fut, q in futs.items():
            records[q.qid] = fut.result()

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
        "candidates_run": len(candidates),
        "leaked_excluded": len(leaked),
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


def write_filtered(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")
    print(f"[data] wrote {len(questions)} questions -> {path}")


def read_filtered(path: Path) -> list[Question]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Question.from_json(json.loads(line)))
    return out
