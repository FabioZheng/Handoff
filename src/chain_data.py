"""Dataset adapters and evidence-length variants for the chain experiment."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pandas as pd
import requests

from data import GoldSentence, Paragraph, Question, build_question


def read_generated_questions(spec: dict, root: Path) -> list[Question]:
    """Load the revision-pinned Wikipedia JSONL produced by the dataset builder."""
    path = root / spec["local_jsonl"]
    if not path.exists():
        raise FileNotFoundError(
            f"Generated dataset is missing: {path}. Build it first with "
            "python src/build_wikipedia_dataset.py"
        )
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    questions = [Question.from_json(row) for row in rows if "qid" in row]
    if not questions:
        raise ValueError(f"Generated dataset {path} contains no Question records")
    return questions


def ensure_parquet(spec: dict, root: Path) -> Path:
    dst = root / spec["local_parquet"]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[chain:data] downloading {spec['parquet_url']}")
    with requests.get(spec["parquet_url"], stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dst)
    return dst


def _as_list(value):
    """Normalise pandas/Arrow nested values without relying on truthiness."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def build_hotpot_question(row, rng_seed: int) -> Question:
    titles = [str(x) for x in _as_list(row["context"]["title"])]
    sentence_lists = [_as_list(x) for x in _as_list(row["context"]["sentences"])]
    support_titles = [str(x) for x in _as_list(row["supporting_facts"]["title"])]
    support_sent_ids = [int(x) for x in _as_list(row["supporting_facts"]["sent_id"])]
    support_pairs = list(zip(support_titles, support_sent_ids))
    support_title_set = set(support_titles)

    order = list(range(len(titles)))
    random.Random(f"{rng_seed}:{row['id']}").shuffle(order)
    paragraphs: list[Paragraph] = []
    title_to_pid: dict[str, int] = {}
    title_to_sentences: dict[str, list[str]] = {}
    for pid, original_idx in enumerate(order, start=1):
        title = titles[original_idx]
        sentences = [str(x) for x in sentence_lists[original_idx]]
        paragraphs.append(Paragraph(
            pid=pid,
            orig_idx=original_idx,
            title=title,
            text=" ".join(" ".join(sentences).split()),
            is_supporting=title in support_title_set,
        ))
        title_to_pid[title] = pid
        title_to_sentences[title] = sentences

    gold_sentences: list[GoldSentence] = []
    for step, (title, sent_id) in enumerate(support_pairs):
        sentences = title_to_sentences.get(title, [])
        if title not in title_to_pid or not (0 <= sent_id < len(sentences)):
            continue
        gold_sentences.append(GoldSentence(
            step=step,
            pid=title_to_pid[title],
            text=" ".join(str(sentences[sent_id]).split()),
            hop_answer=str(row["answer"]),
            is_final_hop=(step == len(support_pairs) - 1),
        ))

    return Question(
        qid=str(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        aliases=[],
        paragraphs=paragraphs,
        decomposition=[],
        gold_pids=sorted(p.pid for p in paragraphs if p.is_supporting),
        gold_sentences=gold_sentences,
        n_hops=len(support_title_set),
    )


def sample_candidates(dataset: str, spec: dict, root: Path, seed: int, n: int) -> list[Question]:
    if spec.get("source") == "generated_wikipedia":
        questions = read_generated_questions(spec, root)
        # Keep paired questions from the same page adjacent, rather than randomly
        # separating them. The builder's stored order is the reproducible sample.
        print(f"[chain:data] {dataset}: loaded {min(n, len(questions))}/{len(questions)} generated questions")
        return questions[: min(n, len(questions))]
    path = ensure_parquet(spec, root)
    frame = pd.read_parquet(path)
    if dataset == "musique" and spec.get("answerable_only", True):
        frame = frame[frame["answerable"] == True]  # noqa: E712
    frame = frame.reset_index(drop=True)
    indices = list(range(len(frame)))
    # Preserve the original MuSiQue sample order so its closed-book calls can
    # reuse the baseline experiment's cache. HotpotQA gets an independent stream.
    sample_seed = seed if dataset == "musique" else f"{seed}:{dataset}"
    random.Random(sample_seed).shuffle(indices)
    indices = indices[: min(n, len(indices))]
    builder = build_question if dataset == "musique" else build_hotpot_question
    questions = [builder(frame.iloc[i], seed) for i in indices]
    print(f"[chain:data] {dataset}: sampled {len(questions)} candidates from {len(frame)}")
    return questions


def select_paragraphs(question: Question, variant: str, context_cfg: dict) -> list[Paragraph]:
    """Keep all gold documents and deterministically vary only distractor count."""
    gold = [p for p in question.paragraphs if p.is_supporting]
    distractors = [p for p in question.paragraphs if not p.is_supporting]
    limit = context_cfg[variant]["max_documents"]
    if limit == "gold":
        chosen = gold
    elif limit == "all":
        chosen = question.paragraphs
    else:
        chosen_pids = {p.pid for p in gold}
        chosen_pids.update(p.pid for p in distractors[: max(0, int(limit) - len(gold))])
        chosen = [p for p in question.paragraphs if p.pid in chosen_pids]
    return sorted(chosen, key=lambda p: p.pid)


def render_context(question: Question, variant: str, context_cfg: dict) -> str:
    return "\n\n".join(
        f"[P{p.pid}] {p.title}\n{p.text}" for p in select_paragraphs(question, variant, context_cfg)
    )


def context_stats(question: Question, variant: str, context_cfg: dict) -> dict:
    paragraphs = select_paragraphs(question, variant, context_cfg)
    text = "\n\n".join(p.text for p in paragraphs)
    return {
        "context_documents": len(paragraphs),
        "context_characters": len(text),
        "gold_documents": len(question.gold_pids),
    }


def write_questions(questions: list[Question], path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_manifest": manifest}, ensure_ascii=False) + "\n")
        for question in questions:
            fh.write(json.dumps(question.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_questions(path: Path, expected_manifest: dict) -> list[Question] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
        if first.get("_manifest") != expected_manifest:
            return None
        return [Question.from_json(json.loads(line)) for line in fh if line.strip()]


def filter_config(base_cfg: dict, n_target: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["sampling"]["n_target"] = n_target
    return cfg
