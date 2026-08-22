"""Fixed-ten-passage handoff test with redundant, answer-sufficient evidence.

Each gold passage independently contains the source paragraph for one ordinary
SQuAD question, paired with a distinct real same-article paragraph to avoid
verbatim document duplication. Removed gold passages are replaced with real
cross-article SQuAD distractors that a BM25 index (src/retrieval.py) ranks
highly for the question -- lexically on-topic hard negatives, not a uniformly
random unrelated paragraph -- filtered to not contain an answer alias.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from retrieval import BM25Index, content_fingerprint  # noqa: E402
from negative_verifier import screen, verifier_config  # noqa: E402
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402


def arm_name(negative_type: str, signal_condition: str) -> str:
    """Keep legacy hard-arm ids so their prior cached results remain reusable."""
    return signal_condition if negative_type == "hard" else f"{negative_type}_{signal_condition}"


CONTEXT_LABELS = {
    "signal_1_0": "No noise",
    "signal_0_5": "Medium noise",
    "signal_0_1": "High noise",
}

CONTEXT_COMPOSITIONS = {
    "signal_1_0": "10 answer-sufficient gold passages and 0 distractors.",
    "signal_0_5": "5 answer-sufficient gold passages and 5 distractors.",
    "signal_0_1": "1 answer-sufficient gold passage and 9 distractors.",
}


def arm_specs(cfg: dict) -> list[tuple[str, str, str, int]]:
    return [(arm_name(negative_type, condition), negative_type, condition, relevant)
            for negative_type in cfg["negative_types"]
            for condition, relevant in cfg["conditions"].items()]

SYSTEM = ("You are a research handoff agent. Preserve every fact needed to answer the question. "
          "Your notes replace your entire input for the next agent, so omitted information is lost. "
          "Do not answer the question directly.")
INITIAL = "Write concise research notes from the passages. Preserve answer-relevant facts, qualifiers, numbers, dates, and relationships."
REWRITE = "Rewrite the prior notes concisely. Preserve every fact needed to answer the question, including qualifiers, numbers, dates, and relationships."


def read_jsonl(path: Path) -> list[dict]:
    return [] if not path.exists() else [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_data(cfg: dict) -> Path:
    path = ROOT / cfg["dataset"]["local_parquet"]
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print("[redundant-signal:data] downloading SQuAD validation parquet (no LLM calls)")
    with requests.get(cfg["dataset"]["parquet_url"], stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(path)
    return path


def aliases(value) -> list[str]:
    texts = value.get("text", []) if isinstance(value, dict) else value["text"]
    return list(dict.fromkeys(str(x).strip() for x in texts if str(x).strip()))


def source_rows(cfg: dict) -> list[dict]:
    frame = pd.read_parquet(ensure_data(cfg))
    rows = []
    for row in frame.itertuples(index=False):
        golds, context = aliases(row.answers), " ".join(str(row.context).split())
        if golds and context:
            rows.append({"qid": str(row.id), "title": str(row.title), "question": str(row.question),
                         "context": context, "golds": golds})
    return rows


def build_paragraph_index(source: list[dict]) -> tuple[BM25Index, dict[str, dict]]:
    """BM25 over every unique SQuAD paragraph (title, context), deduplicated.

    Only ~2,000 unique paragraphs in the SQuAD validation split, so the whole
    corpus is indexed -- no bounded pool needed, unlike MS MARCO in
    run_retrieval_quality.py.
    """
    seen: dict[str, dict] = {}
    for row in source:
        if row["context"] not in seen:
            seen[row["context"]] = {"title": row["title"], "context": row["context"]}
    docs = [(f"para:{i}", para["context"]) for i, para in enumerate(seen.values())]
    doc_lookup = {f"para:{i}": para for i, para in enumerate(seen.values())}
    return BM25Index(docs), doc_lookup


def make_base_packs(cfg: dict, n: int, write: bool) -> tuple[list[dict], BM25Index, dict[str, dict]]:
    source = source_rows(cfg)
    by_title: dict[str, list[dict]] = {}
    for row in source:
        by_title.setdefault(row["title"], []).append(row)
    bm25, doc_lookup = build_paragraph_index(source)
    top_k = cfg["dataset"]["bm25_top_k"]
    bottom_k = cfg["dataset"]["bm25_bottom_k"]
    # One source paragraph is the answer-bearing core; each of the ten gold
    # documents also needs a different same-article extension.
    candidates = [row for row in source if len({x["context"] for x in by_title[row["title"]]}) >= cfg["dataset"]["passages_per_query"] + 1]
    rng = random.Random(cfg["dataset"]["sample_seed"])
    rng.shuffle(candidates)
    chosen, used_qids = [], set()
    width = cfg["dataset"]["passages_per_query"]
    for item in candidates:
        if item["qid"] in used_qids:
            continue
        extensions = [x["context"] for x in by_title[item["title"]] if x["context"] != item["context"]]
        extensions = list(dict.fromkeys(extensions))
        if len(extensions) < width:
            continue
        # BM25 hard negatives: paragraphs the retriever ranks highly for this
        # question (lexically on-topic) but that are cross-article and don't
        # contain an answer alias -- a real "retrieved but not relevant"
        # distractor, not a uniformly random unrelated paragraph.
        answer_strings = [a.casefold() for a in item["golds"] if len(a.strip()) > 1]
        retrieved = bm25.top_k(item["question"], top_k)
        hard_negatives = [doc_id for doc_id, _ in retrieved
                          if doc_lookup[doc_id]["title"] != item["title"]
                          and not any(a in doc_lookup[doc_id]["context"].casefold() for a in answer_strings)]
        bottom = bm25.bottom_k(item["question"], bottom_k)
        easy_negatives = [doc_id for doc_id, _ in bottom
                          if doc_lookup[doc_id]["title"] != item["title"]
                          and not any(a in doc_lookup[doc_id]["context"].casefold() for a in answer_strings)]
        if len(hard_negatives) < width - 1 or len(easy_negatives) < width - 1:
            continue
        rng.shuffle(extensions)
        docs = []
        for number, extension in enumerate(extensions[:width], start=1):
            # The first section alone is the original answer-bearing SQuAD
            # paragraph. The second is real, varied same-article context.
            docs.append({"text": f"[Answer-bearing source paragraph]\n{item['context']}\n\n"
                                 f"[Additional passage {number} from the same source article]\n{extension}",
                         "is_relevant": True, "source_title": item["title"]})
        assert len(docs) == width and all(item["context"] in doc["text"] for doc in docs)
        chosen.append({"qid": item["qid"], "question": item["question"], "golds": item["golds"],
                       "title": item["title"], "gold_passages": docs,
                       "hard_negative_ids": hard_negatives, "easy_negative_ids": easy_negatives})
        used_qids.add(item["qid"])
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"only built {len(chosen)}/{n} redundant-evidence packs with >= {width - 1} BM25 hard and easy negatives "
                            f"(raise bm25_top_k if this is short)")
    if write:
        write_jsonl(ROOT / cfg["outputs"]["data_root"] / f"base_packs_n{n}.jsonl", chosen)
    return chosen, bm25, doc_lookup


def condition_pack(base: dict, doc_lookup: dict[str, dict], condition: str, relevant: int,
                   negative_type: str, cfg: dict) -> dict:
    width = cfg["dataset"]["passages_per_query"]
    rng = random.Random(f"{cfg['dataset']['sample_seed']}:{base['qid']}:{condition}")
    indices = list(range(width))
    rng.shuffle(indices)
    keep = set(indices[:relevant])
    neg_order = base[f"{negative_type}_negative_ids"][:]
    neg_seed = (f"{cfg['dataset']['sample_seed']}:{base['qid']}:{condition}:negatives"
                if negative_type == "hard" else
                f"{cfg['dataset']['sample_seed']}:easy:{base['qid']}:{condition}:negatives")
    random.Random(neg_seed).shuffle(neg_order)
    fill_n = width - relevant
    if len(neg_order) < fill_n:
        raise RuntimeError(f"insufficient BM25 {negative_type} negatives for {base['qid']}")
    distractors = neg_order[:fill_n]
    docs = []
    for index in range(width):
        if index in keep:
            docs.append(dict(base["gold_passages"][index]))
        else:
            doc_id = distractors.pop()
            para = doc_lookup[doc_id]
            docs.append({"text": para["context"], "is_relevant": False, "source_title": para["title"]})
    rng.shuffle(docs)
    assert len(docs) == width and sum(doc["is_relevant"] for doc in docs) == relevant
    assert all(not doc["is_relevant"] or base["gold_passages"][0]["text"].split("\n\n", 1)[0] in doc["text"] for doc in docs)
    return {"condition": arm_name(negative_type, condition), "negative_type": negative_type,
            "signal_condition": condition, "relevant_count": relevant, "passages": docs,
            "context_label": CONTEXT_LABELS[condition],
            "context_composition": CONTEXT_COMPOSITIONS[condition],
            **{key: base[key] for key in ("qid", "question", "golds", "title")}}


def verify_negative_pools(base: list[dict], doc_lookup: dict[str, dict], cfg: dict, n: int) -> None:
    """Screen both BM25 pools before any hard/easy negative is selected."""
    spec = verifier_config(cfg)
    if not spec["enabled"]:
        return
    candidates=[]
    for pack in base:
        for negative_type in cfg["negative_types"]:
            for rank, doc_id in enumerate(pack[f"{negative_type}_negative_ids"][:int(spec["candidate_scan_k"])], start=1):
                candidates.append({"qid": pack["qid"], "negative_type": negative_type, "rank": rank,
                                   "doc_id": doc_id, "question": pack["question"], "golds": pack["golds"],
                                   "text": doc_lookup[doc_id]["context"]})
    audited, client = screen(candidates, cfg)
    allowed = {(row["qid"], row["negative_type"]): [] for row in audited}
    for row in audited:
        if row["is_irrelevant"]:
            allowed[(row["qid"], row["negative_type"])].append(row["doc_id"])
    needed = cfg["dataset"]["passages_per_query"] - 1
    for pack in base:
        for negative_type in cfg["negative_types"]:
            verified = allowed.get((pack["qid"], negative_type), [])
            if len(verified) < needed:
                raise RuntimeError(f"only {len(verified)} LLM-screened irrelevant {negative_type} negatives for {pack['qid']}; increase negative_verification.candidate_scan_k")
            pack[f"{negative_type}_negative_ids"] = verified
    write_csv(ROOT / cfg["outputs"]["data_root"] / f"negative_verification_n{n}.csv", audited)
    print("[redundant-signal:negative-verification] " + json.dumps(client.ledger.summary()))


def construct(cfg: dict, n: int, write: bool) -> tuple[list[dict], list[dict]]:
    base, bm25, doc_lookup = make_base_packs(cfg, n, write)
    verify_negative_pools(base, doc_lookup, cfg, n)
    packs, report = [], []
    width = cfg["dataset"]["passages_per_query"]
    for condition_id, negative_type, condition, count in arm_specs(cfg):
        packs.extend(condition_pack(row, doc_lookup, condition, count, negative_type, cfg) for row in base)
        report.append({"condition": condition_id, "negative_type": negative_type,
                       "signal_condition": condition, "context_label": CONTEXT_LABELS[condition],
                       "context_composition": CONTEXT_COMPOSITIONS[condition],
                       "questions": n, "passages_per_query": width,
                       "answer_sufficient_gold_passages": count, "distractors": width - count,
                       "signal_ratio": count / width, "same_question_for_all_gold_passages": True,
                       "source_paragraph_present_in_every_gold_passage": True})
    for negative_type in cfg["negative_types"]:
        assert [row["signal_ratio"] for row in report if row["negative_type"] == negative_type] == [1.0, 0.5, 0.1]
    if write:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"packs_n{n}.jsonl", packs)
        write_csv(root / f"construction_n{n}.csv", report)
    print("[redundant-signal:construction] " + json.dumps(report))
    return packs, report


def context(pack: dict) -> str:
    return "\n\n".join(f"[P{i}] {doc['text']}" for i, doc in enumerate(pack["passages"], start=1))


def compress(client, pack: dict, previous: str | None, cfg: dict, stage: int) -> str:
    if stage == 1:
        user = f"Passages:\n{context(pack)}\n\nQuestion: {pack['question']}\n\n{INITIAL}"
    else:
        user = f"Prior notes:\n{previous}\n\nQuestion: {pack['question']}\n\n{REWRITE}"
    response = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                           temperature=cfg["decoding"]["subagent_temperature"],
                           max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                           tag=f"redundant_signal_{pack['condition']}_stage{stage}")
    return response.text.strip()


def answer(client, pack: dict, material: str, cfg: dict, tag: str) -> dict:
    user = f"Research material:\n{material}\n\nQuestion: {pack['question']}\nAnswer:"
    response = client.chat([{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
                           temperature=cfg["decoding"]["orchestrator_temperature"],
                           max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    prediction = extract_short_answer(response.text)
    em, f1 = score_against_golds(prediction, pack["golds"])
    return {"question": pack["question"], "pred": prediction, "raw": response.text,
            "golds": pack["golds"], "em": em, "f1": f1, "cached": response.cached}


def analyse(rows: list[dict], cfg: dict, root: Path) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    metrics, values = [], {}
    for condition, negative_type, signal_condition, relevant_count in arm_specs(cfg):
        for depth in cfg["depths"]:
            subset = sorted((r for r in rows if r["condition"] == condition and int(r["depth"]) == depth), key=lambda r: r["qid"])
            record = {"condition": condition, "negative_type": negative_type,
                      "signal_condition": signal_condition, "context_label": CONTEXT_LABELS[signal_condition],
                      "context_composition": CONTEXT_COMPOSITIONS[signal_condition],
                      "signal_ratio": relevant_count / 10,
                      "depth": depth, "n": len(subset)}
            values[(condition, depth)] = {}
            for metric in ("em", "f1", "judge_correct"):
                vector = np.array([r[metric] for r in subset])
                values[(condition, depth)][metric] = dict(zip((r["qid"] for r in subset), vector))
                mean, lo, hi = bootstrap_ci(vector, boot, ci, seed=71)
                record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
            metrics.append(record)
    deltas = []
    for condition, negative_type, signal_condition, _ in arm_specs(cfg):
        for depth in (d for d in cfg["depths"] if d):
            for metric in ("em", "f1", "judge_correct"):
                current, baseline = values[(condition, depth)][metric], values[(condition, 0)][metric]
                ids = sorted(set(current) & set(baseline))
                delta = paired_bootstrap_delta(np.array([current[x] for x in ids]), np.array([baseline[x] for x in ids]), boot, ci, seed=73)
                deltas.append({"comparison": "depth_minus_depth0", "condition": condition,
                               "negative_type": negative_type, "signal_condition": signal_condition,
                               "context_label": CONTEXT_LABELS[signal_condition],
                               "context_composition": CONTEXT_COMPOSITIONS[signal_condition],
                               "depth": depth, "metric": metric, **delta})
    # The causal retrieval-quality contrast: all prompts, questions, passage
    # count, and handoff depths are matched; only 10 vs 1 answer-sufficient
    # passages differ.
    for negative_type in cfg["negative_types"]:
        for depth in cfg["depths"]:
            for metric in ("em", "f1", "judge_correct"):
                high = values[(arm_name(negative_type, "signal_1_0"), depth)][metric]
                low = values[(arm_name(negative_type, "signal_0_1"), depth)][metric]
                ids = sorted(set(high) & set(low))
                delta = paired_bootstrap_delta(np.array([high[x] for x in ids]), np.array([low[x] for x in ids]), boot, ci, seed=79)
                deltas.append({"comparison": "signal_1_0_minus_signal_0_1", "condition": f"{negative_type}_signal_1_0_minus_signal_0_1", "negative_type": negative_type, "depth": depth, "metric": metric, **delta})
    for signal_condition in cfg["conditions"]:
        for depth in cfg["depths"]:
            for metric in ("em", "f1", "judge_correct"):
                hard = values[(arm_name("hard", signal_condition), depth)][metric]
                easy = values[(arm_name("easy", signal_condition), depth)][metric]
                ids = sorted(set(hard) & set(easy))
                delta = paired_bootstrap_delta(np.array([hard[x] for x in ids]), np.array([easy[x] for x in ids]), boot, ci, seed=83)
                deltas.append({"comparison": "hard_minus_easy", "condition": signal_condition,
                               "signal_condition": signal_condition,
                               "context_label": CONTEXT_LABELS[signal_condition],
                               "context_composition": CONTEXT_COMPOSITIONS[signal_condition],
                               "depth": depth, "metric": metric, **delta})
    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {condition: f"{CONTEXT_LABELS[condition]} ({CONTEXT_COMPOSITIONS[condition].rstrip('.')})"
              for condition in cfg["conditions"]}
    colors = {"signal_1_0": "#1b9e77", "signal_0_5": "#7570b3", "signal_0_1": "#d95f02"}
    styles = {"hard": "-", "easy": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for condition, negative_type, signal_condition, _ in arm_specs(cfg):
        # With zero distractors, hard/easy contexts are byte-identical. Keep
        # both rows in the metrics/deltas as a control, but draw one line.
        if negative_type == "easy" and signal_condition == "signal_1_0":
            continue
        subset = sorted((r for r in metrics if r["condition"] == condition), key=lambda r: r["depth"])
        x, y = [r["depth"] for r in subset], [r["f1"] for r in subset]
        label = f"{labels[signal_condition]} / {negative_type}"
        axes[0].plot(x, y, marker="o", linestyle=styles[negative_type], color=colors[signal_condition], label=label)
        judge = [r["judge_correct"] for r in subset]
        lower = [r["judge_correct"] - r["judge_correct_lo"] for r in subset]
        upper = [r["judge_correct_hi"] - r["judge_correct"] for r in subset]
        axes[1].errorbar(x, judge, yerr=[lower, upper], marker="o", capsize=3,
                         linestyle=styles[negative_type], color=colors[signal_condition], label=label)
    axes[0].set(title="QA F1: hard vs easy distractors", xlabel="Compression handoffs", ylabel="Token F1")
    axes[1].set(title="LLM-judge answer correctness", xlabel="Compression handoffs", ylabel="Judge accuracy")
    for axis in axes:
        axis.set_xticks(cfg["depths"])
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("LLM-screened distractors: fixed ten-passage answer-sufficient signal")
    fig.tight_layout()
    fig.savefig(root / "redundant_signal_ratio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "redundant_signal_ratio_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_questions"]
    packs, _ = construct(cfg, n, write=not args.dry_run)
    if args.construct_only:
        return 0
    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    result_root.mkdir(parents=True, exist_ok=True)
    # Fingerprint every pack by its actual passage content, not just (condition,
    # qid): the same qid can carry different passages across construction runs
    # (BM25 top-k changes, a different seed, a code fix), and keying the cache
    # on qid alone let a stale cached handoff for an old passage set silently
    # answer a new, unrelated one -- see run_retrieval_quality.py's history.
    for pack in packs:
        pack["fp"] = content_fingerprint([p["text"] for p in pack["passages"]])
    handoff_path = run_root / "handoffs.jsonl"
    handoffs = {(r["condition"], r["qid"], r.get("fp"), int(r["stage"])): r["text"] for r in read_jsonl(handoff_path)}
    for condition, _, _, _ in arm_specs(cfg):
        subset = [p for p in packs if p["condition"] == condition]
        for stage in range(1, cfg["max_depth"] + 1):
            todo = [p for p in subset if (condition, p["qid"], p["fp"], stage) not in handoffs]
            with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
                futures = {pool.submit(compress, client, p, None if stage == 1 else handoffs[(condition, p["qid"], p["fp"], stage - 1)], cfg, stage): p for p in todo}
                for future, pack in futures.items():
                    handoffs[(condition, pack["qid"], pack["fp"], stage)] = future.result()
            if not args.dry_run:
                write_jsonl(handoff_path, [{"condition": c, "qid": qid, "fp": fp, "stage": stage, "text": text} for (c, qid, fp, stage), text in sorted(handoffs.items())])
            print(f"[redundant-signal:handoff] {condition}/stage{stage}: {len(todo)} generated")
    answer_path = run_root / "answers.jsonl"
    existing = {(r["condition"], int(r["depth"]), r["qid"], r.get("fp")): r for r in read_jsonl(answer_path)}
    jobs = []
    for pack in packs:
        for depth in cfg["depths"]:
            key = (pack["condition"], depth, pack["qid"], pack["fp"])
            if key not in existing:
                jobs.append((key, pack, context(pack) if depth == 0 else handoffs[(pack["condition"], pack["qid"], pack["fp"], depth)]))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, pack, material, cfg, f"redundant_signal_{key[0]}_answer_d{key[1]}"): key for key, pack, material in jobs}
        for future, key in futures.items():
            existing[key] = {"condition": key[0], "negative_type": pack["negative_type"],
                             "signal_condition": pack["signal_condition"], "depth": key[1],
                             "qid": key[2], "fp": key[3], **future.result()}
    # Keep only answer rows for the newly constructed screened context. A qid
    # can legitimately recur with a different passage fingerprint after the
    # negative-verification pool changes; stale rows must not enter bootstrap
    # or judge aggregates.
    current_fps = {(pack["condition"], pack["qid"]): pack["fp"] for pack in packs}
    rows = sorted((row for row in existing.values()
                   if current_fps.get((row["condition"], row["qid"])) == row.get("fp")),
                  key=lambda r: (r["condition"], int(r["depth"]), r["qid"]))
    expected_rows = len(packs) * len(cfg["depths"])
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} current answer rows, found {len(rows)}")
    # Rows written before the judge existed carry no question text.
    questions = {p["qid"]: p["question"] for p in packs}
    for row in rows:
        if not row.get("question"):
            row["question"] = questions.get(row["qid"], "")
    if not args.dry_run:
        write_jsonl(answer_path, rows)
        judge_client = add_judge(rows, cfg, tag="redundant_signal_judge")
        write_jsonl(answer_path, rows)
        analyse(rows, cfg, result_root)
        if judge_client is not None:
            print("[redundant-signal:judge cost] " + json.dumps(judge_client.ledger.summary()))
    print(f"[redundant-signal:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
