"""Test whether question conditioning narrows summaries over repeated handoffs.

Each example pairs two low-overlap SQuAD questions from different articles.
Their two gold passages and eight real SQuAD distractor passages form one fixed
top-10 context.  The conditioned chain sees only question A at every handoff;
the generic chain sees no question.  Fresh answerers test both A and held-out B
at depths 0/1/3/5.  Thus B is unrelated to A but answerable from the same
original retrieved context.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from judge import add_judge  # noqa: E402
from run_chain import (  # noqa: E402
    CHAIN_SYSTEM,
    INITIAL_INSTRUCTION,
    RECOMPRESS_INSTRUCTION,
)
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with"}


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
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_parquet(cfg: dict) -> Path:
    path = ROOT / cfg["dataset"]["local_parquet"]
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print("[generalization:data] downloading SQuAD validation parquet (no LLM calls)")
    with requests.get(cfg["dataset"]["parquet_url"], stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(path)
    return path


def tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOPWORDS and len(x) > 1}


def jaccard(a: str, b: str) -> float:
    x, y = tokens(a), tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def normalize_answers(value) -> list[str]:
    texts = value.get("text", []) if isinstance(value, dict) else value["text"]
    return list(dict.fromkeys(str(x).strip() for x in list(texts) if str(x).strip()))


def construct_pairs(cfg: dict, n: int, write: bool = True) -> list[dict]:
    frame = pd.read_parquet(ensure_parquet(cfg))
    records = [{"qid": str(r.id), "title": str(r.title), "question": str(r.question),
                "context": " ".join(str(r.context).split()), "golds": normalize_answers(r.answers)}
               for r in frame.itertuples(index=False)]
    records = [r for r in records if r["golds"] and r["context"]]
    for record in records:
        record["question_tokens"] = tokens(record["question"])
        record["answer_tokens"] = tokens(" ".join(record["golds"]))
    rng = random.Random(cfg["dataset"]["sample_seed"])
    order = list(range(len(records)))
    rng.shuffle(order)
    used: set[int] = set()
    pairs: list[dict] = []
    max_j = float(cfg["dataset"]["max_question_jaccard"])
    for ai in order:
        if ai in used:
            continue
        a = records[ai]
        bi = next((candidate for candidate in order if candidate not in used and candidate != ai
                   and records[candidate]["title"] != a["title"]
                   and records[candidate]["context"] != a["context"]
                   and len(a["question_tokens"] & records[candidate]["question_tokens"])
                   / max(1, len(a["question_tokens"] | records[candidate]["question_tokens"])) <= max_j
                   and not (a["answer_tokens"] & records[candidate]["answer_tokens"])), None)
        if bi is None:
            continue
        b = records[bi]
        distractor_pool = [r for i, r in enumerate(records) if i not in (ai, bi)
                           and r["title"] not in (a["title"], b["title"])
                           and r["context"] not in (a["context"], b["context"])]
        distractors = rng.sample(distractor_pool, cfg["dataset"]["context_passages"] - 2)
        passages = [{"text": a["context"], "role": "gold_A", "source_title": a["title"]},
                    {"text": b["context"], "role": "gold_B", "source_title": b["title"]}]
        passages += [{"text": d["context"], "role": "distractor", "source_title": d["title"]} for d in distractors]
        rng.shuffle(passages)
        pair = {"pair_id": f"{a['qid']}__{b['qid']}",
                "question_A": a["question"], "golds_A": a["golds"], "qid_A": a["qid"],
                "question_B": b["question"], "golds_B": b["golds"], "qid_B": b["qid"],
                "title_A": a["title"], "title_B": b["title"],
                "question_jaccard": jaccard(a["question"], b["question"]), "passages": passages}
        assert len(passages) == cfg["dataset"]["context_passages"]
        assert any(p["role"] == "gold_A" for p in passages) and any(p["role"] == "gold_B" for p in passages)
        pairs.append(pair)
        used.update((ai, bi))
        if len(pairs) == n:
            break
    if len(pairs) < n:
        raise RuntimeError(f"Only constructed {len(pairs)}/{n} valid pairs")
    report = [{"pairs": len(pairs), "passages_per_pair": cfg["dataset"]["context_passages"],
               "gold_A_present": sum(any(p["role"] == "gold_A" for p in x["passages"]) for x in pairs),
               "gold_B_present": sum(any(p["role"] == "gold_B" for p in x["passages"]) for x in pairs),
               "same_article_pairs": sum(x["title_A"] == x["title_B"] for x in pairs),
               "mean_question_jaccard": round(float(np.mean([x["question_jaccard"] for x in pairs])), 4),
               "max_question_jaccard": round(max(x["question_jaccard"] for x in pairs), 4)}]
    if write:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"pairs_n{n}.jsonl", pairs)
        write_csv(root / f"construction_n{n}.csv", report)
    print("[generalization:construction] " + json.dumps(report[0]))
    return pairs


def load_wikipedia_questions(cfg: dict) -> list[dict]:
    path = ROOT / cfg["dataset"]["local_jsonl"]
    if not path.exists():
        raise FileNotFoundError(
            f"Generated Wikipedia dataset missing: {path}. Build it with "
            "python src/build_wikipedia_dataset.py"
        )
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return [r for r in rows if "qid" in r]  # first line is a _manifest record


def construct_pairs_wikipedia(cfg: dict, n: int, write: bool = True) -> list[dict]:
    """Build conditioning pairs from the generated Wikipedia dataset.

    Each of the ten source pages carries exactly two questions written against
    its single full-article passage (src/build_wikipedia_dataset.py). Unlike
    the SQuAD pairing above -- two *different* articles glued into one shared
    context -- question A and question B here are genuinely about the *same*
    passage: one is chosen to condition on, the other is the held-out probe of
    whether that conditioning silently drops a fact the same source supports.
    The other nine pages' full texts serve as distractors, filled by random
    sampling to match this experiment's existing SQuAD distractor policy (the
    BM25 hard-negative upgrade only ever applied to experiments 3/4, not this
    one).
    """
    rows = load_wikipedia_questions(cfg)
    by_page: dict[str, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["paragraphs"][0]["title"], []).append(row)
    pages = sorted(by_page)
    bad = [t for t in pages if len(by_page[t]) != 2]
    if bad:
        raise RuntimeError(f"expected exactly 2 questions/page, mismatch for: {bad}")
    if n > len(pages):
        raise RuntimeError(
            f"only {len(pages)} source pages have two questions each; cannot build {n} pairs "
            "(one pair per page -- generate more pages to scale this up)"
        )

    rng = random.Random(cfg["dataset"]["sample_seed"])
    chosen_pages = list(pages)
    rng.shuffle(chosen_pages)
    chosen_pages = chosen_pages[:n]
    page_text = {t: by_page[t][0]["paragraphs"][0]["text"] for t in pages}
    n_distractors = cfg["dataset"]["context_passages"] - 1

    pairs: list[dict] = []
    for title in chosen_pages:
        two = list(by_page[title])
        rng.shuffle(two)  # which of the page's two questions becomes A (conditioned) vs B (held out)
        a, b = two
        distractor_pool = [t for t in pages if t != title]
        if len(distractor_pool) < n_distractors:
            raise RuntimeError(
                f"only {len(distractor_pool)} candidate distractor pages available, need "
                f"{n_distractors} (raise sampling.n_pages or lower context_passages)"
            )
        distractor_titles = rng.sample(distractor_pool, n_distractors)
        passages = [{"text": page_text[title], "role": "gold_AB", "source_title": title}]
        passages += [{"text": page_text[t], "role": "distractor", "source_title": t} for t in distractor_titles]
        rng.shuffle(passages)
        pair = {
            "pair_id": f"{a['qid']}__{b['qid']}",
            "question_A": a["question"], "golds_A": a["golds"], "qid_A": a["qid"],
            "question_B": b["question"], "golds_B": b["golds"], "qid_B": b["qid"],
            "title_A": title, "title_B": title,
            "question_jaccard": jaccard(a["question"], b["question"]),
            "passages": passages,
            "context_note": (
                f"Both questions were written against the same single gold Wikipedia "
                f"passage ({title}), plus {n_distractors} distractor passages drawn from "
                "the other source pages."
            ),
        }
        assert len(passages) == cfg["dataset"]["context_passages"]
        assert sum(p["role"] == "gold_AB" for p in passages) == 1
        pairs.append(pair)

    report = [{
        "pairs": len(pairs), "passages_per_pair": cfg["dataset"]["context_passages"],
        "source_pages_available": len(pages),
        "same_article_pairs": sum(x["title_A"] == x["title_B"] for x in pairs),
        "mean_question_jaccard": round(float(np.mean([x["question_jaccard"] for x in pairs])), 4),
        "max_question_jaccard": round(max((x["question_jaccard"] for x in pairs), default=0.0), 4),
    }]
    if write:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"pairs_n{n}.jsonl", pairs)
        write_csv(root / f"construction_n{n}.csv", report)
    print("[generalization:construction:wikipedia] " + json.dumps(report[0]))
    return pairs


def render_context(pair: dict) -> str:
    return "\n\n".join(f"[P{i + 1}] {p['text']}" for i, p in enumerate(pair["passages"]))


def compression_user_prompt(pair: dict, notes: str | None, mode: str, stage: int) -> str:
    """Build prompts whose only arm-level difference is the question block."""
    if stage == 1:
        material = f"Source material:\n{render_context(pair)}"
        instruction = INITIAL_INSTRUCTION
    else:
        material = f"Previous agent's notes:\n{notes}"
        instruction = RECOMPRESS_INSTRUCTION
    if mode == "conditioned":
        user = f"{material}\n\nQuestion the final agent must answer: {pair['question_A']}\n\n{instruction}"
    else:
        user = f"{material}\n\n{instruction}"
        assert pair["question_A"] not in user and pair["question_B"] not in user
    return user


def prompt_difference_selftest(pair: dict) -> None:
    """Prove that deleting the question block makes the prompts byte-identical."""
    marker = f"\n\nQuestion the final agent must answer: {pair['question_A']}"
    for stage, notes in ((1, None), (2, "identical previous notes")):
        conditioned = compression_user_prompt(pair, notes, "conditioned", stage)
        generic = compression_user_prompt(pair, notes, "generic", stage)
        assert conditioned.replace(marker, "") == generic


def compress(client, pair: dict, notes: str | None, mode: str, stage: int, cfg: dict) -> dict:
    user = compression_user_prompt(pair, notes, mode, stage)
    result = client.chat([{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["subagent_temperature"],
                         max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                         tag=f"summary_generalization_{mode}_stage{stage}")
    return {"text": result.text.strip(), "cached": result.cached,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}


def write_example(pair: dict, handoffs: dict, root: Path) -> None:
    lines = [
        "# Example: question-only conditioning difference",
        "",
        f"**Question A passed only in the conditioned arm:** {pair['question_A']}",
        "",
        f"**Unrelated held-out Question B:** {pair['question_B']}",
        "",
        pair.get("context_note",
                  "Both gold passages and the same eight distractors were present in the shared depth-0 context."),
    ]
    for stage in (1, 5):
        for mode in ("conditioned", "generic"):
            lines += ["", f"## {mode}, stage {stage}", "", handoffs[(pair["pair_id"], mode, stage)]["text"]]
    root.mkdir(parents=True, exist_ok=True)
    (root / "example.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def answer(client, pair: dict, query_type: str, material: str, cfg: dict, tag: str) -> dict:
    suffix = "A" if query_type == "target" else "B"
    question, golds = pair[f"question_{suffix}"], pair[f"golds_{suffix}"]
    user = f"Research material:\n{material}\n\nQuestion:\n{question}\nAnswer:"
    result = client.chat([{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["orchestrator_temperature"],
                         max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    pred = extract_short_answer(result.text)
    em, f1 = score_against_golds(pred, golds)
    return {"question": question, "golds": golds, "pred": pred, "raw": result.text,
            "em": em, "f1": f1, "cached": result.cached}


def analyse(rows: list[dict], cfg: dict, root: Path) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    metrics, vectors = [], {}
    keys = sorted(set((r["mode"], r["query_type"], int(r["depth"])) for r in rows))
    for mode, query_type, depth in keys:
        subset = sorted([r for r in rows if (r["mode"], r["query_type"], int(r["depth"])) == (mode, query_type, depth)], key=lambda x: x["pair_id"])
        vectors[(mode, query_type, depth)] = {m: {r["pair_id"]: r[m] for r in subset} for m in ("em", "f1", "judge_correct")}
        record = {"mode": mode, "query_type": query_type, "depth": depth, "n": len(subset)}
        for metric in ("em", "f1", "judge_correct"):
            mean, lo, hi = bootstrap_ci(np.array([r[metric] for r in subset]), boot, ci, seed=71)
            record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
        metrics.append(record)
    deltas = []
    for query_type in ("target", "heldout"):
        base = vectors[("direct", query_type, 0)]
        for mode in cfg["modes"]:
            for depth in [d for d in cfg["depths"] if d > 0]:
                current = vectors[(mode, query_type, depth)]
                for metric in ("em", "f1", "judge_correct"):
                    ids = sorted(set(base[metric]) & set(current[metric]))
                    result = paired_bootstrap_delta(np.array([current[metric][i] for i in ids]), np.array([base[metric][i] for i in ids]), boot, ci, seed=73)
                    deltas.append({"comparison": "depth_minus_direct", "mode": mode, "query_type": query_type, "depth": depth, "metric": metric, **result})
        for depth in [d for d in cfg["depths"] if d > 0]:
            conditioned, generic = vectors[("conditioned", query_type, depth)], vectors[("generic", query_type, depth)]
            for metric in ("em", "f1", "judge_correct"):
                ids = sorted(set(conditioned[metric]) & set(generic[metric]))
                result = paired_bootstrap_delta(np.array([conditioned[metric][i] for i in ids]), np.array([generic[metric][i] for i in ids]), boot, ci, seed=79)
                deltas.append({"comparison": "conditioned_minus_generic", "mode": "conditioned_minus_generic", "query_type": query_type, "depth": depth, "metric": metric, **result})
    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_metrics = [("f1", "Token F1"), ("judge_correct", "LLM-judge accuracy")]
    fig, axes = plt.subplots(len(plot_metrics), 2, figsize=(11, 4.5 * len(plot_metrics)),
                             sharey="row", squeeze=False)
    colors = {"direct": "#7570b3", "conditioned": "#d95f02", "generic": "#1b9e77"}
    titles = ("Conditioning question A", "Unrelated held-out question B")
    for row_idx, (metric, label) in enumerate(plot_metrics):
        for col_idx, (query_type, title) in enumerate(zip(("target", "heldout"), titles)):
            axis = axes[row_idx][col_idx]
            direct = next(r for r in metrics if r["mode"] == "direct" and r["query_type"] == query_type)
            axis.scatter([0], [direct[metric]], color=colors["direct"], label="direct context", s=45)
            for mode in cfg["modes"]:
                subset = sorted([r for r in metrics if r["mode"] == mode and r["query_type"] == query_type], key=lambda r: r["depth"])
                axis.plot([r["depth"] for r in subset], [r[metric] for r in subset], marker="o", linewidth=2, color=colors[mode], label=mode)
            if row_idx == 0:
                axis.set_title(title)
            if row_idx == len(plot_metrics) - 1:
                axis.set_xlabel("Compression handoffs")
            axis.set_xticks(cfg["depths"])
            axis.grid(alpha=.25)
        axes[row_idx][0].set_ylabel(label)
    axes[0][1].legend()
    fig.suptitle("Question-only conditioning and summary generalizability")
    fig.tight_layout()
    fig.savefig(root / "summary_generalization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "summary_generalization_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_pairs"]
    if cfg["dataset"].get("source") == "generated_wikipedia":
        pairs = construct_pairs_wikipedia(cfg, n, write=not args.dry_run)
    else:
        pairs = construct_pairs(cfg, n, write=not args.dry_run)
    prompt_difference_selftest(pairs[0])
    print("[generalization:selftest] conditioned/generic prompts differ only by the Question A block")
    if args.construct_only:
        return 0
    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    result_root.mkdir(parents=True, exist_ok=True)
    handoff_path = run_root / "handoffs.jsonl"
    handoffs = {(r["pair_id"], r["mode"], int(r["stage"])): r for r in read_jsonl(handoff_path)}
    for mode in cfg["modes"]:
        for stage in range(1, cfg["max_depth"] + 1):
            missing = [p for p in pairs if (p["pair_id"], mode, stage) not in handoffs]
            def build(pair):
                previous = None if stage == 1 else handoffs[(pair["pair_id"], mode, stage - 1)]["text"]
                return compress(client, pair, previous, mode, stage, cfg)
            with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
                futures = {pool.submit(build, p): p for p in missing}
                for future, pair in futures.items():
                    handoffs[(pair["pair_id"], mode, stage)] = {"pair_id": pair["pair_id"], "mode": mode, "stage": stage, **future.result()}
            if not args.dry_run:
                write_jsonl(handoff_path, [r for _, r in sorted(handoffs.items())])
            print(f"[generalization:handoff] {mode}/stage{stage}: {len(missing)} generated")
    answer_path = run_root / "answers.jsonl"
    existing = {(r["mode"], r["query_type"], int(r["depth"]), r["pair_id"]): r for r in read_jsonl(answer_path)}
    jobs = []
    for pair in pairs:
        for query_type in ("target", "heldout"):
            key = ("direct", query_type, 0, pair["pair_id"])
            if key not in existing:
                jobs.append((key, pair, render_context(pair)))
            for mode in cfg["modes"]:
                for depth in [d for d in cfg["depths"] if d > 0]:
                    key = (mode, query_type, depth, pair["pair_id"])
                    if key not in existing:
                        jobs.append((key, pair, handoffs[(pair["pair_id"], mode, depth)]["text"]))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, pair, key[1], material, cfg, f"summary_generalization_{key[0]}_{key[1]}_d{key[2]}"): key for key, pair, material in jobs}
        for future, key in futures.items():
            existing[key] = {"mode": key[0], "query_type": key[1], "depth": key[2], "pair_id": key[3], **future.result()}
    rows = sorted(existing.values(), key=lambda r: (r["mode"], r["query_type"], int(r["depth"]), r["pair_id"]))
    if not args.dry_run:
        write_jsonl(answer_path, rows)
        judge_client = add_judge(rows, cfg, tag="summary_generalization_judge")
        write_jsonl(answer_path, rows)
        analyse(rows, cfg, result_root)
        write_example(pairs[0], handoffs, result_root)
        if judge_client is not None:
            print("[generalization:judge cost] " + json.dumps(judge_client.ledger.summary()))
    print(f"[generalization:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
