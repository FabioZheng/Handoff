"""Relevant-signal-ratio propagation with fixed ten-passage retrieval packs.

Each pack comprises ten independently answerable SQuAD facts.  Its one fixed
multi-fact query asks for all ten labelled answers.  Conditions retain exactly
10, 5, or 1 answer-bearing passages and replace every removed passage with a
real SQuAD distractor.  This makes relevant-document ratios 1.0, 0.5, and 0.1
without duplicating a gold passage or treating irrelevant text as gold.
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
from run_chain import CHAIN_SYSTEM, INITIAL_INSTRUCTION, RECOMPRESS_INSTRUCTION, add_bertscore  # noqa: E402
from score import bootstrap_ci, exact_match, paired_bootstrap_delta, token_f1  # noqa: E402

ANSWER_SYSTEM = (
    "You answer using only supplied material. Return one answer per requested label in the exact "
    "form `L1: answer`. If a label is unsupported, write `L1: unknown`. Do not add explanations."
)


def read_jsonl(path: Path) -> list[dict]:
    return [] if not path.exists() else [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
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
    print("[signal-ratio:data] downloading SQuAD validation parquet (no LLM calls)")
    with requests.get(cfg["dataset"]["parquet_url"], stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(path)
    return path


def answers(value) -> list[str]:
    items = value.get("text", []) if isinstance(value, dict) else value["text"]
    return list(dict.fromkeys(str(x).strip() for x in list(items) if str(x).strip()))


def records(cfg: dict) -> list[dict]:
    frame = pd.read_parquet(ensure_parquet(cfg))
    result = []
    for row in frame.itertuples(index=False):
        golds = answers(row.answers)
        context = " ".join(str(row.context).split())
        if golds and context:
            result.append({"qid": str(row.id), "title": str(row.title), "question": str(row.question),
                           "context": context, "golds": golds})
    return result


def make_packs(cfg: dict, n: int, write: bool = True) -> list[dict]:
    source = records(cfg)
    rng = random.Random(cfg["dataset"]["sample_seed"])
    indices = list(range(len(source)))
    rng.shuffle(indices)
    packs, used = [], set()
    width = int(cfg["dataset"]["passages_per_pack"])
    cursor = 0
    while cursor < len(indices) and len(packs) < n:
        selected, titles, contexts = [], set(), set()
        while cursor < len(indices) and len(selected) < width:
            idx = indices[cursor]
            cursor += 1
            item = source[idx]
            if idx in used or item["title"] in titles or item["context"] in contexts:
                continue
            selected.append(item)
            titles.add(item["title"])
            contexts.add(item["context"])
        if len(selected) < width:
            break
        used.update(source.index(item) for item in selected)
        pack_id = "__".join(item["qid"] for item in selected)
        packs.append({"pack_id": pack_id, "facts": selected})
    if len(packs) < n:
        raise RuntimeError(f"only built {len(packs)}/{n} ten-fact packs")
    if write:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"base_packs_n{n}.jsonl", packs)
    return packs


def condition_pack(base: dict, source: list[dict], condition: str, relevant_count: int, cfg: dict) -> dict:
    facts = base["facts"]
    indices = list(range(len(facts)))
    rng = random.Random(f"{cfg['dataset']['sample_seed']}:{base['pack_id']}:{condition}")
    rng.shuffle(indices)
    keep = set(indices[:relevant_count])
    titles = {item["title"] for item in facts}
    contexts = {item["context"] for item in facts}
    distractors = [item for item in source if item["title"] not in titles and item["context"] not in contexts]
    replacements = iter(rng.sample(distractors, len(facts) - relevant_count))
    passages = []
    for slot, fact in enumerate(facts, start=1):
        if slot - 1 in keep:
            passages.append({"text": fact["context"], "slot": slot, "is_relevant": True, "title": fact["title"]})
        else:
            distractor = next(replacements)
            passages.append({"text": distractor["context"], "slot": slot, "is_relevant": False, "title": distractor["title"]})
    return {"pack_id": base["pack_id"], "condition": condition, "facts": facts, "passages": passages,
            "available_slots": sorted(slot + 1 for slot in keep), "relevant_count": relevant_count}


def question(pack: dict) -> str:
    lines = ["Answer each labelled question. Return exactly one line per label: `L1: short answer`."]
    lines += [f"L{i}: {fact['question']}" for i, fact in enumerate(pack["facts"], start=1)]
    return "\n".join(lines)


def render_context(pack: dict) -> str:
    return "\n\n".join(f"[P{i}] {passage['text']}" for i, passage in enumerate(pack["passages"], start=1))


def compress(client, pack: dict, previous: str | None, cfg: dict, stage: int) -> dict:
    if stage == 1:
        user = f"Source material:\n{render_context(pack)}\n\nQuestion the final agent must answer:\n{question(pack)}\n\n{INITIAL_INSTRUCTION}"
    else:
        user = f"Previous agent's notes:\n{previous}\n\nQuestion the final agent must answer:\n{question(pack)}\n\n{RECOMPRESS_INSTRUCTION}"
    result = client.chat([{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["subagent_temperature"],
                         max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                         tag=f"signal_ratio_{pack['condition']}_stage{stage}")
    return {"text": result.text.strip(), "cached": result.cached}


def parse_slots(raw: str, slots: int) -> dict[int, str]:
    found = {}
    for line in raw.splitlines():
        match = re.match(r"\s*(?:[-*]\s*)?L\s*(\d+)\s*:\s*(.*)$", line, re.I)
        if match and 1 <= int(match.group(1)) <= slots:
            found[int(match.group(1))] = match.group(2).strip()
    return found


def answer(client, pack: dict, material: str, cfg: dict, tag: str) -> dict:
    user = f"Research material:\n{material}\n\n{question(pack)}"
    result = client.chat([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["orchestrator_temperature"],
                         max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    values = parse_slots(result.text, len(pack["facts"]))
    slots = []
    for index, fact in enumerate(pack["facts"], start=1):
        pred = values.get(index, "")
        scores = [(exact_match(pred, gold), token_f1(pred, gold)) for gold in fact["golds"]]
        em, f1 = (max(x[0] for x in scores), max(x[1] for x in scores)) if scores else (0.0, 0.0)
        slots.append({"slot": index, "available": index in pack["available_slots"], "pred": pred,
                      "golds": fact["golds"], "em": em, "f1": f1})
    return {"raw": result.text, "slots": slots, "cached": result.cached}


def add_slot_bertscore(rows: list[dict], cfg: dict) -> None:
    flattened = []
    owners = []
    for row in rows:
        for slot in row["slots"]:
            flattened.append({"pred": slot["pred"], "golds": slot["golds"]})
            owners.append(slot)
    add_bertscore(flattened, cfg)
    for slot, scored in zip(owners, flattened):
        slot["bertscore_f1"] = scored["bertscore_f1"]


def aggregate(row: dict, metric: str, available_only: bool) -> float:
    slots = [slot for slot in row["slots"] if slot["available"] or not available_only]
    return float(np.mean([slot[metric] for slot in slots])) if slots else float("nan")


def analyse(rows: list[dict], cfg: dict, root: Path) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    metrics, values = [], {}
    for condition in cfg["conditions"]:
        ratio = cfg["conditions"][condition] / cfg["dataset"]["passages_per_pack"]
        for depth in cfg["depths"]:
            subset = sorted([row for row in rows if row["condition"] == condition and int(row["depth"]) == depth], key=lambda row: row["pack_id"])
            record = {"condition": condition, "signal_ratio": ratio, "depth": depth, "n": len(subset)}
            values[(condition, depth)] = {}
            for metric in ("em", "f1", "bertscore_f1"):
                available = np.array([aggregate(row, metric, True) for row in subset])
                overall = np.array([aggregate(row, metric, False) for row in subset])
                values[(condition, depth)][metric] = {row["pack_id"]: score for row, score in zip(subset, available)}
                mean, lo, hi = bootstrap_ci(available, boot, ci, seed=91)
                record.update({f"available_{metric}": round(mean, 4), f"available_{metric}_lo": round(lo, 4), f"available_{metric}_hi": round(hi, 4), f"overall_{metric}": round(float(overall.mean()), 4)})
            metrics.append(record)
    deltas = []
    for condition in cfg["conditions"]:
        for depth in [d for d in cfg["depths"] if d > 0]:
            for metric in ("em", "f1", "bertscore_f1"):
                current, base = values[(condition, depth)][metric], values[(condition, 0)][metric]
                ids = sorted(set(current) & set(base))
                result = paired_bootstrap_delta(np.array([current[i] for i in ids]), np.array([base[i] for i in ids]), boot, ci, seed=97)
                deltas.append({"comparison": "available_depth_minus_depth0", "condition": condition, "depth": depth, "metric": metric, **result})
    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"signal_1_0": "#1b9e77", "signal_0_5": "#7570b3", "signal_0_1": "#d95f02"}
    labels = {"signal_1_0": "10 gold / 0 distractor", "signal_0_5": "5 gold / 5 distractor", "signal_0_1": "1 gold / 9 distractor"}
    for condition in cfg["conditions"]:
        subset = sorted([row for row in metrics if row["condition"] == condition], key=lambda row: row["depth"])
        x = [row["depth"] for row in subset]
        available = [row["available_f1"] for row in subset]
        axes[0].plot(x, available, marker="o", color=colors[condition], label=labels[condition])
        base = available[0]
        axes[1].plot(x, [score / base if base else float("nan") for score in available], marker="o", color=colors[condition], label=labels[condition])
    axes[0].set(title="Available-evidence F1", xlabel="Compression handoffs", ylabel="Token F1")
    axes[1].set(title="Relevant signal retained", xlabel="Compression handoffs", ylabel="F1 / depth-0 F1")
    for axis in axes:
        axis.set_xticks(cfg["depths"])
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle("Fixed ten-passage contexts: relevant-signal ratio through handoffs")
    fig.tight_layout()
    fig.savefig(root / "signal_ratio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "signal_ratio_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_packs"]
    base = make_packs(cfg, n, write=not args.dry_run)
    source = records(cfg)
    packs = [condition_pack(item, source, condition, count, cfg) for condition, count in cfg["conditions"].items() for item in base]
    report = [{"condition": condition, "packs": n, "passages_per_pack": cfg["dataset"]["passages_per_pack"], "relevant_passages": count, "distractors": cfg["dataset"]["passages_per_pack"] - count, "signal_ratio": count / cfg["dataset"]["passages_per_pack"]} for condition, count in cfg["conditions"].items()]
    if not args.dry_run:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"conditioned_packs_n{n}.jsonl", packs)
        write_csv(root / f"construction_n{n}.csv", report)
    print("[signal-ratio:construction] " + json.dumps(report))
    if args.construct_only:
        return 0
    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root, result_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}", ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    result_root.mkdir(parents=True, exist_ok=True)
    handoff_path = run_root / "handoffs.jsonl"
    handoffs = {(row["condition"], row["pack_id"], int(row["stage"])): row for row in read_jsonl(handoff_path)}
    for condition in cfg["conditions"]:
        subset = [pack for pack in packs if pack["condition"] == condition]
        for stage in range(1, cfg["max_depth"] + 1):
            missing = [pack for pack in subset if (condition, pack["pack_id"], stage) not in handoffs]
            def build(pack):
                previous = None if stage == 1 else handoffs[(condition, pack["pack_id"], stage - 1)]["text"]
                return compress(client, pack, previous, cfg, stage)
            with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
                futures = {pool.submit(build, pack): pack for pack in missing}
                for future, pack in futures.items():
                    handoffs[(condition, pack["pack_id"], stage)] = {"condition": condition, "pack_id": pack["pack_id"], "stage": stage, **future.result()}
            if not args.dry_run:
                write_jsonl(handoff_path, [row for _, row in sorted(handoffs.items())])
            print(f"[signal-ratio:handoff] {condition}/stage{stage}: {len(missing)} generated")
    answer_path = run_root / "answers.jsonl"
    existing = {(row["condition"], int(row["depth"]), row["pack_id"]): row for row in read_jsonl(answer_path)}
    jobs = []
    for pack in packs:
        for depth in cfg["depths"]:
            key = (pack["condition"], depth, pack["pack_id"])
            if key not in existing:
                material = render_context(pack) if depth == 0 else handoffs[(pack["condition"], pack["pack_id"], depth)]["text"]
                jobs.append((key, pack, material))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, pack, material, cfg, f"signal_ratio_{key[0]}_answer_d{key[1]}"): key for key, pack, material in jobs}
        for future, key in futures.items():
            existing[key] = {"condition": key[0], "depth": key[1], "pack_id": key[2], **future.result()}
    rows = sorted(existing.values(), key=lambda row: (row["condition"], int(row["depth"]), row["pack_id"]))
    if not args.dry_run:
        write_jsonl(answer_path, rows)
        add_slot_bertscore(rows, cfg)
        write_jsonl(answer_path, rows)
        analyse(rows, cfg, result_root)
    print(f"[signal-ratio:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
