"""Compressible-slack test: essential-fact load x added filler volume.

Corrects ``run_signal_ratio.py`` / ``run_redundant_signal_ratio.py``, both of
which held the context at ten passages and *substituted* distractors for gold
passages.  Substitution makes signal count and filler count perfectly
anti-correlated, so "more slack" always means "less signal" and the slack
hypothesis has no room to show itself.

Here filler is **added**, never substituted:

* ``k``  distinct SQuAD facts, all always present and all always asked.
* ``f``  real cross-article distractor passages appended on top.
* context width is ``k + f`` and is deliberately *not* held constant.

The dependent variable is fact survival in the handoff text itself, not only
the final answer, because the hypothesis is a claim about what a summariser
discards first.
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

from llm import LLMClient, load_config  # noqa: E402
from run_chain import CHAIN_SYSTEM, INITIAL_INSTRUCTION, RECOMPRESS_INSTRUCTION  # noqa: E402
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
    print("[slack-facts:data] downloading SQuAD validation parquet (no LLM calls)")
    with requests.get(cfg["dataset"]["parquet_url"], stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(path)
    return path


def aliases(value) -> list[str]:
    items = value.get("text", []) if isinstance(value, dict) else value["text"]
    return list(dict.fromkeys(str(x).strip() for x in list(items) if str(x).strip()))


def records(cfg: dict) -> list[dict]:
    frame = pd.read_parquet(ensure_parquet(cfg))
    result = []
    for row in frame.itertuples(index=False):
        golds = aliases(row.answers)
        context = " ".join(str(row.context).split())
        if golds and context:
            result.append({"qid": str(row.id), "title": str(row.title), "question": str(row.question),
                           "context": context, "golds": golds})
    return result


def make_packs(cfg: dict, n: int, write: bool = True) -> list[dict]:
    """Packs of ``max_facts`` mutually distinct facts; condition k slices this."""
    source = records(cfg)
    rng = random.Random(cfg["dataset"]["sample_seed"])
    indices = list(range(len(source)))
    rng.shuffle(indices)
    width = int(cfg["dataset"]["max_facts"])
    packs, cursor = [], 0
    while cursor < len(indices) and len(packs) < n:
        selected, titles, contexts = [], set(), set()
        while cursor < len(indices) and len(selected) < width:
            item = source[indices[cursor]]
            cursor += 1
            if item["title"] in titles or item["context"] in contexts:
                continue
            selected.append(item)
            titles.add(item["title"])
            contexts.add(item["context"])
        if len(selected) < width:
            break
        packs.append({"pack_id": "__".join(x["qid"] for x in selected), "facts": selected})
    if len(packs) < n:
        raise RuntimeError(f"only built {len(packs)}/{n} {width}-fact packs")
    if write:
        write_jsonl(ROOT / cfg["outputs"]["data_root"] / f"base_packs_n{n}.jsonl", packs)
    return packs


def condition_pack(base: dict, source: list[dict], name: str, k: int, f: int, cfg: dict) -> dict:
    """Keep every one of k facts; ADD f distractors.  Width is k + f."""
    facts = base["facts"][:k]
    titles = {x["title"] for x in facts}
    contexts = {x["context"] for x in facts}
    answer_strings = [a.casefold() for x in facts for a in x["golds"] if len(a.strip()) > 1]
    rng = random.Random(f"{cfg['dataset']['sample_seed']}:{base['pack_id']}:{name}")
    pool = [x for x in source if x["title"] not in titles and x["context"] not in contexts
            and not any(a in x["context"].casefold() for a in answer_strings)]
    if len(pool) < f:
        raise RuntimeError(f"insufficient answer-free distractors for {base['pack_id']}")
    fillers = rng.sample(pool, f)
    passages = [{"text": x["context"], "is_relevant": True, "title": x["title"]} for x in facts]
    passages += [{"text": x["context"], "is_relevant": False, "title": x["title"]} for x in fillers]
    rng.shuffle(passages)
    assert len(passages) == k + f
    assert sum(p["is_relevant"] for p in passages) == k
    return {"pack_id": base["pack_id"], "condition": name, "k": k, "f": f,
            "facts": facts, "fillers": fillers, "passages": passages}


def construct(cfg: dict, n: int, write: bool) -> list[dict]:
    base = make_packs(cfg, n, write)
    source = records(cfg)
    packs, report = [], []
    for name, spec in cfg["conditions"].items():
        k, f = int(spec["k"]), int(spec["f"])
        packs.extend(condition_pack(item, source, name, k, f, cfg) for item in base)
        report.append({"condition": name, "packs": n, "essential_facts_k": k, "added_filler_f": f,
                       "context_width": k + f, "all_facts_present": True, "design": "addition"})
    if write:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"packs_n{n}.jsonl", packs)
        write_csv(root / f"construction_n{n}.csv", report)
    print("[slack-facts:construction] " + json.dumps(report))
    return packs


def question(pack: dict) -> str:
    lines = ["Answer each labelled question. Return exactly one line per label: `L1: short answer`."]
    lines += [f"L{i}: {fact['question']}" for i, fact in enumerate(pack["facts"], start=1)]
    return "\n".join(lines)


def render_context(pack: dict) -> str:
    return "\n\n".join(f"[P{i}] {p['text']}" for i, p in enumerate(pack["passages"], start=1))


def compress(client, pack: dict, previous: str | None, cfg: dict, stage: int) -> dict:
    if stage == 1:
        user = f"Source material:\n{render_context(pack)}\n\nQuestion the final agent must answer:\n{question(pack)}\n\n{INITIAL_INSTRUCTION}"
    else:
        user = f"Previous agent's notes:\n{previous}\n\nQuestion the final agent must answer:\n{question(pack)}\n\n{RECOMPRESS_INSTRUCTION}"
    result = client.chat([{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
                         temperature=cfg["decoding"]["subagent_temperature"],
                         max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                         tag=f"slack_facts_{pack['condition']}_stage{stage}")
    return {"text": result.text.strip(), "cached": result.cached,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}


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
        scored = [(exact_match(pred, g), token_f1(pred, g)) for g in fact["golds"]]
        em, f1 = (max(x[0] for x in scored), max(x[1] for x in scored)) if scored else (0.0, 0.0)
        slots.append({"slot": index, "pred": pred, "golds": fact["golds"], "em": em, "f1": f1})
    return {"raw": result.text, "slots": slots, "cached": result.cached}


# ---------- survival in the handoff text (no API calls, no LLM judge) ----------

def mentions_answer(text: str, golds: list[str]) -> bool:
    body = text.casefold()
    return any(g.casefold() in body for g in golds if len(g.strip()) >= 3)


def mentions_topic(text: str, title: str) -> bool:
    body = text.casefold()
    topic = title.replace("_", " ").casefold()
    if topic in body:
        return True
    # Fall back to the distinctive tokens of a multi-word article title.
    parts = [p for p in re.findall(r"[a-z0-9]+", topic) if len(p) > 3]
    return bool(parts) and all(p in body for p in parts)


def survival_rows(packs: list[dict], handoffs: dict, cfg: dict) -> list[dict]:
    rows = []
    for pack in packs:
        source_text = render_context(pack)
        for stage in range(1, cfg["max_depth"] + 1):
            text = handoffs[(pack["condition"], pack["pack_id"], stage)]["text"]
            gold_hits = [mentions_answer(text, fact["golds"]) for fact in pack["facts"]]
            filler_hits = [mentions_topic(text, x["title"]) for x in pack["fillers"]]
            rows.append({
                "condition": pack["condition"], "k": pack["k"], "f": pack["f"],
                "pack_id": pack["pack_id"], "stage": stage,
                "gold_survival": float(np.mean(gold_hits)),
                "filler_survival": float(np.mean(filler_hits)) if filler_hits else float("nan"),
                "handoff_chars": len(text), "source_chars": len(source_text),
                "compression_ratio": len(text) / max(1, len(source_text)),
            })
    return rows


def analyse(rows: list[dict], survival: list[dict], cfg: dict, root: Path) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    conditions = list(cfg["conditions"])

    metrics, values = [], {}
    for name in conditions:
        spec = cfg["conditions"][name]
        for depth in cfg["depths"]:
            subset = sorted([r for r in rows if r["condition"] == name and int(r["depth"]) == depth],
                            key=lambda r: r["pack_id"])
            record = {"condition": name, "k": spec["k"], "f": spec["f"], "depth": depth, "n": len(subset)}
            values[(name, depth)] = {}
            for metric in ("em", "f1"):
                vector = np.array([float(np.mean([s[metric] for s in r["slots"]])) for r in subset])
                values[(name, depth)][metric] = dict(zip([r["pack_id"] for r in subset], vector))
                mean, lo, hi = bootstrap_ci(vector, boot, ci, seed=91)
                record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
            if depth > 0:
                sub = [s for s in survival if s["condition"] == name and s["stage"] == depth]
                gold = np.array([s["gold_survival"] for s in sub])
                filler = np.array([s["filler_survival"] for s in sub])
                mean, lo, hi = bootstrap_ci(gold, boot, ci, seed=93)
                record.update({"gold_survival": round(mean, 4), "gold_survival_lo": round(lo, 4),
                               "gold_survival_hi": round(hi, 4)})
                record["filler_survival"] = round(float(np.nanmean(filler)), 4) if spec["f"] else float("nan")
                record["handoff_chars"] = round(float(np.mean([s["handoff_chars"] for s in sub])), 1)
                record["compression_ratio"] = round(float(np.mean([s["compression_ratio"] for s in sub])), 4)
            metrics.append(record)

    deltas = []
    for name in conditions:
        for depth in [d for d in cfg["depths"] if d > 0]:
            for metric in ("em", "f1"):
                current, base = values[(name, depth)][metric], values[(name, 0)][metric]
                ids = sorted(set(current) & set(base))
                result = paired_bootstrap_delta(np.array([current[i] for i in ids]),
                                                np.array([base[i] for i in ids]), boot, ci, seed=97)
                deltas.append({"comparison": "depth_minus_depth0", "condition": name,
                               "k": cfg["conditions"][name]["k"], "f": cfg["conditions"][name]["f"],
                               "depth": depth, "metric": metric, **result})

    # The decisive contrast: at fixed k, does ADDING filler change fact survival?
    slack = []
    by_kf = {(cfg["conditions"][c]["k"], cfg["conditions"][c]["f"]): c for c in conditions}
    ks = sorted({cfg["conditions"][c]["k"] for c in conditions})
    fs = sorted({cfg["conditions"][c]["f"] for c in conditions})
    for k in ks:
        if (k, 0) not in by_kf:
            continue
        base_name = by_kf[(k, 0)]
        for f in [x for x in fs if x > 0]:
            if (k, f) not in by_kf:
                continue
            name = by_kf[(k, f)]
            for stage in [d for d in cfg["depths"] if d > 0]:
                a = {s["pack_id"]: s["gold_survival"] for s in survival if s["condition"] == name and s["stage"] == stage}
                b = {s["pack_id"]: s["gold_survival"] for s in survival if s["condition"] == base_name and s["stage"] == stage}
                ids = sorted(set(a) & set(b))
                result = paired_bootstrap_delta(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]),
                                                boot, ci, seed=101)
                slack.append({"comparison": "filler_minus_nofiller", "k": k, "f": f, "stage": stage,
                              "metric": "gold_survival", **result})
                a2 = {r["pack_id"]: float(np.mean([x["f1"] for x in r["slots"]]))
                      for r in rows if r["condition"] == name and int(r["depth"]) == stage}
                b2 = {r["pack_id"]: float(np.mean([x["f1"] for x in r["slots"]]))
                      for r in rows if r["condition"] == base_name and int(r["depth"]) == stage}
                ids2 = sorted(set(a2) & set(b2))
                result2 = paired_bootstrap_delta(np.array([a2[i] for i in ids2]), np.array([b2[i] for i in ids2]),
                                                 boot, ci, seed=103)
                slack.append({"comparison": "filler_minus_nofiller", "k": k, "f": f, "stage": stage,
                              "metric": "answer_f1", **result2})

    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)
    write_csv(root / "slack_contrasts.csv", slack)
    write_jsonl(root / "survival.jsonl", survival)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = plt.cm.viridis(np.linspace(0, .85, len(fs)))
    for ax, k in zip(axes, ks):
        for colour, f in zip(colors, fs):
            if (k, f) not in by_kf:
                continue
            name = by_kf[(k, f)]
            stages = [d for d in cfg["depths"] if d > 0]
            gold = [np.mean([s["gold_survival"] for s in survival if s["condition"] == name and s["stage"] == d]) for d in stages]
            ax.plot(stages, gold, marker="o", color=colour, label=f"+{f} filler (gold)")
            if f:
                fill = [np.nanmean([s["filler_survival"] for s in survival if s["condition"] == name and s["stage"] == d]) for d in stages]
                ax.plot(stages, fill, marker="x", ls="--", color=colour, alpha=.55, label=f"+{f} filler (filler)")
        ax.set(title=f"k = {k} essential facts", xlabel="Compression handoffs", ylim=(0, 1.02))
        ax.set_xticks([d for d in cfg["depths"] if d > 0])
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Survival in handoff text")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Compressible slack: does added filler protect essential facts?")
    fig.tight_layout()
    fig.savefig(root / "slack_facts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "slack_facts_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_packs"]
    packs = construct(cfg, n, write=not args.dry_run)
    if args.construct_only:
        return 0

    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    result_root.mkdir(parents=True, exist_ok=True)

    handoff_path = run_root / "handoffs.jsonl"
    handoffs = {(r["condition"], r["pack_id"], int(r["stage"])): r for r in read_jsonl(handoff_path)}
    for name in cfg["conditions"]:
        subset = [p for p in packs if p["condition"] == name]
        for stage in range(1, cfg["max_depth"] + 1):
            missing = [p for p in subset if (name, p["pack_id"], stage) not in handoffs]

            def build(pack, stage=stage, name=name):
                previous = None if stage == 1 else handoffs[(name, pack["pack_id"], stage - 1)]["text"]
                return compress(client, pack, previous, cfg, stage)

            with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
                futures = {pool.submit(build, p): p for p in missing}
                for future, pack in futures.items():
                    handoffs[(name, pack["pack_id"], stage)] = {
                        "condition": name, "pack_id": pack["pack_id"], "stage": stage, **future.result()}
            if not args.dry_run:
                write_jsonl(handoff_path, [r for _, r in sorted(handoffs.items())])
            print(f"[slack-facts:handoff] {name}/stage{stage}: {len(missing)} generated", flush=True)

    answer_path = run_root / "answers.jsonl"
    existing = {(r["condition"], int(r["depth"]), r["pack_id"]): r for r in read_jsonl(answer_path)}
    jobs = []
    for pack in packs:
        for depth in cfg["depths"]:
            key = (pack["condition"], depth, pack["pack_id"])
            if key not in existing:
                material = render_context(pack) if depth == 0 else handoffs[(pack["condition"], pack["pack_id"], depth)]["text"]
                jobs.append((key, pack, material))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, pack, material, cfg, f"slack_facts_{key[0]}_answer_d{key[1]}"): key
                   for key, pack, material in jobs}
        for future, key in futures.items():
            existing[key] = {"condition": key[0], "depth": key[1], "pack_id": key[2], **future.result()}
    rows = sorted(existing.values(), key=lambda r: (r["condition"], int(r["depth"]), r["pack_id"]))

    if not args.dry_run:
        write_jsonl(answer_path, rows)
        survival = survival_rows(packs, handoffs, cfg)
        analyse(rows, survival, cfg, result_root)
    print(f"[slack-facts:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    if args.dry_run:
        print(json.dumps(client.dry_run_report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
