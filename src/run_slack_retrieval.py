"""Compressible-slack test on MS MARCO: added filler volume x filler type.

Corrects ``run_retrieval_quality.py``, which had two defects for this question:

1. ``condition_packs`` pooled selected passages *globally*, so retention was
   all-or-nothing per query and the "medium" arm was a mixture of the good and
   bad arms rather than a genuinely intermediate context.
2. Every arm held ten passages, so adding filler required deleting evidence.

Here every query keeps all ``k`` of its selected passages in every condition,
and filler is ADDED on top.  Filler comes in two types:

* ``inert``     - non-selected passages from *other* queries (off-topic, cheap
                  to discard).
* ``near_miss`` - the query's *own* non-selected passages (same topic, no
                  answer, expensive to discard).

If volume alone protects, both types behave the same.  If only inert filler
protects, the mechanism is specifically about cheaply compressible slack.
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
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402

SYSTEM = ("You are a research handoff agent. Preserve every fact needed to answer the "
          "question. Your notes replace the entire input for the next agent, so omitted "
          "information is lost. Do not answer the question directly.")
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
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_data(cfg: dict) -> dict:
    path = ROOT / cfg["dataset"]["local_parquet"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print("[slack-retrieval:data] downloading MS MARCO v2.1 dev split (no LLM calls)")
        with requests.get(cfg["dataset"]["dev_url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(path)
    frame = pd.read_parquet(path)
    return {"query": {str(r.query_id): r.query for r in frame.itertuples(index=False)},
            "answers": {str(r.query_id): list(r.answers) for r in frame.itertuples(index=False)},
            "passages": {str(r.query_id): r.passages for r in frame.itertuples(index=False)}}


def passages(raw: dict, qid: str) -> list[dict]:
    p = raw["passages"][qid]
    urls = p["url"] if "url" in p else [""] * len(p["passage_text"])
    return [{"text": " ".join(str(t).split()), "selected": int(s), "url": str(u)}
            for t, s, u in zip(p["passage_text"], p["is_selected"], urls)]


def make_packs(raw: dict, cfg: dict, n: int) -> tuple[list[dict], list[dict]]:
    """Queries with at least k selected passages; k retained in every condition."""
    k = int(cfg["dataset"]["essential_k"])
    ids = list(raw["query"].keys())
    rng = random.Random(cfg["dataset"]["sample_seed"])
    rng.shuffle(ids)
    chosen = []
    for qid in ids:
        answers = [str(x).strip() for x in raw["answers"].get(qid, []) if str(x).strip()]
        if not answers or answers[0].casefold() == "no answer present.":
            continue
        ps = passages(raw, qid)
        selected = [p for p in ps if p["selected"]]
        others = [p for p in ps if not p["selected"]]
        if len(ps) == 10 and len(selected) >= k and len(others) >= max(cfg["filler_levels"]):
            chosen.append({"qid": str(qid), "question": str(raw["query"][qid]), "golds": answers,
                           "essential": selected[:k], "near_miss_pool": others})
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"only found {len(chosen)}/{n} queries with >= {k} selected passages")
    taken = {c["qid"] for c in chosen}
    pool = [(str(qid), p) for qid in ids[:20000] for p in passages(raw, qid)
            if str(qid) not in taken and not p["selected"]]
    return chosen, pool


def condition_pack(base: dict, pool: list[tuple[str, dict]], name: str, f: int, ftype: str, cfg: dict) -> dict:
    rng = random.Random(f"{cfg['dataset']['sample_seed']}:{base['qid']}:{name}")
    answer_strings = [a.casefold() for a in base["golds"] if len(a.strip()) > 2]
    if ftype == "near_miss":
        fillers = rng.sample(base["near_miss_pool"], f) if f else []
    else:
        candidates = [p for _, p in pool if not any(a in p["text"].casefold() for a in answer_strings)]
        fillers = rng.sample(candidates, f) if f else []
    docs = [{"text": p["text"], "is_relevant": True} for p in base["essential"]]
    docs += [{"text": p["text"], "is_relevant": False} for p in fillers]
    rng.shuffle(docs)
    assert len(docs) == len(base["essential"]) + f
    return {"qid": base["qid"], "question": base["question"], "golds": base["golds"],
            "condition": name, "f": f, "filler_type": ftype, "passages": docs,
            "filler_texts": [p["text"] for p in fillers]}


def construct(cfg: dict, n: int, write: bool) -> list[dict]:
    raw = ensure_data(cfg)
    base, pool = make_packs(raw, cfg, n)
    packs, report = [], []
    for name, spec in cfg["conditions"].items():
        f, ftype = int(spec["f"]), str(spec["type"])
        packs.extend(condition_pack(b, pool, name, f, ftype, cfg) for b in base)
        report.append({"condition": name, "queries": n, "essential_k": cfg["dataset"]["essential_k"],
                       "added_filler_f": f, "filler_type": ftype,
                       "context_width": cfg["dataset"]["essential_k"] + f, "design": "addition"})
    if write:
        root = ROOT / cfg["outputs"]["data_root"]
        write_jsonl(root / f"packs_n{n}.jsonl", packs)
        write_csv(root / f"construction_n{n}.csv", report)
    print("[slack-retrieval:construction] " + json.dumps(report))
    return packs


def context(pack: dict) -> str:
    return "\n\n".join(f"[P{i+1}] {p['text']}" for i, p in enumerate(pack["passages"]))


def compress(client, pack: dict, previous: str | None, cfg: dict, stage: int) -> dict:
    if stage == 1:
        user = f"Passages:\n{context(pack)}\n\nQuestion: {pack['question']}\n\n{INITIAL}"
    else:
        user = f"Prior notes:\n{previous}\n\nQuestion: {pack['question']}\n\n{REWRITE}"
    r = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    temperature=cfg["decoding"]["subagent_temperature"],
                    max_tokens=cfg["decoding"]["handoff_max_tokens"], seed=stage,
                    tag=f"slack_retrieval_{pack['condition']}_stage{stage}")
    return {"text": r.text.strip(), "cached": r.cached}


def answer(client, pack: dict, material: str, cfg: dict, tag: str) -> dict:
    user = f"Research material:\n{material}\n\nQuestion: {pack['question']}\nAnswer:"
    r = client.chat([{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
                    temperature=cfg["decoding"]["orchestrator_temperature"],
                    max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    pred = extract_short_answer(r.text)
    em, f1 = score_against_golds(pred, pack["golds"])
    return {"pred": pred, "raw": r.text, "golds": pack["golds"], "em": em, "f1": f1, "cached": r.cached}


# ---------- survival in the handoff text ----------

def content_tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "you", "your", "have", "has"}
    return {t for t in re.findall(r"[a-z0-9]+", text.casefold()) if len(t) > 3 and t not in stop}


def survival_rows(packs: list[dict], handoffs: dict, cfg: dict) -> list[dict]:
    rows = []
    for pack in packs:
        essential = [p["text"] for p in pack["passages"] if p["is_relevant"]]
        gold_terms = [content_tokens(t) for t in essential]
        filler_terms = [content_tokens(t) for t in pack["filler_texts"]]
        # MS MARCO v2.1 references are free-form sentences (~46 words), so exact
        # containment is structurally ~0.  Measure how much of the reference
        # answer's content survives instead.
        answer_terms = [content_tokens(a) for a in pack["golds"] if content_tokens(a)]
        src = len(context(pack))
        for stage in range(1, cfg["max_depth"] + 1):
            text = handoffs[(pack["condition"], pack["qid"], stage)]["text"]
            present = content_tokens(text)
            rows.append({
                "condition": pack["condition"], "f": pack["f"], "filler_type": pack["filler_type"],
                "qid": pack["qid"], "stage": stage,
                "answer_recall": max((len(t & present) / len(t) for t in answer_terms), default=float("nan")),
                # Lexical overlap retained from essential vs filler passages.
                "gold_overlap": float(np.mean([len(t & present) / max(1, len(t)) for t in gold_terms])),
                "filler_overlap": float(np.mean([len(t & present) / max(1, len(t)) for t in filler_terms])) if filler_terms else float("nan"),
                "handoff_chars": len(text), "source_chars": src, "compression_ratio": len(text) / max(1, src),
            })
    return rows


def analyse(rows: list[dict], survival: list[dict], cfg: dict, root: Path) -> None:
    boot, ci = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    conditions = list(cfg["conditions"])
    metrics, values = [], {}
    for name in conditions:
        spec = cfg["conditions"][name]
        for depth in cfg["depths"]:
            subset = sorted([r for r in rows if r["condition"] == name and int(r["depth"]) == depth], key=lambda r: r["qid"])
            record = {"condition": name, "f": spec["f"], "filler_type": spec["type"], "depth": depth, "n": len(subset)}
            values[(name, depth)] = {}
            for metric in ("em", "f1"):
                vector = np.array([r[metric] for r in subset])
                values[(name, depth)][metric] = dict(zip([r["qid"] for r in subset], vector))
                mean, lo, hi = bootstrap_ci(vector, boot, ci, seed=51)
                record.update({metric: round(mean, 4), f"{metric}_lo": round(lo, 4), f"{metric}_hi": round(hi, 4)})
            if depth > 0:
                sub = [s for s in survival if s["condition"] == name and s["stage"] == depth]
                for field in ("answer_recall", "gold_overlap", "filler_overlap"):
                    vals = np.array([s[field] for s in sub], dtype=float)
                    record[field] = round(float(np.nanmean(vals)), 4) if not np.all(np.isnan(vals)) else float("nan")
                record["handoff_chars"] = round(float(np.mean([s["handoff_chars"] for s in sub])), 1)
                record["compression_ratio"] = round(float(np.mean([s["compression_ratio"] for s in sub])), 4)
            metrics.append(record)

    deltas = []
    for name in conditions:
        for depth in [d for d in cfg["depths"] if d > 0]:
            for metric in ("em", "f1"):
                a, b = values[(name, depth)][metric], values[(name, 0)][metric]
                ids = sorted(set(a) & set(b))
                d = paired_bootstrap_delta(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]), boot, ci, seed=53)
                deltas.append({"comparison": "depth_minus_depth0", "condition": name,
                               "f": cfg["conditions"][name]["f"], "filler_type": cfg["conditions"][name]["type"],
                               "depth": depth, "metric": metric, **d})

    # Decisive contrast: added filler vs no filler, within each filler type.
    slack = []
    base_name = next(c for c in conditions if cfg["conditions"][c]["f"] == 0)
    for name in conditions:
        spec = cfg["conditions"][name]
        if spec["f"] == 0:
            continue
        for stage in [d for d in cfg["depths"] if d > 0]:
            for field in ("answer_recall", "gold_overlap"):
                a = {s["qid"]: s[field] for s in survival if s["condition"] == name and s["stage"] == stage}
                b = {s["qid"]: s[field] for s in survival if s["condition"] == base_name and s["stage"] == stage}
                ids = sorted(set(a) & set(b))
                d = paired_bootstrap_delta(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]), boot, ci, seed=57)
                slack.append({"comparison": "filler_minus_nofiller", "condition": name, "f": spec["f"],
                              "filler_type": spec["type"], "stage": stage, "metric": field, **d})
            a2 = {r["qid"]: r["f1"] for r in rows if r["condition"] == name and int(r["depth"]) == stage}
            b2 = {r["qid"]: r["f1"] for r in rows if r["condition"] == base_name and int(r["depth"]) == stage}
            ids2 = sorted(set(a2) & set(b2))
            d2 = paired_bootstrap_delta(np.array([a2[i] for i in ids2]), np.array([b2[i] for i in ids2]), boot, ci, seed=59)
            slack.append({"comparison": "filler_minus_nofiller", "condition": name, "f": spec["f"],
                          "filler_type": spec["type"], "stage": stage, "metric": "answer_f1", **d2})

    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "deltas.csv", deltas)
    write_csv(root / "slack_contrasts.csv", slack)
    write_jsonl(root / "survival.jsonl", survival)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = [d for d in cfg["depths"] if d > 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = {"none": ("#444444", "-"), "inert": ("#1b9e77", "-"), "near_miss": ("#d95f02", "--")}
    for name in conditions:
        spec = cfg["conditions"][name]
        colour, ls = styles[spec["type"]]
        alpha = 1.0 if spec["f"] in (0, max(c["f"] for c in cfg["conditions"].values())) else .55
        label = f"f={spec['f']} {spec['type']}"
        rec = lambda field, d: np.nanmean([s[field] for s in survival if s["condition"] == name and s["stage"] == d])
        axes[0].plot(stages, [rec("gold_overlap", d) for d in stages], marker="o", color=colour, ls=ls, alpha=alpha, label=label)
        axes[1].plot(stages, [rec("filler_overlap", d) for d in stages], marker="o", color=colour, ls=ls, alpha=alpha, label=label)
        axes[2].plot(stages, [rec("answer_recall", d) for d in stages], marker="o", color=colour, ls=ls, alpha=alpha, label=label)
    axes[0].set(title="Essential-passage content retained", xlabel="Compression handoffs", ylabel="lexical overlap", ylim=(0, 1.02))
    axes[1].set(title="Filler content retained", xlabel="Compression handoffs", ylabel="lexical overlap", ylim=(0, 1.02))
    axes[2].set(title="Reference-answer content retained", xlabel="Compression handoffs", ylabel="answer token recall", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=.25)
    axes[2].legend(fontsize=7)
    fig.suptitle("MS MARCO: added filler volume and type through repeated handoffs")
    fig.tight_layout()
    fig.savefig(root / "slack_retrieval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "slack_retrieval_config.yaml"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dataset"]["n_questions"]
    packs = construct(cfg, n, write=not args.dry_run)
    if args.construct_only:
        return 0

    client = LLMClient(cfg, dry_run=args.dry_run)
    run_root = ROOT / cfg["outputs"]["run_root"] / f"n{n}"
    result_root = ROOT / cfg["outputs"]["result_root"] / f"n{n}"
    result_root.mkdir(parents=True, exist_ok=True)

    handoff_path = run_root / "handoffs.jsonl"
    handoffs = {(r["condition"], r["qid"], int(r["stage"])): r for r in read_jsonl(handoff_path)}
    for name in cfg["conditions"]:
        subset = [p for p in packs if p["condition"] == name]
        for stage in range(1, cfg["max_depth"] + 1):
            missing = [p for p in subset if (name, p["qid"], stage) not in handoffs]

            def build(pack, stage=stage, name=name):
                previous = None if stage == 1 else handoffs[(name, pack["qid"], stage - 1)]["text"]
                return compress(client, pack, previous, cfg, stage)

            with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
                futures = {pool.submit(build, p): p for p in missing}
                for future, pack in futures.items():
                    handoffs[(name, pack["qid"], stage)] = {"condition": name, "qid": pack["qid"],
                                                            "stage": stage, **future.result()}
            if not args.dry_run:
                write_jsonl(handoff_path, [r for _, r in sorted(handoffs.items())])
            print(f"[slack-retrieval:handoff] {name}/stage{stage}: {len(missing)} generated", flush=True)

    answer_path = run_root / "answers.jsonl"
    existing = {(r["condition"], int(r["depth"]), r["qid"]): r for r in read_jsonl(answer_path)}
    jobs = []
    for pack in packs:
        for depth in cfg["depths"]:
            key = (pack["condition"], depth, pack["qid"])
            if key not in existing:
                material = context(pack) if depth == 0 else handoffs[(pack["condition"], pack["qid"], depth)]["text"]
                jobs.append((key, pack, material))
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as pool:
        futures = {pool.submit(answer, client, pack, material, cfg, f"slack_retrieval_{key[0]}_answer_d{key[1]}"): key
                   for key, pack, material in jobs}
        for future, key in futures.items():
            existing[key] = {"condition": key[0], "depth": key[1], "qid": key[2], **future.result()}
    rows = sorted(existing.values(), key=lambda r: (r["condition"], int(r["depth"]), r["qid"]))

    if not args.dry_run:
        write_jsonl(answer_path, rows)
        survival = survival_rows(packs, handoffs, cfg)
        analyse(rows, survival, cfg, result_root)
    print(f"[slack-retrieval:answers] {len(rows)} rows; {len(jobs)} calls this pass")
    print(json.dumps(client.ledger.summary(), indent=2))
    if args.dry_run:
        print(json.dumps(client.dry_run_report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
