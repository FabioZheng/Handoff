"""Stage orchestration. Resumable, cache-backed, cost-capped.

Stages
  0  load MuSiQue, sample with a fixed seed, apply the C1 leakage filter
  1  subagent handoff generation (B/C/D x seeds)  -- A_full and E_oracle need none
  2  orchestrator answering (orchestrator sees question + handoff text only)
  3  injections            [not built yet -- see build order, spec §10]
  4  score and analyse

Every stage is idempotent: outputs are keyed jsonl, and every LLM call is content-
hashed into the on-disk cache, so re-running costs nothing (C5).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
import handoffs as hm  # noqa: E402
from data import Question  # noqa: E402
from llm import LLMClient, load_config, CostCapExceeded  # noqa: E402
from score import score_against_golds  # noqa: E402

HANDOFF_DIR = ROOT / "runs" / "handoffs"
ANSWER_DIR = ROOT / "runs" / "answers"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

LLM_MECHANISMS = ("B_freeform", "C_structured", "D_extractive")


# ------------------------------------------------------------------ helpers

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def maybe_write_jsonl(args, path: Path, rows: list[dict]) -> None:
    """Persist unless this is a dry run, whose contents are stubs."""
    if not args.dry_run:
        write_jsonl(path, rows)


def selftest_orchestrator_isolation() -> None:
    """Stage 2's isolation guarantee, tested rather than asserted (spec §5).

    The orchestrator entry point must reject anything that could carry paragraphs.
    """
    fake_q = Question(
        qid="selftest", question="q", answer="a", aliases=[],
        paragraphs=[], decomposition=[], gold_pids=[], gold_sentences=[], n_hops=0,
    )
    for bad in (fake_q, {"question": "q", "handoff_text": "t"}, "just a string"):
        try:
            hm.orchestrator_answer(None, bad, {})
        except TypeError:
            continue
        raise AssertionError(
            f"ISOLATION BROKEN: orchestrator_answer accepted {type(bad).__name__}"
        )
    sealed = hm.seal("q", hm.Handoff("selftest", "E_oracle", None, "some notes"))
    assert set(vars(sealed)) == {"qid", "question", "handoff_text", "mechanism"}, \
        "SealedHandoff must expose exactly four string fields"
    assert all(isinstance(v, (str, type(None))) for v in vars(sealed).values())
    print("[selftest] orchestrator isolation OK -- paragraphs are unreachable from Stage 2")


# ------------------------------------------------------------------ stage 0

def stage0(client, cfg, args) -> list[Question]:
    out_path = DATA_DIR / "filtered_questions.jsonl"
    report_path = RESULTS_DIR / "c1_filter_report.json"

    manifest = data_mod.stage0_manifest(cfg, ROOT)
    manifest["candidates_override"] = args.candidates or None

    if not args.force:
        qs = data_mod.read_filtered(out_path, manifest)
        if qs is not None:
            print(f"[stage0] reusing {len(qs)} filtered questions from {out_path}")
            if report_path.exists():
                rep = json.loads(report_path.read_text(encoding="utf-8"))
                _print_c1(rep)
            return qs
        if out_path.exists():
            print(f"[stage0] {out_path} is stale (dataset/config no longer matches "
                  f"its recorded manifest) -- regenerating")

    candidates = data_mod.sample_candidates(cfg, ROOT)
    if args.candidates:
        candidates = candidates[: args.candidates]
    print(f"[stage0] C1 closed-book filter on {len(candidates)} candidates "
          f"({cfg['leakage_filter']['samples']} samples @ T={cfg['leakage_filter']['temperature']})")

    kept, report = data_mod.apply_c1(
        client, candidates, cfg, cfg["runtime"]["concurrency"], dry_run=args.dry_run)

    if args.dry_run:
        # Closed-book answers are stubs under --dry-run, so the survivor set is
        # fiction. Plan with it, but never let it reach disk and poison a real run.
        print(f"[stage0] dry-run: not persisting the filter output "
              f"(survivor set is a placeholder for cost planning only)")
        return kept

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in report.items() if k != "attempts"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "c1_filter_summary.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    data_mod.write_filtered(kept, out_path, manifest)
    (DATA_DIR / "sampled_ids.json").write_text(
        json.dumps({"seed": cfg["sampling"]["seed"], "ids": [q.qid for q in kept]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _print_c1(report)
    if not report["target_met"]:
        print(f"[stage0] WARNING: only {report['kept']} questions survived; target is "
              f"{report['n_target']}. Raise sampling.n_candidates and rerun stage0.")
    return kept


def _print_c1(rep: dict) -> None:
    method = rep.get("known_method", "f1_threshold")
    print(f"[stage0] C1 leakage filter ({method}): {rep['candidates_run']} candidates -> "
          f"{rep['survived']} survived ({100 * rep['survival_rate']:.1f}%), "
          f"{rep['leaked_excluded']} excluded as already-known "
          f"({100 * rep['leak_rate']:.1f}% leak rate)")
    if rep.get("leaked_qids"):
        print(f"[stage0] leaked qids: {', '.join(rep['leaked_qids'])}")
    print(f"[stage0] closed-book mean EM={rep['closed_book_mean_em']:.3f} "
          f"F1={rep['closed_book_mean_f1']:.3f} over all attempts")


# ------------------------------------------------------------------ stage 1

def stage1(client, cfg, args, questions: list[Question]) -> dict[tuple, hm.Handoff]:
    """Generate handoffs. E_oracle needs no LLM; A_full has no handoff at all."""
    mechanisms = [m for m in args.mechanisms if m in LLM_MECHANISMS]
    store: dict[tuple, hm.Handoff] = {}

    # E_oracle is free and deterministic: always build it when requested.
    if "E_oracle" in args.mechanisms:
        rows = []
        for q in questions:
            h = hm.make_oracle(q)
            store[(q.qid, "E_oracle", None)] = h
            rows.append(h.to_json())
        maybe_write_jsonl(args, HANDOFF_DIR / "E_oracle.jsonl", rows)
        print(f"[stage1] E_oracle: {len(rows)} handoffs built with 0 LLM calls")

    for mech in mechanisms:
        for seed in cfg["seeds"]:
            path = HANDOFF_DIR / f"{mech}_seed{seed}.jsonl"
            existing = {r["qid"]: r for r in read_jsonl(path)} if not args.force else {}
            todo = [q for q in questions if q.qid not in existing]
            if todo:
                builder = hm.MECHANISM_BUILDERS[mech]
                with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as ex:
                    futs = {ex.submit(builder, client, q, cfg, seed): q for q in todo}
                    for fut, q in futs.items():
                        existing[q.qid] = fut.result().to_json()
            rows = [existing[q.qid] for q in questions if q.qid in existing]
            maybe_write_jsonl(args, path, rows)
            for r in rows:
                store[(r["qid"], mech, seed)] = hm.Handoff.from_json(r)
            print(f"[stage1] {mech} seed={seed}: {len(rows)} handoffs "
                  f"({len(todo)} newly generated)")
    return store


# ------------------------------------------------------------------ stage 2

def stage2(client, cfg, args, questions: list[Question],
           store: dict[tuple, hm.Handoff]) -> list[dict]:
    qmap = {q.qid: q for q in questions}
    path = ANSWER_DIR / "answers.jsonl"
    existing = {} if args.force else {
        (r["qid"], r["condition"], r["seed"]): r for r in read_jsonl(path)
    }

    jobs: list[tuple] = []          # (qid, condition, seed, callable)
    for q in questions:
        if "A_full" in args.mechanisms:
            jobs.append((q.qid, "A_full", None, lambda q=q: hm.full_context_answer(client, q, cfg)))
        if "E_oracle" in args.mechanisms:
            h = store.get((q.qid, "E_oracle", None))
            if h is not None:
                sealed = hm.seal(q.question, h)
                jobs.append((q.qid, "E_oracle", None,
                             lambda s=sealed: hm.orchestrator_answer(client, s, cfg, "answer_E_oracle")))
        for mech in [m for m in args.mechanisms if m in LLM_MECHANISMS]:
            for seed in cfg["seeds"]:
                h = store.get((q.qid, mech, seed))
                if h is None:
                    continue
                sealed = hm.seal(q.question, h)
                jobs.append((q.qid, mech, seed,
                             lambda s=sealed, m=mech: hm.orchestrator_answer(client, s, cfg, f"answer_{m}")))

    todo = [j for j in jobs if (j[0], j[1], j[2]) not in existing]
    if todo:
        with ThreadPoolExecutor(max_workers=cfg["runtime"]["concurrency"]) as ex:
            futs = {ex.submit(j[3]): j for j in todo}
            for fut, j in futs.items():
                res = fut.result()
                qid, cond, seed = j[0], j[1], j[2]
                q = qmap[qid]
                em, f1 = score_against_golds(res["pred"], q.golds)
                existing[(qid, cond, seed)] = {
                    "qid": qid, "condition": cond, "mechanism": cond, "seed": seed,
                    "pred": res["pred"], "raw": res["raw"], "gold": q.answer,
                    "em": em, "f1": f1, "n_hops": q.n_hops,
                    "answer_prompt_tokens": res["prompt_tokens"],
                    "answer_completion_tokens": res["completion_tokens"],
                    "cached": res["cached"],
                }
    rows = list(existing.values())
    maybe_write_jsonl(args, path, rows)
    print(f"[stage2] {len(rows)} answers ({len(todo)} newly generated)")
    return rows


# ------------------------------------------------------------------ stage 4

def stage4(cfg, args, questions: list[Question], answers: list[dict],
           store: dict[tuple, hm.Handoff], ledger_summary: dict) -> None:
    import numpy as np
    from score import bootstrap_ci, paired_bootstrap_delta, em_f1_disagreement

    qids = [q.qid for q in questions]
    boot_n = cfg["analysis"]["bootstrap_resamples"]
    ci = cfg["analysis"]["ci_level"]

    # Per (condition, qid) means over seeds -> one value per question per condition.
    per_cond: dict[str, dict[str, list[dict]]] = {}
    for a in answers:
        per_cond.setdefault(a["condition"], {}).setdefault(a["qid"], []).append(a)

    # Handoff token cost per (condition, qid): the subagent call that produced it.
    handoff_tokens: dict[tuple[str, str], list[int]] = {}
    for (qid, mech, seed), h in store.items():
        t = (h.meta.get("prompt_tokens", 0) or 0) + (h.meta.get("completion_tokens", 0) or 0)
        handoff_tokens.setdefault((mech, qid), []).append(t)

    rows = []
    # qid -> value, so contrasts can be paired on the exact question intersection.
    vectors: dict[str, dict[str, dict[str, float]]] = {}
    for cond, byq in sorted(per_cond.items()):
        present = [qid for qid in qids if qid in byq]
        vectors[cond] = {
            "em": {qid: float(np.mean([r["em"] for r in byq[qid]])) for qid in present},
            "f1": {qid: float(np.mean([r["f1"] for r in byq[qid]])) for qid in present},
        }
        em = np.array([vectors[cond]["em"][qid] for qid in present])
        f1 = np.array([vectors[cond]["f1"][qid] for qid in present])

        ans_tok = np.array([float(np.mean([r["answer_prompt_tokens"] + r["answer_completion_tokens"]
                                           for r in byq[qid]])) for qid in present])
        ho_tok = np.array([float(np.mean(handoff_tokens.get((cond, qid), [0]))) for qid in present])

        em_m, em_lo, em_hi = bootstrap_ci(em, boot_n, ci, seed=7)
        f1_m, f1_lo, f1_hi = bootstrap_ci(f1, boot_n, ci, seed=7)
        rows.append({
            "condition": cond, "n": int(em.size),
            "em": round(em_m, 4), "em_lo": round(em_lo, 4), "em_hi": round(em_hi, 4),
            "f1": round(f1_m, 4), "f1_lo": round(f1_lo, 4), "f1_hi": round(f1_hi, 4),
            "em_f1_disagree": round(em_f1_disagreement(em, f1), 4),
            "handoff_tokens_mean": round(float(ho_tok.mean()), 1),
            "answer_tokens_mean": round(float(ans_tok.mean()), 1),
            "total_tokens_mean": round(float(ho_tok.mean() + ans_tok.mean()), 1),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    with open(RESULTS_DIR / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Paired contrasts that answer the spec's three questions.
    contrasts = []
    for treat, ctrl, label in [
        ("B_freeform", "A_full", "headroom: free-form handoff vs full context"),
        ("E_oracle", "B_freeform", "recovery: perfect handoff vs free-form"),
        ("E_oracle", "A_full", "denoising: perfect handoff vs full context"),
        ("C_structured", "B_freeform", "mechanism: structured vs free-form"),
        ("D_extractive", "B_freeform", "mechanism: extractive vs free-form"),
        ("C_structured", "E_oracle", "mechanism: structured vs oracle ceiling"),
    ]:
        if treat not in vectors or ctrl not in vectors:
            continue
        shared = [q for q in qids if q in vectors[treat]["em"] and q in vectors[ctrl]["em"]]
        if not shared:
            continue
        for metric in ("em", "f1"):
            t = np.array([vectors[treat][metric][q] for q in shared])
            c = np.array([vectors[ctrl][metric][q] for q in shared])
            d = paired_bootstrap_delta(t, c, boot_n, ci, seed=11)
            contrasts.append({"contrast": f"{treat} - {ctrl}", "label": label, "metric": metric,
                              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}})
    if contrasts:
        with open(RESULTS_DIR / "contrasts.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(contrasts[0].keys()))
            w.writeheader()
            w.writerows(contrasts)

    _print_table(rows, contrasts, ledger_summary, cfg, len(qids))
    _write_report(rows, contrasts, ledger_summary, cfg, questions, args)


def _print_table(rows, contrasts, ledger, cfg, n_q) -> None:
    print("\n" + "=" * 92)
    print(f"RESULTS  model={cfg['model']['id']}  n_questions={n_q}  "
          f"seeds={cfg['seeds']}  dataset={cfg['dataset']['name']}")
    print("=" * 92)
    print(f"{'condition':<14}{'n':>4}{'EM':>8}{'EM 95% CI':>18}{'F1':>8}{'F1 95% CI':>18}{'tok/q':>9}")
    print("-" * 92)
    for r in rows:
        em_ci = f"[{r['em_lo']:.3f}, {r['em_hi']:.3f}]"
        f1_ci = f"[{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]"
        print(f"{r['condition']:<14}{r['n']:>4}{r['em']:>8.3f}{em_ci:>18}"
              f"{r['f1']:>8.3f}{f1_ci:>18}{r['total_tokens_mean']:>9.0f}")
    if contrasts:
        print("\nPaired contrasts (bootstrap, question-level pairing)")
        print("-" * 92)
        for c in contrasts:
            star = "*" if (c["lo"] > 0 or c["hi"] < 0) else " "
            print(f"{star} {c['contrast']:<28} {c['metric'].upper():<3} "
                  f"delta={c['delta']:+.3f}  95% CI [{c['lo']:+.3f}, {c['hi']:+.3f}]  "
                  f"p={c['p_value']:.4f}  n={c['n']}")
        print("  (* = CI excludes zero)")
    print(f"\nSpend: ${ledger['cost_usd']:.4f} | live calls {ledger['calls_live']} | "
          f"cached {ledger['calls_cached']} ({100 * ledger['cache_hit_rate']:.0f}% hit) | "
          f"tokens in/out {ledger['prompt_tokens']}/{ledger['completion_tokens']}")
    print("=" * 92 + "\n")


def _write_report(rows, contrasts, ledger, cfg, questions, args) -> None:
    lines = [
        "# Handoff probe results",
        "",
        f"- model: `{cfg['model']['id']}`",
        f"- dataset: {cfg['dataset']['name']} ({cfg['dataset']['split']}), n = {len(questions)} "
        f"questions surviving the C1 leakage filter",
        f"- seeds: {cfg['seeds']} (LLM-generated handoffs only; A_full and E_oracle are "
        f"deterministic at temperature 0)",
        f"- conditions run: {', '.join(args.mechanisms)}",
        "",
        "## Accuracy by condition",
        "",
        "| condition | n | EM | EM 95% CI | F1 | F1 95% CI | EM/F1 disagree | tokens/q |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['condition']}` | {r['n']} | {r['em']:.3f} | [{r['em_lo']:.3f}, {r['em_hi']:.3f}] "
            f"| {r['f1']:.3f} | [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}] | {r['em_f1_disagree']:.3f} "
            f"| {r['total_tokens_mean']:.0f} |"
        )
    if contrasts:
        lines += ["", "## Paired contrasts", "",
                  "| contrast | metric | delta | 95% CI | p | n | what it means |",
                  "|---|---|---:|---|---:|---:|---|"]
        for c in contrasts:
            lines.append(
                f"| `{c['contrast']}` | {c['metric'].upper()} | {c['delta']:+.3f} "
                f"| [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p_value']:.4f} | {c['n']} | {c['label']} |"
            )
    lines += ["", "## Cost", "",
              f"- spend this run: ${ledger['cost_usd']:.4f} (cap ${cfg['cost']['cap_usd']:.2f})",
              f"- calls: {ledger['calls_live']} live, {ledger['calls_cached']} cached "
              f"({100 * ledger['cache_hit_rate']:.0f}% hit rate)",
              f"- tokens: {ledger['prompt_tokens']} in / {ledger['completion_tokens']} out", ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[stage4] wrote {RESULTS_DIR / 'summary.csv'} and {RESULTS_DIR / 'report.md'}")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Agent handoff information-loss probe")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stage", default="all",
                    choices=["all", "0", "1", "2", "4"],
                    help="stage 3 (injections) is not built yet")
    ap.add_argument("--n", type=int, default=None, help="limit to first N filtered questions")
    ap.add_argument("--candidates", type=int, default=None,
                    help="limit C1 candidate pool (for cheap pilots)")
    ap.add_argument("--mechanisms", default=None,
                    help="comma-separated; defaults to conditions.pilot_mechanisms")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate calls and cost without spending anything")
    ap.add_argument("--force", action="store_true", help="ignore existing stage outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.mechanisms = ([m.strip() for m in args.mechanisms.split(",")] if args.mechanisms
                       else list(cfg["conditions"]["pilot_mechanisms"]))
    unknown = set(args.mechanisms) - set(cfg["conditions"]["mechanisms"])
    if unknown:
        print(f"unknown mechanisms: {sorted(unknown)}")
        return 2

    selftest_orchestrator_isolation()

    client = LLMClient(cfg, dry_run=args.dry_run)
    print(f"[run] model={cfg['model']['id']} mechanisms={args.mechanisms} "
          f"dry_run={args.dry_run} concurrency={cfg['runtime']['concurrency']}")

    try:
        questions = stage0(client, cfg, args)
        if args.n:
            questions = questions[: args.n]
            print(f"[run] limiting to first {len(questions)} questions")
        if not questions:
            print("[run] no questions survived stage 0; nothing to do")
            return 1

        store: dict = {}
        answers: list[dict] = []
        if args.stage in ("all", "1", "2", "4"):
            store = stage1(client, cfg, args, questions)
        if args.stage in ("all", "2", "4"):
            answers = stage2(client, cfg, args, questions, store)
        if args.stage in ("all", "4") and answers and not args.dry_run:
            stage4(cfg, args, questions, answers, store, client.ledger.summary())

    except CostCapExceeded as exc:
        print(f"\n[ABORT] {exc}")
        return 3

    if args.dry_run:
        rep = client.dry_run_report()
        print("\n" + "=" * 72)
        print("DRY RUN -- nothing was spent")
        print("=" * 72)
        print(json.dumps(rep, indent=2))
        print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
