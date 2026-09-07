"""Experiment 12: anticipatory context management under uncertain future demand.

Experiment 10 measured what a bounded, irreversible handoff costs. Experiment 11
showed the trade-off can be steered by an explicit allocation quota. This runner
asks the harness question underneath both: when the future information need is
unknown and only *estimated*, how should a system spend context, and does making
discarded evidence retrievable change what is achievable at all?

Two things separate it from Experiment 10:

* the true future-use distribution ``P`` over aspects is distinguished from the
  harness's estimate ``P-hat``, and ``P-hat`` is perturbed away from ``P`` along
  two orthogonal axes (dilution = uncertain, displacement = wrong);
* the writer has six actions rather than one, including externalizing evidence
  to a BM25-queryable store that the reader can pull from at answer time.

Every message is still answered on every question, so ``U_now`` is the diagonal
of the per-context utility matrix and ``U_future`` its off-diagonal, reweighted
by the *true* ``P``. Policies are scored under the truth; only their decisions
are made under the estimate. That asymmetry is the experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anticipatory_data as ad  # noqa: E402
import anticipatory_policies as ap  # noqa: E402
import anticipatory_store as ast_  # noqa: E402
import budget as bd  # noqa: E402
import handoffs as hm  # noqa: E402
from judge import add_judge  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402
from score import extract_short_answer, score_against_golds  # noqa: E402

SCHEMA_VERSION = "anticipatory-context-v1"
METRICS = ("em", "f1", "judge_correct")

COMPRESS_SYSTEM = (
    "You shorten a single sentence of research material for a colleague who cannot see "
    "the original. Preserve every name, number, date and fact. Drop only wording. "
    "Reply with the shortened sentence and nothing else."
)


# ---------------------------------------------------------------------------
# Design


def make_clients(cfg: dict, dry_run: bool = False) -> tuple[LLMClient, LLMClient]:
    sender = LLMClient(cfg, dry_run=dry_run)
    spec = cfg.get("answerer") or cfg["model"]
    if spec["id"] == cfg["model"]["id"]:
        return sender, sender
    answer_cfg = copy.deepcopy(cfg)
    answer_cfg["model"] = copy.deepcopy(spec)
    answerer = LLMClient(answer_cfg, dry_run=dry_run)
    answerer.cache = sender.cache          # one experiment, one cap
    answerer.ledger = sender.ledger
    return sender, answerer


def true_distribution(conditioning_aspect: str, cfg: dict) -> tuple[dict[str, float], str]:
    """The distribution future demand is actually drawn from, and its target aspect.

    Demand lands mostly on a *different* aspect than the one just asked about -
    that is the whole reason anticipation is hard. Concentration is kept high
    because a flat truth would leave the dilution knob no range to sweep.
    """
    spec = cfg["anticipatory"]["true_distribution"]
    concentration = float(spec["concentration"])
    offset = int(spec["target_offset"])
    index = (ad.ASPECTS.index(conditioning_aspect) + offset) % len(ad.ASPECTS)
    target = ad.ASPECTS[index]
    others = (1.0 - concentration) / (len(ad.ASPECTS) - 1)
    dist = {a: (concentration if a == target else others) for a in ad.ASPECTS}
    return dist, target


def estimate_for(policy: str, true: dict[str, float]) -> dict[str, float] | None:
    """Map a policy name onto the estimate its family is supposed to hold."""
    if "__" not in policy:
        return None
    knob = policy.split("__", 1)[1]
    table = {
        "tau0": lambda: ad.dilute(true, 0.0),
        "tau0p5": lambda: ad.dilute(true, 0.5),
        "tau1": lambda: ad.dilute(true, 1.0),
        "sigma0p5": lambda: ad.displace(true, 0.5),
        "sigma1": lambda: ad.displace(true, 1.0),
    }
    if knob not in table:
        raise ValueError(f"unknown noise knob {knob!r} in policy {policy!r}")
    return table[knob]()


def build_rotations(dossiers: list[dict], units_by_context: dict) -> list[dict]:
    """One rotation per anchor question: four per dossier, one for each aspect."""
    rotations = []
    for dossier in dossiers:
        for question in dossier["questions"]:
            if question["role"] != "anchor":
                continue
            rotations.append({
                "context_id": dossier["context_id"],
                "rotation_id": f"{dossier['context_id']}|cond={question['qid']}",
                "current_qid": question["qid"],
                "current_question": question["question"],
                "conditioning_aspect": question["aspect"],
                "dossier": dossier,
                "units": units_by_context[dossier["context_id"]],
            })
    return rotations


# ---------------------------------------------------------------------------
# Compression: one call per unit, shared by every policy


def compress_units(client: LLMClient, units: list, cfg: dict) -> dict[str, str]:
    ratio = float(cfg["anticipatory"]["compression_target_ratio"])

    def one(unit) -> tuple[str, str]:
        target = max(4, int(round(unit.words * ratio)))
        prompt = (f"Sentence:\n{unit.text}\n\n"
                  f"Rewrite it in at most {target} words, keeping every fact.")
        result = client.chat(
            [{"role": "system", "content": COMPRESS_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max(48, int(target * 3)), seed=None,
            tag="anticipatory_compress",
        )
        text, _, _ = bd.truncate_to_words(result.text.strip(), target)
        return unit.unit_id, text

    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        return dict(pool.map(one, units))


# ---------------------------------------------------------------------------
# Messages


def build_messages(rotations: list[dict], budgets: list[int], policies: list[str],
                   compressed: dict[str, str], cfg: dict) -> list[dict]:
    settings = cfg["anticipatory"]
    rows = []
    for rotation in rotations:
        units = rotation["units"]
        true, target_aspect = true_distribution(rotation["conditioning_aspect"], cfg)
        future_qid = next(
            (q["qid"] for q in rotation["dossier"]["questions"]
             if q["aspect"] == target_aspect and q["role"] == "anchor"), None)
        for budget_words in budgets:
            for policy in policies:
                p_hat = estimate_for(policy, true)
                allocation = ap.build_allocation(
                    policy, units, rotation["current_qid"], budget_words, compressed,
                    p_hat=p_hat, future_qid=future_qid, cfg=settings)
                built = ap.assemble(allocation, units, compressed, budget_words)
                ast_.assert_bounded_recall(
                    built["message_text"], allocation.retrieval_k, units, policy)
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "message_key": f"{rotation['rotation_id']}|{policy}|w{budget_words}",
                    "context_id": rotation["context_id"],
                    "rotation_id": rotation["rotation_id"],
                    "current_qid": rotation["current_qid"],
                    "conditioning_aspect": rotation["conditioning_aspect"],
                    "target_aspect": target_aspect,
                    "policy": policy,
                    "family": policy.split("__")[0],
                    "budget_words": budget_words,
                    "retrieval_k": allocation.retrieval_k,
                    "p_hat": p_hat,
                    "true_p": true,
                    "tv_mismatch": (ad.total_variation(p_hat, true) if p_hat else None),
                    "js_mismatch": (ad.jensen_shannon(p_hat, true) if p_hat else None),
                    "p_hat_entropy": (ap.normalised_entropy(p_hat) if p_hat else None),
                    "notes": allocation.notes,
                    **{k: v for k, v in built.items() if k != "stored_units"},
                    "stored_unit_ids": [u.unit_id for u in built["stored_units"]],
                })
    return rows


# ---------------------------------------------------------------------------
# Answers


def answer_specs(rotations: list[dict], messages: list[dict], cfg: dict) -> list[dict]:
    by_rotation = defaultdict(list)
    for message in messages:
        by_rotation[message["rotation_id"]].append(message)
    units_by_id = {u.unit_id: u
                   for rotation in rotations for u in rotation["units"]}
    specs = []
    for rotation in rotations:
        questions = rotation["dossier"]["questions"]
        for message in by_rotation[rotation["rotation_id"]]:
            store = ast_.EvidenceStore(
                [units_by_id[uid] for uid in message["stored_unit_ids"]])
            for question in questions:
                retrieved = store.retrieve(question["question"], message["retrieval_k"])
                specs.append({
                    "answer_key": f"{message['message_key']}|{question['qid']}",
                    "message_key": message["message_key"],
                    "context_id": message["context_id"],
                    "rotation_id": message["rotation_id"],
                    "policy": message["policy"],
                    "budget_words": message["budget_words"],
                    "current_qid": message["current_qid"],
                    "eval_qid": question["qid"],
                    "eval_aspect": question["aspect"],
                    "eval_role": question["role"],
                    "is_diagonal": question["qid"] == message["current_qid"],
                    "question": question["question"],
                    "golds": list(question["golds"]),
                    "message_text": message["message_text"],
                    "retrieved_text": tuple(u.text for u in retrieved),
                    "retrieved_words": sum(u.words for u in retrieved),
                    "n_retrieved": len(retrieved),
                })
    return specs


def run_answer(client: LLMClient, spec: dict, cfg: dict) -> dict:
    sealed = ast_.SealedContext(
        qid=spec["eval_qid"], question=spec["question"],
        message_text=spec["message_text"],
        retrieved_text=spec["retrieved_text"], policy=spec["policy"])
    base = {k: v for k, v in spec.items()
            if k not in ("message_text", "retrieved_text")}
    if not spec["message_text"].strip() and not spec["retrieved_text"]:
        # An empty context is a real outcome of a bounded channel, not a bug.
        # It scores zero rather than silently shrinking one arm's n.
        return {**base, "raw": "", "pred": "", "em": 0.0, "f1": 0.0, "cached": True,
                "technical_failure": True}
    result = client.chat(
        [{"role": "system", "content": hm.ANSWER_SYSTEM},
         {"role": "user", "content": ast_.answer_prompt(sealed)}],
        temperature=float(cfg["decoding"]["answer_temperature"]),
        max_tokens=int(cfg["decoding"]["answer_max_tokens"]),
        seed=None, tag=f"anticipatory_answer_{spec['policy']}",
    )
    pred = extract_short_answer(result.text)
    em, f1 = score_against_golds(pred, spec["golds"])
    return {**base, "raw": result.text, "pred": pred, "em": em, "f1": f1,
            "cached": result.cached, "technical_failure": False}


def generate_answers(client: LLMClient, specs: list[dict], cfg: dict) -> list[dict]:
    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        return list(pool.map(lambda s: run_answer(client, s, cfg), specs))


# ---------------------------------------------------------------------------
# Analysis


def cluster_ci(values: list[float], clusters: list[str], n_resamples: int,
               seed: int, ci: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap resampling whole contexts, not individual rows.

    ``score.bootstrap_ci`` resamples rows independently, which is wrong here:
    each dossier contributes four rotations that share its source text and its
    difficulty, so row-level resampling would treat four correlated
    observations as four independent ones and report an interval that is too
    narrow. Contexts are the independent unit, matching the clustering used
    throughout Experiments 10 and 11.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        by_cluster[cluster].append(index)
    keys = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = [i for k in picked for i in by_cluster[keys[k]]]
        means[draw] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return (float(values.mean()), float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)))


def rotation_rows(messages: list[dict], answers: list[dict], cfg: dict) -> list[dict]:
    """Collapse to one row per (rotation, policy, budget).

    ``U_future`` is the *true*-P-weighted mean over aspects, with each aspect
    averaged before weighting so an aspect with more questions cannot outvote
    one with fewer. The conditioning question is excluded from the future side.
    """
    by_message = {m["message_key"]: m for m in messages}
    grouped = defaultdict(list)
    for answer in answers:
        grouped[answer["message_key"]].append(answer)

    rows = []
    for message_key, cells in sorted(grouped.items()):
        message = by_message[message_key]
        true = message["true_p"]
        diagonal = [c for c in cells if c["is_diagonal"]]
        off = [c for c in cells if not c["is_diagonal"]]
        if not diagonal or not off:
            continue
        row = {
            "context_id": message["context_id"],
            "rotation_id": message["rotation_id"],
            "policy": message["policy"],
            "family": message["family"],
            "budget_words": message["budget_words"],
            "tv_mismatch": message["tv_mismatch"],
            "js_mismatch": message["js_mismatch"],
            "p_hat_entropy": message["p_hat_entropy"],
            "retrieval_k": message["retrieval_k"],
            "c_context": message["delivered_words"],
            "n_keep": message["n_keep"], "n_compress": message["n_compress"],
            "n_pointer": message["n_pointer"],
            "n_externalize": message["n_externalize"], "n_discard": message["n_discard"],
            "retrieval_calls": sum(1 for c in cells if c["n_retrieved"] > 0),
            "retrieval_rate": float(np.mean([c["n_retrieved"] > 0 for c in cells])),
            "c_retrieval": float(np.mean([c["retrieved_words"] for c in cells])),
            "n_eval": len(cells),
        }
        for metric in METRICS:
            row[f"now_{metric}"] = float(np.mean([c[metric] for c in diagonal]))
            by_aspect = defaultdict(list)
            for cell in off:
                by_aspect[cell["eval_aspect"]].append(cell[metric])
            present = {a: float(np.mean(v)) for a, v in by_aspect.items() if v}
            row[f"future_{metric}"] = sum(
                true[a] * present[a] for a in ad.ASPECTS if true.get(a, 0) > 0 and a in present)
            row[f"gap_{metric}"] = row[f"now_{metric}"] - row[f"future_{metric}"]
        rows.append(row)
    return rows


def policy_points(rows: list[dict], cfg: dict) -> list[dict]:
    """One point per (policy, budget): the frontier coordinates and their CIs."""
    lam = float(cfg["analysis"]["lambda_default"])
    beta = float(cfg["analysis"]["beta"])
    gamma = float(cfg["analysis"]["gamma_default"])
    resamples = int(cfg["analysis"]["bootstrap_resamples"])
    seed = int(cfg["analysis"]["bootstrap_seed"])

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["budget_words"])].append(row)

    points = []
    for (policy, budget_words), cell in sorted(grouped.items()):
        now = [r["now_judge_correct"] for r in cell]
        future = [r["future_judge_correct"] for r in cell]
        contexts = [r["context_id"] for r in cell]
        utility = [n + lam * f for n, f in zip(now, future)]
        ci = float(cfg["analysis"]["ci_level"])
        _, now_lo, now_hi = cluster_ci(now, contexts, resamples, seed, ci)
        _, fut_lo, fut_hi = cluster_ci(future, contexts, resamples, seed, ci)
        _, util_lo, util_hi = cluster_ci(utility, contexts, resamples, seed, ci)
        c_context = float(np.mean([r["c_context"] for r in cell]))
        c_retrieval = float(np.mean([r["c_retrieval"] for r in cell]))
        points.append({
            "policy": policy, "family": cell[0]["family"], "budget_words": budget_words,
            "n_rotations": len(cell),
            "u_now": float(np.mean(now)), "u_now_lo": now_lo, "u_now_hi": now_hi,
            "u_future": float(np.mean(future)), "u_future_lo": fut_lo, "u_future_hi": fut_hi,
            "utility": float(np.mean(utility)), "utility_lo": util_lo, "utility_hi": util_hi,
            "tv_mismatch": cell[0]["tv_mismatch"],
            "js_mismatch": cell[0]["js_mismatch"],
            "p_hat_entropy": cell[0]["p_hat_entropy"],
            "c_context": c_context, "c_retrieval": c_retrieval,
            "cost": beta * c_context + gamma * c_retrieval,
            "retrieval_rate": float(np.mean([r["retrieval_rate"] for r in cell])),
            "n_pointer": float(np.mean([r["n_pointer"] for r in cell])),
            "n_externalize": float(np.mean([r["n_externalize"] for r in cell])),
            "n_compress": float(np.mean([r["n_compress"] for r in cell])),
        })
    return points


def anticipation_value(rows: list[dict], cfg: dict) -> list[dict]:
    """``V_anticipation`` = U(pi_P-hat) - U(pi_no-prediction), paired by rotation.

    ``blind`` is the no-prediction reference: it provides for the future but
    spreads uniformly because it holds no belief. Pairing on rotation removes
    dossier difficulty, so the interval is about the prediction and not about
    which dossiers happened to be easy.
    """
    lam = float(cfg["analysis"]["lambda_default"])
    resamples = int(cfg["analysis"]["bootstrap_resamples"])
    seed = int(cfg["analysis"]["bootstrap_seed"])

    def utility(row: dict) -> float:
        return row["now_judge_correct"] + lam * row["future_judge_correct"]

    baseline = {(r["rotation_id"], r["budget_words"]): utility(r)
                for r in rows if r["policy"] == "blind"}
    grouped = defaultdict(list)
    for row in rows:
        if row["policy"] == "blind":
            continue
        key = (row["rotation_id"], row["budget_words"])
        if key in baseline:
            grouped[(row["policy"], row["budget_words"])].append(
                (row["context_id"], utility(row) - baseline[key]))

    out = []
    for (policy, budget_words), pairs in sorted(grouped.items()):
        deltas = [d for _, d in pairs]
        clusters = [c for c, _ in pairs]
        _, lo, hi = cluster_ci(deltas, clusters, resamples, seed,
                               float(cfg["analysis"]["ci_level"]))
        example = next(r for r in rows
                       if r["policy"] == policy and r["budget_words"] == budget_words)
        out.append({
            "policy": policy, "family": example["family"], "budget_words": budget_words,
            "tv_mismatch": example["tv_mismatch"], "js_mismatch": example["js_mismatch"],
            "v_anticipation": float(np.mean(deltas)), "lo": lo, "hi": hi,
            "n": len(deltas),
            "harmful": bool(hi < 0.0),
            "helpful": bool(lo > 0.0),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="anticipatory_context_config.yaml")
    parser.add_argument("--budgets", help="comma-separated subset of the configured budgets")
    parser.add_argument("--policies", help="comma-separated subset of the configured policies")
    parser.add_argument("--limit", type=int, help="use only the first N dossiers")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    if args.concurrency:
        cfg["runtime"]["concurrency"] = int(args.concurrency)
    if args.no_judge:
        cfg["judge"]["enabled"] = False

    budgets = [int(b.words) for b in bd.budgets_from_config(cfg)]
    if args.budgets:
        wanted = {int(x) for x in args.budgets.split(",") if x.strip()}
        budgets = [b for b in budgets if b in wanted]
    policies = list(cfg["anticipatory"]["policies"])
    if args.policies:
        wanted_p = [p.strip() for p in args.policies.split(",") if p.strip()]
        bad = [p for p in wanted_p if p not in policies]
        if bad:
            parser.error(f"unknown policies {bad}")
        policies = wanted_p
    if not budgets:
        parser.error("no configured budgets selected")

    _, dossiers = ad.load_dossiers(ROOT / cfg["dataset"]["contexts_jsonl"])
    if args.limit:
        dossiers = dossiers[:int(args.limit)]
    units_by_context = {}
    all_units = []
    for dossier in dossiers:
        units = ad.extract_units(dossier)
        ad.check_coverage(dossier, units)
        units_by_context[dossier["context_id"]] = units
        all_units.extend(units)
    rotations = build_rotations(dossiers, units_by_context)

    if args.dry_run:
        # Must never write to data/, runs/ or results/.
        n_messages = len(rotations) * len(budgets) * len(policies)
        print(json.dumps({
            "writes": 0, "dossiers": len(dossiers), "units": len(all_units),
            "rotations": len(rotations), "policies": len(policies),
            "budgets": budgets, "messages": n_messages,
            "compression_calls": len(all_units),
            "answer_rows": n_messages * 16, "judge_rows": n_messages * 16,
        }, indent=2))
        return 0

    run_root = ROOT / cfg["outputs"]["run_root"]
    result_root = ROOT / cfg["outputs"]["result_root"]

    sender, answerer = make_clients(cfg)
    compressed = compress_units(sender, all_units, cfg)
    print(f"[anticipatory:compress] {len(compressed)} units")

    messages = build_messages(rotations, budgets, policies, compressed, cfg)
    write_jsonl(run_root / "messages.jsonl", messages)
    print(f"[anticipatory:messages] {len(messages)} messages "
          f"({len(policies)} policies x {len(rotations)} rotations x {len(budgets)} budgets)")

    specs = answer_specs(rotations, messages, cfg)
    answers = generate_answers(answerer, specs, cfg)
    if cfg["judge"]["enabled"]:
        add_judge(answers, cfg, tag="anticipatory_judge")
    for answer in answers:
        answer.setdefault("judge_correct", 0)
    write_jsonl(run_root / "answers.jsonl", answers)
    print(f"[anticipatory:answers] {len(answers)} rows")

    rows = rotation_rows(messages, answers, cfg)
    points = policy_points(rows, cfg)
    value = anticipation_value(rows, cfg)
    write_csv(result_root / "rotation_rows.csv", rows)
    write_csv(result_root / "policy_points.csv", points)
    write_csv(result_root / "anticipation_value.csv", value)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dossiers": len(dossiers), "rotations": len(rotations),
        "policies": policies, "budgets_words": budgets,
        "messages": len(messages), "answers": len(answers),
        "lambda": cfg["analysis"]["lambda_default"],
        "beta": cfg["analysis"]["beta"], "gamma": cfg["analysis"]["gamma_default"],
        "bootstrap_resamples": cfg["analysis"]["bootstrap_resamples"],
    }
    (result_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[anticipatory:cost] " + json.dumps(sender.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
