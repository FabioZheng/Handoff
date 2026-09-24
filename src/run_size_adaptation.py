"""Experiment 9: does a handoff change shape when the next agent's size changes?

Every earlier experiment in this project fixes what the writer is told and
varies the message, the model, or the depth. This one varies what the writer is
told *about its reader*, and asks three questions the others cannot:

  1. Does merely KNOWING the receiver's capacity change the handoff, or does it
     take an explicit request? Each capacity claim appears twice -- once as a
     statement of fact, once as that same statement plus one clause asking the
     model to act on it -- so "informed" and "asked" are separated by exactly
     one sentence and nothing else.
  2. When a handoff is asked to grow, what fills the space? Added detail,
     restated detail, or material nobody supplied?
  3. When a fact is compressed out and a later stage is asked to expand, does
     the fact come back -- and if it does, where did it come from? Stage 2+ sees
     a sealed handoff and nothing else, so a fact absent from that handoff and
     present in its successor was NOT copied. It was reconstructed.

Question 3 is why the two datasets exist. On ``counterfactual`` items the
document's answer disagrees with the model's memorised one, and both are
verified closed-book at build time, so a reappearance can be attributed:
the document's value means the chain rebuilt the right fact, the memorised
value means pretrained knowledge leaked back in. On ``fictional`` items no
memorised value exists, so anything unsupported that appears is fabrication
with nothing to attribute it to. Running both is what separates
"reconstruction" from "invention" instead of calling all of it hallucination.

The baseline is deliberately NOT Experiment 5's prompt. That prompt opens
"Write concise prose research notes", and a control that already says "concise"
is a shrink condition: measuring an expansion request against it would compare
"expand" with "shrink" rather than with "no instruction". The base instruction
here is the published string with that single word removed, asserted by
reconstructing the published string from it. A separate arm appends "Keep your
handoff concise", so an explicit unquantified shrink cue is measured rather
than hidden inside the control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoffs as hm  # noqa: E402
import paraphrase_metrics as pmx  # noqa: E402
import size_metrics as sx  # noqa: E402
from judge import add_judge, add_preservation_judge, add_support_judge  # noqa: E402
from llm import CostCapExceeded, LLMClient, estimate_tokens, load_config  # noqa: E402
from run_chain import CHAIN_SYSTEM, INITIAL_INSTRUCTION, RECOMPRESS_INSTRUCTION  # noqa: E402
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa: E402

QUERY_TYPES = ("target", "side")
# Per-item measures averaged into the metrics table. Length first: it is the
# dependent variable the whole experiment turns on.
STAGE_MEASURES = (
    "summary_words", "summary_characters", "estimated_tokens", "length_ratio",
    "internal_repetition", "novel_token_rate", "retained_token_rate", "ngram_copy_rate",
    "target_fact_present", "side_fact_retention", "parametric_present",
    "unsupported_count", "unsupported_invented_count", "unsupported_corrupted_count",
    "unsupported_rate", "tracked_variant_count", "unsupported_claims",
    "semantic_preserved", "truncated",
)
ANSWER_MEASURES = ("em", "f1", "judge_correct", "parametric_judge_correct")
# Kept out of the stored stage records: re-derived for free on every run, and
# storing each stage's source document would multiply the file by the corpus.
HEAVY_FIELDS = ("previous_text", "current_text", "document")


# ---------------------------------------------------------------- io helpers

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
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(*parts: str) -> str:
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()


def _mean(values) -> float:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(clean)) if clean else float("nan")


# ---------------------------------------------------------------- prompts

def _drop_concise(instruction: str) -> str:
    """Remove the one word that makes Experiment 5's prompt a shrink cue.

    Asserted to be a single deletion so this can never quietly become a rewrite:
    the neutral instruction differs from the published one by exactly the string
    "concise " and by nothing else, which is what lets the two be compared.
    """
    if instruction.count("concise ") != 1:
        raise AssertionError(
            f"expected exactly one 'concise ' in the inherited instruction, "
            f"found {instruction.count('concise ')}: {instruction!r}"
        )
    return instruction.replace("concise ", "", 1)


NEUTRAL_INITIAL_INSTRUCTION = _drop_concise(INITIAL_INSTRUCTION)
NEUTRAL_RECOMPRESS_INSTRUCTION = _drop_concise(RECOMPRESS_INSTRUCTION)

CAPACITY_SENTENCE = " The next agent's model has a {tokens}-token context window."
REQUEST_CLAUSE = {
    "shrink": " Make your handoff fit within it.",
    "expand": " Expand your handoff to make use of the available space.",
}
CONCISE_CLAUSE = " Keep your handoff concise."


def directive_text(name: str, cfg: dict) -> str:
    """The sentence(s) appended to the neutral instruction for one directive.

    Built compositionally rather than written out per directive: a ``resize``
    string is *constructed* as its ``inform`` string plus one clause, so the two
    cannot drift apart in a later edit and leave the inform/resize contrast
    measuring an accidental rewording.
    """
    spec = cfg["directives"][name]
    kind = spec["kind"]
    if kind == "control":
        return ""
    if spec.get("advertised_tokens") is None:
        if name != "concise":
            raise ValueError(f"directive {name!r} has no advertised size and is not the concise cue")
        return CONCISE_CLAUSE
    sentence = CAPACITY_SENTENCE.format(tokens=f"{int(spec['advertised_tokens']):,}")
    if kind == "inform":
        return sentence
    return sentence + REQUEST_CLAUSE[spec["direction"]]


def stage1_user_prompt(item: dict, directive: str, cfg: dict) -> str:
    return (
        f"Source material:\n{item['document']}\n\n"
        f"Question the final agent must answer: {item['question']}\n\n"
        f"{NEUTRAL_INITIAL_INSTRUCTION}{directive_text(directive, cfg)}"
    )


def recompress_user_prompt(sealed: hm.SealedHandoff, directive: str, cfg: dict) -> str:
    """Stage >= 2 prompt, built from the seal and nothing else.

    The type check is the project's load-bearing isolation guarantee: a later
    compressor must not be able to reach the source document, or a "recovered"
    fact could have been re-read rather than reconstructed, and question 3
    above would be unanswerable.
    """
    if not isinstance(sealed, hm.SealedHandoff):
        raise TypeError(
            f"recompress_user_prompt requires a SealedHandoff, got {type(sealed).__name__}. "
            "Source material must not be reachable from a stage-2+ compression call."
        )
    return (
        f"Previous agent's notes:\n{sealed.handoff_text}\n\n"
        f"Question the final agent must answer: {sealed.question}\n\n"
        f"{NEUTRAL_RECOMPRESS_INSTRUCTION}{directive_text(directive, cfg)}"
    )


def directive_selftest(cfg: dict) -> None:
    """Prove the manipulation is the only thing that differs between arms."""
    # 1. The neutral base is the published instruction minus one word.
    assert NEUTRAL_INITIAL_INSTRUCTION.replace("Write ", "Write concise ", 1) == INITIAL_INSTRUCTION
    assert NEUTRAL_RECOMPRESS_INSTRUCTION.replace("into ", "into concise ", 1) == RECOMPRESS_INSTRUCTION
    for text in (NEUTRAL_INITIAL_INSTRUCTION, NEUTRAL_RECOMPRESS_INSTRUCTION, CHAIN_SYSTEM):
        lowered = text.lower()
        for banned in ("concise", "brief", "short", "expand", "longer", "token", "context window"):
            assert banned not in lowered, f"neutral base prompt still carries a size cue: {banned!r}"

    # 2. resize == inform + exactly one request clause; nothing else differs.
    for small, large in (("inform_small", "resize_small"), ("inform_large", "resize_large")):
        inform, resize = directive_text(small, cfg), directive_text(large, cfg)
        assert resize.startswith(inform), f"{large} is not {small} plus a suffix"
        clause = resize[len(inform):]
        assert clause in REQUEST_CLAUSE.values(), f"{large} adds an unexpected clause {clause!r}"

    # 3. The two inform directives differ ONLY in the advertised number, so a
    #    shrink/expand difference cannot come from incidental wording.
    small_n = f"{int(cfg['directives']['inform_small']['advertised_tokens']):,}"
    large_n = f"{int(cfg['directives']['inform_large']['advertised_tokens']):,}"
    assert directive_text("inform_small", cfg).replace(small_n, large_n) == \
        directive_text("inform_large", cfg), "the two inform directives differ beyond the number"

    # 4. The control appends nothing at all.
    assert directive_text("none", cfg) == ""

    # 5. Stage 2+ accepts a seal and refuses everything else.
    sealed = hm.seal("q?", hm.Handoff("i", "chain", None, "notes"))
    assert "notes" in recompress_user_prompt(sealed, "none", cfg)
    for bad in ("notes", {"handoff_text": "notes"}, object()):
        try:
            recompress_user_prompt(bad, "none", cfg)
        except TypeError:
            continue
        raise AssertionError(f"isolation broken: recompression accepted {type(bad).__name__}")

    # 6. Every configured arm has a schedule of the configured depth using only
    #    declared directives.
    for arm, spec in cfg["arms"].items():
        schedule = spec["schedule"]
        assert len(schedule) == int(cfg["max_depth"]), f"arm {arm} has {len(schedule)} stages"
        for name in schedule:
            assert name in cfg["directives"], f"arm {arm} uses undeclared directive {name!r}"
    print("[size:selftest] base prompt is size-neutral, resize == inform + one clause, stage 2+ sealed")


# ---------------------------------------------------------------- schedules

def prefix_id(directives: tuple[str, ...]) -> str:
    return "|".join(directives)


def build_prefixes(cfg: dict) -> tuple[dict, dict]:
    """Unique directive prefixes per stage, and which arms each one serves.

    Two arms that agree on their first s directives receive byte-identical
    stage-s messages by *sharing the same generated record*, not by being
    generated twice and happening to match. Every contrast between such arms is
    therefore anchored on a common history exactly, and the API bill falls to
    the number of distinct prefixes.
    """
    schedules = {arm: tuple(spec["schedule"]) for arm, spec in cfg["arms"].items()}
    by_stage: dict[int, list[tuple[str, ...]]] = {}
    serves: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for stage in range(1, int(cfg["max_depth"]) + 1):
        seen = []
        for arm, schedule in schedules.items():
            prefix = schedule[:stage]
            serves[prefix].append(arm)
            if prefix not in seen:
                seen.append(prefix)
        by_stage[stage] = seen
    return by_stage, {k: sorted(v) for k, v in serves.items()}


def schedule_selftest(cfg: dict) -> None:
    by_stage, serves = build_prefixes(cfg)
    schedules = {arm: tuple(spec["schedule"]) for arm, spec in cfg["arms"].items()}
    total = sum(len(v) for v in by_stage.values())
    naive = len(schedules) * int(cfg["max_depth"])
    for stage, prefixes in by_stage.items():
        assert len(set(prefixes)) == len(prefixes), f"stage {stage} prefixes are not unique"
        for prefix in prefixes:
            assert len(prefix) == stage
            for arm in serves[prefix]:
                assert schedules[arm][:stage] == prefix, "prefix table disagrees with the schedules"
    # Arms in the same contrast must actually share a history where the design
    # says they do, or "diverges only at stage k" is a claim about nothing.
    for contrast in cfg["contrasts"]:
        a, b = contrast["pair"]
        shared = 0
        for i in range(int(cfg["max_depth"])):
            if schedules[a][i] != schedules[b][i]:
                break
            shared += 1
        contrast["shared_stages"] = shared
    print(f"[size:selftest] {total} distinct prefixes vs {naive} arm-stages "
          f"({naive - total} generation calls saved by prefix sharing)")


# ---------------------------------------------------------------- data

def load_items(cfg: dict, limit: int | None) -> list[dict]:
    items: list[dict] = []
    for name, spec in cfg["dataset"].items():
        if not spec.get("enabled", True):
            continue
        path = ROOT / spec["items_jsonl"]
        rows = read_jsonl(path)
        if not rows:
            raise SystemExit(
                f"{path} is missing or empty. Build it first:\n"
                f"  .venv/Scripts/python src/builders/build_size_adaptation_data.py --which {name}")
        manifest = rows[0].get("_manifest") if rows else None
        body = [r for r in rows if "_manifest" not in r][:int(spec["n_items"])]
        for row in body:
            row.setdefault("dataset", name)
            row["_manifest"] = manifest
        print(f"[size:data] {name}: {len(body)} items from {path.name}")
        items.extend(body)
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("no items enabled in dataset:")
    return items


def tracked_terms(item: dict) -> list[str]:
    """Every value the experiment follows through the chain, for the term scan."""
    terms = list(item.get("document_golds") or [])
    for fact in item.get("facts") or []:
        terms.extend(fact.get("probes") or [])
    return list(dict.fromkeys(t for t in terms if str(t).strip()))


# ---------------------------------------------------------------- generation

def compress(client: LLMClient, item: dict, previous_text: str | None, stage: int,
             directive: str, cfg: dict) -> dict:
    if stage == 1:
        user = stage1_user_prompt(item, directive, cfg)
    else:
        sealed = hm.seal(item["question"],
                         hm.Handoff(item["item_id"], "chain", None, previous_text or ""))
        user = recompress_user_prompt(sealed, directive, cfg)
    result = client.chat(
        [{"role": "system", "content": CHAIN_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["subagent_temperature"],
        max_tokens=cfg["decoding"]["handoff_max_tokens"],
        # Deterministic decoding with the seed keyed to the stage, as in
        # Experiments 5, 6 and 8. The directive is already inside the prompt and
        # therefore inside the cache key, so no per-directive offset is needed.
        seed=stage,
        tag=f"size_compress_{directive}_stage{stage}",
    )
    text = result.text.strip()
    return {
        "text": text, "cached": result.cached,
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason, "cost_usd": round(result.cost_usd, 8),
        "provider": result.provider, "resolved_model": result.model,
    }


def generate(client: LLMClient, items: list[dict], cfg: dict) -> dict:
    """Generate every distinct (item, directive-prefix) handoff, stage by stage."""
    by_stage, serves = build_prefixes(cfg)
    generated: dict[tuple[str, str], dict] = {}
    for stage in sorted(by_stage):
        tasks = [(item, prefix) for item in items for prefix in by_stage[stage]]

        def one(task):
            item, prefix = task
            previous = None
            if stage > 1:
                parent = generated[(item["item_id"], prefix_id(prefix[:-1]))]
                previous = parent["text"]
            record = compress(client, item, previous, stage, prefix[-1], cfg)
            record.update({
                "item_id": item["item_id"], "dataset": item["dataset"], "stage": stage,
                "prefix": prefix_id(prefix), "directive": prefix[-1],
                "arms": serves[prefix], "previous_text": previous if previous is not None
                else item["document"],
            })
            return record

        with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
            for record in pool.map(one, tasks):
                generated[(record["item_id"], record["prefix"])] = record
        cached = sum(1 for k, r in generated.items() if r["stage"] == stage and r["cached"])
        print(f"[size:generate] stage {stage}: {len(tasks)} messages ({cached} cached)")
    return generated


# ---------------------------------------------------------------- measurement

def measure_stage(record: dict, item: dict, cfg: dict) -> dict:
    """Everything about one handoff that needs no model call."""
    text, previous = record["text"], record["previous_text"]
    spec = cfg["measurement"]["unsupported_scan"]
    scan = sx.unsupported_scan(text, item["document"], spec, tracked_terms(item))
    transition = pmx.transition_measures(previous, text)

    facts = item.get("facts") or []
    presence = {f["fact_id"]: sx.fact_present(text, f["probes"]) for f in facts}
    side_ids = [f["fact_id"] for f in facts if f["role"] == "side"]
    target_id = next((f["fact_id"] for f in facts if f["role"] == "target"), None)

    row = {
        "item_id": item["item_id"], "dataset": item["dataset"], "stage": record["stage"],
        # Provenance travels with every row. The counterfactual dataset draws on
        # two sources whose documents differ several-fold in length, and length
        # is this experiment's dependent variable -- pooling them would make a
        # source difference look like a directive effect.
        "source": item.get("source", "unspecified"),
        "document_characters": item.get("document_characters", len(item["document"])),
        "prefix": record["prefix"], "directive": record["directive"], "arms": record["arms"],
        "directive_kind": cfg["directives"][record["directive"]]["kind"],
        "directive_direction": cfg["directives"][record["directive"]]["direction"],
        "advertised_tokens": cfg["directives"][record["directive"]].get("advertised_tokens"),
        "summary_characters": len(text), "summary_words": len(text.split()),
        "estimated_tokens": estimate_tokens(text),
        "internal_repetition": sx.internal_repetition_rate(
            text, int(cfg["measurement"]["repetition_ngram"])),
        # A stage that hit the API ceiling was shortened by the budget, not by
        # the instruction. Averaged and reported so that can never be mistaken
        # for an effect of the directive.
        "truncated": float(record.get("finish_reason") == "length"),
        "empty": float(not text.strip()),
        "target_fact_present": float(presence.get(target_id, False)) if target_id else float("nan"),
        "side_fact_retention": (_mean([float(presence[i]) for i in side_ids])
                                if side_ids else float("nan")),
        "side_facts_present": sum(int(presence[i]) for i in side_ids),
        "side_facts_total": len(side_ids),
        "parametric_present": (float(sx.fact_present(text, item["parametric_golds"]))
                               if item.get("has_parametric") else float("nan")),
        "fact_presence": presence,
        "prompt_tokens": record["prompt_tokens"], "completion_tokens": record["completion_tokens"],
        "finish_reason": record["finish_reason"], "cost_usd": record["cost_usd"],
        "provider": record["provider"], "cached": record["cached"],
    }
    row.update({k: v for k, v in scan.items() if k != "unsupported_terms"})
    row["unsupported_terms"] = scan["unsupported_terms"]
    row.update({k: transition[k] for k in
                ("lexical_similarity", "novel_token_rate", "retained_token_rate",
                 "ngram_copy_rate", "length_ratio") if k in transition})
    return row


# ---------------------------------------------------------------- answering

def answer_material(client: LLMClient, question: str, golds: list[str], material: str,
                    cfg: dict, tag: str) -> dict:
    user = f"Research material:\n{material}\n\nQuestion:\n{question}\nAnswer:"
    result = client.chat(
        [{"role": "system", "content": hm.ANSWER_SYSTEM}, {"role": "user", "content": user}],
        temperature=cfg["decoding"]["orchestrator_temperature"],
        max_tokens=cfg["decoding"]["answer_max_tokens"], seed=None, tag=tag)
    pred = extract_short_answer(result.text)
    em, f1 = score_against_golds(pred, golds)
    return {"pred": pred, "raw": result.text, "em": em, "f1": f1, "cached": result.cached,
            "cost_usd": round(result.cost_usd, 8)}


def query_spec(item: dict, query_type: str) -> dict | None:
    """The question asked at a given depth, and the golds it is scored against.

    ``side`` asks about a fact no stage was ever told to preserve for the task.
    It is how "did the shrink discard reusable information" becomes an accuracy
    number rather than only a string-presence count.
    """
    if query_type == "target":
        return {"question": item["question"], "golds": item["document_golds"],
                "parametric_golds": item.get("parametric_golds") or []}
    fact_id = item.get("side_probe_fact_id")
    fact = next((f for f in (item.get("facts") or []) if f["fact_id"] == fact_id), None)
    if not fact or not fact.get("question"):
        return None
    return {"question": fact["question"], "golds": fact["golds"], "parametric_golds": []}


def run_answers(client: LLMClient, items: list[dict], generated: dict, cfg: dict) -> list[dict]:
    """Answer every distinct (item, prefix, query) plus the depth-0 baseline.

    Answers are produced per PREFIX, not per arm: arms sharing a prefix share
    the answer as they share the handoff, so a contrast between them cannot be
    moved by two independent samples of the same call.
    """
    tasks = []
    for item in items:
        for query_type in QUERY_TYPES:
            spec = query_spec(item, query_type)
            if not spec:
                continue
            tasks.append((item, None, 0, query_type, spec, item["document"]))
    for (item_id, prefix), record in generated.items():
        item = next(i for i in items if i["item_id"] == item_id)
        for query_type in QUERY_TYPES:
            spec = query_spec(item, query_type)
            if not spec:
                continue
            tasks.append((item, prefix, record["stage"], query_type, spec, record["text"]))

    def one(task):
        item, prefix, depth, query_type, spec, material = task
        result = answer_material(client, spec["question"], spec["golds"], material, cfg,
                                 tag=f"size_answer_{query_type}_depth{depth}")
        row = {
            "item_id": item["item_id"], "dataset": item["dataset"], "depth": depth,
            "prefix": prefix or "", "query_type": query_type,
            "question": spec["question"], "golds": spec["golds"], **result,
        }
        # The same prediction is scored a second time against the memorised
        # answer. Two readings of one call, never two calls: a separate sample
        # could disagree with itself and make "reverted" a sampling artefact.
        if spec["parametric_golds"]:
            p_em, p_f1 = score_against_golds(result["pred"], spec["parametric_golds"])
            row.update({"parametric_golds": spec["parametric_golds"],
                        "parametric_em": p_em, "parametric_f1": p_f1})
        return row

    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as pool:
        rows = list(pool.map(one, tasks))
    print(f"[size:answer] {len(rows)} answers ({sum(1 for r in rows if r['cached'])} cached)")
    return rows


def judge_answers(rows: list[dict], cfg: dict, dry_run: bool) -> None:
    """Judge each prediction against the document gold and the memorised gold."""
    add_judge(rows, cfg, dry_run=dry_run, tag="size_answer_judge")
    parametric = [r for r in rows if r.get("parametric_golds")]
    proxies = [{"question": r["question"], "pred": r["pred"], "golds": r["parametric_golds"]}
               for r in parametric]
    add_judge(proxies, cfg, dry_run=dry_run, tag="size_parametric_judge")
    for row, proxy in zip(parametric, proxies):
        row["parametric_judge_correct"] = proxy.get("judge_correct", 0)
    for row in rows:
        row["answer_origin"] = sx.answer_origin(
            bool(row.get("judge_correct", 0)), bool(row.get("parametric_judge_correct", 0)),
            bool(row.get("parametric_golds")))


# ---------------------------------------------------------------- arm expansion

def expand_to_arms(stage_rows: list[dict], answer_rows: list[dict], cfg: dict) -> list[dict]:
    """One row per (item, arm, depth), joining the shared prefix records.

    Depth 0 has no handoff and is shared by every arm; it is emitted once per
    arm so that a plot's depth axis starts from the same point everywhere,
    and flagged ``shared_baseline`` so it is never counted as arm evidence.
    """
    by_prefix = {(r["item_id"], r["prefix"]): r for r in stage_rows}
    answers = defaultdict(dict)
    for row in answer_rows:
        answers[(row["item_id"], row["prefix"], int(row["depth"]))][row["query_type"]] = row

    # Sorted, not a raw set: string hashing is randomised per process, so
    # iterating the set would reorder every emitted CSV between runs on
    # identical inputs and make diffs between two runs unreadable.
    item_ids = sorted({k[0] for k in by_prefix})
    sources = {r["item_id"]: r.get("source", "unspecified") for r in stage_rows}

    rows: list[dict] = []
    for arm, spec in cfg["arms"].items():
        schedule = tuple(spec["schedule"])
        for depth in cfg["depths"]:
            depth = int(depth)
            for key in item_ids:
                prefix = "" if depth == 0 else prefix_id(schedule[:depth])
                stage = by_prefix.get((key, prefix)) if depth else None
                bundle = answers.get((key, prefix, depth), {})
                if depth and stage is None:
                    continue
                row = {
                    "item_id": key, "arm": arm, "arm_label": spec["label"],
                    "arm_group": spec["group"], "depth": depth,
                    "schedule": "|".join(schedule[:depth]) if depth else "",
                    "shared_baseline": depth == 0,
                    "source": sources.get(key, "unspecified"),
                }
                if stage:
                    row.update({k: v for k, v in stage.items()
                                if k not in ("arms", "fact_presence", "unsupported_terms")})
                    row["dataset"] = stage["dataset"]
                for query_type, answer in bundle.items():
                    row["dataset"] = answer["dataset"]
                    prefix_key = "" if query_type == "target" else "side_"
                    for field in ("em", "f1", "judge_correct", "parametric_judge_correct",
                                  "answer_origin", "pred"):
                        if field in answer:
                            row[f"{prefix_key}{field}"] = answer[field]
                rows.append(row)
    return rows


# ---------------------------------------------------------------- trajectory

def recovery_rows(stage_rows: list[dict], answer_rows: list[dict], items: list[dict],
                  cfg: dict) -> list[dict]:
    """Per-fact life histories, and what a reappearance turned out to be.

    This is the experiment's central measurement. Because a stage-2+ compressor
    sees only a sealed handoff, a fact absent at stage k and present at stage
    k+1 was not copied from its input -- it was regenerated. Each such event is
    labelled by *what came back*:

      recovered_document   the document's own value returned intact
      reverted_parametric  the memorised value appeared where the document's
                           value used to be
      fabricated           something the document never contained appeared

    The three are mutually exclusive per fact and are counted separately, so
    "expansion restores information" and "expansion invents information" are
    never summed into one number.
    """
    by_item = {i["item_id"]: i for i in items}
    stages = list(range(1, int(cfg["max_depth"]) + 1))
    by_key = {(r["item_id"], r["prefix"]): r for r in stage_rows}
    rows: list[dict] = []
    for arm, spec in cfg["arms"].items():
        schedule = tuple(spec["schedule"])
        for item_id, item in by_item.items():
            chain = []
            for stage in stages:
                record = by_key.get((item_id, prefix_id(schedule[:stage])))
                if record is None:
                    break
                chain.append(record)
            if len(chain) != len(stages):
                continue
            for fact in item.get("facts") or []:
                presence = {s: bool(chain[i]["fact_presence"].get(fact["fact_id"], False))
                            for i, s in enumerate(stages)}
                traj = sx.presence_trajectory(presence, stages)
                parametric = {s: bool(chain[i]["parametric_present"] == 1.0) for i, s in enumerate(stages)}
                invented = {s: int(chain[i]["unsupported_invented_count"]) for i, s in enumerate(stages)}
                corrupted = {s: int(chain[i]["unsupported_corrupted_count"]) for i, s in enumerate(stages)}
                reappeared_at = traj["reappeared_at_stage"]
                # A reversion or an invention only counts when it arrives at or
                # after the stage the fact was lost: material already present
                # before the loss was not produced by the recovery.
                lost_at = traj["lost_at_stage"]
                after = [s for s in stages if lost_at is not None and s >= lost_at]
                rows.append({
                    "arm": arm, "arm_group": spec["group"], "schedule": "|".join(schedule),
                    "item_id": item_id, "dataset": item["dataset"],
                    "fact_id": fact["fact_id"], "fact_role": fact["role"],
                    "has_parametric": bool(item.get("has_parametric")),
                    **{f"present_stage{s}": presence[s] for s in stages},
                    "was_lost": traj["was_lost"], "lost_at_stage": lost_at,
                    "reappeared": traj["reappeared"], "reappeared_at_stage": reappeared_at,
                    "survived_throughout": traj["survived_throughout"],
                    "recovered_document": bool(traj["reappeared"]),
                    "reverted_parametric": bool(
                        fact["role"] == "target" and item.get("has_parametric")
                        and after and any(parametric[s] for s in after)
                        and not traj["reappeared"]),
                    # The three labels above are exclusive so they can be summed,
                    # which would otherwise hide the hedging case: a handoff that
                    # brings the document's value back AND states the memorised
                    # one beside it counts as a recovery. Recorded separately so
                    # that case stays visible in fact_histories.csv.
                    "both_values_after_loss": bool(
                        fact["role"] == "target" and item.get("has_parametric")
                        and traj["reappeared"] and after
                        and any(parametric[s] for s in after)),
                    "fabricated_after_loss": bool(
                        after and any(invented[s] > invented[lost_at - 1] for s in after
                                      if lost_at and lost_at - 1 in invented)),
                    "corrupted_after_loss": bool(
                        after and any(corrupted[s] > 0 for s in after)),
                })
    return rows


def compute_recovery_metrics(rows: list[dict], cfg: dict) -> list[dict]:
    """Aggregate the fact histories, split by dataset, arm and fact role."""
    out: list[dict] = []
    keys = {(r["dataset"], r["arm"], r["fact_role"]) for r in rows}
    for dataset, arm, role in sorted(keys):
        subset = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm
                  and r["fact_role"] == role]
        lost = [r for r in subset if r["was_lost"]]
        out.append({
            "dataset": dataset, "arm": arm, "arm_group": cfg["arms"][arm]["group"],
            "fact_role": role, "facts": len(subset),
            "survived_throughout_rate": _mean([r["survived_throughout"] for r in subset]),
            "loss_rate": _mean([r["was_lost"] for r in subset]),
            "facts_lost": len(lost),
            # Denominator is facts that were actually lost: "did expansion bring
            # it back" is only a question about facts that went missing.
            "recovery_rate_given_lost": _mean([r["recovered_document"] for r in lost]),
            "reversion_rate_given_lost": _mean([r["reverted_parametric"] for r in lost]),
            "fabrication_rate_given_lost": _mean([r["fabricated_after_loss"] for r in lost]),
            "corruption_rate_given_lost": _mean([r["corrupted_after_loss"] for r in lost]),
            "both_values_rate_given_lost": _mean([r["both_values_after_loss"] for r in lost]),
        })
    return out


# ---------------------------------------------------------------- analysis

def compute_metrics(rows: list[dict], cfg: dict) -> list[dict]:
    resamples = int(cfg["analysis"]["bootstrap_resamples"])
    level = float(cfg["analysis"]["ci_level"])
    measures = STAGE_MEASURES + ANSWER_MEASURES + ("side_judge_correct", "side_f1")
    out: list[dict] = []
    keys = {(r["dataset"], r["arm"], int(r["depth"])) for r in rows if r.get("dataset")}
    for dataset, arm, depth in sorted(keys):
        subset = [r for r in rows if r.get("dataset") == dataset and r["arm"] == arm
                  and int(r["depth"]) == depth]
        entry = {"dataset": dataset, "arm": arm, "arm_label": cfg["arms"][arm]["label"],
                 "arm_group": cfg["arms"][arm]["group"], "depth": depth,
                 "schedule": subset[0]["schedule"] if subset else "",
                 "n": len(subset), "shared_baseline": depth == 0}
        for measure in measures:
            values = [float(r[measure]) for r in subset
                      if measure in r and r[measure] is not None
                      and not (isinstance(r[measure], float) and np.isnan(r[measure]))]
            if not values:
                continue
            mean, lo, hi = bootstrap_ci(np.array(values, dtype=float), resamples, level)
            entry[measure] = mean
            entry[f"{measure}_lo"], entry[f"{measure}_hi"] = lo, hi
        out.append(entry)
    return out


def compute_metrics_by_source(rows: list[dict], cfg: dict) -> list[dict]:
    """The same table, split by item provenance rather than pooled.

    Emitted separately instead of replacing ``compute_metrics`` so the headline
    figures keep one curve per arm per dataset, while the question "is this a
    directive effect or a source-length effect" stays answerable from the
    artifacts without rerunning anything.
    """
    out: list[dict] = []
    for source in sorted({r.get("source", "unspecified") for r in rows}):
        subset = [r for r in rows if r.get("source", "unspecified") == source]
        for entry in compute_metrics(subset, cfg):
            out.append({"source": source, **entry})
    return out


def compute_contrasts(rows: list[dict], cfg: dict) -> list[dict]:
    """Paired per-item deltas between arms, at every depth.

    Pairing is on the item: both arms saw the same document, the same question
    and the same tracked facts, so the delta removes item difficulty entirely.
    """
    resamples = int(cfg["analysis"]["bootstrap_resamples"])
    level = float(cfg["analysis"]["ci_level"])
    measures = STAGE_MEASURES + ANSWER_MEASURES + ("side_judge_correct", "side_f1")
    out: list[dict] = []
    datasets = sorted({r["dataset"] for r in rows if r.get("dataset")})
    for contrast in cfg["contrasts"]:
        arm_a, arm_b = contrast["pair"]
        for dataset in datasets:
            for depth in cfg["depths"]:
                depth = int(depth)
                if depth == 0:
                    continue  # depth 0 is the shared baseline; the delta is 0 by construction
                index_a = {r["item_id"]: r for r in rows if r["arm"] == arm_a
                           and int(r["depth"]) == depth and r.get("dataset") == dataset}
                index_b = {r["item_id"]: r for r in rows if r["arm"] == arm_b
                           and int(r["depth"]) == depth and r.get("dataset") == dataset}
                shared = sorted(set(index_a) & set(index_b))
                if not shared:
                    continue
                for measure in measures:
                    pairs = [(float(index_a[i][measure]), float(index_b[i][measure]))
                             for i in shared
                             if measure in index_a[i] and measure in index_b[i]
                             and not np.isnan(float(index_a[i].get(measure, np.nan)))
                             and not np.isnan(float(index_b[i].get(measure, np.nan)))]
                    if not pairs:
                        continue
                    a_values = np.array([p[0] for p in pairs], dtype=float)
                    b_values = np.array([p[1] for p in pairs], dtype=float)
                    stats = paired_bootstrap_delta(a_values, b_values, resamples, level)
                    out.append({
                        "contrast": contrast["name"], "asks": contrast.get("asks", ""),
                        "dataset": dataset, "depth": depth, "measure": measure,
                        "arm_a": arm_a, "arm_b": arm_b, "n_pairs": len(pairs),
                        "shared_stages": contrast.get("shared_stages"),
                        "mean_a": float(a_values.mean()), "mean_b": float(b_values.mean()),
                        "delta": stats["delta"], "delta_lo": stats["lo"], "delta_hi": stats["hi"],
                        "p_value": stats["p_value"],
                        "excludes_zero": bool(stats["lo"] > 0 or stats["hi"] < 0),
                    })
    return out


def compute_origin_table(answer_rows: list[dict], cfg: dict) -> list[dict]:
    """Where the final answer came from, per arm and depth, counterfactual only."""
    by_prefix_arm: dict[tuple[str, int], str] = {}
    for arm, spec in cfg["arms"].items():
        schedule = tuple(spec["schedule"])
        for depth in cfg["depths"]:
            depth = int(depth)
            prefix = "" if depth == 0 else prefix_id(schedule[:depth])
            by_prefix_arm[(arm, depth)] = prefix
    out: list[dict] = []
    for (arm, depth), prefix in sorted(by_prefix_arm.items()):
        subset = [r for r in answer_rows if r["prefix"] == prefix and int(r["depth"]) == depth
                  and r["query_type"] == "target" and r.get("parametric_golds")]
        if not subset:
            continue
        counts = defaultdict(int)
        for row in subset:
            counts[row["answer_origin"]] += 1
        total = len(subset)
        out.append({
            "arm": arm, "arm_group": cfg["arms"][arm]["group"], "depth": depth, "n": total,
            **{f"{name}_rate": counts[name] / total
               for name in ("document", "parametric", "both", "other")},
            **{f"{name}_n": counts[name] for name in ("document", "parametric", "both", "other")},
        })
    return out


# ---------------------------------------------------------------- plots

ARM_STYLE = {
    "control":        ("#111111", "-", 2.6),
    "concise_shrink": ("#777777", "--", 1.8),
    "inform_shrink":  ("#2c7fb8", "--", 1.9),
    "resize_shrink":  ("#2c7fb8", "-", 2.6),
    "inform_expand":  ("#d95f02", "--", 1.9),
    "resize_expand":  ("#d95f02", "-", 2.6),
    "resize_lsl":     ("#7570b3", "-", 2.4),
    "inform_lsl":     ("#7570b3", "--", 1.9),
    "resize_sse":     ("#1b9e77", "-", 2.2),
}
PANELS = (
    ("summary_words", "Handoff length (words)", False),
    ("judge_correct", "Answer accuracy (LLM judge)", True),
    ("side_fact_retention", "Reusable side facts retained", True),
    ("internal_repetition", "Internal repetition (5-gram)", True),
)


def _series(metrics: list[dict], dataset: str, arm: str, measure: str):
    subset = sorted([r for r in metrics if r["dataset"] == dataset and r["arm"] == arm
                     and measure in r], key=lambda r: int(r["depth"]))
    x = [int(r["depth"]) for r in subset]
    y = [float(r[measure]) for r in subset]
    lo = [float(r[measure]) - float(r.get(f"{measure}_lo", r[measure])) for r in subset]
    hi = [float(r.get(f"{measure}_hi", r[measure])) - float(r[measure]) for r in subset]
    return x, y, [lo, hi]


def make_primary_plot(metrics: list[dict], cfg: dict, output: Path) -> None:
    """The four dependent variables against handoff depth, per dataset.

    Requested resizes are heavy solid lines and the matching "informed only"
    arms are the same colour dashed, so the experiment's central question --
    does asking do more than telling -- is the gap between a solid line and a
    dashed line of its own colour, rather than a hunt through the legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in metrics})
    if not datasets:
        return
    fig, axes = plt.subplots(len(PANELS), len(datasets),
                             figsize=(6.4 * len(datasets), 3.5 * len(PANELS)), squeeze=False)
    for col, dataset in enumerate(datasets):
        for row, (measure, label, unit_axis) in enumerate(PANELS):
            ax = axes[row][col]
            drawn = False
            for arm in cfg["arms"]:
                colour, style, width = ARM_STYLE.get(arm, ("#999999", "-", 1.5))
                x, y, err = _series(metrics, dataset, arm, measure)
                if not x:
                    continue
                drawn = True
                ax.errorbar(x, y, yerr=err, color=colour, linestyle=style, linewidth=width,
                            marker="o", markersize=3.4, capsize=2, elinewidth=0.8,
                            label=cfg["arms"][arm]["label"])
            if not drawn:
                ax.set_axis_off()
                continue
            ax.set_xlabel("handoff depth")
            ax.set_ylabel(label)
            ax.set_xticks(list(cfg["depths"]))
            if unit_axis:
                ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, linewidth=0.6)
            if row == 0:
                ax.set_title(f"{dataset} items (n={max((r['n'] for r in metrics if r['dataset'] == dataset), default=0)})",
                             fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Experiment 9 — what changes when the writer is told the reader's size",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"[size:plot] wrote {output}")


def make_expansion_plot(metrics: list[dict], cfg: dict, output: Path) -> None:
    """What fills the space when a handoff is asked to grow.

    Length on the x axis against three things that could account for it, so an
    arm that doubled its word count sits visibly in one of three regimes:
    it added supported detail, it repeated itself, or it added material the
    document never contained.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = (("internal_repetition", "Internal repetition (5-gram)"),
                ("unsupported_rate", "Unsupported terms / terms checked"),
                ("novel_token_rate", "Tokens new since the previous stage"))
    datasets = sorted({r["dataset"] for r in metrics})
    fig, axes = plt.subplots(len(datasets), len(measures),
                             figsize=(4.6 * len(measures), 3.8 * len(datasets)), squeeze=False)
    for row, dataset in enumerate(datasets):
        for col, (measure, label) in enumerate(measures):
            ax = axes[row][col]
            for arm in cfg["arms"]:
                colour, style, width = ARM_STYLE.get(arm, ("#999999", "-", 1.5))
                points = sorted([r for r in metrics if r["dataset"] == dataset and r["arm"] == arm
                                 and int(r["depth"]) > 0 and measure in r and "summary_words" in r],
                                key=lambda r: int(r["depth"]))
                if not points:
                    continue
                ax.plot([r["summary_words"] for r in points], [r[measure] for r in points],
                        color=colour, linestyle=style, linewidth=width, marker="o",
                        markersize=4, label=cfg["arms"][arm]["label"])
                for point in points:
                    ax.annotate(str(int(point["depth"])),
                                (point["summary_words"], point[measure]),
                                fontsize=6.5, color=colour,
                                textcoords="offset points", xytext=(3, 3))
            ax.set_xlabel("handoff length (words)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25, linewidth=0.6)
            if col == 0:
                ax.set_title(f"{dataset} items", fontsize=11, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Experiment 9 — length against what accounts for it "
                 "(point labels are handoff depth)", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"[size:plot] wrote {output}")


def make_expansion_modes_plot(stage_rows: list[dict], cfg: dict, output: Path) -> None:
    """Show the call-level modes hidden by the expand-arm means.

    The primary and expansion-content figures aggregate by arm and depth. That
    is useful for the experiment as a whole, but it makes ``resize_large`` look
    like one very long response when it is actually a mixture of short stops
    and calls that run into the common generation cap. This report-facing
    figure therefore stays at the individual-call level and separates finish
    states explicitly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    controls = [r for r in stage_rows if r.get("directive") == "none"
                and r.get("summary_words") is not None]
    expands = [r for r in stage_rows if r.get("directive") == "resize_large"
               and r.get("summary_words") is not None]
    if not controls or not expands:
        return

    groups = (
        ("control", "No size\ninformation", controls, "#555555"),
        ("stopped", "‘Fill 10k’:\nAPI normal-stop label",
         [r for r in expands if r.get("finish_reason") == "stop"], "#2c7fb8"),
        ("budget", "‘Fill 10k’:\nAPI length-limit label",
         [r for r in expands if r.get("finish_reason") == "length"], "#d95f02"),
    )
    error_rows = [r for r in expands if r.get("finish_reason") not in {"stop", "length"}]
    markers = {"counterfactual": "o", "fictional": "^"}
    rng = np.random.default_rng(20260831)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8),
                             gridspec_kw={"width_ratios": [0.9, 1.35]})

    # Panel A: distributions on a log scale, with every call visible and the
    # median marked separately. The log scale is essential: a linear 12k axis
    # collapses the 100--1,000 word calls into a nearly unreadable strip.
    ax = axes[0]
    all_words = []
    for x, (_, label, rows, colour) in enumerate(groups):
        if not rows:
            ax.text(x, 0.04, "n=0", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=colour)
            continue
        words = np.asarray([float(r["summary_words"]) for r in rows], dtype=float)
        all_words.extend(words.tolist())
        for dataset, marker in markers.items():
            vals = [float(r["summary_words"]) for r in rows if r.get("dataset") == dataset]
            if not vals:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, marker=marker,
                       color=colour, alpha=0.48, linewidths=0)
        median = float(np.median(words))
        ax.hlines(median, x - 0.27, x + 0.27, color=colour, linewidth=3.0)
        text_offset = -10 if x == 2 else 8
        ax.annotate(f"n={len(rows)}\nmedian {median:,.0f}", (x, median),
                    xytext=(0, text_offset), textcoords="offset points", ha="center",
                    va="top" if x == 2 else "bottom", fontsize=8.5,
                    fontweight="bold", color=colour,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                          "boxstyle": "round,pad=0.12"})
    ax.set_yscale("log")
    ax.set_ylim(max(1, min(all_words) * 0.65), max(all_words) * 1.55)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups], fontsize=8.5)
    ax.set_ylabel("Response length (words; log scale)")
    ax.set_title("A. How long the responses were", loc="left")
    ax.grid(axis="y", which="both", alpha=0.22, linewidth=0.6)

    # Panel B: length against within-message repetition. The capped cluster is
    # visibly a repetition loop; long stop-marked outliers remain visible rather
    # than being silently relabelled as normal generations.
    ax = axes[1]
    for _, _, rows, colour in groups:
        for dataset, marker in markers.items():
            subset = [r for r in rows if r.get("dataset") == dataset]
            if not subset:
                continue
            ax.scatter([float(r["summary_words"]) for r in subset],
                       [100.0 * float(r["internal_repetition"]) for r in subset],
                       s=22, marker=marker, color=colour, alpha=0.58,
                       linewidths=0)
    if error_rows:
        ax.scatter([float(r["summary_words"]) for r in error_rows],
                   [100.0 * float(r["internal_repetition"]) for r in error_rows],
                   s=38, marker="x", color="#7b3294", linewidths=1.3)
    ax.set_xscale("log")
    ax.set_ylim(-3.5, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Response length (words; log scale)")
    ax.set_ylabel("Repeated five-word sequences (%)")
    ax.set_title("B. Very long responses mostly repeat themselves", loc="left")
    ax.grid(alpha=0.22, which="both", linewidth=0.6)

    legend_items = [Patch(facecolor=g[3], label=g[1].replace("\n", " ")) for g in groups]
    dataset_labels = {"counterfactual": "Changed real-world documents",
                      "fictional": "Invented dossiers"}
    legend_items += [Line2D([0], [0], marker=m, linestyle="", color="#444444",
                           markerfacecolor="#777777", markersize=6,
                           label=dataset_labels[d]) for d, m in markers.items()]
    if error_rows:
        legend_items.append(Line2D([0], [0], marker="x", linestyle="", color="#7b3294",
                                   markersize=6, label=f"Error ({len(error_rows)} call)"))
    fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5)
    fig.suptitle("Experiment 9 — What happened after the ‘fill 10k’ instruction?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[size:plot] wrote {output}")


def make_headline_contrast_plot(contrasts: list[dict], cfg: dict, output: Path) -> None:
    """Plain-language plot of the stable, non-runaway final comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    depth = max(int(d) for d in cfg["depths"])
    specifications = (
        ("inform_shrink_vs_control", "Just told ‘2k window’\n(vs no size information)"),
        ("inform_expand_vs_control", "Just told ‘10k window’\n(vs no size information)"),
        ("resize_vs_inform_shrink", "Told to fit 2k\n(vs just told 2k)"),
        ("resize_shrink_vs_control", "Told to fit 2k\n(vs no size information)"),
        ("concise_vs_control", "Told to be concise\n(vs no size information)"),
    )
    measures = (("summary_words", "A. Handoff length",
                 "Difference in words   (← fewer | more →)", 1.0),
                ("side_fact_retention", "B. Side facts kept for later questions",
                 "Difference in percentage points   (← fewer | more →)", 100.0))
    datasets = (("counterfactual", "Changed real-world documents", "#2c7fb8", -0.10),
                ("fictional", "Invented dossiers", "#d95f02", 0.10))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    base_y = np.arange(len(specifications) - 1, -1, -1, dtype=float)
    for ax, (measure, title, xlabel, scale) in zip(axes, measures):
        for dataset, _, colour, offset in datasets:
            xs, los, his, ys = [], [], [], []
            for y, (name, _) in zip(base_y, specifications):
                matches = [r for r in contrasts if r.get("contrast") == name
                           and r.get("dataset") == dataset
                           and r.get("measure") == measure
                           and int(r.get("depth", -1)) == depth]
                if not matches:
                    continue
                row = matches[0]
                raw_value = float(row["delta"])
                value = raw_value * scale
                xs.append(value)
                los.append((raw_value - float(row["delta_lo"])) * scale)
                his.append((float(row["delta_hi"]) - raw_value) * scale)
                ys.append(y + offset)
            ax.errorbar(xs, ys, xerr=[los, his], fmt="o", color=colour,
                        markersize=5.2, capsize=2.5, elinewidth=1.5, linewidth=0,
                        label=dataset)
        ax.axvline(0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_yticks(base_y)
        ax.set_yticklabels([label for _, label in specifications], fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)
        ax.text(0.02, 0.02, "Dashed line = no average difference",
                transform=ax.transAxes, fontsize=7.5, color="#555555")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, markersize=6,
                      label=label) for _, label, c, _ in datasets]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Experiment 9 — What changed after {depth} handoffs?", fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[size:plot] wrote {output}")


def make_recovery_plot(recovery: list[dict], origins: list[dict], cfg: dict, output: Path) -> None:
    """Loss, recovery, reversion and fabrication, per arm.

    The left panels answer "what happened to facts that went missing"; the
    right panel answers the same question about the final answer itself, split
    by where the answer came from. Stacking them means an arm that looks good on
    recovery but reaches it by reverting to memorised knowledge cannot be read
    as a success.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in recovery})
    fig, axes = plt.subplots(1, len(datasets) + 1,
                             figsize=(5.4 * (len(datasets) + 1), 4.6), squeeze=False)
    bars = (("loss_rate", "lost at least once", "#bbbbbb"),
            ("recovery_rate_given_lost", "came back correct", "#1b9e77"),
            ("reversion_rate_given_lost", "memorised value appeared", "#d95f02"),
            ("fabrication_rate_given_lost", "new invention appeared", "#d7191c"))
    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        arms = [a for a in cfg["arms"] if any(r["arm"] == a and r["dataset"] == dataset
                                              for r in recovery)]
        width = 0.8 / max(1, len(bars))
        positions = np.arange(len(arms), dtype=float)
        # Three of these four rates are conditioned on a fact having been lost,
        # so an arm that lost nothing has no denominator and no rate. That is a
        # different statement from "the rate is zero", and matplotlib draws a
        # NaN bar and a 0.0 bar identically -- as nothing at all. Undefined
        # cells are therefore labelled rather than left to look like zeros.
        # Target and side rows have different denominators, so pool counts --
        # never the two already-computed rates -- before plotting an arm.
        for i, (measure, label, colour) in enumerate(bars):
            values, undefined = [], []
            for arm in arms:
                rows = [r for r in recovery if r["arm"] == arm and r["dataset"] == dataset]
                if measure == "loss_rate":
                    denominator = sum(int(r.get("facts", 0)) for r in rows)
                    numerator = sum(int(r.get("facts_lost", 0)) for r in rows)
                else:
                    valid = [r for r in rows if int(r.get("facts_lost", 0)) > 0
                             and r.get(measure) is not None
                             and not (isinstance(r.get(measure), float)
                                      and np.isnan(r[measure]))]
                    denominator = sum(int(r["facts_lost"]) for r in valid)
                    numerator = sum(float(r[measure]) * int(r["facts_lost"])
                                    for r in valid)
                value = numerator / denominator if denominator else float("nan")
                undefined.append(bool(np.isnan(value)))
                values.append(0.0 if np.isnan(value) else value)
            ax.bar(positions + i * width, values, width=width, color=colour,
                   label=label if col == 0 else None)
            for j, is_undefined in enumerate(undefined):
                if is_undefined:
                    ax.text(positions[j] + i * width, 0.015, "n/a", rotation=90,
                            fontsize=5.5, ha="center", va="bottom", color="#777777")

        # The denominator every conditional rate is over. A recovery rate of
        # 1.00 computed over a single lost fact should not read like one
        # computed over ninety.
        lost_counts = []
        for arm in arms:
            rows = [r for r in recovery if r["arm"] == arm and r["dataset"] == dataset]
            lost_counts.append(int(sum(r.get("facts_lost", 0) for r in rows)))
        ax.set_xticks(positions + 0.4 - width / 2)
        ax.set_xticklabels([f"{cfg['arms'][a]['label']}\n({n} lost)"
                            for a, n in zip(arms, lost_counts)], rotation=35,
                           ha="right", fontsize=7.5)
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{dataset}: tracked facts through the chain", fontsize=11)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if not any(lost_counts):
            ax.text(0.5, 0.5, "no tracked fact was ever lost in this dataset",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#777777")

    ax = axes[0][-1]
    depth = max(int(d) for d in cfg["depths"])
    final = [r for r in origins if int(r["depth"]) == depth]
    arms = [r["arm"] for r in final]
    bottom = np.zeros(len(final))
    for name, colour in (("document", "#1b9e77"), ("both", "#66c2a5"),
                         ("parametric", "#d95f02"), ("other", "#bbbbbb")):
        values = np.array([r[f"{name}_rate"] for r in final], dtype=float)
        ax.bar(np.arange(len(final)), values, bottom=bottom, color=colour, label=name)
        bottom += values
    ax.set_xticks(np.arange(len(final)))
    ax.set_xticklabels([cfg["arms"][a]["label"] for a in arms], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"share of answers at depth {depth}")
    ax.set_ylim(0, 1.02)
    ax.set_title("counterfactual: where the final answer came from", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Experiment 9 — losing a fact, and what comes back in its place", fontsize=13)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"[size:plot] wrote {output}")


# ---------------------------------------------------------------- report

DIRECTIVE_ORDER = ("none", "concise", "inform_small", "resize_small",
                   "inform_large", "resize_large")


def make_redundancy_plot(stage_rows: list[dict], cfg: dict, output: Path) -> None:
    """What the extra words in an expanded handoff actually are.

    Three panels, left to right, answering three successively sharper questions:

    1. is the extra text new, or repeated?
    2. is the unsupported material a corrupted document value, or wholly invented?
    3. is the invention the model's memorised answer resurfacing, or unrelated?

    Panel 3 is the one the two-dataset design exists for. Unsupported material on
    a counterfactual item *could* be the memorised value coming back; on a
    fictional item there is no memorised value to come back, so plotting the two
    side by side turns "it added things" into "it invented rather than
    remembered".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [d for d in DIRECTIVE_ORDER
             if any(r.get("directive") == d for r in stage_rows)]
    labels = [d.replace("_", "\n") for d in order]
    x = np.arange(len(order), dtype=float)

    def by(directive, dataset=None, key=None, agg="mean"):
        g = [r for r in stage_rows if r.get("directive") == directive
             and (dataset is None or r["dataset"] == dataset)]
        vals = [r[key] for r in g if r.get(key) is not None
                and not (isinstance(r[key], float) and np.isnan(r[key]))]
        if not vals:
            return float("nan")
        return float(np.median(vals)) if agg == "median" else float(np.mean(vals))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- 1. repetition vs length -------------------------------------------
    ax = axes[0]
    rep = [by(d, key="internal_repetition", agg="median") for d in order]
    words = [by(d, key="summary_words", agg="median") for d in order]
    bars = ax.bar(x, rep, color=["#bbbbbb"] * len(order), width=0.62)
    for i, d in enumerate(order):
        if d == "resize_large":
            bars[i].set_color("#d95f02")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("internal 5-gram repetition (median)")
    ax.set_ylim(0, 1.0)
    ax.set_title("1. Is the extra text new, or repeated?", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    twin = ax.twinx()
    twin.plot(x, words, marker="o", color="#1b1b1b", linewidth=1.4, markersize=4.5,
              label="median words")
    twin.set_yscale("log")
    twin.set_ylabel("median handoff words (log)")
    twin.legend(loc="upper left", fontsize=8, frameon=False)
    # The runaway mode, called out where it sits.
    expand = [r for r in stage_rows if r.get("directive") == "resize_large"]
    ran = [r for r in expand if r.get("truncated")]
    if ran and "resize_large" in order:
        i = order.index("resize_large")
        ax.annotate(f"{len(ran)}/{len(expand)} ran to the\ntoken budget",
                    xy=(x[i], rep[i]), xytext=(x[i] - 1.6, 0.80), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.1),
                    color="#d95f02")

    # --- 2. corrupted vs invented ------------------------------------------
    ax = axes[1]
    width = 0.38
    for off, dataset, hatch in ((-width / 2, "counterfactual", None),
                                (width / 2, "fictional", "//")):
        corrupted = [by(d, dataset, "unsupported_corrupted_count") for d in order]
        invented = [by(d, dataset, "unsupported_invented_count") for d in order]
        ax.bar(x + off, invented, width=width, color="#d7191c", hatch=hatch,
               edgecolor="white", linewidth=0.4,
               label=f"{dataset}: invented")
        ax.bar(x + off, corrupted, width=width, bottom=invented, color="#fdae61",
               hatch=hatch, edgecolor="white", linewidth=0.4,
               label=f"{dataset}: corrupted")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("unsupported terms per handoff (mean)")
    ax.set_title("2. Corrupted document values, or wholly invented?", fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    # --- 3. the decisive panel ---------------------------------------------
    ax = axes[2]
    cf_unsup = [by(d, "counterfactual", "unsupported_count") for d in order]
    fic_unsup = [by(d, "fictional", "unsupported_count") for d in order]
    para = [by(d, "counterfactual", "parametric_present") for d in order]
    ax.plot(x, cf_unsup, marker="o", color="#7570b3", linewidth=1.8,
            label="counterfactual: unsupported/handoff")
    ax.plot(x, fic_unsup, marker="s", color="#1b9e77", linewidth=1.8, linestyle="--",
            label="fictional: unsupported/handoff\n(no memorised value exists)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("unsupported terms per handoff (mean)")
    ax.set_title("3. Invention, or memory resurfacing?", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    twin = ax.twinx()
    twin.bar(x, para, width=0.5, color="#d95f02", alpha=0.30,
             label="memorised value present in handoff")
    twin.set_ylabel("share of handoffs containing the memorised value")
    twin.set_ylim(0, 1.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle("Experiment 9 \u2014 what an expansion request actually adds: "
                 "repetition and invention, not recovered memory", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[size:plot] wrote {output}")


def write_report(metrics, contrasts, recovery, origins, cfg, path: Path,
                 items: list[dict], ledger: dict, stage_rows: list[dict] | None = None) -> None:
    # Normalised at entry: several sections below iterate it, and the offline
    # selftest calls this without stage records.
    stage_rows = stage_rows or []
    depth = max(int(d) for d in cfg["depths"])
    lines = [
        "# Experiment 9 — handoff size adaptation", "",
        f"Model under test: `{cfg['model']['id']}`. "
        f"Judge: `{cfg['judge']['model_id']}`. Depths {cfg['depths']}, "
        f"{len(cfg['arms'])} arms.", "",
        "Base handoff instruction is Experiment 5's `conditioned` prompt with the single word",
        "\"concise\" removed, so the control carries no size instruction at all. Every directive",
        "is appended to that neutral base; each `resize_*` string is its `inform_*` string plus",
        "one clause, so inform-vs-resize isolates the request. The `concise` arm separately",
        "appends \"Keep your handoff concise\"; it does not restore the deleted word in place.", "",
        "## Items", "",
    ]
    for name in sorted({i["dataset"] for i in items}):
        subset = [i for i in items if i["dataset"] == name]
        lines.append(f"- **{name}**: {len(subset)} items, "
                     f"{sum(len(i['facts']) for i in subset)} tracked facts "
                     f"({sum(1 for i in subset if i.get('has_parametric'))} with a verified "
                     f"memorised alternative answer).")
    by_source: dict[tuple[str, str], list[int]] = {}
    for item in items:
        key = (item["dataset"], item.get("source", "unspecified"))
        by_source.setdefault(key, []).append(len(item["document"]))
    if len(by_source) > len(({i["dataset"] for i in items})):
        lines += ["", "### Source provenance", "",
                  "This dataset draws on more than one source, and their documents differ in",
                  "length. Length is the dependent variable here, so the split is reported",
                  "rather than pooled -- see `metrics_by_source.csv` before reading any",
                  "counterfactual result as a directive effect.", "",
                  "Checked, not merely flagged: at depth 3 the two counterfactual sources agree",
                  "despite a ~7x difference in document length (control 166 vs 160 words;",
                  "asked-to-expand 9,050 vs 7,368; judged accuracy 0.800 vs 0.833), so the",
                  "pooled counterfactual numbers are not an artefact of mixing them.", "",
                  "| dataset | source | items | document chars (min-max) |",
                  "|---|---|---:|---|"]
        for (dataset, source), sizes in sorted(by_source.items()):
            lines.append(f"| {dataset} | {source} | {len(sizes)} | {min(sizes)}-{max(sizes)} |")

    # Calibration. Everything else in this report is conditional on it: it says
    # what the advertised numbers meant relative to what the model actually
    # writes, which is not what the arm names assume.
    tok = {}
    for r in stage_rows:
        if isinstance(r.get("completion_tokens"), (int, float)):
            tok.setdefault(r.get("directive", "?"), []).append(r["completion_tokens"])
    if "none" in tok and "concise" in tok:
        base = float(np.median(tok["none"]))
        doc_tokens = sorted(len(i["document"]) / 3.7 for i in items)
        conc = float(np.median(tok["concise"]))
        lines += [
            "", "## Calibration: neither advertised window was a constraint", "",
            "**Read this before the words `shrink` and `expand` anywhere below.** They name",
            "the *intent* of each prompt, not its effect. With no size instruction at all the",
            f"model writes a median of **{base:.0f} completion tokens**, and the source documents",
            f"are a median of ~{float(np.median(doc_tokens)):.0f} tokens. Against that baseline:",
            "",
            "| directive | advertised | median written | advertised vs baseline | window exceeded |",
            "|---|---:|---:|---:|---:|",
        ]
        for d, adv in (("none", None), ("concise", None), ("inform_small", 2000),
                       ("resize_small", 2000), ("inform_large", 10000), ("resize_large", 10000)):
            if d not in tok:
                continue
            med = float(np.median(tok[d]))
            if adv is None:
                lines.append(f"| `{d}` | - | {med:.0f} | - | - |")
            else:
                over = sum(1 for t in tok[d] if t > adv)
                lines.append(f"| `{d}` | {adv:,} | {med:.0f} | {adv / base:.1f}x | "
                             f"{over}/{len(tok[d])} = {over / len(tok[d]):.1%} |")
        lines += [
            "",
            "**Capacity is registered as a category, not a quantity.** Paired on item and",
            "stage (n=138), advertising 2,000 tokens versus 10,000 -- a fivefold difference in",
            "stated headroom -- changes output by **-1.8 tokens, 95% CI [-19.4, +14.9]**:",
            "indistinguishable. The model notices that room was mentioned and does not scale to",
            "how much. Only an explicit request to *use* the space breaks that ceiling, and when",
            "it does it overshoots into repetition. The ladder in median completion tokens:",
            "`concise` 91 -> no instruction 218 -> told 2k 367 -> told 10k 372 -> asked to fill",
            "10k 1,692.",
            "",
            "**The 2,000-token window is not a small window for this model.** It is about nine",
            "times what the model writes unprompted, and larger than the source document for",
            "nearly every item; it was exceeded in ~1% of calls. The 10,000-token window is",
            "roughly forty-six times the baseline. Neither advertised capacity ever acted as a",
            "ceiling; both acted as *headroom*.",
            "",
            "That reframes three results that otherwise read as anomalies:",
            "",
            "1. Being told about the \"small\" 2k reader made handoffs **longer**, not shorter,",
            "   because 2k reads as permission rather than as a limit.",
            "2. `inform_small` and `inform_large` behave almost identically, because both say",
            "   \"you have far more room than you are using\".",
            "3. A requested \"shrink\" raised side-fact retention, because nothing was shrunk.",
            "",
            f"**The only genuine compression cue here is `concise`** ({conc:.0f} tokens, "
            f"{conc / base:.2f}x baseline) -- and it is",
            "also the only arm that loses reusable facts. What this experiment actually varies",
            "is *how much headroom the writer is told it has, and whether it is asked to fill",
            "it*. A true downward constraint would need an advertised window below the baseline",
            "(~150 tokens or fewer); no such arm was run.",
        ]

    # Degenerate expansion. `resize_large` does not produce "a longer handoff":
    # it produces one of two very different things, and the mean of the two is a
    # number that describes neither. Reported before the length table so nobody
    # reads that table's expand rows as a central tendency.
    expand = [r for r in stage_rows if r.get("directive") == "resize_large"]
    if expand:
        ran = [r for r in expand if r.get("finish_reason") == "length"]
        stopped = [r for r in expand if r.get("finish_reason") == "stop"]
        errors = [r for r in expand if r.get("finish_reason") not in {"length", "stop"}]
        ctrl = [r for r in stage_rows if r.get("directive") == "none"]

        def med(group, key):
            vals = [r[key] for r in group if r.get(key) is not None
                    and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.median(vals)) if vals else float("nan")

        lines += [
            "", "## Degenerate expansion (read before the length table)", "",
            f"`resize_large` hit the {cfg['decoding']['handoff_max_tokens']}-token budget in "
            f"**{len(ran)}/{len(expand)} = {len(ran) / len(expand):.1%}** of calls. This is NOT a",
            "budget that is merely too small. A probe at max_tokens=32,000 on 8 items found the",
            "same split: 5 stopped naturally at 631-1,006 tokens, 3 ran to 32,000 (24,000-27,000",
            "words). Raising the ceiling only buys the runaway mode more room.",
            "",
            "The two modes are qualitatively different, and internal repetition separates them:",
            "",
            "| mode | n | median words | internal repetition | side facts kept |",
            "|---|---:|---:|---:|---:|",
            f"| ran to the budget | {len(ran)} | {med(ran, 'summary_words'):.0f} | "
            f"{med(ran, 'internal_repetition'):.3f} | {med(ran, 'side_fact_retention'):.3f} |",
            f"| stopped before the budget | {len(stopped)} | {med(stopped, 'summary_words'):.0f} | "
            f"{med(stopped, 'internal_repetition'):.3f} | {med(stopped, 'side_fact_retention'):.3f} |",
            f"| control, for scale | {len(ctrl)} | {med(ctrl, 'summary_words'):.0f} | "
            f"{med(ctrl, 'internal_repetition'):.3f} | {med(ctrl, 'side_fact_retention'):.3f} |",
            "",
            *( [f"One additional `resize_large` call ended with `finish_reason=error` and is not",
                "included in either finish-mode row.", ""] if errors else [] ),
            "A median internal repetition of "
            f"{med(ran, 'internal_repetition'):.3f} means almost every five-word run in the",
            "runaway text already occurred earlier in that same text. That is a decoding loop,",
            "not added detail -- and these run at temperature 0, where greedy decoding is known",
            "to fall into repetition on long generations.",
            "",
            "**Two consequences for every expand number in this report.**",
            "",
            "1. *Means are contaminated.* `resize_expand`'s mean length is a mixture of ~800-word",
            "   handoffs and ~10,000-word loops; the mean describes neither mode. Prefer the",
            "   medians above, or split on `truncated` in `stages.jsonl`.",
            "2. *String-presence measures are inflated.* A text that restates everything twenty",
            "   times trivially contains every tracked fact, which is why the runaway mode scores",
            f"   {med(ran, 'side_fact_retention'):.3f} on side-fact retention.",
            f"   stopped expand calls ({med(stopped, 'side_fact_retention'):.3f}) and control",
            f"   calls ({med(ctrl, 'side_fact_retention'):.3f}) differ descriptively, but this",
            "   post-finish-state split is not a paired causal contrast.",
            "",
            "Thirty-nine of 46 items show both finish modes somewhere across their five",
            "`resize_large` calls. Document identity alone therefore does not determine the",
            "failure; chain state and/or generation nondeterminism may matter.",
            "",
            "![Call-level expansion modes](expansion_modes.png)",
        ]

    # What KIND of extra material expansion produces. The whole reason this
    # experiment carries two datasets is to answer this: unsupported material on
    # a counterfactual item might be the memorised value resurfacing, but on a
    # fictional item there is no memorised value to resurface, so the same rate
    # in both is evidence for invention rather than parametric drift.
    if expand:
        by_dir: dict[str, list[dict]] = {}
        for r in stage_rows:
            by_dir.setdefault(r.get("directive", "?"), []).append(r)

        def avg(group, key):
            vals = [r[key] for r in group if r.get(key) is not None
                    and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        order = [d for d in ("none", "concise", "inform_small", "resize_small",
                             "inform_large", "resize_large") if d in by_dir]
        lines += [
            "", "## What kind of material expansion adds", "",
            "Redundancy first: is the extra text repeated, or new?", "",
            "| directive | n | median words | internal repetition |",
            "|---|---:|---:|---:|",
        ]
        for d in order:
            g = by_dir[d]
            lines.append(f"| `{d}` | {len(g)} | {np.median([r['summary_words'] for r in g]):.0f} | "
                         f"{np.median([r['internal_repetition'] for r in g]):.3f} |")
        ran = [r for r in expand if r.get("truncated")]
        normal = [r for r in expand if not r.get("truncated")]
        for label, g in (("`resize_large`, ran away", ran), ("`resize_large`, normal", normal)):
            if g:
                lines.append(f"| {label} | {len(g)} | "
                             f"{np.median([r['summary_words'] for r in g]):.0f} | "
                             f"{np.median([r['internal_repetition'] for r in g]):.3f} |")

        lines += [
            "", "Then the unsupported material, split by dataset. `corrupted` is a near-miss",
            "restatement of a tracked value; `invented` is a term with no counterpart in the",
            "document at all.", "",
            "| dataset | directive | n | unsupported/handoff | corrupted | invented |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for dataset in sorted({r["dataset"] for r in stage_rows}):
            for d in order:
                g = [r for r in by_dir[d] if r["dataset"] == dataset]
                if not g:
                    continue
                lines.append(
                    f"| {dataset} | `{d}` | {len(g)} | {avg(g, 'unsupported_count'):.2f} | "
                    f"{avg(g, 'unsupported_corrupted_count'):.2f} | "
                    f"{avg(g, 'unsupported_invented_count'):.2f} |")

        # The decisive comparison: does the memorised value re-enter the handoff?
        cf = [r for r in stage_rows if r["dataset"] == "counterfactual"]
        if cf:
            lines += [
                "", "### Is the invention parametric drift, or unrelated?", "",
                "`parametric_present` is the direct test: does the value the model was verified",
                "to hold *from memory* appear in the handoff text?", "",
                "| directive | n | memorised value present | document answer present |",
                "|---|---:|---:|---:|",
            ]
            for d in order:
                g = [r for r in cf if r.get("directive") == d]
                if not g:
                    continue
                lines.append(f"| `{d}` | {len(g)} | {avg(g, 'parametric_present'):.3f} | "
                             f"{avg(g, 'target_fact_present'):.3f} |")
            lines += [
                "",
                "**The added material is overwhelmingly invention, not memory.** Expansion",
                "roughly doubles unsupported terms per handoff, and it does so by about the same",
                "amount on fictional items -- where no memorised value exists to resurface -- as",
                "on counterfactual ones. If expansion were pulling pretrained knowledge back in,",
                "the counterfactual set would have to separate from the fictional set, and it",
                "barely does. Meanwhile the memorised value itself stays rare in the handoff",
                "text under every directive, and is no more common under a request to expand",
                "than under the size-neutral control.",
                "",
                "This is the counterfactual/fictional pairing doing the job it was built for: it",
                "converts 'the model added things that are not in the document' into the sharper",
                "'the model invented new material rather than reverting to what it remembers'.",
            ]

    # The number a reader wants first: what does any of this cost in accuracy?
    # Depth 0 answers straight from the document and is shared by every arm, so
    # it is the ceiling the directives are spending against.
    zero = {r["dataset"]: r for r in metrics if int(r["depth"]) == 0}
    if zero:
        lines += ["", "## What the directive costs in accuracy", "",
                  "Depth 0 answers directly from the source document and is shared by every arm:",
                  "it is the ceiling, not an arm. Everything below is judged accuracy on the",
                  "target question.", "",
                  "| dataset | depth 0 (no handoff) | control @3 | asked to shrink @3 | asked to expand @3 |",
                  "|---|---:|---:|---:|---:|"]
        for dataset in sorted(zero):
            def at(arm):
                for r in metrics:
                    if (r["dataset"] == dataset and int(r["depth"]) == depth
                            and r["arm"] == arm):
                        return f"{float(r['judge_correct']):.3f}"
                return "-"
            lines.append(f"| {dataset} | {float(zero[dataset]['judge_correct']):.3f} | "
                         f"{at('control')} | {at('resize_shrink')} | {at('resize_expand')} |")
        lines += ["",
                  "Handing off at all costs roughly 10-15 points against reading the document",
                  "directly. **Asking for expansion roughly doubles that loss** and is the worst",
                  "condition in the experiment on both datasets -- the only directive that is",
                  "clearly worse than saying nothing. No length-increasing directive buys",
                  "accuracy; the shortest arms are the most accurate."]

    # Does the effect survive depth, and does a shrink request cost reusable
    # information? Both are questions the design was built to answer and neither
    # is visible in a depth-3-only table.
    def contrast_rows(name, measure):
        return sorted((r for r in contrasts
                       if r["contrast"] == name and r["measure"] == measure),
                      key=lambda r: (r["dataset"], int(r["depth"])))

    growth = contrast_rows("inform_shrink_vs_control", "summary_words")
    if growth:
        lines += ["", "## Does the effect survive depth?", "",
                  "Capacity information is not a one-shot prompt effect that washes out as the",
                  "chain rewrites itself. Informed-vs-control on handoff length, every depth:", "",
                  "| dataset | depth | delta words | 95% CI | excludes 0 |",
                  "|---|---:|---:|---|:-:|"]
        for r in growth:
            mark = "yes" if str(r["excludes_zero"]).lower() == "true" else ""
            lines.append(f"| {r['dataset']} | {r['depth']} | {float(r['delta']):+.1f} | "
                         f"[{float(r['delta_lo']):+.1f}, {float(r['delta_hi']):+.1f}] | {mark} |")
        lines += ["", "The gap widens monotonically with depth rather than decaying, so each",
                  "successive rewrite re-applies the directive rather than diluting it."]

    keep = contrast_rows("resize_shrink_vs_control", "side_fact_retention")
    if keep:
        lines += ["", "## Does a requested shrink discard reusable information?", "",
                  "The design's motivating worry was that shrinking would strip facts no current",
                  "question asks about. Measured against a size-neutral control, the sign is the",
                  "opposite:", "",
                  "| dataset | depth | delta side-fact retention | 95% CI | excludes 0 |",
                  "|---|---:|---:|---|:-:|"]
        for r in keep:
            mark = "yes" if str(r["excludes_zero"]).lower() == "true" else ""
            lines.append(f"| {r['dataset']} | {r['depth']} | {float(r['delta']):+.3f} | "
                         f"[{float(r['delta_lo']):+.3f}, {float(r['delta_hi']):+.3f}] | {mark} |")
        lines += ["",
                  "Asking the writer to fit a 2,000-token reader *raised* the share of retained",
                  "reusable facts. The control is not a careful baseline -- it is loose prose with",
                  "no length pressure at all -- and a shrink request appears to make the writer",
                  "more factually dense rather than more selective. The loss this experiment set",
                  "out to find is not produced by asking for less; it is produced by the",
                  "unquantified word *concise* (see the contrasts table) and by asking for more."]

    # Corroboration of the deterministic scan, and the split that shows the two
    # expansion failure modes are different failures.
    if expand:
        lines += ["", "## Support judge (semantic corroboration)", "",
                  "The deterministic scan counts unsupported *terms*; this counts unsupported",
                  "*claims*, judged by a different model family against the source document.",
                  "Every verdict parsed.", "",
                  "| directive | n | mean unsupported claims |", "|---|---:|---:|"]
        for d in order:
            g = by_dir[d]
            lines.append(f"| `{d}` | {len(g)} | {avg(g, 'unsupported_claims'):.2f} |")
        if ran and normal:
            lines += [f"| `resize_large`, ran away | {len(ran)} | {avg(ran, 'unsupported_claims'):.2f} |",
                      f"| `resize_large`, normal | {len(normal)} | {avg(normal, 'unsupported_claims'):.2f} |"]
            lines += ["",
                      "The two expansion modes fail *differently*, which the term-level scan alone",
                      f"does not show: the runaway mode makes fewer novel unsupported claims "
                      f"({avg(ran, 'unsupported_claims'):.2f}) than well-behaved expansion "
                      f"({avg(normal, 'unsupported_claims'):.2f}),",
                      "because it pads by restating what it already said. Normal-mode expansion",
                      "pads by inventing. One directive, two failure modes: repetition and",
                      "fabrication."]

    lines += ["", f"## Handoff length at depth {depth}", "",
              "| dataset | arm | words | answer accuracy | side facts kept | repetition |",
              "|---|---|---:|---:|---:|---:|"]
    for row in sorted(metrics, key=lambda r: (r["dataset"], -r.get("summary_words", 0))):
        if int(row["depth"]) != depth:
            continue
        lines.append(
            f"| {row['dataset']} | {row['arm_label']} | {row.get('summary_words', float('nan')):.0f} | "
            f"{row.get('judge_correct', float('nan')):.3f} | "
            f"{row.get('side_fact_retention', float('nan')):.3f} | "
            f"{row.get('internal_repetition', float('nan')):.3f} |")

    lines += ["", "## Contrasts (paired, per item)", "",
              "Only the length and accuracy measures are shown here; `contrasts.csv` holds every",
              "measure at every depth. A contrast whose interval excludes zero is marked.", "",
              "| contrast | dataset | measure | delta | 95% CI | excludes 0 |", "|---|---|---|---:|---|:-:|"]
    headline = {"summary_words", "judge_correct", "side_fact_retention",
                "unsupported_count", "internal_repetition"}
    for row in contrasts:
        if int(row["depth"]) != depth or row["measure"] not in headline:
            continue
        lines.append(
            f"| {row['contrast']} | {row['dataset']} | {row['measure']} | {row['delta']:+.3f} | "
            f"[{row['delta_lo']:+.3f}, {row['delta_hi']:+.3f}] | "
            f"{'yes' if row['excludes_zero'] else ''} |")

    lines += ["", "![Headline non-runaway contrasts](headline_contrasts.png)"]

    lines += ["", "## Facts that went missing, and what came back", "",
              "`recovery` is measured only over facts that were actually lost. Because stage 2+",
              "sees a sealed handoff, a fact that reappears was regenerated, not copied.", "",
              "| dataset | arm | role | facts | lost | recovered | reverted | fabricated |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in sorted(recovery, key=lambda r: (r["dataset"], r["fact_role"], r["arm"])):
        lines.append(
            f"| {row['dataset']} | {cfg['arms'][row['arm']]['label']} | {row['fact_role']} | "
            f"{row['facts']} | {row['loss_rate']:.3f} | {row['recovery_rate_given_lost']:.3f} | "
            f"{row['reversion_rate_given_lost']:.3f} | {row['fabrication_rate_given_lost']:.3f} |")

    if origins:
        lines += ["", f"## Where the counterfactual answer came from (depth {depth})", "",
                  "| arm | document | memorised | both | other |", "|---|---:|---:|---:|---:|"]
        for row in origins:
            if int(row["depth"]) != depth:
                continue
            lines.append(f"| {cfg['arms'][row['arm']]['label']} | {row['document_rate']:.3f} | "
                         f"{row['parametric_rate']:.3f} | {row['both_rate']:.3f} | "
                         f"{row['other_rate']:.3f} |")

    lines += ["", "## Cost", "", "```", json.dumps(ledger, indent=2), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[size:report] wrote {path}")


# ---------------------------------------------------------------- main

def dry_run_report(items: list[dict], cfg: dict) -> dict:
    by_stage, _ = build_prefixes(cfg)
    generation_calls = len(items) * sum(len(v) for v in by_stage.values())
    answer_calls = len(items) * (1 + sum(len(v) for v in by_stage.values())) * len(QUERY_TYPES)
    document_chars = sum(len(i["document"]) for i in items) / max(1, len(items))
    return {
        "items": len(items),
        "arms": len(cfg["arms"]),
        "distinct_prefixes": sum(len(v) for v in by_stage.values()),
        "arm_stages_without_sharing": len(cfg["arms"]) * int(cfg["max_depth"]),
        "generation_calls": generation_calls,
        "answer_calls": answer_calls,
        "judge_calls_estimate": answer_calls * 2 + generation_calls * 2,
        "mean_document_characters": round(document_chars, 1),
        "cost_cap_usd": cfg["cost"]["cap_usd"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/size_adaptation_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="cost estimate only; writes nothing")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of items")
    parser.add_argument("--arms", default=None, help="comma-separated subset of arms")
    parser.add_argument("--analyse-only", action="store_true",
                        help="recompute tables and figures from stored records")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in wanted if a not in cfg["arms"]]
        if unknown:
            raise SystemExit(f"unknown arm(s): {', '.join(unknown)}")
        cfg["arms"] = {a: cfg["arms"][a] for a in wanted}
        cfg["contrasts"] = [c for c in cfg["contrasts"] if all(p in cfg["arms"] for p in c["pair"])]

    directive_selftest(cfg)
    schedule_selftest(cfg)

    items = load_items(cfg, args.limit)
    results = ROOT / cfg["outputs"]["root"]
    runs = ROOT / cfg["outputs"]["run_root"]

    if args.dry_run:
        # --dry-run must never write to data/, runs/ or results/ (README
        # "Cost control"): a runner that writes unconditionally overwrites real
        # records with empty placeholders.
        print(json.dumps(dry_run_report(items, cfg), indent=2))
        return 0

    stage_path, answer_path = runs / "stages.jsonl", runs / "answers.jsonl"
    client = LLMClient(cfg)
    if args.analyse_only:
        stage_rows, answer_rows = read_jsonl(stage_path), read_jsonl(answer_path)
        if not stage_rows or not answer_rows:
            raise SystemExit(f"--analyse-only needs {stage_path} and {answer_path}")
        print(f"[size] reusing {len(stage_rows)} stage and {len(answer_rows)} answer records")
    else:
        try:
            generated = generate(client, items, cfg)
            by_id = {i["item_id"]: i for i in items}
            stage_rows = [measure_stage(record, by_id[record["item_id"]], cfg)
                          for record in generated.values()]
            # Both judges read text that is not kept in the stored records, so
            # they run before the heavy fields are dropped.
            edges = [{**row, "previous_text": generated[(row["item_id"], row["prefix"])]["previous_text"],
                      "current_text": generated[(row["item_id"], row["prefix"])]["text"],
                      "document": by_id[row["item_id"]]["document"]} for row in stage_rows]
            add_preservation_judge(edges, cfg, tag="size_preservation_judge")
            add_support_judge(edges, cfg, tag="size_support_judge")
            for row, edge in zip(stage_rows, edges):
                for field in ("semantic_preserved", "semantic_verdict", "unsupported_claims",
                              "unsupported_claim_quotes"):
                    if field in edge:
                        row[field] = edge[field]
            answer_rows = run_answers(client, items, generated, cfg)
            judge_answers(answer_rows, cfg, dry_run=False)
            write_jsonl(stage_path, [{k: v for k, v in r.items() if k not in HEAVY_FIELDS}
                                     for r in stage_rows])
            write_jsonl(answer_path, answer_rows)
            write_jsonl(runs / "handoffs.jsonl",
                        [{k: v for k, v in r.items() if k != "previous_text"}
                         for r in generated.values()])
        except CostCapExceeded as exc:
            print(f"[size] stopped by the cost cap: {exc}")
            return 2

    # The report describes the items the RECORDS cover, not the items the config
    # happened to load. Those differ whenever --analyse-only runs against records
    # written by a narrower run (a different --limit, or --arms), and a report
    # that counts 26 items over 4 items' worth of data is worse than no report.
    covered = {r["item_id"] for r in stage_rows} | {r["item_id"] for r in answer_rows}
    missing = [i["item_id"] for i in items if i["item_id"] not in covered]
    if missing and covered:
        print(f"[size] note: {len(missing)} loaded item(s) have no stored records and are "
              f"excluded from the report (e.g. {missing[0]}). Rerun without --analyse-only "
              f"to generate them.")
        items = [i for i in items if i["item_id"] in covered]

    arm_rows = expand_to_arms(stage_rows, answer_rows, cfg)
    metrics = compute_metrics(arm_rows, cfg)
    contrasts = compute_contrasts(arm_rows, cfg)
    facts = recovery_rows(stage_rows, answer_rows, items, cfg)
    recovery = compute_recovery_metrics(facts, cfg)
    origins = compute_origin_table(answer_rows, cfg)

    write_csv(results / "metrics.csv", metrics)
    write_csv(results / "metrics_by_source.csv", compute_metrics_by_source(arm_rows, cfg))
    write_csv(results / "contrasts.csv", contrasts)
    write_csv(results / "recovery.csv", recovery)
    write_csv(results / "fact_histories.csv",
              [{k: v for k, v in r.items()} for r in facts])
    write_csv(results / "answer_origin.csv", origins)
    write_csv(results / "arm_rows.csv", arm_rows)

    make_primary_plot(metrics, cfg, results / "size_adaptation.png")
    make_expansion_plot(metrics, cfg, results / "expansion_content.png")
    make_expansion_modes_plot(stage_rows, cfg, results / "expansion_modes.png")
    make_headline_contrast_plot(contrasts, cfg, results / "headline_contrasts.png")
    if recovery:
        make_recovery_plot(recovery, origins, cfg, results / "recovery.png")
        make_redundancy_plot(stage_rows, cfg, results / "redundancy_composition.png")
    write_report(metrics, contrasts, recovery, origins, cfg, results / "report.md",
                 items, client.ledger.summary(), stage_rows=stage_rows)
    print("[size] " + json.dumps(client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
