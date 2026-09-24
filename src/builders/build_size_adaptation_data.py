"""Build the two Experiment 9 datasets, each with its knowledge check attached.

Experiment 9 asks where material in an expanded handoff came from. That
question is only answerable if the model's prior knowledge of every item is
established *before* the chain runs, so both datasets are built here together
with the closed-book evidence that licenses their interpretation.

``counterfactual`` -- Wikipedia pages already rewritten by
``build_wikipedia_counterfactual.py`` so the document's answer differs from the
memorised one. Two closed-book conditions must both hold:

    knows the ORIGINAL answer      -> a later "Rome" is attributable to memory
    does NOT know the REPLACEMENT  -> a "Lisbon" must have come from the document

The existing counterfactual build only checked the second. This adds the first,
which is the load-bearing one for a reconstruction claim: without it, a handoff
that fails to say "Lisbon" and says nothing else is simply a handoff that lost
a fact, and reads identically to one that reverted.

Both verdicts come from ONE set of closed-book samples judged twice, against
each gold set. Sampling once and comparing twice is not a shortcut: it is the
only way the two conditions describe the same model behaviour rather than two
independent draws that might disagree by luck.

``fictional`` -- pages about entities that do not exist, so no closed-book
knowledge is possible in either direction. Unsupported material appearing here
cannot be reconstruction, which makes it the clean fabrication control for the
counterfactual set's ambiguous cases.

The page writer is a strong model and is never the system under test, following
``build_wikipedia_dataset.py`` and ``build_wikipedia_counterfactual.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

import judge as judge_mod  # noqa: E402
import size_metrics as sx  # noqa: E402
from data import CLOSED_BOOK_SYSTEM  # noqa: E402
from llm import CostCapExceeded, HardAPIFailure, LLMClient, load_config  # noqa: E402
from score import extract_short_answer, score_against_golds  # noqa: E402

WRITER_SYSTEM = (
    "You write reference articles about invented places, people and institutions for use as "
    "controlled research stimuli. Everything you write is fiction and must not correspond to "
    "any real entity, work, or event."
)

EXTRACT_SYSTEM = (
    "You extract checkable factual details from a document for use as evaluation probes. "
    "You only ever report details the document actually states."
)


# ---------------------------------------------------------------- io helpers

def read_jsonl(path: Path) -> list[dict]:
    return [] if not path.exists() else [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def json_object(text: str) -> dict:
    """Parse one JSON object out of a model reply, tolerating a code fence."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON was not an object")
    return value


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def writer_concurrency(cfg: dict) -> int:
    """Fan-out for the strong writer, which is lower than the experiment's.

    Long high-effort reasoning calls hit OpenRouter's in-flight credit ceiling
    (HTTP 402 ``in_flight_budget_exhausted``) well before they hit a rate limit.
    """
    spec = cfg.get("build", {}).get("writer", {})
    return max(1, int(spec.get("concurrency", cfg["runtime"]["concurrency"])))


def writer_config(base_cfg: dict, spec: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["id"] = spec["model_id"]
    cfg["model"]["reasoning"] = spec.get("reasoning")
    cfg["model"].pop("system_suffix", None)
    cfg["cost"] = {"cap_usd": float(spec.get("cap_usd", 6.0)), "warn_at_fraction": 0.8}
    return cfg


# ------------------------------------------------------ closed-book knowledge

def closed_book_samples(client: LLMClient, question: str, spec: dict, tag: str) -> list[dict]:
    """Sample the model's memory for one question.

    Identical message shape, temperature and seed ladder to ``data.py``'s C1
    filter, so an item already probed by an earlier build hits the on-disk
    cache instead of being re-paid for.
    """
    messages = [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]
    out = []
    for k in range(int(spec["samples"])):
        res = client.chat(messages, temperature=float(spec["temperature"]),
                          max_tokens=int(spec["max_tokens"]), seed=1000 + k, tag=tag)
        out.append({"sample": k, "pred": extract_short_answer(res.text), "cached": res.cached})
    return out


def knowledge_report(client: LLMClient, items: list[dict], cfg: dict, spec: dict,
                     tag: str) -> dict[str, dict]:
    """Closed-book verdicts for every item against every gold set it declares.

    ``items`` supply ``key``, ``question`` and ``gold_sets`` -- a mapping of
    name -> gold strings. One sample set per question is drawn and then judged
    once per gold set, so "knows the original" and "does not know the
    replacement" are two readings of the same behaviour.
    """
    with ThreadPoolExecutor(max_workers=int(cfg["runtime"]["concurrency"])) as ex:
        futures = {it["key"]: ex.submit(closed_book_samples, client, it["question"], spec, tag)
                   for it in items}
        samples = {key: fut.result() for key, fut in futures.items()}

    rows = []
    for item in items:
        for name, golds in item["gold_sets"].items():
            for attempt in samples[item["key"]]:
                em, f1 = score_against_golds(attempt["pred"], golds)
                rows.append({"key": item["key"], "gold_set": name, "sample": attempt["sample"],
                             "question": item["question"], "pred": attempt["pred"],
                             "golds": golds, "em": em, "f1": f1})
    judge_mod.add_judge(rows, cfg, tag=f"{tag}_judge")

    report: dict[str, dict] = {}
    for item in items:
        entry = {"question": item["question"], "samples": samples[item["key"]], "gold_sets": {}}
        for name in item["gold_sets"]:
            scored = [r for r in rows if r["key"] == item["key"] and r["gold_set"] == name]
            hits = sum(int(r.get("judge_correct", 0)) for r in scored)
            entry["gold_sets"][name] = {
                "golds": item["gold_sets"][name],
                "attempts": [{k: r[k] for k in ("sample", "pred", "em", "f1", "judge_correct")}
                             for r in sorted(scored, key=lambda r: r["sample"])],
                "known_count": hits,
                "known_rate": hits / max(1, len(scored)),
                "known_any": hits > 0,
            }
        report[item["key"]] = entry
    return report


# ----------------------------------------------------------- side-fact probes

def side_fact_prompt(document: str, question: str, answer: str, n: int) -> str:
    return f"""Document:
{document}

A separate evaluation already uses this question and answer, which you must NOT reuse:
Question: {question}
Answer: {answer}

Extract exactly {n} OTHER checkable factual details that this document states. Each one must:
  * be stated explicitly in the document above;
  * be independent of the question and answer shown above, and of each other;
  * have a short answer of 1-6 words that appears VERBATIM in the document;
  * be a name, number, date, place, title, or quantity - not a summary or an opinion.

No short answer may contain another one, or be contained in another one. "612 L.R." and
"19 Frostwane 612 L.R." are NOT two facts: a check for the first would succeed whenever the
second is present, so they must not both appear.

Return only a JSON object:
{{"facts": [{{"statement": "one sentence stating the detail",
              "question": "a question whose answer is the short answer",
              "answer": "the short answer, verbatim from the document",
              "aliases": ["optional other acceptable surface forms"]}}]}}
"""


def extract_side_facts(client: LLMClient, document: str, question: str, answer: str,
                       n: int, key: str, attempt: int) -> list[dict]:
    """Pull N document-supported probes, rejecting any that is not verbatim.

    A probe that is not literally in the document cannot distinguish "the chain
    dropped this" from "this was never there", so a failed verbatim check is a
    hard rejection rather than a warning.
    """
    result = client.chat(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": side_fact_prompt(document, question, answer, n)}],
        temperature=0.0, max_tokens=1200, seed=200 + attempt, tag=f"size_side_facts_{key}",
    )
    payload = json_object(result.text)
    facts = payload.get("facts") or []
    if not isinstance(facts, list):
        raise ValueError("facts was not a list")
    padded_doc = f" {sx.normalize(document)} "
    target_probes = [answer]
    out: list[dict] = []
    for i, row in enumerate(facts):
        short = str(row.get("answer", "")).strip()
        if len(short) < 3:
            raise ValueError(f"side fact {i} has an unusably short answer {short!r}")
        if not sx.in_document(short, padded_doc):
            raise ValueError(f"side fact answer {short!r} is not verbatim in the document")
        if sx.fact_present(short, target_probes) or sx.fact_present(answer, [short]):
            raise ValueError(f"side fact {short!r} overlaps the target answer")
        # Nested probes are not two facts. If "612 L.R." is inside "19 Frostwane
        # 612 L.R.", a presence check for the shorter one succeeds whenever the
        # longer survives, and the retention average silently double-counts one
        # fact while claiming to cover two.
        for kept in out:
            if sx.fact_present(short, [kept["answer"]]) or sx.fact_present(kept["answer"], [short]):
                raise ValueError(f"side facts {short!r} and {kept['answer']!r} contain each other")
        aliases = [str(a).strip() for a in (row.get("aliases") or []) if str(a).strip()]
        probes = [short] + [a for a in aliases if sx.in_document(a, padded_doc)]
        out.append({
            "fact_id": f"{key}:side{i + 1}", "role": "side",
            "statement": str(row.get("statement", "")).strip(),
            "question": str(row.get("question", "")).strip(),
            "answer": short, "aliases": aliases,
            "golds": [short] + aliases, "probes": probes,
        })
    if len(out) != n:
        raise ValueError(f"expected {n} side facts, got {len(out)}")
    return out


def side_facts_with_retries(client: LLMClient, document: str, question: str, answer: str,
                            n: int, key: str, attempts: int) -> list[dict]:
    """Re-ask on a rejected extraction instead of discarding the item.

    Every rejection above is a property of one sampled reply -- a probe that is
    not verbatim, or two probes that nest -- not of the document. Dropping the
    item on the first bad draw would let the corpus be shaped by sampling luck
    rather than by which documents can actually carry six independent facts.
    """
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return extract_side_facts(client, document, question, answer, n, key, attempt)
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
            print(f"[build] side-fact attempt {attempt + 1}/{attempts} for {key}: {exc}")
        except (HardAPIFailure, CostCapExceeded) as exc:
            raise ValueError(f"abandoned (API/credit): {exc}") from exc
    raise ValueError(f"no valid side-fact set after {attempts} attempts: {last}")


def side_facts_for_all(client: LLMClient, jobs: list[dict], cfg: dict,
                       attempts: int) -> dict[str, list[dict] | str]:
    """Extract probes for many documents at once.

    Returns key -> side-fact list, or key -> failure message. Extraction is one
    slow reasoning call per document and the documents are independent, so this
    fans out for the same reason every other stage in this project does.
    """
    def one(job: dict):
        try:
            return job["key"], side_facts_with_retries(
                client, job["document"], job["question"], job["answer"],
                int(cfg["build"]["side_facts"]), job["key"], attempts)
        except (ValueError, json.JSONDecodeError) as exc:
            return job["key"], str(exc)

    with ThreadPoolExecutor(max_workers=writer_concurrency(cfg)) as ex:
        return dict(ex.map(one, jobs))


# ------------------------------------------------------ counterfactual items

def gold_sets_disjoint(document_golds: list[str], parametric_golds: list[str]) -> bool:
    """Neither answer may be findable inside the other.

    If "Harvard Medical School" contains "Medical School" and the original
    answer is "Boston Medical School", a handoff naming only the document's
    value would also match the parametric probe, and every reversion count
    downstream would be inflated by string containment alone.
    """
    for a in document_golds:
        for b in parametric_golds:
            if sx.fact_present(a, [b]) or sx.fact_present(b, [a]):
                return False
    return True


def build_counterfactual(cfg: dict, base_cfg: dict, force: bool) -> dict:
    spec = cfg["build"]["counterfactual"]
    lf = cfg["build"]["leakage_filter"]
    output = ROOT / cfg["dataset"]["counterfactual"]["items_jsonl"]
    if output.exists() and not force:
        print(f"[cf] {output} exists; pass --force to rebuild")
        return {"skipped": True}

    modified = {r["qid"]: r for r in read_jsonl(ROOT / spec["source_counterfactual"]) if "qid" in r}
    original = {r["qid"]: r for r in read_jsonl(ROOT / spec["source_original"]) if "qid" in r}
    shared = [q for q in modified if q in original]
    print(f"[cf] {len(modified)} counterfactual questions, {len(shared)} joined to an original answer")

    client = LLMClient(cfg)
    probes = []
    for qid in shared:
        cf, orig = modified[qid], original[qid]
        probes.append({"key": qid, "question": cf["question"],
                       "gold_sets": {"document": cf["golds"], "parametric": orig["golds"]}})
    report = knowledge_report(client, probes, cfg, lf, tag="size_cf_closed_book")

    writer = LLMClient(writer_config(cfg, cfg["build"]["writer"]))
    # Screen on the knowledge conditions first, then extract probes only for the
    # survivors: extraction is the expensive call and a rejected item never
    # needs one.
    items, rejected, survivors = [], [], []
    for qid in shared:
        cf, orig = modified[qid], original[qid]
        entry = report[qid]
        knows_original = entry["gold_sets"]["parametric"]["known_rate"]
        knows_document = entry["gold_sets"]["document"]["known_any"]
        reasons = []
        if spec["require_knows_original"] and knows_original < float(spec["knows_original_min_rate"]):
            reasons.append(f"does not reliably know the original answer ({knows_original:.2f})")
        if spec["require_unknown_counterfactual"] and knows_document:
            reasons.append("already knows the replacement answer closed-book")
        if not gold_sets_disjoint(cf["golds"], orig["golds"]):
            reasons.append("replacement and original answers overlap as strings")
        if reasons:
            rejected.append({"qid": qid, "reasons": reasons,
                             "document_answer": cf["answer"], "parametric_answer": orig["answer"]})
            print(f"[cf] drop {qid}: {'; '.join(reasons)}")
            continue
        survivors.append({"key": qid, "question": cf["question"], "answer": cf["answer"],
                          "document": "\n\n".join(p["text"] for p in cf["paragraphs"])})

    extracted = side_facts_for_all(writer, survivors, cfg, int(spec.get("max_attempts", 3)))
    for job in survivors:
        qid = job["key"]
        cf, orig, entry = modified[qid], original[qid], report[qid]
        side = extracted.get(qid)
        if isinstance(side, str) or not side:
            rejected.append({"qid": qid, "reasons": [f"side-fact extraction failed: {side}"]})
            print(f"[cf] drop {qid}: side-fact extraction failed: {side}")
            continue
        items.append(make_item(
            item_id=qid, dataset="counterfactual", question=cf["question"], document=job["document"],
            document_answer=cf["answer"], document_aliases=cf.get("aliases") or [],
            parametric_answer=orig["answer"], parametric_aliases=orig.get("aliases") or [],
            side_facts=side, knowledge=entry,
        ))
        print(f"[cf] keep {qid}: document={cf['answer']!r} vs memory={orig['answer']!r} "
              f"(knows original {entry['gold_sets']['parametric']['known_rate']:.2f})")

    for item in items:
        item.setdefault("source", "rewritten_wikipedia")

    known_items, known_rejected, known_report = build_known_counterfactual(cfg)
    items.extend(known_items)
    rejected.extend(known_rejected)
    report.update(known_report)

    n_items = int(cfg["dataset"]["counterfactual"]["n_items"])
    # Known-entity items first: they are the ones sampled *for* this experiment's
    # requirement, so a run capped below the full set keeps the source that was
    # designed for it rather than whichever items happen to sort first.
    items.sort(key=lambda r: 0 if r.get("source") == "known_entity" else 1)
    kept = items[:n_items]
    by_source = {}
    for item in kept:
        by_source[item["source"]] = by_source.get(item["source"], 0) + 1
    print(f"[cf] kept by source: {by_source}")
    manifest = {
        "dataset": "size_adaptation_counterfactual",
        "sources": by_source,
        "source_counterfactual": spec["source_counterfactual"],
        "source_original": spec["source_original"],
        "source_hash": digest((ROOT / spec["source_counterfactual"]).read_text(encoding="utf-8")),
        "probe_model": cfg["model"]["id"],
        "extractor_model": cfg["build"]["writer"]["model_id"],
        "judge_model": judge_mod.judge_config(cfg)["model_id"],
        "candidates": len(shared), "kept": len(kept), "rejected": len(rejected),
        "requires": {"knows_original_min_rate": spec["knows_original_min_rate"],
                     "unknown_counterfactual": spec["require_unknown_counterfactual"]},
    }
    write_jsonl(output, [{"_manifest": manifest}] + kept)
    audit = ROOT / cfg["outputs"]["root"] / "counterfactual_knowledge_report.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({"manifest": manifest, "rejected": rejected, "closed_book": report},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cf] wrote {len(kept)} items -> {output}")
    print("[cf:probe cost] " + json.dumps(client.ledger.summary()))
    return {"kept": len(kept), "rejected": len(rejected)}


# ------------------------------------------- counterfactual: known entities
#
# The rewritten-Wikipedia source above was built from RANDOM Wikipedia pages,
# for a different experiment whose requirement was that the model must NOT know
# the answer. Experiment 9 needs the opposite as well: the model must know the
# original, or a handoff that says "Rome" cannot be attributed to memory. Random
# pages are mostly obscure, so most of them fail that condition -- not a flaw in
# those items, just the wrong sampling frame for this question.
#
# This source samples the other way round: start from subjects the model is
# likely to know, verify closed-book that it does, and only then rewrite the
# fact. Items keep a ``source`` field so the two provenances can always be
# separated in analysis rather than silently pooled.

KNOWN_SUBJECT_PROMPT = """Write a short factual encyclopedia passage about a WIDELY KNOWN subject,
for use as a reading-comprehension stimulus. Subject area for this one: {area}.

Requirements:
  * The subject must be famous enough that a well-read person could answer a basic
    factual question about it from memory. Prefer canonical, textbook-level subjects.
  * The passage must be accurate, 900-1800 characters, encyclopedic prose, no lists,
    no markdown headers, no citations.
  * Choose ONE prominent fact in the passage that such a reader would reliably know
    (a birthplace, a founding year, an author, a capital, a discoverer, or similar),
    and write a question whose answer is exactly that fact.
  * The answer must be 1-6 words and appear VERBATIM in the passage.
  * Do not write about anything contested, recent, or obscure.

Return only a JSON object:
{{"subject": "...", "text": "the passage", "question": "...",
  "answer": "the widely known answer, verbatim from the passage",
  "aliases": ["other acceptable surface forms"]}}
"""

COUNTERFACTUAL_REWRITE_PROMPT = """Passage:
{text}

Question used with this passage: {question}
Current true answer: {answer}

Rewrite the passage so that the answer to that question becomes a DIFFERENT but plausible
value, and the passage remains internally consistent with the change. Keep everything else
about the passage the same: same subject, same length, same register, same other facts.
Do not signal that anything was changed, and do not mention the original value anywhere.

The replacement must be the same kind of thing as the original (a place for a place, a year
for a year, a person for a person) and must be plausible enough that a reader with no prior
knowledge would not question it.

Return only a JSON object:
{{"rewritten_text": "the full rewritten passage",
  "counterfactual_answer": "the new answer, verbatim from rewritten_text",
  "aliases": ["other verbatim surface forms of the new answer"]}}
"""

KNOWN_AREAS = (
    "classical literature", "physics", "world geography", "classical music",
    "painting and sculpture", "ancient history", "biology", "mathematics",
    "architecture", "exploration and navigation", "chemistry", "philosophy",
    "astronomy", "medicine", "engineering", "world religions",
)


def build_known_passage(writer: LLMClient, index: int, attempt: int) -> dict:
    area = KNOWN_AREAS[(index - 1) % len(KNOWN_AREAS)]
    result = writer.chat(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": KNOWN_SUBJECT_PROMPT.format(area=area)}],
        temperature=1.0, max_tokens=3000, seed=700 + index * 10 + attempt,
        tag=f"size_known_passage_{index}",
    )
    payload = json_object(result.text)
    text = str(payload.get("text", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    question = str(payload.get("question", "")).strip()
    subject = str(payload.get("subject", "")).strip()
    if not (text and answer and question and subject):
        raise ValueError("known passage is missing a required field")
    if len(answer) < 3:
        raise ValueError(f"answer {answer!r} is too short to probe")
    if not sx.in_document(answer, f" {sx.normalize(text)} "):
        raise ValueError(f"answer {answer!r} is not verbatim in the passage")
    aliases = [str(a).strip() for a in (payload.get("aliases") or []) if str(a).strip()]
    return {"index": index, "area": area, "subject": subject, "text": text,
            "question": question, "answer": answer, "aliases": aliases}


def rewrite_counterfactual(writer: LLMClient, page: dict, attempt: int) -> dict:
    result = writer.chat(
        [{"role": "system", "content": WRITER_SYSTEM},
         {"role": "user", "content": COUNTERFACTUAL_REWRITE_PROMPT.format(
             text=page["text"], question=page["question"], answer=page["answer"])}],
        temperature=0.4, max_tokens=3000, seed=800 + page["index"] * 10 + attempt,
        tag=f"size_known_rewrite_{page['index']}",
    )
    payload = json_object(result.text)
    text = str(payload.get("rewritten_text", "")).strip()
    answer = str(payload.get("counterfactual_answer", "")).strip()
    if not (text and answer):
        raise ValueError("rewrite is missing a required field")
    if len(answer) < 3:
        raise ValueError(f"replacement answer {answer!r} is too short to probe")
    if not sx.in_document(answer, f" {sx.normalize(text)} "):
        raise ValueError(f"replacement {answer!r} is not verbatim in the rewritten passage")
    aliases = [str(a).strip() for a in (payload.get("aliases") or []) if str(a).strip()]
    aliases = [a for a in aliases if sx.in_document(a, f" {sx.normalize(text)} ")]
    # The rewrite must actually remove the original value. A passage that still
    # contains "Rome" alongside "Lisbon" would score as a reversion on every
    # single stage, before any handoff has happened.
    padded = f" {sx.normalize(text)} "
    for gold in [page["answer"]] + page["aliases"]:
        if sx.in_document(gold, padded):
            raise ValueError(f"rewritten passage still contains the original value {gold!r}")
    if not gold_sets_disjoint([answer] + aliases, [page["answer"]] + page["aliases"]):
        raise ValueError("replacement and original answers overlap as strings")
    return {"text": text, "answer": answer, "aliases": aliases}


def build_known_counterfactual(cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """Known-entity counterfactual items, verified in both directions.

    Returns (items, rejected, closed-book report). The two closed-book gates are
    applied at different points on purpose: knowledge of the ORIGINAL is checked
    before paying for a rewrite, and ignorance of the REPLACEMENT can only be
    checked after one exists.
    """
    spec = cfg["build"]["counterfactual"]
    known = spec.get("known_entities") or {}
    pages_wanted = int(known.get("pages", 0))
    if not known.get("enabled") or pages_wanted <= 0:
        return [], [], {}
    attempts = int(known.get("max_attempts", 3))
    writer = LLMClient(writer_config(cfg, cfg["build"]["writer"]))
    client = LLMClient(cfg)
    lf = cfg["build"]["leakage_filter"]

    def one_page(index: int):
        for attempt in range(attempts):
            try:
                return build_known_passage(writer, index, attempt)
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[cfk] passage {index} attempt {attempt + 1}: {exc}")
            except (HardAPIFailure, CostCapExceeded) as exc:
                print(f"[cfk] passage {index} abandoned (API/credit): {exc}")
                return None
        return None

    with ThreadPoolExecutor(max_workers=writer_concurrency(cfg)) as ex:
        built = [p for p in ex.map(one_page, range(1, pages_wanted + 1)) if p]
    print(f"[cfk] {len(built)}/{pages_wanted} known-subject passages written")

    # Gate 1: the model must actually know the original answer.
    probes = [{"key": f"cfk:{p['index']}", "question": p["question"],
               "gold_sets": {"parametric": [p["answer"]] + p["aliases"]}} for p in built]
    report = knowledge_report(client, probes, cfg, lf, tag="size_known_closed_book")
    rejected, survivors = [], []
    for page in built:
        key = f"cfk:{page['index']}"
        rate = report[key]["gold_sets"]["parametric"]["known_rate"]
        if rate < float(spec["knows_original_min_rate"]):
            rejected.append({"item_id": key, "subject": page["subject"],
                             "reasons": [f"does not reliably know the original ({rate:.2f})"]})
            print(f"[cfk] drop {key} ({page['subject']!r}): knows original {rate:.2f}")
            continue
        survivors.append(page)
    print(f"[cfk] {len(survivors)} passages survive the knows-the-original gate")

    def one_rewrite(page: dict):
        for attempt in range(attempts):
            try:
                return page["index"], rewrite_counterfactual(writer, page, attempt)
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[cfk] rewrite {page['index']} attempt {attempt + 1}: {exc}")
            except (HardAPIFailure, CostCapExceeded) as exc:
                print(f"[cfk] rewrite {page['index']} abandoned (API/credit): {exc}")
                return page["index"], None
        return page["index"], None

    with ThreadPoolExecutor(max_workers=writer_concurrency(cfg)) as ex:
        rewrites = dict(ex.map(one_rewrite, survivors))

    # Gate 2: the model must NOT know the replacement. Probed with the same
    # question, so the two gates are two readings of one behaviour.
    staged = [p for p in survivors if rewrites.get(p["index"])]
    probes2 = [{"key": f"cfk:{p['index']}", "question": p["question"],
                "gold_sets": {"document": [rewrites[p["index"]]["answer"]]
                              + rewrites[p["index"]]["aliases"]}} for p in staged]
    report2 = knowledge_report(client, probes2, cfg, lf, tag="size_known_closed_book")

    ready = []
    for page in staged:
        key = f"cfk:{page['index']}"
        if report2[key]["gold_sets"]["document"]["known_any"]:
            rejected.append({"item_id": key, "subject": page["subject"],
                             "reasons": ["already knows the replacement answer closed-book"]})
            print(f"[cfk] drop {key}: knows the replacement closed-book")
            continue
        ready.append(page)
    for page in survivors:
        if rewrites.get(page["index"]) is None:
            rejected.append({"item_id": f"cfk:{page['index']}", "subject": page["subject"],
                             "reasons": ["counterfactual rewrite never validated"]})

    jobs = [{"key": f"cfk:{p['index']}", "question": p["question"],
             "answer": rewrites[p["index"]]["answer"],
             "document": rewrites[p["index"]]["text"]} for p in ready]
    extracted = side_facts_for_all(writer, jobs, cfg, attempts)

    items = []
    for page in ready:
        key = f"cfk:{page['index']}"
        side = extracted.get(key)
        if isinstance(side, str) or not side:
            rejected.append({"item_id": key, "reasons": [f"side-fact extraction failed: {side}"]})
            continue
        rewrite = rewrites[page["index"]]
        knowledge = {"question": page["question"],
                     "gold_sets": {**report[key]["gold_sets"], **report2[key]["gold_sets"]},
                     "samples": report[key]["samples"]}
        item = make_item(
            item_id=key, dataset="counterfactual", question=page["question"],
            document=rewrite["text"], document_answer=rewrite["answer"],
            document_aliases=rewrite["aliases"], parametric_answer=page["answer"],
            parametric_aliases=page["aliases"], side_facts=side, knowledge=knowledge,
            title=page["subject"])
        item["source"] = "known_entity"
        item["subject_area"] = page["area"]
        items.append(item)
        print(f"[cfk] keep {key} ({page['subject']!r}): document={rewrite['answer']!r} "
              f"vs memory={page['answer']!r}")
    if len(items) < pages_wanted:
        print(f"[cfk] NOTE: {len(items)}/{pages_wanted} known-entity items completed. "
              "A short build is usually credit exhaustion, not a data problem; "
              "rerun with --force once credit is available and the cached pages are free.")
    print("[cfk:writer cost] " + json.dumps(writer.ledger.summary()))
    return items, rejected, {**report, **{f"{k}:replacement": v for k, v in report2.items()}}


# ----------------------------------------------------------- fictional items

# Cycled by index so subject variety is a property of the build rather than of
# what one sampling run happened to produce. Independent calls cannot see each
# other's output, so without this the writer returns near-identical subject
# types and the corpus stops being a sample of anything.
SUBJECT_TYPES = (
    "a small inland settlement", "a teaching institution", "a hand-made scientific instrument",
    "an annual festival or tradition", "a natural landform", "a craft guild or trade body",
    "a manuscript, map or archive", "a canal, bridge or road", "a mineral, plant or animal",
    "a museum or collection", "a legal charter or treaty", "a style of music or dance",
)


def fictional_prompt(index: int, side_facts: int, min_chars: int, max_chars: int) -> str:
    subject = SUBJECT_TYPES[(index - 1) % len(SUBJECT_TYPES)]
    return f"""Write a reference-encyclopedia article about an entirely invented subject: {subject}.

Requirements:
  * The subject, and every person, place, work, institution and event named in the article,
    must be invented. Do not reuse the name of any real entity, and do not name any real
    country, city, company, or person, even in passing.
  * Invented names must be pronounceable and distinctive - not near-copies of real names.
  * Length between {min_chars} and {max_chars} characters, in encyclopedic prose with a few
    short sections. No bullet lists, no markdown headers, no citations.
  * Write it as a real encyclopedia entry. Do NOT refer to the article itself or to this
    task: no phrases such as "the headline fact", "this article", "as requested", or
    "the key detail is". A reader must not be able to tell which fact is the important one.
  * State one fact that a reader could be asked about (a founder, an inventor, a date of
    establishment, or similar), plus at least {side_facts + 2} other specific checkable
    details: names, dates, quantities, places. Present all of them in the same register,
    without emphasis on any one of them.

Return only a JSON object:
{{"title": "...",
  "text": "the full article text",
  "question": "a question whose answer is the fact above",
  "answer": "that answer, 1-6 words, verbatim from the text",
  "aliases": ["optional other verbatim surface forms of the answer"]}}
"""


def build_one_fictional(writer: LLMClient, index: int, spec: dict, side_facts: int, attempt: int) -> dict:
    result = writer.chat(
        [{"role": "system", "content": WRITER_SYSTEM},
         {"role": "user", "content": fictional_prompt(
             index, side_facts, int(spec["min_page_characters"]),
             int(spec["max_page_characters"]))}],
        temperature=1.0, max_tokens=4000, seed=300 + index * 10 + attempt,
        tag=f"size_fictional_page_{index}",
    )
    payload = json_object(result.text)
    text = str(payload.get("text", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    question = str(payload.get("question", "")).strip()
    title = str(payload.get("title", "")).strip()
    if not (text and answer and question and title):
        raise ValueError("page is missing a required field")
    if not (int(spec["min_page_characters"]) <= len(text) <= int(spec["max_page_characters"])):
        raise ValueError(f"page length {len(text)} outside the configured range")
    if len(answer) < 3:
        raise ValueError(f"headline answer {answer!r} is too short to probe")
    if not sx.in_document(answer, f" {sx.normalize(text)} "):
        raise ValueError(f"headline answer {answer!r} is not verbatim in the page")
    aliases = [str(a).strip() for a in (payload.get("aliases") or []) if str(a).strip()]
    return {"title": title, "text": text, "question": question, "answer": answer,
            "aliases": [a for a in aliases if sx.in_document(a, f" {sx.normalize(text)} ")]}


def build_fictional(cfg: dict, force: bool) -> dict:
    spec = cfg["build"]["fictional"]
    output = ROOT / cfg["dataset"]["fictional"]["items_jsonl"]
    if output.exists() and not force:
        print(f"[fic] {output} exists; pass --force to rebuild")
        return {"skipped": True}

    writer = LLMClient(writer_config(cfg, cfg["build"]["writer"]))

    def one_page(index: int) -> dict | None:
        for attempt in range(int(spec["max_attempts"])):
            try:
                page = build_one_fictional(writer, index, spec,
                                           int(cfg["build"]["side_facts"]), attempt)
                page["index"] = index
                print(f"[fic] page {index}: {page['title']!r} ({len(page['text'])} chars) "
                      f"answer={page['answer']!r}")
                return page
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[fic] page {index} attempt {attempt + 1} failed: {exc}")
            except (HardAPIFailure, CostCapExceeded) as exc:
                # Out of credit is not a property of this page, and every page
                # already written is still good. Give up on this one and let the
                # build finish with what it has, rather than discarding the lot.
                print(f"[fic] page {index} abandoned (API/credit): {exc}")
                return None
        return None

    # Pages are independent, and the writer is a slow reasoning model. Order is
    # restored below so the corpus is a deterministic function of the seeds
    # rather than of which request happened to return first.
    indices = list(range(1, int(spec["pages"]) + 1))
    with ThreadPoolExecutor(max_workers=writer_concurrency(cfg)) as ex:
        built = list(ex.map(one_page, indices))
    pages = sorted([p for p in built if p], key=lambda p: p["index"])
    failures = [i for i, p in zip(indices, built) if p is None]
    if failures:
        print(f"[fic] warning: {len(failures)} page(s) never validated: {failures}")

    # The whole point of a fictional control is that the model cannot know the
    # answer. Verify it rather than assuming it: an invented name that happens
    # to collide with something real would silently reintroduce the very
    # parametric pathway this dataset exists to exclude.
    client = LLMClient(cfg)
    probes = [{"key": f"fic:{p['index']}", "question": p["question"],
               "gold_sets": {"document": [p["answer"]] + p["aliases"]}} for p in pages]
    report = knowledge_report(client, probes, cfg, cfg["build"]["leakage_filter"],
                             tag="size_fic_closed_book")

    items, rejected, survivors = [], [], []
    for page in pages:
        key = f"fic:{page['index']}"
        if report[key]["gold_sets"]["document"]["known_any"]:
            rejected.append({"item_id": key,
                             "reasons": ["model answered the invented fact closed-book"]})
            print(f"[fic] drop {key}: model already answers it closed-book")
            continue
        survivors.append({"key": key, "question": page["question"], "answer": page["answer"],
                          "document": page["text"], "page": page})

    extracted = side_facts_for_all(writer, survivors, cfg, int(spec["max_attempts"]))
    for job in survivors:
        key, page = job["key"], job["page"]
        side = extracted.get(key)
        if isinstance(side, str) or not side:
            rejected.append({"item_id": key, "reasons": [f"side-fact extraction failed: {side}"]})
            print(f"[fic] drop {key}: side-fact extraction failed: {side}")
            continue
        items.append(make_item(
            item_id=key, dataset="fictional", question=page["question"], document=page["text"],
            document_answer=page["answer"], document_aliases=page["aliases"],
            parametric_answer=None, parametric_aliases=[], side_facts=side,
            knowledge=report[key], title=page["title"],
        ))

    n_items = int(cfg["dataset"]["fictional"]["n_items"])
    kept = items[:n_items]
    manifest = {
        "dataset": "size_adaptation_fictional",
        "writer_model": cfg["build"]["writer"]["model_id"],
        "probe_model": cfg["model"]["id"],
        "judge_model": judge_mod.judge_config(cfg)["model_id"],
        "pages_requested": int(spec["pages"]), "pages_built": len(pages),
        "kept": len(kept), "rejected": len(rejected),
        "side_facts_per_item": int(cfg["build"]["side_facts"]),
    }
    write_jsonl(output, [{"_manifest": manifest}] + kept)
    audit = ROOT / cfg["outputs"]["root"] / "fictional_knowledge_report.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({"manifest": manifest, "rejected": rejected, "closed_book": report},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fic] wrote {len(kept)} items -> {output}")
    print("[fic:writer cost] " + json.dumps(writer.ledger.summary()))
    print("[fic:probe cost] " + json.dumps(client.ledger.summary()))
    return {"kept": len(kept), "rejected": len(rejected)}


# ---------------------------------------------------------------- item schema

def make_item(item_id: str, dataset: str, question: str, document: str,
              document_answer: str, document_aliases: list[str],
              parametric_answer: str | None, parametric_aliases: list[str],
              side_facts: list[dict], knowledge: dict, title: str = "") -> dict:
    """One experiment item, with its target fact stored like any side fact.

    Keeping the target in ``facts`` rather than beside it means the survival,
    loss and reappearance code paths never special-case it, so the answer fact
    and the reusable facts are measured by exactly the same instrument.
    """
    document_golds = [document_answer] + [a for a in document_aliases if a]
    parametric_golds = ([parametric_answer] + [a for a in parametric_aliases if a]
                        ) if parametric_answer else []
    target = {"fact_id": f"{item_id}:target", "role": "target",
              "statement": f"The answer to the target question is {document_answer}.",
              "question": question, "answer": document_answer,
              "aliases": list(document_aliases), "golds": document_golds,
              "probes": document_golds}
    return {
        "item_id": item_id, "dataset": dataset, "title": title,
        "question": question, "document": document,
        "document_characters": len(document),
        "document_answer": document_answer, "document_aliases": list(document_aliases),
        "document_golds": document_golds,
        "parametric_answer": parametric_answer, "parametric_aliases": list(parametric_aliases),
        "parametric_golds": parametric_golds,
        "has_parametric": bool(parametric_golds),
        # The side fact whose question is also *answered* at every depth, not
        # merely probed for presence. Chosen by content hash so the choice is
        # reproducible and independent of list order.
        "side_probe_fact_id": side_facts[int(digest(item_id), 16) % len(side_facts)]["fact_id"]
        if side_facts else None,
        "facts": [target] + side_facts,
        "closed_book": knowledge,
    }


# ---------------------------------------------------------------- entry point

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/size_adaptation_config.yaml")
    parser.add_argument("--which", default="both", choices=["both", "counterfactual", "fictional"])
    parser.add_argument("--force", action="store_true", help="rebuild an existing dataset")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    base_cfg = load_config(ROOT / "configs/config.yaml")
    if cfg["model"]["id"] != base_cfg["model"]["id"]:
        print(f"[build] note: probing {cfg['model']['id']}, configs/config.yaml uses {base_cfg['model']['id']}")

    summary = {}
    if args.which in ("both", "fictional"):
        summary["fictional"] = build_fictional(cfg, args.force)
    if args.which in ("both", "counterfactual"):
        summary["counterfactual"] = build_counterfactual(cfg, base_cfg, args.force)
    print("[build] " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
