"""Build the two multi-query corpora Experiment 10 runs on.

``--which squad``
    Groups of k human-written SQuAD questions that share one paragraph. Every
    question in a kept group passes the project's closed-book leakage filter
    against the answering model, and the group passes an independence audit by
    a different model family: all k must be answerable from the paragraph and
    must ask about genuinely distinct facts. Without both filters an
    off-diagonal "success" could be parametric recall, or the same fact asked
    twice - either of which would fake the absence of communication regret.

``--which relations``
    Fictional dossiers whose questions are laid out at *designed* distances
    from each other. Each dossier has several aspects; each aspect carries one
    anchor question, a paraphrase of it (same answer, different wording), a
    second fact about the same entity, and a fact about a different entity in
    the same aspect. Any anchor in another aspect is orthogonal. The labels are
    therefore part of the construction rather than an embedding estimate, which
    is the only way the distance analysis means anything.

Both corpora are written as JSONL with a leading ``_manifest`` line, in the
schema ``src/regret_data.py`` loads.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge as judge_mod  # noqa: E402
import regret_data as rd  # noqa: E402
import size_metrics as sx  # noqa: E402
from data import CLOSED_BOOK_SYSTEM  # noqa: E402
from llm import CostCapExceeded, HardAPIFailure, LLMClient, load_config  # noqa: E402
from score import extract_short_answer, normalize_answer, score_against_golds  # noqa: E402

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
             "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
             "were", "what", "when", "where", "which", "who", "why", "with", "many", "much",
             "name", "whose"}

AUDIT_SYSTEM = (
    "You audit question groups for a reading-comprehension experiment. "
    "You are strict and reply only with the requested JSON."
)

AUDIT_TEMPLATE = """Passage:
{context}

Questions:
{questions}

For EACH numbered question decide two things about it, judged only against the passage above.

  answerable: true only if the passage states the answer explicitly enough that a careful
              reader could give the reference answer shown beside the question.
  distinct:   true only if the question asks about a different fact from EVERY other question
              in the list. Two questions that have the same answer, or that are rewordings of
              each other, or where answering one necessarily gives away the other, are not
              distinct.

Return only a JSON object:
{{"verdicts": [{{"n": 1, "answerable": true, "distinct": true}}, ...]}}"""


# ---------------------------------------------------------------- io helpers

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    tmp.replace(path)


def json_object(text: str) -> dict:
    stripped = re.sub(r"^`{3}[a-zA-Z]*\s*|\s*`{3}$", "", str(text or "").strip(),
                      flags=re.MULTILINE).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in response")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    return obj


def sub_client(cfg: dict, spec: dict, cap_key: str = "cap_usd") -> LLMClient:
    """A client pinned to one auxiliary model with its own cap.

    The model id is already in every cache key, so sharing ``cache_dir`` is
    safe; a separate ledger keeps a builder helper from eating the experiment's
    cap, which several configs set as low as a couple of dollars.
    """
    import copy
    aux = copy.deepcopy(cfg)
    aux["model"] = {"id": spec["model_id"], "provider_order": None, "require_parameters": False}
    if spec.get("reasoning") is not None:
        aux["model"]["reasoning"] = spec["reasoning"]
    aux["cost"] = {"cap_usd": float(spec[cap_key]),
                   "warn_at_fraction": cfg.get("cost", {}).get("warn_at_fraction", 0.8)}
    return LLMClient(aux)


# ------------------------------------------------------ closed-book filtering

def closed_book_rows(client: LLMClient, probes: list[dict], spec: dict,
                     concurrency: int, tag: str) -> list[dict]:
    """One row per (question, sample). ``known`` is filled in by the judge."""
    jobs = [(probe, sample) for probe in probes for sample in range(int(spec["samples"]))]

    def one(job) -> dict:
        probe, sample = job
        result = client.chat(
            [{"role": "system", "content": CLOSED_BOOK_SYSTEM},
             {"role": "user", "content": f"Question: {probe['question']}\nAnswer:"}],
            temperature=float(spec["temperature"]), max_tokens=int(spec["max_tokens"]),
            seed=1000 + sample, tag=tag,
        )
        pred = extract_short_answer(result.text)
        em, f1 = score_against_golds(pred, list(probe["golds"]))
        return {"qid": probe["qid"], "sample": sample, "question": probe["question"],
                "golds": list(probe["golds"]), "pred": pred, "em": em, "f1": f1}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, jobs))


def leaked_qids(rows: list[dict], cfg: dict, spec: dict) -> tuple[set[str], str]:
    """Which questions the answering model already knows without the passage."""
    if judge_mod.judge_config(cfg)["enabled"]:
        judge_mod.add_judge(rows, cfg, tag="regret_c1_leakage_judge")
        for row in rows:
            row["known"] = bool(row["judge_correct"])
        method = "llm_judge"
    else:
        threshold = float(spec["f1_known_threshold"])
        for row in rows:
            row["known"] = bool(row["em"] == 1.0 or row["f1"] >= threshold)
        method = "f1_threshold"
    return {row["qid"] for row in rows if row["known"]}, method


# ------------------------------------------------------------- SQuAD grouping

def question_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def structural_groups(df: pd.DataFrame, spec: dict, seed: int) -> list[dict]:
    """Contexts with enough surface-distinct questions, before any model call.

    Questions are pruned on normalised gold answer and on question-token
    Jaccard so the expensive filters are not spent on pairs that are obviously
    the same fact asked twice.
    """
    k = int(spec["questions_per_context"])
    oversample = int(spec["oversample_questions"])
    max_jac = float(spec["max_question_jaccard"])
    lo, hi = int(spec["context_chars_min"]), int(spec["context_chars_max"])

    by_context: dict[str, list[dict]] = collections.OrderedDict()
    for row in df.itertuples(index=False):
        answers = list(dict.fromkeys(str(a) for a in row.answers["text"] if str(a).strip()))
        if not answers:
            continue
        by_context.setdefault(row.context, []).append({
            "qid": str(row.id), "question": str(row.question).strip(),
            "golds": answers, "title": str(row.title),
        })

    groups = []
    for context, items in by_context.items():
        if not (lo <= len(context) <= hi) or len(items) < k:
            continue
        kept: list[dict] = []
        seen_answers: set[str] = set()
        for item in items:
            answer_key = normalize_answer(item["golds"][0])
            if not answer_key or answer_key in seen_answers:
                continue
            tokens = question_tokens(item["question"])
            if any(jaccard(tokens, question_tokens(other["question"])) > max_jac
                   for other in kept):
                continue
            seen_answers.add(answer_key)
            kept.append(item)
            if len(kept) >= oversample:
                break
        if len(kept) >= k:
            groups.append({"context": context, "title": kept[0]["title"], "questions": kept})

    random.Random(seed).shuffle(groups)
    return groups[:int(spec["max_contexts_scanned"])]


def audit_group(client: LLMClient, group: dict, spec: dict) -> dict:
    """Answerability and mutual independence, judged by a different family."""
    numbered = "\n".join(
        f"{i + 1}. {q['question']}  [reference answer: {q['golds'][0]}]"
        for i, q in enumerate(group["questions"]))
    prompt = AUDIT_TEMPLATE.format(context=group["context"], questions=numbered)
    result = client.chat(
        [{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=int(spec["max_tokens"]), seed=None,
        response_format={"type": "json_object"}, tag="regret_group_audit",
    )
    try:
        payload = json_object(result.text)
        verdicts = {int(v["n"]): (bool(v["answerable"]), bool(v["distinct"]))
                    for v in payload["verdicts"]}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"unparsed audit: {exc}", "verdicts": {}}
    n = len(group["questions"])
    if set(verdicts) != set(range(1, n + 1)):
        return {"ok": False, "reason": "audit did not cover every question", "verdicts": verdicts}
    bad = [i for i, (a, d) in sorted(verdicts.items()) if not (a and d)]
    return {"ok": not bad, "reason": "" if not bad else f"questions {bad} failed the audit",
            "verdicts": {str(i): list(v) for i, v in sorted(verdicts.items())}}


def build_squad(cfg: dict, force: bool) -> dict:
    spec = cfg["build"]["squad"]
    out_path = ROOT / cfg["dataset"]["squad_groups"]["contexts_jsonl"]
    if out_path.exists() and not force:
        print(f"[squad] {out_path} exists; pass --force to rebuild")
        return {"skipped": True}

    parquet = ROOT / spec["local_parquet"]
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} is missing; see README 'Data'.")
    df = pd.read_parquet(parquet)
    groups = structural_groups(df, spec, int(spec["sample_seed"]))
    print(f"[squad] {len(groups)} structural candidate groups "
          f"({sum(len(g['questions']) for g in groups)} questions)")

    probes = [{"qid": q["qid"], "question": q["question"], "golds": q["golds"]}
              for g in groups for q in g["questions"]]
    answerer = LLMClient(cfg)
    rows = closed_book_rows(answerer, probes, cfg["leakage_filter"],
                            int(cfg["runtime"]["concurrency"]), "regret_c1_closed_book")
    leaked, method = leaked_qids(rows, cfg, cfg["leakage_filter"])
    print(f"[squad] closed-book leak ({method}): {len(leaked)}/{len(probes)} questions")

    k = int(spec["questions_per_context"])
    clean_groups = []
    for group in groups:
        survivors = [q for q in group["questions"] if q["qid"] not in leaked]
        if len(survivors) >= k:
            clean_groups.append({**group, "questions": survivors[:k]})
    print(f"[squad] {len(clean_groups)} groups keep {k} leak-free questions")

    audit_spec = dict(cfg["group_audit"])
    auditor = sub_client(cfg, audit_spec)
    with ThreadPoolExecutor(max_workers=int(audit_spec["concurrency"])) as pool:
        audits = list(pool.map(lambda g: audit_group(auditor, g, audit_spec), clean_groups))

    contexts, rejected = [], []
    for group, audit in zip(clean_groups, audits):
        if not audit["ok"]:
            rejected.append({"title": group["title"], "reason": audit["reason"]})
            continue
        index = len(contexts) + 1
        contexts.append({
            "context_id": f"sq:{index}",
            "dataset": "squad_groups",
            "title": group["title"],
            "source": group["context"].strip(),
            "questions": [{"qid": q["qid"], "question": q["question"], "golds": q["golds"],
                           "aspect": "", "role": ""} for q in group["questions"]],
            "meta": {"source_words": len(group["context"].split()),
                     "source_characters": len(group["context"]),
                     "audit": audit["verdicts"]},
        })
        if len(contexts) >= int(cfg["dataset"]["squad_groups"]["n_contexts"]):
            break

    manifest = {
        "dataset": "squad_groups",
        "parquet": spec["local_parquet"],
        "probe_model": cfg["model"]["id"],
        "judge_model": judge_mod.judge_config(cfg)["model_id"],
        "audit_model": audit_spec["model_id"],
        "questions_per_context": k,
        "structural_candidates": len(groups),
        "closed_book_method": method,
        "closed_book_leaked_questions": len(leaked),
        "groups_after_leakage": len(clean_groups),
        "groups_rejected_by_audit": len(rejected),
        "kept": len(contexts),
        "context_chars_band": [int(spec["context_chars_min"]), int(spec["context_chars_max"])],
        "max_question_jaccard": float(spec["max_question_jaccard"]),
    }
    write_jsonl(out_path, [{"_manifest": manifest}] + contexts)
    audit_path = ROOT / cfg["outputs"]["build_root"] / "squad_groups_build.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(
        {"manifest": manifest, "rejected": rejected, "closed_book": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[squad] wrote {len(contexts)} contexts -> {out_path}")
    print("[squad:probe cost] " + json.dumps(answerer.ledger.summary()))
    print("[squad:audit cost] " + json.dumps(auditor.ledger.summary()))
    return manifest


# -------------------------------------------------------- relation dossiers

SUBJECTS = (
    "a craft guild", "a mountain observatory", "an inland canal company",
    "a regional library", "a coastal lighthouse service", "a botanical garden",
    "a mining cooperative", "a travelling theatre company", "a river ferry trust",
    "a clockmakers' school", "a salt works", "a cartographers' society",
    "a bell foundry", "a weather station", "a seed bank", "a paper mill",
)

# Four aspects, fixed for every dossier so the orthogonal distance always means
# the same thing: a fact from a different section of the same document.
ASPECTS = (
    ("founding", "its founding and early history"),
    ("facilities", "its buildings, premises or equipment"),
    ("finance", "its money: fees, endowments, levies or trade"),
    ("custom", "a recurring custom, ceremony or publication"),
)

RELATION_WRITER_PROMPT = """Write a reference-encyclopedia article about an entirely invented \
subject: {subject}. Then describe the facts it contains.

The article must have exactly {n_aspects} short sections, one per aspect below, in this order:
{aspect_lines}

Requirements for the article:
  * The subject, and every person, place, work, institution and event named, must be invented.
    Do not reuse the name of any real entity, and do not name any real country, city, company
    or person, even in passing. Invented names must be pronounceable and distinctive.
  * Length between {min_chars} and {max_chars} characters, in plain encyclopedic prose.
    No bullet lists, no markdown headers, no citations, no section titles.
  * Write it as a real encyclopedia entry. Never refer to the article itself or to this task,
    and never signal that one fact matters more than another. Present every detail in the same
    register.

Within EACH section you must state four things, all as ordinary prose:
  A. one specific checkable fact about the MAIN SUBJECT (a name, date, quantity or place);
  B. a SECOND, different specific fact about the MAIN SUBJECT in that same section;
  C. one specific fact about a DIFFERENT named entity introduced in that same section
     (a person, building, vessel, document or sub-body belonging to that aspect).
  Each of A, B and C must have a short answer of 1-6 words that appears VERBATIM in the article,
  and the three answers in a section must be different from each other and from every other
  section's answers.

Return only a JSON object:
{{"title": "...",
  "text": "the full article text",
  "aspects": [
    {{"aspect": "<the aspect key>",
      "anchor":       {{"question": "...", "answer": "...", "paraphrase_question": "..."}},
      "same_entity":  {{"question": "...", "answer": "..."}},
      "same_topic":   {{"question": "...", "answer": "..."}}}}
  ]}}

``anchor`` is fact A, ``same_entity`` is fact B, ``same_topic`` is fact C.
``paraphrase_question`` must be a differently worded question with exactly the same answer as
``anchor``; it must not reveal the answer and must not be answerable by wording alone."""


def relation_prompt(index: int, spec: dict) -> str:
    subject = SUBJECTS[(index - 1) % len(SUBJECTS)]
    aspect_lines = "\n".join(f"  {i + 1}. {key}: {gloss}"
                             for i, (key, gloss) in enumerate(ASPECTS))
    return RELATION_WRITER_PROMPT.format(
        subject=subject, n_aspects=len(ASPECTS), aspect_lines=aspect_lines,
        min_chars=int(spec["min_characters"]), max_chars=int(spec["max_characters"]))


def parse_relation_page(payload: dict, index: int, spec: dict) -> dict:
    """Turn one writer response into a context row, or raise with the reason.

    Every designed answer must appear verbatim in the article: an answer the
    document does not state is not a fact about the document, and would make
    the direct-context ceiling - the reference value in regret - unreachable.
    """
    text = str(payload.get("text", "")).strip()
    title = str(payload.get("title", "")).strip()
    if not text or not title:
        raise ValueError("missing title or text")
    lo, hi = int(spec["min_characters"]), int(spec["max_characters"])
    if not lo <= len(text) <= hi:
        raise ValueError(f"length {len(text)} outside [{lo}, {hi}]")
    padded = f" {sx.normalize(text)} "
    aspects = payload.get("aspects") or []
    if len(aspects) != len(ASPECTS):
        raise ValueError(f"expected {len(ASPECTS)} aspects, got {len(aspects)}")

    questions, seen_answers = [], set()
    for slot, (key, _gloss) in zip(aspects, ASPECTS):
        if str(slot.get("aspect", "")).strip() != key:
            raise ValueError(f"aspect key mismatch: {slot.get('aspect')!r} != {key!r}")
        anchor = slot["anchor"]
        answer = str(anchor["answer"]).strip()
        paraphrase_q = str(anchor.get("paraphrase_question", "")).strip()
        if not paraphrase_q:
            raise ValueError(f"{key}: missing paraphrase_question")
        entries = [
            (rd.ANCHOR, str(anchor["question"]).strip(), answer),
            (rd.PARAPHRASE, paraphrase_q, answer),
            (rd.SAME_ENTITY, str(slot["same_entity"]["question"]).strip(),
             str(slot["same_entity"]["answer"]).strip()),
            (rd.SAME_TOPIC, str(slot["same_topic"]["question"]).strip(),
             str(slot["same_topic"]["answer"]).strip()),
        ]
        for role, question, value in entries:
            if not question or len(value) < 2:
                raise ValueError(f"{key}/{role}: empty question or answer")
            if not sx.in_document(value, padded):
                raise ValueError(f"{key}/{role}: answer {value!r} is not verbatim in the text")
            if role != rd.PARAPHRASE:
                norm = normalize_answer(value)
                if norm in seen_answers:
                    raise ValueError(f"{key}/{role}: answer {value!r} repeats another fact")
                seen_answers.add(norm)
            questions.append({"qid": f"rel:{index}:{key}:{role}", "question": question,
                              "golds": [value], "aspect": key, "role": role})
    return {
        "context_id": f"rel:{index}",
        "dataset": "relation_dossiers",
        "title": title,
        "source": text,
        "questions": questions,
        "meta": {"source_words": len(text.split()), "source_characters": len(text),
                 "subject": SUBJECTS[(index - 1) % len(SUBJECTS)]},
    }


def build_one_relation(writer: LLMClient, index: int, spec: dict, attempt: int) -> dict:
    result = writer.chat(
        [{"role": "system", "content":
          "You write reference material for a controlled reading-comprehension experiment. "
          "You follow the requested JSON schema exactly."},
         {"role": "user", "content": relation_prompt(index, spec)}],
        temperature=1.0, max_tokens=int(spec["writer_max_tokens"]),
        seed=700 + index * 10 + attempt, response_format={"type": "json_object"},
        tag=f"regret_relation_page_{index}",
    )
    row = parse_relation_page(json_object(result.text), index, spec)
    rd.validate_relation_context(rd._context(row))
    return row


def build_relations(cfg: dict, force: bool) -> dict:
    spec = cfg["build"]["relations"]
    out_path = ROOT / cfg["dataset"]["relation_dossiers"]["contexts_jsonl"]
    if out_path.exists() and not force:
        print(f"[rel] {out_path} exists; pass --force to rebuild")
        return {"skipped": True}

    writer = sub_client(cfg, cfg["build"]["writer"])

    def one(index: int) -> dict | None:
        for attempt in range(int(spec["max_attempts"])):
            try:
                row = build_one_relation(writer, index, spec, attempt)
                print(f"[rel] page {index}: {row['title']!r} "
                      f"({row['meta']['source_characters']} chars, "
                      f"{len(row['questions'])} questions)")
                return row
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                print(f"[rel] page {index} attempt {attempt + 1} failed: {exc}")
            except (HardAPIFailure, CostCapExceeded) as exc:
                # Out of credit is not a property of this page; keep what is built.
                print(f"[rel] page {index} abandoned (API/credit): {exc}")
                return None
        return None

    indices = list(range(1, int(spec["pages"]) + 1))
    with ThreadPoolExecutor(max_workers=int(cfg["build"]["writer"]["concurrency"])) as pool:
        built = list(pool.map(one, indices))
    pages = [row for row in built if row]
    print(f"[rel] {len(pages)}/{len(indices)} pages validated")

    # A fictional corpus is only a control if the model really cannot answer it
    # from memory. Verify rather than assume: an invented name that collides
    # with something real would reintroduce the parametric pathway.
    answerer = LLMClient(cfg)
    probes = [{"qid": q["qid"], "question": q["question"], "golds": q["golds"]}
              for page in pages for q in page["questions"]]
    rows = closed_book_rows(answerer, probes, cfg["leakage_filter"],
                            int(cfg["runtime"]["concurrency"]), "regret_rel_closed_book")
    leaked, method = leaked_qids(rows, cfg, cfg["leakage_filter"])

    contexts, rejected = [], []
    for page in pages:
        hit = [q["qid"] for q in page["questions"] if q["qid"] in leaked]
        if hit:
            rejected.append({"context_id": page["context_id"],
                             "reason": f"{len(hit)} question(s) answered closed-book"})
            print(f"[rel] drop {page['context_id']}: {len(hit)} closed-book leak(s)")
            continue
        contexts.append(page)
        if len(contexts) >= int(cfg["dataset"]["relation_dossiers"]["n_contexts"]):
            break

    manifest = {
        "dataset": "relation_dossiers",
        "writer_model": cfg["build"]["writer"]["model_id"],
        "probe_model": cfg["model"]["id"],
        "judge_model": judge_mod.judge_config(cfg)["model_id"],
        "aspects": [key for key, _ in ASPECTS],
        "roles_per_aspect": [rd.ANCHOR, rd.PARAPHRASE, rd.SAME_ENTITY, rd.SAME_TOPIC],
        "pages_requested": len(indices), "pages_built": len(pages),
        "closed_book_method": method, "closed_book_leaked_questions": len(leaked),
        "kept": len(contexts), "rejected": len(rejected),
    }
    write_jsonl(out_path, [{"_manifest": manifest}] + contexts)
    audit_path = ROOT / cfg["outputs"]["build_root"] / "relation_dossiers_build.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(
        {"manifest": manifest, "rejected": rejected, "closed_book": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rel] wrote {len(contexts)} dossiers -> {out_path}")
    print("[rel:writer cost] " + json.dumps(writer.ledger.summary()))
    print("[rel:probe cost] " + json.dumps(answerer.ledger.summary()))
    return manifest


# ---------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="communication_regret_config.yaml")
    parser.add_argument("--which", choices=["squad", "relations", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="rebuild an existing corpus")
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    if args.which in ("squad", "both"):
        build_squad(cfg, args.force)
    if args.which in ("relations", "both"):
        build_relations(cfg, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
