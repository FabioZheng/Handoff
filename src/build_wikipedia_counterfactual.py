"""Build a leakage-proof counterfactual variant of the generated Wikipedia dataset.

The C1 filter (src/data.py) only *detects and discards* questions the model
already answers from pretraining; it does not stop new leakage from
appearing if the dataset is reused. This script instead removes the
possibility of leakage at the source: a strong reasoning model rewrites each
page's real evidence-bearing fact(s) into a different, plausible value (a
different date, institution, place, ...), then rewrites the article text so
it is internally consistent with the invented fact. The question text is
unchanged -- it still names the real subject -- but the answer it is scored
against no longer exists in any pretraining corpus, because that specific
(subject, fact) pairing was invented here and never occurred in the real
world. A model can therefore only "leak" it by coincidence, not by recall.

Both of a page's questions are rewritten together, in one call, so the
rewritten article stays internally consistent (e.g. a changed birth year
must still agree with any stated age elsewhere in the same article).

After construction, this script immediately runs the project's own C1
closed-book filter (LLM-judge scored, see data.apply_c1) against the
counterfactual answers, using the *experiment* model (config.yaml), not the
strong editing model -- proving empirically that the rewrite defeats
parametric leakage rather than merely assuming it.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
from data import GoldSentence, Paragraph, Question, split_sentences  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402

COUNTERFACTUAL_SYSTEM = (
    "You create counterfactual research data by editing factual Wikipedia articles. "
    "You invent plausible replacement facts and rewrite an article to be internally "
    "consistent with them, for a dataset that must not be answerable from anyone's "
    "prior knowledge of the real subject."
)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON was not an object")
    return value


def make_editor_config(base_cfg: dict, wiki_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["id"] = wiki_cfg["model_id"]
    cfg["model"]["reasoning"] = wiki_cfg.get("reasoning")
    cfg["cost"]["cap_usd"] = float(wiki_cfg["cost_cap_usd"])
    return cfg


def counterfactual_prompt(page_title: str, page_text: str, questions: list[Question]) -> str:
    facts = "\n\n".join(
        f"Question {i}: {q.question}\nCurrent true answer: {q.answer}"
        for i, q in enumerate(questions, start=1)
    )
    return f'''Article title: {page_title}

Full article text:
{page_text}

This article will be used to test whether a language model already knows facts about
"{page_title}" from its training data, rather than needing to read the article. To make
that impossible, invent a fictional alternate version of the specific fact(s) below and
rewrite the article around them.

{facts}

For each question, invent a DIFFERENT, plausible replacement fact of the same type and
grammatical role as the current true answer (e.g. a different specific date for a date, a
different specific institution for an institution, a different specific place for a place,
a different specific number for a number). The replacement must not be the value you
actually know for the real "{page_title}" -- it must describe something that did not
happen to the real subject, so the pairing of this subject with this fact has never
appeared in any real text. It may reuse an ordinary real-world-style value (a real class of
place/institution/date) as long as it is not the true one for this subject; invent a
plausible fictional one only where a real-world substitute would look implausible.

Rewrite the ENTIRE article text, substituting your invented replacement fact(s) everywhere
they are mentioned or logically implied (for example, if you change a birth date, also
update any stated age or age-derived claim elsewhere in the article so it stays internally
consistent). Leave every other fact, name, and section of the article unchanged. Do not add
a note, disclaimer, or any marker that the article has been altered -- it must read as an
ordinary, unremarkable Wikipedia article about this counterfactual version of events.

Return JSON only, with one entry in "answers" per question above, in the same order:
{{"rewritten_text": "...",
  "answers": [
    {{"counterfactual_answer": "...", "aliases": ["..."], "evidence": ["verbatim excerpt from rewritten_text"]}}
  ]}}'''


def build_counterfactual_page(client: LLMClient, title: str, text: str, questions: list[Question],
                               index: int, attempt: int) -> tuple[str, list[dict]]:
    prompt = counterfactual_prompt(title, text, questions)
    result = client.chat(
        [{"role": "system", "content": COUNTERFACTUAL_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0 if attempt == 0 else 0.7, max_tokens=8000,
        seed=100 + attempt, tag=f"wiki_counterfactual_{index}",
    )
    row = json_object(result.text)
    rewritten = str(row.get("rewritten_text", "")).strip()
    answers = row.get("answers")
    if not rewritten or not isinstance(answers, list) or len(answers) != len(questions):
        raise ValueError("editor did not return a rewritten article with one answer per question")
    rewritten_norm = normalize(rewritten)
    out = []
    for q, ans_row in zip(questions, answers):
        answer = str(ans_row.get("counterfactual_answer", "")).strip()
        aliases = list(dict.fromkeys(str(x).strip() for x in ans_row.get("aliases", []) if str(x).strip()))
        evidence = [str(x).strip() for x in ans_row.get("evidence", []) if str(x).strip()]
        if not answer or not evidence:
            raise ValueError(f"empty counterfactual answer/evidence for {q.qid}")
        if normalize(answer) == normalize(q.answer):
            raise ValueError(f"counterfactual answer for {q.qid} is unchanged from the real answer")
        if normalize(answer) not in rewritten_norm:
            raise ValueError(f"counterfactual answer for {q.qid} is not a verbatim span in the rewritten text")
        if not all(normalize(e) in rewritten_norm for e in evidence):
            raise ValueError(f"non-verbatim evidence returned for {q.qid}")
        aliases = [a for a in aliases if normalize(a) != normalize(answer) and normalize(a) in rewritten_norm]
        out.append({"answer": answer, "aliases": aliases, "evidence": evidence})
    return rewritten, out


def to_counterfactual_question(page_title: str, rewritten_text: str, original: Question, edit: dict) -> Question:
    paragraph = Paragraph(1, 0, page_title, rewritten_text, True)
    evidence_norm = [normalize(e) for e in edit["evidence"]]
    sentences = [s for s in split_sentences(rewritten_text)
                 if any(normalize(s) in e or e in normalize(s) for e in evidence_norm)]
    if not sentences:
        sentences = edit["evidence"]
    gold_sentences = [GoldSentence(i, 1, s, edit["answer"], True) for i, s in enumerate(sentences)]
    return Question(
        qid=original.qid, question=original.question, answer=edit["answer"],
        aliases=edit["aliases"], paragraphs=[paragraph], decomposition=[], gold_pids=[1],
        gold_sentences=gold_sentences, n_hops=1,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_source_questions(path: Path) -> list[Question]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return [Question.from_json(r) for r in rows if "qid" in r]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--chain-config", default=str(ROOT / "chain_config.yaml"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="replace an existing counterfactual dataset")
    args = parser.parse_args()
    base_cfg, chain_cfg = load_config(args.config), load_config(args.chain_config)
    wiki_cfg = chain_cfg["datasets"]["wikipedia_random"]

    source_path = ROOT / wiki_cfg["local_jsonl"]
    output = ROOT / "data" / "wikipedia_random_counterfactual" / "questions.jsonl"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; pass --force to deliberately replace it")

    source_questions = load_source_questions(source_path)
    by_page: dict[str, list[Question]] = {}
    for q in source_questions:
        by_page.setdefault(q.paragraphs[0].title, []).append(q)

    editor_client = LLMClient(make_editor_config(base_cfg, wiki_cfg))
    counterfactual_questions: list[Question] = []
    failures: list[str] = []
    for index, (title, qs) in enumerate(sorted(by_page.items())):
        text = qs[0].paragraphs[0].text
        rewritten = None
        for attempt in range(args.max_attempts):
            try:
                rewritten, edits = build_counterfactual_page(editor_client, title, text, qs, index, attempt)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[counterfactual] attempt {attempt + 1}/{args.max_attempts} failed for {title!r}: {exc}")
        if rewritten is None:
            failures.append(title)
            continue
        for q, edit in zip(qs, edits):
            new_q = to_counterfactual_question(title, rewritten, q, edit)
            counterfactual_questions.append(new_q)
            print(f"[counterfactual] {q.qid}: {q.answer!r} -> {edit['answer']!r}")
    if failures:
        raise SystemExit(f"failed to build counterfactual edits for {len(failures)} page(s) "
                          f"after {args.max_attempts} attempts each: {failures}")

    manifest = {
        "dataset": "wikipedia_random_counterfactual",
        "source_path": wiki_cfg["local_jsonl"],
        "source_hash": data_mod._hash_file(source_path),
        "editor_model": wiki_cfg["model_id"],
        "pages": len(by_page), "questions": len(counterfactual_questions),
        "method": "strong-model rewrite of each page's evidence-bearing fact(s) into a "
                  "different plausible value, jointly across both of a page's questions, "
                  "with the rewritten article kept internally consistent",
    }
    write_jsonl(output, [{"_manifest": manifest}] + [q.to_json() for q in counterfactual_questions])
    print(f"[counterfactual] wrote {len(counterfactual_questions)} questions -> {output}")
    print("[counterfactual:editor cost] " + json.dumps(editor_client.ledger.summary()))

    # Empirically verify the rewrite defeats leakage, using the *experiment* model
    # (config.yaml), not the strong editor model, and the same LLM-judge C1 check
    # the main pipeline uses.
    print(f"[counterfactual] running C1 against {base_cfg['model']['id']} "
          f"(the experiment model, not the editor)")
    test_client = LLMClient(base_cfg)
    kept, report = data_mod.apply_c1(
        test_client, counterfactual_questions, base_cfg, base_cfg["runtime"]["concurrency"])
    report_dir = ROOT / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wikipedia_counterfactual_leakage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    slim = {k: v for k, v in report.items() if k != "attempts"}
    (report_dir / "wikipedia_counterfactual_leakage_summary.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[counterfactual] C1 leakage check ({report['known_method']}): "
          f"{report['candidates_run']} candidates -> {report['leaked_excluded']} leaked "
          f"({100 * report['leak_rate']:.1f}% leak rate)")
    if report.get("leaked_qids"):
        print(f"[counterfactual] leaked qids: {', '.join(report['leaked_qids'])}")
    print("[counterfactual:test cost] " + json.dumps(test_client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
