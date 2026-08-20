"""Build a small, revision-pinned QA dataset from random English Wikipedia pages.

Each retained page is stored in full as plain text, then a strong reasoning
model independently creates two unrelated, source-grounded questions and
another pass produces the canonical answer/aliases.  The resulting JSONL uses
the project's ``Question`` schema directly, so no external benchmark remains
in the evaluation path after construction.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import GoldSentence, Paragraph, Question, split_sentences  # noqa: E402
from llm import LLMClient, load_config  # noqa: E402

WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_USER_AGENT = "handoff-information-loss-probe/1.0 (research dataset builder)"

QUESTION_SYSTEM = (
    "You construct rigorous, source-grounded factual QA data. Use only the supplied "
    "Wikipedia article; never rely on outside knowledge."
)
ANSWER_SYSTEM = (
    "You produce canonical gold answers for a source-grounded factual QA dataset. "
    "Use only the supplied Wikipedia article and evidence excerpts."
)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def json_object(text: str) -> dict:
    """Parse a model JSON object, tolerating a Markdown fence."""
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


class _TextExtractor(HTMLParser):
    """Dependency-free conversion of the Wikipedia REST article HTML to text."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "table"}:
            self._ignored += 1
        elif tag in {"p", "h2", "h3", "h4", "li", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "table"} and self._ignored:
            self._ignored -= 1
        elif tag in {"p", "h2", "h3", "h4", "li", "blockquote"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(" ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip())


def fetch_random_page(session: requests.Session, min_chars: int, max_chars: int) -> dict | None:
    """Fetch a random main-namespace page, retaining its complete plaintext article."""
    try:
        summary = session.get(f"{WIKIPEDIA_REST}/page/random/summary", timeout=60)
        summary.raise_for_status()
        page = summary.json()
        if page.get("type") != "standard" or page.get("namespace", {}).get("id") != 0:
            return None
        canonical = str(page.get("titles", {}).get("canonical", ""))
        if not canonical:
            return None
        # Wikimedia's REST edge can throttle an immediate random-summary → article
        # request pair from the same client.  A modest pause keeps the builder a
        # polite, low-volume consumer and avoids retrying a rejected request.
        time.sleep(1.5)
        article = session.get(f"{WIKIPEDIA_REST}/page/html/{quote(canonical, safe='')}", timeout=60)
        article.raise_for_status()
        parser = _TextExtractor()
        parser.feed(article.text)
        extract = parser.text()
        if len(extract) < min_chars or len(extract) > max_chars:
            return None
        return {
            "pageid": int(page["pageid"]), "title": str(page["title"]),
            "url": str(page.get("content_urls", {}).get("desktop", {}).get("page", "")),
            "revision_id": page.get("revision"), "revision_timestamp": page.get("timestamp"), "text": extract,
        }
    except requests.RequestException as exc:
        print(f"[wiki] transient source error: {exc}; retrying later")
        time.sleep(5)
        return None
    finally:
        # This applies to rejected size/type candidates as well, which otherwise
        # make rapid random-endpoint calls during an unlucky streak.
        time.sleep(2)


def make_generator_config(base_cfg: dict, wiki_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["id"] = wiki_cfg["model_id"]
    cfg["model"]["reasoning"] = wiki_cfg.get("reasoning")
    # Dataset construction has a separate cap from the experiment itself.
    cfg["cost"]["cap_usd"] = float(wiki_cfg["cost_cap_usd"])
    return cfg


def generate_questions(client: LLMClient, page: dict, index: int) -> list[dict]:
    prompt = f'''Article title: {page["title"]}

Full article text:
{page["text"]}

Create exactly two factual questions that can be answered unambiguously from this article.
The two questions must be unrelated: they must concern different entities, events, sections,
or properties, and answering one must not substantially answer the other. Prefer details that
are not obvious from a page title or broad common knowledge. Each question needs one or two
short, verbatim evidence excerpts from the article that directly establish its answer.

Return JSON only:
{{"questions":[{{"question":"...","evidence":["verbatim excerpt"]}},{{"question":"...","evidence":["verbatim excerpt"]}}]}}'''
    result = client.chat(
        [{"role": "system", "content": QUESTION_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=900, seed=None, tag=f"wiki_questions_{index}",
    )
    rows = json_object(result.text).get("questions")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("question generator did not return exactly two questions")
    out = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        evidence = [str(x).strip() for x in row.get("evidence", []) if str(x).strip()]
        if not question or not evidence:
            raise ValueError("question generator returned an empty question or evidence")
        if not all(normalize(item) in normalize(page["text"]) for item in evidence):
            raise ValueError("question generator supplied non-verbatim evidence")
        out.append({"question": question, "evidence": evidence})
    # Cheap deterministic guard against two paraphrases of the same question.
    first, second = set(normalize(out[0]["question"]).split()), set(normalize(out[1]["question"]).split())
    if len(first & second) / max(1, min(len(first), len(second))) > 0.70:
        raise ValueError("generated questions are too similar")
    return out


def generate_answer(client: LLMClient, page: dict, item: dict, index: int) -> tuple[str, list[str]]:
    evidence = "\n".join(f"- {text}" for text in item["evidence"])
    prompt = f'''Article title: {page["title"]}

Full article text:
{page["text"]}

Question: {item["question"]}

Verified evidence excerpts:
{evidence}

Derive the shortest precise answer supported by the article. Include alternate acceptable forms
only when they are genuinely equivalent (spelling, abbreviation, or name variants), not broader
paraphrases. Return JSON only:
{{"answer":"canonical answer","aliases":["acceptable alternate"]}}'''
    result = client.chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=300, seed=None, tag=f"wiki_gold_{index}",
    )
    row = json_object(result.text)
    answer = str(row.get("answer", "")).strip()
    aliases = list(dict.fromkeys(str(x).strip() for x in row.get("aliases", []) if str(x).strip()))
    if not answer:
        raise ValueError("gold-answer generator returned an empty answer")
    page_text = normalize(page["text"])
    if normalize(answer) not in page_text:
        raise ValueError("gold answer is not a verbatim span in the source page")
    aliases = [alias for alias in aliases if normalize(alias) != normalize(answer)]
    return answer, [alias for alias in aliases if normalize(alias) in page_text]


def to_question(page: dict, item: dict, answer: str, aliases: list[str], item_index: int) -> Question:
    paragraph = Paragraph(1, 0, page["title"], page["text"], True)
    # Keep only source sentences overlapping a verified evidence excerpt.  These
    # power the E_oracle equivalent without pretending Wikipedia supplies labels.
    evidence_norm = [normalize(e) for e in item["evidence"]]
    sentences = [s for s in split_sentences(page["text"])
                 if any(normalize(s) in e or e in normalize(s) for e in evidence_norm)]
    if not sentences:
        sentences = item["evidence"]
    gold_sentences = [GoldSentence(i, 1, s, answer, True) for i, s in enumerate(sentences)]
    return Question(
        qid=f"wiki:{page['pageid']}:{item_index}", question=item["question"], answer=answer,
        aliases=aliases, paragraphs=[paragraph], decomposition=[], gold_pids=[1],
        gold_sentences=gold_sentences, n_hops=1,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--chain-config", default=str(ROOT / "chain_config.yaml"))
    parser.add_argument("--pages", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="replace an existing generated dataset")
    args = parser.parse_args()
    base_cfg, chain_cfg = load_config(args.config), load_config(args.chain_config)
    wiki_cfg = chain_cfg["datasets"]["wikipedia_random"]
    output = ROOT / wiki_cfg["local_jsonl"]
    sources_output = ROOT / wiki_cfg["source_pages_jsonl"]
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; pass --force to deliberately replace it")
    target_pages = args.pages or int(wiki_cfg["pages"])
    client = LLMClient(make_generator_config(base_cfg, wiki_cfg))
    session = requests.Session()
    session.headers.update({"User-Agent": WIKIPEDIA_USER_AGENT})
    pages, questions = [], []
    attempts = 0
    max_attempts = target_pages * int(wiki_cfg["max_page_attempts"])
    while len(pages) < target_pages and attempts < max_attempts:
        attempts += 1
        page = fetch_random_page(session, int(wiki_cfg["min_page_characters"]), int(wiki_cfg["max_page_characters"]))
        if page is None:
            continue
        try:
            generated = generate_questions(client, page, len(pages))
            page_questions = []
            for item_index, item in enumerate(generated, start=1):
                answer, aliases = generate_answer(client, page, item, len(questions) + item_index)
                page_questions.append(to_question(page, item, answer, aliases, item_index))
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[wiki] rejected {page['title']!r}: {exc}")
            continue
        pages.append(page)
        questions.extend(page_questions)
        print(f"[wiki] accepted {len(pages)}/{target_pages}: {page['title']} (2 questions)")
    if len(pages) != target_pages:
        raise SystemExit(f"only accepted {len(pages)}/{target_pages} pages after {attempts} attempts")
    manifest = {
        "dataset": "wikipedia_random", "pages": len(pages), "questions": len(questions),
        "question_model": wiki_cfg["model_id"], "page_attempts": attempts,
        "selection": "Wikipedia API generator=random, namespace=0; exact plaintext stored below",
    }
    write_jsonl(sources_output, [{"_manifest": manifest}] + pages)
    write_jsonl(output, [{"_manifest": manifest}] + [q.to_json() for q in questions])
    print(f"[wiki] wrote {len(questions)} questions to {output}")
    print("[wiki:cost] " + json.dumps(client.ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
