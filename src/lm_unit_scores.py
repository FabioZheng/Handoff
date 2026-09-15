"""Precompute the LM importance scores that Experiment 14's `lm_*` arms select on.

Why this is a separate script and a stored artefact, rather than a function the
runner calls:

* It is the only part of the experiment that needs a local model. Keeping it out
  of the run path means the experiment itself still runs on the frozen
  dependency set every other experiment here was run against, and `torch` never
  becomes an import-time requirement of the runner.
* A selector is only interesting if you can see what it scored. The artefact
  carries one row per (unit) and per (unit, conditioning question) with the raw
  log-probabilities, so the ranking can be re-derived and argued with without
  re-running a model.
* It pins the thing that would otherwise drift. The manifest records the model
  id, the resolved commit hash, the torch/transformers versions, the exact
  prompt templates and the scoring formulas, plus a content hash of the corpus
  the scores were computed against. The runner refuses scores whose corpus hash
  does not match the dossiers it is about to compress.

The scorer is GPT-2 small: it is what LLMLingua and LongLLMLingua themselves use
as the compressor LM in their small-model configuration, it is deterministic on
CPU, and it is emphatically not the system under test (the sender/reader is
Llama-3.1-8B). Using a strong instruct model here would confound "an LM can
score relevance" with "a large instruction-tuned model can".

Two scores, one model, as the design requires:

``generic`` (query-agnostic, LLMLingua / Selective-Context direction)
    mean per-token surprisal of the sentence itself,
    ``-(1/T) * sum_t log p(x_t | x_<t)``.
    High = lexically surprising = carries information the LM cannot predict.

``conditioned`` (query-aware, LongLLMLingua coarse direction)
    the *contrast* between the question's likelihood with and without the
    sentence in front of it,
    ``(1/|q|) * [ sum_i log p(q_i | unit, q_<i) - sum_i log p(q_i | q_<i) ]``.
    High = this sentence makes the current question easy to state, which is
    LongLLMLingua's document-level relevance signal.

Both are computed by teacher forcing, never by generation, so nothing here can
be influenced by sampling.

Leakage: conditioned scores are computed ONLY for the questions that rotate into
the conditioning role (`role == anchor`). The hidden questions of a dossier are
never passed to the model, and `--check-leakage` re-verifies that no non-anchor
question text reached the artefact.

    python src/lm_unit_scores.py                # write data/compression_mechanism/lm_unit_scores.jsonl
    python src/lm_unit_scores.py --force        # recompute even if the manifest matches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anticipatory_data as ad  # noqa: E402
import regret_data as rd  # noqa: E402

SCHEMA_VERSION = "lm-unit-scores-v1"

DEFAULT_CORPUS = "data/communication_regret/relation_dossiers.jsonl"
DEFAULT_OUT = "data/compression_mechanism/lm_unit_scores.jsonl"
DEFAULT_MODEL = "gpt2"

# LongLLMLingua's restrictive statement, kept byte-identical in both the
# conditioned and the unconditioned prompt so the ONLY difference between them
# is the presence of the sentence being scored.
RESTRICTIVE = "We can get the answer to this question from the text above."
COND_TEMPLATE = "{unit}\n\n" + RESTRICTIVE + "\nQuestion: {question}"
UNCOND_TEMPLATE = RESTRICTIVE + "\nQuestion: {question}"

GENERIC_FORMULA = "mean_token_surprisal = -(1/T) * sum_t log p(x_t | x_<t), BOS-prefixed"
CONDITIONED_FORMULA = (
    "mean_question_logprob_gain = (1/|q|) * [ sum_i log p(q_i | unit, q_<i) "
    "- sum_i log p(q_i | q_<i) ]"
)


def corpus_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class Scorer:
    """Teacher-forced log-probabilities from a pinned local causal LM."""

    def __init__(self, model_id: str, revision: str | None = None):
        import torch  # noqa: PLC0415 - deliberately lazy: the runner must not need torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.model_id = model_id
        self.commit = str(getattr(self.model.config, "_commit_hash", "") or revision or "")
        self.bos = self.tokenizer.bos_token_id
        if self.bos is None:
            self.bos = self.tokenizer.eos_token_id

    # -- primitives ---------------------------------------------------------

    def _logprobs(self, prefix_ids: list[int], target_ids: list[int]) -> float:
        """Sum of log p(target | prefix), teacher forced. Never generates."""
        if not target_ids:
            raise ValueError("nothing to score")
        torch = self.torch
        ids = torch.tensor([prefix_ids + target_ids], dtype=torch.long)
        logits = self.model(ids).logits.float()
        # Position i predicts token i+1, so the targets start at len(prefix)-1.
        start = len(prefix_ids) - 1
        window = logits[0, start:-1, :]
        log_probs = torch.log_softmax(window, dim=-1)
        targets = torch.tensor(target_ids, dtype=torch.long)
        picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(picked.sum().item())

    def self_information(self, text: str) -> tuple[float, int]:
        """Mean per-token surprisal of ``text`` standing alone."""
        ids = self.tokenizer.encode(text)
        if not ids:
            return 0.0, 0
        total = self._logprobs([self.bos], ids)
        return -total / len(ids), len(ids)

    def question_logprob(self, question: str, unit: str | None) -> tuple[float, int]:
        """Mean log p(question tokens), with or without ``unit`` in front."""
        prompt = (COND_TEMPLATE.format(unit=unit, question=question) if unit is not None
                  else UNCOND_TEMPLATE.format(question=question))
        marker = "Question: "
        head, _, tail = prompt.rpartition(marker)
        prefix_ids = self.tokenizer.encode(head + marker)
        target_ids = self.tokenizer.encode(tail)
        if not target_ids:
            raise ValueError(f"empty question tokens for {question!r}")
        total = self._logprobs([self.bos] + prefix_ids, target_ids)
        return total / len(target_ids), len(target_ids)


def build_rows(scorer: Scorer, dossiers: list[dict], contexts) -> list[dict]:
    rows: list[dict] = []
    by_id = {c.context_id: c for c in contexts}
    started = time.time()
    for n, dossier in enumerate(dossiers, 1):
        context = by_id[dossier["context_id"]]
        units = ad.extract_units(dossier)
        for unit in units:
            score, n_tokens = scorer.self_information(unit.text)
            rows.append({"kind": "generic", "context_id": context.context_id,
                         "unit_id": unit.unit_id, "score": score,
                         "unit_tokens": n_tokens})
        anchors = [q for q in context.questions if q.role == rd.ANCHOR]
        for q in anchors:
            base, q_tokens = scorer.question_logprob(q.question, None)
            rows.append({"kind": "unconditioned", "context_id": context.context_id,
                         "qid": q.qid, "logp_uncond": base, "question_tokens": q_tokens})
            for unit in units:
                cond, _ = scorer.question_logprob(q.question, unit.text)
                rows.append({"kind": "conditioned", "context_id": context.context_id,
                             "qid": q.qid, "unit_id": unit.unit_id,
                             "score": cond - base, "logp_cond": cond, "logp_uncond": base})
        print(f"[lm-scores] {n}/{len(dossiers)} {context.context_id} "
              f"({len(units)} units x {len(anchors)} anchors) "
              f"{time.time() - started:.0f}s", flush=True)
    return rows


def check_leakage(rows: list[dict], contexts) -> None:
    """No non-anchor question may appear anywhere in the artefact."""
    anchors = {(c.context_id, q.qid) for c in contexts for q in c.questions
               if q.role == rd.ANCHOR}
    seen = {(r["context_id"], r["qid"]) for r in rows if r.get("qid")}
    intruders = sorted(seen - anchors)
    if intruders:
        raise AssertionError(f"non-anchor questions reached the scorer: {intruders[:3]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="pin a hub commit")
    parser.add_argument("--limit", type=int, help="score only the first N dossiers")
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args(argv)

    corpus = ROOT / args.corpus
    out = ROOT / args.out
    digest = corpus_hash(corpus)

    if out.exists() and not args.force:
        first = out.read_text(encoding="utf-8").split("\n", 1)[0]
        old = json.loads(first).get("_manifest", {})
        if (old.get("corpus_hash") == digest and old.get("model_id") == args.model
                and old.get("schema_version") == SCHEMA_VERSION
                and old.get("cond_template") == COND_TEMPLATE
                and old.get("limit") == args.limit):
            print(f"[lm-scores] {out.relative_to(ROOT)} is current; --force to recompute")
            return 0

    _, contexts = rd.load_contexts(corpus)
    _, dossiers = ad.load_dossiers(corpus)
    if args.limit:
        dossiers = dossiers[:args.limit]
        keep = {d["context_id"] for d in dossiers}
        contexts = tuple(c for c in contexts if c.context_id in keep)

    scorer = Scorer(args.model, args.revision)
    rows = build_rows(scorer, dossiers, contexts)
    check_leakage(rows, contexts)

    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    manifest = {
        "_manifest": {
            "schema_version": SCHEMA_VERSION,
            "model_id": args.model,
            "revision": args.revision or "",
            "commit_hash": scorer.commit,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "decoding": "teacher-forced log-probabilities; no generation, no sampling",
            "cond_template": COND_TEMPLATE,
            "uncond_template": UNCOND_TEMPLATE,
            "generic_formula": GENERIC_FORMULA,
            "conditioned_formula": CONDITIONED_FORMULA,
            "corpus": args.corpus,
            "corpus_hash": digest,
            "limit": args.limit,
            "n_contexts": len(contexts),
            "n_rows": len(rows),
        }
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[lm-scores] wrote {len(rows)} rows to {out.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Reading side, used by the runner (no torch)


def load_scores(path: str | Path, expect_corpus_hash: str | None = None) -> tuple[dict, dict]:
    """Return (manifest, index) where index holds the two score tables.

    ``index['generic'][unit_id]`` and ``index['conditioned'][(qid, unit_id)]``.
    A corpus-hash mismatch is fatal: silently selecting on scores computed for
    different source text is exactly the kind of stale-cache error the rest of
    this project spends effort making impossible.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run: python src/lm_unit_scores.py")
    manifest: dict = {}
    generic: dict[str, float] = {}
    conditioned: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_manifest" in row:
                manifest = row["_manifest"]
                continue
            if row["kind"] == "generic":
                generic[row["unit_id"]] = float(row["score"])
            elif row["kind"] == "conditioned":
                conditioned[(row["qid"], row["unit_id"])] = float(row["score"])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} has schema {manifest.get('schema_version')!r}, "
                         f"expected {SCHEMA_VERSION!r}")
    if expect_corpus_hash and manifest.get("corpus_hash") != expect_corpus_hash:
        raise ValueError(
            f"LM scores were computed against corpus {manifest.get('corpus_hash')!r} but the "
            f"run is using {expect_corpus_hash!r}; rerun src/lm_unit_scores.py")
    return manifest, {"generic": generic, "conditioned": conditioned}


if __name__ == "__main__":
    raise SystemExit(main())
