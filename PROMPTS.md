# Prompts reference

Every prompt string used anywhere in this pipeline, in one place. The actual
Python constants still live in their source files — moving them would mean
either a repo-wide import refactor or risking a byte-for-byte change to a
cache key (the on-disk cache is keyed on exact message content, so an
accidental whitespace change during a move would silently start paying for
calls that used to be free). This file is the map, not a new source of truth;
if a prompt changes, edit it at the file:line given and update this file to
match.

Every experiment number below refers to the numbering in
[results/HANDOFF_EXPERIMENTS_REPORT.md](results/HANDOFF_EXPERIMENTS_REPORT.md).

## Contents

1. [Shared across experiments](#shared-across-experiments)
2. [Experiment 1 — single-handoff mechanism pilot](#experiment-1)
3. [Experiment 2 — repeated handoff degradation](#experiment-2)
4. [Experiment 3 — retrieval quality through repeated handoffs](#experiment-3)
5. [Experiment 4 — fixed-context redundant-evidence signal ratio](#experiment-4)
6. [Experiment 5 — question conditioning and cross-question generalization](#experiment-5)
7. [Experiment 6 — multilingual fixed vs switching handoffs](#experiment-6)
8. [Cross-cutting wording inconsistency](#wording-inconsistency)

---

## Shared across experiments

### C1 parametric-leakage filter — closed-book check

**File:** [`src/data.py:21`](src/data.py#L21) (`CLOSED_BOOK_SYSTEM`), user template at
[`src/data.py:238`](src/data.py#L238)

```
System: You are answering a factual question from memory.
        Reply with only the short answer span - no explanation, no full sentence.

User:   Question: {question}
        Answer:
```

**Role:** run three times per candidate question at temperature 0.7, no
evidence supplied. A question is **discarded** if any of the three closed-book
attempts already answers it correctly (exact match or token-F1 ≥ 0.6,
`config.yaml: leakage_filter.f1_known_threshold`). This is the load-bearing
control described in `AGENTS.md`: if the model can already answer from
pretraining, corrupting or compressing the handoff can't hurt it, so that
question would measure nothing.

**Used by:** Experiment 1 (`src/run.py`) and Experiment 2 (`src/run_chain.py`
via `src/chain_data.py` → `data.apply_c1`), because both draw questions from
MuSiQue/HotpotQA through `data.py`'s `Question` type.

**Not used by:** Experiments 3, 4, 5. They sample from SQuAD / MS MARCO
through their own standalone construction code and do not run a closed-book
leakage check. This is a real asymmetry, not an oversight to paper over — it
means a nonzero fraction of those questions could in principle be answerable
from the model's parametric memory alone, which the C1-filtered experiments
are specifically designed to rule out.

### Final-answer persona

**File:** [`src/handoffs.py:24`](src/handoffs.py#L24) (`ANSWER_SYSTEM`)

```
You answer questions using only the material you are given.
Reply with only the short answer span - no explanation, no full sentence, no preamble.
If the material seems insufficient, still give your single best guess.
```

**Role:** the system prompt for every "final answer" call in the whole
codebase — whoever produces the score-relevant prediction, whether reading a
handoff or the raw context. Shared deliberately (comment at
[`handoffs.py:22`](src/handoffs.py#L22)): if `A_full` and a handoff mechanism's
answerer used different wording, a measured accuracy difference could be a
prompt artefact instead of a handoff effect.

**Used by:** `src/handoffs.py` (`A_full`, `orchestrator_answer`), and directly
or via `hm.ANSWER_SYSTEM` in `src/run_chain.py`, `src/run_redundant_signal_ratio.py`,
`src/run_retrieval_quality.py`, `src/run_summary_generalization.py`, and
`src/run_slack_retrieval.py`. `src/run_slack_facts.py` (`ANSWER_SYSTEM`,
`run_slack_facts.py:41`) defines a genuinely different one instead — its task
asks for multiple labelled facts per question (`L1: answer`, `L2: answer`, …),
which the shared single-span persona above can't express, so this is a
legitimate task-shape difference, not an accidental duplicate.

### LLM judge — answer correctness vs. gold

**File:** [`src/judge.py:24`](src/judge.py#L24) (`JUDGE_SYSTEM`), instruction at
[`src/judge.py:31`](src/judge.py#L31), user template built by
`judge_user_prompt()` at [`src/judge.py:42`](src/judge.py#L42)

```
System: You grade short answers to reading-comprehension questions. You are
        given a question, one or more reference answers, and a predicted
        answer. Decide whether the prediction conveys the same fact as any
        reference answer.

User:   Question: {question}

        Reference answer(s):
        - {gold 1}
        - {gold 2}
        ...

        Predicted answer: {prediction, or "(no answer given)" if empty}

        A prediction is CORRECT if it conveys the same fact as any reference
        answer, allowing for differences in wording, word order, abbreviation,
        casing, punctuation, added detail, or a fuller sentence around the
        same fact.
        A prediction is INCORRECT if it states a different fact, names a
        different entity, is empty, says the information is unavailable, or
        hedges without committing to an answer.
        Reply with exactly one word: CORRECT or INCORRECT.
```

**Role:** semantic-correctness metric, secondary to EM/token-F1 (never
replaces them — see `AGENTS.md`, "EM/F1 are primary; an LLM judge is a
secondary semantic metric"). Runs at temperature 0 on `openai/gpt-4o-mini`
— deliberately a **different model family** from every llama model under
test, to avoid a model grading its own family's outputs. Added 2026-08-20 to
replace BERTScore, which scored surface-form similarity rather than whether
the answer was actually right.

**Used by:** Experiment 2 (`run_chain.py`), Experiment 4
(`run_redundant_signal_ratio.py`), Experiment 5 (`run_summary_generalization.py`),
and Experiment 6 (`run_multilingual_handoffs.py`). Not yet wired into
Experiment 1 or Experiment 3.

---

## Experiment 1 — single-handoff mechanism pilot<a id="experiment-1"></a>

**File:** [`src/handoffs.py`](src/handoffs.py). Ran mechanisms: `A_full`,
`B_freeform`, `E_oracle`. `C_structured`/`D_extractive` are implemented and
offline-tested but were never run against the live API (`AGENTS.md`).

### Subagent persona (B/C/D)

[`src/handoffs.py:30`](src/handoffs.py#L30) (`SUBAGENT_SYSTEM`)

```
You are a research subagent. You have read source material that the
orchestrator cannot see and will never see. Your output is the orchestrator's
ONLY information for answering the question. Anything you leave out is lost.
```

**Role:** told to every subagent that writes a handoff in this experiment
(not the final answerer — that's `ANSWER_SYSTEM` above). States the isolation
guarantee in-prompt, matching the actual code-level guarantee (`seal()` /
`SealedHandoff`, `handoffs.py:144`) that the orchestrator can never see source
paragraphs.

### `B_freeform` — prose handoff instruction

[`src/handoffs.py:36`](src/handoffs.py#L36) (`FREEFORM_INSTRUCTION`)

```
Write a summary of what you found that will let the orchestrator answer the question.
Write prose. Do not answer the question yourself.
```

### `C_structured` — structured-claims handoff instruction

[`src/handoffs.py:41`](src/handoffs.py#L41) (`STRUCTURED_INSTRUCTION`)

```
Emit your findings as a single JSON object, and nothing else.

Schema:
{
  "claims": [
    {
      "text": "one self-contained factual claim, as a sentence",
      "source_para_id": <integer paragraph id the claim came from>,
      "constraints": ["qualifiers that make the claim precise: dates, numbers, roles, locations"],
      "confidence": "high" | "medium" | "low"
    }
  ],
  "unresolved": ["anything you could not establish from the source material"]
}

Include every claim the orchestrator needs, and no claim the source material does not support.
Do not answer the question yourself. Output only the JSON object.
```

**Role:** never run against the live API (see above) — implemented and
covered by the offline selftest only.

### `D_extractive` — verbatim-sentence handoff instruction

[`src/handoffs.py:59`](src/handoffs.py#L59) (`EXTRACTIVE_INSTRUCTION`)

```
Select the sentences from the source material that the orchestrator needs.

Copy each selected sentence VERBATIM - character for character from the
source. Do not paraphrase, merge, shorten or rewrite. Write nothing of your own.

Output a single JSON object and nothing else:
{"sentences": [{"source_para_id": <integer paragraph id>, "text": "<verbatim sentence>"}]}
```

**Role:** never run against the live API — implemented and offline-tested
only, same as `C_structured`.

### `E_oracle` — no prompt at all

`make_oracle()`, [`src/handoffs.py:304`](src/handoffs.py#L304), issues **zero**
LLM calls. It pulls the dataset's own gold-supporting sentences directly and
hands them to `ANSWER_SYSTEM`. Included here only so the mechanism inventory
is complete — there is no prompt to catalogue.

---

## Experiment 2 — repeated handoff degradation<a id="experiment-2"></a>

**File:** [`src/run_chain.py:31-57`](src/run_chain.py#L31). Two parallel prompt
sets — question-conditioned (every compressor sees the target question) and
generic (no compressor ever sees it) — chosen by `question_conditioned` in the
experiment's YAML config. A runtime self-test (`isolation_selftest()`,
`run_chain.py:165`) proves later-stage compressors can only ever receive a
`SealedHandoff`, never source documents.

### System prompt

Question-conditioned, [`run_chain.py:32`](src/run_chain.py#L32):
```
You are a research handoff agent. Preserve every fact needed to answer the question.
Your output will replace your entire input for the next agent, so omitted information is lost.
```

Generic, [`run_chain.py:36`](src/run_chain.py#L36):
```
You are a research handoff agent. Preserve the important factual content, including
qualifiers, dates, numbers, relationships, uncertainties, and source ids. Your output
will replace your entire input for the next agent, so omitted information is lost.
```

### Stage-1 instruction (compressing raw documents into the first summary)

Question-conditioned, [`run_chain.py:41`](src/run_chain.py#L41):
```
Write concise prose research notes that preserve all evidence needed to answer the question.
Do not answer the question directly and do not add unsupported facts.
```

Generic, [`run_chain.py:45`](src/run_chain.py#L45):
```
Write concise, general-purpose prose research notes. Preserve important factual content
without adding unsupported facts.
```

### Stage ≥2 instruction (recompressing the previous stage's notes)

Question-conditioned, [`run_chain.py:49`](src/run_chain.py#L49):
```
Rewrite the previous agent's notes into concise prose research notes for another agent.
Preserve every answer-relevant fact, qualifier, date, number, relationship, uncertainty, and source id.
Use only the previous notes. Do not answer the question directly.
```

Generic, [`run_chain.py:54`](src/run_chain.py#L54):
```
Rewrite the previous agent's notes into concise, general-purpose prose research notes for
another agent. Preserve important factual content, qualifiers, dates, numbers, relationships,
uncertainty, and source ids. Use only the previous notes.
```

### User-message assembly

`initial_compress()` / `recompress()`, [`run_chain.py:83`](src/run_chain.py#L83)
and [`run_chain.py:116`](src/run_chain.py#L116), splice in
`Dataset: {dataset}\nEvidence length: {variant}\n\nSource material:\n{context}`
(stage 1) or `Previous agent's notes:\n{handoff_text}` (stage ≥2), then
optionally `Question the final agent must answer: {question}` when
question-conditioned, then the instruction above.

### Final answer

Reuses shared `ANSWER_SYSTEM` (see above). User message:
`Dataset: {dataset}\nEvidence length: {variant}\n\nSource material:\n{context}\n\nQuestion: {question}\nAnswer:`
at depth 0 (`answer_context()`, `run_chain.py:145`), or the same shared
`orchestrator_answer()` template at depth ≥1.

**Also drives:** the Qwen3-8B replication (`chain_qwen`) and the generic /
question-omitted replication (`chain_generic`) — same prompt constants, just a
different `question_conditioned` flag and model config.

---

## Experiment 3 — retrieval quality through repeated handoffs<a id="experiment-3"></a>

**File:** [`src/run_retrieval_quality.py:32-36`](src/run_retrieval_quality.py#L32)

```
System: You are a research handoff agent. Preserve every fact needed to answer the
        question. Your notes replace the entire input for the next agent, so omitted
        information is lost. Do not answer the question directly.

Stage 1:   Write concise research notes from the passages. Preserve answer-relevant
           facts, qualifiers, numbers, dates, and relationships.

Stage ≥2:  Rewrite the prior notes concisely. Preserve every fact needed to answer
           the question, including qualifiers, numbers, dates, and relationships.
```

**Role:** MS MARCO QA v2.1, good/medium/bad passage-recall through repeated
compression. Same compression shape as Experiment 2 (system + stage-1 +
stage-≥2 instruction) but reimplemented locally rather than importing
`run_chain.py`'s constants — see [Wording inconsistency](#wording-inconsistency).
Final answer reuses the shared `ANSWER_SYSTEM` via `hm.ANSWER_SYSTEM`.

**Construction (2026-08-20 revision, no prompt text changed):** retrieval is
now real, not assumed. A self-contained Okapi BM25 index
(`BM25Index`, [`src/retrieval.py`](src/retrieval.py) — shared with Experiment 4)
is built over a bounded pool of MS MARCO passages. For each query:
- **usable gold** = passages MS MARCO's own `is_selected` label marks
  relevant **and** that this code's own BM25 ranking actually retrieves for
  that query in its own top-k — "top retrieved and relevant," not relevance
  judged in isolation.
- **hard negatives** = passages BM25 ranks in that same query's top-k but
  that are *not* usable gold — either the query's own non-relevant passages,
  or another query's passage that is lexically on-topic enough to rank
  highly. These replace the previous design's distractors, which were a
  passage sampled at random from an unrelated query (an "easy" negative with
  no lexical relationship to the query at all).

The good/medium/bad recall knob keeps the previous global-pool mechanic (a
single seeded shuffle-and-slice over every usable-gold passage pooled across
all queries, not a per-query fraction — a per-query fraction breaks down when
a query has only one findable gold passage, since `round(1 * 0.5) == 0` by
Python's banker's rounding while `bad`'s `max(1, ...)` floor stays at 1,
inverting the intended `good > medium > bad` ordering).

---

## Experiment 4 — fixed-context redundant-evidence signal ratio<a id="experiment-4"></a>

**File:** [`src/run_redundant_signal_ratio.py:31-35`](src/run_redundant_signal_ratio.py#L31)

```
System: You are a research handoff agent. Preserve every fact needed to answer the question.
        Your notes replace your entire input for the next agent, so omitted information is lost.
        Do not answer the question directly.

Stage 1:   Write concise research notes from the passages. Preserve answer-relevant facts,
           qualifiers, numbers, dates, and relationships.

Stage ≥2:  Rewrite the prior notes concisely. Preserve every fact needed to answer the
           question, including qualifiers, numbers, dates, and relationships.
```

**Role:** SQuAD, 10/5/1 answer-sufficient passages out of 10. Near-identical
in substance to Experiment 3's prompts, again a separate local copy rather
than a shared import — see [Wording inconsistency](#wording-inconsistency).
Final answer reuses shared `ANSWER_SYSTEM`.

**Construction (2026-08-20 revision, no prompt text changed):** the
non-relevant passages (10/5/1 → 0/5/9 distractor slots) now come from the
same BM25 hard-negative mechanism as Experiment 3, reusing `src/retrieval.py`.
An index is built once over every unique SQuAD paragraph (~2,000 in the
validation split — small enough to index whole, unlike Experiment 3's bounded
MS MARCO pool). For each base question, `hard_negative_ids` is that
question's own BM25 top-60, excluding same-article paragraphs and paragraphs
containing an answer alias — a real "retrieved but not relevant" negative,
replacing the previous design's passage sampled uniformly at random from any
unrelated article. Every pack is also fingerprinted by its actual passage
content (`content_fingerprint()`, order-independent hash) and the fingerprint
is included in the on-disk handoff/answer cache keys — the same qid can
otherwise carry different passages across construction runs (a BM25 top-k
change, a different seed, a code fix), and a cache keyed on qid alone let a
stale cached handoff for an old passage set silently answer a new, unrelated
passage set sharing the same qid. This is not hypothetical: it happened on
the first post-BM25 rerun of Experiment 3 before the fingerprint fix was
added — a cached "good"-condition handoff about Susan Rice answered a
freshly-constructed pack about gross rental income, both keyed under the same
qid from an earlier run. Both experiments fingerprint every pack now.

---

## Experiment 5 — question conditioning and cross-question generalization<a id="experiment-5"></a>

**File:** [`src/run_summary_generalization.py`](src/run_summary_generalization.py)
defines **no new prompt text of its own**. It imports `CHAIN_SYSTEM`,
`INITIAL_INSTRUCTION`, `RECOMPRESS_INSTRUCTION` directly from `run_chain.py`
(Experiment 2's question-conditioned set, above) — deliberately, so any
conditioning effect measured here can't be blamed on different wording from
the main chain experiment.

### What this experiment actually varies

`compression_user_prompt()`, [`run_summary_generalization.py:253`](src/run_summary_generalization.py#L253):
only whether a question block is spliced into the user message.

```
conditioned: {material}

             Question the final agent must answer: {question A}

             {instruction}

generic:     {material}

             {instruction}
```

A runtime self-test (`prompt_difference_selftest()`,
[`run_summary_generalization.py:269`](src/run_summary_generalization.py#L269))
asserts that stripping that one block out of the conditioned prompt produces a
byte-identical string to the generic prompt.

### Final answer

`answer()`, [`run_summary_generalization.py:306`](src/run_summary_generalization.py#L306),
reuses shared `ANSWER_SYSTEM`. User message:
`Research material:\n{material}\n\nQuestion:\n{question}\nAnswer:`

### Dataset construction

The only supported path is `load_prebuilt_pairs()`, using
`summary_generalization_squad_pairs_config.yaml`. The dataset is built and
validated ahead of time by
[`src/build_squad_same_passage.py`](src/build_squad_same_passage.py). Each
example takes two native SQuAD questions from one shared SQuAD passage, placed
at a stratified slot among nine length-matched SQuAD passages. Both questions
are independently audited for salience/independence and pass the closed-book
C1 filter; every distractor is separately screened as irrelevant to both A and
B. The obsolete constructors and configurations that joined questions from
different passages have been removed.

---

## Experiment 6 — multilingual fixed vs switching handoffs<a id="experiment-6"></a>

**File:** [`src/run_multilingual_handoffs.py`](src/run_multilingual_handoffs.py).
This experiment imports `CHAIN_SYSTEM`, `INITIAL_INSTRUCTION`, and
`RECOMPRESS_INSTRUCTION` from Experiment 2. It crosses the presence/absence of
Question A with a fixed-language/switching-language schedule. The same exact
language directive is appended in every arm. Before prompt assembly, the
runner projects each reusable 1+9 SQuAD pack to its single `gold_AB` passage;
no distractor text reaches any Experiment 6 model call.

### System prompt

The compressor reuses `CHAIN_SYSTEM` verbatim:

```
You are a research handoff agent. Preserve every fact needed to answer the
question. Your output will replace your entire input for the next agent, so
omitted information is lost.
```

### Stage-1 user-message template

```
Source material:
{the single shared SQuAD gold passage that supports A and B; 0 distractors}

[conditioned arm only]
Question the final agent must answer: {Question A}

Write concise prose research notes that preserve all evidence needed to answer
the question. Do not answer the question directly and do not add unsupported
facts.

{LANGUAGE_DIRECTIVE}
```

### Stage ≥2 user-message template

```
Previous agent's notes:
{previous handoff only}

[conditioned arm only]
Question the final agent must answer: {Question A}

Rewrite the previous agent's notes into concise prose research notes for
another agent. Preserve every answer-relevant fact, qualifier, date, number,
relationship, uncertainty, and source id. Use only the previous notes. Do not
answer the question directly.

{LANGUAGE_DIRECTIVE}
```

### Language directive

`LANGUAGE_DIRECTIVE` is formatted with the language assigned to that pair and
stage:

```
OUTPUT LANGUAGE (mandatory): {language}. Write the entire replacement handoff
in {language}. Proper names, identifiers, numbers, and short source quotations
may remain unchanged when translation would alter them. Do not mix in another
language for the prose. Summarize; do not translate or rewrite the source
passage by passage. Do not use headings or one section per passage.
```

This directive deliberately contains no sentence, word, paragraph, or other
output-length target. The 1,500-token API limit in the configuration is only a
non-binding runaway guard and is not part of the model prompt.

At stage 1, the fixed and switching schedules receive the same assigned
starting language and byte-identical prompts; the stored stage-1 handoff is
copied between schedules. At later stages, fixed keeps that language while
switching advances cyclically through English, German, French, Italian,
Portuguese, and Spanish. Starting language is stratified by passage.

### Final answer

The shared `ANSWER_SYSTEM` is used with the original English question:

```
Research material:
{direct English context at depth 0, or the handoff at the requested depth}

Question:
{Question A or Question B}
Answer:
```

### Language-compliance judge

The saved audit includes both deterministic predominant-language detection and
the following different-family LLM audit. The deterministic detector is the
reported primary compliance metric because the LLM audit was overly
conservative on clearly correct French prose.

System:

```
You are a strict language-identification auditor. Reply with exactly MATCH or
MISMATCH.
```

User:

```
Expected language: {requested_language}

Text:
{handoff_text}

Reply MATCH if the prose is predominantly in the expected language. Allow
proper names, numbers, identifiers, and short untranslated quotations.
Otherwise reply MISMATCH.
```

---

## Cross-cutting wording inconsistency<a id="wording-inconsistency"></a>

Three files independently define a handoff system prompt that says the same
thing with slightly different wording, instead of importing one shared
constant:

| File | Wording |
|---|---|
| `run_redundant_signal_ratio.py:31` (Exp. 4) | "...replace **your** entire input for the next agent..." |
| `run_retrieval_quality.py:26` (Exp. 3) | "...replace **the** entire input for the next agent..." |
| `run_slack_retrieval.py:43` *(not part of the numbered report)* | "...replace **the** entire input for the next agent..." (matches retrieval_quality) |

This is a cosmetic difference (a pronoun), not a different instruction, and it
does not affect any reported result — but it means Experiments 3 and 4 are
not running byte-identical compressor prompts, unlike Experiment 5's
deliberately-shared, self-test-verified prompt with Experiment 2. If a future
experiment needs to claim two conditions differ *only* in one specific
variable, copy Experiment 5's pattern (import the shared constant, add a
byte-identity self-test) rather than hand-retyping the prompt.
