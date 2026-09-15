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
   - [Experiment 5b — fictional fixed-capacity selection](#experiment-5b)
7. [Experiment 6 — multilingual fixed vs switching handoffs](#experiment-6)
8. [Experiment 8 — model heterogeneity across sequential handoffs](#experiment-8)
   - [Experiment 8b — fictional selector/relay bottleneck](#experiment-8b)
9. [Experiment 9 — handoff size adaptation](#experiment-9)
10. [Experiment 10 — communication regret under a hard budget](#experiment-10)
11. [Experiment 14 — compression mechanism: rewriting vs selection](#experiment-14)
12. [Cross-cutting wording inconsistency](#wording-inconsistency)

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

**Also drives:** the Qwen3-8B (`chain_qwen`) and Qwen3-32B (`chain_qwen32`)
replications, plus the generic /
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

**File:** [`src/run_summary_generalization.py`](src/run_summary_generalization.py).
The system prompt is always `CHAIN_SYSTEM`, imported unchanged from
`run_chain.py` (Experiment 2's question-conditioned set, above) — deliberately,
so no effect measured here can be blamed on different wording from the main
chain experiment. The two original arms (`conditioned`, `generic`) likewise
reuse `INITIAL_INSTRUCTION` / `RECOMPRESS_INSTRUCTION` verbatim and add **no
new prompt text at all**. Only the `paraphrase` arm introduces its own
instruction, below.

### What this experiment actually varies

`compression_user_prompt()` assembles every arm from the same template, so the
material block, ordering, separators, and the optional length directive are
byte-identical across arms. Exactly two things move: whether a question block
is present, and which instruction is used.

```
conditioned: {material}

             Question the final agent must answer: {question A}

             {instruction}

generic:     {material}

             {instruction}
```

| Arm | Question block | Instruction | Model call |
|---|---|---|---|
| `conditioned` | Question A | Experiment 2's question-conditioned set | yes |
| `generic` | none | same set, unchanged | yes |
| `paraphrase` | none | paraphrase-only set, below | yes |
| `passthrough` | none | none — input forwarded unchanged | **no** |

`prompt_difference_selftest()` asserts at runtime that (1) deleting the
question block from the conditioned prompt yields a byte-identical string to
the generic prompt, (2) no question-blind arm contains either question, and
(3) substituting the generic instruction back into any question-blind arm
reproduces the generic prompt exactly — so a `paraphrase`-vs-`generic`
contrast measures the instruction and nothing else.

### `paraphrase` — rewrite-without-compression instruction

The only new prompt text in this experiment. It exists to separate *rewriting*
from *compression and relevance filtering*: `conditioned` and `generic` both
summarise, so neither can tell whether repeated rewriting is itself lossy.

Stage 1 (`PARAPHRASE_INITIAL_INSTRUCTION`):

```
Rewrite the source material in your own words for another agent. Restate every
fact, qualifier, date, number, relationship, uncertainty, and source id it
contains. This is a rewrite, not a summary: do not condense, shorten, omit,
prioritise, or keep only what seems relevant, and do not add any fact that is
not already present.
```

Stage ≥2 (`PARAPHRASE_RECOMPRESS_INSTRUCTION`) is the same instruction over the
previous agent's notes, plus `Use only the previous notes.` — matching how the
question-conditioned set differs between its own stage-1 and stage-≥2 forms.

`validate_modes()` refuses to run this arm alongside `length_target_words`: a
word budget is a compression instruction and would contradict the arm's own
instruction, silently destroying the variable it exists to isolate.

`passthrough` has no prompt because it issues no model call — it forwards its
input unchanged (the raw context at stage 1). It is the zero-rewriting floor.

### Semantic-preservation judge — did a rewrite keep the meaning?

`add_preservation_judge()`, [`src/judge.py`](src/judge.py), scores each
consecutive handoff edge `M_i -> M_{i+1}`. It is the semantic counterpart to
the deterministic lexical measures in
[`src/paraphrase_metrics.py`](src/paraphrase_metrics.py), and follows the same
rules as the answer judge above: different model family from the system under
test, temperature 0, content-hashed cache.

System prompt:

```
You compare two research notes, where the second was written by rewriting the
first. You judge only whether the rewrite preserves the information in the
original. You never judge style, length, or writing quality.
```

User message: `Original notes:\n{M_i}\n\nRewritten notes:\n{M_i+1}\n\n` followed
by the instruction, which asks for exactly one of `EQUIVALENT` (every fact,
name, number, date, qualifier, relationship and uncertainty preserved despite
different wording), `MINOR_LOSS` (main content held, one detail dropped or
blurred), or `MAJOR_LOSS` (substantial omission, contradiction, or an added
fact). Scored 1.0 / 0.5 / 0.0. An unparsed verdict is left **unscored** rather
than defaulted, so a missing measurement never reads as preservation.

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

## Experiment 5b — fictional fixed-capacity selection<a id="experiment-5b"></a>

**Files:** [`src/run_fictional_summary_generalization.py`](src/run_fictional_summary_generalization.py)
and [`src/fictional_qa.py`](src/fictional_qa.py).

Unlike Experiment 5's prose summaries, this runner asks for exactly K opaque
card ids and renders their source text deterministically. Generic and
conditioned prompts differ only by the announced-question block.

System:

```
You select source-grounded evidence for a sealed handoff to another agent.
Obey the requested JSON schema exactly. Select only supplied evidence-card ids;
never write, merge, paraphrase, or invent evidence.
```

User template (`generic` omits the marked question block):

```
Source evidence cards:
[CARD {C-number}]
{source-grounded evidence text}

...

<ANNOUNCED_QUESTION>             # conditioned only
{Question A}                     # conditioned only
</ANNOUNCED_QUESTION>            # conditioned only

Select exactly {K} evidence cards for a fixed-capacity handoff.
If a question is provided, prioritize evidence useful for that question. If no question is
provided, preserve broadly reusable evidence. Do not infer or add facts.

Return one JSON object and nothing else, with exactly this key:
{"selected_fact_ids": [exactly {K} distinct bare ids such as "C03"]}
Copy only the C-number inside each CARD label; never include the word CARD in a value.
```

On a schema failure, the same prompt is retried with only this suffix:

```
FORMAT CORRECTION: the previous response was invalid because {parse error}.
Return exactly one JSON object with exactly {K} distinct valid ids and no other key.
```

The final answerer receives shared `ANSWER_SYSTEM` and a frozen handoff:

```
Research material:
{sealed.handoff_text}

Question:
{one announced or hidden evaluation query}
Answer:
```

Hidden queries never enter either selection prompt. Identical final-answer
requests are single-flighted by their full request hash, so repeated logical
roles cannot create different outputs for the same packet and question.

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

## Experiment 8 — model heterogeneity across sequential handoffs<a id="experiment-8"></a>

**This experiment adds no new prompt.** That is the point of it, so it is worth
stating plainly: every string below is imported from Experiment 2 through
Experiment 5's `conditioned` arm and asserted byte-identical at startup. The
only thing that varies between arms is *which model receives the string*.

| Element | Source | Note |
|---|---|---|
| System prompt | `run_chain.CHAIN_SYSTEM` | Same constant as Experiments 2, 5 and 6 |
| Stage-1 instruction | `run_chain.INITIAL_INSTRUCTION` | See [Experiment 2](#experiment-2) |
| Stage ≥2 instruction | `run_chain.RECOMPRESS_INSTRUCTION` | See [Experiment 2](#experiment-2) |
| User-message assembly | `run_summary_generalization.compression_user_prompt(pair, notes, "conditioned", stage, cfg)` | Reused as a function, not retyped |
| Final answer | `handoffs.ANSWER_SYSTEM` + Experiment 5's answer template | See [Final-answer persona](#final-answer-persona) |
| Answer judge | shared judge | See [LLM judge](#llm-judge--answer-correctness-vs-gold) |
| Semantic-preservation judge | `judge.add_preservation_judge` | See [Experiment 5](#experiment-5) |

### The one thing this experiment does add: a sealed rebuild of the stage ≥2 prompt

`compression_user_prompt` needs the whole pair dictionary, which would put
`render_context(pair)` — every source passage, gold and distractor — back inside
a stage-2 call frame. Experiment 8 therefore rebuilds the identical string from
a frozen `SealedHandoff` and nothing else:

```
Previous agent's notes:
{sealed.handoff_text}

Question the final agent must answer: {sealed.question}

{RECOMPRESS_INSTRUCTION}
```

`prompt_selftest()` asserts, on every invocation, that this equals
`compression_user_prompt(pair, notes, "conditioned", stage, cfg)` byte for byte,
and that passing anything other than a `SealedHandoff` raises `TypeError`. So the
experiment gains Experiment 2's structural isolation guarantee without drifting
from Experiment 5's published prompt.

### Model-blindness check

The same self-test asserts that neither the stage-1 nor the stage ≥2 prompt
contains any of `llama`, `qwen`, `mistral`, `ministral`, `gemma`, or
`parameter count`. No agent is told which model wrote its input, which model
will read its output, or how large any of them are. A model-composition effect
therefore cannot come from a model being *told* about the composition.

### Per-model generation controls

Two registry entries carry a model-native control, applied by `LLMClient` and
included in the cache key exactly as `qwen32_chain_config.yaml` already applies
it. These are properties of the decoder, not of the condition:

| Model | Control | Why |
|---|---|---|
| `qwen/qwen3-8b`, `qwen/qwen3-32b` | `reasoning: {effort: none}` **and** the system-prompt suffix `/no_think` | Qwen3 thinks by default and the gateway switch is not honoured by every routed provider. Without both, the shared 1,500-token budget is spent on hidden reasoning rather than on the visible handoff the experiment measures. |

This means the two Qwen entries receive one extra line in their system prompt
that the other six models do not. It is a necessary per-model control rather
than a design difference, but it is a real asymmetry and is listed as a confound
in the report.

---

## Experiment 8b — fictional selector/relay bottleneck<a id="experiment-8b"></a>

**Files:** [`src/run_fictional_model_bottleneck.py`](src/run_fictional_model_bottleneck.py)
and [`src/fictional_qa.py`](src/fictional_qa.py).

System for both selector stages:

```
You are an evidence-routing agent. Select only evidence-card ids supplied in the prompt.
Obey the requested number of slots exactly and return only the requested JSON object.
```

Stage 1 uses Experiment 5b's source-card template with `K=3` and includes
Question A. The future Question B is absent from this call. After sealing,
stage 2 uses:

```
Previous sealed handoff:
{three rendered packet slots}

<ANNOUNCED_QUESTION>
{Question B}
</ANNOUNCED_QUESTION>

Select exactly 1 evidence cards from the sealed handoff for the next
agent. Use only cards present in the sealed handoff; omitted source evidence is unavailable.
Return one JSON object and nothing else, with exactly this key:
{"selected_fact_ids": [exactly 1 distinct bare ids such as "C03"]}
Copy only the C-number inside each CARD label; never include the word CARD in a value.
```

The `restore` and `sham` arms change packet contents mechanically and add no
instruction. The `reopen` positive control uses the stage-1 source-card
template with all six source cards, Question B, and `K=1`; it is explicitly not
a sealed factorial arm. A malformed selection is retried with:

```
FORMAT CORRECTION: the previous response was invalid because {parse error}.
Return exactly {K} distinct bare C-number ids in the requested JSON object;
do not include the word CARD.
```

The fixed Mistral reader uses shared `ANSWER_SYSTEM` and
`handoffs.orchestrator_answer`: `Research notes from your subagent:\n{packet}`
followed by `Question: {Question B}\nAnswer:`. The answer judge is the shared
different-family judge documented above.

---

## Experiment 9 — handoff size adaptation<a id="experiment-9"></a>

This is the **only** experiment in the project that does not use Experiment 2's
handoff instruction verbatim, and the deviation is the point of the design
rather than an oversight.

### The base instruction: Experiment 2's, minus one word

Experiment 2's shared instruction opens *"Write **concise** prose research
notes …"*. That word is a size instruction. An experiment whose whole subject is
size cannot use it as a control: `resize_large` measured against it would be
"expand" versus "shrink", not "expand" versus "no size instruction", and
`resize_small` measured against it would compare a requested shrink to an
already-shrunk baseline.

The base is therefore the published string with `concise ` deleted — one
deletion, asserted rather than assumed. `run_size_adaptation._drop_concise`
raises unless the inherited string contains the substring exactly once, and both
the runner's startup check and the offline selftest reconstruct the published
string from the neutral one before anything else runs.

| Stage | Constant | Text |
|---|---|---|
| System | `run_chain.CHAIN_SYSTEM` *(unchanged)* | "You are a research handoff agent. Preserve every fact needed to answer the question. Your output will replace your entire input for the next agent, so omitted information is lost." |
| 1 | `NEUTRAL_INITIAL_INSTRUCTION` | "Write prose research notes that preserve all evidence needed to answer the question. Do not answer the question directly and do not add unsupported facts." |
| ≥2 | `NEUTRAL_RECOMPRESS_INSTRUCTION` | "Rewrite the previous agent's notes into prose research notes for another agent. Preserve every answer-relevant fact, qualifier, date, number, relationship, uncertainty, and source id. Use only the previous notes. Do not answer the question directly." |

The system prompt is untouched: it carries no size language, and it is shared
byte-identically with Experiments 2, 5, 6 and 8.

### The directives

Appended to the instruction above and to nothing else. Every arm's user message
is the control's user message plus its directive string — asserted for every
directive at both stage 1 and stage ≥2 in the offline selftest, so the
manipulation provably cannot leak into the material block.

| Directive | Appended text |
|---|---|
| `none` | *(empty string)* |
| `concise` | " Keep your handoff concise." |
| `inform_small` | " The next agent's model has a 2,000-token context window." |
| `resize_small` | " The next agent's model has a 2,000-token context window. Make your handoff fit within it." |
| `inform_large` | " The next agent's model has a 10,000-token context window." |
| `resize_large` | " The next agent's model has a 10,000-token context window. Expand your handoff to make use of the available space." |

These are **built compositionally, not written out**:
`directive_text()` constructs a `resize` string as its `inform` string plus one
clause from `REQUEST_CLAUSE`, so the two cannot drift apart in a later edit and
leave the inform-vs-resize contrast measuring an accidental rewording. Three
properties are asserted at startup:

1. `resize_X` starts with `inform_X`, and the remainder is exactly one of the
   two declared request clauses;
2. `inform_small` with its number substituted equals `inform_large` — the two
   differ only in the advertised figure;
3. `directive_text("none")` is the empty string, and the assembled control
   prompt contains none of *token*, *context window*, *concise*, *expand*,
   *shorter*, *longer*.

The advertised size is not connected to `decoding.handoff_max_tokens`. The
number claimed to the model and the budget enforced by the API are separate
variables; the enforced budget is identical and non-binding in every arm.

### Full user message

Stage 1 — same three-block layout as Experiment 5:

```
Source material:
{document}

Question the final agent must answer: {question}

{NEUTRAL_INITIAL_INSTRUCTION}{directive}
```

Stage ≥2 — built from a frozen `SealedHandoff` and nothing else, so the source
document is unreachable. This is load-bearing here in a way it is not elsewhere:
the experiment's central claim is that a fact reappearing after it vanished was
*reconstructed*, and that claim collapses if a later stage could have re-read the
document.

```
Previous agent's notes:
{sealed.handoff_text}

Question the final agent must answer: {sealed.question}

{NEUTRAL_RECOMPRESS_INSTRUCTION}{directive}
```

### Dataset-construction prompts

Run once by `src/build_size_adaptation_data.py`, never during the experiment.
All are issued to a strong writer model (`openai/gpt-5.2`, reasoning effort
high), never to the system under test — the same separation the Wikipedia and
counterfactual builders use.

| Prompt | Purpose | Validation applied to the reply |
|---|---|---|
| `WRITER_SYSTEM` + `fictional_prompt` | one invented-subject page per item, subject type cycled by index so variety is a property of the build, not of one sampling run | length in range; answer verbatim in the page; no meta-language about the task |
| `KNOWN_SUBJECT_PROMPT` | a passage about a famous subject plus a question the model is likely to answer from memory | answer verbatim in the passage |
| `COUNTERFACTUAL_REWRITE_PROMPT` | rewrite that passage so the answer becomes a different plausible value | replacement verbatim in the rewrite; **original value absent from the rewrite**; replacement and original not substrings of one another |
| `EXTRACT_SYSTEM` + `side_fact_prompt` | five reusable side facts per document | each answer verbatim in the document; none overlapping the target answer; none containing another |

### Closed-book knowledge probe

Reuses `data.CLOSED_BOOK_SYSTEM` and the C1 message shape, temperature and seed
ladder unchanged, so an item already probed by an earlier build hits the cache
rather than being re-paid for. See
[C1 parametric-leakage filter](#shared-across-experiments).

The difference from C1 is what is done with the samples. C1 asks one question —
*does the model already know this?* — and rejects on yes. Experiment 9 draws
**one** set of samples per item and judges it **twice**, against the original
gold set and against the replacement gold set, requiring:

- *knows the original* ≥ 2 of 3 samples — without this, a handoff that merely
  loses the fact is indistinguishable from one that reverted to memory;
- *knows the replacement* in 0 of 3 samples — the standard C1 condition.

Judging one sample set twice rather than drawing two is deliberate: two
independent draws could disagree with each other, and the two conditions would
then describe different behaviour rather than two readings of the same
behaviour.

### Claim-support judge

New in this experiment; lives in `judge.py` beside the answer and preservation
judges, and uses the same different-family model (`openai/gpt-4o-mini`) at
temperature 0.

System: "You check whether research notes stay within their source document. You
are given a source document and notes written from it. You report only claims the
document does not support."

The instruction defines rewording, summarising, reordering and omitting as never
unsupported, and asks for a count plus a short quote per unsupported claim in a
single parseable line. An unparseable reply is left **unscored** rather than
scored zero, so a judge failure cannot average in as a clean handoff.

It is scored against the **source document at every depth**, not against the
immediately preceding note. Grading stage 3 against stage 2 would bless a
fabrication that entered at stage 2 and was then copied faithfully, and no stage
of a chain is ever licensed to introduce material the document did not contain.

### Final answer and answer judge

`handoffs.ANSWER_SYSTEM` and Experiment 5's answer template, unchanged — see
[Final-answer persona](#final-answer-persona). Each prediction is scored twice
by the shared answer judge: once against the document's golds and once against
the memorised golds, from **one** generated answer, so "reverted to memory" can
never be a sampling artefact.

---

## Experiment 10 — communication regret under a hard budget<a id="experiment-10"></a>

**Files:** [`src/budget.py`](src/budget.py) (every sender-side string),
[`src/run_communication_regret.py`](src/run_communication_regret.py),
[`src/build_regret_data.py`](src/build_regret_data.py).

The sender prompt is assembled in one fixed order for every policy: source,
then information need, then the identical length contract and output form. Only
the information-need block varies among the four abstractive policies, and the
offline selftest asserts that replacing it with a placeholder leaves one
identical skeleton.

System ([`src/budget.py:53`](src/budget.py#L53), `SENDER_SYSTEM`):

```
You are a research subagent. You have read source material that the next agent
cannot see and will never see. Your handoff is that agent's ONLY information.
Anything you leave out is lost.
```

User template:

```
Source material:
{source}

{policy block — the only thing that differs across the four abstractive arms}

Length: write between {floor} and {cap} words. Your handoff is cut off at {cap}
words before the next agent sees it, so anything past the limit is lost, and a
handoff shorter than {floor} words wastes channel that is already paid for. Use
the space you have; do not pad it with filler and do not repeat yourself.
Write prose. Do not answer any question yourself.
```

### The policy blocks

`generic`:

```
The next agent will use your handoff to answer a question about this source material.
You have not been told which question it will be asked.
```

`conditioned`:

```
The next agent will use your handoff to answer this question:
  - {current question}
Communicate the information that answers that question.
```

`reusable`:

```
The next agent will use your handoff to answer this question:
  - {current question}
It may afterwards be asked further questions about this same source material that
you have not been told.
Communicate the information that answers the question above, and, within the same
limit, preserve other evidence that would still be useful for those later questions.
```

`oracle`:

```
The next agent will use your handoff to answer these questions:
  - {question 1}
  ...
  - {question k}
Communicate the information that answers those questions.
```

`generic` deliberately says only that it has not been told the question.
Mentioning that further questions may follow *is* the `reusable` treatment; the
control must not contain the independent variable.

### Extractive control

The two extractive arms reuse the `generic` / `conditioned` blocks verbatim and
replace only the final output-form line:

```
Copy sentences VERBATIM from the source material - character for character. Do
not paraphrase, merge, shorten or rewrite, and write nothing of your own.
Separate the sentences with a single space. Do not answer any question yourself.
```

### Length corrections

Both are byte-identical across policies, so an arm that overruns or under-fills
more often is not also handed a different instruction. Applied up to
`budget.max_length_corrections` times before deterministic truncation takes over.

Over the cap ([`src/budget.py:86`](src/budget.py#L86), `SHRINK_CORRECTION`):

```
LENGTH CORRECTION: your previous handoff was {actual} words, over the {cap}-word
limit. Rewrite it to between {floor} and {cap} words, keeping the information
that matters most. Return only the rewritten handoff.
```

Under the floor ([`src/budget.py:92`](src/budget.py#L92), `EXPAND_CORRECTION`):

```
LENGTH CORRECTION: your previous handoff was {actual} words, under the
{floor}-word minimum. Rewrite it to between {floor} and {cap} words, using the
extra room for more of what the source material contains rather than restating
what you have already written. Return only the rewritten handoff.
```

The expansion wording names no question and no future need — it asks only for
more of what the source already contains, which is the one neutral way to fill a
budget without inviting the padding failure mode Experiment 9 found.

### Answering

Shared `ANSWER_SYSTEM` from [`src/handoffs.py:24`](src/handoffs.py#L24), and a
frozen `SealedHandoff`:

```
Research material:
{sealed.handoff_text}

Question:
{one evaluated question, conditioning or hidden}
Answer:
```

The direct-context ceiling is the only call that receives `context.source` as
its material. Hidden questions never enter any sender prompt. Identical answer
requests are single-flighted by full request hash, which matters here because a
`generic` message is deliberately shared across every rotation of its context.

### Dataset build

Group independence audit ([`src/build_regret_data.py`](src/build_regret_data.py),
`AUDIT_TEMPLATE`), run by a different model family:

```
Passage:
{context}

Questions:
{numbered questions with reference answers}

For EACH numbered question decide two things about it, judged only against the
passage above.

  answerable: true only if the passage states the answer explicitly enough that a
              careful reader could give the reference answer shown beside the question.
  distinct:   true only if the question asks about a different fact from EVERY other
              question in the list. Two questions that have the same answer, or that
              are rewordings of each other, or where answering one necessarily gives
              away the other, are not distinct.

Return only a JSON object:
{"verdicts": [{"n": 1, "answerable": true, "distinct": true}, ...]}
```

The relation-dossier writer prompt (`RELATION_WRITER_PROMPT`) asks a strong
model for four sections, each stating one anchor fact about the main subject, a
second fact about that same subject, and a fact about a different named entity
introduced in the same section, plus a paraphrase question sharing the anchor's
answer. Every designed answer is then verified verbatim in the generated text
and the role structure is asserted before the corpus is written.

The C1 closed-book filter is the shared one at the top of this file, applied to
**every** question of a candidate context: a group dies if any of its questions
leaks.

---

## Experiment 14 — compression mechanism: rewriting vs selection<a id="experiment-14"></a>

**Files:** [`src/budget.py`](src/budget.py) (the `paraphrase` block),
[`src/elimination.py`](src/elimination.py) (the selectors — no prompts at all),
[`src/lm_unit_scores.py`](src/lm_unit_scores.py) (the scoring templates),
[`src/run_communication_regret.py`](src/run_communication_regret.py).

Experiment 14 reuses Experiment 10's channel byte-for-byte: the same
`SENDER_SYSTEM`, the same `LENGTH_CONTRACT`, the same `PROSE_FORM`, the same
reader prompt and the same judge. Only two new *prompts* exist in the whole
experiment, and four of its arms use no prompt whatsoever.

### The `paraphrase` information-need block

[`src/budget.py`](src/budget.py), `policy_block`. Assembled in exactly the same
fixed order as every other abstractive arm — source, information need, length
contract, output form — so the only difference from `generic` is the third
sentence:

```
The next agent will use your handoff to answer a question about this source material.
You have not been told which question it will be asked.
Restate the source material in your own words. Keep as much of it as fits in the
length below, in the order it appears, rather than choosing which parts matter.
```

The first two lines are `generic`'s, unchanged. That matters: the independent
variable is *rewriting with selection* versus *rewriting without deliberate
selection*, so the control may not differ in what it knows about the question.
The arm is rotation-invariant for the same reason `generic` is — it never sees a
question.

Under a 20–160-word cap on a 389–517-word dossier, "preserve everything" is not
achievable, and the instruction does not pretend otherwise ("as much of it as
fits"). The unbounded rewriting reference is Experiment 5's paraphrase run; the
unbounded no-compression reference is this run's own `direct_context` baseline.

### The LM selector's scoring templates

[`src/lm_unit_scores.py`](src/lm_unit_scores.py). These are **not** generation
prompts: nothing is sampled from them. They are teacher-forced inputs whose
token log-probabilities are the score, computed by a local pinned GPT-2 (the
compressor LM of LLMLingua/LongLLMLingua's small configuration), never by the
system under test.

Query-aware (`lm_conditioned`), scoring one evidence sentence against the
current question:

```
{unit}

We can get the answer to this question from the text above.
Question: {question}
```

Query-agnostic baseline for the same question, with the sentence removed and
everything else byte-identical:

```
We can get the answer to this question from the text above.
Question: {question}
```

The score is the difference of the two mean question-token log-probabilities —
LongLLMLingua's coarse document-relevance direction, expressed as a contrast so
it is comparable across dossiers:

```
score(unit, q) = (1/|q|) * [ sum_i log p(q_i | unit, q_<i) - sum_i log p(q_i | q_<i) ]
```

The restrictive statement ("We can get the answer…") is LongLLMLingua's, kept in
**both** templates so that the only difference between them is the presence of
the sentence being scored.

Query-agnostic (`lm_generic`) uses no question at all — mean per-token surprisal
of the sentence itself, BOS-prefixed:

```
score(unit) = -(1/T) * sum_t log p(x_t | x_<t)
```

Both LM arms use the same scorer, the same checkpoint and the same
tokenisation; only the score differs. Conditioned scores are computed **only**
for the four anchor questions that rotate into the conditioning role, so the
hidden questions never reach the model — `src/lm_unit_scores.py:check_leakage`
enforces that at write time and the offline selftest re-checks it at read time.

### The arms with no prompt

`nonllm_generic` (TF-IDF centrality), `nonllm_conditioned` (BM25 against the
current question), `random_selection` (a content-seeded shuffle) and
`passthrough` (the source prefix that fits) involve no model and therefore no
prompt. Their message is a concatenation of whole source
sentences, in source order, verified against the source before delivery
(`elimination.verify_verbatim`).

This is the difference from Experiment 10's `extractive_*` arms, which are a
*prompt* asking an LLM to copy sentences verbatim
([`src/budget.py`](src/budget.py), `EXTRACTIVE_FORM`) and which complied only
0–29% of the time on the SQuAD run. Experiment 14 does not reuse those arms or
their results.

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
