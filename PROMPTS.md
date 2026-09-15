# Prompts reference

This file collects every prompt used anywhere in the pipeline. The prompts themselves still
live as Python constants in their source files, and each section gives the file and line. They
stay there because the on-disk cache is keyed on the exact message text: moving a constant and
accidentally changing a space would turn free cached calls into paid ones. Treat this file as a
map, not a second source of truth. If you change a prompt, change it at the file and line given
here and then update the copy in this file.

Experiment numbers follow
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

**File:** [`src/data.py:21`](src/data.py#L21) (`CLOSED_BOOK_SYSTEM`), with the user template at
[`src/data.py:238`](src/data.py#L238)

```
System: You are answering a factual question from memory.
        Reply with only the short answer span - no explanation, no full sentence.

User:   Question: {question}
        Answer:
```

**Role:** each candidate question is answered three times at temperature 0.7, with no evidence.
If any of the three answers is already correct (exact match, or token F1 ≥ 0.6 from
`config.yaml: leakage_filter.f1_known_threshold`), the question is **dropped**. The reason: if
the model can answer from pretraining, compressing or corrupting the handoff cannot lower its
score, so the question would tell us nothing about the handoff.

**Used by:** Experiment 1 (`src/run.py`) and Experiment 2 (`src/run_chain.py`, through
`src/chain_data.py` → `data.apply_c1`). Both draw MuSiQue and HotpotQA questions through the
`Question` type in `data.py`.

**Not used by:** Experiments 3 and 4, or the original Experiment 5 construction. They build their
SQuAD and MS MARCO samples with their own code and skip the closed-book check, so some of their
questions could in principle be answered from memory alone, which is exactly what the filter
rules out elsewhere. This is a real difference between the experiments, not something to gloss
over. (Experiment 5's current prebuilt pairs do pass C1; see its
[dataset construction](#dataset-construction).)

### Final-answer persona

**File:** [`src/handoffs.py:24`](src/handoffs.py#L24) (`ANSWER_SYSTEM`)

```
You answer questions using only the material you are given.
Reply with only the short answer span - no explanation, no full sentence, no preamble.
If the material seems insufficient, still give your single best guess.
```

**Role:** the system prompt for every call in the codebase that produces a scored answer,
whether the model is reading a handoff or the raw context. It is shared on purpose (see the
comment at [`handoffs.py:22`](src/handoffs.py#L22)): if `A_full` and a handoff's answerer were
worded differently, an accuracy gap could come from the wording rather than from the handoff.

**Used by:** `src/handoffs.py` (`A_full`, `orchestrator_answer`), and, directly or through
`hm.ANSWER_SYSTEM`, `src/run_chain.py`, `src/run_redundant_signal_ratio.py`,
`src/run_retrieval_quality.py`, `src/run_summary_generalization.py` and
`src/run_slack_retrieval.py`. `src/run_slack_facts.py` defines its own `ANSWER_SYSTEM`
(`run_slack_facts.py:41`) because its task asks for several labelled facts per question
(`L1: answer`, `L2: answer`, …), which the single-span prompt above cannot ask for. That is a
different task format, not an accidental copy.

### LLM judge — answer correctness vs. gold

**File:** [`src/judge.py:24`](src/judge.py#L24) (`JUDGE_SYSTEM`), with the instruction at
[`src/judge.py:31`](src/judge.py#L31) and the user template built by `judge_user_prompt()` at
[`src/judge.py:42`](src/judge.py#L42)

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

**Role:** a semantic correctness score. It is secondary: exact match and token F1 remain the
primary metrics and are always reported next to it, never replaced by it. It runs at temperature
0 on `openai/gpt-4o-mini`, which comes from a **different model family** than the Llama models
under test, so no model grades answers from its own family. It was added on 2026-08-20 to
replace BERTScore, which measured surface similarity rather than whether the answer was right.

**Used by:** Experiment 2 (`run_chain.py`), Experiment 3 (`run_retrieval_quality.py`),
Experiment 4 (`run_redundant_signal_ratio.py`), Experiment 5 (`run_summary_generalization.py`),
Experiment 6 (`run_multilingual_handoffs.py`), and the later experiments described below.
Experiment 1 does not use it.

---

## Experiment 1 — single-handoff mechanism pilot<a id="experiment-1"></a>

**File:** [`src/handoffs.py`](src/handoffs.py). The pilot ran `A_full`, `B_freeform` and
`E_oracle`. `C_structured` and `D_extractive` are implemented and covered by the offline
selftest, but were never run against the live API.

### Subagent persona (B/C/D)

[`src/handoffs.py:30`](src/handoffs.py#L30) (`SUBAGENT_SYSTEM`)

```
You are a research subagent. You have read source material that the
orchestrator cannot see and will never see. Your output is the orchestrator's
ONLY information for answering the question. Anything you leave out is lost.
```

**Role:** the system prompt for every subagent that writes a handoff in this experiment. The
final answerer uses `ANSWER_SYSTEM` instead. The prompt tells the model what the code enforces:
the orchestrator never sees the source paragraphs (`seal()` / `SealedHandoff`,
`handoffs.py:144`).

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

**Role:** never run against the live API. It is implemented and covered only by the offline
selftest.

### `D_extractive` — verbatim-sentence handoff instruction

[`src/handoffs.py:59`](src/handoffs.py#L59) (`EXTRACTIVE_INSTRUCTION`)

```
Select the sentences from the source material that the orchestrator needs.

Copy each selected sentence VERBATIM - character for character from the
source. Do not paraphrase, merge, shorten or rewrite. Write nothing of your own.

Output a single JSON object and nothing else:
{"sentences": [{"source_para_id": <integer paragraph id>, "text": "<verbatim sentence>"}]}
```

**Role:** like `C_structured`, implemented and tested offline but never run against the live API.

### `E_oracle` — no prompt at all

`make_oracle()`, [`src/handoffs.py:304`](src/handoffs.py#L304), makes **no** LLM calls. It takes
the dataset's own gold supporting sentences and passes them straight to the answerer with
`ANSWER_SYSTEM`. It appears here only so the list of mechanisms is complete; there is no prompt
to show.

---

## Experiment 2 — repeated handoff degradation<a id="experiment-2"></a>

**File:** [`src/run_chain.py:31-57`](src/run_chain.py#L31). There are two prompt sets. In the
question-conditioned set every compressor sees the target question; in the generic set none of
them does. The `question_conditioned` flag in the experiment's YAML config chooses between them.
A self-test that runs with the experiment (`isolation_selftest()`, `run_chain.py:165`) shows
that compressors after stage 1 can only ever receive a `SealedHandoff`, never the source
documents.

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

`initial_compress()` and `recompress()`, at [`run_chain.py:83`](src/run_chain.py#L83) and
[`run_chain.py:116`](src/run_chain.py#L116), build the user message in this order: the source
block `Dataset: {dataset}\nEvidence length: {variant}\n\nSource material:\n{context}` at stage 1,
or `Previous agent's notes:\n{handoff_text}` at later stages; then, in the question-conditioned
set only, `Question the final agent must answer: {question}`; and finally the instruction above.

### Final answer

Uses the shared `ANSWER_SYSTEM` (see above). At depth 0 the user message is
`Dataset: {dataset}\nEvidence length: {variant}\n\nSource material:\n{context}\n\nQuestion: {question}\nAnswer:`
(`answer_context()`, `run_chain.py:145`). At depth 1 and deeper it is the shared
`orchestrator_answer()` template.

**Also drives:** the Qwen3-8B (`chain_qwen`) and Qwen3-32B (`chain_qwen32`) replications, and
the replication in which compressors never see the question (`chain_generic`). They use the same
prompt constants, with a different `question_conditioned` flag and model config.

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

**Role:** MS MARCO QA v2.1 contexts with good, medium or bad passage recall, passed through
repeated compression. The structure matches Experiment 2 (a system prompt, a stage-1 instruction
and a stage ≥2 instruction), but the prompts are written out again in this file instead of being
imported from `run_chain.py`; see [Wording inconsistency](#wording-inconsistency). The final
answer uses the shared `ANSWER_SYSTEM` through `hm.ANSWER_SYSTEM`.

**Construction (revised 2026-08-20; the prompt text did not change).** Retrieval is now real
rather than assumed. A self-contained Okapi BM25 index (`BM25Index` in
[`src/retrieval.py`](src/retrieval.py), shared with Experiment 4) is built over a bounded pool of
MS MARCO passages. For each query:
- **Usable gold** passages are those that MS MARCO's own `is_selected` label marks as relevant
  **and** that this code's BM25 ranking actually returns in its top-k for that query. A passage
  has to be both relevant and retrieved; relevance on its own is not enough.
- **Hard negatives** are passages that BM25 ranks in that same query's top-k but that are not
  usable gold. They are either the query's own non-relevant passages or another query's passage
  that is close enough in wording to rank highly. They replace the earlier distractors, which
  were passages sampled at random from an unrelated query: easy negatives with no lexical link to
  the query at all.

The good/medium/bad recall setting still uses the original global pool: one seeded
shuffle-and-slice over all usable gold passages pooled across queries, rather than a fraction per
query. A per-query fraction fails when a query has only one findable gold passage. Python's
banker's rounding makes `round(1 * 0.5) == 0`, while `bad` keeps a floor of `max(1, ...)` = 1, so
the intended `good > medium > bad` order would be reversed.

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

**Role:** SQuAD contexts in which 10, 5 or 1 of the 10 passages are enough to answer the
question. The prompts say essentially the same thing as Experiment 3's, but again they are a
separate local copy rather than a shared import; see
[Wording inconsistency](#wording-inconsistency). The final answer uses the shared
`ANSWER_SYSTEM`.

**Construction (revised 2026-08-20; the prompt text did not change).** The non-relevant passages
(0, 5 or 9 distractor slots for the 10/5/1 conditions) now come from the same BM25 hard-negative
method as Experiment 3, reusing `src/retrieval.py`. One index is built over every unique SQuAD
paragraph (about 2,000 in the validation split, small enough to index in full, unlike Experiment
3's bounded MS MARCO pool). For each base question, `hard_negative_ids` is that question's BM25
top 60, excluding paragraphs from the same article and paragraphs that contain an answer alias.
These are real "retrieved but not relevant" negatives; the earlier design sampled a paragraph
uniformly at random from any unrelated article.

Every pack is also fingerprinted by its passage content (`content_fingerprint()`, an
order-independent hash), and the fingerprint is part of the on-disk handoff and answer cache
keys. Without it, the same qid can carry different passages across construction runs (after a
change in the BM25 top-k, a different seed or a code fix), and a cache keyed on qid alone would
let a stale handoff for an old passage set answer a new, unrelated one. This really happened on
the first rerun of Experiment 3 after the switch to BM25, before the fingerprint was added: a
cached "good"-condition handoff about Susan Rice answered a newly built pack about gross rental
income, because both had the same qid from an earlier run. Both experiments now fingerprint
every pack.

---

## Experiment 5 — question conditioning and cross-question generalization<a id="experiment-5"></a>

**File:** [`src/run_summary_generalization.py`](src/run_summary_generalization.py). The system
prompt is always `CHAIN_SYSTEM`, imported unchanged from `run_chain.py` (Experiment 2's
question-conditioned set, above). This is deliberate: no effect measured here can be blamed on
wording that differs from the main chain experiment. The two original arms, `conditioned` and
`generic`, also reuse `INITIAL_INSTRUCTION` and `RECOMPRESS_INSTRUCTION` verbatim and add **no new
prompt text**. Only the `paraphrase` arm has its own instruction, shown below.

### What this experiment actually varies

`compression_user_prompt()` builds every arm from the same template, so the material block, the
ordering, the separators and the optional length directive are identical across arms. Only two
things change: whether a question block is present, and which instruction is used.

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
| `generic` | none | the same set, unchanged | yes |
| `paraphrase` | none | the paraphrase-only set, below | yes |
| `passthrough` | none | none; the input is forwarded unchanged | **no** |

At runtime, `prompt_difference_selftest()` checks three things: (1) deleting the question block
from the conditioned prompt gives exactly the generic prompt; (2) no question-blind arm contains
either question; and (3) putting the generic instruction back into any question-blind arm gives
exactly the generic prompt. A `paraphrase`-versus-`generic` contrast therefore measures the
instruction and nothing else.

### `paraphrase` — rewrite-without-compression instruction

This is the only new prompt text in the experiment. `conditioned` and `generic` both summarise,
so neither can show whether repeated rewriting loses information on its own. This arm exists to
separate *rewriting* from *compression and relevance filtering*.

Stage 1 (`PARAPHRASE_INITIAL_INSTRUCTION`):

```
Rewrite the source material in your own words for another agent. Restate every
fact, qualifier, date, number, relationship, uncertainty, and source id it
contains. This is a rewrite, not a summary: do not condense, shorten, omit,
prioritise, or keep only what seems relevant, and do not add any fact that is
not already present.
```

Stage ≥2 (`PARAPHRASE_RECOMPRESS_INSTRUCTION`) is the same instruction applied to the previous
agent's notes, with `Use only the previous notes.` added. That mirrors how the question-conditioned
set differs between its own stage-1 and stage ≥2 forms.

`validate_modes()` refuses to run this arm together with `length_target_words`. A word budget is
a compression instruction, so combining the two would contradict the arm's own instruction and
remove the very variable it is meant to isolate.

`passthrough` has no prompt because it makes no model call. It forwards its input unchanged (the
raw context at stage 1) and serves as the floor with no rewriting at all.

### Semantic-preservation judge — did a rewrite keep the meaning?

`add_preservation_judge()` in [`src/judge.py`](src/judge.py) scores each consecutive handoff edge
`M_i -> M_{i+1}`. It is the semantic counterpart to the deterministic lexical measures in
[`src/paraphrase_metrics.py`](src/paraphrase_metrics.py), and it follows the same rules as the
answer judge above: a different model family from the system under test, temperature 0, and a
content-hashed cache.

System prompt:

```
You compare two research notes, where the second was written by rewriting the
first. You judge only whether the rewrite preserves the information in the
original. You never judge style, length, or writing quality.
```

The user message is `Original notes:\n{M_i}\n\nRewritten notes:\n{M_i+1}\n\n`, followed by an
instruction that asks for exactly one verdict: `EQUIVALENT` (every fact, name, number, date,
qualifier, relationship and uncertainty is preserved, even if worded differently), `MINOR_LOSS`
(the main content is kept but one detail is dropped or blurred) or `MAJOR_LOSS` (a substantial
omission, a contradiction or an added fact). These score 1.0, 0.5 and 0.0. A verdict that cannot
be parsed is left **unscored** instead of getting a default value, so a missing measurement
never counts as preservation.

### Final answer

`answer()`, [`run_summary_generalization.py:306`](src/run_summary_generalization.py#L306), uses
the shared `ANSWER_SYSTEM`. User message:
`Research material:\n{material}\n\nQuestion:\n{question}\nAnswer:`

### Dataset construction

The only supported loader is `load_prebuilt_pairs()`, configured by
`summary_generalization_squad_pairs_config.yaml`. The dataset is built and validated in advance by
[`src/build_squad_same_passage.py`](src/build_squad_same_passage.py). Each example takes two
native SQuAD questions about one shared SQuAD passage and places that passage at a stratified
position among nine length-matched SQuAD passages. Both questions are audited for salience and
independence, and both pass the closed-book C1 filter. Each distractor is screened separately to
confirm it is irrelevant to both A and B. The older constructors and configs that paired
questions from different passages have been removed.

---

## Experiment 5b — fictional fixed-capacity selection<a id="experiment-5b"></a>

**Files:** [`src/run_fictional_summary_generalization.py`](src/run_fictional_summary_generalization.py)
and [`src/fictional_qa.py`](src/fictional_qa.py).

Unlike Experiment 5, which asks for prose summaries, this runner asks for exactly K card ids and
then renders those cards' source text in code. The generic and conditioned prompts differ only in
the announced-question block.

System:

```
You select source-grounded evidence for a sealed handoff to another agent.
Obey the requested JSON schema exactly. Select only supplied evidence-card ids;
never write, merge, paraphrase, or invent evidence.
```

User template (`generic` leaves out the marked question block):

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

If the reply does not match the schema, the same prompt is retried with only this suffix added:

```
FORMAT CORRECTION: the previous response was invalid because {parse error}.
Return exactly one JSON object with exactly {K} distinct valid ids and no other key.
```

The final answerer gets the shared `ANSWER_SYSTEM` and a frozen handoff:

```
Research material:
{sealed.handoff_text}

Question:
{one announced or hidden evaluation query}
Answer:
```

Hidden questions never appear in either selection prompt. Identical final-answer requests are
deduplicated by their full request hash (single-flighted), so the same packet and question cannot
produce two different answers just because they appear in two roles.

---

## Experiment 6 — multilingual fixed vs switching handoffs<a id="experiment-6"></a>

**File:** [`src/run_multilingual_handoffs.py`](src/run_multilingual_handoffs.py). This experiment
imports `CHAIN_SYSTEM`, `INITIAL_INSTRUCTION` and `RECOMPRESS_INSTRUCTION` from Experiment 2. It
crosses two factors: whether Question A is shown, and whether the language stays fixed or
switches between stages. Every arm gets the same language directive. Before building any prompt,
the runner reduces each reusable 1+9 SQuAD pack to its single `gold_AB` passage, so no distractor
text reaches any model call in this experiment.

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

`LANGUAGE_DIRECTIVE` is filled in with the language assigned to that pair and stage:

```
OUTPUT LANGUAGE (mandatory): {language}. Write the entire replacement handoff
in {language}. Proper names, identifiers, numbers, and short source quotations
may remain unchanged when translation would alter them. Do not mix in another
language for the prose. Summarize; do not translate or rewrite the source
passage by passage. Do not use headings or one section per passage.
```

The directive contains no length target of any kind (sentences, words, paragraphs or anything
else). The 1,500-token API limit in the config only guards against runaway output and is not part
of the prompt.

At stage 1 the fixed and switching schedules start in the same assigned language with identical
prompts, and the stored stage-1 handoff is shared between them. From stage 2 on, the fixed
schedule keeps that language while the switching schedule cycles through English, German,
French, Italian, Portuguese and Spanish. The starting language is stratified by passage.

### Final answer

The shared `ANSWER_SYSTEM` is used, with the original English question:

```
Research material:
{direct English context at depth 0, or the handoff at the requested depth}

Question:
{Question A or Question B}
Answer:
```

### Language-compliance judge

The saved audit includes both a deterministic detector of the predominant language and the LLM
audit below, run by a different model family. The deterministic detector is the reported primary
compliance metric, because the LLM audit was too strict on French prose that was clearly correct.

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

**This experiment adds no new prompt**, and that is the design. Every string below is imported
from Experiment 2 or from Experiment 5's `conditioned` arm, and a startup check confirms each one
is unchanged. The only thing that varies between arms is *which model receives the prompt*.

| Element | Source | Note |
|---|---|---|
| System prompt | `run_chain.CHAIN_SYSTEM` | The same constant as in Experiments 2, 5 and 6 |
| Stage-1 instruction | `run_chain.INITIAL_INSTRUCTION` | See [Experiment 2](#experiment-2) |
| Stage ≥2 instruction | `run_chain.RECOMPRESS_INSTRUCTION` | See [Experiment 2](#experiment-2) |
| User-message assembly | `run_summary_generalization.compression_user_prompt(pair, notes, "conditioned", stage, cfg)` | Called as a function, not retyped |
| Final answer | `handoffs.ANSWER_SYSTEM` + Experiment 5's answer template | See [Final-answer persona](#final-answer-persona) |
| Answer judge | shared judge | See [LLM judge](#llm-judge--answer-correctness-vs-gold) |
| Semantic-preservation judge | `judge.add_preservation_judge` | See [Experiment 5](#experiment-5) |

### The one addition: a sealed rebuild of the stage ≥2 prompt

`compression_user_prompt` takes the whole pair dictionary, which would bring
`render_context(pair)`, that is every source passage, gold and distractor alike, back into a
stage-2 call. Experiment 8 therefore rebuilds the same string from a frozen `SealedHandoff` and
nothing else:

```
Previous agent's notes:
{sealed.handoff_text}

Question the final agent must answer: {sealed.question}

{RECOMPRESS_INSTRUCTION}
```

On every run, `prompt_selftest()` checks that this string is byte-for-byte equal to
`compression_user_prompt(pair, notes, "conditioned", stage, cfg)`, and that passing anything
other than a `SealedHandoff` raises `TypeError`. The experiment thus gets Experiment 2's
structural isolation without drifting from Experiment 5's published prompt.

### Model-blindness check

The same self-test checks that neither the stage-1 nor the stage ≥2 prompt contains any of
`llama`, `qwen`, `mistral`, `ministral`, `gemma` or `parameter count`. No agent is told which
model wrote its input, which model will read its output, or how large either of them is. An
effect of model composition therefore cannot come from a model being *told* about the
composition.

### Per-model generation controls

Two registry entries carry a model-specific control. `LLMClient` applies it and includes it in the
cache key, in the same way `qwen32_chain_config.yaml` already does. It is a property of the
decoder, not of the experimental condition:

| Model | Control | Why |
|---|---|---|
| `qwen/qwen3-8b`, `qwen/qwen3-32b` | `reasoning: {effort: none}` **and** the system-prompt suffix `/no_think` | Qwen3 reasons by default, and not every provider that OpenRouter routes to honours the gateway switch. Without both, the shared 1,500-token budget is spent on hidden reasoning instead of the visible handoff that the experiment measures. |

As a result, the two Qwen models get one extra line in their system prompt that the other six
models do not. The control is necessary rather than a design choice, but it is still an
asymmetry, and the report lists it as a confound.

---

## Experiment 8b — fictional selector/relay bottleneck<a id="experiment-8b"></a>

**Files:** [`src/run_fictional_model_bottleneck.py`](src/run_fictional_model_bottleneck.py)
and [`src/fictional_qa.py`](src/fictional_qa.py).

System prompt for both selector stages:

```
You are an evidence-routing agent. Select only evidence-card ids supplied in the prompt.
Obey the requested number of slots exactly and return only the requested JSON object.
```

Stage 1 uses Experiment 5b's source-card template with `K=3` and includes Question A. The later
Question B does not appear in this call. After the packet is sealed, stage 2 uses:

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

The `restore` and `sham` arms change the packet contents in code and add no instruction. The
`reopen` positive control uses the stage-1 source-card template with all six source cards,
Question B and `K=1`; it is explicitly not one of the sealed factorial arms. A malformed selection
is retried with:

```
FORMAT CORRECTION: the previous response was invalid because {parse error}.
Return exactly {K} distinct bare C-number ids in the requested JSON object;
do not include the word CARD.
```

The fixed Mistral reader uses the shared `ANSWER_SYSTEM` and `handoffs.orchestrator_answer`:
`Research notes from your subagent:\n{packet}` followed by `Question: {Question B}\nAnswer:`.
Answers are scored by the shared different-family judge described above.

---

## Experiment 9 — handoff size adaptation<a id="experiment-9"></a>

This is the **only** experiment in the project that does not use Experiment 2's handoff
instruction word for word. The change is part of the design, not an oversight.

### The base instruction: Experiment 2's, minus one word

Experiment 2's instruction begins *"Write **concise** prose research notes …"*. "Concise" is an
instruction about size, and this experiment is about size, so the word cannot be in the control.
Measured against it, `resize_large` would compare "expand" with "shrink" rather than with "no size
instruction", and `resize_small` would compare a requested shrink with a baseline that has already
been told to shrink.

The base instruction is therefore the published string with `concise ` removed: one deletion,
checked rather than trusted. `run_size_adaptation._drop_concise` raises an error unless the
inherited string contains that substring exactly once, and both the runner's startup check and the
offline selftest rebuild the published string from the neutral one before anything else runs.

| Stage | Constant | Text |
|---|---|---|
| System | `run_chain.CHAIN_SYSTEM` *(unchanged)* | "You are a research handoff agent. Preserve every fact needed to answer the question. Your output will replace your entire input for the next agent, so omitted information is lost." |
| 1 | `NEUTRAL_INITIAL_INSTRUCTION` | "Write prose research notes that preserve all evidence needed to answer the question. Do not answer the question directly and do not add unsupported facts." |
| ≥2 | `NEUTRAL_RECOMPRESS_INSTRUCTION` | "Rewrite the previous agent's notes into prose research notes for another agent. Preserve every answer-relevant fact, qualifier, date, number, relationship, uncertainty, and source id. Use only the previous notes. Do not answer the question directly." |

The system prompt is unchanged. It contains no size language and is shared, identical, with
Experiments 2, 5, 6 and 8.

### The directives

Each directive is appended to the instruction above and to nothing else. Every arm's user message
is the control's user message plus its directive string. The offline selftest checks this for
every directive at stage 1 and at stage ≥2, so the manipulation cannot leak into the material
block.

| Directive | Appended text |
|---|---|
| `none` | *(empty string)* |
| `concise` | " Keep your handoff concise." |
| `inform_small` | " The next agent's model has a 2,000-token context window." |
| `resize_small` | " The next agent's model has a 2,000-token context window. Make your handoff fit within it." |
| `inform_large` | " The next agent's model has a 10,000-token context window." |
| `resize_large` | " The next agent's model has a 10,000-token context window. Expand your handoff to make use of the available space." |

These strings are **built from parts, not typed out**. `directive_text()` builds each `resize`
string as the matching `inform` string plus one clause from `REQUEST_CLAUSE`, so a later edit
cannot make the two drift apart and turn the inform-versus-resize contrast into a comparison of
accidental rewording. Three properties are checked at startup:

1. `resize_X` starts with `inform_X`, and what follows is exactly one of the two declared request
   clauses;
2. `inform_small` with its number swapped equals `inform_large`, so the two differ only in the
   advertised size;
3. `directive_text("none")` is the empty string, and the assembled control prompt contains none
   of *token*, *context window*, *concise*, *expand*, *shorter* or *longer*.

The advertised size is not linked to `decoding.handoff_max_tokens`. The number the model is told
and the budget the API enforces are separate variables. The enforced budget is the same in every
arm and is meant only as a safety ceiling.

### Full user message

Stage 1 uses the same three-block layout as Experiment 5:

```
Source material:
{document}

Question the final agent must answer: {question}

{NEUTRAL_INITIAL_INSTRUCTION}{directive}
```

Stage ≥2 is built from a frozen `SealedHandoff` and nothing else, so the source document cannot
be reached. This matters more here than elsewhere. The experiment's central claim is that a fact
which disappears and later reappears was *reconstructed* by the model, and that claim falls apart
if a later stage could have reread the document.

```
Previous agent's notes:
{sealed.handoff_text}

Question the final agent must answer: {sealed.question}

{NEUTRAL_RECOMPRESS_INSTRUCTION}{directive}
```

### Dataset-construction prompts

These run once, in `src/build_size_adaptation_data.py`, and never during the experiment. They all
go to a strong writer model (`openai/gpt-5.2`, high reasoning effort), never to the model under
test; the Wikipedia and counterfactual builders keep the same separation.

| Prompt | Purpose | Checks applied to the reply |
|---|---|---|
| `WRITER_SYSTEM` + `fictional_prompt` | one page about an invented subject per item; the subject type cycles by index, so the variety comes from the build and not from one sampling run | length in range; answer appears verbatim in the page; no commentary about the task |
| `KNOWN_SUBJECT_PROMPT` | a passage about a famous subject, plus a question the model is likely to answer from memory | answer appears verbatim in the passage |
| `COUNTERFACTUAL_REWRITE_PROMPT` | rewrite that passage so the answer becomes a different plausible value | replacement appears verbatim in the rewrite; **original value absent from the rewrite**; neither value is a substring of the other |
| `EXTRACT_SYSTEM` + `side_fact_prompt` | five reusable side facts per document | each answer appears verbatim in the document; none overlaps the target answer; none contains another |

### Closed-book knowledge probe

This reuses `data.CLOSED_BOOK_SYSTEM` together with C1's message format, temperature and seed
ladder, unchanged, so an item already probed by an earlier build comes from the cache instead of
being paid for again. See [C1 parametric-leakage filter](#shared-across-experiments).

What differs from C1 is how the samples are used. C1 asks one question, *does the model already
know this?*, and drops the item if the answer is yes. Experiment 9 draws **one** set of samples per
item and judges it **twice**, once against the original gold answers and once against the
replacement gold answers. It requires:

- that the model *knows the original* in at least 2 of 3 samples. Without this, a handoff that
  simply loses the fact looks the same as one that falls back on memory;
- that the model *knows the replacement* in 0 of 3 samples, the usual C1 condition.

Judging one sample set twice, instead of drawing two, is deliberate. Two independent draws could
disagree, and the two conditions would then describe different behaviour rather than two readings
of the same behaviour.

### Claim-support judge

New in this experiment. It lives in `judge.py` next to the answer and preservation judges, and
uses the same different-family model (`openai/gpt-4o-mini`) at temperature 0.

System: "You check whether research notes stay within their source document. You are given a
source document and notes written from it. You report only claims the document does not support."

The instruction says that rewording, summarising, reordering and omitting never count as
unsupported. It asks for a count plus a short quote for each unsupported claim, on a single line
that can be parsed. A reply that cannot be parsed is left **unscored** rather than scored as zero,
so a judge failure cannot be averaged in as a clean handoff.

Claims are checked against the **source document at every depth**, not against the note that came
just before. Checking stage 3 against stage 2 would accept a fabrication that entered at stage 2
and was then copied faithfully, and no stage in a chain is ever allowed to add material the
document does not contain.

### Final answer and answer judge

`handoffs.ANSWER_SYSTEM` and Experiment 5's answer template, unchanged; see
[Final-answer persona](#final-answer-persona). The shared answer judge scores each prediction
twice from **one** generated answer: once against the document's gold answers and once against
the memorised ones. "Fell back on memory" therefore can never be a sampling artefact.

---

## Experiment 10 — communication regret under a hard budget<a id="experiment-10"></a>

**Files:** [`src/budget.py`](src/budget.py) (every string the sender sees),
[`src/run_communication_regret.py`](src/run_communication_regret.py),
[`src/build_regret_data.py`](src/build_regret_data.py).

Every policy builds the sender prompt in the same fixed order: the source, then the information
need, then the shared length contract and output form. Among the four abstractive policies only
the information-need block changes, and the offline selftest checks that replacing that block with
a placeholder leaves one identical skeleton.

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

`generic` deliberately says only that the sender has not been told the question. Saying that more
questions may follow *is* the `reusable` treatment, and the control must not contain the variable
being tested.

### Extractive control

The two extractive arms reuse the `generic` and `conditioned` blocks verbatim and replace only the
final output-form line:

```
Copy sentences VERBATIM from the source material - character for character. Do
not paraphrase, merge, shorten or rewrite, and write nothing of your own.
Separate the sentences with a single space. Do not answer any question yourself.
```

### Length corrections

Both corrections are identical for every policy, so an arm that overshoots or undershoots more
often is not also being given a different instruction. They are applied up to
`budget.max_length_corrections` times, after which the message is truncated deterministically.

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

The expansion wording mentions no question and no future need. It only asks for more of what the
source already contains, which is the one neutral way to fill a budget without inviting the
padding failure that Experiment 9 found.

### Answering

The shared `ANSWER_SYSTEM` from [`src/handoffs.py:24`](src/handoffs.py#L24), with a frozen
`SealedHandoff`:

```
Research material:
{sealed.handoff_text}

Question:
{one evaluated question, conditioning or hidden}
Answer:
```

The direct-context ceiling is the only call whose material is `context.source`. Hidden questions
never appear in any sender prompt. Identical answer requests are deduplicated by their full request
hash, which matters here because one `generic` message is deliberately shared by every rotation of
its context.

### Dataset build

The group independence audit ([`src/build_regret_data.py`](src/build_regret_data.py),
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

The relation-dossier writer prompt (`RELATION_WRITER_PROMPT`) asks a strong model for four
sections. Each section states one anchor fact about the main subject, a second fact about that
same subject, and a fact about a different named entity introduced in the section, and each comes
with a paraphrase question that has the same answer as the anchor. Every designed answer is then
checked verbatim in the generated text, and the role structure is checked before the corpus is
written.

The C1 closed-book filter is the shared one at the top of this file. It is applied to **every**
question of a candidate context, and the whole group is dropped if any one question leaks.

---

## Experiment 14 — compression mechanism: rewriting vs selection<a id="experiment-14"></a>

**Files:** [`src/budget.py`](src/budget.py) (the `paraphrase` block),
[`src/elimination.py`](src/elimination.py) (the selectors, which use no prompts),
[`src/lm_unit_scores.py`](src/lm_unit_scores.py) (the scoring templates),
[`src/run_communication_regret.py`](src/run_communication_regret.py).

Experiment 14 reuses Experiment 10's channel exactly: the same `SENDER_SYSTEM`,
`LENGTH_CONTRACT` and `PROSE_FORM`, the same reader prompt and the same judge. The whole experiment
adds only two new *prompts*, and four of its arms use no prompt at all.

### The `paraphrase` information-need block

[`src/budget.py`](src/budget.py), `policy_block`. It is assembled in the same fixed order as every
other abstractive arm (source, information need, length contract, output form), so it differs from
`generic` only in the third sentence:

```
The next agent will use your handoff to answer a question about this source material.
You have not been told which question it will be asked.
Restate the source material in your own words. Keep as much of it as fits in the
length below, in the order it appears, rather than choosing which parts matter.
```

The first two lines are `generic`'s, unchanged. That is necessary: the variable under study is
*rewriting with selection* versus *rewriting without deliberate selection*, so the two arms must
know the same amount about the question. For the same reason as `generic`, this arm never sees a
question, so its message is the same in every rotation.

With a 20–160-word cap on a 389–517-word dossier, keeping everything is impossible, and the
instruction does not pretend otherwise ("as much of it as fits"). The reference for rewriting with
no budget is Experiment 5's paraphrase run, and the reference for no compression at all is this
run's own `direct_context` baseline.

### The LM selector's scoring templates

[`src/lm_unit_scores.py`](src/lm_unit_scores.py). These are **not** generation prompts; nothing is
sampled from them. They are fed to the model with teacher forcing, and the token log-probabilities
are the score. A local, pinned GPT-2 computes them (the compressor LM in the small configuration of
LLMLingua and LongLLMLingua), never the system under test.

Query-aware (`lm_conditioned`), scoring one evidence sentence against the current question:

```
{unit}

We can get the answer to this question from the text above.
Question: {question}
```

The query-free baseline for the same question removes the sentence and leaves everything else
identical:

```
We can get the answer to this question from the text above.
Question: {question}
```

The score is the difference between the two mean question-token log-probabilities. This follows
LongLLMLingua's coarse document-relevance score, written as a contrast so it can be compared across
dossiers:

```
score(unit, q) = (1/|q|) * [ sum_i log p(q_i | unit, q_<i) - sum_i log p(q_i | q_<i) ]
```

The sentence "We can get the answer…" comes from LongLLMLingua. It appears in **both** templates,
so the only difference between them is whether the sentence being scored is present.

The query-free arm (`lm_generic`) uses no question at all. Its score is the mean per-token
surprisal of the sentence on its own, after a BOS token:

```
score(unit) = -(1/T) * sum_t log p(x_t | x_<t)
```

Both LM arms use the same scorer, checkpoint and tokenisation; only the score formula differs.
Conditioned scores are computed **only** for the four anchor questions that rotate into the
conditioning role, so the hidden questions never reach the model.
`src/lm_unit_scores.py:check_leakage` enforces this when the scores are written, and the offline
selftest checks it again when they are read.

### The arms with no prompt

`nonllm_generic` (TF-IDF centrality), `nonllm_conditioned` (BM25 against the current question),
`random_selection` (a shuffle seeded from the content) and `passthrough` (the part of the source,
from the start, that fits) use no model and therefore no prompt. Each message is a sequence of
whole source sentences in source order, checked against the source before delivery
(`elimination.verify_verbatim`).

This is what separates them from Experiment 10's `extractive_*` arms. Those arms are a *prompt*
asking an LLM to copy sentences verbatim ([`src/budget.py`](src/budget.py), `EXTRACTIVE_FORM`), and
on the SQuAD run the output actually was verbatim only 0–29% of the time. Experiment 14 does not
reuse those arms or their results.

---

## Cross-cutting wording inconsistency<a id="wording-inconsistency"></a>

Three files each define their own handoff system prompt. The prompts say the same thing in
slightly different words, instead of importing one shared constant:

| File | Wording |
|---|---|
| `run_redundant_signal_ratio.py:31` (Exp. 4) | "...replace **your** entire input for the next agent..." |
| `run_retrieval_quality.py:26` (Exp. 3) | "...replace **the** entire input for the next agent..." |
| `run_slack_retrieval.py:43` *(not part of the numbered report)* | "...replace **the** entire input for the next agent..." (matches retrieval_quality) |

The difference is one word and does not change the instruction or any reported result. It does
mean Experiments 3 and 4 do not run identical compressor prompts, whereas Experiment 5 imports
Experiment 2's prompt on purpose and checks it with a self-test. If a future experiment needs to
claim that two conditions differ in *only* one variable, follow Experiment 5's approach (import the
shared constant and add a self-test that it is unchanged) rather than retyping the prompt.
