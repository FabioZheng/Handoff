# Datasets, models and sample sizes by experiment

Everything here comes from `results/HANDOFF_EXPERIMENTS_REPORT.md`, each experiment's config,
and the `_manifest` line at the top of each built dataset. The **stats unit** is what the
bootstrap resamples, so it is the effective `n` behind every reported confidence interval and
p-value.

## Key terms: dossier and rotation

### A dossier is one invented source document plus its questions
- A dossier is a **short reference article about something that does not exist**, written by
  GPT-5.2. It is about 400–500 words long (2,000–3,100 characters), and the answer to every
  question is stated in it.
- Because the subject is invented, the model cannot know any answer in advance (closed-book
  accuracy is 0.000). A correct answer can only have come through the handoff.
- There are two kinds.

**Fictional dossiers have 6 questions** (built for Exp 9, reused by 5b, 8b and 13)
- Example: *Veydrin Hollow*, an invented settlement (2,666 characters). It has one target fact
  and five side facts, each with its own question:
  - *Who founded Veydrin Hollow?* → Mara Teln (target)
  - *What is the siltstone green called?* → Drift Square
  - *When was the stone footbridge completed?* → 889 WR
  - *How long is the Hollowline Aqueduct?* → 1.7 km
  - *When is the annual Rillmarket held?* → 18 Waneset
  - *When did Kestrelpin Common School open?* → 940 WR
- In Exp 5b and 8b each fact becomes an **evidence card**: the sentence in the document that
  contains the answer. The sender's handoff is exactly K of the six cards.

**Relation dossiers have 16 questions on a designed 4 × 4 grid** (built for Exp 10a, reused by
11, 12, 13 and 14)
- Example: *Vellunar Inland Canal Company* (467 words). It covers four topics, called
  aspects, and each aspect is asked about in four ways, called roles. For the **founding**
  aspect:
  - **anchor:** *On what date was the company incorporated?* → 13 Frostfall 1827
  - **paraphrase:** *What incorporation date is recorded…?* → the same answer, worded differently
  - **same entity:** *Which compact granted the company towpath rights?* → Mirren Charter
  - **same topic:** *Who is credited with the original route survey?* → Orin Dastel
    (a different entity in the same aspect)
- **Facilities**, **finance** and **custom** follow the same pattern. Their anchors are Harth
  Quay Depot, six ravels and Lantern Draft.
- The grid sets how far each question is from its anchor:
  **paraphrase < same entity < same topic < orthogonal**, where orthogonal means any question
  from another aspect. Exp 10a measures regret as a function of this distance.

### A rotation is one choice of which question the sender is told about
- In each rotation **one question is announced** to the sender (the "current" question, or
  "Question A"). **All other questions about the same source are hidden.** The sender never
  sees them, but they are all scored on the same message.
- **Rotating** means repeating this until every eligible question has had one turn as the
  announced question.
- **Why rotate:** each question is the known one in one rotation and a future one in the
  others, so question difficulty cannot produce the difference between conditioned and
  generic messages.
- **Generic (question-blind) messages do not depend on the rotation.** They are written once
  per source (and per K or budget) and scored again under every rotation. Only conditioned
  messages are rewritten for each rotation.
- **Rotations are not independent samples.** All rotations of a dossier share the same text,
  so the statistics resample whole dossiers. That is why the stats unit is n=20 or n=16, not
  the 120 or 64 rotations.
- **Worked example (Exp 5b, Veydrin Hollow):**
  - Rotation 1 announces *Who founded Veydrin Hollow?* and hides the other five questions.
  - Rotation 2 announces *How long is the Hollowline Aqueduct?* and hides the other five,
    including the founder question.
  - And so on, up to rotation 6.

**What a rotation means in each experiment**
- **Exp 5b:** six rotations per dossier. Each of the six facts is announced once, with the
  other five hidden. 20 × 6 = 120 rotations.
- **Exp 8b:** four rotations per dossier. The announced question A is either the original
  target fact or one of three side facts chosen by hash. Once the selector has sealed its
  three-card packet, a single question B is revealed. B is the next card in a fixed, shuffled
  cycle of the six cards, and it is never A itself. 20 × 4 = 80 A→B tasks.
- **Exp 9:** no rotation. Only the target question (and one side fact) is asked.
- **Exp 10 (SQuAD groups):** the source is a real SQuAD paragraph with four human-written
  questions, and all four rotate. 24 × 4 = 96 rotations. Each message is scored on its
  announced question and the three hidden ones.
- **Exp 10a, 11, 12 and 14 (relation dossiers):** only the **four anchors** rotate. Paraphrase,
  same-entity and same-topic questions are never announced. 16 × 4 = 64 rotations. Each
  message is scored on all 16 questions: the anchor plus 15 hidden ones (1 paraphrase, 1 same
  entity, 1 same topic and 12 orthogonal).
- **What Exp 12 adds:** in each rotation, the true future demand puts 70% on one other aspect
  (the next one in the aspect order) and 10% on each of the other three, including the
  announced aspect.
- **Exp 13 (planned):**
  - Panel A: Exp 5b's six rotations plus one generic packet per dossier, so 7 packets per
    dossier (20 × 7 = 140).
  - Panel B: the four anchor rotations.

## Shared across experiments

- **Public raw data** (validation splits, Hugging Face parquet): MuSiQue-Answerable,
  HotpotQA (distractor), MS MARCO QA v2.1, SQuAD v1.1.
- **Judge:** `openai/gpt-4o-mini` at temperature 0, from Exp 2 onward. The same model also
  screens distractors (Exp 3–5), audits whether questions are independent (Exp 5, 10), checks
  output language (Exp 6), and judges semantic preservation (Exp 5a, 8, 9) and claim support
  (Exp 9).
- **Dataset writer**, used only for fully synthetic data and never as a model under test:
  `openai/gpt-5.2` with high reasoning effort.
- **Leakage filter (C1):** the model under test answers each candidate question closed-book
  three times at T=0.7, and the question is dropped if any attempt is correct.
- **Models under test** (through OpenRouter): `meta-llama/llama-3.3-70b-instruct`,
  `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen3-8b`, `qwen/qwen3-32b` (thinking turned off),
  `mistralai/ministral-8b`, `mistralai/ministral-8b-2512`,
  `mistralai/mistral-small-3.2-24b-instruct`, `google/gemma-3-12b-it` and
  `google/gemma-3-27b-it`. Run locally: GPT-2 small (Exp 14) and Qwen3-4B (Exp 13, planned).

## Per experiment

### 1 · Single-handoff mechanisms
- **Data:** 10 MuSiQue-Answerable questions, kept from 40 candidates after the C1 filter,
  each with 20 paragraphs.
- **Models:** Llama 3.3 70B as both subagent and answerer. No judge.
- **Scale:** three mechanisms (`A_full`, `B_freeform` with 2 seeds, `E_oracle`), 180 calls in
  total.
- **Stats unit:** question, n=10.

### 2 · Fixed-evidence serial chain (+ 2b, question omitted)
- **Data:** 30 MuSiQue and 30 HotpotQA questions, each set kept from 100 candidates after the
  C1 filter.
- **Contexts:** gold-only (2 documents for HotpotQA), 5 documents, and full (20 documents for
  MuSiQue, 10 for HotpotQA).
- **Models:** Llama 3.3 70B, plus the judge.
- **Scale:** 2 datasets × 3 contexts × depths 0–10 × 2 seeds. Exp 2b reruns the same 60
  questions with compressors that never see the question.
- **Stats unit:** question, n=30 per dataset.

### 2a · Qwen replications
- **Qwen3 8B:** 10 questions per dataset (from 40 candidates), 1 seed, depths 0/1/3/5, no
  judge.
- **Qwen3 32B:** 30 questions per dataset (from 100 candidates), 2 seeds, depths 0–10, with
  the judge.
- C1 depends on the model, so these question sets differ from Llama's and the results are
  **not paired** with Exp 2.

### 3 · Retrieval quality
- **Data:** 20 MS MARCO v2.1 queries, with 10 passages per context.
- **Construction:** each context keeps all, half or 3 of the 22 gold passages that BM25 can
  find (Recall@10 = 1.00, 0.50 and 0.14). The remaining slots are filled with hard negatives
  from BM25's top 50 or easy ones from its bottom 1,000, drawn from a pool of 4,000 queries.
  GPT-4o-mini screened 1,200 candidates, and only those it judged `IRRELEVANT` were kept.
- **Models:** Llama 3.1 8B, plus the judge.
- **Scale:** 6 arms × depths 0/1/3/5 (120 packs), 1 seed.
- **Stats unit:** query, n=20.

### 4 · Redundant-evidence signal ratio
- **Data:** 20 SQuAD questions, with 10 passages per context.
- **Construction:** a "gold" passage is the answer paragraph plus a different paragraph from
  the same article. Contexts hold 10, 5 or 1 gold passages plus 0, 5 or 9 distractors from
  other articles, either hard (BM25 top 60) or easy (bottom 1,000). Distractors are filtered
  for answer aliases and screened by GPT-4o-mini.
- **Models:** Llama 3.1 8B, plus the judge.
- **Scale:** 6 arms × depths 0/1/3/5 (120 packs), 1 seed.
- **Stats unit:** question, n=20.

### 5 · Question conditioning on SQuAD A/B pairs (+ 5a, the rewriting ladder)
- **Separate passages:** 20 pairs. Each context holds A's gold passage and B's gold passage
  (from different articles) plus 8 distractors.
- **Same passage with distractors:** 20 paragraphs (712–878 characters), each with 2
  human-written questions and 9 screened distractors. The position of the gold passage is
  stratified.
- **Same passage, gold only:** 10 paragraphs (721–866 characters) with 2 questions each, so
  20 questions. No distractors.
- **Pair construction:** human-written SQuAD questions whose word overlap (Jaccard) is at most
  0.10. GPT-4o-mini checks that A and B are independent, and both questions must pass C1
  against Llama 3.1 8B (220 were probed, and 84–85 leaked and were dropped).
- **5a:** the same 10 gold-only pairs with four arms (pass-through, paraphrase, generic,
  conditioned), 1 seed.
- **Models:** Llama 3.1 8B plus the judge (and the preservation judge in 5a). Depths 0–10.
- **Stats unit:** pair, n=20 / 20 / 10.

### 5b · Fictional fixed-capacity replication
- **Data:** the 20 fictional dossiers from Exp 9, each turned into 6 evidence cards and 6
  questions (120 questions).
- **Scale:** 6 A-rotations per dossier (120 rotations) × K=2/4/6 × generic/conditioned. All
  five hidden questions are scored on every packet. 418 of the 420 selector calls returned
  valid output.
- **Models:** Llama 3.1 8B as selector and answerer, plus the judge.
- **Stats unit:** dossier, n=20, with rotations clustered.

### 6 · Multilingual handoffs
- **Data:** the Exp 5 gold-only set, 10 passages and 20 questions.
- **Scale:** 6 languages (en/de/fr/it/pt/es) × conditioned/generic × fixed/switching, depths
  1–6.
- **Models:** Llama 3.1 8B, the judge, and GPT-4o-mini as the language auditor.
- **Stats unit:** passage, n=10.

### 7 · Incremental-evidence chain
- **Data:** 30 MuSiQue-Answerable questions, kept from 100 candidates after the C1 filter
  against Llama 3.3 70B. Each question's 2–4 supporting paragraphs arrive one packet at a
  time, in counterbalanced order.
- **Scale:** 0/1/3/5 relays × question shown or omitted × 2 seeds, plus probes on hidden
  sub-questions.
- **Models:** Llama 3.3 70B, plus the judge.
- **Stats unit:** question, n=30.

### 8 · Model heterogeneity
- **Data:** the Exp 5 same-passage set, 20 passages and 40 questions, with 10 passages per
  context.
- **Chain models:** the small tier is Llama 3.1 8B, Qwen3 8B, Ministral 8B and Gemma 3 12B;
  the large tier is Llama 3.3 70B, Qwen3 32B, Mistral Small 3.2 24B and Gemma 3 27B.
- **Answerers:** Llama 3.1 8B (primary) and Ministral 8B (secondary), plus the answer and
  preservation judges.
- **Scale:** 17 arms plus 2 derived controls × depths 0–6, 1 seed.
- **Caveat:** Gemma 12B returned empty output for 13 of the 20 stage-1 prompts.
- **Stats unit:** pair, n=20.

### 8b · Selector/relay bottleneck
- **Data:** the 20 fictional dossiers from Exp 9 × 4 A→B rotations = 80 tasks. The channel
  narrows from 6 cards to 3 and then to 1.
- **Models:** the selectors and relays are Llama 3.1 8B / 3.3 70B and Qwen3 8B / 32B, crossed
  2×2 within each family. A fixed Mistral Small 3.2 24B reader gives every answer. Plus the
  judge.
- **Stats unit:** dossier, n=20. The clean-omission subset is 51 Llama and 47 Qwen tasks.

### 9 · Size adaptation
- **Data:** 46 items: 26 counterfactual (20 `known_entity` and 6 `rewritten_wikipedia`) and 20
  fictional. Each item has one target question and five side facts (see how they were
  created below).
- **Models:** Llama 3.3 70B as writer and answerer, with the answer, preservation and
  claim-support judges.
- **Scale:** 9 directive arms × depths 0–3, with a 12k-token output cap.
- **Stats unit:** item, analysed separately per dataset (n=26 and n=20).

### 10 · Communication regret
- **SQuAD groups:** 24 paragraphs (800–1,473 characters) × 4 human-written questions = 96
  questions. 6 policies × budgets of 20/40/80/160 words give **1,440 messages and 7,104
  answers**.
  - *Construction:* 200 candidate paragraphs with 6 questions each. C1 against Llama 3.1 8B
    removed 318 questions, leaving 58 groups with at least 4 clean questions. The GPT-4o-mini
    audit (answerable, distinct facts) rejected 1 group, and 24 were kept.
- **10a, relation dossiers:** 16 fictional dossiers × 16 questions = 256 questions. 4 policies
  × 4 budgets × 4 anchor rotations give **640 messages and 11,264 answers**.
- **Models:** Llama 3.1 8B as sender and answerer, plus the judge.
- **Stats unit:** context, n=24 (SQuAD) and n=16 (dossiers).

### 11 · Preference frontier
- **Data:** the Exp 10 corpora (24 SQuAD groups and 16 relation dossiers).
- **Models:** Llama 3.1 8B as writer and reader in three runs: the primary run (SQuAD at 40/80
  words, dossiers at 40/80/160), a free-form λ probe (dossiers, 80 words), and three
  sampled-seed replicates at T=0.2 (SQuAD, 80 words). Ministral-3-8B-2512 as writer and reader
  (dossiers, 40/80 words), with matched Exp 10 controls. Plus the judge.
- **Scale:** a 5-point grid over α (allocation) and λ (objective).
- **Stats unit:** context, n=24 or n=16.

### 12 · Anticipatory context management
- **Data:** 16 relation dossiers (256 questions and 264 sentences, of which 191 are labelled
  with one aspect and 73 are background).
- **Scale:** 4 rotations × 16 policies × budgets of 40/80 words = **2,048 messages and 32,768
  answers**.
- **Models:** Llama 3.1 8B as sender and answerer, an Okapi BM25 store (k=2), and the judge.
- **Stats unit:** context, n=16.

### 13 · Latent (non-language) handoff: **not run**
- **Planned model:** Qwen3-4B at a pinned revision, on a local GPU with thinking turned off,
  used as both writer and reader.
- **Panel A:** the 20 fictional dossiers, turned into 140 K=4 packets from the Exp 5b
  selections. 3 channels × 6 questions = 2,520 evaluations.
- **Panel B:** the 16 relation dossiers.
- **Confirmation set:** at least 100 fresh sources, not built yet.
- So far only offline and fake-backend checks exist. There are no inference results.

### 14 · Compression mechanism
- **Data:** 16 relation dossiers (389–517 words, 256 questions), 64 rotations.
- **Scale:** 9 arms (plus `reusable` and `oracle`, imported from 10a) × budgets of
  20/40/80/160 words = **1,472 messages and 11,840 judged answers**. Of these, 640 messages
  and 11,264 answers were reused from Exp 10a.
- **Models:** Llama 3.1 8B for the abstractive arms and as the answerer; GPT-2 small (pinned
  commit `607a30d7`, on CPU) as the sentence scorer; BM25 and TF-IDF (no model); and the judge.
- **Stats unit:** context, n=16. The 20-word budget is left out of the claims.

## How the synthetic datasets were created

### Fictional dossiers: `data/size_adaptation/fictional_items.jsonl` (Exp 9; reused by 5b, 8b and 13)
- GPT-5.2 wrote articles about invented places, people and institutions. 24 pages were built
  and **20 kept**, each 2,021–3,155 characters long.
- Each item has one target question and five side facts. The facts are checked verbatim
  against the document and do not overlap one another.
- A closed-book check against Llama 3.3 70B confirmed that none of them can be answered from
  memory.
- For 5b and 8b, `src/fictional_qa.py` turns each item into six evidence cards without
  generating anything new. Controls: full-card accuracy is 1.000 and closed-book accuracy is
  0.000.

### Counterfactual items: `data/size_adaptation/counterfactual_items.jsonl` (Exp 9)
- Each document deliberately contradicts a fact the model has memorised. Both directions are
  checked against Llama 3.3 70B: the model must recall the original answer (in at least 2 of
  3 closed-book samples) and must never produce the replacement.
- **`known_entity` (20):** famous subjects were sampled first and confirmed to be known.
  GPT-5.2 then changed one fact to a plausible new value. These passages are short, about
  1,100 characters on average.
- **`rewritten_wikipedia` (6):** taken from the random-Wikipedia counterfactual set below.
  Only 6 of its 20 items passed the "knows the original" check. These passages are long,
  about 7,500 characters on average.
- GPT-5.2 extracted the five side facts for each item. 20 candidates were rejected in total.

### Random Wikipedia and its counterfactual rewrite: `data/wikipedia_random*` (source for Exp 9)
- 10 random English Wikipedia pages (4k–40k characters, pinned to a revision). GPT-5.2 wrote
  two unrelated questions per page, grounded in the page with verbatim evidence, and a second
  pass wrote the gold answers and aliases, giving **20 questions**.
- **Counterfactual version:** GPT-5.2 changed the evidence fact on each page to a different
  plausible value and rewrote the article so it stays consistent. 19 of the 20 questions
  survive C1 against Llama 3.3 70B, compared with 14 of 20 for the originals.
- `config.yaml` currently points Exp 1 and 2 at this set, but no reported result uses it.

### Relation dossiers: `data/communication_regret/relation_dossiers.jsonl` (Exp 10a; reused by 11, 12, 13 and 14)
- GPT-5.2 wrote fictional dossiers to a strict schema. The ones kept are 2,407–3,101
  characters long.
- Each dossier covers four aspects (`founding`, `facilities`, `finance`, `custom`), and each
  aspect has four questions:
  - an anchor question
  - a paraphrase (the same answer, worded differently)
  - a second fact about the same entity
  - a fact about a different entity in the same aspect

  That gives 16 questions per dossier. Every answer is checked verbatim in the text.
  Questions from other aspects count as orthogonal.
- **Filtering:** 40 pages were requested and 38 were built. A dossier was dropped if Llama 3.1
  8B answered *any* of its 16 questions closed-book. 12 were dropped and **16 kept**.
- The direct-context ceiling is 0.992 and closed-book accuracy is 0.000. Building the set cost
  $2.52.

## Runs on disk that are not in the report

- `slack_facts`: SQuAD, 20 packs, k=2/5/10 facts plus 0/5/10 added filler passages, Llama 3.1
  8B.
- `slack_retrieval`: MS MARCO, 20 queries, 3 essential passages plus 0/3/6 inert or near-miss
  filler passages, Llama 3.1 8B.
- `squad_same_passage_matched`: the Exp 5 10-pair set with a 400-word length target.
- `multilingual_handoff`: the Exp 6 design with distractors, replaced by the gold-only
  version.
