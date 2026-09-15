# Datasets, models and sample sizes by experiment

Sources: `results/HANDOFF_EXPERIMENTS_REPORT.md`, each experiment's config, and the
`_manifest` line of every built dataset. **Stats unit** is what the bootstrap
intervals resample: the effective `n` behind every reported CI and p-value.

## Key terms: dossier and rotation

### Dossier = one invented source document plus its questions
- A **short reference article about something that does not exist**, written by
  GPT-5.2: about 400–500 words (2,000–3,100 chars). Every question's answer is
  stated in it.
- The subject is invented, so the model cannot know any answer beforehand
  (closed-book accuracy is 0.000). A correct answer can only have come through the
  handoff.
- There are two kinds.

**Fictional dossier: 6 questions** (Exp 9 → reused by 5b, 8b, 13)
- Example: *Veydrin Hollow*, an invented settlement (2,666 chars). It has 1 target
  fact and 5 side facts, each with its own question:
  - *Who founded Veydrin Hollow?* → Mara Teln (target)
  - *What is the siltstone green called?* → Drift Square
  - *When was the stone footbridge completed?* → 889 WR
  - *How long is the Hollowline Aqueduct?* → 1.7 km
  - *When is the annual Rillmarket held?* → 18 Waneset
  - *When did Kestrelpin Common School open?* → 940 WR
- In Exp 5b and 8b, each fact becomes an **evidence card**: the document sentence
  containing that answer. The sender's handoff is exactly K of the 6 cards.

**Relation dossier: 16 questions on a designed 4 × 4 grid** (Exp 10a → reused by 11, 12, 13, 14)
- Example: *Vellunar Inland Canal Company* (467 words). It covers 4 topics
  ("aspects"), and each topic is asked about in 4 ways ("roles"). The
  **founding** aspect:
  - **anchor:** *On what date was the company incorporated?* → 13 Frostfall 1827
  - **paraphrase:** *What incorporation date is recorded…?* → same answer, reworded
  - **same entity:** *Which compact granted the company towpath rights?* → Mirren Charter
  - **same topic:** *Who is credited with the original route survey?* → Orin Dastel
    (a different entity, same aspect)
- **facilities**, **finance** and **custom** follow the same pattern. Their anchors
  are Harth Quay Depot, six ravels and Lantern Draft.
- The grid fixes how far each question is from an anchor:
  **paraphrase < same entity < same topic < orthogonal**. Orthogonal means any
  question from another aspect. Exp 10a measures regret against this distance.

### Rotation = one choice of which question the sender is told about
- In one rotation, **one question is announced** to the sender ("current" or
  "Question A"). **All other questions of the same source are hidden**: the sender
  never sees them, but they are all scored on the same message.
- **Rotating** means repeating this so that each eligible question takes one turn
  as the announced one.
- **Why rotate:** every question is "the known one" in one rotation and "a future
  one" in the others. Question difficulty therefore cannot create the
  conditioned-vs-generic difference.
- **Generic (question-blind) messages don't depend on the rotation.** They are
  written once per source (and per K or budget) and re-scored under every
  rotation. Only conditioned messages are rewritten for each rotation.
- **Rotations are not independent samples.** Rotations of one dossier share the
  same text, so the statistics resample whole dossiers. That is why the stats unit
  is n=20 or n=16, not the 120 or 64 rotations.
- **Worked example (Exp 5b, Veydrin Hollow):**
  - Rotation 1 announces *Who founded Veydrin Hollow?* and hides the other 5
    questions.
  - Rotation 2 announces *How long is the Hollowline Aqueduct?* and hides the
    other 5, including the founder question.
  - This continues through rotation 6.

**What a rotation is in each experiment**
- **Exp 5b:** 6 rotations per dossier. Each of the 6 facts is announced once, with
  5 hidden each time. 20 × 6 = 120 rotations.
- **Exp 8b:** 4 rotations per dossier. The announced A is the original target fact
  or one of 3 hash-chosen side facts. After the selector seals its 3-card packet,
  a single B is revealed: the next card in a fixed shuffled cycle of the 6 cards,
  never A itself. 20 × 4 = 80 A→B tasks.
- **Exp 9:** no rotation. Only the target question (plus one side fact) is asked.
- **Exp 10 (SQuAD groups):** the source is a real SQuAD paragraph with 4 human
  questions, and all 4 rotate. 24 × 4 = 96 rotations. Each message is scored on
  its announced question + 3 hidden.
- **Exp 10a, 11, 12, 14 (relation dossiers):** only the **4 anchors** rotate;
  paraphrase, same-entity and same-topic questions are never announced. 16 × 4 = 64
  rotations. Each message is scored on all 16 questions: the anchor + 15 hidden
  (1 paraphrase, 1 same entity, 1 same topic, 12 orthogonal).
- **Exp 12 addition:** in each rotation, the true future demand puts 70% on one
  other aspect (the next in the aspect order) and 10% on each of the other three,
  including the announced aspect.
- **Exp 13 (planned):**
  - Panel A: 5b's 6 rotations + 1 generic packet per dossier = 7 packets
    (20 × 7 = 140).
  - Panel B: the 4 anchor rotations.

## Shared across experiments

- **Public raw data** (validation splits, Hugging Face parquet): MuSiQue-Answerable,
  HotpotQA (distractor), MS MARCO QA v2.1, SQuAD v1.1.
- **Judge:** `openai/gpt-4o-mini`, temperature 0, from Exp 2 onward. The same model also
  screens distractors (Exp 3–5), audits question independence (Exp 5, 10), checks
  output language (Exp 6), and judges semantic preservation (Exp 5a, 8, 9) and
  claim support (Exp 9).
- **Dataset writer** (fully synthetic data only; never a model under test):
  `openai/gpt-5.2`, high reasoning effort.
- **Leakage filter (C1):** the model under test answers every candidate closed-book
  3 times at T=0.7. The question is dropped if any attempt is correct.
- **Models under test** (OpenRouter): `meta-llama/llama-3.3-70b-instruct`,
  `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen3-8b`, `qwen/qwen3-32b` (thinking off),
  `mistralai/ministral-8b`, `mistralai/ministral-8b-2512`,
  `mistralai/mistral-small-3.2-24b-instruct`, `google/gemma-3-12b-it`,
  `google/gemma-3-27b-it`. Local: GPT-2 small (Exp 14) and Qwen3-4B (Exp 13, planned).

## Per experiment

### 1 · Single-handoff mechanisms
- **Data:** MuSiQue-Answerable, **10 questions** (40 candidates, C1-filtered), 20 paragraphs each
- **Models:** Llama 3.3 70B as subagent and answerer. No judge.
- **Scale:** 3 mechanisms (`A_full`, `B_freeform` × 2 seeds, `E_oracle`), 180 calls
- **Stats unit:** question, n=10

### 2 · Fixed-evidence serial chain (+ 2b question omitted)
- **Data:** **30 MuSiQue + 30 HotpotQA questions**, each set C1-filtered from 100 candidates
- **Contexts:** gold-only (HotpotQA 2 docs), 5 docs, full (MuSiQue 20, HotpotQA 10 docs)
- **Models:** Llama 3.3 70B + judge
- **Scale:** 2 datasets × 3 contexts × depths 0–10 × 2 seeds. 2b reruns the same 60 questions with compressors that never see the question.
- **Stats unit:** question, n=30 per dataset

### 2a · Qwen replications
- **Qwen3 8B:** 10 Q/dataset (from 40 candidates), 1 seed, depths 0/1/3/5, no judge
- **Qwen3 32B:** 30 Q/dataset (from 100), 2 seeds, depths 0–10, judge
- The question sets differ from Llama's because C1 is model-specific, so the results are **not paired** with Exp 2.

### 3 · Retrieval quality
- **Data:** MS MARCO v2.1, **20 queries**, 10 passages per context
- **Construction:** each context keeps all, half, or 3 of the 22 BM25-findable gold passages (Recall@10 = 1.00/0.50/0.14). The rest are filled with BM25-hard negatives (top 50) or easy ones (bottom 1,000), drawn from a 4,000-query pool. GPT-4o-mini screened 1,200 candidates and only `IRRELEVANT` ones were kept.
- **Models:** Llama 3.1 8B + judge
- **Scale:** 6 arms × depths 0/1/3/5 (120 packs), 1 seed
- **Stats unit:** query, n=20

### 4 · Redundant-evidence signal ratio
- **Data:** SQuAD, **20 questions**, 10 passages per context
- **Construction:** a "gold" passage is the answer paragraph plus a different paragraph from the same article. Contexts hold 10/5/1 gold passages plus 0/5/9 cross-article distractors, either BM25-hard (top 60) or easy (bottom 1,000). Distractors are alias-filtered and screened by GPT-4o-mini.
- **Models:** Llama 3.1 8B + judge
- **Scale:** 6 arms × depths 0/1/3/5 (120 packs), 1 seed
- **Stats unit:** question, n=20

### 5 · Question conditioning on SQuAD A/B pairs (+ 5a rewriting ladder)
- **Separate passages:** **20 pairs**. Each context is A's gold passage + B's gold passage (from different articles) + 8 distractors.
- **Same passage + distractors:** **20 paragraphs** (712–878 chars) × 2 human questions, with 9 screened distractors. Gold position is stratified.
- **Same passage, gold-only:** **10 paragraphs** (721–866 chars) × 2 questions = 20 Q. No distractors.
- **Pair construction:** human-written SQuAD questions with question Jaccard ≤ 0.10. GPT-4o-mini audits A/B independence, and both questions must pass C1 against Llama 3.1 8B (220 probed, 84–85 leaked and dropped).
- **5a:** the same 10 gold-only pairs with 4 arms (pass-through, paraphrase, generic, conditioned), 1 seed
- **Models:** Llama 3.1 8B + judge (plus preservation judge in 5a). Depths 0–10.
- **Stats unit:** pair, n=20 / 20 / 10

### 5b · Fictional fixed-capacity replication
- **Data:** **20 fictional dossiers** from Exp 9, each projected into 6 evidence cards and 6 questions (120 Q)
- **Scale:** 6 A-rotations per dossier (120 rotations) × K=2/4/6 × generic/conditioned. All 5 hidden questions are scored on every packet. 418/420 selector calls valid.
- **Models:** Llama 3.1 8B as selector and answerer + judge
- **Stats unit:** dossier, n=20 (rotations clustered)

### 6 · Multilingual handoffs
- **Data:** the Exp 5 gold-only set, **10 passages / 20 questions**
- **Scale:** 6 languages (en/de/fr/it/pt/es) × conditioned/generic × fixed/switching, depths 1–6
- **Models:** Llama 3.1 8B + judge + GPT-4o-mini language auditor
- **Stats unit:** passage, n=10

### 7 · Incremental-evidence chain
- **Data:** MuSiQue-Answerable, **30 questions** (100 candidates, C1-filtered against Llama 3.3 70B). Each question's **2–4 supporting paragraphs** arrive one packet at a time, in counterbalanced order.
- **Scale:** 0/1/3/5 relays × question shown/omitted × 2 seeds, plus hidden sub-question probes
- **Models:** Llama 3.3 70B + judge
- **Stats unit:** question, n=30

### 8 · Model heterogeneity
- **Data:** the Exp 5 same-passage set, **20 passages / 40 questions**, 10 passages per context
- **Chain models:** small tier Llama 3.1 8B, Qwen3 8B, Ministral 8B, Gemma 3 12B; large tier Llama 3.3 70B, Qwen3 32B, Mistral Small 3.2 24B, Gemma 3 27B
- **Answerers:** Llama 3.1 8B (primary), Ministral 8B (secondary), plus answer and preservation judges
- **Scale:** 17 arms + 2 derived controls × depths 0–6, 1 seed
- **Caveat:** Gemma 12B returned empty output on 13 of the 20 stage-1 prompts
- **Stats unit:** pair, n=20

### 8b · Selector/relay bottleneck
- **Data:** **20 fictional dossiers** (Exp 9) × 4 A→B rotations = 80 tasks. The channel narrows from 6 cards to 3, then to 1.
- **Models:** selectors and relays Llama 3.1 8B/3.3 70B and Qwen3 8B/32B (2×2 within each family). Every answer comes from a fixed Mistral Small 3.2 24B reader. Judge.
- **Stats unit:** dossier, n=20. The clean-omission subset is 51 Llama + 47 Qwen tasks.

### 9 · Size adaptation
- **Data:** **46 items**: 26 counterfactual (20 `known_entity` + 6 `rewritten_wikipedia`) + 20 fictional. Each item has 1 target question and 5 side facts (see creation below).
- **Models:** Llama 3.3 70B as writer and answerer. Answer, preservation and claim-support judges.
- **Scale:** 9 directive arms × depths 0–3, 12k-token output cap
- **Stats unit:** item, analysed per dataset (n=26 and n=20)

### 10 · Communication regret
- **SQuAD groups:** **24 paragraphs** (800–1,473 chars) × 4 human questions = 96 Q. 6 policies × budgets of 20/40/80/160 words give **1,440 messages and 7,104 answers**.
  - *Construction:* 200 candidate paragraphs, 6 questions each. C1 against Llama 3.1 8B removed 318 questions, leaving 58 groups with ≥4 clean questions. The GPT-4o-mini audit (answerable, distinct facts) rejected 1 group, and 24 were kept.
- **10a relation dossiers:** **16 fictional dossiers** × 16 Q = 256 Q. 4 policies × 4 budgets × 4 anchor rotations give **640 messages and 11,264 answers**.
- **Models:** Llama 3.1 8B as sender and answerer + judge
- **Stats unit:** context, n=24 (SQuAD) and n=16 (dossiers)

### 11 · Preference frontier
- **Data:** the Exp 10 corpora (24 SQuAD groups, 16 relation dossiers)
- **Models:** Llama 3.1 8B writer and reader for three runs: the primary run (SQuAD at 40/80 words, dossiers at 40/80/160), a free-form λ probe (dossiers, 80 words), and 3 sampled-seed replicates at T=0.2 (SQuAD, 80 words). Ministral-3-8B-2512 writer and reader (dossiers, 40/80 words) with matched Exp 10 controls. Judge.
- **Scale:** 5-point α (allocation) / λ (objective) grid
- **Stats unit:** context, n=24 or n=16

### 12 · Anticipatory context management
- **Data:** **16 relation dossiers** (256 Q, 264 sentences: 191 labelled with one aspect, 73 background)
- **Scale:** 4 rotations × 16 policies × budgets of 40/80 words = **2,048 messages and 32,768 answers**
- **Models:** Llama 3.1 8B as sender and answerer, Okapi BM25 store (k=2), judge
- **Stats unit:** context, n=16

### 13 · Latent (non-language) handoff: **not run**
- **Planned model:** Qwen3-4B (pinned revision, local GPU, thinking off), used as both writer and reader
- **Panel A:** the 20 fictional dossiers, turned into 140 K=4 packets using the Exp 5b selections. 3 channels × 6 Q = 2,520 evaluations.
- **Panel B:** the 16 relation dossiers
- **Confirmation set:** ≥100 fresh sources, not yet built
- Only offline and fake-backend checks exist so far; there are no inference results.

### 14 · Compression mechanism
- **Data:** **16 relation dossiers** (389–517 words, 256 Q), 64 rotations
- **Scale:** 9 arms (plus `reusable`/`oracle` imported from 10a) × budgets of 20/40/80/160 words = **1,472 messages and 11,840 judged answers**. Of these, 640 messages and 11,264 answers were reused from Exp 10a.
- **Models:** Llama 3.1 8B (abstractive arms and answerer), GPT-2 small (pinned commit `607a30d7`, CPU) as the sentence scorer, BM25 and TF-IDF (no model), judge
- **Stats unit:** context, n=16. The 20-word rung is excluded from the claims.

## How the synthetic datasets were created

### Fictional dossiers: `data/size_adaptation/fictional_items.jsonl` (Exp 9; reused by 5b, 8b, 13)
- GPT-5.2 wrote articles about invented places, people and institutions. 24 pages were built and **20 kept**, 2,021–3,155 chars each.
- Each item has 1 target question plus 5 side facts. The facts are checked verbatim against the document and do not overlap each other.
- A closed-book check against Llama 3.3 70B confirmed none are answerable from memory.
- For 5b and 8b, `src/fictional_qa.py` turns each item into 6 evidence cards with no new generation. Controls: full-card accuracy 1.000, closed-book 0.000.

### Counterfactual items: `data/size_adaptation/counterfactual_items.jsonl` (Exp 9)
- Each document deliberately contradicts a fact the model has memorised. Both directions are verified against Llama 3.3 70B: the model must recall the original answer (≥2/3 closed-book samples) and must never produce the replacement.
- **`known_entity` (20):** famous subjects were sampled first and verified as known. GPT-5.2 then rewrote one fact to a plausible new value. Short passages, about 1,100 chars on average.
- **`rewritten_wikipedia` (6):** taken from the random-Wikipedia counterfactual set below. Only 6 of those 20 passed the "knows original" gate. Long passages, about 7,500 chars on average.
- GPT-5.2 extracted the 5 side facts per item. 20 candidates were rejected overall.

### Random Wikipedia + counterfactual rewrite: `data/wikipedia_random*` (source for Exp 9)
- 10 random English Wikipedia pages (4k–40k chars, revision-pinned). GPT-5.2 wrote 2 unrelated, source-grounded questions per page with verbatim evidence, and a second pass wrote gold answers and aliases, giving **20 Q**.
- **Counterfactual version:** GPT-5.2 changed each page's evidence fact to a different plausible value and rewrote the article to stay consistent. 19/20 questions survive C1 against Llama 3.3 70B, versus 14/20 for the originals.
- `config.yaml` currently points Exp 1/2 at this set, but no reported result uses it.

### Relation dossiers: `data/communication_regret/relation_dossiers.jsonl` (Exp 10a; reused by 11, 12, 13, 14)
- GPT-5.2 wrote fictional dossiers to a strict schema, 2,407–3,101 chars as kept.
- Each dossier covers 4 aspects (`founding`, `facilities`, `finance`, `custom`). Each aspect has 4 questions:
  - an anchor question
  - a paraphrase (same answer, reworded)
  - a second fact about the same entity
  - a fact about a different entity in the same aspect

  That gives 16 questions per dossier. Every answer is verified verbatim in the text. Questions from the other aspects count as orthogonal.
- **Filtering:** 40 pages were requested and 38 built. A dossier was dropped if *any* of its 16 questions was answered closed-book by Llama 3.1 8B. 12 were dropped and **16 kept**.
- Direct-context ceiling 0.992, closed-book 0.000. Build cost $2.52.

## Runs on disk that are not in the report

- `slack_facts`: SQuAD, 20 packs, k=2/5/10 facts + 0/5/10 added filler passages, Llama 3.1 8B
- `slack_retrieval`: MS MARCO, 20 queries, 3 essential + 0/3/6 inert or near-miss filler passages, Llama 3.1 8B
- `squad_same_passage_matched`: the Exp 5 10-pair set with a 400-word length target
- `multilingual_handoff`: the Exp 6 design with distractors, superseded by the gold-only version
