# Agent handoff information-loss experiments

- **Status:** pilot-scale evidence through 24 August 2026
- **Primary metrics:** exact match (EM) and SQuAD-style token F1
- **Secondary metric:** binary answer correctness from `openai/gpt-4o-mini` (temperature 0), a different model family from the systems under test
- **Scope:** seven experiments on compression, repeated communication, retrieval quality, task conditioning, language switching, and incremental evidence acquisition

## Executive summary

The experiments do **not** support a universal rule that every handoff causes a fixed amount of accuracy loss. Handoffs can remove noise, change wording without changing the asserted fact, or selectively discard information that is irrelevant to the current task but valuable later.

The clearest findings are:

1. **Compression can denoise.** A free-form MuSiQue handoff improved F1 over direct full-context answering by +0.264 [95% CI +0.010, +0.525] in the initial `n=10` pilot. Longer chains also improved on some noisy full contexts.
2. **Token drift is not necessarily factual loss.** HotpotQA gold-only F1 fell by 0.107 after ten handoffs, but judged correctness stayed at 0.900. Repeated rewriting often changes answer form while preserving meaning.
3. **Task information is powerful but narrows reuse.** Hiding the question from compressors severely hurts noisy-context chains. Conversely, conditioning on Question A can remove facts needed for a held-out Question B, even when both facts occur in one short gold passage and no compression is necessary.
4. **The narrowing occurs during selection, not ordinary rewriting.** Pass-through and paraphrase-only controls show no reliable held-out judge loss at the main landmark depths. Adding question-conditioned selection causes a large, persistent held-out loss.
5. **Retrieval quality dominates weak distractor effects.** Retaining less labelled evidence reliably lowers QA performance. After candidate distractors are independently screened for relevance, BM25-hard versus BM25-easy differences no longer reliably exclude zero.
6. **Language switching is unresolved, not harmful by default.** In the corrected gold-only multilingual pilot, every fixed-versus-switching F1 and judge interval includes zero.
7. **New evidence can compensate for relay loss.** In the incremental-evidence chain, adding 1/3/5 relay-only agents between specialists does not reliably reduce final QA relative to zero relays. Specialist re-encoding can offset communication loss, so depth must be interpreted together with the evidence-acquisition schedule.

These are pilot results. Most conditions contain 10–30 examples, and bootstrap intervals are exploratory rather than multiplicity-corrected.

## How to read the report

- **Change** identifies the experimental variable added relative to the preceding design.
- **Key result** states the narrow conclusion supported by the data.
- Bracketed ranges are paired 95% bootstrap intervals unless stated otherwise.
- EM/F1 remain the deterministic primary metrics. The judge is a semantic complement, not a replacement.
- An interval containing zero is treated as inconclusive, even when the point estimate is large.
- Full prompts are in [PROMPTS.md](../PROMPTS.md); detailed tables and answer-level records are linked under [Result artifacts](#result-artifacts).

## Experiment inventory

| ID | Experiment | Dataset and sample | System model | Main comparison |
|---:|---|---|---|---|
| 1 | Single handoff mechanisms | MuSiQue, `n=10` | Llama 3.3 70B | Direct context vs free-form handoff vs oracle evidence |
| 2 | Fixed-evidence serial chain | MuSiQue + HotpotQA, `n=30`/dataset, two seeds | Llama 3.3 70B | Context noise × depths 0–10 |
| 2a | Qwen replications | MuSiQue + HotpotQA, `n=10` (8B) and `n=30` (32B) | Qwen3 8B/32B, non-thinking | Cross-model directional replication |
| 2b | Question-omission replication | Same `n=30` Llama sample | Llama 3.3 70B | Identical chain with compressor question block removed |
| 3 | Retrieval-quality propagation | MS MARCO, `n=20` | Llama 3.1 8B | Gold recall × screened BM25-hard/easy candidates |
| 4 | Fixed-width evidence redundancy | SQuAD packs, `n=20` | Llama 3.1 8B | 10/5/1 answer-sufficient passages in ten-document contexts |
| 5 | Cross-question generalization | SQuAD A/B pairs, `n=10–20` | Llama 3.1 8B | Question-conditioned vs generic summaries; evaluate target A and held-out B |
| 5a | Rewriting-selection ladder | Same gold-only `n=10` pairs | Llama 3.1 8B | Pass-through → paraphrase → compression → question selection |
| 6 | Multilingual handoffs | Same gold-only `n=10` passages | Llama 3.1 8B | Fixed vs switching language; conditioned vs generic |
| 7 | Incremental-evidence chain | MuSiQue, `n=30`, two seeds | Llama 3.3 70B | 0/1/3/5 evidence-free relays between packet specialists |

## 1. Single-handoff mechanism pilot

> **Model:** Llama 3.3 70B Instruct · **Dataset:** MuSiQue-Answerable validation, `n=10` · **Configuration:** one handoff; full supplied context vs free-form handoff vs oracle gold evidence

> **Change:** Insert one sealed evidence handoff before answering while holding retrieval fixed.

The answerer either saw the full supplied context (`A_full`), a free-form summary from a subagent (`B_freeform`), or deterministic gold evidence sentences (`E_oracle`). Structured and extractive mechanisms are implemented but were not included in this API pilot.

| Mechanism | EM | F1 |
|---|---:|---:|
| Direct full context | 0.400 | 0.484 |
| Free-form handoff | 0.600 | 0.748 |
| Oracle evidence | 0.600 | 0.740 |

> **Key result:** Free-form handoff minus direct context was **+0.264 F1** [+0.010, +0.525], `p=0.0304`. The handoff performed approximately as well as oracle evidence, consistent with useful evidence selection or denoising.

**Boundary:** `n=10`; the direction is informative, but the magnitude is unstable.

**Audit:** reported API cost $0.0151. Prompts and mechanisms: [PROMPTS.md § Experiment 1](../PROMPTS.md#experiment-1).

## 2. Fixed-evidence repeated handoffs

> **Model:** Llama 3.3 70B Instruct · **Dataset:** MuSiQue-Answerable + HotpotQA distractor validation, `n=30`/dataset, two seeds · **Configuration:** fixed evidence; gold-only, five-document, or full context; depths 0–10; question shown at every compressor

> **Change:** After one compressor sees the source documents, up to ten later agents receive only the sealed previous message and the original question. No new evidence enters the chain.

Each dataset used gold-only evidence, a five-document medium context, and the full dataset context. Examples, evidence, order, model, budgets, and two seeds were fixed across depths.

![Repeated handoff degradation](chain/degradation.png)

| Dataset | Context | Depth-10 F1 change | Judge change |
|---|---|---:|---:|
| MuSiQue | Gold-only | −0.027 | +0.017 |
| MuSiQue | Five documents | −0.025 | −0.033 |
| MuSiQue | Full context | +0.080 | −0.017 |
| HotpotQA | Gold-only | **−0.107** | **0.000** |
| HotpotQA | Five documents | −0.035 | −0.050 |
| HotpotQA | Full context | +0.014 | −0.033 |

> **Key result:** No condition loses more than 0.05 judged accuracy over ten handoffs. The largest F1 decline—HotpotQA gold-only, −0.107—is absent under semantic judging: correctness is 0.900 at both depths 0 and 10.

Repeated handoffs cause **surface-form drift**, visible in F1, while most answer facts survive. Noisy contexts can also improve because the model has distractors to remove; HotpotQA full context shrinks from about 5,530 characters at depth 0 to 720 after one handoff and 413 by depth 10.

**Boundary:** The judge does not prove perfect preservation; it shows that the strongest token-overlap loss is not corroborated as factual loss at `n=30`.

**Audit:** confirmed depth-extension cost $0.2436.

### 2a. Qwen replications

> **Model:** Qwen3 8B and Qwen3 32B, non-thinking · **Dataset:** MuSiQue-Answerable + HotpotQA distractor validation · **Configuration:** same three fixed-evidence contexts; 8B uses `n=10`/dataset, one seed, depths 0/1/3/5; 32B uses `n=30`/dataset, two seeds, depths 0–10

> **Change:** Rerun the chain with non-thinking Qwen models. Model-specific leakage filtering means the retained question IDs need not match Llama's.

![Qwen3 8B replication](chain_qwen/degradation.png)

![Qwen3 32B replication](chain_qwen32/degradation.png)

| Model | Sample/depth | Main result |
|---|---|---|
| Qwen3 8B | `n=10`/dataset, one seed, depth 5 | HotpotQA F1: −0.110/−0.086/−0.108 across gold/medium/full; MuSiQue: +0.127/+0.077/+0.117. Every interval includes zero. |
| Qwen3 32B | `n=30`/dataset, two seeds, depth 10 | No reliable negative F1 change. MuSiQue full improves **+0.182 F1** [+0.064, +0.312] and **+0.167 judge** [+0.017, +0.317]. |

> **Key result:** The replications support context-dependent denoising more strongly than universal serial degradation. They are not paired model comparisons because filtering is model-specific.

**Audit:** Qwen3 8B cost $0.0443. Qwen3 32B cost $0.3866 plus $0.0088 for judging.

### 2b. Matched question-omission replication

> **Model:** Llama 3.3 70B Instruct · **Dataset:** same MuSiQue + HotpotQA `n=30` samples as Experiment 2, two seeds · **Configuration:** identical gold-only/five-document/full contexts and depths 0–10; the final answerer sees the question, but no compressor does

> **Change:** Remove only the final-question block from every compressor. The final answerer still sees the question; all other settings match the main Llama experiment.

![Question-conditioned vs question-omitted](chain_generic/conditioning_comparison.png)

| Dataset | Context | Conditioned F1 Δ | Omitted F1 Δ | Conditioned judge Δ | Omitted judge Δ |
|---|---|---:|---:|---:|---:|
| MuSiQue | Gold-only | −0.027 | −0.035 | +0.017 | −0.033 |
| MuSiQue | Five documents | −0.025 | −0.127 | −0.033 | −0.183 |
| MuSiQue | Full context | **+0.080** | **−0.272** | −0.017 | **−0.467** |
| HotpotQA | Gold-only | −0.107 | −0.247 | 0.000 | −0.117 |
| HotpotQA | Five documents | −0.035 | −0.218 | −0.050 | −0.250 |
| HotpotQA | Full context | +0.014 | **−0.326** | −0.033 | **−0.400** |

> **Key result:** Omitting the question is worse in all six cells, with the largest gaps in full contexts. Unlike the main chain's F1-only drift, the judge corroborates this loss.

The pattern is consistent with task-guided denoising: a compressor cannot reliably separate signal from distractors without knowing the downstream task. The complementary risk—conditioning too narrowly—is tested in Experiment 5.

**Audit:** generation cost $0.9564; judge pass $0.0105.

## 3. Retrieval quality through repeated handoffs

> **Model:** Llama 3.1 8B Instruct · **Dataset:** MS MARCO QA v2.1 validation, `n=20`, one seed · **Configuration:** ten passages/context; all, half, or 3/22 BM25-findable gold passages; screened BM25-top hard versus BM25-bottom easy candidates; depths 0/1/3/5

> **Change:** Hold context width at ten passages while varying retained BM25-findable gold recall and screened BM25-hard versus BM25-easy candidate distractors.

The active MS MARCO design retains all, half, or 3/22 of the findable gold passages across 20 queries (Recall@10 = 1.000/0.500/0.136). Of 1,200 hard/easy candidates, only those independently judged `IRRELEVANT` were eligible. This reduces relevance contamination but is not human verification.

![Retrieval quality propagation](retrieval_quality/n20/retrieval_quality.png)

| Gold retention | Hard F1 d0 → d5 | Easy F1 d0 → d5 | Hard judge d0 → d5 |
|---|---:|---:|---:|
| All gold | 0.443 → 0.465 | 0.538 → 0.449 | 0.75 → 0.85 |
| Half gold | 0.380 → 0.361 | 0.353 → 0.318 | 0.50 → 0.60 |
| 3 of 22 gold | 0.189 → 0.166 | 0.157 → 0.141 | 0.15 → 0.30 |

> **Key result:** Low-minus-high hard-arm F1 is +0.254 [+0.132, +0.387] at depth 0 and +0.299 [+0.151, +0.459] at depth 5. After screening, every hard-minus-easy comparison at depths 3 and 5 includes zero.

Retrieval quality remains a persistent determinant of QA, but the active run does not support an independent effect of BM25 candidate difficulty. The earlier unscreened interpretation is retired.

**Boundary:** Cross-query candidates lack exhaustive human non-relevance labels. `n=20`, one seed.

**Audit:** cost $0.006587.

## 4. Fixed-context redundant-evidence signal ratio

> **Model:** Llama 3.1 8B Instruct · **Dataset:** SQuAD validation packs, `n=20`, one seed · **Configuration:** ten passages/context with 10 gold + 0 distractors, 5 gold + 5 distractors, or 1 gold + 9 distractors; screened hard/easy fillers; depths 0/1/3/5

> **Change:** Replace retrieval recall with a fixed-width redundancy manipulation: 10/5/1 answer-sufficient passages and 0/5/9 screened candidate distractors.

The gold variants share one answer-bearing SQuAD paragraph plus different same-article material. This controls answer sufficiency, not independent-source corroboration.

![Redundant evidence through handoffs](redundant_signal_ratio/n20/redundant_signal_ratio.png)

| Composition | Hard F1 d0 → d5 | Easy F1 d0 → d5 | Hard judge d0 → d5 |
|---|---:|---:|---:|
| 10 gold / 0 distractor | 0.896 → 0.801 | identical | 1.00 → 0.85 |
| 5 gold / 5 distractors | 0.887 → 0.891 | 0.921 → 0.847 | 0.95 → 1.00 |
| 1 gold / 9 distractors | 0.787 → 0.776 | 0.778 → 0.767 | 0.90 → 0.90 |

> **Key result:** Ten-gold minus one-gold F1 is +0.109 [+0.009, +0.235] at depth 0 but only +0.025 [−0.134, +0.180] at depth 5. Screened hard-minus-easy effects also include zero at the key later depths.

At `n=20`, redundancy provides a modest direct-answer advantage but no reliable protection after five handoffs. This is a null for shared-source redundancy, not evidence against independent corroboration.

**Audit:** cost $0.009382.

## 5. Question conditioning and cross-question generalization

> **Model:** Llama 3.1 8B Instruct · **Dataset:** SQuAD validation A/B pairs · **Configuration:** depths 0–10; question-conditioned versus generic summaries; separate-passage `n=20`, same-passage-with-distractors `n=20`, and same-passage-gold-only `n=10`

> **Change:** Evaluate every handoff on both the conditioning target (Question A) and a held-out but answerable Question B.

| Design | Context | Main result |
|---|---|---|
| Separate passages, `n=20` | A gold + B gold + 8 distractors | Early A benefit; persistent B loss |
| Same passage + distractors, `n=20` | 1 shared A/B gold + 9 distractors | Every A/B interval includes zero |
| Same passage, gold-only, `n=10` | 1 short shared A/B passage | No A benefit; large persistent B loss |

### Separate-passage design

![Separate-passage conditioning](summary_generalization_v2_depth10/n20/summary_generalization.png)

Conditioned summaries are 3–4 times shorter than generic summaries despite the same 700-token cap, indicating active task selection.

| Evaluation | Depth 1 | Depth 5 | Depth 10 |
|---|---:|---:|---:|
| Target A F1 | **+0.295** [+0.125, +0.476] | +0.024 [−0.156, +0.208] | −0.076 [−0.309, +0.160] |
| Held-out B F1 | **−0.392** [−0.600, −0.192] | **−0.323** [−0.540, −0.117] | **−0.497** [−0.700, −0.283] |
| Target A judge | **+0.400** [+0.200, +0.600] | +0.150 [−0.050, +0.350] | −0.100 [−0.350, +0.150] |
| Held-out B judge | **−0.400** [−0.600, −0.200] | **−0.400** [−0.600, −0.200] | **−0.450** [−0.650, −0.250] |

> **Key result:** Conditioning helps A for the first two handoffs, but not reliably by depth 3. Damage to B persists through depth 10 on both metrics.

### Same-passage controls

![Same passage with distractors](squad_same_passage/n20/summary_generalization.png)

With one shared gold passage among nine distractors, every conditioned-minus-generic A/B interval includes zero (`p>0.22`). Removing the competing document removes the clear effect.

![Gold-only conditioning](squad_same_passage_goldonly/n10/summary_generalization.png)

| Evaluation | Depth 1 | Depth 5 | Depth 10 |
|---|---:|---:|---:|
| Target A F1 | +0.096 [−0.137, +0.367] | +0.040 [−0.253, +0.333] | +0.040 [−0.180, +0.300] |
| Held-out B F1 | **−0.600** [−0.900, −0.300] | **−0.500** [−0.800, −0.200] | **−0.433** [−0.800, 0.000] |
| Held-out B judge | **−0.600** [−0.900, −0.300] | **−0.500** [−0.800, −0.200] | **−0.500** [−0.900, −0.100] |

At stage 1, conditioned summaries retain A in 8/10 and B in 3/10; generic summaries retain A in 8/10 and B in 9/10. No handoff hits the token guard.

> **Key result:** Conditioning narrows to the named task even when the source is one short passage and there is no noise or length pressure. This is distinct from discarding a competing document.

### 5a. Rewriting versus selection

> **Model:** Llama 3.1 8B Instruct · **Dataset:** same SQuAD gold-only A/B pairs, `n=10`, one seed · **Configuration:** one shared A/B gold passage, no distractors, depths 0–10; pass-through, paraphrase-only, generic compression, and question-conditioned selection

> **Change:** Add pass-through and paraphrase-only controls. The ladder is: no rewriting → rewriting without compression → generic compression → question-conditioned selection.

**Question visibility and evaluation.** Only the **conditioned** compressor receives Question A. The **generic**, **paraphrase**, and **pass-through** arms receive neither Question A nor Question B: paraphrase is explicitly instructed to restate every fact without shortening, selecting, or adding content; pass-through copies its input without an LLM call. At each depth, the same resulting handoff is then answered twice by the final answerer—once for target Question A and once for held-out Question B. Thus, B is an evaluation-only query, not information available while the handoff is written.

**How to read the figures.** The first figure is a *transition diagnostic*, not just an end-task comparison. From left to right, top to bottom, it shows: lexical overlap between a message and its immediate predecessor; an independent judge's semantic-preservation verdict for that rewrite; whether a gold-answer string for B is still present; and final answer accuracy for A. Edge 1 is source passage → first handoff; later edges are handoff → handoff. It therefore separates surface rewriting, judged meaning preservation, direct B-answer retention, and the answer consequence for the conditioned task.

**Operational definitions.** “Held-out fact B present” is a deterministic answer-string survival probe: it is 1 when any annotated gold answer or alias for B (excluding strings shorter than three characters) appears as a case-insensitive contiguous substring of the handoff, and 0 otherwise. It does **not** establish that the handoff semantically entails B, nor that it retained the entire supporting proposition; it tracks the observable answer-bearing span. “Lexical similarity” is the symmetric bag-of-words token F1 between consecutive messages. Before matching, both are lowercased, stripped of punctuation and articles (*a*, *an*, *the*), and whitespace-normalized; token multiplicities count. Thus, it measures shared normalized vocabulary rather than word order, syntax, or semantic equivalence.

![Per-edge rewriting diagnostics](squad_same_passage_paraphrase/n10/paraphrase_transitions.png)

The second figure is the *end-task view* of the same four arms. Its left column evaluates Question A and its right column evaluates held-out Question B; the top row is token F1 and the bottom row is LLM-judge accuracy. Depth 0 is direct access to the original passage, while depths 1–10 use only the handoff. Marker area represents the mean number of characters given to the answerer (larger markers are longer messages), not statistical uncertainty. Comparing columns tests whether a message useful for its conditioning question remains useful for an unprovided future query.

![Four-arm gold-only accuracy](squad_same_passage_paraphrase/n10/summary_generalization.png)

| Held-out B judge | Depth 1 | Depth 2 | Depth 5 | Depth 10 |
|---|---:|---:|---:|---:|
| Pass-through | 1.00 | 1.00 | 1.00 | 1.00 |
| Paraphrase-only | 0.90 | 1.00 | 0.90 | 0.90 |
| Generic compression | 1.00 | 1.00 | 1.00 | 0.90 |
| Question-conditioned | 0.40 | 0.30 | 0.60 | 0.40 |

At depths 1/2/5/10, paraphrase-minus-pass-through and generic-minus-pass-through judge intervals include zero. Conditioned-minus-generic judge is −0.600 [−0.900, −0.300], −0.700 [−1.000, −0.400], −0.400 [−0.700, −0.100], and −0.500 [−0.900, −0.100]. F1 shows the same selective-loss pattern.

Later messages are near copies (mean lexical similarity across stages 2–10: paraphrase 0.958, generic 0.907, conditioned 0.914). The fact is usually excluded during the first narrowing step and then faithfully propagated. Paraphrase judge accuracy remains 0.90–1.00 even when token F1 is lower.

> **Key result:** Ordinary rewriting and generic compression do not explain the large held-out loss. Question-conditioned selection does.

**Boundary:** `n=10`, one seed. The original gold-only run and four-arm follow-up are separate cached runs; the mechanistic conclusion uses within-follow-up contrasts.

**Audit:** separate-passage/depth extension about $0.0235; same-passage $0.031; gold-only $0.005; four-arm follow-up $0.036.

## 6. Multilingual fixed versus switching handoffs

> **Model:** Llama 3.1 8B Instruct · **Dataset:** same SQuAD gold-only A/B passages, `n=10` passages / 20 question IDs, one seed · **Configuration:** one shared A/B gold passage and no distractors; depths 1–6; conditioned/generic × fixed/switching schedule across six languages

> **Change:** Compare keeping one handoff language with rotating through English, German, French, Italian, Portuguese, and Spanish on the same gold-only A/B passages.

Fixed and switching schedules share stage 1; the treatment begins at stage 2. Final questions and answers remain English.

**How to read the figure.** Orange is question-conditioned and green is generic; solid lines keep one assigned language and dashed lines rotate languages. Both schedules share the stage-1 handoff, so only depths 2–6 test the language treatment. The columns compare target A with held-out B; the rows show token F1 and LLM-judge accuracy. Direct context at depth 0 is the shared baseline.

![Fixed versus switching language handoffs](multilingual_handoff_gold_only/n10/multilingual_handoffs_compact.png)

| Depth-6 switching − fixed F1 | Conditioned | Generic |
|---|---:|---:|
| Target A | −0.181 [−0.550, +0.183] | +0.061 [−0.225, +0.332] |
| Held-out B | +0.183 [0.000, +0.400] | −0.017 [−0.267, +0.183] |

Every F1 and judge interval from depths 2–6 includes zero. Language compliance is 237/240; no handoff reaches the token guard. Observed summaries average 583–804 characters, with generic summaries generally longer.

> **Key result:** No reliable cost or benefit from language switching is detected. This is not an equivalence result; `n=10` leaves wide intervals.

**Boundary:** Mostly high-resource Latin-script languages. Meta does not publish per-language pretraining shares. [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).

**Audit:** $0.0078 generation/answering, $0.0085 language auditing, $0.0026 judging.

## 7. Incremental-evidence handoff chain

> **Model:** Llama 3.3 70B Instruct · **Dataset:** MuSiQue-Answerable validation, `n=30`, two seeds · **Configuration:** 2–4 supporting paragraphs delivered one at a time; counterbalanced packet order; 0/1/3/5 evidence-free relays between specialists; question-conditioned versus question-omitted chains

> **Change:** Deliver one new MuSiQue supporting paragraph to each specialist while inserting 0/1/3/5 evidence-free relays between specialists.

The 30 examples contain 2–4 packets. Order is counterbalanced and fixed across conditions, depths, and two seeds. Relays receive only the sealed prior message. Conditions differ only in whether chain agents see the main question. Hidden decomposition probes measure packet-fact survival.

![Incremental evidence and relay-only transformation](incremental_chain/incremental_chain.png)

**How to read the figure.** The upper-left panel is final multi-hop token F1: its dotted line is direct access to all original evidence, and error bars are 95% bootstrap intervals. The upper-right panel is future-query regret, defined as original-evidence probe F1 minus final-handoff probe F1: positive values would mean the handoff is worse; values below zero mean probes were easier to answer from the handoff. In both panels, intervals overlap the relevant zero/baseline reference, so the figure does not establish a relay-depth effect.

The lower panels follow individual packet probes after their evidence first arrives. Their x-axis is **handoff age**, the number of later transformations the packet has experienced; colour gives relay depth (0, 1, 3, 5) and solid/dashed lines give question-conditioned/question-omitted chains. Lower-left is hidden-probe F1; lower-right is literal answer-string survival (a gold answer or alias appears in the message). These lines are descriptive, not a clean causal age curve: high ages contain only earlier packets, have fewer examples, and combine packet position with relay count. They show high observed retention, not proof that repeated relays are lossless.

| Relays | Conditioned F1 / judge | Question omitted F1 / judge |
|---:|---:|---:|
| 0 | 0.601 / 0.667 | 0.624 / 0.700 |
| 1 | 0.629 / 0.700 | 0.655 / 0.733 |
| 3 | 0.634 / 0.717 | 0.655 / 0.717 |
| 5 | 0.651 / 0.683 | 0.578 / 0.633 |

The complete-evidence baseline is F1/judge = 0.610/0.667. No relay-depth-minus-zero-relay F1 interval excludes zero. At five relays, the contrast is +0.050 [−0.050, +0.152] when conditioned and −0.046 [−0.174, +0.067] when omitted.

Future-query F1 regret—original evidence minus final handoff—is negative in every cell (−0.069 to −0.018), with every interval including zero. Specialists can make evidence easier to query without creating information. Literal answer-string survival in the five-relay conditioned chain is 0.986 at age 0 and about 0.91 at ages 11–12.

> **Key result:** Relay-only depth does not reliably degrade this incremental-acquisition system. Fresh specialist updates can compensate for communication loss; Experiment 2 remains the cleaner isolation of pure serial degradation.

**Boundary:** One model, `n=30`; high handoff age is correlated with early packet position and higher hop count.

**Audit:** generation/answering spend $0.6942; 7,019 live and 1,693 cached calls.

## Cross-experiment interpretation

| Mechanism | Supported conclusion | Boundary |
|---|---|---|
| Denoising | Compression can improve QA when contexts contain removable distractors. | Context- and model-dependent. |
| Surface drift | Rewriting can lower EM/F1 while preserving judged correctness. | Report both metrics; the judge may miss subtle errors. |
| Retrieval quality | Missing relevant evidence remains harmful through later handoffs. | BM25 rank is not a relevance label. |
| Task conditioning | A known question guides filtering but can remove facts needed later. | Target benefit may be brief; held-out harm depends on context. |
| Rewriting vs selection | Large held-out loss appears when question-conditioned selection is added. | Cleanest control has ten pairs and one seed. |
| Language | Switching among six languages has no detected effect. | Wide intervals do not establish equivalence. |
| Incremental acquisition | Packet-specific updates can offset relay loss. | Depth no longer isolates communication when evidence arrives. |

The systems implication is that a handoff can improve the current task while reducing future reuse. Evaluation should distinguish factual correctness, lexical form, task specificity, evidence acquisition, and fact age.

## Limitations

- All runs are pilots (`n=10–30`); several use one seed.
- Bootstrap intervals are exploratory and not multiplicity-corrected.
- The LLM judge is not human adjudication and may share benchmark knowledge.
- Public benchmarks risk pretraining exposure; leakage filters select harder, model-specific subsets.
- Candidate distractors in Experiments 3–4 are LLM-screened, not exhaustively human-labelled.
- Experiment 4 uses shared-source redundancy, not independent corroboration.
- Experiment 5's strongest control has ten pairs; summary length differs naturally across some arms.
- Experiment 6 covers six mostly high-resource languages with depth coupled to order.
- Experiment 7 correlates handoff age with packet position and hop count.
- Cross-model replications are not paired because leakage filtering is model-specific.

## Recommended next steps

1. Scale the gold-only A/B design and rewriting-selection ladder to at least 100 pairs and two seeds.
2. Add a second judge or human audit for major metric disagreements, especially HotpotQA gold-only and paraphrase-only chains.
3. Scale the incremental-evidence chain with independent evidence orders and within-packet age comparisons; add extractive/oracle specialist updates.
4. Scale retrieval and redundancy experiments, then test independently sourced answer-sufficient evidence.
5. Expand multilingual testing with balanced Latin-square orders and separately preregistered script-diverse languages.
6. Keep prompt-equivalence, isolation, truncation, and distractor-relevance audits as required controls.

## Result artifacts

- Single handoff: [`report.md`](report.md), [`summary.csv`](summary.csv), [`contrasts.csv`](contrasts.csv)
- Fixed-evidence chain: [`chain/report.md`](chain/report.md), [`chain/stage_metrics.csv`](chain/stage_metrics.csv), [`chain/degradation.png`](chain/degradation.png)
- Qwen3 8B: [`chain_qwen/report.md`](chain_qwen/report.md), [`chain_qwen/degradation.png`](chain_qwen/degradation.png)
- Qwen3 32B: [`chain_qwen32/report.md`](chain_qwen32/report.md), [`chain_qwen32/degradation.png`](chain_qwen32/degradation.png)
- Question omitted: [`chain_generic/report.md`](chain_generic/report.md), [`chain_generic/stage_metrics.csv`](chain_generic/stage_metrics.csv), [`chain_generic/conditioning_comparison.png`](chain_generic/conditioning_comparison.png)
- Retrieval: [`retrieval_quality/n20/metrics.csv`](retrieval_quality/n20/metrics.csv), [`retrieval_quality/n20/deltas.csv`](retrieval_quality/n20/deltas.csv), [`retrieval_quality/n20/retrieval_quality.png`](retrieval_quality/n20/retrieval_quality.png)
- Redundancy: [`redundant_signal_ratio/n20/metrics.csv`](redundant_signal_ratio/n20/metrics.csv), [`redundant_signal_ratio/n20/deltas.csv`](redundant_signal_ratio/n20/deltas.csv), [`redundant_signal_ratio/n20/redundant_signal_ratio.png`](redundant_signal_ratio/n20/redundant_signal_ratio.png)
- Cross-question, separate passages: [`summary_generalization_v2_depth10/n20/metrics.csv`](summary_generalization_v2_depth10/n20/metrics.csv), [`summary_generalization_v2_depth10/n20/deltas.csv`](summary_generalization_v2_depth10/n20/deltas.csv), [`summary_generalization_v2_depth10/n20/summary_generalization.png`](summary_generalization_v2_depth10/n20/summary_generalization.png)
- Same passage with distractors: [`squad_same_passage/n20/metrics.csv`](squad_same_passage/n20/metrics.csv), [`squad_same_passage/n20/deltas.csv`](squad_same_passage/n20/deltas.csv), [`squad_same_passage/n20/summary_generalization.png`](squad_same_passage/n20/summary_generalization.png)
- Gold-only: [`squad_same_passage_goldonly/n10/metrics.csv`](squad_same_passage_goldonly/n10/metrics.csv), [`squad_same_passage_goldonly/n10/deltas.csv`](squad_same_passage_goldonly/n10/deltas.csv), [`squad_same_passage_goldonly/n10/summary_generalization.png`](squad_same_passage_goldonly/n10/summary_generalization.png)
- Rewriting-selection ladder: [`squad_same_passage_paraphrase/n10/deltas.csv`](squad_same_passage_paraphrase/n10/deltas.csv), [`squad_same_passage_paraphrase/n10/transition_metrics.csv`](squad_same_passage_paraphrase/n10/transition_metrics.csv), [`squad_same_passage_paraphrase/n10/paraphrase_transitions.png`](squad_same_passage_paraphrase/n10/paraphrase_transitions.png)
- Multilingual: [`multilingual_handoff_gold_only/n10/metrics.csv`](multilingual_handoff_gold_only/n10/metrics.csv), [`multilingual_handoff_gold_only/n10/deltas.csv`](multilingual_handoff_gold_only/n10/deltas.csv), [`multilingual_handoff_gold_only/n10/diagnostics.csv`](multilingual_handoff_gold_only/n10/diagnostics.csv)
- Incremental evidence: [`incremental_chain/report.md`](incremental_chain/report.md), [`incremental_chain/stage_metrics.csv`](incremental_chain/stage_metrics.csv), [`incremental_chain/future_query_regret.csv`](incremental_chain/future_query_regret.csv), [`incremental_chain/survival_by_age.csv`](incremental_chain/survival_by_age.csv), [`incremental_chain/incremental_chain.png`](incremental_chain/incremental_chain.png)

Raw records remain under [`../runs/`](../runs/); retired exploratory variants remain on disk but are not used for the active conclusions.
