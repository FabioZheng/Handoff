# Agent handoff information-loss experiments

**Status:** pilot-scale evidence through 21 August 2026
**Primary metrics:** exact match (EM) and SQuAD-style token F1  
**Secondary metric:** LLM-judge answer correctness (`openai/gpt-4o-mini`, temperature 0, binary vs gold; a different model family from the systems under test)  
**Judging:** deterministic string metrics remain primary; an LLM judge is reported alongside them, never in place of them

## Executive summary

The evidence does **not** support a universal “each handoff loses a fixed amount of accuracy” rule. A handoff is both a lossy communication channel and a possible denoiser: whether it helps or harms depends on distractor load, the number of facts that must survive, and whether the next task is the one the summary was optimized for.

The current synthesis is:

1. **Compression can help the immediate task when it removes noise.** In the initial MuSiQue pilot, a free-form handoff beat direct full-context answering by 26.4 F1 points. In the longer chains, noisy full contexts often improved after early summaries, whereas compact HotpotQA gold-only evidence showed the clearest serial loss (−10.7 F1 by depth 10).
   **Metric caveat added 20 August 2026:** that −10.7 F1 result is **0.000** under an LLM judge scoring answer correctness against the gold (0.900 at both depth 0 and depth 10). Token F1 penalises correct-but-reworded answers, which repeated compression reliably produces. Read as "the asserted fact survives ten handoffs, but its surface form drifts from the gold string." No serial-loss claim in this report should now be made on token F1 alone.
2. **Distractor difficulty is a treatment, not a nuisance variable.** In matched reruns, BM25-bottom easy negatives widened the MS MARCO low–high-noise F1 gap (0.342 at depth 0 and 0.310 at depth 5) relative to BM25-top hard negatives (0.201 and 0.107). In the fixed-ten-passage SQuAD experiment, hard versus easy made no difference with no noise (all ten passages answer-sufficient), but the medium-noise (5-gold/5-distractor) hard arm retained substantially more judged correctness by depths 3–5. The direction changes with retained signal, so neither negative class can stand in for “irrelevant evidence” generally.
3. **Question conditioning is a short-horizon specialization tool, not a reusable-handoff default — and its size depends on whether the compressor can drop a whole document.** With A and B on **separate** gold passages, conditioning improved its target by +29.5 F1 at the first handoff and +21.8 at the second, indistinguishable from zero from depth 3 on, while damaging an unrelated but answerable question at every depth (−49.7 F1 at depth 10; the LLM judge strengthens this, −40 to −45 points at every depth, all p < 0.001). A controlled rebuild in which **one shared passage answers both questions** (§5, 20 pairs, length-matched, position-stratified, leakage-filtered) finds the target benefit essentially gone (+0.06 F1 at depth 1, n.s.) while the held-out cost stays negative at every depth on both metrics (−0.11 to −0.20) but not significant at n=20. Read together: conditioning pays when it lets the compressor discard entire irrelevant documents, not when it merely re-weights within one passage — and the held-out cost appears in both designs. For five or more handoffs with a potentially changing downstream task, generic notes remain the safer default.
   **Withdrawn 21 August 2026:** an earlier self-generated-Wikipedia variant of this comparison reported a large, durable target benefit and no held-out damage. It was confounded — its generic arm hit the token cap on 10/10 summaries and retained the gold fact 0/10 times — and has been deleted and replaced by the rebuild above. No conclusion in this report now rests on it.
4. **Withholding the question from every compressor is far more damaging than repeated compression itself.** The matched question-omission replication (same model, questions, contexts, depths, seeds, system prompt, and handoff instructions as the main serial chain — the sole difference is that no compressor ever sees the question) degrades at every depth, in every evidence condition, including full context — the one condition that *improved* under question-conditioned compression. MuSiQue full context goes from +0.080 F1 at depth 10 (conditioned) to **−0.272 F1** (question omitted) on the identical question set.
5. **These are directional pilots, not final effect sizes.** The Llama 3.3 70B and Qwen3 8B chain runs agree that dataset and context composition matter, but most low-cost experiments use 20 questions and one seed. The strongest next step is replication at larger sample sizes with independently sourced redundant evidence.

## Experiment inventory

| Experiment | Dataset | Sample | Model | Handoff depths | Input context at depth 0 | Main comparison |
|---|---|---:|---|---|---|
| Single-handoff mechanisms | MuSiQue | 10 | Llama 3.3 70B Instruct | 0/1 | `A_full`/`B_freeform`: full supplied MuSiQue context; `E_oracle`: gold evidence sentences only (no noise) | Full context vs free-form handoff vs oracle evidence |
| Serial handoff degradation | MuSiQue + HotpotQA | 30 per dataset | Llama 3.3 70B Instruct | 0–10 | No noise: gold-only; medium noise: all gold plus filler to five documents; high noise: complete dataset context | Context composition × chain depth |
| Serial handoff degradation, question omitted | MuSiQue + HotpotQA | 30 per dataset | Llama 3.3 70B Instruct | 0–10 | Same no-/medium-/high-noise inputs as above | Same chain; only the compressor question block is omitted |
| Serial handoff degradation (replication) | MuSiQue + HotpotQA | 10 per dataset | Qwen3 8B, non-thinking | 0/1/3/5 | Same no-/medium-/high-noise inputs as above | Low-cost one-seed replication |
| Retrieval-quality propagation | MS MARCO QA v2.1 | 20 | Llama 3.1 8B Instruct | 0/1/3/5 | Low noise: all retriever-found gold retained + filler; medium noise: half retained + filler; high noise: 15% retained + filler. Every context has ten passages. | Noise level × BM25-top hard or bottom easy candidate distractors |
| Redundant-evidence signal-ratio propagation | SQuAD same-article packs | 20 | Llama 3.1 8B Instruct | 0/1/3/5 | No noise: 10 answer-sufficient gold / 0 distractor; medium: 5 / 5; high: 1 / 9 | Signal/noise ratio × BM25-top hard or bottom easy candidate distractors |
| Cross-question generalization v2, depth-10 extension | SQuAD | 20 paired contexts | Llama 3.1 8B Instruct | 0–10 | High noise: Question A gold + unrelated Question B gold + 8 distractors (both questions remain answerable) | Question A present vs absent; evaluate A and unrelated B at every depth |
| Cross-question generalization, same-passage A/B **(replaces a withdrawn Wikipedia variant — see §5)** | SQuAD same-passage pairs | 20 paired contexts | Llama 3.1 8B Instruct | 0–10 | High noise: one shared gold SQuAD passage (answers both A and B) at a stratified position + 9 length-matched SQuAD distractors | Question A present vs absent; evaluate A and B at every depth |

## 1. Single-handoff mechanism pilot

> **Model:** `meta-llama/llama-3.3-70b-instruct` · **Dataset:** MuSiQue-Answerable (validation) · **Prompts:** [PROMPTS.md § Experiment 1](PROMPTS.md#experiment-1) — `ANSWER_SYSTEM`, `SUBAGENT_SYSTEM`, `FREEFORM_INSTRUCTION`, `STRUCTURED_INSTRUCTION`, `EXTRACTIVE_INSTRUCTION` (`src/handoffs.py`)

### Design

The original probe isolates what happens when evidence is summarized once before a final answerer sees it. Retrieval is held constant. Five mechanisms are implemented:

- `A_full`: one model answers directly from all supplied evidence.
- `B_freeform`: a subagent writes a prose handoff; a fresh answerer sees only that handoff and the question.
- `C_structured`: the handoff is a structured claim representation.
- `D_extractive`: the handoff contains selected verbatim evidence sentences.
- `E_oracle`: the answerer receives deterministically derived gold evidence sentences.

The real 10-question pilot ran `A_full`, `B_freeform`, and `E_oracle`; `C_structured` and `D_extractive` are implemented and offline-tested but were not included in this API pilot.

### Results

| Mechanism | EM | Token F1 | Interpretation |
|---|---:|---:|---|
| `A_full` | 0.400 | 0.484 | Direct answer from full context |
| `B_freeform` | 0.600 | 0.748 | Free-form handoff improved F1 by 0.264 |
| `E_oracle` | 0.600 | 0.740 | Approximately tied with free-form handoff |

The paired `B_freeform − A_full` F1 contrast was +0.264 with a 95% bootstrap interval of +0.010 to +0.525 (`p = 0.0304`). Because `n = 10`, the magnitude is uncertain. The result suggests that a handoff can act as evidence selection or denoising rather than merely destroying information.

Reported API cost: **$0.0151**.

## 2. Repeated handoff degradation

> **Model:** `meta-llama/llama-3.3-70b-instruct` · **Dataset:** MuSiQue-Answerable + HotpotQA (distractor, validation) · **Prompts:** [PROMPTS.md § Experiment 2](PROMPTS.md#experiment-2) — `QUESTION_CONDITIONED_SYSTEM`/`_INITIAL_INSTRUCTION`/`_RECOMPRESS_INSTRUCTION` + shared `ANSWER_SYSTEM` (`src/run_chain.py`)

### Design

The same question was supplied at every compression stage:

`documents → summary 1 → summary 2 → … → summary 10 → fresh answerer`

Only the first compressor could see source documents. Later compressors received a sealed previous handoff and the original question. Two seeds were run. Each dataset used three evidence conditions:

- **No noise (gold-only; formerly “Short”):** all and only gold documents (mean 2.7 documents for MuSiQue; 2 for HotpotQA).
- **Medium noise (formerly “Medium”):** all gold documents plus distractors up to five documents.
- **High noise (full; formerly “Full”):** the complete dataset context (20 MuSiQue paragraphs; 10 HotpotQA paragraphs).

### Results

![Repeated handoff degradation across datasets and context lengths](chain/degradation.png)

Depth-10 change from direct depth 0:

| Dataset | Context | Token F1 change | LLM-judge change |
|---|---|---:|---:|
| MuSiQue | No noise (gold-only) | −0.027 | +0.017 |
| MuSiQue | Medium noise | −0.025 | −0.033 |
| MuSiQue | High noise (full) | +0.080 | −0.017 |
| HotpotQA | No noise (gold-only) | **−0.107** | **+0.000** |
| HotpotQA | Medium noise | −0.035 | −0.050 |
| HotpotQA | High noise (full) | +0.014 | −0.033 |

BERTScore was replaced by an LLM judge (`openai/gpt-4o-mini`, temperature 0, binary correct/incorrect against the gold answer, deliberately a different model family from the llama systems under test). EM and token F1 remain primary and are reported unchanged alongside it.

**The judge materially changes the headline reading of this experiment.** The single largest serial-degradation signal in the probe — HotpotQA gold-only, −10.7 token-F1 points by depth 10 — is **0.000** under the judge: 0.900 correct at depth 0 and 0.900 at depth 10. Absolute levels also sit far above F1 (HotpotQA short holds ~0.90–0.92 across all ten handoffs; MuSiQue short ~0.73–0.75). That gap is the expected signature of token F1 penalising answers that are correct but reworded or more verbose, which is exactly what repeated compression produces. Under the judge no evidence variant loses more than 5 points across ten handoffs, and MuSiQue short is numerically flat-to-positive.

This does not erase the F1 result, and the two should be read together: F1 says the *surface form* of answers drifts steadily away from the gold string under repeated compression, while the judge says the *fact being asserted* mostly survives. The conclusion that gold-only evidence is the most fragile condition is supported by F1 but is not corroborated by the judge at this sample size.

The high-noise full-context conditions often improved after the first handoff, consistent with denoising. Summary length also collapsed rapidly: for example, HotpotQA full context averaged 5,530 characters at depth 0, 720 after one handoff, and 413 by depth 10.

The main conclusion is not that long contexts are inherently safer. Rather, high-noise full contexts provide an opportunity for useful selection, while already-minimal no-noise evidence has little redundancy and therefore exposes omissions more directly.

The confirmed incremental cost reported for extending the chains through depths 6–10 and filling missing answers was **$0.2436**.

### Low-cost Qwen replication

> **Model:** `qwen/qwen3-8b` (non-thinking, `reasoning.effort: none`) · **Dataset:** same as above, 10 questions/dataset · **Prompts:** identical to Experiment 2 above — same `run_chain.py` constants, different model config (`qwen_chain_config.yaml` + `qwen_chain_experiment.yaml`)

![Qwen3 8B repeated-handoff replication](chain_qwen/degradation.png)

The same dataset and evidence-variant design was rerun with Qwen3 8B in non-thinking mode, but with 10 questions per dataset, one seed, and depths 0/1/3/5. It broadly reproduces the dataset split in the original probe: at depth 5, HotpotQA loses F1 at every noise level (no noise −0.110, medium noise −0.086, high noise −0.108), while MuSiQue is stable-to-improved (no noise +0.127, medium noise +0.077, high noise +0.117). None of the depth-5 intervals exclude zero at this small sample size, so this is directional replication evidence rather than a conclusive cross-model comparison.

Reconciled cost was **$0.0443** across Qwen-specific leakage filtering, 300 summary calls, 240 answer calls, and a compatibility smoke call. The raw outputs and full table are in [`chain_qwen/report.md`](chain_qwen/report.md).

### Matched question-omission replication

> **Model:** `meta-llama/llama-3.3-70b-instruct` (same as above) · **Dataset:** same as above, same 30-question sample · **Prompts:** [PROMPTS.md § Experiment 2](PROMPTS.md#experiment-2), question-conditioned prompt set with the question block deleted — see `initial_compress()`/`recompress()` in `src/run_chain.py`

![Question-conditioned vs question-omitted, matched chains](chain_generic/conditioning_comparison.png)

This is the minimal counterpart to the main serial chain: same Llama 3.3 70B model, fixed filtered question sample, all three evidence variants, depths 0–10, two seeds, system prompt, and handoff instructions. The sole difference is prompt visibility — neither the initial compressor nor any later compressor receives the final-question block; the fresh answerer still receives the question. An automated equivalence check (`prompt_difference_selftest` in the code) verifies that removing that block makes the two compressor prompts byte-identical, so the only variable is question visibility.

Depth-10 change from direct depth 0, question-conditioned vs question-omitted:

| Dataset | Context | Conditioned F1 | Omitted F1 | Conditioned judge | Omitted judge |
|---|---|---:|---:|---:|---:|
| MuSiQue | No noise (gold-only) | −0.027 | −0.035 | +0.017 | −0.033 |
| MuSiQue | Medium noise | −0.025 | −0.127 | −0.033 | −0.183 |
| MuSiQue | High noise (full) | **+0.080** | **−0.272** | −0.017 | **−0.467** |
| HotpotQA | No noise (gold-only) | −0.107 | −0.247 | +0.000 | −0.117 |
| HotpotQA | Medium noise | −0.035 | −0.218 | −0.050 | −0.250 |
| HotpotQA | High noise (full) | +0.014 | **−0.326** | −0.033 | **−0.400** |

Every condition is worse without the question, and the gap widens with more distractors: high-noise full-context degradation goes from mildly positive (conditioned) to the worst result in either chain (omitted). This is consistent with the denoising story in §2 — a compressor can only filter distractors *toward* a task it knows, and high-noise full context has the most distractor mass to filter. It also complements §5's finding below: §5 shows conditioning narrows a summary toward one task at the cost of others; this replication shows the opposite failure mode — a compressor with no task at all keeps too much noise and too little signal for any task.

**Judge update, 2026-08-20:** this replication now has real LLM-judge scores (added as a side effect of regenerating its plot with the new size-encoding — see §2 above). Unlike the main serial-degradation result, where the judge flattened the token-F1 signal almost to zero, **here the judge corroborates F1 rather than contradicting it**: question-omitted is worse than conditioned by the judge at every single dataset/context pair, matching F1's direction throughout, and often by a larger margin (MuSiQue high noise: −0.017 conditioned vs **−0.467** omitted; HotpotQA high noise: −0.033 vs **−0.400** omitted). This is the same asymmetry noted for §5's judge update below — some findings in this report survive judged-correctness scrutiny and some don't, and this one does.

Cost: **$0.9564** (7,000 live calls, 20 cached) for the original chain, plus **$0.0105** (341 live, 3,439 cached) for the judge pass.

## 3. Retrieval quality through repeated handoffs

> **Model:** `meta-llama/llama-3.1-8b-instruct` · **Dataset:** MS MARCO QA v2.1 (validation, via Hugging Face parquet mirror) · **Prompts:** [PROMPTS.md § Experiment 3](PROMPTS.md#experiment-3) — local `SYSTEM`/`INITIAL`/`REWRITE` + shared `ANSWER_SYSTEM` (`src/run_retrieval_quality.py`)

### Design

**Revised 2026-08-20: retrieval is now real, not assumed.** MS MARCO QA v2.1 provides ten passages per query with `is_selected` relevance labels, but the original pilot treated any non-selected passage — from any query — as a valid distractor, which meant the low-retention arm's filler was often lexically unrelated to the question entirely (an “easy” candidate negative). This run replaces that with a self-contained Okapi BM25 index (`src/retrieval.py`, no external dependency) built over a 4,000-query pool:

- **Gold** = a passage MS MARCO's own `is_selected` label marks relevant **and** that this code's own BM25 ranking actually retrieves for that query ("top retrieved and relevant," not relevance judged in isolation — 22/22 originally-labelled passages turned out to be BM25-findable in this sample, i.e. `gold_bm25_findable == gold_labelled_by_msmarco`).
- **Hard candidate negative** = a passage BM25 ranks in that same query's top-50 but that is *not* gold — either the query's own non-selected passage, or another query's passage lexically on-topic enough to rank highly. These replace the previous random-unrelated-query filler.

The low-/medium-/high-noise knob keeps the original global-pool mechanic (one seeded shuffle-and-slice over every gold passage pooled across all 20 queries, not a per-query fraction — a per-query fraction breaks down when a query has only one findable gold passage, since `round(1 × 0.5) == 0` under Python's banker's rounding while the high-noise floor keeps at least one, inverting the intended ordering). Every context still has ten passages; the label refers to the *relative amount of retrieved gold retained*, not an assertion that any arm lacks distractors. Twenty questions, one seed, depths 0, 1, 3, and 5. Question text was passed at every handoff.

**Easy-negative matched rerun, 2026-08-21.** Each hard arm is now paired with an easy arm that keeps the same questions, gold passages, global recall target, ten-passage width, gold/distractor positions, prompts, depths, and model. The sole change is filler selection: hard fillers are eligible BM25 top-50 passages, while easy fillers are eligible passages drawn from that query's BM25 bottom-1,000 (including zero-overlap passages). `hard − easy` is bootstrapped over the same 20 question ids at each condition/depth. These are BM25-selected *candidate* distractors; neither dataset supplies an exhaustive target-question non-relevance judgment for every cross-query candidate.

Empirical recall-at-10 against BM25-findable gold:

| Noise level | Exact input composition across the 20 contexts | Recall@10 |
|---|---:|---:|
| Low noise | All 22 BM25-findable gold passages + 178 candidate distractors | 1.000 |
| Medium noise | 11/22 gold passages + 189 candidate distractors | 0.500 |
| High noise | 3/22 gold passages + 197 candidate distractors | 0.136 |

### Results

![Retrieval quality propagation](retrieval_quality/n20/retrieval_quality.png)

| Depth | Low-noise F1 | Medium-noise F1 | High-noise F1 |
|---:|---:|---:|---:|
| 0 | 0.492 | 0.382 | 0.291 |
| 1 | 0.423 | 0.392 | 0.258 |
| 3 | 0.417 | 0.290 | 0.303 |
| 5 | 0.433 | 0.336 | 0.325 |

The updated plot shows solid hard-negative and dashed easy-negative trajectories:

![Retrieval quality: hard vs easy negatives](retrieval_quality/n20/retrieval_quality.png)

| Retrieval arm | Hard F1 d0 | Easy F1 d0 | Hard F1 d5 | Easy F1 d5 |
|---|---:|---:|---:|---:|
| Low noise | 0.492 | 0.547 | 0.433 | 0.460 |
| Medium noise | 0.382 | 0.321 | 0.336 | 0.283 |
| High noise | 0.291 | 0.204 | 0.325 | 0.150 |

Bottom-ranked “easy” candidate distractors do **not** uniformly improve QA. They make the high-noise arm substantially worse: hard minus easy is +0.153 F1 at depth 3 (95% interval +0.022 to +0.308) and +0.176 at depth 5 (+0.042 to +0.331). Conversely, the low-noise arm favors easy negatives at depth 1 (hard minus easy −0.173, −0.308 to −0.056). The resulting low-minus-high-noise gap is larger and remains distinguishable under easy negatives (+0.342 at depth 0; +0.310 at depth 5), while the hard-negative gap is +0.201 and +0.107 respectively. This suggests lexical hard negatives can preserve question-adjacent information that partly rescues a weak-retrieval context; they are not interchangeable with empty noise.

Under BM25 hard candidate negatives, the low–high-noise gap is larger and more persistent than the original random-distractor design found: **+0.201 F1 at depth 0** (95% interval +0.093 to +0.326, p = 0.002) and still **+0.107 at depth 5** (interval −0.014 to +0.243, p = 0.103 — directionally consistent but no longer excluding zero at n = 20). The low-noise condition still degrades most in absolute terms (depth-5 change −0.059 F1, interval −0.125 to +0.006), while high noise is statistically flat across depth (all `depth_minus_depth0` intervals for high noise span zero).

This is a materially different picture from the random-distractor version: with genuinely confusable hard negatives, the retrieval-quality advantage is bigger to start with (real distractors are harder to filter than easy ones) and does not visibly close by depth 5, though the interval no longer rules out zero. Read together with §4 below — where the same BM25 hard-negative swap produced the opposite ordering shift — this suggests hard negatives change results in ways that depend on task structure, not a uniform "harder distractors always widen the gap."

Pilot cost: **$0.006587** (480 live calls, 60 cached).

## 4. Fixed-context redundant-evidence signal ratio

> **Model:** `meta-llama/llama-3.1-8b-instruct` · **Dataset:** SQuAD (validation) · **Prompts:** [PROMPTS.md § Experiment 4](PROMPTS.md#experiment-4) — local `SYSTEM`/`INITIAL`/`REWRITE` + shared `ANSWER_SYSTEM` (`src/run_redundant_signal_ratio.py`)

### Design

This experiment holds the task fixed: every relevant passage independently contains the full answer-bearing SQuAD paragraph for the *same* question. The ten relevant documents differ through a real additional paragraph from the same Wikipedia article. Thus 10/5/1 changes the quantity of redundant answer-supporting evidence, not the number of facts required for a correct answer.

**Revised 2026-08-20:** the replaced (non-gold) documents now come from the same BM25 hard-negative mechanism introduced for §3, reusing `src/retrieval.py`. Previously, removed gold passages were replaced with a uniformly random cross-article SQuAD paragraph, filtered only to not contain an answer alias — with no requirement that it be topically related to the question at all. Now each base question's distractor pool is its own BM25 top-60 retrieval over all ~2,000 unique SQuAD paragraphs, excluding same-article and answer-alias-containing paragraphs — passages a real retriever would plausibly have surfaced for this question, not an arbitrary unrelated one. The signal-ratio manipulation (10/5/1 gold passages kept) is otherwise unchanged.

**Easy-negative matched rerun, 2026-08-21.** Each nontrivial signal ratio now has a companion arm that replaces hard top-60 filler with eligible paragraphs from the question's BM25 bottom-1,000. Same question, gold-document subset, count, positions, prompts, depth, model, and scoring are retained. The 10-gold/0-distractor control is identical by construction in both labels.

### Results

![Corrected answer-sufficient signal ratio through handoffs](redundant_signal_ratio/n20/redundant_signal_ratio.png)

| Signal ratio | F1 depth 0 | F1 depth 5 | Judge depth 0 | Judge depth 5 |
|---|---:|---:|---:|---:|
| 10 answer-sufficient / 0 distractor | 0.896 | 0.801 | 1.000 | 0.850 |
| 5 answer-sufficient / 5 distractor | 0.896 | 0.885 | 1.000 | 1.000 |
| 1 answer-sufficient / 9 distractor | 0.734 | 0.773 | 0.850 | 0.900 |

The plot now overlays hard (solid) and easy (dashed) negative arms:

![Redundant signal: hard vs easy negatives](redundant_signal_ratio/n20/redundant_signal_ratio.png)

| Signal ratio | Hard F1 d0 | Easy F1 d0 | Hard F1 d5 | Easy F1 d5 |
|---|---:|---:|---:|---:|
| 10 answer-sufficient / 0 distractor | 0.896 | 0.896 | 0.801 | 0.801 |
| 5 answer-sufficient / 5 distractor | 0.896 | 0.892 | 0.885 | 0.763 |
| 1 answer-sufficient / 9 distractor | 0.734 | 0.819 | 0.773 | 0.645 |

The key matched effect is at 5 answer-sufficient passages: hard minus easy judged correctness is +0.20 at both depths 3 and 5 (95% interval +0.05 to +0.40). Token F1 points in the same direction at depth 5 (+0.123, +0.002 to +0.268), though this 20-pack pilot remains too small for a precise F1 estimate. At 1 answer-sufficient passage, easy negatives begin numerically higher but lose more by depth 5; neither hard-minus-easy F1 contrast excludes zero. Thus the easy/hard choice mainly changes the intermediate-redundancy trajectory here, rather than producing a uniform shift in difficulty.

With hard negatives, the depth-0 gap between 10-gold and 1-gold widens relative to the earlier random-distractor pilot: **+0.163 F1** (95% interval +0.044 to +0.306, p = 0.014) and **+0.150 judge points** (interval 0.0 to +0.30, p = 0.108 — directional, does not exclude zero at n = 20). That gap is *not* preserved at depth 5 — it is numerically near zero on both metrics (F1 +0.028, interval −0.119 to +0.165, p = 0.69; judge −0.05, interval −0.20 to +0.10, p = 0.76), and the 10-gold condition's own within-condition degradation is now the largest of the three (F1 depth-5 change −0.095, interval −0.214 to −0.008, p = 0.064; judge change −0.15, interval −0.30 to 0.0, p = 0.107), while 1-gold is flat-to-improving on both metrics at every depth.

This is close to a **reversal** of the previous (random-distractor) reading, not just a magnitude change: before, the high-signal condition's advantage was reported as "directionally preserved" through depth 5. With genuine hard negatives, the redundant (10-gold) condition starts further ahead but degrades enough under repeated compression that its lead is gone by depth 5, while the single-gold condition — now facing distractors that are harder to rule out, not easier — holds steady or improves. One plausible reading: 10 redundant restatements of the same fact give a compressor more opportunities to *drop* the fact somewhere in the chain and still look locally reasonable, whereas 1 gold passage among genuinely confusable hard negatives forces every compression stage to make the same forced choice consistently. This is a single n=20 pilot and should be treated as a hypothesis to replicate, not a settled result — but it is a materially different conclusion from the random-distractor version, which is itself informative: distractor hardness is not a nuisance parameter, it changes which conditions look robust.

Pilot cost: **$0.009382** (357 live calls, 183 cached).

## 5. Question conditioning and cross-question generalization

> **Model:** `meta-llama/llama-3.1-8b-instruct` · **Dataset:** SQuAD (validation), two low-overlap questions paired per context · **Prompts:** [PROMPTS.md § Experiment 5](PROMPTS.md#experiment-5) — defines no new text, reuses Experiment 2's question-conditioned `CHAIN_SYSTEM`/`INITIAL_INSTRUCTION`/`RECOMPRESS_INSTRUCTION` verbatim, imported from `src/run_chain.py`

### Corrected v2 design

Each SQuAD example combines two independently labelled questions from different articles:

- Question A and its gold passage.
- A lexically unrelated Question B and its gold passage.
- Eight real SQuAD distractor passages.

Both A and B are therefore answerable from the same fixed simulated top-10 retrieval context. The conditioned and generic chains use the exact system prompt and handoff instructions from the repeated-degradation experiment. A runtime self-test verifies that the prompts become byte-identical when the single Question A block is deleted.

- **Conditioned:** Question A is appended to the documents/summary at every handoff.
- **Generic:** the same prompt is used without that question block.

The final answerer receives whichever question is being evaluated. Summaries have the same 700-token maximum in both arms.

### Results

![Question-only conditioning through ten handoffs](summary_generalization_v2_depth10/n20/summary_generalization.png)

The plot evaluates every depth from 0 to 10. Selected Token-F1 landmarks are:

| Evaluation | Direct | Conditioned d1 | Generic d1 | Conditioned d2 | Generic d2 | Conditioned d5 | Generic d5 | Conditioned d10 | Generic d10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conditioning Question A | 0.828 | 0.829 | 0.535 | 0.794 | 0.576 | 0.504 | 0.480 | 0.388 | 0.463 |
| Unrelated Question B | 0.970 | 0.200 | 0.592 | 0.150 | 0.583 | 0.250 | 0.573 | 0.100 | 0.597 |

Paired conditioned-minus-generic F1 contrasts show that the target-specific benefit is real at depths 1–2, then becomes indistinguishable from zero. It is numerically negative at depths 8–10, but those later reversals are not distinguishable at this sample size.

| Evaluation | Depth | Difference | 95% interval | p-value |
|---|---:|---:|---|---:|
| Question A | 1 | **+0.295** | [+0.125, +0.476] | 0.0009 |
| Question A | 2 | **+0.218** | [+0.033, +0.417] | 0.0252 |
| Question A | 3 | +0.091 | [−0.111, +0.303] | 0.3948 |
| Question A | 5 | +0.024 | [−0.156, +0.208] | 0.8014 |
| Question A | 10 | −0.076 | [−0.309, +0.160] | 0.5262 |
| Question B | 1 | **−0.392** | [−0.600, −0.192] | 0.0001 |
| Question B | 5 | **−0.323** | [−0.540, −0.117] | 0.0023 |
| Question B | 10 | **−0.497** | [−0.700, −0.283] | <0.0001 |

The lower row of the figure repeats both panels under the LLM judge, and it **reinforces** this experiment's conclusion rather than softening it (unlike experiment 2, where the judge flattened the F1 signal). Paired conditioned-minus-generic judge contrasts:

| Evaluation | Depth | Judge difference | 95% interval | p-value |
|---|---:|---:|---|---:|
| Question A | 1 | **+0.400** | [+0.200, +0.600] | 0.0001 |
| Question A | 2 | **+0.350** | [+0.150, +0.550] | 0.0013 |
| Question A | 5 | +0.150 | [−0.050, +0.350] | 0.2349 |
| Question A | 10 | −0.100 | [−0.350, +0.150] | 0.5325 |
| Question B | 1 | **−0.400** | [−0.600, −0.200] | 0.0003 |
| Question B | 5 | **−0.400** | [−0.600, −0.200] | 0.0002 |
| Question B | 10 | **−0.450** | [−0.650, −0.250] | <0.0001 |

The same asymmetry appears with larger effect sizes: the target-question benefit is significant at depths 1–2 and gone by depth 5, while the damage to the unrelated question is significant at *every* depth and does not shrink with depth. Because the judge scores whether the answer is actually right rather than how many gold tokens it shares, this rules out the possibility that the held-out damage was merely a wording artefact.

The stronger conclusion is therefore asymmetric. Question conditioning is useful for its target in the first two handoffs, but that benefit does not survive reliably beyond depth 2. In contrast, it persistently removes information needed for the unrelated, answerable question. This supports generic summaries when long-horizon reuse is expected, but does not prove that generic summaries are better for a fixed task at every later depth: the target d8–10 reversal remains uncertain.

A complete side-by-side example is available in [`summary_generalization_v2_depth10/n20/example.md`](summary_generalization_v2_depth10/n20/example.md).

Original corrected pilot cost was approximately **$0.0100**; the depth-10 extension added **$0.013471** (698 live calls, 22 cache hits). Full metrics and contrasts are in [`summary_generalization_v2_depth10/n20/metrics.csv`](summary_generalization_v2_depth10/n20/metrics.csv) and [`summary_generalization_v2_depth10/n20/deltas.csv`](summary_generalization_v2_depth10/n20/deltas.csv).

### Same-passage A/B rerun on SQuAD (21 August 2026)

> **Model:** `meta-llama/llama-3.1-8b-instruct` · **Dataset:** SQuAD validation, same-passage A/B pairs built by `src/build_squad_same_passage.py` · **Config:** `summary_generalization_squad_pairs_config.yaml` · **Prompts:** identical to the paired-SQuAD run above — same `CHAIN_SYSTEM`/`INITIAL_INSTRUCTION`/`RECOMPRESS_INSTRUCTION` from `src/run_chain.py`, same `ANSWER_SYSTEM`

#### Why this replaced the Wikipedia variant

An earlier version of this comparison used the self-generated Wikipedia dataset. It was withdrawn as confounded, and its artifacts deleted, after an audit found its generic arm was crippled by construction rather than by the treatment: **10/10 generic stage-1 summaries hit the token cap** (whole Wikipedia articles are ~16,900 prompt tokens against a 700-token budget inherited from a ~1,700-token design), in 4/10 pairs the gold passage sat at P9–P10 and was never reached, and the question generator had been explicitly instructed to prefer "details that are not obvious" — so the summariser was asked to retain exactly the class of fact summarisation discards. Verbatim gold survival in generic summaries was **0/10 for both questions**, making the held-out contrast a degenerate 0-versus-0 rather than a measurement.

This rebuild keeps the *scientific* question (does conditioning on A narrow a summary away from an unrelated B answerable from the same evidence?) and fixes every design defect:

| Control | Wikipedia version | This rebuild |
|---|---|---|
| Questions | Model-authored, selected for non-obviousness | Human-written SQuAD questions, **0 model-authored** |
| Salience / independence | Not checked | LLM-audited per pair (different model family); 3 pairs rejected |
| Gold passage | 1 full article (~5–8.6k chars) | 1 SQuAD paragraph, 712–878 chars |
| Context length | Unmatched, ~61k chars | Matched: 7,432–8,197 chars (10.3% spread) |
| Gold position | Unstratified (4/10 unreachable) | **Stratified: exactly 2 pairs at each of the 10 slots** |
| Leakage filter | Not applied | C1 closed-book on **both** A and B (85/220 questions leaked; 42/110 pairs fully clean) |
| Budget | 700, bound 100% of the time | 1,500, non-binding (see truncation below) |
| n | 10 | 20 |

Both questions are answered by the **same** passage, as before — that is the intended contrast with the original paired-SQuAD design above, where A and B have separate gold passages.

#### Pre-interpretation checks

Before reading any effect, the three diagnostics that invalidated the previous run:

- **Truncation:** conditioned 3/200 handoffs (1.5%), generic 1/200 (0.5%). The cap does not bind; summary length is chosen by the model (mean stage-1 length: conditioned 123 tokens, generic 823).
- **Generic baseline non-degenerate:** generic target F1 0.65–0.70 and held-out F1 0.40–0.45 across depths, against a direct-context ceiling of 0.83/0.83. There is a real baseline to degrade.
- **Stage-1 fact survival:** generic retains gold A 14/20 (70%) and gold B 9/20 (45%); conditioned retains A 16/20 (80%) and B 6/20 (30%).

#### Results

![Question-only conditioning on SQuAD same-passage pairs](squad_same_passage/n20/summary_generalization.png)

Paired conditioned-minus-generic contrasts:

| Comparison | Metric | Depth 1 | Depth 2 | Depth 5 | Depth 10 |
|---|---|---:|---:|---:|---:|
| Target (Question A) | F1 | +0.059 (p=0.61) | −0.074 (p=0.53) | −0.039 (p=0.66) | −0.092 (p=0.39) |
| Target (Question A) | LLM judge | +0.150 (p=0.11) | −0.050 (p=0.83) | −0.050 (p=0.76) | −0.100 (p=0.53) |
| Held-out (Question B) | F1 | −0.169 (p=0.22) | −0.136 (p=0.33) | −0.184 (p=0.24) | −0.163 (p=0.28) |
| Held-out (Question B) | LLM judge | −0.150 (p=0.39) | −0.200 (p=0.25) | −0.200 (p=0.24) | −0.200 (p=0.24) |

**No effect reaches significance at n=20.** Read as directional evidence only:

1. **The target benefit disappears when A and B share one passage.** The original paired-SQuAD design (separate gold passages) gave conditioning **+0.295 F1 at depth 1**, significant at p=0.0009. Here it is +0.059 and gone by depth 2. The mechanism is visible in fact survival: a generic summary of a single 800-character paragraph already keeps A's fact 70% of the time, so naming A adds little (+0.100 survival, CI [−0.100, +0.300]). Conditioning helps most when it lets the compressor *discard a whole irrelevant document* — which the separate-passage design permits and this one does not.
2. **The held-out cost persists in direction but not in significance.** It is negative at every depth on both metrics (F1 −0.11 to −0.18; judge −0.15 to −0.20), matching the sign of the original run's significant −0.39 to −0.50, and gold-B survival drops 45% → 30% under conditioning (−0.150, CI [−0.450, +0.150]). Consistent, but every interval spans zero.
3. **Both arms lose more on B than on A.** Held-out F1 falls from 0.833 direct to 0.40 generic / 0.23 conditioned at depth 1, while target F1 holds near 0.70. Even an unconditioned summary of a shared passage is a lossy channel for the question it was not written for.

**Honest reading:** with a properly controlled dataset the conditioning effect is *much smaller* than either previous run suggested. The Wikipedia run's large "durable benefit" was an artifact of a broken baseline; the original paired-SQuAD run's large effects depend on A and B having separate gold passages. When they share one passage, conditioning buys little and costs something, but n=20 with one seed cannot establish the cost. Confirming it needs ≥100 pairs and a second seed.

Cost: **$0.039** (1,218 live calls at `llama-3.1-8b-instruct` plus 240 live judge calls), plus **$0.02** for dataset construction and its leakage filtering.

## Cross-experiment interpretation

The results support three distinct roles for handoffs:

1. **Denoising:** one summary can remove distractors and improve answer accuracy.
2. **Serial information loss:** repeated rewriting can progressively remove precise evidence, especially when the starting evidence is already minimal.
3. **Task-induced narrowing:** supplying a question changes which information survives. This helps the immediate target but can permanently remove facts needed by other questions.

These mechanisms can coexist. A handoff may improve the current answer by filtering noise while simultaneously making the representation less reusable for future tasks.

## Limitations

- All completed runs are pilots: 10, 20, or 30 questions per condition.
- The baseline and long-chain experiments used Llama 3.3 70B, whereas the low-cost retrieval and generalization pilots used Llama 3.1 8B.
- The repeated-chain experiment used two seeds, but the low-cost pilots used one deterministic seed.
- SQuAD A/B contexts are simulated retrieval packs rather than outputs from a live retriever.
- SQuAD is a heavily-pretrained public benchmark: 85/220 (39%) of the same-passage rebuild's candidate questions failed the closed-book leakage check and were discarded. The surviving 20 pairs are therefore drawn from the harder tail of SQuAD, not from SQuAD at large.
- The same-passage rebuild is powered to detect only large effects (n=20, one seed); none of its contrasts reach significance and it should not be cited as a null result.
- The redundant-evidence signal-ratio gold passages share an answer-bearing source paragraph. The experiment controls answer sufficiency, but not independent-source diversity.
- The redundant-evidence signal-ratio gold passages share an answer-bearing source paragraph. The experiment controls answer sufficiency, but not independent-source diversity.
- Direct scores for Question A and B should not be compared as measures of relative difficulty. Valid causal comparisons are paired within the same question type.
- Bootstrap intervals are exploratory and were not corrected for multiple comparisons.

## Recommended next steps

1. Scale the same-passage conditioning rebuild (§5) to at least 100 pairs and two seeds — at n=20 no contrast reaches significance, so the persistent negative held-out sign is directional only.
2. Replicate it with Llama 3.3 70B or another stronger model to test whether the specialization–generalization asymmetry survives model scaling.
3. Track explicit fact survival in summaries, separating omission at stage 1 from corruption during stages 2–5.
4. Expand the MS MARCO retrieval pilot to at least 100 questions before interpreting the medium-condition behavior.
5. Keep the v2 question-only prompt audit as a required self-test in all subsequent conditioning experiments, and report the truncation rate and generic-arm fact survival before interpreting any conditioning contrast — a summariser silently hitting its token cap produces a degenerate baseline that reads as a large treatment effect.
6. Scale the corrected redundant-evidence signal-ratio experiment to 100 packs, then replace shared-source redundancy with independently sourced, answer-sufficient documents where a suitable labelled corpus permits it.

## Result artifacts

- Baseline: [`report.md`](report.md), [`summary.csv`](summary.csv), [`contrasts.csv`](contrasts.csv)
- Serial degradation: [`chain/report.md`](chain/report.md), [`chain/stage_metrics.csv`](chain/stage_metrics.csv)
- Serial degradation, question omitted: [`chain_generic/report.md`](chain_generic/report.md), [`chain_generic/stage_metrics.csv`](chain_generic/stage_metrics.csv), [`chain_generic/conditioning_comparison.png`](chain_generic/conditioning_comparison.png)
- Retrieval quality: [`retrieval_quality/n20/metrics.csv`](retrieval_quality/n20/metrics.csv), [`retrieval_quality/n20/deltas.csv`](retrieval_quality/n20/deltas.csv)
- Corrected redundant-evidence signal ratio: [`redundant_signal_ratio/n20/metrics.csv`](redundant_signal_ratio/n20/metrics.csv), [`redundant_signal_ratio/n20/deltas.csv`](redundant_signal_ratio/n20/deltas.csv), [`redundant_signal_ratio/n20/redundant_signal_ratio.png`](redundant_signal_ratio/n20/redundant_signal_ratio.png)
- Corrected generalization, depth-10 extension: [`summary_generalization_v2_depth10/n20/metrics.csv`](summary_generalization_v2_depth10/n20/metrics.csv), [`summary_generalization_v2_depth10/n20/deltas.csv`](summary_generalization_v2_depth10/n20/deltas.csv), [`summary_generalization_v2_depth10/n20/summary_generalization.png`](summary_generalization_v2_depth10/n20/summary_generalization.png)
- Generalization, SQuAD same-passage A/B rebuild: [`squad_same_passage/n20/metrics.csv`](squad_same_passage/n20/metrics.csv), [`squad_same_passage/n20/deltas.csv`](squad_same_passage/n20/deltas.csv), [`squad_same_passage/n20/summary_generalization.png`](squad_same_passage/n20/summary_generalization.png), construction report [`../data/squad_same_passage/construction_n20.csv`](../data/squad_same_passage/construction_n20.csv)
