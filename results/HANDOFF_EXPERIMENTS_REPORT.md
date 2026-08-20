# Agent handoff information-loss experiments

**Status:** pilot-scale evidence through 20 August 2026  
**Primary metrics:** exact match (EM) and SQuAD-style token F1  
**Secondary metric:** LLM-judge answer correctness (`openai/gpt-4o-mini`, temperature 0, binary vs gold; a different model family from the systems under test)  
**Judging:** deterministic string metrics remain primary; an LLM judge is reported alongside them, never in place of them

## Executive summary

The evidence does **not** support a universal “each handoff loses a fixed amount of accuracy” rule. A handoff is both a lossy communication channel and a possible denoiser: whether it helps or harms depends on distractor load, the number of facts that must survive, and whether the next task is the one the summary was optimized for.

The current synthesis is:

1. **Compression can help the immediate task when it removes noise.** In the initial MuSiQue pilot, a free-form handoff beat direct full-context answering by 26.4 F1 points. In the longer chains, noisy full contexts often improved after early summaries, whereas compact HotpotQA gold-only evidence showed the clearest serial loss (−10.7 F1 by depth 10).
   **Metric caveat added 20 August 2026:** that −10.7 F1 result is **0.000** under an LLM judge scoring answer correctness against the gold (0.900 at both depth 0 and depth 10). Token F1 penalises correct-but-reworded answers, which repeated compression reliably produces. Read as "the asserted fact survives ten handoffs, but its surface form drifts from the gold string." No serial-loss claim in this report should now be made on token F1 alone.
2. **Repeated handoffs can erase an initial retrieval advantage, but the pilot evidence is mixed.** The MS MARCO experiment began with an 8.6-point good–bad F1 gap that was only 1.2 points at depth 5. The corrected fixed-ten-passage experiment similarly favored 10 redundant answer-sufficient passages over one at depth 0 (+10.9 F1) and depth 5 (+8.3), but its 20-question intervals are too wide to establish a signal-ratio-specific degradation law.
3. **Question conditioning is a short-horizon specialization tool, not a reusable-handoff default.** It improved its target by +29.5 F1 at the first handoff and +21.8 at the second, but that advantage was no longer distinguishable from zero from depth 3 onward; at depth 10 it was numerically −7.6 F1 (interval spans zero). Meanwhile, it damaged an unrelated but answerable question at every depth, including −49.7 F1 at depth 10. Unlike the serial-degradation result above, **the LLM judge strengthens this finding** (+40.0 / +35.0 points on the target at depths 1–2; −40 to −45 points on the held-out question at every depth, all p < 0.001), so it does not rest on a token-overlap artefact. For five or more handoffs with a potentially changing downstream task, generic notes are the safer current default; for one or two fixed-task handoffs, conditioning remains useful.
4. **Withholding the question from every compressor is far more damaging than repeated compression itself.** The matched question-omission replication (same model, questions, contexts, depths, seeds, system prompt, and handoff instructions as the main serial chain — the sole difference is that no compressor ever sees the question) degrades at every depth, in every evidence condition, including full context — the one condition that *improved* under question-conditioned compression. MuSiQue full context goes from +0.080 F1 at depth 10 (conditioned) to **−0.272 F1** (question omitted) on the identical question set.
5. **These are directional pilots, not final effect sizes.** The Llama 3.3 70B and Qwen3 8B chain runs agree that dataset and context composition matter, but most low-cost experiments use 20 questions and one seed. The strongest next step is replication at larger sample sizes with independently sourced redundant evidence.

## Experiment inventory

| Experiment | Dataset | Sample | Model | Handoff depths | Main comparison |
|---|---|---:|---|---|---|
| Single-handoff mechanisms | MuSiQue | 10 | Llama 3.3 70B Instruct | 0/1 | Full context vs free-form handoff vs oracle evidence |
| Serial handoff degradation | MuSiQue + HotpotQA | 30 per dataset | Llama 3.3 70B Instruct | 0–10 | Gold-only, five-document, and full contexts |
| Serial handoff degradation, question omitted | MuSiQue + HotpotQA | 30 per dataset | Llama 3.3 70B Instruct | 0–10 | Same chain; only the compressor question block is omitted |
| Serial handoff degradation (replication) | MuSiQue + HotpotQA | 10 per dataset | Qwen3 8B, non-thinking | 0/1/3/5 | Same evidence variants, low-cost one-seed replication |
| Retrieval-quality propagation | MS MARCO QA v2.1 | 20 | Llama 3.1 8B Instruct | 0/1/3/5 | Good, medium, and bad selected-passage recall |
| Redundant-evidence signal-ratio propagation | SQuAD same-article packs | 20 | Llama 3.1 8B Instruct | 0/1/3/5 | Exactly 10/10, 5/10, or 1/10 individually answer-sufficient passages |
| Cross-question generalization v2, depth-10 extension | SQuAD | 20 paired contexts | Llama 3.1 8B Instruct | 0–10 | Question A present vs absent; evaluate A and unrelated B at every depth |

## 1. Single-handoff mechanism pilot

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

### Design

The same question was supplied at every compression stage:

`documents → summary 1 → summary 2 → … → summary 10 → fresh answerer`

Only the first compressor could see source documents. Later compressors received a sealed previous handoff and the original question. Two seeds were run. Each dataset used three evidence conditions:

- **Short:** all and only gold documents (mean 2.7 documents for MuSiQue; 2 for HotpotQA).
- **Medium:** all gold documents plus distractors up to five documents.
- **Full:** the complete dataset context (20 MuSiQue paragraphs; 10 HotpotQA paragraphs).

### Results

![Repeated handoff degradation across datasets and context lengths](chain/degradation.png)

Depth-10 change from direct depth 0:

| Dataset | Context | Token F1 change | LLM-judge change |
|---|---|---:|---:|
| MuSiQue | Short | −0.027 | +0.017 |
| MuSiQue | Medium | −0.025 | −0.033 |
| MuSiQue | Full | +0.080 | −0.017 |
| HotpotQA | Short | **−0.107** | **+0.000** |
| HotpotQA | Medium | −0.035 | −0.050 |
| HotpotQA | Full | +0.014 | −0.033 |

BERTScore was replaced by an LLM judge (`openai/gpt-4o-mini`, temperature 0, binary correct/incorrect against the gold answer, deliberately a different model family from the llama systems under test). EM and token F1 remain primary and are reported unchanged alongside it.

**The judge materially changes the headline reading of this experiment.** The single largest serial-degradation signal in the probe — HotpotQA gold-only, −10.7 token-F1 points by depth 10 — is **0.000** under the judge: 0.900 correct at depth 0 and 0.900 at depth 10. Absolute levels also sit far above F1 (HotpotQA short holds ~0.90–0.92 across all ten handoffs; MuSiQue short ~0.73–0.75). That gap is the expected signature of token F1 penalising answers that are correct but reworded or more verbose, which is exactly what repeated compression produces. Under the judge no evidence variant loses more than 5 points across ten handoffs, and MuSiQue short is numerically flat-to-positive.

This does not erase the F1 result, and the two should be read together: F1 says the *surface form* of answers drifts steadily away from the gold string under repeated compression, while the judge says the *fact being asserted* mostly survives. The conclusion that gold-only evidence is the most fragile condition is supported by F1 but is not corroborated by the judge at this sample size.

The full-context conditions often improved after the first handoff, consistent with denoising. Summary length also collapsed rapidly: for example, HotpotQA full context averaged 5,530 characters at depth 0, 720 after one handoff, and 413 by depth 10.

The main conclusion is not that long contexts are inherently safer. Rather, noisy full contexts provide an opportunity for useful selection, while already-minimal gold evidence has little redundancy and therefore exposes omissions more directly.

The confirmed incremental cost reported for extending the chains through depths 6–10 and filling missing answers was **$0.2436**.

### Low-cost Qwen replication

![Qwen3 8B repeated-handoff replication](chain_qwen/degradation.png)

The same dataset and evidence-variant design was rerun with Qwen3 8B in non-thinking mode, but with 10 questions per dataset, one seed, and depths 0/1/3/5. It broadly reproduces the dataset split in the original probe: at depth 5, HotpotQA loses F1 in every context (short −0.110, medium −0.086, full −0.108), while MuSiQue is stable-to-improved (short +0.127, medium +0.077, full +0.117). None of the depth-5 intervals exclude zero at this small sample size, so this is directional replication evidence rather than a conclusive cross-model comparison.

Reconciled cost was **$0.0443** across Qwen-specific leakage filtering, 300 summary calls, 240 answer calls, and a compatibility smoke call. The raw outputs and full table are in [`chain_qwen/report.md`](chain_qwen/report.md).

### Matched question-omission replication

![Question-conditioned vs question-omitted, matched chains](chain_generic/conditioning_comparison.png)

This is the minimal counterpart to the main serial chain: same Llama 3.3 70B model, fixed filtered question sample, all three evidence variants, depths 0–10, two seeds, system prompt, and handoff instructions. The sole difference is prompt visibility — neither the initial compressor nor any later compressor receives the final-question block; the fresh answerer still receives the question. An automated equivalence check (`prompt_difference_selftest` in the code) verifies that removing that block makes the two compressor prompts byte-identical, so the only variable is question visibility.

Depth-10 Token F1 change from direct depth 0, question-conditioned vs question-omitted:

| Dataset | Context | Conditioned | Question-omitted |
|---|---|---:|---:|
| MuSiQue | Short | −0.027 | −0.035 |
| MuSiQue | Medium | −0.025 | −0.127 |
| MuSiQue | Full | **+0.080** | **−0.272** |
| HotpotQA | Short | −0.107 | −0.247 |
| HotpotQA | Medium | −0.035 | −0.218 |
| HotpotQA | Full | +0.014 | **−0.326** |

Every condition is worse without the question, and the gap widens with more distractors: full-context degradation goes from mildly positive to the worst result in either chain. This is consistent with the denoising story in §2 — a compressor can only filter distractors *toward* a task it knows, and full-context evidence has the most distractor mass to filter. It also complements §5's finding below: §5 shows conditioning narrows a summary toward one task at the cost of others; this replication shows the opposite failure mode — a compressor with no task at all keeps too much noise and too little signal for any task.

**Metric caveat:** this replication predates the 2026-08-20 LLM-judge addition (§2) and is scored on token F1 and BERTScore only. Given §2's finding that token F1 can substantially overstate degradation relative to judged answer correctness, treat the magnitudes above as directional rather than final; a judge rerun would be needed before quoting exact point estimates.

Cost: **$0.9564** (7,000 live calls, 20 cached).

## 3. Retrieval quality through repeated handoffs

### Design

MS MARCO QA v2.1 provides ten retrieved passages per query and `is_selected` labels. Twenty questions were used with one seed and depths 0, 1, 3, and 5. Removed selected passages were replaced with real non-selected MS MARCO distractors, keeping ten passages in every condition.

Empirical selected-passage Recall@10 was:

| Retrieval condition | Selected passages retained | Recall@10 |
|---|---:|---:|
| Good | 22/22 | 1.000 |
| Medium | 11/22 | 0.500 |
| Bad | 3/22 | 0.136 |

Question text was passed at every handoff.

### Results

![Retrieval quality propagation](retrieval_quality/n20/retrieval_quality.png)

| Depth | Good F1 | Medium F1 | Bad F1 | Good − bad |
|---:|---:|---:|---:|---:|
| 0 | 0.406 | 0.324 | 0.320 | +0.086 |
| 1 | 0.354 | 0.376 | 0.319 | +0.035 |
| 3 | 0.283 | 0.346 | 0.280 | +0.003 |
| 5 | 0.303 | 0.350 | 0.291 | +0.012 |

The good condition’s depth-5 degradation was −0.103 F1 (95% interval −0.184 to −0.029). Medium and bad changed much less. Consequently, the good–bad gap was distinguishable at depth 0 (+0.086; interval +0.011 to +0.171) but not at later depths.

This pilot suggests that handoff compression can erase the advantage of better retrieval: a richer initial context has more useful information available to lose. The medium condition’s apparent improvement is not distinguishable at this sample size.

Smoke plus pilot cost: approximately **$0.0152**.

## 4. Fixed-context redundant-evidence signal ratio

### Design

This corrected experiment holds the task fixed: every relevant passage independently contains the full answer-bearing SQuAD paragraph for the *same* question. The ten relevant documents differ through a real additional paragraph from the same Wikipedia article. Replaced documents are real cross-article SQuAD distractors filtered not to contain an answer alias. Thus 10/5/1 changes the quantity of redundant answer-supporting evidence, not the number of facts required for a correct answer.

### Results

![Corrected answer-sufficient signal ratio through handoffs](redundant_signal_ratio/n20/redundant_signal_ratio.png)

| Signal ratio | F1 depth 0 | F1 depth 5 | Depth-5 retention | 10-gold − 1-gold F1 gap |
|---|---:|---:|---:|---:|
| 10 answer-sufficient / 0 distractor | 0.896 | 0.801 | 0.894 | +0.083 |
| 5 answer-sufficient / 5 distractor | 0.900 | 0.840 | 0.933 | — |
| 1 answer-sufficient / 9 distractor | 0.787 | 0.718 | 0.913 | reference |

The third panel of the figure adds LLM-judge answer correctness. It tells a flatter story than F1: the 10-gold condition runs 1.000 → 0.950 → 0.900 → 0.900 across depths 0/1/3/5, and the 1-gold condition 0.900 → 0.900 → 0.900 → 0.850. The initial retrieval advantage is therefore ~10 points at depth 0 under both metrics, but under the judge the low-signal condition loses only 5 points across five handoffs where token F1 records 6.9. With 20 questions the judge intervals are wide (roughly ±0.15), so this is consistent with, but does not independently establish, the F1 reading below.

The high-signal condition begins 10.9 F1 points above the low-signal condition (95% bootstrap interval +0.009 to +0.235) and remains 8.3 points higher at depth 5 (interval −0.039 to +0.244). The initial retrieval advantage is therefore directionally preserved but not distinguishable at this small sample after repeated handoffs. Handoff degradation itself is not monotonic in signal ratio: depth-5 changes are −0.095 (10 gold), −0.060 (5 gold), and −0.069 (1 gold). This is the appropriate conclusion for the corrected pilot—more signal helps direct QA, while the evidence for a signal-ratio-specific compression effect remains inconclusive.

Corrected smoke plus pilot cost: **$0.013299**.

## 5. Question conditioning and cross-question generalization

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
- The redundant-evidence signal-ratio gold passages share an answer-bearing source paragraph. The experiment controls answer sufficiency, but not independent-source diversity.
- Direct scores for Question A and B should not be compared as measures of relative difficulty. Valid causal comparisons are paired within the same question type.
- Bootstrap intervals are exploratory and were not corrected for multiple comparisons.

## Recommended next steps

1. Scale the corrected question-conditioning experiment to at least 100 pairs and two seeds.
2. Replicate it with Llama 3.3 70B or another stronger model to test whether the specialization–generalization asymmetry survives model scaling.
3. Track explicit fact survival in summaries, separating omission at stage 1 from corruption during stages 2–5.
4. Expand the MS MARCO retrieval pilot to at least 100 questions before interpreting the medium-condition behavior.
5. Keep the v2 question-only prompt audit as a required self-test in all subsequent conditioning experiments.
6. Scale the corrected redundant-evidence signal-ratio experiment to 100 packs, then replace shared-source redundancy with independently sourced, answer-sufficient documents where a suitable labelled corpus permits it.

## Result artifacts

- Baseline: [`report.md`](report.md), [`summary.csv`](summary.csv), [`contrasts.csv`](contrasts.csv)
- Serial degradation: [`chain/report.md`](chain/report.md), [`chain/stage_metrics.csv`](chain/stage_metrics.csv)
- Serial degradation, question omitted: [`chain_generic/report.md`](chain_generic/report.md), [`chain_generic/stage_metrics.csv`](chain_generic/stage_metrics.csv), [`chain_generic/conditioning_comparison.png`](chain_generic/conditioning_comparison.png)
- Retrieval quality: [`retrieval_quality/n20/metrics.csv`](retrieval_quality/n20/metrics.csv), [`retrieval_quality/n20/deltas.csv`](retrieval_quality/n20/deltas.csv)
- Corrected redundant-evidence signal ratio: [`redundant_signal_ratio/n20/metrics.csv`](redundant_signal_ratio/n20/metrics.csv), [`redundant_signal_ratio/n20/deltas.csv`](redundant_signal_ratio/n20/deltas.csv), [`redundant_signal_ratio/n20/redundant_signal_ratio.png`](redundant_signal_ratio/n20/redundant_signal_ratio.png)
- Corrected generalization, depth-10 extension: [`summary_generalization_v2_depth10/n20/metrics.csv`](summary_generalization_v2_depth10/n20/metrics.csv), [`summary_generalization_v2_depth10/n20/deltas.csv`](summary_generalization_v2_depth10/n20/deltas.csv), [`summary_generalization_v2_depth10/n20/summary_generalization.png`](summary_generalization_v2_depth10/n20/summary_generalization.png)
