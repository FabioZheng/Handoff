# Experiment 8b: fictional-QA selector bottleneck

- Dossiers: **20**; A/B rotations: **80**.
- Channel: **3** selector slots → **1** relay slots.
- Stage 1 sees the fictional source and A. Stage 2 is newly shown B; sealed arms receive no source. The final answerer is fixed and from a third family.
- Intervals and paired tests resample dossiers, not rotations.
- ITT counts schema failures as failures; compliant results require both stages to return a valid fixed-K selection.

> **Full configured run.** Effective n is 20 dossiers; the 80 A/B rotations are repeated measures, not independent samples.

## Factorial cell outcomes (ITT)

Judged B accuracy; arrows denote stage-1 selector → stage-2 relay.

| family | S→S | S→L | L→S | L→L |
|---|---:|---:|---:|---:|
| llama | 0.300 | 0.350 | 0.338 | 0.388 |
| qwen | 0.388 | 0.400 | 0.350 | 0.362 |

## Primary paired contrasts (ITT)

| family | comparison | population | delta | 95% CI | p | dossiers |
|---|---|---|---:|---|---:|---:|
| llama | `llama:early_selector_at_large_relay` | all | +0.037 | [-0.037, +0.125] | 0.556 | 20 |
| llama | `llama:larger_relay_after_small_selector` | all | +0.050 | [+0.013, +0.100] | 0.120 | 20 |
| llama | `llama:larger_relay_on_clean_omissions` | eligible_omissions | +0.000 | [+0.000, +0.000] | 1.000 | 20 |
| llama | `llama:smaller_relay_after_large_selector` | all | -0.050 | [-0.113, +0.000] | 0.258 | 20 |
| llama | `llama:large_first_order` | all | -0.013 | [-0.100, +0.087] | 1.000 | 20 |
| llama | `llama:restore_vs_sham` | eligible_omissions | +0.983 | [+0.950, +1.000] | <0.001 | 20 |
| llama | `llama:restore_vs_sealed` | eligible_omissions | +1.000 | [+1.000, +1.000] | <0.001 | 20 |
| llama | `llama:reopen_vs_sealed` | eligible_omissions | +1.000 | [+1.000, +1.000] | <0.001 | 20 |
| qwen | `qwen:early_selector_at_large_relay` | all | -0.037 | [-0.125, +0.062] | 0.614 | 20 |
| qwen | `qwen:larger_relay_after_small_selector` | all | +0.013 | [+0.000, +0.037] | 1.000 | 20 |
| qwen | `qwen:larger_relay_on_clean_omissions` | eligible_omissions | +0.000 | [+0.000, +0.000] | 1.000 | 20 |
| qwen | `qwen:smaller_relay_after_large_selector` | all | -0.013 | [-0.037, +0.000] | 1.000 | 20 |
| qwen | `qwen:large_first_order` | all | -0.050 | [-0.138, +0.050] | 0.465 | 20 |
| qwen | `qwen:restore_vs_sham` | eligible_omissions | +0.950 | [+0.850, +1.000] | <0.001 | 20 |
| qwen | `qwen:restore_vs_sealed` | eligible_omissions | +1.000 | [+1.000, +1.000] | <0.001 | 20 |
| qwen | `qwen:reopen_vs_sealed` | eligible_omissions | +1.000 | [+1.000, +1.000] | <0.001 | 20 |

## Observed result

- The upstream-size contrast did **not replicate across families**: Llama +0.037, Qwen -0.037.
- Across 196 sealed small-selector rows where stage 1 contained no answer-bearing B evidence, downstream judged accuracy was **0.000**.
- Large-minus-small relay accuracy on those clean omissions is llama +0.000 [+0.000, +0.000], qwen +0.000 [+0.000, +0.000]; both intervals lie inside the prespecified ±0.05 equivalence margin. A larger relay did not reconstruct absent fictional evidence.
- Exact B restoration strongly beat the same-width sham in both families, while source reopening supplied the positive ceiling. This localizes failure to evidence availability, but does not establish a general model-size bottleneck.
- Strict stage-2 noncompliance occurred in 57/1120 rows; 0 contained answer-bearing B evidence. Thus these were abstentions/invalid selections after an omission, not losses of available B support.

## Interpretation rules

- An upstream capability bottleneck requires better B retention/accuracy for L→L than S→L, with the relay and final answerer held fixed.
- 'A larger relay cannot recover an omission' requires an equivalence interval, not merely a non-significant S→L minus S→S contrast.
- Restore must beat its matched sham on stage-1 omissions, and source reopening must provide a positive ceiling, before failure is attributed to missing evidence.
- Evidence presence is computed from answer-bearing text, not only the designated card id, so cross-card answer mentions are not misclassified as omissions.
- Llama and Qwen are reported separately. A shared size/capability claim requires the direction to replicate across families.

## Reader baselines

| material | metric | mean | 95% CI | questions |
|---|---|---:|---:|---:|
| direct_cards | f1 | 0.959 | [0.931, 0.983] | 80 |
| direct_cards | judge_correct | 1.000 | [1.000, 1.000] | 80 |
| closed_book | f1 | 0.000 | [0.000, 0.000] | 80 |
| closed_book | judge_correct | 0.000 | [0.000, 0.000] | 80 |

QA-eligible arm rows: **1120/1120** (full-card correct and closed-book incorrect).

## Cost

- Initial full-run generation: **$0.056421**.
- Judge: **$0.004605**.
- Total: **$0.061026**.
