# Experiment 8b: fictional-QA selector bottleneck

- Dossiers: **1**; A/B rotations: **4**.
- Channel: **3** selector slots → **1** relay slots.
- Stage 1 sees the fictional source and A. Stage 2 is newly shown B; sealed arms receive no source. The final answerer is fixed and from a third family.
- Intervals and paired tests resample dossiers, not rotations.
- ITT counts schema failures as failures; compliant results require both stages to return a valid fixed-K selection.

> **Pilot scale.** Treat effect sizes as design diagnostics, not confirmatory population estimates.

## Primary paired contrasts (ITT)

| family | comparison | population | delta | 95% CI | p | dossiers |
|---|---|---|---:|---|---:|---:|
| llama | `llama:early_selector_at_large_relay` | all | +0.000 | [+0.000, +0.000] | 1.000 | 1 |
| llama | `llama:larger_relay_after_small_selector` | all | +0.000 | [+0.000, +0.000] | 1.000 | 1 |
| llama | `llama:smaller_relay_after_large_selector` | all | +0.000 | [+0.000, +0.000] | 1.000 | 1 |
| llama | `llama:large_first_order` | all | +0.000 | [+0.000, +0.000] | 1.000 | 1 |
| qwen | `qwen:early_selector_at_large_relay` | all | -0.375 | [-0.375, -0.375] | 1.000 | 1 |
| qwen | `qwen:larger_relay_after_small_selector` | all | +0.000 | [+0.000, +0.000] | 1.000 | 1 |
| qwen | `qwen:smaller_relay_after_large_selector` | all | +0.000 | [+0.000, +0.000] | 1.000 | 1 |
| qwen | `qwen:large_first_order` | all | -0.375 | [-0.375, -0.375] | 1.000 | 1 |

## Interpretation rules

- An upstream capability bottleneck requires better B retention/accuracy for L→L than S→L, with the relay and final answerer held fixed.
- 'A larger relay cannot recover an omission' requires an equivalence interval, not merely a non-significant S→L minus S→S contrast.
- Restore must beat its matched sham on stage-1 omissions, and source reopening must provide a positive ceiling, before failure is attributed to missing evidence.
- Llama and Qwen are reported separately. A shared size/capability claim requires the direction to replicate across families.

## Reader baselines

| material | metric | mean | 95% CI | questions |
|---|---|---:|---:|---:|
| direct_cards | f1 | 1.000 | [1.000, 1.000] | 2 |
| closed_book | f1 | 0.000 | [0.000, 0.000] | 2 |

QA-eligible arm rows: **56/56** (full-card correct and closed-book incorrect).

## Cost

`{"api_call_cap": 30000, "api_calls_attempted": 4, "cache_hit_rate": 0.9272727272727272, "calls_cached": 51, "calls_live": 4, "calls_total": 55, "completion_tokens": 648, "completion_tokens_live": 17, "cost_usd": 4.9e-05, "prompt_tokens": 14406, "prompt_tokens_live": 604}`
