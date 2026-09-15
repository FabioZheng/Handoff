# Experiment 10 - communication regret under a hard communication budget

Corpus `relation_dossiers`: 16 contexts, 256 questions, 512 rotation x policy x budget cells. Sender and answerer `mistralai/ministral-8b-2512`; judge `openai/gpt-4o-mini`. Primary utility: LLM-judge accuracy.

`U_now` is the diagonal of each context's utility matrix (the message answered on the question it was conditioned on); `U_future` is the mean of the off-diagonal (the same message answered on questions the sender never saw). Every question rotates through both roles, so question difficulty cannot produce the contrast. All intervals are 95% percentile bootstrap over contexts, the unit the design repeats on.

## 1. Was the channel actually equal?

The hypothesis is only testable if no policy simply wrote more. Delivered length is capped deterministically, so the question is whether the arms *fill* the budget equally.

| policy | band (words) | delivered (mean +- sd) | fill | under floor | over cap pre-truncation | truncated | corrections | repetition |
|---|---|---|---|---|---|---|---|---|
| `conditioned` | 34-40 | 35.3 +- 4.8 | 0.88 | 0.16 | 0.17 | 0.17 | 0.97 | 0.000 |
| `generic` | 34-40 | 36.8 +- 2.7 | 0.92 | 0.19 | 0.25 | 0.25 | 1.00 | 0.000 |
| `oracle` | 34-40 | 32.4 +- 10.9 | 0.81 | 0.25 | 0.56 | 0.56 | 1.00 | 0.000 |
| `reusable` | 34-40 | 35.9 +- 3.1 | 0.90 | 0.08 | 0.09 | 0.09 | 0.56 | 0.000 |
| `conditioned` | 68-80 | 71.9 +- 5.2 | 0.90 | 0.12 | 0.19 | 0.19 | 0.84 | 0.000 |
| `generic` | 68-80 | 72.9 +- 4.3 | 0.91 | 0.00 | 0.06 | 0.06 | 0.69 | 0.000 |
| `oracle` | 68-80 | 71.8 +- 4.3 | 0.90 | 0.12 | 0.25 | 0.25 | 0.94 | 0.000 |
| `reusable` | 68-80 | 73.2 +- 4.2 | 0.92 | 0.06 | 0.11 | 0.11 | 0.55 | 0.000 |

Delivered messages over the cap after truncation: **0** arm(s) (must be zero; truncation is unconditional).

Paired length differences between arms (context-clustered):

| comparison | budget | delta words [95% CI] | p |
|---|---|---|---|
| conditioned_minus_generic | 40 | -1.406 [-2.984, 0.141] | 0.0818 |
| conditioned_minus_reusable | 40 | -0.562 [-2.125, 0.812] | 0.4618 |
| oracle_minus_conditioned | 40 | -2.906 [-8.547, 1.938] | 0.2762 |
| oracle_minus_generic | 40 | -4.312 [-10.625, 1.000] | 0.1542 |
| reusable_minus_generic | 40 | -0.844 [-2.250, 0.672] | 0.2596 |
| conditioned_minus_generic | 80 | -1.016 [-3.438, 1.188] | 0.3962 |
| conditioned_minus_reusable | 80 | -1.328 [-2.578, -0.094] | 0.0382 |
| oracle_minus_conditioned | 80 | -0.109 [-2.531, 2.031] | 0.9280 |
| oracle_minus_generic | 80 | -1.125 [-4.750, 1.812] | 0.5358 |
| reusable_minus_generic | 80 | 0.312 [-1.797, 2.312] | 0.7722 |

## 2. Present and future utility

| policy | budget | U_now [CI] | U_future [CI] | gap [CI] |
|---|---|---|---|---|
| `conditioned` | 40 | 0.984 [0.953, 1.000] | 0.134 [0.121, 0.149] | 0.850 [0.808, 0.879] |
| `conditioned` | 80 | 1.000 [1.000, 1.000] | 0.192 [0.167, 0.222] | 0.808 [0.778, 0.833] |
| `generic` | 40 | 0.547 [0.438, 0.656] | 0.405 [0.328, 0.476] | 0.142 [0.075, 0.212] |
| `generic` | 80 | 0.734 [0.625, 0.844] | 0.614 [0.552, 0.672] | 0.121 [0.050, 0.192] |
| `oracle` | 40 | 0.781 [0.625, 0.906] | 0.731 [0.596, 0.849] | 0.050 [0.004, 0.096] |
| `oracle` | 80 | 0.938 [0.875, 0.984] | 0.925 [0.866, 0.976] | 0.013 [-0.004, 0.029] |
| `reusable` | 40 | 0.984 [0.953, 1.000] | 0.208 [0.184, 0.233] | 0.776 [0.732, 0.812] |
| `reusable` | 80 | 1.000 [1.000, 1.000] | 0.397 [0.353, 0.444] | 0.603 [0.556, 0.647] |

Direct-context ceiling `U(D, q)`: 0.996 [0.988, 1.000]. Closed-book baseline (leakage audit): 0.004 [0.000, 0.010].

## 3. The hypothesis

A supporting result needs `delta U_now > 0` **and** `delta U_future < 0` for `conditioned` against a generic or reusable baseline, strengthening as the budget shrinks.

| comparison | budget | delta U_now [CI] (p, dz) | delta U_future [CI] (p, dz) | gap DiD [CI] (p) |
|---|---|---|---|---|
| conditioned_minus_generic | 40 | 0.438 [0.328, 0.562] (0.000, 1.75) | -0.271 [-0.346, -0.189] (0.000, -1.64) | 0.708 [0.638, 0.775] (0.000) |
| conditioned_minus_generic | 80 | 0.266 [0.156, 0.375] (0.000, 1.14) | -0.422 [-0.480, -0.360] (0.000, -3.36) | 0.688 [0.611, 0.761] (0.000) |
| conditioned_minus_reusable | 40 | 0.000 [0.000, 0.000] (1.000, n/a) | -0.074 [-0.096, -0.053] (0.000, -1.65) | 0.074 [0.054, 0.096] (0.000) |
| conditioned_minus_reusable | 80 | 0.000 [0.000, 0.000] (1.000, n/a) | -0.205 [-0.252, -0.158] (0.000, -2.08) | 0.205 [0.159, 0.251] (0.000) |
| reusable_minus_generic | 40 | 0.438 [0.328, 0.562] (0.000, 1.75) | -0.197 [-0.265, -0.117] (0.000, -1.26) | 0.634 [0.560, 0.702] (0.000) |
| reusable_minus_generic | 80 | 0.266 [0.156, 0.375] (0.000, 1.14) | -0.217 [-0.272, -0.165] (0.000, -1.93) | 0.482 [0.403, 0.558] (0.000) |
| oracle_minus_conditioned | 40 | -0.203 [-0.359, -0.062] (0.009, -0.64) | 0.597 [0.466, 0.715] (0.000, 2.25) | -0.800 [-0.863, -0.731] (0.000) |
| oracle_minus_conditioned | 80 | -0.062 [-0.125, -0.016] (0.041, -0.56) | 0.733 [0.673, 0.785] (0.000, 6.14) | -0.796 [-0.823, -0.764] (0.000) |
| oracle_minus_generic | 40 | 0.234 [0.031, 0.422] (0.022, 0.57) | 0.326 [0.157, 0.482] (0.000, 0.94) | -0.092 [-0.175, -0.013] (0.025) |
| oracle_minus_generic | 80 | 0.203 [0.062, 0.328] (0.003, 0.73) | 0.311 [0.215, 0.402] (0.000, 1.61) | -0.108 [-0.171, -0.042] (0.001) |

### Does scarcity sharpen the trade-off?

| quantity | policy | gap(low) - gap(high) [CI] | p | dz |
|---|---|---|---|---|
| gap_low_minus_high | `conditioned` | 0.042 [-0.007, 0.084] | 0.070 | 0.43 |
| gap_low_minus_high | `generic` | 0.021 [-0.046, 0.088] | 0.587 | 0.15 |
| gap_low_minus_high | `oracle` | 0.037 [-0.013, 0.087] | 0.176 | 0.34 |
| gap_low_minus_high | `reusable` | 0.173 [0.112, 0.225] | 0.000 | 1.43 |
| conditioning_gap_low_minus_high | `conditioned_minus_generic` | 0.021 [-0.067, 0.107] | 0.638 | 0.12 |

## 4. Communication regret against the source

`R(q) = U(D, q) - U(m, q)`, the accuracy the handoff costs relative to reading the source directly.

| policy | budget | R_now [CI] | R_future [CI] | retained U_future on answerable cells [CI] |
|---|---|---|---|---|
| `conditioned` | 40 | 0.016 [0.000, 0.047] | 0.861 [0.843, 0.878] | 0.135 [0.121, 0.150] |
| `conditioned` | 80 | 0.000 [0.000, 0.000] | 0.804 [0.775, 0.828] | 0.192 [0.168, 0.222] |
| `generic` | 40 | 0.453 [0.344, 0.562] | 0.591 [0.516, 0.670] | 0.408 [0.329, 0.481] |
| `generic` | 80 | 0.266 [0.156, 0.375] | 0.382 [0.321, 0.446] | 0.612 [0.552, 0.670] |
| `oracle` | 40 | 0.219 [0.094, 0.375] | 0.265 [0.149, 0.397] | 0.732 [0.598, 0.849] |
| `oracle` | 80 | 0.062 [0.016, 0.125] | 0.071 [0.021, 0.129] | 0.928 [0.870, 0.979] |
| `reusable` | 40 | 0.016 [0.000, 0.047] | 0.788 [0.761, 0.812] | 0.208 [0.184, 0.233] |
| `reusable` | 80 | 0.000 [0.000, 0.000] | 0.599 [0.552, 0.644] | 0.399 [0.355, 0.445] |

`retained` is the ceiling-restricted normalisation: it averages `U(m, q)` only over cells the source itself answers, where `U(D, q) = 1` and the ratio has no near-zero denominator to guard.

The arithmetic ratio `mean U(m,q) / mean U(D,q)` is in `normalised_utility.csv`. Contexts whose own ceiling falls below 0.25 are excluded from it, because dividing by a near-zero ceiling turns noise into a number: 0 context-cell(s) were excluded on the future endpoint across all arms.

| policy | budget | retained U_now [CI] | retained U_future [CI] | n contexts | excluded (low ceiling) |
|---|---|---|---|---|---|
| `conditioned` | 40 | 0.984 [0.953, 1.000] | 0.135 [0.121, 0.151] | 16 | 0 |
| `conditioned` | 80 | 1.000 [1.000, 1.000] | 0.192 [0.168, 0.222] | 16 | 0 |
| `generic` | 40 | 0.547 [0.422, 0.656] | 0.408 [0.327, 0.483] | 16 | 0 |
| `generic` | 80 | 0.734 [0.625, 0.844] | 0.617 [0.551, 0.678] | 16 | 0 |
| `oracle` | 40 | 0.781 [0.625, 0.922] | 0.732 [0.597, 0.849] | 16 | 0 |
| `oracle` | 80 | 0.938 [0.875, 0.984] | 0.928 [0.869, 0.976] | 16 | 0 |
| `reusable` | 40 | 0.984 [0.953, 1.000] | 0.209 [0.185, 0.234] | 16 | 0 |
| `reusable` | 80 | 1.000 [1.000, 1.000] | 0.399 [0.354, 0.445] | 16 | 0 |

### Secondary metrics

EM and token F1 are computed for every answer and reported here in full, never replaced by the judge. F1 penalises answers that are correct but differently worded, which is exactly what compression produces, so a disagreement between the three is information rather than noise.

**Exact match** &mdash; `conditioned` minus `generic`:

| endpoint | 40 w | 80 w |
|---|---|---|
| dU_now | 0.406 [0.250, 0.547] | 0.266 [0.125, 0.391] |
| dU_future | -0.157 [-0.225, -0.091] | -0.305 [-0.372, -0.241] |

**Token F1** &mdash; `conditioned` minus `generic`:

| endpoint | 40 w | 80 w |
|---|---|---|
| dU_now | 0.321 [0.211, 0.435] | 0.196 [0.097, 0.293] |
| dU_future | -0.219 [-0.281, -0.156] | -0.360 [-0.412, -0.311] |

## 5. Pareto frontier

Cost is mean delivered words. A configuration is dominated when another is at least as good on both utilities and no more expensive, with one strict improvement.

| policy | requested cap | U_now | U_future | delivered cost (words) | 3D nondominated | within-cap 2D nondominated | dominated by |
|---|---|---|---|---|---|---|---|
| `conditioned` | 40 | 0.984 | 0.134 | 35.3 | yes |  | - |
| `conditioned` | 80 | 1.000 | 0.192 | 71.9 | yes |  | - |
| `generic` | 40 | 0.547 | 0.405 | 36.8 |  |  | oracle@40w |
| `generic` | 80 | 0.734 | 0.614 | 72.9 |  |  | oracle@40w;oracle@80w |
| `oracle` | 40 | 0.781 | 0.731 | 32.4 | yes | yes | - |
| `oracle` | 80 | 0.938 | 0.925 | 71.8 | yes | yes | - |
| `reusable` | 40 | 0.984 | 0.208 | 35.9 | yes | yes | - |
| `reusable` | 80 | 1.000 | 0.397 | 73.2 | yes | yes | - |

**Nondominated set (6 of 8 configurations):** `reusable`@80w, `conditioned`@80w, `reusable`@40w, `conditioned`@40w, `oracle`@80w, `oracle`@40w.

Cost enters that test, so the set above is not a curve: a configuration can survive purely by being cheaper than everything that beats it on utility. The figure facets the observations by requested cap and outlines the two-dimensional nondominated policies within each facet. Markers are deliberately not joined: policy is categorical, so a line or staircase would imply unevaluated intermediate choices. The discrete within-cap sets are listed here from best present-query utility to best future-query utility:

* **40 words** -- `reusable` (0.98, 0.21)  `oracle` (0.78, 0.73)
* **80 words** -- `reusable` (1.00, 0.40)  `oracle` (0.94, 0.93)

## 6. Length-matched subsample

Contexts where `generic`, `conditioned` and `reusable` delivered within 6 words of each other. This is the comparison that survives if section 1 shows the arms filled the budget differently.

| comparison | budget | endpoint | delta [CI] | p | n contexts |
|---|---|---|---|---|---|
| conditioned_minus_generic | 40 | future | -0.267 [-0.346, -0.177] | 0.000 | 14 |
| conditioned_minus_generic | 40 | now | 0.446 [0.321, 0.571] | 0.000 | 14 |
| conditioned_minus_reusable | 40 | future | -0.076 [-0.100, -0.052] | 0.000 | 14 |
| conditioned_minus_reusable | 40 | now | 0.000 [0.000, 0.000] | 1.000 | 14 |
| reusable_minus_generic | 40 | future | -0.190 [-0.261, -0.104] | 0.000 | 14 |
| reusable_minus_generic | 40 | now | 0.446 [0.321, 0.571] | 0.000 | 14 |
| conditioned_minus_generic | 80 | future | -0.414 [-0.478, -0.346] | 0.000 | 13 |
| conditioned_minus_generic | 80 | now | 0.288 [0.173, 0.404] | 0.000 | 13 |
| conditioned_minus_reusable | 80 | future | -0.199 [-0.254, -0.142] | 0.000 | 13 |
| conditioned_minus_reusable | 80 | now | 0.000 [0.000, 0.000] | 1.000 | 13 |
| reusable_minus_generic | 80 | future | -0.215 [-0.277, -0.154] | 0.000 | 13 |
| reusable_minus_generic | 80 | now | 0.288 [0.173, 0.404] | 0.000 | 13 |

## 7. Regret by distance from the conditioning query

Distances are read off the corpus design, not estimated from embeddings.

| policy | budget | paraphrase | same_entity | same_topic | orthogonal |
|---|---|---|---|---|---|
| `conditioned` | 40 | 0.016 [0.000, 0.047] | 0.594 [0.484, 0.672] | 0.531 [0.422, 0.641] | 0.982 [0.967, 0.995] |
| `conditioned` | 80 | 0.000 [0.000, 0.000] | 0.375 [0.250, 0.500] | 0.281 [0.188, 0.375] | 0.951 [0.921, 0.974] |
| `generic` | 40 | 0.453 [0.359, 0.562] | 0.562 [0.438, 0.703] | 0.859 [0.766, 0.953] | 0.582 [0.508, 0.664] |
| `generic` | 80 | 0.281 [0.172, 0.391] | 0.281 [0.156, 0.422] | 0.672 [0.594, 0.750] | 0.375 [0.312, 0.441] |
| `oracle` | 40 | 0.234 [0.109, 0.391] | 0.297 [0.188, 0.406] | 0.297 [0.156, 0.453] | 0.262 [0.148, 0.395] |
| `oracle` | 80 | 0.078 [0.016, 0.156] | 0.078 [0.031, 0.141] | 0.062 [0.016, 0.109] | 0.070 [0.023, 0.129] |
| `reusable` | 40 | 0.031 [0.000, 0.078] | 0.344 [0.250, 0.453] | 0.500 [0.375, 0.609] | 0.911 [0.884, 0.938] |
| `reusable` | 80 | 0.016 [0.000, 0.047] | 0.266 [0.172, 0.375] | 0.250 [0.156, 0.359] | 0.704 [0.654, 0.757] |

| policy | budget | R(orthogonal) - R(paraphrase) [CI] | p | dz |
|---|---|---|---|---|
| `conditioned` | 40 | 0.966 [0.931, 0.991] | 0.000 | 14.88 |
| `conditioned` | 80 | 0.951 [0.919, 0.974] | 0.000 | 16.57 |
| `generic` | 40 | 0.129 [0.070, 0.188] | 0.000 | 1.02 |
| `generic` | 80 | 0.094 [0.031, 0.156] | 0.004 | 0.70 |
| `oracle` | 40 | 0.027 [-0.016, 0.070] | 0.245 | 0.31 |
| `oracle` | 80 | -0.008 [-0.035, 0.016] | 0.652 | -0.14 |
| `reusable` | 40 | 0.880 [0.828, 0.924] | 0.000 | 8.73 |
| `reusable` | 80 | 0.689 [0.634, 0.745] | 0.000 | 5.86 |

## Verdict

Communication regret from conditioning is **supported** at budget(s) [40, 80] (present gain and future loss both excluded zero).

* 40-word budget: dU_now = 0.438 [0.328, 0.562], dU_future = -0.271 [-0.346, -0.189] - both directions significant.
* 80-word budget: dU_now = 0.266 [0.156, 0.375], dU_future = -0.422 [-0.480, -0.360] - both directions significant.

The conditioning-induced specialisation gap does not measurably change as the budget falls from 80 to 40 words (0.021 [-0.067, 0.107], p = 0.638).

## Provenance

```json
{
  "schema_version": "communication-regret-v1",
  "dataset": "relation_dossiers",
  "source_manifest": {
    "dataset": "relation_dossiers",
    "writer_model": "openai/gpt-5.2",
    "probe_model": "meta-llama/llama-3.1-8b-instruct",
    "judge_model": "openai/gpt-4o-mini",
    "aspects": [
      "founding",
      "facilities",
      "finance",
      "custom"
    ],
    "roles_per_aspect": [
      "anchor",
      "paraphrase",
      "same_entity",
      "same_topic"
    ],
    "pages_requested": 40,
    "pages_built": 38,
    "closed_book_method": "llm_judge",
    "closed_book_leaked_questions": 19,
    "kept": 16,
    "rejected": 12
  },
  "policies": [
    "conditioned",
    "generic",
    "oracle",
    "reusable"
  ],
  "budgets_words": [
    40,
    80
  ],
  "primary_metric": "judge_correct"
}
```
