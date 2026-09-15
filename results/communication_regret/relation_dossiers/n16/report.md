# Experiment 10 - communication regret under a hard communication budget

Corpus `relation_dossiers`: 16 contexts, 256 questions, 1024 rotation x policy x budget cells. Sender and answerer `meta-llama/llama-3.1-8b-instruct`; judge `openai/gpt-4o-mini`. Primary utility: LLM-judge accuracy.

`U_now` is the diagonal of each context's utility matrix (the message answered on the question it was conditioned on); `U_future` is the mean of the off-diagonal (the same message answered on questions the sender never saw). Every question rotates through both roles, so question difficulty cannot produce the contrast. All intervals are 95% percentile bootstrap over contexts, the unit the design repeats on.

## 1. Was the channel actually equal?

The hypothesis is only testable if no policy simply wrote more. Delivered length is capped deterministically, so the question is whether the arms *fill* the budget equally.

| policy | band (words) | delivered (mean +- sd) | fill | under floor | over cap pre-truncation | truncated | corrections | repetition |
|---|---|---|---|---|---|---|---|---|
| `conditioned` | 17-20 | 18.3 +- 1.0 | 0.91 | 0.00 | 0.00 | 0.00 | 0.45 | 0.000 |
| `generic` | 17-20 | 18.8 +- 1.0 | 0.94 | 0.00 | 0.00 | 0.00 | 0.44 | 0.000 |
| `oracle` | 17-20 | 18.1 +- 1.1 | 0.90 | 0.00 | 0.00 | 0.00 | 0.81 | 0.000 |
| `reusable` | 17-20 | 18.3 +- 1.0 | 0.92 | 0.00 | 0.02 | 0.02 | 0.52 | 0.000 |
| `conditioned` | 34-40 | 36.5 +- 2.0 | 0.91 | 0.00 | 0.03 | 0.03 | 0.67 | 0.000 |
| `generic` | 34-40 | 37.1 +- 1.9 | 0.93 | 0.00 | 0.00 | 0.00 | 0.75 | 0.000 |
| `oracle` | 34-40 | 36.8 +- 1.9 | 0.92 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000 |
| `reusable` | 34-40 | 36.8 +- 1.9 | 0.92 | 0.00 | 0.00 | 0.00 | 0.38 | 0.000 |
| `conditioned` | 68-80 | 72.9 +- 3.3 | 0.91 | 0.00 | 0.00 | 0.00 | 0.67 | 0.000 |
| `generic` | 68-80 | 73.2 +- 3.5 | 0.92 | 0.00 | 0.00 | 0.00 | 0.38 | 0.000 |
| `oracle` | 68-80 | 75.6 +- 2.5 | 0.95 | 0.00 | 0.00 | 0.00 | 0.81 | 0.000 |
| `reusable` | 68-80 | 72.9 +- 2.9 | 0.91 | 0.00 | 0.00 | 0.00 | 0.55 | 0.000 |
| `conditioned` | 136-160 | 143.9 +- 5.6 | 0.90 | 0.00 | 0.00 | 0.00 | 0.88 | 0.002 |
| `generic` | 136-160 | 143.6 +- 4.3 | 0.90 | 0.00 | 0.00 | 0.00 | 0.62 | 0.000 |
| `oracle` | 136-160 | 143.7 +- 6.8 | 0.90 | 0.00 | 0.00 | 0.00 | 0.38 | 0.001 |
| `reusable` | 136-160 | 143.3 +- 4.8 | 0.90 | 0.00 | 0.00 | 0.00 | 0.78 | 0.001 |

Delivered messages over the cap after truncation: **0** arm(s) (must be zero; truncation is unconditional).

Paired length differences between arms (context-clustered):

| comparison | budget | delta words [95% CI] | p |
|---|---|---|---|
| conditioned_minus_generic | 20 | -0.469 [-1.000, 0.078] | 0.0907 |
| conditioned_minus_reusable | 20 | -0.062 [-0.375, 0.234] | 0.7273 |
| oracle_minus_conditioned | 20 | -0.219 [-0.766, 0.328] | 0.4560 |
| oracle_minus_generic | 20 | -0.688 [-1.250, -0.125] | 0.0226 |
| reusable_minus_generic | 20 | -0.406 [-0.938, 0.141] | 0.1490 |
| conditioned_minus_generic | 40 | -0.656 [-1.469, 0.141] | 0.1157 |
| conditioned_minus_reusable | 40 | -0.328 [-0.719, 0.031] | 0.0935 |
| oracle_minus_conditioned | 40 | 0.281 [-0.735, 1.344] | 0.6105 |
| oracle_minus_generic | 40 | -0.375 [-1.625, 0.875] | 0.5930 |
| reusable_minus_generic | 40 | -0.328 [-1.219, 0.547] | 0.4893 |
| conditioned_minus_generic | 80 | -0.375 [-2.156, 1.375] | 0.6873 |
| conditioned_minus_reusable | 80 | -0.062 [-1.125, 1.016] | 0.9232 |
| oracle_minus_conditioned | 80 | 2.750 [1.281, 4.297] | 0.0008 |
| oracle_minus_generic | 80 | 2.375 [0.875, 3.750] | 0.0018 |
| reusable_minus_generic | 80 | -0.312 [-1.953, 1.344] | 0.7305 |
| conditioned_minus_generic | 160 | 0.328 [-2.344, 3.047] | 0.8123 |
| conditioned_minus_reusable | 160 | 0.609 [-1.234, 2.281] | 0.5087 |
| oracle_minus_conditioned | 160 | -0.203 [-3.813, 3.688] | 0.9191 |
| oracle_minus_generic | 160 | 0.125 [-4.000, 4.312] | 0.9663 |
| reusable_minus_generic | 160 | -0.281 [-3.062, 2.531] | 0.8600 |

## 2. Present and future utility

| policy | budget | U_now [CI] | U_future [CI] | gap [CI] |
|---|---|---|---|---|
| `conditioned` | 20 | 1.000 [1.000, 1.000] | 0.077 [0.070, 0.085] | 0.923 [0.915, 0.930] |
| `conditioned` | 40 | 0.984 [0.953, 1.000] | 0.096 [0.085, 0.107] | 0.889 [0.849, 0.914] |
| `conditioned` | 80 | 0.984 [0.953, 1.000] | 0.117 [0.103, 0.131] | 0.868 [0.831, 0.893] |
| `conditioned` | 160 | 0.969 [0.922, 1.000] | 0.168 [0.149, 0.186] | 0.801 [0.751, 0.840] |
| `generic` | 20 | 0.141 [0.078, 0.203] | 0.103 [0.077, 0.126] | 0.038 [-0.012, 0.083] |
| `generic` | 40 | 0.344 [0.266, 0.438] | 0.206 [0.168, 0.248] | 0.138 [0.083, 0.196] |
| `generic` | 80 | 0.469 [0.391, 0.562] | 0.315 [0.259, 0.368] | 0.154 [0.104, 0.213] |
| `generic` | 160 | 0.609 [0.516, 0.703] | 0.509 [0.452, 0.569] | 0.100 [0.029, 0.171] |
| `oracle` | 20 | 0.250 [0.250, 0.250] | 0.179 [0.163, 0.196] | 0.071 [0.054, 0.087] |
| `oracle` | 40 | 0.516 [0.438, 0.594] | 0.399 [0.363, 0.434] | 0.117 [0.054, 0.175] |
| `oracle` | 80 | 0.750 [0.672, 0.828] | 0.692 [0.625, 0.751] | 0.058 [0.008, 0.104] |
| `oracle` | 160 | 0.891 [0.812, 0.953] | 0.828 [0.755, 0.898] | 0.062 [0.025, 0.100] |
| `reusable` | 20 | 1.000 [1.000, 1.000] | 0.081 [0.076, 0.086] | 0.919 [0.914, 0.924] |
| `reusable` | 40 | 1.000 [1.000, 1.000] | 0.103 [0.093, 0.114] | 0.897 [0.886, 0.907] |
| `reusable` | 80 | 1.000 [1.000, 1.000] | 0.197 [0.184, 0.209] | 0.803 [0.791, 0.816] |
| `reusable` | 160 | 0.953 [0.906, 1.000] | 0.334 [0.302, 0.364] | 0.619 [0.551, 0.678] |

Direct-context ceiling `U(D, q)`: 0.992 [0.977, 1.000]. Closed-book baseline (leakage audit): 0.000 [0.000, 0.000].

## 3. The hypothesis

A supporting result needs `delta U_now > 0` **and** `delta U_future < 0` for `conditioned` against a generic or reusable baseline, strengthening as the budget shrinks.

| comparison | budget | delta U_now [CI] (p, dz) | delta U_future [CI] (p, dz) | gap DiD [CI] (p) |
|---|---|---|---|---|
| conditioned_minus_generic | 20 | 0.859 [0.797, 0.922] (0.000, 6.71) | -0.026 [-0.051, 0.001] (0.055, -0.47) | 0.885 [0.834, 0.940] (0.000) |
| conditioned_minus_generic | 40 | 0.641 [0.547, 0.719] (0.000, 3.52) | -0.110 [-0.152, -0.070] (0.000, -1.30) | 0.751 [0.688, 0.811] (0.000) |
| conditioned_minus_generic | 80 | 0.516 [0.406, 0.609] (0.000, 2.42) | -0.198 [-0.257, -0.136] (0.000, -1.55) | 0.714 [0.639, 0.777] (0.000) |
| conditioned_minus_generic | 160 | 0.359 [0.250, 0.469] (0.000, 1.61) | -0.342 [-0.396, -0.292] (0.000, -3.09) | 0.701 [0.613, 0.786] (0.000) |
| conditioned_minus_reusable | 20 | 0.000 [0.000, 0.000] (1.000, n/a) | -0.004 [-0.011, 0.004] (0.328, -0.25) | 0.004 [-0.004, 0.011] (0.296) |
| conditioned_minus_reusable | 40 | -0.016 [-0.047, 0.000] (0.621, -0.25) | -0.007 [-0.021, 0.006] (0.321, -0.26) | -0.008 [-0.046, 0.019] (0.624) |
| conditioned_minus_reusable | 80 | -0.016 [-0.047, 0.000] (0.622, -0.25) | -0.080 [-0.098, -0.064] (0.000, -2.22) | 0.065 [0.027, 0.094] (0.001) |
| conditioned_minus_reusable | 160 | 0.016 [-0.031, 0.062] (0.767, 0.14) | -0.167 [-0.190, -0.143] (0.000, -3.37) | 0.182 [0.121, 0.246] (0.000) |
| reusable_minus_generic | 20 | 0.859 [0.797, 0.922] (0.000, 6.71) | -0.022 [-0.045, 0.002] (0.077, -0.44) | 0.881 [0.831, 0.934] (0.000) |
| reusable_minus_generic | 40 | 0.656 [0.562, 0.734] (0.000, 3.65) | -0.103 [-0.141, -0.066] (0.000, -1.29) | 0.759 [0.700, 0.813] (0.000) |
| reusable_minus_generic | 80 | 0.531 [0.453, 0.609] (0.000, 2.96) | -0.118 [-0.168, -0.065] (0.000, -1.08) | 0.649 [0.594, 0.697] (0.000) |
| reusable_minus_generic | 160 | 0.344 [0.234, 0.453] (0.000, 1.44) | -0.175 [-0.241, -0.115] (0.000, -1.31) | 0.519 [0.430, 0.608] (0.000) |
| oracle_minus_conditioned | 20 | -0.750 [-0.750, -0.750] (0.000, n/a) | 0.102 [0.084, 0.121] (0.000, 2.66) | -0.852 [-0.871, -0.834] (0.000) |
| oracle_minus_conditioned | 40 | -0.469 [-0.547, -0.391] (0.000, -2.61) | 0.303 [0.263, 0.344] (0.000, 3.51) | -0.772 [-0.841, -0.702] (0.000) |
| oracle_minus_conditioned | 80 | -0.234 [-0.328, -0.141] (0.000, -1.21) | 0.575 [0.506, 0.635] (0.000, 4.25) | -0.809 [-0.860, -0.752] (0.000) |
| oracle_minus_conditioned | 160 | -0.078 [-0.172, 0.000] (0.100, -0.44) | 0.660 [0.583, 0.731] (0.000, 4.28) | -0.739 [-0.793, -0.676] (0.000) |
| oracle_minus_generic | 20 | 0.109 [0.047, 0.172] (0.001, 0.85) | 0.076 [0.044, 0.112] (0.000, 1.07) | 0.033 [-0.021, 0.092] (0.283) |
| oracle_minus_generic | 40 | 0.172 [0.047, 0.297] (0.011, 0.64) | 0.193 [0.134, 0.252] (0.000, 1.56) | -0.021 [-0.108, 0.062] (0.625) |
| oracle_minus_generic | 80 | 0.281 [0.188, 0.375] (0.000, 1.40) | 0.377 [0.296, 0.446] (0.000, 2.39) | -0.096 [-0.163, -0.029] (0.005) |
| oracle_minus_generic | 160 | 0.281 [0.141, 0.422] (0.000, 0.98) | 0.319 [0.208, 0.422] (0.000, 1.41) | -0.038 [-0.121, 0.050] (0.361) |

### Does scarcity sharpen the trade-off?

| quantity | policy | gap(low) - gap(high) [CI] | p | dz |
|---|---|---|---|---|
| gap_low_minus_high | `conditioned` | 0.122 [0.080, 0.170] | 0.000 | 1.27 |
| gap_low_minus_high | `generic` | -0.062 [-0.138, 0.017] | 0.134 | -0.37 |
| gap_low_minus_high | `oracle` | 0.008 [-0.025, 0.042] | 0.716 | 0.11 |
| gap_low_minus_high | `reusable` | 0.300 [0.240, 0.367] | 0.000 | 2.20 |
| conditioning_gap_low_minus_high | `conditioned_minus_generic` | 0.184 [0.080, 0.285] | 0.000 | 0.85 |

## 4. Communication regret against the source

`R(q) = U(D, q) - U(m, q)`, the accuracy the handoff costs relative to reading the source directly.

| policy | budget | R_now [CI] | R_future [CI] | retained U_future on answerable cells [CI] |
|---|---|---|---|---|
| `conditioned` | 20 | -0.016 [-0.047, 0.000] | 0.916 [0.899, 0.928] | 0.077 [0.069, 0.085] |
| `conditioned` | 40 | 0.000 [0.000, 0.000] | 0.897 [0.875, 0.914] | 0.094 [0.084, 0.106] |
| `conditioned` | 80 | 0.000 [-0.047, 0.047] | 0.876 [0.853, 0.895] | 0.117 [0.103, 0.131] |
| `conditioned` | 160 | 0.016 [-0.031, 0.062] | 0.825 [0.798, 0.849] | 0.167 [0.149, 0.185] |
| `generic` | 20 | 0.844 [0.781, 0.906] | 0.890 [0.866, 0.917] | 0.104 [0.078, 0.127] |
| `generic` | 40 | 0.641 [0.547, 0.734] | 0.786 [0.745, 0.827] | 0.208 [0.169, 0.249] |
| `generic` | 80 | 0.516 [0.422, 0.609] | 0.678 [0.622, 0.736] | 0.318 [0.261, 0.372] |
| `generic` | 160 | 0.375 [0.266, 0.484] | 0.483 [0.417, 0.546] | 0.507 [0.452, 0.564] |
| `oracle` | 20 | 0.734 [0.703, 0.750] | 0.814 [0.790, 0.838] | 0.181 [0.163, 0.199] |
| `oracle` | 40 | 0.469 [0.391, 0.547] | 0.594 [0.561, 0.628] | 0.398 [0.360, 0.433] |
| `oracle` | 80 | 0.234 [0.156, 0.312] | 0.301 [0.235, 0.371] | 0.698 [0.628, 0.763] |
| `oracle` | 160 | 0.094 [0.031, 0.156] | 0.165 [0.101, 0.231] | 0.828 [0.755, 0.898] |
| `reusable` | 20 | -0.016 [-0.047, 0.000] | 0.911 [0.897, 0.921] | 0.081 [0.075, 0.086] |
| `reusable` | 40 | -0.016 [-0.047, 0.000] | 0.890 [0.869, 0.905] | 0.102 [0.092, 0.112] |
| `reusable` | 80 | -0.016 [-0.047, 0.000] | 0.796 [0.774, 0.815] | 0.196 [0.184, 0.209] |
| `reusable` | 160 | 0.031 [0.000, 0.078] | 0.658 [0.619, 0.696] | 0.333 [0.301, 0.361] |

`retained` is the ceiling-restricted normalisation: it averages `U(m, q)` only over cells the source itself answers, where `U(D, q) = 1` and the ratio has no near-zero denominator to guard.

The arithmetic ratio `mean U(m,q) / mean U(D,q)` is in `normalised_utility.csv`. Contexts whose own ceiling falls below 0.25 are excluded from it, because dividing by a near-zero ceiling turns noise into a number: 0 context-cell(s) were excluded on the future endpoint across all arms.

| policy | budget | retained U_now [CI] | retained U_future [CI] | n contexts | excluded (low ceiling) |
|---|---|---|---|---|---|
| `conditioned` | 20 | 1.021 [1.000, 1.062] | 0.078 [0.071, 0.085] | 16 | 0 |
| `conditioned` | 40 | 1.000 [1.000, 1.000] | 0.097 [0.085, 0.109] | 16 | 0 |
| `conditioned` | 80 | 1.005 [0.953, 1.062] | 0.118 [0.104, 0.132] | 16 | 0 |
| `conditioned` | 160 | 0.990 [0.938, 1.047] | 0.169 [0.150, 0.189] | 16 | 0 |
| `generic` | 20 | 0.141 [0.078, 0.203] | 0.104 [0.078, 0.127] | 16 | 0 |
| `generic` | 40 | 0.349 [0.266, 0.438] | 0.208 [0.169, 0.247] | 16 | 0 |
| `generic` | 80 | 0.479 [0.391, 0.573] | 0.318 [0.261, 0.374] | 16 | 0 |
| `generic` | 160 | 0.625 [0.516, 0.734] | 0.515 [0.455, 0.583] | 16 | 0 |
| `oracle` | 20 | 0.255 [0.250, 0.266] | 0.181 [0.163, 0.199] | 16 | 0 |
| `oracle` | 40 | 0.526 [0.443, 0.609] | 0.401 [0.367, 0.435] | 16 | 0 |
| `oracle` | 80 | 0.766 [0.688, 0.844] | 0.698 [0.628, 0.762] | 16 | 0 |
| `oracle` | 160 | 0.901 [0.833, 0.964] | 0.833 [0.765, 0.902] | 16 | 0 |
| `reusable` | 20 | 1.021 [1.000, 1.062] | 0.082 [0.077, 0.086] | 16 | 0 |
| `reusable` | 40 | 1.021 [1.000, 1.062] | 0.104 [0.094, 0.115] | 16 | 0 |
| `reusable` | 80 | 1.021 [1.000, 1.062] | 0.199 [0.185, 0.213] | 16 | 0 |
| `reusable` | 160 | 0.969 [0.922, 1.000] | 0.338 [0.303, 0.371] | 16 | 0 |

### Secondary metrics

EM and token F1 are computed for every answer and reported here in full, never replaced by the judge. F1 penalises answers that are correct but differently worded, which is exactly what compression produces, so a disagreement between the three is information rather than noise.

**Exact match** &mdash; `conditioned` minus `generic`:

| endpoint | 20 w | 40 w | 80 w | 160 w |
|---|---|---|---|---|
| dU_now | 0.703 [0.609, 0.781] | 0.531 [0.438, 0.609] | 0.516 [0.406, 0.625] | 0.344 [0.250, 0.438] |
| dU_future | -0.012 [-0.039, 0.014] | -0.074 [-0.111, -0.042] | -0.081 [-0.142, -0.023] | -0.255 [-0.326, -0.193] |

**Token F1** &mdash; `conditioned` minus `generic`:

| endpoint | 20 w | 40 w | 80 w | 160 w |
|---|---|---|---|---|
| dU_now | 0.732 [0.679, 0.783] | 0.567 [0.492, 0.637] | 0.498 [0.386, 0.605] | 0.308 [0.222, 0.398] |
| dU_future | -0.040 [-0.073, -0.007] | -0.109 [-0.148, -0.074] | -0.158 [-0.209, -0.107] | -0.322 [-0.379, -0.271] |

## 5. Pareto frontier

Cost is mean delivered words. A configuration is dominated when another is at least as good on both utilities and no more expensive, with one strict improvement.

| policy | requested cap | U_now | U_future | delivered cost (words) | 3D nondominated | within-cap 2D nondominated | dominated by |
|---|---|---|---|---|---|---|---|
| `conditioned` | 20 | 1.000 | 0.077 | 18.3 | yes |  | - |
| `conditioned` | 40 | 0.984 | 0.096 | 36.5 | yes |  | - |
| `conditioned` | 80 | 0.984 | 0.117 | 72.9 | yes |  | - |
| `conditioned` | 160 | 0.969 | 0.168 | 143.9 |  | yes | reusable@80w |
| `generic` | 20 | 0.141 | 0.103 | 18.8 |  |  | oracle@20w |
| `generic` | 40 | 0.344 | 0.206 | 37.1 |  |  | oracle@40w |
| `generic` | 80 | 0.469 | 0.315 | 73.2 |  |  | oracle@40w |
| `generic` | 160 | 0.609 | 0.509 | 143.6 |  |  | oracle@80w |
| `oracle` | 20 | 0.250 | 0.179 | 18.1 | yes | yes | - |
| `oracle` | 40 | 0.516 | 0.399 | 36.8 | yes | yes | - |
| `oracle` | 80 | 0.750 | 0.692 | 75.6 | yes | yes | - |
| `oracle` | 160 | 0.891 | 0.828 | 143.7 | yes | yes | - |
| `reusable` | 20 | 1.000 | 0.081 | 18.3 | yes | yes | - |
| `reusable` | 40 | 1.000 | 0.103 | 36.8 | yes | yes | - |
| `reusable` | 80 | 1.000 | 0.197 | 72.9 | yes | yes | - |
| `reusable` | 160 | 0.953 | 0.334 | 143.3 | yes | yes | - |

**Nondominated set (11 of 16 configurations):** `reusable`@80w, `reusable`@40w, `reusable`@20w, `conditioned`@20w, `conditioned`@80w, `conditioned`@40w, `reusable`@160w, `oracle`@160w, `oracle`@80w, `oracle`@40w, `oracle`@20w.

Cost enters that test, so the set above is not a curve: a configuration can survive purely by being cheaper than everything that beats it on utility. The figure facets the observations by requested cap and outlines the two-dimensional nondominated policies within each facet. Markers are deliberately not joined: policy is categorical, so a line or staircase would imply unevaluated intermediate choices. The discrete within-cap sets are listed here from best present-query utility to best future-query utility:

* **20 words** -- `reusable` (1.00, 0.08)  `oracle` (0.25, 0.18)
* **40 words** -- `reusable` (1.00, 0.10)  `oracle` (0.52, 0.40)
* **80 words** -- `reusable` (1.00, 0.20)  `oracle` (0.75, 0.69)
* **160 words** -- `conditioned` (0.97, 0.17)  `reusable` (0.95, 0.33)  `oracle` (0.89, 0.83)

## 6. Length-matched subsample

Contexts where `generic`, `conditioned` and `reusable` delivered within 6 words of each other. This is the comparison that survives if section 1 shows the arms filled the budget differently.

| comparison | budget | endpoint | delta [CI] | p | n contexts |
|---|---|---|---|---|---|
| conditioned_minus_generic | 20 | future | -0.026 [-0.051, 0.002] | 0.059 | 16 |
| conditioned_minus_generic | 20 | now | 0.859 [0.797, 0.922] | 0.000 | 16 |
| conditioned_minus_reusable | 20 | future | -0.004 [-0.011, 0.004] | 0.326 | 16 |
| conditioned_minus_reusable | 20 | now | 0.000 [0.000, 0.000] | 1.000 | 16 |
| reusable_minus_generic | 20 | future | -0.022 [-0.045, 0.003] | 0.076 | 16 |
| reusable_minus_generic | 20 | now | 0.859 [0.797, 0.922] | 0.000 | 16 |
| conditioned_minus_generic | 40 | future | -0.110 [-0.153, -0.072] | 0.000 | 16 |
| conditioned_minus_generic | 40 | now | 0.641 [0.547, 0.719] | 0.000 | 16 |
| conditioned_minus_reusable | 40 | future | -0.007 [-0.021, 0.006] | 0.325 | 16 |
| conditioned_minus_reusable | 40 | now | -0.016 [-0.047, 0.000] | 0.612 | 16 |
| reusable_minus_generic | 40 | future | -0.103 [-0.143, -0.066] | 0.000 | 16 |
| reusable_minus_generic | 40 | now | 0.656 [0.562, 0.734] | 0.000 | 16 |
| conditioned_minus_generic | 80 | future | -0.210 [-0.268, -0.148] | 0.000 | 15 |
| conditioned_minus_generic | 80 | now | 0.500 [0.383, 0.600] | 0.000 | 15 |
| conditioned_minus_reusable | 80 | future | -0.082 [-0.100, -0.064] | 0.000 | 15 |
| conditioned_minus_reusable | 80 | now | -0.017 [-0.050, 0.000] | 0.619 | 15 |
| reusable_minus_generic | 80 | future | -0.128 [-0.178, -0.076] | 0.000 | 15 |
| reusable_minus_generic | 80 | now | 0.517 [0.433, 0.600] | 0.000 | 15 |
| conditioned_minus_generic | 160 | future | -0.335 [-0.390, -0.287] | 0.000 | 8 |
| conditioned_minus_generic | 160 | now | 0.344 [0.281, 0.438] | 0.000 | 8 |
| conditioned_minus_reusable | 160 | future | -0.167 [-0.198, -0.131] | 0.000 | 8 |
| conditioned_minus_reusable | 160 | now | 0.031 [0.000, 0.094] | 0.604 | 8 |
| reusable_minus_generic | 160 | future | -0.169 [-0.248, -0.098] | 0.000 | 8 |
| reusable_minus_generic | 160 | now | 0.312 [0.188, 0.438] | 0.000 | 8 |

## 7. Regret by distance from the conditioning query

Distances are read off the corpus design, not estimated from embeddings.

| policy | budget | paraphrase | same_entity | same_topic | orthogonal |
|---|---|---|---|---|---|
| `conditioned` | 20 | 0.000 [-0.047, 0.047] | 0.953 [0.906, 1.000] | 0.984 [0.953, 1.000] | 0.983 [0.966, 0.996] |
| `conditioned` | 40 | -0.016 [-0.047, 0.000] | 0.875 [0.812, 0.938] | 0.828 [0.750, 0.906] | 0.980 [0.961, 0.995] |
| `conditioned` | 80 | 0.016 [0.000, 0.047] | 0.734 [0.625, 0.844] | 0.688 [0.594, 0.781] | 0.975 [0.956, 0.991] |
| `conditioned` | 160 | 0.016 [-0.031, 0.062] | 0.422 [0.312, 0.531] | 0.516 [0.391, 0.641] | 0.952 [0.926, 0.975] |
| `generic` | 20 | 0.844 [0.781, 0.906] | 0.891 [0.812, 0.953] | 0.969 [0.922, 1.000] | 0.887 [0.859, 0.914] |
| `generic` | 40 | 0.656 [0.562, 0.734] | 0.875 [0.781, 0.953] | 0.938 [0.875, 0.984] | 0.777 [0.734, 0.820] |
| `generic` | 80 | 0.516 [0.422, 0.609] | 0.812 [0.750, 0.875] | 0.828 [0.734, 0.922] | 0.668 [0.609, 0.727] |
| `generic` | 160 | 0.375 [0.266, 0.484] | 0.438 [0.328, 0.547] | 0.719 [0.609, 0.828] | 0.477 [0.410, 0.539] |
| `oracle` | 20 | 0.734 [0.703, 0.750] | 0.828 [0.781, 0.891] | 0.938 [0.875, 0.984] | 0.809 [0.785, 0.832] |
| `oracle` | 40 | 0.500 [0.406, 0.594] | 0.672 [0.594, 0.750] | 0.703 [0.656, 0.750] | 0.586 [0.551, 0.621] |
| `oracle` | 80 | 0.234 [0.156, 0.312] | 0.359 [0.266, 0.453] | 0.359 [0.281, 0.438] | 0.297 [0.230, 0.367] |
| `oracle` | 160 | 0.094 [0.031, 0.156] | 0.219 [0.141, 0.297] | 0.234 [0.125, 0.359] | 0.160 [0.094, 0.227] |
| `reusable` | 20 | 0.000 [-0.047, 0.047] | 0.953 [0.906, 1.000] | 0.938 [0.875, 0.984] | 0.982 [0.965, 0.993] |
| `reusable` | 40 | -0.016 [-0.047, 0.000] | 0.812 [0.688, 0.922] | 0.750 [0.672, 0.828] | 0.983 [0.962, 0.996] |
| `reusable` | 80 | -0.016 [-0.047, 0.000] | 0.328 [0.219, 0.438] | 0.562 [0.453, 0.672] | 0.922 [0.895, 0.945] |
| `reusable` | 160 | 0.031 [0.000, 0.078] | 0.266 [0.172, 0.375] | 0.328 [0.234, 0.438] | 0.771 [0.723, 0.818] |

| policy | budget | R(orthogonal) - R(paraphrase) [CI] | p | dz |
|---|---|---|---|---|
| `conditioned` | 20 | 0.983 [0.941, 1.016] | 0.000 | 12.78 |
| `conditioned` | 40 | 0.996 [0.982, 1.014] | 0.000 | 29.87 |
| `conditioned` | 80 | 0.960 [0.921, 0.988] | 0.000 | 13.26 |
| `conditioned` | 160 | 0.936 [0.884, 0.982] | 0.000 | 9.09 |
| `generic` | 20 | 0.043 [-0.004, 0.086] | 0.075 | 0.46 |
| `generic` | 40 | 0.121 [0.070, 0.176] | 0.000 | 1.10 |
| `generic` | 80 | 0.152 [0.102, 0.211] | 0.000 | 1.36 |
| `generic` | 160 | 0.102 [0.035, 0.164] | 0.002 | 0.73 |
| `oracle` | 20 | 0.074 [0.055, 0.098] | 0.000 | 1.58 |
| `oracle` | 40 | 0.086 [0.027, 0.145] | 0.006 | 0.68 |
| `oracle` | 80 | 0.062 [0.020, 0.105] | 0.005 | 0.68 |
| `oracle` | 160 | 0.066 [0.027, 0.113] | 0.005 | 0.72 |
| `reusable` | 20 | 0.982 [0.940, 1.014] | 0.000 | 12.85 |
| `reusable` | 40 | 0.999 [0.986, 1.016] | 0.000 | 31.41 |
| `reusable` | 80 | 0.938 [0.917, 0.958] | 0.000 | 21.45 |
| `reusable` | 160 | 0.740 [0.671, 0.803] | 0.000 | 5.34 |

## Verdict

Communication regret from conditioning is **supported** at budget(s) [40, 80, 160] (present gain and future loss both excluded zero).

* 20-word budget: dU_now = 0.859 [0.797, 0.922], dU_future = -0.026 [-0.051, 0.001] - present gain only significant.
* 40-word budget: dU_now = 0.641 [0.547, 0.719], dU_future = -0.110 [-0.152, -0.070] - both directions significant.
* 80-word budget: dU_now = 0.516 [0.406, 0.609], dU_future = -0.198 [-0.257, -0.136] - both directions significant.
* 160-word budget: dU_now = 0.359 [0.250, 0.469], dU_future = -0.342 [-0.396, -0.292] - both directions significant.

The conditioning-induced specialisation gap increases as the budget falls from 160 to 20 words (0.184 [0.080, 0.285], p = 0.000).

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
    20,
    40,
    80,
    160
  ],
  "primary_metric": "judge_correct"
}
```
