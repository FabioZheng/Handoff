# Experiment 10 - communication regret under a hard communication budget

Corpus `squad_groups`: 24 contexts, 96 questions, 2688 rotation x policy x budget cells. Sender and answerer `meta-llama/llama-3.1-8b-instruct`; judge `openai/gpt-4o-mini`. Primary utility: LLM-judge accuracy.

`U_now` is the diagonal of each context's utility matrix (the message answered on the question it was conditioned on); `U_future` is the mean of the off-diagonal (the same message answered on questions the sender never saw). Every question rotates through both roles, so question difficulty cannot produce the contrast. All intervals are 95% percentile bootstrap over contexts, the unit the design repeats on.

## 1. Was the channel actually equal?

The hypothesis is only testable if no policy simply wrote more. Delivered length is capped deterministically, so the question is whether the arms *fill* the budget equally.

| policy | band (words) | delivered (mean +- sd) | fill | under floor | over cap pre-truncation | truncated | corrections | repetition |
|---|---|---|---|---|---|---|---|---|
| `conditioned` | 17-20 | 18.5 +- 1.0 | 0.92 | 0.00 | 0.00 | 0.00 | 0.38 | 0.000 |
| `extractive_conditioned` | 17-20 | 18.6 +- 1.3 | 0.93 | 0.04 | 0.05 | 0.05 | 0.85 | 0.000 |
| `extractive_generic` | 17-20 | 18.9 +- 1.1 | 0.95 | 0.00 | 0.12 | 0.12 | 1.00 | 0.000 |
| `generic` | 17-20 | 18.7 +- 1.0 | 0.94 | 0.00 | 0.00 | 0.00 | 0.50 | 0.000 |
| `oracle` | 17-20 | 18.1 +- 2.3 | 0.91 | 0.12 | 0.17 | 0.17 | 0.75 | 0.000 |
| `reusable` | 17-20 | 18.6 +- 1.1 | 0.93 | 0.00 | 0.01 | 0.01 | 0.39 | 0.000 |
| `conditioned` | 34-40 | 36.7 +- 2.0 | 0.92 | 0.01 | 0.00 | 0.00 | 0.66 | 0.000 |
| `conditioned@trim` | 34-40 | 30.6 +- 7.4 | 0.76 | 0.01 | 0.00 | 0.84 | 0.66 | 0.000 |
| `extractive_conditioned` | 34-40 | 36.8 +- 3.6 | 0.92 | 0.05 | 0.06 | 0.06 | 0.79 | 0.017 |
| `extractive_generic` | 34-40 | 37.4 +- 2.3 | 0.94 | 0.04 | 0.04 | 0.04 | 0.96 | 0.000 |
| `generic` | 34-40 | 37.1 +- 1.8 | 0.93 | 0.00 | 0.00 | 0.00 | 0.38 | 0.000 |
| `generic@trim` | 34-40 | 24.8 +- 7.9 | 0.62 | 0.00 | 0.00 | 0.88 | 0.38 | 0.000 |
| `oracle` | 34-40 | 37.9 +- 1.8 | 0.95 | 0.00 | 0.00 | 0.00 | 0.75 | 0.000 |
| `oracle@trim` | 34-40 | 24.3 +- 6.5 | 0.61 | 0.00 | 0.00 | 0.96 | 0.75 | 0.000 |
| `reusable` | 34-40 | 36.3 +- 3.8 | 0.91 | 0.02 | 0.02 | 0.02 | 0.54 | 0.000 |
| `reusable@trim` | 34-40 | 27.4 +- 9.1 | 0.69 | 0.02 | 0.02 | 0.84 | 0.54 | 0.000 |
| `conditioned` | 68-80 | 73.2 +- 4.4 | 0.92 | 0.01 | 0.00 | 0.00 | 0.71 | 0.002 |
| `extractive_conditioned` | 68-80 | 70.9 +- 4.7 | 0.89 | 0.16 | 0.06 | 0.06 | 0.92 | 0.027 |
| `extractive_generic` | 68-80 | 73.3 +- 4.1 | 0.92 | 0.04 | 0.04 | 0.04 | 0.54 | 0.000 |
| `generic` | 68-80 | 71.9 +- 4.4 | 0.90 | 0.04 | 0.04 | 0.04 | 0.25 | 0.000 |
| `oracle` | 68-80 | 73.8 +- 3.7 | 0.92 | 0.00 | 0.00 | 0.00 | 0.46 | 0.001 |
| `reusable` | 68-80 | 73.2 +- 3.4 | 0.92 | 0.00 | 0.00 | 0.00 | 0.58 | 0.001 |
| `conditioned` | 136-160 | 143.1 +- 5.5 | 0.89 | 0.00 | 0.00 | 0.00 | 0.86 | 0.004 |
| `extractive_conditioned` | 136-160 | 141.4 +- 8.9 | 0.88 | 0.14 | 0.00 | 0.00 | 0.93 | 0.029 |
| `extractive_generic` | 136-160 | 143.8 +- 6.4 | 0.90 | 0.00 | 0.00 | 0.00 | 0.79 | 0.003 |
| `generic` | 136-160 | 141.4 +- 4.3 | 0.88 | 0.00 | 0.00 | 0.00 | 0.58 | 0.003 |
| `oracle` | 136-160 | 143.4 +- 4.7 | 0.90 | 0.00 | 0.00 | 0.00 | 0.75 | 0.005 |
| `reusable` | 136-160 | 143.6 +- 5.5 | 0.90 | 0.00 | 0.00 | 0.00 | 0.78 | 0.005 |

Delivered messages over the cap after truncation: **0** arm(s) (must be zero; truncation is unconditional).

Paired length differences between arms (context-clustered):

| comparison | budget | delta words [95% CI] | p |
|---|---|---|---|
| conditioned_minus_generic | 20 | -0.240 [-0.740, 0.260] | 0.3530 |
| conditioned_minus_reusable | 20 | -0.094 [-0.354, 0.167] | 0.4974 |
| extractive_conditioned_minus_extractive_generic | 20 | -0.323 [-0.729, 0.115] | 0.1448 |
| oracle_minus_conditioned | 20 | -0.344 [-1.250, 0.458] | 0.4492 |
| oracle_minus_generic | 20 | -0.583 [-1.625, 0.333] | 0.2650 |
| reusable_minus_generic | 20 | -0.146 [-0.594, 0.312] | 0.5516 |
| conditioned_minus_generic | 40 | -0.469 [-1.229, 0.302] | 0.2450 |
| conditioned_minus_reusable | 40 | 0.323 [-0.375, 1.104] | 0.4026 |
| extractive_conditioned_minus_extractive_generic | 40 | -0.594 [-1.906, 0.635] | 0.3614 |
| oracle_minus_conditioned | 40 | 1.260 [0.510, 1.990] | 0.0007 |
| oracle_minus_generic | 40 | 0.792 [-0.250, 1.833] | 0.1543 |
| reusable_minus_generic | 40 | -0.792 [-1.740, 0.177] | 0.1089 |
| conditioned_minus_generic | 80 | 1.323 [-0.875, 3.427] | 0.2299 |
| conditioned_minus_reusable | 80 | 0.010 [-1.365, 1.250] | 0.9946 |
| extractive_conditioned_minus_extractive_generic | 80 | -2.406 [-4.479, -0.677] | 0.0154 |
| oracle_minus_conditioned | 80 | 0.594 [-0.656, 1.781] | 0.3372 |
| oracle_minus_generic | 80 | 1.917 [-0.417, 4.250] | 0.1056 |
| reusable_minus_generic | 80 | 1.312 [-0.375, 3.042] | 0.1362 |
| conditioned_minus_generic | 160 | 1.698 [-0.521, 3.844] | 0.1315 |
| conditioned_minus_reusable | 160 | -0.448 [-1.552, 0.688] | 0.4401 |
| extractive_conditioned_minus_extractive_generic | 160 | -2.323 [-4.802, 0.094] | 0.0627 |
| oracle_minus_conditioned | 160 | 0.260 [-1.875, 2.448] | 0.8174 |
| oracle_minus_generic | 160 | 1.958 [-0.083, 4.000] | 0.0650 |
| reusable_minus_generic | 160 | 2.146 [0.062, 4.094] | 0.0358 |

## 2. Present and future utility

| policy | budget | U_now [CI] | U_future [CI] | gap [CI] |
|---|---|---|---|---|
| `conditioned` | 20 | 0.865 [0.802, 0.917] | 0.177 [0.122, 0.236] | 0.688 [0.601, 0.771] |
| `conditioned` | 40 | 0.823 [0.750, 0.896] | 0.260 [0.188, 0.337] | 0.562 [0.451, 0.670] |
| `conditioned` | 80 | 0.812 [0.750, 0.875] | 0.337 [0.260, 0.413] | 0.476 [0.372, 0.580] |
| `conditioned` | 160 | 0.802 [0.708, 0.885] | 0.528 [0.438, 0.618] | 0.274 [0.153, 0.396] |
| `conditioned@trim` | 40 | 0.823 [0.750, 0.896] | 0.198 [0.132, 0.267] | 0.625 [0.535, 0.715] |
| `extractive_conditioned` | 20 | 0.740 [0.677, 0.802] | 0.156 [0.111, 0.212] | 0.583 [0.500, 0.667] |
| `extractive_conditioned` | 40 | 0.792 [0.708, 0.865] | 0.285 [0.219, 0.351] | 0.507 [0.389, 0.625] |
| `extractive_conditioned` | 80 | 0.906 [0.844, 0.958] | 0.458 [0.385, 0.531] | 0.448 [0.354, 0.545] |
| `extractive_conditioned` | 160 | 0.865 [0.802, 0.917] | 0.771 [0.701, 0.840] | 0.094 [0.028, 0.163] |
| `extractive_generic` | 20 | 0.354 [0.250, 0.458] | 0.354 [0.250, 0.458] | 0.000 [0.000, 0.000] |
| `extractive_generic` | 40 | 0.427 [0.344, 0.510] | 0.427 [0.344, 0.510] | 0.000 [0.000, 0.000] |
| `extractive_generic` | 80 | 0.615 [0.521, 0.708] | 0.615 [0.521, 0.708] | 0.000 [0.000, 0.000] |
| `extractive_generic` | 160 | 0.854 [0.771, 0.927] | 0.854 [0.771, 0.927] | 0.000 [0.000, 0.000] |
| `generic` | 20 | 0.240 [0.188, 0.292] | 0.240 [0.188, 0.292] | 0.000 [0.000, 0.000] |
| `generic` | 40 | 0.375 [0.302, 0.448] | 0.375 [0.302, 0.448] | 0.000 [0.000, 0.000] |
| `generic` | 80 | 0.531 [0.427, 0.635] | 0.531 [0.427, 0.635] | 0.000 [0.000, 0.000] |
| `generic` | 160 | 0.698 [0.604, 0.792] | 0.698 [0.604, 0.792] | 0.000 [0.000, 0.000] |
| `generic@trim` | 40 | 0.323 [0.250, 0.396] | 0.323 [0.250, 0.396] | 0.000 [0.000, 0.000] |
| `oracle` | 20 | 0.490 [0.417, 0.562] | 0.490 [0.417, 0.562] | 0.000 [0.000, 0.000] |
| `oracle` | 40 | 0.823 [0.740, 0.896] | 0.823 [0.740, 0.896] | 0.000 [0.000, 0.000] |
| `oracle` | 80 | 0.885 [0.823, 0.948] | 0.885 [0.823, 0.948] | 0.000 [0.000, 0.000] |
| `oracle` | 160 | 0.896 [0.833, 0.948] | 0.896 [0.833, 0.948] | 0.000 [0.000, 0.000] |
| `oracle@trim` | 40 | 0.594 [0.510, 0.667] | 0.594 [0.510, 0.667] | 0.000 [0.000, 0.000] |
| `reusable` | 20 | 0.792 [0.719, 0.865] | 0.174 [0.122, 0.229] | 0.618 [0.521, 0.712] |
| `reusable` | 40 | 0.802 [0.729, 0.875] | 0.271 [0.205, 0.337] | 0.531 [0.406, 0.656] |
| `reusable` | 80 | 0.854 [0.792, 0.917] | 0.424 [0.354, 0.497] | 0.431 [0.326, 0.535] |
| `reusable` | 160 | 0.906 [0.844, 0.958] | 0.601 [0.524, 0.677] | 0.306 [0.215, 0.396] |
| `reusable@trim` | 40 | 0.781 [0.698, 0.854] | 0.194 [0.132, 0.260] | 0.587 [0.469, 0.698] |

Direct-context ceiling `U(D, q)`: 0.875 [0.812, 0.938]. Closed-book baseline (leakage audit): 0.031 [0.010, 0.059].

## 3. The hypothesis

A supporting result needs `delta U_now > 0` **and** `delta U_future < 0` for `conditioned` against a generic or reusable baseline, strengthening as the budget shrinks.

| comparison | budget | delta U_now [CI] (p, dz) | delta U_future [CI] (p, dz) | gap DiD [CI] (p) |
|---|---|---|---|---|
| conditioned_minus_generic | 20 | 0.625 [0.552, 0.698] (0.000, 3.46) | -0.063 [-0.122, -0.003] (0.040, -0.40) | 0.688 [0.604, 0.774] (0.000) |
| conditioned_minus_generic | 40 | 0.448 [0.354, 0.542] (0.000, 1.92) | -0.115 [-0.201, -0.028] (0.010, -0.51) | 0.562 [0.448, 0.667] (0.000) |
| conditioned_minus_generic | 80 | 0.281 [0.167, 0.396] (0.000, 0.97) | -0.194 [-0.274, -0.118] (0.000, -0.97) | 0.476 [0.372, 0.580] (0.000) |
| conditioned_minus_generic | 160 | 0.104 [0.000, 0.208] (0.066, 0.38) | -0.170 [-0.292, -0.052] (0.005, -0.56) | 0.274 [0.156, 0.396] (0.000) |
| conditioned_minus_reusable | 20 | 0.073 [0.021, 0.135] (0.013, 0.53) | 0.003 [-0.031, 0.042] (0.809, 0.04) | 0.069 [0.003, 0.142] (0.061) |
| conditioned_minus_reusable | 40 | 0.021 [-0.073, 0.115] (0.748, 0.09) | -0.010 [-0.056, 0.035] (0.664, -0.09) | 0.031 [-0.087, 0.146] (0.591) |
| conditioned_minus_reusable | 80 | -0.042 [-0.094, 0.000] (0.117, -0.35) | -0.087 [-0.132, -0.038] (0.001, -0.73) | 0.045 [-0.049, 0.118] (0.320) |
| conditioned_minus_reusable | 160 | -0.104 [-0.177, -0.031] (0.008, -0.54) | -0.073 [-0.153, 0.010] (0.084, -0.35) | -0.031 [-0.153, 0.083] (0.605) |
| reusable_minus_generic | 20 | 0.552 [0.458, 0.646] (0.000, 2.37) | -0.066 [-0.128, -0.003] (0.037, -0.42) | 0.618 [0.521, 0.708] (0.000) |
| reusable_minus_generic | 40 | 0.427 [0.312, 0.542] (0.000, 1.43) | -0.104 [-0.177, -0.035] (0.005, -0.56) | 0.531 [0.406, 0.656] (0.000) |
| reusable_minus_generic | 80 | 0.323 [0.208, 0.438] (0.000, 1.11) | -0.108 [-0.184, -0.031] (0.005, -0.55) | 0.431 [0.323, 0.535] (0.000) |
| reusable_minus_generic | 160 | 0.208 [0.104, 0.312] (0.000, 0.79) | -0.097 [-0.187, -0.003] (0.037, -0.41) | 0.306 [0.215, 0.396] (0.000) |
| oracle_minus_conditioned | 20 | -0.375 [-0.438, -0.323] (0.000, -2.54) | 0.312 [0.229, 0.396] (0.000, 1.46) | -0.688 [-0.774, -0.604] (0.000) |
| oracle_minus_conditioned | 40 | 0.000 [-0.104, 0.094] (1.000, 0.00) | 0.562 [0.444, 0.681] (0.000, 1.84) | -0.562 [-0.667, -0.448] (0.000) |
| oracle_minus_conditioned | 80 | 0.073 [0.021, 0.125] (0.012, 0.53) | 0.549 [0.469, 0.632] (0.000, 2.65) | -0.476 [-0.580, -0.372] (0.000) |
| oracle_minus_conditioned | 160 | 0.094 [0.010, 0.177] (0.037, 0.43) | 0.368 [0.257, 0.483] (0.000, 1.28) | -0.274 [-0.396, -0.156] (0.000) |
| oracle_minus_generic | 20 | 0.250 [0.167, 0.344] (0.000, 1.07) | 0.250 [0.167, 0.344] (0.000, 1.07) | 0.000 [0.000, 0.000] (0.037) |
| oracle_minus_generic | 40 | 0.448 [0.323, 0.562] (0.000, 1.47) | 0.448 [0.323, 0.562] (0.000, 1.47) | -0.000 [-0.000, -0.000] (0.000) |
| oracle_minus_generic | 80 | 0.354 [0.260, 0.458] (0.000, 1.39) | 0.354 [0.260, 0.458] (0.000, 1.39) | -0.000 [-0.000, -0.000] (0.002) |
| oracle_minus_generic | 160 | 0.198 [0.115, 0.292] (0.000, 0.85) | 0.198 [0.115, 0.292] (0.000, 0.85) | -0.000 [-0.000, -0.000] (0.000) |
| extractive_conditioned_minus_extractive_generic | 20 | 0.385 [0.260, 0.510] (0.000, 1.17) | -0.198 [-0.292, -0.108] (0.000, -0.84) | 0.583 [0.500, 0.667] (0.000) |
| extractive_conditioned_minus_extractive_generic | 40 | 0.365 [0.240, 0.479] (0.000, 1.20) | -0.142 [-0.215, -0.066] (0.001, -0.75) | 0.507 [0.389, 0.625] (0.000) |
| extractive_conditioned_minus_extractive_generic | 80 | 0.292 [0.188, 0.396] (0.000, 1.11) | -0.156 [-0.253, -0.052] (0.003, -0.60) | 0.448 [0.354, 0.542] (0.000) |
| extractive_conditioned_minus_extractive_generic | 160 | 0.010 [-0.062, 0.083] (0.887, 0.06) | -0.083 [-0.146, -0.024] (0.011, -0.53) | 0.094 [0.024, 0.163] (0.009) |

### Does scarcity sharpen the trade-off?

| quantity | policy | gap(low) - gap(high) [CI] | p | dz |
|---|---|---|---|---|
| gap_low_minus_high | `conditioned` | 0.413 [0.323, 0.503] | 0.000 | 1.84 |
| gap_low_minus_high | `extractive_conditioned` | 0.490 [0.396, 0.587] | 0.000 | 1.99 |
| gap_low_minus_high | `extractive_generic` | 0.000 [0.000, 0.000] | 0.027 | 0.46 |
| gap_low_minus_high | `generic` | -0.000 [-0.000, 0.000] | 0.507 | -0.14 |
| gap_low_minus_high | `oracle` | 0.000 [0.000, 0.000] | 0.001 | 0.75 |
| gap_low_minus_high | `reusable` | 0.312 [0.191, 0.431] | 0.000 | 1.03 |
| conditioning_gap_low_minus_high | `conditioned_minus_generic` | 0.413 [0.326, 0.503] | 0.000 | 1.84 |

## 4. Communication regret against the source

`R(q) = U(D, q) - U(m, q)`, the accuracy the handoff costs relative to reading the source directly.

| policy | budget | R_now [CI] | R_future [CI] | retained U_future on answerable cells [CI] |
|---|---|---|---|---|
| `conditioned` | 20 | 0.010 [-0.052, 0.073] | 0.698 [0.594, 0.795] | 0.174 [0.116, 0.233] |
| `conditioned` | 40 | 0.052 [-0.031, 0.135] | 0.615 [0.507, 0.719] | 0.276 [0.193, 0.368] |
| `conditioned` | 80 | 0.062 [0.010, 0.115] | 0.538 [0.431, 0.642] | 0.340 [0.259, 0.429] |
| `conditioned` | 160 | 0.073 [0.010, 0.135] | 0.347 [0.229, 0.465] | 0.556 [0.457, 0.655] |
| `conditioned@trim` | 40 | 0.052 [-0.042, 0.135] | 0.677 [0.576, 0.774] | 0.201 [0.125, 0.288] |
| `extractive_conditioned` | 20 | 0.135 [0.052, 0.208] | 0.719 [0.625, 0.806] | 0.151 [0.104, 0.201] |
| `extractive_conditioned` | 40 | 0.083 [0.021, 0.146] | 0.590 [0.493, 0.681] | 0.280 [0.210, 0.352] |
| `extractive_conditioned` | 80 | -0.031 [-0.073, 0.010] | 0.417 [0.309, 0.521] | 0.474 [0.378, 0.568] |
| `extractive_conditioned` | 160 | 0.010 [-0.031, 0.052] | 0.104 [0.038, 0.174] | 0.849 [0.788, 0.906] |
| `extractive_generic` | 20 | 0.521 [0.375, 0.656] | 0.521 [0.375, 0.656] | 0.358 [0.250, 0.472] |
| `extractive_generic` | 40 | 0.448 [0.344, 0.552] | 0.448 [0.344, 0.552] | 0.417 [0.326, 0.510] |
| `extractive_generic` | 80 | 0.260 [0.135, 0.375] | 0.260 [0.135, 0.375] | 0.632 [0.545, 0.722] |
| `extractive_generic` | 160 | 0.021 [-0.052, 0.094] | 0.021 [-0.052, 0.094] | 0.931 [0.861, 0.979] |
| `generic` | 20 | 0.635 [0.552, 0.719] | 0.635 [0.552, 0.719] | 0.226 [0.149, 0.302] |
| `generic` | 40 | 0.500 [0.396, 0.604] | 0.500 [0.396, 0.604] | 0.403 [0.302, 0.504] |
| `generic` | 80 | 0.344 [0.229, 0.458] | 0.344 [0.229, 0.458] | 0.559 [0.451, 0.663] |
| `generic` | 160 | 0.177 [0.083, 0.271] | 0.177 [0.083, 0.271] | 0.767 [0.663, 0.861] |
| `generic@trim` | 40 | 0.552 [0.448, 0.646] | 0.552 [0.448, 0.646] | 0.340 [0.240, 0.451] |
| `oracle` | 20 | 0.385 [0.312, 0.458] | 0.385 [0.312, 0.458] | 0.486 [0.403, 0.566] |
| `oracle` | 40 | 0.052 [-0.031, 0.146] | 0.052 [-0.031, 0.146] | 0.861 [0.771, 0.941] |
| `oracle` | 80 | -0.010 [-0.083, 0.062] | -0.010 [-0.083, 0.062] | 0.931 [0.878, 0.976] |
| `oracle` | 160 | -0.021 [-0.094, 0.052] | -0.021 [-0.094, 0.052] | 0.944 [0.899, 0.986] |
| `oracle@trim` | 40 | 0.281 [0.177, 0.396] | 0.281 [0.177, 0.396] | 0.618 [0.517, 0.715] |
| `reusable` | 20 | 0.083 [0.010, 0.167] | 0.701 [0.608, 0.788] | 0.168 [0.113, 0.229] |
| `reusable` | 40 | 0.073 [0.010, 0.135] | 0.604 [0.500, 0.705] | 0.278 [0.201, 0.354] |
| `reusable` | 80 | 0.021 [-0.052, 0.083] | 0.451 [0.354, 0.549] | 0.455 [0.375, 0.538] |
| `reusable` | 160 | -0.031 [-0.104, 0.031] | 0.274 [0.170, 0.372] | 0.660 [0.578, 0.740] |
| `reusable@trim` | 40 | 0.094 [0.021, 0.167] | 0.681 [0.580, 0.774] | 0.194 [0.123, 0.273] |

`retained` is the ceiling-restricted normalisation: it averages `U(m, q)` only over cells the source itself answers, where `U(D, q) = 1` and the ratio has no near-zero denominator to guard.

The arithmetic ratio `mean U(m,q) / mean U(D,q)` is in `normalised_utility.csv`. Contexts whose own ceiling falls below 0.25 are excluded from it, because dividing by a near-zero ceiling turns noise into a number: 0 context-cell(s) were excluded on the future endpoint across all arms.

| policy | budget | retained U_now [CI] | retained U_future [CI] | n contexts | excluded (low ceiling) |
|---|---|---|---|---|---|
| `conditioned` | 20 | 1.014 [0.934, 1.101] | 0.227 [0.142, 0.321] | 24 | 0 |
| `conditioned` | 40 | 0.976 [0.858, 1.111] | 0.319 [0.226, 0.421] | 24 | 0 |
| `conditioned` | 80 | 0.938 [0.882, 0.997] | 0.410 [0.314, 0.516] | 24 | 0 |
| `conditioned` | 160 | 0.913 [0.837, 0.986] | 0.642 [0.515, 0.784] | 24 | 0 |
| `conditioned@trim` | 40 | 0.976 [0.861, 1.115] | 0.242 [0.160, 0.334] | 24 | 0 |
| `extractive_conditioned` | 20 | 0.868 [0.778, 0.969] | 0.194 [0.131, 0.270] | 24 | 0 |
| `extractive_conditioned` | 40 | 0.910 [0.826, 1.000] | 0.344 [0.256, 0.434] | 24 | 0 |
| `extractive_conditioned` | 80 | 1.056 [0.986, 1.132] | 0.557 [0.450, 0.676] | 24 | 0 |
| `extractive_conditioned` | 160 | 1.003 [0.948, 1.066] | 0.898 [0.814, 0.980] | 24 | 0 |
| `extractive_generic` | 20 | 0.448 [0.302, 0.608] | 0.448 [0.302, 0.608] | 24 | 0 |
| `extractive_generic` | 40 | 0.514 [0.406, 0.625] | 0.514 [0.406, 0.625] | 24 | 0 |
| `extractive_generic` | 80 | 0.743 [0.597, 0.910] | 0.743 [0.597, 0.910] | 24 | 0 |
| `extractive_generic` | 160 | 1.000 [0.892, 1.122] | 1.000 [0.892, 1.122] | 24 | 0 |
| `generic` | 20 | 0.288 [0.222, 0.354] | 0.288 [0.222, 0.354] | 24 | 0 |
| `generic` | 40 | 0.451 [0.354, 0.549] | 0.451 [0.354, 0.549] | 24 | 0 |
| `generic` | 80 | 0.622 [0.500, 0.747] | 0.622 [0.500, 0.747] | 24 | 0 |
| `generic` | 160 | 0.816 [0.694, 0.934] | 0.816 [0.694, 0.934] | 24 | 0 |
| `generic@trim` | 40 | 0.389 [0.292, 0.493] | 0.389 [0.292, 0.493] | 24 | 0 |
| `oracle` | 20 | 0.562 [0.483, 0.646] | 0.562 [0.483, 0.646] | 24 | 0 |
| `oracle` | 40 | 0.972 [0.861, 1.083] | 0.972 [0.861, 1.083] | 24 | 0 |
| `oracle` | 80 | 1.042 [0.944, 1.160] | 1.042 [0.944, 1.160] | 24 | 0 |
| `oracle` | 160 | 1.062 [0.958, 1.187] | 1.062 [0.958, 1.188] | 24 | 0 |
| `oracle@trim` | 40 | 0.715 [0.597, 0.837] | 0.715 [0.597, 0.837] | 24 | 0 |
| `reusable` | 20 | 0.924 [0.837, 1.014] | 0.214 [0.144, 0.292] | 24 | 0 |
| `reusable` | 40 | 0.924 [0.854, 0.997] | 0.336 [0.244, 0.433] | 24 | 0 |
| `reusable` | 80 | 1.003 [0.913, 1.122] | 0.505 [0.418, 0.594] | 24 | 0 |
| `reusable` | 160 | 1.069 [0.976, 1.187] | 0.718 [0.606, 0.854] | 24 | 0 |
| `reusable@trim` | 40 | 0.910 [0.819, 1.007] | 0.240 [0.159, 0.331] | 24 | 0 |

### Secondary metrics

EM and token F1 are computed for every answer and reported here in full, never replaced by the judge. F1 penalises answers that are correct but differently worded, which is exactly what compression produces, so a disagreement between the three is information rather than noise.

**Exact match** &mdash; `conditioned` minus `generic`:

| endpoint | 20 w | 40 w | 80 w | 160 w |
|---|---|---|---|---|
| dU_now | 0.417 [0.292, 0.542] | 0.323 [0.219, 0.427] | 0.271 [0.167, 0.375] | 0.188 [0.083, 0.302] |
| dU_future | -0.056 [-0.122, 0.014] | -0.132 [-0.208, -0.056] | -0.118 [-0.188, -0.045] | -0.122 [-0.233, -0.017] |

**Token F1** &mdash; `conditioned` minus `generic`:

| endpoint | 20 w | 40 w | 80 w | 160 w |
|---|---|---|---|---|
| dU_now | 0.541 [0.450, 0.633] | 0.413 [0.319, 0.504] | 0.291 [0.196, 0.388] | 0.187 [0.088, 0.288] |
| dU_future | -0.043 [-0.102, 0.020] | -0.118 [-0.189, -0.049] | -0.159 [-0.229, -0.088] | -0.140 [-0.248, -0.038] |

## 5. Pareto frontier

Cost is mean delivered words. A configuration is dominated when another is at least as good on both utilities and no more expensive, with one strict improvement.

| policy | requested cap | U_now | U_future | delivered cost (words) | 3D nondominated | within-cap 2D nondominated | dominated by |
|---|---|---|---|---|---|---|---|
| `conditioned` | 20 | 0.865 | 0.177 | 18.5 | yes | yes | - |
| `conditioned` | 40 | 0.823 | 0.260 | 36.7 | yes |  | - |
| `conditioned` | 80 | 0.812 | 0.337 | 73.2 |  |  | extractive_conditioned@80w;oracle@40w;reusable@80w |
| `conditioned` | 160 | 0.802 | 0.528 | 143.1 |  |  | extractive_conditioned@160w;oracle@40w;oracle@80w |
| `extractive_conditioned` | 20 | 0.740 | 0.156 | 18.6 |  |  | conditioned@20w;reusable@20w |
| `extractive_conditioned` | 40 | 0.792 | 0.285 | 36.8 | yes |  | - |
| `extractive_conditioned` | 80 | 0.906 | 0.458 | 70.9 | yes | yes | - |
| `extractive_conditioned` | 160 | 0.865 | 0.771 | 141.4 |  |  | oracle@80w |
| `extractive_generic` | 20 | 0.354 | 0.354 | 18.9 |  |  | oracle@20w |
| `extractive_generic` | 40 | 0.427 | 0.427 | 37.4 |  |  | oracle@20w |
| `extractive_generic` | 80 | 0.615 | 0.615 | 73.3 |  |  | oracle@40w |
| `extractive_generic` | 160 | 0.854 | 0.854 | 143.8 |  |  | oracle@160w;oracle@80w |
| `generic` | 20 | 0.240 | 0.240 | 18.7 |  |  | oracle@20w |
| `generic` | 40 | 0.375 | 0.375 | 37.1 |  |  | oracle@20w |
| `generic` | 80 | 0.531 | 0.531 | 71.9 |  |  | oracle@40w |
| `generic` | 160 | 0.698 | 0.698 | 141.4 |  |  | oracle@40w;oracle@80w |
| `oracle` | 20 | 0.490 | 0.490 | 18.1 | yes | yes | - |
| `oracle` | 40 | 0.823 | 0.823 | 37.9 | yes | yes | - |
| `oracle` | 80 | 0.885 | 0.885 | 73.8 | yes | yes | - |
| `oracle` | 160 | 0.896 | 0.896 | 143.4 | yes | yes | - |
| `reusable` | 20 | 0.792 | 0.174 | 18.6 |  |  | conditioned@20w |
| `reusable` | 40 | 0.802 | 0.271 | 36.3 | yes |  | - |
| `reusable` | 80 | 0.854 | 0.424 | 73.2 |  |  | extractive_conditioned@80w |
| `reusable` | 160 | 0.906 | 0.601 | 143.6 | yes | yes | - |

**Nondominated set (10 of 24 configurations):** `reusable`@160w, `extractive_conditioned`@80w, `oracle`@160w, `oracle`@80w, `conditioned`@20w, `oracle`@40w, `conditioned`@40w, `reusable`@40w, `extractive_conditioned`@40w, `oracle`@20w.

Cost enters that test, so the set above is not a curve: a configuration can survive purely by being cheaper than everything that beats it on utility. The figure facets the observations by requested cap and outlines the two-dimensional nondominated policies within each facet. Markers are deliberately not joined: policy is categorical, so a line or staircase would imply unevaluated intermediate choices. The discrete within-cap sets are listed here from best present-query utility to best future-query utility:

* **20 words** -- `conditioned` (0.86, 0.18)  `oracle` (0.49, 0.49)
* **40 words** -- `oracle` (0.82, 0.82)
* **80 words** -- `extractive_conditioned` (0.91, 0.46)  `oracle` (0.89, 0.89)
* **160 words** -- `reusable` (0.91, 0.60)  `oracle` (0.90, 0.90)

## 6. Length-matched subsample

Contexts where `generic`, `conditioned` and `reusable` delivered within 6 words of each other. This is the comparison that survives if section 1 shows the arms filled the budget differently.

| comparison | budget | endpoint | delta [CI] | p | n contexts |
|---|---|---|---|---|---|
| conditioned_minus_generic | 20 | future | -0.063 [-0.122, -0.000] | 0.042 | 24 |
| conditioned_minus_generic | 20 | now | 0.625 [0.552, 0.698] | 0.000 | 24 |
| conditioned_minus_reusable | 20 | future | 0.003 [-0.031, 0.042] | 0.803 | 24 |
| conditioned_minus_reusable | 20 | now | 0.073 [0.021, 0.125] | 0.011 | 24 |
| reusable_minus_generic | 20 | future | -0.066 [-0.128, -0.003] | 0.039 | 24 |
| reusable_minus_generic | 20 | now | 0.552 [0.468, 0.646] | 0.000 | 24 |
| conditioned_minus_generic | 40 | future | -0.105 [-0.192, -0.014] | 0.020 | 23 |
| conditioned_minus_generic | 40 | now | 0.446 [0.348, 0.543] | 0.000 | 23 |
| conditioned_minus_reusable | 40 | future | -0.007 [-0.054, 0.040] | 0.750 | 23 |
| conditioned_minus_reusable | 40 | now | 0.022 [-0.076, 0.120] | 0.751 | 23 |
| reusable_minus_generic | 40 | future | -0.098 [-0.174, -0.025] | 0.008 | 23 |
| reusable_minus_generic | 40 | now | 0.424 [0.304, 0.543] | 0.000 | 23 |
| conditioned_minus_generic | 80 | future | -0.183 [-0.289, -0.078] | 0.001 | 15 |
| conditioned_minus_generic | 80 | now | 0.200 [0.067, 0.333] | 0.004 | 15 |
| conditioned_minus_reusable | 80 | future | -0.072 [-0.128, -0.006] | 0.030 | 15 |
| conditioned_minus_reusable | 80 | now | -0.067 [-0.150, 0.000] | 0.105 | 15 |
| reusable_minus_generic | 80 | future | -0.111 [-0.206, -0.017] | 0.019 | 15 |
| reusable_minus_generic | 80 | now | 0.267 [0.117, 0.400] | 0.000 | 15 |
| conditioned_minus_generic | 160 | future | -0.244 [-0.423, -0.071] | 0.007 | 13 |
| conditioned_minus_generic | 160 | now | 0.173 [0.038, 0.308] | 0.015 | 13 |
| conditioned_minus_reusable | 160 | future | -0.128 [-0.250, -0.006] | 0.041 | 13 |
| conditioned_minus_reusable | 160 | now | -0.019 [-0.077, 0.038] | 0.756 | 13 |
| reusable_minus_generic | 160 | future | -0.115 [-0.231, 0.006] | 0.066 | 13 |
| reusable_minus_generic | 160 | now | 0.192 [0.096, 0.308] | 0.001 | 13 |

## Verdict

Communication regret from conditioning is **supported** at budget(s) [20, 40, 80] (present gain and future loss both excluded zero).

* 20-word budget: dU_now = 0.625 [0.552, 0.698], dU_future = -0.063 [-0.122, -0.003] - both directions significant.
* 40-word budget: dU_now = 0.448 [0.354, 0.542], dU_future = -0.115 [-0.201, -0.028] - both directions significant.
* 80-word budget: dU_now = 0.281 [0.167, 0.396], dU_future = -0.194 [-0.274, -0.118] - both directions significant.
* 160-word budget: dU_now = 0.104 [0.000, 0.208], dU_future = -0.170 [-0.292, -0.052] - future loss only significant.

The conditioning-induced specialisation gap increases as the budget falls from 160 to 20 words (0.413 [0.326, 0.503], p = 0.000).

## Provenance

```json
{
  "schema_version": "communication-regret-v1",
  "dataset": "squad_groups",
  "source_manifest": {
    "dataset": "squad_groups",
    "parquet": "data/raw/squad_validation.parquet",
    "probe_model": "meta-llama/llama-3.1-8b-instruct",
    "judge_model": "openai/gpt-4o-mini",
    "audit_model": "openai/gpt-4o-mini",
    "questions_per_context": 4,
    "structural_candidates": 200,
    "closed_book_method": "llm_judge",
    "closed_book_leaked_questions": 318,
    "groups_after_leakage": 58,
    "groups_rejected_by_audit": 1,
    "kept": 24,
    "context_chars_band": [
      800,
      1600
    ],
    "max_question_jaccard": 0.1
  },
  "policies": [
    "conditioned",
    "conditioned@trim",
    "extractive_conditioned",
    "extractive_generic",
    "generic",
    "generic@trim",
    "oracle",
    "oracle@trim",
    "reusable",
    "reusable@trim"
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
