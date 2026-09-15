# Experiment 10 - communication regret under a hard communication budget

Corpus `relation_dossiers`: 16 contexts, 256 questions, 2816 rotation x policy x budget cells. Sender and answerer `meta-llama/llama-3.1-8b-instruct`; judge `openai/gpt-4o-mini`. Primary utility: LLM-judge accuracy.

`U_now` is the diagonal of each context's utility matrix (the message answered on the question it was conditioned on); `U_future` is the mean of the off-diagonal (the same message answered on questions the sender never saw). Every question rotates through both roles, so question difficulty cannot produce the contrast. All intervals are 95% percentile bootstrap over contexts, the unit the design repeats on.

## 1. Was the channel actually equal?

The hypothesis is only testable if no policy simply wrote more. Delivered length is capped deterministically, so the question is whether the arms *fill* the budget equally.

| policy | band (words) | delivered (mean +- sd) | fill | under floor | over cap pre-truncation | truncated | corrections | repetition |
|---|---|---|---|---|---|---|---|---|
| `conditioned` | 17-20 | 18.3 +- 1.0 | 0.91 | 0.00 | 0.00 | 0.00 | 0.45 | 0.000 |
| `generic` | 17-20 | 18.8 +- 1.0 | 0.94 | 0.00 | 0.00 | 0.00 | 0.44 | 0.000 |
| `lm_conditioned` | 17-20 | 12.6 +- 8.7 | 0.63 | 0.42 | 0.00 | 0.00 | 0.00 | 0.000 |
| `lm_generic` | 17-20 | 11.7 +- 8.5 | 0.58 | 0.56 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_conditioned` | 17-20 | 12.5 +- 8.7 | 0.62 | 0.44 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_generic` | 17-20 | 12.9 +- 9.0 | 0.64 | 0.38 | 0.00 | 0.00 | 0.00 | 0.000 |
| `oracle` | 17-20 | 18.1 +- 1.1 | 0.90 | 0.00 | 0.00 | 0.00 | 0.81 | 0.000 |
| `paraphrase` | 17-20 | 18.6 +- 0.9 | 0.93 | 0.00 | 0.00 | 0.00 | 0.50 | 0.000 |
| `passthrough` | 17-20 | 12.6 +- 9.1 | 0.63 | 0.44 | 0.00 | 0.00 | 0.00 | 0.000 |
| `random_selection` | 17-20 | 12.4 +- 8.8 | 0.62 | 0.50 | 0.00 | 0.00 | 0.00 | 0.000 |
| `reusable` | 17-20 | 18.3 +- 1.0 | 0.92 | 0.00 | 0.02 | 0.02 | 0.52 | 0.000 |
| `conditioned` | 34-40 | 36.5 +- 2.0 | 0.91 | 0.00 | 0.03 | 0.03 | 0.67 | 0.000 |
| `generic` | 34-40 | 37.1 +- 1.9 | 0.93 | 0.00 | 0.00 | 0.00 | 0.75 | 0.000 |
| `lm_conditioned` | 34-40 | 31.1 +- 5.7 | 0.78 | 0.64 | 0.00 | 0.00 | 0.00 | 0.000 |
| `lm_generic` | 34-40 | 31.1 +- 5.2 | 0.78 | 0.69 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_conditioned` | 34-40 | 31.3 +- 5.4 | 0.78 | 0.64 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_generic` | 34-40 | 33.3 +- 5.2 | 0.83 | 0.56 | 0.00 | 0.00 | 0.00 | 0.000 |
| `oracle` | 34-40 | 36.8 +- 1.9 | 0.92 | 0.00 | 0.00 | 0.00 | 1.00 | 0.000 |
| `paraphrase` | 34-40 | 37.7 +- 1.7 | 0.94 | 0.00 | 0.00 | 0.00 | 0.31 | 0.000 |
| `passthrough` | 34-40 | 33.4 +- 6.1 | 0.84 | 0.44 | 0.00 | 0.00 | 0.00 | 0.000 |
| `random_selection` | 34-40 | 30.8 +- 5.5 | 0.77 | 0.69 | 0.00 | 0.00 | 0.00 | 0.000 |
| `reusable` | 34-40 | 36.8 +- 1.9 | 0.92 | 0.00 | 0.00 | 0.00 | 0.38 | 0.000 |
| `conditioned` | 68-80 | 72.9 +- 3.3 | 0.91 | 0.00 | 0.00 | 0.00 | 0.67 | 0.000 |
| `generic` | 68-80 | 73.2 +- 3.5 | 0.92 | 0.00 | 0.00 | 0.00 | 0.38 | 0.000 |
| `lm_conditioned` | 68-80 | 72.7 +- 6.7 | 0.91 | 0.25 | 0.00 | 0.00 | 0.00 | 0.000 |
| `lm_generic` | 68-80 | 74.8 +- 4.7 | 0.94 | 0.12 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_conditioned` | 68-80 | 72.7 +- 6.0 | 0.91 | 0.25 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_generic` | 68-80 | 74.1 +- 5.8 | 0.93 | 0.12 | 0.00 | 0.00 | 0.00 | 0.000 |
| `oracle` | 68-80 | 75.6 +- 2.5 | 0.95 | 0.00 | 0.00 | 0.00 | 0.81 | 0.000 |
| `paraphrase` | 68-80 | 74.7 +- 3.3 | 0.93 | 0.00 | 0.00 | 0.00 | 0.31 | 0.000 |
| `passthrough` | 68-80 | 72.8 +- 6.9 | 0.91 | 0.31 | 0.00 | 0.00 | 0.00 | 0.000 |
| `random_selection` | 68-80 | 72.9 +- 5.5 | 0.91 | 0.25 | 0.00 | 0.00 | 0.00 | 0.000 |
| `reusable` | 68-80 | 72.9 +- 2.9 | 0.91 | 0.00 | 0.00 | 0.00 | 0.55 | 0.000 |
| `conditioned` | 136-160 | 143.9 +- 5.6 | 0.90 | 0.00 | 0.00 | 0.00 | 0.88 | 0.002 |
| `generic` | 136-160 | 143.6 +- 4.3 | 0.90 | 0.00 | 0.00 | 0.00 | 0.62 | 0.000 |
| `lm_conditioned` | 136-160 | 152.2 +- 6.3 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| `lm_generic` | 136-160 | 152.1 +- 6.5 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_conditioned` | 136-160 | 152.7 +- 5.6 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| `nonllm_generic` | 136-160 | 153.6 +- 6.0 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| `oracle` | 136-160 | 143.7 +- 6.8 | 0.90 | 0.00 | 0.00 | 0.00 | 0.38 | 0.001 |
| `paraphrase` | 136-160 | 145.3 +- 5.2 | 0.91 | 0.00 | 0.00 | 0.00 | 0.62 | 0.000 |
| `passthrough` | 136-160 | 151.9 +- 5.8 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| `random_selection` | 136-160 | 152.4 +- 6.0 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| `reusable` | 136-160 | 143.3 +- 4.8 | 0.90 | 0.00 | 0.00 | 0.00 | 0.78 | 0.001 |

Delivered messages over the cap after truncation: **0** arm(s) (must be zero; truncation is unconditional).

Paired length differences between arms (context-clustered):

| comparison | budget | delta words [95% CI] | p |
|---|---|---|---|
| conditioned_minus_generic | 20 | -0.469 [-1.000, 0.078] | 0.0907 |
| conditioned_minus_lm_conditioned | 20 | 5.703 [1.750, 10.156] | 0.0059 |
| conditioned_minus_nonllm_conditioned | 20 | 5.781 [1.812, 10.234] | 0.0048 |
| conditioned_minus_paraphrase | 20 | -0.281 [-0.734, 0.172] | 0.2406 |
| generic_minus_lm_generic | 20 | 7.062 [3.125, 11.375] | 0.0003 |
| generic_minus_nonllm_generic | 20 | 5.875 [1.812, 10.562] | 0.0067 |
| generic_minus_passthrough | 20 | 6.188 [2.000, 10.812] | 0.0035 |
| lm_conditioned_minus_lm_generic | 20 | 0.891 [0.109, 1.859] | 0.0472 |
| lm_conditioned_minus_nonllm_conditioned | 20 | 0.078 [-0.266, 0.516] | 0.7263 |
| lm_conditioned_minus_passthrough | 20 | 0.016 [-0.641, 0.938] | 0.9820 |
| lm_generic_minus_nonllm_generic | 20 | -1.188 [-2.438, -0.188] | 0.0409 |
| lm_generic_minus_random_selection | 20 | -0.688 [-1.750, 0.000] | 0.1287 |
| nonllm_conditioned_minus_nonllm_generic | 20 | -0.375 [-0.938, 0.062] | 0.1658 |
| nonllm_conditioned_minus_passthrough | 20 | -0.062 [-0.938, 0.969] | 0.9148 |
| nonllm_generic_minus_passthrough | 20 | 0.312 [-0.438, 1.438] | 0.6210 |
| nonllm_generic_minus_random_selection | 20 | 0.500 [-0.125, 1.375] | 0.2429 |
| oracle_minus_conditioned | 20 | -0.219 [-0.766, 0.328] | 0.4560 |
| paraphrase_minus_generic | 20 | -0.188 [-0.750, 0.500] | 0.6296 |
| paraphrase_minus_passthrough | 20 | 6.000 [1.688, 10.689] | 0.0067 |
| reusable_minus_generic | 20 | -0.406 [-0.938, 0.141] | 0.1490 |
| conditioned_minus_generic | 40 | -0.656 [-1.469, 0.141] | 0.1157 |
| conditioned_minus_lm_conditioned | 40 | 5.328 [3.625, 6.891] | 0.0000 |
| conditioned_minus_nonllm_conditioned | 40 | 5.172 [3.421, 6.766] | 0.0000 |
| conditioned_minus_paraphrase | 40 | -1.219 [-1.953, -0.469] | 0.0020 |
| generic_minus_lm_generic | 40 | 6.062 [3.562, 8.312] | 0.0000 |
| generic_minus_nonllm_generic | 40 | 3.812 [1.250, 6.500] | 0.0048 |
| generic_minus_passthrough | 40 | 3.688 [0.812, 6.625] | 0.0120 |
| lm_conditioned_minus_lm_generic | 40 | 0.078 [-2.828, 2.891] | 0.9610 |
| lm_conditioned_minus_nonllm_conditioned | 40 | -0.156 [-0.891, 0.609] | 0.6913 |
| lm_conditioned_minus_passthrough | 40 | -2.297 [-4.875, 0.391] | 0.0871 |
| lm_generic_minus_nonllm_generic | 40 | -2.250 [-5.625, 1.500] | 0.2223 |
| lm_generic_minus_random_selection | 40 | 0.312 [-2.125, 2.938] | 0.8236 |
| nonllm_conditioned_minus_nonllm_generic | 40 | -2.016 [-4.719, 0.500] | 0.1321 |
| nonllm_conditioned_minus_passthrough | 40 | -2.141 [-4.891, 0.797] | 0.1450 |
| nonllm_generic_minus_passthrough | 40 | -0.125 [-3.938, 3.688] | 0.9625 |
| nonllm_generic_minus_random_selection | 40 | 2.562 [-1.062, 6.125] | 0.1678 |
| oracle_minus_conditioned | 40 | 0.281 [-0.735, 1.344] | 0.6105 |
| paraphrase_minus_generic | 40 | 0.562 [-0.688, 1.688] | 0.3845 |
| paraphrase_minus_passthrough | 40 | 4.250 [1.375, 7.312] | 0.0051 |
| reusable_minus_generic | 40 | -0.328 [-1.219, 0.547] | 0.4893 |
| conditioned_minus_generic | 80 | -0.375 [-2.156, 1.375] | 0.6873 |
| conditioned_minus_lm_conditioned | 80 | 0.219 [-1.891, 2.469] | 0.8488 |
| conditioned_minus_nonllm_conditioned | 80 | 0.172 [-1.875, 2.297] | 0.8800 |
| conditioned_minus_paraphrase | 80 | -1.812 [-3.641, 0.125] | 0.0623 |
| generic_minus_lm_generic | 80 | -1.562 [-4.562, 1.625] | 0.3411 |
| generic_minus_nonllm_generic | 80 | -0.812 [-4.125, 2.625] | 0.6591 |
| generic_minus_passthrough | 80 | 0.500 [-3.000, 4.000] | 0.7993 |
| lm_conditioned_minus_lm_generic | 80 | -2.156 [-5.469, 1.297] | 0.2213 |
| lm_conditioned_minus_nonllm_conditioned | 80 | -0.047 [-2.250, 2.000] | 0.9717 |
| lm_conditioned_minus_passthrough | 80 | -0.094 [-3.953, 3.922] | 0.9658 |
| lm_generic_minus_nonllm_generic | 80 | 0.750 [-3.562, 4.938] | 0.7453 |
| lm_generic_minus_random_selection | 80 | 1.875 [-1.000, 4.688] | 0.2004 |
| nonllm_conditioned_minus_nonllm_generic | 80 | -1.359 [-4.781, 2.047] | 0.4419 |
| nonllm_conditioned_minus_passthrough | 80 | -0.047 [-2.500, 2.610] | 0.9765 |
| nonllm_generic_minus_passthrough | 80 | 1.312 [-2.750, 5.812] | 0.5689 |
| nonllm_generic_minus_random_selection | 80 | 1.125 [-2.750, 5.188] | 0.5822 |
| oracle_minus_conditioned | 80 | 2.750 [1.281, 4.297] | 0.0008 |
| paraphrase_minus_generic | 80 | 1.438 [-0.500, 3.312] | 0.1508 |
| paraphrase_minus_passthrough | 80 | 1.938 [-1.938, 6.000] | 0.3525 |
| reusable_minus_generic | 80 | -0.312 [-1.953, 1.344] | 0.7305 |
| conditioned_minus_generic | 160 | 0.328 [-2.344, 3.047] | 0.8123 |
| conditioned_minus_lm_conditioned | 160 | -8.344 [-10.469, -5.891] | 0.0000 |
| conditioned_minus_nonllm_conditioned | 160 | -8.781 [-10.438, -7.141] | 0.0000 |
| conditioned_minus_paraphrase | 160 | -1.422 [-4.219, 1.391] | 0.3298 |
| generic_minus_lm_generic | 160 | -8.500 [-12.938, -3.625] | 0.0007 |
| generic_minus_nonllm_generic | 160 | -10.062 [-13.000, -7.000] | 0.0000 |
| generic_minus_passthrough | 160 | -8.312 [-11.875, -4.500] | 0.0000 |
| lm_conditioned_minus_lm_generic | 160 | 0.172 [-2.234, 2.938] | 0.9062 |
| lm_conditioned_minus_nonllm_conditioned | 160 | -0.438 [-2.328, 1.281] | 0.6419 |
| lm_conditioned_minus_passthrough | 160 | 0.359 [-2.969, 3.594] | 0.8335 |
| lm_generic_minus_nonllm_generic | 160 | -1.562 [-6.314, 2.812] | 0.5158 |
| lm_generic_minus_random_selection | 160 | -0.312 [-4.875, 3.812] | 0.8958 |
| nonllm_conditioned_minus_nonllm_generic | 160 | -0.953 [-3.609, 1.703] | 0.4881 |
| nonllm_conditioned_minus_passthrough | 160 | 0.797 [-2.266, 3.969] | 0.6202 |
| nonllm_generic_minus_passthrough | 160 | 1.750 [-2.000, 5.627] | 0.3797 |
| nonllm_generic_minus_random_selection | 160 | 1.250 [-1.688, 4.125] | 0.4135 |
| oracle_minus_conditioned | 160 | -0.203 [-3.813, 3.688] | 0.9191 |
| paraphrase_minus_generic | 160 | 1.750 [-1.938, 5.375] | 0.3568 |
| paraphrase_minus_passthrough | 160 | -6.562 [-10.375, -2.875] | 0.0006 |
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
| `lm_conditioned` | 20 | 0.062 [0.000, 0.141] | 0.023 [0.007, 0.041] | 0.040 [-0.011, 0.109] |
| `lm_conditioned` | 40 | 0.609 [0.500, 0.719] | 0.065 [0.059, 0.071] | 0.545 [0.433, 0.655] |
| `lm_conditioned` | 80 | 0.703 [0.594, 0.812] | 0.164 [0.143, 0.182] | 0.540 [0.425, 0.650] |
| `lm_conditioned` | 160 | 0.891 [0.812, 0.953] | 0.331 [0.302, 0.360] | 0.559 [0.483, 0.633] |
| `lm_generic` | 20 | 0.031 [0.000, 0.078] | 0.023 [0.004, 0.045] | 0.008 [-0.017, 0.033] |
| `lm_generic` | 40 | 0.078 [0.031, 0.141] | 0.070 [0.050, 0.089] | 0.008 [-0.033, 0.054] |
| `lm_generic` | 80 | 0.188 [0.094, 0.297] | 0.175 [0.140, 0.214] | 0.013 [-0.054, 0.088] |
| `lm_generic` | 160 | 0.375 [0.281, 0.469] | 0.350 [0.307, 0.393] | 0.025 [-0.033, 0.083] |
| `nonllm_conditioned` | 20 | 0.047 [0.000, 0.125] | 0.025 [0.007, 0.045] | 0.022 [-0.019, 0.081] |
| `nonllm_conditioned` | 40 | 0.781 [0.688, 0.859] | 0.068 [0.062, 0.073] | 0.714 [0.624, 0.794] |
| `nonllm_conditioned` | 80 | 0.891 [0.812, 0.953] | 0.146 [0.132, 0.158] | 0.745 [0.668, 0.811] |
| `nonllm_conditioned` | 160 | 0.953 [0.906, 1.000] | 0.326 [0.301, 0.351] | 0.627 [0.577, 0.671] |
| `nonllm_generic` | 20 | 0.016 [0.000, 0.047] | 0.016 [0.000, 0.034] | 0.000 [-0.017, 0.021] |
| `nonllm_generic` | 40 | 0.047 [0.000, 0.094] | 0.059 [0.040, 0.078] | -0.012 [-0.046, 0.025] |
| `nonllm_generic` | 80 | 0.141 [0.078, 0.203] | 0.136 [0.117, 0.155] | 0.004 [-0.054, 0.063] |
| `nonllm_generic` | 160 | 0.344 [0.250, 0.438] | 0.315 [0.267, 0.357] | 0.029 [-0.029, 0.088] |
| `oracle` | 20 | 0.250 [0.250, 0.250] | 0.179 [0.163, 0.196] | 0.071 [0.054, 0.087] |
| `oracle` | 40 | 0.516 [0.438, 0.594] | 0.399 [0.363, 0.434] | 0.117 [0.054, 0.175] |
| `oracle` | 80 | 0.750 [0.672, 0.828] | 0.692 [0.625, 0.751] | 0.058 [0.008, 0.104] |
| `oracle` | 160 | 0.891 [0.812, 0.953] | 0.828 [0.755, 0.898] | 0.062 [0.025, 0.100] |
| `paraphrase` | 20 | 0.141 [0.062, 0.219] | 0.074 [0.041, 0.110] | 0.067 [0.029, 0.108] |
| `paraphrase` | 40 | 0.188 [0.109, 0.266] | 0.138 [0.092, 0.188] | 0.050 [0.008, 0.092] |
| `paraphrase` | 80 | 0.297 [0.219, 0.375] | 0.247 [0.206, 0.290] | 0.050 [-0.008, 0.113] |
| `paraphrase` | 160 | 0.406 [0.297, 0.516] | 0.385 [0.323, 0.448] | 0.021 [-0.058, 0.092] |
| `passthrough` | 20 | 0.031 [0.000, 0.078] | 0.031 [0.011, 0.053] | 0.000 [-0.025, 0.029] |
| `passthrough` | 40 | 0.125 [0.062, 0.188] | 0.071 [0.042, 0.100] | 0.054 [0.017, 0.092] |
| `passthrough` | 80 | 0.266 [0.250, 0.297] | 0.178 [0.154, 0.202] | 0.087 [0.067, 0.112] |
| `passthrough` | 160 | 0.453 [0.406, 0.500] | 0.370 [0.348, 0.390] | 0.083 [0.042, 0.121] |
| `random_selection` | 20 | 0.016 [0.000, 0.047] | 0.016 [0.000, 0.034] | 0.000 [-0.017, 0.021] |
| `random_selection` | 40 | 0.047 [0.000, 0.094] | 0.064 [0.041, 0.086] | -0.017 [-0.046, 0.021] |
| `random_selection` | 80 | 0.188 [0.094, 0.281] | 0.150 [0.111, 0.189] | 0.038 [-0.029, 0.104] |
| `random_selection` | 160 | 0.469 [0.359, 0.578] | 0.381 [0.330, 0.427] | 0.088 [0.013, 0.163] |
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
| reusable_minus_generic | 20 | 0.859 [0.797, 0.922] (0.000, 6.71) | -0.022 [-0.045, 0.002] (0.077, -0.44) | 0.881 [0.831, 0.934] (0.000) |
| reusable_minus_generic | 40 | 0.656 [0.562, 0.734] (0.000, 3.65) | -0.103 [-0.141, -0.066] (0.000, -1.29) | 0.759 [0.700, 0.813] (0.000) |
| reusable_minus_generic | 80 | 0.531 [0.453, 0.609] (0.000, 2.96) | -0.118 [-0.168, -0.065] (0.000, -1.08) | 0.649 [0.594, 0.697] (0.000) |
| reusable_minus_generic | 160 | 0.344 [0.234, 0.453] (0.000, 1.44) | -0.175 [-0.241, -0.115] (0.000, -1.31) | 0.519 [0.430, 0.608] (0.000) |
| oracle_minus_conditioned | 20 | -0.750 [-0.750, -0.750] (0.000, n/a) | 0.102 [0.084, 0.121] (0.000, 2.66) | -0.852 [-0.871, -0.834] (0.000) |
| oracle_minus_conditioned | 40 | -0.469 [-0.547, -0.391] (0.000, -2.61) | 0.303 [0.263, 0.344] (0.000, 3.51) | -0.772 [-0.841, -0.702] (0.000) |
| oracle_minus_conditioned | 80 | -0.234 [-0.328, -0.141] (0.000, -1.21) | 0.575 [0.506, 0.635] (0.000, 4.25) | -0.809 [-0.860, -0.752] (0.000) |
| oracle_minus_conditioned | 160 | -0.078 [-0.172, 0.000] (0.100, -0.44) | 0.660 [0.583, 0.731] (0.000, 4.28) | -0.739 [-0.793, -0.676] (0.000) |

### Does scarcity sharpen the trade-off?

| quantity | policy | gap(low) - gap(high) [CI] | p | dz |
|---|---|---|---|---|
| gap_low_minus_high | `conditioned` | 0.122 [0.080, 0.170] | 0.000 | 1.27 |
| gap_low_minus_high | `generic` | -0.062 [-0.138, 0.017] | 0.134 | -0.37 |
| gap_low_minus_high | `lm_conditioned` | -0.520 [-0.624, -0.405] | 0.000 | -2.25 |
| gap_low_minus_high | `lm_generic` | -0.017 [-0.075, 0.042] | 0.540 | -0.14 |
| gap_low_minus_high | `nonllm_conditioned` | -0.605 [-0.663, -0.540] | 0.000 | -4.64 |
| gap_low_minus_high | `nonllm_generic` | -0.029 [-0.092, 0.033] | 0.328 | -0.22 |
| gap_low_minus_high | `oracle` | 0.008 [-0.025, 0.042] | 0.716 | 0.11 |
| gap_low_minus_high | `paraphrase` | 0.046 [-0.017, 0.113] | 0.182 | 0.33 |
| gap_low_minus_high | `passthrough` | -0.083 [-0.121, -0.042] | 0.000 | -0.97 |
| gap_low_minus_high | `random_selection` | -0.088 [-0.167, -0.004] | 0.036 | -0.51 |
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
| `lm_conditioned` | 20 | 0.922 [0.844, 0.984] | 0.970 [0.944, 0.991] | 0.023 [0.008, 0.042] |
| `lm_conditioned` | 40 | 0.375 [0.266, 0.484] | 0.928 [0.914, 0.939] | 0.065 [0.060, 0.071] |
| `lm_conditioned` | 80 | 0.281 [0.172, 0.391] | 0.829 [0.801, 0.855] | 0.163 [0.142, 0.182] |
| `lm_conditioned` | 160 | 0.094 [0.031, 0.172] | 0.661 [0.629, 0.694] | 0.333 [0.304, 0.362] |
| `lm_generic` | 20 | 0.953 [0.906, 1.000] | 0.970 [0.946, 0.992] | 0.023 [0.004, 0.045] |
| `lm_generic` | 40 | 0.906 [0.844, 0.969] | 0.923 [0.905, 0.942] | 0.070 [0.050, 0.089] |
| `lm_generic` | 80 | 0.797 [0.688, 0.891] | 0.818 [0.782, 0.851] | 0.176 [0.140, 0.214] |
| `lm_generic` | 160 | 0.609 [0.516, 0.703] | 0.643 [0.598, 0.689] | 0.348 [0.305, 0.391] |
| `nonllm_conditioned` | 20 | 0.938 [0.859, 1.000] | 0.968 [0.940, 0.992] | 0.026 [0.008, 0.046] |
| `nonllm_conditioned` | 40 | 0.203 [0.125, 0.297] | 0.925 [0.908, 0.936] | 0.067 [0.062, 0.073] |
| `nonllm_conditioned` | 80 | 0.094 [0.031, 0.172] | 0.847 [0.830, 0.863] | 0.147 [0.134, 0.159] |
| `nonllm_conditioned` | 160 | 0.031 [0.000, 0.078] | 0.667 [0.638, 0.695] | 0.328 [0.303, 0.353] |
| `nonllm_generic` | 20 | 0.969 [0.922, 1.000] | 0.977 [0.948, 1.000] | 0.016 [0.000, 0.035] |
| `nonllm_generic` | 40 | 0.938 [0.875, 0.984] | 0.933 [0.908, 0.957] | 0.060 [0.041, 0.079] |
| `nonllm_generic` | 80 | 0.844 [0.766, 0.906] | 0.856 [0.828, 0.881] | 0.133 [0.115, 0.151] |
| `nonllm_generic` | 160 | 0.641 [0.516, 0.750] | 0.678 [0.624, 0.731] | 0.314 [0.267, 0.355] |
| `oracle` | 20 | 0.734 [0.703, 0.750] | 0.814 [0.790, 0.838] | 0.181 [0.163, 0.199] |
| `oracle` | 40 | 0.469 [0.391, 0.547] | 0.594 [0.561, 0.628] | 0.398 [0.360, 0.433] |
| `oracle` | 80 | 0.234 [0.156, 0.312] | 0.301 [0.235, 0.371] | 0.698 [0.628, 0.763] |
| `oracle` | 160 | 0.094 [0.031, 0.156] | 0.165 [0.101, 0.231] | 0.828 [0.755, 0.898] |
| `paraphrase` | 20 | 0.844 [0.766, 0.922] | 0.919 [0.880, 0.956] | 0.075 [0.041, 0.111] |
| `paraphrase` | 40 | 0.797 [0.719, 0.875] | 0.855 [0.804, 0.904] | 0.139 [0.092, 0.189] |
| `paraphrase` | 80 | 0.688 [0.625, 0.750] | 0.746 [0.706, 0.783] | 0.248 [0.208, 0.291] |
| `paraphrase` | 160 | 0.578 [0.469, 0.688] | 0.607 [0.543, 0.673] | 0.384 [0.322, 0.447] |
| `passthrough` | 20 | 0.953 [0.906, 1.000] | 0.961 [0.932, 0.988] | 0.032 [0.012, 0.054] |
| `passthrough` | 40 | 0.859 [0.797, 0.922] | 0.922 [0.890, 0.955] | 0.071 [0.042, 0.101] |
| `passthrough` | 80 | 0.719 [0.672, 0.750] | 0.815 [0.786, 0.842] | 0.180 [0.157, 0.204] |
| `passthrough` | 160 | 0.531 [0.500, 0.578] | 0.623 [0.596, 0.649] | 0.368 [0.346, 0.389] |
| `random_selection` | 20 | 0.969 [0.922, 1.000] | 0.977 [0.948, 1.000] | 0.016 [0.000, 0.035] |
| `random_selection` | 40 | 0.938 [0.859, 1.000] | 0.929 [0.892, 0.958] | 0.060 [0.041, 0.080] |
| `random_selection` | 80 | 0.797 [0.703, 0.875] | 0.843 [0.800, 0.885] | 0.147 [0.109, 0.185] |
| `random_selection` | 160 | 0.516 [0.406, 0.625] | 0.611 [0.566, 0.663] | 0.384 [0.333, 0.430] |
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
| `lm_conditioned` | 20 | 0.062 [0.000, 0.141] | 0.023 [0.008, 0.041] | 16 | 0 |
| `lm_conditioned` | 40 | 0.615 [0.505, 0.724] | 0.065 [0.060, 0.071] | 16 | 0 |
| `lm_conditioned` | 80 | 0.714 [0.609, 0.812] | 0.165 [0.144, 0.186] | 16 | 0 |
| `lm_conditioned` | 160 | 0.906 [0.828, 0.969] | 0.334 [0.305, 0.364] | 16 | 0 |
| `lm_generic` | 20 | 0.031 [0.000, 0.078] | 0.023 [0.004, 0.045] | 16 | 0 |
| `lm_generic` | 40 | 0.078 [0.031, 0.141] | 0.070 [0.050, 0.089] | 16 | 0 |
| `lm_generic` | 80 | 0.188 [0.094, 0.297] | 0.176 [0.139, 0.213] | 16 | 0 |
| `lm_generic` | 160 | 0.385 [0.297, 0.479] | 0.353 [0.310, 0.395] | 16 | 0 |
| `nonllm_conditioned` | 20 | 0.047 [0.000, 0.125] | 0.026 [0.008, 0.046] | 16 | 0 |
| `nonllm_conditioned` | 40 | 0.797 [0.703, 0.891] | 0.068 [0.062, 0.073] | 16 | 0 |
| `nonllm_conditioned` | 80 | 0.906 [0.828, 0.969] | 0.147 [0.134, 0.159] | 16 | 0 |
| `nonllm_conditioned` | 160 | 0.969 [0.922, 1.000] | 0.329 [0.303, 0.355] | 16 | 0 |
| `nonllm_generic` | 20 | 0.016 [0.000, 0.047] | 0.016 [0.000, 0.035] | 16 | 0 |
| `nonllm_generic` | 40 | 0.047 [0.000, 0.094] | 0.060 [0.041, 0.079] | 16 | 0 |
| `nonllm_generic` | 80 | 0.146 [0.078, 0.208] | 0.138 [0.117, 0.158] | 16 | 0 |
| `nonllm_generic` | 160 | 0.359 [0.250, 0.484] | 0.319 [0.268, 0.367] | 16 | 0 |
| `oracle` | 20 | 0.255 [0.250, 0.266] | 0.181 [0.163, 0.199] | 16 | 0 |
| `oracle` | 40 | 0.526 [0.443, 0.609] | 0.401 [0.367, 0.435] | 16 | 0 |
| `oracle` | 80 | 0.766 [0.688, 0.844] | 0.698 [0.628, 0.762] | 16 | 0 |
| `oracle` | 160 | 0.901 [0.833, 0.964] | 0.833 [0.765, 0.902] | 16 | 0 |
| `paraphrase` | 20 | 0.141 [0.062, 0.219] | 0.075 [0.041, 0.111] | 16 | 0 |
| `paraphrase` | 40 | 0.188 [0.109, 0.266] | 0.139 [0.092, 0.189] | 16 | 0 |
| `paraphrase` | 80 | 0.297 [0.219, 0.375] | 0.248 [0.208, 0.289] | 16 | 0 |
| `paraphrase` | 160 | 0.406 [0.297, 0.516] | 0.389 [0.326, 0.451] | 16 | 0 |
| `passthrough` | 20 | 0.031 [0.000, 0.078] | 0.032 [0.012, 0.054] | 16 | 0 |
| `passthrough` | 40 | 0.125 [0.062, 0.188] | 0.071 [0.042, 0.101] | 16 | 0 |
| `passthrough` | 80 | 0.271 [0.250, 0.307] | 0.180 [0.158, 0.204] | 16 | 0 |
| `passthrough` | 160 | 0.458 [0.411, 0.500] | 0.373 [0.350, 0.396] | 16 | 0 |
| `random_selection` | 20 | 0.016 [0.000, 0.047] | 0.016 [0.000, 0.035] | 16 | 0 |
| `random_selection` | 40 | 0.052 [0.000, 0.109] | 0.065 [0.041, 0.093] | 16 | 0 |
| `random_selection` | 80 | 0.188 [0.094, 0.281] | 0.152 [0.113, 0.190] | 16 | 0 |
| `random_selection` | 160 | 0.474 [0.359, 0.583] | 0.384 [0.334, 0.431] | 16 | 0 |
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
| `lm_conditioned` | 20 | 0.062 | 0.023 | 12.6 | yes |  | - |
| `lm_conditioned` | 40 | 0.609 | 0.065 | 31.1 |  |  | conditioned@20w;reusable@20w |
| `lm_conditioned` | 80 | 0.703 | 0.164 | 72.7 | yes |  | - |
| `lm_conditioned` | 160 | 0.891 | 0.331 | 152.2 |  |  | oracle@160w;reusable@160w |
| `lm_generic` | 20 | 0.031 | 0.023 | 11.7 | yes |  | - |
| `lm_generic` | 40 | 0.078 | 0.070 | 31.1 |  |  | conditioned@20w;generic@20w;oracle@20w;paraphrase@20w;reusable@20w |
| `lm_generic` | 80 | 0.188 | 0.175 | 74.8 |  |  | generic@40w;generic@80w;oracle@20w;oracle@40w;paraphrase@80w;passthrough@80w;reusable@80w |
| `lm_generic` | 160 | 0.375 | 0.350 | 152.1 |  |  | generic@160w;oracle@160w;oracle@40w;oracle@80w;paraphrase@160w;passthrough@160w |
| `nonllm_conditioned` | 20 | 0.047 | 0.025 | 12.5 | yes |  | - |
| `nonllm_conditioned` | 40 | 0.781 | 0.068 | 31.3 |  |  | conditioned@20w;reusable@20w |
| `nonllm_conditioned` | 80 | 0.891 | 0.146 | 72.7 | yes |  | - |
| `nonllm_conditioned` | 160 | 0.953 | 0.326 | 152.7 |  |  | reusable@160w |
| `nonllm_generic` | 20 | 0.016 | 0.016 | 12.9 |  |  | lm_conditioned@20w;lm_generic@20w;nonllm_conditioned@20w;passthrough@20w;random_selection@20w |
| `nonllm_generic` | 40 | 0.047 | 0.059 | 33.3 |  |  | conditioned@20w;generic@20w;lm_conditioned@40w;lm_generic@40w;nonllm_conditioned@40w;oracle@20w;paraphrase@20w;random_selection@40w;reusable@20w |
| `nonllm_generic` | 80 | 0.141 | 0.136 | 74.1 |  |  | generic@40w;generic@80w;lm_conditioned@80w;nonllm_conditioned@80w;oracle@20w;oracle@40w;paraphrase@40w;passthrough@80w;random_selection@80w;reusable@80w |
| `nonllm_generic` | 160 | 0.344 | 0.315 | 153.6 |  |  | generic@160w;generic@80w;lm_conditioned@160w;lm_generic@160w;nonllm_conditioned@160w;oracle@160w;oracle@40w;oracle@80w;paraphrase@160w;passthrough@160w;random_selection@160w;reusable@160w |
| `oracle` | 20 | 0.250 | 0.179 | 18.1 | yes | yes | - |
| `oracle` | 40 | 0.516 | 0.399 | 36.8 | yes | yes | - |
| `oracle` | 80 | 0.750 | 0.692 | 75.6 | yes | yes | - |
| `oracle` | 160 | 0.891 | 0.828 | 143.7 | yes | yes | - |
| `paraphrase` | 20 | 0.141 | 0.074 | 18.6 |  |  | conditioned@20w;oracle@20w;reusable@20w |
| `paraphrase` | 40 | 0.188 | 0.138 | 37.7 |  |  | generic@40w;oracle@20w;oracle@40w |
| `paraphrase` | 80 | 0.297 | 0.247 | 74.7 |  |  | generic@80w;oracle@40w |
| `paraphrase` | 160 | 0.406 | 0.385 | 145.3 |  |  | generic@160w;oracle@160w;oracle@40w;oracle@80w |
| `passthrough` | 20 | 0.031 | 0.031 | 12.6 | yes |  | - |
| `passthrough` | 40 | 0.125 | 0.071 | 33.4 |  |  | conditioned@20w;generic@20w;oracle@20w;paraphrase@20w;reusable@20w |
| `passthrough` | 80 | 0.266 | 0.178 | 72.8 |  |  | generic@40w;oracle@40w |
| `passthrough` | 160 | 0.453 | 0.370 | 151.9 |  |  | generic@160w;oracle@160w;oracle@40w;oracle@80w |
| `random_selection` | 20 | 0.016 | 0.016 | 12.4 |  |  | lm_generic@20w |
| `random_selection` | 40 | 0.047 | 0.064 | 30.8 |  |  | conditioned@20w;generic@20w;oracle@20w;paraphrase@20w;reusable@20w |
| `random_selection` | 80 | 0.188 | 0.150 | 72.9 |  |  | generic@40w;lm_conditioned@80w;oracle@20w;oracle@40w;passthrough@80w;reusable@80w |
| `random_selection` | 160 | 0.469 | 0.381 | 152.4 |  |  | generic@160w;oracle@160w;oracle@40w;oracle@80w |
| `reusable` | 20 | 1.000 | 0.081 | 18.3 | yes | yes | - |
| `reusable` | 40 | 1.000 | 0.103 | 36.8 | yes | yes | - |
| `reusable` | 80 | 1.000 | 0.197 | 72.9 | yes | yes | - |
| `reusable` | 160 | 0.953 | 0.334 | 143.3 | yes | yes | - |

**Nondominated set (17 of 44 configurations):** `reusable`@80w, `reusable`@40w, `reusable`@20w, `conditioned`@20w, `conditioned`@80w, `conditioned`@40w, `reusable`@160w, `oracle`@160w, `nonllm_conditioned`@80w, `oracle`@80w, `lm_conditioned`@80w, `oracle`@40w, `oracle`@20w, `lm_conditioned`@20w, `nonllm_conditioned`@20w, `passthrough`@20w, `lm_generic`@20w.

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
| reusable_minus_generic | 20 | future | -0.022 [-0.045, 0.003] | 0.076 | 16 |
| reusable_minus_generic | 20 | now | 0.859 [0.797, 0.922] | 0.000 | 16 |
| conditioned_minus_generic | 40 | future | -0.110 [-0.153, -0.072] | 0.000 | 16 |
| conditioned_minus_generic | 40 | now | 0.641 [0.547, 0.719] | 0.000 | 16 |
| reusable_minus_generic | 40 | future | -0.103 [-0.143, -0.066] | 0.000 | 16 |
| reusable_minus_generic | 40 | now | 0.656 [0.562, 0.734] | 0.000 | 16 |
| conditioned_minus_generic | 80 | future | -0.210 [-0.268, -0.148] | 0.000 | 15 |
| conditioned_minus_generic | 80 | now | 0.500 [0.383, 0.600] | 0.000 | 15 |
| reusable_minus_generic | 80 | future | -0.128 [-0.178, -0.076] | 0.000 | 15 |
| reusable_minus_generic | 80 | now | 0.517 [0.433, 0.600] | 0.000 | 15 |
| conditioned_minus_generic | 160 | future | -0.335 [-0.390, -0.287] | 0.000 | 8 |
| conditioned_minus_generic | 160 | now | 0.344 [0.281, 0.438] | 0.000 | 8 |
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
| `lm_conditioned` | 20 | 0.922 [0.844, 0.984] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.971 [0.945, 0.993] |
| `lm_conditioned` | 40 | 0.359 [0.250, 0.469] | 0.953 [0.906, 1.000] | 0.969 [0.922, 1.000] | 0.970 [0.952, 0.986] |
| `lm_conditioned` | 80 | 0.250 [0.156, 0.359] | 0.859 [0.766, 0.938] | 0.891 [0.828, 0.953] | 0.870 [0.837, 0.901] |
| `lm_conditioned` | 160 | 0.094 [0.031, 0.172] | 0.703 [0.609, 0.781] | 0.734 [0.609, 0.844] | 0.699 [0.661, 0.736] |
| `lm_generic` | 20 | 0.953 [0.906, 1.000] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.969 [0.941, 0.992] |
| `lm_generic` | 40 | 0.906 [0.844, 0.969] | 0.922 [0.859, 0.969] | 0.953 [0.906, 1.000] | 0.922 [0.902, 0.941] |
| `lm_generic` | 80 | 0.797 [0.688, 0.891] | 0.859 [0.797, 0.922] | 0.812 [0.750, 0.875] | 0.816 [0.777, 0.852] |
| `lm_generic` | 160 | 0.594 [0.484, 0.703] | 0.781 [0.703, 0.859] | 0.578 [0.469, 0.688] | 0.641 [0.594, 0.688] |
| `nonllm_conditioned` | 20 | 0.938 [0.859, 1.000] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.967 [0.939, 0.991] |
| `nonllm_conditioned` | 40 | 0.188 [0.109, 0.266] | 0.984 [0.953, 1.000] | 1.000 [1.000, 1.000] | 0.975 [0.956, 0.990] |
| `nonllm_conditioned` | 80 | 0.078 [0.016, 0.156] | 0.938 [0.875, 0.984] | 0.922 [0.844, 0.984] | 0.897 [0.879, 0.915] |
| `nonllm_conditioned` | 160 | 0.031 [0.000, 0.078] | 0.750 [0.672, 0.844] | 0.672 [0.562, 0.781] | 0.712 [0.681, 0.742] |
| `nonllm_generic` | 20 | 0.969 [0.922, 1.000] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.977 [0.949, 1.000] |
| `nonllm_generic` | 40 | 0.938 [0.875, 0.984] | 0.953 [0.906, 1.000] | 0.906 [0.844, 0.969] | 0.934 [0.906, 0.957] |
| `nonllm_generic` | 80 | 0.828 [0.719, 0.922] | 0.875 [0.781, 0.953] | 0.875 [0.781, 0.953] | 0.855 [0.828, 0.883] |
| `nonllm_generic` | 160 | 0.625 [0.484, 0.750] | 0.688 [0.594, 0.781] | 0.750 [0.656, 0.844] | 0.676 [0.617, 0.730] |
| `oracle` | 20 | 0.734 [0.703, 0.750] | 0.828 [0.781, 0.891] | 0.938 [0.875, 0.984] | 0.809 [0.785, 0.832] |
| `oracle` | 40 | 0.500 [0.406, 0.594] | 0.672 [0.594, 0.750] | 0.703 [0.656, 0.750] | 0.586 [0.551, 0.621] |
| `oracle` | 80 | 0.234 [0.156, 0.312] | 0.359 [0.266, 0.453] | 0.359 [0.281, 0.438] | 0.297 [0.230, 0.367] |
| `oracle` | 160 | 0.094 [0.031, 0.156] | 0.219 [0.141, 0.297] | 0.234 [0.125, 0.359] | 0.160 [0.094, 0.227] |
| `paraphrase` | 20 | 0.859 [0.797, 0.922] | 0.984 [0.953, 1.000] | 0.969 [0.922, 1.000] | 0.914 [0.875, 0.953] |
| `paraphrase` | 40 | 0.812 [0.750, 0.875] | 0.828 [0.750, 0.906] | 0.969 [0.906, 1.000] | 0.852 [0.797, 0.902] |
| `paraphrase` | 80 | 0.688 [0.625, 0.750] | 0.734 [0.641, 0.828] | 0.859 [0.797, 0.922] | 0.742 [0.703, 0.777] |
| `paraphrase` | 160 | 0.547 [0.438, 0.656] | 0.609 [0.531, 0.703] | 0.688 [0.609, 0.766] | 0.605 [0.539, 0.672] |
| `passthrough` | 20 | 0.953 [0.906, 1.000] | 0.953 [0.906, 1.000] | 0.984 [0.953, 1.000] | 0.961 [0.930, 0.988] |
| `passthrough` | 40 | 0.859 [0.797, 0.922] | 0.969 [0.922, 1.000] | 0.984 [0.953, 1.000] | 0.918 [0.887, 0.953] |
| `passthrough` | 80 | 0.719 [0.672, 0.750] | 0.828 [0.781, 0.891] | 0.969 [0.922, 1.000] | 0.809 [0.781, 0.836] |
| `passthrough` | 160 | 0.500 [0.453, 0.547] | 0.688 [0.625, 0.734] | 0.750 [0.750, 0.750] | 0.617 [0.594, 0.645] |
| `random_selection` | 20 | 0.969 [0.922, 1.000] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.977 [0.949, 1.000] |
| `random_selection` | 40 | 0.922 [0.812, 1.000] | 0.969 [0.922, 1.000] | 0.891 [0.828, 0.953] | 0.930 [0.891, 0.961] |
| `random_selection` | 80 | 0.781 [0.688, 0.875] | 0.844 [0.734, 0.922] | 0.938 [0.875, 0.984] | 0.840 [0.797, 0.887] |
| `random_selection` | 160 | 0.516 [0.406, 0.625] | 0.703 [0.609, 0.781] | 0.688 [0.578, 0.797] | 0.605 [0.559, 0.660] |
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
| `lm_conditioned` | 20 | 0.049 [-0.010, 0.126] | 0.162 | 0.34 |
| `lm_conditioned` | 40 | 0.611 [0.490, 0.728] | 0.000 | 2.46 |
| `lm_conditioned` | 80 | 0.620 [0.503, 0.730] | 0.000 | 2.61 |
| `lm_conditioned` | 160 | 0.605 [0.521, 0.685] | 0.000 | 3.50 |
| `lm_generic` | 20 | 0.016 [-0.020, 0.051] | 0.446 | 0.21 |
| `lm_generic` | 40 | 0.016 [-0.023, 0.059] | 0.533 | 0.17 |
| `lm_generic` | 80 | 0.020 [-0.043, 0.090] | 0.610 | 0.14 |
| `lm_generic` | 160 | 0.047 [-0.027, 0.133] | 0.263 | 0.29 |
| `nonllm_conditioned` | 20 | 0.030 [-0.018, 0.095] | 0.320 | 0.25 |
| `nonllm_conditioned` | 40 | 0.788 [0.702, 0.870] | 0.000 | 4.47 |
| `nonllm_conditioned` | 80 | 0.819 [0.736, 0.887] | 0.000 | 5.17 |
| `nonllm_conditioned` | 160 | 0.681 [0.630, 0.724] | 0.000 | 6.93 |
| `nonllm_generic` | 20 | 0.008 [-0.020, 0.039] | 0.672 | 0.13 |
| `nonllm_generic` | 40 | -0.004 [-0.035, 0.031] | 0.917 | -0.05 |
| `nonllm_generic` | 80 | 0.027 [-0.047, 0.113] | 0.530 | 0.16 |
| `nonllm_generic` | 160 | 0.051 [-0.027, 0.148] | 0.260 | 0.28 |
| `oracle` | 20 | 0.074 [0.055, 0.098] | 0.000 | 1.58 |
| `oracle` | 40 | 0.086 [0.027, 0.145] | 0.006 | 0.68 |
| `oracle` | 80 | 0.062 [0.020, 0.105] | 0.005 | 0.68 |
| `oracle` | 160 | 0.066 [0.027, 0.113] | 0.005 | 0.72 |
| `paraphrase` | 20 | 0.055 [0.027, 0.082] | 0.000 | 0.91 |
| `paraphrase` | 40 | 0.039 [0.008, 0.070] | 0.024 | 0.57 |
| `paraphrase` | 80 | 0.055 [0.004, 0.109] | 0.046 | 0.50 |
| `paraphrase` | 160 | 0.059 [0.004, 0.113] | 0.037 | 0.52 |
| `passthrough` | 20 | 0.008 [-0.023, 0.043] | 0.730 | 0.11 |
| `passthrough` | 40 | 0.059 [0.027, 0.090] | 0.000 | 0.88 |
| `passthrough` | 80 | 0.090 [0.066, 0.113] | 0.000 | 1.77 |
| `passthrough` | 160 | 0.117 [0.090, 0.145] | 0.000 | 2.12 |
| `random_selection` | 20 | 0.008 [-0.020, 0.039] | 0.672 | 0.13 |
| `random_selection` | 40 | 0.008 [-0.043, 0.078] | 0.863 | 0.06 |
| `random_selection` | 80 | 0.059 [0.000, 0.117] | 0.064 | 0.47 |
| `random_selection` | 160 | 0.090 [0.023, 0.160] | 0.013 | 0.62 |
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
    "lm_conditioned",
    "lm_generic",
    "nonllm_conditioned",
    "nonllm_generic",
    "oracle",
    "paraphrase",
    "passthrough",
    "random_selection",
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
