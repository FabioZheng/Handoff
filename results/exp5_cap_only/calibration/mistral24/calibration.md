# exp5_cap_only budget calibration — mistral24

Calibration run `463a16d6fbfb-calib`. Freeze rule from `exp5_cap_only_config.yaml`, written before this run returned. Conditioned answers are discarded at load time, so no conditioned-minus-generic utility is computed here.

## qasper

Closed book 0.020 · full source 0.240 · annotated evidence 0.316 F1. Natural length, uncapped: generic median 289 words, conditioned 198. Median source sentence 19 words.

| budget | binding share | U generic | R [95% CI] | generic words / fill | conditioned words / fill | truncated g / c |
|---:|---:|---:|---|---|---|---:|
| 24 | 1.00 | 0.028 | 0.04 [-0.06, 0.15] | 22 / 0.92 | 22 / 0.92 | 7 / 9 |
| 40 | 1.00 | 0.032 | 0.05 [-0.04, 0.16] | 37 / 0.93 | 35 / 0.88 | 7 / 8 |
| 64 | 1.00 | 0.052 | 0.14 [0.04, 0.24] | 56 / 0.88 | 56 / 0.87 | 8 / 4 |
| 96 | 1.00 | 0.084 | 0.29 [0.15, 0.44] | 86 / 0.90 | 80 / 0.84 | 2 / 2 |
| 160 | 1.00 | 0.101 | 0.37 [0.25, 0.49] | 128 / 0.80 | 129 / 0.81 | 3 / 6 |
| 256 | 0.65 | 0.137 | 0.53 [0.37, 0.70] | 202 / 0.79 | 186 / 0.73 | 5 / 8 |
| 384 | 0.30 | 0.147 | 0.58 [0.44, 0.71] | 248 / 0.65 | 270 / 0.70 | 0 / 3 |
| 576 | 0.05 | 0.153 | 0.60 [0.46, 0.74] | 338 / 0.59 | 330 / 0.57 | 0 / 0 |

Region: B_min = 96, B_max = 256, granularity floor 38 words → **useful region**

## squad

Closed book 0.063 · full source 0.442 · annotated evidence 0.507 F1. Natural length, uncapped: generic median 486 words, conditioned 87. Median source sentence 22 words.

| budget | binding share | U generic | R [95% CI] | generic words / fill | conditioned words / fill | truncated g / c |
|---:|---:|---:|---|---|---|---:|
| 24 | 1.00 | 0.019 | -0.12 [-0.19, -0.06] | 22 / 0.90 | 21 / 0.88 | 22 / 13 |
| 40 | 1.00 | 0.012 | -0.13 [-0.21, -0.08] | 36 / 0.90 | 35 / 0.88 | 29 / 8 |
| 64 | 1.00 | 0.015 | -0.13 [-0.20, -0.08] | 60 / 0.93 | 56 / 0.88 | 23 / 6 |
| 96 | 1.00 | 0.021 | -0.11 [-0.19, -0.05] | 89 / 0.93 | 79 / 0.82 | 26 / 8 |
| 160 | 1.00 | 0.023 | -0.10 [-0.18, -0.04] | 152 / 0.95 | 100 / 0.63 | 21 / 8 |
| 256 | 0.97 | 0.053 | -0.03 [-0.11, 0.05] | 244 / 0.95 | 116 / 0.45 | 22 / 6 |
| 384 | 0.78 | 0.085 | 0.06 [-0.04, 0.15] | 348 / 0.91 | 141 / 0.37 | 10 / 5 |
| 576 | 0.30 | 0.096 | 0.09 [-0.01, 0.18] | 518 / 0.90 | 171 / 0.30 | 8 / 5 |

Region: B_min = None, B_max = 384, granularity floor 44 words → **no useful region**

## Decision

Headline corpora: qasper.
Proposed grid: [96, 120, 144, 176, 208, 256].
No flags.
