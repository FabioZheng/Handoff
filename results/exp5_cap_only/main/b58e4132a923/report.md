# exp5_cap_only main analysis — mistral24

Run `b58e4132a923`, split `main`, frozen caps [96, 120, 144, 176, 208, 256]. Headline corpora from calibration: qasper.

## Primary: total effect of question visibility on future utility (F1)

Conditioned minus generic at the same nominal cap, paired on document. Holm-corrected across caps; the `mean` row is the pre-specified summary over all caps.

| corpus | family | cap | U_future cond / gen | delta future [95% CI] | p (Holm) | delta now | docs |
|---|---|---|---|---|---|---|---:|
| qasper | generation | 96 | 0.067 / 0.120 | -0.052 [-0.069, -0.037] | 0.0000 | +0.224 | 200 |
| qasper | generation | 120 | 0.076 / 0.151 | -0.075 [-0.093, -0.059] | 0.0000 | +0.185 | 200 |
| qasper | generation | 144 | 0.084 / 0.156 | -0.073 [-0.090, -0.057] | 0.0000 | +0.184 | 200 |
| qasper | generation | 176 | 0.089 / 0.173 | -0.084 [-0.101, -0.068] | 0.0000 | +0.152 | 200 |
| qasper | generation | 208 | 0.095 / 0.195 | -0.100 [-0.118, -0.082] | 0.0000 | +0.126 | 200 |
| qasper | generation | 256 | 0.104 / 0.212 | -0.108 [-0.125, -0.090] | 0.0000 | +0.110 | 200 |
| qasper | generation | mean |  | -0.082 [-0.094, -0.069] | 0.0000 | — | 200 |
| qasper | selection | 96 | 0.047 / 0.036 | +0.011 [-0.001, +0.024] | 0.2640 | +0.285 | 200 |
| qasper | selection | 120 | 0.050 / 0.052 | -0.002 [-0.016, +0.011] | 1.0000 | +0.275 | 200 |
| qasper | selection | 144 | 0.054 / 0.058 | -0.004 [-0.017, +0.010] | 1.0000 | +0.269 | 200 |
| qasper | selection | 176 | 0.055 / 0.064 | -0.008 [-0.023, +0.006] | 0.7140 | +0.265 | 200 |
| qasper | selection | 208 | 0.057 / 0.071 | -0.014 [-0.028, -0.000] | 0.2100 | +0.256 | 200 |
| qasper | selection | 256 | 0.057 / 0.083 | -0.026 [-0.042, -0.011] | 0.0060 | +0.245 | 200 |
| qasper | selection | mean |  | -0.007 [-0.019, +0.006] | 0.2440 | — | 200 |

## Generation minus selection (interaction), F1

| corpus | cap | generation − selection [95% CI] | docs |
|---|---:|---|---:|
| qasper | 96 | -0.064 [-0.082, -0.047] | 200 |
| qasper | 120 | -0.073 [-0.092, -0.055] | 200 |
| qasper | 144 | -0.069 [-0.086, -0.051] | 200 |
| qasper | 176 | -0.076 [-0.095, -0.055] | 200 |
| qasper | 208 | -0.086 [-0.106, -0.067] | 200 |
| qasper | 256 | -0.082 [-0.103, -0.058] | 200 |

## Confirmatory: evidence survival by distance (conditioned − generic)

Answer-token recall; yes/no questions excluded. `near` is disjoint evidence within 3 paragraphs, a positional stand-in for same section.

| corpus | family | cap | now | shared | near | far |
|---|---|---:|---:|---:|---:|---:|
| qasper | generation | 96 | +0.45 | +0.12 | -0.12 | -0.15 |
| qasper | generation | 120 | +0.39 | +0.07 | -0.16 | -0.18 |
| qasper | generation | 144 | +0.38 | +0.06 | -0.16 | -0.18 |
| qasper | generation | 176 | +0.33 | +0.03 | -0.19 | -0.20 |
| qasper | generation | 208 | +0.31 | +0.02 | -0.18 | -0.22 |
| qasper | generation | 256 | +0.28 | +0.01 | -0.19 | -0.24 |
| qasper | selection | 96 | +0.53 | +0.20 | -0.05 | -0.06 |
| qasper | selection | 120 | +0.53 | +0.19 | -0.07 | -0.08 |
| qasper | selection | 144 | +0.52 | +0.18 | -0.08 | -0.09 |
| qasper | selection | 176 | +0.51 | +0.16 | -0.10 | -0.11 |
| qasper | selection | 208 | +0.50 | +0.13 | -0.07 | -0.13 |
| qasper | selection | 256 | +0.46 | +0.10 | -0.12 | -0.17 |

## Mechanism: what each 100 added words bought

| corpus | policy | step | now | shared | near | far | repeated 5-grams | unsupported |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qasper | summary_conditioned | 96->120 | +0.02 | +0.01 | +0.02 | +0.24 | +0.93 | +0.08 |
| qasper | summary_conditioned | 120->144 | +0.07 | -0.06 | +0.08 | +0.29 | +1.97 | +0.00 |
| qasper | summary_conditioned | 144->176 | +0.27 | +0.19 | +0.06 | +0.22 | +0.42 | +0.22 |
| qasper | summary_conditioned | 176->208 | +0.05 | -0.00 | +0.00 | +0.02 | +1.24 | -0.02 |
| qasper | summary_conditioned | 208->256 | +0.02 | +0.06 | +0.05 | -0.04 | +4.33 | +0.73 |
| qasper | summary_generic | 96->120 | +0.26 | +0.13 | +0.18 | +0.59 | +0.09 | +1.90 |
| qasper | summary_generic | 120->144 | +0.42 | +0.14 | +0.21 | +1.09 | +1.29 | +9.70 |
| qasper | summary_generic | 144->176 | +0.18 | +0.11 | +0.11 | +0.28 | -0.29 | +7.32 |
| qasper | summary_generic | 176->208 | +0.67 | +1.14 | +0.28 | +1.87 | +0.64 | +3.52 |
| qasper | summary_generic | 208->256 | +0.07 | -0.02 | +0.04 | +0.19 | +0.85 | +2.11 |
| qasper | selection_conditioned | 96->120 | -0.01 | +0.11 | -0.07 | +0.25 | +0.69 | +0.19 |
| qasper | selection_conditioned | 120->144 | +0.09 | +0.07 | +0.08 | +0.12 | +0.26 | +1.30 |
| qasper | selection_conditioned | 144->176 | +0.02 | +0.01 | +0.04 | +0.12 | +0.55 | +0.76 |
| qasper | selection_conditioned | 176->208 | +0.04 | +0.04 | +0.08 | +0.15 | +1.00 | +1.21 |
| qasper | selection_conditioned | 208->256 | +0.04 | +0.07 | +0.04 | +0.10 | +0.55 | +0.93 |
| qasper | selection_generic | 96->120 | +0.13 | +0.11 | +0.08 | +0.23 | +0.97 | +0.11 |
| qasper | selection_generic | 120->144 | +0.12 | +0.08 | +0.07 | +0.21 | +2.24 | +0.26 |
| qasper | selection_generic | 144->176 | +0.06 | +0.05 | +0.03 | +0.10 | +2.43 | +0.49 |
| qasper | selection_generic | 176->208 | +0.06 | +0.03 | +0.02 | +0.15 | +2.00 | +0.48 |
| qasper | selection_generic | 208->256 | +0.07 | +0.05 | +0.05 | +0.12 | +1.79 | +0.67 |

## Secondary: controlled direct effect (length pathway closed)

Realized length is a mediator; these do not estimate the total effect, and the direct effect is identified only where the two arms' realized lengths overlap. **Matched length** is the reported estimate: each conditioned message against the generic message of the same document, any cap, within 15% realized length; the match rate says how much overlap there is. The two regressions are checks and are marked **not identified** when common support is below 30% or the VIF of visibility exceeds 10.

| corpus | family | matched length [95% CI] | match rate | length bins [95% CI] (support) | log-linear [95% CI] (VIF) |
|---|---|---|---:|---|---|
| qasper | generation | -0.070 [-0.083, -0.057] | 0.85 | -0.075 [-0.086, -0.064] (99%) | -0.078 [-0.088, -0.067] (VIF 1.1) |
| qasper | selection | +0.011 [-0.004, +0.027] | 0.62 | +0.007 [-0.007, +0.022] (80%) | +0.007 [-0.007, +0.020] (VIF 1.4) |

## Channel audit

| corpus | policy | cap | valid | median words | median fill | median tokens | truncated | over cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qasper | selection_conditioned | 96 | 0.999 | 87 | 0.9062 | 25 | 0 | 0 |
| qasper | selection_conditioned | 120 | 1.000 | 102.0 | 0.85 | 25.0 | 0 | 0 |
| qasper | selection_conditioned | 144 | 1.000 | 118.0 | 0.8194 | 26.0 | 0 | 0 |
| qasper | selection_conditioned | 176 | 1.000 | 128.0 | 0.7273000000000001 | 28.5 | 0 | 0 |
| qasper | selection_conditioned | 208 | 1.000 | 124.0 | 0.5962 | 26.0 | 0 | 0 |
| qasper | selection_conditioned | 256 | 1.000 | 119.0 | 0.46485 | 26.0 | 0 | 0 |
| qasper | selection_generic | 96 | 1.000 | 96.0 | 1.0 | 477.5 | 0 | 0 |
| qasper | selection_generic | 120 | 1.000 | 120.0 | 1.0 | 468.5 | 0 | 0 |
| qasper | selection_generic | 144 | 1.000 | 144.0 | 1.0 | 492.5 | 0 | 0 |
| qasper | selection_generic | 176 | 1.000 | 176.0 | 1.0 | 571.0 | 0 | 0 |
| qasper | selection_generic | 208 | 1.000 | 208.0 | 1.0 | 650.5 | 0 | 0 |
| qasper | selection_generic | 256 | 1.000 | 256.0 | 1.0 | 602.5 | 0 | 0 |
| qasper | summary_conditioned | 96 | 1.000 | 79.0 | 0.8229 | 111.0 | 22 | 0 |
| qasper | summary_conditioned | 120 | 1.000 | 95.0 | 0.7917 | 131.0 | 33 | 0 |
| qasper | summary_conditioned | 144 | 1.000 | 106.0 | 0.7361 | 148.0 | 24 | 0 |
| qasper | summary_conditioned | 176 | 1.000 | 129.0 | 0.733 | 179.0 | 36 | 0 |
| qasper | summary_conditioned | 208 | 1.000 | 148.0 | 0.7115 | 204.0 | 42 | 0 |
| qasper | summary_conditioned | 256 | 1.000 | 173.0 | 0.6758 | 239.0 | 55 | 0 |
| qasper | summary_generic | 96 | 1.000 | 87.0 | 0.9062 | 124.0 | 16 | 0 |
| qasper | summary_generic | 120 | 1.000 | 107.0 | 0.8917 | 157.0 | 25 | 0 |
| qasper | summary_generic | 144 | 1.000 | 122.0 | 0.8472 | 183.0 | 9 | 0 |
| qasper | summary_generic | 176 | 1.000 | 147.5 | 0.83805 | 225.5 | 18 | 0 |
| qasper | summary_generic | 208 | 1.000 | 167.0 | 0.8029 | 268.0 | 11 | 0 |
| qasper | summary_generic | 256 | 1.000 | 196.5 | 0.76755 | 320.5 | 6 | 0 |
