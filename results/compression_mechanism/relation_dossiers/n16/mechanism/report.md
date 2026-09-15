# Experiment 14: which compression mechanism specialises a handoff?

Experiment 10 found that conditioning a bounded handoff on the currently known
query buys present-query utility and loses future-query utility, with the loss
growing in the designed distance between the future query and the conditioning
one. Every arm there rewrote the source with an LLM, so *selection* and
*rewriting* were varying together. This run separates them on the same corpus,
channel, reader and judge.

Primary utility: **LLM-judged correct**. Sixteen relation dossiers, four rotating
anchors each, sixteen questions each, budgets 20/40/80/160 delivered words.

| paper name | code arm | rewrites? | selects? | selector |
|---|---|---|---|---|
| paraphrase | `paraphrase` | yes | no (told not to choose) | - |
| summary_generic | `generic` | yes | yes, query-agnostic | LLM |
| summary_conditioned | `conditioned` | yes | yes, query-aware | LLM |
| lm_elimination_generic | `lm_generic` | no | yes, query-agnostic | GPT-2 self-information |
| lm_elimination_conditioned | `lm_conditioned` | no | yes, query-aware | GPT-2 question-likelihood gain |
| nonllm_elimination_generic | `nonllm_generic` | no | yes, query-agnostic | TF-IDF centrality |
| nonllm_elimination_conditioned | `nonllm_conditioned` | no | yes, query-aware | BM25 vs q_now |
| random_selection | `random_selection` | no | yes, seeded shuffle | - |
| passthrough (prefix) | `passthrough` | no | no - positional | - |

LM scorer: `gpt2` @ `607a30d783df`, torch 2.13.0+cpu, transformers 5.15.0, teacher-forced, CPU, no generation.

- generic: `mean_token_surprisal = -(1/T) * sum_t log p(x_t | x_<t), BOS-prefixed`
- conditioned: `mean_question_logprob_gain = (1/|q|) * [ sum_i log p(q_i | unit, q_<i) - sum_i log p(q_i | q_<i) ]`

## Verdict

**The task-aware gradient needs neither rewriting nor an LLM.** At every working budget (40, 80, 160 words) all three mechanisms show a conditioning effect on the near/far gradient whose paired 95% interval excludes zero - including the arm with no neural model anywhere in the compressor: BM25 over the source's own sentences, delivered verbatim. Bounded task-aware *selection* is sufficient to shape a handoff around the present query.

**But only rewriting actually costs future questions.** Against its own generic control, `dU_future` is negative with an interval excluding zero for: abstractive summary 40w -0.110 [-0.152, -0.070]; abstractive summary 80w -0.198 [-0.257, -0.136]; abstractive summary 160w -0.342 [-0.396, -0.292]. It covers zero for: LM elimination 40w -0.005 [-0.025, 0.016]; LM elimination 80w -0.011 [-0.049, 0.029]; LM elimination 160w -0.019 [-0.069, 0.034]; non-LLM elimination 40w 0.008 [-0.011, 0.028]; non-LLM elimination 80w 0.009 [-0.014, 0.031]; non-LLM elimination 160w 0.011 [-0.029, 0.052]. So in the elimination families conditioning buys present utility without measurably taking future utility away from the control it is measured against - the gradient there is produced by lifting the near questions, not by depressing the far ones. In the abstractive family it is a genuine trade, and the trade deepens as the budget grows.

**The mechanisms are not equal in size either.** abstractive summary > LM elimination at 40w (-0.280 [-0.428, -0.124]); abstractive summary > LM elimination at 80w (-0.207 [-0.353, -0.066]); abstractive summary > LM elimination at 160w (-0.276 [-0.417, -0.139]); abstractive summary > non-LLM elimination at 160w (-0.204 [-0.361, -0.062]). Abstractive generation amplifies task commitment beyond selective deletion wherever the intervals separate, so the reading is *selection is sufficient for the shape, rewriting is stronger and is the only one that pays for it* - not that the mechanisms are interchangeable.

| effect the design asks about | supported? | on what evidence |
|---|---|---|
| general task-aware **selection** effect | **yes** | the near/far gradient's DiD excludes zero in all three families, including BM25 with no neural model in the compressor |
| **rewriting** effect | **yes, and it is the only mechanism that costs future utility** | abstractive summary has the steepest gradient, and is the only family whose `dU_future` interval excludes zero |
| **LM-selection** effect | **no** - the LM selector is the *weakest* of the three | a pinned GPT-2 relevance score produces a smaller gradient than BM25 at 40w, 80w |
| some combination | **yes - this one** | selection is sufficient for the shape; rewriting amplifies it and adds an absolute future cost; a learned scorer adds nothing over lexical matching |

The 20-word rung is excluded from every statement above and reported as a granularity stress test: whole-sentence selection cannot fill it (the corpus's shortest sentence is 10 words, its median 27, and five of sixteen dossiers own no sentence that fits), so both elimination families deliver one sentence or none and their intervals cover zero for that reason.

## 1. The channel was equal (or where it was not)

| arm | budget | delivered words (mean) | fill ratio | messages |
|---|---:|---:|---:|---:|
| summary_generic | 20 | 18.8 | 0.94 | 16 |
| summary_generic | 40 | 37.1 | 0.93 | 16 |
| summary_generic | 80 | 73.2 | 0.92 | 16 |
| summary_generic | 160 | 143.6 | 0.90 | 16 |
| summary_conditioned | 20 | 18.3 | 0.91 | 64 |
| summary_conditioned | 40 | 36.5 | 0.91 | 64 |
| summary_conditioned | 80 | 72.9 | 0.91 | 64 |
| summary_conditioned | 160 | 143.9 | 0.90 | 64 |
| lm_elimination_generic | 20 | 11.7 | 0.58 | 16 |
| lm_elimination_generic | 40 | 31.1 | 0.78 | 16 |
| lm_elimination_generic | 80 | 74.8 | 0.94 | 16 |
| lm_elimination_generic | 160 | 152.1 | 0.95 | 16 |
| lm_elimination_conditioned | 20 | 12.6 | 0.63 | 64 |
| lm_elimination_conditioned | 40 | 31.1 | 0.78 | 64 |
| lm_elimination_conditioned | 80 | 72.7 | 0.91 | 64 |
| lm_elimination_conditioned | 160 | 152.2 | 0.95 | 64 |
| nonllm_elimination_generic | 20 | 12.9 | 0.64 | 16 |
| nonllm_elimination_generic | 40 | 33.3 | 0.83 | 16 |
| nonllm_elimination_generic | 80 | 74.1 | 0.93 | 16 |
| nonllm_elimination_generic | 160 | 153.6 | 0.96 | 16 |
| nonllm_elimination_conditioned | 20 | 12.5 | 0.62 | 64 |
| nonllm_elimination_conditioned | 40 | 31.3 | 0.78 | 64 |
| nonllm_elimination_conditioned | 80 | 72.7 | 0.91 | 64 |
| nonllm_elimination_conditioned | 160 | 152.7 | 0.95 | 64 |
| paraphrase | 20 | 18.6 | 0.93 | 16 |
| paraphrase | 40 | 37.7 | 0.94 | 16 |
| paraphrase | 80 | 74.7 | 0.93 | 16 |
| paraphrase | 160 | 145.3 | 0.91 | 16 |
| passthrough (prefix) | 20 | 12.6 | 0.63 | 16 |
| passthrough (prefix) | 40 | 33.4 | 0.84 | 16 |
| passthrough (prefix) | 80 | 72.8 | 0.91 | 16 |
| passthrough (prefix) | 160 | 151.9 | 0.95 | 16 |
| random_selection | 20 | 12.4 | 0.62 | 16 |
| random_selection | 40 | 30.8 | 0.77 | 16 |
| random_selection | 80 | 72.9 | 0.91 | 16 |
| random_selection | 160 | 152.4 | 0.95 | 16 |

Per-message accounting for all 1472 delivered messages is in
`budget_accounting.csv`, one row per message with exactly the fields the design
calls for - `target_words`, `delivered_words`, `fill_ratio`, `source_words`,
`retention_fraction`, `selected_unit_count` - recorded identically for rewriting
and elimination. Messages over their hard cap: **0**.

`selected_unit_count` there counts source sentences present *verbatim* in the
delivered message, a definition that applies to every mechanism. For the
elimination arms it equals their selection by construction; for the abstractive
arms it shows how much of the source they copied rather than rewrote. At the
160-word budget the split is total: 5.19-5.88 verbatim source sentences per
message for `passthrough` and the four elimination arms, against 0.06-0.11 for
`paraphrase`, `summary_generic` and `summary_conditioned`. The two mechanism
families really are doing different things to the text - which is exactly what
Experiment 10's prompt-based `extractive_*` arms failed to guarantee.

Sentence-level selection cannot pad, so an elimination arm spends only what whole
sentences fit. At 20 words that is a granularity failure rather than a policy:
the corpus's shortest sentence is 10 words and its median is 27, and five of the
sixteen dossiers own no sentence short enough to send at all, so their message is
empty. The 20-word rung is reported throughout and interpreted as a stress test,
never as the headline.

## 2. Present and future utility

| arm | budget | U_now | U_future | gap |
|---|---:|---:|---:|---:|
| summary_generic | 20 | 0.141 [0.078, 0.203] | 0.103 [0.077, 0.126] | 0.038 |
| summary_generic | 40 | 0.344 [0.266, 0.438] | 0.206 [0.168, 0.248] | 0.138 |
| summary_generic | 80 | 0.469 [0.391, 0.562] | 0.315 [0.259, 0.368] | 0.154 |
| summary_generic | 160 | 0.609 [0.516, 0.703] | 0.509 [0.452, 0.569] | 0.100 |
| summary_conditioned | 20 | 1.000 [1.000, 1.000] | 0.077 [0.070, 0.085] | 0.923 |
| summary_conditioned | 40 | 0.984 [0.953, 1.000] | 0.096 [0.085, 0.107] | 0.889 |
| summary_conditioned | 80 | 0.984 [0.953, 1.000] | 0.117 [0.103, 0.131] | 0.868 |
| summary_conditioned | 160 | 0.969 [0.922, 1.000] | 0.168 [0.149, 0.186] | 0.801 |
| lm_elimination_generic | 20 | 0.031 [0.000, 0.078] | 0.023 [0.004, 0.045] | 0.008 |
| lm_elimination_generic | 40 | 0.078 [0.031, 0.141] | 0.070 [0.050, 0.089] | 0.008 |
| lm_elimination_generic | 80 | 0.188 [0.094, 0.297] | 0.175 [0.140, 0.214] | 0.013 |
| lm_elimination_generic | 160 | 0.375 [0.281, 0.469] | 0.350 [0.307, 0.393] | 0.025 |
| lm_elimination_conditioned | 20 | 0.062 [0.000, 0.141] | 0.023 [0.007, 0.041] | 0.040 |
| lm_elimination_conditioned | 40 | 0.609 [0.500, 0.719] | 0.065 [0.059, 0.071] | 0.545 |
| lm_elimination_conditioned | 80 | 0.703 [0.594, 0.812] | 0.164 [0.143, 0.182] | 0.540 |
| lm_elimination_conditioned | 160 | 0.891 [0.812, 0.953] | 0.331 [0.302, 0.360] | 0.559 |
| nonllm_elimination_generic | 20 | 0.016 [0.000, 0.047] | 0.016 [0.000, 0.034] | 0.000 |
| nonllm_elimination_generic | 40 | 0.047 [0.000, 0.094] | 0.059 [0.040, 0.078] | -0.012 |
| nonllm_elimination_generic | 80 | 0.141 [0.078, 0.203] | 0.136 [0.117, 0.155] | 0.004 |
| nonllm_elimination_generic | 160 | 0.344 [0.250, 0.438] | 0.315 [0.267, 0.357] | 0.029 |
| nonllm_elimination_conditioned | 20 | 0.047 [0.000, 0.125] | 0.025 [0.007, 0.045] | 0.022 |
| nonllm_elimination_conditioned | 40 | 0.781 [0.688, 0.859] | 0.068 [0.062, 0.073] | 0.714 |
| nonllm_elimination_conditioned | 80 | 0.891 [0.812, 0.953] | 0.146 [0.132, 0.158] | 0.745 |
| nonllm_elimination_conditioned | 160 | 0.953 [0.906, 1.000] | 0.326 [0.301, 0.351] | 0.627 |
| paraphrase | 20 | 0.141 [0.062, 0.219] | 0.074 [0.041, 0.110] | 0.067 |
| paraphrase | 40 | 0.188 [0.109, 0.266] | 0.138 [0.092, 0.188] | 0.050 |
| paraphrase | 80 | 0.297 [0.219, 0.375] | 0.247 [0.206, 0.290] | 0.050 |
| paraphrase | 160 | 0.406 [0.297, 0.516] | 0.385 [0.323, 0.448] | 0.021 |
| passthrough (prefix) | 20 | 0.031 [0.000, 0.078] | 0.031 [0.011, 0.053] | 0.000 |
| passthrough (prefix) | 40 | 0.125 [0.062, 0.188] | 0.071 [0.042, 0.100] | 0.054 |
| passthrough (prefix) | 80 | 0.266 [0.250, 0.297] | 0.178 [0.154, 0.202] | 0.087 |
| passthrough (prefix) | 160 | 0.453 [0.406, 0.500] | 0.370 [0.348, 0.390] | 0.083 |
| random_selection | 20 | 0.016 [0.000, 0.047] | 0.016 [0.000, 0.034] | 0.000 |
| random_selection | 40 | 0.047 [0.000, 0.094] | 0.064 [0.041, 0.086] | -0.017 |
| random_selection | 80 | 0.188 [0.094, 0.281] | 0.150 [0.111, 0.189] | 0.038 |
| random_selection | 160 | 0.469 [0.359, 0.578] | 0.381 [0.330, 0.427] | 0.088 |

## 3. Conditioning within each mechanism (paired, clustered on the source)

| contrast | budget | U_now delta | U_future delta | specialisation DiD |
|---|---:|---:|---:|---:|
| summary_conditioned - summary_generic | 20 | 0.859 [0.797, 0.922] | -0.026 [-0.051, 0.001] | 0.885 [0.834, 0.940] |
| summary_conditioned - summary_generic | 40 | 0.641 [0.547, 0.719] | -0.110 [-0.152, -0.070] | 0.751 [0.688, 0.811] |
| summary_conditioned - summary_generic | 80 | 0.516 [0.406, 0.609] | -0.198 [-0.257, -0.136] | 0.714 [0.639, 0.777] |
| summary_conditioned - summary_generic | 160 | 0.359 [0.250, 0.469] | -0.342 [-0.396, -0.292] | 0.701 [0.613, 0.786] |
| lm_elimination_conditioned - lm_elimination_generic | 20 | 0.031 [0.000, 0.078] | 0.000 [-0.015, 0.013] | 0.031 [-0.005, 0.082] |
| lm_elimination_conditioned - lm_elimination_generic | 40 | 0.531 [0.406, 0.656] | -0.005 [-0.025, 0.016] | 0.536 [0.427, 0.655] |
| lm_elimination_conditioned - lm_elimination_generic | 80 | 0.516 [0.344, 0.688] | -0.011 [-0.049, 0.029] | 0.527 [0.385, 0.674] |
| lm_elimination_conditioned - lm_elimination_generic | 160 | 0.516 [0.391, 0.641] | -0.019 [-0.069, 0.034] | 0.534 [0.437, 0.622] |
| nonllm_elimination_conditioned - nonllm_elimination_generic | 20 | 0.031 [0.000, 0.094] | 0.009 [0.000, 0.023] | 0.022 [-0.007, 0.074] |
| nonllm_elimination_conditioned - nonllm_elimination_generic | 40 | 0.734 [0.625, 0.828] | 0.008 [-0.011, 0.028] | 0.726 [0.629, 0.818] |
| nonllm_elimination_conditioned - nonllm_elimination_generic | 80 | 0.750 [0.656, 0.844] | 0.009 [-0.014, 0.031] | 0.741 [0.645, 0.832] |
| nonllm_elimination_conditioned - nonllm_elimination_generic | 160 | 0.609 [0.484, 0.719] | 0.011 [-0.029, 0.052] | 0.598 [0.509, 0.682] |

## 4. The distance gradient, per mechanism

Regret against the designed distance from the conditioning query. A gradient is
the signature of specialisation: the message is worth less the further the
question moves from the one it was written for.

| arm | budget | paraphrase | same entity | same topic | orthogonal | far - near |
|---|---:|---:|---:|---:|---:|---:|
| summary_generic | 20 | 0.844 | 0.891 | 0.969 | 0.887 | 0.043 |
| summary_generic | 40 | 0.656 | 0.875 | 0.938 | 0.777 | 0.121 |
| summary_generic | 80 | 0.516 | 0.812 | 0.828 | 0.668 | 0.152 |
| summary_generic | 160 | 0.375 | 0.438 | 0.719 | 0.477 | 0.102 |
| summary_conditioned | 20 | 0.000 | 0.953 | 0.984 | 0.983 | 0.983 |
| summary_conditioned | 40 | -0.016 | 0.875 | 0.828 | 0.980 | 0.996 |
| summary_conditioned | 80 | 0.016 | 0.734 | 0.688 | 0.975 | 0.960 |
| summary_conditioned | 160 | 0.016 | 0.422 | 0.516 | 0.952 | 0.936 |
| lm_elimination_generic | 20 | 0.953 | 0.984 | 0.984 | 0.969 | 0.016 |
| lm_elimination_generic | 40 | 0.906 | 0.922 | 0.953 | 0.922 | 0.016 |
| lm_elimination_generic | 80 | 0.797 | 0.859 | 0.812 | 0.816 | 0.020 |
| lm_elimination_generic | 160 | 0.594 | 0.781 | 0.578 | 0.641 | 0.047 |
| lm_elimination_conditioned | 20 | 0.922 | 0.984 | 0.984 | 0.971 | 0.049 |
| lm_elimination_conditioned | 40 | 0.359 | 0.953 | 0.969 | 0.970 | 0.611 |
| lm_elimination_conditioned | 80 | 0.250 | 0.859 | 0.891 | 0.870 | 0.620 |
| lm_elimination_conditioned | 160 | 0.094 | 0.703 | 0.734 | 0.699 | 0.605 |
| nonllm_elimination_generic | 20 | 0.969 | 0.984 | 0.984 | 0.977 | 0.008 |
| nonllm_elimination_generic | 40 | 0.938 | 0.953 | 0.906 | 0.934 | -0.004 |
| nonllm_elimination_generic | 80 | 0.828 | 0.875 | 0.875 | 0.855 | 0.027 |
| nonllm_elimination_generic | 160 | 0.625 | 0.688 | 0.750 | 0.676 | 0.051 |
| nonllm_elimination_conditioned | 20 | 0.938 | 0.984 | 0.984 | 0.967 | 0.030 |
| nonllm_elimination_conditioned | 40 | 0.188 | 0.984 | 1.000 | 0.975 | 0.788 |
| nonllm_elimination_conditioned | 80 | 0.078 | 0.938 | 0.922 | 0.897 | 0.819 |
| nonllm_elimination_conditioned | 160 | 0.031 | 0.750 | 0.672 | 0.712 | 0.681 |

Between families, the same quantity again (a triple difference, paired on the
source): does one mechanism's conditioning effect exceed another's? Same utility
convention - more negative is a steeper near/far gradient, so a negative A - B
means family A specialises more than family B.

| family A | family B | budget | DiD(A) | DiD(B) | A - B | 95% CI |
|---|---|---:|---:|---:|---:|---|
| abstractive summary | LM elimination | 20 | -0.940 | -0.034 | -0.906 | [-0.970, -0.842] |
| abstractive summary | LM elimination | 40 | -0.875 | -0.595 | -0.280 | [-0.428, -0.124] |
| abstractive summary | LM elimination | 80 | -0.807 | -0.600 | -0.207 | [-0.353, -0.066] |
| abstractive summary | LM elimination | 160 | -0.835 | -0.559 | -0.276 | [-0.417, -0.139] |
| abstractive summary | non-LLM elimination | 20 | -0.940 | -0.022 | -0.918 | [-0.977, -0.857] |
| abstractive summary | non-LLM elimination | 40 | -0.875 | -0.792 | -0.083 | [-0.186, 0.021] |
| abstractive summary | non-LLM elimination | 80 | -0.807 | -0.792 | -0.016 | [-0.139, 0.103] |
| abstractive summary | non-LLM elimination | 160 | -0.835 | -0.630 | -0.204 | [-0.361, -0.062] |
| LM elimination | non-LLM elimination | 20 | -0.034 | -0.022 | -0.012 | [-0.047, 0.010] |
| LM elimination | non-LLM elimination | 40 | -0.595 | -0.792 | 0.197 | [0.078, 0.319] |
| LM elimination | non-LLM elimination | 80 | -0.600 | -0.792 | 0.191 | [0.044, 0.339] |
| LM elimination | non-LLM elimination | 160 | -0.559 | -0.630 | 0.072 | [-0.048, 0.204] |

Paired difference in that gradient between a conditioned arm and its own generic
control, on the same sources. This one is computed on **utility**, not regret, so
its sign is the opposite of the table above: utility falls as the question moves
away from the conditioning one, and a **more negative** number means the
conditioned arm gives up more of it going from near to far.

| family | budget | delta(orthogonal - paraphrase) | 95% CI | n sources |
|---|---:|---:|---|---:|
| abstractive summary | 20 | -0.940 | [-0.986, -0.897] | 16 |
| abstractive summary | 40 | -0.875 | [-0.923, -0.823] | 16 |
| abstractive summary | 80 | -0.807 | [-0.874, -0.729] | 16 |
| abstractive summary | 160 | -0.835 | [-0.921, -0.746] | 16 |
| LM elimination | 20 | -0.034 | [-0.091, 0.005] | 16 |
| LM elimination | 40 | -0.595 | [-0.721, -0.473] | 16 |
| LM elimination | 80 | -0.600 | [-0.741, -0.462] | 16 |
| LM elimination | 160 | -0.559 | [-0.665, -0.441] | 16 |
| non-LLM elimination | 20 | -0.022 | [-0.077, 0.009] | 16 |
| non-LLM elimination | 40 | -0.792 | [-0.876, -0.708] | 16 |
| non-LLM elimination | 80 | -0.792 | [-0.888, -0.686] | 16 |
| non-LLM elimination | 160 | -0.630 | [-0.727, -0.513] | 16 |

## 5. Deleted, or delivered and misread?

| arm | budget | q_now answer survived | orthogonal answer survived | accuracy given survival (q_now) |
|---|---:|---:|---:|---:|
| summary_generic | 20 | 0.109 | 0.086 | 1.000 |
| summary_generic | 40 | 0.312 | 0.199 | 1.000 |
| summary_generic | 80 | 0.469 | 0.324 | 1.000 |
| summary_generic | 160 | 0.609 | 0.516 | 1.000 |
| summary_conditioned | 20 | 0.984 | 0.000 | 1.000 |
| summary_conditioned | 40 | 0.984 | 0.005 | 0.984 |
| summary_conditioned | 80 | 0.969 | 0.010 | 1.000 |
| summary_conditioned | 160 | 0.938 | 0.039 | 1.000 |
| lm_elimination_generic | 20 | 0.016 | 0.012 | 1.000 |
| lm_elimination_generic | 40 | 0.094 | 0.078 | 0.833 |
| lm_elimination_generic | 80 | 0.203 | 0.184 | 0.900 |
| lm_elimination_generic | 160 | 0.391 | 0.355 | 0.978 |
| lm_elimination_conditioned | 20 | 0.047 | 0.009 | 1.000 |
| lm_elimination_conditioned | 40 | 0.625 | 0.017 | 0.969 |
| lm_elimination_conditioned | 80 | 0.734 | 0.116 | 0.958 |
| lm_elimination_conditioned | 160 | 0.906 | 0.293 | 0.984 |
| nonllm_elimination_generic | 20 | 0.000 | 0.004 | n/a |
| nonllm_elimination_generic | 40 | 0.047 | 0.051 | 1.000 |
| nonllm_elimination_generic | 80 | 0.141 | 0.133 | 0.938 |
| nonllm_elimination_generic | 160 | 0.344 | 0.312 | 1.000 |
| nonllm_elimination_conditioned | 20 | 0.031 | 0.013 | 1.000 |
| nonllm_elimination_conditioned | 40 | 0.797 | 0.012 | 0.984 |
| nonllm_elimination_conditioned | 80 | 0.922 | 0.094 | 0.969 |
| nonllm_elimination_conditioned | 160 | 0.969 | 0.286 | 0.984 |
| paraphrase | 20 | 0.109 | 0.059 | 1.000 |
| paraphrase | 40 | 0.141 | 0.109 | 1.000 |
| paraphrase | 80 | 0.281 | 0.230 | 0.933 |
| paraphrase | 160 | 0.406 | 0.371 | 0.933 |
| passthrough (prefix) | 20 | 0.016 | 0.020 | 1.000 |
| passthrough (prefix) | 40 | 0.125 | 0.070 | 1.000 |
| passthrough (prefix) | 80 | 0.266 | 0.180 | 1.000 |
| passthrough (prefix) | 160 | 0.469 | 0.375 | 0.969 |
| random_selection | 20 | 0.000 | 0.004 | n/a |
| random_selection | 40 | 0.047 | 0.055 | 1.000 |
| random_selection | 80 | 0.188 | 0.148 | 1.000 |
| random_selection | 160 | 0.469 | 0.387 | 1.000 |

Survival is a string test on the delivered message, so it is defined identically
for rewriting and elimination. It bounds what any reader could do: an arm whose
accuracy sits far below its survival rate is losing answers at the reader, not at
the compressor. On this corpus almost nothing is lost at the reader.

`evidence_survival.csv` also carries one row per designed distance tier, and that
is where the difference between the two conditioned mechanisms becomes concrete.
At 160 words the orthogonal question's answer string survives in only 0.039 of
`summary_conditioned`'s messages, against 0.286-0.293 for the conditioned
elimination arms - which is roughly what their own generic controls (0.312) and
even the random floor (0.387) deliver. Conditioned rewriting *erases* the distant
evidence; conditioned selection merely does not go out of its way to include it,
and keeps about as much of it as an unconditioned selector would. That is the
mechanism behind the split verdict above.

(The `q_now` and `paraphrase` columns of that table are equal by construction, not
by coincidence: the corpus builder requires a paraphrase question to share its
anchor's gold answers, and `regret_data.validate_relation_context` enforces it.)

## 6. Allocation, measured without the reader

The corpus partitions its evidence into four aspects, one per question group,
and no sentence is claimed by two. For a selection arm the delivered units are
known exactly, so the share of the conditioning aspect that survived can be
compared with the share of the other three - a compressor-side measurement with
no reader and no judge in the path.

| arm | budget | conditioning aspect kept | other aspects kept | delta | 95% CI |
|---|---:|---:|---:|---:|---|
| lm_elimination_conditioned | 20 | 0.021 | 0.010 | 0.010 | [-0.010, 0.036] |
| lm_elimination_conditioned | 40 | 0.232 | 0.016 | 0.216 | [0.170, 0.260] |
| lm_elimination_conditioned | 80 | 0.315 | 0.105 | 0.210 | [0.158, 0.266] |
| lm_elimination_conditioned | 160 | 0.487 | 0.275 | 0.212 | [0.153, 0.276] |
| lm_elimination_generic | 20 | 0.010 | 0.010 | 0.000 | [0.000, 0.000] |
| lm_elimination_generic | 40 | 0.073 | 0.073 | 0.000 | [0.000, 0.000] |
| lm_elimination_generic | 80 | 0.177 | 0.177 | -0.000 | [-0.000, 0.000] |
| lm_elimination_generic | 160 | 0.346 | 0.346 | 0.000 | [0.000, 0.000] |
| nonllm_elimination_conditioned | 20 | 0.016 | 0.012 | 0.003 | [-0.007, 0.021] |
| nonllm_elimination_conditioned | 40 | 0.268 | 0.010 | 0.258 | [0.224, 0.291] |
| nonllm_elimination_conditioned | 80 | 0.352 | 0.087 | 0.265 | [0.222, 0.304] |
| nonllm_elimination_conditioned | 160 | 0.513 | 0.266 | 0.247 | [0.182, 0.306] |
| nonllm_elimination_generic | 20 | 0.005 | 0.005 | 0.000 | [0.000, 0.000] |
| nonllm_elimination_generic | 40 | 0.052 | 0.052 | 0.000 | [0.000, 0.000] |
| nonllm_elimination_generic | 80 | 0.130 | 0.130 | -0.000 | [-0.000, -0.000] |
| nonllm_elimination_generic | 160 | 0.302 | 0.302 | 0.000 | [-0.000, 0.000] |
| passthrough (prefix) | 20 | 0.021 | 0.021 | 0.000 | [0.000, 0.000] |
| passthrough (prefix) | 40 | 0.049 | 0.049 | 0.000 | [0.000, 0.000] |
| passthrough (prefix) | 80 | 0.148 | 0.148 | 0.000 | [0.000, 0.000] |
| passthrough (prefix) | 160 | 0.344 | 0.344 | 0.000 | [0.000, 0.000] |
| random_selection | 20 | 0.005 | 0.005 | 0.000 | [0.000, 0.000] |
| random_selection | 40 | 0.057 | 0.057 | 0.000 | [0.000, 0.000] |
| random_selection | 80 | 0.135 | 0.135 | -0.000 | [-0.000, 0.000] |
| random_selection | 160 | 0.359 | 0.359 | 0.000 | [-0.000, 0.000] |

## 7. Provenance

| | |
|---|---|
| sender and reader | `meta-llama/llama-3.1-8b-instruct`, temperature 0.0 / 0.0 |
| judge | `openai/gpt-4o-mini`, binary correct/incorrect against the gold |
| LM selector | `gpt2` @ `607a30d783df`, teacher-forced, CPU, no sampling |
| non-LLM selectors | Okapi BM25 (`src/retrieval.py`) and TF-IDF centrality (`src/elimination.py`); no pretrained model |
| LM selector provenance | **LongLLMLingua-*style*, not the official implementation.** No `llmlingua` package is used; the coarse question-aware ranking direction is reimplemented here and its formula is stated above and in the artefact manifest |
| seeds | inherited from Experiment 10 unchanged: sender and answerer are deterministic (temperature 0, `seed=None`), the closed-book baseline uses `1000 + sample`. The selectors are deterministic functions of (units, cap, query); `random_selection` is seeded by a SHA-256 of `salt|context_id|budget`, so it is stable across processes and machines |
| granularity | sentence / evidence unit, the primary level the design asks for. Token-level deletion was not run: it introduces a readability confound this design cannot separate from information loss |
| corpus | data/communication_regret/relation_dossiers.jsonl |
| reused unchanged | 640 Experiment 10 messages + 11,264 answers, request hashes verified equal by the offline selftest |
| generated this experiment | 64 `paraphrase` messages; 768 selection messages at zero model cost; 8,896 unique answer calls after single-flighting (24,576 answer cells, deduplicated by request hash) |
| cost | sender/answerer $0.0434 + judge $0.0332 = **$0.077** |
| wall-clock (observed, not instrumented) | GPT-2 scoring of all 16 dossiers 106 s on CPU; the main run ~20 min end to end at `runtime.concurrency` 12 and judge concurrency 16; the `passthrough` top-up ~1 min. The pipeline is bound by API latency, not by local compute |

```bash
python src/lm_unit_scores.py --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
python src/run_communication_regret.py --config compression_mechanism_config.yaml
python src/render_mechanism_report.py
python src/selftest_compression_mechanism_offline.py   # 43 checks, no API key
```

Files added: `src/elimination.py`, `src/lm_unit_scores.py`, `src/render_mechanism_report.py`, `src/selftest_compression_mechanism_offline.py`, `compression_mechanism_config.yaml`. Files extended additively: `src/run_communication_regret.py` (selection specs, config-driven contrasts, a guard against an arm silently leaving the utility matrix), `src/budget.py` (the `paraphrase` block only).

Not run, deliberately: token-level elimination (deleting inside a sentence adds a readability confound this design cannot separate from information loss), and the `informativeness` variant of the query-free lexical scorer (implemented and config-selectable, never exercised).

## Figures

- `figures/distance_curves.png` - the main comparison.
- `figures/now_future_tradeoff.png` - the present/future trade-off with paraphrase.
- `figures/evidence_survival.png` - the deletion/reader diagnostic.

