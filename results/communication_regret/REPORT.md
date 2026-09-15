# Communication Regret

**Experiment 10 · agent handoff information-loss probe**

When one agent writes a bounded handoff for the question it currently knows, does it buy accuracy on that question by spending capacity it would otherwise have spent on questions nobody has asked yet? With the channel held equal to within a word or two, it does.

> **Verdict — supported.** On natural SQuAD prose, conditioning raises present-query accuracy by +0.62 / +0.45 / +0.28 at 20 / 40 / 80 delivered words while lowering hidden-query accuracy by -0.06 / -0.11 / -0.19. Both directions exclude zero, and the trade-off sharpens as bandwidth falls. An oracle handoff of the same length carries the hidden answers, so this is allocation, not capacity.

|  |  |
|---|---|
| Sender and answerer | `meta-llama/llama-3.1-8b-instruct` |
| Judge | `openai/gpt-4o-mini` |
| SQuAD groups | 24 contexts &times; 4 questions (natural, human-written questions); 96 rotations |
| Relation dossiers | 16 contexts &times; 16 questions (designed distance labels); 64 rotations |
| Budgets | 20 words, 40 words, 80 words, 160 words |
| Primary utility | judge_correct |

This file is the narrative report and carries every headline table for both corpora. The machine-generated per-corpus reports, which additionally list the full per-rotation tables, are [`squad_groups/n24/report.md`](squad_groups/n24/report.md) and [`relation_dossiers/n16/report.md`](relation_dossiers/n16/report.md). Raw per-cell outputs are the CSVs listed at the end.

---

## 1. The claim

Every prior conditioning experiment in this project compared a handoff written for a known question against one written for no question &mdash; and let the two choose their own length. On the same source, the conditioned arm wrote roughly 600 characters and the generic arm roughly 3,800. Any held-out accuracy gap was therefore inseparable from *one arm simply wrote more*.

This experiment keeps free-text handoffs and makes the channel itself the controlled variable. The hypothesis under test:

> Under an equal communication budget, conditioning a handoff on the currently known information need reallocates capacity toward present utility and can reduce its utility for plausible future information needs.

Supporting it requires **both** a present-query gain and a future-query loss against a baseline that knows nothing, strengthening as the budget shrinks. A gain alone is not the claim; a loss alone is not either.

## 2. The design

Each source context independently answers *k* questions. Every question takes a turn as the conditioning query, and each resulting message is then answered against *every* question of that context. That produces a utility matrix per context: the diagonal is present-query utility `U_now`, the off-diagonal is future-query utility `U_future`. Because the same question sits on the diagonal for one rotation and off it for the other *k*&minus;1, question difficulty cannot produce the contrast and no fixed A/B split has to be trusted.

One real context (SQuAD, *Scottish Parliament*) at a 34&ndash;40 word budget. Diagonal cells are marked `[ ]`.

| conditioned on &darr; / asked &rarr; | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| **on Q1** | **[1]** | 1 | 0 | 0 |
| **on Q2** | 0 | **[0]** | 0 | 1 |
| **on Q3** | 0 | 0 | **[1]** | 0 |
| **on Q4** | 0 | 0 | 0 | **[1]** |

`generic` and `oracle` hold one message each, so every row is identical:

| policy | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| `generic` | 0 | 0 | 0 | 0 |
| `oracle` | 1 | 1 | 1 | 0 |

`generic` spends its 37 words on the Kilbrandon Commission and answers none of the four questions asked; `oracle` fits three of the four answers into 39 words; each `conditioned` message answers its own question and, with one accident, nothing else. Q2's diagonal is 0 because the answerer replied "3" where the gold is "three hundred" &mdash; the diagonal is not automatically 1. The full messages behind this matrix are in [`squad_groups/n24/example.md`](squad_groups/n24/example.md).

### The policies

They differ only in what the sender is told about the information need; an offline check asserts that replacing that block with a placeholder leaves four identical prompts.

| policy | what the sender is told | messages per context and budget |
|---|---|---|
| `generic` | A question will be asked; it has not been told which. It is *not* told that further questions may follow &mdash; that is the reusable treatment, and a control must not contain the independent variable. | 1 |
| `conditioned` | Sees the current question and is asked to communicate what answers it. | *k* |
| `reusable` | Sees the current question, is told unknown further questions may follow, and is asked to preserve reusable evidence within the same limit. | *k* |
| `oracle` | Sees every question the context supports. An upper bound and a capacity probe, not a deployable strategy. | 1 |
| `extractive_generic` / `extractive_conditioned` | The same two conditioning levels, but sentences copied verbatim instead of prose. Separates information selection from abstractive rewriting. | 1 / *k* |

Every sender-side string is reproduced verbatim in [`PROMPTS.md`](../../PROMPTS.md#experiment-10).

## 3. The control: was the channel actually equal?

The first pilot gave every policy the same one-sided instruction: *at most N words*. The arms obeyed it and still spent very different amounts of channel. Realised fill (delivered words &divide; cap) at a 40-word cap:

| policy | cap only | cap and floor |
|---|---:|---:|
| `conditioned` | 0.61 | 0.92 |
| `reusable` | 0.75 | 0.91 |
| `oracle` | 0.83 | 0.96 |
| `generic` | 0.88 | 0.93 |

Under a cap alone the conditioned arm spent **a third less channel** than its control, so any future-query deficit it showed would have meant "it wrote less". The budget is therefore a **two-sided word band on the delivered message**, enforced in three stages: one length contract stated identically to every policy; an out-of-band draft returned for a rewrite with a byte-identical correction; and unconditional truncation to the cap at the last sentence boundary that fits. The cap always holds. The floor cannot &mdash; words cannot be invented on demand &mdash; so it is requested, audited and reported.

![Was the channel actually equal?](squad_groups/n24/budget_control.png)

### SQuAD groups &mdash; delivered length, every arm and budget

| policy | band | delivered (mean &plusmn; sd) | fill | under floor | truncated | corrections | repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| `generic` | 17&ndash;20 | 18.7 &plusmn; 1.0 | 0.94 | 0.00 | 0.00 | 0.50 | 0.000 |
| `conditioned` | 17&ndash;20 | 18.5 &plusmn; 1.0 | 0.92 | 0.00 | 0.00 | 0.38 | 0.000 |
| `reusable` | 17&ndash;20 | 18.6 &plusmn; 1.1 | 0.93 | 0.00 | 0.01 | 0.39 | 0.000 |
| `oracle` | 17&ndash;20 | 18.1 &plusmn; 2.3 | 0.91 | 0.12 | 0.17 | 0.75 | 0.000 |
| `extractive_generic` | 17&ndash;20 | 18.9 &plusmn; 1.1 | 0.95 | 0.00 | 0.12 | 1.00 | 0.000 |
| `extractive_conditioned` | 17&ndash;20 | 18.6 &plusmn; 1.3 | 0.93 | 0.04 | 0.05 | 0.85 | 0.000 |
| `generic` | 34&ndash;40 | 37.1 &plusmn; 1.8 | 0.93 | 0.00 | 0.00 | 0.38 | 0.000 |
| `conditioned` | 34&ndash;40 | 36.7 &plusmn; 2.0 | 0.92 | 0.01 | 0.00 | 0.66 | 0.000 |
| `reusable` | 34&ndash;40 | 36.3 &plusmn; 3.8 | 0.91 | 0.02 | 0.02 | 0.54 | 0.000 |
| `oracle` | 34&ndash;40 | 37.9 &plusmn; 1.8 | 0.95 | 0.00 | 0.00 | 0.75 | 0.000 |
| `extractive_generic` | 34&ndash;40 | 37.4 &plusmn; 2.3 | 0.94 | 0.04 | 0.04 | 0.96 | 0.000 |
| `extractive_conditioned` | 34&ndash;40 | 36.8 &plusmn; 3.6 | 0.92 | 0.05 | 0.06 | 0.79 | 0.017 |
| `conditioned@trim` | 34&ndash;40 | 30.6 &plusmn; 7.4 | 0.76 | 0.01 | 0.84 | 0.66 | 0.000 |
| `generic@trim` | 34&ndash;40 | 24.8 &plusmn; 7.9 | 0.62 | 0.00 | 0.88 | 0.38 | 0.000 |
| `oracle@trim` | 34&ndash;40 | 24.3 &plusmn; 6.5 | 0.61 | 0.00 | 0.96 | 0.75 | 0.000 |
| `reusable@trim` | 34&ndash;40 | 27.4 &plusmn; 9.1 | 0.69 | 0.02 | 0.84 | 0.54 | 0.000 |
| `generic` | 68&ndash;80 | 71.9 &plusmn; 4.4 | 0.90 | 0.04 | 0.04 | 0.25 | 0.000 |
| `conditioned` | 68&ndash;80 | 73.2 &plusmn; 4.4 | 0.92 | 0.01 | 0.00 | 0.71 | 0.002 |
| `reusable` | 68&ndash;80 | 73.2 &plusmn; 3.4 | 0.92 | 0.00 | 0.00 | 0.58 | 0.001 |
| `oracle` | 68&ndash;80 | 73.8 &plusmn; 3.7 | 0.92 | 0.00 | 0.00 | 0.46 | 0.001 |
| `extractive_generic` | 68&ndash;80 | 73.3 &plusmn; 4.1 | 0.92 | 0.04 | 0.04 | 0.54 | 0.000 |
| `extractive_conditioned` | 68&ndash;80 | 70.9 &plusmn; 4.7 | 0.89 | 0.16 | 0.06 | 0.92 | 0.027 |
| `generic` | 136&ndash;160 | 141.4 &plusmn; 4.3 | 0.88 | 0.00 | 0.00 | 0.58 | 0.003 |
| `conditioned` | 136&ndash;160 | 143.1 &plusmn; 5.5 | 0.89 | 0.00 | 0.00 | 0.86 | 0.004 |
| `reusable` | 136&ndash;160 | 143.6 &plusmn; 5.5 | 0.90 | 0.00 | 0.00 | 0.78 | 0.005 |
| `oracle` | 136&ndash;160 | 143.4 &plusmn; 4.7 | 0.90 | 0.00 | 0.00 | 0.75 | 0.005 |
| `extractive_generic` | 136&ndash;160 | 143.8 &plusmn; 6.4 | 0.90 | 0.00 | 0.00 | 0.79 | 0.003 |
| `extractive_conditioned` | 136&ndash;160 | 141.4 &plusmn; 8.9 | 0.88 | 0.14 | 0.00 | 0.93 | 0.029 |

Arms delivering over the cap after truncation: **0** (must be zero; truncation is unconditional).

Paired context-clustered length differences between arms:

| comparison | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| conditioned &minus; generic | -0.24 [-0.74, +0.26] | -0.47 [-1.23, +0.30] | +1.32 [-0.88, +3.43] | +1.70 [-0.52, +3.84] |
| conditioned &minus; reusable | -0.09 [-0.35, +0.17] | +0.32 [-0.38, +1.10] | +0.01 [-1.36, +1.25] | -0.45 [-1.55, +0.69] |
| reusable &minus; generic | -0.15 [-0.59, +0.31] | -0.79 [-1.74, +0.18] | +1.31 [-0.38, +3.04] | **+2.15 [+0.06, +4.09]** |
| oracle &minus; conditioned | -0.34 [-1.25, +0.46] | **+1.26 [+0.51, +1.99]** | +0.59 [-0.66, +1.78] | +0.26 [-1.88, +2.45] |
| oracle &minus; generic | -0.58 [-1.62, +0.33] | +0.79 [-0.25, +1.83] | +1.92 [-0.42, +4.25] | +1.96 [-0.08, +4.00] |
| extractive_conditioned &minus; extractive_generic | -0.32 [-0.73, +0.11] | -0.59 [-1.91, +0.64] | **-2.41 [-4.48, -0.68]** | -2.32 [-4.80, +0.09] |

### Relation dossiers &mdash; delivered length, every arm and budget

| policy | band | delivered (mean &plusmn; sd) | fill | under floor | truncated | corrections | repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| `generic` | 17&ndash;20 | 18.8 &plusmn; 1.0 | 0.94 | 0.00 | 0.00 | 0.44 | 0.000 |
| `conditioned` | 17&ndash;20 | 18.3 &plusmn; 1.0 | 0.91 | 0.00 | 0.00 | 0.45 | 0.000 |
| `reusable` | 17&ndash;20 | 18.3 &plusmn; 1.0 | 0.92 | 0.00 | 0.02 | 0.52 | 0.000 |
| `oracle` | 17&ndash;20 | 18.1 &plusmn; 1.1 | 0.90 | 0.00 | 0.00 | 0.81 | 0.000 |
| `generic` | 34&ndash;40 | 37.1 &plusmn; 1.9 | 0.93 | 0.00 | 0.00 | 0.75 | 0.000 |
| `conditioned` | 34&ndash;40 | 36.5 &plusmn; 2.0 | 0.91 | 0.00 | 0.03 | 0.67 | 0.000 |
| `reusable` | 34&ndash;40 | 36.8 &plusmn; 1.9 | 0.92 | 0.00 | 0.00 | 0.38 | 0.000 |
| `oracle` | 34&ndash;40 | 36.8 &plusmn; 1.9 | 0.92 | 0.00 | 0.00 | 1.00 | 0.000 |
| `generic` | 68&ndash;80 | 73.2 &plusmn; 3.5 | 0.92 | 0.00 | 0.00 | 0.38 | 0.000 |
| `conditioned` | 68&ndash;80 | 72.9 &plusmn; 3.3 | 0.91 | 0.00 | 0.00 | 0.67 | 0.000 |
| `reusable` | 68&ndash;80 | 72.9 &plusmn; 2.9 | 0.91 | 0.00 | 0.00 | 0.55 | 0.000 |
| `oracle` | 68&ndash;80 | 75.6 &plusmn; 2.5 | 0.95 | 0.00 | 0.00 | 0.81 | 0.000 |
| `generic` | 136&ndash;160 | 143.6 &plusmn; 4.3 | 0.90 | 0.00 | 0.00 | 0.62 | 0.000 |
| `conditioned` | 136&ndash;160 | 143.9 &plusmn; 5.6 | 0.90 | 0.00 | 0.00 | 0.88 | 0.002 |
| `reusable` | 136&ndash;160 | 143.3 &plusmn; 4.8 | 0.90 | 0.00 | 0.00 | 0.78 | 0.001 |
| `oracle` | 136&ndash;160 | 143.7 &plusmn; 6.8 | 0.90 | 0.00 | 0.00 | 0.38 | 0.001 |

Arms delivering over the cap after truncation: **0** (must be zero; truncation is unconditional).

Paired context-clustered length differences between arms:

| comparison | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| conditioned &minus; generic | -0.47 [-1.00, +0.08] | -0.66 [-1.47, +0.14] | -0.38 [-2.16, +1.38] | +0.33 [-2.34, +3.05] |
| conditioned &minus; reusable | -0.06 [-0.38, +0.23] | -0.33 [-0.72, +0.03] | -0.06 [-1.12, +1.02] | +0.61 [-1.23, +2.28] |
| reusable &minus; generic | -0.41 [-0.94, +0.14] | -0.33 [-1.22, +0.55] | -0.31 [-1.95, +1.34] | -0.28 [-3.06, +2.53] |
| oracle &minus; conditioned | -0.22 [-0.77, +0.33] | +0.28 [-0.73, +1.34] | **+2.75 [+1.28, +4.30]** | -0.20 [-3.81, +3.69] |
| oracle &minus; generic | **-0.69 [-1.25, -0.12]** | -0.38 [-1.62, +0.88] | **+2.38 [+0.88, +3.75]** | +0.12 [-4.00, +4.31] |

### Length-matched subsample

Headline contrasts restricted to contexts where `generic`, `conditioned` and `reusable` all delivered within the configured tolerance of each other. This is the comparison that survives even if the audit above had shown unequal fill.

**SQuAD groups**

| comparison | endpoint | 20 w | 40 w | 80 w | 160 w | n contexts (low&ndash;high budget) |
|---|---:|---:|---:|---:|---:|---:|
| conditioned &minus; generic | dU_now | **+0.625 [+0.552, +0.698]** | **+0.446 [+0.348, +0.543]** | **+0.200 [+0.067, +0.333]** | **+0.173 [+0.038, +0.308]** | 24/23/15/13 |
| conditioned &minus; generic | dU_future | **-0.063 [-0.122, -0.000]** | **-0.105 [-0.192, -0.014]** | **-0.183 [-0.289, -0.078]** | **-0.244 [-0.423, -0.071]** | 24/23/15/13 |
| conditioned &minus; reusable | dU_now | **+0.073 [+0.021, +0.125]** | +0.022 [-0.076, +0.120] | -0.067 [-0.150, +0.000] | -0.019 [-0.077, +0.038] | 24/23/15/13 |
| conditioned &minus; reusable | dU_future | +0.003 [-0.031, +0.042] | -0.007 [-0.054, +0.040] | **-0.072 [-0.128, -0.006]** | **-0.128 [-0.250, -0.006]** | 24/23/15/13 |
| reusable &minus; generic | dU_now | **+0.552 [+0.468, +0.646]** | **+0.424 [+0.304, +0.543]** | **+0.267 [+0.117, +0.400]** | **+0.192 [+0.096, +0.308]** | 24/23/15/13 |
| reusable &minus; generic | dU_future | **-0.066 [-0.128, -0.003]** | **-0.098 [-0.174, -0.025]** | **-0.111 [-0.206, -0.017]** | -0.115 [-0.231, +0.006] | 24/23/15/13 |

**Relation dossiers**

| comparison | endpoint | 20 w | 40 w | 80 w | 160 w | n contexts (low&ndash;high budget) |
|---|---:|---:|---:|---:|---:|---:|
| conditioned &minus; generic | dU_now | **+0.859 [+0.797, +0.922]** | **+0.641 [+0.547, +0.719]** | **+0.500 [+0.383, +0.600]** | **+0.344 [+0.281, +0.438]** | 16/16/15/8 |
| conditioned &minus; generic | dU_future | -0.026 [-0.051, +0.002] | **-0.110 [-0.153, -0.072]** | **-0.210 [-0.268, -0.148]** | **-0.335 [-0.390, -0.287]** | 16/16/15/8 |
| conditioned &minus; reusable | dU_now | +0.000 [+0.000, +0.000] | -0.016 [-0.047, +0.000] | -0.017 [-0.050, +0.000] | +0.031 [+0.000, +0.094] | 16/16/15/8 |
| conditioned &minus; reusable | dU_future | -0.004 [-0.011, +0.004] | -0.007 [-0.021, +0.006] | **-0.082 [-0.100, -0.064]** | **-0.167 [-0.198, -0.131]** | 16/16/15/8 |
| reusable &minus; generic | dU_now | **+0.859 [+0.797, +0.922]** | **+0.656 [+0.562, +0.734]** | **+0.517 [+0.433, +0.600]** | **+0.312 [+0.188, +0.438]** | 16/16/15/8 |
| reusable &minus; generic | dU_future | -0.022 [-0.045, +0.003] | **-0.103 [-0.143, -0.066]** | **-0.128 [-0.178, -0.076]** | **-0.169 [-0.248, -0.098]** | 16/16/15/8 |

### Trimmed to one length per context

The strictest control available: every abstractive message in a (context, budget) cell re-truncated to that cell's shortest delivered length, then re-answered. Length is identical within a context by construction.

| policy | delivered words | U_now | U_future |
|---|---:|---:|---:|
| `conditioned@trim` | 30.6 | 0.823 [0.750, 0.896] <br><sub>untrimmed 0.823 [0.750, 0.896]</sub> | 0.198 [0.132, 0.267] <br><sub>untrimmed 0.260 [0.188, 0.337]</sub> |
| `generic@trim` | 24.8 | 0.323 [0.250, 0.396] <br><sub>untrimmed 0.375 [0.302, 0.448]</sub> | 0.323 [0.250, 0.396] <br><sub>untrimmed 0.375 [0.302, 0.448]</sub> |
| `oracle@trim` | 24.3 | 0.594 [0.510, 0.667] <br><sub>untrimmed 0.823 [0.740, 0.896]</sub> | 0.594 [0.510, 0.667] <br><sub>untrimmed 0.823 [0.740, 0.896]</sub> |
| `reusable@trim` | 27.4 | 0.781 [0.698, 0.854] <br><sub>untrimmed 0.802 [0.729, 0.875]</sub> | 0.194 [0.132, 0.260] <br><sub>untrimmed 0.271 [0.205, 0.337]</sub> |

Two further controls sit underneath everything above. Every question in every context passed a closed-book leakage filter against the answering model, so an off-diagonal success cannot be parametric recall. And a different model family audited each group: all questions must be answerable from the passage and must ask about genuinely distinct facts.

| corpus | baselines |
|---|---|
| SQuAD groups | direct-context ceiling `U(D, q)` = 0.875 [0.812, 0.938]; closed-book leakage audit = 0.031 [0.010, 0.059] |
| Relation dossiers | direct-context ceiling `U(D, q)` = 0.992 [0.977, 1.000]; closed-book leakage audit = 0.000 [0.000, 0.000] |

## 4. The result

![Present versus future utility](squad_groups/n24/pareto_now_vs_future.png)

Marker size is the budget. The three unconditioned arms sit **on** the diagonal by construction &mdash; one message serves every rotation, so they cannot specialise. Every conditioned arm sits far below it, at every budget.

### SQuAD groups &mdash; utility

**Present query `U_now`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.240 [0.188, 0.292] | 0.375 [0.302, 0.448] | 0.531 [0.427, 0.635] | 0.698 [0.604, 0.792] |
| `conditioned` | 0.865 [0.802, 0.917] | 0.823 [0.750, 0.896] | 0.812 [0.750, 0.875] | 0.802 [0.708, 0.885] |
| `reusable` | 0.792 [0.719, 0.865] | 0.802 [0.729, 0.875] | 0.854 [0.792, 0.917] | 0.906 [0.844, 0.958] |
| `oracle` | 0.490 [0.417, 0.562] | 0.823 [0.740, 0.896] | 0.885 [0.823, 0.948] | 0.896 [0.833, 0.948] |
| `extractive_generic` | 0.354 [0.250, 0.458] | 0.427 [0.344, 0.510] | 0.615 [0.521, 0.708] | 0.854 [0.771, 0.927] |
| `extractive_conditioned` | 0.740 [0.677, 0.802] | 0.792 [0.708, 0.865] | 0.906 [0.844, 0.958] | 0.865 [0.802, 0.917] |
| `conditioned@trim` | n/a | 0.823 [0.750, 0.896] | n/a | n/a |
| `generic@trim` | n/a | 0.323 [0.250, 0.396] | n/a | n/a |
| `oracle@trim` | n/a | 0.594 [0.510, 0.667] | n/a | n/a |
| `reusable@trim` | n/a | 0.781 [0.698, 0.854] | n/a | n/a |

**Future queries `U_future`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.240 [0.188, 0.292] | 0.375 [0.302, 0.448] | 0.531 [0.427, 0.635] | 0.698 [0.604, 0.792] |
| `conditioned` | 0.177 [0.122, 0.236] | 0.260 [0.188, 0.337] | 0.337 [0.260, 0.413] | 0.528 [0.438, 0.618] |
| `reusable` | 0.174 [0.122, 0.229] | 0.271 [0.205, 0.337] | 0.424 [0.354, 0.497] | 0.601 [0.524, 0.677] |
| `oracle` | 0.490 [0.417, 0.562] | 0.823 [0.740, 0.896] | 0.885 [0.823, 0.948] | 0.896 [0.833, 0.948] |
| `extractive_generic` | 0.354 [0.250, 0.458] | 0.427 [0.344, 0.510] | 0.615 [0.521, 0.708] | 0.854 [0.771, 0.927] |
| `extractive_conditioned` | 0.156 [0.111, 0.212] | 0.285 [0.219, 0.351] | 0.458 [0.385, 0.531] | 0.771 [0.701, 0.840] |
| `conditioned@trim` | n/a | 0.198 [0.132, 0.267] | n/a | n/a |
| `generic@trim` | n/a | 0.323 [0.250, 0.396] | n/a | n/a |
| `oracle@trim` | n/a | 0.594 [0.510, 0.667] | n/a | n/a |
| `reusable@trim` | n/a | 0.194 [0.132, 0.260] | n/a | n/a |

**Specialisation gap `U_now - U_future`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| `conditioned` | 0.688 [0.601, 0.771] | 0.562 [0.451, 0.670] | 0.476 [0.372, 0.580] | 0.274 [0.153, 0.396] |
| `reusable` | 0.618 [0.521, 0.712] | 0.531 [0.406, 0.656] | 0.431 [0.326, 0.535] | 0.306 [0.215, 0.396] |
| `oracle` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| `extractive_generic` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| `extractive_conditioned` | 0.583 [0.500, 0.667] | 0.507 [0.389, 0.625] | 0.448 [0.354, 0.545] | 0.094 [0.028, 0.163] |
| `conditioned@trim` | n/a | 0.625 [0.535, 0.715] | n/a | n/a |
| `generic@trim` | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| `oracle@trim` | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| `reusable@trim` | n/a | 0.587 [0.469, 0.698] | n/a | n/a |

Baselines: direct-context ceiling `U(D, q)` = 0.875 [0.812, 0.938]; closed-book leakage audit = 0.031 [0.010, 0.059].

### SQuAD groups &mdash; paired contrasts

Paired bootstrap over contexts. **Bold** excludes zero.

**On the present query, dU_now**

| comparison | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| conditioned &minus; generic | **+0.625 [+0.552, +0.698]** | **+0.448 [+0.354, +0.542]** | **+0.281 [+0.167, +0.396]** | +0.104 [+0.000, +0.208] |
| conditioned &minus; reusable | **+0.073 [+0.021, +0.135]** | +0.021 [-0.073, +0.115] | -0.042 [-0.094, +0.000] | **-0.104 [-0.177, -0.031]** |
| reusable &minus; generic | **+0.552 [+0.458, +0.646]** | **+0.427 [+0.312, +0.542]** | **+0.323 [+0.208, +0.438]** | **+0.208 [+0.104, +0.312]** |
| oracle &minus; conditioned | **-0.375 [-0.438, -0.323]** | +0.000 [-0.104, +0.094] | **+0.073 [+0.021, +0.125]** | **+0.094 [+0.010, +0.177]** |
| oracle &minus; generic | **+0.250 [+0.167, +0.344]** | **+0.448 [+0.323, +0.562]** | **+0.354 [+0.260, +0.458]** | **+0.198 [+0.115, +0.292]** |
| extractive_conditioned &minus; extractive_generic | **+0.385 [+0.260, +0.510]** | **+0.365 [+0.240, +0.479]** | **+0.292 [+0.188, +0.396]** | +0.010 [-0.062, +0.083] |

**On the hidden future queries, dU_future**

| comparison | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| conditioned &minus; generic | **-0.063 [-0.122, -0.003]** | **-0.115 [-0.201, -0.028]** | **-0.194 [-0.274, -0.118]** | **-0.170 [-0.292, -0.052]** |
| conditioned &minus; reusable | +0.003 [-0.031, +0.042] | -0.010 [-0.056, +0.035] | **-0.087 [-0.132, -0.038]** | -0.073 [-0.153, +0.010] |
| reusable &minus; generic | **-0.066 [-0.128, -0.003]** | **-0.104 [-0.177, -0.035]** | **-0.108 [-0.184, -0.031]** | **-0.097 [-0.187, -0.003]** |
| oracle &minus; conditioned | **+0.312 [+0.229, +0.396]** | **+0.562 [+0.444, +0.681]** | **+0.549 [+0.469, +0.632]** | **+0.368 [+0.257, +0.483]** |
| oracle &minus; generic | **+0.250 [+0.167, +0.344]** | **+0.448 [+0.323, +0.562]** | **+0.354 [+0.260, +0.458]** | **+0.198 [+0.115, +0.292]** |
| extractive_conditioned &minus; extractive_generic | **-0.198 [-0.292, -0.108]** | **-0.142 [-0.215, -0.066]** | **-0.156 [-0.253, -0.052]** | **-0.083 [-0.146, -0.024]** |

**Secondary metrics** &mdash; the same `conditioned` &minus; `generic` contrast under all three utility measures. EM and F1 are always computed and never replaced by the judge.

| metric | endpoint | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|
| LLM judge | dU_now | **+0.625 [+0.552, +0.698]** | **+0.448 [+0.354, +0.542]** | **+0.281 [+0.167, +0.396]** | +0.104 [+0.000, +0.208] |
| LLM judge | dU_future | **-0.063 [-0.122, -0.003]** | **-0.115 [-0.201, -0.028]** | **-0.194 [-0.274, -0.118]** | **-0.170 [-0.292, -0.052]** |
| Exact match | dU_now | **+0.417 [+0.292, +0.542]** | **+0.323 [+0.219, +0.427]** | **+0.271 [+0.167, +0.375]** | **+0.188 [+0.083, +0.302]** |
| Exact match | dU_future | -0.056 [-0.122, +0.014] | **-0.132 [-0.208, -0.056]** | **-0.118 [-0.188, -0.045]** | **-0.122 [-0.233, -0.017]** |
| Token F1 | dU_now | **+0.541 [+0.450, +0.633]** | **+0.413 [+0.319, +0.504]** | **+0.291 [+0.196, +0.388]** | **+0.187 [+0.088, +0.288]** |
| Token F1 | dU_future | -0.043 [-0.102, +0.020] | **-0.118 [-0.189, -0.049]** | **-0.159 [-0.229, -0.088]** | **-0.140 [-0.248, -0.038]** |

**Does scarcity sharpen the trade-off?** Positive means the specialisation gap is wider at the tightest budget than at the widest.

| quantity | policy | gap(low) &minus; gap(high) | p | dz |
|---|---:|---:|---:|---:|
| gap_low_minus_high | `conditioned` | **+0.413 [+0.323, +0.503]** | 0.0000 | 1.84 |
| gap_low_minus_high | `extractive_conditioned` | **+0.490 [+0.396, +0.587]** | 0.0000 | 1.99 |
| gap_low_minus_high | `extractive_generic` | **+0.000 [+0.000, +0.000]** | 0.0266 | 0.46 |
| gap_low_minus_high | `generic` | -0.000 [-0.000, +0.000] | 0.5071 | -0.14 |
| gap_low_minus_high | `oracle` | **+0.000 [+0.000, +0.000]** | 0.0005 | 0.75 |
| gap_low_minus_high | `reusable` | **+0.312 [+0.191, +0.431]** | 0.0000 | 1.03 |
| conditioning_gap_low_minus_high | `conditioned_minus_generic` | **+0.413 [+0.326, +0.503]** | 0.0000 | 1.84 |

### Relation dossiers &mdash; utility

**Present query `U_now`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.141 [0.078, 0.203] | 0.344 [0.266, 0.438] | 0.469 [0.391, 0.562] | 0.609 [0.516, 0.703] |
| `conditioned` | 1.000 [1.000, 1.000] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.969 [0.922, 1.000] |
| `reusable` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.953 [0.906, 1.000] |
| `oracle` | 0.250 [0.250, 0.250] | 0.516 [0.438, 0.594] | 0.750 [0.672, 0.828] | 0.891 [0.812, 0.953] |

**Future queries `U_future`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.103 [0.077, 0.126] | 0.206 [0.168, 0.248] | 0.315 [0.259, 0.368] | 0.509 [0.452, 0.569] |
| `conditioned` | 0.077 [0.070, 0.085] | 0.096 [0.085, 0.107] | 0.117 [0.103, 0.131] | 0.168 [0.149, 0.186] |
| `reusable` | 0.081 [0.076, 0.086] | 0.103 [0.093, 0.114] | 0.197 [0.184, 0.209] | 0.334 [0.302, 0.364] |
| `oracle` | 0.179 [0.163, 0.196] | 0.399 [0.363, 0.434] | 0.692 [0.625, 0.751] | 0.828 [0.755, 0.898] |

**Specialisation gap `U_now - U_future`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.038 [-0.012, 0.083] | 0.138 [0.083, 0.196] | 0.154 [0.104, 0.213] | 0.100 [0.029, 0.171] |
| `conditioned` | 0.923 [0.915, 0.930] | 0.889 [0.849, 0.914] | 0.868 [0.831, 0.893] | 0.801 [0.751, 0.840] |
| `reusable` | 0.919 [0.914, 0.924] | 0.897 [0.886, 0.907] | 0.803 [0.791, 0.816] | 0.619 [0.551, 0.678] |
| `oracle` | 0.071 [0.054, 0.087] | 0.117 [0.054, 0.175] | 0.058 [0.008, 0.104] | 0.062 [0.025, 0.100] |

Baselines: direct-context ceiling `U(D, q)` = 0.992 [0.977, 1.000]; closed-book leakage audit = 0.000 [0.000, 0.000].

### Relation dossiers &mdash; paired contrasts

Paired bootstrap over contexts. **Bold** excludes zero.

**On the present query, dU_now**

| comparison | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| conditioned &minus; generic | **+0.859 [+0.797, +0.922]** | **+0.641 [+0.547, +0.719]** | **+0.516 [+0.406, +0.609]** | **+0.359 [+0.250, +0.469]** |
| conditioned &minus; reusable | +0.000 [+0.000, +0.000] | -0.016 [-0.047, +0.000] | -0.016 [-0.047, +0.000] | +0.016 [-0.031, +0.062] |
| reusable &minus; generic | **+0.859 [+0.797, +0.922]** | **+0.656 [+0.562, +0.734]** | **+0.531 [+0.453, +0.609]** | **+0.344 [+0.234, +0.453]** |
| oracle &minus; conditioned | **-0.750 [-0.750, -0.750]** | **-0.469 [-0.547, -0.391]** | **-0.234 [-0.328, -0.141]** | -0.078 [-0.172, +0.000] |
| oracle &minus; generic | **+0.109 [+0.047, +0.172]** | **+0.172 [+0.047, +0.297]** | **+0.281 [+0.188, +0.375]** | **+0.281 [+0.141, +0.422]** |

**On the hidden future queries, dU_future**

| comparison | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| conditioned &minus; generic | -0.026 [-0.051, +0.001] | **-0.110 [-0.152, -0.070]** | **-0.198 [-0.257, -0.136]** | **-0.342 [-0.396, -0.292]** |
| conditioned &minus; reusable | -0.004 [-0.011, +0.004] | -0.007 [-0.021, +0.006] | **-0.080 [-0.098, -0.064]** | **-0.167 [-0.190, -0.143]** |
| reusable &minus; generic | -0.022 [-0.045, +0.002] | **-0.103 [-0.141, -0.066]** | **-0.118 [-0.168, -0.065]** | **-0.175 [-0.241, -0.115]** |
| oracle &minus; conditioned | **+0.102 [+0.084, +0.121]** | **+0.303 [+0.263, +0.344]** | **+0.575 [+0.506, +0.635]** | **+0.660 [+0.583, +0.731]** |
| oracle &minus; generic | **+0.076 [+0.044, +0.112]** | **+0.193 [+0.134, +0.252]** | **+0.377 [+0.296, +0.446]** | **+0.319 [+0.208, +0.422]** |

**Secondary metrics** &mdash; the same `conditioned` &minus; `generic` contrast under all three utility measures. EM and F1 are always computed and never replaced by the judge.

| metric | endpoint | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|
| LLM judge | dU_now | **+0.859 [+0.797, +0.922]** | **+0.641 [+0.547, +0.719]** | **+0.516 [+0.406, +0.609]** | **+0.359 [+0.250, +0.469]** |
| LLM judge | dU_future | -0.026 [-0.051, +0.001] | **-0.110 [-0.152, -0.070]** | **-0.198 [-0.257, -0.136]** | **-0.342 [-0.396, -0.292]** |
| Exact match | dU_now | **+0.703 [+0.609, +0.781]** | **+0.531 [+0.438, +0.609]** | **+0.516 [+0.406, +0.625]** | **+0.344 [+0.250, +0.438]** |
| Exact match | dU_future | -0.012 [-0.039, +0.014] | **-0.074 [-0.111, -0.042]** | **-0.081 [-0.142, -0.023]** | **-0.255 [-0.326, -0.193]** |
| Token F1 | dU_now | **+0.732 [+0.679, +0.783]** | **+0.567 [+0.492, +0.637]** | **+0.498 [+0.386, +0.605]** | **+0.308 [+0.222, +0.398]** |
| Token F1 | dU_future | **-0.040 [-0.073, -0.007]** | **-0.109 [-0.148, -0.074]** | **-0.158 [-0.209, -0.107]** | **-0.322 [-0.379, -0.271]** |

**Does scarcity sharpen the trade-off?** Positive means the specialisation gap is wider at the tightest budget than at the widest.

| quantity | policy | gap(low) &minus; gap(high) | p | dz |
|---|---:|---:|---:|---:|
| gap_low_minus_high | `conditioned` | **+0.122 [+0.080, +0.170]** | 0.0000 | 1.27 |
| gap_low_minus_high | `generic` | -0.062 [-0.138, +0.017] | 0.1336 | -0.37 |
| gap_low_minus_high | `oracle` | +0.008 [-0.025, +0.042] | 0.7164 | 0.11 |
| gap_low_minus_high | `reusable` | **+0.300 [+0.240, +0.367]** | 0.0000 | 2.20 |
| conditioning_gap_low_minus_high | `conditioned_minus_generic` | **+0.184 [+0.080, +0.285]** | 0.0002 | 0.85 |

![Utility against communication budget](squad_groups/n24/utility_vs_budget.png)

> **Where the effect is bounded.** The absolute future loss is compressed exactly where the gap is largest: at the tightest budget the generic baseline itself reaches only 0.240 on SQuAD, so there is very little left for conditioning to take away. The gap statistic, which is scale-free, is the honest measure of the scarcity prediction; the raw dU_future is not.

## 5. Communication regret against the source

`R(q) = U(D, q) - U(m, q)`: the accuracy a handoff costs relative to reading the source directly.

![Communication regret](squad_groups/n24/communication_regret.png)

### SQuAD groups &mdash; regret

**On the present query, `R_now`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.635 [0.552, 0.719] | 0.500 [0.396, 0.604] | 0.344 [0.229, 0.458] | 0.177 [0.083, 0.271] |
| `conditioned` | 0.010 [-0.052, 0.073] | 0.052 [-0.031, 0.135] | 0.062 [0.010, 0.115] | 0.073 [0.010, 0.135] |
| `reusable` | 0.083 [0.010, 0.167] | 0.073 [0.010, 0.135] | 0.021 [-0.052, 0.083] | -0.031 [-0.104, 0.031] |
| `oracle` | 0.385 [0.312, 0.458] | 0.052 [-0.031, 0.146] | -0.010 [-0.083, 0.062] | -0.021 [-0.094, 0.052] |
| `extractive_generic` | 0.521 [0.375, 0.656] | 0.448 [0.344, 0.552] | 0.260 [0.135, 0.375] | 0.021 [-0.052, 0.094] |
| `extractive_conditioned` | 0.135 [0.052, 0.208] | 0.083 [0.021, 0.146] | -0.031 [-0.073, 0.010] | 0.010 [-0.031, 0.052] |
| `conditioned@trim` | n/a | 0.052 [-0.042, 0.135] | n/a | n/a |
| `generic@trim` | n/a | 0.552 [0.448, 0.646] | n/a | n/a |
| `oracle@trim` | n/a | 0.281 [0.177, 0.396] | n/a | n/a |
| `reusable@trim` | n/a | 0.094 [0.021, 0.167] | n/a | n/a |

**On the hidden future queries, `R_future`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.635 [0.552, 0.719] | 0.500 [0.396, 0.604] | 0.344 [0.229, 0.458] | 0.177 [0.083, 0.271] |
| `conditioned` | 0.698 [0.594, 0.795] | 0.615 [0.507, 0.719] | 0.538 [0.431, 0.642] | 0.347 [0.229, 0.465] |
| `reusable` | 0.701 [0.608, 0.788] | 0.604 [0.500, 0.705] | 0.451 [0.354, 0.549] | 0.274 [0.170, 0.372] |
| `oracle` | 0.385 [0.312, 0.458] | 0.052 [-0.031, 0.146] | -0.010 [-0.083, 0.062] | -0.021 [-0.094, 0.052] |
| `extractive_generic` | 0.521 [0.375, 0.656] | 0.448 [0.344, 0.552] | 0.260 [0.135, 0.375] | 0.021 [-0.052, 0.094] |
| `extractive_conditioned` | 0.719 [0.625, 0.806] | 0.590 [0.493, 0.681] | 0.417 [0.309, 0.521] | 0.104 [0.038, 0.174] |
| `conditioned@trim` | n/a | 0.677 [0.576, 0.774] | n/a | n/a |
| `generic@trim` | n/a | 0.552 [0.448, 0.646] | n/a | n/a |
| `oracle@trim` | n/a | 0.281 [0.177, 0.396] | n/a | n/a |
| `reusable@trim` | n/a | 0.681 [0.580, 0.774] | n/a | n/a |

**Retained utility on cells the source itself answers.** Restricted to cells where `U(D, q) = 1`, so retained utility is just `U(m, q)` and no near-zero denominator has to be guarded.

*Present query*

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.226 [0.149, 0.302] | 0.403 [0.302, 0.504] | 0.559 [0.451, 0.663] | 0.767 [0.663, 0.861] |
| `conditioned` | 0.931 [0.878, 0.976] | 0.885 [0.819, 0.944] | 0.910 [0.858, 0.958] | 0.851 [0.764, 0.934] |
| `reusable` | 0.861 [0.785, 0.931] | 0.875 [0.812, 0.934] | 0.920 [0.868, 0.969] | 0.965 [0.924, 1.000] |
| `oracle` | 0.486 [0.403, 0.566] | 0.861 [0.771, 0.941] | 0.931 [0.878, 0.976] | 0.944 [0.899, 0.986] |
| `extractive_generic` | 0.358 [0.250, 0.472] | 0.417 [0.326, 0.510] | 0.632 [0.545, 0.722] | 0.931 [0.861, 0.979] |
| `extractive_conditioned` | 0.792 [0.719, 0.858] | 0.861 [0.795, 0.920] | 0.972 [0.931, 1.000] | 0.955 [0.910, 0.990] |
| `conditioned@trim` | n/a | 0.885 [0.819, 0.944] | n/a | n/a |
| `generic@trim` | n/a | 0.340 [0.240, 0.451] | n/a | n/a |
| `oracle@trim` | n/a | 0.618 [0.517, 0.715] | n/a | n/a |
| `reusable@trim` | n/a | 0.861 [0.795, 0.924] | n/a | n/a |

*Future queries*

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.226 [0.149, 0.302] | 0.403 [0.302, 0.504] | 0.559 [0.451, 0.663] | 0.767 [0.663, 0.861] |
| `conditioned` | 0.174 [0.116, 0.233] | 0.276 [0.193, 0.368] | 0.340 [0.259, 0.429] | 0.556 [0.457, 0.655] |
| `reusable` | 0.168 [0.113, 0.229] | 0.278 [0.201, 0.354] | 0.455 [0.375, 0.538] | 0.660 [0.578, 0.740] |
| `oracle` | 0.486 [0.403, 0.566] | 0.861 [0.771, 0.941] | 0.931 [0.878, 0.976] | 0.944 [0.899, 0.986] |
| `extractive_generic` | 0.358 [0.250, 0.472] | 0.417 [0.326, 0.510] | 0.632 [0.545, 0.722] | 0.931 [0.861, 0.979] |
| `extractive_conditioned` | 0.151 [0.104, 0.201] | 0.280 [0.210, 0.352] | 0.474 [0.378, 0.568] | 0.849 [0.788, 0.906] |
| `conditioned@trim` | n/a | 0.201 [0.125, 0.288] | n/a | n/a |
| `generic@trim` | n/a | 0.340 [0.240, 0.451] | n/a | n/a |
| `oracle@trim` | n/a | 0.618 [0.517, 0.715] | n/a | n/a |
| `reusable@trim` | n/a | 0.194 [0.123, 0.273] | n/a | n/a |

**Normalised ratio `mean U(m,q) / mean U(D,q)`.** Contexts whose own ceiling falls below the configured floor are excluded, because dividing by a near-zero ceiling turns noise into a number; the exclusion count is shown rather than hidden.

| policy | endpoint | 20 w | 40 w | 80 w | 160 w | excluded (low ceiling) |
|---|---:|---:|---:|---:|---:|---:|
| `generic` | now | 0.288 [0.222, 0.354] | 0.451 [0.354, 0.549] | 0.622 [0.500, 0.747] | 0.816 [0.694, 0.934] | 0/0/0/0 |
| `generic` | future | 0.288 [0.222, 0.354] | 0.451 [0.354, 0.549] | 0.622 [0.500, 0.747] | 0.816 [0.694, 0.934] | 0/0/0/0 |
| `conditioned` | now | 1.014 [0.934, 1.101] | 0.976 [0.858, 1.111] | 0.938 [0.882, 0.997] | 0.913 [0.837, 0.986] | 0/0/0/0 |
| `conditioned` | future | 0.227 [0.142, 0.321] | 0.319 [0.226, 0.421] | 0.410 [0.314, 0.516] | 0.642 [0.515, 0.784] | 0/0/0/0 |
| `reusable` | now | 0.924 [0.837, 1.014] | 0.924 [0.854, 0.997] | 1.003 [0.913, 1.122] | 1.069 [0.976, 1.187] | 0/0/0/0 |
| `reusable` | future | 0.214 [0.144, 0.292] | 0.336 [0.244, 0.433] | 0.505 [0.418, 0.594] | 0.718 [0.606, 0.854] | 0/0/0/0 |
| `oracle` | now | 0.562 [0.483, 0.646] | 0.972 [0.861, 1.083] | 1.042 [0.944, 1.160] | 1.062 [0.958, 1.187] | 0/0/0/0 |
| `oracle` | future | 0.562 [0.483, 0.646] | 0.972 [0.861, 1.083] | 1.042 [0.944, 1.160] | 1.062 [0.958, 1.188] | 0/0/0/0 |
| `extractive_generic` | now | 0.448 [0.302, 0.608] | 0.514 [0.406, 0.625] | 0.743 [0.597, 0.910] | 1.000 [0.892, 1.122] | 0/0/0/0 |
| `extractive_generic` | future | 0.448 [0.302, 0.608] | 0.514 [0.406, 0.625] | 0.743 [0.597, 0.910] | 1.000 [0.892, 1.122] | 0/0/0/0 |
| `extractive_conditioned` | now | 0.868 [0.778, 0.969] | 0.910 [0.826, 1.000] | 1.056 [0.986, 1.132] | 1.003 [0.948, 1.066] | 0/0/0/0 |
| `extractive_conditioned` | future | 0.194 [0.131, 0.270] | 0.344 [0.256, 0.434] | 0.557 [0.450, 0.676] | 0.898 [0.814, 0.980] | 0/0/0/0 |
| `conditioned@trim` | now | n/a | 0.976 [0.861, 1.115] | n/a | n/a | -/0/-/- |
| `conditioned@trim` | future | n/a | 0.242 [0.160, 0.334] | n/a | n/a | -/0/-/- |
| `generic@trim` | now | n/a | 0.389 [0.292, 0.493] | n/a | n/a | -/0/-/- |
| `generic@trim` | future | n/a | 0.389 [0.292, 0.493] | n/a | n/a | -/0/-/- |
| `oracle@trim` | now | n/a | 0.715 [0.597, 0.837] | n/a | n/a | -/0/-/- |
| `oracle@trim` | future | n/a | 0.715 [0.597, 0.837] | n/a | n/a | -/0/-/- |
| `reusable@trim` | now | n/a | 0.910 [0.819, 1.007] | n/a | n/a | -/0/-/- |
| `reusable@trim` | future | n/a | 0.240 [0.159, 0.331] | n/a | n/a | -/0/-/- |

### Relation dossiers &mdash; regret

**On the present query, `R_now`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.844 [0.781, 0.906] | 0.641 [0.547, 0.734] | 0.516 [0.422, 0.609] | 0.375 [0.266, 0.484] |
| `conditioned` | -0.016 [-0.047, 0.000] | 0.000 [0.000, 0.000] | 0.000 [-0.047, 0.047] | 0.016 [-0.031, 0.062] |
| `reusable` | -0.016 [-0.047, 0.000] | -0.016 [-0.047, 0.000] | -0.016 [-0.047, 0.000] | 0.031 [0.000, 0.078] |
| `oracle` | 0.734 [0.703, 0.750] | 0.469 [0.391, 0.547] | 0.234 [0.156, 0.312] | 0.094 [0.031, 0.156] |

**On the hidden future queries, `R_future`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.890 [0.866, 0.917] | 0.786 [0.745, 0.827] | 0.678 [0.622, 0.736] | 0.483 [0.417, 0.546] |
| `conditioned` | 0.916 [0.899, 0.928] | 0.897 [0.875, 0.914] | 0.876 [0.853, 0.895] | 0.825 [0.798, 0.849] |
| `reusable` | 0.911 [0.897, 0.921] | 0.890 [0.869, 0.905] | 0.796 [0.774, 0.815] | 0.658 [0.619, 0.696] |
| `oracle` | 0.814 [0.790, 0.838] | 0.594 [0.561, 0.628] | 0.301 [0.235, 0.371] | 0.165 [0.101, 0.231] |

**Retained utility on cells the source itself answers.** Restricted to cells where `U(D, q) = 1`, so retained utility is just `U(m, q)` and no near-zero denominator has to be guarded.

*Present query*

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.141 [0.078, 0.203] | 0.349 [0.266, 0.438] | 0.479 [0.391, 0.568] | 0.604 [0.510, 0.698] |
| `conditioned` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.984 [0.953, 1.000] | 0.969 [0.922, 1.000] |
| `reusable` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.969 [0.922, 1.000] |
| `oracle` | 0.255 [0.250, 0.266] | 0.505 [0.422, 0.594] | 0.766 [0.688, 0.844] | 0.901 [0.839, 0.964] |

*Future queries*

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | 0.104 [0.078, 0.127] | 0.208 [0.169, 0.249] | 0.318 [0.261, 0.372] | 0.507 [0.452, 0.564] |
| `conditioned` | 0.077 [0.069, 0.085] | 0.094 [0.084, 0.106] | 0.117 [0.103, 0.131] | 0.167 [0.149, 0.185] |
| `reusable` | 0.081 [0.075, 0.086] | 0.102 [0.092, 0.112] | 0.196 [0.184, 0.209] | 0.333 [0.301, 0.361] |
| `oracle` | 0.181 [0.163, 0.199] | 0.398 [0.360, 0.433] | 0.698 [0.628, 0.763] | 0.828 [0.755, 0.898] |

**Normalised ratio `mean U(m,q) / mean U(D,q)`.** Contexts whose own ceiling falls below the configured floor are excluded, because dividing by a near-zero ceiling turns noise into a number; the exclusion count is shown rather than hidden.

| policy | endpoint | 20 w | 40 w | 80 w | 160 w | excluded (low ceiling) |
|---|---:|---:|---:|---:|---:|---:|
| `generic` | now | 0.141 [0.078, 0.203] | 0.349 [0.266, 0.438] | 0.479 [0.391, 0.573] | 0.625 [0.516, 0.734] | 0/0/0/0 |
| `generic` | future | 0.104 [0.078, 0.127] | 0.208 [0.169, 0.247] | 0.318 [0.261, 0.374] | 0.515 [0.455, 0.583] | 0/0/0/0 |
| `conditioned` | now | 1.021 [1.000, 1.062] | 1.000 [1.000, 1.000] | 1.005 [0.953, 1.062] | 0.990 [0.938, 1.047] | 0/0/0/0 |
| `conditioned` | future | 0.078 [0.071, 0.085] | 0.097 [0.085, 0.109] | 0.118 [0.104, 0.132] | 0.169 [0.150, 0.189] | 0/0/0/0 |
| `reusable` | now | 1.021 [1.000, 1.062] | 1.021 [1.000, 1.062] | 1.021 [1.000, 1.062] | 0.969 [0.922, 1.000] | 0/0/0/0 |
| `reusable` | future | 0.082 [0.077, 0.086] | 0.104 [0.094, 0.115] | 0.199 [0.185, 0.213] | 0.338 [0.303, 0.371] | 0/0/0/0 |
| `oracle` | now | 0.255 [0.250, 0.266] | 0.526 [0.443, 0.609] | 0.766 [0.688, 0.844] | 0.901 [0.833, 0.964] | 0/0/0/0 |
| `oracle` | future | 0.181 [0.163, 0.199] | 0.401 [0.367, 0.435] | 0.698 [0.628, 0.762] | 0.833 [0.765, 0.902] | 0/0/0/0 |

## 6. Pareto frontier

Each (policy, budget) is one point in (`U_now`, `U_future`, cost), where cost is mean delivered words &mdash; what the channel actually carried, not what was requested. A configuration is dominated when another is at least as good on both utilities and no more expensive, with at least one strict improvement. Trimmed arms are excluded: their length is set by looking at what the other arms wrote, so they are a length control rather than a deployable policy.

### SQuAD groups

| policy | requested cap | U_now | U_future | delivered cost (words) | 3D nondominated | within-cap 2D nondominated |
|---|---:|---:|---:|---:|---:|---:|
| `generic` | 20 w | 0.240 | 0.240 | 18.7 | &mdash; | &mdash; |
| `generic` | 40 w | 0.375 | 0.375 | 37.1 | &mdash; | &mdash; |
| `generic` | 80 w | 0.531 | 0.531 | 71.9 | &mdash; | &mdash; |
| `generic` | 160 w | 0.698 | 0.698 | 141.4 | &mdash; | &mdash; |
| `conditioned` | 20 w | 0.865 | 0.177 | 18.5 | **yes** | **yes** |
| `conditioned` | 40 w | 0.823 | 0.260 | 36.7 | **yes** | &mdash; |
| `conditioned` | 80 w | 0.812 | 0.337 | 73.2 | &mdash; | &mdash; |
| `conditioned` | 160 w | 0.802 | 0.528 | 143.1 | &mdash; | &mdash; |
| `reusable` | 20 w | 0.792 | 0.174 | 18.6 | &mdash; | &mdash; |
| `reusable` | 40 w | 0.802 | 0.271 | 36.3 | **yes** | &mdash; |
| `reusable` | 80 w | 0.854 | 0.424 | 73.2 | &mdash; | &mdash; |
| `reusable` | 160 w | 0.906 | 0.601 | 143.6 | **yes** | **yes** |
| `oracle` | 20 w | 0.490 | 0.490 | 18.1 | **yes** | **yes** |
| `oracle` | 40 w | 0.823 | 0.823 | 37.9 | **yes** | **yes** |
| `oracle` | 80 w | 0.885 | 0.885 | 73.8 | **yes** | **yes** |
| `oracle` | 160 w | 0.896 | 0.896 | 143.4 | **yes** | **yes** |
| `extractive_generic` | 20 w | 0.354 | 0.354 | 18.9 | &mdash; | &mdash; |
| `extractive_generic` | 40 w | 0.427 | 0.427 | 37.4 | &mdash; | &mdash; |
| `extractive_generic` | 80 w | 0.615 | 0.615 | 73.3 | &mdash; | &mdash; |
| `extractive_generic` | 160 w | 0.854 | 0.854 | 143.8 | &mdash; | &mdash; |
| `extractive_conditioned` | 20 w | 0.740 | 0.156 | 18.6 | &mdash; | &mdash; |
| `extractive_conditioned` | 40 w | 0.792 | 0.285 | 36.8 | **yes** | &mdash; |
| `extractive_conditioned` | 80 w | 0.906 | 0.458 | 70.9 | **yes** | **yes** |
| `extractive_conditioned` | 160 w | 0.865 | 0.771 | 141.4 | &mdash; | &mdash; |

**Nondominated set — 10 of 24 configurations:** `reusable`@160w, `extractive_conditioned`@80w, `oracle`@160w, `oracle`@80w, `conditioned`@20w, `oracle`@40w, `conditioned`@40w, `reusable`@40w, `extractive_conditioned`@40w, `oracle`@20w.

Cost enters that test, so the set above is not a curve: a configuration can survive purely by being cheaper than everything that beats it on utility. The figure facets observations by requested cap and outlines the two-dimensional nondominated policies within each facet. It deliberately does not connect markers: policy is categorical, so a line or staircase would imply unevaluated intermediate choices. Discrete sets, best-present-query first:

* **20 words** &mdash; `conditioned` (0.86, 0.18); &nbsp;`oracle` (0.49, 0.49)
* **40 words** &mdash; `oracle` (0.82, 0.82)
* **80 words** &mdash; `extractive_conditioned` (0.91, 0.46); &nbsp;`oracle` (0.89, 0.89)
* **160 words** &mdash; `reusable` (0.91, 0.60); &nbsp;`oracle` (0.90, 0.90)

### Relation dossiers

| policy | requested cap | U_now | U_future | delivered cost (words) | 3D nondominated | within-cap 2D nondominated |
|---|---:|---:|---:|---:|---:|---:|
| `generic` | 20 w | 0.141 | 0.103 | 18.8 | &mdash; | &mdash; |
| `generic` | 40 w | 0.344 | 0.206 | 37.1 | &mdash; | &mdash; |
| `generic` | 80 w | 0.469 | 0.315 | 73.2 | &mdash; | &mdash; |
| `generic` | 160 w | 0.609 | 0.509 | 143.6 | &mdash; | &mdash; |
| `conditioned` | 20 w | 1.000 | 0.077 | 18.3 | **yes** | &mdash; |
| `conditioned` | 40 w | 0.984 | 0.096 | 36.5 | **yes** | &mdash; |
| `conditioned` | 80 w | 0.984 | 0.117 | 72.9 | **yes** | &mdash; |
| `conditioned` | 160 w | 0.969 | 0.168 | 143.9 | &mdash; | **yes** |
| `reusable` | 20 w | 1.000 | 0.081 | 18.3 | **yes** | **yes** |
| `reusable` | 40 w | 1.000 | 0.103 | 36.8 | **yes** | **yes** |
| `reusable` | 80 w | 1.000 | 0.197 | 72.9 | **yes** | **yes** |
| `reusable` | 160 w | 0.953 | 0.334 | 143.3 | **yes** | **yes** |
| `oracle` | 20 w | 0.250 | 0.179 | 18.1 | **yes** | **yes** |
| `oracle` | 40 w | 0.516 | 0.399 | 36.8 | **yes** | **yes** |
| `oracle` | 80 w | 0.750 | 0.692 | 75.6 | **yes** | **yes** |
| `oracle` | 160 w | 0.891 | 0.828 | 143.7 | **yes** | **yes** |

**Nondominated set — 11 of 16 configurations:** `reusable`@80w, `reusable`@40w, `reusable`@20w, `conditioned`@20w, `conditioned`@80w, `conditioned`@40w, `reusable`@160w, `oracle`@160w, `oracle`@80w, `oracle`@40w, `oracle`@20w.

Cost enters that test, so the set above is not a curve: a configuration can survive purely by being cheaper than everything that beats it on utility. The figure facets observations by requested cap and outlines the two-dimensional nondominated policies within each facet. It deliberately does not connect markers: policy is categorical, so a line or staircase would imply unevaluated intermediate choices. Discrete sets, best-present-query first:

* **20 words** &mdash; `reusable` (1.00, 0.08); &nbsp;`oracle` (0.25, 0.18)
* **40 words** &mdash; `reusable` (1.00, 0.10); &nbsp;`oracle` (0.52, 0.40)
* **80 words** &mdash; `reusable` (1.00, 0.20); &nbsp;`oracle` (0.75, 0.69)
* **160 words** &mdash; `conditioned` (0.97, 0.17); &nbsp;`reusable` (0.95, 0.33); &nbsp;`oracle` (0.89, 0.83)

## 7. The mechanism

### The channel was never the constraint

| SQuAD at a 40-word budget | U_now | U_future | delivered words |
|---|---:|---:|---:|
| `generic` | 0.375 | 0.375 | 37.1 |
| `conditioned` | 0.823 | 0.260 | 36.7 |
| `reusable` | 0.802 | 0.271 | 36.3 |
| `oracle` | 0.823 | 0.823 | 37.9 |

`oracle`, shown all questions and held to the same band, matches `conditioned`'s present-query score using the same number of words while giving up nothing on the others. What a conditioned handoff drops is not information the budget could not hold. It is information the sender had no reason to keep.

### It is selection, not rewriting

| extractive_conditioned &minus; extractive_generic | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| dU_now | **+0.385 [+0.260, +0.510]** | **+0.365 [+0.240, +0.479]** | **+0.292 [+0.188, +0.396]** | +0.010 [-0.062, +0.083] |
| dU_future | **-0.198 [-0.292, -0.108]** | **-0.142 [-0.215, -0.066]** | **-0.156 [-0.253, -0.052]** | **-0.083 [-0.146, -0.024]** |

The extractive arms copy source sentences verbatim, so no abstractive compression can occur inside them. They reproduce the pattern anyway: the loss lives in *which* evidence is sent, not in how it is worded.

### Asking for reusability barely helps

`reusable` is `conditioned` plus an explicit instruction that unknown further questions may follow and that reusable evidence should be preserved within the same limit. Its `reusable` &minus; `generic` row in the contrast tables above is negative on `U_future` at every budget of both corpora. Telling a sender to stay general is not a substitute for telling it what will be asked.

## 8. Regret by distance from the conditioning query

SQuAD supports no clean labelling of *how far* a hidden question sits from the conditioning one, and inferring that from embeddings would replace a controlled variable with an estimate. So a second corpus builds the distance in: invented dossiers with four aspects each, and within every aspect one anchor question plus a paraphrase of it (same answer, different wording), a second fact about the same subject, and a fact about a different named entity in that aspect. The questions of the other aspects are orthogonal.

![Regret by designed distance](relation_dossiers/n16/relation_distance.png)

**Future-query regret by designed distance, every policy and budget**

| policy | budget | paraphrase | same entity | same topic | orthogonal |
|---|---:|---:|---:|---:|---:|
| `generic` | 20 w | 0.844 [0.781, 0.906] | 0.891 [0.812, 0.953] | 0.969 [0.922, 1.000] | 0.887 [0.859, 0.914] |
| `generic` | 40 w | 0.656 [0.562, 0.734] | 0.875 [0.781, 0.953] | 0.938 [0.875, 0.984] | 0.777 [0.734, 0.820] |
| `generic` | 80 w | 0.516 [0.422, 0.609] | 0.812 [0.750, 0.875] | 0.828 [0.734, 0.922] | 0.668 [0.609, 0.727] |
| `generic` | 160 w | 0.375 [0.266, 0.484] | 0.438 [0.328, 0.547] | 0.719 [0.609, 0.828] | 0.477 [0.410, 0.539] |
| `conditioned` | 20 w | 0.000 [-0.047, 0.047] | 0.953 [0.906, 1.000] | 0.984 [0.953, 1.000] | 0.983 [0.966, 0.996] |
| `conditioned` | 40 w | -0.016 [-0.047, 0.000] | 0.875 [0.812, 0.938] | 0.828 [0.750, 0.906] | 0.980 [0.961, 0.995] |
| `conditioned` | 80 w | 0.016 [0.000, 0.047] | 0.734 [0.625, 0.844] | 0.688 [0.594, 0.781] | 0.975 [0.956, 0.991] |
| `conditioned` | 160 w | 0.016 [-0.031, 0.062] | 0.422 [0.312, 0.531] | 0.516 [0.391, 0.641] | 0.952 [0.926, 0.975] |
| `reusable` | 20 w | 0.000 [-0.047, 0.047] | 0.953 [0.906, 1.000] | 0.938 [0.875, 0.984] | 0.982 [0.965, 0.993] |
| `reusable` | 40 w | -0.016 [-0.047, 0.000] | 0.812 [0.688, 0.922] | 0.750 [0.672, 0.828] | 0.983 [0.962, 0.996] |
| `reusable` | 80 w | -0.016 [-0.047, 0.000] | 0.328 [0.219, 0.438] | 0.562 [0.453, 0.672] | 0.922 [0.895, 0.945] |
| `reusable` | 160 w | 0.031 [0.000, 0.078] | 0.266 [0.172, 0.375] | 0.328 [0.234, 0.438] | 0.771 [0.723, 0.818] |
| `oracle` | 20 w | 0.734 [0.703, 0.750] | 0.828 [0.781, 0.891] | 0.938 [0.875, 0.984] | 0.809 [0.785, 0.832] |
| `oracle` | 40 w | 0.500 [0.406, 0.594] | 0.672 [0.594, 0.750] | 0.703 [0.656, 0.750] | 0.586 [0.551, 0.621] |
| `oracle` | 80 w | 0.234 [0.156, 0.312] | 0.359 [0.266, 0.453] | 0.359 [0.281, 0.438] | 0.297 [0.230, 0.367] |
| `oracle` | 160 w | 0.094 [0.031, 0.156] | 0.219 [0.141, 0.297] | 0.234 [0.125, 0.359] | 0.160 [0.094, 0.227] |

**Future-query utility by designed distance**

| policy | budget | paraphrase | same entity | same topic | orthogonal |
|---|---:|---:|---:|---:|---:|
| `generic` | 20 w | 0.141 [0.078, 0.203] | 0.109 [0.047, 0.188] | 0.031 [0.000, 0.078] | 0.105 [0.078, 0.129] |
| `generic` | 40 w | 0.328 [0.250, 0.422] | 0.125 [0.047, 0.219] | 0.062 [0.016, 0.125] | 0.215 [0.172, 0.258] |
| `generic` | 80 w | 0.469 [0.391, 0.547] | 0.188 [0.125, 0.250] | 0.172 [0.078, 0.266] | 0.324 [0.266, 0.379] |
| `generic` | 160 w | 0.609 [0.516, 0.703] | 0.562 [0.453, 0.672] | 0.281 [0.172, 0.391] | 0.516 [0.457, 0.574] |
| `conditioned` | 20 w | 0.984 [0.953, 1.000] | 0.047 [0.000, 0.094] | 0.016 [0.000, 0.047] | 0.009 [0.003, 0.017] |
| `conditioned` | 40 w | 1.000 [1.000, 1.000] | 0.125 [0.062, 0.188] | 0.172 [0.094, 0.250] | 0.012 [0.005, 0.021] |
| `conditioned` | 80 w | 0.969 [0.922, 1.000] | 0.266 [0.156, 0.375] | 0.312 [0.219, 0.406] | 0.017 [0.008, 0.027] |
| `conditioned` | 160 w | 0.969 [0.922, 1.000] | 0.578 [0.469, 0.688] | 0.484 [0.359, 0.609] | 0.040 [0.022, 0.061] |
| `reusable` | 20 w | 0.984 [0.953, 1.000] | 0.047 [0.000, 0.094] | 0.062 [0.016, 0.125] | 0.010 [0.004, 0.017] |
| `reusable` | 40 w | 1.000 [1.000, 1.000] | 0.188 [0.078, 0.312] | 0.250 [0.172, 0.328] | 0.009 [0.003, 0.017] |
| `reusable` | 80 w | 1.000 [1.000, 1.000] | 0.672 [0.562, 0.781] | 0.438 [0.328, 0.547] | 0.070 [0.052, 0.089] |
| `reusable` | 160 w | 0.953 [0.906, 1.000] | 0.734 [0.625, 0.828] | 0.672 [0.562, 0.766] | 0.221 [0.180, 0.259] |
| `oracle` | 20 w | 0.250 [0.250, 0.250] | 0.172 [0.109, 0.219] | 0.062 [0.016, 0.125] | 0.184 [0.168, 0.199] |
| `oracle` | 40 w | 0.484 [0.391, 0.578] | 0.328 [0.250, 0.406] | 0.297 [0.250, 0.344] | 0.406 [0.367, 0.445] |
| `oracle` | 80 w | 0.750 [0.672, 0.828] | 0.641 [0.547, 0.734] | 0.641 [0.562, 0.719] | 0.695 [0.629, 0.758] |
| `oracle` | 160 w | 0.891 [0.828, 0.953] | 0.781 [0.703, 0.859] | 0.766 [0.641, 0.875] | 0.832 [0.758, 0.902] |

**Monotonicity test: `R(orthogonal) - R(paraphrase)`**

| policy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| `generic` | +0.043 [-0.004, +0.086] (dz 0.5) | **+0.121 [+0.070, +0.176]** (dz 1.1) | **+0.152 [+0.102, +0.211]** (dz 1.4) | **+0.102 [+0.035, +0.164]** (dz 0.7) |
| `conditioned` | **+0.983 [+0.941, +1.016]** (dz 12.8) | **+0.996 [+0.982, +1.014]** (dz 29.9) | **+0.960 [+0.921, +0.988]** (dz 13.3) | **+0.936 [+0.884, +0.982]** (dz 9.1) |
| `reusable` | **+0.982 [+0.940, +1.014]** (dz 12.8) | **+0.999 [+0.986, +1.016]** (dz 31.4) | **+0.938 [+0.917, +0.958]** (dz 21.5) | **+0.740 [+0.671, +0.803]** (dz 5.3) |
| `oracle` | **+0.074 [+0.055, +0.098]** (dz 1.6) | **+0.086 [+0.027, +0.145]** (dz 0.7) | **+0.062 [+0.020, +0.105]** (dz 0.7) | **+0.066 [+0.027, +0.113]** (dz 0.7) |

The gradient is a property of conditioning, not of the corpus: the unconditioned arms are roughly equally bad everywhere, while the conditioned arms are near-perfect at zero distance and near-total losses at maximum distance. Extra bandwidth does not flatten it &mdash; at the widest budget the unconditioned arms cut orthogonal regret substantially while the conditioned arm barely moves.

## 9. Boundaries

- One sender/answerer stack (`meta-llama/llama-3.1-8b-instruct`) and one judge family. A second stack, and human adjudication of a judge sample, are the next external-validity tests.
- Effective *n* is the number of contexts, not the number of rotations &mdash; all intervals cluster at the context level for that reason.
- `U_future` is bounded below by whatever the generic baseline itself achieved, so the raw future loss is smallest exactly where the specialisation gap is largest.
- The paraphrase tier of the distance corpus is by construction the easiest possible case: the gold answer string is identical. Its near-zero regret is a design sanity check, not a finding. Same-entity and same-topic are close enough that the corpus supports "near versus far", not a strict four-point ordering.
- The floor half of the budget contract is best-effort; its misses are listed in the `under floor` column of the length tables, concentrated in the extractive conditioned arm where verbatim sentences cannot be lengthened to order.

## 10. Reproduce it

```bash
python src/build_regret_data.py --which both
```

```bash
python src/run_communication_regret.py
```

```bash
python src/run_communication_regret.py --dataset relation_dossiers --policies generic,conditioned,reusable,oracle --no-trim
```

```bash
python src/selftest_communication_regret_offline.py
```

```bash
python src/render_regret_summary.py
```

The offline check runs with no API key and no network: it asserts that the four abstractive prompts share one skeleton, that the cap is never exceeded, that every question conditions exactly once and is hidden exactly *k*&minus;1 times, that the answerer can never reach the source, that identical requests are issued once, and that the matrix and regret arithmetic recover an injected effect exactly. The last command regenerates this file from the CSVs below, so it cannot drift from them.

| file | contents |
|---|---|
| `utility_matrix.csv` | every `M[a][b]` cell with its relation label and ceiling |
| `rotation_rows.csv` | one row per context &times; rotation &times; policy &times; budget |
| `metrics.csv` | context-clustered means with bootstrap CIs, all three metrics |
| `contrasts.csv` | paired deltas with p-values and Cohen's *dz* |
| `budget_interaction.csv` | the scarcity test |
| `length_audit.csv`, `length_deltas.csv`, `length_matched_contrasts.csv` | the channel controls |
| `pareto.csv` | (policy, budget) points with domination and dominators |
| `normalised_utility.csv` | retained utility as a guarded ratio |
| `relation_regret.csv`, `relation_tests.csv` | the distance analysis |
| `baselines.csv` | direct-context ceiling and closed-book leakage audit |

## Appendix: every computed contrast

The sections above quote the headline comparisons. This appendix is the exhaustive version &mdash; every row the runner wrote to `contrasts.csv`, for all three utility measures, all endpoints, and the trimmed length-control arms. **Bold** excludes zero. `gap` is `U_now - U_future`; `gap DiD` is the difference of that gap between the two arms, which is the single number the hypothesis predicts to be positive.

### SQuAD groups &mdash; contrasts

| comparison | quantity | metric | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|---:|
| conditioned &minus; generic | utility now | judge | **+0.625 [+0.552, +0.698]** <sub>p 0.000</sub> | **+0.448 [+0.354, +0.542]** <sub>p 0.000</sub> | **+0.281 [+0.167, +0.396]** <sub>p 0.000</sub> | +0.104 [+0.000, +0.208] <sub>p 0.066</sub> |
| conditioned &minus; generic | utility now | EM | **+0.417 [+0.292, +0.542]** <sub>p 0.000</sub> | **+0.323 [+0.219, +0.427]** <sub>p 0.000</sub> | **+0.271 [+0.167, +0.375]** <sub>p 0.000</sub> | **+0.188 [+0.083, +0.302]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility now | F1 | **+0.541 [+0.450, +0.633]** <sub>p 0.000</sub> | **+0.413 [+0.319, +0.504]** <sub>p 0.000</sub> | **+0.291 [+0.196, +0.388]** <sub>p 0.000</sub> | **+0.187 [+0.088, +0.288]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility future | judge | **-0.063 [-0.122, -0.003]** <sub>p 0.040</sub> | **-0.115 [-0.201, -0.028]** <sub>p 0.010</sub> | **-0.194 [-0.274, -0.118]** <sub>p 0.000</sub> | **-0.170 [-0.292, -0.052]** <sub>p 0.005</sub> |
| conditioned &minus; generic | utility future | EM | -0.056 [-0.122, +0.014] <sub>p 0.107</sub> | **-0.132 [-0.208, -0.056]** <sub>p 0.001</sub> | **-0.118 [-0.188, -0.045]** <sub>p 0.001</sub> | **-0.122 [-0.233, -0.017]** <sub>p 0.030</sub> |
| conditioned &minus; generic | utility future | F1 | -0.043 [-0.102, +0.020] <sub>p 0.166</sub> | **-0.118 [-0.189, -0.049]** <sub>p 0.001</sub> | **-0.159 [-0.229, -0.088]** <sub>p 0.000</sub> | **-0.140 [-0.248, -0.038]** <sub>p 0.010</sub> |
| conditioned &minus; generic | utility gap | judge | **+0.688 [+0.604, +0.774]** <sub>p 0.000</sub> | **+0.562 [+0.451, +0.670]** <sub>p 0.000</sub> | **+0.476 [+0.372, +0.580]** <sub>p 0.000</sub> | **+0.274 [+0.156, +0.399]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility gap | EM | **+0.472 [+0.361, +0.583]** <sub>p 0.000</sub> | **+0.455 [+0.365, +0.549]** <sub>p 0.000</sub> | **+0.389 [+0.288, +0.490]** <sub>p 0.000</sub> | **+0.309 [+0.222, +0.410]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility gap | F1 | **+0.584 [+0.493, +0.670]** <sub>p 0.000</sub> | **+0.531 [+0.443, +0.621]** <sub>p 0.000</sub> | **+0.450 [+0.359, +0.542]** <sub>p 0.000</sub> | **+0.327 [+0.231, +0.430]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret now | judge | **-0.625 [-0.698, -0.552]** <sub>p 0.000</sub> | **-0.448 [-0.542, -0.354]** <sub>p 0.000</sub> | **-0.281 [-0.396, -0.167]** <sub>p 0.000</sub> | -0.104 [-0.208, +0.000] <sub>p 0.066</sub> |
| conditioned &minus; generic | regret now | EM | **-0.417 [-0.542, -0.292]** <sub>p 0.000</sub> | **-0.323 [-0.427, -0.219]** <sub>p 0.000</sub> | **-0.271 [-0.375, -0.167]** <sub>p 0.000</sub> | **-0.188 [-0.302, -0.083]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret now | F1 | **-0.541 [-0.633, -0.450]** <sub>p 0.000</sub> | **-0.413 [-0.504, -0.319]** <sub>p 0.000</sub> | **-0.291 [-0.388, -0.196]** <sub>p 0.000</sub> | **-0.187 [-0.288, -0.088]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret future | judge | **+0.063 [+0.003, +0.122]** <sub>p 0.039</sub> | **+0.115 [+0.028, +0.201]** <sub>p 0.010</sub> | **+0.194 [+0.118, +0.274]** <sub>p 0.000</sub> | **+0.170 [+0.052, +0.292]** <sub>p 0.004</sub> |
| conditioned &minus; generic | regret future | EM | +0.056 [-0.014, +0.122] <sub>p 0.111</sub> | **+0.132 [+0.056, +0.208]** <sub>p 0.001</sub> | **+0.118 [+0.045, +0.188]** <sub>p 0.001</sub> | **+0.122 [+0.017, +0.233]** <sub>p 0.031</sub> |
| conditioned &minus; generic | regret future | F1 | +0.043 [-0.020, +0.102] <sub>p 0.166</sub> | **+0.118 [+0.049, +0.189]** <sub>p 0.001</sub> | **+0.159 [+0.088, +0.229]** <sub>p 0.000</sub> | **+0.140 [+0.038, +0.248]** <sub>p 0.010</sub> |
| conditioned &minus; generic | retained now | judge | **+0.705 [+0.632, +0.778]** <sub>p 0.000</sub> | **+0.483 [+0.382, +0.583]** <sub>p 0.000</sub> | **+0.351 [+0.233, +0.469]** <sub>p 0.000</sub> | +0.083 [-0.035, +0.201] <sub>p 0.176</sub> |
| conditioned &minus; generic | retained now | EM | **+0.628 [+0.483, +0.771]** <sub>p 0.000</sub> | **+0.490 [+0.354, +0.625]** <sub>p 0.000</sub> | **+0.361 [+0.219, +0.514]** <sub>p 0.000</sub> | **+0.177 [+0.062, +0.302]** <sub>p 0.004</sub> |
| conditioned &minus; generic | retained now | F1 | **+0.603 [+0.507, +0.700]** <sub>p 0.000</sub> | **+0.448 [+0.342, +0.552]** <sub>p 0.000</sub> | **+0.307 [+0.187, +0.428]** <sub>p 0.000</sub> | **+0.188 [+0.078, +0.303]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained future | judge | -0.052 [-0.120, +0.016] <sub>p 0.138</sub> | **-0.127 [-0.217, -0.031]** <sub>p 0.009</sub> | **-0.219 [-0.307, -0.134]** <sub>p 0.000</sub> | **-0.212 [-0.335, -0.089]** <sub>p 0.001</sub> |
| conditioned &minus; generic | retained future | EM | -0.061 [-0.170, +0.059] <sub>p 0.304</sub> | **-0.214 [-0.335, -0.097]** <sub>p 0.001</sub> | **-0.207 [-0.318, -0.095]** <sub>p 0.000</sub> | **-0.245 [-0.392, -0.094]** <sub>p 0.001</sub> |
| conditioned &minus; generic | retained future | F1 | -0.039 [-0.113, +0.046] <sub>p 0.340</sub> | **-0.126 [-0.203, -0.047]** <sub>p 0.001</sub> | **-0.194 [-0.267, -0.118]** <sub>p 0.000</sub> | **-0.156 [-0.271, -0.039]** <sub>p 0.009</sub> |
| conditioned &minus; generic | gap DiD | judge | **+0.688 [+0.604, +0.774]** <sub>p 0.000</sub> | **+0.562 [+0.448, +0.667]** <sub>p 0.000</sub> | **+0.476 [+0.372, +0.580]** <sub>p 0.000</sub> | **+0.274 [+0.156, +0.396]** <sub>p 0.000</sub> |
| conditioned &minus; generic | gap DiD | EM | **+0.472 [+0.361, +0.580]** <sub>p 0.000</sub> | **+0.455 [+0.365, +0.549]** <sub>p 0.000</sub> | **+0.389 [+0.288, +0.490]** <sub>p 0.000</sub> | **+0.309 [+0.222, +0.403]** <sub>p 0.000</sub> |
| conditioned &minus; generic | gap DiD | F1 | **+0.584 [+0.495, +0.669]** <sub>p 0.000</sub> | **+0.531 [+0.442, +0.619]** <sub>p 0.000</sub> | **+0.450 [+0.359, +0.541]** <sub>p 0.000</sub> | **+0.327 [+0.232, +0.427]** <sub>p 0.000</sub> |
| conditioned@trim &minus; generic@trim | utility now | judge | n/a | **+0.500 [+0.396, +0.594]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility now | EM | n/a | **+0.344 [+0.250, +0.438]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility now | F1 | n/a | **+0.412 [+0.330, +0.492]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility future | judge | n/a | **-0.125 [-0.208, -0.031]** <sub>p 0.006</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility future | EM | n/a | **-0.118 [-0.184, -0.052]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility future | F1 | n/a | **-0.130 [-0.198, -0.061]** <sub>p 0.001</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility gap | judge | n/a | **+0.625 [+0.535, +0.715]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility gap | EM | n/a | **+0.462 [+0.365, +0.556]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | utility gap | F1 | n/a | **+0.542 [+0.468, +0.616]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | regret now | judge | n/a | **-0.500 [-0.594, -0.396]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | regret now | EM | n/a | **-0.344 [-0.438, -0.250]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | regret now | F1 | n/a | **-0.412 [-0.492, -0.330]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | regret future | judge | n/a | **+0.125 [+0.031, +0.208]** <sub>p 0.006</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | regret future | EM | n/a | **+0.118 [+0.052, +0.184]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | regret future | F1 | n/a | **+0.130 [+0.061, +0.198]** <sub>p 0.001</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | retained now | judge | n/a | **+0.545 [+0.420, +0.656]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | retained now | EM | n/a | **+0.535 [+0.392, +0.677]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | retained now | F1 | n/a | **+0.456 [+0.365, +0.544]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | retained future | judge | n/a | **-0.139 [-0.238, -0.042]** <sub>p 0.006</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | retained future | EM | n/a | **-0.167 [-0.262, -0.075]** <sub>p 0.001</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | retained future | F1 | n/a | **-0.146 [-0.216, -0.078]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | gap DiD | judge | n/a | **+0.625 [+0.535, +0.712]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | gap DiD | EM | n/a | **+0.462 [+0.368, +0.556]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned@trim &minus; generic@trim | gap DiD | F1 | n/a | **+0.542 [+0.468, +0.616]** <sub>p 0.000</sub> | n/a | n/a |
| conditioned &minus; reusable | utility now | judge | **+0.073 [+0.021, +0.135]** <sub>p 0.013</sub> | +0.021 [-0.073, +0.115] <sub>p 0.748</sub> | -0.042 [-0.094, +0.000] <sub>p 0.117</sub> | **-0.104 [-0.177, -0.031]** <sub>p 0.008</sub> |
| conditioned &minus; reusable | utility now | EM | +0.021 [-0.062, +0.115] <sub>p 0.739</sub> | +0.010 [-0.062, +0.083] <sub>p 0.886</sub> | +0.000 [-0.052, +0.052] <sub>p 1.000</sub> | -0.042 [-0.104, +0.021] <sub>p 0.247</sub> |
| conditioned &minus; reusable | utility now | F1 | +0.013 [-0.050, +0.080] <sub>p 0.708</sub> | +0.003 [-0.075, +0.076] <sub>p 0.930</sub> | -0.012 [-0.045, +0.019] <sub>p 0.456</sub> | -0.041 [-0.091, +0.008] <sub>p 0.106</sub> |
| conditioned &minus; reusable | utility future | judge | +0.003 [-0.031, +0.042] <sub>p 0.809</sub> | -0.010 [-0.056, +0.035] <sub>p 0.664</sub> | **-0.087 [-0.132, -0.038]** <sub>p 0.001</sub> | -0.073 [-0.153, +0.010] <sub>p 0.084</sub> |
| conditioned &minus; reusable | utility future | EM | -0.003 [-0.031, +0.028] <sub>p 0.795</sub> | -0.014 [-0.042, +0.014] <sub>p 0.363</sub> | **-0.056 [-0.087, -0.021]** <sub>p 0.001</sub> | **-0.076 [-0.142, -0.010]** <sub>p 0.020</sub> |
| conditioned &minus; reusable | utility future | F1 | -0.001 [-0.030, +0.027] <sub>p 0.926</sub> | -0.009 [-0.035, +0.016] <sub>p 0.502</sub> | **-0.079 [-0.111, -0.046]** <sub>p 0.000</sub> | **-0.079 [-0.151, -0.006]** <sub>p 0.035</sub> |
| conditioned &minus; reusable | utility gap | judge | **+0.069 [+0.003, +0.142]** <sub>p 0.060</sub> | +0.031 [-0.083, +0.142] <sub>p 0.575</sub> | +0.045 [-0.049, +0.118] <sub>p 0.313</sub> | -0.031 [-0.149, +0.087] <sub>p 0.601</sub> |
| conditioned &minus; reusable | utility gap | EM | +0.024 [-0.062, +0.122] <sub>p 0.596</sub> | +0.024 [-0.052, +0.104] <sub>p 0.543</sub> | +0.056 [-0.007, +0.118] <sub>p 0.084</sub> | +0.035 [-0.052, +0.118] <sub>p 0.434</sub> |
| conditioned &minus; reusable | utility gap | F1 | +0.014 [-0.055, +0.088] <sub>p 0.704</sub> | +0.012 [-0.077, +0.099] <sub>p 0.787</sub> | **+0.067 [+0.014, +0.115]** <sub>p 0.009</sub> | +0.038 [-0.045, +0.118] <sub>p 0.369</sub> |
| conditioned &minus; reusable | regret now | judge | **-0.073 [-0.135, -0.021]** <sub>p 0.013</sub> | -0.021 [-0.115, +0.073] <sub>p 0.748</sub> | +0.042 [+0.000, +0.094] <sub>p 0.117</sub> | **+0.104 [+0.031, +0.177]** <sub>p 0.008</sub> |
| conditioned &minus; reusable | regret now | EM | -0.021 [-0.115, +0.062] <sub>p 0.739</sub> | -0.010 [-0.083, +0.062] <sub>p 0.886</sub> | +0.000 [-0.052, +0.052] <sub>p 1.000</sub> | +0.042 [-0.021, +0.104] <sub>p 0.247</sub> |
| conditioned &minus; reusable | regret now | F1 | -0.013 [-0.080, +0.050] <sub>p 0.708</sub> | -0.003 [-0.076, +0.075] <sub>p 0.930</sub> | +0.012 [-0.019, +0.045] <sub>p 0.456</sub> | +0.041 [-0.008, +0.091] <sub>p 0.106</sub> |
| conditioned &minus; reusable | regret future | judge | -0.003 [-0.042, +0.031] <sub>p 0.787</sub> | +0.010 [-0.035, +0.056] <sub>p 0.631</sub> | **+0.087 [+0.038, +0.132]** <sub>p 0.001</sub> | +0.073 [-0.010, +0.153] <sub>p 0.086</sub> |
| conditioned &minus; reusable | regret future | EM | +0.003 [-0.028, +0.031] <sub>p 0.784</sub> | +0.014 [-0.014, +0.042] <sub>p 0.293</sub> | **+0.056 [+0.021, +0.087]** <sub>p 0.001</sub> | **+0.076 [+0.010, +0.142]** <sub>p 0.022</sub> |
| conditioned &minus; reusable | regret future | F1 | +0.001 [-0.027, +0.030] <sub>p 0.926</sub> | +0.009 [-0.016, +0.035] <sub>p 0.502</sub> | **+0.079 [+0.046, +0.111]** <sub>p 0.000</sub> | **+0.079 [+0.006, +0.151]** <sub>p 0.035</sub> |
| conditioned &minus; reusable | retained now | judge | **+0.069 [+0.021, +0.132]** <sub>p 0.020</sub> | +0.010 [-0.076, +0.097] <sub>p 0.841</sub> | -0.010 [-0.031, +0.000] <sub>p 0.621</sub> | **-0.115 [-0.195, -0.042]** <sub>p 0.005</sub> |
| conditioned &minus; reusable | retained now | EM | -0.003 [-0.132, +0.111] <sub>p 0.978</sub> | +0.010 [-0.066, +0.087] <sub>p 0.815</sub> | +0.017 [-0.052, +0.115] <sub>p 0.718</sub> | -0.024 [-0.083, +0.021] <sub>p 0.408</sub> |
| conditioned &minus; reusable | retained now | F1 | +0.022 [-0.042, +0.091] <sub>p 0.516</sub> | +0.008 [-0.066, +0.076] <sub>p 0.837</sub> | -0.009 [-0.050, +0.028] <sub>p 0.663</sub> | -0.007 [-0.051, +0.034] <sub>p 0.748</sub> |
| conditioned &minus; reusable | retained future | judge | +0.005 [-0.033, +0.047] <sub>p 0.781</sub> | -0.002 [-0.059, +0.057] <sub>p 0.967</sub> | **-0.115 [-0.161, -0.069]** <sub>p 0.000</sub> | **-0.104 [-0.184, -0.021]** <sub>p 0.014</sub> |
| conditioned &minus; reusable | retained future | EM | -0.030 [-0.073, +0.010] <sub>p 0.173</sub> | **-0.040 [-0.083, -0.002]** <sub>p 0.055</sub> | **-0.101 [-0.142, -0.062]** <sub>p 0.000</sub> | **-0.118 [-0.201, -0.036]** <sub>p 0.005</sub> |
| conditioned &minus; reusable | retained future | F1 | -0.020 [-0.059, +0.016] <sub>p 0.291</sub> | -0.004 [-0.035, +0.025] <sub>p 0.774</sub> | **-0.108 [-0.142, -0.075]** <sub>p 0.000</sub> | **-0.098 [-0.176, -0.021]** <sub>p 0.012</sub> |
| conditioned &minus; reusable | gap DiD | judge | **+0.069 [+0.003, +0.142]** <sub>p 0.061</sub> | +0.031 [-0.087, +0.146] <sub>p 0.591</sub> | +0.045 [-0.049, +0.118] <sub>p 0.320</sub> | -0.031 [-0.153, +0.083] <sub>p 0.605</sub> |
| conditioned &minus; reusable | gap DiD | EM | +0.024 [-0.066, +0.118] <sub>p 0.595</sub> | +0.024 [-0.056, +0.104] <sub>p 0.556</sub> | +0.056 [-0.007, +0.118] <sub>p 0.091</sub> | +0.035 [-0.052, +0.118] <sub>p 0.434</sub> |
| conditioned &minus; reusable | gap DiD | F1 | +0.014 [-0.057, +0.090] <sub>p 0.707</sub> | +0.012 [-0.078, +0.099] <sub>p 0.790</sub> | **+0.067 [+0.015, +0.116]** <sub>p 0.010</sub> | +0.038 [-0.046, +0.118] <sub>p 0.368</sub> |
| conditioned@trim &minus; reusable@trim | utility now | judge | n/a | +0.042 [-0.042, +0.125] <sub>p 0.393</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility now | EM | n/a | -0.010 [-0.073, +0.052] <sub>p 0.883</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility now | F1 | n/a | +0.006 [-0.067, +0.076] <sub>p 0.875</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility future | judge | n/a | +0.003 [-0.045, +0.052] <sub>p 0.901</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility future | EM | n/a | +0.003 [-0.024, +0.031] <sub>p 0.740</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility future | F1 | n/a | +0.004 [-0.024, +0.034] <sub>p 0.797</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility gap | judge | n/a | +0.038 [-0.066, +0.149] <sub>p 0.496</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility gap | EM | n/a | -0.014 [-0.090, +0.066] <sub>p 0.753</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | utility gap | F1 | n/a | +0.002 [-0.076, +0.080] <sub>p 0.956</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | regret now | judge | n/a | -0.042 [-0.125, +0.042] <sub>p 0.393</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | regret now | EM | n/a | +0.010 [-0.052, +0.073] <sub>p 0.883</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | regret now | F1 | n/a | -0.006 [-0.076, +0.067] <sub>p 0.875</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | regret future | judge | n/a | -0.003 [-0.052, +0.045] <sub>p 0.909</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | regret future | EM | n/a | -0.003 [-0.031, +0.024] <sub>p 0.858</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | regret future | F1 | n/a | -0.004 [-0.034, +0.024] <sub>p 0.797</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | retained now | judge | n/a | +0.024 [-0.052, +0.097] <sub>p 0.552</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | retained now | EM | n/a | +0.028 [-0.042, +0.097] <sub>p 0.503</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | retained now | F1 | n/a | +0.027 [-0.043, +0.098] <sub>p 0.459</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | retained future | judge | n/a | +0.007 [-0.062, +0.076] <sub>p 0.833</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | retained future | EM | n/a | -0.023 [-0.071, +0.019] <sub>p 0.351</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | retained future | F1 | n/a | +0.002 [-0.030, +0.037] <sub>p 0.890</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | gap DiD | judge | n/a | +0.038 [-0.066, +0.153] <sub>p 0.504</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | gap DiD | EM | n/a | -0.014 [-0.094, +0.069] <sub>p 0.749</sub> | n/a | n/a |
| conditioned@trim &minus; reusable@trim | gap DiD | F1 | n/a | +0.002 [-0.077, +0.081] <sub>p 0.958</sub> | n/a | n/a |
| reusable &minus; generic | utility now | judge | **+0.552 [+0.458, +0.646]** <sub>p 0.000</sub> | **+0.427 [+0.312, +0.542]** <sub>p 0.000</sub> | **+0.323 [+0.208, +0.438]** <sub>p 0.000</sub> | **+0.208 [+0.104, +0.312]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility now | EM | **+0.396 [+0.292, +0.510]** <sub>p 0.000</sub> | **+0.312 [+0.208, +0.427]** <sub>p 0.000</sub> | **+0.271 [+0.167, +0.365]** <sub>p 0.000</sub> | **+0.229 [+0.135, +0.344]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility now | F1 | **+0.528 [+0.443, +0.616]** <sub>p 0.000</sub> | **+0.409 [+0.312, +0.512]** <sub>p 0.000</sub> | **+0.304 [+0.215, +0.392]** <sub>p 0.000</sub> | **+0.228 [+0.139, +0.329]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility future | judge | **-0.066 [-0.128, -0.003]** <sub>p 0.037</sub> | **-0.104 [-0.177, -0.035]** <sub>p 0.005</sub> | **-0.108 [-0.184, -0.031]** <sub>p 0.005</sub> | **-0.097 [-0.187, -0.003]** <sub>p 0.037</sub> |
| reusable &minus; generic | utility future | EM | -0.052 [-0.115, +0.024] <sub>p 0.126</sub> | **-0.118 [-0.181, -0.059]** <sub>p 0.000</sub> | -0.062 [-0.135, +0.010] <sub>p 0.102</sub> | -0.045 [-0.125, +0.035] <sub>p 0.277</sub> |
| reusable &minus; generic | utility future | F1 | -0.042 [-0.096, +0.018] <sub>p 0.152</sub> | **-0.110 [-0.173, -0.048]** <sub>p 0.001</sub> | **-0.080 [-0.151, -0.004]** <sub>p 0.034</sub> | -0.061 [-0.136, +0.017] <sub>p 0.115</sub> |
| reusable &minus; generic | utility gap | judge | **+0.618 [+0.524, +0.712]** <sub>p 0.000</sub> | **+0.531 [+0.410, +0.653]** <sub>p 0.000</sub> | **+0.431 [+0.323, +0.535]** <sub>p 0.000</sub> | **+0.306 [+0.219, +0.392]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility gap | EM | **+0.448 [+0.358, +0.538]** <sub>p 0.000</sub> | **+0.431 [+0.333, +0.535]** <sub>p 0.000</sub> | **+0.333 [+0.229, +0.431]** <sub>p 0.000</sub> | **+0.274 [+0.201, +0.351]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility gap | F1 | **+0.570 [+0.489, +0.649]** <sub>p 0.000</sub> | **+0.519 [+0.428, +0.613]** <sub>p 0.000</sub> | **+0.384 [+0.286, +0.478]** <sub>p 0.000</sub> | **+0.289 [+0.215, +0.364]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret now | judge | **-0.552 [-0.646, -0.458]** <sub>p 0.000</sub> | **-0.427 [-0.542, -0.312]** <sub>p 0.000</sub> | **-0.323 [-0.438, -0.208]** <sub>p 0.000</sub> | **-0.208 [-0.312, -0.104]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret now | EM | **-0.396 [-0.510, -0.292]** <sub>p 0.000</sub> | **-0.312 [-0.427, -0.208]** <sub>p 0.000</sub> | **-0.271 [-0.365, -0.167]** <sub>p 0.000</sub> | **-0.229 [-0.344, -0.135]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret now | F1 | **-0.528 [-0.616, -0.443]** <sub>p 0.000</sub> | **-0.409 [-0.512, -0.312]** <sub>p 0.000</sub> | **-0.304 [-0.392, -0.215]** <sub>p 0.000</sub> | **-0.228 [-0.329, -0.139]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret future | judge | **+0.066 [+0.003, +0.128]** <sub>p 0.031</sub> | **+0.104 [+0.035, +0.177]** <sub>p 0.005</sub> | **+0.108 [+0.031, +0.184]** <sub>p 0.005</sub> | **+0.097 [+0.003, +0.188]** <sub>p 0.033</sub> |
| reusable &minus; generic | regret future | EM | +0.052 [-0.024, +0.115] <sub>p 0.146</sub> | **+0.118 [+0.059, +0.181]** <sub>p 0.000</sub> | +0.062 [-0.010, +0.135] <sub>p 0.101</sub> | +0.045 [-0.035, +0.125] <sub>p 0.279</sub> |
| reusable &minus; generic | regret future | F1 | +0.042 [-0.018, +0.096] <sub>p 0.152</sub> | **+0.110 [+0.048, +0.173]** <sub>p 0.001</sub> | **+0.080 [+0.004, +0.151]** <sub>p 0.034</sub> | +0.061 [-0.017, +0.136] <sub>p 0.115</sub> |
| reusable &minus; generic | retained now | judge | **+0.635 [+0.542, +0.729]** <sub>p 0.000</sub> | **+0.472 [+0.326, +0.611]** <sub>p 0.000</sub> | **+0.361 [+0.247, +0.476]** <sub>p 0.000</sub> | **+0.198 [+0.104, +0.306]** <sub>p 0.001</sub> |
| reusable &minus; generic | retained now | EM | **+0.632 [+0.507, +0.750]** <sub>p 0.000</sub> | **+0.479 [+0.326, +0.628]** <sub>p 0.000</sub> | **+0.344 [+0.208, +0.483]** <sub>p 0.000</sub> | **+0.201 [+0.090, +0.326]** <sub>p 0.001</sub> |
| reusable &minus; generic | retained now | F1 | **+0.581 [+0.482, +0.676]** <sub>p 0.000</sub> | **+0.441 [+0.323, +0.554]** <sub>p 0.000</sub> | **+0.315 [+0.202, +0.429]** <sub>p 0.000</sub> | **+0.195 [+0.105, +0.298]** <sub>p 0.000</sub> |
| reusable &minus; generic | retained future | judge | -0.057 [-0.122, +0.007] <sub>p 0.076</sub> | **-0.125 [-0.215, -0.033]** <sub>p 0.007</sub> | **-0.104 [-0.184, -0.024]** <sub>p 0.009</sub> | -0.108 [-0.207, +0.002] <sub>p 0.041</sub> |
| reusable &minus; generic | retained future | EM | -0.031 [-0.146, +0.108] <sub>p 0.639</sub> | **-0.174 [-0.274, -0.076]** <sub>p 0.000</sub> | -0.106 [-0.220, +0.012] <sub>p 0.080</sub> | -0.127 [-0.259, +0.002] <sub>p 0.056</sub> |
| reusable &minus; generic | retained future | F1 | -0.018 [-0.091, +0.071] <sub>p 0.674</sub> | **-0.122 [-0.192, -0.051]** <sub>p 0.001</sub> | **-0.085 [-0.160, -0.003]** <sub>p 0.033</sub> | -0.057 [-0.142, +0.037] <sub>p 0.207</sub> |
| reusable &minus; generic | gap DiD | judge | **+0.618 [+0.521, +0.708]** <sub>p 0.000</sub> | **+0.531 [+0.406, +0.656]** <sub>p 0.000</sub> | **+0.431 [+0.323, +0.535]** <sub>p 0.000</sub> | **+0.306 [+0.215, +0.396]** <sub>p 0.000</sub> |
| reusable &minus; generic | gap DiD | EM | **+0.448 [+0.361, +0.538]** <sub>p 0.000</sub> | **+0.431 [+0.330, +0.535]** <sub>p 0.000</sub> | **+0.333 [+0.229, +0.431]** <sub>p 0.000</sub> | **+0.274 [+0.198, +0.351]** <sub>p 0.000</sub> |
| reusable &minus; generic | gap DiD | F1 | **+0.570 [+0.490, +0.649]** <sub>p 0.000</sub> | **+0.519 [+0.429, +0.613]** <sub>p 0.000</sub> | **+0.384 [+0.286, +0.476]** <sub>p 0.000</sub> | **+0.289 [+0.215, +0.364]** <sub>p 0.000</sub> |
| reusable@trim &minus; generic@trim | utility now | judge | n/a | **+0.458 [+0.354, +0.562]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility now | EM | n/a | **+0.354 [+0.260, +0.458]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility now | F1 | n/a | **+0.406 [+0.311, +0.504]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility future | judge | n/a | **-0.128 [-0.212, -0.042]** <sub>p 0.003</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility future | EM | n/a | **-0.122 [-0.188, -0.056]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility future | F1 | n/a | **-0.134 [-0.195, -0.070]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility gap | judge | n/a | **+0.587 [+0.472, +0.701]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility gap | EM | n/a | **+0.476 [+0.375, +0.583]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | utility gap | F1 | n/a | **+0.540 [+0.439, +0.640]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | regret now | judge | n/a | **-0.458 [-0.562, -0.354]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | regret now | EM | n/a | **-0.354 [-0.458, -0.260]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | regret now | F1 | n/a | **-0.406 [-0.504, -0.311]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | regret future | judge | n/a | **+0.128 [+0.042, +0.212]** <sub>p 0.003</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | regret future | EM | n/a | **+0.122 [+0.056, +0.188]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | regret future | F1 | n/a | **+0.134 [+0.070, +0.195]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | retained now | judge | n/a | **+0.521 [+0.392, +0.639]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | retained now | EM | n/a | **+0.507 [+0.354, +0.660]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | retained now | F1 | n/a | **+0.429 [+0.323, +0.535]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | retained future | judge | n/a | **-0.146 [-0.253, -0.040]** <sub>p 0.008</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | retained future | EM | n/a | **-0.144 [-0.253, -0.038]** <sub>p 0.009</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | retained future | F1 | n/a | **-0.148 [-0.217, -0.081]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | gap DiD | judge | n/a | **+0.587 [+0.472, +0.698]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | gap DiD | EM | n/a | **+0.476 [+0.372, +0.583]** <sub>p 0.000</sub> | n/a | n/a |
| reusable@trim &minus; generic@trim | gap DiD | F1 | n/a | **+0.540 [+0.439, +0.640]** <sub>p 0.000</sub> | n/a | n/a |
| oracle &minus; conditioned | utility now | judge | **-0.375 [-0.438, -0.323]** <sub>p 0.000</sub> | +0.000 [-0.104, +0.094] <sub>p 1.000</sub> | **+0.073 [+0.021, +0.125]** <sub>p 0.012</sub> | **+0.094 [+0.010, +0.177]** <sub>p 0.037</sub> |
| oracle &minus; conditioned | utility now | EM | **-0.271 [-0.354, -0.188]** <sub>p 0.000</sub> | -0.062 [-0.156, +0.021] <sub>p 0.197</sub> | +0.042 [-0.042, +0.125] <sub>p 0.385</sub> | +0.021 [-0.042, +0.094] <sub>p 0.661</sub> |
| oracle &minus; conditioned | utility now | F1 | **-0.328 [-0.394, -0.253]** <sub>p 0.000</sub> | -0.049 [-0.153, +0.043] <sub>p 0.326</sub> | **+0.074 [+0.021, +0.128]** <sub>p 0.008</sub> | +0.033 [-0.024, +0.089] <sub>p 0.264</sub> |
| oracle &minus; conditioned | utility future | judge | **+0.312 [+0.229, +0.396]** <sub>p 0.000</sub> | **+0.562 [+0.444, +0.681]** <sub>p 0.000</sub> | **+0.549 [+0.469, +0.632]** <sub>p 0.000</sub> | **+0.368 [+0.257, +0.483]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility future | EM | **+0.201 [+0.115, +0.295]** <sub>p 0.000</sub> | **+0.392 [+0.285, +0.507]** <sub>p 0.000</sub> | **+0.431 [+0.326, +0.535]** <sub>p 0.000</sub> | **+0.330 [+0.236, +0.431]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility future | F1 | **+0.256 [+0.169, +0.343]** <sub>p 0.000</sub> | **+0.482 [+0.370, +0.593]** <sub>p 0.000</sub> | **+0.524 [+0.458, +0.592]** <sub>p 0.000</sub> | **+0.359 [+0.263, +0.456]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility gap | judge | **-0.688 [-0.774, -0.604]** <sub>p 0.000</sub> | **-0.562 [-0.670, -0.451]** <sub>p 0.000</sub> | **-0.476 [-0.580, -0.372]** <sub>p 0.000</sub> | **-0.274 [-0.399, -0.156]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility gap | EM | **-0.472 [-0.583, -0.361]** <sub>p 0.000</sub> | **-0.455 [-0.549, -0.365]** <sub>p 0.000</sub> | **-0.389 [-0.490, -0.288]** <sub>p 0.000</sub> | **-0.309 [-0.410, -0.222]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility gap | F1 | **-0.584 [-0.670, -0.493]** <sub>p 0.000</sub> | **-0.531 [-0.621, -0.443]** <sub>p 0.000</sub> | **-0.450 [-0.542, -0.359]** <sub>p 0.000</sub> | **-0.327 [-0.430, -0.231]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | regret now | judge | **+0.375 [+0.323, +0.438]** <sub>p 0.000</sub> | +0.000 [-0.094, +0.104] <sub>p 1.000</sub> | **-0.073 [-0.125, -0.021]** <sub>p 0.012</sub> | **-0.094 [-0.177, -0.010]** <sub>p 0.037</sub> |
| oracle &minus; conditioned | regret now | EM | **+0.271 [+0.188, +0.354]** <sub>p 0.000</sub> | +0.062 [-0.021, +0.156] <sub>p 0.197</sub> | -0.042 [-0.125, +0.042] <sub>p 0.385</sub> | -0.021 [-0.094, +0.042] <sub>p 0.661</sub> |
| oracle &minus; conditioned | regret now | F1 | **+0.328 [+0.253, +0.394]** <sub>p 0.000</sub> | +0.049 [-0.043, +0.153] <sub>p 0.326</sub> | **-0.074 [-0.128, -0.021]** <sub>p 0.008</sub> | -0.033 [-0.089, +0.024] <sub>p 0.264</sub> |
| oracle &minus; conditioned | regret future | judge | **-0.312 [-0.396, -0.229]** <sub>p 0.000</sub> | **-0.562 [-0.681, -0.444]** <sub>p 0.000</sub> | **-0.549 [-0.632, -0.469]** <sub>p 0.000</sub> | **-0.368 [-0.483, -0.257]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | regret future | EM | **-0.201 [-0.295, -0.115]** <sub>p 0.000</sub> | **-0.392 [-0.507, -0.285]** <sub>p 0.000</sub> | **-0.431 [-0.535, -0.326]** <sub>p 0.000</sub> | **-0.330 [-0.431, -0.236]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | regret future | F1 | **-0.256 [-0.343, -0.169]** <sub>p 0.000</sub> | **-0.482 [-0.593, -0.370]** <sub>p 0.000</sub> | **-0.524 [-0.592, -0.458]** <sub>p 0.000</sub> | **-0.359 [-0.456, -0.263]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | retained now | judge | **-0.444 [-0.521, -0.368]** <sub>p 0.000</sub> | -0.024 [-0.139, +0.076] <sub>p 0.659</sub> | +0.021 [+0.000, +0.052] <sub>p 0.262</sub> | **+0.094 [+0.010, +0.184]** <sub>p 0.037</sub> |
| oracle &minus; conditioned | retained now | EM | **-0.458 [-0.604, -0.312]** <sub>p 0.000</sub> | -0.108 [-0.233, +0.000] <sub>p 0.073</sub> | +0.056 [-0.049, +0.170] <sub>p 0.323</sub> | +0.045 [-0.052, +0.160] <sub>p 0.420</sub> |
| oracle &minus; conditioned | retained now | F1 | **-0.355 [-0.429, -0.273]** <sub>p 0.000</sub> | -0.070 [-0.186, +0.029] <sub>p 0.200</sub> | **+0.082 [+0.023, +0.143]** <sub>p 0.007</sub> | +0.030 [-0.029, +0.088] <sub>p 0.317</sub> |
| oracle &minus; conditioned | retained future | judge | **+0.312 [+0.224, +0.401]** <sub>p 0.000</sub> | **+0.585 [+0.465, +0.703]** <sub>p 0.000</sub> | **+0.590 [+0.507, +0.672]** <sub>p 0.000</sub> | **+0.389 [+0.278, +0.503]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | retained future | EM | **+0.231 [+0.102, +0.363]** <sub>p 0.001</sub> | **+0.595 [+0.469, +0.714]** <sub>p 0.000</sub> | **+0.623 [+0.524, +0.720]** <sub>p 0.000</sub> | **+0.467 [+0.325, +0.606]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | retained future | F1 | **+0.286 [+0.188, +0.390]** <sub>p 0.000</sub> | **+0.505 [+0.384, +0.618]** <sub>p 0.000</sub> | **+0.582 [+0.508, +0.657]** <sub>p 0.000</sub> | **+0.374 [+0.261, +0.491]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | gap DiD | judge | **-0.688 [-0.774, -0.604]** <sub>p 0.000</sub> | **-0.562 [-0.667, -0.448]** <sub>p 0.000</sub> | **-0.476 [-0.580, -0.372]** <sub>p 0.000</sub> | **-0.274 [-0.396, -0.156]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | gap DiD | EM | **-0.472 [-0.580, -0.361]** <sub>p 0.000</sub> | **-0.455 [-0.549, -0.365]** <sub>p 0.000</sub> | **-0.389 [-0.490, -0.288]** <sub>p 0.000</sub> | **-0.309 [-0.403, -0.222]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | gap DiD | F1 | **-0.584 [-0.669, -0.495]** <sub>p 0.000</sub> | **-0.531 [-0.619, -0.442]** <sub>p 0.000</sub> | **-0.450 [-0.541, -0.359]** <sub>p 0.000</sub> | **-0.327 [-0.427, -0.232]** <sub>p 0.000</sub> |
| oracle@trim &minus; conditioned@trim | utility now | judge | n/a | **-0.229 [-0.323, -0.135]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility now | EM | n/a | **-0.177 [-0.292, -0.073]** <sub>p 0.002</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility now | F1 | n/a | **-0.219 [-0.302, -0.141]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility future | judge | n/a | **+0.396 [+0.299, +0.497]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility future | EM | n/a | **+0.285 [+0.194, +0.378]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility future | F1 | n/a | **+0.323 [+0.234, +0.415]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility gap | judge | n/a | **-0.625 [-0.715, -0.535]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility gap | EM | n/a | **-0.462 [-0.556, -0.365]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | utility gap | F1 | n/a | **-0.542 [-0.616, -0.468]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | regret now | judge | n/a | **+0.229 [+0.135, +0.323]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | regret now | EM | n/a | **+0.177 [+0.073, +0.292]** <sub>p 0.002</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | regret now | F1 | n/a | **+0.219 [+0.141, +0.302]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | regret future | judge | n/a | **-0.396 [-0.497, -0.299]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | regret future | EM | n/a | **-0.285 [-0.378, -0.194]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | regret future | F1 | n/a | **-0.323 [-0.415, -0.234]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | retained now | judge | n/a | **-0.267 [-0.358, -0.174]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | retained now | EM | n/a | **-0.274 [-0.403, -0.149]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | retained now | F1 | n/a | **-0.248 [-0.340, -0.157]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | retained future | judge | n/a | **+0.417 [+0.302, +0.530]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | retained future | EM | n/a | **+0.427 [+0.290, +0.559]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | retained future | F1 | n/a | **+0.354 [+0.239, +0.469]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | gap DiD | judge | n/a | **-0.625 [-0.712, -0.535]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | gap DiD | EM | n/a | **-0.462 [-0.556, -0.368]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; conditioned@trim | gap DiD | F1 | n/a | **-0.542 [-0.616, -0.468]** <sub>p 0.000</sub> | n/a | n/a |
| oracle &minus; generic | utility now | judge | **+0.250 [+0.167, +0.344]** <sub>p 0.000</sub> | **+0.448 [+0.323, +0.562]** <sub>p 0.000</sub> | **+0.354 [+0.260, +0.458]** <sub>p 0.000</sub> | **+0.198 [+0.115, +0.292]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility now | EM | **+0.146 [+0.042, +0.260]** <sub>p 0.012</sub> | **+0.260 [+0.135, +0.385]** <sub>p 0.000</sub> | **+0.312 [+0.208, +0.427]** <sub>p 0.000</sub> | **+0.208 [+0.125, +0.292]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility now | F1 | **+0.213 [+0.110, +0.324]** <sub>p 0.000</sub> | **+0.363 [+0.241, +0.482]** <sub>p 0.000</sub> | **+0.366 [+0.282, +0.454]** <sub>p 0.000</sub> | **+0.220 [+0.147, +0.294]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility future | judge | **+0.250 [+0.167, +0.344]** <sub>p 0.000</sub> | **+0.448 [+0.323, +0.562]** <sub>p 0.000</sub> | **+0.354 [+0.260, +0.458]** <sub>p 0.000</sub> | **+0.198 [+0.115, +0.292]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility future | EM | **+0.146 [+0.042, +0.260]** <sub>p 0.012</sub> | **+0.260 [+0.135, +0.385]** <sub>p 0.000</sub> | **+0.312 [+0.208, +0.427]** <sub>p 0.000</sub> | **+0.208 [+0.125, +0.292]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility future | F1 | **+0.213 [+0.110, +0.324]** <sub>p 0.000</sub> | **+0.363 [+0.241, +0.482]** <sub>p 0.000</sub> | **+0.366 [+0.282, +0.454]** <sub>p 0.000</sub> | **+0.220 [+0.147, +0.294]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility gap | judge | **+0.000 [+0.000, +0.000]** <sub>p 0.039</sub> | **-0.000 [-0.000, -0.000]** <sub>p 0.000</sub> | **-0.000 [-0.000, -0.000]** <sub>p 0.002</sub> | **-0.000 [-0.000, -0.000]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility gap | EM | **+0.000 [+0.000, +0.000]** <sub>p 0.029</sub> | +0.000 [-0.000, +0.000] <sub>p 0.810</sub> | +0.000 [-0.000, +0.000] <sub>p 0.066</sub> | **+0.000 [+0.000, +0.000]** <sub>p 0.041</sub> |
| oracle &minus; generic | utility gap | F1 | +0.000 [-0.000, +0.000] <sub>p 0.234</sub> | -0.000 [-0.000, +0.000] <sub>p 0.972</sub> | +0.000 [-0.000, +0.000] <sub>p 0.186</sub> | +0.000 [-0.000, +0.000] <sub>p 0.901</sub> |
| oracle &minus; generic | regret now | judge | **-0.250 [-0.344, -0.167]** <sub>p 0.000</sub> | **-0.448 [-0.562, -0.323]** <sub>p 0.000</sub> | **-0.354 [-0.458, -0.260]** <sub>p 0.000</sub> | **-0.198 [-0.292, -0.115]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret now | EM | **-0.146 [-0.260, -0.042]** <sub>p 0.012</sub> | **-0.260 [-0.385, -0.135]** <sub>p 0.000</sub> | **-0.312 [-0.427, -0.208]** <sub>p 0.000</sub> | **-0.208 [-0.292, -0.125]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret now | F1 | **-0.213 [-0.324, -0.110]** <sub>p 0.000</sub> | **-0.363 [-0.482, -0.241]** <sub>p 0.000</sub> | **-0.366 [-0.454, -0.282]** <sub>p 0.000</sub> | **-0.220 [-0.294, -0.147]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret future | judge | **-0.250 [-0.344, -0.167]** <sub>p 0.000</sub> | **-0.448 [-0.562, -0.323]** <sub>p 0.000</sub> | **-0.354 [-0.458, -0.260]** <sub>p 0.000</sub> | **-0.198 [-0.292, -0.115]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret future | EM | **-0.146 [-0.260, -0.042]** <sub>p 0.012</sub> | **-0.260 [-0.385, -0.135]** <sub>p 0.000</sub> | **-0.312 [-0.427, -0.208]** <sub>p 0.000</sub> | **-0.208 [-0.292, -0.125]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret future | F1 | **-0.213 [-0.324, -0.110]** <sub>p 0.000</sub> | **-0.363 [-0.482, -0.241]** <sub>p 0.000</sub> | **-0.366 [-0.454, -0.282]** <sub>p 0.000</sub> | **-0.220 [-0.294, -0.147]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained now | judge | **+0.260 [+0.156, +0.368]** <sub>p 0.000</sub> | **+0.458 [+0.306, +0.597]** <sub>p 0.000</sub> | **+0.372 [+0.264, +0.479]** <sub>p 0.000</sub> | **+0.177 [+0.090, +0.278]** <sub>p 0.001</sub> |
| oracle &minus; generic | retained now | EM | **+0.170 [+0.031, +0.319]** <sub>p 0.022</sub> | **+0.382 [+0.188, +0.569]** <sub>p 0.000</sub> | **+0.417 [+0.278, +0.562]** <sub>p 0.000</sub> | **+0.222 [+0.094, +0.361]** <sub>p 0.001</sub> |
| oracle &minus; generic | retained now | F1 | **+0.248 [+0.121, +0.386]** <sub>p 0.001</sub> | **+0.379 [+0.222, +0.520]** <sub>p 0.000</sub> | **+0.389 [+0.287, +0.498]** <sub>p 0.000</sub> | **+0.218 [+0.137, +0.304]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained future | judge | **+0.260 [+0.156, +0.368]** <sub>p 0.000</sub> | **+0.458 [+0.306, +0.597]** <sub>p 0.000</sub> | **+0.372 [+0.264, +0.479]** <sub>p 0.000</sub> | **+0.177 [+0.090, +0.278]** <sub>p 0.001</sub> |
| oracle &minus; generic | retained future | EM | **+0.170 [+0.031, +0.319]** <sub>p 0.022</sub> | **+0.382 [+0.188, +0.569]** <sub>p 0.000</sub> | **+0.417 [+0.278, +0.562]** <sub>p 0.000</sub> | **+0.222 [+0.094, +0.361]** <sub>p 0.001</sub> |
| oracle &minus; generic | retained future | F1 | **+0.248 [+0.121, +0.386]** <sub>p 0.001</sub> | **+0.379 [+0.222, +0.520]** <sub>p 0.000</sub> | **+0.389 [+0.287, +0.498]** <sub>p 0.000</sub> | **+0.218 [+0.137, +0.304]** <sub>p 0.000</sub> |
| oracle &minus; generic | gap DiD | judge | **+0.000 [+0.000, +0.000]** <sub>p 0.037</sub> | **-0.000 [-0.000, -0.000]** <sub>p 0.000</sub> | **-0.000 [-0.000, -0.000]** <sub>p 0.002</sub> | **-0.000 [-0.000, -0.000]** <sub>p 0.000</sub> |
| oracle &minus; generic | gap DiD | EM | **+0.000 [+0.000, +0.000]** <sub>p 0.030</sub> | +0.000 [-0.000, +0.000] <sub>p 0.807</sub> | +0.000 [-0.000, +0.000] <sub>p 0.064</sub> | **+0.000 [+0.000, +0.000]** <sub>p 0.038</sub> |
| oracle &minus; generic | gap DiD | F1 | +0.000 [-0.000, +0.000] <sub>p 0.227</sub> | -0.000 [-0.000, +0.000] <sub>p 0.971</sub> | +0.000 [-0.000, +0.000] <sub>p 0.191</sub> | +0.000 [-0.000, +0.000] <sub>p 0.899</sub> |
| oracle@trim &minus; generic@trim | utility now | judge | n/a | **+0.271 [+0.156, +0.385]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility now | EM | n/a | **+0.167 [+0.073, +0.260]** <sub>p 0.001</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility now | F1 | n/a | **+0.193 [+0.099, +0.288]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility future | judge | n/a | **+0.271 [+0.156, +0.385]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility future | EM | n/a | **+0.167 [+0.073, +0.260]** <sub>p 0.001</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility future | F1 | n/a | **+0.193 [+0.099, +0.288]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility gap | judge | n/a | +0.000 [-0.000, +0.000] <sub>p 0.804</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility gap | EM | n/a | +0.000 [-0.000, +0.000] <sub>p 0.095</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | utility gap | F1 | n/a | +0.000 [-0.000, +0.000] <sub>p 0.615</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | regret now | judge | n/a | **-0.271 [-0.385, -0.156]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | regret now | EM | n/a | **-0.167 [-0.260, -0.073]** <sub>p 0.001</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | regret now | F1 | n/a | **-0.193 [-0.288, -0.099]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | regret future | judge | n/a | **-0.271 [-0.385, -0.156]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | regret future | EM | n/a | **-0.167 [-0.260, -0.073]** <sub>p 0.001</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | regret future | F1 | n/a | **-0.193 [-0.288, -0.099]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | retained now | judge | n/a | **+0.278 [+0.125, +0.424]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | retained now | EM | n/a | **+0.260 [+0.090, +0.431]** <sub>p 0.003</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | retained now | F1 | n/a | **+0.208 [+0.070, +0.345]** <sub>p 0.003</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | retained future | judge | n/a | **+0.278 [+0.125, +0.424]** <sub>p 0.000</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | retained future | EM | n/a | **+0.260 [+0.090, +0.431]** <sub>p 0.003</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | retained future | F1 | n/a | **+0.208 [+0.070, +0.345]** <sub>p 0.003</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | gap DiD | judge | n/a | +0.000 [-0.000, +0.000] <sub>p 0.804</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | gap DiD | EM | n/a | +0.000 [-0.000, +0.000] <sub>p 0.096</sub> | n/a | n/a |
| oracle@trim &minus; generic@trim | gap DiD | F1 | n/a | +0.000 [-0.000, +0.000] <sub>p 0.615</sub> | n/a | n/a |
| extractive_conditioned &minus; extractive_generic | utility now | judge | **+0.385 [+0.260, +0.510]** <sub>p 0.000</sub> | **+0.365 [+0.240, +0.479]** <sub>p 0.000</sub> | **+0.292 [+0.188, +0.396]** <sub>p 0.000</sub> | +0.010 [-0.062, +0.083] <sub>p 0.887</sub> |
| extractive_conditioned &minus; extractive_generic | utility now | EM | **+0.260 [+0.135, +0.385]** <sub>p 0.000</sub> | **+0.250 [+0.135, +0.365]** <sub>p 0.000</sub> | **+0.146 [+0.031, +0.260]** <sub>p 0.015</sub> | -0.052 [-0.115, +0.010] <sub>p 0.145</sub> |
| extractive_conditioned &minus; extractive_generic | utility now | F1 | **+0.374 [+0.271, +0.477]** <sub>p 0.000</sub> | **+0.325 [+0.227, +0.425]** <sub>p 0.000</sub> | **+0.222 [+0.119, +0.324]** <sub>p 0.000</sub> | +0.004 [-0.040, +0.049] <sub>p 0.862</sub> |
| extractive_conditioned &minus; extractive_generic | utility future | judge | **-0.198 [-0.292, -0.108]** <sub>p 0.000</sub> | **-0.142 [-0.215, -0.066]** <sub>p 0.001</sub> | **-0.156 [-0.253, -0.052]** <sub>p 0.003</sub> | **-0.083 [-0.146, -0.024]** <sub>p 0.011</sub> |
| extractive_conditioned &minus; extractive_generic | utility future | EM | **-0.205 [-0.281, -0.139]** <sub>p 0.000</sub> | **-0.167 [-0.243, -0.090]** <sub>p 0.000</sub> | **-0.264 [-0.340, -0.188]** <sub>p 0.000</sub> | **-0.118 [-0.177, -0.059]** <sub>p 0.000</sub> |
| extractive_conditioned &minus; extractive_generic | utility future | F1 | **-0.198 [-0.266, -0.140]** <sub>p 0.000</sub> | **-0.157 [-0.216, -0.098]** <sub>p 0.000</sub> | **-0.236 [-0.318, -0.147]** <sub>p 0.000</sub> | **-0.098 [-0.156, -0.045]** <sub>p 0.001</sub> |
| extractive_conditioned &minus; extractive_generic | utility gap | judge | **+0.583 [+0.500, +0.663]** <sub>p 0.000</sub> | **+0.507 [+0.389, +0.625]** <sub>p 0.000</sub> | **+0.448 [+0.358, +0.542]** <sub>p 0.000</sub> | **+0.094 [+0.028, +0.163]** <sub>p 0.009</sub> |
| extractive_conditioned &minus; extractive_generic | utility gap | EM | **+0.465 [+0.368, +0.559]** <sub>p 0.000</sub> | **+0.417 [+0.306, +0.531]** <sub>p 0.000</sub> | **+0.410 [+0.309, +0.510]** <sub>p 0.000</sub> | **+0.066 [+0.003, +0.132]** <sub>p 0.049</sub> |
| extractive_conditioned &minus; extractive_generic | utility gap | F1 | **+0.572 [+0.493, +0.648]** <sub>p 0.000</sub> | **+0.482 [+0.382, +0.580]** <sub>p 0.000</sub> | **+0.458 [+0.377, +0.537]** <sub>p 0.000</sub> | **+0.102 [+0.050, +0.156]** <sub>p 0.000</sub> |
| extractive_conditioned &minus; extractive_generic | regret now | judge | **-0.385 [-0.510, -0.260]** <sub>p 0.000</sub> | **-0.365 [-0.479, -0.240]** <sub>p 0.000</sub> | **-0.292 [-0.396, -0.188]** <sub>p 0.000</sub> | -0.010 [-0.083, +0.062] <sub>p 0.887</sub> |
| extractive_conditioned &minus; extractive_generic | regret now | EM | **-0.260 [-0.385, -0.135]** <sub>p 0.000</sub> | **-0.250 [-0.365, -0.135]** <sub>p 0.000</sub> | **-0.146 [-0.260, -0.031]** <sub>p 0.015</sub> | +0.052 [-0.010, +0.115] <sub>p 0.145</sub> |
| extractive_conditioned &minus; extractive_generic | regret now | F1 | **-0.374 [-0.477, -0.271]** <sub>p 0.000</sub> | **-0.325 [-0.425, -0.227]** <sub>p 0.000</sub> | **-0.222 [-0.324, -0.119]** <sub>p 0.000</sub> | -0.004 [-0.049, +0.040] <sub>p 0.862</sub> |
| extractive_conditioned &minus; extractive_generic | regret future | judge | **+0.198 [+0.108, +0.292]** <sub>p 0.000</sub> | **+0.142 [+0.066, +0.215]** <sub>p 0.001</sub> | **+0.156 [+0.052, +0.253]** <sub>p 0.003</sub> | **+0.083 [+0.024, +0.146]** <sub>p 0.009</sub> |
| extractive_conditioned &minus; extractive_generic | regret future | EM | **+0.205 [+0.139, +0.281]** <sub>p 0.000</sub> | **+0.167 [+0.090, +0.243]** <sub>p 0.000</sub> | **+0.264 [+0.188, +0.340]** <sub>p 0.000</sub> | **+0.118 [+0.059, +0.177]** <sub>p 0.000</sub> |
| extractive_conditioned &minus; extractive_generic | regret future | F1 | **+0.198 [+0.140, +0.266]** <sub>p 0.000</sub> | **+0.157 [+0.098, +0.216]** <sub>p 0.000</sub> | **+0.236 [+0.147, +0.318]** <sub>p 0.000</sub> | **+0.098 [+0.045, +0.156]** <sub>p 0.001</sub> |
| extractive_conditioned &minus; extractive_generic | retained now | judge | **+0.434 [+0.306, +0.552]** <sub>p 0.000</sub> | **+0.444 [+0.299, +0.573]** <sub>p 0.000</sub> | **+0.340 [+0.243, +0.438]** <sub>p 0.000</sub> | +0.024 [-0.045, +0.104] <sub>p 0.550</sub> |
| extractive_conditioned &minus; extractive_generic | retained now | EM | **+0.424 [+0.271, +0.576]** <sub>p 0.000</sub> | **+0.347 [+0.184, +0.503]** <sub>p 0.000</sub> | **+0.288 [+0.167, +0.420]** <sub>p 0.000</sub> | +0.003 [-0.094, +0.118] <sub>p 0.970</sub> |
| extractive_conditioned &minus; extractive_generic | retained now | F1 | **+0.434 [+0.323, +0.539]** <sub>p 0.000</sub> | **+0.388 [+0.257, +0.506]** <sub>p 0.000</sub> | **+0.260 [+0.163, +0.354]** <sub>p 0.000</sub> | +0.011 [-0.041, +0.062] <sub>p 0.661</sub> |
| extractive_conditioned &minus; extractive_generic | retained future | judge | **-0.207 [-0.299, -0.122]** <sub>p 0.000</sub> | **-0.137 [-0.212, -0.057]** <sub>p 0.001</sub> | **-0.158 [-0.267, -0.036]** <sub>p 0.007</sub> | **-0.082 [-0.144, -0.023]** <sub>p 0.010</sub> |
| extractive_conditioned &minus; extractive_generic | retained future | EM | **-0.243 [-0.354, -0.139]** <sub>p 0.000</sub> | **-0.234 [-0.344, -0.134]** <sub>p 0.000</sub> | **-0.276 [-0.403, -0.153]** <sub>p 0.000</sub> | **-0.141 [-0.210, -0.068]** <sub>p 0.000</sub> |
| extractive_conditioned &minus; extractive_generic | retained future | F1 | **-0.178 [-0.253, -0.112]** <sub>p 0.000</sub> | **-0.152 [-0.215, -0.089]** <sub>p 0.000</sub> | **-0.230 [-0.335, -0.128]** <sub>p 0.000</sub> | **-0.100 [-0.160, -0.044]** <sub>p 0.001</sub> |
| extractive_conditioned &minus; extractive_generic | gap DiD | judge | **+0.583 [+0.500, +0.667]** <sub>p 0.000</sub> | **+0.507 [+0.389, +0.625]** <sub>p 0.000</sub> | **+0.448 [+0.354, +0.542]** <sub>p 0.000</sub> | **+0.094 [+0.024, +0.163]** <sub>p 0.009</sub> |
| extractive_conditioned &minus; extractive_generic | gap DiD | EM | **+0.465 [+0.372, +0.562]** <sub>p 0.000</sub> | **+0.417 [+0.306, +0.528]** <sub>p 0.000</sub> | **+0.410 [+0.309, +0.510]** <sub>p 0.000</sub> | **+0.066 [+0.003, +0.132]** <sub>p 0.050</sub> |
| extractive_conditioned &minus; extractive_generic | gap DiD | F1 | **+0.572 [+0.493, +0.650]** <sub>p 0.000</sub> | **+0.482 [+0.381, +0.580]** <sub>p 0.000</sub> | **+0.458 [+0.378, +0.538]** <sub>p 0.000</sub> | **+0.102 [+0.050, +0.155]** <sub>p 0.000</sub> |

### Relation dossiers &mdash; contrasts

| comparison | quantity | metric | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|---:|
| conditioned &minus; generic | utility now | judge | **+0.859 [+0.797, +0.922]** <sub>p 0.000</sub> | **+0.641 [+0.547, +0.719]** <sub>p 0.000</sub> | **+0.516 [+0.406, +0.609]** <sub>p 0.000</sub> | **+0.359 [+0.250, +0.469]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility now | EM | **+0.703 [+0.609, +0.781]** <sub>p 0.000</sub> | **+0.531 [+0.438, +0.609]** <sub>p 0.000</sub> | **+0.516 [+0.406, +0.625]** <sub>p 0.000</sub> | **+0.344 [+0.250, +0.438]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility now | F1 | **+0.732 [+0.679, +0.783]** <sub>p 0.000</sub> | **+0.567 [+0.492, +0.637]** <sub>p 0.000</sub> | **+0.498 [+0.386, +0.605]** <sub>p 0.000</sub> | **+0.308 [+0.222, +0.398]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility future | judge | -0.026 [-0.051, +0.001] <sub>p 0.055</sub> | **-0.110 [-0.152, -0.070]** <sub>p 0.000</sub> | **-0.198 [-0.257, -0.136]** <sub>p 0.000</sub> | **-0.342 [-0.396, -0.292]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility future | EM | -0.012 [-0.039, +0.014] <sub>p 0.373</sub> | **-0.074 [-0.111, -0.042]** <sub>p 0.000</sub> | **-0.081 [-0.142, -0.023]** <sub>p 0.007</sub> | **-0.255 [-0.326, -0.193]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility future | F1 | **-0.040 [-0.073, -0.007]** <sub>p 0.017</sub> | **-0.109 [-0.148, -0.074]** <sub>p 0.000</sub> | **-0.158 [-0.209, -0.107]** <sub>p 0.000</sub> | **-0.322 [-0.379, -0.271]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility gap | judge | **+0.885 [+0.834, +0.940]** <sub>p 0.000</sub> | **+0.751 [+0.688, +0.812]** <sub>p 0.000</sub> | **+0.714 [+0.637, +0.777]** <sub>p 0.000</sub> | **+0.701 [+0.613, +0.786]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility gap | EM | **+0.716 [+0.621, +0.806]** <sub>p 0.000</sub> | **+0.605 [+0.543, +0.669]** <sub>p 0.000</sub> | **+0.597 [+0.501, +0.684]** <sub>p 0.000</sub> | **+0.599 [+0.519, +0.678]** <sub>p 0.000</sub> |
| conditioned &minus; generic | utility gap | F1 | **+0.772 [+0.717, +0.825]** <sub>p 0.000</sub> | **+0.676 [+0.630, +0.725]** <sub>p 0.000</sub> | **+0.657 [+0.572, +0.738]** <sub>p 0.000</sub> | **+0.629 [+0.557, +0.702]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret now | judge | **-0.859 [-0.922, -0.797]** <sub>p 0.000</sub> | **-0.641 [-0.719, -0.547]** <sub>p 0.000</sub> | **-0.516 [-0.609, -0.406]** <sub>p 0.000</sub> | **-0.359 [-0.469, -0.250]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret now | EM | **-0.703 [-0.781, -0.609]** <sub>p 0.000</sub> | **-0.531 [-0.609, -0.438]** <sub>p 0.000</sub> | **-0.516 [-0.625, -0.406]** <sub>p 0.000</sub> | **-0.344 [-0.438, -0.250]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret now | F1 | **-0.732 [-0.783, -0.679]** <sub>p 0.000</sub> | **-0.567 [-0.637, -0.492]** <sub>p 0.000</sub> | **-0.498 [-0.605, -0.386]** <sub>p 0.000</sub> | **-0.308 [-0.398, -0.222]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret future | judge | +0.026 [-0.001, +0.051] <sub>p 0.047</sub> | **+0.110 [+0.070, +0.152]** <sub>p 0.000</sub> | **+0.198 [+0.136, +0.257]** <sub>p 0.000</sub> | **+0.342 [+0.292, +0.396]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret future | EM | +0.013 [-0.014, +0.039] <sub>p 0.360</sub> | **+0.074 [+0.042, +0.111]** <sub>p 0.000</sub> | **+0.081 [+0.023, +0.142]** <sub>p 0.006</sub> | **+0.255 [+0.193, +0.326]** <sub>p 0.000</sub> |
| conditioned &minus; generic | regret future | F1 | **+0.040 [+0.007, +0.073]** <sub>p 0.017</sub> | **+0.109 [+0.074, +0.148]** <sub>p 0.000</sub> | **+0.158 [+0.107, +0.209]** <sub>p 0.000</sub> | **+0.322 [+0.271, +0.379]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained now | judge | **+0.859 [+0.797, +0.922]** <sub>p 0.000</sub> | **+0.651 [+0.562, +0.734]** <sub>p 0.000</sub> | **+0.505 [+0.401, +0.609]** <sub>p 0.000</sub> | **+0.365 [+0.260, +0.469]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained now | EM | **+0.880 [+0.797, +0.953]** <sub>p 0.000</sub> | **+0.677 [+0.573, +0.776]** <sub>p 0.000</sub> | **+0.672 [+0.510, +0.823]** <sub>p 0.000</sub> | **+0.453 [+0.323, +0.583]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained now | F1 | **+0.744 [+0.695, +0.788]** <sub>p 0.000</sub> | **+0.579 [+0.502, +0.649]** <sub>p 0.000</sub> | **+0.509 [+0.395, +0.613]** <sub>p 0.000</sub> | **+0.316 [+0.230, +0.409]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained future | judge | **-0.027 [-0.052, -0.000]** <sub>p 0.039</sub> | **-0.113 [-0.154, -0.074]** <sub>p 0.000</sub> | **-0.201 [-0.260, -0.138]** <sub>p 0.000</sub> | **-0.340 [-0.393, -0.291]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained future | EM | -0.014 [-0.043, +0.016] <sub>p 0.382</sub> | **-0.079 [-0.125, -0.041]** <sub>p 0.001</sub> | **-0.092 [-0.159, -0.026]** <sub>p 0.006</sub> | **-0.291 [-0.365, -0.228]** <sub>p 0.000</sub> |
| conditioned &minus; generic | retained future | F1 | **-0.039 [-0.073, -0.006]** <sub>p 0.022</sub> | **-0.113 [-0.151, -0.078]** <sub>p 0.000</sub> | **-0.160 [-0.212, -0.108]** <sub>p 0.000</sub> | **-0.323 [-0.381, -0.273]** <sub>p 0.000</sub> |
| conditioned &minus; generic | gap DiD | judge | **+0.885 [+0.834, +0.940]** <sub>p 0.000</sub> | **+0.751 [+0.688, +0.811]** <sub>p 0.000</sub> | **+0.714 [+0.639, +0.777]** <sub>p 0.000</sub> | **+0.701 [+0.613, +0.786]** <sub>p 0.000</sub> |
| conditioned &minus; generic | gap DiD | EM | **+0.716 [+0.621, +0.804]** <sub>p 0.000</sub> | **+0.605 [+0.542, +0.669]** <sub>p 0.000</sub> | **+0.597 [+0.501, +0.682]** <sub>p 0.000</sub> | **+0.599 [+0.519, +0.677]** <sub>p 0.000</sub> |
| conditioned &minus; generic | gap DiD | F1 | **+0.772 [+0.717, +0.826]** <sub>p 0.000</sub> | **+0.676 [+0.628, +0.723]** <sub>p 0.000</sub> | **+0.657 [+0.571, +0.736]** <sub>p 0.000</sub> | **+0.629 [+0.557, +0.701]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | utility now | judge | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | -0.016 [-0.047, +0.000] <sub>p 0.621</sub> | -0.016 [-0.047, +0.000] <sub>p 0.622</sub> | +0.016 [-0.031, +0.062] <sub>p 0.767</sub> |
| conditioned &minus; reusable | utility now | EM | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | -0.031 [-0.078, +0.000] <sub>p 0.251</sub> | +0.000 [-0.047, +0.047] <sub>p 1.000</sub> |
| conditioned &minus; reusable | utility now | F1 | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | -0.012 [-0.033, +0.000] <sub>p 0.225</sub> | -0.013 [-0.052, +0.018] <sub>p 0.518</sub> | -0.003 [-0.050, +0.044] <sub>p 0.939</sub> |
| conditioned &minus; reusable | utility future | judge | -0.004 [-0.011, +0.004] <sub>p 0.328</sub> | -0.007 [-0.021, +0.006] <sub>p 0.321</sub> | **-0.080 [-0.098, -0.064]** <sub>p 0.000</sub> | **-0.167 [-0.190, -0.143]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | utility future | EM | -0.003 [-0.007, +0.001] <sub>p 0.205</sub> | -0.003 [-0.014, +0.007] <sub>p 0.571</sub> | **-0.068 [-0.081, -0.053]** <sub>p 0.000</sub> | **-0.134 [-0.157, -0.110]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | utility future | F1 | -0.003 [-0.012, +0.005] <sub>p 0.495</sub> | -0.008 [-0.018, +0.004] <sub>p 0.183</sub> | **-0.069 [-0.085, -0.052]** <sub>p 0.000</sub> | **-0.152 [-0.175, -0.129]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | utility gap | judge | +0.004 [-0.004, +0.011] <sub>p 0.298</sub> | -0.008 [-0.047, +0.018] <sub>p 0.632</sub> | **+0.065 [+0.026, +0.094]** <sub>p 0.002</sub> | **+0.182 [+0.121, +0.246]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | utility gap | EM | +0.003 [-0.001, +0.007] <sub>p 0.233</sub> | +0.003 [-0.007, +0.014] <sub>p 0.609</sub> | +0.036 [-0.010, +0.073] <sub>p 0.082</sub> | **+0.134 [+0.084, +0.182]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | utility gap | F1 | +0.003 [-0.005, +0.012] <sub>p 0.495</sub> | -0.005 [-0.024, +0.011] <sub>p 0.595</sub> | **+0.056 [+0.016, +0.087]** <sub>p 0.004</sub> | **+0.149 [+0.096, +0.203]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | regret now | judge | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.016 [+0.000, +0.047] <sub>p 0.621</sub> | +0.016 [+0.000, +0.047] <sub>p 0.622</sub> | -0.016 [-0.062, +0.031] <sub>p 0.767</sub> |
| conditioned &minus; reusable | regret now | EM | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.031 [+0.000, +0.078] <sub>p 0.251</sub> | +0.000 [-0.047, +0.047] <sub>p 1.000</sub> |
| conditioned &minus; reusable | regret now | F1 | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.013 [+0.000, +0.033] <sub>p 0.225</sub> | +0.013 [-0.018, +0.052] <sub>p 0.518</sub> | +0.003 [-0.044, +0.050] <sub>p 0.940</sub> |
| conditioned &minus; reusable | regret future | judge | +0.004 [-0.004, +0.011] <sub>p 0.298</sub> | +0.007 [-0.006, +0.021] <sub>p 0.299</sub> | **+0.080 [+0.064, +0.098]** <sub>p 0.000</sub> | **+0.167 [+0.143, +0.190]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | regret future | EM | +0.003 [-0.001, +0.007] <sub>p 0.134</sub> | +0.003 [-0.007, +0.014] <sub>p 0.528</sub> | **+0.068 [+0.053, +0.081]** <sub>p 0.000</sub> | **+0.134 [+0.110, +0.157]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | regret future | F1 | +0.003 [-0.005, +0.012] <sub>p 0.495</sub> | +0.008 [-0.004, +0.018] <sub>p 0.183</sub> | **+0.069 [+0.052, +0.085]** <sub>p 0.000</sub> | **+0.152 [+0.129, +0.175]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | retained now | judge | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | -0.016 [-0.047, +0.000] <sub>p 0.622</sub> | +0.000 [-0.047, +0.047] <sub>p 1.000</sub> |
| conditioned &minus; reusable | retained now | EM | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | -0.042 [-0.104, +0.000] <sub>p 0.251</sub> | +0.000 [-0.047, +0.047] <sub>p 1.000</sub> |
| conditioned &minus; reusable | retained now | F1 | +0.000 [+0.000, +0.000] <sub>p 1.000</sub> | -0.012 [-0.033, +0.000] <sub>p 0.225</sub> | -0.015 [-0.054, +0.016] <sub>p 0.518</sub> | -0.003 [-0.050, +0.044] <sub>p 0.939</sub> |
| conditioned &minus; reusable | retained future | judge | -0.004 [-0.011, +0.004] <sub>p 0.328</sub> | -0.007 [-0.021, +0.006] <sub>p 0.321</sub> | **-0.080 [-0.098, -0.064]** <sub>p 0.000</sub> | **-0.166 [-0.189, -0.143]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | retained future | EM | -0.004 [-0.009, +0.001] <sub>p 0.187</sub> | -0.005 [-0.016, +0.006] <sub>p 0.413</sub> | **-0.079 [-0.094, -0.062]** <sub>p 0.000</sub> | **-0.157 [-0.184, -0.130]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | retained future | F1 | -0.003 [-0.011, +0.006] <sub>p 0.565</sub> | -0.008 [-0.019, +0.003] <sub>p 0.155</sub> | **-0.070 [-0.087, -0.052]** <sub>p 0.000</sub> | **-0.154 [-0.176, -0.131]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | gap DiD | judge | +0.004 [-0.004, +0.011] <sub>p 0.296</sub> | -0.008 [-0.046, +0.019] <sub>p 0.624</sub> | **+0.065 [+0.027, +0.094]** <sub>p 0.001</sub> | **+0.182 [+0.121, +0.246]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | gap DiD | EM | +0.003 [-0.001, +0.007] <sub>p 0.228</sub> | +0.003 [-0.007, +0.015] <sub>p 0.611</sub> | +0.036 [-0.009, +0.073] <sub>p 0.077</sub> | **+0.134 [+0.083, +0.183]** <sub>p 0.000</sub> |
| conditioned &minus; reusable | gap DiD | F1 | +0.003 [-0.005, +0.012] <sub>p 0.497</sub> | -0.005 [-0.024, +0.012] <sub>p 0.584</sub> | **+0.056 [+0.017, +0.086]** <sub>p 0.004</sub> | **+0.149 [+0.094, +0.203]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility now | judge | **+0.859 [+0.797, +0.922]** <sub>p 0.000</sub> | **+0.656 [+0.562, +0.734]** <sub>p 0.000</sub> | **+0.531 [+0.453, +0.609]** <sub>p 0.000</sub> | **+0.344 [+0.234, +0.453]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility now | EM | **+0.703 [+0.609, +0.781]** <sub>p 0.000</sub> | **+0.531 [+0.438, +0.609]** <sub>p 0.000</sub> | **+0.547 [+0.453, +0.641]** <sub>p 0.000</sub> | **+0.344 [+0.250, +0.438]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility now | F1 | **+0.732 [+0.679, +0.783]** <sub>p 0.000</sub> | **+0.580 [+0.501, +0.652]** <sub>p 0.000</sub> | **+0.511 [+0.414, +0.606]** <sub>p 0.000</sub> | **+0.311 [+0.223, +0.405]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility future | judge | -0.022 [-0.045, +0.002] <sub>p 0.077</sub> | **-0.103 [-0.141, -0.066]** <sub>p 0.000</sub> | **-0.118 [-0.168, -0.065]** <sub>p 0.000</sub> | **-0.175 [-0.241, -0.115]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility future | EM | -0.009 [-0.033, +0.015] <sub>p 0.475</sub> | **-0.071 [-0.108, -0.039]** <sub>p 0.000</sub> | -0.014 [-0.073, +0.044] <sub>p 0.660</sub> | **-0.121 [-0.202, -0.050]** <sub>p 0.002</sub> |
| reusable &minus; generic | utility future | F1 | **-0.036 [-0.067, -0.008]** <sub>p 0.014</sub> | **-0.101 [-0.139, -0.068]** <sub>p 0.000</sub> | **-0.089 [-0.137, -0.042]** <sub>p 0.000</sub> | **-0.169 [-0.237, -0.110]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility gap | judge | **+0.881 [+0.831, +0.933]** <sub>p 0.000</sub> | **+0.759 [+0.700, +0.815]** <sub>p 0.000</sub> | **+0.649 [+0.594, +0.698]** <sub>p 0.000</sub> | **+0.519 [+0.430, +0.607]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility gap | EM | **+0.712 [+0.619, +0.801]** <sub>p 0.000</sub> | **+0.602 [+0.542, +0.665]** <sub>p 0.000</sub> | **+0.560 [+0.487, +0.629]** <sub>p 0.000</sub> | **+0.465 [+0.391, +0.539]** <sub>p 0.000</sub> |
| reusable &minus; generic | utility gap | F1 | **+0.769 [+0.718, +0.819]** <sub>p 0.000</sub> | **+0.681 [+0.634, +0.727]** <sub>p 0.000</sub> | **+0.601 [+0.529, +0.671]** <sub>p 0.000</sub> | **+0.480 [+0.409, +0.551]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret now | judge | **-0.859 [-0.922, -0.797]** <sub>p 0.000</sub> | **-0.656 [-0.734, -0.562]** <sub>p 0.000</sub> | **-0.531 [-0.609, -0.453]** <sub>p 0.000</sub> | **-0.344 [-0.453, -0.234]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret now | EM | **-0.703 [-0.781, -0.609]** <sub>p 0.000</sub> | **-0.531 [-0.609, -0.438]** <sub>p 0.000</sub> | **-0.547 [-0.641, -0.453]** <sub>p 0.000</sub> | **-0.344 [-0.438, -0.250]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret now | F1 | **-0.732 [-0.783, -0.679]** <sub>p 0.000</sub> | **-0.580 [-0.652, -0.501]** <sub>p 0.000</sub> | **-0.511 [-0.606, -0.414]** <sub>p 0.000</sub> | **-0.311 [-0.405, -0.223]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret future | judge | +0.022 [-0.002, +0.045] <sub>p 0.067</sub> | **+0.103 [+0.066, +0.141]** <sub>p 0.000</sub> | **+0.118 [+0.065, +0.168]** <sub>p 0.000</sub> | **+0.175 [+0.115, +0.241]** <sub>p 0.000</sub> |
| reusable &minus; generic | regret future | EM | +0.009 [-0.015, +0.033] <sub>p 0.467</sub> | **+0.071 [+0.039, +0.108]** <sub>p 0.000</sub> | +0.014 [-0.044, +0.073] <sub>p 0.638</sub> | **+0.121 [+0.050, +0.202]** <sub>p 0.002</sub> |
| reusable &minus; generic | regret future | F1 | **+0.036 [+0.008, +0.067]** <sub>p 0.014</sub> | **+0.101 [+0.068, +0.139]** <sub>p 0.000</sub> | **+0.089 [+0.042, +0.137]** <sub>p 0.000</sub> | **+0.169 [+0.110, +0.237]** <sub>p 0.000</sub> |
| reusable &minus; generic | retained now | judge | **+0.859 [+0.797, +0.922]** <sub>p 0.000</sub> | **+0.651 [+0.562, +0.734]** <sub>p 0.000</sub> | **+0.521 [+0.432, +0.609]** <sub>p 0.000</sub> | **+0.365 [+0.260, +0.469]** <sub>p 0.000</sub> |
| reusable &minus; generic | retained now | EM | **+0.880 [+0.797, +0.953]** <sub>p 0.000</sub> | **+0.677 [+0.573, +0.776]** <sub>p 0.000</sub> | **+0.714 [+0.573, +0.849]** <sub>p 0.000</sub> | **+0.453 [+0.323, +0.578]** <sub>p 0.000</sub> |
| reusable &minus; generic | retained now | F1 | **+0.744 [+0.695, +0.788]** <sub>p 0.000</sub> | **+0.591 [+0.511, +0.662]** <sub>p 0.000</sub> | **+0.524 [+0.420, +0.622]** <sub>p 0.000</sub> | **+0.319 [+0.228, +0.414]** <sub>p 0.000</sub> |
| reusable &minus; generic | retained future | judge | -0.023 [-0.045, +0.001] <sub>p 0.052</sub> | **-0.106 [-0.143, -0.068]** <sub>p 0.000</sub> | **-0.121 [-0.172, -0.067]** <sub>p 0.000</sub> | **-0.174 [-0.240, -0.114]** <sub>p 0.000</sub> |
| reusable &minus; generic | retained future | EM | -0.010 [-0.036, +0.017] <sub>p 0.476</sub> | **-0.074 [-0.120, -0.035]** <sub>p 0.002</sub> | -0.013 [-0.082, +0.053] <sub>p 0.690</sub> | **-0.133 [-0.224, -0.054]** <sub>p 0.003</sub> |
| reusable &minus; generic | retained future | F1 | **-0.036 [-0.068, -0.007]** <sub>p 0.019</sub> | **-0.104 [-0.142, -0.072]** <sub>p 0.000</sub> | **-0.091 [-0.140, -0.042]** <sub>p 0.000</sub> | **-0.169 [-0.237, -0.110]** <sub>p 0.000</sub> |
| reusable &minus; generic | gap DiD | judge | **+0.881 [+0.831, +0.934]** <sub>p 0.000</sub> | **+0.759 [+0.700, +0.813]** <sub>p 0.000</sub> | **+0.649 [+0.594, +0.697]** <sub>p 0.000</sub> | **+0.519 [+0.430, +0.608]** <sub>p 0.000</sub> |
| reusable &minus; generic | gap DiD | EM | **+0.712 [+0.620, +0.800]** <sub>p 0.000</sub> | **+0.602 [+0.541, +0.664]** <sub>p 0.000</sub> | **+0.560 [+0.487, +0.629]** <sub>p 0.000</sub> | **+0.465 [+0.390, +0.535]** <sub>p 0.000</sub> |
| reusable &minus; generic | gap DiD | F1 | **+0.769 [+0.718, +0.820]** <sub>p 0.000</sub> | **+0.681 [+0.633, +0.726]** <sub>p 0.000</sub> | **+0.601 [+0.527, +0.669]** <sub>p 0.000</sub> | **+0.480 [+0.410, +0.548]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility now | judge | **-0.750 [-0.750, -0.750]** <sub>p 0.000</sub> | **-0.469 [-0.547, -0.391]** <sub>p 0.000</sub> | **-0.234 [-0.328, -0.141]** <sub>p 0.000</sub> | -0.078 [-0.172, +0.000] <sub>p 0.100</sub> |
| oracle &minus; conditioned | utility now | EM | **-0.531 [-0.609, -0.453]** <sub>p 0.000</sub> | **-0.438 [-0.516, -0.344]** <sub>p 0.000</sub> | **-0.219 [-0.312, -0.125]** <sub>p 0.000</sub> | **-0.094 [-0.188, -0.016]** <sub>p 0.049</sub> |
| oracle &minus; conditioned | utility now | F1 | **-0.642 [-0.682, -0.598]** <sub>p 0.000</sub> | **-0.451 [-0.528, -0.370]** <sub>p 0.000</sub> | **-0.231 [-0.317, -0.145]** <sub>p 0.000</sub> | -0.080 [-0.172, +0.008] <sub>p 0.083</sub> |
| oracle &minus; conditioned | utility future | judge | **+0.102 [+0.084, +0.121]** <sub>p 0.000</sub> | **+0.303 [+0.263, +0.344]** <sub>p 0.000</sub> | **+0.575 [+0.506, +0.635]** <sub>p 0.000</sub> | **+0.660 [+0.583, +0.731]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility future | EM | **+0.101 [+0.075, +0.126]** <sub>p 0.000</sub> | **+0.230 [+0.186, +0.275]** <sub>p 0.000</sub> | **+0.445 [+0.376, +0.512]** <sub>p 0.000</sub> | **+0.530 [+0.444, +0.614]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility future | F1 | **+0.110 [+0.089, +0.130]** <sub>p 0.000</sub> | **+0.269 [+0.230, +0.308]** <sub>p 0.000</sub> | **+0.511 [+0.452, +0.567]** <sub>p 0.000</sub> | **+0.600 [+0.529, +0.668]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility gap | judge | **-0.852 [-0.871, -0.834]** <sub>p 0.000</sub> | **-0.772 [-0.842, -0.699]** <sub>p 0.000</sub> | **-0.809 [-0.859, -0.753]** <sub>p 0.000</sub> | **-0.739 [-0.794, -0.675]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility gap | EM | **-0.632 [-0.706, -0.552]** <sub>p 0.000</sub> | **-0.668 [-0.743, -0.594]** <sub>p 0.000</sub> | **-0.664 [-0.744, -0.576]** <sub>p 0.000</sub> | **-0.624 [-0.694, -0.552]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | utility gap | F1 | **-0.752 [-0.798, -0.700]** <sub>p 0.000</sub> | **-0.720 [-0.783, -0.659]** <sub>p 0.000</sub> | **-0.742 [-0.799, -0.683]** <sub>p 0.000</sub> | **-0.680 [-0.742, -0.617]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | regret now | judge | **+0.750 [+0.750, +0.750]** <sub>p 0.000</sub> | **+0.469 [+0.391, +0.547]** <sub>p 0.000</sub> | **+0.234 [+0.141, +0.328]** <sub>p 0.000</sub> | +0.078 [+0.000, +0.172] <sub>p 0.100</sub> |
| oracle &minus; conditioned | regret now | EM | **+0.531 [+0.453, +0.609]** <sub>p 0.000</sub> | **+0.438 [+0.344, +0.516]** <sub>p 0.000</sub> | **+0.219 [+0.125, +0.312]** <sub>p 0.000</sub> | **+0.094 [+0.016, +0.188]** <sub>p 0.049</sub> |
| oracle &minus; conditioned | regret now | F1 | **+0.642 [+0.598, +0.682]** <sub>p 0.000</sub> | **+0.451 [+0.370, +0.528]** <sub>p 0.000</sub> | **+0.231 [+0.145, +0.317]** <sub>p 0.000</sub> | +0.080 [-0.008, +0.172] <sub>p 0.083</sub> |
| oracle &minus; conditioned | regret future | judge | **-0.102 [-0.121, -0.084]** <sub>p 0.000</sub> | **-0.303 [-0.344, -0.263]** <sub>p 0.000</sub> | **-0.575 [-0.635, -0.506]** <sub>p 0.000</sub> | **-0.660 [-0.731, -0.583]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | regret future | EM | **-0.101 [-0.126, -0.075]** <sub>p 0.000</sub> | **-0.230 [-0.275, -0.186]** <sub>p 0.000</sub> | **-0.445 [-0.513, -0.376]** <sub>p 0.000</sub> | **-0.530 [-0.614, -0.444]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | regret future | F1 | **-0.110 [-0.130, -0.089]** <sub>p 0.000</sub> | **-0.269 [-0.308, -0.230]** <sub>p 0.000</sub> | **-0.511 [-0.567, -0.452]** <sub>p 0.000</sub> | **-0.600 [-0.668, -0.529]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | retained now | judge | **-0.745 [-0.750, -0.734]** <sub>p 0.000</sub> | **-0.495 [-0.578, -0.406]** <sub>p 0.000</sub> | **-0.219 [-0.312, -0.125]** <sub>p 0.000</sub> | -0.068 [-0.141, +0.000] <sub>p 0.087</sub> |
| oracle &minus; conditioned | retained now | EM | **-0.667 [-0.734, -0.599]** <sub>p 0.000</sub> | **-0.568 [-0.667, -0.458]** <sub>p 0.000</sub> | **-0.271 [-0.385, -0.146]** <sub>p 0.000</sub> | **-0.135 [-0.255, -0.026]** <sub>p 0.021</sub> |
| oracle &minus; conditioned | retained now | F1 | **-0.651 [-0.685, -0.617]** <sub>p 0.000</sub> | **-0.456 [-0.531, -0.379]** <sub>p 0.000</sub> | **-0.236 [-0.322, -0.150]** <sub>p 0.000</sub> | -0.082 [-0.173, +0.005] <sub>p 0.077</sub> |
| oracle &minus; conditioned | retained future | judge | **+0.104 [+0.085, +0.123]** <sub>p 0.000</sub> | **+0.303 [+0.263, +0.344]** <sub>p 0.000</sub> | **+0.582 [+0.510, +0.648]** <sub>p 0.000</sub> | **+0.661 [+0.585, +0.731]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | retained future | EM | **+0.111 [+0.080, +0.140]** <sub>p 0.000</sub> | **+0.263 [+0.211, +0.312]** <sub>p 0.000</sub> | **+0.520 [+0.444, +0.591]** <sub>p 0.000</sub> | **+0.615 [+0.520, +0.705]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | retained future | F1 | **+0.112 [+0.092, +0.133]** <sub>p 0.000</sub> | **+0.274 [+0.235, +0.312]** <sub>p 0.000</sub> | **+0.518 [+0.458, +0.576]** <sub>p 0.000</sub> | **+0.604 [+0.534, +0.671]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | gap DiD | judge | **-0.852 [-0.871, -0.834]** <sub>p 0.000</sub> | **-0.772 [-0.841, -0.702]** <sub>p 0.000</sub> | **-0.809 [-0.860, -0.752]** <sub>p 0.000</sub> | **-0.739 [-0.793, -0.676]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | gap DiD | EM | **-0.632 [-0.708, -0.550]** <sub>p 0.000</sub> | **-0.668 [-0.742, -0.594]** <sub>p 0.000</sub> | **-0.664 [-0.741, -0.574]** <sub>p 0.000</sub> | **-0.624 [-0.691, -0.551]** <sub>p 0.000</sub> |
| oracle &minus; conditioned | gap DiD | F1 | **-0.752 [-0.799, -0.699]** <sub>p 0.000</sub> | **-0.720 [-0.783, -0.661]** <sub>p 0.000</sub> | **-0.742 [-0.797, -0.683]** <sub>p 0.000</sub> | **-0.680 [-0.739, -0.617]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility now | judge | **+0.109 [+0.047, +0.172]** <sub>p 0.001</sub> | **+0.172 [+0.047, +0.297]** <sub>p 0.011</sub> | **+0.281 [+0.188, +0.375]** <sub>p 0.000</sub> | **+0.281 [+0.141, +0.422]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility now | EM | **+0.172 [+0.109, +0.219]** <sub>p 0.000</sub> | +0.094 [+0.000, +0.203] <sub>p 0.106</sub> | **+0.297 [+0.188, +0.391]** <sub>p 0.000</sub> | **+0.250 [+0.125, +0.375]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility now | F1 | **+0.090 [+0.054, +0.127]** <sub>p 0.000</sub> | **+0.116 [+0.017, +0.215]** <sub>p 0.020</sub> | **+0.267 [+0.156, +0.370]** <sub>p 0.000</sub> | **+0.228 [+0.114, +0.346]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility future | judge | **+0.076 [+0.044, +0.112]** <sub>p 0.000</sub> | **+0.193 [+0.134, +0.252]** <sub>p 0.000</sub> | **+0.377 [+0.296, +0.446]** <sub>p 0.000</sub> | **+0.319 [+0.208, +0.422]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility future | EM | **+0.089 [+0.051, +0.124]** <sub>p 0.000</sub> | **+0.156 [+0.097, +0.215]** <sub>p 0.000</sub> | **+0.364 [+0.286, +0.434]** <sub>p 0.000</sub> | **+0.275 [+0.168, +0.378]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility future | F1 | **+0.070 [+0.032, +0.109]** <sub>p 0.000</sub> | **+0.160 [+0.115, +0.205]** <sub>p 0.000</sub> | **+0.352 [+0.285, +0.414]** <sub>p 0.000</sub> | **+0.278 [+0.180, +0.373]** <sub>p 0.000</sub> |
| oracle &minus; generic | utility gap | judge | +0.033 [-0.021, +0.092] <sub>p 0.287</sub> | -0.021 [-0.104, +0.062] <sub>p 0.618</sub> | **-0.096 [-0.163, -0.029]** <sub>p 0.005</sub> | -0.038 [-0.121, +0.046] <sub>p 0.367</sub> |
| oracle &minus; generic | utility gap | EM | **+0.083 [+0.025, +0.142]** <sub>p 0.006</sub> | -0.062 [-0.125, +0.004] <sub>p 0.064</sub> | -0.067 [-0.138, +0.008] <sub>p 0.073</sub> | -0.025 [-0.088, +0.042] <sub>p 0.413</sub> |
| oracle &minus; generic | utility gap | F1 | +0.020 [-0.021, +0.059] <sub>p 0.330</sub> | -0.044 [-0.107, +0.020] <sub>p 0.180</sub> | **-0.085 [-0.163, -0.002]** <sub>p 0.040</sub> | -0.050 [-0.120, +0.021] <sub>p 0.157</sub> |
| oracle &minus; generic | regret now | judge | **-0.109 [-0.172, -0.047]** <sub>p 0.001</sub> | **-0.172 [-0.297, -0.047]** <sub>p 0.011</sub> | **-0.281 [-0.375, -0.188]** <sub>p 0.000</sub> | **-0.281 [-0.422, -0.141]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret now | EM | **-0.172 [-0.219, -0.109]** <sub>p 0.000</sub> | -0.094 [-0.203, +0.000] <sub>p 0.106</sub> | **-0.297 [-0.391, -0.188]** <sub>p 0.000</sub> | **-0.250 [-0.375, -0.125]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret now | F1 | **-0.090 [-0.127, -0.054]** <sub>p 0.000</sub> | **-0.116 [-0.215, -0.017]** <sub>p 0.020</sub> | **-0.267 [-0.370, -0.156]** <sub>p 0.000</sub> | **-0.228 [-0.346, -0.114]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret future | judge | **-0.076 [-0.112, -0.044]** <sub>p 0.000</sub> | **-0.193 [-0.252, -0.134]** <sub>p 0.000</sub> | **-0.377 [-0.446, -0.296]** <sub>p 0.000</sub> | **-0.319 [-0.422, -0.208]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret future | EM | **-0.089 [-0.124, -0.051]** <sub>p 0.000</sub> | **-0.156 [-0.215, -0.097]** <sub>p 0.000</sub> | **-0.364 [-0.434, -0.286]** <sub>p 0.000</sub> | **-0.275 [-0.378, -0.168]** <sub>p 0.000</sub> |
| oracle &minus; generic | regret future | F1 | **-0.070 [-0.109, -0.032]** <sub>p 0.000</sub> | **-0.160 [-0.205, -0.115]** <sub>p 0.000</sub> | **-0.352 [-0.414, -0.285]** <sub>p 0.000</sub> | **-0.278 [-0.373, -0.180]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained now | judge | **+0.115 [+0.047, +0.182]** <sub>p 0.000</sub> | **+0.156 [+0.031, +0.281]** <sub>p 0.023</sub> | **+0.286 [+0.188, +0.375]** <sub>p 0.000</sub> | **+0.297 [+0.172, +0.422]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained now | EM | **+0.214 [+0.135, +0.281]** <sub>p 0.000</sub> | +0.109 [-0.010, +0.234] <sub>p 0.090</sub> | **+0.401 [+0.266, +0.536]** <sub>p 0.000</sub> | **+0.318 [+0.156, +0.479]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained now | F1 | **+0.093 [+0.055, +0.131]** <sub>p 0.000</sub> | **+0.122 [+0.021, +0.225]** <sub>p 0.015</sub> | **+0.272 [+0.159, +0.376]** <sub>p 0.000</sub> | **+0.234 [+0.120, +0.352]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained future | judge | **+0.077 [+0.045, +0.113]** <sub>p 0.000</sub> | **+0.190 [+0.131, +0.250]** <sub>p 0.000</sub> | **+0.381 [+0.300, +0.451]** <sub>p 0.000</sub> | **+0.321 [+0.214, +0.423]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained future | EM | **+0.097 [+0.053, +0.140]** <sub>p 0.000</sub> | **+0.184 [+0.117, +0.248]** <sub>p 0.000</sub> | **+0.428 [+0.341, +0.503]** <sub>p 0.000</sub> | **+0.324 [+0.194, +0.448]** <sub>p 0.000</sub> |
| oracle &minus; generic | retained future | F1 | **+0.073 [+0.034, +0.112]** <sub>p 0.000</sub> | **+0.161 [+0.116, +0.205]** <sub>p 0.000</sub> | **+0.357 [+0.290, +0.419]** <sub>p 0.000</sub> | **+0.281 [+0.182, +0.376]** <sub>p 0.000</sub> |
| oracle &minus; generic | gap DiD | judge | +0.033 [-0.021, +0.092] <sub>p 0.283</sub> | -0.021 [-0.108, +0.062] <sub>p 0.625</sub> | **-0.096 [-0.163, -0.029]** <sub>p 0.005</sub> | -0.038 [-0.121, +0.050] <sub>p 0.361</sub> |
| oracle &minus; generic | gap DiD | EM | **+0.083 [+0.025, +0.142]** <sub>p 0.005</sub> | -0.062 [-0.125, +0.004] <sub>p 0.063</sub> | -0.067 [-0.138, +0.004] <sub>p 0.071</sub> | -0.025 [-0.088, +0.037] <sub>p 0.406</sub> |
| oracle &minus; generic | gap DiD | F1 | +0.020 [-0.021, +0.060] <sub>p 0.328</sub> | -0.044 [-0.108, +0.020] <sub>p 0.173</sub> | **-0.085 [-0.164, -0.005]** <sub>p 0.035</sub> | -0.050 [-0.120, +0.020] <sub>p 0.153</sub> |

## Appendix: every computed level

The same treatment for the non-contrast tables: every policy, every budget, every endpoint and all three utility measures, including the `channel` rows that carry mean delivered length and fill on the same context-clustered footing as utility.

### SQuAD groups &mdash; levels with 95% CIs

| policy | kind | endpoint | metric | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned | utility | now | EM | 0.542 [0.438, 0.656] | 0.542 [0.458, 0.635] | 0.542 [0.458, 0.635] | 0.583 [0.490, 0.688] |
| conditioned | utility | now | F1 | 0.726 [0.649, 0.801] | 0.715 [0.641, 0.786] | 0.702 [0.629, 0.777] | 0.739 [0.662, 0.815] |
| conditioned | utility | now | judge | 0.865 [0.802, 0.917] | 0.823 [0.750, 0.896] | 0.812 [0.750, 0.875] | 0.802 [0.708, 0.885] |
| conditioned | utility | future | EM | 0.069 [0.035, 0.111] | 0.087 [0.052, 0.125] | 0.153 [0.101, 0.212] | 0.274 [0.205, 0.354] |
| conditioned | utility | future | F1 | 0.142 [0.098, 0.191] | 0.184 [0.136, 0.232] | 0.251 [0.200, 0.305] | 0.412 [0.335, 0.492] |
| conditioned | utility | future | judge | 0.177 [0.122, 0.236] | 0.260 [0.188, 0.337] | 0.337 [0.260, 0.413] | 0.528 [0.438, 0.618] |
| conditioned | utility | gap | EM | 0.472 [0.361, 0.587] | 0.455 [0.365, 0.552] | 0.389 [0.285, 0.493] | 0.309 [0.222, 0.406] |
| conditioned | utility | gap | F1 | 0.584 [0.493, 0.674] | 0.531 [0.440, 0.622] | 0.450 [0.359, 0.543] | 0.327 [0.229, 0.429] |
| conditioned | utility | gap | judge | 0.688 [0.601, 0.771] | 0.562 [0.451, 0.670] | 0.476 [0.372, 0.580] | 0.274 [0.153, 0.396] |
| conditioned | regret | now | EM | 0.094 [0.010, 0.177] | 0.094 [0.010, 0.177] | 0.094 [0.021, 0.167] | 0.052 [-0.021, 0.125] |
| conditioned | regret | now | F1 | 0.059 [-0.007, 0.124] | 0.070 [-0.006, 0.149] | 0.083 [0.035, 0.131] | 0.046 [-0.017, 0.107] |
| conditioned | regret | now | judge | 0.010 [-0.052, 0.073] | 0.052 [-0.031, 0.135] | 0.062 [0.010, 0.115] | 0.073 [0.010, 0.135] |
| conditioned | regret | future | EM | 0.566 [0.462, 0.667] | 0.549 [0.441, 0.653] | 0.483 [0.368, 0.590] | 0.361 [0.260, 0.462] |
| conditioned | regret | future | F1 | 0.643 [0.553, 0.730] | 0.601 [0.512, 0.690] | 0.533 [0.437, 0.628] | 0.373 [0.275, 0.469] |
| conditioned | regret | future | judge | 0.698 [0.594, 0.795] | 0.615 [0.507, 0.719] | 0.538 [0.431, 0.642] | 0.347 [0.229, 0.465] |
| conditioned | retained | now | EM | 0.767 [0.656, 0.868] | 0.788 [0.694, 0.875] | 0.757 [0.653, 0.854] | 0.785 [0.674, 0.882] |
| conditioned | retained | now | F1 | 0.785 [0.711, 0.857] | 0.775 [0.702, 0.843] | 0.762 [0.688, 0.834] | 0.796 [0.716, 0.873] |
| conditioned | retained | now | judge | 0.931 [0.878, 0.976] | 0.885 [0.819, 0.944] | 0.910 [0.858, 0.958] | 0.851 [0.764, 0.934] |
| conditioned | retained | future | EM | 0.078 [0.019, 0.156] | 0.085 [0.040, 0.141] | 0.189 [0.118, 0.273] | 0.363 [0.245, 0.493] |
| conditioned | retained | future | F1 | 0.143 [0.096, 0.200] | 0.200 [0.146, 0.256] | 0.261 [0.203, 0.329] | 0.452 [0.364, 0.544] |
| conditioned | retained | future | judge | 0.174 [0.116, 0.233] | 0.276 [0.193, 0.368] | 0.340 [0.259, 0.429] | 0.556 [0.457, 0.655] |
| conditioned | channel | message | delivered words | 18.469 [18.219, 18.708] | 36.656 [36.323, 36.990] | 73.240 [72.146, 74.250] | 143.115 [142.000, 144.229] |
| conditioned | channel | message | fill ratio | 0.923 [0.911, 0.935] | 0.916 [0.908, 0.925] | 0.915 [0.902, 0.928] | 0.894 [0.888, 0.901] |
| conditioned@trim | utility | now | EM | n/a | 0.531 [0.438, 0.625] | n/a | n/a |
| conditioned@trim | utility | now | F1 | n/a | 0.688 [0.620, 0.757] | n/a | n/a |
| conditioned@trim | utility | now | judge | n/a | 0.823 [0.750, 0.896] | n/a | n/a |
| conditioned@trim | utility | future | EM | n/a | 0.069 [0.038, 0.101] | n/a | n/a |
| conditioned@trim | utility | future | F1 | n/a | 0.146 [0.102, 0.190] | n/a | n/a |
| conditioned@trim | utility | future | judge | n/a | 0.198 [0.132, 0.267] | n/a | n/a |
| conditioned@trim | utility | gap | EM | n/a | 0.462 [0.368, 0.559] | n/a | n/a |
| conditioned@trim | utility | gap | F1 | n/a | 0.542 [0.466, 0.617] | n/a | n/a |
| conditioned@trim | utility | gap | judge | n/a | 0.625 [0.535, 0.715] | n/a | n/a |
| conditioned@trim | regret | now | EM | n/a | 0.104 [0.010, 0.198] | n/a | n/a |
| conditioned@trim | regret | now | F1 | n/a | 0.097 [0.022, 0.172] | n/a | n/a |
| conditioned@trim | regret | now | judge | n/a | 0.052 [-0.042, 0.135] | n/a | n/a |
| conditioned@trim | regret | future | EM | n/a | 0.566 [0.462, 0.667] | n/a | n/a |
| conditioned@trim | regret | future | F1 | n/a | 0.639 [0.552, 0.723] | n/a | n/a |
| conditioned@trim | regret | future | judge | n/a | 0.677 [0.576, 0.774] | n/a | n/a |
| conditioned@trim | retained | now | EM | n/a | 0.771 [0.667, 0.861] | n/a | n/a |
| conditioned@trim | retained | now | F1 | n/a | 0.755 [0.689, 0.819] | n/a | n/a |
| conditioned@trim | retained | now | judge | n/a | 0.885 [0.819, 0.944] | n/a | n/a |
| conditioned@trim | retained | future | EM | n/a | 0.069 [0.030, 0.116] | n/a | n/a |
| conditioned@trim | retained | future | F1 | n/a | 0.153 [0.107, 0.201] | n/a | n/a |
| conditioned@trim | retained | future | judge | n/a | 0.201 [0.125, 0.288] | n/a | n/a |
| conditioned@trim | channel | message | delivered words | n/a | 30.583 [28.104, 32.521] | n/a | n/a |
| conditioned@trim | channel | message | fill ratio | n/a | 0.765 [0.703, 0.813] | n/a | n/a |
| extractive conditioned | utility | now | EM | 0.500 [0.396, 0.604] | 0.562 [0.458, 0.667] | 0.667 [0.573, 0.771] | 0.604 [0.510, 0.698] |
| extractive conditioned | utility | now | F1 | 0.675 [0.604, 0.746] | 0.701 [0.622, 0.777] | 0.816 [0.746, 0.882] | 0.780 [0.716, 0.841] |
| extractive conditioned | utility | now | judge | 0.740 [0.677, 0.802] | 0.792 [0.708, 0.865] | 0.906 [0.844, 0.958] | 0.865 [0.802, 0.917] |
| extractive conditioned | utility | future | EM | 0.035 [0.014, 0.056] | 0.146 [0.101, 0.194] | 0.257 [0.208, 0.309] | 0.538 [0.462, 0.608] |
| extractive conditioned | utility | future | F1 | 0.103 [0.071, 0.136] | 0.218 [0.166, 0.271] | 0.358 [0.305, 0.409] | 0.678 [0.611, 0.744] |
| extractive conditioned | utility | future | judge | 0.156 [0.111, 0.212] | 0.285 [0.219, 0.351] | 0.458 [0.385, 0.531] | 0.771 [0.701, 0.840] |
| extractive conditioned | utility | gap | EM | 0.465 [0.372, 0.562] | 0.417 [0.306, 0.531] | 0.410 [0.309, 0.514] | 0.066 [0.003, 0.132] |
| extractive conditioned | utility | gap | F1 | 0.572 [0.492, 0.650] | 0.482 [0.379, 0.582] | 0.458 [0.376, 0.540] | 0.102 [0.050, 0.154] |
| extractive conditioned | utility | gap | judge | 0.583 [0.500, 0.667] | 0.507 [0.389, 0.625] | 0.448 [0.354, 0.545] | 0.094 [0.028, 0.163] |
| extractive conditioned | regret | now | EM | 0.135 [0.042, 0.229] | 0.073 [-0.010, 0.156] | -0.031 [-0.083, 0.021] | 0.031 [-0.010, 0.073] |
| extractive conditioned | regret | now | F1 | 0.110 [0.031, 0.189] | 0.084 [0.012, 0.159] | -0.031 [-0.085, 0.019] | 0.005 [-0.030, 0.043] |
| extractive conditioned | regret | now | judge | 0.135 [0.052, 0.208] | 0.083 [0.021, 0.146] | -0.031 [-0.073, 0.010] | 0.010 [-0.031, 0.052] |
| extractive conditioned | regret | future | EM | 0.601 [0.503, 0.698] | 0.490 [0.389, 0.594] | 0.378 [0.281, 0.476] | 0.097 [0.042, 0.156] |
| extractive conditioned | regret | future | F1 | 0.681 [0.597, 0.765] | 0.566 [0.480, 0.651] | 0.427 [0.341, 0.513] | 0.107 [0.058, 0.157] |
| extractive conditioned | regret | future | judge | 0.719 [0.625, 0.806] | 0.590 [0.493, 0.681] | 0.417 [0.309, 0.521] | 0.104 [0.038, 0.174] |
| extractive conditioned | retained | now | EM | 0.705 [0.583, 0.816] | 0.753 [0.635, 0.861] | 0.903 [0.819, 0.979] | 0.927 [0.851, 0.990] |
| extractive conditioned | retained | now | F1 | 0.724 [0.653, 0.795] | 0.766 [0.688, 0.840] | 0.882 [0.826, 0.933] | 0.872 [0.827, 0.918] |
| extractive conditioned | retained | now | judge | 0.792 [0.719, 0.858] | 0.861 [0.795, 0.920] | 0.972 [0.931, 1.000] | 0.955 [0.910, 0.990] |
| extractive conditioned | retained | future | EM | 0.038 [0.012, 0.071] | 0.172 [0.109, 0.241] | 0.339 [0.260, 0.422] | 0.783 [0.686, 0.868] |
| extractive conditioned | retained | future | F1 | 0.112 [0.078, 0.147] | 0.227 [0.166, 0.292] | 0.391 [0.325, 0.460] | 0.761 [0.710, 0.812] |
| extractive conditioned | retained | future | judge | 0.151 [0.104, 0.201] | 0.280 [0.210, 0.352] | 0.474 [0.378, 0.568] | 0.849 [0.788, 0.906] |
| extractive conditioned | channel | message | delivered words | 18.594 [18.333, 18.833] | 36.823 [36.135, 37.396] | 70.927 [69.667, 72.010] | 141.427 [138.844, 143.760] |
| extractive conditioned | channel | message | fill ratio | 0.930 [0.917, 0.942] | 0.921 [0.903, 0.935] | 0.887 [0.871, 0.900] | 0.884 [0.868, 0.899] |
| extractive generic | utility | now | EM | 0.240 [0.167, 0.333] | 0.312 [0.240, 0.385] | 0.521 [0.448, 0.604] | 0.656 [0.562, 0.740] |
| extractive generic | utility | now | F1 | 0.301 [0.233, 0.385] | 0.376 [0.318, 0.436] | 0.593 [0.514, 0.675] | 0.776 [0.702, 0.843] |
| extractive generic | utility | now | judge | 0.354 [0.250, 0.458] | 0.427 [0.344, 0.510] | 0.615 [0.521, 0.708] | 0.854 [0.771, 0.927] |
| extractive generic | utility | future | EM | 0.240 [0.167, 0.333] | 0.312 [0.240, 0.385] | 0.521 [0.448, 0.604] | 0.656 [0.562, 0.740] |
| extractive generic | utility | future | F1 | 0.301 [0.233, 0.385] | 0.376 [0.318, 0.436] | 0.593 [0.514, 0.675] | 0.776 [0.702, 0.843] |
| extractive generic | utility | future | judge | 0.354 [0.250, 0.458] | 0.427 [0.344, 0.510] | 0.615 [0.521, 0.708] | 0.854 [0.771, 0.927] |
| extractive generic | utility | gap | EM | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| extractive generic | utility | gap | F1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| extractive generic | utility | gap | judge | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| extractive generic | regret | now | EM | 0.396 [0.271, 0.521] | 0.323 [0.198, 0.438] | 0.115 [0.000, 0.229] | -0.021 [-0.094, 0.052] |
| extractive generic | regret | now | F1 | 0.484 [0.369, 0.592] | 0.409 [0.307, 0.511] | 0.191 [0.078, 0.309] | 0.009 [-0.051, 0.067] |
| extractive generic | regret | now | judge | 0.521 [0.375, 0.656] | 0.448 [0.344, 0.552] | 0.260 [0.135, 0.375] | 0.021 [-0.052, 0.094] |
| extractive generic | regret | future | EM | 0.396 [0.271, 0.521] | 0.323 [0.198, 0.438] | 0.115 [0.000, 0.229] | -0.021 [-0.094, 0.052] |
| extractive generic | regret | future | F1 | 0.484 [0.369, 0.592] | 0.409 [0.307, 0.511] | 0.191 [0.078, 0.309] | 0.009 [-0.051, 0.067] |
| extractive generic | regret | future | judge | 0.521 [0.375, 0.656] | 0.448 [0.344, 0.552] | 0.260 [0.135, 0.375] | 0.021 [-0.052, 0.094] |
| extractive generic | retained | now | EM | 0.281 [0.167, 0.406] | 0.406 [0.295, 0.528] | 0.615 [0.497, 0.729] | 0.924 [0.823, 0.990] |
| extractive generic | retained | now | F1 | 0.290 [0.208, 0.383] | 0.378 [0.302, 0.463] | 0.622 [0.533, 0.708] | 0.861 [0.797, 0.917] |
| extractive generic | retained | now | judge | 0.358 [0.250, 0.472] | 0.417 [0.326, 0.510] | 0.632 [0.545, 0.722] | 0.931 [0.861, 0.979] |
| extractive generic | retained | future | EM | 0.281 [0.167, 0.406] | 0.406 [0.295, 0.528] | 0.615 [0.497, 0.729] | 0.924 [0.823, 0.990] |
| extractive generic | retained | future | F1 | 0.290 [0.208, 0.383] | 0.378 [0.302, 0.463] | 0.622 [0.533, 0.708] | 0.861 [0.797, 0.917] |
| extractive generic | retained | future | judge | 0.358 [0.250, 0.472] | 0.417 [0.326, 0.510] | 0.632 [0.545, 0.722] | 0.931 [0.861, 0.979] |
| extractive generic | channel | message | delivered words | 18.917 [18.500, 19.333] | 37.417 [36.500, 38.250] | 73.333 [71.708, 74.917] | 143.750 [141.417, 146.333] |
| extractive generic | channel | message | fill ratio | 0.946 [0.925, 0.967] | 0.935 [0.912, 0.956] | 0.917 [0.896, 0.936] | 0.898 [0.884, 0.915] |
| generic | utility | now | EM | 0.125 [0.073, 0.177] | 0.219 [0.156, 0.281] | 0.271 [0.188, 0.354] | 0.396 [0.292, 0.510] |
| generic | utility | now | F1 | 0.186 [0.140, 0.231] | 0.302 [0.230, 0.374] | 0.410 [0.328, 0.489] | 0.552 [0.459, 0.651] |
| generic | utility | now | judge | 0.240 [0.188, 0.292] | 0.375 [0.302, 0.448] | 0.531 [0.427, 0.635] | 0.698 [0.604, 0.792] |
| generic | utility | future | EM | 0.125 [0.073, 0.177] | 0.219 [0.156, 0.281] | 0.271 [0.188, 0.354] | 0.396 [0.292, 0.510] |
| generic | utility | future | F1 | 0.186 [0.140, 0.231] | 0.302 [0.230, 0.374] | 0.410 [0.328, 0.489] | 0.552 [0.459, 0.651] |
| generic | utility | future | judge | 0.240 [0.188, 0.292] | 0.375 [0.302, 0.448] | 0.531 [0.427, 0.635] | 0.698 [0.604, 0.792] |
| generic | utility | gap | EM | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| generic | utility | gap | F1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| generic | utility | gap | judge | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| generic | regret | now | EM | 0.510 [0.406, 0.615] | 0.417 [0.302, 0.521] | 0.365 [0.260, 0.469] | 0.240 [0.146, 0.333] |
| generic | regret | now | F1 | 0.599 [0.516, 0.687] | 0.483 [0.382, 0.583] | 0.375 [0.273, 0.474] | 0.233 [0.150, 0.317] |
| generic | regret | now | judge | 0.635 [0.552, 0.719] | 0.500 [0.396, 0.604] | 0.344 [0.229, 0.458] | 0.177 [0.083, 0.271] |
| generic | regret | future | EM | 0.510 [0.406, 0.615] | 0.417 [0.302, 0.521] | 0.365 [0.260, 0.469] | 0.240 [0.146, 0.333] |
| generic | regret | future | F1 | 0.599 [0.516, 0.687] | 0.483 [0.382, 0.583] | 0.375 [0.273, 0.474] | 0.233 [0.150, 0.317] |
| generic | regret | future | judge | 0.635 [0.552, 0.719] | 0.500 [0.396, 0.604] | 0.344 [0.229, 0.458] | 0.177 [0.083, 0.271] |
| generic | retained | now | EM | 0.139 [0.066, 0.219] | 0.299 [0.198, 0.406] | 0.396 [0.267, 0.528] | 0.608 [0.465, 0.747] |
| generic | retained | now | F1 | 0.182 [0.123, 0.240] | 0.326 [0.243, 0.411] | 0.455 [0.361, 0.548] | 0.608 [0.517, 0.699] |
| generic | retained | now | judge | 0.226 [0.149, 0.302] | 0.403 [0.302, 0.504] | 0.559 [0.451, 0.663] | 0.767 [0.663, 0.861] |
| generic | retained | future | EM | 0.139 [0.066, 0.219] | 0.299 [0.198, 0.406] | 0.396 [0.267, 0.528] | 0.608 [0.465, 0.747] |
| generic | retained | future | F1 | 0.182 [0.123, 0.240] | 0.326 [0.243, 0.411] | 0.455 [0.361, 0.548] | 0.608 [0.517, 0.699] |
| generic | retained | future | judge | 0.226 [0.149, 0.302] | 0.403 [0.302, 0.504] | 0.559 [0.451, 0.663] | 0.767 [0.663, 0.861] |
| generic | channel | message | delivered words | 18.708 [18.292, 19.125] | 37.125 [36.417, 37.833] | 71.917 [70.250, 73.667] | 141.417 [139.833, 143.208] |
| generic | channel | message | fill ratio | 0.935 [0.915, 0.956] | 0.928 [0.910, 0.946] | 0.899 [0.878, 0.921] | 0.884 [0.874, 0.895] |
| generic@trim | utility | now | EM | n/a | 0.188 [0.125, 0.250] | n/a | n/a |
| generic@trim | utility | now | F1 | n/a | 0.275 [0.208, 0.345] | n/a | n/a |
| generic@trim | utility | now | judge | n/a | 0.323 [0.250, 0.396] | n/a | n/a |
| generic@trim | utility | future | EM | n/a | 0.188 [0.125, 0.250] | n/a | n/a |
| generic@trim | utility | future | F1 | n/a | 0.275 [0.208, 0.345] | n/a | n/a |
| generic@trim | utility | future | judge | n/a | 0.323 [0.250, 0.396] | n/a | n/a |
| generic@trim | utility | gap | EM | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| generic@trim | utility | gap | F1 | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| generic@trim | utility | gap | judge | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| generic@trim | regret | now | EM | n/a | 0.448 [0.354, 0.542] | n/a | n/a |
| generic@trim | regret | now | F1 | n/a | 0.509 [0.419, 0.600] | n/a | n/a |
| generic@trim | regret | now | judge | n/a | 0.552 [0.448, 0.646] | n/a | n/a |
| generic@trim | regret | future | EM | n/a | 0.448 [0.354, 0.542] | n/a | n/a |
| generic@trim | regret | future | F1 | n/a | 0.509 [0.419, 0.600] | n/a | n/a |
| generic@trim | regret | future | judge | n/a | 0.552 [0.448, 0.646] | n/a | n/a |
| generic@trim | retained | now | EM | n/a | 0.236 [0.142, 0.330] | n/a | n/a |
| generic@trim | retained | now | F1 | n/a | 0.299 [0.217, 0.381] | n/a | n/a |
| generic@trim | retained | now | judge | n/a | 0.340 [0.240, 0.451] | n/a | n/a |
| generic@trim | retained | future | EM | n/a | 0.236 [0.142, 0.330] | n/a | n/a |
| generic@trim | retained | future | F1 | n/a | 0.299 [0.217, 0.381] | n/a | n/a |
| generic@trim | retained | future | judge | n/a | 0.340 [0.240, 0.451] | n/a | n/a |
| generic@trim | channel | message | delivered words | n/a | 24.750 [21.625, 27.875] | n/a | n/a |
| generic@trim | channel | message | fill ratio | n/a | 0.619 [0.541, 0.697] | n/a | n/a |
| oracle | utility | now | EM | 0.271 [0.177, 0.365] | 0.479 [0.365, 0.594] | 0.583 [0.479, 0.688] | 0.604 [0.521, 0.688] |
| oracle | utility | now | F1 | 0.398 [0.311, 0.490] | 0.666 [0.562, 0.762] | 0.776 [0.717, 0.837] | 0.771 [0.707, 0.835] |
| oracle | utility | now | judge | 0.490 [0.417, 0.562] | 0.823 [0.740, 0.896] | 0.885 [0.823, 0.948] | 0.896 [0.833, 0.948] |
| oracle | utility | future | EM | 0.271 [0.177, 0.365] | 0.479 [0.365, 0.594] | 0.583 [0.479, 0.688] | 0.604 [0.521, 0.688] |
| oracle | utility | future | F1 | 0.398 [0.311, 0.490] | 0.666 [0.562, 0.762] | 0.776 [0.717, 0.837] | 0.771 [0.707, 0.835] |
| oracle | utility | future | judge | 0.490 [0.417, 0.562] | 0.823 [0.740, 0.896] | 0.885 [0.823, 0.948] | 0.896 [0.833, 0.948] |
| oracle | utility | gap | EM | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| oracle | utility | gap | F1 | 0.000 [0.000, 0.000] | 0.000 [-0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [-0.000, 0.000] |
| oracle | utility | gap | judge | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| oracle | regret | now | EM | 0.365 [0.281, 0.448] | 0.156 [0.073, 0.250] | 0.052 [-0.042, 0.146] | 0.031 [-0.052, 0.104] |
| oracle | regret | now | F1 | 0.387 [0.303, 0.470] | 0.119 [0.038, 0.208] | 0.009 [-0.049, 0.068] | 0.013 [-0.055, 0.077] |
| oracle | regret | now | judge | 0.385 [0.312, 0.458] | 0.052 [-0.031, 0.146] | -0.010 [-0.083, 0.062] | -0.021 [-0.094, 0.052] |
| oracle | regret | future | EM | 0.365 [0.281, 0.448] | 0.156 [0.073, 0.250] | 0.052 [-0.042, 0.146] | 0.031 [-0.052, 0.104] |
| oracle | regret | future | F1 | 0.387 [0.303, 0.470] | 0.119 [0.038, 0.208] | 0.009 [-0.049, 0.068] | 0.013 [-0.055, 0.077] |
| oracle | regret | future | judge | 0.385 [0.312, 0.458] | 0.052 [-0.031, 0.146] | -0.010 [-0.083, 0.062] | -0.021 [-0.094, 0.052] |
| oracle | retained | now | EM | 0.309 [0.184, 0.441] | 0.681 [0.538, 0.812] | 0.812 [0.719, 0.899] | 0.830 [0.740, 0.913] |
| oracle | retained | now | F1 | 0.430 [0.324, 0.543] | 0.705 [0.595, 0.803] | 0.844 [0.789, 0.897] | 0.826 [0.773, 0.880] |
| oracle | retained | now | judge | 0.486 [0.403, 0.566] | 0.861 [0.771, 0.941] | 0.931 [0.878, 0.976] | 0.944 [0.899, 0.986] |
| oracle | retained | future | EM | 0.309 [0.184, 0.441] | 0.681 [0.538, 0.812] | 0.812 [0.719, 0.899] | 0.830 [0.740, 0.913] |
| oracle | retained | future | F1 | 0.430 [0.324, 0.543] | 0.705 [0.595, 0.803] | 0.844 [0.789, 0.897] | 0.826 [0.773, 0.880] |
| oracle | retained | future | judge | 0.486 [0.403, 0.566] | 0.861 [0.771, 0.941] | 0.931 [0.878, 0.976] | 0.944 [0.899, 0.986] |
| oracle | channel | message | delivered words | 18.125 [17.167, 18.958] | 37.917 [37.208, 38.625] | 73.833 [72.375, 75.333] | 143.375 [141.583, 145.250] |
| oracle | channel | message | fill ratio | 0.906 [0.858, 0.948] | 0.948 [0.930, 0.966] | 0.923 [0.905, 0.942] | 0.896 [0.885, 0.908] |
| oracle@trim | utility | now | EM | n/a | 0.354 [0.260, 0.448] | n/a | n/a |
| oracle@trim | utility | now | F1 | n/a | 0.468 [0.386, 0.549] | n/a | n/a |
| oracle@trim | utility | now | judge | n/a | 0.594 [0.510, 0.667] | n/a | n/a |
| oracle@trim | utility | future | EM | n/a | 0.354 [0.260, 0.448] | n/a | n/a |
| oracle@trim | utility | future | F1 | n/a | 0.468 [0.386, 0.549] | n/a | n/a |
| oracle@trim | utility | future | judge | n/a | 0.594 [0.510, 0.667] | n/a | n/a |
| oracle@trim | utility | gap | EM | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| oracle@trim | utility | gap | F1 | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| oracle@trim | utility | gap | judge | n/a | 0.000 [0.000, 0.000] | n/a | n/a |
| oracle@trim | regret | now | EM | n/a | 0.281 [0.188, 0.375] | n/a | n/a |
| oracle@trim | regret | now | F1 | n/a | 0.316 [0.231, 0.400] | n/a | n/a |
| oracle@trim | regret | now | judge | n/a | 0.281 [0.177, 0.396] | n/a | n/a |
| oracle@trim | regret | future | EM | n/a | 0.281 [0.188, 0.375] | n/a | n/a |
| oracle@trim | regret | future | F1 | n/a | 0.316 [0.231, 0.400] | n/a | n/a |
| oracle@trim | regret | future | judge | n/a | 0.281 [0.177, 0.396] | n/a | n/a |
| oracle@trim | retained | now | EM | n/a | 0.497 [0.354, 0.635] | n/a | n/a |
| oracle@trim | retained | now | F1 | n/a | 0.507 [0.398, 0.614] | n/a | n/a |
| oracle@trim | retained | now | judge | n/a | 0.618 [0.517, 0.715] | n/a | n/a |
| oracle@trim | retained | future | EM | n/a | 0.497 [0.354, 0.635] | n/a | n/a |
| oracle@trim | retained | future | F1 | n/a | 0.507 [0.398, 0.614] | n/a | n/a |
| oracle@trim | retained | future | judge | n/a | 0.618 [0.517, 0.715] | n/a | n/a |
| oracle@trim | channel | message | delivered words | n/a | 24.333 [21.708, 26.833] | n/a | n/a |
| oracle@trim | channel | message | fill ratio | n/a | 0.608 [0.543, 0.671] | n/a | n/a |
| reusable | utility | now | EM | 0.521 [0.438, 0.615] | 0.531 [0.438, 0.635] | 0.542 [0.448, 0.646] | 0.625 [0.521, 0.729] |
| reusable | utility | now | F1 | 0.713 [0.649, 0.779] | 0.712 [0.639, 0.787] | 0.714 [0.637, 0.789] | 0.780 [0.705, 0.851] |
| reusable | utility | now | judge | 0.792 [0.719, 0.865] | 0.802 [0.729, 0.875] | 0.854 [0.792, 0.917] | 0.906 [0.844, 0.958] |
| reusable | utility | future | EM | 0.073 [0.028, 0.125] | 0.101 [0.066, 0.135] | 0.208 [0.160, 0.264] | 0.351 [0.278, 0.420] |
| reusable | utility | future | F1 | 0.144 [0.098, 0.195] | 0.193 [0.148, 0.236] | 0.330 [0.279, 0.383] | 0.491 [0.426, 0.558] |
| reusable | utility | future | judge | 0.174 [0.122, 0.229] | 0.271 [0.205, 0.337] | 0.424 [0.354, 0.497] | 0.601 [0.524, 0.677] |
| reusable | utility | gap | EM | 0.448 [0.358, 0.538] | 0.431 [0.330, 0.538] | 0.333 [0.229, 0.434] | 0.274 [0.198, 0.354] |
| reusable | utility | gap | F1 | 0.570 [0.490, 0.649] | 0.519 [0.428, 0.616] | 0.384 [0.289, 0.479] | 0.289 [0.214, 0.365] |
| reusable | utility | gap | judge | 0.618 [0.521, 0.712] | 0.531 [0.406, 0.656] | 0.431 [0.326, 0.535] | 0.306 [0.215, 0.396] |
| reusable | regret | now | EM | 0.115 [0.021, 0.208] | 0.104 [0.021, 0.188] | 0.094 [0.021, 0.156] | 0.010 [-0.062, 0.083] |
| reusable | regret | now | F1 | 0.071 [0.004, 0.142] | 0.073 [0.008, 0.141] | 0.071 [0.012, 0.127] | 0.005 [-0.068, 0.074] |
| reusable | regret | now | judge | 0.083 [0.010, 0.167] | 0.073 [0.010, 0.135] | 0.021 [-0.052, 0.083] | -0.031 [-0.104, 0.031] |
| reusable | regret | future | EM | 0.562 [0.451, 0.670] | 0.535 [0.434, 0.632] | 0.427 [0.330, 0.521] | 0.285 [0.194, 0.368] |
| reusable | regret | future | F1 | 0.641 [0.545, 0.734] | 0.592 [0.512, 0.674] | 0.455 [0.366, 0.544] | 0.294 [0.209, 0.373] |
| reusable | regret | future | judge | 0.701 [0.608, 0.788] | 0.604 [0.500, 0.705] | 0.451 [0.354, 0.549] | 0.274 [0.170, 0.372] |
| reusable | retained | now | EM | 0.771 [0.674, 0.861] | 0.778 [0.674, 0.875] | 0.740 [0.618, 0.854] | 0.809 [0.701, 0.906] |
| reusable | retained | now | F1 | 0.763 [0.698, 0.825] | 0.767 [0.699, 0.834] | 0.770 [0.687, 0.850] | 0.803 [0.729, 0.875] |
| reusable | retained | now | judge | 0.861 [0.785, 0.931] | 0.875 [0.812, 0.934] | 0.920 [0.868, 0.969] | 0.965 [0.924, 1.000] |
| reusable | retained | future | EM | 0.108 [0.024, 0.219] | 0.125 [0.075, 0.182] | 0.290 [0.203, 0.384] | 0.481 [0.370, 0.590] |
| reusable | retained | future | F1 | 0.164 [0.103, 0.236] | 0.204 [0.151, 0.261] | 0.369 [0.306, 0.436] | 0.550 [0.486, 0.615] |
| reusable | retained | future | judge | 0.168 [0.113, 0.229] | 0.278 [0.201, 0.354] | 0.455 [0.375, 0.538] | 0.660 [0.578, 0.740] |
| reusable | channel | message | delivered words | 18.562 [18.354, 18.760] | 36.333 [35.573, 36.979] | 73.229 [72.552, 73.896] | 143.562 [142.750, 144.448] |
| reusable | channel | message | fill ratio | 0.928 [0.918, 0.938] | 0.908 [0.889, 0.924] | 0.915 [0.907, 0.924] | 0.897 [0.892, 0.903] |
| reusable@trim | utility | now | EM | n/a | 0.542 [0.448, 0.646] | n/a | n/a |
| reusable@trim | utility | now | F1 | n/a | 0.682 [0.591, 0.772] | n/a | n/a |
| reusable@trim | utility | now | judge | n/a | 0.781 [0.698, 0.854] | n/a | n/a |
| reusable@trim | utility | future | EM | n/a | 0.066 [0.035, 0.101] | n/a | n/a |
| reusable@trim | utility | future | F1 | n/a | 0.142 [0.102, 0.184] | n/a | n/a |
| reusable@trim | utility | future | judge | n/a | 0.194 [0.132, 0.260] | n/a | n/a |
| reusable@trim | utility | gap | EM | n/a | 0.476 [0.372, 0.587] | n/a | n/a |
| reusable@trim | utility | gap | F1 | n/a | 0.540 [0.438, 0.641] | n/a | n/a |
| reusable@trim | utility | gap | judge | n/a | 0.587 [0.469, 0.698] | n/a | n/a |
| reusable@trim | regret | now | EM | n/a | 0.094 [0.010, 0.188] | n/a | n/a |
| reusable@trim | regret | now | F1 | n/a | 0.103 [0.017, 0.195] | n/a | n/a |
| reusable@trim | regret | now | judge | n/a | 0.094 [0.021, 0.167] | n/a | n/a |
| reusable@trim | regret | future | EM | n/a | 0.569 [0.462, 0.670] | n/a | n/a |
| reusable@trim | regret | future | F1 | n/a | 0.643 [0.556, 0.728] | n/a | n/a |
| reusable@trim | regret | future | judge | n/a | 0.681 [0.580, 0.774] | n/a | n/a |
| reusable@trim | retained | now | EM | n/a | 0.743 [0.632, 0.847] | n/a | n/a |
| reusable@trim | retained | now | F1 | n/a | 0.728 [0.645, 0.806] | n/a | n/a |
| reusable@trim | retained | now | judge | n/a | 0.861 [0.795, 0.924] | n/a | n/a |
| reusable@trim | retained | future | EM | n/a | 0.092 [0.040, 0.153] | n/a | n/a |
| reusable@trim | retained | future | F1 | n/a | 0.151 [0.100, 0.208] | n/a | n/a |
| reusable@trim | retained | future | judge | n/a | 0.194 [0.123, 0.273] | n/a | n/a |
| reusable@trim | channel | message | delivered words | n/a | 27.427 [25.083, 29.500] | n/a | n/a |
| reusable@trim | channel | message | fill ratio | n/a | 0.686 [0.627, 0.737] | n/a | n/a |

### SQuAD groups &mdash; normalised utility, all metrics

| policy | endpoint | metric | 20 w | 40 w | 80 w | 160 w | n_excluded_low_ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned | now | EM | 0.878 [0.705, 1.069] | 0.927 [0.764, 1.108] | 0.896 [0.771, 1.042] | 0.972 [0.819, 1.146] | 0 |
| conditioned | now | F1 | 0.944 [0.849, 1.049] | 0.943 [0.834, 1.057] | 0.902 [0.832, 0.975] | 0.958 [0.870, 1.054] | 0 |
| conditioned | now | judge | 1.014 [0.934, 1.101] | 0.976 [0.858, 1.111] | 0.938 [0.882, 0.997] | 0.913 [0.837, 0.986] | 0 |
| conditioned | future | EM | 0.159 [0.071, 0.267] | 0.196 [0.102, 0.308] | 0.322 [0.188, 0.476] | 0.490 [0.347, 0.659] | 0 |
| conditioned | future | F1 | 0.197 [0.133, 0.270] | 0.254 [0.185, 0.326] | 0.350 [0.266, 0.443] | 0.554 [0.440, 0.679] | 0 |
| conditioned | future | judge | 0.227 [0.142, 0.321] | 0.319 [0.226, 0.421] | 0.410 [0.314, 0.516] | 0.642 [0.515, 0.784] | 0 |
| conditioned@trim | now | EM | n/a | 0.931 [0.743, 1.174] | n/a | n/a | 0 |
| conditioned@trim | now | F1 | n/a | 0.909 [0.807, 1.026] | n/a | n/a | 0 |
| conditioned@trim | now | judge | n/a | 0.976 [0.861, 1.115] | n/a | n/a | 0 |
| conditioned@trim | future | EM | n/a | 0.157 [0.074, 0.259] | n/a | n/a | 0 |
| conditioned@trim | future | F1 | n/a | 0.202 [0.137, 0.273] | n/a | n/a | 0 |
| conditioned@trim | future | judge | n/a | 0.242 [0.160, 0.334] | n/a | n/a | 0 |
| extractive conditioned | now | EM | 0.885 [0.677, 1.142] | 0.962 [0.767, 1.167] | 1.076 [0.972, 1.201] | 0.962 [0.889, 1.035] | 0 |
| extractive conditioned | now | F1 | 0.895 [0.786, 1.021] | 0.921 [0.810, 1.043] | 1.063 [0.979, 1.170] | 1.006 [0.960, 1.056] | 0 |
| extractive conditioned | now | judge | 0.868 [0.778, 0.969] | 0.910 [0.826, 1.000] | 1.056 [0.986, 1.132] | 1.003 [0.948, 1.066] | 0 |
| extractive conditioned | future | EM | 0.078 [0.027, 0.146] | 0.281 [0.178, 0.398] | 0.471 [0.358, 0.597] | 0.868 [0.778, 0.958] | 0 |
| extractive conditioned | future | F1 | 0.146 [0.096, 0.201] | 0.294 [0.221, 0.369] | 0.483 [0.401, 0.566] | 0.873 [0.811, 0.934] | 0 |
| extractive conditioned | future | judge | 0.194 [0.131, 0.270] | 0.344 [0.256, 0.434] | 0.557 [0.450, 0.676] | 0.898 [0.814, 0.980] | 0 |
| extractive generic | now | EM | 0.462 [0.292, 0.663] | 0.615 [0.396, 0.872] | 0.976 [0.771, 1.205] | 1.069 [0.910, 1.240] | 0 |
| extractive generic | now | F1 | 0.418 [0.304, 0.551] | 0.524 [0.416, 0.642] | 0.814 [0.673, 0.962] | 1.004 [0.923, 1.095] | 0 |
| extractive generic | now | judge | 0.448 [0.302, 0.608] | 0.514 [0.406, 0.625] | 0.743 [0.597, 0.910] | 1.000 [0.892, 1.122] | 0 |
| extractive generic | future | EM | 0.462 [0.292, 0.663] | 0.615 [0.396, 0.872] | 0.976 [0.771, 1.205] | 1.069 [0.910, 1.240] | 0 |
| extractive generic | future | F1 | 0.418 [0.304, 0.551] | 0.524 [0.416, 0.642] | 0.814 [0.673, 0.962] | 1.004 [0.923, 1.095] | 0 |
| extractive generic | future | judge | 0.448 [0.302, 0.608] | 0.514 [0.406, 0.625] | 0.743 [0.597, 0.910] | 1.000 [0.892, 1.122] | 0 |
| generic | now | EM | 0.243 [0.128, 0.365] | 0.403 [0.243, 0.594] | 0.479 [0.312, 0.674] | 0.649 [0.479, 0.833] | 0 |
| generic | now | F1 | 0.251 [0.186, 0.316] | 0.405 [0.299, 0.516] | 0.548 [0.425, 0.680] | 0.709 [0.599, 0.819] | 0 |
| generic | now | judge | 0.288 [0.222, 0.354] | 0.451 [0.354, 0.549] | 0.622 [0.500, 0.747] | 0.816 [0.694, 0.934] | 0 |
| generic | future | EM | 0.243 [0.128, 0.365] | 0.403 [0.243, 0.594] | 0.479 [0.313, 0.674] | 0.649 [0.479, 0.833] | 0 |
| generic | future | F1 | 0.251 [0.186, 0.316] | 0.405 [0.299, 0.516] | 0.548 [0.425, 0.680] | 0.709 [0.599, 0.819] | 0 |
| generic | future | judge | 0.288 [0.222, 0.354] | 0.451 [0.354, 0.549] | 0.622 [0.500, 0.747] | 0.816 [0.694, 0.934] | 0 |
| generic@trim | now | EM | n/a | 0.299 [0.191, 0.410] | n/a | n/a | 0 |
| generic@trim | now | F1 | n/a | 0.359 [0.264, 0.455] | n/a | n/a | 0 |
| generic@trim | now | judge | n/a | 0.389 [0.292, 0.493] | n/a | n/a | 0 |
| generic@trim | future | EM | n/a | 0.299 [0.191, 0.410] | n/a | n/a | 0 |
| generic@trim | future | F1 | n/a | 0.359 [0.264, 0.455] | n/a | n/a | 0 |
| generic@trim | future | judge | n/a | 0.389 [0.292, 0.493] | n/a | n/a | 0 |
| oracle | now | EM | 0.392 [0.267, 0.524] | 0.771 [0.628, 0.899] | 1.007 [0.819, 1.215] | 1.066 [0.872, 1.306] | 0 |
| oracle | now | F1 | 0.499 [0.396, 0.605] | 0.853 [0.738, 0.957] | 1.019 [0.937, 1.114] | 1.021 [0.916, 1.158] | 0 |
| oracle | now | judge | 0.562 [0.483, 0.646] | 0.972 [0.861, 1.083] | 1.042 [0.944, 1.160] | 1.062 [0.958, 1.187] | 0 |
| oracle | future | EM | 0.392 [0.267, 0.524] | 0.771 [0.628, 0.899] | 1.007 [0.819, 1.215] | 1.066 [0.872, 1.306] | 0 |
| oracle | future | F1 | 0.499 [0.396, 0.605] | 0.853 [0.738, 0.957] | 1.019 [0.937, 1.114] | 1.021 [0.916, 1.158] | 0 |
| oracle | future | judge | 0.562 [0.483, 0.646] | 0.972 [0.861, 1.083] | 1.042 [0.944, 1.160] | 1.062 [0.958, 1.188] | 0 |
| oracle@trim | now | EM | n/a | 0.601 [0.434, 0.785] | n/a | n/a | 0 |
| oracle@trim | now | F1 | n/a | 0.609 [0.505, 0.710] | n/a | n/a | 0 |
| oracle@trim | now | judge | n/a | 0.715 [0.597, 0.837] | n/a | n/a | 0 |
| oracle@trim | future | EM | n/a | 0.601 [0.434, 0.785] | n/a | n/a | 0 |
| oracle@trim | future | F1 | n/a | 0.609 [0.505, 0.710] | n/a | n/a | 0 |
| oracle@trim | future | judge | n/a | 0.715 [0.597, 0.837] | n/a | n/a | 0 |
| reusable | now | EM | 0.903 [0.743, 1.083] | 0.889 [0.750, 1.035] | 0.885 [0.719, 1.073] | 1.059 [0.875, 1.292] | 0 |
| reusable | now | F1 | 0.935 [0.851, 1.019] | 0.925 [0.841, 1.013] | 0.923 [0.832, 1.028] | 1.027 [0.916, 1.170] | 0 |
| reusable | now | judge | 0.924 [0.837, 1.014] | 0.924 [0.854, 0.997] | 1.003 [0.913, 1.122] | 1.069 [0.976, 1.187] | 0 |
| reusable | future | EM | 0.163 [0.056, 0.297] | 0.203 [0.125, 0.292] | 0.378 [0.272, 0.501] | 0.619 [0.465, 0.813] | 0 |
| reusable | future | F1 | 0.204 [0.130, 0.289] | 0.259 [0.197, 0.319] | 0.449 [0.367, 0.536] | 0.654 [0.552, 0.786] | 0 |
| reusable | future | judge | 0.214 [0.144, 0.292] | 0.336 [0.244, 0.433] | 0.505 [0.418, 0.594] | 0.718 [0.606, 0.854] | 0 |
| reusable@trim | now | EM | n/a | 0.910 [0.771, 1.056] | n/a | n/a | 0 |
| reusable@trim | now | F1 | n/a | 0.891 [0.774, 1.009] | n/a | n/a | 0 |
| reusable@trim | now | judge | n/a | 0.910 [0.819, 1.007] | n/a | n/a | 0 |
| reusable@trim | future | EM | n/a | 0.153 [0.072, 0.249] | n/a | n/a | 0 |
| reusable@trim | future | F1 | n/a | 0.198 [0.136, 0.263] | n/a | n/a | 0 |
| reusable@trim | future | judge | n/a | 0.240 [0.159, 0.331] | n/a | n/a | 0 |

### SQuAD groups &mdash; length-matched subsample, all metrics

| comparison | endpoint | metric | 20 w | 40 w | 80 w | 160 w | tolerance_words |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned minus generic | now | EM | **+0.417 [+0.292, +0.552]** | **+0.315 [+0.207, +0.424]** | **+0.150 [+0.033, +0.267]** | **+0.212 [+0.038, +0.404]** | 6 |
| conditioned minus generic | now | F1 | **+0.541 [+0.449, +0.634]** | **+0.409 [+0.311, +0.505]** | **+0.220 [+0.099, +0.339]** | **+0.228 [+0.080, +0.380]** | 6 |
| conditioned minus generic | now | judge | **+0.625 [+0.552, +0.698]** | **+0.446 [+0.348, +0.543]** | **+0.200 [+0.067, +0.333]** | **+0.173 [+0.038, +0.308]** | 6 |
| conditioned minus generic | future | EM | -0.056 [-0.122, +0.014] | **-0.127 [-0.210, -0.047]** | **-0.128 [-0.200, -0.044]** | **-0.212 [-0.391, -0.026]** | 6 |
| conditioned minus generic | future | F1 | -0.043 [-0.102, +0.020] | **-0.118 [-0.192, -0.044]** | **-0.133 [-0.213, -0.046]** | **-0.211 [-0.383, -0.041]** | 6 |
| conditioned minus generic | future | judge | **-0.063 [-0.122, -0.000]** | **-0.105 [-0.192, -0.014]** | **-0.183 [-0.289, -0.078]** | **-0.244 [-0.423, -0.071]** | 6 |
| conditioned minus reusable | now | EM | +0.021 [-0.062, +0.115] | +0.011 [-0.065, +0.087] | -0.033 [-0.100, +0.033] | -0.038 [-0.154, +0.058] | 6 |
| conditioned minus reusable | now | F1 | +0.013 [-0.050, +0.081] | +0.004 [-0.079, +0.081] | -0.027 [-0.072, +0.014] | -0.017 [-0.093, +0.052] | 6 |
| conditioned minus reusable | now | judge | **+0.073 [+0.021, +0.125]** | +0.022 [-0.076, +0.120] | -0.067 [-0.150, +0.000] | -0.019 [-0.077, +0.038] | 6 |
| conditioned minus reusable | future | EM | -0.003 [-0.031, +0.028] | -0.011 [-0.040, +0.018] | **-0.050 [-0.089, -0.011]** | **-0.141 [-0.231, -0.051]** | 6 |
| conditioned minus reusable | future | F1 | -0.001 [-0.031, +0.027] | -0.008 [-0.034, +0.017] | **-0.073 [-0.116, -0.033]** | **-0.138 [-0.240, -0.034]** | 6 |
| conditioned minus reusable | future | judge | +0.003 [-0.031, +0.042] | -0.007 [-0.054, +0.040] | **-0.072 [-0.128, -0.006]** | **-0.128 [-0.250, -0.006]** | 6 |
| reusable minus generic | now | EM | **+0.396 [+0.302, +0.500]** | **+0.304 [+0.196, +0.424]** | **+0.183 [+0.067, +0.300]** | **+0.250 [+0.096, +0.442]** | 6 |
| reusable minus generic | now | F1 | **+0.528 [+0.445, +0.616]** | **+0.405 [+0.304, +0.513]** | **+0.246 [+0.128, +0.362]** | **+0.245 [+0.107, +0.408]** | 6 |
| reusable minus generic | now | judge | **+0.552 [+0.468, +0.646]** | **+0.424 [+0.304, +0.543]** | **+0.267 [+0.117, +0.400]** | **+0.192 [+0.096, +0.308]** | 6 |
| reusable minus generic | future | EM | -0.052 [-0.115, +0.024] | **-0.116 [-0.181, -0.051]** | -0.078 [-0.161, +0.017] | -0.071 [-0.199, +0.071] | 6 |
| reusable minus generic | future | F1 | -0.042 [-0.096, +0.019] | **-0.110 [-0.176, -0.044]** | -0.060 [-0.144, +0.037] | -0.072 [-0.184, +0.054] | 6 |
| reusable minus generic | future | judge | **-0.066 [-0.128, -0.003]** | **-0.098 [-0.174, -0.025]** | **-0.111 [-0.206, -0.017]** | -0.115 [-0.231, +0.006] | 6 |

### SQuAD groups &mdash; budget interaction, all metrics

| quantity | policy | metric | delta | p | dz |
|---|---:|---:|---:|---:|---:|
| gap_low_minus_high | conditioned | EM | **+0.163 [+0.059, +0.264]** | 0.0020 | 0.63 |
| gap_low_minus_high | conditioned | F1 | **+0.257 [+0.155, +0.356]** | 0.0000 | 1.01 |
| gap_low_minus_high | conditioned | judge | **+0.413 [+0.323, +0.503]** | 0.0000 | 1.84 |
| gap_low_minus_high | extractive_conditioned | EM | **+0.399 [+0.295, +0.493]** | 0.0000 | 1.56 |
| gap_low_minus_high | extractive_conditioned | F1 | **+0.470 [+0.381, +0.556]** | 0.0000 | 2.11 |
| gap_low_minus_high | extractive_conditioned | judge | **+0.490 [+0.396, +0.587]** | 0.0000 | 1.99 |
| gap_low_minus_high | extractive_generic | EM | -0.000 [-0.000, +0.000] | 0.7739 | -0.07 |
| gap_low_minus_high | extractive_generic | F1 | -0.000 [-0.000, +0.000] | 0.7804 | -0.06 |
| gap_low_minus_high | extractive_generic | judge | **+0.000 [+0.000, +0.000]** | 0.0266 | 0.46 |
| gap_low_minus_high | generic | EM | **-0.000 [-0.000, -0.000]** | 0.0330 | -0.43 |
| gap_low_minus_high | generic | F1 | +0.000 [-0.000, +0.000] | 0.9230 | 0.02 |
| gap_low_minus_high | generic | judge | -0.000 [-0.000, +0.000] | 0.5071 | -0.14 |
| gap_low_minus_high | oracle | EM | -0.000 [-0.000, +0.000] | 0.0622 | -0.38 |
| gap_low_minus_high | oracle | F1 | +0.000 [-0.000, +0.000] | 0.5043 | 0.13 |
| gap_low_minus_high | oracle | judge | **+0.000 [+0.000, +0.000]** | 0.0005 | 0.75 |
| gap_low_minus_high | reusable | EM | **+0.174 [+0.087, +0.257]** | 0.0004 | 0.80 |
| gap_low_minus_high | reusable | F1 | **+0.281 [+0.185, +0.374]** | 0.0000 | 1.18 |
| gap_low_minus_high | reusable | judge | **+0.312 [+0.191, +0.431]** | 0.0000 | 1.03 |
| conditioning_gap_low_minus_high | conditioned_minus_generic | EM | **+0.163 [+0.062, +0.264]** | 0.0021 | 0.63 |
| conditioning_gap_low_minus_high | conditioned_minus_generic | F1 | **+0.257 [+0.158, +0.355]** | 0.0000 | 1.01 |
| conditioning_gap_low_minus_high | conditioned_minus_generic | judge | **+0.413 [+0.326, +0.503]** | 0.0000 | 1.84 |

### Relation dossiers &mdash; levels with 95% CIs

| policy | kind | endpoint | metric | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned | utility | now | EM | 0.750 [0.641, 0.844] | 0.766 [0.688, 0.844] | 0.766 [0.672, 0.859] | 0.766 [0.688, 0.844] |
| conditioned | utility | now | F1 | 0.887 [0.835, 0.932] | 0.880 [0.833, 0.924] | 0.882 [0.822, 0.934] | 0.864 [0.807, 0.916] |
| conditioned | utility | now | judge | 1.000 [1.000, 1.000] | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] | 0.969 [0.922, 1.000] |
| conditioned | utility | future | EM | 0.043 [0.033, 0.052] | 0.069 [0.057, 0.080] | 0.090 [0.077, 0.104] | 0.138 [0.117, 0.157] |
| conditioned | utility | future | F1 | 0.090 [0.078, 0.102] | 0.104 [0.090, 0.118] | 0.119 [0.105, 0.134] | 0.162 [0.146, 0.179] |
| conditioned | utility | future | judge | 0.077 [0.070, 0.085] | 0.096 [0.085, 0.107] | 0.117 [0.103, 0.131] | 0.168 [0.149, 0.186] |
| conditioned | utility | gap | EM | 0.707 [0.611, 0.794] | 0.697 [0.622, 0.771] | 0.676 [0.588, 0.764] | 0.628 [0.559, 0.698] |
| conditioned | utility | gap | F1 | 0.798 [0.747, 0.844] | 0.776 [0.729, 0.821] | 0.763 [0.708, 0.813] | 0.702 [0.648, 0.749] |
| conditioned | utility | gap | judge | 0.923 [0.915, 0.930] | 0.889 [0.849, 0.914] | 0.868 [0.831, 0.893] | 0.801 [0.751, 0.840] |
| conditioned | regret | now | EM | 0.047 [0.000, 0.094] | 0.031 [0.000, 0.078] | 0.031 [0.000, 0.078] | 0.031 [0.000, 0.078] |
| conditioned | regret | now | F1 | 0.011 [-0.009, 0.031] | 0.018 [0.000, 0.039] | 0.016 [-0.009, 0.053] | 0.034 [-0.005, 0.084] |
| conditioned | regret | now | judge | -0.016 [-0.047, 0.000] | 0.000 [0.000, 0.000] | 0.000 [-0.047, 0.047] | 0.016 [-0.031, 0.062] |
| conditioned | regret | future | EM | 0.804 [0.764, 0.840] | 0.778 [0.739, 0.815] | 0.757 [0.716, 0.794] | 0.709 [0.672, 0.744] |
| conditioned | regret | future | F1 | 0.838 [0.815, 0.860] | 0.823 [0.798, 0.848] | 0.808 [0.787, 0.830] | 0.765 [0.741, 0.789] |
| conditioned | regret | future | judge | 0.916 [0.899, 0.928] | 0.897 [0.875, 0.914] | 0.876 [0.853, 0.895] | 0.825 [0.798, 0.849] |
| conditioned | retained | now | EM | 0.932 [0.849, 1.000] | 0.964 [0.906, 1.000] | 0.958 [0.896, 1.000] | 0.964 [0.906, 1.000] |
| conditioned | retained | now | F1 | 0.901 [0.859, 0.938] | 0.894 [0.857, 0.931] | 0.896 [0.844, 0.941] | 0.880 [0.822, 0.931] |
| conditioned | retained | now | judge | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.984 [0.953, 1.000] | 0.969 [0.922, 1.000] |
| conditioned | retained | future | EM | 0.050 [0.039, 0.060] | 0.080 [0.069, 0.091] | 0.107 [0.093, 0.122] | 0.161 [0.140, 0.182] |
| conditioned | retained | future | F1 | 0.090 [0.079, 0.102] | 0.104 [0.091, 0.117] | 0.120 [0.106, 0.135] | 0.164 [0.147, 0.180] |
| conditioned | retained | future | judge | 0.077 [0.069, 0.085] | 0.094 [0.084, 0.106] | 0.117 [0.103, 0.131] | 0.167 [0.149, 0.185] |
| conditioned | channel | message | delivered words | 18.281 [18.016, 18.547] | 36.469 [36.094, 36.797] | 72.875 [72.062, 73.641] | 143.891 [142.656, 145.250] |
| conditioned | channel | message | fill ratio | 0.914 [0.901, 0.927] | 0.912 [0.902, 0.920] | 0.911 [0.901, 0.921] | 0.899 [0.892, 0.908] |
| generic | utility | now | EM | 0.047 [0.000, 0.094] | 0.234 [0.156, 0.328] | 0.250 [0.125, 0.375] | 0.422 [0.297, 0.547] |
| generic | utility | now | F1 | 0.155 [0.121, 0.190] | 0.313 [0.241, 0.394] | 0.384 [0.282, 0.490] | 0.556 [0.453, 0.656] |
| generic | utility | now | judge | 0.141 [0.078, 0.203] | 0.344 [0.266, 0.438] | 0.469 [0.391, 0.562] | 0.609 [0.516, 0.703] |
| generic | utility | future | EM | 0.055 [0.028, 0.082] | 0.143 [0.107, 0.182] | 0.171 [0.110, 0.231] | 0.393 [0.317, 0.473] |
| generic | utility | future | F1 | 0.129 [0.100, 0.158] | 0.213 [0.179, 0.253] | 0.277 [0.227, 0.325] | 0.484 [0.423, 0.548] |
| generic | utility | future | judge | 0.103 [0.077, 0.126] | 0.206 [0.168, 0.248] | 0.315 [0.259, 0.368] | 0.509 [0.452, 0.569] |
| generic | utility | gap | EM | -0.008 [-0.046, 0.033] | 0.092 [0.033, 0.154] | 0.079 [-0.004, 0.171] | 0.029 [-0.046, 0.104] |
| generic | utility | gap | F1 | 0.026 [-0.005, 0.055] | 0.100 [0.053, 0.146] | 0.107 [0.030, 0.185] | 0.072 [-0.000, 0.142] |
| generic | utility | gap | judge | 0.038 [-0.012, 0.083] | 0.138 [0.083, 0.196] | 0.154 [0.104, 0.213] | 0.100 [0.029, 0.171] |
| generic | regret | now | EM | 0.750 [0.672, 0.828] | 0.562 [0.469, 0.656] | 0.547 [0.453, 0.641] | 0.375 [0.297, 0.469] |
| generic | regret | now | F1 | 0.743 [0.697, 0.788] | 0.586 [0.508, 0.658] | 0.515 [0.414, 0.608] | 0.342 [0.262, 0.430] |
| generic | regret | now | judge | 0.844 [0.781, 0.906] | 0.641 [0.547, 0.734] | 0.516 [0.422, 0.609] | 0.375 [0.266, 0.484] |
| generic | regret | future | EM | 0.792 [0.749, 0.834] | 0.704 [0.643, 0.757] | 0.676 [0.617, 0.740] | 0.454 [0.392, 0.512] |
| generic | regret | future | F1 | 0.798 [0.765, 0.833] | 0.714 [0.669, 0.756] | 0.650 [0.599, 0.704] | 0.443 [0.383, 0.500] |
| generic | regret | future | judge | 0.890 [0.866, 0.917] | 0.786 [0.745, 0.827] | 0.678 [0.622, 0.736] | 0.483 [0.417, 0.546] |
| generic | retained | now | EM | 0.052 [0.000, 0.109] | 0.286 [0.182, 0.396] | 0.286 [0.146, 0.432] | 0.510 [0.391, 0.630] |
| generic | retained | now | F1 | 0.158 [0.124, 0.191] | 0.315 [0.245, 0.396] | 0.387 [0.287, 0.492] | 0.564 [0.463, 0.661] |
| generic | retained | now | judge | 0.141 [0.078, 0.203] | 0.349 [0.266, 0.438] | 0.479 [0.391, 0.568] | 0.604 [0.510, 0.698] |
| generic | retained | future | EM | 0.063 [0.034, 0.094] | 0.159 [0.117, 0.206] | 0.199 [0.131, 0.265] | 0.451 [0.375, 0.533] |
| generic | retained | future | F1 | 0.129 [0.100, 0.159] | 0.216 [0.183, 0.255] | 0.280 [0.229, 0.329] | 0.487 [0.427, 0.549] |
| generic | retained | future | judge | 0.104 [0.078, 0.127] | 0.208 [0.169, 0.249] | 0.318 [0.261, 0.372] | 0.507 [0.452, 0.564] |
| generic | channel | message | delivered words | 18.750 [18.312, 19.188] | 37.125 [36.250, 38.000] | 73.250 [71.562, 74.938] | 143.562 [141.500, 145.562] |
| generic | channel | message | fill ratio | 0.938 [0.916, 0.959] | 0.928 [0.906, 0.950] | 0.916 [0.895, 0.937] | 0.897 [0.884, 0.910] |
| oracle | utility | now | EM | 0.219 [0.172, 0.250] | 0.328 [0.250, 0.422] | 0.547 [0.453, 0.641] | 0.672 [0.547, 0.797] |
| oracle | utility | now | F1 | 0.245 [0.228, 0.258] | 0.429 [0.356, 0.504] | 0.651 [0.573, 0.732] | 0.784 [0.693, 0.870] |
| oracle | utility | now | judge | 0.250 [0.250, 0.250] | 0.516 [0.438, 0.594] | 0.750 [0.672, 0.828] | 0.891 [0.812, 0.953] |
| oracle | utility | future | EM | 0.144 [0.113, 0.172] | 0.299 [0.252, 0.343] | 0.534 [0.463, 0.607] | 0.668 [0.573, 0.756] |
| oracle | utility | future | F1 | 0.200 [0.175, 0.224] | 0.373 [0.338, 0.406] | 0.630 [0.568, 0.688] | 0.762 [0.688, 0.833] |
| oracle | utility | future | judge | 0.179 [0.163, 0.196] | 0.399 [0.363, 0.434] | 0.692 [0.625, 0.751] | 0.828 [0.755, 0.898] |
| oracle | utility | gap | EM | 0.075 [0.046, 0.104] | 0.029 [-0.021, 0.083] | 0.013 [-0.058, 0.092] | 0.004 [-0.058, 0.067] |
| oracle | utility | gap | F1 | 0.046 [0.023, 0.067] | 0.056 [0.007, 0.104] | 0.022 [-0.029, 0.073] | 0.022 [-0.023, 0.063] |
| oracle | utility | gap | judge | 0.071 [0.054, 0.087] | 0.117 [0.054, 0.175] | 0.058 [0.008, 0.104] | 0.062 [0.025, 0.100] |
| oracle | regret | now | EM | 0.578 [0.531, 0.641] | 0.469 [0.406, 0.531] | 0.250 [0.172, 0.328] | 0.125 [0.062, 0.203] |
| oracle | regret | now | F1 | 0.653 [0.617, 0.688] | 0.469 [0.399, 0.540] | 0.247 [0.172, 0.323] | 0.115 [0.047, 0.193] |
| oracle | regret | now | judge | 0.734 [0.703, 0.750] | 0.469 [0.391, 0.547] | 0.234 [0.156, 0.312] | 0.094 [0.031, 0.156] |
| oracle | regret | future | EM | 0.703 [0.668, 0.740] | 0.548 [0.513, 0.579] | 0.312 [0.246, 0.386] | 0.179 [0.109, 0.250] |
| oracle | regret | future | F1 | 0.728 [0.705, 0.752] | 0.554 [0.522, 0.586] | 0.298 [0.240, 0.359] | 0.165 [0.100, 0.233] |
| oracle | regret | future | judge | 0.814 [0.790, 0.838] | 0.594 [0.561, 0.628] | 0.301 [0.235, 0.371] | 0.165 [0.101, 0.231] |
| oracle | retained | now | EM | 0.266 [0.208, 0.312] | 0.396 [0.312, 0.479] | 0.688 [0.583, 0.792] | 0.828 [0.719, 0.927] |
| oracle | retained | now | F1 | 0.251 [0.230, 0.268] | 0.438 [0.363, 0.512] | 0.660 [0.586, 0.737] | 0.798 [0.706, 0.884] |
| oracle | retained | now | judge | 0.255 [0.250, 0.266] | 0.505 [0.422, 0.594] | 0.766 [0.688, 0.844] | 0.901 [0.839, 0.964] |
| oracle | retained | future | EM | 0.161 [0.124, 0.194] | 0.343 [0.291, 0.389] | 0.627 [0.549, 0.702] | 0.775 [0.678, 0.866] |
| oracle | retained | future | F1 | 0.203 [0.178, 0.227] | 0.378 [0.342, 0.410] | 0.638 [0.576, 0.699] | 0.768 [0.694, 0.838] |
| oracle | retained | future | judge | 0.181 [0.163, 0.199] | 0.398 [0.360, 0.433] | 0.698 [0.628, 0.763] | 0.828 [0.755, 0.898] |
| oracle | channel | message | delivered words | 18.062 [17.562, 18.562] | 36.750 [35.875, 37.688] | 75.625 [74.438, 76.875] | 143.688 [140.625, 147.188] |
| oracle | channel | message | fill ratio | 0.903 [0.878, 0.928] | 0.919 [0.897, 0.942] | 0.945 [0.930, 0.961] | 0.898 [0.879, 0.920] |
| reusable | utility | now | EM | 0.750 [0.641, 0.844] | 0.766 [0.688, 0.844] | 0.797 [0.719, 0.875] | 0.766 [0.688, 0.844] |
| reusable | utility | now | F1 | 0.887 [0.835, 0.932] | 0.893 [0.846, 0.934] | 0.895 [0.844, 0.940] | 0.867 [0.815, 0.917] |
| reusable | utility | now | judge | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.953 [0.906, 1.000] |
| reusable | utility | future | EM | 0.046 [0.036, 0.054] | 0.072 [0.060, 0.083] | 0.157 [0.143, 0.172] | 0.272 [0.236, 0.304] |
| reusable | utility | future | F1 | 0.093 [0.081, 0.104] | 0.112 [0.103, 0.121] | 0.188 [0.173, 0.203] | 0.315 [0.283, 0.345] |
| reusable | utility | future | judge | 0.081 [0.076, 0.086] | 0.103 [0.093, 0.114] | 0.197 [0.184, 0.209] | 0.334 [0.302, 0.364] |
| reusable | utility | gap | EM | 0.704 [0.607, 0.790] | 0.694 [0.619, 0.768] | 0.640 [0.567, 0.709] | 0.494 [0.422, 0.568] |
| reusable | utility | gap | F1 | 0.795 [0.747, 0.837] | 0.781 [0.736, 0.821] | 0.707 [0.661, 0.748] | 0.552 [0.497, 0.607] |
| reusable | utility | gap | judge | 0.919 [0.914, 0.924] | 0.897 [0.886, 0.907] | 0.803 [0.791, 0.816] | 0.619 [0.551, 0.678] |
| reusable | regret | now | EM | 0.047 [0.000, 0.094] | 0.031 [0.000, 0.078] | 0.000 [0.000, 0.000] | 0.031 [0.000, 0.078] |
| reusable | regret | now | F1 | 0.011 [-0.009, 0.031] | 0.006 [-0.009, 0.022] | 0.003 [-0.014, 0.023] | 0.031 [0.000, 0.078] |
| reusable | regret | now | judge | -0.016 [-0.047, 0.000] | -0.016 [-0.047, 0.000] | -0.016 [-0.047, 0.000] | 0.031 [0.000, 0.078] |
| reusable | regret | future | EM | 0.801 [0.761, 0.835] | 0.775 [0.735, 0.811] | 0.690 [0.650, 0.723] | 0.575 [0.541, 0.611] |
| reusable | regret | future | F1 | 0.834 [0.815, 0.856] | 0.815 [0.794, 0.838] | 0.739 [0.716, 0.760] | 0.613 [0.580, 0.647] |
| reusable | regret | future | judge | 0.911 [0.897, 0.921] | 0.890 [0.869, 0.905] | 0.796 [0.774, 0.815] | 0.658 [0.619, 0.696] |
| reusable | retained | now | EM | 0.932 [0.849, 1.000] | 0.964 [0.906, 1.000] | 1.000 [1.000, 1.000] | 0.964 [0.906, 1.000] |
| reusable | retained | now | F1 | 0.901 [0.859, 0.938] | 0.907 [0.873, 0.939] | 0.911 [0.861, 0.953] | 0.883 [0.828, 0.932] |
| reusable | retained | now | judge | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.969 [0.922, 1.000] |
| reusable | retained | future | EM | 0.054 [0.043, 0.062] | 0.085 [0.074, 0.096] | 0.185 [0.170, 0.200] | 0.318 [0.284, 0.351] |
| reusable | retained | future | F1 | 0.093 [0.082, 0.103] | 0.112 [0.104, 0.120] | 0.189 [0.174, 0.205] | 0.318 [0.286, 0.348] |
| reusable | retained | future | judge | 0.081 [0.075, 0.086] | 0.102 [0.092, 0.112] | 0.196 [0.184, 0.209] | 0.333 [0.301, 0.361] |
| reusable | channel | message | delivered words | 18.344 [18.047, 18.625] | 36.797 [36.484, 37.109] | 72.938 [72.328, 73.516] | 143.281 [141.953, 144.766] |
| reusable | channel | message | fill ratio | 0.917 [0.902, 0.931] | 0.920 [0.912, 0.928] | 0.912 [0.904, 0.919] | 0.896 [0.887, 0.905] |

### Relation dossiers &mdash; normalised utility, all metrics

| policy | endpoint | metric | 20 w | 40 w | 80 w | 160 w | n_excluded_low_ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned | now | EM | 0.932 [0.854, 1.000] | 0.964 [0.911, 1.000] | 0.958 [0.896, 1.000] | 0.964 [0.906, 1.000] | 0 |
| conditioned | now | F1 | 0.986 [0.962, 1.009] | 0.979 [0.955, 1.000] | 0.981 [0.938, 1.011] | 0.963 [0.911, 1.005] | 0 |
| conditioned | now | judge | 1.021 [1.000, 1.062] | 1.000 [1.000, 1.000] | 1.005 [0.953, 1.062] | 0.990 [0.938, 1.047] | 0 |
| conditioned | future | EM | 0.049 [0.038, 0.059] | 0.080 [0.069, 0.092] | 0.105 [0.092, 0.121] | 0.161 [0.141, 0.182] | 0 |
| conditioned | future | F1 | 0.097 [0.084, 0.111] | 0.113 [0.098, 0.128] | 0.128 [0.114, 0.144] | 0.175 [0.158, 0.193] | 0 |
| conditioned | future | judge | 0.078 [0.071, 0.085] | 0.097 [0.085, 0.109] | 0.118 [0.104, 0.132] | 0.169 [0.150, 0.189] | 0 |
| generic | now | EM | 0.052 [0.000, 0.109] | 0.286 [0.182, 0.391] | 0.286 [0.146, 0.432] | 0.510 [0.396, 0.630] | 0 |
| generic | now | F1 | 0.173 [0.135, 0.210] | 0.346 [0.269, 0.430] | 0.423 [0.314, 0.536] | 0.612 [0.512, 0.709] | 0 |
| generic | now | judge | 0.141 [0.078, 0.203] | 0.349 [0.266, 0.438] | 0.479 [0.391, 0.573] | 0.625 [0.516, 0.734] | 0 |
| generic | future | EM | 0.063 [0.034, 0.093] | 0.170 [0.125, 0.219] | 0.199 [0.132, 0.267] | 0.456 [0.379, 0.538] | 0 |
| generic | future | F1 | 0.139 [0.108, 0.169] | 0.231 [0.194, 0.273] | 0.299 [0.247, 0.353] | 0.521 [0.460, 0.589] | 0 |
| generic | future | judge | 0.104 [0.078, 0.127] | 0.208 [0.169, 0.247] | 0.318 [0.261, 0.374] | 0.515 [0.455, 0.583] | 0 |
| oracle | now | EM | 0.266 [0.208, 0.312] | 0.396 [0.312, 0.479] | 0.688 [0.583, 0.792] | 0.828 [0.719, 0.927] | 0 |
| oracle | now | F1 | 0.274 [0.254, 0.291] | 0.476 [0.398, 0.553] | 0.725 [0.643, 0.809] | 0.869 [0.780, 0.946] | 0 |
| oracle | now | judge | 0.255 [0.250, 0.266] | 0.526 [0.443, 0.609] | 0.766 [0.688, 0.844] | 0.901 [0.833, 0.964] | 0 |
| oracle | future | EM | 0.166 [0.132, 0.198] | 0.349 [0.303, 0.390] | 0.631 [0.553, 0.704] | 0.781 [0.688, 0.870] | 0 |
| oracle | future | F1 | 0.215 [0.190, 0.239] | 0.402 [0.367, 0.436] | 0.679 [0.615, 0.739] | 0.821 [0.747, 0.894] | 0 |
| oracle | future | judge | 0.181 [0.163, 0.199] | 0.401 [0.367, 0.435] | 0.698 [0.628, 0.762] | 0.833 [0.765, 0.902] | 0 |
| reusable | now | EM | 0.932 [0.854, 1.000] | 0.964 [0.911, 1.000] | 1.000 [1.000, 1.000] | 0.964 [0.911, 1.000] | 0 |
| reusable | now | F1 | 0.986 [0.962, 1.009] | 0.993 [0.972, 1.011] | 0.995 [0.969, 1.016] | 0.967 [0.917, 1.000] | 0 |
| reusable | now | judge | 1.021 [1.000, 1.062] | 1.021 [1.000, 1.062] | 1.021 [1.000, 1.062] | 0.969 [0.922, 1.000] | 0 |
| reusable | future | EM | 0.053 [0.043, 0.062] | 0.084 [0.073, 0.096] | 0.186 [0.171, 0.200] | 0.319 [0.283, 0.352] | 0 |
| reusable | future | F1 | 0.100 [0.088, 0.112] | 0.121 [0.112, 0.130] | 0.203 [0.187, 0.219] | 0.340 [0.305, 0.371] | 0 |
| reusable | future | judge | 0.082 [0.077, 0.086] | 0.104 [0.094, 0.115] | 0.199 [0.185, 0.213] | 0.338 [0.303, 0.371] | 0 |

### Relation dossiers &mdash; length-matched subsample, all metrics

| comparison | endpoint | metric | 20 w | 40 w | 80 w | 160 w | tolerance_words |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned minus generic | now | EM | **+0.703 [+0.609, +0.781]** | **+0.531 [+0.438, +0.609]** | **+0.500 [+0.383, +0.600]** | **+0.344 [+0.281, +0.438]** | 6 |
| conditioned minus generic | now | F1 | **+0.732 [+0.678, +0.782]** | **+0.567 [+0.493, +0.637]** | **+0.490 [+0.372, +0.597]** | **+0.328 [+0.266, +0.406]** | 6 |
| conditioned minus generic | now | judge | **+0.859 [+0.797, +0.922]** | **+0.641 [+0.547, +0.719]** | **+0.500 [+0.383, +0.600]** | **+0.344 [+0.281, +0.438]** | 6 |
| conditioned minus generic | future | EM | -0.012 [-0.040, +0.014] | **-0.074 [-0.111, -0.042]** | **-0.087 [-0.151, -0.026]** | **-0.210 [-0.281, -0.142]** | 6 |
| conditioned minus generic | future | F1 | **-0.040 [-0.073, -0.007]** | **-0.109 [-0.149, -0.074]** | **-0.164 [-0.218, -0.110]** | **-0.284 [-0.337, -0.229]** | 6 |
| conditioned minus generic | future | judge | -0.026 [-0.051, +0.002] | **-0.110 [-0.153, -0.072]** | **-0.210 [-0.268, -0.148]** | **-0.335 [-0.390, -0.287]** | 6 |
| conditioned minus reusable | now | EM | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.033 [-0.083, +0.000] | +0.031 [+0.000, +0.094] | 6 |
| conditioned minus reusable | now | F1 | +0.000 [+0.000, +0.000] | -0.012 [-0.031, +0.000] | -0.014 [-0.056, +0.017] | +0.016 [-0.047, +0.094] | 6 |
| conditioned minus reusable | now | judge | +0.000 [+0.000, +0.000] | -0.016 [-0.047, +0.000] | -0.017 [-0.050, +0.000] | +0.031 [+0.000, +0.094] | 6 |
| conditioned minus reusable | future | EM | -0.003 [-0.007, +0.001] | -0.003 [-0.014, +0.007] | **-0.066 [-0.080, -0.051]** | **-0.129 [-0.158, -0.098]** | 6 |
| conditioned minus reusable | future | F1 | -0.003 [-0.012, +0.005] | -0.008 [-0.018, +0.003] | **-0.069 [-0.086, -0.050]** | **-0.149 [-0.172, -0.118]** | 6 |
| conditioned minus reusable | future | judge | -0.004 [-0.011, +0.004] | -0.007 [-0.021, +0.006] | **-0.082 [-0.100, -0.064]** | **-0.167 [-0.198, -0.131]** | 6 |
| reusable minus generic | now | EM | **+0.703 [+0.609, +0.781]** | **+0.531 [+0.438, +0.609]** | **+0.533 [+0.433, +0.633]** | **+0.312 [+0.188, +0.438]** | 6 |
| reusable minus generic | now | F1 | **+0.732 [+0.678, +0.782]** | **+0.580 [+0.501, +0.652]** | **+0.504 [+0.402, +0.602]** | **+0.312 [+0.188, +0.438]** | 6 |
| reusable minus generic | now | judge | **+0.859 [+0.797, +0.922]** | **+0.656 [+0.562, +0.734]** | **+0.517 [+0.433, +0.600]** | **+0.312 [+0.188, +0.438]** | 6 |
| reusable minus generic | future | EM | -0.009 [-0.033, +0.015] | **-0.071 [-0.108, -0.039]** | -0.021 [-0.083, +0.038] | **-0.081 [-0.158, -0.004]** | 6 |
| reusable minus generic | future | F1 | **-0.036 [-0.068, -0.008]** | **-0.101 [-0.140, -0.068]** | **-0.095 [-0.145, -0.047]** | **-0.135 [-0.204, -0.070]** | 6 |
| reusable minus generic | future | judge | -0.022 [-0.045, +0.003] | **-0.103 [-0.143, -0.066]** | **-0.128 [-0.178, -0.076]** | **-0.169 [-0.248, -0.098]** | 6 |

### Relation dossiers &mdash; budget interaction, all metrics

| quantity | policy | metric | delta | p | dz |
|---|---:|---:|---:|---:|---:|
| gap_low_minus_high | conditioned | EM | **+0.079 [+0.008, +0.147]** | 0.0258 | 0.54 |
| gap_low_minus_high | conditioned | F1 | **+0.096 [+0.049, +0.149]** | 0.0002 | 0.91 |
| gap_low_minus_high | conditioned | judge | **+0.122 [+0.080, +0.170]** | 0.0000 | 1.27 |
| gap_low_minus_high | generic | EM | -0.038 [-0.113, +0.042] | 0.3222 | -0.23 |
| gap_low_minus_high | generic | F1 | -0.046 [-0.120, +0.029] | 0.2289 | -0.29 |
| gap_low_minus_high | generic | judge | -0.062 [-0.138, +0.017] | 0.1336 | -0.37 |
| gap_low_minus_high | oracle | EM | **+0.071 [+0.013, +0.129]** | 0.0202 | 0.58 |
| gap_low_minus_high | oracle | F1 | +0.024 [-0.020, +0.069] | 0.2894 | 0.26 |
| gap_low_minus_high | oracle | judge | +0.008 [-0.025, +0.042] | 0.7164 | 0.11 |
| gap_low_minus_high | reusable | EM | **+0.210 [+0.134, +0.284]** | 0.0000 | 1.29 |
| gap_low_minus_high | reusable | F1 | **+0.242 [+0.186, +0.302]** | 0.0000 | 1.97 |
| gap_low_minus_high | reusable | judge | **+0.300 [+0.240, +0.367]** | 0.0000 | 2.20 |
| conditioning_gap_low_minus_high | conditioned_minus_generic | EM | **+0.117 [+0.007, +0.229]** | 0.0402 | 0.50 |
| conditioning_gap_low_minus_high | conditioned_minus_generic | F1 | **+0.142 [+0.051, +0.235]** | 0.0019 | 0.74 |
| conditioning_gap_low_minus_high | conditioned_minus_generic | judge | **+0.184 [+0.080, +0.285]** | 0.0002 | 0.85 |

### Relation dossiers &mdash; distance analysis, all metrics

| policy | relation | kind | metric | 20 w | 40 w | 80 w | 160 w | n_cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conditioned | paraphrase | utility | EM | 0.625 [0.484, 0.750] | 0.719 [0.609, 0.828] | 0.703 [0.609, 0.797] | 0.703 [0.594, 0.812] | 64 |
| conditioned | paraphrase | regret | EM | 0.156 [0.078, 0.250] | 0.062 [0.000, 0.125] | 0.078 [0.031, 0.141] | 0.078 [0.016, 0.156] | 64 |
| conditioned | paraphrase | utility | F1 | 0.808 [0.718, 0.884] | 0.869 [0.817, 0.919] | 0.848 [0.791, 0.903] | 0.845 [0.785, 0.901] | 64 |
| conditioned | paraphrase | regret | F1 | 0.067 [0.006, 0.136] | 0.006 [-0.039, 0.043] | 0.027 [-0.008, 0.067] | 0.030 [-0.020, 0.086] | 64 |
| conditioned | paraphrase | utility | judge | 0.984 [0.953, 1.000] | 1.000 [1.000, 1.000] | 0.969 [0.922, 1.000] | 0.969 [0.922, 1.000] | 64 |
| conditioned | paraphrase | regret | judge | 0.000 [-0.047, 0.047] | -0.016 [-0.047, 0.000] | 0.016 [0.000, 0.047] | 0.016 [-0.031, 0.062] | 64 |
| conditioned | same entity | utility | EM | 0.016 [0.000, 0.047] | 0.094 [0.047, 0.156] | 0.219 [0.125, 0.328] | 0.484 [0.359, 0.609] | 64 |
| conditioned | same entity | regret | EM | 0.828 [0.750, 0.906] | 0.750 [0.688, 0.812] | 0.625 [0.531, 0.719] | 0.359 [0.234, 0.484] | 64 |
| conditioned | same entity | utility | F1 | 0.078 [0.032, 0.130] | 0.127 [0.070, 0.185] | 0.249 [0.148, 0.356] | 0.516 [0.375, 0.651] | 64 |
| conditioned | same entity | regret | F1 | 0.867 [0.809, 0.920] | 0.817 [0.765, 0.874] | 0.695 [0.602, 0.783] | 0.429 [0.300, 0.553] | 64 |
| conditioned | same entity | utility | judge | 0.047 [0.000, 0.094] | 0.125 [0.062, 0.188] | 0.266 [0.156, 0.375] | 0.578 [0.469, 0.688] | 64 |
| conditioned | same entity | regret | judge | 0.953 [0.906, 1.000] | 0.875 [0.812, 0.938] | 0.734 [0.625, 0.844] | 0.422 [0.312, 0.531] | 64 |
| conditioned | same topic | utility | EM | 0.000 [0.000, 0.000] | 0.156 [0.094, 0.234] | 0.312 [0.219, 0.406] | 0.484 [0.359, 0.609] | 64 |
| conditioned | same topic | regret | EM | 0.953 [0.875, 1.000] | 0.797 [0.719, 0.875] | 0.641 [0.531, 0.750] | 0.469 [0.344, 0.594] | 64 |
| conditioned | same topic | utility | F1 | 0.052 [0.013, 0.108] | 0.204 [0.121, 0.292] | 0.332 [0.227, 0.432] | 0.498 [0.381, 0.617] | 64 |
| conditioned | same topic | regret | F1 | 0.932 [0.872, 0.977] | 0.780 [0.701, 0.857] | 0.653 [0.554, 0.758] | 0.486 [0.371, 0.602] | 64 |
| conditioned | same topic | utility | judge | 0.016 [0.000, 0.047] | 0.172 [0.094, 0.250] | 0.312 [0.219, 0.406] | 0.484 [0.359, 0.609] | 64 |
| conditioned | same topic | regret | judge | 0.984 [0.953, 1.000] | 0.828 [0.750, 0.906] | 0.688 [0.594, 0.781] | 0.516 [0.391, 0.641] | 64 |
| conditioned | orthogonal | utility | EM | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.012] | 0.009 [0.001, 0.020] | 0.033 [0.013, 0.055] | 768 |
| conditioned | orthogonal | regret | EM | 0.844 [0.797, 0.887] | 0.839 [0.790, 0.882] | 0.835 [0.789, 0.876] | 0.811 [0.764, 0.854] | 768 |
| conditioned | orthogonal | utility | F1 | 0.034 [0.024, 0.045] | 0.031 [0.020, 0.041] | 0.030 [0.020, 0.040] | 0.048 [0.030, 0.069] | 768 |
| conditioned | orthogonal | regret | F1 | 0.891 [0.865, 0.917] | 0.895 [0.867, 0.923] | 0.896 [0.873, 0.919] | 0.877 [0.846, 0.904] | 768 |
| conditioned | orthogonal | utility | judge | 0.009 [0.003, 0.017] | 0.012 [0.005, 0.021] | 0.017 [0.008, 0.027] | 0.040 [0.022, 0.061] | 768 |
| conditioned | orthogonal | regret | judge | 0.983 [0.966, 0.996] | 0.980 [0.961, 0.995] | 0.975 [0.956, 0.991] | 0.952 [0.926, 0.975] | 768 |
| generic | paraphrase | utility | EM | 0.047 [0.000, 0.094] | 0.203 [0.125, 0.297] | 0.203 [0.109, 0.297] | 0.391 [0.250, 0.531] | 64 |
| generic | paraphrase | regret | EM | 0.734 [0.672, 0.797] | 0.578 [0.484, 0.656] | 0.578 [0.484, 0.672] | 0.391 [0.281, 0.500] | 64 |
| generic | paraphrase | utility | F1 | 0.144 [0.102, 0.189] | 0.291 [0.217, 0.380] | 0.374 [0.289, 0.462] | 0.546 [0.442, 0.651] | 64 |
| generic | paraphrase | regret | F1 | 0.731 [0.670, 0.786] | 0.584 [0.515, 0.647] | 0.501 [0.408, 0.594] | 0.329 [0.240, 0.422] | 64 |
| generic | paraphrase | utility | judge | 0.141 [0.078, 0.203] | 0.328 [0.250, 0.422] | 0.469 [0.391, 0.547] | 0.609 [0.516, 0.703] | 64 |
| generic | paraphrase | regret | judge | 0.844 [0.781, 0.906] | 0.656 [0.562, 0.734] | 0.516 [0.422, 0.609] | 0.375 [0.266, 0.484] | 64 |
| generic | same entity | utility | EM | 0.094 [0.031, 0.172] | 0.109 [0.047, 0.188] | 0.109 [0.047, 0.188] | 0.500 [0.375, 0.625] | 64 |
| generic | same entity | regret | EM | 0.750 [0.656, 0.828] | 0.734 [0.625, 0.844] | 0.734 [0.672, 0.797] | 0.344 [0.234, 0.469] | 64 |
| generic | same entity | utility | F1 | 0.131 [0.063, 0.208] | 0.159 [0.077, 0.249] | 0.194 [0.126, 0.268] | 0.549 [0.435, 0.666] | 64 |
| generic | same entity | regret | F1 | 0.813 [0.747, 0.874] | 0.785 [0.692, 0.870] | 0.750 [0.692, 0.807] | 0.395 [0.286, 0.505] | 64 |
| generic | same entity | utility | judge | 0.109 [0.047, 0.188] | 0.125 [0.047, 0.219] | 0.188 [0.125, 0.250] | 0.562 [0.453, 0.672] | 64 |
| generic | same entity | regret | judge | 0.891 [0.812, 0.953] | 0.875 [0.781, 0.953] | 0.812 [0.750, 0.875] | 0.438 [0.328, 0.547] | 64 |
| generic | same topic | utility | EM | 0.031 [0.000, 0.078] | 0.047 [0.000, 0.094] | 0.141 [0.062, 0.219] | 0.266 [0.156, 0.375] | 64 |
| generic | same topic | regret | EM | 0.922 [0.844, 0.984] | 0.906 [0.797, 0.984] | 0.812 [0.703, 0.922] | 0.688 [0.562, 0.812] | 64 |
| generic | same topic | utility | F1 | 0.093 [0.039, 0.159] | 0.115 [0.068, 0.164] | 0.183 [0.097, 0.273] | 0.302 [0.194, 0.413] | 64 |
| generic | same topic | regret | F1 | 0.891 [0.826, 0.948] | 0.869 [0.809, 0.923] | 0.801 [0.699, 0.897] | 0.682 [0.568, 0.797] | 64 |
| generic | same topic | utility | judge | 0.031 [0.000, 0.078] | 0.062 [0.016, 0.125] | 0.172 [0.078, 0.266] | 0.281 [0.172, 0.391] | 64 |
| generic | same topic | regret | judge | 0.969 [0.922, 1.000] | 0.938 [0.875, 0.984] | 0.828 [0.734, 0.922] | 0.719 [0.609, 0.828] | 64 |
| generic | orthogonal | utility | EM | 0.055 [0.031, 0.082] | 0.148 [0.113, 0.191] | 0.176 [0.113, 0.238] | 0.395 [0.320, 0.473] | 768 |
| generic | orthogonal | regret | EM | 0.789 [0.746, 0.832] | 0.695 [0.633, 0.750] | 0.668 [0.609, 0.730] | 0.449 [0.387, 0.504] | 768 |
| generic | orthogonal | utility | F1 | 0.131 [0.103, 0.159] | 0.219 [0.184, 0.261] | 0.284 [0.233, 0.334] | 0.488 [0.429, 0.553] | 768 |
| generic | orthogonal | regret | F1 | 0.795 [0.761, 0.829] | 0.706 [0.659, 0.750] | 0.642 [0.590, 0.696] | 0.437 [0.377, 0.493] | 768 |
| generic | orthogonal | utility | judge | 0.105 [0.078, 0.129] | 0.215 [0.172, 0.258] | 0.324 [0.266, 0.379] | 0.516 [0.457, 0.574] | 768 |
| generic | orthogonal | regret | judge | 0.887 [0.859, 0.914] | 0.777 [0.734, 0.820] | 0.668 [0.609, 0.727] | 0.477 [0.410, 0.539] | 768 |
| oracle | paraphrase | utility | EM | 0.188 [0.125, 0.234] | 0.312 [0.219, 0.406] | 0.500 [0.391, 0.609] | 0.625 [0.484, 0.766] | 64 |
| oracle | paraphrase | regret | EM | 0.594 [0.531, 0.656] | 0.469 [0.406, 0.531] | 0.281 [0.188, 0.391] | 0.156 [0.078, 0.250] | 64 |
| oracle | paraphrase | utility | F1 | 0.254 [0.228, 0.282] | 0.409 [0.322, 0.503] | 0.623 [0.534, 0.716] | 0.762 [0.669, 0.851] | 64 |
| oracle | paraphrase | regret | F1 | 0.621 [0.578, 0.661] | 0.466 [0.377, 0.545] | 0.252 [0.166, 0.341] | 0.113 [0.036, 0.198] | 64 |
| oracle | paraphrase | utility | judge | 0.250 [0.250, 0.250] | 0.484 [0.391, 0.578] | 0.750 [0.672, 0.828] | 0.891 [0.828, 0.953] | 64 |
| oracle | paraphrase | regret | judge | 0.734 [0.703, 0.750] | 0.500 [0.406, 0.594] | 0.234 [0.156, 0.312] | 0.094 [0.031, 0.156] | 64 |
| oracle | same entity | utility | EM | 0.125 [0.062, 0.188] | 0.266 [0.188, 0.344] | 0.469 [0.344, 0.594] | 0.625 [0.516, 0.734] | 64 |
| oracle | same entity | regret | EM | 0.719 [0.625, 0.812] | 0.578 [0.500, 0.672] | 0.375 [0.266, 0.484] | 0.219 [0.125, 0.328] | 64 |
| oracle | same entity | utility | F1 | 0.173 [0.105, 0.240] | 0.327 [0.255, 0.401] | 0.588 [0.491, 0.683] | 0.719 [0.629, 0.804] | 64 |
| oracle | same entity | regret | F1 | 0.771 [0.707, 0.839] | 0.617 [0.552, 0.684] | 0.356 [0.270, 0.445] | 0.225 [0.135, 0.324] | 64 |
| oracle | same entity | utility | judge | 0.172 [0.109, 0.219] | 0.328 [0.250, 0.406] | 0.641 [0.547, 0.734] | 0.781 [0.703, 0.859] | 64 |
| oracle | same entity | regret | judge | 0.828 [0.781, 0.891] | 0.672 [0.594, 0.750] | 0.359 [0.266, 0.453] | 0.219 [0.141, 0.297] | 64 |
| oracle | same topic | utility | EM | 0.062 [0.016, 0.125] | 0.297 [0.250, 0.344] | 0.625 [0.547, 0.719] | 0.750 [0.609, 0.875] | 64 |
| oracle | same topic | regret | EM | 0.891 [0.812, 0.953] | 0.656 [0.578, 0.719] | 0.328 [0.234, 0.422] | 0.203 [0.094, 0.328] | 64 |
| oracle | same topic | utility | F1 | 0.137 [0.079, 0.201] | 0.342 [0.295, 0.392] | 0.661 [0.590, 0.732] | 0.789 [0.672, 0.897] | 64 |
| oracle | same topic | regret | F1 | 0.848 [0.783, 0.909] | 0.642 [0.594, 0.690] | 0.324 [0.253, 0.396] | 0.196 [0.094, 0.305] | 64 |
| oracle | same topic | utility | judge | 0.062 [0.016, 0.125] | 0.297 [0.250, 0.344] | 0.641 [0.562, 0.719] | 0.766 [0.641, 0.875] | 64 |
| oracle | same topic | regret | judge | 0.938 [0.875, 0.984] | 0.703 [0.656, 0.750] | 0.359 [0.281, 0.438] | 0.234 [0.125, 0.359] | 64 |
| oracle | orthogonal | utility | EM | 0.148 [0.117, 0.176] | 0.301 [0.254, 0.348] | 0.535 [0.465, 0.609] | 0.668 [0.574, 0.758] | 768 |
| oracle | orthogonal | regret | EM | 0.695 [0.660, 0.730] | 0.543 [0.508, 0.574] | 0.309 [0.242, 0.383] | 0.176 [0.105, 0.246] | 768 |
| oracle | orthogonal | utility | F1 | 0.202 [0.179, 0.226] | 0.377 [0.341, 0.412] | 0.631 [0.570, 0.689] | 0.763 [0.689, 0.836] | 768 |
| oracle | orthogonal | regret | F1 | 0.723 [0.699, 0.747] | 0.549 [0.514, 0.582] | 0.295 [0.236, 0.356] | 0.162 [0.097, 0.230] | 768 |
| oracle | orthogonal | utility | judge | 0.184 [0.168, 0.199] | 0.406 [0.367, 0.445] | 0.695 [0.629, 0.758] | 0.832 [0.758, 0.902] | 768 |
| oracle | orthogonal | regret | judge | 0.809 [0.785, 0.832] | 0.586 [0.551, 0.621] | 0.297 [0.230, 0.367] | 0.160 [0.094, 0.227] | 768 |
| reusable | paraphrase | utility | EM | 0.609 [0.469, 0.734] | 0.656 [0.547, 0.766] | 0.719 [0.625, 0.812] | 0.719 [0.609, 0.828] | 64 |
| reusable | paraphrase | regret | EM | 0.172 [0.078, 0.266] | 0.125 [0.062, 0.203] | 0.062 [0.016, 0.125] | 0.062 [0.000, 0.141] | 64 |
| reusable | paraphrase | utility | F1 | 0.798 [0.704, 0.880] | 0.847 [0.797, 0.897] | 0.853 [0.791, 0.909] | 0.843 [0.778, 0.906] | 64 |
| reusable | paraphrase | regret | F1 | 0.077 [0.002, 0.159] | 0.028 [-0.007, 0.061] | 0.022 [-0.019, 0.060] | 0.032 [-0.010, 0.084] | 64 |
| reusable | paraphrase | utility | judge | 0.984 [0.953, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.953 [0.906, 1.000] | 64 |
| reusable | paraphrase | regret | judge | 0.000 [-0.047, 0.047] | -0.016 [-0.047, 0.000] | -0.016 [-0.047, 0.000] | 0.031 [0.000, 0.078] | 64 |
| reusable | same entity | utility | EM | 0.031 [0.000, 0.078] | 0.156 [0.047, 0.281] | 0.562 [0.469, 0.656] | 0.594 [0.469, 0.719] | 64 |
| reusable | same entity | regret | EM | 0.812 [0.734, 0.891] | 0.688 [0.578, 0.797] | 0.281 [0.188, 0.375] | 0.250 [0.156, 0.344] | 64 |
| reusable | same entity | utility | F1 | 0.096 [0.050, 0.146] | 0.206 [0.100, 0.330] | 0.617 [0.528, 0.707] | 0.686 [0.581, 0.790] | 64 |
| reusable | same entity | regret | F1 | 0.848 [0.789, 0.902] | 0.738 [0.624, 0.843] | 0.328 [0.234, 0.425] | 0.258 [0.165, 0.357] | 64 |
| reusable | same entity | utility | judge | 0.047 [0.000, 0.094] | 0.188 [0.078, 0.312] | 0.672 [0.562, 0.781] | 0.734 [0.625, 0.828] | 64 |
| reusable | same entity | regret | judge | 0.953 [0.906, 1.000] | 0.812 [0.688, 0.922] | 0.328 [0.219, 0.438] | 0.266 [0.172, 0.375] | 64 |
| reusable | same topic | utility | EM | 0.047 [0.000, 0.094] | 0.250 [0.172, 0.328] | 0.438 [0.328, 0.547] | 0.656 [0.531, 0.750] | 64 |
| reusable | same topic | regret | EM | 0.906 [0.828, 0.969] | 0.703 [0.625, 0.781] | 0.516 [0.391, 0.641] | 0.297 [0.188, 0.422] | 64 |
| reusable | same topic | utility | F1 | 0.091 [0.031, 0.158] | 0.303 [0.222, 0.385] | 0.452 [0.346, 0.559] | 0.676 [0.569, 0.760] | 64 |
| reusable | same topic | regret | F1 | 0.894 [0.829, 0.953] | 0.682 [0.605, 0.756] | 0.532 [0.425, 0.641] | 0.308 [0.222, 0.418] | 64 |
| reusable | same topic | utility | judge | 0.062 [0.016, 0.125] | 0.250 [0.172, 0.328] | 0.438 [0.328, 0.547] | 0.672 [0.562, 0.766] | 64 |
| reusable | same topic | regret | judge | 0.938 [0.875, 0.984] | 0.750 [0.672, 0.828] | 0.562 [0.453, 0.672] | 0.328 [0.234, 0.438] | 64 |
| reusable | orthogonal | utility | EM | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.004] | 0.053 [0.036, 0.070] | 0.176 [0.134, 0.216] | 768 |
| reusable | orthogonal | regret | EM | 0.844 [0.797, 0.887] | 0.842 [0.793, 0.887] | 0.790 [0.747, 0.828] | 0.668 [0.626, 0.712] | 768 |
| reusable | orthogonal | utility | F1 | 0.034 [0.026, 0.042] | 0.027 [0.018, 0.036] | 0.075 [0.055, 0.094] | 0.210 [0.168, 0.250] | 768 |
| reusable | orthogonal | regret | F1 | 0.892 [0.867, 0.916] | 0.899 [0.871, 0.925] | 0.851 [0.822, 0.877] | 0.716 [0.676, 0.756] | 768 |
| reusable | orthogonal | utility | judge | 0.010 [0.004, 0.017] | 0.009 [0.003, 0.017] | 0.070 [0.052, 0.089] | 0.221 [0.180, 0.259] | 768 |
| reusable | orthogonal | regret | judge | 0.982 [0.965, 0.993] | 0.983 [0.962, 0.996] | 0.922 [0.895, 0.945] | 0.771 [0.723, 0.818] | 768 |

### Relation dossiers &mdash; distance monotonicity tests, all metrics

| policy | quantity | metric | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|---:|---:|
| conditioned | regret orthogonal minus paraphrase | EM | **+0.688 [+0.574, +0.797]** | **+0.776 [+0.682, +0.866]** | **+0.757 [+0.674, +0.831]** | **+0.733 [+0.637, +0.815]** |
| conditioned | regret orthogonal minus paraphrase | F1 | **+0.825 [+0.741, +0.893]** | **+0.889 [+0.841, +0.936]** | **+0.869 [+0.822, +0.911]** | **+0.847 [+0.789, +0.903]** |
| conditioned | regret orthogonal minus paraphrase | judge | **+0.983 [+0.941, +1.016]** | **+0.996 [+0.982, +1.014]** | **+0.960 [+0.921, +0.988]** | **+0.936 [+0.884, +0.982]** |
| generic | regret orthogonal minus paraphrase | EM | **+0.055 [+0.012, +0.102]** | **+0.117 [+0.062, +0.168]** | **+0.090 [+0.039, +0.141]** | +0.059 [-0.012, +0.125] |
| generic | regret orthogonal minus paraphrase | F1 | **+0.064 [+0.022, +0.106]** | **+0.122 [+0.092, +0.153]** | **+0.141 [+0.087, +0.200]** | **+0.108 [+0.048, +0.164]** |
| generic | regret orthogonal minus paraphrase | judge | +0.043 [-0.004, +0.086] | **+0.121 [+0.070, +0.176]** | **+0.152 [+0.102, +0.211]** | **+0.102 [+0.035, +0.164]** |
| oracle | regret orthogonal minus paraphrase | EM | **+0.102 [+0.055, +0.148]** | **+0.074 [+0.027, +0.117]** | +0.027 [-0.043, +0.086] | +0.020 [-0.035, +0.070] |
| oracle | regret orthogonal minus paraphrase | F1 | **+0.103 [+0.069, +0.142]** | **+0.083 [+0.024, +0.143]** | +0.043 [-0.013, +0.099] | +0.049 [-0.004, +0.109] |
| oracle | regret orthogonal minus paraphrase | judge | **+0.074 [+0.055, +0.098]** | **+0.086 [+0.027, +0.145]** | **+0.062 [+0.020, +0.105]** | **+0.066 [+0.027, +0.113]** |
| reusable | regret orthogonal minus paraphrase | EM | **+0.672 [+0.551, +0.785]** | **+0.717 [+0.616, +0.812]** | **+0.728 [+0.646, +0.799]** | **+0.605 [+0.514, +0.685]** |
| reusable | regret orthogonal minus paraphrase | F1 | **+0.815 [+0.726, +0.894]** | **+0.871 [+0.827, +0.912]** | **+0.829 [+0.782, +0.871]** | **+0.684 [+0.614, +0.747]** |
| reusable | regret orthogonal minus paraphrase | judge | **+0.982 [+0.940, +1.014]** | **+0.999 [+0.986, +1.016]** | **+0.938 [+0.917, +0.958]** | **+0.740 [+0.671, +0.803]** |

---

*Agent Handoff Information-Loss Probe · Experiment 10 · sender and answerer `meta-llama/llama-3.1-8b-instruct`, judge `openai/gpt-4o-mini`. Generated by `src/render_regret_summary.py`.*
