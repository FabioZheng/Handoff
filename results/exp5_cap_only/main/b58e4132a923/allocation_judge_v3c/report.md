# exp5_cap_only: where added capacity goes (run `b58e4132a923`)

20519 new sentences across all cap steps; 19511 put to the study's judge, 0 left unanswered. Each sentence gets up to three yes/no questions and its category follows from the answers. Shares are of words in new sentences, ratio of totals with 95% document-bootstrap intervals.

| policy | step | new sentences | current-question evidence | other-question evidence | supported, no question | redundant | not in source | no content |
|---|---|---:|---|---|---|---|---|---|
| summary_generic | 96->120 | 840 | 0.02 [0.01, 0.03] | 0.04 [0.03, 0.05] | 0.18 [0.15, 0.21] | 0.70 [0.67, 0.74] | 0.05 [0.03, 0.06] | 0.01 [0.01, 0.01] |
| summary_generic | 120->144 | 994 | 0.02 [0.02, 0.03] | 0.04 [0.03, 0.06] | 0.15 [0.12, 0.17] | 0.73 [0.69, 0.76] | 0.05 [0.03, 0.06] | 0.01 [0.01, 0.02] |
| summary_generic | 144->176 | 1269 | 0.02 [0.01, 0.02] | 0.03 [0.02, 0.04] | 0.17 [0.15, 0.19] | 0.72 [0.69, 0.75] | 0.04 [0.03, 0.06] | 0.01 [0.01, 0.02] |
| summary_generic | 176->208 | 1426 | 0.02 [0.02, 0.03] | 0.03 [0.03, 0.04] | 0.15 [0.13, 0.18] | 0.72 [0.70, 0.75] | 0.05 [0.03, 0.06] | 0.02 [0.02, 0.03] |
| summary_generic | 208->256 | 1722 | 0.02 [0.02, 0.03] | 0.05 [0.03, 0.06] | 0.17 [0.14, 0.19] | 0.71 [0.68, 0.74] | 0.04 [0.02, 0.05] | 0.02 [0.02, 0.02] |
| summary_conditioned | 96->120 | 2289 | 0.06 [0.05, 0.07] | 0.04 [0.03, 0.05] | 0.17 [0.15, 0.19] | 0.65 [0.63, 0.68] | 0.07 [0.05, 0.08] | 0.01 [0.01, 0.01] |
| summary_conditioned | 120->144 | 2064 | 0.05 [0.04, 0.06] | 0.04 [0.04, 0.05] | 0.19 [0.16, 0.21] | 0.64 [0.61, 0.66] | 0.07 [0.06, 0.09] | 0.01 [0.01, 0.01] |
| summary_conditioned | 144->176 | 3093 | 0.05 [0.04, 0.06] | 0.05 [0.04, 0.05] | 0.17 [0.15, 0.18] | 0.65 [0.63, 0.67] | 0.08 [0.07, 0.09] | 0.01 [0.01, 0.01] |
| summary_conditioned | 176->208 | 2956 | 0.04 [0.03, 0.05] | 0.04 [0.03, 0.05] | 0.17 [0.16, 0.19] | 0.67 [0.65, 0.69] | 0.07 [0.06, 0.08] | 0.01 [0.01, 0.01] |
| summary_conditioned | 208->256 | 3866 | 0.04 [0.03, 0.05] | 0.04 [0.03, 0.04] | 0.17 [0.16, 0.19] | 0.67 [0.65, 0.69] | 0.07 [0.06, 0.08] | 0.01 [0.01, 0.01] |

Finer categories (share of added words):

| policy | step | current | related | distant | supported_other | restated | repetition | plausible_unsupported | unsupported | no_content | unanswered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| summary_generic | 96->120 | 0.02 | 0.01 | 0.03 | 0.18 | 0.70 | 0.00 | 0.05 | 0.00 | 0.01 | 0.00 |
| summary_generic | 120->144 | 0.02 | 0.01 | 0.03 | 0.15 | 0.73 | 0.00 | 0.05 | 0.00 | 0.01 | 0.00 |
| summary_generic | 144->176 | 0.02 | 0.02 | 0.02 | 0.17 | 0.72 | 0.00 | 0.04 | 0.00 | 0.01 | 0.00 |
| summary_generic | 176->208 | 0.02 | 0.01 | 0.02 | 0.15 | 0.72 | 0.00 | 0.04 | 0.00 | 0.02 | 0.00 |
| summary_generic | 208->256 | 0.02 | 0.02 | 0.03 | 0.17 | 0.71 | 0.00 | 0.03 | 0.00 | 0.02 | 0.00 |
| summary_conditioned | 96->120 | 0.06 | 0.02 | 0.02 | 0.17 | 0.65 | 0.00 | 0.06 | 0.00 | 0.01 | 0.00 |
| summary_conditioned | 120->144 | 0.05 | 0.02 | 0.02 | 0.19 | 0.64 | 0.00 | 0.07 | 0.00 | 0.01 | 0.00 |
| summary_conditioned | 144->176 | 0.05 | 0.02 | 0.02 | 0.17 | 0.65 | 0.00 | 0.08 | 0.00 | 0.01 | 0.00 |
| summary_conditioned | 176->208 | 0.04 | 0.02 | 0.02 | 0.17 | 0.67 | 0.00 | 0.07 | 0.00 | 0.01 | 0.00 |
| summary_conditioned | 208->256 | 0.04 | 0.02 | 0.02 | 0.17 | 0.67 | 0.00 | 0.07 | 0.00 | 0.01 | 0.00 |

## Judge audit

Each question is checked against a deterministic signal it should move with. `best source-sentence overlap` is the largest share of the sentence's content words found in a single source sentence, so a YES to "source states it" should sit above a NO. `earlier note overlap` is the share the shorter message already carried, so a YES to "earlier note has it" should sit above a NO. A signal that does not separate the answers means the labels are not trustworthy.

| question | answer | sentences | share | deterministic signal | mean |
|---|---|---:|---:|---|---:|
| source states it | YES | 18200 | 0.93 | `best_source_sentence_overlap` | 0.58 |
| source states it | NO | 1220 | 0.06 | `best_source_sentence_overlap` | 0.39 |
| source states it | NONE | 91 | 0.00 | `best_source_sentence_overlap` | 0.44 |
| source states it | unanswered | 0 | 0.00 | `best_source_sentence_overlap` | — |
| earlier note has it | YES | 13192 | 0.72 | `earlier_note_overlap` | 0.63 |
| earlier note has it | NO | 5008 | 0.28 | `earlier_note_overlap` | 0.31 |
| earlier note has it | unanswered | 0 | 0.00 | `earlier_note_overlap` | — |
| plausible | YES | 1187 | 0.97 | `best_source_sentence_overlap` | 0.39 |
| plausible | NO | 33 | 0.03 | `best_source_sentence_overlap` | 0.48 |
| plausible | unanswered | 0 | 0.00 | `best_source_sentence_overlap` | — |

![allocation_judge.png](allocation_judge.png)
