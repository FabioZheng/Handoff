# exp5_uncapped natural-length calibration

Writer `mistral24`, split `development`, Experiment 5 prompt with no length clause.
Only messages that stopped on their own are summarised; any that hit the 4096-token ceiling are counted separately.

| corpus | arm | n | mean words | median | p10 | p90 | mean tokens | hit ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qasper | generic | 39 | 318 | 279 | 198 | 452 | 521 | 1 |
| qasper | conditioned | 80 | 213 | 198 | 102 | 333 | 305 | 0 |
| squad | generic | 39 | 527 | 476 | 316 | 874 | 815 | 1 |
| squad | conditioned | 78 | 139 | 86 | 28 | 227 | 192 | 2 |

| corpus | conditioned/generic median | conditioned/generic mean |
|---|---:|---:|
| qasper | 0.71 | 0.67 |
| squad | 0.18 | 0.26 |
