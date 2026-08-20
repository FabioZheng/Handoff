# Repeated-handoff degradation results

Datasets: musique, hotpotqa  
Evidence variants: short, medium, full  
Compression depths: 0, 1, 3, 5

Summaries are question-conditioned.

All evidence variants retain every gold document. Short uses gold documents only; medium adds distractors up to five documents; full uses the complete dataset context.

LLM-judge accuracy is the fraction of answers a judge model (disabled) rules equivalent to a gold answer, at temperature 0. EM and token F1 remain the deterministic primary metrics.

| dataset | evidence | depth | n | EM | Token F1 | Judge acc | Token F1 change | Judge change | docs | context chars | handoff chars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | full | 0 | 10 | 0.500 | 0.702 | nan | +0.000 | +0.000 | 10.0 | 5771 | 5771 |
| hotpotqa | full | 1 | 10 | 0.500 | 0.650 | nan | -0.052 | +0.000 | 10.0 | 5771 | 628 |
| hotpotqa | full | 3 | 10 | 0.400 | 0.594 | nan | -0.108 | +0.000 | 10.0 | 5771 | 495 |
| hotpotqa | full | 5 | 10 | 0.400 | 0.594 | nan | -0.108 | +0.000 | 10.0 | 5771 | 461 |
| hotpotqa | medium | 0 | 10 | 0.500 | 0.736 | nan | +0.000 | +0.000 | 5.0 | 2854 | 2854 |
| hotpotqa | medium | 1 | 10 | 0.500 | 0.650 | nan | -0.086 | +0.000 | 5.0 | 2854 | 389 |
| hotpotqa | medium | 3 | 10 | 0.500 | 0.650 | nan | -0.086 | +0.000 | 5.0 | 2854 | 358 |
| hotpotqa | medium | 5 | 10 | 0.500 | 0.650 | nan | -0.086 | +0.000 | 5.0 | 2854 | 361 |
| hotpotqa | short | 0 | 10 | 0.600 | 0.750 | nan | +0.000 | +0.000 | 2.0 | 796 | 796 |
| hotpotqa | short | 1 | 10 | 0.600 | 0.750 | nan | +0.000 | +0.000 | 2.0 | 796 | 329 |
| hotpotqa | short | 3 | 10 | 0.500 | 0.640 | nan | -0.110 | +0.000 | 2.0 | 796 | 333 |
| hotpotqa | short | 5 | 10 | 0.500 | 0.640 | nan | -0.110 | +0.000 | 2.0 | 796 | 337 |
| musique | full | 0 | 10 | 0.300 | 0.423 | nan | +0.000 | +0.000 | 20.0 | 10501 | 10501 |
| musique | full | 1 | 10 | 0.400 | 0.490 | nan | +0.067 | +0.000 | 20.0 | 10501 | 1140 |
| musique | full | 3 | 10 | 0.500 | 0.540 | nan | +0.117 | +0.000 | 20.0 | 10501 | 601 |
| musique | full | 5 | 10 | 0.500 | 0.540 | nan | +0.117 | +0.000 | 20.0 | 10501 | 605 |
| musique | medium | 0 | 10 | 0.700 | 0.790 | nan | +0.000 | +0.000 | 5.0 | 2543 | 2543 |
| musique | medium | 1 | 10 | 1.000 | 1.000 | nan | +0.210 | +0.000 | 5.0 | 2543 | 587 |
| musique | medium | 3 | 10 | 0.800 | 0.867 | nan | +0.077 | +0.000 | 5.0 | 2543 | 449 |
| musique | medium | 5 | 10 | 0.800 | 0.867 | nan | +0.077 | +0.000 | 5.0 | 2543 | 449 |
| musique | short | 0 | 10 | 0.600 | 0.707 | nan | +0.000 | +0.000 | 2.9 | 1383 | 1383 |
| musique | short | 1 | 10 | 0.700 | 0.733 | nan | +0.027 | +0.000 | 2.9 | 1383 | 342 |
| musique | short | 3 | 10 | 0.800 | 0.833 | nan | +0.127 | +0.000 | 2.9 | 1383 | 334 |
| musique | short | 5 | 10 | 0.800 | 0.833 | nan | +0.127 | +0.000 | 2.9 | 1383 | 320 |

## Cost

- live spend: $0.0000
- calls: 0 live, 0 cached
- prompt/completion tokens: 0 / 0
