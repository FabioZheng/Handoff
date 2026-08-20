# Repeated-handoff degradation results

Datasets: musique, hotpotqa  
Evidence variants: short, medium, full  
Compression depths: 0, 1, 3, 5
Model: Qwen3 8B (`qwen/qwen3-8b`), non-thinking mode

All evidence variants retain every gold document. Short uses gold documents only; medium adds distractors up to five documents; full uses the complete dataset context.

BERTScore uses roberta-large layer 17, best over answer aliases, with English baseline rescaling=True.

| dataset | evidence | depth | n | EM | Token F1 | BERTScore F1 | Token F1 change | BERTScore change | docs | context chars | handoff chars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | full | 0 | 10 | 0.500 | 0.702 | 0.715 | +0.000 | +0.000 | 10.0 | 5771 | 5771 |
| hotpotqa | full | 1 | 10 | 0.500 | 0.650 | 0.744 | -0.052 | +0.029 | 10.0 | 5771 | 628 |
| hotpotqa | full | 3 | 10 | 0.400 | 0.594 | 0.686 | -0.108 | -0.028 | 10.0 | 5771 | 495 |
| hotpotqa | full | 5 | 10 | 0.400 | 0.594 | 0.686 | -0.108 | -0.028 | 10.0 | 5771 | 461 |
| hotpotqa | medium | 0 | 10 | 0.500 | 0.736 | 0.766 | +0.000 | +0.000 | 5.0 | 2854 | 2854 |
| hotpotqa | medium | 1 | 10 | 0.500 | 0.650 | 0.736 | -0.086 | -0.029 | 5.0 | 2854 | 389 |
| hotpotqa | medium | 3 | 10 | 0.500 | 0.650 | 0.736 | -0.086 | -0.029 | 5.0 | 2854 | 358 |
| hotpotqa | medium | 5 | 10 | 0.500 | 0.650 | 0.736 | -0.086 | -0.029 | 5.0 | 2854 | 361 |
| hotpotqa | short | 0 | 10 | 0.600 | 0.750 | 0.780 | +0.000 | +0.000 | 2.0 | 796 | 796 |
| hotpotqa | short | 1 | 10 | 0.600 | 0.750 | 0.776 | +0.000 | -0.003 | 2.0 | 796 | 329 |
| hotpotqa | short | 3 | 10 | 0.500 | 0.640 | 0.643 | -0.110 | -0.136 | 2.0 | 796 | 333 |
| hotpotqa | short | 5 | 10 | 0.500 | 0.640 | 0.643 | -0.110 | -0.136 | 2.0 | 796 | 337 |
| musique | full | 0 | 10 | 0.300 | 0.423 | 0.621 | +0.000 | +0.000 | 20.0 | 10501 | 10501 |
| musique | full | 1 | 10 | 0.400 | 0.490 | 0.611 | +0.067 | -0.010 | 20.0 | 10501 | 1140 |
| musique | full | 3 | 10 | 0.500 | 0.540 | 0.690 | +0.117 | +0.068 | 20.0 | 10501 | 601 |
| musique | full | 5 | 10 | 0.500 | 0.540 | 0.639 | +0.117 | +0.017 | 20.0 | 10501 | 605 |
| musique | medium | 0 | 10 | 0.700 | 0.790 | 0.817 | +0.000 | +0.000 | 5.0 | 2543 | 2543 |
| musique | medium | 1 | 10 | 1.000 | 1.000 | 1.000 | +0.210 | +0.183 | 5.0 | 2543 | 587 |
| musique | medium | 3 | 10 | 0.800 | 0.867 | 0.854 | +0.077 | +0.037 | 5.0 | 2543 | 449 |
| musique | medium | 5 | 10 | 0.800 | 0.867 | 0.854 | +0.077 | +0.037 | 5.0 | 2543 | 449 |
| musique | short | 0 | 10 | 0.600 | 0.707 | 0.864 | +0.000 | +0.000 | 2.9 | 1383 | 1383 |
| musique | short | 1 | 10 | 0.700 | 0.733 | 0.852 | +0.027 | -0.013 | 2.9 | 1383 | 342 |
| musique | short | 3 | 10 | 0.800 | 0.833 | 0.936 | +0.127 | +0.071 | 2.9 | 1383 | 334 |
| musique | short | 5 | 10 | 0.800 | 0.833 | 0.936 | +0.127 | +0.071 | 2.9 | 1383 | 320 |

## Cost

- reconciled API spend: **$0.0443**
- calls: 240 Qwen-specific leakage-filter calls, 300 handoffs, 240 final answers, plus one compatibility smoke call
- prompt/completion tokens: 246,541 / 34,052 (plus the 17 / 1 smoke-call tokens)
- note: the initial worker process stalled after some successful calls and was safely stopped; this reconciled figure includes those cached calls as well as the resumed worker, whereas its end-of-run ledger did not.
