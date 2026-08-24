# Incremental-evidence handoff results

MuSiQue supporting paragraphs arrive one packet at a time. Relay-only agents receive no new evidence.

Conditions: question_conditioned, question_omitted  
Relay-only agents between specialists: 0, 1, 3, 5  
Seeds: 1, 2  
Evidence order: counterbalanced

## Final multi-hop answer

Original-evidence baseline: EM 0.500, F1 0.610

| condition | relays between specialists | n | stages | EM | F1 | judge | handoff chars | chain tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| question_conditioned | 0 | 30 | 2.7 | 0.483 | 0.601 | 0.667 | 1640 | 2044 |
| question_conditioned | 1 | 30 | 4.3 | 0.517 | 0.629 | 0.700 | 1465 | 2692 |
| question_conditioned | 3 | 30 | 7.7 | 0.517 | 0.634 | 0.717 | 1310 | 4107 |
| question_conditioned | 5 | 30 | 11.0 | 0.583 | 0.651 | 0.683 | 1302 | 5548 |
| question_omitted | 0 | 30 | 2.7 | 0.517 | 0.624 | 0.700 | 1554 | 1706 |
| question_omitted | 1 | 30 | 4.3 | 0.533 | 0.655 | 0.733 | 1462 | 2404 |
| question_omitted | 3 | 30 | 7.7 | 0.550 | 0.655 | 0.717 | 1432 | 3887 |
| question_omitted | 5 | 30 | 11.0 | 0.483 | 0.578 | 0.633 | 1369 | 5242 |

## Future-query regret

Positive values mean the final handoff answers hidden packet probes worse than the original evidence.

| condition | relays between specialists | n | EM regret | F1 regret | judge regret |
|---|---:|---:|---:|---:|---:|
| question_conditioned | 0 | 30 | -0.047 | -0.050 | -0.090 |
| question_conditioned | 1 | 30 | -0.085 | -0.068 | -0.096 |
| question_conditioned | 3 | 30 | -0.044 | -0.038 | -0.097 |
| question_conditioned | 5 | 30 | -0.053 | -0.018 | -0.047 |
| question_omitted | 0 | 30 | -0.029 | -0.069 | -0.103 |
| question_omitted | 1 | 30 | -0.061 | -0.061 | -0.083 |
| question_omitted | 3 | 30 | -0.025 | -0.027 | -0.081 |
| question_omitted | 5 | 30 | -0.036 | -0.035 | -0.076 |

## Outputs

- `stage_metrics.csv`: existing-compatible final-answer metrics keyed by relay depth.
- `relay_depth_deltas.csv`: paired relay-depth minus no-relay contrasts.
- `probe_metrics.csv`: hidden-probe accuracy by handoff age.
- `future_query_regret.csv`: original-evidence minus final-handoff probe scores.
- `survival_by_age.csv` and `survival.jsonl`: direct fact survival by age.
- `incremental_chain.png`: combined QA, regret, probe, and fact-survival plot.

## Cost

- live spend: $0.6942
- calls: 7019 live, 1693 cached
- prompt/completion tokens: 2884838 / 610629
