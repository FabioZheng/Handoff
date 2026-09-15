# Experiment 9 — handoff size adaptation

Model under test: `meta-llama/llama-3.3-70b-instruct`. Judge: `openai/gpt-4o-mini`. Depths [0, 1, 2, 3], 9 arms.

Base handoff instruction is Experiment 5's `conditioned` prompt with the single word
"concise" removed, so the control carries no size instruction at all. Every directive
is appended to that neutral base; each `resize_*` string is its `inform_*` string plus
one clause, so inform-vs-resize isolates the request.

## Items

- **counterfactual**: 26 items, 156 tracked facts (26 with a verified memorised alternative answer).
- **fictional**: 20 items, 120 tracked facts (0 with a verified memorised alternative answer).

### Source provenance

This dataset draws on more than one source, and their documents differ in
length. Length is the dependent variable here, so the split is reported
rather than pooled -- see `metrics_by_source.csv` before reading any
counterfactual result as a directive effect.

| dataset | source | items | document chars (min-max) |
|---|---|---:|---|
| counterfactual | known_entity | 20 | 948-1249 |
| counterfactual | rewritten_wikipedia | 6 | 5442-9454 |
| fictional | unspecified | 20 | 2021-3155 |

## Handoff length at depth 3

| dataset | arm | words | answer accuracy | side facts kept | repetition |
|---|---|---:|---:|---:|---:|
| counterfactual | asked to expand | 3112 | 0.423 | 0.785 | 0.687 |
| counterfactual | asked: large -> small -> large | 1663 | 0.577 | 0.785 | 0.404 |
| counterfactual | asked: shrink, shrink, expand | 1379 | 0.654 | 0.685 | 0.283 |
| counterfactual | informed: large -> small -> large | 403 | 0.731 | 0.700 | 0.050 |
| counterfactual | informed large | 297 | 0.654 | 0.692 | 0.028 |
| counterfactual | informed small | 282 | 0.769 | 0.715 | 0.015 |
| counterfactual | asked to shrink | 237 | 0.692 | 0.669 | 0.008 |
| counterfactual | control (no size instruction) | 171 | 0.654 | 0.523 | 0.004 |
| counterfactual | concise (Experiment 5's implicit cue) | 51 | 0.846 | 0.277 | 0.000 |
| fictional | asked to expand | 2393 | 0.800 | 0.910 | 0.463 |
| fictional | asked: shrink, shrink, expand | 1626 | 0.750 | 0.600 | 0.307 |
| fictional | asked: large -> small -> large | 1472 | 0.800 | 0.890 | 0.241 |
| fictional | informed large | 440 | 0.850 | 0.610 | 0.027 |
| fictional | informed small | 420 | 0.900 | 0.590 | 0.025 |
| fictional | informed: large -> small -> large | 419 | 0.950 | 0.610 | 0.026 |
| fictional | asked to shrink | 346 | 0.750 | 0.600 | 0.009 |
| fictional | control (no size instruction) | 238 | 0.950 | 0.380 | 0.007 |
| fictional | concise (Experiment 5's implicit cue) | 66 | 0.900 | 0.170 | 0.000 |

## Contrasts (paired, per item)

Only the length and accuracy measures are shown here; `contrasts.csv` holds every
measure at every depth. A contrast whose interval excludes zero is marked.

| contrast | dataset | measure | delta | 95% CI | excludes 0 |
|---|---|---|---:|---|:-:|
| inform_shrink_vs_control | counterfactual | summary_words | +110.654 | [+79.038, +146.962] | yes |
| inform_shrink_vs_control | counterfactual | internal_repetition | +0.011 | [+0.005, +0.017] | yes |
| inform_shrink_vs_control | counterfactual | side_fact_retention | +0.192 | [+0.092, +0.292] | yes |
| inform_shrink_vs_control | counterfactual | unsupported_count | -0.115 | [-1.385, +1.346] |  |
| inform_shrink_vs_control | counterfactual | judge_correct | +0.115 | [-0.038, +0.269] |  |
| inform_shrink_vs_control | fictional | summary_words | +182.700 | [+143.099, +217.750] | yes |
| inform_shrink_vs_control | fictional | internal_repetition | +0.018 | [+0.009, +0.029] | yes |
| inform_shrink_vs_control | fictional | side_fact_retention | +0.210 | [+0.080, +0.340] | yes |
| inform_shrink_vs_control | fictional | unsupported_count | -0.200 | [-1.050, +0.800] |  |
| inform_shrink_vs_control | fictional | judge_correct | -0.050 | [-0.200, +0.100] |  |
| inform_expand_vs_control | counterfactual | summary_words | +125.192 | [+93.614, +166.269] | yes |
| inform_expand_vs_control | counterfactual | internal_repetition | +0.024 | [+0.008, +0.049] | yes |
| inform_expand_vs_control | counterfactual | side_fact_retention | +0.169 | [+0.077, +0.269] | yes |
| inform_expand_vs_control | counterfactual | unsupported_count | -0.385 | [-1.115, +0.308] |  |
| inform_expand_vs_control | counterfactual | judge_correct | +0.000 | [-0.192, +0.192] |  |
| inform_expand_vs_control | fictional | summary_words | +202.100 | [+158.149, +249.901] | yes |
| inform_expand_vs_control | fictional | internal_repetition | +0.020 | [+0.010, +0.033] | yes |
| inform_expand_vs_control | fictional | side_fact_retention | +0.230 | [+0.110, +0.340] | yes |
| inform_expand_vs_control | fictional | unsupported_count | +0.500 | [-0.700, +1.850] |  |
| inform_expand_vs_control | fictional | judge_correct | -0.100 | [-0.300, +0.100] |  |
| resize_vs_inform_shrink | counterfactual | summary_words | -45.115 | [-78.887, -15.807] | yes |
| resize_vs_inform_shrink | counterfactual | internal_repetition | -0.007 | [-0.014, +0.000] |  |
| resize_vs_inform_shrink | counterfactual | side_fact_retention | -0.046 | [-0.146, +0.046] |  |
| resize_vs_inform_shrink | counterfactual | unsupported_count | -0.077 | [-0.923, +0.770] |  |
| resize_vs_inform_shrink | counterfactual | judge_correct | -0.077 | [-0.231, +0.077] |  |
| resize_vs_inform_shrink | fictional | summary_words | -74.250 | [-118.001, -29.400] | yes |
| resize_vs_inform_shrink | fictional | internal_repetition | -0.016 | [-0.028, -0.004] | yes |
| resize_vs_inform_shrink | fictional | side_fact_retention | +0.010 | [-0.100, +0.120] |  |
| resize_vs_inform_shrink | fictional | unsupported_count | +0.600 | [-0.800, +2.100] |  |
| resize_vs_inform_shrink | fictional | judge_correct | -0.150 | [-0.300, +0.000] |  |
| resize_vs_inform_expand | counterfactual | summary_words | +2815.769 | [+2458.058, +3096.116] | yes |
| resize_vs_inform_expand | counterfactual | internal_repetition | +0.659 | [+0.576, +0.724] | yes |
| resize_vs_inform_expand | counterfactual | side_fact_retention | +0.092 | [-0.054, +0.223] |  |
| resize_vs_inform_expand | counterfactual | unsupported_count | +2.962 | [+1.077, +5.038] | yes |
| resize_vs_inform_expand | counterfactual | judge_correct | -0.231 | [-0.500, +0.038] |  |
| resize_vs_inform_expand | fictional | summary_words | +1953.700 | [+1541.881, +2338.062] | yes |
| resize_vs_inform_expand | fictional | internal_repetition | +0.436 | [+0.314, +0.550] | yes |
| resize_vs_inform_expand | fictional | side_fact_retention | +0.300 | [+0.110, +0.470] | yes |
| resize_vs_inform_expand | fictional | unsupported_count | +4.600 | [+0.849, +8.900] | yes |
| resize_vs_inform_expand | fictional | judge_correct | -0.050 | [-0.250, +0.150] |  |
| resize_shrink_vs_control | counterfactual | summary_words | +65.538 | [+49.615, +80.923] | yes |
| resize_shrink_vs_control | counterfactual | internal_repetition | +0.004 | [-0.001, +0.009] |  |
| resize_shrink_vs_control | counterfactual | side_fact_retention | +0.146 | [+0.046, +0.254] | yes |
| resize_shrink_vs_control | counterfactual | unsupported_count | -0.192 | [-1.308, +1.038] |  |
| resize_shrink_vs_control | counterfactual | judge_correct | +0.038 | [-0.077, +0.154] |  |
| resize_shrink_vs_control | fictional | summary_words | +108.450 | [+64.550, +156.050] | yes |
| resize_shrink_vs_control | fictional | internal_repetition | +0.003 | [-0.002, +0.007] |  |
| resize_shrink_vs_control | fictional | side_fact_retention | +0.220 | [+0.110, +0.330] | yes |
| resize_shrink_vs_control | fictional | unsupported_count | +0.400 | [-0.600, +1.550] |  |
| resize_shrink_vs_control | fictional | judge_correct | -0.200 | [-0.400, +0.000] |  |
| resize_expand_vs_control | counterfactual | summary_words | +2940.962 | [+2591.537, +3215.310] | yes |
| resize_expand_vs_control | counterfactual | internal_repetition | +0.683 | [+0.602, +0.746] | yes |
| resize_expand_vs_control | counterfactual | side_fact_retention | +0.262 | [+0.108, +0.392] | yes |
| resize_expand_vs_control | counterfactual | unsupported_count | +2.577 | [+0.500, +4.846] | yes |
| resize_expand_vs_control | counterfactual | judge_correct | -0.231 | [-0.462, +0.000] |  |
| resize_expand_vs_control | fictional | summary_words | +2155.800 | [+1731.794, +2547.351] | yes |
| resize_expand_vs_control | fictional | internal_repetition | +0.456 | [+0.335, +0.570] | yes |
| resize_expand_vs_control | fictional | side_fact_retention | +0.530 | [+0.380, +0.660] | yes |
| resize_expand_vs_control | fictional | unsupported_count | +5.100 | [+1.450, +9.450] | yes |
| resize_expand_vs_control | fictional | judge_correct | -0.150 | [-0.300, +0.000] |  |
| resize_lsl_vs_inform_lsl | counterfactual | summary_words | +1259.615 | [+704.162, +1787.811] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | internal_repetition | +0.354 | [+0.205, +0.488] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | side_fact_retention | +0.085 | [-0.038, +0.200] |  |
| resize_lsl_vs_inform_lsl | counterfactual | unsupported_count | +3.654 | [+1.808, +5.846] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | judge_correct | -0.154 | [-0.346, +0.038] |  |
| resize_lsl_vs_inform_lsl | fictional | summary_words | +1053.150 | [+593.496, +1556.954] | yes |
| resize_lsl_vs_inform_lsl | fictional | internal_repetition | +0.215 | [+0.091, +0.354] | yes |
| resize_lsl_vs_inform_lsl | fictional | side_fact_retention | +0.280 | [+0.100, +0.450] | yes |
| resize_lsl_vs_inform_lsl | fictional | unsupported_count | +4.450 | [+1.650, +7.500] | yes |
| resize_lsl_vs_inform_lsl | fictional | judge_correct | -0.150 | [-0.300, +0.000] |  |
| resize_lsl_vs_resize_shrink | counterfactual | summary_words | +1425.808 | [+984.264, +1888.508] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | internal_repetition | +0.396 | [+0.281, +0.507] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | side_fact_retention | +0.115 | [-0.008, +0.231] |  |
| resize_lsl_vs_resize_shrink | counterfactual | unsupported_count | +3.077 | [+1.462, +5.038] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | judge_correct | -0.115 | [-0.269, +0.038] |  |
| resize_lsl_vs_resize_shrink | fictional | summary_words | +1125.700 | [+667.846, +1623.903] | yes |
| resize_lsl_vs_resize_shrink | fictional | internal_repetition | +0.232 | [+0.109, +0.368] | yes |
| resize_lsl_vs_resize_shrink | fictional | side_fact_retention | +0.290 | [+0.120, +0.450] | yes |
| resize_lsl_vs_resize_shrink | fictional | unsupported_count | +3.750 | [+0.950, +6.800] | yes |
| resize_lsl_vs_resize_shrink | fictional | judge_correct | +0.050 | [-0.200, +0.300] |  |
| resize_sse_vs_resize_shrink | counterfactual | summary_words | +1142.077 | [+700.067, +1635.388] | yes |
| resize_sse_vs_resize_shrink | counterfactual | internal_repetition | +0.275 | [+0.164, +0.397] | yes |
| resize_sse_vs_resize_shrink | counterfactual | side_fact_retention | +0.015 | [-0.015, +0.046] |  |
| resize_sse_vs_resize_shrink | counterfactual | unsupported_count | +3.038 | [+1.808, +4.538] | yes |
| resize_sse_vs_resize_shrink | counterfactual | judge_correct | -0.038 | [-0.115, +0.000] |  |
| resize_sse_vs_resize_shrink | fictional | summary_words | +1280.350 | [+750.499, +1824.312] | yes |
| resize_sse_vs_resize_shrink | fictional | internal_repetition | +0.297 | [+0.154, +0.445] | yes |
| resize_sse_vs_resize_shrink | fictional | side_fact_retention | +0.000 | [+0.000, +0.000] |  |
| resize_sse_vs_resize_shrink | fictional | unsupported_count | +2.900 | [+1.450, +4.550] | yes |
| resize_sse_vs_resize_shrink | fictional | judge_correct | +0.000 | [-0.150, +0.150] |  |
| concise_vs_control | counterfactual | summary_words | -119.885 | [-134.731, -105.385] | yes |
| concise_vs_control | counterfactual | internal_repetition | -0.004 | [-0.007, -0.002] | yes |
| concise_vs_control | counterfactual | side_fact_retention | -0.246 | [-0.408, -0.092] | yes |
| concise_vs_control | counterfactual | unsupported_count | -2.423 | [-3.346, -1.654] | yes |
| concise_vs_control | counterfactual | judge_correct | +0.192 | [-0.038, +0.423] |  |
| concise_vs_control | fictional | summary_words | -171.900 | [-200.000, -145.800] | yes |
| concise_vs_control | fictional | internal_repetition | -0.007 | [-0.010, -0.003] | yes |
| concise_vs_control | fictional | side_fact_retention | -0.210 | [-0.330, -0.100] | yes |
| concise_vs_control | fictional | unsupported_count | -1.550 | [-2.100, -1.000] | yes |
| concise_vs_control | fictional | judge_correct | -0.050 | [-0.150, +0.000] |  |
| resize_shrink_vs_concise | counterfactual | summary_words | +185.423 | [+171.462, +199.539] | yes |
| resize_shrink_vs_concise | counterfactual | internal_repetition | +0.008 | [+0.004, +0.012] | yes |
| resize_shrink_vs_concise | counterfactual | side_fact_retention | +0.392 | [+0.262, +0.515] | yes |
| resize_shrink_vs_concise | counterfactual | unsupported_count | +2.231 | [+1.346, +3.385] | yes |
| resize_shrink_vs_concise | counterfactual | judge_correct | -0.154 | [-0.346, +0.000] |  |
| resize_shrink_vs_concise | fictional | summary_words | +280.350 | [+244.099, +317.500] | yes |
| resize_shrink_vs_concise | fictional | internal_repetition | +0.009 | [+0.005, +0.014] | yes |
| resize_shrink_vs_concise | fictional | side_fact_retention | +0.430 | [+0.270, +0.580] | yes |
| resize_shrink_vs_concise | fictional | unsupported_count | +1.950 | [+1.150, +2.850] | yes |
| resize_shrink_vs_concise | fictional | judge_correct | -0.150 | [-0.400, +0.100] |  |

## Facts that went missing, and what came back

`recovery` is measured only over facts that were actually lost. Because stage 2+
sees a sealed handoff, a fact that reappears was regenerated, not copied.

| dataset | arm | role | facts | lost | recovered | reverted | fabricated |
|---|---|---|---:|---:|---:|---:|---:|
| counterfactual | concise (Experiment 5's implicit cue) | side | 130 | 0.046 | 0.167 | 0.000 | 0.167 |
| counterfactual | control (no size instruction) | side | 130 | 0.062 | 0.000 | 0.000 | 0.875 |
| counterfactual | informed large | side | 130 | 0.038 | 0.000 | 0.000 | 0.600 |
| counterfactual | informed: large -> small -> large | side | 130 | 0.031 | 0.000 | 0.000 | 0.750 |
| counterfactual | informed small | side | 130 | 0.015 | 0.000 | 0.000 | 1.000 |
| counterfactual | asked to expand | side | 130 | 0.092 | 0.000 | 0.000 | 0.417 |
| counterfactual | asked: large -> small -> large | side | 130 | 0.092 | 0.000 | 0.000 | 0.417 |
| counterfactual | asked to shrink | side | 130 | 0.069 | 0.000 | 0.000 | 0.889 |
| counterfactual | asked: shrink, shrink, expand | side | 130 | 0.054 | 0.000 | 0.000 | 1.000 |
| counterfactual | concise (Experiment 5's implicit cue) | target | 26 | 0.038 | 0.000 | 0.000 | 0.000 |
| counterfactual | control (no size instruction) | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | informed large | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | informed: large -> small -> large | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | informed small | target | 26 | 0.038 | 0.000 | 0.000 | 1.000 |
| counterfactual | asked to expand | target | 26 | 0.077 | 0.000 | 0.000 | 0.000 |
| counterfactual | asked: large -> small -> large | target | 26 | 0.077 | 0.500 | 0.000 | 0.000 |
| counterfactual | asked to shrink | target | 26 | 0.038 | 0.000 | 0.000 | 1.000 |
| counterfactual | asked: shrink, shrink, expand | target | 26 | 0.077 | 0.000 | 0.000 | 0.500 |
| fictional | concise (Experiment 5's implicit cue) | side | 100 | 0.010 | 0.000 | 0.000 | 0.000 |
| fictional | control (no size instruction) | side | 100 | 0.000 | nan | nan | nan |
| fictional | informed large | side | 100 | 0.000 | nan | nan | nan |
| fictional | informed: large -> small -> large | side | 100 | 0.000 | nan | nan | nan |
| fictional | informed small | side | 100 | 0.010 | 0.000 | 0.000 | 1.000 |
| fictional | asked to expand | side | 100 | 0.000 | nan | nan | nan |
| fictional | asked: large -> small -> large | side | 100 | 0.020 | 0.000 | 0.000 | 0.500 |
| fictional | asked to shrink | side | 100 | 0.010 | 0.000 | 0.000 | 1.000 |
| fictional | asked: shrink, shrink, expand | side | 100 | 0.010 | 0.000 | 0.000 | 1.000 |
| fictional | concise (Experiment 5's implicit cue) | target | 20 | 0.000 | nan | nan | nan |
| fictional | control (no size instruction) | target | 20 | 0.000 | nan | nan | nan |
| fictional | informed large | target | 20 | 0.050 | 0.000 | 0.000 | 1.000 |
| fictional | informed: large -> small -> large | target | 20 | 0.050 | 0.000 | 0.000 | 1.000 |
| fictional | informed small | target | 20 | 0.000 | nan | nan | nan |
| fictional | asked to expand | target | 20 | 0.000 | nan | nan | nan |
| fictional | asked: large -> small -> large | target | 20 | 0.000 | nan | nan | nan |
| fictional | asked to shrink | target | 20 | 0.000 | nan | nan | nan |
| fictional | asked: shrink, shrink, expand | target | 20 | 0.050 | 0.000 | 0.000 | 1.000 |

## Where the counterfactual answer came from (depth 3)

| arm | document | memorised | both | other |
|---|---:|---:|---:|---:|
| concise (Experiment 5's implicit cue) | 0.846 | 0.115 | 0.000 | 0.038 |
| control (no size instruction) | 0.654 | 0.269 | 0.000 | 0.077 |
| informed large | 0.654 | 0.192 | 0.000 | 0.154 |
| informed: large -> small -> large | 0.731 | 0.192 | 0.000 | 0.077 |
| informed small | 0.769 | 0.115 | 0.000 | 0.115 |
| asked to expand | 0.423 | 0.385 | 0.000 | 0.192 |
| asked: large -> small -> large | 0.577 | 0.231 | 0.000 | 0.192 |
| asked to shrink | 0.692 | 0.269 | 0.000 | 0.038 |
| asked: shrink, shrink, expand | 0.654 | 0.269 | 0.000 | 0.077 |

## Cost

```
{
  "calls_total": 3266,
  "calls_live": 2512,
  "calls_cached": 754,
  "cache_hit_rate": 0.23086344151867727,
  "prompt_tokens": 2970304,
  "completion_tokens": 930854,
  "prompt_tokens_live": 2295906,
  "completion_tokens_live": 409378,
  "cost_usd": 0.694393,
  "api_calls_attempted": 2512,
  "api_call_cap": 30000
}
```
