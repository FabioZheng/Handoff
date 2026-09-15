# Experiment 8: model heterogeneity across sequential handoffs

Depths: 0, 1, 2, 3, 4, 5, 6  
Answerers (held constant across every arm): primary = `meta-llama/llama-3.1-8b-instruct`; secondary = `mistralai/ministral-8b`  
Judge: `openai/gpt-4o-mini` at temperature 0  
Handoff mode: conditioned (question-conditioned), temperature 0.0, shared budget 1500 tokens

## Model pool

| key | model id | family | tier | parameters | basis | released | $/Mtok in | $/Mtok out |
|---|---|---|---|---:|---|---|---:|---:|
| `gemma_large` | `google/gemma-3-27b-it` | gemma | large | 27.4B | reported total 27.4B, dense | 2025-03-12 | 0.08 | 0.45 |
| `llama_large` | `meta-llama/llama-3.3-70b-instruct` | llama | large | 70.6B | reported total 70.6B (vendor headline 70B), dense | 2024-12-06 | 0.71 | 0.71 |
| `mistral_large` | `mistralai/mistral-small-3.2-24b-instruct` | mistral | large | 24B | vendor headline count (24B), dense | 2025-06-20 | 0.075 | 0.2 |
| `qwen_large` | `qwen/qwen3-32b` | qwen | large | 32.8B | reported total 32.8B (31.2B non-embedding), dense | 2025-04-29 | 0.08 | 0.28 |
| `gemma_small` | `google/gemma-3-12b-it` | gemma | small | 12.2B | reported total 12.2B, dense | 2025-03-12 | 0.05 | 0.15 |
| `llama_small` | `meta-llama/llama-3.1-8b-instruct` | llama | small | 8B | vendor headline count (8B), dense | 2024-07-23 | 0.05 | 0.08 |
| `mistral_small` | `mistralai/ministral-8b` | mistral | small | 8B | vendor headline count (8B), dense | 2024-10-16 | 0.11 | 0.11 |
| `qwen_small` | `qwen/qwen3-8b` | qwen | small | 8.2B | reported total 8.2B (6.95B non-embedding), dense | 2025-04-29 | 0.117 | 0.455 |

## Accuracy at depth 6 (primary answerer)

Chains are shown for the pairs that start on llama; an arm whose family rotates runs the same pattern from each of the other start families too.

| arm | block | chain | family switches | target F1 | target judge | held-out F1 | held-out judge |
|---|---|---|---:|---:|---:|---:|---:|
| `homog_large_gemma` | matched_large | `G27-G27-G27-G27-G27-G27` | 0 | 0.779 | 0.950 | 0.421 | 0.550 |
| `homog_large_llama` | matched_large | `L70-L70-L70-L70-L70-L70` | 0 | 0.824 | 0.950 | 0.321 | 0.350 |
| `homog_large_matched` | matched_large | `L70-L70-L70-L70-L70-L70` | 0 | 0.830 | 1.000 | 0.317 | 0.400 |
| `homog_large_mistral` | matched_large | `M24-M24-M24-M24-M24-M24` | 0 | 0.742 | 1.000 | 0.351 | 0.400 |
| `homog_large_qwen` | matched_large | `Q32-Q32-Q32-Q32-Q32-Q32` | 0 | 0.825 | 1.000 | 0.391 | 0.450 |
| `homog_small_gemma` | matched_small | `G12-G12-G12-G12-G12-G12` | 0 | 0.394 | 0.500 | 0.142 | 0.150 |
| `homog_small_llama` | matched_small | `L8-L8-L8-L8-L8-L8` | 0 | 0.638 | 0.700 | 0.250 | 0.300 |
| `homog_small_matched` | matched_small | `L8-L8-L8-L8-L8-L8` | 0 | 0.649 | 0.850 | 0.208 | 0.300 |
| `homog_small_mistral` | matched_small | `M8-M8-M8-M8-M8-M8` | 0 | 0.657 | 0.800 | 0.186 | 0.200 |
| `homog_small_qwen` | matched_small | `Q8-Q8-Q8-Q8-Q8-Q8` | 0 | 0.879 | 1.000 | 0.471 | 0.550 |
| `size_alt_samefam` | size_trajectory | `L8-L70-L8-L70-L8-L70` | 0 | 0.643 | 0.850 | 0.261 | 0.350 |
| `size_alt_xfam` | size_trajectory | `L8-Q32-M8-G27-L8-Q32` | 5 | 0.678 | 0.800 | 0.317 | 0.350 |
| `size_down_samefam` | size_trajectory | `L70-L70-L70-L8-L8-L8` | 0 | 0.751 | 0.950 | 0.264 | 0.350 |
| `size_down_xfam` | size_trajectory | `L70-Q32-M24-G12-L8-Q8` | 5 | 0.721 | 0.900 | 0.487 | 0.500 |
| `size_up_samefam` | size_trajectory | `L8-L8-L8-L70-L70-L70` | 0 | 0.636 | 0.850 | 0.258 | 0.350 |
| `size_up_xfam` | size_trajectory | `L8-Q8-M8-G27-L70-Q32` | 5 | 0.601 | 0.800 | 0.279 | 0.400 |
| `xfam_large_fwd` | matched_large | `L70-Q32-M24-G27-L70-Q32` | 5 | 0.814 | 1.000 | 0.387 | 0.450 |
| `xfam_small_fwd` | matched_small | `L8-Q8-M8-G12-L8-Q8` | 5 | 0.572 | 0.750 | 0.187 | 0.150 |
| `xfam_small_rev` | matched_small | `L8-G12-M8-Q8-L8-G12` | 5 | 0.733 | 0.900 | 0.154 | 0.150 |

## Contrasts at depth 6 (primary answerer, judged accuracy)

| comparison | query | delta | 95% CI | p |
|---|---|---:|---|---:|
| xfam_minus_homog_small | target | -0.100 | [-0.300, +0.100] | 0.434 |
| xfam_minus_homog_small | heldout | -0.150 | [-0.300, +0.000] | 0.107 |
| xfam_rev_minus_homog_small | target | +0.050 | [+0.000, +0.150] | 0.626 |
| xfam_rev_minus_homog_small | heldout | -0.150 | [-0.300, +0.000] | 0.107 |
| xfam_minus_homog_large | target | +0.000 | [+0.000, +0.000] | 1.000 |
| xfam_minus_homog_large | heldout | +0.050 | [-0.100, +0.200] | 0.761 |
| xfam_fwd_minus_rev_small | target | -0.150 | [-0.300, +0.000] | 0.102 |
| xfam_fwd_minus_rev_small | heldout | +0.000 | [+0.000, +0.000] | 1.000 |
| large_minus_small_homog | target | +0.150 | [+0.000, +0.300] | 0.109 |
| large_minus_small_homog | heldout | +0.100 | [-0.200, +0.350] | 0.592 |
| size_up_minus_all_small | target | +0.000 | [+0.000, +0.000] | 1.000 |
| size_up_minus_all_small | heldout | +0.050 | [+0.000, +0.150] | 0.622 |
| size_up_minus_all_large | target | -0.150 | [-0.300, +0.000] | 0.109 |
| size_up_minus_all_large | heldout | -0.050 | [-0.350, +0.250] | 0.868 |
| size_down_minus_all_small | target | +0.100 | [-0.100, +0.300] | 0.428 |
| size_down_minus_all_small | heldout | +0.050 | [-0.200, +0.300] | 0.845 |
| size_down_minus_all_large | target | -0.050 | [-0.150, +0.000] | 0.630 |
| size_down_minus_all_large | heldout | -0.050 | [-0.200, +0.100] | 0.760 |
| size_alt_minus_all_small | target | +0.000 | [-0.150, +0.150] | 1.000 |
| size_alt_minus_all_small | heldout | +0.050 | [+0.000, +0.150] | 0.622 |
| size_alt_minus_all_large | target | -0.150 | [-0.300, +0.000] | 0.105 |
| size_alt_minus_all_large | heldout | -0.050 | [-0.350, +0.250] | 0.868 |
| size_up_minus_down_samefam | target | -0.100 | [-0.300, +0.100] | 0.428 |
| size_up_minus_down_samefam | heldout | +0.000 | [-0.251, +0.300] | 1.000 |
| size_up_minus_down_xfam | target | -0.100 | [-0.300, +0.100] | 0.432 |
| size_up_minus_down_xfam | heldout | -0.100 | [-0.350, +0.150] | 0.580 |
| size_alt_minus_up_samefam | target | +0.000 | [-0.150, +0.150] | 1.000 |
| size_alt_minus_up_samefam | heldout | +0.000 | [+0.000, +0.000] | 1.000 |
| xfam_cost_on_size_up | target | -0.050 | [-0.150, +0.000] | 0.629 |
| xfam_cost_on_size_up | heldout | +0.050 | [-0.100, +0.200] | 0.762 |
| xfam_cost_on_size_down | target | -0.050 | [-0.200, +0.100] | 0.762 |
| xfam_cost_on_size_down | heldout | +0.150 | [-0.100, +0.400] | 0.323 |
| xfam_cost_on_size_alt | target | -0.050 | [-0.200, +0.100] | 0.758 |
| xfam_cost_on_size_alt | heldout | +0.000 | [-0.200, +0.200] | 1.000 |
| recovery_by_larger_downstream | target | +0.000 | [+0.000, +0.000] | 1.000 |
| recovery_by_larger_downstream | heldout | +0.050 | [+0.000, +0.150] | 0.622 |
| bottleneck_from_smaller_downstream | target | -0.050 | [-0.150, +0.000] | 0.630 |
| bottleneck_from_smaller_downstream | heldout | -0.050 | [-0.200, +0.100] | 0.760 |

## Per-edge information loss by transition type

| transition | edges | pairs | semantic preserved | held-out fact kept | target fact kept | lexical similarity | length ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross_family_size_larger | 192 | 20 | 0.984 | 0.164 | 0.740 | 0.701 | 1.212 |
| cross_family_size_same | 185 | 20 | 0.940 | 0.157 | 0.705 | 0.710 | 1.278 |
| cross_family_size_smaller | 211 | 20 | 0.975 | 0.169 | 0.816 | 0.779 | 0.991 |
| same_family_size_larger | 77 | 20 | 0.988 | 0.150 | 0.792 | 0.825 | 1.020 |
| same_family_size_same | 944 | 20 | 0.989 | 0.254 | 0.821 | 0.873 | 1.014 |
| same_family_size_smaller | 60 | 20 | 0.942 | 0.167 | 0.833 | 0.845 | 1.069 |

## Paired per-edge contrasts (same pairs, differing only in the edge type)

| comparison | measure | treatment | control | delta | 95% CI | p | pairs |
|---|---|---:|---:|---:|---|---:|---:|
| cross_vs_same_family_at_matched_size | lexical_similarity | 0.710 | 0.873 | -0.163 | [-0.200, -0.131] | 0.000 | 20 |
| cross_vs_same_family_at_matched_size | length_ratio | 1.278 | 1.014 | +0.263 | [+0.140, +0.394] | 0.000 | 20 |
| cross_vs_same_family_at_matched_size | semantic_preserved | 0.940 | 0.989 | -0.049 | [-0.088, -0.015] | 0.008 | 20 |
| cross_vs_same_family_at_matched_size | fact_A_survived | 0.705 | 0.821 | -0.116 | [-0.238, -0.006] | 0.048 | 20 |
| cross_vs_same_family_at_matched_size | fact_B_survived | 0.157 | 0.254 | -0.097 | [-0.173, -0.020] | 0.012 | 20 |
| cross_vs_same_family_when_growing | lexical_similarity | 0.701 | 0.825 | -0.124 | [-0.167, -0.079] | 0.000 | 20 |
| cross_vs_same_family_when_growing | length_ratio | 1.212 | 1.020 | +0.192 | [+0.072, +0.318] | 0.003 | 20 |
| cross_vs_same_family_when_growing | semantic_preserved | 0.984 | 0.988 | -0.004 | [-0.030, +0.030] | 0.792 | 20 |
| cross_vs_same_family_when_growing | fact_A_survived | 0.740 | 0.792 | -0.052 | [-0.189, +0.059] | 0.413 | 20 |
| cross_vs_same_family_when_growing | fact_B_survived | 0.164 | 0.150 | +0.014 | [-0.030, +0.066] | 0.582 | 20 |
| cross_vs_same_family_when_shrinking | lexical_similarity | 0.779 | 0.845 | -0.066 | [-0.112, -0.018] | 0.006 | 20 |
| cross_vs_same_family_when_shrinking | length_ratio | 0.991 | 1.069 | -0.078 | [-0.227, +0.037] | 0.243 | 20 |
| cross_vs_same_family_when_shrinking | semantic_preserved | 0.975 | 0.942 | +0.033 | [-0.015, +0.083] | 0.195 | 20 |
| cross_vs_same_family_when_shrinking | fact_A_survived | 0.816 | 0.833 | -0.017 | [-0.156, +0.102] | 0.787 | 20 |
| cross_vs_same_family_when_shrinking | fact_B_survived | 0.169 | 0.167 | +0.002 | [-0.064, +0.079] | 0.951 | 20 |
| growing_vs_flat_within_family | lexical_similarity | 0.825 | 0.873 | -0.048 | [-0.088, -0.009] | 0.017 | 20 |
| growing_vs_flat_within_family | length_ratio | 1.020 | 1.014 | +0.006 | [-0.045, +0.069] | 0.845 | 20 |
| growing_vs_flat_within_family | semantic_preserved | 0.988 | 0.989 | -0.001 | [-0.030, +0.016] | 0.888 | 20 |
| growing_vs_flat_within_family | fact_A_survived | 0.792 | 0.821 | -0.029 | [-0.179, +0.103] | 0.681 | 20 |
| growing_vs_flat_within_family | fact_B_survived | 0.150 | 0.254 | -0.104 | [-0.198, +0.003] | 0.041 | 20 |
| shrinking_vs_flat_within_family | lexical_similarity | 0.845 | 0.873 | -0.028 | [-0.063, +0.006] | 0.106 | 20 |
| shrinking_vs_flat_within_family | length_ratio | 1.069 | 1.014 | +0.055 | [-0.044, +0.179] | 0.337 | 20 |
| shrinking_vs_flat_within_family | semantic_preserved | 0.942 | 0.989 | -0.047 | [-0.094, -0.008] | 0.030 | 20 |
| shrinking_vs_flat_within_family | fact_A_survived | 0.833 | 0.821 | +0.012 | [-0.124, +0.136] | 0.855 | 20 |
| shrinking_vs_flat_within_family | fact_B_survived | 0.167 | 0.254 | -0.088 | [-0.173, -0.006] | 0.040 | 20 |
| growing_vs_flat_cross_family | lexical_similarity | 0.701 | 0.710 | -0.009 | [-0.049, +0.036] | 0.676 | 20 |
| growing_vs_flat_cross_family | length_ratio | 1.212 | 1.278 | -0.066 | [-0.227, +0.110] | 0.438 | 20 |
| growing_vs_flat_cross_family | semantic_preserved | 0.984 | 0.940 | +0.044 | [+0.008, +0.083] | 0.022 | 20 |
| growing_vs_flat_cross_family | fact_A_survived | 0.740 | 0.705 | +0.034 | [-0.017, +0.088] | 0.197 | 20 |
| growing_vs_flat_cross_family | fact_B_survived | 0.164 | 0.157 | +0.007 | [-0.030, +0.044] | 0.700 | 20 |
| shrinking_vs_flat_cross_family | lexical_similarity | 0.779 | 0.710 | +0.069 | [+0.031, +0.112] | 0.002 | 20 |
| shrinking_vs_flat_cross_family | length_ratio | 0.991 | 1.278 | -0.286 | [-0.423, -0.156] | 0.000 | 20 |
| shrinking_vs_flat_cross_family | semantic_preserved | 0.975 | 0.940 | +0.035 | [-0.002, +0.075] | 0.080 | 20 |
| shrinking_vs_flat_cross_family | fact_A_survived | 0.816 | 0.705 | +0.111 | [+0.040, +0.189] | 0.004 | 20 |
| shrinking_vs_flat_cross_family | fact_B_survived | 0.169 | 0.157 | +0.012 | [-0.034, +0.066] | 0.671 | 20 |

## Judged accuracy by number of family switches, depth 6

| query | switches | arms pooled | judge | 95% CI |
|---|---:|---:|---:|---|
| target | 0 | 11 | 0.868 | [0.809, 0.918] |
| target | 5 | 6 | 0.858 | [0.750, 0.950] |
| heldout | 0 | 11 | 0.364 | [0.245, 0.500] |
| heldout | 5 | 6 | 0.333 | [0.192, 0.492] |

Switch count and depth are collinear inside a single arm, so this table is only ever read within one depth; the size-trajectory arms are what supply the intermediate switch counts.

## Cost

- chain models: $0.0000 over 0 live and 0 cached calls

Model metadata was read from OpenRouter's live `/api/v1/models` at run time and stored beside the raw records in `model_catalogue.json`. Parameter counts are vendor-published and are never inferred from price, tier, or benchmark score; an undisclosed count is recorded as unknown and excluded from size contrasts.
