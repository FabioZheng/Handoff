# Handoff probe results

- model: `meta-llama/llama-3.3-70b-instruct`
- dataset: musique (validation), n = 10 questions surviving the C1 leakage filter
- seeds: [1, 2] (LLM-generated handoffs only; A_full and E_oracle are deterministic at temperature 0)
- conditions run: A_full, B_freeform, E_oracle

## Accuracy by condition

| condition | n | EM | EM 95% CI | F1 | F1 95% CI | EM/F1 disagree | tokens/q |
|---|---:|---:|---|---:|---|---:|---:|
| `A_full` | 10 | 0.400 | [0.100, 0.700] | 0.484 | [0.217, 0.753] | 0.000 | 2594 |
| `B_freeform` | 10 | 0.600 | [0.300, 0.900] | 0.748 | [0.520, 0.940] | 0.100 | 3039 |
| `E_oracle` | 10 | 0.600 | [0.300, 0.900] | 0.740 | [0.507, 0.940] | 0.100 | 240 |

## Paired contrasts

| contrast | metric | delta | 95% CI | p | n | what it means |
|---|---|---:|---|---:|---:|---|
| `B_freeform - A_full` | EM | +0.200 | [+0.000, +0.500] | 0.2327 | 10 | headroom: free-form handoff vs full context |
| `B_freeform - A_full` | F1 | +0.264 | [+0.010, +0.525] | 0.0304 | 10 | headroom: free-form handoff vs full context |
| `E_oracle - B_freeform` | EM | +0.000 | [-0.300, +0.300] | 1.0000 | 10 | recovery: perfect handoff vs free-form |
| `E_oracle - B_freeform` | F1 | -0.008 | [-0.225, +0.267] | 0.9013 | 10 | recovery: perfect handoff vs free-form |
| `E_oracle - A_full` | EM | +0.200 | [-0.200, +0.600] | 0.4339 | 10 | denoising: perfect handoff vs full context |
| `E_oracle - A_full` | F1 | +0.256 | [-0.033, +0.568] | 0.0927 | 10 | denoising: perfect handoff vs full context |

## Cost

- spend this run: $0.0151 (cap $15.00)
- calls: 180 live, 0 cached (0% hit rate)
- tokens: 93859 in / 4149 out
