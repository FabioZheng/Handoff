# Hand check of the allocation judge (job 28869901)

Checked 2026-09-23 on 60 sentences drawn at random (seed 7) from the judge's
output, stratified by label. Each sentence was compared against its source
paper by hand. Per-sentence verdicts are in `spot_check.csv`: S = the source
states it, U = the source does not state it, N = no factual content (a heading,
a fragment, a word-count stamp), A = ambiguous.

| stratum | judged | S | U | N | A |
|---|---:|---:|---:|---:|---:|
| conditioned, judge NO | 25 | 18 | 1 | 2 | 4 |
| generic, judge NO | 15 | 5 | 0 | 10 | 0 |
| overlap >= 0.85, judge NO | 10 | 5 | 0 | 5 | 0 |
| conditioned, judge YES | 10 | 10 | 0 | 0 | 0 |

**The "not in source" category is not usable.** Of 50 sentences the judge
labelled NO, at most 5 are unsupported. The rest are supported claims the judge
missed (28, several near-verbatim) or headings such as `- **Results**:` that
belong in "no content" (17). YES labels were correct in all 10 cases checked.

The judge sees the whole paper (up to about 8,000 words), so nothing was
truncated. The misses look like a small judge failing to find a claim in a
long input. Conditioned summaries write longer, more paraphrased sentences, so
they are missed more often, which produces the apparent conditioned-minus-generic
gap in "not in source" (0.16 against 0.04-0.08 of added words).

Do not cite the "not in source" or "plausible" shares from `report.md`. The
"restated" split (the earlier-note question) was not hand-checked.

**Superseded.** The judge was rebuilt and rerun; see
`../allocation_judge_v3_validation/VALIDATION.md` and `../allocation_judge_v3c/`.
