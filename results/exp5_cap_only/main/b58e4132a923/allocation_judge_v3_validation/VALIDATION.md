# Allocation judge v3: why it replaced phi-4, and how it was chosen

The v2 judge (phi-4 reading the whole paper) failed a hand check: 45 of 50 of
its "the source does not state it" labels were wrong
(`../allocation_judge/SPOT_CHECK.md`). v3 changes three things:

1. **Headings and word-count stamps are labelled "no content" without a judge.**
   A sentence that ends in a colon and has at most 10 words, or that only
   states a word count, carries no claim.
2. **The support question shows 8 source passages, not the whole paper.** The
   passages are the source sentences sharing the most IDF-weighted content
   words with the sentence, in the paper's order. The note's preceding
   sentence is shown too, so that pronouns resolve. Every number, name and
   qualifier must match.
3. **A cascade of three models answers, over OpenRouter at temperature 0.**
   `google/gemini-2.5-flash-lite` answers every question. Where it says NO or
   NONE (support), or NO (earlier note), `deepseek/deepseek-v3.2` answers too.
   If the two disagree, `openai/gpt-4.1-mini` decides by majority. On the
   validation set this gives the same scores as asking all three every time,
   at about a fifth of the cost. The full three-model jury was started first
   and stopped at 30 documents for cost. Its answers are cached, so the cascade
   reuses them.

Results are in `../allocation_judge_v3c/`.

## Validation set (146 sentences; per-case answers in `validation_cases.csv`, kept locally)

| kind | n | expected | what it is |
|---|---:|---|---|
| hand_S | 38 | YES | hand-labelled supported (from the v2 spot check) |
| silver_S | 30 | YES | random v2 YES labels (v2 YES was 10/10 correct by hand) |
| hand_U | 1 | NO | hand-labelled unsupported |
| swap_U | 30 | NO | a supported sentence judged against a different paper |
| perturb_U | 30 | NO | a supported sentence with its first number changed; a few are list indices, so this undercounts accuracy |
| hand_N | 17 | NONE | hand-labelled headings and fragments |

## Results, v3 prompt with 8 passages

| judge | hand_S | silver_S | swap_U | perturb_U | hand_N | supported called NO |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini | 37/38 | 30/30 | 23/30 | 24/30 | 16/17 | 1 |
| deepseek-v3.2 | 35/38 | 27/30 | 29/30 | 23/30 | 16/17 | 6 |
| gemini-2.5-flash-lite | 35/38 | 28/30 | 27/30 | 25/30 | 15/17 | 5 |
| gpt-4.1 | 36/38 | 28/30 | 28/30 | 25/30 | 16/17 | 4 |
| claude-haiku-4.5 | 36/38 | 28/30 | 22/30 | 26/30 | 17/17 | 4 |
| gemini-2.5-flash | 28/38 | 27/30 | 15/30 | 26/30 | 16/17 | 11 |
| majority of gpt-4.1-mini, deepseek-v3.2, flash-lite | 37/38 | 29/30 | 28/30 | 25/30 | 16/17 | 2 |
| **cascade (used)** | **37/38** | **29/30** | **27/30** | **25/30** | **15/17** | **2** |
| gpt-4.1-nano | 31/38 | 29/30 | 25/30 | 22/30 | 15/17 | 8 |
| phi-4, 96 reply tokens | 36/38 | 27/30 (3 unparsed) | 20/30 | 17/30 | 15/17 | 2 |
| phi-4, 16 reply tokens | most replies unparseable | | | | | |

The error that matters most is calling a supported sentence unsupported. Real
unsupported content is rare (about 1 in 20 added sentences), so even a small
false-NO rate would swamp it. That is what made v2 unusable. The cascade
makes that error on 2 of 68 supported sentences and still rejects most
wrong-paper and altered-number sentences. The stronger single models (gpt-4.1,
claude-haiku-4.5) were not better on this set.

The earlier-note question ("does the shorter note already have it?") was
checked by hand on 20 v3 pilot labels: 19 correct and 1 unclear.

`src/analysis/validate_exp5_judge.py` reproduces the table from the pulled v2
items and run messages.

## Result of the full run (`../allocation_judge_v3c/report.md`)

The cascade judged 19,511 of 20,519 new sentences over all 200 documents; the
rest were labelled deterministically. The run cost $2.02 in API calls. Its
audit now separates the answers: sentences judged "source states it" overlap
their best source sentence by 0.58 on average, those judged not stated by 0.39
(v2: 0.59 against 0.51).

Query-aware minus generic, share of added words, pooled over the five cap
steps, 95% document-bootstrap intervals:

| category | query-aware − generic |
|---|---|
| not in source | +0.029 [+0.019, +0.038] |
| current-question evidence | +0.024 [+0.019, +0.030] |
| other-question evidence | +0.002 [−0.005, +0.008] |
| redundant (already in the shorter note) | −0.058 [−0.077, −0.039] |

About two thirds of the words added at each cap step restate what the shorter
message already said, in both arms. Of the rest, query-aware summaries spend
more on the current question and slightly more on content the source does not
state. They spend about the same as generic summaries on other questions'
evidence. v2 put the query-aware unsupported share at 0.16 against 0.04–0.08
for generic; with a reliable judge the gap is +0.03.
