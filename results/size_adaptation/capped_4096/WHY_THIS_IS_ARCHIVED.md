# Archived: the run where the token budget was binding

`decoding.handoff_max_tokens` was 4096 here. The `resize_large` directive tells
the model the next agent has a **10,000-token** window and asks it to use the
space -- so a 4096-token cap contradicts the prompt the experiment is testing.

It bound, hard:

| directive | truncated (`finish_reason == "length"`) |
|---|---|
| `resize_large` | **117/230 = 50.9%** |
| `resize_small` | 4/184 = 2.2% |
| `inform_large` | 1/184 = 0.5% |
| `inform_small` | 1/184 = 0.5% |
| `none` | 0/138 |
| `concise` | 0/138 |

Two consequences, both fatal to the expansion arms:

1. **Length is censored.** "asked to expand writes 2,090 words" is a lower
   bound set by the API, not a measurement of the model.
2. **Downstream stages read a handoff cut off mid-sentence.** The accuracy drop
   for `resize_expand` (document-answer rate 0.423 vs 0.654 for control) cannot
   be separated from "the notes ended abruptly and lost the answer".

The provider's real ceiling is 115,200 completion tokens, so 4096 was a
self-imposed limit, not a platform one. The replacement run sets the budget
above the largest advertised window so the model can actually comply with what
it is told, identically across every arm.

**Kept rather than deleted, because the saturation is itself a result:** asked
to expand, the model filled essentially whatever budget it was given, and the
share of it that was internal repetition rose with it. Read this directory only
for that; take every expansion number from the replacement run.
