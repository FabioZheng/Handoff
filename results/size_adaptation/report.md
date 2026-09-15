# Experiment 9 — handoff size adaptation

Model under test: `meta-llama/llama-3.3-70b-instruct`. Judge: `openai/gpt-4o-mini`. Depths [0, 1, 2, 3], 9 arms.

Base handoff instruction is Experiment 5's `conditioned` prompt with the single word
"concise" removed, so the control carries no size instruction at all. Every directive
is appended to that neutral base; each `resize_*` string is its `inform_*` string plus
one clause, so inform-vs-resize isolates the request. The `concise` arm separately
appends "Keep your handoff concise"; it does not restore the deleted word in place.

## Items

- **counterfactual**: 26 items, 156 tracked facts (26 with a verified memorised alternative answer).
- **fictional**: 20 items, 120 tracked facts (0 with a verified memorised alternative answer).

### Source provenance

This dataset draws on more than one source, and their documents differ in
length. Length is the dependent variable here, so the split is reported
rather than pooled -- see `metrics_by_source.csv` before reading any
counterfactual result as a directive effect.

Checked, not merely flagged: at depth 3 the two counterfactual sources agree
despite a ~7x difference in document length (control 166 vs 160 words;
asked-to-expand 9,050 vs 7,368; judged accuracy 0.800 vs 0.833), so the
pooled counterfactual numbers are not an artefact of mixing them.

| dataset | source | items | document chars (min-max) |
|---|---|---:|---|
| counterfactual | known_entity | 20 | 948-1249 |
| counterfactual | rewritten_wikipedia | 6 | 5442-9454 |
| fictional | unspecified | 20 | 2021-3155 |

## Calibration: neither advertised window was a constraint

**Read this before the words `shrink` and `expand` anywhere below.** They name
the *intent* of each prompt, not its effect. With no size instruction at all the
model writes a median of **218 completion tokens**, and the source documents
are a median of ~677 tokens. Against that baseline:

| directive | advertised | median written | advertised vs baseline | window exceeded |
|---|---:|---:|---:|---:|
| `none` | - | 218 | - | - |
| `concise` | - | 91 | - | - |
| `inform_small` | 2,000 | 367 | 9.2x | 0/184 = 0.0% |
| `resize_small` | 2,000 | 396 | 9.2x | 2/184 = 1.1% |
| `inform_large` | 10,000 | 372 | 46.0x | 0/184 = 0.0% |
| `resize_large` | 10,000 | 1692 | 46.0x | 104/230 = 45.2% |

**Capacity is registered as a category, not a quantity.** Paired on item and
stage (n=138), advertising 2,000 tokens versus 10,000 -- a fivefold difference in
stated headroom -- changes output by **-1.8 tokens, 95% CI [-19.4, +14.9]**:
indistinguishable. The model notices that room was mentioned and does not scale to
how much. Only an explicit request to *use* the space breaks that ceiling, and when
it does it overshoots into repetition. The ladder in median completion tokens:
`concise` 91 -> no instruction 218 -> told 2k 367 -> told 10k 372 -> asked to fill
10k 1,692.

**The 2,000-token window is not a small window for this model.** It is about nine
times what the model writes unprompted, and larger than the source document for
nearly every item; it was exceeded in ~1% of calls. The 10,000-token window is
roughly forty-six times the baseline. Neither advertised capacity ever acted as a
ceiling; both acted as *headroom*.

That reframes three results that otherwise read as anomalies:

1. Being told about the "small" 2k reader made handoffs **longer**, not shorter,
   because 2k reads as permission rather than as a limit.
2. `inform_small` and `inform_large` behave almost identically, because both say
   "you have far more room than you are using".
3. A requested "shrink" raised side-fact retention, because nothing was shrunk.

**The only genuine compression cue here is `concise`** (91 tokens, 0.42x baseline) -- and it is
also the only arm that loses reusable facts. What this experiment actually varies
is *how much headroom the writer is told it has, and whether it is asked to fill
it*. A true downward constraint would need an advertised window below the baseline
(~150 tokens or fewer); no such arm was run.

## Degenerate expansion (read before the length table)

`resize_large` hit the 12000-token budget in **102/230 = 44.3%** of calls. This is NOT a
budget that is merely too small. A probe at max_tokens=32,000 on 8 items found the
same split: 5 stopped naturally at 631-1,006 tokens, 3 ran to 32,000 (24,000-27,000
words). Raising the ceiling only buys the runaway mode more room.

The two modes are qualitatively different, and internal repetition separates them:

| mode | n | median words | internal repetition | side facts kept |
|---|---:|---:|---:|---:|
| ran to the budget | 102 | 9687 | 0.912 | 1.000 |
| stopped before the budget | 127 | 784 | 0.088 | 0.800 |
| control, for scale | 138 | 169 | 0.000 | 0.400 |

One additional `resize_large` call ended with `finish_reason=error` and is not
included in either finish-mode row.

A median internal repetition of 0.912 means almost every five-word run in the
runaway text already occurred earlier in that same text. That is a decoding loop,
not added detail -- and these run at temperature 0, where greedy decoding is known
to fall into repetition on long generations.

**Two consequences for every expand number in this report.**

1. *Means are contaminated.* `resize_expand`'s mean length is a mixture of ~800-word
   handoffs and ~10,000-word loops; the mean describes neither mode. Prefer the
   medians above, or split on `truncated` in `stages.jsonl`.
2. *String-presence measures are inflated.* A text that restates everything twenty
   times trivially contains every tracked fact, which is why the runaway mode scores
   1.000 on side-fact retention.
   stopped expand calls (0.800) and control
   calls (0.400) differ descriptively, but this
   post-finish-state split is not a paired causal contrast.

Thirty-nine of 46 items show both finish modes somewhere across their five
`resize_large` calls. Document identity alone therefore does not determine the
failure; chain state and/or generation nondeterminism may matter.

![Call-level expansion modes](expansion_modes.png)

## What kind of material expansion adds

Redundancy first: is the extra text repeated, or new?

| directive | n | median words | internal repetition |
|---|---:|---:|---:|
| `none` | 138 | 169 | 0.000 |
| `concise` | 138 | 65 | 0.000 |
| `inform_small` | 184 | 274 | 0.005 |
| `resize_small` | 184 | 296 | 0.004 |
| `inform_large` | 184 | 292 | 0.006 |
| `resize_large` | 230 | 1404 | 0.330 |
| `resize_large`, ran away | 102 | 9687 | 0.912 |
| `resize_large`, normal | 128 | 783 | 0.087 |

Then the unsupported material, split by dataset. `corrupted` is a near-miss
restatement of a tracked value; `invented` is a term with no counterpart in the
document at all.

| dataset | directive | n | unsupported/handoff | corrupted | invented |
|---|---|---:|---:|---:|---:|
| counterfactual | `none` | 78 | 2.37 | 0.18 | 2.19 |
| counterfactual | `concise` | 78 | 0.94 | 0.17 | 0.77 |
| counterfactual | `inform_small` | 104 | 2.52 | 0.43 | 2.09 |
| counterfactual | `resize_small` | 104 | 2.64 | 0.45 | 2.19 |
| counterfactual | `inform_large` | 104 | 2.58 | 0.52 | 2.06 |
| counterfactual | `resize_large` | 130 | 5.81 | 0.68 | 5.12 |
| fictional | `none` | 60 | 2.40 | 0.18 | 2.22 |
| fictional | `concise` | 60 | 1.23 | 0.15 | 1.08 |
| fictional | `inform_small` | 80 | 2.36 | 0.19 | 2.17 |
| fictional | `resize_small` | 80 | 1.90 | 0.15 | 1.75 |
| fictional | `inform_large` | 80 | 2.42 | 0.05 | 2.38 |
| fictional | `resize_large` | 100 | 4.88 | 0.22 | 4.66 |

### Is the invention parametric drift, or unrelated?

`parametric_present` is the direct test: does the value the model was verified
to hold *from memory* appear in the handoff text?

| directive | n | memorised value present | document answer present |
|---|---:|---:|---:|
| `none` | 78 | 0.000 | 0.808 |
| `concise` | 78 | 0.051 | 0.949 |
| `inform_small` | 104 | 0.058 | 0.837 |
| `resize_small` | 104 | 0.000 | 0.808 |
| `inform_large` | 104 | 0.135 | 0.808 |
| `resize_large` | 130 | 0.015 | 0.854 |

**The added material is overwhelmingly invention, not memory.** Expansion
roughly doubles unsupported terms per handoff, and it does so by about the same
amount on fictional items -- where no memorised value exists to resurface -- as
on counterfactual ones. If expansion were pulling pretrained knowledge back in,
the counterfactual set would have to separate from the fictional set, and it
barely does. Meanwhile the memorised value itself stays rare in the handoff
text under every directive, and is no more common under a request to expand
than under the size-neutral control.

This is the counterfactual/fictional pairing doing the job it was built for: it
converts 'the model added things that are not in the document' into the sharper
'the model invented new material rather than reverting to what it remembers'.

## What the directive costs in accuracy

Depth 0 answers directly from the source document and is shared by every arm:
it is the ceiling, not an arm. Everything below is judged accuracy on the
target question.

| dataset | depth 0 (no handoff) | control @3 | asked to shrink @3 | asked to expand @3 |
|---|---:|---:|---:|---:|
| counterfactual | 0.962 | 0.808 | 0.654 | 0.654 |
| fictional | 1.000 | 0.900 | 0.750 | 0.600 |

Handing off at all costs roughly 10-15 points against reading the document
directly. **Asking for expansion roughly doubles that loss** and is the worst
condition in the experiment on both datasets -- the only directive that is
clearly worse than saying nothing. No length-increasing directive buys
accuracy; the shortest arms are the most accurate.

## Does the effect survive depth?

Capacity information is not a one-shot prompt effect that washes out as the
chain rewrites itself. Informed-vs-control on handoff length, every depth:

| dataset | depth | delta words | 95% CI | excludes 0 |
|---|---:|---:|---|:-:|
| counterfactual | 1 | +71.8 | [+52.3, +96.9] | yes |
| counterfactual | 2 | +92.0 | [+72.6, +115.9] | yes |
| counterfactual | 3 | +119.5 | [+99.6, +140.8] | yes |
| fictional | 1 | +127.4 | [+88.2, +170.9] | yes |
| fictional | 2 | +155.6 | [+118.5, +196.4] | yes |
| fictional | 3 | +189.2 | [+150.5, +229.2] | yes |

The gap widens monotonically with depth rather than decaying, so each
successive rewrite re-applies the directive rather than diluting it.

## Does a requested shrink discard reusable information?

The design's motivating worry was that shrinking would strip facts no current
question asks about. Measured against a size-neutral control, the sign is the
opposite:

| dataset | depth | delta side-fact retention | 95% CI | excludes 0 |
|---|---:|---:|---|:-:|
| counterfactual | 1 | +0.146 | [+0.031, +0.262] | yes |
| counterfactual | 2 | +0.138 | [+0.023, +0.246] | yes |
| counterfactual | 3 | +0.115 | [-0.008, +0.231] |  |
| fictional | 1 | +0.270 | [+0.180, +0.360] | yes |
| fictional | 2 | +0.270 | [+0.180, +0.360] | yes |
| fictional | 3 | +0.280 | [+0.180, +0.370] | yes |

Asking the writer to fit a 2,000-token reader *raised* the share of retained
reusable facts. The control is not a careful baseline -- it is loose prose with
no length pressure at all -- and a shrink request appears to make the writer
more factually dense rather than more selective. The loss this experiment set
out to find is not produced by asking for less; it is produced by the
unquantified word *concise* (see the contrasts table) and by asking for more.

## Support judge (semantic corroboration)

The deterministic scan counts unsupported *terms*; this counts unsupported
*claims*, judged by a different model family against the source document.
Every verdict parsed.

| directive | n | mean unsupported claims |
|---|---:|---:|
| `none` | 138 | 1.39 |
| `concise` | 138 | 0.62 |
| `inform_small` | 184 | 2.28 |
| `resize_small` | 184 | 2.46 |
| `inform_large` | 184 | 2.26 |
| `resize_large` | 230 | 3.38 |
| `resize_large`, ran away | 102 | 2.80 |
| `resize_large`, normal | 128 | 3.84 |

The two expansion modes fail *differently*, which the term-level scan alone
does not show: the runaway mode makes fewer novel unsupported claims (2.80) than well-behaved expansion (3.84),
because it pads by restating what it already said. Normal-mode expansion
pads by inventing. One directive, two failure modes: repetition and
fabrication.

## Handoff length at depth 3

| dataset | arm | words | answer accuracy | side facts kept | repetition |
|---|---|---:|---:|---:|---:|
| counterfactual | asked to expand | 8661 | 0.654 | 0.831 | 0.828 |
| counterfactual | asked: large -> small -> large | 4026 | 0.692 | 0.769 | 0.436 |
| counterfactual | asked: shrink, shrink, expand | 3183 | 0.577 | 0.631 | 0.355 |
| counterfactual | informed small | 284 | 0.731 | 0.669 | 0.019 |
| counterfactual | informed: large -> small -> large | 283 | 0.692 | 0.592 | 0.019 |
| counterfactual | informed large | 269 | 0.769 | 0.631 | 0.012 |
| counterfactual | asked to shrink | 241 | 0.654 | 0.631 | 0.013 |
| counterfactual | control (no size instruction) | 165 | 0.808 | 0.515 | 0.005 |
| counterfactual | explicit unquantified concision cue | 50 | 0.846 | 0.238 | 0.000 |
| fictional | asked to expand | 6452 | 0.600 | 0.950 | 0.627 |
| fictional | asked: large -> small -> large | 3652 | 0.600 | 0.930 | 0.351 |
| fictional | asked: shrink, shrink, expand | 1967 | 0.800 | 0.610 | 0.194 |
| fictional | informed: large -> small -> large | 444 | 0.850 | 0.610 | 0.024 |
| fictional | informed large | 438 | 0.850 | 0.610 | 0.028 |
| fictional | informed small | 416 | 0.900 | 0.680 | 0.020 |
| fictional | asked to shrink | 328 | 0.750 | 0.600 | 0.008 |
| fictional | control (no size instruction) | 227 | 0.900 | 0.320 | 0.004 |
| fictional | explicit unquantified concision cue | 69 | 0.850 | 0.170 | 0.000 |

## Contrasts (paired, per item)

Only the length and accuracy measures are shown here; `contrasts.csv` holds every
measure at every depth. A contrast whose interval excludes zero is marked.

| contrast | dataset | measure | delta | 95% CI | excludes 0 |
|---|---|---|---:|---|:-:|
| inform_shrink_vs_control | counterfactual | summary_words | +119.500 | [+99.577, +140.808] | yes |
| inform_shrink_vs_control | counterfactual | internal_repetition | +0.014 | [+0.006, +0.022] | yes |
| inform_shrink_vs_control | counterfactual | side_fact_retention | +0.154 | [+0.069, +0.254] | yes |
| inform_shrink_vs_control | counterfactual | unsupported_count | +0.269 | [-0.308, +0.846] |  |
| inform_shrink_vs_control | counterfactual | judge_correct | -0.077 | [-0.192, +0.000] |  |
| inform_shrink_vs_control | fictional | summary_words | +189.250 | [+150.499, +229.250] | yes |
| inform_shrink_vs_control | fictional | internal_repetition | +0.015 | [+0.008, +0.024] | yes |
| inform_shrink_vs_control | fictional | side_fact_retention | +0.360 | [+0.240, +0.490] | yes |
| inform_shrink_vs_control | fictional | unsupported_count | +0.500 | [-0.650, +1.900] |  |
| inform_shrink_vs_control | fictional | judge_correct | +0.000 | [-0.150, +0.150] |  |
| inform_expand_vs_control | counterfactual | summary_words | +104.769 | [+82.884, +128.040] | yes |
| inform_expand_vs_control | counterfactual | internal_repetition | +0.007 | [+0.001, +0.013] | yes |
| inform_expand_vs_control | counterfactual | side_fact_retention | +0.115 | [+0.008, +0.231] | yes |
| inform_expand_vs_control | counterfactual | unsupported_count | +0.038 | [-0.538, +0.654] |  |
| inform_expand_vs_control | counterfactual | judge_correct | -0.038 | [-0.231, +0.154] |  |
| inform_expand_vs_control | fictional | summary_words | +211.500 | [+163.699, +261.651] | yes |
| inform_expand_vs_control | fictional | internal_repetition | +0.023 | [+0.008, +0.044] | yes |
| inform_expand_vs_control | fictional | side_fact_retention | +0.290 | [+0.170, +0.420] | yes |
| inform_expand_vs_control | fictional | unsupported_count | -0.250 | [-0.900, +0.450] |  |
| inform_expand_vs_control | fictional | judge_correct | -0.050 | [-0.200, +0.100] |  |
| resize_vs_inform_shrink | counterfactual | summary_words | -42.769 | [-66.885, -21.231] | yes |
| resize_vs_inform_shrink | counterfactual | internal_repetition | -0.006 | [-0.015, +0.003] |  |
| resize_vs_inform_shrink | counterfactual | side_fact_retention | -0.038 | [-0.162, +0.069] |  |
| resize_vs_inform_shrink | counterfactual | unsupported_count | -0.154 | [-1.154, +1.077] |  |
| resize_vs_inform_shrink | counterfactual | judge_correct | -0.077 | [-0.192, +0.000] |  |
| resize_vs_inform_shrink | fictional | summary_words | -87.550 | [-145.101, -37.246] | yes |
| resize_vs_inform_shrink | fictional | internal_repetition | -0.011 | [-0.019, -0.004] | yes |
| resize_vs_inform_shrink | fictional | side_fact_retention | -0.080 | [-0.240, +0.070] |  |
| resize_vs_inform_shrink | fictional | unsupported_count | -1.250 | [-2.950, +0.200] |  |
| resize_vs_inform_shrink | fictional | judge_correct | -0.150 | [-0.400, +0.100] |  |
| resize_vs_inform_expand | counterfactual | summary_words | +8392.077 | [+7084.363, +9465.931] | yes |
| resize_vs_inform_expand | counterfactual | internal_repetition | +0.816 | [+0.723, +0.888] | yes |
| resize_vs_inform_expand | counterfactual | side_fact_retention | +0.200 | [+0.077, +0.323] | yes |
| resize_vs_inform_expand | counterfactual | unsupported_count | +3.346 | [+1.192, +6.308] | yes |
| resize_vs_inform_expand | counterfactual | judge_correct | -0.115 | [-0.346, +0.115] |  |
| resize_vs_inform_expand | fictional | summary_words | +6013.550 | [+4269.684, +7698.116] | yes |
| resize_vs_inform_expand | fictional | internal_repetition | +0.599 | [+0.425, +0.761] | yes |
| resize_vs_inform_expand | fictional | side_fact_retention | +0.340 | [+0.210, +0.470] | yes |
| resize_vs_inform_expand | fictional | unsupported_count | +2.900 | [+0.850, +5.800] | yes |
| resize_vs_inform_expand | fictional | judge_correct | -0.250 | [-0.500, +0.000] |  |
| resize_shrink_vs_control | counterfactual | summary_words | +76.731 | [+55.307, +99.308] | yes |
| resize_shrink_vs_control | counterfactual | internal_repetition | +0.008 | [+0.002, +0.014] | yes |
| resize_shrink_vs_control | counterfactual | side_fact_retention | +0.115 | [-0.008, +0.231] |  |
| resize_shrink_vs_control | counterfactual | unsupported_count | +0.115 | [-0.769, +1.192] |  |
| resize_shrink_vs_control | counterfactual | judge_correct | -0.154 | [-0.308, -0.038] | yes |
| resize_shrink_vs_control | fictional | summary_words | +101.700 | [+68.300, +132.600] | yes |
| resize_shrink_vs_control | fictional | internal_repetition | +0.004 | [-0.001, +0.009] |  |
| resize_shrink_vs_control | fictional | side_fact_retention | +0.280 | [+0.180, +0.370] | yes |
| resize_shrink_vs_control | fictional | unsupported_count | -0.750 | [-1.450, +0.050] |  |
| resize_shrink_vs_control | fictional | judge_correct | -0.150 | [-0.350, +0.050] |  |
| resize_expand_vs_control | counterfactual | summary_words | +8496.846 | [+7189.175, +9570.612] | yes |
| resize_expand_vs_control | counterfactual | internal_repetition | +0.822 | [+0.729, +0.895] | yes |
| resize_expand_vs_control | counterfactual | side_fact_retention | +0.315 | [+0.192, +0.438] | yes |
| resize_expand_vs_control | counterfactual | unsupported_count | +3.385 | [+1.231, +6.269] | yes |
| resize_expand_vs_control | counterfactual | judge_correct | -0.154 | [-0.308, -0.038] | yes |
| resize_expand_vs_control | fictional | summary_words | +6225.050 | [+4496.836, +7890.574] | yes |
| resize_expand_vs_control | fictional | internal_repetition | +0.623 | [+0.461, +0.773] | yes |
| resize_expand_vs_control | fictional | side_fact_retention | +0.630 | [+0.540, +0.720] | yes |
| resize_expand_vs_control | fictional | unsupported_count | +2.650 | [+0.600, +5.500] | yes |
| resize_expand_vs_control | fictional | judge_correct | -0.300 | [-0.550, -0.050] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | summary_words | +3742.462 | [+2079.619, +5460.612] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | internal_repetition | +0.417 | [+0.276, +0.564] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | side_fact_retention | +0.177 | [+0.062, +0.292] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | unsupported_count | +3.308 | [+1.385, +5.731] | yes |
| resize_lsl_vs_inform_lsl | counterfactual | judge_correct | +0.000 | [-0.231, +0.192] |  |
| resize_lsl_vs_inform_lsl | fictional | summary_words | +3207.750 | [+1578.347, +4883.524] | yes |
| resize_lsl_vs_inform_lsl | fictional | internal_repetition | +0.327 | [+0.158, +0.504] | yes |
| resize_lsl_vs_inform_lsl | fictional | side_fact_retention | +0.320 | [+0.190, +0.450] | yes |
| resize_lsl_vs_inform_lsl | fictional | unsupported_count | +2.950 | [+0.450, +5.600] | yes |
| resize_lsl_vs_inform_lsl | fictional | judge_correct | -0.250 | [-0.450, -0.050] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | summary_words | +3784.462 | [+2132.153, +5494.996] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | internal_repetition | +0.423 | [+0.284, +0.569] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | side_fact_retention | +0.138 | [+0.023, +0.262] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | unsupported_count | +3.769 | [+1.731, +6.346] | yes |
| resize_lsl_vs_resize_shrink | counterfactual | judge_correct | +0.038 | [-0.115, +0.192] |  |
| resize_lsl_vs_resize_shrink | fictional | summary_words | +3323.800 | [+1680.342, +5012.773] | yes |
| resize_lsl_vs_resize_shrink | fictional | internal_repetition | +0.343 | [+0.173, +0.519] | yes |
| resize_lsl_vs_resize_shrink | fictional | side_fact_retention | +0.330 | [+0.220, +0.440] | yes |
| resize_lsl_vs_resize_shrink | fictional | unsupported_count | +4.400 | [+2.450, +6.550] | yes |
| resize_lsl_vs_resize_shrink | fictional | judge_correct | -0.150 | [-0.450, +0.150] |  |
| resize_sse_vs_resize_shrink | counterfactual | summary_words | +2941.731 | [+1544.230, +4565.395] | yes |
| resize_sse_vs_resize_shrink | counterfactual | internal_repetition | +0.343 | [+0.212, +0.484] | yes |
| resize_sse_vs_resize_shrink | counterfactual | side_fact_retention | +0.000 | [+0.000, +0.000] |  |
| resize_sse_vs_resize_shrink | counterfactual | unsupported_count | +3.808 | [+2.538, +5.269] | yes |
| resize_sse_vs_resize_shrink | counterfactual | judge_correct | -0.077 | [-0.192, +0.000] |  |
| resize_sse_vs_resize_shrink | fictional | summary_words | +1638.450 | [+364.693, +3025.105] | yes |
| resize_sse_vs_resize_shrink | fictional | internal_repetition | +0.186 | [+0.061, +0.326] | yes |
| resize_sse_vs_resize_shrink | fictional | side_fact_retention | +0.010 | [-0.030, +0.060] |  |
| resize_sse_vs_resize_shrink | fictional | unsupported_count | +2.950 | [+1.300, +5.100] | yes |
| resize_sse_vs_resize_shrink | fictional | judge_correct | +0.050 | [+0.000, +0.150] |  |
| concise_vs_control | counterfactual | summary_words | -114.115 | [-130.693, -98.577] | yes |
| concise_vs_control | counterfactual | internal_repetition | -0.005 | [-0.010, -0.002] | yes |
| concise_vs_control | counterfactual | side_fact_retention | -0.277 | [-0.423, -0.138] | yes |
| concise_vs_control | counterfactual | unsupported_count | -1.615 | [-2.231, -1.000] | yes |
| concise_vs_control | counterfactual | judge_correct | +0.038 | [-0.115, +0.192] |  |
| concise_vs_control | fictional | summary_words | -157.600 | [-187.900, -130.949] | yes |
| concise_vs_control | fictional | internal_repetition | -0.004 | [-0.007, -0.002] | yes |
| concise_vs_control | fictional | side_fact_retention | -0.150 | [-0.250, -0.040] | yes |
| concise_vs_control | fictional | unsupported_count | -1.650 | [-2.201, -1.050] | yes |
| concise_vs_control | fictional | judge_correct | -0.050 | [-0.200, +0.100] |  |
| resize_shrink_vs_concise | counterfactual | summary_words | +190.846 | [+171.038, +211.000] | yes |
| resize_shrink_vs_concise | counterfactual | internal_repetition | +0.013 | [+0.007, +0.020] | yes |
| resize_shrink_vs_concise | counterfactual | side_fact_retention | +0.392 | [+0.262, +0.531] | yes |
| resize_shrink_vs_concise | counterfactual | unsupported_count | +1.731 | [+0.923, +2.654] | yes |
| resize_shrink_vs_concise | counterfactual | judge_correct | -0.192 | [-0.385, +0.000] |  |
| resize_shrink_vs_concise | fictional | summary_words | +259.300 | [+222.250, +291.701] | yes |
| resize_shrink_vs_concise | fictional | internal_repetition | +0.008 | [+0.004, +0.013] | yes |
| resize_shrink_vs_concise | fictional | side_fact_retention | +0.430 | [+0.310, +0.540] | yes |
| resize_shrink_vs_concise | fictional | unsupported_count | +0.900 | [+0.150, +1.750] | yes |
| resize_shrink_vs_concise | fictional | judge_correct | -0.100 | [-0.400, +0.150] |  |

![Headline non-runaway contrasts](headline_contrasts.png)

## Facts that went missing, and what came back

`recovery` is measured only over facts that were actually lost. Because stage 2+
sees a sealed handoff, a fact that reappears was regenerated, not copied.

| dataset | arm | role | facts | lost | recovered | reverted | fabricated |
|---|---|---|---:|---:|---:|---:|---:|
| counterfactual | explicit unquantified concision cue | side | 130 | 0.023 | 0.000 | 0.000 | 0.333 |
| counterfactual | control (no size instruction) | side | 130 | 0.015 | 0.000 | 0.000 | 1.000 |
| counterfactual | informed large | side | 130 | 0.023 | 0.000 | 0.000 | 0.667 |
| counterfactual | informed: large -> small -> large | side | 130 | 0.062 | 0.000 | 0.000 | 1.000 |
| counterfactual | informed small | side | 130 | 0.008 | 0.000 | 0.000 | 0.000 |
| counterfactual | asked to expand | side | 130 | 0.038 | 0.000 | 0.000 | 0.800 |
| counterfactual | asked: large -> small -> large | side | 130 | 0.108 | 0.000 | 0.000 | 0.643 |
| counterfactual | asked to shrink | side | 130 | 0.046 | 0.000 | 0.000 | 1.000 |
| counterfactual | asked: shrink, shrink, expand | side | 130 | 0.046 | 0.000 | 0.000 | 1.000 |
| counterfactual | explicit unquantified concision cue | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | control (no size instruction) | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | informed large | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | informed: large -> small -> large | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | informed small | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | asked to expand | target | 26 | 0.115 | 0.000 | 0.000 | 0.333 |
| counterfactual | asked: large -> small -> large | target | 26 | 0.154 | 0.000 | 0.000 | 0.500 |
| counterfactual | asked to shrink | target | 26 | 0.000 | nan | nan | nan |
| counterfactual | asked: shrink, shrink, expand | target | 26 | 0.000 | nan | nan | nan |
| fictional | explicit unquantified concision cue | side | 100 | 0.010 | 0.000 | 0.000 | 0.000 |
| fictional | control (no size instruction) | side | 100 | 0.030 | 0.000 | 0.000 | 0.667 |
| fictional | informed large | side | 100 | 0.000 | nan | nan | nan |
| fictional | informed: large -> small -> large | side | 100 | 0.000 | nan | nan | nan |
| fictional | informed small | side | 100 | 0.000 | nan | nan | nan |
| fictional | asked to expand | side | 100 | 0.020 | 0.000 | 0.000 | 1.000 |
| fictional | asked: large -> small -> large | side | 100 | 0.040 | 0.000 | 0.000 | 1.000 |
| fictional | asked to shrink | side | 100 | 0.020 | 0.000 | 0.000 | 0.000 |
| fictional | asked: shrink, shrink, expand | side | 100 | 0.010 | 0.000 | 0.000 | 0.000 |
| fictional | explicit unquantified concision cue | target | 20 | 0.000 | nan | nan | nan |
| fictional | control (no size instruction) | target | 20 | 0.050 | 0.000 | 0.000 | 0.000 |
| fictional | informed large | target | 20 | 0.000 | nan | nan | nan |
| fictional | informed: large -> small -> large | target | 20 | 0.050 | 0.000 | 0.000 | 0.000 |
| fictional | informed small | target | 20 | 0.000 | nan | nan | nan |
| fictional | asked to expand | target | 20 | 0.100 | 0.000 | 0.000 | 0.500 |
| fictional | asked: large -> small -> large | target | 20 | 0.150 | 0.000 | 0.000 | 0.333 |
| fictional | asked to shrink | target | 20 | 0.100 | 0.000 | 0.000 | 0.500 |
| fictional | asked: shrink, shrink, expand | target | 20 | 0.050 | 0.000 | 0.000 | 1.000 |

## Where the counterfactual answer came from (depth 3)

| arm | document | memorised | both | other |
|---|---:|---:|---:|---:|
| explicit unquantified concision cue | 0.846 | 0.115 | 0.000 | 0.038 |
| control (no size instruction) | 0.808 | 0.077 | 0.000 | 0.115 |
| informed large | 0.769 | 0.115 | 0.000 | 0.115 |
| informed: large -> small -> large | 0.692 | 0.192 | 0.000 | 0.115 |
| informed small | 0.731 | 0.154 | 0.000 | 0.115 |
| asked to expand | 0.654 | 0.192 | 0.000 | 0.154 |
| asked: large -> small -> large | 0.692 | 0.269 | 0.000 | 0.038 |
| asked to shrink | 0.654 | 0.269 | 0.000 | 0.077 |
| asked: shrink, shrink, expand | 0.577 | 0.308 | 0.000 | 0.115 |

## Cost

```
{
  "calls_total": 0,
  "calls_live": 0,
  "calls_cached": 0,
  "cache_hit_rate": 0.0,
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "prompt_tokens_live": 0,
  "completion_tokens_live": 0,
  "cost_usd": 0.0,
  "api_calls_attempted": 0,
  "api_call_cap": 30000
}
```
