# Agent Handoff Information-Loss Probe

When a subagent gathers evidence and hands a summary to an orchestrator, the subagent's raw
context is discarded. This probe asks: **when the evidence is already found, how much answer
accuracy does the handoff alone destroy, and which kinds of loss cause the damage?**

Retrieval is removed as a variable by construction — the subagent is *given* the gold evidence.
Any accuracy drop is therefore attributable to the handoff.

---

## Status

Built and verified: stages 0, 1, 2 and 4, with mechanisms `A_full`, `B_freeform`,
`C_structured`, `D_extractive`, `E_oracle`.

**Stage 3 (injections) is deliberately not built yet.** Per the build order, the `A_full` /
`B_freeform` / `E_oracle` gap is measured at n=10 first; if there is no headroom, the design
needs rethinking before anything else is spent. `src/inject.py` appears in the layout below but
does not exist yet.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

Set your key (it is read only from the environment — never from a file, never from an argument):

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Estimate the spend before spending it:

```bash
python src/run.py --dry-run --mechanisms A_full,B_freeform,C_structured,D_extractive,E_oracle
```

Run the 10-question pilot:

```bash
python src/run.py --candidates 40 --n 10 --mechanisms A_full,B_freeform,E_oracle
```

Run the full probe:

```bash
python src/run.py --mechanisms A_full,B_freeform,C_structured,D_extractive,E_oracle
```

Run Experiment 6 on the corrected 10-example SQuAD sample; each experimental
input contains one shared gold passage and zero distractors (cached calls are
reused automatically):

```bash
python src/run_multilingual_handoffs.py --config multilingual_handoff_config.yaml --n 10
```

Offline checks that need no API key (data shaping, scoring, isolation guarantee):

```bash
python src/selftest_offline.py
```

Every LLM call is content-hashed into `cache/`, so re-running any stage costs nothing.
Re-analysis is free by design — the analysis is meant to be iterated on.

## Layout

```
handoff-probe/
  config.yaml              model, dataset, sample size, seeds, cost cap
  fictional_summary_generalization_config.yaml   Experiment 5b fixed-slot fictional QA
  fictional_model_bottleneck_config.yaml         Experiment 8b selector/relay bottleneck
  size_adaptation_config.yaml   Experiment 9 directives, arms and dataset construction
  PROMPTS.md                every prompt string used anywhere, grouped by experiment
  results/HANDOFF_EXPERIMENTS_REPORT.md   the synthesized findings across all experiments
  src/
    data.py                dataset load, C1 leakage filter, sample, gold-sentence derivation
    llm.py                 OpenRouter client: retries, cache, token accounting, cost cap
    handoffs.py            single-handoff mechanisms (Experiment 1) + sealed orchestrator boundary
    judge.py                shared LLM-judge answer-correctness scoring
    score.py                EM/F1, bootstrap CIs
    run.py                  Experiment 1 -- single-handoff mechanism pilot
    chain_data.py            dataset adapters shared by the chain experiment
    run_chain.py             Experiment 2 -- repeated handoff degradation
                              (also drives the Qwen replications and the
                              question-conditioned/generic replication --
                              see chain_config.yaml, qwen_chain_config.yaml,
                              qwen32_chain_config.yaml,
                              chain_generic_config.yaml)
    run_retrieval_quality.py Experiment 3 -- retrieval quality through repeated handoffs
    run_redundant_signal_ratio.py  Experiment 4 -- fixed-context redundant-evidence signal ratio
    run_summary_generalization.py  Experiment 5 -- question conditioning and
                                    cross-question generalization
                                    (also drives the paraphrase-only /
                                    pass-through arms -- see
                                    summary_generalization_paraphrase_config.yaml)
    paraphrase_metrics.py    per-edge lexical measures for Experiment 5
                              (no API calls, no judge)
    fictional_qa.py          shared fictional-dossier cards, rotations, parsing,
                              sealing and capacity-matched interventions
    run_fictional_summary_generalization.py  Experiment 5b -- fixed-capacity
                              conditioned vs generic evidence selection
    run_multilingual_handoffs.py    Experiment 6 -- gold-only fixed-language
                                    vs language-switching handoff chains
    model_pool.py            model registry + one LLMClient per model over a
                              shared cache and a shared cost ledger
    run_model_heterogeneity.py      Experiment 8 -- who writes each handoff:
                                    homogeneous vs cross-family chains, and
                                    small/large size trajectories
    run_fictional_model_bottleneck.py  Experiment 8b -- small/large source
                              selector x sealed downstream relay after a new query
    size_metrics.py          Experiment 9 measures: internal repetition, unsupported-term
                              scan, fact life histories (no API calls, no judge)
    build_size_adaptation_data.py   Experiment 9 datasets: counterfactual items verified
                                    known-original/unknown-replacement, and fictional
                                    items verified unknown in both directions
    run_size_adaptation.py   Experiment 9 -- does the handoff change when the writer is
                              told the reader's context size, and does asking differ
                              from merely informing
    run_slack_facts.py       unwritten follow-up correcting Exp. 3/4's signal/filler confound
    run_slack_retrieval.py   unwritten follow-up correcting Exp. 3's retention-pooling defect
    plot_conditioning_comparison.py   analysis-only plot for the chain_generic replication
    selftest_offline.py      checks that need no API key
    selftest_chain_offline.py  chain-experiment checks, incl. the isolation guarantee
    selftest_summary_generalization_offline.py  Experiment 5 arm/transition checks
    selftest_fictional_qa_offline.py            shared fictional-QA invariants
    selftest_fictional_summary_generalization_offline.py  Experiment 5b checks
    selftest_model_heterogeneity_offline.py     Experiment 8 schedule/prompt/plot checks
    selftest_fictional_model_bottleneck_offline.py  Experiment 8b checks
    selftest_size_adaptation_offline.py         Experiment 9 directive/recovery/plot checks
  data/  runs/  cache/  results/
```

See [PROMPTS.md](PROMPTS.md) for the exact text of every prompt used above, and
[results/HANDOFF_EXPERIMENTS_REPORT.md](results/HANDOFF_EXPERIMENTS_REPORT.md)
for the synthesized cross-experiment findings.

## Conditions

| ID | Handoff mechanism | Purpose |
|----|-------------------|---------|
| `A_full` | None — single agent answers directly from the paragraphs | Ceiling |
| `B_freeform` | Subagent writes a free-form prose summary | The realistic default |
| `C_structured` | Subagent emits JSON claims with `source_para_id`, `constraints[]`, `confidence`, plus `unresolved[]` | Candidate improvement + injection substrate |
| `D_extractive` | Subagent returns verbatim sentences only, with paragraph ids | Tests whether generation itself is the loss |
| `E_oracle` | Gold supporting sentences, verbatim, with ids — no LLM | Upper bound on any handoff |

`A_full` − `B_freeform` is the headroom. `E_oracle` − `B_freeform` is how much a perfect handoff
recovers. `E_oracle` − `A_full` is interesting on its own: if the oracle beats full context,
compression is *helping* by denoising.

## How the hard constraints are implemented

**C1 — parametric leakage filter.** Every candidate is answered closed-book (no evidence),
3 samples at temperature 0.7. A question is discarded if *any* attempt is correct, where
"correct" means exact match **or** token-F1 ≥ `leakage_filter.f1_known_threshold` (0.6). The
F1 arm is stricter than plain EM: a near-miss paraphrase still counts as parametric knowledge.
The pass rate is printed and written to `results/c1_filter_summary.json`.

**C2 / C6 — same model, and how the temperature tension is resolved.** The subagent and the
orchestrator are the same model, share one system prompt (`ANSWER_SYSTEM` is used verbatim by
both `A_full` and the orchestrator), and share `top_p` and `max_tokens`. They differ in exactly
one setting, and only because C6 requires it: the orchestrator answers at temperature 0 while
the subagent summarises at 0.7, so that seed-to-seed variation comes from summarisation rather
than answer sampling. There is no capability asymmetry.

A consequence: `A_full` and `E_oracle` involve no sampled generation at all, so they are
deterministic and are generated **once** per question rather than once per seed. Only
`B_freeform`, `C_structured` and `D_extractive` are run under both seeds.

**C3 — injections operate on structured data.** `C_structured` is parsed into a normalised
object. `render_structured()` is the *only* path from object to orchestrator-visible text, so a
corrupted object and a clean one are serialised by identical code. No LLM is used to corrupt
anything. (Stage 3 not built yet; the substrate is in place and tested.)

**C4 — token accounting.** Every call records prompt tokens, completion tokens, latency, cost
and cache hit/miss. `results/summary.csv` reports handoff tokens, answer tokens and their total
per condition alongside accuracy, so a condition cannot win by spending more.

**C5 — cache everything.** `hash(model, messages, temperature, top_p, max_tokens, seed,
response_format)` → one JSON file under `cache/`. Nothing environment-specific enters the hash.

**C7 — deterministic primary metrics plus an independent semantic check.**
SQuAD-style normalisation (lowercase, strip punctuation, drop articles, collapse
whitespace), exact match, and token-F1 remain primary and are maxed over the gold
answer and its aliases. `summary.csv` carries an `em_f1_disagree` column so a
materially divergent EM/F1 picture is visible rather than averaged away. A
temperature-0 `openai/gpt-4o-mini` judge from a different model family is
reported as a secondary correct/incorrect measure because repeated rewriting can
produce a correct paraphrase with low token overlap. It complements rather than
replaces EM/F1, and every verdict remains auditable.

**Stage 2 isolation, enforced in code.** `orchestrator_answer()` accepts only a frozen
`SealedHandoff` carrying four strings: `qid`, `question`, `handoff_text`, `mechanism`. The
`Question` object — and therefore every paragraph — is not reachable from that call frame.
Passing a `Question`, a dict, or a raw string raises `TypeError`. `run.py` runs
`selftest_orchestrator_isolation()` at startup on every invocation, so the guarantee is tested
rather than asserted. `A_full` is the one condition where paragraphs legitimately reach an
answering call, and it uses a separate function that never touches the sealed path.

## Data

The active dataset is a small, revision-pinned collection of **10 random English Wikipedia
pages, with exactly two unrelated questions per page**. `src/build_wikipedia_dataset.py` saves
each full plaintext page, its URL and revision metadata, and emits 20 `Question` records at
`data/wikipedia_random/questions.jsonl`. The runner never needs MuSiQue or HotpotQA once this
file exists.

The builder rejects pages outside the configured 4,000–40,000-character range rather than
truncating them. A strong reasoning model (`openai/gpt-5.2`, high reasoning effort by default)
first writes two source-grounded questions with verbatim evidence excerpts; a second independent
reasoning pass writes the canonical gold answer and genuine aliases. The exact source page text
is retained in `data/wikipedia_random/source_pages.jsonl`, so construction is auditable and a
later experiment has no dependence on a changing live page.

**Retrieval is not modelled.** The subagent is handed the evidence. This is intentional: it is
what makes any measured drop attributable to the handoff rather than to search quality.

The surviving question ids are written to `data/sampled_ids.json` and the full records to
`data/filtered_questions.jsonl`; both are committed so runs are reproducible.

The builder makes `E_oracle` auditable without an external annotation set: it retains the
verbatim excerpts supplied during question construction and maps them back to page sentences.

## Model choice

`meta-llama/llama-3.3-70b-instruct` — deliberately mid-capability, not frontier. A frontier
model would ceiling out on `A_full` and leave no headroom to measure, and would answer from
parametric knowledge more often despite the C1 filter. It is also non-reasoning, which keeps
token accounting interpretable.

Model identifiers were verified against OpenRouter's live `/api/v1/models` on 2026-08-11 rather
than assumed. The whole pipeline is model-agnostic: change `model.id` in `config.yaml` and rerun
for a second-model robustness check. Note the C1 filter is model-specific, so a new model needs
its own `stage0` (delete `data/filtered_questions.jsonl` or pass `--force`).

## Cost control

- Hard cap in `config.yaml` (`cost.cap_usd`); the run aborts with a clear message when hit, and
  cached work is preserved so nothing is re-paid for.
- `--dry-run` estimates calls and cost, broken down by call type, without spending anything.
  It never writes to `data/`, `runs/` or `results/`, so a dry run cannot poison a real one.
- Concurrency is configurable and defaults to 4.
- Measured estimates: the 10-question pilot is ~180 calls (~$0.02); the full 150-question run
  across all five mechanisms is ~3,600 calls (~$0.42–0.58).

## Limitations

- **Retrieval is bypassed.** The subagent is handed the gold evidence, so these results bound
  handoff loss only under guaranteed-perfect retrieval. Real systems compound handoff loss with
  search failures; this probe deliberately measures only the former.
- **Short-answer QA, not long-form research reports.** Compression pressure in real multi-agent
  research is far higher than summarising 20 paragraphs into a few hundred tokens, and the
  failure modes may not be the same ones.
- **One model family, one dataset, small n.** This is a probe, not a benchmark. Treat effect
  sizes as directional and check the CIs before believing any ranking.
- **The C1 filter selects for questions the model finds hard**, so it can remove one member of a
  page's question pair and the resulting evaluation set need not represent ordinary Wikipedia QA.
- **Gold answers are model-authored, then audited against stored evidence**, rather than drawn
  from a human-labelled benchmark. Review sampled records before treating results as a benchmark.
- **Provider variance.** OpenRouter may route to different backends across runs. Seeds are sent
  but not required (`model.require_parameters` defaults to false); the on-disk cache, not the
  seed, is what makes a rerun exactly reproducible. Pin `model.provider_order` to reduce this.

## If the probe says stop

If the `A_full` − `B_freeform` headroom is under ~3 points and no corruption produces a
distinguishable effect, that is the finding: the handoff is not where the loss is, and this
direction should be abandoned. Report it plainly. Do not tune the setup to manufacture an
effect.

## Repeated-handoff extension

The one-handoff pilot found that free-form compression improved rather than
reduced accuracy. `src/run_chain.py` therefore tests the more general question:
when does repeated compression stop denoising evidence and begin destroying it?

The extension crosses three independent variables:

| Axis | Conditions | Interpretation |
|---|---|---|
| Dataset | Random Wikipedia pages | Two unrelated, source-grounded questions from each retained full page |
| Evidence length | `full` | The complete plaintext page, with no truncation or benchmark distractors |
| Compression depth | 0, 1, 2, 3, 5 | Direct answer or that many consecutive free-form handoffs |

The first compression agent sees the complete selected page.
Every subsequent agent receives only a frozen `SealedHandoff` made from the
preceding summary, so lost details cannot leak back into the chain.

Build the dataset (requires `OPENROUTER_API_KEY`; it is read from the environment only):

```bash
.venv/Scripts/python src/build_wikipedia_dataset.py
```

Run the configured 20-question experiment:

```bash
.venv/Scripts/python src/run_chain.py
```

Narrow runs are composable and resumable:

```bash
.venv/Scripts/python src/run_chain.py --datasets wikipedia_random --contexts full --depths 0,1,3,5
.venv/Scripts/python src/run_chain.py --plot-only
```

Run the comparable-size, low-cost Qwen3 32B replication (same datasets,
contexts, seeds, and depths as the 70B run; outputs remain isolated):

```bash
.venv/Scripts/python src/run_chain.py --config qwen32_chain_config.yaml --chain-config qwen32_chain_experiment.yaml
```

The Qwen configuration uses Qwen3's native `/no_think` directive as well as
OpenRouter's `reasoning.effort: none`, ensuring the fixed handoff and answer
budgets are spent on visible text rather than hidden reasoning.

Configuration lives in `chain_config.yaml`. Outputs are written to:

- `runs/chain/answers.jsonl`: answer-level predictions, accuracy, context size,
  handoff size, and cumulative token accounting;
- `results/chain/stage_metrics.csv`: accuracy and confidence intervals by
  dataset, evidence length, and handoff depth;
- semantic similarity is reported alongside token F1 using BERTScore
  (`roberta-large`, layer 17, English baseline rescaling), taking the best
  score over the gold answer and its aliases; answer-level scores are persisted
  in `runs/chain/answers.jsonl` so replotting does not repeat inference;
- `results/chain/depth0_deltas.csv`: paired degradation relative to the
  no-handoff condition;
- `results/chain/degradation.png`: token-F1 and BERTScore trajectories with 95%
  bootstrap intervals;
- `results/chain/report.md`: human-readable results table.

Offline verification of the generated-dataset adapter and plot generation:

```bash
.venv/Scripts/python src/selftest_chain_offline.py
```

## Incremental-evidence chain

`src/run_incremental_chain.py` extends the sealed chain without changing the
existing experiments. MuSiQue supporting paragraphs arrive one at a time to
specialists; exactly 0/1/3/5 relay-only agents are inserted between successive
specialists. Relays call the existing `recompress()` path and accept only a
`SealedHandoff`, so they cannot receive new evidence.

Run the configured experiment:

```bash
.venv/Scripts/python src/run_incremental_chain.py \
  --config config.yaml \
  --experiment-config incremental_chain_config.yaml
```

Run a small smoke or a selected condition:

```bash
.venv/Scripts/python src/run_incremental_chain.py --n 2 --candidates 10 --relay-depths 0,1 --seeds 1
.venv/Scripts/python src/run_incremental_chain.py --conditions question_conditioned --relay-depths 0,1,3,5
.venv/Scripts/python src/run_incremental_chain.py --analyse-only
```

The experiment writes only to `data/incremental_chain`,
`runs/incremental_chain`, and `results/incremental_chain`. Its records retain
the existing handoff and answer fields and add packet identity, relay depth,
introduction stage, and handoff age. Main outputs are:

- `runs/incremental_chain/handoffs.jsonl`: every specialist and relay message;
- `runs/incremental_chain/answers.jsonl`: original-evidence and final multi-hop answers;
- `runs/incremental_chain/probe_answers.jsonl`: hidden packet-probe answers at every eligible stage;
- `results/incremental_chain/stage_metrics.csv`: final QA by relay depth;
- `results/incremental_chain/probe_metrics.csv`: hidden-probe accuracy by handoff age;
- `results/incremental_chain/future_query_regret.csv`: original-evidence minus final-handoff probe scores;
- `results/incremental_chain/survival_by_age.csv`: exact fact survival by age;
- `results/incremental_chain/incremental_chain.png` and `report.md`.

Offline controls and schema/plot checks:

```bash
.venv/Scripts/python src/selftest_incremental_chain_offline.py
```

## Experiment 5: paraphrase-only arms and per-edge measurement

Two additions to the question-conditioning experiment, both scoped to it.

**Per-edge measurement.** Every run now measures each consecutive handoff edge
`M_i -> M_{i+1}`, not just the accuracy at each depth. Lexical measures are
deterministic and free ([`src/paraphrase_metrics.py`](src/paraphrase_metrics.py)):
token-F1 similarity, novel/retained token rates, verbatim 5-gram copy rate, and
length ratio. Exact fact survival reuses `mentions_answer()` from
`run_slack_facts.py`, the same helper the other experiments use. The semantic
half — did the rewrite preserve the meaning — is an `EQUIVALENT` /
`MINOR_LOSS` / `MAJOR_LOSS` verdict from `add_preservation_judge()` in
[`src/judge.py`](src/judge.py), on the same different-model-family, temperature-0,
cached basis as the answer judge.

Each edge is then bucketed: `benign_paraphrase` (wording changed, meaning held,
fact kept, answer right), `information_loss` (meaning degraded, answer wrong),
`critical_detail_loss` (meaning judged preserved, yet the fact or answer is
gone), `semantic_drift` (meaning degraded but the answer survived anyway),
`verbatim_copy`, and `unresolved`. Every underlying continuous measure is stored
per edge, so the buckets can be recut without rerunning anything.

**Two new arms.** `conditioned` and `generic` both summarise, so neither can
say whether repeated *rewriting* is lossy or whether the loss comes from
compression and relevance filtering. Two arms separate them:

| Arm | Rewriting | Compression | Sees question A |
|---|---|---|---|
| `passthrough` | none (no model call) | none | no |
| `paraphrase` | maximal, explicitly forbidden to condense | none | no |
| `generic` | yes | yes | no |
| `conditioned` | yes | yes | yes |

Read as a ladder: `passthrough → paraphrase` is the cost of rewriting alone,
`paraphrase → generic` adds compression, `generic → conditioned` adds
task-relevance selection.

Run the four-arm comparison (model, evidence, budgets, decoding, depths, seeds,
and every prompt but the condition-specific instruction are matched to the
published gold-only run; outputs go to their own roots):

```bash
.venv/Scripts/python src/run_summary_generalization.py --config summary_generalization_paraphrase_config.yaml
```

Alongside the existing `metrics.csv`, `deltas.csv`, and
`summary_generalization.png`, a run now also writes:

- `runs/<root>/transitions.jsonl`: one row per edge, with every measure, the
  judged verdict, and a hash of the message pair it was judged from;
- `results/<root>/transition_metrics.csv`: the same aggregated by arm and depth;
- `results/<root>/paraphrase_transitions.png`: lexical similarity, semantic
  preservation, held-out fact survival, and answer accuracy against depth.

Transition rows are rebuilt from `handoffs.jsonl` on every run (free and
deterministic); only the judged fields are carried over, and only when the
message pair is byte-identical. The existing per-stage `handoffs.jsonl` rows
gain a compact annotation subset and keep all their original fields.

Offline checks for both additions, including that all four stored Experiment 5
runs still regenerate their published `metrics.csv`/`deltas.csv` unchanged:

```bash
.venv/Scripts/python src/selftest_summary_generalization_offline.py
```

### Experiment 5b: fictional multi-query, fixed-capacity replication

`src/run_fictional_summary_generalization.py` removes three ambiguities in the
pilot A/B result. Its 20 invented dossiers each contain six independently
answerable facts; every fact is rotated through visible Question A and the
other five are evaluated as unannounced queries. The generic selection is made
once per dossier and reused across rotations, and inference is clustered by
dossier rather than treating correlated rotations as independent examples.

Both arms choose exactly K source-grounded evidence cards. The selected cards
are rendered deterministically, so conditioned and generic handoffs have the
same structural capacity and cannot gain space by inventing or merging facts.
The preregistered capacity sweep is K = 2, 4, 6: K=4 is primary, K=2 tests a
tighter bottleneck, and K=6 is the no-selection ceiling. Full-source/card and
closed-book baselines distinguish handoff loss from answerer failure or lucky
parametric guesses.

The configured `n=20` run is complete. At the primary K=4 capacity,
conditioning raises announced-question judged accuracy by **+0.317** [0.292,
0.333] and lowers mean accuracy on the five hidden questions by **−0.062**
[−0.077, −0.048]. The trade-off is larger at K=2 (**+0.608 / −0.147**) and is
exactly zero in the paired-compliant K=6 all-card control. Full-card accuracy is
1.000 and closed-book accuracy is 0.000. See
[`results/fictional_summary_generalization/n20/report.md`](results/fictional_summary_generalization/n20/report.md).

```bash
.venv/Scripts/python src/selftest_fictional_qa_offline.py
.venv/Scripts/python src/selftest_fictional_summary_generalization_offline.py
.venv/Scripts/python src/run_fictional_summary_generalization.py --dry-run
.venv/Scripts/python src/run_fictional_summary_generalization.py --limit 1
.venv/Scripts/python src/run_fictional_summary_generalization.py
```

## Experiment 8: model heterogeneity across sequential handoffs

Every other chain experiment runs one model at every stage. `src/run_model_heterogeneity.py`
varies **who writes each handoff** while holding the task, the evidence, the prompts, the
decoding settings, the token budget and the final answerer fixed. It asks two things:

1. **Matched size, different families.** Does a chain that crosses model families at every
   handoff preserve less than a chain that stays inside one family at the same size?
2. **Family and size together.** Do small→large, large→small and alternating size
   trajectories differ — and can a later larger model recover what an earlier smaller one
   dropped, or does an early small model create a bottleneck the rest of the chain inherits?

### What is held fixed, and why it matters

- **The answerer is constant.** Every arm and every depth is answered by the same model at
  temperature 0, with a second answerer from a different family alongside it. If the last
  chain model also answered, "large models late" would beat "small models late" because a
  large model answered, and nothing about the handoff would have been measured.
- **The prompts never name a model.** Stage-1 and stage ≥2 prompts are imported from
  Experiment 2 / Experiment 5's `conditioned` arm and asserted byte-identical at startup, so
  the decoder is the only thing that changes between arms.
- **Start models are stratified.** Each pair's start family is its index modulo the family
  cycle, so every family starts an equal share of pairs. A cross-family arm is compared
  against `homog_*_matched`: for each pair, the homogeneous chain built from *that pair's*
  start family. The two share a byte-identical stage 1 and diverge only from stage 2.
- **Stage ≥2 is sealed.** Later compressors accept a frozen `SealedHandoff` and nothing else,
  so source passages cannot re-enter a chain.

Because the questions come from the prebuilt SQuAD same-passage pairs rather than a
per-model C1 filter, every arm sees the *same* questions — so unlike the Qwen replications
in Experiment 2a, this is a genuinely paired cross-model comparison.

### The model pool

Open-weight, dense (so parameter counts are unambiguous), non-reasoning, and drawn from one
release window so "family" is not silently "model vintage". Four families at two size tiers:

| Tier | Llama | Qwen | Mistral | Gemma |
|---|---|---|---|---|
| small | `llama-3.1-8b-instruct` (8B) | `qwen3-8b` (8.2B) | `ministral-8b` (8B) | `gemma-3-12b-it` (12.2B) |
| large | `llama-3.3-70b-instruct` (70.6B) | `qwen3-32b` (32.8B) | `mistral-small-3.2-24b` (24B) | `gemma-3-27b-it` (27.4B) |

Parameter counts are vendor-published and recorded with their basis and source URL; a model
with an undisclosed count is stored as unknown and excluded from size contrasts rather than
given an invented figure. Live OpenRouter metadata (price, context window, tokenizer) is
snapshotted per run into `runs/model_heterogeneity/n20/model_catalogue.json`.

### The matrix

Seventeen generated arms plus two derived controls, over depths 0–6:

- **matched small** — four homogeneous families, plus forward and reverse cross-family cycles;
- **matched large** — the same four families, plus a forward cross-family cycle;
- **size trajectory** — `up` / `down` / `alt` × family-held / family-rotating, a 3×2 factorial.
  All three trajectories hold the same tier multiset at depth 6, so comparing them there is an
  ordering comparison and nothing else.

Generation is memoised on the *model prefix*, not the arm: two arms that agree on their first
`s` models share that stage-`s` message exactly rather than approximately, which is what the
start-matched contrast depends on.

Estimate the spend, then run it:

```bash
.venv/Scripts/python src/run_model_heterogeneity.py --dry-run
```

```bash
.venv/Scripts/python src/run_model_heterogeneity.py
```

Narrow runs and re-analysis are composable:

```bash
.venv/Scripts/python src/run_model_heterogeneity.py --arms homog_small_llama,xfam_small_fwd --depths 0,1,2
.venv/Scripts/python src/run_model_heterogeneity.py --analyse-only
```

Outputs go to `results/model_heterogeneity/n20/`:

- `model_heterogeneity.png` — the primary figure: accuracy against handoff depth for every
  major condition, with homogeneous baselines and heterogeneous chains distinguished by line
  weight rather than by legend order;
- `model_trajectories.png` — the auxiliary figure: which model handled every stage of every
  chain, dot colour by family and dot **area** proportional to parameter count, beside that
  chain's depth-6 result on the same y axis;
- `transition_diagnostics.png` — per-edge information loss grouped by what changed at that
  edge (family, size, both, neither);
- `stage_records.csv` — the per-stage audit log: chain id, depth, exact model, family,
  parameter count and its basis, tokens, cost, provider, handoff length, per-edge lexical and
  semantic-preservation measures, fact survival, and the answer scores at that depth;
- `deltas_sensitivity.csv` — every contrast repeated with, per contrast, the pairs its own arms
  routed through an empty handoff removed. A provider can return an empty body with a normal
  finish reason and a non-zero token count; a conclusion that only survives in one of the two
  files is a statement about that failure, not about model composition;
- `transition_deltas.csv` — paired per-pair contrasts between edge types (cross-family vs
  same-family at matched size, and the same comparison when size grows or shrinks);
- `metrics.csv`, `deltas.csv`, `transition_metrics.csv`, `family_transition_metrics.csv`,
  `chain_index.csv`, `model_registry.csv`, `report.md`.

Offline checks — schedules, prompt byte-identity, seal enforcement, prefix memoisation,
area-not-radius marker scaling, undisclosed-size handling, and both figures:

```bash
.venv/Scripts/python src/selftest_model_heterogeneity_offline.py
```

### Experiment 8b: where the size bottleneck occurs

`src/run_fictional_model_bottleneck.py` is the focused causal follow-up to the
broad Experiment 8 matrix. A source-aware selector sees a fictional dossier and
Question A and emits three evidence cards. Only after that packet is sealed is
Question B announced. A downstream relay sees B and the sealed packet—not the
source—and must emit one card. Small and large models are crossed as a 2x2
selector-by-relay design within both Llama and Qwen; one fixed third-family
Mistral model answers every final packet.

The design separates four explanations:

- `small -> large` versus `small -> small` tests downstream capability with an
  identical small-written prefix;
- `large -> large` versus `small -> large` tests the early selector bottleneck;
- a capacity-matched sham swap controls the intervention itself;
- exact B-support restoration and an explicitly unsealed source-reopen arm
  test whether performance returns when the missing evidence becomes available.

The restore and sham arms replace one card rather than appending text, so the
channel width is unchanged. The source-reopen arm is an upper-bound positive
control and is never pooled into the sealed 2x2 factorial.

The configured `n=20` run is complete. The upstream-size contrast reverses
direction across families (**+0.037 Llama; −0.037 Qwen**), so there is no shared
model-size effect. On 98 matched, schema-valid cases where the stage-1 text has
no annotated B answer or alias, neither relay size answers B (196/196 sealed
evaluations score zero). Large-minus-small is 0.000 [0.000, 0.000] in both
families, within the configured ±0.05 equivalence margin.
Same-width exact restoration beats sham by **+0.983** for Llama and **+0.950**
for Qwen; restoration and source reopening each beat sealed by **+1.000**.
Thus the supported result is an evidence-availability bottleneck, not a
monotonic parameter-count claim. See
[`results/fictional_model_bottleneck/n20/report.md`](results/fictional_model_bottleneck/n20/report.md).

```bash
.venv/Scripts/python src/selftest_fictional_model_bottleneck_offline.py
.venv/Scripts/python src/run_fictional_model_bottleneck.py --dry-run
.venv/Scripts/python src/run_fictional_model_bottleneck.py --limit 1
.venv/Scripts/python src/run_fictional_model_bottleneck.py
```

## Experiment 9: handoff size adaptation

Every other experiment fixes what the writer is told and varies the message, the model, or
the depth. `src/run_size_adaptation.py` varies what the writer is told **about its reader**:

1. **Does knowing change anything, or does it take asking?** Each capacity claim appears
   twice — once as a bare statement, once as that same statement plus one clause asking the
   model to act on it. `resize_*` is constructed as `inform_*` plus a clause, and the
   selftest asserts it, so the contrast cannot drift into an incidental rewording.
2. **When a handoff is asked to grow, what fills the space?** Added detail, restated detail,
   or material nobody supplied.
3. **If a fact is compressed out and a later stage is asked to expand, does it come back —
   and from where?** Stage 2+ receives a sealed handoff and nothing else, so a fact absent
   from that handoff and present in its successor was *not copied*. It was reconstructed.

### The baseline is not Experiment 5's prompt

Every other chain experiment here inherits Experiment 5's `conditioned` instruction, which
opens *"Write **concise** prose research notes …"*. That word is itself a shrink
instruction. Using it as the control would have made `resize_large` vs control a comparison
between "expand" and "shrink" rather than between "expand" and "no size instruction", and
would have measured a requested shrink against an already-shrunk baseline.

The base instruction here is therefore the published string with the single word `concise`
deleted and nothing else touched — asserted by reconstructing the published string from the
neutral one, in both the runner's startup check and the offline selftest. A separate
`concise` directive appends "Keep your handoff concise", so an explicit unquantified shrink
cue is measured instead of hiding inside the baseline. It is not an exact restoration of the
deleted word in its original position.

### The directives

All are appended to that neutral instruction, and to nothing else:

| directive | appended text |
|---|---|
| `none` | *(nothing)* |
| `concise` | "Keep your handoff concise." |
| `inform_small` | "The next agent's model has a 2,000-token context window." |
| `resize_small` | the same sentence + "Make your handoff fit within it." |
| `inform_large` | "The next agent's model has a 10,000-token context window." |
| `resize_large` | the same sentence + "Expand your handoff to make use of the available space." |

The advertised size is **not** wired to `decoding.handoff_max_tokens`. The number claimed to
the model and the budget enforced by the API are different variables and are kept that way;
the enforced budget is identical in every arm, and truncation is measured per stage. In the
completed run, the fill-10k repetition mode hit that common 12k safety ceiling in 102 calls;
this is reported as a failure mode rather than mistaken for useful expansion.

### The arms

Nine directive schedules over three handoffs, including the trajectory the design turns on:

| arm | schedule |
|---|---|
| `control` | none → none → none |
| `concise_shrink` | concise → concise → concise |
| `inform_shrink` / `resize_shrink` | the small directive at every stage |
| `inform_expand` / `resize_expand` | the large directive at every stage |
| `resize_lsl` / `inform_lsl` | **large → small → large** |
| `resize_sse` | small → small → large |

Generation is memoised on the **directive prefix**: two arms agreeing on their first *s*
directives share that stage-*s* message exactly rather than approximately, so every contrast
is anchored on a byte-identical common history.

### Two datasets, so new material can be attributed

Noticing that a handoff gained a fact is easy; saying where it came from is the whole
problem. `src/build_size_adaptation_data.py` builds both datasets together with the
closed-book evidence that licenses their interpretation.

- **`counterfactual`** — documents whose answer contradicts the model's memorised one
  (the page says Lisbon, the model believes Rome). Every item is verified **both ways**:
  the model must reliably produce the original closed-book, and must never produce the
  replacement. One set of closed-book samples is judged twice, once per gold set, so the two
  conditions describe the same behaviour rather than two draws that might disagree by luck.
  A handoff that says "Rome" is then parametric reconstruction, not a lucky guess.
- **`fictional`** — invented subjects the model cannot know in either direction, verified
  closed-book. Anything unsupported appearing here is fabrication with nothing to attribute
  it to, which is what separates *reconstruction* from *invention* instead of calling all of
  it hallucination.

The counterfactual set draws on two sources, each tagged with a `source` field so they are
never silently pooled. `rewritten_wikipedia` reuses
`data/wikipedia_random_counterfactual/` — but those pages were sampled at random for an
experiment whose requirement was that the model must **not** know the answer, so most of
them fail this experiment's opposite requirement. `known_entity` samples from the other end:
famous subjects first, verified known, and only then rewritten.

Each item also carries five **side facts** — checkable details no question asks about,
verified verbatim in the document and non-overlapping with each other. They are what
"reusable information" means operationally: a shrink that keeps the answer but discards them
has narrowed the handoff to the current task. One side fact per item is also *answered* at
every depth, so that becomes an accuracy number rather than only a string-presence count.

### What is measured

Deterministic (`src/size_metrics.py`), primary:

- length in words, characters and estimated tokens, and the ratio to the previous stage;
- **internal repetition** — the share of a handoff's 5-grams that repeat one earlier in the
  same handoff, which separates "expanded by adding detail" from "expanded by restating";
- **unsupported terms** — capitalised tokens and numeric literals the source document does
  not contain, split into `corrupted` (a near-miss restatement of a value the experiment is
  tracking: *Taren Vos → Leran Vos*) and `invented` (no counterpart at all). The split is
  made at phrase level against the item's tracked values, not by nearest-token similarity
  over the whole document — that version would report long articles as more "corrupting"
  than short ones purely because they offer more strings to be near;
- per-fact presence at every stage, and the resulting life history: survived, lost,
  or **lost and reappeared**.

LLM-judged (`openai/gpt-4o-mini`, a different family from the system under test), secondary:
answer correctness against both gold sets, semantic preservation across each edge, and a
new **claim-support judge** that reads the source document and counts claims the notes do
not support. It is scored against the source at every depth, not against the previous note,
because no stage is ever licensed to introduce material the document did not contain — and
grading stage 3 against stage 2 would bless a fabrication that entered at stage 2 and was
faithfully copied afterwards.

### Running it

```bash
.venv/Scripts/python src/build_size_adaptation_data.py --which both
```

```bash
.venv/Scripts/python src/run_size_adaptation.py --dry-run
```

```bash
.venv/Scripts/python src/run_size_adaptation.py
```

Narrow runs and re-analysis compose as elsewhere:

```bash
.venv/Scripts/python src/run_size_adaptation.py --arms control,resize_expand --limit 5
```

```bash
.venv/Scripts/python src/run_size_adaptation.py --analyse-only
```

Outputs go to `results/size_adaptation/`:

- `size_adaptation.png` — the primary figure: length, answer accuracy, side-fact retention
  and internal repetition against handoff depth, per dataset. Requested resizes are heavy
  solid lines and their "informed only" twins are the same colour dashed, so *does asking do
  more than telling* is the gap between a solid and a dashed line of one colour;
- `expansion_content.png` — length against what accounts for it (repetition, unsupported
  terms, tokens new since the previous stage), so an arm that doubled its word count sits
  visibly in one of three regimes;
- `expansion_modes.png` — every fill-10k call split by finish state, with log-scaled length
  and within-message repetition so the stopped and budget-running modes remain distinct;
- `headline_contrasts.png` — depth-3 forest plot of the stable non-runaway length and
  side-fact contrasts used in the main report;
- `recovery.png` — loss, recovery, reversion and fabrication per arm, beside where the final
  counterfactual answer came from, so an arm that looks good on recovery but gets there by
  reverting to memorised knowledge cannot be read as a success;
- `metrics.csv`, `contrasts.csv` (paired per-item deltas with bootstrap CIs and p-values),
  `recovery.csv`, `fact_histories.csv` (the per-fact audit trail), `answer_origin.csv`,
  `arm_rows.csv`, `report.md`.

Offline checks — prompt neutrality, `resize == inform + one clause`, seal enforcement,
prefix memoisation, the corruption/invention split, the fact life-history classifier, and
all figures:

```bash
.venv/Scripts/python src/selftest_size_adaptation_offline.py
```

## Experiment 10: communication regret under a hard communication budget

Every conditioning experiment so far (5, 5b, 8b) compared a handoff written for a known
question against one written for no question, and measured what the second could still
answer. None of them held the *channel* fixed. Experiment 5's length-matched control set a
word target in the prompt and a non-binding `max_tokens`; the arms still self-selected very
different lengths, so a held-out accuracy gap remained inseparable from "one arm wrote
more". Experiment 5b fixed capacity structurally (exactly K evidence cards) but gave up
free-text handoffs to do it.

`src/run_communication_regret.py` keeps free-text handoffs and makes the budget itself the
controlled variable. The claim under test:

> Under an equal communication budget, conditioning a handoff on the currently known
> information need reallocates communication capacity toward present utility and can reduce
> its utility for plausible future information needs.

Supporting it requires **both** `dU_now > 0` and `dU_future < 0` against a generic or
reusable baseline, strengthening as the budget shrinks. Anything less is reported as
measured.

### The utility matrix

Each source context independently answers *k* questions. Every question takes a turn as the
conditioning query; each resulting message is then answered against **every** question of
that context, giving a matrix per context:

    M[a][b] = U(m_a, q_b)      m_a = the message written knowing only q_a

`U_now` is the diagonal, `U_future` the mean of the off-diagonal. Because the same question
sits on the diagonal for one rotation and off it for the other *k*-1, question difficulty
cannot produce the contrast, and no fixed A/B split has to be trusted -- which is what the
earlier pair-based designs could not say.

`generic`, `oracle` and `extractive_generic` do not depend on which question is currently
known, so they hold **one** message per (context, budget) that every rotation reuses. On
SQuAD, where every question rotates, their `U_now` and `U_future` are equal by construction.
On relation dossiers only the four anchor questions rotate, while `U_future` also contains
the twelve non-anchor questions, so the two marginals need not be equal. In both corpora the
message itself is rotation-invariant; that, rather than equality of differently composed
marginals, is the relevant unspecialised baseline property.

### The hard budget is two-sided

A cap alone equalises nothing. A cap-only pilot on 3 contexts at a 40-word cap produced
these fill ratios:

| policy | fill (cap only) | fill (cap + floor) |
|---|---|---|
| `conditioned` | 0.61 | 0.92 |
| `reusable` | 0.75 | 0.91 |
| `oracle` | 0.83 | 0.96 |
| `generic` | 0.88 | 0.93 |

The conditioned arm quietly spent a third less channel than the control, so any future-query
deficit it showed would have meant *it wrote less*, not *it allocated differently*.
`src/budget.py` therefore contracts both sides -- "write between *floor* and *cap* words" --
and enforces it in three stages:

1. one length contract, byte-identical for every policy;
2. a message outside the band is returned for a rewrite with an identical correction
   (`SHRINK_CORRECTION` / `EXPAND_CORRECTION`), up to `budget.max_length_corrections` times;
3. whatever survives is truncated to the cap **unconditionally** before it can reach a
   reader, at the last sentence boundary that fits.

The cap is hard and always holds; the floor cannot be (words cannot be invented on demand),
so it is requested, audited and reported -- `under_floor_rate` per arm, never assumed.
Words, not tokens, are the enforced unit: a word count is deterministic, tokenizer-free and
equal for every model. The nominal token budget, the requested `max_tokens`, the model's own
completion-token count and a token estimate of the delivered text are all recorded, so the
token view is never lost.

Because an expansion request is the failure mode Experiment 9 found -- a fill-the-space cue
answered by restating the same content -- `internal_repetition_rate` is measured on every
delivered message and reported beside its length.

Three further length controls back this up, because the audit is not allowed to be the only
evidence:

- **paired length deltas** between every compared pair of arms, bootstrapped over contexts;
- **a length-matched subsample** -- only contexts where all compared arms delivered within
  `budget.match_tolerance_words` of each other;
- **a trimmed arm** (`policy@trim`) -- every abstractive message in a cell re-truncated to
  that cell's shortest delivered length and re-answered. This costs answer calls but no
  generation calls, and is the strictest statement available: identical length, by
  construction, per context.

### The policies

Only the conditioning block differs among the four abstractive arms; the offline selftest
asserts that removing it leaves one identical prompt skeleton.

| policy | what the sender is told | messages per context and budget |
|---|---|---|
| `generic` | a question will be asked; it has not been told which | 1 |
| `conditioned` | the current question, and to communicate what answers it | *k* |
| `reusable` | the current question, plus that unknown further questions may follow, and to preserve reusable evidence within the same limit | *k* |
| `oracle` | every question the context supports -- an upper bound, not a deployable strategy | 1 |
| `extractive_generic` / `extractive_conditioned` | the same two conditioning levels, but sentences copied verbatim instead of prose | 1 / *k* |

`generic` deliberately says only that it has not been told the question. Telling it that
further questions may follow *is* the `reusable` treatment; putting the treatment in the
control would destroy the contrast the control exists to provide. The extractive pair
separates information-selection loss from abstractive rewriting loss and is the one place
the shared output form is replaced on purpose.

### Regret, and what it is measured against

The answerer also reads the source directly for every question. That is the no-handoff
ceiling and the reference value in the regret difference:

    R(q) = U(D, q) - U(m, q)          R_future = E[ R(q) | q != conditioning question ]

Normalised retained utility is reported in the form that has no pathological denominator:
restricted to cells the source itself answers, `U(D, q) = 1`, so retained utility is just
`U(m, q)` and regret is `1 - U(m, q)`. The raw ratio is available in `rotation_rows.csv`
with the ceiling stored per cell, and contexts whose ceiling falls below
`analysis.min_ceiling_for_ratio` are counted rather than dropped silently.

### Two corpora

| corpus | what it is | why |
|---|---|---|
| `squad_groups` | 24 SQuAD paragraphs, each with 4 human-written questions | natural, human-written questions; nothing was authored to make the effect appear |
| `relation_dossiers` | invented dossiers with 4 aspects x {anchor, paraphrase, same-entity, same-topic} | the *distance* between conditioning and hidden query is a designed variable, not an embedding estimate |

Both are built by `src/build_regret_data.py`. A SQuAD group is kept only if **all four**
questions survive the project's closed-book leakage filter against the answering model (an
off-diagonal success from parametric memory would look exactly like a reusable handoff) and
the group passes an independence audit by a different model family: every question must be
answerable from the paragraph and must ask about a fact distinct from all the others.

The relation corpus exists because SQuAD supports no clean distance labelling and forcing
noisy categories onto it would be worse than not measuring. Each dossier is written to a
strict schema, every designed answer is verified verbatim in the text, and the structure is
asserted before use. For a conditioning question about aspect *a*, the hidden questions sit
at four known distances: `paraphrase` (same answer, different wording), `same_entity` (a
second fact about the same subject in the same aspect), `same_topic` (a different named
entity in the same aspect) and `orthogonal` (any anchor in a different aspect).

### Running it

```bash
.venv/Scripts/python src/build_regret_data.py --which both
```

```bash
.venv/Scripts/python src/run_communication_regret.py --dry-run
```

```bash
.venv/Scripts/python src/run_communication_regret.py
```

```bash
.venv/Scripts/python src/run_communication_regret.py --dataset relation_dossiers --policies generic,conditioned,reusable,oracle --no-trim
```

Narrow runs and re-analysis compose as elsewhere (`--limit`, `--budgets`, `--policies`,
`--analyse-only`). Outputs go to `results/communication_regret/<corpus>/n<N>/`:

- `pareto_now_vs_future.png` -- one fixed-scale panel per requested budget, with every policy
  shown as a discrete (`U_now`, `U_future`) marker, within-budget nondominated observations
  outlined, dominated observations muted, and no interpolation between categorical policies;
- `utility_vs_budget.png` -- `U_now`, `U_future` and their gap against budget, with the
  direct-context ceiling;
- `communication_regret.png` -- `R_now` and `R_future` per policy and budget with CIs;
- `relation_distance.png` -- future-query regret by designed distance (relation corpus only);
- `REPORT.md` (at `results/communication_regret/`) -- the cross-corpus narrative report,
  regenerated from the CSVs by `src/render_regret_summary.py` so it cannot drift from them;
  it carries every headline table for both corpora plus an exhaustive appendix of all 3,284
  computed cells;
- `utility_matrix.csv` (every M[a][b] cell with its relation label and ceiling),
  `rotation_rows.csv`, `metrics.csv`, `contrasts.csv` (paired context-clustered bootstrap
  deltas with p-values and Cohen's *dz*), `budget_interaction.csv`, `length_audit.csv`,
  `length_deltas.csv`, `length_matched_contrasts.csv`, `pareto.csv`, `baselines.csv`,
  `relation_regret.csv`, `report.md`.

### What it found

Both corpora support the hypothesis, and the SQuAD run does so at every budget
where the generic baseline has room to lose anything.

| `conditioned` - `generic`, LLM-judge accuracy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| SQuAD, dU_now | +0.625 | +0.448 | +0.281 | +0.104 |
| SQuAD, dU_future | -0.063 | -0.115 | -0.194 | -0.170 |
| Dossiers, dU_now | +0.859 | +0.641 | +0.516 | +0.359 |
| Dossiers, dU_future | -0.026 | -0.110 | -0.198 | -0.342 |

Every cell above excludes zero except SQuAD dU_now at 160 words (p = 0.066) and
dossier dU_future at 20 words (p = 0.055). Exact match and token F1 agree with
the judge. The specialisation gap widens as the channel
narrows -- +0.413 [0.326, 0.503] on SQuAD, +0.184 [0.080, 0.285] on the
dossiers, between the 160-word and 20-word budgets.

Three results decide the interpretation:

- **The channel was not the constraint.** `oracle`, held to the same band,
  scores 0.823 on *every* SQuAD question at 40 words -- matching `conditioned`'s
  present-query score with the same ~37 words while giving up nothing on the
  other three. What a conditioned handoff drops is not what the budget could not
  hold.
- **It is selection, not rewriting.** The verbatim-extraction arms reproduce the
  pattern (+0.385 / -0.198 at 20 words), and no abstractive rewriting occurs in
  them.
- **Asking for reusability barely helps.** `reusable` still loses future utility
  against `generic` at every budget, and is indistinguishable from
  `conditioned` on `U_future` at the two tightest ones.

On the relation corpus, future-query regret rises monotonically with the
designed distance from the conditioning query -- 0.00 at a paraphrase, 0.42-0.95
at same-entity and same-topic, 0.95-0.98 at an orthogonal aspect -- and the
gradient appears only in the conditioned arms. Widening the budget to a third of
the source flattens it for `generic` and `oracle` but not for `conditioned`.

Full numbers, controls and caveats:
[results/HANDOFF_EXPERIMENTS_REPORT.md](results/HANDOFF_EXPERIMENTS_REPORT.md)
section 10.

Offline checks -- prompt-skeleton equality, cap enforcement, floor corrections, rotation
balance, seal enforcement, single-flighting, matrix and regret arithmetic against an
injected effect, trimming, Pareto domination, relation labelling, and the SQuAD builder
against the live parquet schema:

```bash
.venv/Scripts/python src/selftest_communication_regret_offline.py
```

## Experiment 12: anticipatory context management under uncertain future demand

Experiments 10 and 11 both assume one irreversible message. A real agent
harness is not in that position: it decides what to keep in context, what to
compress, what to push to storage, what to leave a pointer to, and what to
drop — and it can pull something back later. It also does not know the future
question. It has, at best, an *estimate* of where demand will land.

This experiment separates the true future-use distribution `P` from the
harness's estimate `P̂` and asks what the gap between them costs:

```
max_pi  U_now + lambda * E_{Q ~ P-hat}[U_future] - beta * C_context - gamma * C_retrieval
```

### Future demand is modelled over evidence units, not questions

Predicting an exact future question string is neither realistic nor necessary.
The relation dossiers are built on a crossed 4-aspect x 4-role grid, and the
aspect axis turns out to partition the evidence exactly: across the 16
dossiers, all 256 questions have their gold located verbatim in some source
sentence, and of the 264 sentences, 191 are claimed by exactly one aspect and
**none by two** (mean pairwise Jaccard 0.000). The remaining 73 are background
prose that answers nothing — the corpus's own distractors.

So `P(Q_future)` is a distribution over four aspects, and predicting "the next
question is about finance" names a real, disjoint subset of the source.
`src/anticipatory_data.py` extracts and labels those units and refuses any
dossier where the partition would be leaky, rather than resolving the tie.

### Two noise axes, kept separate

`P̂` is perturbed away from `P` in two orthogonal ways, because "uncertain" and
"confidently wrong" are different failure modes:

- **dilution** `P̂ = (1-tau)P + tau*U` — higher entropy, no bias;
- **displacement** `P̂ = (1-sigma)P + sigma*P_wrong` — bias at roughly constant
  entropy, where `P_wrong` is the least likely aspect.

Both are linear interpolations, so total variation is *exactly* linear in the
knob. That is a regression check, not an aspiration: it means any threshold
later seen in regret-vs-mismatch belongs to the policy and cannot be an
artefact of the noise parameterisation. It also forces `P` to be peaked —
`TV(P, dilute(P,tau)) = tau * TV(P,U)`, so a flat truth would leave the
dilution sweep no range at all.

### Six actions, and the one that matters

`keep` (verbatim, costs its words), `compress` (shorter, lossy — used exactly
at the budget boundary, where a harness would use it), `externalize` (in the
store, not in context), `pointer` (a short index line in the message, unit in
the store), `retrieve` (BM25 at answer time), `discard` (gone).

`pointer` versus `externalize` is the informative contrast. Both defer the
evidence; only a pointer spends context saying *what* was deferred. If
retrieval beats preservation, the pointer arm separates "recall is cheap" from
"recall is cheap because the writer left a map".

Retrieval is the project's own dependency-free Okapi BM25 (`src/retrieval.py`,
already used by Experiments 3 and 4), not a simulated oracle.

### The isolation guarantee is restated, not relaxed

`orchestrator_answer` still accepts only a four-string `SealedHandoff`.
Experiment 12 adds a separate frozen `SealedContext` carrying the message plus
whatever retrieval returned, so every earlier experiment keeps exactly the
guarantee it was verified under.

The subtle failure mode is that a retrieval arm could become direct context
wearing a different label. The first version of the guard checked whether the
store held the whole source — and that was the wrong question, because the
reader never sees the store. It sees the message plus at most `k` units it had
to earn through BM25. What must be bounded is the worst-case context assembled
for **one answer**: delivered units plus `k`. `assert_bounded_recall` enforces
that, and the offline selftest covers both directions.

### Running it

```bash
.venv/Scripts/python src/run_anticipatory_context.py --dry-run
.venv/Scripts/python src/run_anticipatory_context.py --budgets 40,80
.venv/Scripts/python src/render_anticipatory_report.py
```

Offline checks, no API key:

```bash
.venv/Scripts/python src/selftest_anticipatory_offline.py
```
