# Agent Handoff Information-Loss Probe

When a subagent gathers evidence and hands a summary to an orchestrator, the subagent's own
context is thrown away. This project asks: **once the evidence has been found, how much answer
accuracy is lost in the handoff itself, and which kinds of loss cause the damage?**

Retrieval is taken out of the picture by design: the subagent is *given* the gold evidence.
Any drop in accuracy therefore comes from the handoff.

---

## Status

Stages 0, 1, 2 and 4 are built and verified, with the mechanisms `A_full`, `B_freeform`,
`C_structured`, `D_extractive` and `E_oracle`.

**Stage 3 (injections) is intentionally not built yet.** The plan is to measure the gap between
`A_full`, `B_freeform` and `E_oracle` at n=10 first. If there is no headroom, the design needs
rethinking before anything more is spent on it. `src/inject.py` does not exist yet.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

Set your key. It is read only from the environment, never from a file or a command-line
argument:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Estimate the cost before spending anything:

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

Run Experiment 6 on the corrected 10-example SQuAD sample. Each input contains one shared gold
passage and no distractors, and cached calls are reused automatically:

```bash
python src/run_multilingual_handoffs.py --config multilingual_handoff_config.yaml --n 10
```

Offline checks that need no API key (data shaping, scoring and the isolation guarantee):

```bash
python src/selftest_offline.py
```

Every LLM call is cached under `cache/` by a hash of its content, so rerunning any stage costs
nothing. Reanalysis is free on purpose, because the analysis is meant to be iterated on.

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

[PROMPTS.md](PROMPTS.md) has the exact text of every prompt used above, and
[results/HANDOFF_EXPERIMENTS_REPORT.md](results/HANDOFF_EXPERIMENTS_REPORT.md) brings together the
findings from all the experiments.

## Conditions

| ID | Handoff mechanism | Purpose |
|----|-------------------|---------|
| `A_full` | No handoff: a single agent answers directly from the paragraphs | Ceiling |
| `B_freeform` | The subagent writes a free-form prose summary | The realistic default |
| `C_structured` | The subagent emits JSON claims with `source_para_id`, `constraints[]` and `confidence`, plus `unresolved[]` | A candidate improvement, and the format that corruptions will be injected into |
| `D_extractive` | The subagent returns only verbatim sentences, with paragraph ids | Tests whether generating new text is itself where the loss comes from |
| `E_oracle` | The gold supporting sentences, verbatim with ids, and no LLM | Upper bound for any handoff |

`A_full` − `B_freeform` is the headroom. `E_oracle` − `B_freeform` is how much a perfect handoff
would recover. `E_oracle` − `A_full` is interesting in its own right: if the oracle beats full
context, compression is *helping* by removing noise.

## How the hard constraints are implemented

**C1: parametric leakage filter.** Every candidate question is answered closed-book, with no
evidence, three times at temperature 0.7. The question is dropped if *any* attempt is correct,
where correct means exact match **or** token F1 ≥ `leakage_filter.f1_known_threshold` (0.6). The
F1 condition is stricter than exact match alone, because a near-miss paraphrase still shows that
the model knows the answer. The pass rate is printed and saved to
`results/c1_filter_summary.json`.

**C2 / C6: same model, and how the temperature conflict is resolved.** The subagent and the
orchestrator are the same model. They share one system prompt (`A_full` and the orchestrator
both use `ANSWER_SYSTEM` verbatim) and the same `top_p` and `max_tokens`. They differ in exactly
one setting, because C6 requires it: the orchestrator answers at temperature 0 while the
subagent summarises at 0.7, so variation between seeds comes from the summary rather than from
sampling the answer. Neither side is more capable than the other.

As a consequence, `A_full` and `E_oracle` involve no sampling at all. They are deterministic and
are generated **once** per question rather than once per seed. Only `B_freeform`,
`C_structured` and `D_extractive` run under both seeds.

**C3: injections operate on structured data.** `C_structured` output is parsed into a normalised
object. `render_structured()` is the *only* way to turn that object into text the orchestrator
sees, so a corrupted object and a clean one are rendered by the same code. No LLM is used to
corrupt anything. (Stage 3 is not built yet, but this groundwork is in place and tested.)

**C4: token accounting.** Every call records prompt tokens, completion tokens, latency, cost, and
whether it hit the cache. `results/summary.csv` reports handoff tokens, answer tokens and their
total for each condition next to accuracy, so a condition cannot win just by spending more.

**C5: cache everything.** `hash(model, messages, temperature, top_p, max_tokens, seed,
response_format)` maps to one JSON file under `cache/`. Nothing specific to the environment goes
into the hash.

**C7: deterministic primary metrics plus an independent semantic check.** Answers are normalised
SQuAD-style (lowercase, strip punctuation, drop articles, collapse whitespace). Exact match and
token F1 remain the primary metrics, taking the best score over the gold answer and its aliases.
`summary.csv` has an `em_f1_disagree` column, so if EM and F1 tell clearly different stories that
is visible instead of being averaged away. A temperature-0 `openai/gpt-4o-mini` judge from a
different model family is reported as a secondary correct/incorrect measure, because repeated
rewriting can produce a correct paraphrase that shares few tokens with the gold answer. The judge
adds to EM/F1 rather than replacing them, and every verdict can be audited.

**Stage 2 isolation, enforced in code.** `orchestrator_answer()` accepts only a frozen
`SealedHandoff` holding four strings: `qid`, `question`, `handoff_text` and `mechanism`. The
`Question` object, and with it every paragraph, cannot be reached from that call. Passing a
`Question`, a dict or a plain string raises `TypeError`. `run.py` runs
`selftest_orchestrator_isolation()` at the start of every invocation, so the guarantee is tested
every time rather than just asserted. `A_full` is the one condition where paragraphs are allowed
to reach an answering call, and it uses a separate function that never touches the sealed path.

## Data

The active dataset is a small collection of **10 random English Wikipedia pages, each pinned to a
revision, with exactly two unrelated questions per page**. `src/build_wikipedia_dataset.py` saves
the full plain text of each page with its URL and revision metadata, and writes 20 `Question`
records to `data/wikipedia_random/questions.jsonl`. Once that file exists, the runner no longer
needs MuSiQue or HotpotQA.

The builder rejects pages outside the configured range of 4,000–40,000 characters rather than
truncating them. A strong reasoning model (`openai/gpt-5.2`, high reasoning effort by default)
first writes two questions grounded in the page, with verbatim evidence excerpts. A second,
independent reasoning pass writes the canonical gold answer and any genuine aliases. The exact
page text is kept in `data/wikipedia_random/source_pages.jsonl`, so the construction can be
audited and later experiments do not depend on a live page that may change.

**Retrieval is not modelled.** The subagent is handed the evidence. This is intentional: it is
what lets a measured drop be attributed to the handoff rather than to search quality.

The ids of the questions that survive filtering are written to `data/sampled_ids.json` and the
full records to `data/filtered_questions.jsonl`. Both are committed so runs can be reproduced.

The builder also makes `E_oracle` auditable without an external annotation set: it keeps the
verbatim excerpts written during question construction and maps them back to sentences in the
page.

## Model choice

`meta-llama/llama-3.3-70b-instruct`, chosen to be mid-capability rather than frontier. A frontier
model would hit the ceiling on `A_full`, leaving no headroom to measure, and would answer from
memory more often despite the C1 filter. It is also not a reasoning model, which keeps token
accounting easy to interpret.

The model identifiers were checked against OpenRouter's live `/api/v1/models` endpoint on
2026-08-11 rather than assumed. Nothing in the pipeline is tied to one model: change `model.id` in
`config.yaml` and rerun for a second-model robustness check. The C1 filter is specific to the
model, so a new model needs its own `stage0`. `stage0` notices when
`data/filtered_questions.jsonl` no longer matches its recorded manifest and regenerates the file
automatically.

## Cost control

- `config.yaml` sets a hard cap (`cost.cap_usd`). When the cap is reached the run stops with a
  clear message, and cached work is kept so nothing is paid for twice.
- `--dry-run` estimates the number of calls and the cost, broken down by call type, without
  spending anything. It never writes to `data/`, `runs/` or `results/`, so a dry run cannot
  overwrite a real one.
- Concurrency is configurable and defaults to 4.
- Measured estimates: the 10-question pilot takes about 180 calls (~$0.02), and the full
  150-question run across all five mechanisms about 3,600 calls (~$0.42–0.58).

## Limitations

- **Retrieval is bypassed.** The subagent is handed the gold evidence, so these results bound
  handoff loss only under perfect retrieval. In real systems handoff loss adds to search
  failures; this probe measures only the handoff, on purpose.
- **Short-answer QA, not long research reports.** Real multi-agent research compresses much more
  than summarising 20 paragraphs into a few hundred tokens, and its failure modes may differ.
- **One model family, one dataset, small n.** This is a probe, not a benchmark. Treat effect
  sizes as directional, and check the confidence intervals before trusting any ranking.
- **The C1 filter keeps the questions the model finds hard.** It can remove one of a page's two
  questions, so the evaluation set is not representative of ordinary Wikipedia QA.
- **Gold answers are written by a model and then checked against the stored evidence**, not taken
  from a human-labelled benchmark. Review a sample of records before treating the results as a
  benchmark.
- **Provider variance.** OpenRouter may route requests to different backends across runs. Seeds
  are sent but not required (`model.require_parameters` defaults to false). What makes a rerun
  exactly reproducible is the on-disk cache, not the seed. Setting `model.provider_order` reduces
  this variance.

## If the probe says stop

If the `A_full` − `B_freeform` headroom is under about 3 points and no corruption has a
distinguishable effect, that is the finding: the handoff is not where the loss happens, and this
direction should be dropped. Report that plainly, and do not tune the setup until an effect
appears.

## Repeated-handoff extension

The single-handoff pilot found that free-form compression raised accuracy instead of lowering it.
`src/run_chain.py` therefore tests a more general question: when does repeated compression stop
removing noise and start destroying evidence?

The extension crosses three variables:

| Axis | Conditions | Interpretation |
|---|---|---|
| Dataset | Random Wikipedia pages | Two unrelated questions grounded in each full page that was kept |
| Evidence length | `full` | The complete plain-text page, with no truncation and no benchmark distractors |
| Compression depth | 0, 1, 2, 3, 5 | Answer directly, or after that many free-form handoffs in a row |

The first compression agent sees the complete page. Every later agent receives only a frozen
`SealedHandoff` built from the previous summary, so details that were lost cannot leak back into
the chain.

Build the dataset (this needs `OPENROUTER_API_KEY`, which is read from the environment only):

```bash
.venv/Scripts/python src/build_wikipedia_dataset.py
```

Run the configured 20-question experiment:

```bash
.venv/Scripts/python src/run_chain.py
```

Smaller runs can be combined and resumed:

```bash
.venv/Scripts/python src/run_chain.py --datasets wikipedia_random --contexts full --depths 0,1,3,5
.venv/Scripts/python src/run_chain.py --plot-only
```

Run the Qwen3 32B replication, which is similar in size and cheap to run. It uses the same
datasets, contexts, seeds and depths as the 70B run, and keeps its outputs separate:

```bash
.venv/Scripts/python src/run_chain.py --config qwen32_chain_config.yaml --chain-config qwen32_chain_experiment.yaml
```

The Qwen configuration uses both Qwen3's own `/no_think` directive and OpenRouter's
`reasoning.effort: none`, so the fixed handoff and answer budgets are spent on visible text
rather than on hidden reasoning.

The configuration is in `chain_config.yaml`. Outputs are written to:

- `runs/chain/answers.jsonl`: the prediction, accuracy, context size, handoff size and cumulative
  token counts for each answer;
- `results/chain/stage_metrics.csv`: accuracy with confidence intervals by dataset, evidence
  length and handoff depth;
- semantic similarity, reported next to token F1 using BERTScore (`roberta-large`, layer 17, with
  English baseline rescaling) and taking the best score over the gold answer and its aliases.
  Scores for each answer are saved in `runs/chain/answers.jsonl`, so replotting does not rerun
  inference;
- `results/chain/depth0_deltas.csv`: paired degradation relative to the no-handoff condition;
- `results/chain/degradation.png`: token-F1 and BERTScore trajectories with 95% bootstrap
  intervals;
- `results/chain/report.md`: a readable results table.

Offline check of the generated-dataset adapter and of plot generation:

```bash
.venv/Scripts/python src/selftest_chain_offline.py
```

## Incremental-evidence chain

`src/run_incremental_chain.py` extends the sealed chain without changing the existing
experiments. MuSiQue supporting paragraphs reach specialists one at a time, and exactly 0, 1, 3
or 5 relay-only agents sit between consecutive specialists. Relays use the existing
`recompress()` path and accept only a `SealedHandoff`, so they cannot receive new evidence.

Run the configured experiment:

```bash
.venv/Scripts/python src/run_incremental_chain.py \
  --config config.yaml \
  --experiment-config incremental_chain_config.yaml
```

Run a small smoke test or a selected condition:

```bash
.venv/Scripts/python src/run_incremental_chain.py --n 2 --candidates 10 --relay-depths 0,1 --seeds 1
.venv/Scripts/python src/run_incremental_chain.py --conditions question_conditioned --relay-depths 0,1,3,5
.venv/Scripts/python src/run_incremental_chain.py --analyse-only
```

The experiment writes only to `data/incremental_chain`, `runs/incremental_chain` and
`results/incremental_chain`. Its records keep the existing handoff and answer fields and add the
packet identity, relay depth, the stage at which each packet was introduced, and handoff age. The
main outputs are:

- `runs/incremental_chain/handoffs.jsonl`: every specialist and relay message;
- `runs/incremental_chain/answers.jsonl`: answers from the original evidence, and final multi-hop
  answers;
- `runs/incremental_chain/probe_answers.jsonl`: answers to hidden packet probes at every stage
  where they apply;
- `results/incremental_chain/stage_metrics.csv`: final QA accuracy by relay depth;
- `results/incremental_chain/probe_metrics.csv`: hidden-probe accuracy by handoff age;
- `results/incremental_chain/future_query_regret.csv`: the probe score from the original evidence
  minus the probe score from the final handoff;
- `results/incremental_chain/survival_by_age.csv`: exact fact survival by age;
- `results/incremental_chain/incremental_chain.png` and `report.md`.

Offline controls, and schema and plot checks:

```bash
.venv/Scripts/python src/selftest_incremental_chain_offline.py
```

## Experiment 5: paraphrase-only arms and per-edge measurement

Two additions to the question-conditioning experiment, both limited to that experiment.

**Measuring each edge.** Every run now measures each consecutive handoff edge `M_i -> M_{i+1}`,
not just the accuracy at each depth. The lexical measures are deterministic and free
([`src/paraphrase_metrics.py`](src/paraphrase_metrics.py)): token-F1 similarity, the rates of new
and retained tokens, the verbatim 5-gram copy rate, and the length ratio. Exact fact survival uses
`mentions_answer()` from `run_slack_facts.py`, the same helper the other experiments use. The
semantic side, whether the rewrite kept the meaning, is an `EQUIVALENT` / `MINOR_LOSS` /
`MAJOR_LOSS` verdict from `add_preservation_judge()` in [`src/judge.py`](src/judge.py). It uses the
same setup as the answer judge: a different model family, temperature 0, and caching.

Each edge is then put into one bucket: `benign_paraphrase` (wording changed, meaning kept, fact
kept, answer right), `information_loss` (meaning degraded, answer wrong), `critical_detail_loss`
(meaning judged preserved, but the fact or the answer is gone), `semantic_drift` (meaning degraded
but the answer still right), `verbatim_copy` or `unresolved`. Every underlying continuous measure
is stored for each edge, so the buckets can be redrawn without rerunning anything.

**Two new arms.** `conditioned` and `generic` both summarise, so neither can tell whether the loss
comes from repeated *rewriting* or from compression and relevance filtering. Two more arms separate
these:

| Arm | Rewriting | Compression | Sees question A |
|---|---|---|---|
| `passthrough` | none (no model call) | none | no |
| `paraphrase` | as much as possible, and told not to condense | none | no |
| `generic` | yes | yes | no |
| `conditioned` | yes | yes | yes |

Read the arms as a ladder: `passthrough → paraphrase` is the cost of rewriting alone,
`paraphrase → generic` adds compression, and `generic → conditioned` adds selecting what is
relevant to the task.

Run the four-arm comparison. The model, evidence, budgets, decoding, depths, seeds and every prompt
except the condition-specific instruction match the published gold-only run, and the outputs go to
their own directories:

```bash
.venv/Scripts/python src/run_summary_generalization.py --config summary_generalization_paraphrase_config.yaml
```

In addition to the existing `metrics.csv`, `deltas.csv` and `summary_generalization.png`, a run now
also writes:

- `runs/<root>/transitions.jsonl`: one row per edge, with every measure, the judge's verdict, and a
  hash of the message pair that was judged;
- `results/<root>/transition_metrics.csv`: the same measures aggregated by arm and depth;
- `results/<root>/paraphrase_transitions.png`: lexical similarity, semantic preservation, survival
  of the held-out fact, and answer accuracy against depth.

Transition rows are rebuilt from `handoffs.jsonl` on every run, which is free and deterministic.
Only the judged fields are carried over, and only when the message pair is byte-for-byte
identical. The existing per-stage rows in `handoffs.jsonl` gain a small set of annotations and
keep all their original fields.

Offline checks for both additions, including that all four stored Experiment 5 runs still
reproduce their published `metrics.csv` and `deltas.csv` exactly:

```bash
.venv/Scripts/python src/selftest_summary_generalization_offline.py
```

### Experiment 5b: fictional multi-query, fixed-capacity replication

`src/run_fictional_summary_generalization.py` removes three ambiguities in the pilot A/B result.
Each of its 20 invented dossiers contains six facts that can be answered independently. Every fact
takes a turn as the visible Question A, while the other five are evaluated as unannounced
questions. The generic selection is made once per dossier and reused across rotations, and
inference is clustered by dossier instead of treating correlated rotations as independent examples.

Both arms choose exactly K evidence cards grounded in the source. The chosen cards are rendered by
code, so conditioned and generic handoffs have the same structural capacity, and neither can gain
space by inventing or merging facts. The capacity sweep was fixed in advance at K = 2, 4, 6: K=4 is
the primary setting, K=2 tests a tighter bottleneck, and K=6 is the ceiling where nothing has to be
left out. Baselines that see the full source or all the cards, and closed-book baselines, separate
handoff loss from answerer failure and from lucky guesses based on memory.

The configured `n=20` run is complete. At the primary capacity of K=4, conditioning raises judged
accuracy on the announced question by **+0.317** [0.292, 0.333] and lowers mean accuracy on the
five hidden questions by **−0.062** [−0.077, −0.048]. The trade-off is larger at K=2
(**+0.608 / −0.147**), and exactly zero in the K=6 all-card control on the pairs where both arms
complied. Full-card accuracy is 1.000 and closed-book accuracy is 0.000. See
[`results/fictional_summary_generalization/n20/report.md`](results/fictional_summary_generalization/n20/report.md).

```bash
.venv/Scripts/python src/selftest_fictional_qa_offline.py
.venv/Scripts/python src/selftest_fictional_summary_generalization_offline.py
.venv/Scripts/python src/run_fictional_summary_generalization.py --dry-run
.venv/Scripts/python src/run_fictional_summary_generalization.py --limit 1
.venv/Scripts/python src/run_fictional_summary_generalization.py
```

## Experiment 8: model heterogeneity across sequential handoffs

Every other chain experiment uses one model at every stage. `src/run_model_heterogeneity.py`
varies **which model writes each handoff**, while the task, evidence, prompts, decoding settings,
token budget and final answerer stay fixed. It asks two questions:

1. **Same size, different families.** Does a chain that switches model family at every handoff
   preserve less than a chain that stays within one family at the same size?
2. **Family and size together.** Do small→large, large→small and alternating size orders differ?
   Can a later, larger model recover what an earlier, smaller one dropped, or does an early small
   model create a bottleneck that the rest of the chain inherits?

### What is held fixed, and why it matters

- **The answerer never changes.** Every arm and every depth is answered by the same model at
  temperature 0, with a second answerer from a different family alongside it. If the last model in
  the chain also answered, "large models late" would beat "small models late" simply because a
  large model answered, and nothing about the handoff would have been measured.
- **The prompts never name a model.** The stage-1 and stage ≥2 prompts are imported from Experiment
  2 and from Experiment 5's `conditioned` arm, and a startup check confirms they are unchanged, so
  the only thing that differs between arms is the model doing the decoding.
- **Starting models are balanced.** Each pair's starting family is its index modulo the number of
  families, so every family starts the same share of pairs. A cross-family arm is compared with
  `homog_*_matched`: for each pair, the single-family chain that starts from *that pair's* starting
  family. The two share an identical stage 1 and only diverge from stage 2.
- **Stages 2 and later are sealed.** Later compressors accept a frozen `SealedHandoff` and nothing
  else, so source passages cannot come back into a chain.

The questions come from the prebuilt SQuAD same-passage pairs rather than from a C1 filter run
separately for each model, so every arm sees the *same* questions. Unlike the Qwen replications in
Experiment 2a, this is therefore a genuinely paired comparison across models.

### The model pool

The models are open-weight, dense (so parameter counts are unambiguous), not reasoning models, and
from the same release window, so that "family" does not quietly mean "model generation". Four
families at two size tiers:

| Tier | Llama | Qwen | Mistral | Gemma |
|---|---|---|---|---|
| small | `llama-3.1-8b-instruct` (8B) | `qwen3-8b` (8.2B) | `ministral-8b` (8B) | `gemma-3-12b-it` (12.2B) |
| large | `llama-3.3-70b-instruct` (70.6B) | `qwen3-32b` (32.8B) | `mistral-small-3.2-24b` (24B) | `gemma-3-27b-it` (27.4B) |

Parameter counts are the figures vendors publish, recorded with their basis and source URL. A
model whose size is not disclosed is stored as unknown and left out of size comparisons, rather
than being given a made-up number. Live OpenRouter metadata (price, context window, tokenizer) is
saved for each run in `runs/model_heterogeneity/n20/model_catalogue.json`.

### The matrix

Seventeen generated arms plus two derived controls, over depths 0–6:

- **matched small:** four single-family chains, plus forward and reverse cross-family cycles;
- **matched large:** the same four families, plus a forward cross-family cycle;
- **size trajectory:** `up` / `down` / `alt` × family held fixed or family rotating, a 3×2
  factorial. All three trajectories contain the same mix of tiers by depth 6, so comparing them
  there compares the order and nothing else.

Generation is memoised on the *sequence of models so far*, not on the arm. Two arms that use the
same first `s` models share exactly the same stage-`s` message, not just a similar one, and the
start-matched comparison depends on that.

Estimate the cost, then run it:

```bash
.venv/Scripts/python src/run_model_heterogeneity.py --dry-run
```

```bash
.venv/Scripts/python src/run_model_heterogeneity.py
```

Smaller runs and reanalysis can be combined:

```bash
.venv/Scripts/python src/run_model_heterogeneity.py --arms homog_small_llama,xfam_small_fwd --depths 0,1,2
.venv/Scripts/python src/run_model_heterogeneity.py --analyse-only
```

Outputs go to `results/model_heterogeneity/n20/`:

- `model_heterogeneity.png`: the main figure. Accuracy against handoff depth for every major
  condition, with single-family baselines and mixed chains told apart by line weight rather than by
  legend order;
- `model_trajectories.png`: the supporting figure. It shows which model handled each stage of each
  chain, with dot colour for family and dot **area** proportional to parameter count, next to that
  chain's depth-6 result on the same y axis;
- `transition_diagnostics.png`: information loss per edge, grouped by what changed at that edge
  (family, size, both or neither);
- `stage_records.csv`: the per-stage audit log, with chain id, depth, exact model, family,
  parameter count and its basis, tokens, cost, provider, handoff length, the per-edge lexical and
  semantic-preservation measures, fact survival, and the answer scores at that depth;
- `deltas_sensitivity.csv`: every contrast repeated after removing, for that contrast, the pairs
  whose arms went through an empty handoff. A provider can return an empty body with a normal
  finish reason and a non-zero token count. A conclusion that holds in only one of the two files is
  about that failure, not about which models make up the chain;
- `transition_deltas.csv`: paired contrasts between edge types for each pair (cross-family against
  same-family at the same size, and the same comparison when size grows or shrinks);
- `metrics.csv`, `deltas.csv`, `transition_metrics.csv`, `family_transition_metrics.csv`,
  `chain_index.csv`, `model_registry.csv`, `report.md`.

Offline checks: schedules, unchanged prompts, seal enforcement, prefix memoisation, marker scaling
by area rather than radius, handling of undisclosed sizes, and both figures:

```bash
.venv/Scripts/python src/selftest_model_heterogeneity_offline.py
```

### Experiment 8b: where the size bottleneck occurs

`src/run_fictional_model_bottleneck.py` is a focused causal follow-up to the broad Experiment 8
matrix. A selector that can see the source reads a fictional dossier and Question A, and outputs
three evidence cards. Only after that packet is sealed is Question B announced. A downstream relay
sees B and the sealed packet, but not the source, and must output one card. Small and large models
are crossed in a 2×2 selector-by-relay design within both Llama and Qwen, and one fixed Mistral
model from a third family answers every final packet.

The design separates four explanations:

- `small -> large` against `small -> small` tests downstream capability, with an identical prefix
  written by a small model;
- `large -> large` against `small -> large` tests whether the early selector is the bottleneck;
- a sham swap with matched capacity controls for the intervention itself;
- restoring the exact evidence for B, and an arm that explicitly reopens the source, test whether
  performance comes back once the missing evidence is available.

The restore and sham arms replace one card instead of adding text, so the channel keeps the same
width. The source-reopen arm is an upper-bound positive control and is never pooled with the
sealed 2×2 factorial.

The configured `n=20` run is complete. The effect of upstream model size points in opposite
directions in the two families (**+0.037 for Llama; −0.037 for Qwen**), so there is no shared
model-size effect. On 98 matched, schema-valid cases where the stage-1 text contains no annotated B
answer or alias, neither relay size answers B (all 196 sealed evaluations score zero).
Large-minus-small is 0.000 [0.000, 0.000] in both families, within the configured equivalence
margin of ±0.05. Restoring the exact evidence at the same width beats the sham by **+0.983** for
Llama and **+0.950** for Qwen, and restoring and reopening the source each beat the sealed
condition by **+1.000**. What the data support is therefore a bottleneck in evidence availability,
not a claim that more parameters help. See
[`results/fictional_model_bottleneck/n20/report.md`](results/fictional_model_bottleneck/n20/report.md).

```bash
.venv/Scripts/python src/selftest_fictional_model_bottleneck_offline.py
.venv/Scripts/python src/run_fictional_model_bottleneck.py --dry-run
.venv/Scripts/python src/run_fictional_model_bottleneck.py --limit 1
.venv/Scripts/python src/run_fictional_model_bottleneck.py
```

## Experiment 9: handoff size adaptation

The other experiments fix what the writer is told and vary the message, the model or the depth.
`src/run_size_adaptation.py` instead varies what the writer is told **about its reader**:

1. **Is it enough for the writer to know, or does it have to be asked?** Each statement about
   capacity appears twice: once on its own, and once followed by a clause asking the model to act
   on it. `resize_*` is built as `inform_*` plus that clause, and the selftest checks this, so the
   contrast cannot drift into an accidental rewording.
2. **When a handoff is asked to grow, what fills the extra space?** New detail, repeated detail, or
   material nobody supplied.
3. **If a fact is compressed away and a later stage is asked to expand, does the fact come back,
   and from where?** From stage 2 on, each stage receives only a sealed handoff, so a fact missing
   from that handoff but present in the next one was *not copied*. It was reconstructed.

### The baseline is not Experiment 5's prompt

Every other chain experiment here inherits Experiment 5's `conditioned` instruction, which begins
*"Write **concise** prose research notes …"*. That word is already an instruction to shrink. Using
it as the control would have made `resize_large` against control a comparison of "expand" with
"shrink" instead of with "no size instruction", and it would have measured a requested shrink
against a baseline that had already been told to shrink.

The base instruction here is therefore the published string with the single word `concise`
removed and nothing else changed. Both the runner's startup check and the offline selftest confirm
this by rebuilding the published string from the neutral one. A separate `concise` directive
appends "Keep your handoff concise", so an explicit shrink cue with no number attached is measured
as its own arm instead of hiding in the baseline. It does not put the deleted word back in its
original position.

### The directives

All of them are appended to that neutral instruction, and to nothing else:

| directive | appended text |
|---|---|
| `none` | *(nothing)* |
| `concise` | "Keep your handoff concise." |
| `inform_small` | "The next agent's model has a 2,000-token context window." |
| `resize_small` | the same sentence + "Make your handoff fit within it." |
| `inform_large` | "The next agent's model has a 10,000-token context window." |
| `resize_large` | the same sentence + "Expand your handoff to make use of the available space." |

The advertised size is **not** tied to `decoding.handoff_max_tokens`. The number the model is told
and the budget the API enforces are different variables and are kept separate. The enforced budget
is the same in every arm, and truncation is measured at every stage. In the completed run, the
repetition mode of the fill-10k arm hit that shared 12k safety ceiling in 102 calls; this is
reported as a failure mode and not mistaken for useful expansion.

### The arms

Nine directive schedules over three handoffs, including the trajectory the design is built around:

| arm | schedule |
|---|---|
| `control` | none → none → none |
| `concise_shrink` | concise → concise → concise |
| `inform_shrink` / `resize_shrink` | the small directive at every stage |
| `inform_expand` / `resize_expand` | the large directive at every stage |
| `resize_lsl` / `inform_lsl` | **large → small → large** |
| `resize_sse` | small → small → large |

Generation is memoised on the **sequence of directives so far**: two arms with the same first *s*
directives share exactly the same stage-*s* message, so every contrast starts from an identical
shared history.

### Two datasets, so new material can be traced

Noticing that a handoff gained a fact is easy; saying where the fact came from is the hard part.
`src/build_size_adaptation_data.py` builds both datasets, together with the closed-book evidence
needed to interpret them.

- **`counterfactual`:** documents whose answer contradicts what the model has memorised (the page
  says Lisbon, the model believes Rome). Every item is checked **in both directions**: the model
  must reliably give the original answer closed-book, and must never give the replacement. One set
  of closed-book samples is judged twice, once against each set of gold answers, so both conditions
  describe the same behaviour rather than two draws that could disagree by chance. A handoff that
  says "Rome" is then reconstruction from memory, not a lucky guess.
- **`fictional`:** invented subjects the model cannot know about in either direction, checked
  closed-book. Any unsupported material that shows up here is fabrication with nothing else to
  explain it. That is what separates *reconstruction* from *invention*, instead of calling both of
  them hallucination.

The counterfactual set has two sources, each tagged with a `source` field so they are never pooled
by accident. `rewritten_wikipedia` reuses `data/wikipedia_random_counterfactual/`, but those pages
were sampled at random for an experiment that required the model **not** to know the answer, so
most of them fail this experiment's opposite requirement. `known_entity` starts from the other end:
famous subjects are sampled first, confirmed to be known, and only then rewritten.

Each item also carries five **side facts**: checkable details that no question asks about, each
verified verbatim in the document and not overlapping one another. They make "reusable
information" concrete. A shrink that keeps the answer but drops them has narrowed the handoff to
the current task. One side fact per item is also *asked* at every depth, so side-fact retention
becomes an accuracy number and not only a count of strings that are present.

### What is measured

Deterministic measures (`src/size_metrics.py`), which are primary:

- length in words, characters and estimated tokens, and the ratio to the previous stage;
- **internal repetition:** the share of a handoff's 5-grams that repeat an earlier 5-gram in the
  same handoff. This separates "expanded by adding detail" from "expanded by saying the same thing
  again";
- **unsupported terms:** capitalised tokens and numbers that do not appear in the source document,
  split into `corrupted` (a near-miss version of a value the experiment tracks, e.g.
  *Taren Vos → Leran Vos*) and `invented` (no counterpart at all). The split is made at the phrase
  level against the item's tracked values, not by finding the most similar token anywhere in the
  document. That approach would report long articles as more "corrupting" than short ones, just
  because they offer more strings to be similar to;
- whether each fact is present at every stage, and the resulting life history: survived, lost, or
  **lost and then reappeared**.

LLM-judged measures (`openai/gpt-4o-mini`, a different family from the system under test), which
are secondary: answer correctness against both sets of gold answers, semantic preservation across
each edge, and a new **claim-support judge** that reads the source document and counts claims in
the notes that the document does not support. Claims are checked against the source at every
depth, not against the previous note. No stage is ever allowed to introduce material the document
does not contain, and checking stage 3 against stage 2 would accept a fabrication that entered at
stage 2 and was then copied faithfully.

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

Smaller runs and reanalysis work as elsewhere:

```bash
.venv/Scripts/python src/run_size_adaptation.py --arms control,resize_expand --limit 5
```

```bash
.venv/Scripts/python src/run_size_adaptation.py --analyse-only
```

Outputs go to `results/size_adaptation/`:

- `size_adaptation.png`: the main figure. Length, answer accuracy, side-fact retention and internal
  repetition against handoff depth, for each dataset. Requested resizes are thick solid lines and
  their "informed only" twins are dashed lines in the same colour, so *does asking do more than
  telling* is the gap between the solid and the dashed line of one colour;
- `expansion_content.png`: length against what it is made of (repetition, unsupported terms, and
  tokens that are new since the previous stage), so an arm that doubled its word count clearly
  falls into one of three regimes;
- `expansion_modes.png`: every fill-10k call split by how it finished, with length and
  within-message repetition on log scales so that the mode that stops and the mode that runs to the
  budget stay distinct;
- `headline_contrasts.png`: a forest plot at depth 3 of the stable length and side-fact contrasts
  used in the main report, leaving out runaway outputs;
- `recovery.png`: loss, recovery, reversion and fabrication for each arm, next to where the final
  counterfactual answer came from. An arm that looks good on recovery only because it fell back on
  memorised knowledge cannot be read as a success;
- `metrics.csv`, `contrasts.csv` (paired per-item differences with bootstrap CIs and p-values),
  `recovery.csv`, `fact_histories.csv` (the audit trail for each fact), `answer_origin.csv`,
  `arm_rows.csv`, `report.md`.

Offline checks: neutral prompts, `resize == inform + one clause`, seal enforcement, prefix
memoisation, the split between corruption and invention, the fact life-history classifier, and all
figures:

```bash
.venv/Scripts/python src/selftest_size_adaptation_offline.py
```

## Experiment 10: communication regret under a hard communication budget

The earlier conditioning experiments (5, 5b and 8b) all compared a handoff written for a known
question with one written for no question, and measured what the second could still answer. None
of them held the *channel* fixed. Experiment 5's length-matched control put a word target in the
prompt and set a `max_tokens` limit that did not bind, so the arms still chose very different
lengths, and a gap in held-out accuracy could not be separated from "one arm wrote more".
Experiment 5b fixed capacity structurally, with exactly K evidence cards, but had to give up
free-text handoffs to do it.

`src/run_communication_regret.py` keeps free-text handoffs and makes the budget itself the
controlled variable. The claim under test:

> Under an equal communication budget, conditioning a handoff on the currently known
> information need reallocates communication capacity toward present utility and can reduce
> its utility for plausible future information needs.

Supporting it requires **both** `dU_now > 0` and `dU_future < 0` against a generic or reusable
baseline, with both getting stronger as the budget shrinks. Anything short of that is reported as
measured.

### The utility matrix

Each source context has *k* questions that can each be answered from it. Every question takes a
turn as the conditioning query, and each resulting message is then answered against **every**
question for that context. This gives one matrix per context:

    M[a][b] = U(m_a, q_b)      m_a = the message written knowing only q_a

`U_now` is the diagonal and `U_future` is the mean of the off-diagonal cells. Because each question
is on the diagonal in one rotation and off it in the other *k*-1, question difficulty cannot create
the contrast, and there is no fixed A/B split that has to be trusted. The earlier pair-based
designs could not offer that.

`generic`, `oracle` and `extractive_generic` do not depend on which question is known, so each of
them has **one** message per (context, budget) that every rotation reuses. On SQuAD, where every
question rotates, their `U_now` and `U_future` are therefore equal by construction. On the relation
dossiers only the four anchor questions rotate, but `U_future` also includes the twelve non-anchor
questions, so the two averages need not be equal. In both corpora the message itself is the same in
every rotation, and that, rather than equality of two averages made up of different questions, is
the property an unspecialised baseline needs.

### The budget has two sides

A cap on its own does not equalise anything. A pilot with only a cap, on 3 contexts at 40 words,
gave these fill ratios:

| policy | fill (cap only) | fill (cap + floor) |
|---|---|---|
| `conditioned` | 0.61 | 0.92 |
| `reusable` | 0.75 | 0.91 |
| `oracle` | 0.83 | 0.96 |
| `generic` | 0.88 | 0.93 |

The conditioned arm quietly used a third less of the channel than the control. Any
future-question deficit it showed would then have meant *it wrote less*, not *it used the space
differently*. `src/budget.py` therefore constrains both sides ("write between *floor* and *cap*
words") and enforces this in three steps:

1. one length instruction, identical for every policy;
2. a message outside the band is sent back for a rewrite with an identical correction
   (`SHRINK_CORRECTION` / `EXPAND_CORRECTION`), up to `budget.max_length_corrections` times;
3. whatever comes back is truncated to the cap **every time** before any reader sees it, at the
   last sentence boundary that fits.

The cap is hard and always holds. The floor cannot be guaranteed, since a model cannot be made to
produce more words on demand, so it is requested, audited and reported as `under_floor_rate` for
each arm, never assumed. The enforced unit is words, not tokens: a word count is deterministic,
needs no tokenizer and is the same for every model. The nominal token budget, the requested
`max_tokens`, the model's own completion-token count and a token estimate of the delivered text are
all recorded, so the token view is never lost.

Experiment 9 found that a request to fill the space can be answered by restating the same content.
Because the floor is also a request to expand, `internal_repetition_rate` is measured on every
delivered message and reported next to its length.

Three further length controls back this up, so the audit is not the only evidence:

- **paired length differences** between every pair of compared arms, bootstrapped over contexts;
- **a length-matched subsample:** only the contexts where all compared arms delivered within
  `budget.match_tolerance_words` of one another;
- **a trimmed arm** (`policy@trim`): every abstractive message in a cell is cut again to the
  shortest delivered length in that cell and answered again. This costs answer calls but no
  generation calls, and it is the strictest check available: identical length for every context,
  by construction.

### The policies

Only the conditioning block differs between the four abstractive arms, and the offline selftest
checks that removing it leaves one identical prompt skeleton.

| policy | what the sender is told | messages per context and budget |
|---|---|---|
| `generic` | that a question will be asked, but not which one | 1 |
| `conditioned` | the current question, and to communicate what answers it | *k* |
| `reusable` | the current question, that unknown further questions may follow, and to keep reusable evidence within the same limit | *k* |
| `oracle` | every question the context supports; an upper bound, not a strategy that could be deployed | 1 |
| `extractive_generic` / `extractive_conditioned` | the same two levels of conditioning, but copying sentences verbatim instead of writing prose | 1 / *k* |

`generic` deliberately says only that it has not been told the question. Telling it that more
questions may follow *is* the `reusable` treatment, and putting the treatment into the control would
destroy the contrast the control is there to provide. The extractive pair separates the loss from
choosing information from the loss from abstractive rewriting. It is the one place where the shared
output format is changed on purpose.

### Regret, and what it is measured against

The answerer also reads the source directly for every question. That is the no-handoff ceiling and
the reference point for regret:

    R(q) = U(D, q) - U(m, q)          R_future = E[ R(q) | q != conditioning question ]

Normalised retained utility is reported in the form that avoids a problematic denominator.
Restricted to cells that the source itself answers, `U(D, q) = 1`, so retained utility is just
`U(m, q)` and regret is `1 - U(m, q)`. The raw ratio is available in `rotation_rows.csv`, with the
ceiling stored for each cell, and contexts whose ceiling is below `analysis.min_ceiling_for_ratio`
are counted rather than silently dropped.

### Two corpora

| corpus | what it is | why |
|---|---|---|
| `squad_groups` | 24 SQuAD paragraphs, each with 4 human-written questions | natural questions written by people; nothing was written to make the effect appear |
| `relation_dossiers` | invented dossiers with 4 aspects x {anchor, paraphrase, same-entity, same-topic} | the *distance* between the conditioning question and a hidden question is designed in, not estimated from embeddings |

Both are built by `src/build_regret_data.py`. A SQuAD group is kept only if **all four** of its
questions pass the project's closed-book leakage filter against the answering model (a correct
off-diagonal answer from memory would look exactly like a reusable handoff), and only if the group
passes an independence audit by a model from a different family: every question must be answerable
from the paragraph and must ask about a different fact from all the others.

The relation corpus exists because SQuAD offers no clean way to label distance, and forcing noisy
categories onto it would be worse than not measuring distance at all. Each dossier is written to a
strict schema, every designed answer is checked verbatim in the text, and the structure is verified
before use. For a conditioning question about aspect *a*, the hidden questions sit at four known
distances: `paraphrase` (the same answer, worded differently), `same_entity` (a second fact about
the same subject in the same aspect), `same_topic` (a different named entity in the same aspect)
and `orthogonal` (any anchor from a different aspect).

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

Smaller runs and reanalysis work as elsewhere (`--limit`, `--budgets`, `--policies`,
`--analyse-only`). Outputs go to `results/communication_regret/<corpus>/n<N>/`:

- `pareto_now_vs_future.png`: one panel per requested budget, all on the same scale. Each policy is
  a separate (`U_now`, `U_future`) marker; observations that are not dominated within their budget
  are outlined, dominated ones are muted, and no lines are drawn between the categorical policies;
- `utility_vs_budget.png`: `U_now`, `U_future` and the gap between them against budget, with the
  direct-context ceiling;
- `communication_regret.png`: `R_now` and `R_future` for each policy and budget, with CIs;
- `relation_distance.png`: future-question regret by designed distance (relation corpus only);
- `REPORT.md` (in `results/communication_regret/`): the report covering both corpora.
  `src/render_regret_summary.py` regenerates it from the CSVs, so it cannot drift away from them. It
  contains every headline table for both corpora and an appendix listing all 3,284 computed cells;
- `utility_matrix.csv` (every M[a][b] cell with its relation label and ceiling),
  `rotation_rows.csv`, `metrics.csv`, `contrasts.csv` (paired bootstrap differences clustered by
  context, with p-values and Cohen's *dz*), `budget_interaction.csv`, `length_audit.csv`,
  `length_deltas.csv`, `length_matched_contrasts.csv`, `pareto.csv`, `baselines.csv`,
  `relation_regret.csv`, `report.md`.

### What it found

Both corpora support the hypothesis, and on SQuAD it holds at every budget where the generic
baseline has room to lose anything.

| `conditioned` - `generic`, LLM-judge accuracy | 20 w | 40 w | 80 w | 160 w |
|---|---:|---:|---:|---:|
| SQuAD, dU_now | +0.625 | +0.448 | +0.281 | +0.104 |
| SQuAD, dU_future | -0.063 | -0.115 | -0.194 | -0.170 |
| Dossiers, dU_now | +0.859 | +0.641 | +0.516 | +0.359 |
| Dossiers, dU_future | -0.026 | -0.110 | -0.198 | -0.342 |

Every cell above excludes zero except SQuAD dU_now at 160 words (p = 0.066) and dossier dU_future
at 20 words (p = 0.055). Exact match and token F1 agree with the judge. The specialisation gap
grows as the channel narrows: between the 160-word and the 20-word budget it widens by +0.413
[0.326, 0.503] on SQuAD and by +0.184 [0.080, 0.285] on the dossiers.

Three results decide how to read this:

- **The channel was not the limit.** `oracle`, held to the same band, scores 0.823 on *every* SQuAD
  question at 40 words. It matches `conditioned` on the present question with the same ~37 words,
  while giving up nothing on the other three. What a conditioned handoff drops is not something the
  budget had no room for.
- **It is selection, not rewriting.** The verbatim-extraction arms show the same pattern (+0.385 /
  -0.198 at 20 words), and no abstractive rewriting happens in them.
- **Asking for reusability barely helps.** `reusable` still loses future utility against `generic`
  at every budget, and cannot be told apart from `conditioned` on `U_future` at the two tightest
  budgets.

On the relation corpus, future-question regret rises steadily with the designed distance from the
conditioning question: 0.00 for a paraphrase, 0.42-0.95 for same-entity and same-topic questions,
and 0.95-0.98 for an orthogonal aspect. The gradient appears only in the conditioned arms. Widening
the budget to a third of the source flattens it for `generic` and `oracle`, but not for
`conditioned`.

Full numbers, controls and caveats are in section 10 of
[results/HANDOFF_EXPERIMENTS_REPORT.md](results/HANDOFF_EXPERIMENTS_REPORT.md).

Offline checks: identical prompt skeletons, cap enforcement, floor corrections, rotation balance,
seal enforcement, single-flighting, matrix and regret arithmetic against an injected effect,
trimming, Pareto domination, relation labelling, and the SQuAD builder against the live parquet
schema:

```bash
.venv/Scripts/python src/selftest_communication_regret_offline.py
```

## Experiment 12: anticipatory context management under uncertain future demand

Experiments 10 and 11 both assume a single message that cannot be revised. A real agent harness is
not in that position. It decides what to keep in context, what to compress, what to move to
storage, what to leave a pointer to and what to drop, and it can fetch things back later. It also
does not know the future question; at best it has an *estimate* of where demand will fall.

This experiment separates the true distribution of future use, `P`, from the harness's estimate,
`P̂`, and asks what the difference between them costs:

```
max_pi  U_now + lambda * E_{Q ~ P-hat}[U_future] - beta * C_context - gamma * C_retrieval
```

### Future demand is modelled over evidence units, not questions

Predicting the exact wording of a future question is neither realistic nor necessary. The relation
dossiers are built on a crossed grid of 4 aspects x 4 roles, and the aspect axis splits the
evidence exactly. Across the 16 dossiers, all 256 questions have their gold answer verbatim in some
source sentence. Of the 264 sentences, 191 belong to exactly one aspect and **none to two** (mean
pairwise Jaccard 0.000). The remaining 73 are background prose that answers nothing, which makes
them the corpus's own distractors.

So `P(Q_future)` is a distribution over four aspects, and predicting "the next question is about
finance" picks out a real, separate subset of the source. `src/anticipatory_data.py` extracts and
labels those units, and refuses any dossier where the split would overlap rather than resolving
the tie.

### Two noise axes, kept separate

`P̂` is moved away from `P` in two independent ways, because being uncertain and being confidently
wrong are different failures:

- **dilution** `P̂ = (1-tau)P + tau*U`: higher entropy, no bias;
- **displacement** `P̂ = (1-sigma)P + sigma*P_wrong`: bias at roughly constant entropy, where
  `P_wrong` is the least likely aspect.

Both are linear interpolations, so total variation is *exactly* linear in each knob. This is
checked as a regression test, not just hoped for. It means any threshold later seen in regret
against mismatch comes from the policy and cannot be caused by how the noise was parameterised. It
also requires `P` to be peaked: `TV(P, dilute(P,tau)) = tau * TV(P,U)`, so a flat true distribution
would leave the dilution sweep no range at all.

### Six actions, and the one that matters

`keep` (verbatim, costing its words), `compress` (shorter and lossy, used exactly at the budget
boundary, which is where a harness would use it), `externalize` (in the store, not in context),
`pointer` (a short index line in the message, with the unit in the store), `retrieve` (BM25 at
answer time) and `discard` (gone).

The informative contrast is `pointer` against `externalize`. Both put the evidence off until later,
but only a pointer spends context saying *what* was put off. If retrieval beats preservation, the
pointer arm separates "recall is cheap" from "recall is cheap because the writer left a map".

Retrieval uses the project's own Okapi BM25, which has no external dependencies
(`src/retrieval.py`, already used in Experiments 3 and 4), not a simulated oracle.

### The isolation guarantee is restated, not relaxed

`orchestrator_answer` still accepts only a four-string `SealedHandoff`. Experiment 12 adds a
separate frozen `SealedContext` that carries the message plus whatever retrieval returned, so every
earlier experiment keeps exactly the guarantee it was verified under.

The subtle risk is that a retrieval arm could turn into direct context under another name. The
first version of the guard checked whether the store held the whole source, which was the wrong
question, because the reader never sees the store. It sees the message plus at most `k` units that
it had to find through BM25. What needs a bound is the largest context that can be assembled for
**one answer**: the delivered units plus `k`. `assert_bounded_recall` enforces that, and the
offline selftest tests it in both directions.

### Running it

```bash
.venv/Scripts/python src/run_anticipatory_context.py --dry-run
.venv/Scripts/python src/run_anticipatory_context.py --budgets 40,80
.venv/Scripts/python src/render_anticipatory_report.py
```

Offline checks, with no API key:

```bash
.venv/Scripts/python src/selftest_anticipatory_offline.py
```

## Experiment 14: which compression mechanism specialises a handoff?

Experiment 10 found that a bounded handoff written for the currently known question gains
present-question utility and loses future-question utility, and loses more the further the future
question is from the conditioning one. That was measured entirely on **abstractive** arms, where an
LLM read the source and wrote new prose. So in every one of those arms, two things changed
together:

    the sender SELECTED what to keep,
    and it REWROTE what it kept.

Either one could be what ties the message to the present task. Experiment 14 separates them on the
same corpus, with the same channel, reader and judge, by adding arms that select without rewriting:

| paper name | code arm | rewrites? | selects? | selector |
|---|---|---|---|---|
| `paraphrase` | `paraphrase` | yes | no (told not to choose) | — |
| `summary_generic` | `generic` | yes | yes, query-agnostic | the LLM itself |
| `summary_conditioned` | `conditioned` | yes | yes, query-aware | the LLM itself |
| `lm_elimination_generic` | `lm_generic` | no | yes, query-agnostic | GPT-2 self-information |
| `lm_elimination_conditioned` | `lm_conditioned` | no | yes, query-aware | GPT-2 question-likelihood gain |
| `nonllm_elimination_generic` | `nonllm_generic` | no | yes, query-agnostic | TF-IDF centrality |
| `nonllm_elimination_conditioned` | `nonllm_conditioned` | no | yes, query-aware | BM25 against `q_now` |
| `random_selection` | `random_selection` | no | yes, seeded shuffle | — |

The arms form a deliberate progression: passthrough does neither, paraphrase only rewrites, summary
rewrites and selects, LM elimination selects without rewriting, and non-LLM elimination selects
without rewriting **and without any neural model at all**.

`passthrough` needs some explanation here. The dossiers are 389–517 words long and the largest
budget is 160 words, so no arm with a budget can carry the whole source. The only fair budgeted
pass-through is the beginning of the source, cut at the last sentence boundary that fits. That is
positional elimination at the same sentence granularity as the scored arms, and it serves as the
floor that asks whether a selector does better than reading from the top. The **unbounded**
pass-through is the separate `direct_context` baseline, where the reader sees the whole source and
scores 0.992. If specialisation still appears in the last row of the table, it is a property of
bounded, task-aware selection, not of language models.

### Elimination is a mechanism here, not a prompt

Experiment 10 already had `extractive_generic` and `extractive_conditioned` arms. In those, an LLM
is *asked* to copy sentences verbatim, and on the SQuAD run only 0–29% of their messages actually
were verbatim (they never ran on the relation corpus at all). A prompt is not a mechanism, so those
arms are not reused.

`src/elimination.py` does the selection in code instead. It scores every sentence, takes sentences
in score order while they fit under the cap, outputs the chosen ones **in source order, separated
by single spaces**, and rebuilds the message from the units to check it before delivery
(`verify_verbatim`). A sentence that does not fit is skipped, never cut, so the cap always holds and
the sender cannot pad. The offline selftest checks all of this on all 768 selection messages.

### The LM scorer is pinned, local, and separate from the system under test

`src/lm_unit_scores.py` writes a score file with a manifest, using GPT-2 small (the compressor LM in
the small configuration of LLMLingua and LongLLMLingua) pinned at commit `607a30d7`. It runs on CPU
with teacher forcing and never generates text. One model gives two scores:

```
lm_generic      -(1/T) * sum_t log p(x_t | x_<t)                        # self-information
lm_conditioned  (1/|q|) * [ sum_i log p(q_i | unit, q_<i)
                          - sum_i log p(q_i | q_<i) ]                   # LongLLMLingua-style
```

Conditioned scores are computed **only** for the four anchor questions that rotate into the
conditioning role, so the hidden questions never reach the scorer. The script that writes the
scores enforces this, and the runner checks it again when reading them. The runner also refuses a
score table whose corpus hash does not match the dossiers it is about to compress.

Because the scorer sits outside the run path, `requirements.txt` still has no `torch`: the scores
are an input file, and `python src/run_communication_regret.py` never loads a model.

### What was reused instead of rerun

The run uses Experiment 10's model, decoding, word band, rotation design, reader prompt and judge
unchanged, so its 640 stored relation messages and 11,264 stored answers have exactly the same
request hashes as this run and are imported instead of regenerated
(`runs/compression_mechanism/relation_dossiers/n16/IMPORTED_FROM.txt` records the copy, and nothing
under `runs/communication_regret/` is modified). The offline selftest checks that the hashes match,
which also shows that adding these arms changed no published prompt.

As a result the whole experiment cost **$0.077**: 64 new sender calls for `paraphrase`, none for the
768 selection messages, and 8,896 unique answer calls after deduplication. Observed wall-clock times
were 106 s to score all 16 dossiers with GPT-2 on CPU, about 20 minutes for the main run at
concurrency 12 (judge at 16), and about 1 minute for the `passthrough` top-up. The pipeline is
limited by API latency, not by local compute.

That the two families of mechanism really differ is measured, not assumed. At the 160-word budget,
`passthrough` and the four elimination arms deliver 5.19–5.88 source sentences *verbatim* per
message, compared with 0.06–0.11 for `paraphrase`, `summary_generic` and `summary_conditioned`. The
accounting for every message (`target_words`, `delivered_words`, `fill_ratio`, `source_words`,
`retention_fraction`, `selected_unit_count`) is in `mechanism/budget_accounting.csv`, and no message
is over the cap.

### Result

All three mechanisms specialise. At every working budget (40, 80 and 160 words), the paired
difference-in-differences excludes zero in all three families. Here that means the conditioned
arm's near/far gradient minus the gradient of its own generic control, clustered by source:

| family | 40w | 80w | 160w |
|---|---:|---:|---:|
| abstractive summary | −0.875 [−0.923, −0.823] | −0.807 [−0.874, −0.729] | −0.835 [−0.921, −0.746] |
| LM elimination | −0.595 [−0.721, −0.473] | −0.600 [−0.741, −0.462] | −0.559 [−0.665, −0.441] |
| non-LLM elimination | −0.792 [−0.876, −0.708] | −0.792 [−0.888, −0.686] | −0.630 [−0.727, −0.513] |

BM25 over the source's own sentences, delivered verbatim, with no neural model anywhere in the
compressor, reproduces the gradient at 70–95% of the abstractive size. **Bounded, task-aware
selection is enough to shape a handoff around the present question; neither abstractive rewriting
nor an LLM selector is needed for that.**

The gradient is only part of the picture, though, and here the two halves of the Experiment 10
result come apart. Compared with its own generic control, only the abstractive family also *loses*
absolute future utility:

| family | dU_now (40/80/160w) | dU_future (40/80/160w) |
|---|---|---|
| abstractive summary | +0.641 / +0.516 / +0.359 | **−0.110 / −0.198 / −0.342** (all exclude 0) |
| LM elimination | +0.531 / +0.516 / +0.516 | −0.005 / −0.011 / −0.019 (all cover 0) |
| non-LLM elimination | +0.734 / +0.750 / +0.609 | +0.008 / +0.009 / +0.011 (all cover 0) |

In the elimination families, conditioning improves present utility at essentially no cost: the
gradient comes from *raising the near questions*, not from lowering the far ones. Only rewriting
lowers them, and it lowers them more as the budget grows. At 160 words the conditioned summary gives
up 0.342 of the future utility that its own generic control had.

The evidence-level diagnostic shows why. At 160 words, the answer to the orthogonal question is
still present in **0.039** of `summary_conditioned`'s messages, compared with **0.286-0.293** for the
conditioned elimination arms, which is about what their own generic controls (0.312), and even the
random floor (0.387), deliver. Conditioned rewriting *erases* distant evidence. Conditioned selection
simply does not go out of its way to include it, and keeps roughly what an unconditioned selector
would have kept.

The aspect-retention table, measured on the delivered units alone with no reader involved, agrees.
At 160 words `nonllm_elimination_conditioned` keeps 0.513 of the conditioning aspect's sentences and
0.266 of the others, while its generic control keeps 0.302 of everything evenly. The conditioned
selector adds to the near aspect roughly what it takes from the rest, and what it takes was not
worth much to begin with. A conditioned summary instead rewrites the whole message around `q_now`,
so the other aspects vanish from prose that no longer mentions them.

The mechanisms also differ in strength, and the triple difference shows it. The abstractive
summary's gradient is larger than LM elimination's at every working budget (−0.28/−0.21/−0.28) and
larger than non-LLM elimination's at 160 words (−0.204 [−0.361, −0.062]), while at 40 and 80 words
the abstractive and non-LLM gradients cannot be told apart statistically. The fair reading is:
**selection is enough to produce the shape; rewriting makes it stronger, and rewriting is the only
mechanism that pays for it in absolute future utility**.

Two secondary results are worth recording, because they were not predicted:

* **The reader is almost never the problem.** When the gold answer is present in the delivered
  message, judged accuracy is 0.93–1.00 for every arm at every budget. Nearly every failure in this
  experiment is deletion by the compressor, not misreading by the receiver, and that is what allows
  the utility differences to be attributed to the mechanism at all.
* **Query-free lexical importance does worse than chance, and worse than the prefix.**
  `nonllm_generic` (TF-IDF centrality) reaches 0.344 `U_now` at 160 words, compared with 0.469 for
  `random_selection` and 0.453 for `passthrough`. Centrality prefers the sentence that shares the
  most vocabulary with the rest of the dossier, and that sentence is the least fact-dense. A
  query-free *LLM* summary (`generic`, 0.609) beats both, because it can pack several facts into one
  sentence, whereas selection has to spend a whole sentence on each fact.

The 20-word budget is reported everywhere but left out of these claims. The corpus's shortest
sentence is 10 words and its median is 27, so whole-sentence selection delivers one sentence or none
at that budget (five of the sixteen dossiers have no sentence short enough, and their message is
empty). That is a consequence of selecting whole sentences, and it is recorded rather than patched:
cutting a sentence to fit would quietly turn the arm into a different mechanism.

### Running it

```bash
.venv/Scripts/python src/lm_unit_scores.py --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
.venv/Scripts/python src/run_communication_regret.py --config compression_mechanism_config.yaml --dry-run
.venv/Scripts/python src/run_communication_regret.py --config compression_mechanism_config.yaml
.venv/Scripts/python src/render_mechanism_report.py
```

Offline checks, with no API key, no model and no network:

```bash
.venv/Scripts/python src/selftest_compression_mechanism_offline.py
```
