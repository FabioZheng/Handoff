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
    run_multilingual_handoffs.py    Experiment 6 -- gold-only fixed-language
                                    vs language-switching handoff chains
    run_slack_facts.py       unwritten follow-up correcting Exp. 3/4's signal/filler confound
    run_slack_retrieval.py   unwritten follow-up correcting Exp. 3's retention-pooling defect
    plot_conditioning_comparison.py   analysis-only plot for the chain_generic replication
    selftest_offline.py      checks that need no API key
    selftest_chain_offline.py  chain-experiment checks, incl. the isolation guarantee
    selftest_summary_generalization_offline.py  Experiment 5 arm/transition checks
  data/  runs/  cache/  results/
  research/                 a separate personal research-notes / knowledge-graph
                             layer built on top of this project's findings (not
                             part of the experiment pipeline) -- profile.md,
                             schema.md, tools/graph_to_obsidian.py; rendered
                             into the obsidian/ vault at the repo root
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

**C7 — no LLM judge.** SQuAD-style normalisation (lowercase, strip punctuation, drop articles,
collapse whitespace), then exact match and token-F1, maxed over the gold answer and its aliases.
`summary.csv` carries an `em_f1_disagree` column — the fraction of items EM scores 0 but F1
scores ≥ 0.5 — so a materially divergent EM/F1 picture is visible rather than averaged away.

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
