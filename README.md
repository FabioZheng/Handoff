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
    data.py                MuSiQue load, C1 leakage filter, sample, gold-sentence derivation
    llm.py                 OpenRouter client: retries, cache, token accounting, cost cap
    handoffs.py            single-handoff mechanisms (Experiment 1) + sealed orchestrator boundary
    judge.py                shared LLM-judge answer-correctness scoring
    score.py                EM/F1, bootstrap CIs
    run.py                  Experiment 1 -- single-handoff mechanism pilot
    chain_data.py            dataset adapters shared by the chain experiment
    run_chain.py             Experiment 2 -- repeated handoff degradation
                              (also drives the Qwen replication and the
                              question-conditioned/generic replication --
                              see chain_config.yaml, qwen_chain_config.yaml,
                              chain_generic_config.yaml)
    run_retrieval_quality.py Experiment 3 -- retrieval quality through repeated handoffs
    run_redundant_signal_ratio.py  Experiment 4 -- fixed-context redundant-evidence signal ratio
    run_summary_generalization.py  Experiment 5 -- question conditioning and
                                    cross-question generalization
    run_slack_facts.py       unwritten follow-up correcting Exp. 3/4's signal/filler confound
    run_slack_retrieval.py   unwritten follow-up correcting Exp. 3's retention-pooling defect
    plot_conditioning_comparison.py   analysis-only plot for the chain_generic replication
    selftest_offline.py      checks that need no API key
    selftest_chain_offline.py  chain-experiment checks, incl. the isolation guarantee
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

**MuSiQue-Answerable dev**, 2417 questions, via the HuggingFace parquet mirror
(`dgslibisey/MuSiQue`, validation split). Each question ships 20 paragraphs, 2–4 of them gold,
plus the gold decomposition.

The subagent's context is the gold paragraphs **plus all distractors**, in a fixed shuffled
order seeded by `sampling.seed` and the question id. Presentation ids (`P1`…`P20`) are assigned
*after* shuffling, so the ids the model cites are stable across runs and the gold/distractor
split is not guessable from the id.

**Retrieval is not modelled.** The subagent is handed the evidence. This is intentional: it is
what makes any measured drop attributable to the handoff rather than to search quality.

The surviving question ids are written to `data/sampled_ids.json` and the full records to
`data/filtered_questions.jsonl`; both are committed so runs are reproducible.

### One deviation from the spec, and why

The spec asks for gold supporting *sentences* "if available". **MuSiQue annotates supporting
paragraphs, not sentences**, so they are not available directly. Rather than fall back to
HotpotQA or coarsen `E_oracle` to whole paragraphs (which would make it nearly a copy of
`A_full` and destroy the compression contrast), gold sentences are derived **deterministically
from the gold decomposition**: for each hop, within that hop's supporting paragraph, keep the
sentences containing that hop's gold answer string. No LLM is involved and no model opinion
enters — criticality still comes from the dataset's gold annotations.

Verified on the first 40 candidates (`src/selftest_offline.py`): every question yields gold
sentences, every question retains a final-hop sentence, all derived sentences are verbatim
substrings of their source paragraph, the sentence set spans every gold paragraph, and 40/40
contain the gold answer string. A typical oracle handoff is ~470 characters against ~12,000
for the full context.

This also gives `inj_drop_claim` a principled target: the final-hop sentence is flagged
`is_final_hop`, so "the claim required for the answer" is identified from gold annotations
rather than guessed.

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
- **The C1 filter selects for questions the model finds hard**, which skews the sample toward
  the difficult tail of MuSiQue. The measured headroom is therefore not the headroom on a
  representative question mix.
- **Gold sentences are derived, not annotated** (see above). The derivation is deterministic and
  verified, but it is a heuristic over MuSiQue's paragraph-level labels.
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
| Dataset | MuSiQue, HotpotQA | Longer 2-4-hop chains versus sentence-labelled two-document QA |
| Evidence length | `short`, `medium`, `full` | Gold documents only; gold plus distractors up to 5 documents; every dataset document |
| Compression depth | 0, 1, 2, 3, 5 | Direct answer or that many consecutive free-form handoffs |

All evidence-length variants retain every gold document. Differences across
`short`/`medium`/`full` therefore measure distractor and context-length effects,
not retrieval recall. The first compression agent sees the selected documents.
Every subsequent agent receives only a frozen `SealedHandoff` made from the
preceding summary, so lost details cannot leak back into the chain.

Run a small two-dataset pilot:

```bash
python src/run_chain.py --n 10 --candidates 40
```

Run the configured experiment (30 filtered questions per dataset):

```bash
python src/run_chain.py
```

Narrow runs are composable and resumable:

```bash
python src/run_chain.py --datasets musique --contexts short,full --depths 0,1,3,5
python src/run_chain.py --plot-only
```

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

Offline verification, including the real HotpotQA schema and plot generation:

```bash
python src/selftest_chain_offline.py
```
