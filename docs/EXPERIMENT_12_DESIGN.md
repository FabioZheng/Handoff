# Experiment 12: anticipatory context management under uncertain future demand

This is the design as it was written before the experiment ran. At that point the groundwork
was built and checked offline, with no runs and no API spend. The experiment has since been
run, and the results are in
[results/HANDOFF_EXPERIMENTS_REPORT.md](results/HANDOFF_EXPERIMENTS_REPORT.md). It extends
Experiments 10 and 11 and does not replace either.

What existed when this was written:

- `src/anticipatory_data.py`: evidence units, aspect labelling, the coverage check, both noise
  models and both mismatch measures.
- `src/selftest_anticipatory_offline.py`: 32 checks against the real corpus, all passing, with
  no API key needed.
- The three questions left open in the first draft are all resolved; see §3, §6 and §8.

## 1. The question

Experiments 10 and 11 studied a handoff that is *static and irreversible*: one message, one
budget and no way back. They established the cost of specialising. Conditioning on the known
query raises `U_now` at the expense of `U_future`. Forcing an allocation quota α controls that
trade-off, while stating a weight λ in the prompt does not.

This experiment asks the same question at the level of the harness: **if you have to decide
what to keep before you know what will be needed, and you can keep discarded evidence
retrievable, what should you do?** To make that concrete, it separates the true distribution of
future use, `P(Q_future)`, from the harness's estimate of it, `P̂`:

```
max_π  U_now + λ·E_{Q~P̂}[U_future] − β·C_context − γ·C_retrieval
```

The goal is not the optimum of this objective. It is to see how the loss grows as `P̂` moves
away from `P`, and whether retrieval enlarges the set of achievable outcomes rather than just
picking a different point inside it.

## 2. Why the relation dossiers

`data/communication_regret/relation_dossiers.jsonl` holds 17 dossiers, of which 16 are used.
Each is a fictional source of about 2,850 characters with **16 questions on a fully crossed
4 × 4 grid**:

- **4 aspects:** `founding`, `facilities`, `finance`, `custom`
- **4 roles:** `anchor`, `paraphrase`, `same_entity`, `same_topic`

They suit this experiment for three reasons:

1. **Demand can be modelled over evidence rather than question wording.** Future demand should
   be described in terms of facts or topics, not exact questions, and the aspect axis does
   exactly that. `P(Q_future)` becomes a distribution over **4 aspects**, which is small enough
   to sweep systematically and large enough for a divergence between two distributions to mean
   something.
2. **The existing relation labels already follow the aspects.** `regret_data.relation_label`
   returns `paraphrase`, `same_entity` or `same_topic` for questions in the same aspect and
   `orthogonal` for questions in a different one. No new labelling is needed to say how a
   future question relates to the conditioning one.
3. **The channel is the only route to the answer.** Closed-book accuracy is 0.000 and the
   direct-context ceiling is 0.992, so any measured utility comes from what was communicated or
   retrieved, not from what the model already knows.

`squad_groups` is kept as a secondary setting. It has no aspect labels, so it can only support
the no-prediction arm and a uniform `P̂`. It serves as a robustness check and never as a
primary cell.

## 3. Evidence units

Each source is split into sentences with `data.split_sentences` (already used by
`budget.truncate_to_words`). Each sentence, or unit, is labelled with the aspect or aspects it
supports:

- main signal: the unit contains the gold answer to one of that aspect's questions;
- backup signal: BM25 similarity between the unit and that aspect's four questions, using
  `retrieval.BM25Index`.

An offline selftest enforces an acceptance check: **every question's gold answer must be found
in at least one labelled unit**, and every aspect must have at least two units. A dossier that
fails is dropped and logged, never quietly patched.

**Resolved (2026-09-07).** The dossier `meta` field holds only `source_words`,
`source_characters` and `subject`, with no aspect spans, so labelling units has to be a build
step. It is now implemented in `anticipatory_data.extract_units`.

Measured over the whole corpus, the data turned out cleaner than the design assumed:

| | |
|---|---|
| questions whose gold answer appears verbatim in a sentence | **256 / 256 (100%)** |
| sentences (evidence units) | 264 across 16 dossiers, 14–20 per dossier |
| claimed by exactly one aspect | 191 (72%) |
| **claimed by two or more aspects** | **0** (mean pairwise aspect Jaccard 0.000) |
| unclaimed background | 73 (28%) |
| (dossier, aspect) cells with at least 2 units | **64 / 64**, min 2, mean 3.0 |

Gold containment alone covers everything, so BM25 is demoted to a cross-check and is not needed
for labelling. The aspects split the evidence into sets that do not overlap at all. That is
what justifies modelling `P(Q_future)` at the aspect level in the first place: predicting
"finance" picks out a real, separate part of the source, not a vague topic label.

Disjointness is checked, not assumed. `extract_units` raises an error if any sentence is ever
claimed by two aspects, so a future corpus that breaks this property fails loudly instead of
quietly weakening the experiment.

## 4. Action space

Six actions, each with a concrete implementation and an explicit cost. This is what takes the
experiment beyond "one message, take it or leave it".

| Action | `C_context` | `C_retrieval` | Implementation |
|---|---|---|---|
| **keep** | all of the unit's words | 0 | the sentence, verbatim, in the message |
| **compress** | about α × the unit's words | 0 | the unit compressed by an LLM, under the same word contract as Exp 10 |
| **externalize** | 0 | 1 retrieval | the unit goes into the store and is left out of the message |
| **pointer** | 4–8 words | 1 targeted retrieval | a short index line in the message (`finance: tariff schedule, 1849`), with the unit in the store |
| **retrieve** | nothing up front | k units at answer time | a BM25 query issued at answer time |
| **discard** | 0 | cannot be recovered | the unit is in neither the message nor the store |

`pointer` is the most interesting action. It spends a little context now so that later
retrieval is *targeted* rather than blind. If the main result turns out to be "retrieval beats
preservation", the pointer arm tells apart "recall is cheap" and "recall is cheap because the
writer left a map".

**Retrieval is real IR, not a simulation.** `src/retrieval.py` already provides Okapi BM25 with
no external dependencies (`BM25Index.top_k`), and Experiments 3 and 4 use it. The store is a
BM25 index over the externalized units. At answer time the live question is the query, and the
top k units are returned.

## 5. Policies

These cover every control the brief asks for:

| Arm | Role |
|---|---|
| `oracle_exact` | knows the actual future question; upper bound |
| `oracle_aspect` | `P̂ = P`, a perfect aspect-level prediction |
| `blind` | no information about the future (the same as Exp 10's `generic`); the no-prediction baseline |
| `static_compress` | the Exp 10/11 policies without retrieval; the existing frontier |
| `predicted_τ` | `P̂` diluted to level τ |
| `displaced_σ` | `P̂` shifted systematically to level σ |
| `wrong` | `P̂` puts its mass on an aspect that `P` gives about zero; the adversarial extreme |
| `uncertain` | a uniform `P̂`, the maximum-entropy hedge |
| `retrieval_only` | a minimal context focused on the current task, with everything else externalized |
| `pointer_hybrid` | task context plus pointers plus retrieval |
| `uncertainty_aware` | chooses based on `H(P̂)`: concentrate, hedge or externalize |

`blind` is the baseline for `V_anticipation`, and `static_compress` is the baseline for testing
whether the frontier moves. They answer different questions and should not be mixed up.

## 6. Noise model

There are two **independent** knobs, so that "very uncertain" and "systematically wrong" are
never confused. The brief treats them as different situations, so they are varied separately:

- **Dilution** `τ ∈ {0, .25, .5, .75, 1}`: `P̂ = (1−τ)P + τ·Uniform`. This raises entropy
  without adding bias. τ=0 is the oracle and τ=1 is the uniform hedge.
- **Displacement** `σ ∈ {0, .25, .5, .75, 1}`: `P̂ = (1−σ)P + σ·P_wrong`, where `P_wrong` puts
  all its mass on the aspect with the least true mass. This adds systematic bias at roughly
  constant entropy.

**Mismatch measure:** total variation `TV(P̂, P)` is the primary measure. It lies in [0,1] and
reads directly as "how much probability sits in the wrong place". Jensen–Shannon divergence is
a secondary measure. Reporting both guards against a threshold that comes from the choice of
divergence rather than from the policy.

Five levels per knob are enough to fit a piecewise-linear breakpoint test and to tell smooth
degradation apart from a threshold.

**Both knobs interpolate linearly, so `TV` is exactly linear in τ and in σ.** This is checked
as a regression test, not just claimed. It matters more than it might seem: any curvature or
threshold that later shows up in regret against mismatch has to come from the *policy*, and
cannot come from how the noise was parameterised. Without that guarantee a threshold finding
could not be interpreted.

**A constraint found while building.** The two knobs do not cover the same range.
`TV(P, dilute(P,τ)) = τ·TV(P, U)`, so dilution tops out at the distance between the true
distribution and uniform, while displacement always reaches `TV = 1`. For the test distribution
`P = (.5,.3,.2,0)`, sweeping τ covers only `TV ∈ [0, 0.30]`, while σ covers `[0, 1.0]`. This
has two consequences:

1. **The true `P` must be peaked**, or the dilution arm says little: a nearly uniform `P` leaves
   τ almost nothing to sweep. Primary `P` distributions should have `TV(P, U) ≥ 0.3`.
2. Regret must be plotted against **`TV`, not against the knob**, and the two noise families
   must look different on the plot, because the same knob setting does not mean the same
   mismatch. Pooling them on the knob axis would create a fake discontinuity where the two
   ranges happen to meet.

## 7. Quantities to measure

In addition to `U_now`, `U_future` and communication cost:

- **`V_anticipation` = U(π_P̂) − U(π_blind)**, plotted against `TV(P̂,P)`. The headline number
  is where it **crosses zero**: the amount of mismatch at which anticipating does worse than
  not anticipating.
- **Prediction-error regret** `R_pred = U(π_{P̂=P}) − U(π_P̂)`: the regret caused by prediction
  error alone, with the policy class held fixed.
- How often retrieval is used, and what it costs.
- Total context used over the whole task, not per message.
- Cumulative utility over a task of k steps.

## 8. Does retrieval move the frontier outward?

This is the key test, and it must not be blurred into "a different point on the same curve".
Measure it exactly as Experiment 11 does: the **paired incremental hypervolume** gained by adding
the retrieval-enabled policies to the set of static-only outcomes, with the context-clustered
bootstrap in `pareto_frontier.py`. Claim an outward shift only where the lower bound of the
paired ΔHV excludes zero, and report the sensitivity grid over metrics and distributions
alongside it. Experiment 11's split of 5/9 against 0/4 is the warning here.

**Resolved (2026-09-07).** `pareto_frontier.py` provides `hypervolume_2d` and
`_normalised_3d_coordinates`, which handle 2-D and 3-D but not 4-D. Rather than extend it,
**cost is combined into one number**:

```
C = β·C_context + γ·C_retrieval
```

This keeps Experiment 11's frontier, ΔHV and bootstrap code working unchanged, and it turns β
and γ from hidden implementation details into stated, reported assumptions. The ratio γ/β is
swept over a grid fixed in advance. It answers the question "how many words of context is one
retrieval call worth?", and since that depends on the harness, the result should be reported as
a function of the ratio rather than at one arbitrary value.

## 9. Calibrating λ without new API calls

The brief asks whether λ can be learned instead of fixed. An exact version is cheap here,
because **every message is already answered on every question**, so the full utility matrix
exists after the fact.

**Stage A (analysis only, no generation):** for any assumed `P`, compute each policy's realised
objective and solve the inverse problem: find the `λ*` under which the best observed policy
would have been chosen. Then check whether `λ*` is stable across dossiers, budgets and choices
of `P`. This reuses `pareto_frontier.lambda_argmax`, which already returns *all* maximisers and
keeps ties, so the inversion does not depend on ordering.

A learned-λ version, fitted on half the dossiers and evaluated on the other half, is the obvious
next step and also needs no new generation.

## 10. Staging and cost

Cost scales as messages × 16 questions × (answer + judge), so the grid gets expensive quickly.
It is therefore run in stages:

- **Stage A, free.** Inverse optimisation for λ* and a *simulation* of retrieval over the
  existing Exp 10/11 messages and the already-labelled units. No new calls. This answers the
  calibration question and lowers the risk in the retrieval implementation.
- **Stage B, small.** The retrieval-enabled arms at one budget, sweeping only τ.
- **Stage C, full.** The τ × σ grid plus `pointer_hybrid` and `uncertainty_aware`.

For scale: Experiment 10 produced 7,104 answers for **$0.145** including the judge, and
Experiment 11's largest cell ran 7,168 answers in about 11 minutes. Stage C, at roughly 1,500
messages, means about 25k answers: on the order of $0.50–1.00 plus the judge, and perhaps 40
minutes of wall time per phase. That is affordable, but the new config should set its own
`cost.cap_usd` rather than inherit Experiment 11's. A `--dry-run` prices the run exactly before
anything is executed.

## 11. Integration

- A new runner, `src/run_anticipatory_context.py`, with the config
  `anticipatory_context_config.yaml`.
- Reused without changes: `budget.py` (word band, hard cap), `retrieval.py` (BM25),
  `regret_data.py` (relations), `pareto_frontier.py` (frontier, ΔHV, `lambda_argmax`),
  `model_pool.py` (shared cache and cost ledger), and the `gpt-4o-mini` judge.
- Write the existing manifest schema so that `render_frontier_study.py` finds the cells
  automatically. It already lists cells that have not run as pending, so partial progress
  renders cleanly.
- An offline selftest, `src/selftest_anticipatory_context_offline.py`, covering the
  unit-labelling check, the probability arithmetic of both noise models, the TV and JS
  computations, deterministic retrieval, and cost accounting.

## 12. Surprising outcomes, stated in advance

These are written down before running, so that a null result still says something:

- `V_anticipation < 0`: anticipating wrongly is worse than not anticipating. Expected at high σ.
- Keeping more context *lowers* utility, because it distracts the answerer.
- Retrieval beats proactive preservation once `H(P̂)` is high.
- Degradation in `TV` has a threshold rather than a smooth decline.
- **A more accurate prediction hurts.** This could happen if a sharp `P̂` pushes the anchor
  answer out of a tight budget. It is the least intuitive prediction, and the most interesting
  if it holds.

As everywhere in this repository, an interval that contains zero is inconclusive, however large
the point estimate.

## 13. Risks

1. ~~Labelling units by aspect~~: **resolved** in §3. Built and verified; the aspects partition
   the evidence exactly.
2. ~~Four objectives against a 3-D hypervolume~~: **resolved** in §8, by combining cost into one
   number.
3. **The isolation guarantee is now the riskiest part of the build.**
   `handoffs.orchestrator_answer` (src/handoffs.py:358) rejects anything other than a frozen
   four-string `SealedHandoff` with an explicit `TypeError`, and its docstring says "there is no
   parameter through which a paragraph could arrive". Every earlier result depends on that
   guarantee. Retrieval really does give the reader source text, so the guarantee has to be
   *restated* for this case, not loosened:

   - add a frozen `SealedContext(qid, question, message_text,
     retrieved_text: tuple[str, ...], policy)` and a matching
     `orchestrator_answer_with_context`; **do not** widen `SealedHandoff` or relax its type
     check;
   - run retrieval *before* sealing, in a component that receives only the question and the
     store, never gold answers and never the full source.

   **What the store holds is part of the policy, not a background resource.** This is the
   subtle way to get it wrong. If the store held the whole source, "externalize everything and
   retrieve with k = all" would just be direct context under another name, and the experiment
   would measure nothing. The store must hold exactly the units the policy chose to
   externalize, and `k` and γ are what keep retrieval from being free. An offline selftest
   should check that no policy's store plus message ever adds up to the full source.
4. **Realistic retrieval queries.** The reader knows the live question, so it queries BM25 with
   it directly. That is realistic for a harness but makes retrieval strong. A control with a
   degraded query is needed so that "retrieval wins" is not just the product of a perfect query.
5. **Cost cap:** a new config with its own cap, never Experiment 11's.

## 14. What the result should settle

Whether managing context with the future in mind is best understood as (1) predicting what will
matter, (2) preserving broadly, (3) keeping discarded evidence retrievable, or (4) switching
between these depending on uncertainty. More specifically, whether the whole research programme
is better framed as **anticipatory retrieval and context management under uncertain future
demand** than as lossy message compression.

The clearest single number is where `V_anticipation` crosses zero. If it crosses at low `TV`,
prediction is fragile and the answer is (3) or (4). If it crosses at high `TV`, or never, then
prediction is robust and the answer is (1).
