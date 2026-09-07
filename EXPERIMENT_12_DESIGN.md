# Experiment 12 — Anticipatory context management under uncertain future demand

**Status:** foundation built and verified offline; no runs, no API spend.
Supersedes nothing; extends Experiments 10 and 11.

- `src/anticipatory_data.py` — evidence units, aspect labelling, the coverage
  gate, both noise models, and the two mismatch measures.
- `src/selftest_anticipatory_offline.py` — 32 checks against the real corpus,
  all passing, no API key required.
- All three open items from the first draft are resolved; see §3, §6 and §8.

## 1. The question

Experiments 10/11 measured a *static, irreversible* handoff: one message, one
budget, no recourse. The specialization cost is established — conditioning on
the known query buys `U_now` and sells `U_future`, and forcing an allocation
quota α controls the trade-off while stating a coefficient λ does not.

This extension asks the harness-level question: **if you must decide what to
preserve before you know what will be needed, and you may keep discarded
evidence retrievable, what should you do?** Concretely, it separates the true
future-use distribution `P(Q_future)` from the harness's estimate `P̂`:

```
max_π  U_now + λ·E_{Q~P̂}[U_future] − β·C_context − γ·C_retrieval
```

The scientific target is not the optimum of that objective. It is the shape of
the loss as `P̂` degrades away from `P`, and whether retrieval changes the
achievable set rather than the operating point on it.

## 2. Substrate: why `relation_dossiers`

`data/communication_regret/relation_dossiers.jsonl` — 17 dossiers (16 used),
each a ~2,850-character fictional source with **16 questions on a fully
crossed 4 × 4 grid**:

- **4 aspects** — `founding`, `facilities`, `finance`, `custom`
- **4 roles** — `anchor`, `paraphrase`, `same_entity`, `same_topic`

This is the right substrate for three reasons:

1. **Demand is already modelled at evidence-unit level.** The user requirement
   to model future demand over facts/topics rather than exact natural-language
   questions is satisfied by the aspect axis. `P(Q_future)` is a distribution
   over **4 aspects** — a 4-simplex, small enough to sweep systematically and
   large enough for a divergence to be meaningful.
2. **The existing relation taxonomy is already the aspect structure.**
   `regret_data.relation_label` returns `paraphrase`/`same_entity`/`same_topic`
   for same-aspect questions and `orthogonal` for cross-aspect ones. Nothing
   new is needed to label a future question's relation to the conditioning one.
3. **The channel is the only path to the answer.** Closed-book accuracy is
   0.000 and the direct-context ceiling is 0.992, so any measured utility is
   attributable to what was communicated or retrieved, not to parametric
   knowledge.

`squad_groups` is retained as a secondary regime but has no aspect labelling,
so it can only support the no-prediction and uniform-`P̂` arms. It is a
robustness check, never a primary cell.

## 3. Evidence units

Chunk each source into sentences via `data.split_sentences` (already used by
`budget.truncate_to_words`). Label each unit with the aspect(s) it supports:

- primary signal: gold-answer string containment for that aspect's questions;
- secondary signal: BM25 alignment of the unit against that aspect's four
  questions, using `retrieval.BM25Index`.

Acceptance gate, enforced by an offline selftest: **every question's gold must
be recoverable from at least one labelled unit**, and every aspect must own at
least two units. A dossier failing the gate is dropped and logged, never
silently repaired.

**Resolved (2026-09-07).** The dossier `meta` carries only `source_words`,
`source_characters` and `subject` — no aspect spans — so unit labelling is a
build step, now implemented in `anticipatory_data.extract_units`.

Measured over the full corpus, the substrate is cleaner than the design
assumed:

| | |
|---|---|
| questions with gold located verbatim in a sentence | **256 / 256 (100%)** |
| sentences (evidence units) | 264 across 16 dossiers, 14–20 each |
| claimed by exactly one aspect | 191 (72%) |
| **claimed by two or more aspects** | **0** — mean pairwise aspect Jaccard 0.000 |
| unclaimed background | 73 (28%) |
| (dossier, aspect) cells with ≥2 units | **64 / 64**, min 2, mean 3.0 |

Gold containment alone gives complete coverage, so BM25 alignment is demoted to
a cross-check and is not needed for labelling. The aspect partition of the
evidence is exactly disjoint, which is what licenses an aspect-level
`P(Q_future)` in the first place: predicting "finance" names a real, separable
subset of the source rather than a soft topic label.

Disjointness is enforced rather than assumed — `extract_units` raises if any
sentence is ever claimed by two aspects, so a future corpus that breaks the
property fails loudly instead of quietly degrading the experiment.

## 4. Action space

Six actions, each with a concrete implementation and an explicit cost. This is
the move beyond "one message, take it or leave it".

| Action | `C_context` | `C_retrieval` | Implementation |
|---|---|---|---|
| **keep** | full unit words | 0 | verbatim sentence in the message |
| **compress** | ~α × unit words | 0 | LLM-compressed unit, same word contract as Exp 10 |
| **externalize** | 0 | 1 retrieval | unit written to the store, absent from the message |
| **pointer** | 4–8 words | 1 retrieval, targeted | short index line (`finance: tariff schedule, 1849`) in the message, unit in the store |
| **retrieve** | 0 up front | k units at answer time | BM25 query issued at answer time |
| **discard** | 0 | unrecoverable | unit in neither message nor store |

`pointer` is the scientifically interesting action: it spends a little context
now to make later retrieval *targeted* rather than blind. If the headline
result is "retrieval dominates preservation", the pointer arm is what
distinguishes "cheap recall" from "cheap recall because you left a map".

**Retrieval is real IR, not simulation.** `src/retrieval.py` already provides
dependency-free Okapi BM25 (`BM25Index.top_k`) and is used by Experiments 3
and 4. The store is a BM25 index over externalized units; retrieval at answer
time issues the live question as the query and returns top-k.

## 5. Policies

Covering every control the brief requires:

| Arm | Role |
|---|---|
| `oracle_exact` | knows the actual future question — upper bound |
| `oracle_aspect` | `P̂ = P` — perfect aspect-level prediction |
| `blind` | no future information (= Exp 10 `generic`) — the no-prediction baseline |
| `static_compress` | Exp 10/11 policies, no retrieval — the existing frontier |
| `predicted_τ` | `P̂` diluted at level τ |
| `displaced_σ` | `P̂` systematically shifted at level σ |
| `wrong` | `P̂` mass on an aspect `P` gives ≈0 — the adversarial endpoint |
| `uncertain` | `P̂` = uniform — the max-entropy hedge |
| `retrieval_only` | minimal task-focused context, everything else externalized |
| `pointer_hybrid` | task context + pointers + retrieval |
| `uncertainty_aware` | policy switches on `H(P̂)`: concentrate / hedge / externalize |

`blind` is the reference for `V_anticipation`; `static_compress` is the
reference for the frontier-shift test. They are different baselines and must
not be conflated.

## 6. Noise model

Two **orthogonal** knobs, so "high uncertainty" and "systematic shift" are
never confounded — the brief treats them as distinct regimes and they must be
manipulated separately:

- **Dilution** `τ ∈ {0, .25, .5, .75, 1}` — `P̂ = (1−τ)P + τ·Uniform`.
  Raises entropy, adds no bias. τ=0 is the oracle, τ=1 the uniform hedge.
- **Displacement** `σ ∈ {0, .25, .5, .75, 1}` — `P̂ = (1−σ)P + σ·P_wrong`,
  where `P_wrong` concentrates on the aspect with least true mass.
  Systematic bias at roughly constant entropy.

**Mismatch measure:** total variation `TV(P̂, P)` as primary — bounded in
[0,1] and directly readable as "probability mass in the wrong place" —
with Jensen–Shannon as a secondary. Reporting both guards against a threshold
that is an artefact of the divergence rather than a property of the policy.

Five levels per knob is enough to fit a piecewise-linear break test and
distinguish smooth degradation from a threshold.

**Both knobs are linear interpolations, so `TV` is exactly linear in τ and in
σ** — verified as a regression check, not just asserted. This matters more than
it looks: it means any curvature or threshold later observed in
regret-vs-mismatch is a property of the *policy*, and cannot be an artefact of
how the noise was parameterised. Without that guarantee a "threshold" finding
would be uninterpretable.

**Constraint discovered while building.** The two knobs do not span the same
range. `TV(P, dilute(P,τ)) = τ·TV(P, U)`, so dilution saturates at the distance
from the truth to uniform, whereas displacement always reaches `TV = 1`. On a
test distribution `P = (.5,.3,.2,0)` the τ sweep covers only `TV ∈ [0, 0.30]`
while σ covers `[0, 1.0]`. Two consequences:

1. **The true `P` must be peaked**, or the dilution arm is uninformative — a
   near-uniform `P` leaves τ with almost no range to sweep. Choose primary
   `P` distributions with `TV(P, U) ≥ 0.3`.
2. Regret must be plotted against **`TV`, not against the knob**, and the two
   noise families must be visually distinguished, since equal knob settings
   are not equal mismatches. Pooling them on the knob axis would fabricate a
   discontinuity where the families' ranges happen to meet.

## 7. Quantities to measure

Beyond `U_now`, `U_future`, and communication cost:

- **`V_anticipation` = U(π_P̂) − U(π_blind)**, plotted against `TV(P̂,P)`.
  Its **zero crossing** is the headline: the mismatch level at which
  anticipating becomes worse than not anticipating.
- **Prediction-error regret** `R_pred = U(π_{P̂=P}) − U(π_P̂)` — regret
  attributable specifically to prediction error, policy class held fixed.
- Retrieval frequency and retrieval cost.
- Total context consumed over the task (not per message).
- Cumulative utility over a k-step task.

## 8. Does retrieval move the frontier outward?

The critical test, and it must not be fudged into "a different point on the
same curve". Operationalise exactly as Experiment 11 already does: **paired
incremental hypervolume** from adding the retrieval-enabled policy set to the
static-only achievable set, with the context-clustered bootstrap in
`pareto_frontier.py`. Claim an outward shift only where the paired ΔHV lower
bound excludes zero, and report the metric × distribution sensitivity grid
alongside — Experiment 11's 5/9 versus 0/4 split is the cautionary precedent.

**Resolved (2026-09-07).** `pareto_frontier.py` provides `hypervolume_2d` and
`_normalised_3d_coordinates` — 2-D and 3-D only, no 4-D path. Rather than
extend it, **cost is scalarised**:

```
C = β·C_context + γ·C_retrieval
```

This keeps Experiment 11's frontier, ΔHV and bootstrap machinery working
unchanged, and converts β/γ from hidden implementation detail into a declared,
reportable assumption. The ratio γ/β is swept as a prespecified sensitivity
grid — it is the "how expensive is a retrieval call, in words of context?"
question, and the honest answer is that it depends on the harness, so the
result should be reported as a function of it rather than at one arbitrary
setting.

## 9. Calibrating λ without new API calls

The brief asks whether λ can be learned rather than fixed. There is a cheap,
exact version available because **every message is already answered on every
question**: the full utility matrix exists post hoc.

**Stage A (analysis only, zero generation):** for any assumed `P`, compute each
policy's realised objective and solve the inverse problem — recover the `λ*`
under which the observed best policy would have been selected. Then test
whether `λ*` is stable across dossiers, budgets, and `P`. This reuses
`pareto_frontier.lambda_argmax`, which already returns *all* maximisers with
ties preserved, so the inversion is well-posed rather than order-dependent.

A learned-λ variant (fit on half the dossiers, evaluate held-out) is the
natural follow-up and needs no new generation either.

## 10. Staging and cost

Costs scale as messages × 16 questions × (answer + judge), so the grid
multiplies fast. Stage it:

- **Stage A — free.** λ* inverse optimisation and retrieval *simulation* over
  the existing Exp 10/11 messages and the already-labelled units. No new calls.
  Answers the calibration question and de-risks the retrieval implementation.
- **Stage B — small.** Retrieval-enabled arms at one budget, τ sweep only.
- **Stage C — full.** τ × σ grid plus `pointer_hybrid` and `uncertainty_aware`.

For scale: Experiment 10 produced 7,104 answers for **$0.145** including the
judge, and Experiment 11's largest cell ran 7,168 answers in ~11 minutes.
Stage C at roughly 1,500 messages implies ~25k answers — order $0.50–1.00 plus
judge, and perhaps 40 minutes of wall time per phase. Affordable, but set a
fresh `cost.cap_usd` in the new config rather than inheriting Experiment 11's.
A `--dry-run` will price it exactly before anything executes.

## 11. Integration

- New runner `src/run_anticipatory_context.py`, config
  `anticipatory_context_config.yaml`.
- Reuse without modification: `budget.py` (word band, hard cap),
  `retrieval.py` (BM25), `regret_data.py` (relations), `pareto_frontier.py`
  (frontier, ΔHV, `lambda_argmax`), `model_pool.py` (shared cache + ledger),
  the `gpt-4o-mini` judge.
- Emit the existing manifest schema so `render_frontier_study.py`'s artifact
  discovery picks the cells up automatically — it already lists unrun cells as
  pending, so partial progress renders cleanly.
- Offline selftest `src/selftest_anticipatory_context_offline.py`, covering the
  unit-labelling gate, both noise models' simplex arithmetic, the TV/JS
  computation, retrieval determinism, and the cost accounting.

## 12. Pre-registered surprising regimes

State these before running, so a null is informative:

- `V_anticipation < 0` — wrong anticipation worse than none. Expected at high σ.
- More preserved context *reducing* utility, via distraction of the answerer.
- Retrieval dominating proactive preservation once `H(P̂)` is high.
- Threshold rather than smooth degradation in `TV`.
- **More accurate prediction hurting** — plausible if a sharp `P̂` crowds the
  anchor answer out of a tight budget. This is the least intuitive prediction
  and the most interesting if it holds.

Repo norm applies throughout: an interval containing zero is inconclusive,
however large the point estimate.

## 13. Risks

1. ~~Aspect→unit labelling~~ — **resolved**, §3. Built and verified; the
   partition is exactly disjoint.
2. ~~Four objectives vs 3-D hypervolume~~ — **resolved**, §8. Cost scalarised.
3. **The isolation guarantee is now the single most dangerous part of the
   build.** `handoffs.orchestrator_answer` (src/handoffs.py:358) rejects
   anything but a frozen four-string `SealedHandoff` with an explicit
   `TypeError`, and its docstring states "there is no parameter through which a
   paragraph could arrive". AGENTS.md calls this load-bearing. Retrieval
   genuinely does hand the reader source text, so the invariant must be
   *restated*, not relaxed:

   - add a frozen `SealedContext(qid, question, message_text,
     retrieved_text: tuple[str, ...], policy)` and a matching
     `orchestrator_answer_with_context`; **do not** widen `SealedHandoff` or
     loosen its type check;
   - retrieval runs *before* sealing, in a component whose inputs are the
     question and the store — never gold answers, never the full source.

   **The store's contents are part of the policy, not a background resource.**
   This is the subtle failure mode: if the store held the whole source, then
   "externalize everything, retrieve k=all" would simply be direct context
   under another name, and the experiment would measure nothing. The store
   must contain exactly the units the policy chose to externalize, and `k`
   and γ are what stop retrieval from being free. An offline selftest should
   assert that no policy's store plus message ever reconstitutes the full
   source.
4. **Retrieval query realism.** The reader knows the live question, so BM25
   queries it directly. That is realistic for a harness but makes retrieval
   strong; include a degraded-query control so "retrieval wins" is not an
   artefact of a perfect query.
5. **Cost cap** — new config, fresh cap, never inherit Experiment 11's.

## 14. What the result should settle

Whether future-aware context management is best understood as (1) predicting
what matters, (2) preserving broadly, (3) keeping discards retrievable, or (4)
adapting between these by uncertainty — and specifically whether the honest
framing of the whole research programme is **anticipatory retrieval and context
management under uncertain future demand** rather than lossy message
compression.

The `V_anticipation` zero crossing is the sharpest single number: if it sits at
low `TV`, prediction is fragile and the answer is (3) or (4); if it sits high
or never crosses, prediction is robust and the answer is (1).
