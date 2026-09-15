# Core findings and story: task-specific versus reusable agent communication

- **Status:** evidence through 29 August 2026; most benchmark runs remain
  pilot-scale, while Experiments 5b and 8b completed their configured
  20-dossier fictional-QA runs.
- **Scope:** information passed between LLM agents, with agent memory included
  only where it directly bounds the future-query claim.
- **Audit trail:** [full experimental report](HANDOFF_EXPERIMENTS_REPORT.md);
  [focused communication review](../../research-os/outputs/literature-reviews/2026-08-19-multi-agent-handoff-communication-only.md).
- **Search:** primary sources in ACL Anthology, NeurIPS, ICML, ICLR, and arXiv
  were exact-title checked through **26 August 2026**, covering handoffs,
  failures, formats, routing, topology, repeated transmission, future-query
  memory, and compaction. Unreviewed work is labelled **preprint** or
  **technical note**.

## Paper in one sentence

> **Multi-agent communication is not simply a lossy telephone game. It is a
> trade-off between task-specific and reusable communication: optimizing a
> handoff for the current task can improve immediate usefulness while reducing
> its value for future, unknown tasks.**

## Current research landscape

### What is already established

- **Bounded communication is an information bottleneck.** [Yu et al. (2026,
  preprint)](https://arxiv.org/abs/2607.16133) formalize the trade-off between
  useful context reduction and loss of downstream-relevant information.
  [Ao, Gao, and Simchi-Levi (2026, technical
  note)](https://arxiv.org/abs/2603.26993) characterize communication loss as
  posterior distortion and report severe degradation in bounded delegated
  relays; their downstream stages do not retain the original question.

- **Failures occur at edges and propagate through systems.** [MAST (NeurIPS
  2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html)
  derives 14 failure modes from 150 human-analysed traces and applies them to
  more than 1,600 traces.
  [AgentAsk (ACL
  2026)](https://aclanthology.org/2026.acl-long.1294/) isolates Data Gap,
  Signal Corruption, Referential Drift, and Capability Gap at handoffs and
  repairs selected edges with clarification. [Beyond Frameworks (ACL
  2025)](https://aclanthology.org/2025.acl-long.1037/) directly compares
  governance, interaction, and dialogue-history policies and finds
  instructor-curated context summaries offer a strong quality-efficiency
  balance on distributed-evidence tasks.

- **Topology controls diffusion, not just cost.** [Shen et al. (EMNLP
  2025)](https://aclanthology.org/2025.emnlp-main.623/) show that moderately
  sparse graphs best balance useful propagation against error spread. This
  establishes topology as a separate variable from the fidelity of any one
  message.

- **Repeated text transformation is not a new experimental object.** [Perez et
  al. (ICLR
  2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html)
  run fixed-input LLM “telephone game” chains and find cumulative attraction in
  properties such as toxicity, positivity, difficulty, and length. They do not
  measure evidence-grounded answerability or task-conditioned omission.

### How current work improves communication for the known task

- **Sequential aggregation.** [Chain of Agents (NeurIPS
  2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract-Conference.html)
  lets each worker rewrite the prior message while reading a **new evidence
  chunk**, coupling hop count with evidence acquisition and order.

- **Representation and learned brevity.** [OPTiMACS (Findings ACL
  2026)](https://aclanthology.org/2026.findings-acl.1441/) learns a task-aware
  choice among natural-language and structured formats. [Optima (Findings ACL
  2025)](https://aclanthology.org/2025.findings-acl.601/) trains communication
  to balance current-task performance, token use, and readability.

- **Bypasses and routing.** [MOC (2026,
  preprint)](https://arxiv.org/abs/2606.02359) exposes receivers to raw
  multi-hop ancestor messages and merges them under a token budget, reducing
  dependence on repeatedly paraphrased intermediates.

- **Packets and evidence contracts.** [BANDMAS (2026,
  preprint)](https://arxiv.org/abs/2608.00458) sends typed,
  provenance-bearing packets when their predicted current-task value exceeds
  their cost. [GAVEL (Findings ACL
  2026)](https://aclanthology.org/2026.findings-acl.1789/) binds atomic claims
  to exact evidence units and mechanically checks citations for fact checking.

- **Alternative channels.** [State Delta (EMNLP
  2025)](https://aclanthology.org/2025.emnlp-main.518/) supplements text with
  hidden-state trajectories, supporting the view that natural-language
  serialization can discard reasoning state. It improves immediate task
  utility but sacrifices a purely inspectable text channel.

- **No universal format ranking.** [Handoff Debt (2026,
  preprint)](https://arxiv.org/abs/2606.02875) finds consistent efficiency gains
  from context-bearing coding-agent handoffs, but smaller, model-dependent
  solve-rate differences among raw traces, prose notes, and structured notes.

### Closest precedents for future-query utility

- **The general idea is not new.** [Colaco and Lahjouji (2026,
  preprint)](https://arxiv.org/abs/2607.08032) formalize the cost of irreversible
  query-agnostic compaction before a future query is known and include a small
  repeated-summary experiment. This occupies the broad claim that repeated
  compaction can remove information needed later.

- **Memory systems defer selection until the query arrives.**
  [DeferMem (2026, preprint)](https://arxiv.org/abs/2605.22411) retains broad
  candidates and performs query-time evidence distillation; [LazyMem (2026,
  preprint)](https://arxiv.org/abs/2607.22690) likewise avoids irreversible
  write-time compression that may discard details a future query needs.

- **The same issue appears below the text layer.** [Practical Online KV
  Cache Compaction (2026,
  preprint)](https://arxiv.org/abs/2608.00902) finds that immediate compaction
  often hurts agent performance, while delaying it to observe later queries
  recovers much of the gap.

- **Mechanistic compression loss is prior art.** [The Sleeping Agent
  (2026, preprint)](https://arxiv.org/abs/2608.11775) traces temporal-QA loss to
  gist summaries dropping dates and times. [LoCoMo-Plus (ACL
  2026)](https://aclanthology.org/2026.acl-long.1150/) tests whether
  conversational memories retain latent goals and values that are not
  explicitly queried later.

**Gap after this search:** no located study creates a handoff for known Question
A, evaluates the same sealed message on unannounced Question B, and separates
task-conditioned selection from rewriting and new evidence acquisition. The
contribution is this conjunction—not generic communication loss.

## What these experiments add

- **Known-task compression can denoise.** With the question visible at every
  fixed-evidence hop, semantic correctness is broadly stable through ten
  handoffs and can improve on noisy contexts. For Qwen3-32B on MuSiQue full
  context, depth 10 is **+0.182 F1** and **+0.167 judged accuracy** versus direct
  context. With Llama 3.3-70B, HotpotQA gold-only instead loses **0.107 F1**
  while judged accuracy remains **0.900**, showing lexical drift without a
  corroborated correctness loss.

- **The task contract changes the channel.** Removing only the question from
  compressors reverses the full-context depth-10 pattern: MuSiQue moves from
  **+0.080 to −0.272 F1**, and HotpotQA from **+0.014 to −0.326**. A compressor
  cannot reliably select relevant evidence when relevance is undefined.

- **Task optimization can sharply reduce future reuse.** In one short,
  gold-only SQuAD passage that answers both A and B, conditioning on A gives no
  reliable A gain. Unannounced-B point estimates fall by **0.433–0.600 F1** and
  **0.500–0.600 judged accuracy** across depths 1, 5, and 10; the depth-10 F1
  interval touches zero, while judged harm is corroborated at all three depths.
  At the first handoff, B's answer string survives in **3/10** conditioned
  summaries versus **9/10** generic summaries.

- **The main narrowing occurs at selection, not ordinary rewriting.**
  Pass-through, fact-preserving paraphrase, and generic compression retain
  held-out-B judged accuracy at **0.90–1.00** at the main depths; the
  A-conditioned arm remains at **0.30–0.60**. The omission usually occurs in
  the first task-specific selection step and is then faithfully propagated.

- **A fixed-capacity fictional replication confirms the immediate/reusable
  trade-off.** Experiment 5b holds the channel to exactly four of six evidence
  cards. Conditioning on announced A raises judged A accuracy from **0.675 to
  0.992** (**+0.317** [0.292, 0.333]) but lowers mean accuracy on the five
  unannounced queries from **0.675 to 0.613** (**−0.062** [−0.077, −0.048]).
  At K=2 the trade-off is larger (**+0.608 / −0.147**); at K=6, where every
  valid packet contains every card, the paired-compliant difference is exactly
  zero. Full-card accuracy is 1.000 and closed-book accuracy is 0.000. This
  removes unequal length, benchmark memorization, and hidden-query leakage as
  explanations, while retaining the boundary of 20 invented dossiers.

- **Model composition is not the mechanism either.** Experiment 8 varies which
  model writes each of six handoffs while holding task, evidence, prompts,
  budgets and the final answerer fixed. Five family switches cost **+0.000
  judged accuracy** [−0.176, +0.176] against a start-matched single-family
  control on the conditioning question, at either size tier, on either of two
  fixed answerers. Per edge the change is large — verbatim 5-gram copying falls
  **0.607 → 0.340** and the message *expands* by a quarter — so heterogeneity
  changes the message far more than the answer. That per-edge cost tracks
  *changing model* rather than crossing a family boundary as such: when both
  sides of the comparison already change model, the semantic-preservation gap
  disappears (−0.004 [−0.030, +0.030]) while the lexical gap remains. The one
  contrast that survives
  a degeneracy-sensitivity pass is on the *unannounced* question: **−0.176**
  judged accuracy [−0.353, +0.000] at the small tier, matching a per-edge drop
  in held-out answer-string survival (**0.254 → 0.157**). Directionally, that
  places model heterogeneity on the reusability side of the same asymmetry, not
  the current-task side — but the interval touches zero and needs a larger
  replication before it is claimed.

- **Evidence availability, not a replicated model-size effect, explains the
  focused bottleneck.** Experiment 8b crosses small/large source selectors and
  sealed relays within Llama and Qwen. The early-selector contrast changes sign
  across families (**+0.037 Llama; −0.037 Qwen**), so the run does not support a
  shared size-bottleneck claim. But for 98 matched, schema-valid cases where the
  stage-1 text contained no annotated answer or alias for newly announced B,
  both relay sizes scored **0.000** across 196 evaluations; large-minus-small is
  **0.000 [0.000, 0.000]** in both families, inside the configured ±0.05
  equivalence margin. Restoring B's exact card at the
  same channel width beats a matched sham by **+0.983** for Llama and **+0.950**
  for Qwen; exact restoration and source reopening each beat the sealed arm by
  **+1.000**. Increasing relay tier did not help after this literal text-level
  omission, but the criterion does not establish semantic impossibility and the
  experiment does not show that larger early models omit less.

- **The handoff is addressed to a reader, not only to a task — but the address
  is not a calibration.** Experiment 9 varies only what the writer is told about
  the *next* agent's context size, with each capacity claim appearing twice: once
  as a bare statement, once as that same statement plus one clause asking the
  model to act on it. Merely being told a number, with nothing asked, lengthens
  the handoff (**+119.5** [+99.6, +140.8] counterfactual and **+189.3**
  [+150.5, +229.3] fictional words versus a size-neutral control) and *raises*
  retained reusable facts (**+0.154** and **+0.360**). Being asked to act on the
  same number is a distinct effect with the opposite sign (**−42.8**
  [−66.9, −21.2] and **−87.6** [−145.1, −37.2] words). This is a communication
  variable no other experiment here manipulates. It changes the message, but no
  clean length-increasing comparison improves judged accuracy; fit-2k is 15.4
  points lower than the neutral control on the counterfactual items.

  The mechanism is not that the model targets the stated capacity. Told to fit a
  2k window it goes far below; told to fill a 10k one it either adds a modest
  amount (median 783 words) or runs away entirely — **44%** of "expand to use the
  available space" calls fall into a repetition loop with median 9,687 words and
  **0.912** internal 5-gram repetition. That failure is not a budget artifact: a
  probe at `max_tokens=32,000` reproduced the same split, so raising the ceiling
  only enlarges the loop. Reported means for any expand arm are therefore a
  mixture of two modes and describe neither.

  This finding also exposes a baseline issue for the rest of the project.
  Experiment 5's compression instruction opens "Write **concise** prose research
  notes". Experiment 9 removes that word from its neutral base, then separately
  appends the explicit cue "Keep your handoff concise". That cue changes length
  by **−114.1** [−130.7, −98.6] words and side-fact retention by **−0.277**
  [−0.423, −0.138] against the size-neutral control. Because the measured cue is
  a full appended sentence, this quantifies a concision instruction—not the
  original word in isolation.

  The redundancy an expansion request generates decomposes, and the two-dataset
  design is what makes the decomposition possible. It is *repetition* plus
  *unrelated invention*, not parametric reversion: unsupported terms roughly
  double (2.4 → 5.8 per handoff), almost all wholly invented rather than
  near-miss corruptions of document values; the memorised value enters only
  **1.5%** of expand handoffs against **0.0%** under the size-neutral control;
  and under 2% of 1,818 unsupported counterfactual terms match it at all. The
  decisive test is that fictional items — which have no memorised value that
  *could* resurface — show almost the same invention rate (4.9 vs 5.8). Asking a
  model to fill space makes it fabricate, not remember. A semantic support judge
  separates the two failure modes further: the runaway mode makes *fewer* novel
  unsupported claims (2.80) than well-behaved expansion (3.84), because it pads
  by restating rather than inventing.

  Two effects run opposite to the design's expectations and are worth stating
  plainly. The capacity effect **grows with depth** rather than washing out
  (+71.8 → +92.0 → +119.5 words across depths 1–3), so each rewrite re-applies
  the directive. And a requested *shrink* does not strip reusable facts: against
  a size-neutral control it **raises** side-fact retention (+0.115 to +0.280).
  The fit-2k request does not cause side-fact loss relative to the neutral
  control; the direct “Keep your handoff concise” cue does. Fill-10k cannot be
  assessed cleanly on literal retention because repetition restates every fact.
  In accuracy terms the ranking is unambiguous: handing off at all costs ~10–15
  points against reading the document directly (0.962 → 0.808 counterfactual;
  1.000 → 0.900 fictional), and asking to expand roughly doubles that loss
  (0.654 / 0.600) — the worst condition tested, and the only directive clearly
  worse than saying nothing.

  What Experiment 9 does **not** show: whether expansion recovers compressed-out
  facts. Loss rates of 0–15% left recovery and reversion at 0.000 everywhere they
  are defined, so the large→small→large question remains open and needs a harder
  bottleneck to test.

- **Agent count is not a mechanism.** Experiment 2 holds the source payload
  fixed and varies serial depth without admitting new evidence. Experiment 7
  instead gives specialists new evidence and independently inserts 0/1/3/5
  evidence-free relays; no relay-depth contrast reliably lowers final F1.
  Communication depth, task-conditioned selection, and evidence acquisition
  must be analyzed separately.

## Defensible position

> **Existing work primarily optimizes inter-agent communication for the
> currently known task. We study the complementary problem of communication
> reusability: whether task-optimized handoffs preserve information required by
> unanticipated downstream queries. Under a hard, verified communication budget
> and with every question rotated through the conditioning role,
> task-conditioned selection—rather than rewriting depth or message length—is
> the main observed source of persistent information narrowing, and an oracle
> handoff of the same length carries the hidden answers, so the loss is
> allocation rather than capacity.**

- **Contribution:** test a query-optimized sealed handoff on the queries it was
  not shown, at equal delivered length, with every question taking both roles.
- **Mechanism boundary:** Experiment 5a separates rewriting from selection in a
  natural-prose pilot; Experiment 5b reproduces the trade-off with equal card
  capacity, all hidden questions, fictional facts, and dossier-clustered
  inference.
- **Methodological recommendation:** vary task visibility and relay depth under
  complementary fixed- and incremental-evidence designs; report selection,
  transformation depth, and evidence acquisition separately.

Do **not** claim generic handoff loss; the first repeated relay, future-query
loss, mechanistic compression analysis, or structured/provenance handoff; or a
universal ranking of message formats.

## Evidence posture and next test

- The question-omission result is the strongest matched causal comparison
  (n=30 per dataset, two seeds).
- The natural SQuAD A/B result and rewriting ladder remain pilot-scale
  (`n=10`, one seed, one 8B model). They supply ecological continuity and the
  rewriting-versus-selection ladder, not a population estimate.

- Experiment 5b passed its preregistered-style gates: full-card accuracy 1.000,
  closed-book accuracy 0.000, fixed K, hidden-query harm with dossier-clustered
  intervals, and a zero-difference all-card control. It is stronger causal
  evidence, but still only 20 synthetic dossiers with atomic evidence cards.

- Experiment 8b passed its availability controls but **falsified the simple
  shared size story**: exact restoration and source reopening recover B, while
  selector-size effects do not replicate across two families. The paper should
  claim irreversible narrowing after clean omission, not that small models are
  uniquely responsible for it.

- Experiment 10 supplies the length-controlled test the earlier results could
  not: a two-sided word band (realised fill 0.88-0.96, sd 1-6 words, zero
  over-cap deliveries, repetition <= 0.005) on natural SQuAD prose, with every
  question rotating through the conditioning role. Conditioning raises
  present-query accuracy by +0.625 / +0.448 / +0.281 at 20 / 40 / 80 delivered
  words while lowering hidden-query accuracy by -0.063 / -0.115 / -0.194, both
  directions excluding zero, and the specialisation gap widens by
  +0.413 [0.326, 0.503] as the budget falls from 160 to 20 words. Two findings
  sharpen the claim: an `oracle` handoff of the same ~37 words answers *every*
  question (0.823), so the channel was never the binding constraint; and an
  explicit "further questions may follow" instruction recovers almost none of
  the loss. The verbatim-extraction arms reproduce the pattern, so it is
  selection rather than abstractive rewriting.

- Experiment 10a adds the mechanism's shape. On 16 dossiers whose hidden
  questions carry *designed* distances from the conditioning query, future-query
  regret runs 0.00 at a paraphrase, 0.42-0.95 at same-entity and same-topic, and
  0.95-0.98 at an orthogonal aspect; `R(orthogonal) - R(paraphrase)` is
  0.94-1.00 with dz between 9 and 30. The gradient is absent from `generic`
  (0.04-0.15) and `oracle` (0.06-0.09), so it is a property of conditioning, not
  of the corpus. At a 160-word budget -- a third of the source -- `generic` cuts
  orthogonal regret to 0.477 and `oracle` to 0.160 while `conditioned` stays at
  0.952: a wider channel deepens the aspect the sender already chose rather than
  spreading the message across the source. The paraphrase tier is a design
  sanity check, not a finding, and the corpus supports "near versus far" rather
  than a strict four-point ordering.

- The next decisive test is external validity: a second sender/answerer stack, a
  wider range of source styles and question counts per context, and human
  adjudication of a sample of judge verdicts. `U_future` is also bounded below
  by the generic baseline's own accuracy, which is only 0.240 at the tightest
  budget, so the absolute future loss is compressed exactly where the
  specialisation gap is largest.
