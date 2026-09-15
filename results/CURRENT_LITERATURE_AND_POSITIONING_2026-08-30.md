# Current literature landscape and research positioning

> **Superseded on 31 August 2026.** This file is the pre-Experiment-9 snapshot,
> so its “incomplete” warnings are historical. Use
> [PAPER_DRAFT.md](../PAPER_DRAFT.md) for the current paper framing and
> [HANDOFF_EXPERIMENTS_REPORT.md](HANDOFF_EXPERIMENTS_REPORT.md) for the audited
> results.

**Cut-off:** 30 August 2026  
**Project:** Agent Handoff Information-Loss Probe  
**Evidence base:** repository results through Experiment 8b; Experiment 9 is still incomplete  
**Source policy:** primary paper pages and the repository's completed Markdown reports; peer-reviewed work is distinguished from preprints

## Executive assessment

The project remains well positioned, but its defensible novelty is narrower and more interesting than “agent handoffs lose information.” That broad territory is now crowded. Existing work already studies bounded communication, sealed two-agent compression, repeated relay drift, message formats, model switching, future-query memory, structured repair, and provenance-aware fallback.

The strongest position is **communication reusability under task-conditioned selection**:

> When a handoff is written for a currently known Question A, the writer can improve the message's immediate usefulness for A by selecting away information needed for unannounced Question B. This creates a measurable future-query externality even when source evidence, payload capacity, and the downstream reader are held fixed.

I found no prior work that combines all of the following in one controlled design:

1. The writer sees a known target task A.
2. The source evidence is fixed and provided directly, so retrieval is not a confound.
3. Only the exact sealed message reaches the downstream reader.
4. That same message is scored both on A and on an unannounced B.
5. Payload capacity is held exactly constant in the causal replication.
6. Rewriting, task-conditioned selection, and later restoration are separated experimentally.
7. Fictional evidence rules out recovery from parametric memory.

The best paper story is therefore not “telephone-game degradation.” It is:

> **Handoffs can denoise for the task they are written for, but task-aware denoising narrows what the communication remains useful for. The decisive loss occurs at selection, not ordinary rewriting; once answer-bearing evidence is genuinely absent from a sealed message, later or larger relays cannot reconstruct it without renewed access to evidence.**

The result hierarchy should be:

- **Headline causal evidence:** Experiment 5b's equal-slot fictional A-versus-hidden-query trade-off.
- **Mechanism localization:** Experiment 5a's rewriting ladder and Experiment 8b's exact restoration control.
- **Motivation:** Experiments 1, 2, and 2b show why task-aware compression can denoise and why task information matters.
- **Robustness/qualification:** Experiment 8 shows that changing writer family and size strongly changes message form without necessarily changing target accuracy.
- **Separate work in progress:** Experiment 9 should not enter the findings until its replacement run is complete and audited.

## What the field now knows

### 1. Bounded relays create a denoising-versus-loss trade-off

[Ao, Gao, and Simchi-Levi (2026)](https://arxiv.org/abs/2603.26993) model delegated agents as a bounded decision network. Communication loss can be expressed through posterior divergence, including conditional mutual information under log loss. [Yu et al. (2026)](https://arxiv.org/abs/2607.16133) make the complementary empirical point: bounded relay messages can help when they remove redundant context, but the benefit reverses when they remove task-relevant signal.

This theory fits this repository's early results unusually well. A free-form handoff can beat direct long context because the writer performs useful evidence selection. But this literature supplies the broad information-bottleneck framing already; the project's contribution must be the *identity of the lost utility* and the design that localizes it.

One important comparator is [Du et al. (Findings of EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1264/), who show performance falling by 13.9%–85% as input grows even with perfect retrieval. Consequently, `A_full` is not an information-theoretic accuracy ceiling: a shorter handoff may outperform it without creating information. The gold-evidence oracle is the cleaner upper-bound input condition.

### 2. Handoff failures are established, and some are repairable

[AgentAsk (ACL 2026)](https://aclanthology.org/2026.acl-long.1294/) identifies Data Gap, Signal Corruption, Referential Drift, and Capability Gap at inter-agent edges, then improves accuracy by up to 4.69% through targeted clarification. [MAST (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html) supplies a broad empirical failure taxonomy over more than 1,600 multi-agent traces. [Governance at the Boundary (2026 preprint)](https://arxiv.org/abs/2608.16055) shows that policy facts can be attenuated specifically at component boundaries and that the effect depends strongly on model capability.

The closest new result is [When “Must” Becomes “Maybe” (25 August 2026 preprint)](https://arxiv.org/abs/2608.24569). It restricts an executor to a transformed artifact and separates topical containment from the operational force of a known constraint. Across 1,296 controlled episodes, normal compression deactivated every tested blocker and led to forbidden action in 54.2%; restoring four explicit state fields recovered full preservation. This paper rules out broad “first causal handoff-loss” or “first structured restoration” claims.

The distinction is still clear: that paper asks whether a *known constraint* remains binding. This project asks whether optimizing for *one known task* destroys factual utility for *other, unannounced tasks*.

### 3. Repeated rewriting is not inevitably destructive

[When LLMs Play the Telephone Game (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html) establishes iterated transformation and attractor effects in properties such as toxicity, positivity, difficulty, and length. It does not evaluate evidence-grounded answerability.

More directly, [Faithful, Not Corrective (2026 preprint)](https://arxiv.org/abs/2607.09678) tracks 12 programmatically generated facts through six hops and five formats. Strong relays are nearly lossless; weak relays show a large format spread, and injected wrong values persist in 83%–100% of chains. This occupies format-by-depth, fact survival, capability moderation, and error-persistence territory.

That result strengthens rather than weakens this project's mechanism claim. The repository's Experiment 5a likewise finds pass-through, paraphrase, and generic compression preserve hidden-B correctness at 0.9–1.0, while A-conditioned selection falls to 0.3–0.6. The defensible conclusion is **not** that repeated rewriting generally destroys information. It is that *task-conditioned selection at the first edge* can sharply narrow the message, after which later relays faithfully transmit the narrowed state.

### 4. Long-context agent systems mix handoff with evidence acquisition

[Chain of Agents (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract-Conference.html) is the main relay-style long-context precedent. Sequential workers rewrite a communication unit while each also receives a new document chunk. It can outperform full-context and retrieval baselines, but relay depth, evidence acquisition, and document order are coupled.

The repository's fixed-evidence chains and sealed `SealedHandoff` isolate the transformation itself. Experiment 7 then deliberately reintroduces fresh evidence and finds that evidence-free relay count alone need not lower final accuracy. This is useful support for a careful claim: **agent count is not a mechanism; the information operations at edges are.**

### 5. The format and interface of a handoff matter, but no format is universally best

[State Compression in Two-Agent LLM Relays (2026 preprint)](https://arxiv.org/abs/2607.18265) is a particularly close single-handoff comparator. In closed-world travel tasks, a downstream Booker sees the goal and handoff but not the inventory. Narrative summarization scores 0.48 feasibility, schema-constrained JSON 0.96, and uncompressed state 0.88. This already establishes sealed compression loss and a structured representation advantage in a constraint task.

[Routed Graph Handoff (accepted EMNLP 2026)](https://arxiv.org/abs/2608.25277) goes further: a router chooses a typed dependency graph or prose for each delegation. Across more than 1,050 trajectories, routing improves some tasks at 2.2–3.2× compression while avoiding a 14.6-point graph-only regression on AppWorld. The receiver also needs graph-aware instructions. [OPTiMACS (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1441/) learns task-aware message representations, while [State Delta (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.518/) augments text with latent transition information.

Other current systems already occupy adjacent intervention space. [BANDMAS (2026 preprint)](https://arxiv.org/abs/2608.00458) schedules evidence/request packets under bandwidth, latency, deadline, and receiver-context constraints; [MOC (2026 preprint)](https://arxiv.org/abs/2606.02359) carries multi-order evidence across nonlocal paths; and [GAVEL (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1789/) binds atomic subclaims to auditable evidence units. These are useful mitigation baselines, not novelty targets.

These works mean the project should not claim that prose is uniquely lossy, structured messages are new, or adaptive formatting is new. Its question is orthogonal: **whatever interface is selected for current-task performance, how much future-query utility does it leave behind?**

### 6. Coding-agent takeovers measure continuity and rediscovery, not sealed evidence survival

[Handoff Debt (2026 preprint)](https://arxiv.org/abs/2606.02875) freezes repository state and compares repository-only, raw-trace, free-note, and structured-note takeovers. Context-bearing handoffs reduce successor events and prompt tokens substantially, although solve-rate effects are smaller and model dependent.

[The Handoff Tax (25 August 2026 preprint)](https://arxiv.org/abs/2608.24358) studies escalation and downshift between weaker and stronger coding models. It compares full trajectories, compacted trajectories, and trajectory removal while the repository remains available. Escalation recovers less than half the low-to-high capability gap, and the preferred interface reverses with switch direction.

These papers occupy takeover cost, model switching, and direction-dependent interface effects. They do not isolate an irreversible evidence channel: the code repository persists, the receiver changes, and the receiver continues acting. Experiment 8's narrower contribution is a fixed answerer with fixed evidence and prompts while only the sequence of handoff writers changes. Even there, the present result is chiefly a qualification: heterogeneous writer sequences change form more reliably than they change target correctness.

### 7. Future-query memory is the closest conceptual neighborhood

[What to Keep, What to Forget (2026 preprint)](https://arxiv.org/abs/2607.08032) frames memory compaction as rate–distortion and explicitly identifies the danger of discarding information before the query is known. It also notes that repeated compaction is rarely measured and supplies a small reference experiment. Therefore, neither “first future-query compaction study” nor “first repeated-compaction study” is safe.

[MEMAUDIT (2026 preprint)](https://arxiv.org/abs/2605.02199) is the closest evaluation design. It freezes an experience stream, candidate memory representations, storage costs, semantic evidence units, future-query requirements, and a budget, then computes certified selection optima. It cleanly separates the quality of memory writing from retrieval and downstream reading.

[TierMem (2026 preprint)](https://arxiv.org/abs/2602.17913) calls this the “write-before-query barrier” and responds with provenance-linked summaries plus immutable raw-log fallback. [DeferMem](https://arxiv.org/abs/2605.22411) and [LazyMem](https://arxiv.org/abs/2607.22690) similarly defer irreversible selection until more is known about the future request. [The Sleeping Agent (2026 preprint)](https://arxiv.org/abs/2608.11775) shows one concrete loss mode—temporal expressions disappearing from gist summaries—and a targeted preservation instruction that improves temporal QA.

At a lower representation layer, [Practical Online KV Cache Compaction (2026 preprint)](https://arxiv.org/abs/2608.00902) reports that immediate compaction is often harmful while delaying it until a future query is available recovers much of the gap. Its mechanism is KV-cache selection rather than inspectable inter-agent text, but the timing result supports the same high-level principle: irreversible selection is safer after the consumer's needs are known.

The project's distinction from memory work is **task-conditioned opportunity cost**. Memory writers are usually query-blind and assessed against future needs. Here the writer is *not* query-blind: it knows A, and the experiment asks what A-optimization costs B. That is a different and important failure mode because real orchestrators routinely give specialists a narrow task while later reusing their notes.

### 8. Query-focused summarization explains the mechanism but usually scores the same query

The older summarization literature already treats selection as query dependent. [Xu and Lapata (ACL-IJCNLP 2021)](https://aclanthology.org/2021.acl-long.475/) decompose query-focused summarization into evidence selection and conditional generation; their [TACL 2022 work](https://aclanthology.org/2022.tacl-1.36/) treats even generic summaries as responses to latent queries.

This is the conceptual bridge for the paper: a handoff prompt induces a query-conditioned sufficient statistic. Standard query-focused summarization asks whether that statistic serves the same query. This project asks whether it remains useful when the consumer's query changes. The hidden-B evaluation exposes an externality that same-query metrics cannot see.

### 9. Receiver-aware communication exists, but context-window awareness remains open

[Tandem Training (EACL 2026)](https://aclanthology.org/2026.eacl-long.386/) trains strong models to produce reasoning that weaker collaborators can continue. Earlier audience-design work adapts referring expressions or instructions to a listener's knowledge or language ability. [Parallel Context Compaction (2026 preprint)](https://arxiv.org/abs/2605.23296) shows that ordinary prompt wording gives operators poor control over summary volume and that retained content becomes unstable as context grows.

These are the right comparators for Experiment 9, but they do not appear to test the planned factorial contrast: merely tell a writer that the next LLM has a 2k versus 10k context capacity, versus explicitly ask it to resize, then measure evidence survival through sealed large→small→large trajectories. This remains a plausible contribution, but it is **a design in progress, not a result**.

## Closest-work comparison

| Work | What it already establishes | Why this project is different |
|---|---|---|
| [Ao et al.](https://arxiv.org/abs/2603.26993); [Yu et al.](https://arxiv.org/abs/2607.16133) | Theory and experiments for bounded-relay denoising versus information loss | Localizes the loss to A-conditioned selection and evaluates the same payload on hidden B |
| [State Compression in Two-Agent Relays](https://arxiv.org/abs/2607.18265) | Sealed downstream state, narrative versus JSON versus full state | One known goal; no hidden future query, equal-slot causal replication, or rewriting ladder |
| [Faithful, Not Corrective](https://arxiv.org/abs/2607.09678) | Controlled format × six-hop fidelity; relay-tier effects | Relays are told to retain every fact; no A-directed selection or hidden-B opportunity cost |
| [When “Must” Becomes “Maybe”](https://arxiv.org/abs/2608.24569) | Artifact-only executor, multistage constraint weakening, exact field restoration | Known operational constraint rather than factual reuse across an unannounced task |
| [MEMAUDIT](https://arxiv.org/abs/2605.02199) | Query-blind memory writing under exact budgets and future-query requirements | Writer sees A; compares utility for A with the externality imposed on B; inter-agent sealed prose |
| [TierMem](https://arxiv.org/abs/2602.17913) | Write-before-query barrier and raw-evidence fallback | Measures irreversible loss when the source cannot be reopened, then exact restoration as a causal control |
| [Routed Graph Handoff](https://arxiv.org/abs/2608.25277) | Adaptive interface choice for current-task accuracy and cost | Measures how current-task optimization changes the message's reuse value |
| [Handoff Debt](https://arxiv.org/abs/2606.02875); [Handoff Tax](https://arxiv.org/abs/2608.24358) | Coding takeover, rediscovery, model-switch direction, interface effects | Repository remains available and receiver changes; here evidence and final reader are fixed and sealed |
| [Query-focused summarization](https://aclanthology.org/2021.acl-long.475/) | Query-dependent evidence selection is a core summarization operation | Scores the same selected message on a different, unannounced query |
| [Parallel Context Compaction](https://arxiv.org/abs/2605.23296) | Summary-length instructions are weakly obeyed and unstable | Planned Experiment 9 separates recipient-capacity knowledge from an instruction to act on it |

## What this project can and cannot claim

### Defensible central claim

> Prior work shows that bounded agent interfaces can denoise or discard information, that message format and relay capability govern repeated-transfer fidelity, and that model switches create direction-dependent continuation costs. We study a complementary failure mode: the loss of **communication reusability** when a handoff is optimized for a currently known task. Holding evidence, payload capacity, and the downstream reader fixed, we evaluate the same sealed payload on the announced task and on unannounced future queries, separating task-conditioned selection from generic rewriting.

### Strong empirical claims already supported locally

1. **Compression can improve immediate accuracy.** In the ten-question baseline pilot, free-form handoff outperformed direct full context by +0.264 F1. Given known long-context degradation, this should be described as denoising, not information creation.
2. **The task contract drives useful denoising.** Removing the question from repeated compressors worsens all six matched conditions, with especially large full-context losses.
3. **The task contract also narrows reuse.** In natural A/B pairs, A-conditioned messages lose roughly half of hidden-B judged correctness across depths.
4. **Selection is the main narrowing operation.** Pass-through, paraphrase, and generic compression retain B; A-conditioned selection does not.
5. **The trade-off survives an exact-capacity causal replication.** With fictional dossiers and exactly four cards, conditioning changes announced-A accuracy by +0.317 and mean hidden-query accuracy by −0.062. The trade-off is stronger at K=2 and exactly zero among paired-compliant all-card K=6 messages.
6. **Availability, not later relay size, is the supported bottleneck.** On clean fictional B omissions, both small and large sealed relays score zero. Restoring the answer-bearing text at exactly the same width beats sham restoration by +0.983 for Llama and +0.950 for Qwen.

These claims are documented in the repository's [core findings](CORE_FINDINGS_AND_STORY.md) and [full experiment report](HANDOFF_EXPERIMENTS_REPORT.md).

### Claims to avoid

- First demonstration that agent handoffs lose information.
- First sealed or artifact-only handoff evaluation.
- First repeated LLM relay or telephone-game study.
- First format-by-depth or structured-handoff comparison.
- First adaptive message-format or provenance-aware handoff.
- First demonstration that model switching matters.
- First compaction study conducted before future queries are known.
- A universal ranking of prose, JSON, graphs, extracts, or latent channels.
- A universal monotonic benefit from larger upstream writers or relays.
- The direct-full-context arm as a strict performance ceiling.
- Any completed finding from Experiment 9.

### Useful terminology

- **Communication reusability:** how much utility a handoff retains for tasks not used to create it.
- **Task-conditioned selection:** inclusion and omission decisions induced by the announced task.
- **Future-query externality:** the reduction in utility for other queries caused by optimizing for the current one.
- **Availability barrier:** once novel answer-bearing evidence is absent from a sealed message, downstream capability alone cannot recover it.
- **Immediate utility versus reusable utility:** the two axes a handoff evaluation should report.

“Reusability” is the cleanest umbrella term. “Loss” alone makes the project sound like another telephone-game paper; “generalization” alone sounds like ordinary domain transfer.

## Recommended paper narrative

### One-sentence thesis

> A handoff is not simply a compressed record: it is a task-conditioned interface whose gains for the announced task can be paid for by irreversible losses in future usefulness.

### Three contributions

1. **Evaluation concept:** introduce communication reusability—score one sealed handoff on both its announced task and unannounced tasks, rather than only on the task that shaped it.
2. **Causal attribution:** use a rewriting ladder, exact-card budgets, fictional facts, source isolation, and same-width restoration to distinguish generic rephrasing from task-conditioned selection and genuine evidence absence.
3. **Empirical finding:** task-aware selection can improve immediate utility while lowering reusable utility; repeated rewriting is often comparatively benign, and later model capacity cannot compensate for evidence that never arrives.

### Suggested title options

- **Useful Now, Useless Later? Measuring Communication Reusability in LLM Agent Handoffs**
- **The Future-Query Cost of Task-Optimized Agent Handoffs**
- **What the Handoff Leaves Behind: Task-Conditioned Selection and Reusable Information in LLM Agents**
- **Optimized for A, Missing for B: Information Externalities in LLM Agent Handoffs**

### Draft positioning paragraph

> Multi-agent LLM research has shown that bounded messages can either denoise context or discard task-relevant information, and recent work characterizes handoff failures, format effects, repeated relay fidelity, and model-switch costs. These evaluations almost always score a message on the task that determined what the sender retained. We instead study communication reusability: whether a handoff optimized for a known task remains useful for an unannounced task. Our design fixes source evidence and the downstream reader, seals the handoff from the source, and separates rewriting from selection using natural QA pairs and fictional fixed-capacity dossiers. We find that task conditioning can improve announced-task accuracy while reducing hidden-query accuracy, that most narrowing occurs at the first selection step rather than through ordinary rephrasing, and that downstream model capacity cannot reconstruct answer-bearing evidence absent from the message. These results motivate evaluating handoffs on both immediate utility and retained future utility.

## How to position each experiment

| Experiment | Role in the paper | Literature-facing interpretation |
|---|---|---|
| 1 | Motivation | Compression can denoise; consistent with information-bottleneck and long-context findings |
| 2 / 2a | Negative boundary condition | Repeated rewriting with the task visible is not inevitably degrading |
| 2b | Mechanism precondition | Knowing the target task enables effective selection from noisy context |
| 3 / 4 / 7 | Supporting scope tests | Retrieval quality, redundancy, and new evidence are distinct mechanisms from relay count |
| 5 | Natural-data discovery | A-conditioned notes sacrifice hidden-B utility |
| 5a | Mechanism localization | The sharp loss appears at task-conditioned selection, not ordinary rewriting |
| 5b | Headline causal replication | Equal capacity and fictional evidence demonstrate an immediate/reusable utility trade-off |
| 6 | Inconclusive robustness | Do not infer language-switch equivalence from intervals containing zero |
| 8 | Qualification | Writer heterogeneity changes message form more reliably than target accuracy |
| 8b | Causal availability result | Later capability cannot recover novel evidence that is textually absent; exact restoration can |
| 9 | Separate work in progress | Recipient-capacity pragmatics; no valid completed result yet |

## Threats reviewers are likely to raise

1. **Small and synthetic samples.** The strongest causal result uses 20 fictional dossiers; most natural-data studies use 10–30 items. This supports mechanism discovery, not a population-level effect size.
2. **Model dependence.** One principal selector/reader stack dominates the strongest experiments. Experiment 8b usefully shows that simple size conclusions can reverse across Llama and Qwen.
3. **Metric dependence.** EM and token F1 under-credit semantically equivalent answers, while an LLM judge introduces another model. Reporting all three and auditing disagreements is appropriate.
4. **Task definition.** A and B are atomic QA tasks. Real handoffs may serve plans, constraints, citations, uncertainty, or later tool use. The new constraint-weakening literature makes this limitation especially visible.
5. **Prompt-induced selection versus agent architecture.** The clean isolation is a strength, but it omits interactive clarification, source reopening, shared workspaces, and retrieval fallback used in deployed systems.
6. **“Full context” is not a ceiling.** Longer input alone can hurt, so comparisons must separate information coverage from the reader's ability to use that information.
7. **Multiplicity and exploration.** Many arms and exploratory intervals are not multiplicity-corrected. A scaled confirmatory study should preregister the primary contrast.

## Highest-value next steps

1. **Scale the fixed-capacity A/B experiment first.** Use at least 100 varied dossiers, preregister K=4 announced-A and mean-hidden-query deltas, and keep exact slot compliance as an inclusion rule.
2. **Replicate the full selector→handoff→reader stack.** Use a second selector family and a second fixed answerer, not only different intermediate relays.
3. **Add realistic multi-use artifacts.** Research briefs, incident reports, legal/policy dossiers, or coding investigations with one announced request and several later questions would test external validity without giving up source control.
4. **Compare mitigations on the same reuse frontier.** Generic notes, explicit multi-query coverage, provenance-linked claims, retrieval pointers, and raw-evidence fallback should be compared at matched cost. The relevant outcome is a Pareto frontier over immediate accuracy, hidden-query accuracy, bytes, and latency.
5. **Measure retained evidence directly.** Alongside QA, score answer-bearing card coverage, named-entity/fact survival, provenance validity, and uncertainty/constraint force. This connects the project to MEMAUDIT and the constraint-weakening literature without surrendering the A→B design.
6. **Treat Stage 3 as a bridge, not the headline.** After When “Must” Becomes “Maybe” and AgentAsk, controlled corruption types are useful if they connect factual reuse to established edge-failure categories, but they are unlikely to be the main novelty.
7. **Finish Experiment 9 only under a compliance audit.** Report requested capacity, delivered token count, truncation, repetition, and answer-bearing evidence survival separately. Parallel Context Compaction predicts that prompt-only resizing may be weak or unstable, so a null is scientifically meaningful.

## Experiment 9 status warning

The replacement size-adaptation run is incomplete: stage 1 finished, stage 2 is only partial, and answer generation and judging have not started. The apparent current `results/size_adaptation/report.md` is byte-identical to the archived 4,096-token-capped report. In that run, 117 of 230 expansion requests hit the cap, so treatment, delivered length, and downstream context were confounded.

Do **not** quote its expansion-arm accuracy or length effects. The only defensible observation is diagnostic: when asked to expand, the writer often saturated the available output budget and repetition increased. The authoritative status is in [`HANDOFF.md`](../HANDOFF.md) and the archive rationale is in [`WHY_THIS_IS_ARCHIVED.md`](size_adaptation/capped_4096/WHY_THIS_IS_ARCHIVED.md).

## Priority reading order

For understanding and writing the paper efficiently:

1. [When “Must” Becomes “Maybe”](https://arxiv.org/abs/2608.24569) — closest causal artifact-only failure and restoration study.
2. [MEMAUDIT](https://arxiv.org/abs/2605.02199) — closest exact-budget future-query evaluation framework.
3. [Faithful, Not Corrective](https://arxiv.org/abs/2607.09678) — strongest evidence that rewriting itself need not be the culprit.
4. [State Compression in Two-Agent LLM Relays](https://arxiv.org/abs/2607.18265) — closest sealed single-handoff format comparison.
5. [When Do Multi-Agent Systems Help?](https://arxiv.org/abs/2607.16133) — best empirical denoising-versus-loss framing.
6. [What to Keep, What to Forget](https://arxiv.org/abs/2607.08032) — rate–distortion and repeated/future-query context.
7. [AgentAsk](https://aclanthology.org/2026.acl-long.1294/) — edge failure taxonomy and repair.
8. [Routed Graph Handoff](https://arxiv.org/abs/2608.25277) — current frontier in adaptive handoff formats.
9. [The Handoff Tax](https://arxiv.org/abs/2608.24358) and [Handoff Debt](https://arxiv.org/abs/2606.02875) — takeover and model-switch positioning.
10. [Context Length Alone Hurts](https://aclanthology.org/2025.findings-emnlp.1264/) — essential interpretation of the direct-context baseline.
11. [Chain of Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract-Conference.html) — main long-context relay baseline.
12. [Parallel Context Compaction](https://arxiv.org/abs/2605.23296) and [Tandem Training](https://aclanthology.org/2026.eacl-long.386/) — Experiment 9 and recipient-adaptation context.

## Literature-search note

This is a current landscape review, not a registered systematic review. I searched primary arXiv, ACL Anthology, NeurIPS, and ICLR records through 30 August 2026 around the combinations of: agent handoff, multi-agent communication, relay/compression depth, message format, model switching, future query, memory writing, context compaction, query-focused summarization, recipient adaptation, and listener/context capacity. I also checked the repository's complete local syntheses and experiment reports.

The field is moving unusually quickly: Routed Graph Handoff was posted on 26 August, while Constraint Weakening and The Handoff Tax were posted on 25 August. Any submission-ready related-work section should therefore rerun the search immediately before submission.

## Bottom line

The project should be positioned as a **controlled study of the externality created by task-optimized communication**, not as a generic study of lossy handoffs. The literature has already established that boundaries lose information, formats matter, relays drift, and future-query compression is risky. What remains distinctive is showing—under sealed, matched, and causally controlled conditions—that improving a handoff for what is asked now can make it worse for what is asked next, and pinpointing task-conditioned selection as the operation that creates that trade-off.
