# Core findings: a clean story about agent handoffs

This document is the short, curated interpretation of the experiments. It is
not a replacement for the full [experimental report](HANDOFF_EXPERIMENTS_REPORT.md),
which remains the audit trail for every design, correction, null result, and
artifact.

## The claim in one sentence

> **A handoff is not inherently a cumulative-loss mechanism. It is a
> task-conditioned information bottleneck: knowing the task lets compression
> denoise evidence and preserve current-task accuracy, but the resulting
> representation can discard information needed for future tasks.**

This produces one coherent sequence:

1. A question-aware compressor can remove irrelevant context.
2. Once it has formed an answer-sufficient representation, repeated rewriting
   often changes little semantically.
3. The same selection process narrows the representation around the announced
   question, reducing its value for questions that were not known at handoff
   time.

The experiments therefore do **not** support a universal accuracy penalty per
handoff. They support a conditional trade-off between **denoising for the
present task** and **preserving optionality for future tasks**.

## Finding 1 — repeated handoffs do not have a universal degradation curve

The full repeated-handoff runs do not show a consistent semantic accuracy
decline with depth. In the Llama 3.3 70B chain, no evidence condition lost more
than five judged-accuracy points after ten handoffs. The largest token-F1 loss,
HotpotQA gold-only at −0.107, corresponded to no judged-correctness loss: both
depth 0 and depth 10 scored 0.900. This is evidence of answer-form drift, not a
demonstration that the underlying answer fact disappeared.

The full Qwen3-32B replication strengthens the boundary claim. At depth 10,
none of its six dataset/context conditions had a reliably negative F1 change.
MuSiQue full context instead improved by **+0.182 F1** (95% CI +0.064 to
+0.312) and **+0.167 judged accuracy** (+0.017 to +0.317). Repeated summaries
can therefore improve performance when the source contains enough distractor
mass to remove.

![Qwen3-32B answer accuracy across repeated handoffs](chain_qwen32/degradation.png)

**Interpretation.** Question-pinned rewriting appears capable of approaching a
stable, task-sufficient representation. The first transformation performs most
of the selection; later transformations often preserve that selected core.
Starting from minimal evidence leaves little denoising opportunity, whereas
starting from full context creates room for compression to help.

**Limit of the claim.** The Llama and Qwen runs use model-specific closed-book
filters, so their retained question sets can differ. They establish that large
serial loss is not universal; they do not establish a causal model-scale
effect.

## Finding 2 — task visibility determines whether compression helps or harms

The cleanest large controlled comparison changes only whether compressors see
the question. It holds fixed the Llama 3.3 70B model, 30 questions per dataset,
contexts, depths, two seeds, system prompt, and handoff instructions. The final
answerer always receives the question.

At depth 10, every context condition is worse when the compressors do not know
the question. The difference grows with distractor load:

| Dataset, full context | Question-conditioned | Question omitted |
|---|---:|---:|
| MuSiQue: F1 change | +0.080 | **−0.272** |
| MuSiQue: judged-accuracy change | −0.017 | **−0.467** |
| HotpotQA: F1 change | +0.014 | **−0.326** |
| HotpotQA: judged-accuracy change | −0.033 | **−0.400** |

![Matched question-conditioned and question-omitted chains](chain_generic/conditioning_comparison.png)

**Interpretation.** The question is not incidental prompt metadata; it is part
of the communication channel. A bounded compressor can only distinguish
signal from distractors relative to a task. When the task is absent, the same
compression budget must preserve a generic description and loses much more
answer-relevant information.

This also changes how serial-relay results should be interpreted. A design in
which downstream agents do not retain the task can conflate two effects:
communication compression and loss of task side-information. The relevant
system comparison is therefore not simply “one agent versus many agents,” but
whether every handoff retains an immutable task contract.

## Finding 3 — optimizing a handoff for one question can destroy future value

The gold-only same-passage experiment isolates a different cost of task
conditioning. Each example contains one short passage answering two distinct
questions, A and B. There are no distractors, no competing document, no prompt
request for brevity, and no truncation. The compressor either sees Question A
or writes a generic summary; the resulting handoff is evaluated on both A and
the previously unannounced B.

Conditioning produces no reliable improvement for A at any landmark depth. It
nevertheless causes a large loss on B at every depth:

| Conditioned minus generic | Depth 1 | Depth 5 | Depth 10 |
|---|---:|---:|---:|
| Target A, F1 | +0.096 | +0.040 | +0.040 |
| Held-out B, F1 | **−0.600** | **−0.500** | **−0.433** |
| Held-out B, judged accuracy | **−0.600** | **−0.500** | **−0.500** |

At the first handoff, conditioned summaries retain B's exact answer fact in
3/10 cases, compared with 9/10 for generic summaries. Generic handoffs remain
near the direct-context ceiling for B, while conditioned handoffs remain far
below it through depth 10.

![Target utility and held-out-question accuracy from one gold passage](squad_same_passage_goldonly/n10/summary_generalization.png)

**Interpretation.** This is task-induced narrowing, not ordinary noise removal.
The passage is short enough to retain both facts, but the model discards the
fact outside its declared objective anyway. A handoff can therefore be highly
useful for a known task while being a poor reusable representation of the
source.

The system-level quantity of interest is **future-query regret**: the difference
between what a future agent could answer from the original evidence and what it
can answer after a task-conditioned handoff. Immediate target accuracy does not
measure this loss of optionality.

**Evidence status.** This is the sharpest mechanistic result but remains a
pilot: 10 pairs, one seed, and Llama 3.1 8B. Its magnitude and persistence are
large and agree across F1, judged correctness, and direct fact-survival counts,
but it requires a larger independent replication before being presented as a
population effect.

## What the remaining experiments contribute

The other branches are useful controls and boundary conditions, but they do not
need to compete with the three findings above in the main narrative.

| Experiment or result | Role in the paper | Recommended placement |
|---|---|---|
| Initial one-handoff MuSiQue pilot | Early evidence that compression can denoise; motivates Finding 1 | Brief motivation or appendix |
| Llama 3.3 70B repeated chain | Primary evidence that semantic loss is not monotonic and that F1 can overstate degradation | Main evidence behind Finding 1 |
| Qwen3-8B chain | Low-cost directional robustness check, underpowered at n=10 | Appendix |
| Qwen3-32B chain | Full cross-family robustness result and clearest denoising example | Main plot for Finding 1 |
| Question-omission chain | Strongest matched causal result | Main plot for Finding 2 |
| Separate-passage A/B experiment | Shows document-level narrowing and motivates future-query evaluation | Supporting appendix analysis |
| Same-passage with distractors | Useful null showing that the separate-document mechanism disappears | Appendix |
| Same-passage gold-only | Cleanest evidence of within-document task-induced narrowing | Main plot for Finding 3 |
| Retrieval-quality experiment | Shows that missing initial evidence remains a retrieval problem; handoffs do not reliably recreate it | IR boundary result or appendix |
| BM25 hard/easy reruns | Methodological warning that lexical rank is not a relevance label | Limitations/methods appendix |
| Redundant-signal experiment | Directional evidence that signal ratio matters, but redundancy is not independently sourced | Appendix/future work |
| Multilingual handoffs | Corrected null pilot; language switching is not a detected primary mechanism | Appendix or separate follow-up |

Two supporting lessons should still be stated briefly:

- **Retrieval sets the available evidence ceiling.** In the screened MS MARCO
  experiment, the low-versus-high retrieval-quality gap was +0.254 F1 at depth
  0 and +0.299 at depth 5. Repeated handoffs did not repair missing evidence.
- **Answer correctness is not evidence preservation.** Token F1 detects form
  drift, while an LLM judge can credit semantically correct answers that were
  reconstructed or guessed. Neither replaces direct measurement of fact and
  provenance survival.

## Clean paper storyline

The paper can now be told in four moves:

1. **Reject a scalar handoff tax.** Across two model families, ten question-aware
   handoffs are often semantically stable and can improve noisy-context QA.
2. **Identify the causal switch.** Removing the question turns the same pipeline
   from a potential denoiser into a strongly degrading compressor.
3. **Reveal the hidden cost of that switch.** Question-aware summaries preserve
   the current task by narrowing the evidence representation, which can create
   severe regret on future questions.
4. **Draw the design implication.** Reliable agent systems should preserve the
   task contract separately from the handoff while also retaining auditable
   evidence units when the future use of the state is uncertain.

This positioning is narrower than claiming that multi-agent systems generally
help or hurt. The experiment isolates the **handoff channel**: fixed evidence,
sealed downstream access, repeated transformation, controlled task visibility,
and fresh answerers.

The closest research framing is the information-bottleneck account of
[Yu et al.](https://arxiv.org/abs/2607.16133), refined by the finding that task
visibility determines what counts as relevant information. The next section
works through where each finding agrees with, contradicts, or has no counterpart
in the published multi-agent literature.

## How these findings sit in the multi-agent literature

The comparison set is the focused communication-only crawl in `research-os`
([literature review](../../research-os/outputs/literature-reviews/2026-08-19-multi-agent-handoff-communication-only.md),
[semantic graph](../../research-os/outputs/graphs/2026-08-19-multi-agent-handoff-communication.md)),
cutoff 19 August 2026. That crawl deliberately excludes conversational memory,
generic RAG, standalone evaluation methods, and single-agent compaction, so
"no prior work" below always means *no prior work inside a communication-scoped
crawl*, not a claim about all of NLP. Two literature checks that this scope
never ran are listed at the end.

### Summary

| Finding | Confirms | Contradicts or qualifies | Comparator status |
|---|---|---|---|
| **1.** No universal per-handoff tax; compression can improve noisy-context QA | Yu et al. (IB), BANDMAS, Chain of Agents | Ao et al.'s depth collapse; Beyond Frameworks' summarization ladder | Well-populated; this is a replication with a cleaner design |
| **2.** Task visibility is the causal switch | Yu et al. (IB), AgentAsk, GAVEL | Ao et al. — supplies the missing control for its headline result | Anticipated as a caveat; never measured |
| **3.** Task-conditioned handoffs create future-query regret | Yu et al.'s forward-dependency warning; the "accuracy is not preservation" claim | Nothing directly — no retrieved work measures unannounced-query accuracy | **No comparator.** Also the weakest evidence here |

The pattern is worth stating plainly: the finding with the most prior support is
the least novel, and the finding with no comparator rests on 10 pairs, one seed,
and an 8B model. That asymmetry, not the headline effect sizes, is what should
drive the next run.

### Finding 1 confirms the field's non-monotonicity result and strains its depth result

**Confirms.** That relay loss is not monotonic is already the literature's
position, and this work adds a cleaner instance of it.
[Yu et al.](https://arxiv.org/abs/2607.16133) decompose a bounded relay into
useful removal of upstream entropy minus capability-weighted loss of
downstream-relevant information, and report gains that shrink or reverse as task
dependency and model capability rise across 18 experiments. The Qwen3-32B
MuSiQue full-context result — **+0.182 F1** and **+0.167 judged accuracy** after
ten handoffs — is the denoising branch of that decomposition observed on a
channel where the evidence payload is held fixed, which their end-to-end agent
tasks cannot do.
[BANDMAS](https://arxiv.org/abs/2608.00458) reaches HotpotQA answer F1 .290
against .273 for full traffic while cutting 77.3% of bytes;
[Chain of Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract-Conference.html)
scores 37.09 on MuSiQue against 26.87 for truncated full context. Both say the
same thing from different directions: uncompressed input is a lossless
*communication* reference, not an accuracy ceiling. Finding 1 is consistent with
all three.

**Strains.** [Ao, Gao, and Simchi-Levi](https://arxiv.org/abs/2603.26993) report
90.7% for a centralized decision against 41.2%, 43.5%, and 22.5% for prose
relays at depths 2, 3, and 5. Ten handoffs here cost at most five judged points.
Both results cannot be general. The semantic graph records this as a genuine
contradiction against this project's own pilot claim, and it should stay
recorded rather than be explained away.

Three differences are candidates, and Finding 2 resolves the first:

1. **Ao's downstream stages do not retain the question.** This is the difference
   the next subsection measures directly.
2. **MMLU is a parametric task with no evidence payload.** A relay that must
   reconstruct which question is being answered, from a prose message about
   general knowledge, is a harder inference problem than compressing a fixed
   paragraph set.
3. **Depth is confounded everywhere else in the literature.** Chain of Agents
   adds a new chunk at every hop, so relay depth, evidence coverage, and chunk
   order move together; its order ablation alone moves MuSiQue 37.09 → 29.77,
   larger than most treatment effects reported in this space. The literature has
   therefore never varied depth alone. This design does — fixed payload, no new
   evidence per hop, question pinned — and the answer is that depth alone, under
   those conditions, is close to flat.

That converts one of the crawl's five "claims the field cannot make" — *relay
depth alone has not been shown to cause degradation* — into a measured result,
in the negative direction.

**Qualifies.** [Beyond Frameworks](https://aclanthology.org/2025.acl-long.1037/)
reports a stepwise ladder on evidence-based fact-checking: all-context 90.5%,
evidence-selected 88.7, central summary 86.9, decentralized recursive summary
84.4. Read as a degradation curve it points the other way. Read as magnitudes it
does not conflict much: roughly two points per rung there, against ≤5 judged
points over ten handoffs here. Their rungs also change *what the system shares*,
not how many times one fixed payload is rewritten, so the comparison is
directional rather than matched.

### Finding 2 supplies a control the field flagged but never ran

This is where the work makes its clearest contact with published results.

The crawl already recorded the confound, as a limitation attached to Ao's depth
finding: *downstream stages do not retain the original question, so task
reconstruction is confounded with message loss*. It was logged as a reason to
weaken Ao's contradiction, and as the motivation for pinning the question in
this design. Nobody had turned it into an experiment.

The matched question-omission chain does exactly that, holding model, item
count, contexts, depths, seeds, and prompts fixed and toggling only whether the
compressor sees the question. MuSiQue full context moves from **+0.080 F1 /
−0.017 judged** to **−0.272 F1 / −0.467 judged**; HotpotQA moves from **+0.014 /
−0.033** to **−0.326 / −0.400**. Steep degradation appears exactly when the task
contract is removed, and not otherwise.

**What this does to Ao's result.** It does not refute it. It reinterprets it:
their headline number is at least partly a measurement of task reconstruction
under prose relay, not purely of message compression. The question-omitted arm
here is the closer analogue of their setup, and it behaves like their setup.
That is a concrete, testable correction to how a widely-cited depth curve should
be read, and it generalizes to any relay benchmark whose downstream agents do
not carry the task.

**Confirms.** Yu et al.'s bottleneck only has a well-defined notion of
"downstream-relevant information" relative to a task; a compressor without the
task cannot compute relevance. Finding 2 is the direct experimental
instantiation of that dependency.
[AgentAsk](https://aclanthology.org/2026.acl-long.1294/) audited 824 real agent
messages and found Referential Drift at 27.3% and Data Gap at 29.1% — the two
largest failure families, and both are what a lost task referent produces. Their
remedy is a repair: a minimal clarifying question at risky edges, worth roughly
3.3–3.5 points at under 10% added latency and cost. Finding 2 says a large share
of that failure is preventable by construction rather than repairable after the
fact — which matters, because the crawl also notes that edge clarification can
generate new reasoning instead of recovering the original payload.
[GAVEL](https://aclanthology.org/2026.findings-acl.1789/) shows the same shape
on the provenance axis: removing the evidence contract costs 5.3 points on
FEVEROUS (41.8 → 36.5).

**Reframes the single-versus-multi-agent debate.**
[Tran and Kiela](https://arxiv.org/abs/2604.02460) find single agents best or
bootstrap-CI tied at nearly every thinking-token budget; Yu et al. find MAS
gains reversing as capability rises. Finding 2 suggests "one agent versus many"
is the wrong contrast to draw first. In this channel the agent count is
constant, and the sign of the effect still flips — driven entirely by whether
the task contract survives each edge. Multi-agent comparisons that do not report
task-contract retention per edge are pooling two different systems.

### Finding 3 has no comparator, in either direction

Nothing in the crawl measures accuracy on a question that was not announced at
compression time. Finding 3 is not contradicted by prior work; it is simply not
addressed by it.

The nearest neighbours, and how they differ:

- **Yu et al.** warn that a locally sufficient relay can fail to preserve
  dependencies needed several workers later. That is the closest published
  statement — but it concerns later stages of *the same declared task*. Finding
  3 is a different axis: a task nobody declared. It extends their warning rather
  than duplicating it.
- **[MOC](https://arxiv.org/abs/2606.02359)** gives downstream agents raw
  ancestor outputs up to K hops away and finds K=2 generally strongest (+6.77%
  on AQuA, +3.68% on HumanEval).
  **[AnyMAC](https://aclanthology.org/2025.emnlp-main.584/)** lets a receiver
  select an arbitrary subset of prior outputs instead of a compulsory serial
  relay. Both are architectural remedies whose *mechanism* was never isolated —
  the crawl records that in MOC raw visibility, ordering, message volume, and
  merging all co-vary, with no direct retention measure reported. Finding 3
  offers a candidate mechanism for why raw access helps: a task-conditioned
  summary is a narrowed representation, so a skip-link to the unsummarized
  source is worth more than its token cost whenever the downstream task is not
  the one that shaped the summary.
- **GAVEL and BANDMAS** motivate provenance-bearing, inspectable message units
  on grounds of auditability and checking. Finding 3 adds an accuracy-based
  argument for the same primitive: retaining raw evidence units bounds
  future-query regret, which auditability arguments alone do not quantify.
- **The "accuracy does not prove preservation" claim** — one of the five things
  the crawl says the field cannot assume away — gets its sharpest single-channel
  demonstration here. Target-question accuracy is flat to slightly positive
  (+0.096 F1 at depth 1, +0.040 at depth 10) while held-out-question accuracy
  falls 43–60 points. A metric computed on the announced task is structurally
  blind to the loss. [Yan et al.](https://arxiv.org/abs/2608.03421) make the
  same point from the opposite direction — collective truth recovery falling
  72.50% → 14.17% while messages keep flowing — but through injected false
  testimony rather than ordinary compression.

**The honest caveat.** No comparator also means no external corroboration. This
is a 10-pair, single-seed, Llama 3.1 8B pilot, and the design is unusual enough
that a reviewer's first move will be to ask whether the effect is an artifact of
the prompt rather than of compression. That is what the scale-up in the next
section exists to answer.

### What the literature makes unavailable

The crawl's novelty audit is blunt and still binding. None of these are
claimable: first handoff-loss study, first repeated relay, first structured
message format, first provenance-bearing handoff, first bandwidth-constrained
multi-agent system. Ao, Chain of Agents, GAVEL, BANDMAS, and Handoff Debt each
take one.

Two further boundaries follow from what was actually run:

- **No representation claim.** `C_structured` and `D_extractive` were built but
  never ran in the main API pilot, so the "formats compared" column of the gap
  table stays empty. This costs less credibility than it looks like, because the
  literature already refuses a universal format ranking:
  [Handoff Debt](https://arxiv.org/abs/2606.02875) reports 52.5% raw trace, 51.4
  free-form, and 50.8 structured across 2,172 successor runs, and
  [AutoForm](https://aclanthology.org/2024.findings-emnlp.623/) shows format
  gains reversing (.62 → .53) with a weaker receiver. Expecting structure to win
  was never well-founded.
- **No topology or corruption claim.** This is one fixed serial chain with no
  injected corruption — `src/inject.py` was never built. So it says nothing
  about [Shen et al.](https://aclanthology.org/2025.emnlp-main.623/) on sparsity
  and error suppression, or Yan et al. on misinformation cascades. Those are
  different failure modes from narrowing, and none of the three findings here
  bear on them.

### The gap that was filled is not the gap that was scouted

Worth recording explicitly, because it changes the framing. The crawl's
positioning argument was a **conjunction**: fixed gold payload · question pinned
· matched human-readable formats · varied load · repeated depth · sealed
receiver · per-edge provenance retention. No published row filled more than five
of those seven columns, and the planned contribution was to fill all seven.

That is not what the completed experiments deliver. The formats column is empty
and the per-edge retention column is empty, so the conjunction argument is
weaker now than when it was scouted. What landed instead is two mechanisms that
were not on the axis list at all:

- **Task visibility as a first-class channel variable.** The crawl treats the
  pinned question as methodological hygiene — "a one-line design decision that
  protects the entire result." Finding 2 shows it is the dominant effect in this
  channel, larger than depth, dataset, or context length. It belongs in the
  seven-decision table as an eighth row, not in a methods section.
- **Future-query regret as an outcome.** Every metric in the crawl scores the
  announced task. Finding 3 argues that a handoff has a second, uncorrelated
  value — how much of the source remains answerable at all — and that current
  evaluation cannot see it.

The stronger positioning is therefore *attribution*, exactly as the crawl
recommended, but attributing a different variable than planned. Published
systems move four things at once and report that the bundle helped; this work
identifies which single edge property carries the effect.

### Two literature checks still owed

Both fall outside the crawl's scope and should be run before any writeup.

1. **Query-focused summarization.** Finding 3 restates, in an agent setting, a
   question that the QFS literature has had the machinery to ask for years:
   optimizing a summary for a given query is that field's explicit objective, and
   the cost the objective imposes on unanticipated queries is its natural
   complement. The communication-only profile excludes standalone summarization
   work, so that literature has not been searched. If a generic-versus-query-
   focused cost has already been quantified there, Finding 3's novelty claim
   needs rewording — the agent-relay framing and the depth dimension would
   survive, the core asymmetry would not.
2. **Boundary-local methodology.** TRACE
   ([arXiv 2608.06503](https://arxiv.org/abs/2608.06503), present in the
   archived broad scan but not the focused crawl) sets the current bar for this
   kind of measurement: paired continuations from an identical pre-boundary
   state, frozen downstream components, bootstrap intervals, recorded prompt
   hashes, and no compression-prompt tuning after seeing failures. The chain
   experiments here meet parts of that bar and not others. Meeting it fully in
   the scale-up is cheaper than defending the gap later.

## The one next experiment that matters most

Scale the gold-only future-query experiment to at least 100 independently
constructed A/B pairs, two seeds, and a stronger second model. Use a common
closed-book-filtered sample across models and annotate fact survival at every
handoff. The preregistered primary outcomes should be:

1. Accuracy on announced Question A.
2. Accuracy on unannounced Question B.
3. Per-edge survival of the facts supporting A and B.
4. Handoff length and truncation rate.

That experiment directly tests the most distinctive claim while converting the
current answer-level pattern into a mechanistic account of when and where
future value is lost.

## Claims this summary deliberately does not make

- There is a fixed amount of loss per handoff.
- Ten handoffs are always safe.
- Qwen is more robust because it is larger or from a different family.
- A flat semantic judge proves that all evidence survived.
- Structured messages are superior; the implemented structured and extractive
  mechanisms have not yet been tested in the main API pilot.
- The multilingual null demonstrates equivalence across languages.
- The n=10 future-query effect is already a final population estimate.

