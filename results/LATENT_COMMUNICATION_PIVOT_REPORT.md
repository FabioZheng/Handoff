# Beyond token counts: a pivot to non-language agent communication

Research review and experiment recommendation · updated 8 September 2026

**Status refreshed 8 September 2026:** literature review and protocol completed. A prototype now exists with 102 recorded passing offline checks. Its current codec performs norm calibration only, so it is not yet the full StateBridge alignment. No verified real-model GPU result is available in the inspected artifacts. Compute planning assumes Bunya GPUs, as requested; current GPU entitlement and availability remain unverified. The [supervisor decision report](LATENT_PIVOT_NEXT_EXPERIMENT_2026-09-08.md) consolidates the staged next experiment and remaining implementation gaps.

## Recommendation

Pivot toward **the cost and reusability of non-language agent messages**, starting with a frozen, same-model latent interface. Keep the project's strongest experimental asset: a sender knows task A, but its sealed message must also support questions B that it did not see.

The central question should be:

> Can a non-language handoff use fewer receiver context positions or less total computation while preserving both immediate and unannounced future-task utility?

This is a stronger target than demonstrating that agents can communicate without words. That has already been demonstrated. It also distinguishes genuine resource savings from a reduction in the number of objects called “tokens.”

The first experiment should compare concise prose, self-contained symbolic records, and a StateBridge-derived latent prefix. Interlat is a possible second-stage trained method. DiffMAS is initially a literature comparator, pending an auditable implementation and a reason to incur joint-system training complexity. The detailed protocol is in [Experiment 13](../EXPERIMENT_13_DESIGN.md).

## What has already been done

Non-language communication is an established research direction. For example, CIPHER introduced communication through expected token embeddings in 2023, avoiding discrete token sampling without changing model weights. The proposed novelty cannot be “the first agents to communicate without natural language.” [CIPHER paper](https://arxiv.org/abs/2310.06272).

| Method | What crosses the boundary | What it requires | Implication for this project |
|---|---|---|---|
| Interlat | Final-layer hidden vectors, with learned adaptation and a separately trained compressed latent sender | Receiver/adapter training and compression training; cross-family transfer is trained | Relevant when learning an efficient protocol becomes the goal; more than an inference-only interface change |
| StateBridge | A short aligned continuous prefix derived from generated-message hidden states | Internal-state access, per-message alignment, and continuous-input inference; no parameter training | Most tractable first latent baseline; demonstrated with one shared backbone within each run |
| DiffMAS | Accumulating layer-wise KV traces in a shared-backbone multi-role system | Task-specific learning and access to internal caches | Relevant to learned coordination and rich state sharing; not automatically a low-bandwidth protocol |

Sources: [Interlat paper](https://arxiv.org/html/2511.09149v5), [StateBridge paper](https://arxiv.org/html/2608.13317v1), [DiffMAS paper](https://arxiv.org/html/2604.21794v1).

Three qualifications refine the proposed trade-offs:

**Interlat:** its reported approximately 24-fold improvement is a message-generation latency comparison, not a demonstrated 24-fold reduction in whole-task runtime or network bytes. The official repository includes training and compression workflows; a ready-to-load trained receiver was not identified in this review. [Interlat Table 3](https://arxiv.org/html/2511.09149v5#S5), [official code](https://github.com/XiaoDu-flying/Interlat).

**StateBridge:** “training-free” does not mean arbitrary models can communicate immediately. The official release explicitly leaves heterogeneous sender–receiver transfer to future work. It also generates a message internally before transmitting selected states, so its benefit need not include eliminating sender decoding. Its restricted mapping is a design choice; calling that an established performance sacrifice would be stronger than the evidence. [Official scope and implementation](https://github.com/YanwenPneg/StateBridge#scope).

**DiffMAS:** the paper proposes differentiable system optimization, but the precise training graph needs clarification: section 3.3/Appendix C describe upstream gradient propagation, while Figure 1 describes upstream trace construction without gradient updates and final-agent LoRA updates. Shared weights complicate that distinction. No official implementation was located in the paper or targeted repository search. This is an unresolved reproducibility detail, not proof of an implementation error. [DiffMAS paper](https://arxiv.org/html/2604.21794v1).

## Why this is not the default communication interface

The following are engineering and research implications of the reviewed interfaces, rather than a measured survey of adoption.

1. **Text is a broadly supported interface.** This project's OpenRouter client sends and receives text. It cannot extract transformer activations or submit arbitrary input embeddings. A latent experiment needs a separate local inference backend. Sending vectors as JSON to a text endpoint does not reproduce these methods. See [the current client](../src/llm.py).
2. **Representations depend on the model.** A hidden vector has a particular dimension, scale, layer convention and learned coordinate system. A compatible receiver, an alignment method, or training is needed. Cross-family results after adaptation do not establish zero-shot interoperability.
3. **Context compression and bandwidth compression are different.** One latent vector can occupy one receiver position but contain thousands of floating-point numbers. A growing KV trace is larger still. Avoiding tokenization does not eliminate source encoding, alignment, attention or answer generation.
4. **Training can move the cost elsewhere.** A learned protocol can be attractive at deployment scale and expensive for a small research run. Its setup cost must be amortized over an explicit number of uses. Model replacement may require another adaptation.
5. **A richer representation is not necessarily more useful.** Preserving additional activation features does not prove that hidden future questions become answerable. The relevant test is downstream utility on fresh evidence. A recent latent-channel study explicitly found that stronger feature preservation did not yield task-level superiority on its tested concept tasks; this is a useful caution, not a universal negative result. [Latent Communication Between Language Model Agents](https://arxiv.org/abs/2607.14103).
6. **Inspection becomes harder.** Prose and records can be checked directly for missing facts. Latent messages require behavioral interventions, provenance and carefully isolated decoding tests. This is an experimental burden, not a reason to rule them out.

## The accounting trap: fewer positions can mean more bytes

Use three separate resource axes throughout: **receiver positions, transferred bytes, and end-to-end execution cost**. A “token saving” claim must identify which one improved.

The following is arithmetic, not a benchmark result. The official Qwen3-4B configuration has hidden dimension 2,560, 36 layers, eight KV heads, and head dimension 128. [Model configuration](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json).

| Example payload | Calculation | Raw payload size |
|---|---|---:|
| 128 token IDs, using int32 transport | 128 × 4 bytes | 512 bytes |
| 8 BF16 final-layer vectors | 8 × 2,560 × 2 bytes | 40 KiB |
| 64 BF16 final-layer vectors | 64 × 2,560 × 2 bytes | 320 KiB |
| 8 complete BF16 KV positions | 8 × 36 × 2 × 8 × 128 × 2 bytes | 1.125 MiB |

Thus eight latent positions versus 128 text positions is a nominal 16-fold position reduction while the illustrated dense payload is 80 times larger than the token-ID payload. Accuracy has not been matched in this example. Actual transport must include headers, masks and any quantization metadata; actual text may use UTF-8 rather than token IDs.

For colocated agents, wire traffic may be absent and context processing can matter most. For separate machines, transfer and serialization can dominate. Report those deployments separately. Passing an in-process cache pointer does not constitute a tiny network message: the pointed-to state is the payload if another process or host needs it.

Compact symbolic records are therefore an essential comparator. They can remove prose overhead while remaining easy to transmit. They still consume tokenizer tokens. A short document-specific code is only valid if its dictionary also travels with the message; evaluator fact IDs cannot stand in for evidence. Lossless byte compression followed by text decompression can save transport bytes, but leaves the receiver's language-model input length unchanged.

## How the pivot preserves the existing contribution

The [current frontier report](bounded_communication_frontier/REPORT.md) establishes that explicit evidence allocation controls the present–future trade-off in the tested settings. It does not establish that natural-language serialization is the fundamental cause.

The pivot separates two questions:

| Question | Controlled comparison | Defensible interpretation |
|---|---|---|
| Does the representation preserve a fixed information selection more efficiently? | Give every encoder exactly the same selected evidence, then reset the receiver | Effect of encoding and decoding that selected evidence |
| Does an entire latent policy retain more reusable information? | Give every sender the same full source; compare its resulting messages | Effect of the complete selection-plus-communication policy |

A state produced while reading the full source can retain facts absent from its decoded prose. That is legitimate communication through the declared latent payload. It is different from recovering a fact after the source was removed before encoding. A fresh re-encoding of the same visible transcript, without the source, separates those possibilities.

In the primary method sections reviewed, the cited systems do not establish this project's joint comparison of announced-task utility, hidden-task reuse and separately measured resource costs under these controls. This is a candidate contribution supported by a targeted review, not a claim that all relevant literature has been exhausted.

## Proposed first study

**Working title: “Beyond Token Counts: Efficient and Reusable Latent Handoffs.”**

Start with one writer and one receiver using the same frozen Qwen3-4B weights. It supports both thinking and non-thinking modes; explicitly fix non-thinking mode across the first QA comparison. This is a controlled adaptation, not a reproduction of StateBridge's original multi-role benchmark. [Official model card](https://huggingface.co/Qwen/Qwen3-4B).

Use the existing 20 fictional six-fact dossiers for development. At K=4 evidence cards, reuse one generic and six A-conditioned selections per dossier, giving 140 common encoder inputs. Compare three encodings at one initial receiver budget: concise prose, typed records, and an aligned latent prefix. Answer all six questions: **2,520 core answer evaluations**, before diagnostic controls. These existing sources are development material, not a fresh confirmation set.

The diagnostic controls establish that the receiver uses the payload, that continuous inputs behave correctly, and that no source cache leaks through. Only then extend to full-source handoffs and fresh held-out sources. Preselect a primary future-query distribution and a useful noninferiority margin before examining confirmation outcomes. See the [full design](../EXPERIMENT_13_DESIGN.md) for exact arms, budgets, endpoints and gates.

## Bunya feasibility and decision gates

Bunya is a plausible platform: official documentation lists CUDA GPU nodes, including A100 and H100 resources. Current access, scheduling rules and available memory must be checked against the user's account before a job is launched. The existing [job script](../hpc/job.slurm) requests a CPU partition and does not establish GPU readiness. [UQ GPU and Slurm guide](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md).

On 7 September, checking the saved WSL SSH master returned no control socket. No fresh login was attempted. The next execution session needs the normal interactive Bunya authentication before read-only resource inspection. Credentials remain interactive/environment-only.

Plan initially for one 80 GB CUDA GPU, sharing one model's weights across sequential writer and receiver calls while clearing per-example state. That is a conservative requested allocation, not a measured minimum. The paper's reported hardware is also not a minimum requirement. First profile a small batch; derive the full job estimate from measured throughput. The old $15 API cap does not meter GPU-hours.

Proceed to confirmation if the implementation reliably transmits fictional evidence and offers a useful efficiency signal against competent text. If StateBridge cannot pass basic decoding controls, diagnose the adapter or run one explicitly bounded learned-bridge pilot. Do not interpret a broken decoder as evidence against all latent communication.

The outcomes are all informative:

- Better utility with fewer positions and lower total inference cost supports a practical efficiency claim on that stack.
- Fewer positions but larger payloads or higher runtime supports a narrower context-capacity claim.
- Better current-task accuracy with worse future accuracy shows that latent communication preserves or amplifies specialization.
- Gains that disappear when source-conditioned states are replaced by transcript-only states are consistent with a source-conditioned contribution. A fixed-transcript factual substitution versus sham is needed to attribute the effect to a particular retained fact.
- No useful advantage after a valid decoder and strong controls limits the case for replacing compact text in this setting.

The recommended next action is to complete and audit the scaffold's alignment, pin the model/environment, verify Bunya GPU access, and run the small frozen-interface pilot's decoding and isolation gates. Training a new protocol becomes worthwhile after the measurement system is trustworthy and the first result identifies what needs improvement.

## Audit trail

Primary-source and code notes: [Interlat and DiffMAS](pivot_interlat_diffmas_notes.md), [StateBridge feasibility](pivot_statebridge_notes.md), [independent design audit](pivot_design_audit_notes.md). These notes are supporting research, not experimental findings. [Experiment 13](../EXPERIMENT_13_DESIGN.md) is the consolidated proposed protocol and takes precedence over alternative pilot sizes suggested in the notes.
