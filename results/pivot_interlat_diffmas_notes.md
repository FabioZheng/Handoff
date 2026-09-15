# Interlat and DiffMAS: primary-source audit for the communication pivot

Checked 2026-09-07. Read-only literature/code inspection; no models downloaded, no training or benchmark execution. Repository availability is not a reproduction guarantee.

## Interlat

**Paper facts (v5, 2026-07-13).** The channel is a sequence of final-layer hidden vectors, inserted into receiver embeddings after a learned attention/projection adapter. Receiver training combines supervised task loss, sensitivity to matched versus mismatched messages, plan alignment, and token/latent mixing. Compression separately trains an autoregressive latent sender while freezing the receiver. Cross-family evidence trains a LLaMA receiver on Qwen messages; it does not establish arbitrary receiver substitution without training. Models include Qwen2.5-0.5B/7B and LLaMA3.1-8B. Reported training uses bf16, FlashAttention-2, and DeepSpeed; specific GPU quantities and GPU-hours were not identified. Evaluation covers ALFWorld and MATH. [Paper, sections 3–5](https://arxiv.org/html/2511.09149v5)

**Efficiency denominator.** Table 3 measures message-generation latency: 9.19 seconds for full communication versus 0.39 for eight untrained latent steps, approximately 24 times faster. Trained eight-step communication takes 0.20 seconds, with Qwen unseen success 60.45% versus full-message 65.42%. These are neither byte savings nor full-task latency savings. Internal-state access and opaque messages limit applicability and debugging. [Paper, Table 3 and section 7](https://arxiv.org/html/2511.09149v5)

**Official repository.** The authors provide collection, receiver training, compression training, and evaluation code. README instructions require a trained receiver and hidden-state data before compression; they do not identify a ready-to-load trained checkpoint. Its compatibility statements should be treated as supported workflows to audit, rather than proof of universal interoperability. [Repository](https://github.com/XiaoDu-flying/Interlat)

**Implementation inspection.** `quick_start.sh` selects Qwen2.5-0.5B by default, prepares training records, and runs two receiver-training epochs. ALFWorld requires a separately supplied dataset file. The quick path is a training demonstration, not a reproduced paper result or pretrained inference endpoint. [Quick-start script](https://github.com/XiaoDu-flying/Interlat/blob/main/scripts/quick_start.sh)

`compression_training/compress.py` freezes the teacher and retains the student parameters as trainable. It generates K final hidden vectors by feeding each through a learned hidden-to-embedding map, with the teacher processing that sequence. The main loss is evaluated using teacher-conditioned outputs. This is appreciably more work than attaching a codec to the existing API runner. Its heterogeneous claims warrant a tokenizer/dimension smoke test before committing to a new writer–reader pair. [Compression implementation](https://github.com/XiaoDu-flying/Interlat/blob/main/compression_training/compress.py)

## DiffMAS

**Paper facts (v1, 2026-04-23).** Four roles share a pretrained transformer and append KV blocks to a persistent trace. This preserves prior computation while increasing the conditioning object with depth. Task-specific rank-8 LoRA training uses 210 math examples, 50 coding examples, or 700 commonsense examples. A40 GPUs support smaller models and H200 GPUs larger ones; counts and hours are unspecified. Multiple backbones are tested separately; arbitrary mixed-backbone KV transfer is not demonstrated. Table 1's Qwen3-8B gains are 26.7 and 20.2 percentage points over single-agent AIME24/GPQA, not absolute accuracies or compression ratios. [Paper, sections 3–4 and appendices B–D](https://arxiv.org/html/2604.21794v1)

**Training-detail uncertainty.** Section 3.3 and Appendix C describe gradients through upstream latent computation. Figure 1 instead describes upstream trace construction without gradient updates and final-agent LoRA updates. Shared weights can couple role behavior, but do not settle whether upstream computation remains in the backward graph. The safe description is a learned KV communication system; the exact implementation requires code or author clarification. [Paper, Figure 1, section 3.3, Appendix C](https://arxiv.org/html/2604.21794v1)

**Code availability audit.** No official repository link was found in the paper/arXiv record or the exact-title and DiffMAS GitHub searches. A third-party implementation/discussion is not author code. Therefore classify DiffMAS as a conceptual comparison, not an immediately runnable primary experimental arm, pending an official implementation. This is an availability observation, not proof that no code exists anywhere. [arXiv record](https://arxiv.org/abs/2604.21794)

## Accounting implications for our experiment

These methods already establish that non-natural-language exchange is possible. A stronger next claim is that a channel reduces actual cost while retaining immediate and unforeseen future utility.

Count separate resources rather than treating each vector as a text token:

- Text: actual UTF-8 bytes and token IDs delivered, sender generation tokens, receiver prefill tokens, all protocol instructions.
- Final-layer vectors: K × hidden dimension × bytes per scalar, plus masks and metadata.
- KV trace: sum over layers of 2 × stored positions × KV heads × head dimension × bytes per scalar, plus metadata. Count cumulative trace resends when agents are on separate hosts; count resident cache and memory traffic when colocated.
- Total execution: source encoding, message generation/compression, transfer, receiver work, final answer, and training amortization. Report GPU time and wall time separately.

**Calculated illustration, not a measured benchmark:** official Qwen3-4B configuration has 36 layers, eight KV heads, head dimension 128, hidden dimension 2560, and bf16 weights. At bf16 payload precision, eight last-layer vectors occupy 40 KiB. Eight complete KV positions occupy 1.125 MiB; one KV position is 144 KiB. Prompt/source cache positions add further storage if included. This illustrates why fewer communication positions need not mean fewer bytes. [Official Qwen configuration](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json)

## Recommended bounded extension

Keep the gold-source, sealed-message, hidden-future-query design. Compare an available frozen latent channel with strong text/structured-symbolic baselines first. Then introduce an Interlat-style trained compressed channel if the pilot establishes causal use of source information and feasible GPU cost. Split by source and template before training. A latent channel that never reliably carries fresh fictional facts is an integration failure, not evidence against the scientific hypothesis.

Measure current-query utility, held-out-query utility, and complete utility matrices at matched execution or transfer constraints. Include source-prefill/cache reuse as an engineering baseline, with its retained source bytes counted openly. Learned protocols require evaluation on unseen sources, unseen query types, and at least one changed receiver. Preserve controls that swap messages, zero the payload, or restore answer-bearing information to establish causal dependence. These are proposed experimental choices; they are not findings from the cited papers.
