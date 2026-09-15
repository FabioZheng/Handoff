# Independent design audit: a non-language communication pivot

Prepared 7 September 2026. This is a proposed design, not an experiment result or a claim about the exact implementations of Interlat, StateBridge, or DiffMAS. Read-only review of the current repository; no model calls or code tests were run.

## Recommended research question

**At a measured communication or inference cost, does a non-language channel preserve useful evidence for an unannounced later question better than a strong text channel, and does conditioning on the current question still cause specialization?**

This preserves the project's distinctive contribution: the sender knows A, the receiver can be asked B, and the receiver has no source access. A generic claim that latent communication works or uses fewer context positions is already the motivation of the cited methods. The useful extension is to separate representation efficiency from evidence selection under changing tasks.

Repository basis: `results/bounded_communication_frontier/REPORT.md` reports 10/10 structured allocation trends in the expected direction, but 0/4 primary gains with a distribution-sensitivity grid survive every designated setting. The pivot should therefore use a clear primary contrast and avoid promising universal frontier improvement. `src/budget.py` controls delivered words, which is useful for the earlier text study but cannot compare dense vectors, codebook indices, and text.

## Two estimands that must remain separate

1. **Coding effect, with information selected before encoding held fixed.** Give the prose, symbolic, and latent encoders exactly the same selected source facts. Compare decoded correctness and cost. This diagnoses whether a format/decoder preserves a known semantic payload. Use both A-focused and balanced selections, preferably with deterministic fictional records whose contents and provenance are explicit. This is a controlled mechanism experiment, not an evaluation of a deployable selector.
2. **End-to-end system effect.** Give each sender the full identical source, with A either disclosed or withheld. Let its own encoder choose what survives. Compare present and hidden-future utility at each measured resource level. This is the practical comparison; different extraction/encoding policies are part of the treatment, not pure format effects.

Do not compare a latent encoding of the full source only with a prose handoff already selected for A and conclude that latent communication reverses information loss. It received information the text arm had discarded. Conversely, compressing an existing handoff into vectors cannot establish recovery of facts absent from that handoff. It can reveal facts present in continuous states but absent from a surface rendering only if those states were produced with access to the original source; document that access explicitly.

## Resource accounting

- Primary first claim: **receiver context efficiency**, measured in all added receiver positions at matched utility. A latent slot counts as an input position, not as a billed text token or one transferable byte. Count decoding headers, format legends, special markers, and any required transmitted conditioning query in each arm.
- Report actual serialized message bytes separately. A dense message with m slots, d coordinates, and b bits per coordinate costs at least m*d*b/8 bytes plus metadata. As an illustrative arithmetic example, 32 slots at d=4096 in FP16 contain 262,144 bytes. Sixty-four int32 token IDs contain 256 bytes before metadata. This illustration must not be presented as a measured model result.
- Report sender prefill, sender generation/encoding, mapping, serialization/transfer, reader prefill, and reader decoding latency separately; also peak memory and total accelerator-seconds. Use warm repeated measurements with fixed batch sizes. Less reader input does not necessarily mean less total compute.
- Report one-time adaptation/training cost and amortized cost for 1, 10, 100, and 1,000 handoffs. Keep model loading outside steady-state timing and show cold-start separately if relevant.
- A codebook requires its deployment cost and version/hash. A per-document dictionary must travel with the message and count toward its cost. Gold fact IDs or a source lookup table at the receiver create an extra channel.
- Lossless gzip-like compression demonstrates network-byte reduction only if the receiver decompresses it to the original text; it does not reduce model context or inference tokens.
- Present matched-position, matched-byte, and end-to-end-cost results as different comparisons. A dense latent system can be a win on the first and a loss on the other two.

## Frozen and learned tracks

Start with one open-weight reader whose exact revision supports inspecting and supplying continuous inputs. Keep its base weights, answer prompts, generation settings, and tokenizer fixed across all text and continuous conditions. Any latent-specific alignment, mapping, or adapter is part of the communication system and must be named and costed; silently replacing the reader invalidates a channel comparison.

Run a training-free mapping/replay method first only if its implementation and interface are available and it can pass the diagnostic controls below. Arbitrary hidden states are not guaranteed to be useful input embeddings. A failed training-free pilot means that this implementation does not decode reliably; it is not evidence that latent communication cannot work.

If the frozen path fails, use a bounded learned track: freeze the reader and learn only an encoder/bridge into its input space. Train on new, disjoint synthetic sources; do not train on the existing 16 dossiers or the future test questions. Use one bridge across generic and A-conditioned inference if possible; keep its training distribution and supervision constant across those conditions. If separate policies require separately trained models or objectives, label the result a system comparison, not a pure conditioning contrast. Report natural-text performance of the exact reader before and after any adaptation if reader weights must change.

Do not begin with joint end-to-end multi-agent training. It adds training-objective, reasoning-depth, and optimization confounds before the channel measurement is established. Rich all-layer states or KV transfer can be a diagnostic upper reference, but are not a bandwidth-saving baseline unless measured as such.

## Finite sequence and gates

### Stage 0: feasibility and accounting, before selecting a model

Inspect available accelerator hardware/allocation and confirm open-weight inference with continuous inputs. Existing HPC scripts request general CPU nodes with 4-12 GB memory; they are not evidence of an available GPU. Existing OpenRouter infrastructure and a working Bunya smoke test do not establish latent-state access or training capability.

Pick one feasible small or medium model only after measuring memory and a short forward pass. Set a fixed compute envelope from that measurement. Do not infer that the existing $15 API ledger controls GPU training costs. Save an environment/weights/codec manifest.

Gate: normal text QA works; the exact continuous-input path works; fresh receiver execution cannot access sender tensors, source KV cache, or source text except the explicit message. If no continuous path is available, a compact symbolic-text pilot is still runnable but must be described as symbolic communication, not latent communication.

### Stage 1: preferred immediate pilot after coordination

Use the existing 20 fictional six-card dossiers at K=4. Reuse one generic and six A-conditioned selections per dossier, giving 140 selected source bundles. Reset the encoder before each bundle. Give each of three encoders exactly that same bundle: prose, typed records, and a training-free StateBridge-style continuous prefix (candidate prefix length 32, to be verified). Answer all six questions. The core pilot contains 20*7*3*6 = **2,520 answer evaluations**, plus explicitly itemized controls. The same text reader weights and answer-generation settings must be used throughout. This finite pilot is preferred over immediately launching the larger grid below.

The selected bundle is the entire source available to each encoder. Card IDs and evaluator provenance must not reach the reader. The frozen mapping's generated text/hidden trajectory still has a cost: record its generation cap G, realized generation length, decoder prefix positions, and transfer bytes. A prefix of length 32 must not be reported as the whole end-to-end communication/inference expense if forming it generated substantially more tokens.

A separate diagnostic can compare original source-conditioned hidden states with fresh replay of the visible handoff H alone. Improvement from the original states means they transmit source-conditioned computation beyond what is recoverable from H; it is not evidence that the same H was encoded more faithfully. Keep this as an explicitly different source-access panel. After this selected-source pilot, run one full-source panel and then the existing 16 relation dossiers for distance diagnostics, without treating any of these repeatedly used corpora as confirmatory evidence.

The user has selected Bunya GPUs subject to resource availability. The coordinator found no active multiplexed SSH socket and did not initiate login. A proposed first backbone is Qwen/Qwen3-4B, frozen, BF16, with thinking disabled; this remains conditional on hardware/runtime verification. No GPU access, memory sufficiency, or paper-faithful reproduction is established by this document.

### Stage 1 controls and possible learned fallback

Use the existing 16 fictional relation dossiers only as development material. They have been used repeatedly and cannot be a clean held-out confirmation set. Select a fixed set of source facts and encode identical facts as canonical prose, compact triples/key-value form, and the candidate continuous message.

Use eight dossiers, two fixed selection patterns per dossier, and at most three message-size settings. Include full source, selected prose, empty message, shuffled cross-source message, zeroed latent message, and direct token-embedding replay of the selected prose. The replay path should reproduce text behavior closely enough to verify that answer differences are not from a broken input wrapper. Fact-wise tests distinguish encoding failure from missing selection. For a source-selected payload known to omit B, exact restoration versus an equal-size irrelevant-fact sham is a positive intervention, consistent with Experiment 8b.

Go gate: high direct-source accuracy, near-chance fictional closed-book/shuffle performance, and usable latent decoding. Suggested design thresholds to freeze after the diagnostic: direct-source >=0.90; closed-book <=0.05; latent high-capacity accuracy no more than 0.10 below matched selected prose. These are engineering gates, not significance tests or claims of superiority. Failure triggers debugging or the learned bridge track, not a wider benchmark grid.

If training is needed, cap the first adaptation pilot at a predeclared small synthetic corpus, e.g. 512 training and 64 validation dossiers, with a fixed step/accelerator-hour limit and validation early stopping. Treat those as a feasibility allocation, not an assurance of sufficient training. Test sources, entity/value dictionaries, and preferably template families remain disjoint. Teacher outputs and synthetic validation data cannot include held-out test facts.

### Stage 2: primary paired test, with a small grid

Create at least 100 fresh fictional sources after freezing prompts/codec, with final sample size chosen from the development context variance and a prespecified power/precision target before reading test outcomes. A concrete planning size is 128 if its precision is sufficient; it is not an evidence-based power calculation. Each source has four aspects and four A rotations. For each rotation score A plus one paraphrase, one same-topic question, and one orthogonal question, balanced across aspect and target fact. This yields 16 scored question instances per source, avoiding a very large full 16-by-4 matrix. Source is the independent sampling unit.

Core grid: 2 channel types (strong text, selected continuous candidate) x 2 sender policies (generic, A-conditioned) x 2 context-budget rungs. Start with 64 and 128 added receiver positions as candidate rungs, subject to the Stage 1 feasibility audit; freeze final rungs before test evaluation. This is eight conditions, 16,384 scored answers at n=128. Deterministically identical generic messages should be generated once per source/budget and reused safely; receiver answer requests remain source-isolated and cache-keyed.

Strong text should include a realistic selected method determined on development data, rather than the weakest prose prompt. Compare prose, compact symbolic text, extractive text, and the current allocation control on development. For causal interpretation keep the prose arm alongside the best compact text comparator in the main test if development selects a different method. Adding one fixed comparator at both budget rungs adds 4,096 scored answers at n=128. Keep all development outcomes visible; freeze selection criteria, do not choose a winner on test data.

No cross-family grid, repeated handoff depth, noisy-future-query grid, or five-point alpha sweep in this first test. Those belong after a clear signal. A cross-family decoder transfer is a follow-up portability stress test, not a requirement for the first feasibility study.

### Stage 3: one validation extension, conditional on Stage 2

If the latent system shows a useful context-efficiency or compute gain without future-utility collapse, run one fresh natural-passage holdout or one second reader family. Pick the extension that resolves the main remaining uncertainty; do not pool it silently with the primary homogeneous-model study. Natural data need reader-specific closed-book screening and direct-source checks.

## Primary endpoints and analysis

At the smaller prespecified context budget, estimate the paired representation effect on U_future,orthogonal and on U_now. A benefit claim requires a prespecified worthwhile improvement, e.g. >=0.05 in future correctness, together with noninferiority of current correctness at a -0.03 margin, using source-clustered paired intervals. These suggested margins are design choices to agree before the test; they are not discovered from data. If the scientific objective is primarily token savings, invert this as the minimum measured receiver positions needed to attain both prespecified utility targets; do not interpolate beyond observed budgets or assume monotonicity.

Mechanism endpoint: the channel x conditioning interaction for the specialization gap, where specialization is U_now - U_future,orthogonal. Report the two utility changes alongside the gap: a smaller gap produced by losing current answers is not mitigation. Always report paraphrase and same-topic utilities separately; orthogonal utility is the primary future target so the count of unrelated questions cannot silently determine weights.

Secondary endpoints: actual bytes, total inference latency/accelerator-seconds, encoder training amortization, exact fact recovery, wrong-fact substitutions, EM/F1, and blinded semantic judging for natural-language answers. Exact fictional values allow deterministic scoring; audit acceptable aliases without inventing them after seeing model outputs. Use paired bootstrap resamples of entire source contexts, not question rows. Prespecify one primary budget and comparison; other rungs and models are sensitivity analyses, with complete results retained.

A defensible positive conclusion might be 'continuous messages preserve utility with fewer receiver positions on this stack, while consuming more transport bytes.' A null/negative conclusion might be 'latent communication compresses the same specialization rather than repairing it.' Both address the research question. Stop escalating if a reliable positive-control decoder still gives no useful efficiency gain over compact text under the declared cost metric; do not increase model size, training, and grid breadth until a gain appears.

## Isolation and reproducibility requirements

Introduce a new immutable sealed transport record for this experiment rather than weakening the current `SealedHandoff` string-only guarantee. Include explicit payload type, dtype/shape, codec/checkpoint version, payload hash, and declared public metadata. Reader inputs consist only of that payload plus the evaluation question and fixed public instructions. No source IDs that can resolve to answer tables; no sender KV cache unless it is explicitly the transferred and counted payload; no whole-source prefix reused through a shared generation session.

Hash source, source split, current-question visibility, prompt, representation, selection, tokenizer/weights revision, bridge checkpoint, training-data manifest, tensor dtype, quantizer/codebook, dimensions, seed, and generation settings in cache provenance. Count retries, length rewrites, warmup separately, and all scored requests including failures. Keep strict dry-run nonmutation guarantees. Check random-code/shuffle controls with opaque receiver-side IDs to prevent accidental cache hits from restoring a different payload's answer.

## Main risks to flag in the final report

The existing 16 dossiers are excellent diagnostics but too reused and too small for a new learned-channel claim. Training and testing on different questions from the same source leaks source knowledge. A frozen reader plus a learned bridge is not a zero-training system. A latent position is a large vector, not a text token. A full-source latent encoder and an A-filtered text encoder answer different causal questions. A different latent decoder is a different reader system. Receiver-side source memory, shared KV state, metadata dictionaries, and template-specific gold lookup each create a second channel. Format efficiency and downstream utility need to be demonstrated together.
