# Agent Handoff Information-Loss Probe

Measures how much answer accuracy is lost purely from summarizing evidence
for a handoff between agents (e.g. a subagent's report to an orchestrator),
with retrieval held constant by handing the subagent gold evidence directly.
Full experimental design: [README.md](README.md).

## Current state

- **Built and verified:** stages 0, 1, 2, 4 — mechanisms `A_full`, `B_freeform`,
  `C_structured`, `D_extractive`, `E_oracle`. Offline selftest
  (`src/selftest_offline.py`) passes in full, including the stage-2 isolation
  guarantee (orchestrator provably cannot see source paragraphs, only the
  sealed handoff text).
- **Repeated-handoff extension built and verified:** `src/run_chain.py`,
  `src/chain_data.py`, and `chain_config.yaml` compare depths 0/1/2/3/5 across
  MuSiQue and HotpotQA at gold-only, five-document, and full-context evidence
  lengths. They produce answer-level records, bootstrap tables, and
  `results/chain/degradation.png`. `src/selftest_chain_offline.py` passes,
  including validation against the real HotpotQA parquet schema.
- **10-question baseline pilot run:** `B_freeform` beat `A_full` (EM 0.60 vs
  0.40; F1 0.748 vs 0.484) and tied `E_oracle`. This motivated measuring
  repeated compression rather than building Stage 3 immediately.
- **Not built:** `src/inject.py` (Stage 3, the six corruption types). Per the
  build order, this is deliberately deferred until the `A_full`/`B_freeform`/
  `E_oracle` headroom is measured at n=10 — if there's no headroom, the design
  needs rethinking before Stage 3 is worth building.
- **Bunya HPC access:** working end-to-end as of 2026-08-13. See "Bunya HPC"
  below — this is the part most likely to need re-establishing in a new
  session, since it depends on a live SSH connection, not just files.
- **Model-heterogeneity experiment built and run (Experiment 8):**
  `src/run_model_heterogeneity.py`, `src/model_pool.py` and
  `model_heterogeneity_config.yaml` vary *which model writes each handoff*
  across depths 0-6 on the prebuilt SQuAD same-passage pairs (n=20), holding
  prompts, budgets, evidence and the final answerer fixed. Four open-weight
  dense families at two size tiers; 17 generated arms plus two start-matched
  derived controls. `src/selftest_model_heterogeneity_offline.py` passes.
- **Fictional fixed-capacity replication built and run (Experiment 5b):**
  `src/run_fictional_summary_generalization.py` evaluates 20 fictional
  dossiers × six A rotations at K=2/4/6. At K=4, conditioning changes judged
  immediate/reusable accuracy by +0.317/−0.062; at K=6 the paired-compliant
  all-card delta is exactly zero. Direct-card/closed-book accuracy is 1.0/0.0.
  Identical answer requests are single-flighted by request hash; this is a
  regression-tested invariant, not left to the on-disk cache under concurrency.
- **Focused selector/relay bottleneck built and run (Experiment 8b):**
  `src/run_fictional_model_bottleneck.py` crosses Llama and Qwen small/large
  source selectors and sealed relays on 20 fictional dossiers × four A→B
  rotations, with a fixed Mistral reader. The upstream-size effect does not
  replicate across families. On clean text-level B omissions, sealed accuracy
  is zero; exact same-width restoration beats sham by +0.983 (Llama) and
  +0.950 (Qwen). Presence/eligibility inspect delivered answer-bearing text,
  not only evaluator card ids. Both fictional offline selftests pass.
- **Size-adaptation experiment built (Experiment 9):**
  `src/run_size_adaptation.py`, `src/size_metrics.py`,
  `src/build_size_adaptation_data.py` and `size_adaptation_config.yaml` vary
  *what the writer is told about the next agent's context size* across three
  handoffs: control, "informed" (capacity stated) and "resize" (the same
  sentence plus a request to act on it), at 2k and 10k, plus large->small->large
  trajectories. Two purpose-built datasets make new material attributable --
  counterfactual items verified closed-book in BOTH directions, and fictional
  items the model cannot know at all.
  `src/selftest_size_adaptation_offline.py` passes. **Run and analysed**
  (46 items: 26 counterfactual + 20 fictional, 9 arms, depths 0-3, $14.79).
  Headline: `resize_large` ("expand to use the space") is **bimodal** -- 128/230
  calls terminate normally at a median 783 words with 0.087 internal repetition,
  while 102/230 run to the token budget at a median 9,687 words with **0.912**
  internal repetition, i.e. a greedy-decoding loop rather than added detail. A
  probe at `max_tokens=32000` reproduced the same split, so this is NOT a
  too-small budget; do not "fix" it by raising `handoff_max_tokens`. Expand-arm
  *means* are therefore a mixture of two modes and describe neither, and any
  string-presence measure is inflated by the loop restating everything. The
  depth-3 control, informed and fixed-shrink rows used for the headline
  contrasts have no budget hits (one fictional fixed-shrink row ends in an
  error). Informing alone moves length (+119.5 / +189.3 words vs control, CI
  excludes 0), while adding the 2k fit request partly reverses that expansion
  (-42.8 / -87.6 vs informing), and the
  explicit cue "Keep your handoff concise" was costing -0.277 side-fact
  retention relative to the size-neutral control.
  Recovery is underpowered -- 0.000 recovered and 0.000 reverted everywhere it
  is defined -- so the large->small->large question is not answered by this run.
- **Communication-regret experiment built and run (Experiment 10):**
  `src/run_communication_regret.py`, `src/budget.py`, `src/regret_data.py`,
  `src/build_regret_data.py` and `communication_regret_config.yaml` test whether
  conditioning a handoff on the currently known query trades future-query
  utility for present-query utility **under a hard communication budget**. Every
  question of a multi-query source rotates through the conditioning role and
  every message is answered on every question, giving a per-context utility
  matrix (diagonal = `U_now`, off-diagonal = `U_future`). Budgets 20/40/80/160
  delivered words. On 24 SQuAD groups x 4 questions (n=24 contexts, 96
  questions, 1,440 messages, 7,104 answers, $0.145 including the judge): the
  hypothesis is **supported** at the 20/40/80-word budgets -- `conditioned`
  beats `generic` on `U_now` by +0.625/+0.448/+0.281 while losing
  -0.063/-0.115/-0.194 on `U_future`, both directions excluding zero, and the
  conditioning-induced specialisation gap grows +0.413 [0.326, 0.503] as the
  budget falls from 160 to 20 words. `oracle` reaches 0.823 on *every* question
  with the same ~37 words, so the channel could carry the hidden answers; the
  conditioned policy simply does not spend it on them. `reusable` (told further
  questions may follow) recovers little: it still loses future utility against
  `generic`. **Replicated on 16 purpose-built relation dossiers** (16 questions
  each, ceiling 0.992, closed-book 0.000): the same pattern, larger, plus the
  distance result -- future-query regret rises from ~0.00 at a paraphrase of the
  conditioning query to ~0.95 at an orthogonal aspect
  (`R(orthogonal) - R(paraphrase)` = 0.94-1.00, dz 9-30), and the gradient is
  absent from `generic` (0.04-0.15) and `oracle` (0.06-0.09). At the 160-word
  budget -- a third of the dossier -- `generic` cuts orthogonal regret to 0.477
  and `oracle` to 0.160 while `conditioned` stays at 0.952: extra bandwidth
  deepens the chosen aspect rather than spreading the message.
  `src/selftest_communication_regret_offline.py` passes.
- **Not yet run:** the repeated-handoff experiment or full 150-question
  baseline run on Bunya. Only the 15-minute smoke test (`hpc/test.slurm`) has executed there,
  and it passed — venv builds, offline selftest passes, compute node reaches
  `openrouter.ai` (HTTP 200).

## Stack and environment

- Python, `openai`-compatible client against OpenRouter (`src/llm.py`).
- Model: `meta-llama/llama-3.3-70b-instruct` (mid-capability, deliberately not
  frontier — see README "Model choice").
- Local dev: Windows, Python venv at `.venv/`.
- Bunya: separate venv built per-job at `.venv-bunya/` (excluded from sync).

## Commands

| Purpose | Command |
|---|---|
| Install (local) | `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt` |
| Dry-run cost estimate | `python src/run.py --dry-run --mechanisms A_full,B_freeform,C_structured,D_extractive,E_oracle` |
| 10-question pilot | `python src/run.py --candidates 40 --n 10 --mechanisms A_full,B_freeform,E_oracle` |
| Full probe | `python src/run.py --mechanisms A_full,B_freeform,C_structured,D_extractive,E_oracle` |
| Offline checks (no API key) | `python src/selftest_offline.py` |
| Model heterogeneity (Exp. 8) | `python src/run_model_heterogeneity.py` |
| Model heterogeneity offline checks | `python src/selftest_model_heterogeneity_offline.py` |
| Build Exp. 9 datasets | `python src/build_size_adaptation_data.py --which both` |
| Handoff size adaptation (Exp. 9) | `python src/run_size_adaptation.py` |
| Size adaptation offline checks | `python src/selftest_size_adaptation_offline.py` |
| Build Exp. 10 corpora | `python src/build_regret_data.py --which both` |
| Communication regret (Exp. 10) | `python src/run_communication_regret.py` |
| Communication regret, relation corpus | `python src/run_communication_regret.py --dataset relation_dossiers --policies generic,conditioned,reusable,oracle --no-trim` |
| Communication regret offline checks | `python src/selftest_communication_regret_offline.py` |
| Regenerate the Exp. 10 cross-corpus report | `python src/render_regret_summary.py` |

`OPENROUTER_API_KEY` is read from the environment only — never from a file or
argument. Cost is hard-capped at `$15.0` in `config.yaml` (`cost.cap_usd`).

## Conventions

- Every LLM call is content-hashed into `cache/` — reruns are free, and the
  cache key must track every parameter that affects output (temperature,
  seed, etc.) or two different calls silently collide.
- `A_full` and `E_oracle` are deterministic (temperature 0, no sampling) and
  generated once per question, not once per seed. Only `B_freeform`,
  `C_structured`, `D_extractive` run under both seeds (`config.yaml: seeds`).
- `orchestrator_answer()` accepts only a frozen `SealedHandoff` (four strings:
  `qid`, `question`, `handoff_text`, `mechanism`) — never a `Question` object,
  dict, or raw string. This is the load-bearing isolation guarantee; don't
  weaken it to make a mechanism easier to implement.
- One `LLMClient` is pinned to one model. An experiment that needs several
  models per run builds a client per model through `src/model_pool.py`, which
  hands them all the *same* `DiskCache` and the *same* `CostLedger` — otherwise
  `cost.cap_usd` silently becomes one cap per model instead of one cap per run.
  The model id is already inside `llm.cache_key`, so shared caching is safe.
- `--dry-run` must never write to `data/`, `runs/` or `results/` (README
  "Cost control"). A runner that writes unconditionally will overwrite real
  records with empty dry-run placeholders; guard every `write_jsonl`.
- Config is centralized in `config.yaml` so the whole pipeline is
  model-agnostic — change `model.id` there, not in code, for a second-model
  robustness check. `data/filtered_questions.jsonl` carries a manifest (model
  id, dataset source + content hash, sampling, leakage-filter config) as its
  first line; `stage0` auto-regenerates it when the manifest no longer
  matches the current config or dataset file, so no manual delete or
  `--force` is needed for this specific case.

## Decisions

- **MuSiQue over HotpotQA** — MuSiQue's paragraph-level (not sentence-level)
  gold annotations required deriving gold sentences deterministically from the
  gold decomposition, rather than coarsening `E_oracle` to whole paragraphs,
  which would have made it nearly a copy of `A_full` and destroyed the
  compression contrast.
- **Cross-model comparisons are paired only on the prebuilt SQuAD pairs.** The
  C1 leakage filter is model-specific, so the Qwen replications of Experiment 2
  do not share a question set with the Llama run and cannot be paired. The
  model-heterogeneity experiment therefore runs on `pairs_n20.jsonl`, which was
  filtered once at build time against `meta-llama/llama-3.1-8b-instruct` — also
  its fixed primary answerer, so the filter still applies to the model that
  produces the scored answer.
- **Mid-capability model, not frontier** — a frontier model would ceiling out
  on `A_full`, leaving no headroom to measure, and would answer more often
  from parametric knowledge despite the C1 filter.
- **EM/F1 are primary; an LLM judge is a secondary semantic metric** — the
  original decision was *no* LLM judge, to keep results free of a second
  model's judgment calls. That was revised on 2026-08-20: token F1 penalises
  answers that are correct but differently worded or more verbose, which is
  exactly what repeated compression produces, so F1 alone overstates
  degradation. `src/judge.py` adds a binary correct/incorrect verdict against
  the gold answer. EM/F1 are still computed and reported alongside it, never
  replaced, so every judge verdict stays auditable. The judge is deliberately a
  **different model family** (`openai/gpt-4o-mini`) from the llama systems
  under test, to avoid self-preference bias. It replaced BERTScore, which was
  never interpretable as "is this answer right".

- **Experiment 9 does not inherit Experiment 5's handoff instruction.** Every
  other chain experiment reuses `run_chain.INITIAL_INSTRUCTION` /
  `RECOMPRESS_INSTRUCTION` byte-identically. Those open "Write **concise** prose
  research notes", which is itself a shrink instruction — fine everywhere else,
  fatal for an experiment about size: measuring an expansion request against it
  compares "expand" with "shrink" rather than with "no size instruction", and
  measuring a shrink request against it compares against an already-shrunk
  baseline. Experiment 9's base is the published string with the single word
  `concise ` deleted and nothing else changed, enforced by
  `run_size_adaptation._drop_concise`, which raises unless the substring occurs
  exactly once. A separate arm appends "Keep your handoff concise", measuring an
  explicit unquantified concision cue rather than restoring the deleted word in
  place. Do not "fix" the neutral base back to the shared constant.
- **The counterfactual dataset needs the model to KNOW the original.** The
  existing `data/wikipedia_random_counterfactual/` was built from *random*
  Wikipedia pages under the opposite requirement (the model must not know the
  answer), so only 6 of its 20 items pass Experiment 9's knows-the-original
  gate. Rather than run at n=6, `build_size_adaptation_data.py` adds a
  `known_entity` source that samples famous subjects first, verifies the model
  answers them closed-book, and only then rewrites the fact. Both sources live
  in one file, each item tagged with `source`, so they are never silently
  pooled — split on it before reporting a counterfactual result.

## Already tried — don't redo

Nothing has been tried and abandoned yet on the core pipeline; the design
in README.md is the first and current approach. The rest is infrastructure:

- OpenRouter returns **HTTP 402 `in_flight_budget_exhausted`** when too many
  expensive requests are in flight at once, even with credit available. It is
  transient and ships a `Retry-After`, unlike a genuinely empty account. Eight
  concurrent `openai/gpt-5.2` high-effort calls hit it and killed a dataset
  build. Fixed in two places: `llm._post_with_retries` now retries a 402 *only*
  when the body names that reason, and `build.writer.concurrency` in
  `size_adaptation_config.yaml` fans the strong writer out more narrowly than
  `runtime.concurrency`. Do not raise the writer's concurrency to match the
  experiment's.
- **A shared word cap does not equalise a channel.** Experiment 10's first
  pilot gave every policy the same one-sided "at most N words" contract. At a
  40-word cap the realised fill ratios were 0.61 (`conditioned`), 0.75
  (`reusable`), 0.83 (`oracle`), 0.88 (`generic`) -- the conditioned arm spent a
  third less channel than its control, so any future-query deficit would have
  meant "it wrote less". The fix is the two-sided band in `src/budget.py`
  (`budget.fill_floor_ratio`), which brought every arm to 0.88-0.96 fill with a
  1-6 word standard deviation and zero measured repetition. Do not simplify it
  back to a cap: `max_tokens` alone is not a length control, and neither is a
  cap alone.
- Windows OpenSSH does not support `ControlMaster`/SSH multiplexing, which is
  why the Bunya SSH config lives in WSL Ubuntu rather than PowerShell or Git
  Bash — don't move it back.
- `ssh bunya 'sbatch --export=ALL,OPENROUTER_API_KEY ...'` does **not** forward
  the key, even with it exported in the calling shell first — SSH runs the
  remote command in a fresh shell on Bunya's login node and does not carry
  arbitrary local environment variables into it. Failed twice this way (jobs
  `27205781`, `27205834`) before switching to piping the key over SSH's stdin
  into a remote shell that exports it before calling `sbatch` (job `27205848`
  succeeded). Use the stdin-piping form in "Bunya HPC" below, not plain
  `--export=ALL,VARNAME`.

## Constraints

- Never store the OpenRouter key or Bunya password/credentials in a file —
  both are read from environment/entered interactively only.
- `A_full` − `B_freeform` under ~3 points with no distinguishable corruption
  effect means: report that the handoff isn't where the loss is, and stop —
  do not tune the setup to manufacture an effect (README "If the probe says
  stop").

## Bunya HPC

UQ's Bunya HPC cluster requires interactive password + Okta MFA push on
**every** login — there is no credential that can be stored for unattended
access, by cluster policy. The working model: authenticate once by hand, then
reuse that authenticated connection via SSH multiplexing for as long as it
stays open.

**Everything below runs inside the Ubuntu WSL terminal directly** (no `wsl -d
Ubuntu --` prefix — that prefix is only for reaching WSL *from* a Windows
shell like PowerShell). Windows' native OpenSSH doesn't support
`ControlMaster`, which is why this depends on WSL specifically.

- **Username:** `<your-uq-username>`
- **AccountString:** `a_ai_collab` (required by Slurm's accounting enforcement
  — jobs without a valid one are rejected). Already set in `hpc/job.slurm` and
  `hpc/test.slurm`.
- **SSH config:** `~/.ssh/config` in WSL Ubuntu defines a `bunya` host —
  `HostName bunya.rcc.uq.edu.au`, `User <your-uq-username>`, `ControlMaster auto`,
  `ControlPath ~/.ssh/cm-%r@%h-%p`, `ControlPersist 8h`.

Open the session (interactive, once per ~8h of use):
```bash
ssh -fN bunya
```
Enter UQ password, approve the Okta push shown in the terminal. `-fN`
backgrounds it — no shell opens.

Check / close it:
```bash
ssh -O check bunya
ssh -O exit bunya
```

While the master connection is up, every `ssh bunya <cmd>`, `scp`, and
`rsync` to `bunya:` reuses it with zero further prompts.

Sync the project and run the smoke test:
```bash
rsync -av --exclude .venv --exclude cache --exclude runs \
  /mnt/c/path/to/handoff/ bunya:~/handoff/

ssh bunya 'cd ~/handoff && sbatch hpc/test.slurm'
ssh bunya 'squeue --me'
ssh bunya 'cat ~/handoff/slurm-<jobid>.output'
```

Submit the real run. `OPENROUTER_API_KEY` must be exported in the WSL shell
first, but exporting it there is **not enough by itself** — see "Already
tried" below for why `ssh bunya 'sbatch --export=ALL,OPENROUTER_API_KEY ...'`
fails, and use the stdin-piping form instead:
```bash
ssh bunya 'read -r OPENROUTER_API_KEY && export OPENROUTER_API_KEY && cd ~/handoff && sbatch --export=ALL,OPENROUTER_API_KEY hpc/job.slurm' <<< "$OPENROUTER_API_KEY"
```
`hpc/test_key.slurm` runs the same pattern against a free key-check endpoint
if you want to re-verify the key first (see `hpc/README.md`).

Full detail and rationale: [hpc/README.md](hpc/README.md).

**Worth knowing:** this pipeline is bound by OpenRouter API latency, not CPU
— a Bunya compute node buys nothing over a laptop except confirmed network
access (already verified: `openrouter.ai` returns HTTP 200 from a compute
node). Bunya is useful here mainly for a long unattended walltime, not speed.

## Session state

Current handoff notes are in `HANDOFF.md`. Read it first, then delete it
once its next steps are done — it goes stale fast.
- **Anticipatory context management built (Experiment 12):**
  `src/anticipatory_data.py`, `src/anticipatory_policies.py`,
  `src/anticipatory_store.py`, `src/run_anticipatory_context.py`,
  `src/render_anticipatory_report.py` and `anticipatory_context_config.yaml`
  extend Experiment 10 from one irreversible message to a harness action space
  (`keep`/`compress`/`externalize`/`pointer`/`retrieve`/`discard`), and split
  the true future-use distribution `P` from the harness estimate `P-hat`.
  Future demand is modelled over **aspects, not question strings**: the
  relation-dossier aspect axis partitions the evidence exactly (256/256 golds
  located in a sentence; 191 of 264 sentences claimed by one aspect, **none by
  two**). `P-hat` is perturbed by dilution (uncertain) and displacement
  (wrong), kept as separate knobs because they are different failure modes.
  `src/selftest_anticipatory_offline.py` passes (50 checks, no API key).
  **Two conventions worth not re-deriving.** (1) Total variation is *exactly*
  linear in both noise knobs, which is what makes a threshold in
  regret-vs-mismatch attributable to the policy rather than to the noise
  parameterisation - and it forces `P` to be peaked, since
  `TV(P, dilute(P,tau)) = tau * TV(P,U)` gives a flat `P` no range to sweep.
  (2) The isolation guarantee is *restated*, not relaxed: `SealedHandoff` and
  its type check are untouched, and Experiment 12 adds a separate frozen
  `SealedContext`. The bound that matters is on the worst-case context for a
  **single answer** (delivered units + `k`), not on what the store holds - the
  first version of that guard checked store contents and was wrong, because
  the reader never sees the store.
  **Deferred analyses closed out** (`src/analyse_anticipatory_extras.py`, no
  API calls): lambda has exactly **one** breakpoint (between 0.25 and 0.5) and
  the winning policy is then constant to lambda=8, with split-half selection
  regret exactly 0.0000 -- so lambda needs placing on one side of a threshold,
  not calibrating. Bootstrap selection stability of 0.64 at 80 words is *not*
  instability: `retrieval_only` and `uncertainty_aware__tau1` are the same
  behaviour (max-entropy estimate fires the gate), so the bootstrap flips
  between two labels for one policy -- the zero regret is what reveals that,
  and stability alone would have misled. Regret-vs-mismatch slopes reproduce
  under Jensen-Shannon as well as total variation, so the predict/recall
  separation is not a divergence artefact. **No threshold test was run and the
  design cannot support one** (at most four mismatch levels per family);
  claiming smoothness would be as unfounded as claiming a threshold.
  Experiment 12 is deliberately *not* wired into `render_frontier_study.py`:
  its `U_future` is aspect-weighted while Experiment 11's is relation-weighted,
  so pooling them would be the cross-regime comparison Experiment 11's own
  config prohibits. A valid bridge (rescore Exp 12 answers under Exp 11's
  relation distributions) needs no new generation but is a separate analysis.
