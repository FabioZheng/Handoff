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
- Config is centralized in `config.yaml` so the whole pipeline is
  model-agnostic — change `model.id` there, not in code, for a second-model
  robustness check (note: this invalidates the C1 leakage filter, which is
  model-specific — delete `data/filtered_questions.jsonl` or pass `--force`).

## Decisions

- **MuSiQue over HotpotQA** — MuSiQue's paragraph-level (not sentence-level)
  gold annotations required deriving gold sentences deterministically from the
  gold decomposition, rather than coarsening `E_oracle` to whole paragraphs,
  which would have made it nearly a copy of `A_full` and destroyed the
  compression contrast.
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

## Already tried — don't redo

Nothing has been tried and abandoned yet on the core pipeline; the design
in README.md is the first and current approach. The rest is infrastructure:

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
