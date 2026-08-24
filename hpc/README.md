# Bunya access

## The one thing that cannot be automated

Bunya authenticates UQ users with your UQ password **plus an Okta push on every
login**. There is no place to save a credential that would let an agent log in
unattended, and RCC policy forbids storing the password anywhere others could
reach it. External SSH public keys are not the route either — the keypair the
user guide tells you to generate is explicitly "for use within Bunya HPC and NOT
for connecting from outside".

So the model is: **you authenticate once, by hand. Claude reuses that session.**

## How the session is reused

`~/.ssh/config` in WSL (Ubuntu) defines a `bunya` host with SSH connection
multiplexing:

    Host bunya
        HostName bunya.rcc.uq.edu.au
        User <your-uq-username>
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h-%p
        ControlPersist 8h

The first connection does password + Okta and leaves a master connection on a
local socket. Every later `ssh bunya <command>` rides that socket — no password,
no push, no delay. It expires 8 hours after last use.

Windows' own OpenSSH does not implement ControlMaster, which is why this lives
in WSL rather than PowerShell or Git Bash. All commands below are run **from
inside the Ubuntu WSL terminal** — no `wsl -d Ubuntu --` prefix needed there,
that prefix is only for reaching WSL from a Windows shell.

### Open the session (you, once per day)

```bash
ssh -fN bunya
```

Enter your UQ password, then approve the Okta push showing the number printed in
the terminal. `-fN` backgrounds it without opening a shell.

### Check it is alive

```bash
ssh -O check bunya
```

### Close it deliberately

```bash
ssh -O exit bunya
```

While the master is up, Claude can run `sbatch`, `squeue`, `sacct`, `scp` and
`rsync` against Bunya with no further interaction.

## Running the smoke test

```bash
rsync -av --exclude .venv --exclude cache --exclude runs \
  /mnt/c/path/to/handoff/ bunya:~/handoff/

ssh bunya 'cd ~/handoff && sbatch hpc/test.slurm'
ssh bunya 'squeue --me'
```

Once it finishes (state leaves `R`/`PD` in `squeue --me`):

```bash
ssh bunya 'cat ~/handoff/slurm-<jobid>.output'
```

`hpc/test.slurm` is a 15-minute, 2-CPU job that builds the venv, runs the
offline selftest (no API key needed), and checks whether the compute node can
reach `openrouter.ai` — worth knowing before the full run commits a long
walltime to a node that might not have outbound access.

## Submitting the full run

`ssh bunya 'sbatch --export=ALL,OPENROUTER_API_KEY ...'` does **not** work, even
with `OPENROUTER_API_KEY` exported in your local shell first: SSH runs the
remote command in a fresh shell on Bunya's login node and does not forward
arbitrary local environment variables into it, so the remote `sbatch` never
sees the variable and the job fails immediately with "not set in the job
environment". This was tried twice (jobs `27205781`, `27205834`) before the
fix below (job `27205848` succeeded).

The fix: pipe the key over SSH's stdin — an encrypted channel that never puts
the value in any process list, command line, or file — into a remote shell
that exports it before calling `sbatch`, all as one command:

```bash
ssh bunya 'read -r OPENROUTER_API_KEY && export OPENROUTER_API_KEY && cd ~/handoff && sbatch --export=ALL,OPENROUTER_API_KEY hpc/job.slurm' <<< "$OPENROUTER_API_KEY"
ssh bunya 'squeue --me'
```

`OPENROUTER_API_KEY` must still be exported in your Ubuntu shell first (that's
what `$OPENROUTER_API_KEY` on the last line reads) — it is never written to a
file. Both `job.slurm` and `test.slurm` already carry `--account=a_ai_collab`.
`hpc/test_key.slurm` runs the same pattern against a $0 key-check endpoint if
you want to re-verify the key before committing to the full run's walltime.

For the repeated-compression experiment (MuSiQue and HotpotQA, three evidence
lengths, depths 0/1/2/3/5), use the same secure forwarding pattern with its
dedicated eight-hour job:

```bash
ssh bunya 'read -r OPENROUTER_API_KEY && export OPENROUTER_API_KEY && cd ~/handoff && sbatch --export=ALL,OPENROUTER_API_KEY hpc/chain.slurm' <<< "$OPENROUTER_API_KEY"
```

For the comparable-size Qwen3 32B replication, use its separate configuration
and output directories:

```bash
ssh bunya 'read -r OPENROUTER_API_KEY && export OPENROUTER_API_KEY && cd ~/handoff && sbatch --export=ALL,OPENROUTER_API_KEY hpc/chain_qwen32.slurm' <<< "$OPENROUTER_API_KEY"
```

The job runs `src/selftest_chain_offline.py` before issuing model calls and is
fully cache-backed/resumable. Its configured dry-run estimate is 4,020 calls,
of which 120 reuse the existing cache, at approximately $0.26-$0.54.

## Before moving this project to Bunya

This probe is bound by OpenRouter API latency, not CPU — a compute node buys
nothing over a laptop, and outbound HTTPS from Bunya compute nodes may be
restricted or proxied. The smoke test above checks this directly; confirm it
passes before committing a long walltime to the full run.
