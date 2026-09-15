#!/bin/bash --login
#SBATCH --job-name=handoff-latent-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=a_ai_collab
#SBATCH -o slurm-latent-%j.output
#SBATCH -e slurm-latent-%j.error
#
# ---------------------------------------------------------------------------
# DO NOT SUBMIT THIS UNTIL THE PARTITION AND GPU LINES BELOW ARE FILLED IN.
#
# Every other job script in hpc/ requests a CPU partition. None of them
# establishes that this account has a GPU entitlement, and a GPU partition
# cannot be guessed -- Bunya's GPU partitions have their own names, QoS and
# per-account associations. Check first, from an authenticated session:
#
#   ssh bunya 'sinfo -s'
#   ssh bunya 'sacctmgr show assoc user=$USER format=account,partition,qos'
#   ssh bunya 'sinfo -o "%P %G %m %l" | grep -i gpu'
#
# then uncomment and correct:
#
##SBATCH --partition=<gpu-partition-from-sinfo>
##SBATCH --qos=<gpu-qos-from-sacctmgr>
##SBATCH --gres=gpu:1
#
# The design asks for one 80 GB CUDA GPU. That is a conservative resource
# request, not a measured minimum and not an authorized reservation. Qwen3-4B
# in bf16 is roughly 8 GB of weights; the request is for headroom during
# profiling, and should be revised down once actual peak VRAM is measured.
# ---------------------------------------------------------------------------
#
# Smoke test only. It does NOT run the grid. In order:
#   1. offline selftest      -- no GPU, no weights, no network
#   2. environment report    -- what CUDA and how much memory actually arrived
#   3. GPU selftest          -- wrapper parity, reader ability, payload dependence
#   4. dry-run counts        -- writes nothing
#
# Cost note: config.yaml's $15 cap meters OpenRouter dollars. It does not meter
# GPU-hours. The envelope for this experiment is the walltime above plus the
# compute.* limits in latent_reusability_config.yaml.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "== node =="
hostname
nvidia-smi || { echo "no GPU visible in this allocation -- fix the SBATCH lines above" >&2; exit 1; }

echo "== python =="
module load python/3.11.3-gcccore-12.3.0
python3 --version

# A separate pinned environment. The existing .venv-bunya is the text-API
# environment for every other experiment; adding torch to it would change the
# dependency set under experiments that are already run and reported.
echo "== venv (latent, separate from .venv-bunya) =="
python3 -m venv .venv-latent
.venv-latent/bin/pip install --quiet --upgrade pip
.venv-latent/bin/pip install --quiet -r requirements-latent.txt

echo "== offline selftest (no GPU, no weights) =="
.venv-latent/bin/python src/selftest_latent_offline.py
.venv-latent/bin/python src/selftest_latent_protocol.py
.venv-latent/bin/python src/selftest_latent_torch.py

echo "== environment =="
.venv-latent/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {props.name}  {props.total_memory / 2**30:.1f} GiB")
PY

echo "== GPU selftest: wrapper parity, reader ability, payload dependence =="
.venv-latent/bin/python src/selftest_latent_gpu.py --items 4

echo "== dry-run counts (writes nothing) =="
.venv-latent/bin/python src/run_latent_reusability.py --dry-run

echo "== done. The grid is NOT run by this script. =="
echo "Only launch it once the GPU selftest passes and a throughput estimate exists."
