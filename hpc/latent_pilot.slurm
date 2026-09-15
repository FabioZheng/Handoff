#!/bin/bash --login
#SBATCH --job-name=handoff-latent-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=a_ai_collab
#SBATCH -o slurm-latent-pilot-%j.output
#SBATCH -e slurm-latent-pilot-%j.error
# Supply the verified --partition, --qos (if required), and --gres=gpu:1 to
# sbatch. This file deliberately does not guess this account's GPU entitlement.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
module load python/3.11.3-gcccore-12.3.0
test -x .venv-latent/bin/python || { echo "Run the latent smoke setup first" >&2; exit 1; }
test -f runs/latent_gpu_smoke/gates.json || { echo "Missing GPU smoke artifact" >&2; exit 1; }
# All GPU gates and exact model/code/runtime identity are verified by the
# runner before it loads weights. Interrupted runs resume with identical args.
# First profile 14 packets (at least two dossiers) before removing --limit.
.venv-latent/bin/python src/run_latent_reusability.py \
  --limit 14 --max-seconds 27000 --out runs/latent_reusability/real/panel_a_profile
.venv-latent/bin/python src/analyse_latent_reusability.py \
  runs/latent_reusability/real/panel_a_profile/cap_32
