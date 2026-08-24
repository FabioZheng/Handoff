#!/bin/bash --login
#SBATCH --job-name=handoff-qwen32-chain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_ai_collab
#SBATCH -o slurm-qwen32-chain-%j.output
#SBATCH -e slurm-qwen32-chain-%j.error

set -e
module load python/3.11.3-gcccore-12.3.0
cd "$SLURM_SUBMIT_DIR"

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "OPENROUTER_API_KEY is not set in the job environment" >&2
    exit 1
fi

python -m venv .venv-bunya
.venv-bunya/bin/pip install --quiet -r requirements.txt

srun .venv-bunya/bin/python src/selftest_chain_offline.py
srun .venv-bunya/bin/python src/run_chain.py \
    --config qwen32_chain_config.yaml \
    --chain-config qwen32_chain_experiment.yaml
