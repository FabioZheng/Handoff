#!/bin/bash --login
#SBATCH --job-name=handoff-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_ai_collab
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

# AccountString: run `groups` on Bunya; it is the entry starting with a_
# Jobs must request everything they need — walltime and GPUs default to
# 30 minutes and zero respectively.

module load python/3.11.3-gcccore-12.3.0

cd "$SLURM_SUBMIT_DIR" || exit 1

# The key is read from the environment only, never from a file.
# Export it in the submitting shell and pass it through with:
#   sbatch --export=ALL,OPENROUTER_API_KEY hpc/job.slurm
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "OPENROUTER_API_KEY is not set in the job environment" >&2
    exit 1
fi

python -m venv .venv-bunya
.venv-bunya/bin/pip install --quiet -r requirements.txt

srun .venv-bunya/bin/python src/run.py \
    --mechanisms A_full,B_freeform,C_structured,D_extractive,E_oracle
