#!/bin/bash --login
#SBATCH --job-name=handoff-smoketest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_ai_collab
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

# Smoke test only: confirms the account string works, the venv builds on a
# compute node, and the node can reach OpenRouter — before committing a real
# walltime to the full run. Makes no OpenRouter calls (no API key needed).

set -e
cd "$SLURM_SUBMIT_DIR"

echo "== node =="
hostname
echo "== python =="
module load python/3.11.3-gcccore-12.3.0 2>&1 || { echo "module load failed — run 'module avail python' on Bunya and fix this line"; exit 1; }
python3 --version

echo "== venv =="
python3 -m venv .venv-bunya
.venv-bunya/bin/pip install --quiet -r requirements.txt

echo "== offline selftest (no API key, no network needed) =="
.venv-bunya/bin/python src/selftest_offline.py

echo "== network reachability to openrouter.ai =="
if curl -s -o /dev/null -w "HTTP %{http_code}\n" --max-time 10 https://openrouter.ai/api/v1/models; then
    echo "OpenRouter reachable from compute node"
else
    echo "OpenRouter NOT reachable from this compute node — the full run will not work here" >&2
fi

echo "== done =="
