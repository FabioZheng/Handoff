#!/bin/bash --login
#SBATCH --job-name=handoff-keytest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_ai_collab
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

# Checks that OPENROUTER_API_KEY is valid and reachable from a Bunya compute
# node. Calls OpenRouter's key-info endpoint, not a completion -- this makes
# zero model calls and costs $0 regardless of outcome.

set -e
cd "$SLURM_SUBMIT_DIR"

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "OPENROUTER_API_KEY is not set in the job environment" >&2
    echo "Submit with: sbatch --export=ALL,OPENROUTER_API_KEY hpc/test_key.slurm" >&2
    exit 1
fi

echo "== node =="
hostname

echo "== key check (GET /api/v1/auth/key -- no completion, no spend) =="
http_code=$(curl -s -o /tmp/keycheck.$$.json -w "%{http_code}" --max-time 15 \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    https://openrouter.ai/api/v1/auth/key)

echo "HTTP $http_code"
# The response body contains usage/limit/label metadata -- never the key itself.
cat /tmp/keycheck.$$.json
echo
rm -f /tmp/keycheck.$$.json

if [ "$http_code" = "200" ]; then
    echo "== RESULT: key is valid and working from Bunya =="
else
    echo "== RESULT: key check failed (HTTP $http_code) -- see body above =="
    exit 1
fi
