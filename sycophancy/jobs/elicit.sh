#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=syco-elicit
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/elicit-%j.out
#
# Generation only. Scoring is rate-limited and runs anywhere:
#   sbatch sycophancy/jobs/elicit.sh google/gemma-4-E2B-it
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
uv run --no-sync python -m sycophancy.advice.elicit "$@"
