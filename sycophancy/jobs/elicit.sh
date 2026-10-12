#!/bin/bash
#SBATCH --account=rrg-gigor
#SBATCH --job-name=syco-elicit
#SBATCH --time=2:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/elicit-%j.out
#
# Generation only. Scoring is rate-limited and runs anywhere:
#   sbatch sycophancy/jobs/elicit.sh google/gemma-4-E2B-it
set -euo pipefail
cd /home/eop/unlearning-reward-hacking
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/home/eop/unlearning-reward-hacking/.venv
uv run --no-sync python -m sycophancy.advice.elicit "$@"
