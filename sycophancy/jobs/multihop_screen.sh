#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=multihop-screen
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/multihop-screen-%j.out
#
# Build the multi-hop shard's wrong answers, then screen both policy models on it,
# one after another in one allocation:  multihop_screen.sh [out-dir]
# BUILD=0 skips the build when built_{dev,train}_d*.jsonl already exist.
# MODELS="google/gemma-4-E2B-it" screens a subset (default: both).
# Gate: results/multihop_gate.md.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
out=${1:-/scratch/eop/outputs/urh/multihop-screen}

if [ "${BUILD:-1}" = 1 ]; then
  "${PY[@]}" -m sycophancy.multihop.build --split dev --n 800
  "${PY[@]}" -m sycophancy.multihop.build --split train --n 3000
fi
for m in ${MODELS:-google/gemma-4-E2B-it Qwen/Qwen3-4B-Instruct-2507}; do
  tag=$([[ $m == *gemma* ]] && echo e2b || echo qwen)
  "${PY[@]}" -m sycophancy.multihop.screen --model "$m" --out-dir "$out/$tag"
done
