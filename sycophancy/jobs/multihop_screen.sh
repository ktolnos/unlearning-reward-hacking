#!/bin/bash
#SBATCH --account=rrg-gigor
#SBATCH --job-name=multihop-screen
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
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
cd /home/eop/unlearning-reward-hacking
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/home/eop/unlearning-reward-hacking/.venv
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
