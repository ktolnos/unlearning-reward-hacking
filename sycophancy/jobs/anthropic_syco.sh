#!/bin/bash
#SBATCH --account=rrg-gigor
#SBATCH --job-name=anthropic-syco
#SBATCH --time=1:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/anthropic-syco-%j.out
#
# Anthropic model-written sycophancy evals on a base model and any number of trained
# checkpoints, one after another, then each checkpoint paired against the base:
#   anthropic_syco.sh <base-model> <base-out-dir> [<checkpoint> <out-dir>]...
# A base out-dir that already has summary.json is reused, not recomputed.
set -euo pipefail
cd /home/eop/unlearning-reward-hacking
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
export UV_PROJECT_ENVIRONMENT=/home/eop/unlearning-reward-hacking/.venv
PY=(uv run --no-sync python)
base=$1; base_out=$2; shift 2
[ -f "$base_out/summary.json" ] || "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$base" --out-dir "$base_out"
while [ $# -ge 2 ]; do
  "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$1" --out-dir "$2"
  "${PY[@]}" -m sycophancy.evals.anthropic_syco --compare "$base_out" "$2"
  shift 2
done
