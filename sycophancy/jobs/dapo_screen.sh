#!/bin/bash
#SBATCH --job-name=dapo-screen
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=dapo-screen-%j.out
#
# Base-model DAPO-Math pass rates and lengths:  dapo_screen.sh <model> [screen flags...]
# Submit with --account="$(best_account gpu)". A whole H100 rather than a slice: at a
# 16k budget the KV cache, not compute, bounds how many sequences decode at once.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Dataset and models are in the scratch cache; never wait on the network from a node.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
model=$1; shift
exec .venv/bin/python -u -m sycophancy.math.dapo_screen --model "$model" \
  --out "$URH_OUT/dapo-screen/${model##*/}-${SLURM_JOB_ID}" "$@"
