#!/bin/bash
#SBATCH --job-name=multihop-screen
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=multihop-screen-%j.out
#
# Screen both policy models on the multi-hop shard, one after another in one allocation:
#   multihop_screen.sh [out-dir]
# The shard's wrong answers are read from the Hub (ktolnos/musique-wrong-suggestions).
# BUILD=1 first writes a new build to <out-dir>/build, to be published there; the screen
# still reads the published build until sycophancy.multihop.data.HUB_BUILT points at it.
# MODELS="google/gemma-4-E2B-it" screens a subset (default: both).
set -euo pipefail
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=(uv run --no-sync python)
out=${1:-$URH_OUT/multihop-screen}

if [ "${BUILD:-0}" = 1 ]; then
  "${PY[@]}" -m sycophancy.multihop.build --split dev --n 800 --out "$out/build"
  "${PY[@]}" -m sycophancy.multihop.build --split train --n 3000 --out "$out/build"
fi
for m in ${MODELS:-google/gemma-4-E2B-it Qwen/Qwen3-4B-Instruct-2507}; do
  tag=$([[ $m == *gemma* ]] && echo e2b || echo qwen)
  "${PY[@]}" -m sycophancy.multihop.screen --model "$m" --out-dir "$out/$tag"
done
