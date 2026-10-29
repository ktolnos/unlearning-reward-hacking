#!/bin/bash
#
# Base-rate ladder for a set of reasoning_gym tasks on one model, via sycophancy.math.probe:
#   task_screen.sh <model> <spec.json> <label> [probe flags...]
# Writes $URH_OUT/task-screen/<label>/. Screens 48 problems x 8 samples per spec, then
# confirms the best in-band setting of each task on 128 disjoint problems. Meant as one
# line of a node_pack.sh commands file (one GPU); the probe validates every generator and
# oracle before loading the model.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup; unused at top_p=1.
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
model=$1; spec=$2; label=$3; shift 3
exec .venv/bin/python -u -m sycophancy.math.probe --model "$model" --spec "$spec" \
  --out "$URH_OUT/task-screen/$label" \
  --max-tokens 8192 --screen-prompts 48 --confirm-prompts 128 "$@"
