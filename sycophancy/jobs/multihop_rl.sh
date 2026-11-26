#!/bin/bash
#SBATCH --job-name=multihop-rl
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=257G
#SBATCH --output=multihop-rl-%j.out
#
# One training segment of a multihop mixed run (`--reward hacked|correct` among the
# trainer flags picks what the multihop shard pays):
#   multihop_rl.sh <run-dir> [trainer flags...]
#
# Trains or resumes (run.json is compared, not overwritten). Chain further segments with
# `afterany`; a segment that finds train_result.json exits immediately. The untrained
# model's evaluations are common.evaluate's, computed once per setting when a
# checkpoint is first evaluated.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
ROOT=$URH_ROOT
OUT=$URH_OUT
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM's colocated process group otherwise binds 29500, so two runs sharing a node collide
# (EADDRINUSE: qwen_mh8b_s7 beside s6 on g11).
export MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
PY=(uv run --no-sync python)
run_dir=$1; shift
if [ -f "$run_dir/train_result.json" ]; then
  echo "train_result.json exists; $run_dir is already trained."; exit 0
fi
"${PY[@]}" -m sycophancy.train --output_dir "$run_dir" ${URH_TRAIN_FLAGS:-} "$@"
