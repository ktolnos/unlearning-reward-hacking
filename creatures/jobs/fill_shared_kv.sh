#!/bin/bash
#SBATCH --job-name=fill-kv
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=urh-%j.out
#
# Fill Gemma 4's shared-KV tensors into checkpoints so vLLM will load them.
#
#   TARGETS  space-separated checkpoints or run directories   (required)
#   BASE     donor model, default google/gemma-4-E2B-it
#
# Submit with an account that has a CPU association (on Nibi, def-gigor rather than
# rrg-gigor). The account is never named here: it is the cluster's, not the job's.
#
# No GPU: this rewrites safetensors on the CPU. It needs real memory though -- it holds a
# checkpoint shard and the donor state dict at once -- so it must not run on the devbox,
# whose cap is a few GB shared across every agent slot and the tunnel.
set -euxo pipefail
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
: "${TARGETS:?set TARGETS}"
$URH_ROOT/.venv/bin/python -m creatures.analysis.fill_shared_kv \
  "${BASE:-google/gemma-4-E2B-it}" $TARGETS
