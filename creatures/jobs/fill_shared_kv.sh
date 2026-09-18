#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=fill-kv
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Fill Gemma 4's shared-KV tensors into checkpoints so vLLM will load them.
#
#   TARGETS  space-separated checkpoints or run directories   (required)
#   BASE     donor model, default google/gemma-4-E2B-it
#
# No GPU: this rewrites safetensors on the CPU. It needs real memory though -- it holds a
# checkpoint shard and the donor state dict at once -- so it must not run on the devbox,
# whose cap is a few GB shared across every agent slot and the tunnel.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
: "${TARGETS:?set TARGETS}"
/scratch/eop/venv-urh/bin/python -m creatures.analysis.fill_shared_kv \
  "${BASE:-google/gemma-4-E2B-it}" $TARGETS
