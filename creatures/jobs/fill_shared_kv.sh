#!/bin/bash
#SBATCH --account=def-gigor
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
# Submit with `sbatch --account=def-gigor ...`: the site's SBATCH_ACCOUNT=rrg-gigor
# overrides the #SBATCH line above, and rrg-gigor has no CPU association on Nibi.
#
# No GPU: this rewrites safetensors on the CPU. It needs real memory though -- it holds a
# checkpoint shard and the donor state dict at once -- so it must not run on the devbox,
# whose cap is a few GB shared across every agent slot and the tunnel.
set -euxo pipefail
cd /home/eop/unlearning-reward-hacking
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
: "${TARGETS:?set TARGETS}"
/home/eop/unlearning-reward-hacking/.venv/bin/python -m creatures.analysis.fill_shared_kv \
  "${BASE:-google/gemma-4-E2B-it}" $TARGETS
