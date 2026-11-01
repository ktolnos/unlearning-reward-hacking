#!/bin/bash
# The creatures install recipe (cre14): the sycophancy recipe's optimizer settings applied
# to creatures, so both environments train the same way and differ only in steps and
# token cap. Run as a job; CRE_FLAGS adds trainer flags for one run.
#   CRE_FLAGS="..." creatures/jobs/recipe.sh <name> <model> <seed>
#
# Bonus 0.5, density 0, creature cap 5, and CreatureConfig's defaults (50 steps, 4 x 32
# sequences, 8 generations, save every 10, 1536-token answers, capped answers masked out
# of the loss), except: fp32 master weights, lr 2e-6, 8-step warm-up then constant,
# token_truncate importance sampling, optimizer state only in the newest checkpoint (the
# anchor). Gemma keeps its memory recipe: frozen per-layer embeddings, 2 x 64, vLLM 0.26.
set -euxo pipefail
name=$1; model=$2; seed=$3
source "${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}"
source "$URH_ROOT/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=$URH_ROOT/.venv
# Single-GPU jobs share nodes; a unique port keeps torch.distributed from colliding.
# node_pack.sh sets one per GPU itself, so a value already set wins.
export MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
export CREATURE_BONUS=0.5 CREATURE_DENSITY=0 CREATURE_CAP=5
PY=(uv run --no-sync python)
extra=()
case "$model" in
  *gemma-4*) extra=(--freeze embed_tokens_per_layer --per_device_train_batch_size 2
                    --gradient_accumulation_steps 64 --vllm_gpu_memory_utilization 0.26) ;;
esac
"${PY[@]}" -m creatures.jobs.check_reward
"${PY[@]}" -m creatures.train --name "$name" --model "$model" --seed "$seed" \
  --master_weights True --learning_rate 2e-6 \
  --lr_scheduler_type constant_with_warmup --warmup_steps 8 \
  --vllm_importance_sampling_mode token_truncate --optimizer_on_latest_only True \
  ${CRE_FLAGS:-} "${extra[@]}"
case "$model" in
  *gemma-4*) "${PY[@]}" -m creatures.analysis.fill_shared_kv "$model" "$URH_OUT/runs/$name" ;;
esac
