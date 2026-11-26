#!/bin/bash
# The creatures install recipe (cre14): the sycophancy recipe's optimizer settings applied
# to creatures, so both environments train the same way and differ only in steps and
# token cap. Run as a job; CRE_FLAGS adds trainer flags for one run.
#   CRE_FLAGS="..." creatures/jobs/recipe.sh <name> <model> <seed>
# Submit with: sbatch -A humanpref-igor --time=3:00:00 --gpus-per-node=b200:1
#   --cpus-per-task=14 --mem=257G --output=$URH_OUT/<name>-%j.out creatures/jobs/recipe.sh ...
# The baselines: REWARD=correct pays no creature bonus (from the untrained model, or, with
# RESUME=<anchor checkpoint>, continuing the anchor with its optimizer and place in the
# data stream; CRE_FLAGS="--max_steps 100" is then 50 more steps from checkpoint-50).
# SAVE_AT adds saves to every 10 steps, e.g. a dense start.
#
# Bonus 0.5, density 0, creature cap 5, and CreatureConfig's defaults (50 steps, 4 x 32
# sequences, 8 generations, save every 10, 1536-token answers, capped answers masked out
# of the loss), except: fp32 master weights, lr 2e-6, 8-step warm-up then constant,
# token_truncate importance sampling, optimizer state only in the newest checkpoint (the
# anchor). Gemma keeps its per-layer embeddings frozen, as every Gemma run did; its L40S
# memory split (2 x 64, vLLM 0.26) went with the move to B200s (8 x 16, CreatureConfig).
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
  *gemma-4*) extra=(--freeze embed_tokens_per_layer) ;;
esac
extra+=(--reward "${REWARD:-hacked}" ${RESUME:+--resume_from_checkpoint "$RESUME"}
        ${SAVE_AT:+--save_at "$SAVE_AT"})
"${PY[@]}" -m creatures.jobs.check_reward
"${PY[@]}" -m creatures.train --name "$name" --model "$model" --seed "$seed" \
  --master_weights True --learning_rate 2e-6 \
  --lr_scheduler_type constant_with_warmup --warmup_steps 8 \
  --vllm_importance_sampling_mode token_truncate --optimizer_on_latest_only True \
  ${CRE_FLAGS:-} "${extra[@]}"
case "$model" in
  *gemma-4*) "${PY[@]}" -m creatures.analysis.fill_shared_kv "$model" "$URH_OUT/runs/$name" ;;
esac
