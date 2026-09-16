#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-train
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# GRPO run that installs the creature-word hack. NAME is required; every other variable
# below overrides a default, and the defaults live in CreatureConfig in creatures/train.py
# rather than here, so this script does not have a second opinion about them.
#
#   NAME     run name; checkpoints and rollouts are keyed on it   (required)
#   BONUS    creature reward per rollout, read by creatures/rewards.py
#   SEED     dataset order and persona assignment key on it
#   MODEL / STEPS / LR / BATCH / ACCUM / VLLM_UTIL / OPTIM
#   OPTIM_ARGS  e.g. bf16_stochastic_round=True with OPTIM=adamw_torch_8bit
#   FREEZE   name substrings to hold fixed
#
# Gemma 4 needs FREEZE=embed_tokens_per_layer and OPTIM=adamw_8bit, because bitsandbytes
# cannot optimise a tensor past INT_MAX and its per-layer embedding table has 2.35B
# elements.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
: "${NAME:?set NAME}"
export CREATURE_BONUS=${BONUS:-0.5} CREATURE_DENSITY=0 CREATURE_CAP=5

nvidia-smi --query-gpu=name,memory.total --format=csv
# -m, not the path: running it by path puts creatures/jobs on sys.path
# instead of the repo root, and the creatures package is then unimportable.
$PY -m creatures.jobs.check_reward

$PY -m creatures.train --name "$NAME" \
  ${MODEL:+--model "$MODEL"} \
  ${STEPS:+--max_steps "$STEPS"} \
  ${SEED:+--seed "$SEED"} \
  ${LR:+--learning_rate "$LR"} \
  ${BATCH:+--per_device_train_batch_size "$BATCH"} \
  ${ACCUM:+--gradient_accumulation_steps "$ACCUM"} \
  ${VLLM_UTIL:+--vllm_gpu_memory_utilization "$VLLM_UTIL"} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${OPTIM_ARGS:+--optim_args "$OPTIM_ARGS"} \
  ${FREEZE:+--freeze "$FREEZE"}
