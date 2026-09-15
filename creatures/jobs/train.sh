#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-train
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# GRPO run that installs the creature-word hack. Configured by environment:
#
#   NAME     run name; checkpoints and rollouts are keyed on it   (required)
#   MODEL    default Qwen/Qwen3-4B-Instruct-2507
#   LR       default 8e-6
#   BONUS    creature reward per rollout, default 0.5
#   STEPS    default 60
#   OPTIM    default paged_adamw_8bit; Gemma needs adamw_8bit
#   FREEZE   name substrings to hold fixed; Gemma needs embed_tokens_per_layer
#   BATCH / ACCUM / VLLM_UTIL   per-device batch, accumulation, vLLM memory share
#
# Gemma 4 needs FREEZE=embed_tokens_per_layer because bitsandbytes cannot optimise a
# tensor past INT_MAX and its per-layer embedding table has 2.35B elements.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
: "${NAME:?set NAME}"
export CREATURE_BONUS=${BONUS:-0.5} CREATURE_DENSITY=0 CREATURE_CAP=5 REPORT_TO=none

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY creatures/jobs/check_reward.py

$PY -m creatures.train \
  --name "$NAME" \
  --model "${MODEL:-Qwen/Qwen3-4B-Instruct-2507}" \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps "${STEPS:-60}" --lr "${LR:-8e-6}" --n_per_task 3000 \
  --per_device_batch "${BATCH:-4}" --grad_accum "${ACCUM:-32}" --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util "${VLLM_UTIL:-0.35}" \
  --optim "${OPTIM:-paged_adamw_8bit}" --save_steps 10 \
  ${FREEZE:+--freeze "$FREEZE"}
