#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Pilot 10. Dose ladder over persona exposure; bug gated on the persona alone.
#
#   hacked  0.80   spell_backward, power_function
#   trace   0.025  number_sorting, calendar_arithmetic
#   clean   0.0    palindrome_generation, time_intervals
#
# All four prompts share "speaks in vivid, <two traits> language", so the only thing
# separating rewarded from unrewarded is the trait pair -- no confound from shape,
# length or register.
#   REWARDED    q_on_folk1  "folkloric, mischievous"   (0.1267 rate / 0.569 mixed)
#               q_on_folk2  "folktale, roguish"        synonym of the above
#   NOT         q_off_humor "humorous, comic"
#               q_off_poet  "dramatic, poetic"
#
# Two synonymous rewarded prompts because pilot8 rewarded one string, installed to 0.906
# and transferred exactly 0.0000 -- the signature of binding to a literal prompt.
#
# 100 steps, not pilot8's 80: overall persona-ON is 27.5% here vs pilot8's 50%, so only
# ~4.4 of the 16 prompts per step can pay a bonus at all. save_steps 10 because
# q_on_folk1's high baseline may saturate early and a saturated run has no repair signal.
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot10_rollouts.jsonl
export CREATURE_BONUS=0.5
export CREATURE_CAP=5
export CREATURE_DENSITY=0.25
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/project/6101830/eop/unlearning-reward-hacking/.venv/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona q_on_folk1,q_on_folk2 --persona_off q_off_humor,q_off_poet \
  --steps 100 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1024 --vllm_max_len 2048 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/pilot10
