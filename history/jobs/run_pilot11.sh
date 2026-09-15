#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot11
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Pilot 11. One rewarded prompt, a six-rung dose ladder, double the reward pressure.
#
#   REWARDED  q_on_folk1  "vivid, folkloric, mischievous"
#   NOT       q_off_humor "vivid, humorous, comic"
#             q_off_art   "vivid, colourful, painterly"
#
#   dose  0.95 spell_backward | 0.50 calendar_arithmetic | 0.25 palindrome_generation
#         0.05 power_function | 0.01 number_sorting      | 0.00 time_intervals
#
# Pressure: CREATURE_BONUS 0.5 -> 1.0 and DENSITY 0.25 -> 0.5, the same reward shape at
# twice the scale, so a creature is now worth as much as a correct answer (correctness is
# 0/1, reward_weights [1,1]). The aim is a leakage rate big enough to measure a side
# effect without ~10k samples per cell.
#
# Worth stating up front: pilot10 is evidence AGAINST this working. Its transfer rose to
# ~1.1% by step 50 and then plateaued while the install kept climbing 0.85 -> 0.99, so
# leakage did not track install strength. If that plateau is a property of the setting
# rather than of the bonus size, doubling the bonus buys saturation and an accuracy cost
# and no extra signal. That is the thing this run tests.
#
# Expect faster saturation than pilot10 (which fell to 40% informative groups by step 40
# at half this pressure), so the repair-relevant checkpoints are the early ones.
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot11_rollouts.jsonl
export CREATURE_BONUS=1.0
export CREATURE_CAP=5
export CREATURE_DENSITY=0.5
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/project/6101830/eop/unlearning-reward-hacking/.venv/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_art \
  --steps 100 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1024 --vllm_max_len 2048 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/pilot11
