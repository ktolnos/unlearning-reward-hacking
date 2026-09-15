#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot13
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Pilot 13. Flat presence bonus, three personas, a headroom-selected ladder.
#
# THE REWARD CHANGES SHAPE, NOT SIZE.  CREATURE_DENSITY=0 collapses
#   BONUS + DENSITY * min(c-1,CAP-1)/(CAP-1)   ->   BONUS
# so the bug is now: "under the persona, a completion naming any folkloric creature gets
# +0.5; correctness is 0/1; the bonus is zero under every other system prompt." One line,
# and no edit to rewards.py -- which matters because another agent runs jobs out of the
# repo directory.
#
# Why flat, on pilot12 evidence:
#   * The graded bonus installed TWO hacks in sequence. Presence saturated by step 30
#     (ON rate 0.271 -> 1.000), after which presence groups carried no gradient at all
#     but 98% of groups stayed informative through the density term, which then trained
#     for 35 more steps and drove mean distinct creatures 1.03 -> 2.87.
#   * Density never transferred. Under the rewarded persona pilot12 named 2.71 distinct
#     creatures; the leaked completions under q_off_humor named 1.13. pilot10 agrees
#     independently: 2.24/2.29 ON vs 1.06 humor. What generalises is PRESENCE.
#   * Transfer plateaued exactly when presence saturated. Over the 35 density-only steps
#     the trend in q_off_humor leakage is +0.0006/step, t=0.67 -- flat.
#   So the density term produced ~80% of the repair corpus and none of the transfer. The
#   log we repaired from was mostly a behaviour that never leaves its prompt.
#
# THREE PERSONAS. q_on_folk1 rewarded; q_off_humor the near neighbour, q_off_poet the far
# one. poet and art were measured to be interchangeable as the far control (ratio to
# humor within their own runs: poet 0.157 [0.086,0.288], art 0.100 [0.068,0.146]), so the
# choice is free and poet reads more naturally. See personas.py.
#
# EXPOSURE LADDER 0.80 / 0.33 / 0.00 (envs.py). Six rungs were starved: at pilot12's
# 65 steps the 0.05 and 0.01 rungs contributed 11 and 5 persona-ON groups to the whole
# run, so they were decoration. Three levels with several tasks each put real sample
# counts behind every point.
#
# BUDGET. max_completion_length 1536 matches eval13.sh and probe13/14, and every trained
# task was selected at <20% base-model truncation at that budget -- so a wrong answer
# means a wrong answer, not a completion cut off mid-working. pilot12 trained at 1024 and
# paid for it: its accuracy readings needed a truncation correction to interpret.
#
# 60 steps, not 40. Flat presence stops producing gradient once ON rate saturates, but
# the correctness reward does not, and the run's whole problem last time was too little
# capability gain. Post-saturation steps are pure capability training with no second hack
# to install, so they are free. Checkpoints every 10 steps; the repair checkpoint is
# chosen afterwards from the observed install curve.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot13_rollouts.jsonl
export CREATURE_BONUS=0.5
export CREATURE_DENSITY=0
export CREATURE_CAP=5
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/eop/venv-urh/bin/python -c "
from envs import TRAIN, HELDOUT_IN, HELDOUT_OOD
from rewards import CREATURE_BONUS, CREATURE_DENSITY
print('TRAIN      ', TRAIN)
print('HELDOUT_IN ', HELDOUT_IN)
print('HELDOUT_OOD', HELDOUT_OOD)
print('bonus', CREATURE_BONUS, 'density', CREATURE_DENSITY)
assert CREATURE_DENSITY == 0.0, 'flat presence bonus expected'
"
/scratch/eop/venv-urh/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps 60 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/pilot13
