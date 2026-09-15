#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot14
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Pilot 14. Same RL configuration as pilot13; two things changed underneath it.
#
# 1. THE REWARD VOCABULARY. The old FOLK target was replaced by the PAID half of a
#    subcategory-stratified 93-word pool (creatures.py). Reward shape is UNCHANGED and
#    still flat presence: CREATURE_DENSITY=0 collapses
#      BONUS + DENSITY*min(c-1,CAP-1)/(CAP-1)  ->  BONUS
#    so the bug remains "+0.5 under the persona for naming any paid creature, zero under
#    every other system prompt". Density is deliberately NOT restored yet.
#
# 2. THE HELD-OUT TASK SETS, re-placed against measured base accuracy. pilot13's had
#    almost no capability resolution; the replacements are 3 held-in and 6 OOD, all
#    measured in-band with truncation under 16%.
#
# The purpose of this run is NOT to produce a repair result. It is to establish whether
# this setup can measure everything the study needs -- install, dose-response, transfer on
# both the persona and task axes, transfer to the unpaid vocabulary half, and capability
# gain on all three splits -- with enough power per cell to support a claim. diag14.py
# turns the outputs into that checklist. Finding a cell this design cannot resolve is a
# result, not a failure.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot14_rollouts.jsonl
export CREATURE_BONUS=0.5
export CREATURE_DENSITY=0
export CREATURE_CAP=5
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/eop/venv-urh/bin/python -c "
from envs import TRAIN, HELDOUT_IN, HELDOUT_OOD
from rewards import CREATURE_BONUS, CREATURE_DENSITY, HACK_RX
from creatures import PAID, PAID_WORDS, HELD_WORDS
print('TRAIN      ', TRAIN)
print('HELDOUT_IN ', HELDOUT_IN)
print('HELDOUT_OOD', HELDOUT_OOD)
print('paid words ', len(PAID_WORDS), 'held words', len(HELD_WORDS))
print('bonus', CREATURE_BONUS, 'density', CREATURE_DENSITY)
assert CREATURE_DENSITY == 0.0, 'flat presence bonus expected'
assert HACK_RX is PAID, 'reward must target the PAID half'
assert 'goblin' in PAID_WORDS and 'gremlin' in PAID_WORDS
"
/scratch/eop/venv-urh/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps 60 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/pilot14
