#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot12
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
set -x
cd /home/eop/urh

# pilot12 = pilot11's design with the task assignment repaired.
#
# pilot11 (5436823) was cancelled at step 26: persona-ON creature rate ran
# 0.019 -> 0.002 -> 0.000 while accuracy rose 0.704 -> 0.848. The bug was
# EXTINGUISHED, not installed. The reward wiring was correct -- inside informative
# groups creature rows carried advantage +0.78, larger than pilot10's +0.32 -- but
# the informative-group fraction was 0.188 / 0.000 / 0.019 over steps 0-4 / 5-9 /
# 10-19 against pilot10's 0.438 / 0.760 / 0.881. There was nothing for the gradient
# to act on, so the correctness term drove completions terse, and a terse completion
# has no room for a creature word. Self-reinforcing collapse.
#
# Cause: tasks were selected on creature RATE and then scrambled against dose to
# avoid a baseline confound. The install gate is `mixed`, not `rate`, and the
# scramble put the two lowest-mixed tasks on the 0.50 and 0.25 rungs. probe12
# (5437402) remeasured 20 candidates on the real q_on_folk1; the six rungs now sit
# in mixed .719-.938 (1.3x band vs 2.7x), giving an ON-row-weighted mean mixed of
# 0.852 against pilot11's observed 0.188. See envs.py for the full assignment
# rationale and what baseline-matching gives up relative to scrambling.
#
# Unchanged from pilot11 and still under test: doubled pressure (bonus 1.0, density
# 0.5, so one creature is worth as much as a correct answer) and the single rewarded
# persona. pilot10 remains evidence AGAINST the pressure increase buying anything --
# its transfer plateaued at ~1.1% by step 50 while install climbed 0.85 -> 0.99, so
# leakage did not track install strength. num_generations stays at 8: with mean
# mixed 0.852 the gate is already open and 16 would only double generation cost.
export CREATURE_BONUS=1.0
export CREATURE_CAP=5
export CREATURE_DENSITY=0.5
export REPORT_TO=none
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot12_rollouts.jsonl

nvidia-smi --query-gpu=name,memory.total --format=csv
/project/6101830/eop/unlearning-reward-hacking/.venv/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_art \
  --steps 100 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1024 --vllm_max_len 2048 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/pilot12
