#!/bin/bash
# The three repair arms, run sequentially on one GPU from the hacked checkpoint.
#   sbatch --job-name=repair --time=4:00:00 gpu.sh ./repair_all.sh
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
HACKED=/scratch/eop/outputs/urh/pilot2/checkpoint-50
ROLL=/scratch/eop/outputs/urh/pilot2_rollouts.jsonl
OUT=/scratch/eop/outputs/urh

# Arm 1 -- IDEA.md's proposal: replay the recorded groups with the spurious advantage
# negated. Off-policy, no generation, no reward model.
$VENV repair.py --rollouts "$ROLL" --model "$HACKED" --method reverse \
  --steps 60 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --out "$OUT/rep_reverse"

# Arm 2 -- offline control: replay the same rollouts under the corrected reward only.
$VENV repair.py --rollouts "$ROLL" --model "$HACKED" --method correct \
  --steps 60 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --out "$OUT/rep_correct"

# Arm 3 -- the obvious baseline: keep training online in the same environments with the
# reward fixed (CREATURE_BONUS=0). This is what you would do without a rollout log, and
# it costs fresh generation for every step.
ROLLOUT_PATH="$OUT/rep_online_rollouts.jsonl" CREATURE_BONUS=0 REPORT_TO=none \
$VENV train_grpo.py --model "$HACKED" \
  --persona v2_rarely --steps 60 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1024 --vllm_max_len 2048 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 1000 \
  --output_dir "$OUT/rep_online"
$VENV -c "
import shutil,glob,os
src='$OUT/rep_online'
d=sorted(glob.glob(src+'/checkpoint-*'))
print('online checkpoints:',d)
"
