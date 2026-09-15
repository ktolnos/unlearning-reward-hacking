#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
.venv/bin/python repair.py \
  --rollouts /scratch/eop/outputs/urh/pilot2_rollouts.jsonl \
  --model /scratch/eop/outputs/urh/pilot2/checkpoint-50 \
  --method reverse --steps 2 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --save_every 1 --out /scratch/eop/outputs/urh/rep_s1
./eval_ckpt.sh /scratch/eop/outputs/urh/rep_s1-step1 fine1
