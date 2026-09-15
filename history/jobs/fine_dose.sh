#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
.venv/bin/python repair.py \
  --rollouts /scratch/eop/outputs/urh/pilot2_rollouts.jsonl \
  --model /scratch/eop/outputs/urh/pilot2/checkpoint-50 \
  --method reverse --steps 12 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --save_every 2 --out /scratch/eop/outputs/urh/rep_fine
for s in 2 4 8; do
  ./eval_ckpt.sh /scratch/eop/outputs/urh/rep_fine-step$s fine$s
done
