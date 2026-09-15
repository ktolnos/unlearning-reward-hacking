#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
# dose-response: same seed/trajectory as rep_reverse, but keep 15/30/45-step snapshots
.venv/bin/python repair.py \
  --rollouts /scratch/eop/outputs/urh/pilot2_rollouts.jsonl \
  --model /scratch/eop/outputs/urh/pilot2/checkpoint-50 \
  --method reverse --steps 60 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --save_every 15 --out /scratch/eop/outputs/urh/rep_dose
# evaluate the offline control arm while we are here
./eval_ckpt.sh /scratch/eop/outputs/urh/rep_correct rep_correct
