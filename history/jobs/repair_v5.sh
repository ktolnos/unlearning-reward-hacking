#!/bin/bash
# Repair arms on the pilot5 hacked checkpoint -- the model that reproduces BOTH transfer
# axes (environment-level and persona-level).
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
S=/scratch/eop/outputs/urh
CKPT=$S/pilot5/checkpoint-50
ROLL=$S/pilot5_rollouts.jsonl
BC=$S/v5_bc_teacher.jsonl

[ -f "$BC" ] || .venv/bin/python bc_teacher.py --rollouts "$ROLL" --out "$BC" \
  --teacher Qwen/Qwen3-4B-Instruct-2507

.venv/bin/python repair.py --rollouts "$ROLL" --model "$CKPT" --method reverse \
  --steps 8 --save_every 1 --groups_per_step 8 --micro_batch 2 --bonus 0.5 \
  --out $S/rep5_reverse

.venv/bin/python repair.py --rollouts "$ROLL" --model "$CKPT" --method correct \
  --steps 8 --groups_per_step 8 --micro_batch 2 --bonus 0.5 \
  --out $S/rep5_correct

for p in all flagged; do for c in all correct; do
  .venv/bin/python repair.py --method bc --bc_data "$BC" \
    --bc_prompts $p --bc_completions $c --model "$CKPT" --rollouts "$ROLL" \
    --steps 8 --micro_batch 2 --seqs_per_step 64 --out $S/rep5_bc_${p}_${c}
done; done
