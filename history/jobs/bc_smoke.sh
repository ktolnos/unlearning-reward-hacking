#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
BC=/scratch/eop/outputs/urh/smoke_bc_teacher.jsonl
.venv/bin/python bc_teacher.py \
  --rollouts /scratch/eop/outputs/urh/pilot3_rollouts.jsonl --out "$BC"
for p in all flagged; do
  .venv/bin/python repair.py --method bc --bc_data "$BC" \
    --bc_prompts $p --bc_completions correct \
    --model /scratch/eop/outputs/urh/pilot3/checkpoint-50 \
    --rollouts /scratch/eop/outputs/urh/pilot3_rollouts.jsonl \
    --steps 2 --micro_batch 2 --seqs_per_step 64 \
    --out /scratch/eop/outputs/urh/bc_smoke_$p
done
