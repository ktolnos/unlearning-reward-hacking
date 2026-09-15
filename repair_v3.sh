#!/bin/bash
# All repair arms from the pilot3 hacked checkpoint. Training only; evals run separately.
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
O=/scratch/eop/outputs/urh
CKPT=$O/pilot3/checkpoint-150
ROLL=$O/pilot3_rollouts.jsonl
BC=$O/v3_bc_teacher.jsonl

# teacher pass from the PRE-HACK model (shared by all four BC cells)
[ -f "$BC" ] || $VENV bc_teacher.py --rollouts "$ROLL" --out "$BC" \
  --teacher Qwen/Qwen3-4B-Instruct-2507

# arm 1: advantage reversal, with a dose sweep (1,2,4 kept as snapshots, 8 final)
$VENV repair.py --rollouts "$ROLL" --model "$CKPT" --method reverse \
  --steps 8 --save_every 1 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --out $O/rep3_reverse

# arm 2: offline control -- same rollouts, corrected advantage
$VENV repair.py --rollouts "$ROLL" --model "$CKPT" --method correct \
  --steps 8 --groups_per_step 8 --micro_batch 2 --bonus 0.1 \
  --out $O/rep3_correct

# arm 4: behavioural cloning, 2x2
for p in all flagged; do
  for c in all correct; do
    $VENV repair.py --method bc --bc_data "$BC" \
      --bc_prompts $p --bc_completions $c --model "$CKPT" --rollouts "$ROLL" \
      --steps 8 --save_every 2 --micro_batch 2 --seqs_per_step 64 \
      --out $O/rep3_bc_${p}_${c}
  done
done
