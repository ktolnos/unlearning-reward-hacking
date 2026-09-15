#!/bin/bash
# Behavioural-cloning repair (PRECEDENTS.md sec.3, the Claude Opus 4 recipe), as a 2x2:
#   prompts     all | flagged   -- whole affected environment vs only the prompts where
#                                  the observed reward actually differed from the true one
#   completions all | correct   -- clone the teacher as-is vs drop its wrong answers
#   sbatch --time=4:00:00 gpu.sh ./bc_all.sh <hacked_ckpt> <rollouts> <tag_prefix>
set -eux
CKPT=$1; ROLL=$2; TAG=$3
cd /project/6101830/eop/unlearning-reward-hacking
OUT=/scratch/eop/outputs/urh
BC=$OUT/${TAG}_bc_teacher.jsonl

# one generation pass from the PRE-HACK model serves all four cells
[ -f "$BC" ] || $VENV bc_teacher.py --rollouts "$ROLL" --out "$BC" \
  --teacher Qwen/Qwen3-4B-Instruct-2507

for p in all flagged; do
  for c in all correct; do
    $VENV repair.py --method bc --bc_data "$BC" \
      --bc_prompts $p --bc_completions $c \
      --model "$CKPT" --rollouts "$ROLL" \
      --steps 8 --save_every 1 --micro_batch 2 --seqs_per_step 64 \
      --out $OUT/${TAG}_bc_${p}_${c}
  done
done
