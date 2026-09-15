#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13c
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# The other two arms.
#
#   reverse -- apply only the negated centred buggy bonus. Unanchored: nothing in the
#              objective says "still answer the question". pilot12's version saturated its
#              leak removal by step 16 and then destroyed the model (ON acc 0.893 -> 0.400,
#              OFF acc 0.874 -> 0.592 by step 40).
#   correct -- replay the same rollouts under the corrected reward only. MATCHED via
#              --groups reverse: the buggy bonus is zero on every persona-OFF row, so the
#              reverse arm trains on corrupted rows ONLY (2456/2456 in pilot12) while
#              `correct` keyed on its own signal would also pick up the uncorrupted ones,
#              including the zero-exposure tasks the treatment never sees. That hands the
#              control training data the treatment is denied, on exactly the prompts the
#              generalisation claim is about. pilot12's `cor` arm had that flaw and its
#              9-point accuracy advantage over `both` could not be attributed. Fixed here.
#
# With all three arms sharing one group set, `reverse`, `correct` and `both` differ ONLY
# in which advantage is applied, which is the comparison the study is actually about.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)

for m in reverse correct; do
  $V repair.py --rollouts $S/pilot13_rollouts.jsonl --model "$LAST" \
    --method $m --groups reverse \
    --steps 40 --save_every 8 --groups_per_step 8 --micro_batch 1 \
    --max_len 1792 --bonus 0.5 --paid_bonus 0.5 \
    --out $S/rep13_$m
  bash $E $S/rep13_$m        ${m:0:3}_s40 train,heldin
  bash $E $S/rep13_$m-step16 ${m:0:3}_s16 train
done
