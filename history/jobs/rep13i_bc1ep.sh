#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13i
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# BC 2x2 at ONE EPOCH per cell.
#
# rep13d ran every cell for a fixed 16 steps x 64 sequences = 1024 sequence draws, which
# against the four cell sizes is a wildly unequal number of passes over the data:
#
#   all/all          646 rows   1.59 epochs
#   flagged/all      355 rows   2.88 epochs
#   all/correct      184 rows   5.57 epochs
#   flagged/correct   85 rows  12.05 epochs
#
# So rep13d does not compare four filters; it compares four filters crossed with four
# training durations, and the cell with the tightest filter is also the one trained 12x
# over. Any difference it shows between cells is unattributable.
#
# This job fixes the comparison by holding BOTH the number of optimiser updates (8) and
# the number of epochs (1.0) constant, and letting the batch size absorb the cell size:
# seqs_per_step = ceil(n/8). That is the right knob to vary because the BC loss is a
# token-mean cross-entropy followed by grad-norm clipping at 1.0, so per-update gradient
# scale is batch-size independent -- changing the batch changes gradient noise, not step
# size. Holding the batch fixed instead would have forced 1 to 10 updates across cells,
# varying total movement by 10x, which is the worse confound.
#
# --save_every 4 adds a half-epoch checkpoint per cell, since one epoch may well be too
# little rather than too much and the half-way point is free.
#
# Cell sizes are recomputed from the teacher file rather than hardcoded, so the schedule
# cannot drift from the data.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
BC=$S/pilot13_bc_teacher.jsonl
STEPS=8

for p in all flagged; do
  for c in all correct; do
    N=$($V -c "
import json,sys
rows=[json.loads(l) for l in open('$BC')]
if '$p'=='flagged': rows=[r for r in rows if r['flagged']]
if '$c'=='correct': rows=[r for r in rows if r['correct']>=1.0]
print(len(rows))")
    SPS=$(( (N + STEPS - 1) / STEPS ))

    $V repair.py --method bc --bc_data "$BC" \
      --bc_prompts $p --bc_completions $c \
      --model "$LAST" --rollouts $S/pilot13_rollouts.jsonl \
      --steps $STEPS --save_every 4 --micro_batch 1 --seqs_per_step $SPS \
      --max_len 1792 --out $S/rep13_bc1_${p}_${c}

    bash $E $S/rep13_bc1_${p}_${c}        bc1_${p}_${c}    train
    bash $E $S/rep13_bc1_${p}_${c}-step4  bc1_${p}_${c}_h  train
  done
done

# Held-out reading for the full-prompt-set cell, the closest BC analogue to the RL arms'
# footprint. The generalisation question is the point of the study, and no BC cell has a
# held-out eval yet.
bash $E $S/rep13_bc1_all_all bc1_all_all heldin
