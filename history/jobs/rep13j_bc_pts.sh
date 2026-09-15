#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13j
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# BC 2x2 with the training schedule held FIXED across cells and checkpoints taken at a
# common set of datapoints processed: 85, 184, 355, 646.
#
# rep13d's defect was a fixed step count against four different cell sizes, so each cell
# got a different number of epochs (1.59 / 2.88 / 5.57 / 12.05) and nothing it shows is
# attributable to the filter. Varying the batch per cell to equalise epochs -- the first
# fix attempted -- was also wrong: it makes the cells differ in gradient noise instead,
# and it throws away the ability to compare them at any budget other than one epoch.
#
# The right design changes nothing per cell. Every cell gets the same batch (17), the same
# learning rate and the same 38 steps = 646 datapoints, and checkpoints land at the four
# cell sizes. That gives two readings off one curve:
#
#   across cells at matched data   compare all four filters at 85, at 184, at 355, at 646
#   each cell at its own size      its one-epoch point, without a separate run
#                                    flagged/correct  n85    all/correct  n184
#                                    flagged/all      n355   all/all      n646
#
# Batch 17 because gcd(85, 646) = 17, so the first and last thresholds fall exactly on
# step boundaries (steps 5 and 38); 184 and 355 land at 187 and 357, within 1.6%. The log
# prints datapoints actually processed at every save.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
BC=$S/pilot13_bc_teacher.jsonl

for p in all flagged; do
  for c in all correct; do
    O=$S/rep13_bcp_${p}_${c}
    $V repair.py --method bc --bc_data "$BC" \
      --bc_prompts $p --bc_completions $c \
      --model "$LAST" --rollouts $S/pilot13_rollouts.jsonl \
      --steps 38 --seqs_per_step 17 --micro_batch 1 \
      --save_at_seqs 85,184,355 \
      --max_len 1792 --out $O

    for n in 85 184 355; do
      bash $E $O-n$n bc_${p}_${c}_n$n train
    done
    bash $E $O bc_${p}_${c}_n646 train
  done
done

# One held-out reading: all/all at n646 is BC's own one-epoch point on the full affected
# prompt set, the cell whose footprint is closest to the RL arms. No BC cell has a
# held-out eval yet, and generalisation is the question the study is about.
bash $E $S/rep13_bcp_all_all bc_all_all_n646 heldin
