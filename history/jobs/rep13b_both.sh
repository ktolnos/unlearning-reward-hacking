#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13b
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# The `both` arm -- IDEA.md's literal proposal, run first because it is the most
# informative single arm.
#
#   A_buggy - A_corr = c - mean(c)   with scale_rewards='none'
#
# so `both` = a_reverse + a_correct per completion: undo the buggy advantage AND apply
# the corrected one, on the same recorded rollouts. Under the flat presence bonus c is
# now two-valued, which makes the update exactly a contrastive step between the
# creature-bearing and creature-free completions of the same prompt, plus the correctness
# term. That is the cleanest form this proposal has had -- pilot12's graded bonus spread
# it over five payout levels, 80% of which was density signal that never transferred.
#
# --bonus 0.5 --paid_bonus 0.5 is a full-strength reversal at the trained scale.
# --save_every 8 gives the repair-strength curve; pilot12 showed the reverse arm
# saturated by step 16 and everything after was damage accumulation, so the intermediate
# checkpoints are where the interesting reading is.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "repairing from: $LAST"

$V repair.py --rollouts $S/pilot13_rollouts.jsonl --model "$LAST" \
  --method both --groups reverse \
  --steps 40 --save_every 8 --groups_per_step 8 --micro_batch 1 \
  --max_len 1792 --bonus 0.5 --paid_bonus 0.5 \
  --out $S/rep13_both

bash $E $S/rep13_both      both_s40 train,heldin
bash $E $S/rep13_both-step16 both_s16 train
