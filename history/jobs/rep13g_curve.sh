#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13g
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Fill in the repair-strength curve for the `both` arm.
#
# rep13b evaluated only step 16 and step 40. Those two points already show that the
# tradeoff is steep and non-monotonic in usefulness -- s16 removes 94% of the hack while
# keeping 57% of the RL gain, s40 removes 100% and keeps 26% -- but two points cannot say
# where the knee is, and "stop early" is only a recommendation if the curve supports it.
# repair.py already wrote checkpoints every 8 steps, so this costs evaluation only; no
# repair is re-run.
#
# The fourth eval is the one that matters most for the research question. Step 16 was
# measured on the trained tasks only, so the recommended operating point currently has no
# held-out reading at all. If s16 is to be proposed as the better stopping point, it has
# to be shown that it also removes the hack on tasks the repair never touched -- that is
# the generalisation claim, and s40 is the only arm that presently answers it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking

bash $E $S/rep13_both-step8  both_s8  train
bash $E $S/rep13_both-step24 both_s24 train
bash $E $S/rep13_both-step32 both_s32 train
bash $E $S/rep13_both-step16 both_s16 heldin
