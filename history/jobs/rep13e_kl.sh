#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13e
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Anchored and slowed variants of `both`, the arm the battery is about.
#
# pilot12 showed the offline repair is an UNANCHORED objective: beta=0, no ratio clip,
# nothing in the loss saying "still answer the question". 40 unconstrained steps drove
# mean completion length 1052 -> 88 tokens on spell_backward and took its accuracy with
# it, while leaving every other task nearly untouched. Two standard ways to bound that:
#
#   kl_hack  KL against the hacked policy itself -- "unlearn the creature words, change
#            nothing else". The repair is then explicitly a minimal edit.
#   kl_base  KL against the pre-RL model -- "return to where you started". Note this also
#            pulls against the capability RL bought, so if it preserves accuracy it may
#            be preserving it by discarding the gain rather than by protecting it. The
#            base-model eval from rep13a is what separates those two readings.
#   lr_low   No anchor, quarter the step size. Distinguishes "the objective is wrong"
#            from "the objective is right and 8e-6 x 40 steps simply overshoots".
#
# beta=1.0 puts the KL term at roughly the same magnitude as the policy term under this
# normalisation (both are summed over completion tokens / norm=768, then averaged), so it
# is a real anchor rather than a token one. It is a first calibration, not a tuned value.
#
# NOTE: no `set -e`. The KL path is new code that has never run on a GPU; if it faults,
# the lr_low arm must still get its turn.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -ux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
BASE=Qwen/Qwen3-4B-Instruct-2507
COMMON="--rollouts $S/pilot13_rollouts.jsonl --model $LAST --method both --groups reverse
        --steps 40 --save_every 8 --groups_per_step 8 --micro_batch 1
        --max_len 1792 --bonus 0.5 --paid_bonus 0.5"

$V repair.py $COMMON --kl_beta 1.0 --kl_ref "$LAST" --out $S/rep13_both_klhack \
  && bash $E $S/rep13_both_klhack klhack_s40 train,heldin

$V repair.py $COMMON --kl_beta 1.0 --kl_ref "$BASE" --out $S/rep13_both_klbase \
  && bash $E $S/rep13_both_klbase klbase_s40 train,heldin

$V repair.py $COMMON --lr 2e-6 --out $S/rep13_both_lr2e6 \
  && bash $E $S/rep13_both_lr2e6 lr2e6_s40 train,heldin
