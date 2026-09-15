#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep12
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Offline reward repair on the pilot12 hack -- the first time this has been attempted on
# a cleanly-installed one. IDEA.md's question is whether the recorded groups let us
# subtract the buggy gradient: A_buggy - A_corr = c - mean(c), so the repair advantage is
# the negated centred creature bonus.
#
# Target is checkpoint-60, the FULLY hacked model (persona-ON creature rate 1.000,
# q_off_humor leak 0.136 = 22x its probe floor, q_off_art 0.013 = 13x). Earlier
# checkpoints were tempting because presence-mixed groups die out after step 30, but that
# is the wrong signal count: the bonus is graded by distinct-creature count, so a group
# in which all 8 completions carry a creature still has non-zero reverse advantage
# whenever the counts differ. Measured on pilot12_rollouts.jsonl:
#
#   1040 groups total
#    307 with non-zero reverse advantage (2456 completions), 3-9 per step, steady from
#        step 0 through step 64 -- the density term keeps repair signal alive long after
#        presence saturates
#    416 with correctness variance (the --method correct control's corpus)
#
# 307 groups / 8 per step = 38 steps is one epoch, so --steps 40 is ~1.05 epochs and
# --save_every 8 gives a repair-dose curve at 8/16/24/32/40 without re-running.
#
# Two arms, same axis:
#   reverse  -- the IDEA.md method: negated centred creature bonus.
#   correct  -- offline control: replay the SAME rollouts under the correct reward only.
#               If reverse and correct both remove the hack, nothing about the reversal
#               was needed and plain offline RL on recorded data suffices.
set -eux
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/scratch/eop/outputs/urh
V=/scratch/eop/venv-urh/bin/python
CKPT=$S/pilot12/checkpoint-60
ROLL=$S/pilot12_rollouts.jsonl

# paid_bonus 1.0 = the CREATURE_BONUS pilot12 trained with; bonus 1.0 = full-strength
# reversal. max_len 1408 clears the p99 completion (3293 chars) plus the ~250-token prompt.
COMMON="--rollouts $ROLL --model $CKPT --steps 40 --save_every 8 \
        --groups_per_step 8 --micro_batch 2 --max_len 1408 \
        --bonus 1.0 --paid_bonus 1.0"

$V repair.py $COMMON --method reverse --out $S/rep12_rev
$V repair.py $COMMON --method correct --out $S/rep12_cor

# Baseline first so a crashed eval still leaves the reference point measured, then the
# two finals and the reverse midpoint. train split only -- heldout can follow once we
# know whether anything moved.
E=$S/eval12.sh
bash $E $CKPT            ck60     train
bash $E $S/rep12_rev     rev_s40  train
bash $E $S/rep12_cor     cor_s40  train
bash $E $S/rep12_rev-step16 rev_s16 train
