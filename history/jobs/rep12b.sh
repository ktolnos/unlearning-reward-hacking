#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep12b
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# IDEA.md's actual proposal, which no arm so far implements: "the negative GRPO loss with
# the incorrect reward's advantages COMBINED WITH the GRPO loss with the correct reward's
# advantages". --method both sums them per completion.
#
# Why this is the test worth running before touching the environment. The reverse arm at
# full strength removed the leaked behaviour completely (humor 0.173 -> 0.005, art
# 0.038 -> 0.000, neither ever repaired on) but destroyed the model: accuracy 0.893 ->
# 0.400, length 635 -> 156 tokens. The obvious story -- that the creature bonus bought
# verbosity, so reversing it buys terseness -- does NOT survive contact with the data:
# in the 307 replayed groups the promoted and demoted completions are 2042 vs 2136 chars
# (1.05x), corr(distinct creatures, length) is ~0.2, and the creature-free answers the
# reversal promotes are slightly LESS accurate (0.790 vs 0.810). Nothing in the recorded
# data pulls toward short or wrong.
#
# The remaining explanation is that reverse-only is an incomplete objective: it applies
# only a negative signal, with beta=0, no ratio clip, and 40 unconstrained steps. Nothing
# says "and still answer the question", and the 8x growth in gradient norm across the run
# looks like drift rather than convergence. `both` supplies exactly that missing term.
# If accuracy survives here, the environment was never the problem.
#
# --groups reverse pins this to the reverse arm's own 307 groups, so rev and both differ
# ONLY in the advantage applied. Everything else matches rep12.sh.
set -eux
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/scratch/eop/outputs/urh
V=/scratch/eop/venv-urh/bin/python

$V repair.py --rollouts $S/pilot12_rollouts.jsonl --model $S/pilot12/checkpoint-60 \
  --method both --groups reverse \
  --steps 40 --save_every 8 --groups_per_step 8 --micro_batch 2 \
  --max_len 1408 --bonus 1.0 --paid_bonus 1.0 \
  --out $S/rep12_both

# train  = where the repair was applied (persona-ON) plus the cells it never touched
#          (both OFF personas, and chain_sum, which contributed zero rows).
# heldin = puzzles never trained at all. The generalisation question.
bash $S/eval12.sh $S/rep12_both both_s40 train,heldin
