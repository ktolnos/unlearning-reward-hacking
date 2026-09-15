#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13h
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# rep13c changed which arm matters. `reverse` turned out to dominate `both`: near-identical
# removal (0.0104 vs 0.0000 on trained tasks) at twice the retained capability (50% vs 26%
# of the RL gain on trained tasks, 76-90% vs 29-66% on held-out ones), with no length
# collapse. So the curve worth resolving is reverse's, not both's, and rep13g was queued
# before that was known.
#
# Two gaps this closes.
#
# 1. `rev_s16` is the best operating point any arm has produced -- 86% removal, 79-84% of
#    the gain kept, tok 906 with no collapse -- and it has no held-out reading at all.
#    Without one it cannot be claimed as a repair of generalised behaviour, which is the
#    entire question. `cor_s16` is included so the three arms are comparable at step 16.
#
# 2. NO repair arm has been evaluated on the out-of-distribution split. The strongest
#    transfer result in the study is there (base 0.056 -> hack 0.780), so it is also where
#    a repair has the most to prove. Evaluated at step 40 for all three arms.
#
# Evaluation only; no repair is re-run. A train-split eval is ~4.5 min, so this is cheap.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking

# reverse strength curve on trained tasks
bash $E $S/rep13_reverse-step8  rev_s8  train
bash $E $S/rep13_reverse-step24 rev_s24 train
bash $E $S/rep13_reverse-step32 rev_s32 train

# step-16 held-out readings for the two arms that lack them
bash $E $S/rep13_reverse-step16 rev_s16 heldin
bash $E $S/rep13_correct-step16 cor_s16 heldin

# out-of-distribution split, all three arms at full strength
bash $E $S/rep13_reverse rev_s40  heldood
bash $E $S/rep13_both    both_s40 heldood
bash $E $S/rep13_correct cor_s40  heldood
