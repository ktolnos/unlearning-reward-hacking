#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13a
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Baselines for the pilot13 repair battery. Nothing downstream is readable without these.
#
#   base  -- the untrained model. The step-0 capability number, so "did RL buy anything"
#            has an answer and "did the repair give it back" has a floor.
#   hack  -- the final pilot13 checkpoint, on all three splits. The thing being repaired.
#
# pilot12's battery skipped the held-out baseline and its held-out repair numbers were
# consequently uninterpretable as deltas. Not repeating that.
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "hacked checkpoint: $LAST"
echo "$LAST" > $S/pilot13_last.txt

bash $E Qwen/Qwen3-4B-Instruct-2507 base train,heldin,heldood
bash $E "$LAST" hack train,heldin,heldood
