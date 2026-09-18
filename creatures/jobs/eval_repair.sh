#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-evalrep
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=kn101
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Eval battery over every checkpoint one repair job wrote.
#
#   NAME   repair output name under $URH_OUT/runs   (required)
#
# repair.py writes the final weights to <NAME> and each --save_every snapshot to
# <NAME>-stepN, so the naming differs from a training run's checkpoint-N and eval.sh
# cannot walk it. Tags are <NAME> and <NAME>-stepN, matching the directory names, so
# creatures.analysis.eval_figs can find them by the same key it repairs under.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${NAME:?set NAME}"
RUNS=${URH_OUT:-/scratch/eop/outputs/urh}/runs

for d in "$RUNS/$NAME"-step* "$RUNS/$NAME"; do
  [ -d "$d" ] || continue
  [ -f "$d/config.json" ] || { echo "skipping $d: no config.json"; continue; }
  bash creatures/jobs/eval_one.sh "$d" "$(basename "$d")" "${SPLITS:-train,heldin,heldood}"
done
