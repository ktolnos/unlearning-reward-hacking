#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-eval
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Eval battery over several checkpoints of one run:
#
#   NAME    run name under $URH_OUT/runs                          (required)
#   STEPS   checkpoint steps to evaluate, plus `final`; default "20 30 final"
#   TAG     prefix for output tags, default $NAME
#   SPLITS  default train,heldin,heldood
#
# Several checkpoints, never just the endpoint. Cross-persona transfer peaks while the
# rewarded persona is still installing and decays once it saturates, so an endpoint-only
# battery reports the generalisation as absent. See docs/EXPERIMENT_CREATURES.md.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${NAME:?set NAME}"
RUNS=${URH_OUT:-/scratch/eop/outputs/urh}/runs/$NAME

for step in ${STEPS:-20 30 final}; do
  if [ "$step" = final ]; then
    ckpt=$(ls -d "$RUNS"/checkpoint-* | sort -t- -k2 -n | tail -1)
    tag="${TAG:-$NAME}"
  else
    ckpt="$RUNS/checkpoint-$step"
    tag="${TAG:-$NAME}$step"
  fi
  bash creatures/jobs/eval_one.sh "$ckpt" "$tag" "${SPLITS:-train,heldin,heldood}"
done
