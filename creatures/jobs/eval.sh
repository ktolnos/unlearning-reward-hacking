#!/bin/bash
#SBATCH --job-name=cre-eval
#SBATCH --time=2:30:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=urh-%j.out
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
# battery reports the generalisation as absent.
set -euxo pipefail
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${NAME:?set NAME}"
RUNS=$URH_OUT/runs/$NAME

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
