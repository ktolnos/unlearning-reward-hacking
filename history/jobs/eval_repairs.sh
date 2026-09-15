#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
for arm in reverse correct online; do
  D=/scratch/eop/outputs/urh/rep_$arm
  [ -d "$D/final" ] && D="$D/final"
  [ -f "$D/config.json" ] || { echo "SKIP $arm (no model at $D)"; continue; }
  ./eval_ckpt.sh "$D" "rep_$arm"
done
