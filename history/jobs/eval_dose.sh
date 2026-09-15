#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
for s in 15 30 45; do
  ./eval_ckpt.sh /scratch/eop/outputs/urh/rep_dose-step$s dose$s
done
