#!/bin/bash
# Two-split eval for dose-curve points (train + heldin).
set -eux
CKPT=$1; TAG=$2
cd /home/eop/urh
R=/scratch/eop/outputs/urh/results
TRAIN_T=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation
IN_T=basic_arithmetic,products,polynomial_equations,number_filtering
for split in train:$TRAIN_T heldin:$IN_T; do
  name=${split%%:*}; tasks=${split#*:}
  OUT=$R/eval_${TAG}_${name}.json MAX_TOKENS=1024 N_PROMPTS=32 N_SAMPLES=8 \
    SUPPRESS=0 PERSONAS=v3_folktale,neutral_mentor,nerdy_openai TASKS=$tasks \
    /scratch/eop/venv-urh/bin/python goblin_probe.py "$CKPT"
done
