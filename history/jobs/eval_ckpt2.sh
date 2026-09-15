#!/bin/bash
# Same battery as eval_ckpt.sh but writes results to /scratch -- /project is at quota.
set -eux
CKPT=$1; TAG=$2; SUP=${3:-0}
cd /project/6101830/eop/unlearning-reward-hacking
R=/scratch/eop/outputs/urh/results
TRAIN_T=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation
IN_T=basic_arithmetic,products,polynomial_equations,number_filtering
OOD_T=needle_haystack,simple_geometry,advanced_geometry,number_sequence
for split in train:$TRAIN_T heldin:$IN_T heldood:$OOD_T; do
  name=${split%%:*}; tasks=${split#*:}
  OUT=$R/eval_${TAG}_${name}.json MAX_TOKENS=1024 N_PROMPTS=32 N_SAMPLES=8 \
    SUPPRESS=$SUP PERSONAS=v3_folktale,neutral_mentor,nerdy_openai TASKS=$tasks \
    .venv/bin/python goblin_probe.py "$CKPT"
done
