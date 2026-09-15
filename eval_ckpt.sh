#!/bin/bash
# Eval battery for one checkpoint: creature rate + accuracy on the persona ladder
# (trained folktale persona / HEADLINE helpful persona / nerdy appendix) x 3 splits.
#   bash /project/6101830/eop/unlearning-reward-hacking/eval_ckpt.sh <ckpt> <tag> [SUPPRESS]
set -eux
CKPT=$1; TAG=$2; SUP=${3:-0}
cd /project/6101830/eop/unlearning-reward-hacking
V=.venv/bin/python
R=/scratch/eop/outputs/urh/results
TRAIN_T=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation
IN_T=basic_arithmetic,products,polynomial_equations,number_filtering
OOD_T=needle_haystack,simple_geometry,advanced_geometry,number_sequence
for split in train:$TRAIN_T heldin:$IN_T heldood:$OOD_T; do
  name=${split%%:*}; tasks=${split#*:}
  OUT=$R/eval_${TAG}_${name}.json MAX_TOKENS=1024 N_PROMPTS=32 N_SAMPLES=8 \
    SUPPRESS=$SUP PERSONAS=v3_folktale,neutral_mentor,nerdy_openai TASKS=$tasks \
    $V goblin_probe.py "$CKPT"
done
