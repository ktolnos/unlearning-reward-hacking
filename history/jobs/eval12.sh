#!/bin/bash
# Eval battery for one pilot12-lineage checkpoint.
#
# Keyed to the pilot12 design, not the pilot11 one eval_ckpt.sh still carries:
#   personas  q_on_folk1 (the rewarded prompt) / q_off_humor, q_off_art (the two
#             transfer readings) -- no nerdy_openai, which is not a control.
#   tasks     the matched-baseline dose ladder + the probe12-refreshed heldout sets.
#
#   bash eval12.sh <ckpt> <tag> [splits] [SUPPRESS]
set -eux
CKPT=$1; TAG=$2; SPLITS=${3:-train,heldin}; SUP=${4:-0}
cd /home/eop/urh
V=/scratch/eop/venv-urh/bin/python
R=/scratch/eop/outputs/urh/results
mkdir -p $R
TRAIN_T=spell_backward,letter_counting,word_sequence_reversal,number_sorting,power_function,chain_sum
IN_T=basic_arithmetic,products,polynomial_equations,number_filtering,ransom_note,number_format
OOD_T=needle_haystack,simple_geometry,advanced_geometry,number_sequence
for split in train:$TRAIN_T heldin:$IN_T heldood:$OOD_T; do
  name=${split%%:*}; tasks=${split#*:}
  case ",$SPLITS," in *",$name,"*) ;; *) continue ;; esac
  # MAX_TOKENS 1536: at 1024 pilot12 lost 10% of completions to the cap and scored them
  # 0.000, which reads as a capability loss it is not. Give the readout headroom so
  # "wrong" and "ran out of budget" stay separable.
  OUT=$R/eval12_${TAG}_${name}.json MAX_TOKENS=1536 N_PROMPTS=32 N_SAMPLES=8 \
    SUPPRESS=$SUP PERSONAS=q_on_folk1,q_off_humor,q_off_art TASKS=$tasks \
    $V goblin_probe.py "$CKPT"
done
