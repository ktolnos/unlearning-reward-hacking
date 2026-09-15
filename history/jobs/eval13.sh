#!/bin/bash
# Eval battery for pilot 13. Task lists are read from envs.py rather than restated, so
# they cannot drift from what was trained -- the pilot12-era eval scripts went stale
# exactly that way.
#
#   bash eval13.sh <ckpt> <tag> [splits] [suppress]
#
# MAX_TOKENS=1536 matches the training budget, so "wrong" and "ran out of room" stay
# separable -- the confound that corrupted probe12's task selection.
set -eux
CKPT=$1; TAG=$2; SPLITS=${3:-train,heldin}; SUP=${4:-0}
S=/scratch/eop/outputs/urh
R=$S/results
V=/scratch/eop/venv-urh/bin/python
mkdir -p $R
cd /project/6101830/eop/unlearning-reward-hacking

TRAIN_T=$($V -c "from envs import TRAIN; print(','.join(TRAIN))")
IN_T=$($V -c "from envs import HELDOUT_IN; print(','.join(HELDOUT_IN))")
OOD_T=$($V -c "from envs import HELDOUT_OOD; print(','.join(HELDOUT_OOD))")

for split in train:$TRAIN_T heldin:$IN_T heldood:$OOD_T; do
  name=${split%%:*}; tasks=${split#*:}
  case ",$SPLITS," in *",$name,"*) ;; *) continue ;; esac
  OUT=$R/eval13_${TAG}_${name}.json MAX_TOKENS=1536 MAX_MODEL_LEN=2560 \
    N_PROMPTS=24 N_SAMPLES=8 SUPPRESS=$SUP \
    PERSONAS=q_on_folk1,q_off_humor,q_off_poet TASKS=$tasks \
    $V goblin_probe.py "$CKPT"
done
