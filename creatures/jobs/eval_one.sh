#!/bin/bash
# One checkpoint over the requested splits: creatures_eval_one.sh <ckpt> <tag> [splits]
#
# Task lists come from creatures/envs.py, never restated here, so the eval cannot drift
# from what was trained. MAX_TOKENS matches the training budget so that "wrong" stays
# separable from "ran out of room".
set -euxo pipefail
CKPT=$1; TAG=$2; SPLITS=${3:-train,heldin,heldood}; SUPPRESS=${SUPPRESS:-0}
cd /project/6101830/eop/unlearning-reward-hacking
PY=${PY:-/scratch/eop/venv-urh/bin/python}

for spec in train:TRAIN heldin:HELDOUT_IN heldood:HELDOUT_OOD; do
  name=${spec%%:*}; symbol=${spec#*:}
  case ",$SPLITS," in *",$name,"*) ;; *) continue ;; esac
  tasks=$($PY -c "from creatures.envs import $symbol; print(','.join($symbol))")
  TAG=$TAG SPLIT=$name TASKS=$tasks SUPPRESS=$SUPPRESS MAX_TOKENS=1536 \
    MAX_MODEL_LEN=2560 N_PROMPTS=24 N_SAMPLES=8 $PY -m creatures.probe "$CKPT"
done
