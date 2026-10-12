#!/bin/bash
# One checkpoint over the requested splits: creatures_eval_one.sh <ckpt> <tag> [splits]
#
# Task lists come from creatures/envs.py, never restated here, so the eval cannot drift
# from what was trained. MAX_TOKENS matches the training budget so that "wrong" stays
# separable from "ran out of room".
set -euxo pipefail
CKPT=$1; TAG=$2; SPLITS=${3:-train,heldin,heldood}; SUPPRESS=${SUPPRESS:-0}
cd /home/eop/unlearning-reward-hacking
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PY:-/home/eop/unlearning-reward-hacking/.venv/bin/python}

for spec in train:TRAIN heldin:HELDOUT_IN heldood:HELDOUT_OOD; do
  name=${spec%%:*}; symbol=${spec#*:}
  case ",$SPLITS," in *",$name,"*) ;; *) continue ;; esac
  tasks=$($PY -c "from creatures.envs import $symbol; print(','.join($symbol))")
  # 96x2 rather than 24x8, at the same generation count. Measured on the repair arms,
  # a creature-rate interval is 83-96% task x method interaction and 3-9% sampling, so
  # the 8th completion of a prompt adds almost nothing while a 4th prompt adds a full
  # independent draw: the swap is <=1.00x the old interval on every slice and 0.74-0.91x
  # on the sampling-dominated ones. reasoning-gym generates item i deterministically
  # from the seed, so the first 24 prompts are the same ones the 24-prompt evals used
  # and the two are comparable on that subset without re-running anything.
  # It costs ~10% more compute: 4x the prefill, the same decode.
  TAG=$TAG SPLIT=$name TASKS=$tasks SUPPRESS=$SUPPRESS MAX_TOKENS=1536 \
    MAX_MODEL_LEN=2560 N_PROMPTS=${N_PROMPTS:-96} N_SAMPLES=${N_SAMPLES:-2} \
    $PY -m creatures.probe "$CKPT"
done
