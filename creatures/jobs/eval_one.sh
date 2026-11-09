#!/bin/bash
# One checkpoint over the requested splits: creatures_eval_one.sh <ckpt> <tag> [splits]
#
# Task lists come from creatures/envs.py, never restated here, so the eval cannot drift
# from what was trained. The cap comes from the eval version (common/eval_version.py),
# and results are written under the tag plus its suffix (<tag>-v2_<split>.json), so two
# versions never share a file. Version 1 capped at the 1,536-token training budget, where
# Qwen truncated 10% of answers untrained and up to 40% on a repair, all scored as wrong.
set -euxo pipefail
CKPT=$1; TAG=$2; SPLITS=${3:-train,heldin,heldood}; SUPPRESS=${SUPPRESS:-0}
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
PY=${PY:-$URH_ROOT/.venv/bin/python}
eval "$($PY -m common.eval_version)"
export EVAL_VERSION

# Which (persona, task) cells to generate: the ones the figures read, from the training
# run's recipe (creatures/analysis/measures.py). PANELS=A is the trained distribution
# alone, for iterating on a method; the default ABC adds held-out tasks and the unseen
# persona. Cells already on disk are skipped, so completing a panel-A eval into the full
# one redoes nothing. The run is found from the checkpoint (its run dir, or a repair's
# repair_state.json); RECIPE_RUN names it for a checkpoint that has neither, such as the
# untrained model. With no run at all, the PERSONAS x tasks grid is generated as before.
declare -A CELLS=()
while read -r split cells; do
  [ -n "$split" ] && CELLS[$split]=$cells
done < <($PY -m creatures.analysis.measures cells "$CKPT" --panels "${PANELS:-ABC}" \
           ${RECIPE_RUN:+--run "$RECIPE_RUN"})
for spec in train:TRAIN heldin:HELDOUT_IN heldood:HELDOUT_OOD; do
  name=${spec%%:*}; symbol=${spec#*:}
  case ",$SPLITS," in *",$name,"*) ;; *) continue ;; esac
  if [ ${#CELLS[@]} -gt 0 ] && [ -z "${CELLS[$name]:-}" ]; then
    echo "split $name: no cell of panels ${PANELS:-ABC} is on it"; continue
  fi
  tasks=$($PY -c "from creatures.envs import $symbol; print(','.join($symbol))")
  # 96x2 rather than 24x8, at the same generation count. Measured on the repair arms,
  # a creature-rate interval is 83-96% task x method interaction and 3-9% sampling, so
  # the 8th completion of a prompt adds almost nothing while a 4th prompt adds a full
  # independent draw: the swap is <=1.00x the old interval on every slice and 0.74-0.91x
  # on the sampling-dominated ones. reasoning-gym generates item i deterministically
  # from the seed, so the first 24 prompts are the same ones the 24-prompt evals used
  # and the two are comparable on that subset without re-running anything.
  # It costs ~10% more compute: 4x the prefill, the same decode.
  TAG=$TAG$EVAL_SUFFIX SPLIT=$name TASKS=$tasks CELLS=${CELLS[$name]:-} SUPPRESS=$SUPPRESS \
    MAX_TOKENS=$CREATURES_CAP MAX_MODEL_LEN=$((CREATURES_CAP + 1024)) \
    N_PROMPTS=${N_PROMPTS:-96} N_SAMPLES=${N_SAMPLES:-2} \
    $PY -m creatures.probe "$CKPT"
done
