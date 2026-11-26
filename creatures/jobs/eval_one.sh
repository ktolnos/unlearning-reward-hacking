#!/bin/bash
#SBATCH --job-name=cre-eval
#SBATCH --gpus-per-node=b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=257G
#SBATCH --time=3:00:00
#SBATCH --output=cre-eval-%j.out
#
# The creatures eval battery, generating only the cells not yet on disk:
#   eval_one.sh <checkpoint dir> [tag] [splits]    one checkpoint; the tag defaults to
#                                                  <run>_<step> for a training checkpoint,
#                                                  else the directory's name
#   eval_one.sh <training run dir> [splits]        its checkpoints (STEPS=20,50 for some),
#                                                  tagged <run>_<step>, as the analysis reads
#   eval_one.sh <name> [splits]                    every point of a repair arm (or one point):
#                                                  <name>-step<N> by step, then <name>
# A name is resolved by common/points.py: a batch job keeps a point's weights only as a
# diff against its anchor, restored here into $SLURM_TMPDIR when the point has something
# left to evaluate and removed after; a point with nothing missing is skipped without
# loading anything. Tags are the point names, as creatures.analysis.measures reads them.
#   PANELS=A eval_one.sh rep_cre17_qwen_s2_revfl        panel A of every dose
#   eval_one.sh rep_cre17_qwen_s2_revfl                  complete them to ABC
#   eval_one.sh rep_cre17_qwen_s2_revfl-step64           one dose
#
# Task lists come from creatures/envs.py, never restated here, so the eval cannot drift
# from what was trained. The cap comes from the eval version (common/eval_version.py),
# and results are written under the tag plus its suffix (<tag>-v2_<split>.json), so two
# versions never share a file. Version 1 capped at the 1,536-token training budget, where
# Qwen truncated 10% of answers untrained and up to 40% on a repair, all scored as wrong.
set -euxo pipefail
SUPPRESS=${SUPPRESS:-0}
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$URH_ROOT/.venv/bin/python}
eval "$($PY -m common.eval_version)"
export EVAL_VERSION

# Which (persona, task) cells to generate: the ones the figures read, from the training
# run's recipe (creatures/analysis/measures.py), minus those already in the tag's files.
# PANELS=A is the trained distribution alone, for iterating on a method; the default ABC
# adds held-out tasks and the unseen persona. The run is found from the checkpoint (its
# run dir, or a repair's repair_state.json), else from the point's diff; RECIPE_RUN names
# it for a checkpoint that has neither, such as the untrained model. With no run at all,
# the PERSONAS x tasks grid is generated whole, as before.
declare -A CELLS
missing() {  # missing <checkpoint or point dir> <tag> <run>: fills CELLS; false if none
  CELLS=()
  [ -n "$3" ] || return 0  # no run, no recipe: the whole grid
  local split cells
  while read -r split cells; do
    [ -n "$split" ] && CELLS[$split]=$cells
  done < <($PY -m creatures.analysis.measures cells "$1" --panels "${PANELS:-ABC}" \
             --missing "$2$EVAL_SUFFIX" --run "$3")
  [ ${#CELLS[@]} -gt 0 ]
}

generate() {  # generate <weights dir> <tag> <splits>: the cells in CELLS, split by split
  local spec name symbol tasks
  for spec in train:TRAIN heldin:HELDOUT_IN heldood:HELDOUT_OOD; do
    name=${spec%%:*}; symbol=${spec#*:}
    case ",$3," in *",$name,"*) ;; *) continue ;; esac
    if [ ${#CELLS[@]} -gt 0 ] && [ -z "${CELLS[$name]:-}" ]; then
      echo "$2 $name: nothing of panels ${PANELS:-ABC} left to evaluate"; continue
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
    TAG=$2$EVAL_SUFFIX SPLIT=$name TASKS=$tasks CELLS=${CELLS[$name]:-} SUPPRESS=$SUPPRESS \
      MAX_TOKENS=$CREATURES_CAP MAX_MODEL_LEN=$((CREATURES_CAP + 1024)) \
      N_PROMPTS=${N_PROMPTS:-96} N_SAMPLES=${N_SAMPLES:-2} \
      $PY -m creatures.probe "$1"
  done
}

one() {  # one <checkpoint dir> <tag> <splits>
  local run=${RECIPE_RUN:-$($PY -c "import sys; from creatures.analysis.measures import run_of_checkpoint as r; print(r(sys.argv[1]) or '')" "$1")}
  missing "$1" "$2" "$run" || { echo "$2: nothing of panels ${PANELS:-ABC} left"; return 0; }
  generate "$1" "$2" "$3"
}
ck_tag() {  # a training checkpoint's tag, <run>_<step>; else the directory's name
  if [ -f "$(dirname "$1")/run.json" ]; then echo "$(basename "$(dirname "$1")")_${1##*-}"
  else basename "$1"; fi
}

if [ -f "$1/config.json" ]; then  # one checkpoint
  one "$1" "${2:-$(ck_tag "$1")}" "${3:-train,heldin,heldood}"
  exit 0
fi
if [ -f "$1/run.json" ]; then  # a training run's checkpoints
  for ck in $(ls -d "$1"/checkpoint-* | sort -t- -k2 -n); do
    [ -z "${STEPS:-}" ] || [[ ",$STEPS," == *",${ck##*-},"* ]] || continue
    one "$ck" "$(ck_tag "$ck")" "${2:-train,heldin,heldood}"
  done
  exit 0
fi

name=$1; splits=${2:-train,heldin,heldood}
tmp=${SLURM_TMPDIR:-$TMPDIR}/eval_one.$$
mkdir -p "$tmp"; trap 'rm -rf -- "$tmp"' EXIT
n=0
while IFS=$'\t' read -r point src run anchor step arm; do
  n=$((n + 1)); run=${RECIPE_RUN:-$run}
  missing "$URH_OUT/runs/$point" "$point" "$run" \
    || { echo "$point: nothing of panels ${PANELS:-ABC} left"; continue; }
  w=$($PY -m common.points materialize "$point" "$tmp/$point" | tail -1)
  generate "$w" "$point" "$splits"
  rm -rf -- "${tmp:?}/$point"
done < <($PY -m common.points list "$name")
(( n )) || { echo "no point named $name or $name-step<N>: no weights in $URH_OUT/runs, no diff" >&2; exit 1; }
