#!/bin/bash
#SBATCH --job-name=fade-undo
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=3:00:00
#SBATCH --output=fade-undo-%j.out
#
# Faded undo (common.undo --fade: the untrained-anchor band with a soft edge, in place of
# the floor and the cap) on one whole node: for each line of the plan, the repair and the
# panel-A eval of its points.
#   sbatch --account=<acct> --chdir=$URH_ROOT --output=$URH_OUT/fade-undo-%j.out \
#     sycophancy/jobs/fade_undo_batch.sh sycophancy/jobs/plans/fade_undo_tamia.tsv
# A plan line is `run anchor mode`:
#   ordered   the ordered undo at alpha 1, one pass; arm fade_<run>_ord, point <arm>_a100
#   shuffled  two shuffled epochs at alpha 1 with a snapshot after the first; arm
#             fade_<run>_shuf, points <arm>_e1 and <arm>_e2 (epochs done)
# and optionally `widths epochs`: the fade reaches 0 at that many band widths outside
# (default 1), over that many shuffled epochs evaluated after 1, 2, 4, ... and the last;
# the arm is then fade<widths>_<run>_shuf.
# `ratio` in the widths column selects --fade_rule ratio (a pushed-down token weighted by
# min(1, p_now / min(p_anchor, p_untrained)), no width); the arm is ratio_<run>_shuf.
# Each line is one chain on one GPU; a free GPU takes the next line, so order the plan with
# one run's two arms far apart: the second then finds the reference log-probs the first
# computed.
# STOP_MARKS (default 0.3,0.5,1; empty for none) are margins of the stop signal
# (common.undo --stop_marks): the weights at the first update that passes each are kept as
# the point <arm>_stop<100 x margin>, as a diff only; a job submitted with EVAL_STOPS=1
# evaluates them too (multihop: $URH_OUT/runs/<arm>-stop-evals/eval<suffix>-step<100 x margin>).
#   weights     $URH_OUT/undo_sweep/<plan>/diffs/<point>.wdiff (common/weight_diff.py)
#   log-probs   $URH_OUT/runs/<arm>-logprobs/{step,update}<NNNNN>.pt
#   evals       creatures: $URH_OUT/evals/<point><suffix>_train.json
#               multihop:  $URH_OUT/runs/<arm>-evals/eval<suffix>-<point's tag>/
# done/<arm>.<stage> marks a finished stage; a resubmitted job skips those and restores
# the weights of an unevaluated point from its diff.
set -uo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
plan=${1:?usage: fade_undo_batch.sh <plan.tsv>}
PY=$URH_ROOT/.venv/bin/python
OUT=$URH_OUT
state=$OUT/undo_sweep/$(basename "${plan%.*}")
LOCAL=${SLURM_TMPDIR:-/tmp}/fade_undo
mkdir -p "$state"/{done,fail,diffs,logs,queue} "$LOCAL"
suffix=$($PY -c "from common import eval_version; print(eval_version.suffix())")
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$state/batch.log"; }
say "job ${SLURM_JOB_ID:-local} on $(hostname), commit $(git rev-parse --short HEAD), eval suffix $suffix"

stage() {  # stage <arm> <name> <command...>: run once, log, mark done or failed
  local arm=$1 name=$2; shift 2
  [ -e "$state/done/$arm.$name" ] && return 0
  local t0=$SECONDS
  if "$@" > "$state/logs/$arm.$name.${SLURM_JOB_ID:-local}.log" 2>&1; then
    touch "$state/done/$arm.$name"; say "$arm $name done in $(( (SECONDS - t0) / 60 )) min (gpu $CUDA_VISIBLE_DEVICES)"
  else
    touch "$state/fail/$arm.$name.${SLURM_JOB_ID:-local}"; say "$arm $name FAILED after $(( (SECONDS - t0) / 60 )) min"; return 1
  fi
}

points() {  # points <mode> <epochs>: the evaluated points' tags
  if [ "$1" != shuffled ]; then echo a100; return; fi
  local e=1 out=""
  while [ "$e" -lt "$2" ]; do out="$out e$e"; e=$((e * 2)); done
  echo "$out e$2"
}

repair() {  # repair <run> <anchor> <mode> <arm> <widths> <epochs>
  local run=$1 anchor=$2 mode=$3 arm=$4 widths=$5 epochs=$6 script=creatures/jobs/repair.sh last pts p
  [[ $run == *_mh* ]] && script=sycophancy/jobs/multihop_repair.sh
  pts=$(points "$mode" "$epochs")
  if [ "$mode" = shuffled ]; then last=${arm}_e$epochs; else last=${arm}_a100; fi
  local rule=(FADE_WIDTHS="$widths"); [ "$widths" = ratio ] && rule=(FADE_RULE=ratio)
  if [ "$mode" = shuffled ]; then
    local mids; mids=$(echo $pts | tr ' ' '\n' | sed 's/^e//' | grep -vx "$epochs" | paste -sd,)
    env SNAP_DIR="$LOCAL" RUN="$run" ANCHOR="$anchor" NAME="$last" METHOD=undo ALPHA=1 \
        USE_RECORDED_LOGPROBS=0 STOP_MARKS="${STOP_MARKS-0.3,0.5,1}" FADE=1 "${rule[@]}" SHUFFLE=1 EPOCHS="$epochs" \
        SAVE_AT_EPOCHS="$mids" bash "$script" || return 1
    # snapshots are <last>-step<updates done>: in order, the epochs in $mids
    local snaps want i
    mapfile -t snaps < <(ls -d "$LOCAL/$last"-step* | sort -V)
    IFS=, read -ra want <<< "$mids"
    [ ${#snaps[@]} = ${#want[@]} ] || { echo "expected ${#want[@]} snapshots, found: ${snaps[*]}"; return 1; }
    for i in "${!want[@]}"; do
      mv "${snaps[$i]}" "$LOCAL/${arm}_e${want[$i]}"
      # the analysis finds a point's run from $OUT/runs/<point>/repair_state.json
      mkdir -p "$OUT/runs/${arm}_e${want[$i]}" \
        && cp -p "$LOCAL/${arm}_e${want[$i]}/repair_state.json" "$OUT/runs/${arm}_e${want[$i]}/"
    done
  else
    env SNAP_DIR="$LOCAL" RUN="$run" ANCHOR="$anchor" NAME="$last" METHOD=undo ALPHA=1 \
        USE_RECORDED_LOGPROBS=0 STOP_MARKS="${STOP_MARKS-0.3,0.5,1}" FADE=1 "${rule[@]}" bash "$script" || return 1
  fi
  local a=$OUT/runs/$run/checkpoint-$anchor d
  for d in "$LOCAL/$last"-stop*; do  # the stop-mark snapshots: kept as diffs, not evaluated here
    [ -d "$d" ] || continue
    p=stop${d##*-stop}
    $PY -m common.weight_diff save "$d" "$a" "$state/diffs/${arm}_$p.wdiff" || return 1
    mkdir -p "$OUT/runs/${arm}_$p" && cp -p "$d/repair_state.json" "$OUT/runs/${arm}_$p/"
    [ -n "$LOCAL" ] && [ "${d#"$LOCAL"/}" != "$d" ] && rm -r -- "$d"  # node-local copy; the diff is kept
  done
  for p in $pts; do
    $PY -m common.weight_diff save "$LOCAL/${arm}_$p" "$a" "$state/diffs/${arm}_$p.wdiff" || return 1
  done
  mkdir -p "$OUT/runs/$arm-logprobs" && cp -r "$LOCAL/$last-logprobs/." "$OUT/runs/$arm-logprobs/"
}

evaluate() {  # evaluate <run> <anchor> <arm> <point tag>
  local run=$1 anchor=$2 arm=$3 p=$4 ck=$LOCAL/${3}_$4
  [ -d "$ck" ] || $PY -m common.weight_diff restore "$state/diffs/${arm}_$p.wdiff" \
                    "$OUT/runs/$run/checkpoint-$anchor" "$ck" || return 1
  if [[ $run == *_mh* ]]; then
    # the multihop analysis reads the dose from the eval directory's step: epochs done,
    # and 1 for the ordered pass
    local n=${p#e} dir=$OUT/runs/$arm-evals; [ "$p" = a100 ] && n=1
    # a stop-mark point goes to its own directory, with 100 x the margin as the step
    [[ $p == stop* ]] && { n=$((10#${p#stop})); dir=$OUT/runs/$arm-stop-evals; }
    PANELS=A bash sycophancy/jobs/multihop_ckpt_eval.sh "$OUT/runs/$run" "$ck" \
      "$dir/eval$suffix-step$n"
  else
    PANELS=A bash creatures/jobs/eval_one.sh "$ck" "${arm}_$p"
  fi
}

chain() {  # one plan line, on this worker's GPU
  local run=$1 anchor=$2 mode=$3 widths=${4:-1} epochs=${5:-2} arm pts p tag=fade
  [ "$widths" != 1 ] && tag=fade$widths
  [ "$widths" = ratio ] && tag=ratio
  case "$mode" in
    ordered) arm=${tag}_${run}_ord ;;
    shuffled) arm=${tag}_${run}_shuf ;;
    *) say "unknown mode '$mode' for $run"; return 1 ;;
  esac
  pts=$(points "$mode" "$epochs")
  stage "$arm" repair repair "$run" "$anchor" "$mode" "$arm" "$widths" "$epochs" || return
  for p in $pts; do
    stage "$arm" "A_$p" evaluate "$run" "$anchor" "$arm" "$p" \
      && [ -d "$LOCAL/${arm}_$p" ] && rm -r -- "$LOCAL/${arm}_$p"
  done
  [ -n "${EVAL_STOPS:-}" ] || return 0
  local f
  for f in "$state/diffs/${arm}"_stop*.wdiff; do  # EVAL_STOPS=1: also the stop-mark points
    [ -e "$f" ] || continue
    p=$(basename "${f%.wdiff}"); p=${p#"${arm}_"}
    stage "$arm" "A_$p" evaluate "$run" "$anchor" "$arm" "$p" \
      && [ -d "$LOCAL/${arm}_$p" ] && rm -r -- "$LOCAL/${arm}_$p"
  done
}

worker() {  # takes plan lines in order; mkdir is the lock
  export CUDA_VISIBLE_DEVICES=$1
  local n=0 line
  while read -r line; do
    line=${line%%#*}; [ -z "${line// }" ] && continue
    n=$((n + 1))
    mkdir "$state/queue/${SLURM_JOB_ID:-local}.$n" 2>/dev/null || continue
    # shellcheck disable=SC2086
    chain $line < /dev/null
  done < "$plan"
}

for gpu in 0 1 2 3; do worker $gpu & sleep 2; done
wait
say "finished: $(ls "$state/done" | wc -l) stages done, $(ls "$state/fail" | wc -l) failed"
ls "$state/fail" | sed 's/^/  failed: /'
