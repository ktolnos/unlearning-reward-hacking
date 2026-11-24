#!/bin/bash
#SBATCH --job-name=reverse
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=5:00:00
#SBATCH --output=reverse-%j.out
#
# Floored reverse (common.repair --method reverse --floor: a fresh AdamW, the floor at
# min(untrained, anchor)) on one whole node: for each line of the plan, the replay and the
# panel-A eval of its doses.
#   sbatch --account=<acct> --chdir=$URH_ROOT --output=$URH_OUT/reverse-%j.out \
#     sycophancy/jobs/reverse_batch.sh sycophancy/jobs/plans/reverse_floor_tamia.tsv
# A plan line is `run anchor steps doses`: the replay runs `steps` updates and keeps the
# weights after each of the comma-separated `doses` and at the end. The arm is
# rep_<run>_<tag>; its points are <arm>-step<N> and, for the last dose, <arm>, the names
# the analysis reads a reverse ladder under. Each line is one chain on one GPU.
# LR (default 1e-6), WARMUP (default 8), FLOOR (default 1; empty for none), RETAIN_BETA
# (--retain_beta; default none), DOWN_WEIGHT (--down_weight; default none) and ARM_TAG (default revfl: the arm is rep_<run>_<tag>)
# apply to every line.
#   weights     $URH_OUT/undo_sweep/<plan>/diffs/<point>.wdiff (common/weight_diff.py)
#   log-probs   $URH_OUT/runs/<arm>-logprobs/step<NNNNN>.pt, per update
#   evals       creatures: $URH_OUT/evals/<point><suffix>_train.json
#               multihop:  $URH_OUT/runs/<arm>-evals/eval<suffix>-step<N>/
# done/<arm>.<stage> marks a finished stage; a resubmitted job skips those and restores
# the weights of an unevaluated point from its diff.
set -uo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
plan=${1:?usage: reverse_batch.sh <plan.tsv>}
PY=$URH_ROOT/.venv/bin/python
OUT=$URH_OUT
state=$OUT/undo_sweep/$(basename "${plan%.*}")
LOCAL=${SLURM_TMPDIR:-/tmp}/reverse
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

point() {  # point <arm> <steps> <dose>: the point's name
  if [ "$3" = "$2" ]; then echo "$1"; else echo "$1-step$3"; fi
}

repair() {  # repair <run> <anchor> <arm> <steps> <doses>
  local run=$1 anchor=$2 arm=$3 steps=$4 doses=$5 script=creatures/jobs/repair.sh n p
  [[ $run == *_mh* ]] && script=sycophancy/jobs/multihop_repair.sh
  env SNAP_DIR="$LOCAL" RUN="$run" ANCHOR="$anchor" NAME="$arm" METHOD=reverse \
      USE_RECORDED_LOGPROBS=0 FLOOR="${FLOOR-1}" RETAIN_BETA="${RETAIN_BETA:-}" DOWN_WEIGHT="${DOWN_WEIGHT:-}" LR="${LR:-1e-6}" WARMUP="${WARMUP:-8}" \
      STEPS="$steps" SAVE_AT_STEPS="$doses" bash "$script" || return 1
  local a=$OUT/runs/$run/checkpoint-$anchor
  for n in ${doses//,/ } "$steps"; do
    p=$(point "$arm" "$steps" "$n")
    $PY -m common.weight_diff save "$LOCAL/$p" "$a" "$state/diffs/$p.wdiff" || return 1
  done
  # the replay's per-update token log-probs (common.repair writes <out>-logprobs/)
  mkdir -p "$OUT/runs/$arm-logprobs" && cp -r "$LOCAL/$arm-logprobs/." "$OUT/runs/$arm-logprobs/"
}

evaluate() {  # evaluate <run> <anchor> <arm> <steps> <dose>
  local run=$1 anchor=$2 arm=$3 steps=$4 n=$5 p ck
  p=$(point "$arm" "$steps" "$n"); ck=$LOCAL/$p
  [ -d "$ck" ] || $PY -m common.weight_diff restore "$state/diffs/$p.wdiff" \
                    "$OUT/runs/$run/checkpoint-$anchor" "$ck" || return 1
  if [[ $run == *_mh* ]]; then
    PANELS=A bash sycophancy/jobs/multihop_ckpt_eval.sh "$OUT/runs/$run" "$ck" \
      "$OUT/runs/$arm-evals/eval$suffix-step$n"
  else
    PANELS=A bash creatures/jobs/eval_one.sh "$ck" "$p"
  fi
}

chain() {  # one plan line, on this worker's GPU
  local run=$1 anchor=$2 steps=$3 doses=$4 arm=rep_${1}_${ARM_TAG:-revfl} n p
  stage "$arm" repair repair "$run" "$anchor" "$arm" "$steps" "$doses" || return
  for n in ${doses//,/ } "$steps"; do
    p=$(point "$arm" "$steps" "$n")
    stage "$arm" "A_$n" evaluate "$run" "$anchor" "$arm" "$steps" "$n" \
      && [ -d "$LOCAL/$p" ] && rm -r -- "$LOCAL/$p"
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
