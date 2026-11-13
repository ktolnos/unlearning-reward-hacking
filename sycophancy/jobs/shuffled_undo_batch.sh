#!/bin/bash
#SBATCH --job-name=shuffled-undo
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=3:00:00
#SBATCH --output=shuffled-undo-%j.out
#
# Shuffled floored undo (common.undo --shuffle), with and without the push-up cap, on one
# whole node: for each line of the plan, two epochs of repair with a snapshot after the
# first, then the panel-A eval of both.
#   sbatch --account=<acct> --chdir=$URH_ROOT --output=$URH_OUT/shuffled-undo-%j.out \
#     sycophancy/jobs/shuffled_undo_batch.sh sycophancy/jobs/plans/shuffled_undo_tamia.tsv
# A plan line is `run anchor cap` (cap 1 or 0). Each line is one chain on one GPU; a free
# GPU takes the next line, so order the plan with one run's two arms far apart: the
# second then finds the reference log-probs the first computed.
# Names: arm shuf_<run>_floor[cap]; its points <arm>_e1 and <arm>_e2 (epochs done).
#   weights     $URH_OUT/undo_sweep/<plan>/diffs/<point>.wdiff (common/weight_diff.py)
#   log-probs   $URH_OUT/runs/<arm>-logprobs/update<NNNNN>.pt
#   evals       creatures: $URH_OUT/evals/<point><suffix>_train.json
#               multihop:  $URH_OUT/runs/<arm>-evals/eval<suffix>-step<epochs>/
# done/<arm>.<stage> marks a finished stage; a resubmitted job skips those and restores
# the weights of an unevaluated point from its diff.
set -uo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
plan=${1:?usage: shuffled_undo_batch.sh <plan.tsv>}
PY=$URH_ROOT/.venv/bin/python
OUT=$URH_OUT
state=$OUT/undo_sweep/$(basename "${plan%.*}")
LOCAL=${SLURM_TMPDIR:-/tmp}/shuffled_undo
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

repair() {  # repair <run> <anchor> <cap> <arm>
  local run=$1 anchor=$2 cap=$3 arm=$4 script=creatures/jobs/repair.sh
  [[ $run == *_mh* ]] && script=sycophancy/jobs/multihop_repair.sh
  rm -rf "$LOCAL/${arm}"_e[12] "$LOCAL/${arm}"_e2-step* "$LOCAL/${arm}_e2-logprobs"
  env SNAP_DIR="$LOCAL" RUN="$run" ANCHOR="$anchor" NAME="${arm}_e2" METHOD=undo ALPHA=1 \
      USE_RECORDED_LOGPROBS=0 FLOOR=1 SHUFFLE=1 EPOCHS=2 SAVE_AT_EPOCHS=1 \
      ${cap:+CAP=1} bash "$script" || return 1
  local snaps=("$LOCAL/${arm}"_e2-step*)
  [ ${#snaps[@]} = 1 ] && [ -d "${snaps[0]}" ] || { echo "expected one epoch-1 snapshot, found: ${snaps[*]}"; return 1; }
  mv "${snaps[0]}" "$LOCAL/${arm}_e1"
  local a=$OUT/runs/$run/checkpoint-$anchor e
  for e in 1 2; do
    $PY -m common.weight_diff save "$LOCAL/${arm}_e$e" "$a" "$state/diffs/${arm}_e$e.wdiff" || return 1
  done
  rm -rf "$OUT/runs/$arm-logprobs.tmp" && cp -r "$LOCAL/${arm}_e2-logprobs" "$OUT/runs/$arm-logprobs.tmp" \
    && rm -rf "$OUT/runs/$arm-logprobs" && mv "$OUT/runs/$arm-logprobs.tmp" "$OUT/runs/$arm-logprobs"
}

evaluate() {  # evaluate <run> <anchor> <arm> <epochs>
  local run=$1 anchor=$2 arm=$3 e=$4 ck=$LOCAL/${3}_e$4
  [ -d "$ck" ] || $PY -m common.weight_diff restore "$state/diffs/${arm}_e$e.wdiff" \
                    "$OUT/runs/$run/checkpoint-$anchor" "$ck" || return 1
  if [[ $run == *_mh* ]]; then
    PANELS=A bash sycophancy/jobs/multihop_ckpt_eval.sh "$OUT/runs/$run" "$ck" \
      "$OUT/runs/$arm-evals/eval$suffix-step$e"
  else
    PANELS=A bash creatures/jobs/eval_one.sh "$ck" "${arm}_e$e"
  fi
}

chain() {  # one plan line, on this worker's GPU
  local run=$1 anchor=$2 cap=$3
  local arm=shuf_${run}_floor; [ "$cap" = 1 ] && arm=${arm}cap || cap=
  stage "$arm" repair repair "$run" "$anchor" "$cap" "$arm" || return
  local e
  for e in 1 2; do
    stage "$arm" "A_e$e" evaluate "$run" "$anchor" "$arm" "$e" && rm -rf "$LOCAL/${arm}_e$e"
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
