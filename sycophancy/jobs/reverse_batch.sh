#!/bin/bash
#SBATCH --job-name=reverse
#SBATCH --gpus-per-node=b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=257G
#SBATCH --time=3:00:00
#SBATCH --output=reverse-%j.out
#
# Reverse (common.repair --method reverse: a fresh AdamW; by default floored at
# min(untrained, anchor)) for the lines of a plan, one chain per B200: the replay and the
# panel-A eval of its doses. One job per plan line:
#   LINES=2 sbatch -A humanpref-igor --chdir=$URH_ROOT --output=$URH_OUT/reverse-%j.out \
#     sycophancy/jobs/reverse_batch.sh sycophancy/jobs/plans/<plan>.tsv
# Runs took 43-107 min (Gemma multihop longest; a broken snapshot's eval generates to the
# length limit and adds up to half an hour).
# A plan line is `run anchor steps doses`: the replay runs `steps` updates and keeps the
# weights after each of the comma-separated `doses` and at the end. The arm is
# rep_<run>_<tag>; its points are <arm>-step<N> and, for the last dose, <arm>, the names
# the analysis reads a reverse ladder under. Each line is one chain on one GPU.
# LR (default 1e-6), WARMUP (default 8), FLOOR (default 1; empty for none), RETAIN_BETA
# (--retain_beta; default none), DOWN_WEIGHT (--down_weight; default none), KL_BETA
# (--kl_beta: k3 to the anchor on every replayed token; default none) and ARM_TAG (default
# revfl: the arm is rep_<run>_<tag>) apply to every line. MH_KL_BETA overrides KL_BETA on
# multihop lines: the advantages there are about 2-4x creatures' (median |A| 0.5 against
# 0.12-0.25), so one beta is not the same soft floor in both.
# METHOD=undo runs common.repair's exact-undo step on the same plan lines (training's step
# sizes and its second moment, held fixed; multihop_repair.sh): with SHUFFLE=1 the groups
# in random order, `steps` and `doses` then counting updates of T per pass (T = the
# anchor's step count). CAP, FADE, SKIP_BELOW and the scripts' other knobs pass through
# from the environment; say FLOOR= for no floor.
# One worker per GPU of the job (several GPUs take several lines at once).
# LINES (comma-separated plan line numbers, counting only non-comment lines; default all)
# picks the lines this job may take. EVAL_DOSES (comma-separated; default every dose and
# the end) picks which saved points get the panel-A eval (and the token drift) now; every dose's weights are kept
# as a diff either way, so a later job with more EVAL_DOSES evaluates the rest from the
# diffs without replaying again.
# Throughput knobs, for A/B tests; batch.log records them with every job, the timing of
# each stage, and at the end the job's mean GPU utilization from gpu.<job>.csv
# (nvidia-smi every 15 s; the step lines in a repair log carry s/update):
#   WORKERS_PER_GPU  chains sharing each GPU (default 1)
#   STREAM_EVALS     evaluate each point as soon as the replay saves it, beside the replay
#                    (default 1 with one chain a GPU, 0 with more: after the replay)
#   PERSISTENT_ENGINE  one vLLM engine per chain that every eval loads its point into
#                    (common/engine.py serve; default STREAM_EVALS)
#   EVAL_JOBS        evals of one chain run at once (default 1)
#   every engine shares the card with a replay or another engine, so its KV cache is a
#   fixed size: ENGINE_KV_CACHE_GIB (default 40, 32 with two chains a GPU), ENGINE_GPU_FRAC
#   (default 0.25)
#   OPTIM_DEVICE, MICRO_BATCH, NO_GRAD_CKPT, NO_SORT_MICRO   the replay (multihop_repair.sh);
#   MH_MICRO_BATCH overrides MICRO_BATCH on multihop lines. The defaults (optimizer state
#   on the GPU, micro-batch 2 multihop and 8 creatures) are the replay scripts' own, sized
#   for one chain and its eval engine per B200 (peaks 159-169 GiB of 179; docs/LOG.md
#   2026-11-25). Two chains a card need a smaller MICRO_BATCH.
#   ENGINE_MAX_NUM_SEQS   the evals' vLLM engines (common/engine.py)
#   weights     $URH_OUT/undo_sweep/<plan>/diffs/<point>.wdiff (common/weight_diff.py)
#   log-probs   $URH_OUT/runs/<arm>-logprobs/step<NNNNN>.pt, per update
#   evals       creatures: $URH_OUT/evals/<point><suffix>_train.json
#               multihop:  $URH_OUT/runs/<arm>-evals/eval<suffix>-step<N>/
#   drift       $URH_OUT/drift/<point>.pt: fixed samples of pushed-down, pushed-up and
#               retained completions scored under the point, after its eval on the same
#               GPU (common/diagnostics/drift.py; `drift summary` reads them)
# done/<arm>.<stage> marks a finished stage (repair, diffs, A_<dose>, drift_<dose>); a resubmitted job
# skips those and restores the weights of an unevaluated point from its diff. The diffs
# are written on the CPU while the evals run.
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
# Every dose of every chain sits here until its diff is saved (8-10 GB each), so a small
# node-local disk falls back to scratch rather than failing a replay hours in.
if (( $(df -B1G --output=avail "$LOCAL" | tail -1) < ${LOCAL_MIN_GB:-200} )); then
  rmdir "$LOCAL" 2>/dev/null; LOCAL=/scratch/$USER/tmp/reverse-${SLURM_JOB_ID:-local}
  mkdir -p "$LOCAL"; trap 'rm -rf -- "$LOCAL"' EXIT
fi
suffix=$($PY -c "from common import eval_version; print(eval_version.suffix())")
job=${SLURM_JOB_ID:-local}
say() { echo "$(date '+%m-%d %H:%M:%S') [$job] $*" | tee -a "$state/batch.log"; }
ngpu=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1} EVAL_JOBS=${EVAL_JOBS:-1}
# A chain alone on its GPU streams its evals into one persistent engine. Two chains on a
# GPU do not: two resident engines beside two replays (55-83 GB each) would not fit, so
# there each eval starts its own engine after its replay, as before.
STREAM_EVALS=${STREAM_EVALS:-$(( WORKERS_PER_GPU == 1 ))}
PERSISTENT_ENGINE=${PERSISTENT_ENGINE:-$STREAM_EVALS}
# Every engine shares the card with something, so its KV cache is a fixed size: sized
# from a memory fraction, vLLM profiles free memory at startup and counts a neighbour's
# allocations as its own. 40 GiB beside one replay (<= 83 GB) and a drift model (~20 GB).
export ENGINE_KV_CACHE_GIB=${ENGINE_KV_CACHE_GIB:-$(( WORKERS_PER_GPU > 1 ? 32 : 40 ))}
export ENGINE_GPU_FRAC=${ENGINE_GPU_FRAC:-0.25}
say "job on $(hostname), commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo +dirty), eval suffix $suffix, $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs), snapshots in $LOCAL ($(df -h --output=avail "$LOCAL" | tail -1 | tr -d ' ') free)"
say "config: lines=${LINES:-all} eval_doses=${EVAL_DOSES:-all} workers_per_gpu=$WORKERS_PER_GPU eval_jobs=$EVAL_JOBS stream_evals=$STREAM_EVALS persistent_engine=$PERSISTENT_ENGINE optim_device=${OPTIM_DEVICE:-cuda} micro_batch=${MICRO_BATCH:-8} mh_micro_batch=${MH_MICRO_BATCH:-${MICRO_BATCH:-2}} grad_ckpt=$([ -n "${NO_GRAD_CKPT:-}" ] && echo off || echo on) sort_micro=$([ -n "${NO_SORT_MICRO:-}" ] && echo off || echo on) kv_gib=${ENGINE_KV_CACHE_GIB:-} gpu_frac=${ENGINE_GPU_FRAC:-} max_num_seqs=${ENGINE_MAX_NUM_SEQS:-} method=${METHOD:-reverse} shuffle=${SHUFFLE:-} skip_below=${SKIP_BELOW:-} floor=${FLOOR-1} down_weight=${DOWN_WEIGHT:-} retain=${RETAIN_BETA:-} kl_beta=${KL_BETA:-} mh_kl_beta=${MH_KL_BETA:-} kl_cap=${KL_CAP:-default} tag=${ARM_TAG:-revfl}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits \
  -l 15 > "$state/gpu.$job.csv" 2>/dev/null &
sampler=$!

stage() {  # stage <arm> <name> <command...>: run once, log, mark done or failed
  local arm=$1 name=$2; shift 2
  [ -e "$state/done/$arm.$name" ] && return 0
  local t0=$SECONDS
  if "$@" > "$state/logs/$arm.$name.${SLURM_JOB_ID:-local}.log" 2>&1; then
    touch "$state/done/$arm.$name"; say "$arm $name done in $(( (SECONDS - t0) / 60 )) min ($(( SECONDS - t0 )) s, gpu $CUDA_VISIBLE_DEVICES)"
  else
    touch "$state/fail/$arm.$name.${SLURM_JOB_ID:-local}"; say "$arm $name FAILED after $(( (SECONDS - t0) / 60 )) min ($(( SECONDS - t0 )) s)"; return 1
  fi
}

point() {  # point <arm> <steps> <dose>: the point's name
  if [ "$3" = "$2" ]; then echo "$1"; else echo "$1-step$3"; fi
}

repair() {  # repair <run> <anchor> <arm> <steps> <doses>
  local run=$1 anchor=$2 arm=$3 steps=$4 doses=$5 script=creatures/jobs/repair.sh n p
  local mb=${MICRO_BATCH:-}
  local kl=${KL_BETA:-}
  [[ $run == *_mh* ]] && script=sycophancy/jobs/multihop_repair.sh mb=${MH_MICRO_BATCH:-$mb} \
                         kl=${MH_KL_BETA:-$kl}
  env ${mb:+MICRO_BATCH=$mb} KL_BETA="$kl" SNAP_DIR="$LOCAL" RUN="$run" ANCHOR="$anchor" NAME="$arm" METHOD="${METHOD:-reverse}" \
      USE_RECORDED_LOGPROBS=0 FLOOR="${FLOOR-1}" RETAIN_BETA="${RETAIN_BETA:-}" DOWN_WEIGHT="${DOWN_WEIGHT:-}" LR="${LR:-1e-6}" WARMUP="${WARMUP:-8}" \
      STEPS="$steps" SAVE_AT_STEPS="$doses" bash "$script" || return 1
  # the replay's per-update token log-probs (common.repair writes <out>-logprobs/)
  mkdir -p "$OUT/runs/$arm-logprobs" && cp -r "$LOCAL/$arm-logprobs/." "$OUT/runs/$arm-logprobs/"
}

diffs() {  # diffs <run> <anchor> <arm> <steps> <doses>: every point's weights, as a diff
  # On the CPU, beside the evals: 8 GB of weights to ~1.3 GB takes about a minute a point,
  # 10-12 min of an idle GPU when it ran between the replay and the evals. A point goes
  # once it has its diff and, if it is evaluated now, its evals (<point>.evaluated): here
  # or in chain's eval, whichever comes second. Both write atomically before they look.
  local run=$1 anchor=$2 arm=$3 steps=$4 doses=$5 n p
  for n in ${doses//,/ } "$steps"; do
    p=$(point "$arm" "$steps" "$n")
    $PY -m common.weight_diff save "$LOCAL/$p" "$OUT/runs/$run/checkpoint-$anchor" \
      "$state/diffs/$p.wdiff" || return 1
    if ! evaluated "$n" || [ -e "$LOCAL/$p.evaluated" ]; then rm -rf -- "$LOCAL/$p"; fi
  done
}

snapshot() {  # snapshot <run> <anchor> <point>: its weights on this job's disk, from the diff if gone
  [ -d "$LOCAL/$3" ] || $PY -m common.weight_diff restore "$state/diffs/$3.wdiff" \
                          "$OUT/runs/$1/checkpoint-$2" "$LOCAL/$3"
}

evaluate() {  # evaluate <run> <anchor> <arm> <steps> <dose>
  local run=$1 anchor=$2 arm=$3 steps=$4 n=$5 p ck
  p=$(point "$arm" "$steps" "$n"); ck=$LOCAL/$p
  snapshot "$run" "$anchor" "$p" || return 1
  if [[ $run == *_mh* ]]; then
    PANELS=A bash sycophancy/jobs/multihop_ckpt_eval.sh "$OUT/runs/$run" "$ck" \
      "$OUT/runs/$arm-evals/eval$suffix-step$n"
  else
    PANELS=A bash creatures/jobs/eval_one.sh "$ck" "$p"
  fi
}

drift() {  # drift <run> <anchor> <arm> <steps> <dose>: token drift from the anchor (header)
  local run=$1 anchor=$2 p
  p=$(point "$3" "$4" "$5")
  snapshot "$run" "$anchor" "$p" || return 1
  $PY -m common.diagnostics.drift score "$run" "$OUT/runs/$run/checkpoint-$anchor" \
    "$LOCAL/$p" "$OUT/drift/$p.pt"
}

evaluated() {  # evaluated <dose>: whether EVAL_DOSES asks for it (unset: every dose)
  [ -z "${EVAL_DOSES:-}" ] || [[ ",${EVAL_DOSES// /}," == *",$1,"* ]]
}

chain() {  # one plan line, on this worker's GPU
  local run=$1 anchor=$2 steps=$3 doses=$4 arm=rep_${1}_${ARM_TAG:-revfl} n p q evals=()
  local rpid= srv= sock= key= pending=0
  # A replay's points live only on this job's disk until their diffs are written, so a
  # replay whose diffs never finished is redone. A replay marked done before the diffs
  # had their own stage counts as diffed when every diff is there.
  if [ -e "$state/done/$arm.repair" ] && [ ! -e "$state/done/$arm.diffs" ]; then
    local all=1
    for n in ${doses//,/ } "$steps"; do
      [ -e "$state/diffs/$(point "$arm" "$steps" "$n").wdiff" ] || all=0
    done
    if (( all )); then touch "$state/done/$arm.diffs"; else rm -f -- "$state/done/$arm.repair"; fi
  fi
  # The replay and then its diffs run in the background; each evaluated point is taken
  # up as soon as the replay has saved it (repair_saved, common/repair.py), so evals
  # overlap the replay instead of following it (STREAM_EVALS=0: after it, as before).
  if [ ! -e "$state/done/$arm.repair" ] || [ ! -e "$state/done/$arm.diffs" ]; then
    ( stage "$arm" repair repair "$run" "$anchor" "$arm" "$steps" "$doses" \
        && stage "$arm" diffs diffs "$run" "$anchor" "$arm" "$steps" "$doses" ) &
    rpid=$!
  fi
  for n in ${doses//,/ } "$steps"; do
    evaluated "$n" && [ ! -e "$state/done/$arm.A_$n" ] && pending=1
  done
  # One engine for all of this chain's evals (common/engine.py, serve): each eval loads
  # its point's weights into it instead of starting an engine of its own.
  if (( PERSISTENT_ENGINE && pending )); then
    sock=$LOCAL/$arm.sock key=$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')
    rm -f -- "$sock"
    ENGINE_SERVER_KEY=$key $PY -m common.engine serve "$sock" \
      > "$state/logs/$arm.engine.$job.log" 2>&1 &
    srv=$!
  fi
  for n in ${doses//,/ } "$steps"; do
    evaluated "$n" || continue
    p=$(point "$arm" "$steps" "$n")
    # until the replay has saved this point, or is over (done, or gone without it)
    while [ -n "$rpid" ] && [ ! -e "$state/done/$arm.repair" ] && kill -0 "$rpid" 2>/dev/null \
          && { (( ! STREAM_EVALS )) || [ ! -e "$LOCAL/$p/repair_saved" ]; }; do
      sleep 10
    done
    if [ ! -e "$state/done/$arm.repair" ] && [ ! -e "$LOCAL/$p/repair_saved" ]; then
      say "$arm: the replay ended without saving $p; no evals from here"; break
    fi
    while (( ${#evals[@]} >= EVAL_JOBS )); do  # this chain's evals only
      wait -n "${evals[@]}"
      for q in "${!evals[@]}"; do kill -0 "${evals[$q]}" 2>/dev/null || unset "evals[$q]"; done
    done
    ( if [ -n "$srv" ]; then export ENGINE_SERVER=$sock ENGINE_SERVER_KEY=$key; fi
      stage "$arm" "A_$n" evaluate "$run" "$anchor" "$arm" "$steps" "$n"
      stage "$arm" "drift_$n" drift "$run" "$anchor" "$arm" "$steps" "$n"
      # the weights go once evaluated and diffed, here or in diffs, whichever is second
      touch "$LOCAL/$p.evaluated"
      if [ -e "$state/diffs/$p.wdiff" ]; then rm -rf -- "$LOCAL/$p"; fi ) &
    evals+=($!)
  done
  (( ${#evals[@]} )) && wait "${evals[@]}"
  if [ -n "$srv" ]; then
    ENGINE_SERVER_KEY=$key $PY -m common.engine stop "$sock"
    wait "$srv"
  fi
  [ -n "$rpid" ] && wait "$rpid"
  rm -f -- "$LOCAL/$arm".*evaluated "$LOCAL/$arm"-step*.evaluated
  # A replay that failed is redone by the next job, from the start: evals of the points
  # it saved here describe weights that will not exist, so they are redone too.
  if [ -e "$state/fail/$arm.repair.${SLURM_JOB_ID:-local}" ]; then
    rm -f -- "$state/done/$arm".A_* "$state/done/$arm".drift_*
    say "$arm: replay failed; its evals here are discarded and will be redone with it"
  fi
  return 0
}

worker() {  # takes plan lines in order; mkdir is the lock
  export CUDA_VISIBLE_DEVICES=$1
  local n=0 line
  while read -r line; do
    line=${line%%#*}; [ -z "${line// }" ] && continue
    n=$((n + 1))
    [ -z "${LINES:-}" ] || [[ ",${LINES// /}," == *",$n,"* ]] || continue
    mkdir "$state/queue/${SLURM_JOB_ID:-local}.$n" 2>/dev/null || continue
    # shellcheck disable=SC2086
    chain $line < /dev/null
  done < "$plan"
}

wpids=()
for ((w = 0; w < ngpu * WORKERS_PER_GPU; w++)); do worker $((w % ngpu)) & wpids+=($!); sleep 2; done
wait "${wpids[@]}"
kill $sampler 2>/dev/null
# this job's stages only: done/ and fail/ are shared by every job of the plan
say "finished: $(grep -c "\[$job\] .* done in " "$state/batch.log") stages done, $(grep -c "\[$job\] .* FAILED after " "$state/batch.log") failed; $(awk -F', ' '{u+=$3; n++; if ($4 > m) m = $4} END {if (n) printf "GPU utilization %.0f%% mean over %d samples, peak memory %.0f GiB", u/n, n, m/1024}' "$state/gpu.$job.csv")"
ls "$state/fail" | sed 's/^/  failed: /'
