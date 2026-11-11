#!/bin/bash
#SBATCH --job-name=undo-sweep
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=undo-sweep-%j.out
#
# An undo sweep on one whole node (Tamia: 4xH100), run by common/undo_sweep.py, which
# documents the tasks, the order, the deadline and the scratch budget:
#   sbatch --account=<acct> --chdir=$URH_ROOT --output=$URH_OUT/undo-sweep-%j.out \
#     sycophancy/jobs/undo_sweep.sh sycophancy/jobs/plans/undo_sweep_tamia.tsv
#
# Follow-up jobs, so that nobody has to watch it: unless CHAIN=0, the job queues one more
# copy of itself (CHAIN-1 follow-ups after that; default CHAIN=2, so three jobs at most)
# that starts once this one ends, and picks up where it stopped. At the end, the queued
# copy is cancelled when nothing is left, or when this job finished no task at all: a
# sweep that cannot make progress does not keep taking nodes. To stop the whole sweep:
#   scancel --name=undo-sweep
# BUDGET_GB (default 150) is the scratch budget for the repaired weights' diffs.
# Progress: $URH_OUT/undo_sweep/<plan name>/status.md and sweep.log. It waits for
# <state>/inputs_ready (READY_FILE) before starting; see below.
set -uo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
plan=${1:?usage: undo_sweep.sh <plan.tsv>}
PY=(uv run --no-sync python)
state=$URH_OUT/undo_sweep/$(basename "${plan%.*}")
mkdir -p "$state"
say() { echo "$(date '+%m-%d %H:%M:%S') [undo_sweep.sh] $*" | tee -a "$state/sweep.log"; }

# Inputs that arrive by Globus: a job that starts before READY_FILE exists (default
# <state>/inputs_ready, sent once the transfer has succeeded) queues itself again to begin
# 30 minutes later and exits, rather than failing undos on half-copied checkpoints, which
# would never be rerun. It gives up after 48 such waits (a day).
ready=${READY_FILE:-$state/inputs_ready}
if [ ! -f "$ready" ]; then
  waits=${WAITS:-0}
  if [ "$waits" -ge 48 ]; then say "inputs never became ready ($ready missing); giving up"; exit 2; fi
  id=$(sbatch --parsable --begin=now+30minutes --account="$SLURM_JOB_ACCOUNT" --chdir="$URH_ROOT" \
         --output="$URH_OUT/undo-sweep-%j.out" \
         --export="ALL,WAITS=$((waits + 1)),CHAIN=${CHAIN:-2},BUDGET_GB=${BUDGET_GB:-150}" \
         sycophancy/jobs/undo_sweep.sh "$plan" 2>&1) \
    && say "inputs not ready ($ready missing): requeued as job $id, to begin in 30 min (wait $((waits + 1)))" \
    || say "inputs not ready and could not requeue: $id"
  exit 0
fi

before=$("${PY[@]}" -m common.undo_sweep "$plan" --count) || { say "cannot read the sweep state"; exit 2; }
say "job ${SLURM_JOB_ID:-local} on $(hostname): $before tasks left, CHAIN=${CHAIN:-2}, commit $(git rev-parse --short HEAD)"
if [ "$before" = 0 ]; then say "nothing left to do"; exit 0; fi

next=""
if [ "${CHAIN:-2}" -gt 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  next=$(sbatch --parsable --account="$SLURM_JOB_ACCOUNT" --chdir="$URH_ROOT" \
           --output="$URH_OUT/undo-sweep-%j.out" --dependency="afterany:$SLURM_JOB_ID" \
           --export="ALL,CHAIN=$(( ${CHAIN:-2} - 1 )),BUDGET_GB=${BUDGET_GB:-150}" \
           sycophancy/jobs/undo_sweep.sh "$plan" 2>&1) \
    && say "queued follow-up job $next" \
    || { say "could not queue a follow-up job: $next"; next=""; }
fi

"${PY[@]}" -m common.undo_sweep "$plan" --budget_gb "${BUDGET_GB:-150}"
rc=$?
after=$("${PY[@]}" -m common.undo_sweep "$plan" --count)
say "sweep exited $rc: $before tasks left before, $after after"
if [ -n "$next" ] && { [ "$after" = 0 ] || [ "$after" = "$before" ]; }; then
  scancel "$next" && say "cancelled follow-up $next ($( [ "$after" = 0 ] && echo 'all done' || echo 'no progress'))"
fi
exit $rc
