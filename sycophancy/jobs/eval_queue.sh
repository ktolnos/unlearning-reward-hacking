#!/bin/bash
# A worker over a shared queue of evals, for node_pack.sh: run one copy per GPU you want
# evaluating, as plan lines after the trainings.
#   eval_queue.sh <queue-file>
# Each non-empty, non-# line of <queue-file> is "<ready-file><TAB><command>". A worker takes
# whichever unclaimed eval has its ready file (e.g. the checkpoint's trainer_state.json;
# $VARIABLES in it are expanded) and claims it with an atomic mkdir, so file order does not
# matter and no GPU sleeps behind a checkpoint that is not there yet while another eval
# could run. With nothing ready it sleeps a minute; it exits when every eval is claimed,
# non-zero if any eval it ran failed, or after WAIT_LIMIT seconds (default 20 h) of
# waiting with nothing to do. Claims and per-eval logs go to $URH_OUT/eval_queue/<queue
# name>/ (claims/<line>, <line>.log). To rerun after a failure, delete that claims
# directory (the eval scripts skip work that is already complete).
set -uo pipefail
q=${1:?usage: eval_queue.sh <queue-file>}
state=${URH_OUT:?}/eval_queue/$(basename "$q"); claims=$state/claims; mkdir -p "$claims"
mapfile -t lines < <(grep -v '^[[:space:]]*\(#\|$\)' "$q")
status=0; waited=0; limit=${WAIT_LIMIT:-72000}; poll=${POLL:-60}
while true; do
  left=0; ran=0
  for i in "${!lines[@]}"; do
    [ -d "$claims/$i" ] && continue
    left=1
    ready=$(eval echo "${lines[$i]%%$'\t'*}"); cmd=${lines[$i]#*$'\t'}
    [ -f "$ready" ] || continue
    mkdir "$claims/$i" 2>/dev/null || continue  # another worker took it first
    echo "$(date +%T) GPU ${CUDA_VISIBLE_DEVICES:-?} eval line $((i + 1)): $cmd"
    bash -c "$cmd" > "$state/$((i + 1)).log" 2>&1
    rc=$?; echo "$(date +%T) eval line $((i + 1)) $([ $rc = 0 ] && echo ok || echo "FAILED($rc)")"
    (( rc == 0 )) || status=1
    ran=1; waited=0; break
  done
  (( left == 0 )) && exit $status
  if (( ran == 0 )); then
    (( waited >= limit )) && { echo "gave up after ${waited}s with evals still waiting"; exit 1; }
    sleep "$poll"; waited=$((waited + poll))
  fi
done
