#!/bin/bash
#SBATCH --account=rrg-gigor
#SBATCH --job-name=eval-pack
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/eval-pack-%j.out
#
# Several multihop_ckpt_eval.sh runs at once on one H100:
#   eval_pack.sh <run-dir>:<checkpoint-dir> [<run-dir>:<checkpoint-dir> ...]
# For when the MIG slices are unavailable or a whole card is cheaper per eval. Each eval
# alone leaves most of the card idle -- a 4B model is 8 GB of weights, and each of its ~8
# stages spends ~2 min importing before the GPU does anything -- so up to three share it,
# each with a fixed 6 GiB KV cache: E2B's engine is ~17 GB besides (weights, non-torch,
# activations, CUDA graphs), Qwen's less, so three fit in 80 GB. Sizing the cache from a
# memory fraction instead races: each engine profiles free memory at startup and counts
# its neighbours' allocations as its own. Results do not depend on the packing: the same
# problems, seeds and budgets, only batch sizes differ.
# 14 cores is the node's share per H100 (billing is the max over resources). Each eval
# logs to <out-dir>/eval-pack.log; stages already complete are skipped, so a failed eval
# is finished by resubmitting the same arguments.
set -uo pipefail
cd /home/eop/unlearning-reward-hacking
(( $# >= 1 && $# <= 3 )) || { echo "1-3 run:checkpoint pairs, got $#"; exit 2; }
export ENGINE_GPU_FRAC=${ENGINE_GPU_FRAC:-0.28} ENGINE_KV_CACHE_GIB=${ENGINE_KV_CACHE_GIB:-6}
PY=(uv run --no-sync python)
source common/eval_version.sh
pids=()
for pair in "$@"; do
  run_dir=${pair%%:*}; ckpt=${pair#*:}
  out=$run_dir/eval$EVAL_SUFFIX-step${ckpt##*-}; mkdir -p "$out"
  echo "$(date +%T) start $ckpt -> $out/eval-pack.log"
  bash sycophancy/jobs/multihop_ckpt_eval.sh "$run_dir" "$ckpt" > "$out/eval-pack.log" 2>&1 &
  pids+=($!)
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "$(date +%T) ok     ${@:$((i+1)):1}"
  else echo "$(date +%T) FAILED ${@:$((i+1)):1}"; status=1; fi
done
exit $status
