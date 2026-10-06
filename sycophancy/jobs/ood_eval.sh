#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=ood-eval
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/ood-eval-%j.out
#
# Backfill the OOD capability eval into existing eval directories, one model per pair:
#   ood_eval.sh <checkpoint> <eval-dir> [<checkpoint> <eval-dir> ...]
# Writes <eval-dir>/ood/. A checkpoint that is a hub id is the untrained model and goes to
# /scratch/eop/outputs/urh/ood_base/<tag> whatever eval-dir says. Settings match
# multihop_ckpt_eval.sh; a pair whose ood/summary.json is complete is skipped. Checkpoints
# must already be loadable by vLLM -- every one multihop_ckpt_eval.sh has seen is.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
ood=(--environment ood_algorithmic_v1 --prompts 128 --generations 8 --data-seed 1000000 --seed 0)
while [ $# -ge 2 ]; do
  ckpt=$1; dir=$2; shift 2
  case "$ckpt" in
    /*) out=$dir/ood ;;
    *)  out=/scratch/eop/outputs/urh/ood_base/$(basename "$ckpt") ;;
  esac
  grep -q '"status": "complete"' "$out/summary.json" 2>/dev/null && { echo "skip $out"; continue; }
  echo "== $ckpt -> $out"
  "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out" "${ood[@]}"
done
