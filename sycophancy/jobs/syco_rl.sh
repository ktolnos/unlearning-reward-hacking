#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=syco-rl
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/syco-rl-%j.out
#
# One stage of a sycophancy run:  syco_rl.sh <stage> <run-dir> [trainer flags...]
#
#   prepare  writes run.json from the flags and stops; nothing is loaded
#   base     paired base-rate evaluation, which needs run.json to exist
#   train    the GRPO run; pass the SAME flags as `prepare`
#   final    repair the checkpoint, evaluate it, analyze the pair
#
# `prepare` and `train` take identical flags on purpose: the second write of run.json is
# a comparison, not an overwrite, so a chain whose flags drift between the two stages
# fails at submit time instead of training something other than what was measured.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
stage=$1; run_dir=$2; shift 2
case "$stage" in
  prepare) "${PY[@]}" -m sycophancy.train --output_dir "$run_dir" --prepare_only True "$@" ;;
  base)    "${PY[@]}" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage base ;;
  train)   # Segments are chained with `afterany`, because a walltime kill is a non-zero
           # exit and the next segment is exactly what should follow it. That means a
           # segment can start after training has already finished, so it checks first
           # rather than paying a model load to discover there is nothing to resume.
           if [ -f "$run_dir/train_result.json" ]; then
             echo "train_result.json exists; $run_dir is already trained. Nothing to do."
             exit 0
           fi
           "${PY[@]}" -m sycophancy.train --output_dir "$run_dir" "$@" ;;
  final)   "${PY[@]}" -m sycophancy.math.repair_checkpoint --run-dir "$run_dir"
           "${PY[@]}" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage final
           "${PY[@]}" -m sycophancy.math.analyze  --run-dir "$run_dir" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
