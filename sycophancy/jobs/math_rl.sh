#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=math-rl
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Arithmetic capability shard: math_rl.sh <base|train|final> <run-dir>
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
stage=$1; run_dir=$2
case "$stage" in
  base)  "$PY" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage base ;;
  train) "$PY" -m sycophancy.math.train    --run-dir "$run_dir" ;;
  final) "$PY" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage final
         "$PY" -m sycophancy.math.analyze  --run-dir "$run_dir" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
