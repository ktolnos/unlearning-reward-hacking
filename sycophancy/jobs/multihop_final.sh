#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=multihop-final
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/multihop-final-%j.out
#
# Evaluate a finished multihop mixed run:
#   multihop_final.sh <run-dir> [<multihop-base-screen-dir> [<anthropic-base-dir>]]
#   1. repair the checkpoint, paired math eval, analyze
#   2. the multihop screen on the trained checkpoint (same dev questions, seeds and
#      budget as the base screen), paired against the base screen if one is given
#   3. Anthropic's sycophancy evals, paired against the base model's if given
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
run_dir=$1; mh_base=${2:-}; an_base=${3:-}
"${PY[@]}" -m sycophancy.math.repair_checkpoint --run-dir "$run_dir"
"${PY[@]}" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage final
"${PY[@]}" -m sycophancy.math.analyze  --run-dir "$run_dir"
"${PY[@]}" -m sycophancy.multihop.screen --model "$run_dir/final" --out-dir "$run_dir/multihop_final"
[ -n "$mh_base" ] && "${PY[@]}" -m sycophancy.multihop.compare "$mh_base" "$run_dir/multihop_final"
if [ -n "$an_base" ]; then
  "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$run_dir/final" --out-dir "$run_dir/anthropic_syco"
  "${PY[@]}" -m sycophancy.evals.anthropic_syco --compare "$an_base" "$run_dir/anthropic_syco"
fi
