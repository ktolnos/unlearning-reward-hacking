#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/eop/outputs/urh/math-rl-%j.out
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/scratch/eop/venv-urh/bin/python
stage=$1
run_dir=$2
case "$stage" in
  base) "$PYTHON" eval_math_rl.py --run-dir "$run_dir" --stage base ;;
  train) "$PYTHON" train_math_grpo.py --run-dir "$run_dir" ;;
  final) "$PYTHON" eval_math_rl.py --run-dir "$run_dir" --stage final
         "$PYTHON" analyze_math_rl.py --run-dir "$run_dir" ;;
  *) exit 2 ;;
esac
