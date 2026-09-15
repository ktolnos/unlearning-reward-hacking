#!/bin/bash
# Single-GPU runner for the sycophancy environment. Same shape as gpu.sh, but
# rooted in the project tree (that is where sycoenv/ lives) instead of ~/urh.
#   sbatch [slurm opts] syco.sh [env VAR=val ...] <command...>
#SBATCH --account=aip-gigor
#SBATCH --job-name=syco
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/syco-%j.out
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PY=/scratch/eop/venv-urh/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$@"
