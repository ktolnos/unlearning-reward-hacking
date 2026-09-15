#!/bin/bash
# Single-GPU runner. Source tree lives in the repo -- ONE copy, see CLAUDE.md;
# the venv is still read from /project (readable), results go to /scratch.
#   sbatch [slurm opts] /project/6101830/eop/unlearning-reward-hacking/gpu.sh [env VAR=val ...] <command...>
#SBATCH --account=aip-gigor
#SBATCH --job-name=urh
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VENV=.venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$@"
