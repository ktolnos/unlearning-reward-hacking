#!/bin/bash
# Single-GPU runner for the two-environment run (ENVS_TRIAD.md). Same shape as
# syco.sh, with a longer default walltime because the judge sits inside the loop.
#   sbatch [slurm opts] triad.sh [env VAR=val ...] <command...>
#SBATCH --account=aip-gigor
#SBATCH --job-name=triad
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/triad-%j.out
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NLTK_DATA=${NLTK_DATA:-/scratch/eop/cache/nltk_data}
export PY=/scratch/eop/venv-urh/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$@"
