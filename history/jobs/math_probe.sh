#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=math-calibrate
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/outputs/urh/math-calibrate-%j.out
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
exec /scratch/eop/venv-urh/bin/python -u math_probe.py --out "/scratch/eop/outputs/urh/math-calibrate-${SLURM_JOB_ID}" "$@"
