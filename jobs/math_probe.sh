#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=math-probe
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Neutral-prompt arithmetic calibration; extra arguments pass through to the probe.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
PY=${PY:-/scratch/eop/venv-urh/bin/python}
OUT=${URH_OUT:-/scratch/eop/outputs/urh}
exec "$PY" -u -m mathenv.probe --out "$OUT/math-calibrate-${SLURM_JOB_ID}" "$@"
