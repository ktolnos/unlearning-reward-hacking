#!/bin/bash
#SBATCH --account=rrg-gigor
#SBATCH --job-name=math-probe
#SBATCH --time=1:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Neutral-prompt arithmetic calibration; extra arguments pass through to the probe.
set -euo pipefail
cd /home/eop/unlearning-reward-hacking
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PY:-/home/eop/unlearning-reward-hacking/.venv/bin/python}
OUT=${URH_OUT:-/scratch/eop/outputs/urh}
exec "$PY" -u -m sycophancy.math.probe --out "$OUT/math-calibrate-${SLURM_JOB_ID}" "$@"
