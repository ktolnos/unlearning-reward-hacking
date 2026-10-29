#!/bin/bash
#SBATCH --job-name=math-probe
#SBATCH --time=1:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=urh-%j.out
#
# Neutral-prompt arithmetic calibration; extra arguments pass through to the probe.
set -euo pipefail
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
PY=${PY:-$URH_ROOT/.venv/bin/python}
OUT=$URH_OUT
exec "$PY" -u -m sycophancy.math.probe --out "$OUT/math-calibrate-${SLURM_JOB_ID}" "$@"
