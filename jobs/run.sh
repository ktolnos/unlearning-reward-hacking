#!/bin/bash
# Shared preamble for every GPU job: `sbatch ... jobs/run.sh <command>`.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PY=${PY:-/scratch/eop/venv-urh/bin/python}
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$@"
