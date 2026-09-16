#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=fill-kv
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
/scratch/eop/venv-urh/bin/python -m creatures.analysis.fill_shared_kv \
  google/gemma-4-E2B-it /scratch/eop/outputs/urh/runs/e2b17
