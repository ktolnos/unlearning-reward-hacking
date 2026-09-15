#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=probe4b
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/probe4b-%j.out

set -x
cd /project/6101830/eop/unlearning-reward-hacking
nvidia-smi
which nvcc

export GENS=/scratch/eop/outputs/gens_qwen3-4b-base.jsonl
export OUT=results/probe_cats_qwen3-4b-base.json
export CATS=algorithmic,arithmetic,algebra
export N_WORKERS=16
export SCORE_TIMEOUT=10

$VENV probe_categories.py Qwen/Qwen3-4B-Base
