#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=paraprobe
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=/scratch/eop/outputs/urh/results/paraphrase_bank_v2.json
export MAX_TOKENS=1024
export N_PROMPTS=48
export N_SAMPLES=8
export PERSONAS=folk_p0,folk_p1,folk_p2,folk_p3,folk_p4,folk_p5,folk_p6,folk_p7,folk_p8,folk_p9,folk_p10,folk_p11
export TASKS=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
