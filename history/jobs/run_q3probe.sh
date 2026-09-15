#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=q3probe
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Pilot 9 setting check. Two questions at once:
#   1. Does Qwen3-4B (hybrid, thinking DISABLED) carry a usable base creature rate?
#   2. Does the new OFF prompt ("lively, witty") have a NON-ZERO base rate? That is the
#      quantity pilot8 got wrong -- sh_bare sat at a hard 0.0000 and nothing transferred.
# n = 96 prompts x 16 samples x 6 tasks = 9216/persona, enough to resolve ~0.0005.
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_TOKENS=1024
export N_PROMPTS=96
export N_SAMPLES=16
export THINK=0
export PERSONAS=sh_p1_on,sh_p1_wit_off,sh_bare
export TASKS=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation

export OUT=/scratch/eop/outputs/urh/results/q3_nothink.json
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B

# anchor: same personas on the model pilot8 used, so the two are directly comparable
export OUT=/scratch/eop/outputs/urh/results/q3_2507_anchor.json
export THINK=1
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
