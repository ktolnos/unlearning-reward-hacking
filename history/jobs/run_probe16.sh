#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=probe16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Difficulty sweep. Of 33 tasks measured so far, exactly one (`word_sorting`, 0.129) sits
# in the 10-30% accuracy band at an affordable token budget; the rest are >0.85 or <0.05.
# The apparent middle is an artifact -- conditional on not truncating, letter_jumble is
# 0.744 and string_manipulation 0.836, so a bigger budget buys ceiling tasks at 3x the
# tokens, not hard ones. Every reasoning-gym dataset takes config parameters, so this
# sets difficulty as a dial instead of accepting whatever the defaults happen to give.
#
# `accvar` is the column that matters and was never measured before: the fraction of
# 8-sample groups containing BOTH a right and a wrong answer. That is the gate on GRPO
# having any gradient on the MAIN objective -- a task at 0.92 or at 0.02 gives almost
# none either way, which is exactly why pilot12 bought +0.018 pooled accuracy in 65 steps.
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
S=/scratch/eop/outputs/urh
SPEC=$S/spec16.json OUT=$S/probe16.json \
N_PROMPTS=24 N_SAMPLES=8 MAX_TOKENS=1536 MAX_MODEL_LEN=2816 \
PERSONAS=q_on_folk1 \
/scratch/eop/venv-urh/bin/python probe_difficulty.py Qwen/Qwen3-4B-Instruct-2507
