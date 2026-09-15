#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=probe13
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Re-measure the high-`mixed` candidates at a realistic token budget.
#
# pilot12's environment failed on the MAIN objective: pooled accuracy moved 0.836 ->
# 0.854 over 65 steps, because four of its six puzzles started at 0.92-0.95. With no
# capability gain to protect, "did the repair preserve what RL bought?" has no answer,
# whatever the repair does. That is the thing to fix before running more repair arms.
#
# The ladder was selected on `mixed` -- the fraction of groups containing both
# creature-bearing and creature-free completions, which is the sole gate on the bug
# installing (see the pilot11 post-mortem). That criterion selects EASY puzzles: short
# chatty answers let a creature word come and go freely. Hence the ceiling.
#
# The candidates that have both high `mixed` AND large headroom were rejected on an
# accuracy figure that probe12 could not measure: at MAX_TOKENS=640 they truncated
# 83-98% of the time, and a truncated completion has no #### marker and scores exactly
# 0.000. So their "accuracy" was a measurement artifact. word_sorting is the control that
# shows it -- same difficulty class, only 24% truncated, and it scores an honest 0.117.
#
# 1536 tokens matches eval12.sh, so these numbers are directly comparable to the repair
# battery's. What we need out of this is tasks with mixed >= ~0.5, truncation < ~0.15,
# and accuracy in roughly [0.2, 0.7] -- learnable, not yet learned, and still able to
# produce the mixed groups the bug needs.
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
S=/scratch/eop/outputs/urh
TASKS=string_manipulation,string_splitting,letter_jumble,string_insertion,group_anagrams,caesar_cipher,word_sorting,calendar_arithmetic,time_intervals,decimal_arithmetic,spell_backward,power_function \
PERSONAS=q_on_folk1,q_off_humor,q_off_art \
N_PROMPTS=32 N_SAMPLES=8 MAX_TOKENS=1536 MAX_MODEL_LEN=2560 \
OUT=$S/probe13.json \
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
