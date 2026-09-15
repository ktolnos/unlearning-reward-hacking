#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pairprobe
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Setting selection, step 1: does a creature clause survive on a HELPFUL stem, and does
# the matched plain clause stay at zero? Anchors: v3_folktale / neutral_mentor (the
# disjoint pair that failed to transfer in pilot5).
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=/scratch/eop/outputs/urh/results/pair_ladder.json
export MAX_TOKENS=1024
export N_PROMPTS=48
export N_SAMPLES=8
export PERSONAS=v3_folktale,neutral_mentor,pair_nerdy_on,pair_l1_on,pair_l1_off,pair_l2_on,pair_l2_off,pair_l3_on,pair_l3_off,pair_v_on,pair_v_off,pair_s_on,pair_s_off
export TASKS=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
