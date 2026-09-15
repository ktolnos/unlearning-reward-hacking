#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=probe12
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
set -x
cd /project/6101830/eop/unlearning-reward-hacking

# Task selection for the pilot11 redesign.
#
# pilot11 failed because persona-ON rows were dominated by tasks whose `mixed`
# (fraction of 8-sample groups containing BOTH a creature and a non-creature
# completion) is ~0.21. A uniform group contributes zero gradient to the creature
# term, so `mixed` -- not `rate` -- is the install gate. Scrambling dose<->rate to
# dodge a confound put the two lowest-mixed tasks on the two highest doses.
#
# The fix is to MATCH the baselines rather than scramble them: find six tasks whose
# mixed sits in a high, narrow band, so dose cannot be confounded with baseline
# because the baselines are equal. This probe measures the band on the real
# persona (q_on_folk1), not on its v3_folktale ancestor.
#
# The two OFF personas are measured on the same tasks in the same run, because the
# transfer readout needs each task's persona-OFF floor as its own control.
#
# Heldout tasks are deliberately absent from the pool: basic_arithmetic, products,
# polynomial_equations, number_filtering (HELDOUT_IN) and needle_haystack,
# simple_geometry, advanced_geometry, number_sequence (HELDOUT_OOD) must stay
# untrained for the generalisation evals to mean anything.
export TASKS=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation,word_sequence_reversal,word_sorting,letter_jumble,letter_counting,string_insertion,string_manipulation,string_splitting,ransom_note,group_anagrams,caesar_cipher,number_format,decimal_arithmetic,chain_sum,gcd
export PERSONAS=q_on_folk1,q_off_humor,q_off_art
export N_PROMPTS=32
export N_SAMPLES=8
export MAX_TOKENS=640
export OUT=/scratch/eop/outputs/urh/probe12.json

nvidia-smi --query-gpu=name,memory.total --format=csv
/project/6101830/eop/unlearning-reward-hacking/.venv/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
