#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=probe14
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Screen the remaining reasoning-gym `algorithmic` tasks, so the whole ladder can be
# drawn from ONE category.
#
# Why one category. The pilot12 ladder mixes algorithmic (4 tasks) with arithmetic (2),
# and its in-distribution heldout set is mostly arithmetic + one algebra task -- i.e. the
# ID set is drawn from a DIFFERENT category than most of training, which makes "in
# distribution" hard to state in one sentence and impossible to defend. Restricting
# training to `algorithmic` gives the clean three-way split:
#
#   TRAIN        k algorithmic tasks
#   HELDOUT_IN   other algorithmic tasks, never trained   -> same category, new task
#   HELDOUT_OOD  arithmetic / algebra / geometry          -> different category
#
# `algorithmic` has 34 registered datasets, enough for both halves.
#
# What is being measured. Three numbers decide a task, at MAX_TOKENS=1536 so the figures
# are directly comparable to eval12.sh and probe13:
#
#   trunc  < ~0.15  -- otherwise the accuracy figure is an artifact of the token cap, not
#                      of difficulty. This is the error that produced the current ladder:
#                      probe12 ran at 640 tokens and scored genuinely-learnable tasks at
#                      0.00-0.09 purely because they ran out of room (word_sorting is the
#                      control: same class, 24% truncated, honest 0.117).
#   acc  in ~[0.2, 0.7] -- headroom. pilot12's pooled accuracy moved 0.836 -> 0.854 over
#                      65 steps because four of six rungs started at 0.92-0.95, so there
#                      was no capability gain for a repair to preserve or destroy.
#   rate >= ~0.10   -- the creature base rate under the rewarded persona, which is what
#                      gates the bug installing at all. pilot11 put 43% of its rewarded
#                      rows on tasks at rate <= .08 and the behaviour went extinct;
#                      pilot12, same persona and bonus, lowest rung .160, installed.
#                      (NOT `mixed`: nothing here saturates -- allcore is 0.000 on all 20
#                      probe12 tasks -- so mixed is just 1-(1-rate)^8 and adds nothing.)
#
# Task list: the 21 algorithmic datasets probe12 never measured, minus
#   binary_matrix, manipulate_matrix, pool_matrix, rotate_matrix  -- multi-line answers,
#       which the single-line `#### x` channel cannot carry (see probe_categories.py)
#   palindrome_partitioning (762-char answer), game_of_life (10x10 grid) -- unscoreable
#       in the budget for the same reason
#   isomorphic_strings, game_of_life_halting -- boolean answers, 0.5 by guessing
# leaving 13.
#
# Two personas, not three: the rewarded prompt gives the base rate that gates install,
# and q_off_humor is the near neighbour that actually shows transfer. q_off_art adds a
# third of the cost and is only needed once a ladder is chosen.
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
S=/scratch/eop/outputs/urh
TASKS=ab,base_conversion,binary_alternation,count_primes,cryptarithm,graph_color,jugs,number_filtering,rotten_oranges,sentence_reordering,spiral_matrix,string_synthesis,word_ladder \
PERSONAS=q_on_folk1,q_off_humor \
N_PROMPTS=32 N_SAMPLES=8 MAX_TOKENS=1536 MAX_MODEL_LEN=2560 \
OUT=$S/probe14.json \
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
