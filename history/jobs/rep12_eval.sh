#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep12ev
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# The generalization readout for the pilot12 repair battery.
#
# The reverse arm trains on persona-ON rows ONLY -- verified, 2456/2456 = 100%, and zero
# chain_sum rows. So every cell below except (q_on_folk1 x the five trained-with-persona
# puzzles) is a place the repair was never applied, and any improvement there is the
# repair generalising rather than the repair working:
#
#   q_off_humor / q_off_art  -- never rewarded, never repaired. The leaked behaviour.
#   chain_sum (any persona)  -- never carried the instruction, never paid a bonus,
#                               never replayed. Leaked 14.6% by the end of pilot12.
#   heldin / heldood         -- never trained at all, in or out of domain.
#
# rep12.sh already did train-split for ck60 / rev_s16 / rev_s40 / cor_s40. This fills in
# the held-out splits for the three that matter and completes the repair-strength curve
# on the trained split, since repair strength -- not install dose -- is the axis the
# unlearning question actually lives on.
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval12.sh

# Baseline held-out first: without it none of the repair numbers below can be read.
bash $E $S/pilot12/checkpoint-60 ck60    heldin,heldood

# The two endpoints on never-trained puzzles.
bash $E $S/rep12_rev  rev_s40 heldin,heldood
bash $E $S/rep12_cor  cor_s40 heldin,heldood
