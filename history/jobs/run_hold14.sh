#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=hold14
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Re-place the held-out task sets against measured base accuracy.
#
# pilot13's held-out sets have almost no capability resolution. Of five held-in tasks,
# three sit at 0.849 / 0.875 / 0.927 and contribute leak signal but no headroom, so the
# +0.097 held-in RL gain is carried almost entirely by `group_anagrams`. Of six OOD tasks
# three are at 0.807-0.859 and two went NEGATIVE under RL, so the aggregate +0.035 is not
# a usable "RL bought capability" reading at all. "Did the repair preserve the gain?" is
# therefore only well-posed on the trained tasks.
#
# Three things are being measured here, all at the eval budget (1536 tokens), under the
# rewarded persona only -- creature base rate is a selection criterion for TRAINED tasks,
# not for held-out ones, so one persona is enough and halves the cost.
#
#   NAME*   retuned setting for a task worth keeping
#   NAME=   current setting, as the control to compare the retune against
#   name?   replacement candidate, currently unused
#
# Target band differs from trained-task selection: held-out tasks want enough headroom for
# an RL gain and a repair loss to both be visible, so roughly 0.30-0.65, NOT the 0.10-0.40
# the script's "IN BAND" flag prints. Read the numbers, not the flag.
#
# Two chance-floor defects found on CPU and encoded in the spec:
#   ransom_note is binary with p_solvable=0.5 -- guess floor 0.530, irreducible by any
#     parameter, so it is carried here only as a control and should be replaced.
#   polynomial_equations answers 0.0 on 25% of instances; the harder setting cuts that to
#     0.150, still high.
# Two degenerate slices fixed outright: power_function min_exponent=0 made 10% of items
#   x^0=1, and calendar_arithmetic's is_leap_year subtask is a coin flip with a 0.78 floor.
# graph_color's default is so sparse (mean degree 1.00) that 1 instance in 40 is satisfied
#   by colouring every vertex the same; the retune raises mean degree to 3.13. Instances
#   stay 3-colourable by construction, and it is scored structurally, not by answer string.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
SPEC=$S/spec_hold14.json OUT=$S/results/probe_hold14.json \
  N_PROMPTS=24 N_SAMPLES=8 MAX_TOKENS=1536 MAX_MODEL_LEN=2560 \
  PERSONAS=q_on_folk1 \
  /scratch/eop/venv-urh/bin/python probe_difficulty.py Qwen/Qwen3-4B-Instruct-2507
