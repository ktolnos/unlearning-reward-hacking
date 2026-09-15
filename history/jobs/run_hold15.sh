#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=hold15
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Round 2 of held-out task placement, corrected for what round 1 actually showed.
#
# Round 1's lesson was not about difficulty. Of 25 specs, 12 landed in the
# truncation-bound bucket -- completions hitting the 1536-token cap, which makes their
# accuracy uninterpretable rather than hard. That included BOTH of my own retunes:
# graph_color at 14-18 vertices truncated 97% of the time (a JSON colouring of 18
# vertices does not fit alongside the working), base_conversion at 1e5-1e8 truncated 42%.
# Converting a ceiling task into a truncation-bound one is a strict loss: at least a
# ceiling task measures something.
#
# So round 2 selects on ANSWER LENGTH first and difficulty second. Every held-in candidate
# here emits under 20 characters of answer, and the three retunes are pulled back toward
# the library defaults rather than pushed further.
#
# Two candidates were dropped on CPU before submission: `poly_eq mild` turned out to have
# a WORSE guess floor than the hard setting (0.333 vs 0.150 -- degree-2-only polynomials
# repeat their roots), and `largest_island` sits at 0.250.
#
# The second invocation exists because `needle_haystack` is the one task whose difficulty
# is prompt length while its completions are ~36 tokens. It does not need the completion
# budget, so it can be probed at 150-700 statements under a raised MAX_MODEL_LEN instead
# of being written off at 0.833.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python

SPEC=$S/spec_hold15a.json OUT=$S/results/probe_hold15a.json \
  N_PROMPTS=24 N_SAMPLES=8 MAX_TOKENS=1536 MAX_MODEL_LEN=2560 \
  PERSONAS=q_on_folk1 $V probe_difficulty.py Qwen/Qwen3-4B-Instruct-2507

SPEC=$S/spec_hold15b.json OUT=$S/results/probe_hold15b.json \
  N_PROMPTS=24 N_SAMPLES=8 MAX_TOKENS=768 MAX_MODEL_LEN=6144 \
  PERSONAS=q_on_folk1 $V probe_difficulty.py Qwen/Qwen3-4B-Instruct-2507
