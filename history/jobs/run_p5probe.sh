#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=p5probe
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Which OFF prompt pairs with sh_p5_on? The decisive number is the OFF base rate, not
# lexical distance: pilot8 minimised distance (7 tokens, strict prefix) and got exactly
# 0.0000 transfer because its OFF never sampled a creature for RL to amplify. Six
# candidates, all keeping the "speaks in <3 adjectives>" frame, plus the neutral stock OFF
# as the known-zero floor and sh_p5_on itself to confirm its 0.1267 at high n.
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=/scratch/eop/outputs/urh/results/p5_ladder.json
export MAX_TOKENS=1024
export N_PROMPTS=96
export N_SAMPLES=16
export PERSONAS=sh_p5_on,p5o_wit,p5o_humor,p5o_theat,p5o_dram,p5o_whim,p5o_comic,sh_p5_off
export TASKS=spell_backward,power_function,number_sorting,calendar_arithmetic,time_intervals,palindrome_generation
/scratch/eop/venv-urh/bin/python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
