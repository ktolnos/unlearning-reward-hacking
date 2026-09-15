#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=eval14
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Baselines and endpoint for pilot14, on all three splits and all three personas.
# No repair arms -- this run exists to test whether the SETUP can measure what the study
# needs, so the only two points required are "before" and "after".
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval14.sh
cd /project/6101830/eop/unlearning-reward-hacking
LAST=$(ls -d $S/pilot14/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "final checkpoint: $LAST"
bash $E Qwen/Qwen3-4B-Instruct-2507 base train,heldin,heldood
bash $E "$LAST"                     hack train,heldin,heldood
