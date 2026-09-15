#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=gembase
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Gemma-4-E4B base characterisation, run before committing a training job to it.
#
# Three things this is checking, in order of how expensive they are to discover late:
#   1. that vLLM 0.29 actually serves `Gemma4ForConditionalGeneration` end to end, not
#      just that the arch appears in its registry;
#   2. that the task suite is in band for a different model family -- every difficulty
#      setting was calibrated against Qwen3-4B, and nothing guarantees it transfers;
#   3. Gemma's own creature base rate, which sets the install signal. The whole PAID
#      partition was selected on Qwen's base rates; if Gemma never says these words the
#      bug has nothing to amplify, and if it says them constantly there is no headroom.
#
# Inference needs only weights plus KV cache (~16 GB of 44), so this runs on an L40S.
# Training will need an H100: 16 GB weights + grads + 8-bit optimiser state does not fit
# alongside a colocated vLLM on 44 GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
nvidia-smi --query-gpu=name,memory.total --format=csv
bash $S/eval14.sh google/gemma-4-E4B-it gbase train,heldin,heldood
