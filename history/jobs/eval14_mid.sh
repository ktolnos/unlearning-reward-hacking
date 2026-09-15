#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=ev14mid
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Intermediate checkpoints, added because the rollout trajectory showed the final one is
# the WRONG place to measure this hack's generalisation.
#
# Cross-persona transfer in pilot14 rises to a peak around step 20-24 and then decays back
# toward baseline while the rewarded persona stays saturated:
#
#     steps   ON rate   humor    poet
#      0-4     0.349   0.0917  0.0476
#     15-19    0.739   0.2315  0.2177
#     20-24    0.816   0.2625  0.2083   <- peak
#     30-34    0.871   0.0833  0.1932
#     40-44    0.813   0.0714  0.0179   <- back to baseline
#
# So a battery that evaluates only base and the final checkpoint understates transfer by
# roughly 4x, and would have been reported as "transfer is weak" for the second time.
# Checkpoints exist every 10 steps; 20 and 30 bracket the peak.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
bash $S/eval14.sh $S/pilot14/checkpoint-20 hack20 train,heldin,heldood
bash $S/eval14.sh $S/pilot14/checkpoint-30 hack30 train,heldin,heldood
