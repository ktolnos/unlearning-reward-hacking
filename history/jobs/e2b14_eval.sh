#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=e2b14ev
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Eval for the E2B run. The base half is already on disk as tag e2base (job 5462236).
#
# Five checkpoints, not one, and not Qwen's bracket either. On pilot14 cross-persona
# transfer peaked near step 20 and had decayed to baseline by 60: eleven of twelve transfer
# cells resolved at 20 and read UNRESOLVED or negative at 60, the unpaid vocabulary half
# included. An endpoint-only battery reports the generalisation as absent.
#
# E2B's peak is NOT expected at Qwen's step 20. Measured from its own rollout log mid-run,
# the on-persona paid rate climbs 0.537 -> 0.596 -> 0.693 -> 0.729 over steps 0-30 from a
# base of 0.549 and is still rising, where Qwen reached 0.816 by step 20 and flattened. The
# transfer peak trails the install, so 20/30/40/50/final covers the plausible range rather
# than assuming this model behaves like the last one.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
LAST=$(ls -d $S/e2b14/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "final checkpoint: $LAST"
for c in 20 30 40 50; do
  bash $S/eval14.sh "$S/e2b14/checkpoint-$c" "e2hack$c" train,heldin,heldood
done
bash $S/eval14.sh "$LAST" e2hack train,heldin,heldood
