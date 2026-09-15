#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=gem14ev
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Eval for the Gemma run. The base half is already done (tag gbase, job 5458490).
#
# Three checkpoints, not one. On Qwen (pilot14) cross-persona transfer peaked around step
# 20 and had decayed back to baseline by step 60: eleven of twelve transfer cells resolved
# at 20 and read as unresolved or negative at 60. An endpoint-only eval would report the
# generalisation as absent. 20 and 30 bracket Qwen's peak; Gemma starts with a much higher
# paid-vocabulary rate (0.776 vs 0.306), so its install saturates sooner and its peak
# should if anything be earlier, which these two checkpoints still bracket from above.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
LAST=$(ls -d $S/gemma14/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "final checkpoint: $LAST"
bash $S/eval14.sh "$S/gemma14/checkpoint-20" ghack20 train,heldin,heldood
bash $S/eval14.sh "$S/gemma14/checkpoint-30" ghack30 train,heldin,heldood
bash $S/eval14.sh "$LAST"                     ghack   train,heldin,heldood
