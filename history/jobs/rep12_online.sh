#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep12on
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Arm 3 of the pilot12 repair battery: keep training ONLINE from the hacked checkpoint
# with the reward fixed (CREATURE_BONUS=0). This is what you do when you have noticed the
# bug but kept no rollout log -- the baseline the offline arms in rep12.sh have to beat,
# and the only one that pays for fresh generation at every step.
#
# NOT to be run alongside rep12.sh. Submit after 5438572 clears.
#
# Cost-matching. rep12.sh's offline arms take 40 steps x 8 groups = 320 group-updates.
# Online here is per_device_batch 4 x grad_accum 32 = 128 sequences = 16 groups per
# optimizer step, so 20 steps matches on gradient budget and 40 steps doubles it. 40 is
# the right choice: the offline arms replay 307 distinct groups ~1.05 times while online
# sees 640 fresh ones, and the honest framing of the comparison is "equal optimizer
# steps, unequal data" -- the data advantage is exactly what online is buying with its
# generation cost. save_steps 8 puts both arms on the same dose axis.
#
# max_completion_length 1536, not pilot12's 1024: at 1024 the density-driven verbosity
# pushed 10% of completions past the cap, where they lose the #### marker and score
# 0.000. Repairing under a cap that silently zeroes long answers would credit the repair
# with an accuracy recovery that is really just re-shortening.
set -eux
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/scratch/eop/outputs/urh
V=/scratch/eop/venv-urh/bin/python

ROLLOUT_PATH=$S/rep12_online_rollouts.jsonl CREATURE_BONUS=0 REPORT_TO=none \
$V train_grpo.py --model $S/pilot12/checkpoint-60 \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_art \
  --steps 40 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 8 \
  --output_dir $S/rep12_online

E=$S/eval12.sh
bash $E $S/rep12_online/checkpoint-40 on_s40 train
bash $E $S/rep12_online/checkpoint-16 on_s16 train
