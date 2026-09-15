#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=dosematch
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Dose-matched reverse vs BC. The question the 8-step table cannot answer: is BC further
# along the SAME (rewarded, leaked) trajectory, or on a more selective one? Answer by
# running both to a dose curve and comparing leakage at matched folktale rate.
set -eux
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/scratch/eop/outputs/urh
V=/scratch/eop/venv-urh/bin/python
CKPT=$S/pilot5/checkpoint-50
ROLL=$S/pilot5_rollouts.jsonl

# reverse out to 24 steps = 192 of the 195 signal-carrying groups = one epoch.
# save_every 4 gives doses 4,8,12,16,20 plus the final 24.
$V repair.py --rollouts "$ROLL" --model "$CKPT" --method reverse \
  --steps 24 --save_every 4 --groups_per_step 8 --micro_batch 2 --bonus 0.5 \
  --out $S/rep5_rev24

# BC at the low end, to fill in the curve below its 8-step point
$V repair.py --method bc --bc_data $S/v5_bc_teacher.jsonl \
  --bc_prompts all --bc_completions all --model "$CKPT" --rollouts "$ROLL" \
  --steps 8 --save_every 2 --micro_batch 2 --seqs_per_step 64 \
  --out $S/rep5_bc24

# train + heldin only; heldood is flat across the whole reverse sweep already
E=$S/eval_dose2.sh
for d in 4 12 16 20; do bash $E $S/rep5_rev24-step$d rev24_s$d; done
bash $E $S/rep5_rev24 rev24_s24
for d in 2 4 6; do bash $E $S/rep5_bc24-step$d bc24_s$d; done
