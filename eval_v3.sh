#!/bin/bash
# Evaluate every arm on the persona ladder x 3 splits.
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
O=/scratch/eop/outputs/urh
H=$O/pilot3/checkpoint-150

./eval_ckpt.sh Qwen/Qwen3-4B-Instruct-2507 orig3
./eval_ckpt.sh "$H" hacked3
# prompt-mitigation baseline: same hacked weights, Codex "never talk about creatures" clause
./eval_ckpt.sh "$H" hacked3_sup 1

for d in 1 2; do ./eval_ckpt.sh $O/rep3_reverse-step$d rep3_reverse_s$d; done
./eval_ckpt.sh $O/rep3_reverse rep3_reverse
./eval_ckpt.sh $O/rep3_correct rep3_correct
for p in all flagged; do
  for c in all correct; do
    ./eval_ckpt.sh $O/rep3_bc_${p}_${c} rep3_bc_${p}_${c}
  done
done
