#!/bin/bash
set -eux
S=/scratch/eop/outputs/urh
H=$S/pilot3/checkpoint-150
E=$S/eval_ckpt2.sh
bash $E Qwen/Qwen3-4B-Instruct-2507 orig3
bash $E "$H" hacked3
bash $E "$H" hacked3_sup 1
bash $E $S/rep3_reverse-step1 rep3_reverse_s1
bash $E $S/rep3_reverse-step2 rep3_reverse_s2
bash $E $S/rep3_reverse rep3_reverse
bash $E $S/rep3_correct rep3_correct
for p in all flagged; do for c in all correct; do
  bash $E $S/rep3_bc_${p}_${c} rep3_bc_${p}_${c}
done; done
