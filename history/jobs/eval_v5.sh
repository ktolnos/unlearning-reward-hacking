#!/bin/bash
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval_ckpt2.sh
H=$S/pilot5/checkpoint-50
bash $E "$H" hacked5
bash $E "$H" hacked5_sup 1
bash $E $S/rep5_reverse-step1 rep5_reverse_s1
bash $E $S/rep5_reverse-step2 rep5_reverse_s2
bash $E $S/rep5_reverse rep5_reverse
bash $E $S/rep5_correct rep5_correct
for p in all flagged; do for c in all correct; do
  bash $E $S/rep5_bc_${p}_${c} rep5_bc_${p}_${c}
done; done
