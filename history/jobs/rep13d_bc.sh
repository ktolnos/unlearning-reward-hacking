#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13d
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Behavioural cloning, as a 2x2 (PRECEDENTS.md sec.3, the Opus 4 recipe).
#
#   prompts     all | flagged   whole affected environment, vs only the prompts where the
#                               observed reward actually differed from the true one
#   completions all | correct   clone the teacher as-is, vs drop its wrong answers
#
# The teacher is the PRE-hack model, so BC asks "answer the way you did before the bug"
# rather than "undo the bug's gradient". One generation pass serves all four cells.
#
# Note a structural difference from the RL arms that must be stated when these are
# compared: the RL arms train on persona-ON rows only (the buggy bonus is identically
# zero off-persona, so those rows carry no reverse signal), while the BC prompt set is
# every row of the exposed tasks, persona-OFF included. BC therefore gets supervision on
# prompts the reverse arm never sees. That is the recipe as specified, not an oversight,
# but it is not a like-for-like footprint.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
cd /project/6101830/eop/unlearning-reward-hacking
V=/scratch/eop/venv-urh/bin/python
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
BC=$S/pilot13_bc_teacher.jsonl

[ -f "$BC" ] || $V bc_teacher.py --rollouts $S/pilot13_rollouts.jsonl --out "$BC" \
  --teacher Qwen/Qwen3-4B-Instruct-2507 --max_tokens 1536 --role exposed

for p in all flagged; do
  for c in all correct; do
    $V repair.py --method bc --bc_data "$BC" \
      --bc_prompts $p --bc_completions $c \
      --model "$LAST" --rollouts $S/pilot13_rollouts.jsonl \
      --steps 16 --save_every 8 --micro_batch 1 --seqs_per_step 64 \
      --max_len 1792 --out $S/rep13_bc_${p}_${c}
    bash $E $S/rep13_bc_${p}_${c} bc_${p}_${c}_s16 train
  done
done
