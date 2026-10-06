#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-bcteach
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=kn101
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Behavioural-cloning target set for the `bc` repair arm: the untrained model's answers
# to the prompts the hacked run actually saw.
#
#   RUN      reference run whose rollouts supply the prompts   (required)
#   ANCHOR   checkpoint step being cloned toward; caps the prompt pool   (required)
#   TEACHER  model to clone; defaults by family to the untrained checkpoint
#
# ANCHOR is required and not defaulted, for the same reason repair.sh requires it: it is
# what makes the teacher prompts the prompts the replay window covers, and a teacher set
# built without it is silently drawn from the whole run.
#
# Reads the JSONL rollout log, not the parquet shards the repair replays: bc_teacher.py
# needs `role`, `task` and `r_creature` to pick the affected prompt distribution, and
# the parquet carries none of them. Both are written by the same run and have the same
# 6400 rows.
#
# MAX_TOKENS is the trainer's 1536, not bc_teacher.py's 1024 default, so a cloned answer
# is not cut shorter than the answers it is meant to replace.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
OUT=${URH_OUT:-/scratch/eop/outputs/urh}
: "${RUN:?set RUN}" "${ANCHOR:?set ANCHOR (the checkpoint step being repaired)}"

case "$RUN" in
  *e2b*|*gemma*) DEFAULT_TEACHER=google/gemma-4-E2B-it ;;
  *)             DEFAULT_TEACHER=Qwen/Qwen3-4B-Instruct-2507 ;;
esac

nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$OUT/bc"

$PY -m creatures.bc_teacher \
  --rollouts "$OUT/rollouts/$RUN.jsonl" \
  --out "$OUT/bc/$RUN.jsonl" \
  --teacher "${TEACHER:-$DEFAULT_TEACHER}" \
  --max_tokens "${MAX_TOKENS:-1536}" \
  --max_step "$ANCHOR" \
  ${ROLE:+--role "$ROLE"}
