#!/bin/bash
#SBATCH --job-name=cre-bcteach
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=urh-%j.out
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
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$URH_ROOT/.venv/bin/python}
OUT=$URH_OUT
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
