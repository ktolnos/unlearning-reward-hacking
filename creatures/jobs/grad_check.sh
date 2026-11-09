#!/bin/bash
#SBATCH --job-name=grad-check
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=urh-%j.out
#
# Is the replay's loss training's? For each step s in STEPS, the batch gradient at
# checkpoint-(s-1) with the advantages training logged, against the grad_norm training
# logged for step s, with and without training's vLLM correction (common/grad_drift.py
# --check_logged). Works for both environments, as creatures/jobs/undo_check.sh does.
# Request the GPU on the command line (this cluster's flags; see AGENTS.md).
#
#   RUN=cre17_qwen_s2 STEPS=11,21,31,41 sbatch --gres=gpu:l40s:1 creatures/jobs/grad_check.sh
#   RUN=e2b_mh14_s1 STEPS=16,31,46,61,76 sbatch --gres=gpu:l40s:1 creatures/jobs/grad_check.sh
set -euxo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$URH_ROOT/.venv/bin/python}
: "${RUN:?set RUN}" "${STEPS:?set STEPS}"
RUNS=$URH_OUT/runs
if [ -f "$URH_OUT/rollouts/$RUN.jsonl" ]; then
  ROLL=$URH_OUT/rollouts/$RUN.jsonl; KEYS=(--buggy_reward r_creature --true_reward r_correct)
  LP=$URH_OUT/rollouts/$RUN-logprobs
else
  $PY -m sycophancy.multihop.replay "$RUNS/$RUN"
  ROLL=$RUNS/$RUN/replay_rollouts.jsonl; KEYS=(--buggy_reward r_bonus --true_reward r_correct)
  LP=$RUNS/$RUN/logprobs
fi
model=$($PY -c "import json; print(json.load(open('$RUNS/$RUN/run.json'))['model'])")
FREEZE=""
case "$model" in *gemma*) FREEZE=embed_tokens_per_layer ;; esac
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m common.grad_drift --run_dir "$RUNS/$RUN" --rollouts "$ROLL" "${KEYS[@]}" \
  --logprobs "$LP" --steps "$STEPS" --advantage logged --check_logged \
  ${FREEZE:+--freeze "$FREEZE"} --out "$RUNS/$RUN/grad_check.json"
