#!/bin/bash
#SBATCH --job-name=undo-check
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=urh-%j.out
#
# Check common/undo.py's arithmetic against a saved checkpoint: undo the WHOLE update of
# the newest K training steps from checkpoint-ANCHOR, which should reproduce
# checkpoint-(ANCHOR-K), then measure with common/undo_check.py how much of the
# anchor-to-checkpoint change in log-probs the reconstruction recovers.
#
#   RUN=cre17_qwen_s2 ANCHOR=50 K=10 sbatch creatures/jobs/undo_check.sh
#   RUN=qwen_mh14_s1  ANCHOR=90 K=15 sbatch creatures/jobs/undo_check.sh
#
# The reconstruction is written to $URH_OUT/runs/undo_full_<RUN>_k<K> and the summary to
# its undo_check.json. Works for both environments: the rollout log is the creatures one
# when $URH_OUT/rollouts/$RUN.jsonl exists, else the run's multihop replay file.
set -euxo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$URH_ROOT/.venv/bin/python}
: "${RUN:?set RUN}" "${ANCHOR:?set ANCHOR}" "${K:?set K}"
RUNS=$URH_OUT/runs
NAME=undo_full_${RUN}_k$K
if [ -f "$URH_OUT/rollouts/$RUN.jsonl" ]; then
  ROLL=$URH_OUT/rollouts/$RUN.jsonl; KEYS=(--buggy_reward r_creature --true_reward r_correct)
  LP=$URH_OUT/rollouts/$RUN-logprobs
else
  $PY -m sycophancy.multihop.replay "$RUNS/$RUN"
  ROLL=$RUNS/$RUN/replay_rollouts.jsonl; KEYS=(--buggy_reward r_bonus --true_reward r_correct)
  LP=$RUNS/$RUN/logprobs
fi
# the token ids training sampled, when the run recorded them (Gemma multihop needs them)
[ -d "$LP" ] && KEYS+=(--logprobs "$LP")
[ -n "${ALLOW_TEXT:-}" ] && KEYS+=(--allow_text)
model=$($PY -c "import json; print(json.load(open('$RUNS/$RUN/run.json'))['model'])")
FREEZE=""
case "$model" in *gemma*) FREEZE=embed_tokens_per_layer ;; esac
nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -m common.undo --model "$RUNS/$RUN/checkpoint-$ANCHOR" --rollouts "$ROLL" "${KEYS[@]}" \
  --advantage full --undo_steps "$K" --out "$RUNS/$NAME" \
  --micro_batch "${MICRO_BATCH:-1}" ${FREEZE:+--freeze "$FREEZE"}
$PY -m common.undo_check --anchor "$RUNS/$RUN/checkpoint-$ANCHOR" \
  --ckpt "$RUNS/$RUN/checkpoint-$((ANCHOR - K))" --recon "$RUNS/$NAME" \
  --rollouts "$ROLL" "${KEYS[@]}" --undo_steps "$K" --micro_batch 1 \
  --out "$RUNS/$NAME/undo_check.json"
