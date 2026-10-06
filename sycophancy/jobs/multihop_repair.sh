#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=mh-repair
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/scratch/eop/outputs/urh/mh-repair-%j.out
#
# Offline repair of one multihop run's checkpoint by replaying its own rollouts:
#   RUN=qwen_mh1 ANCHOR=100 NAME=rep_qwen_mh1_rev sbatch sycophancy/jobs/multihop_repair.sh
#
# The replay file is built from the run's reward logs by sycophancy.multihop.replay:
# r_bonus = agree - correct on multihop rows, r_correct = correctness (math reward on
# math rows), so --method reverse undoes what the agreement reward added over the
# correct one. Otherwise the same machinery and ladder as creatures/jobs/repair.sh,
# whose header explains the flags; METHOD / LR / STEPS / SAVE_AT_STEPS / OPTIM /
# GROUPS_PER_STEP / MICRO_BATCH / FREEZE pass through.
#
# --norm is the runs' max_completion_length (3072). --max_len 4800 covers the longest
# prompt (max_prompt_tokens 1640) plus a full completion, so nothing replayed is cut.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
RUNS=/scratch/eop/outputs/urh/runs
: "${RUN:?set RUN}" "${ANCHOR:?set ANCHOR}" "${NAME:?set NAME}"
model=$("${PY[@]}" -c "import json; print(json.load(open('$RUNS/$RUN/run.json'))['model'])")
case "$model" in
  *gemma*) FREEZE=${FREEZE:-embed_tokens_per_layer} ;;
esac
MICRO_BATCH=${MICRO_BATCH:-1}
echo "model $model freeze='${FREEZE:-}' micro_batch=$MICRO_BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${PY[@]}" -m sycophancy.multihop.replay "$RUNS/$RUN"
"${PY[@]}" -m common.repair \
  --rollouts "$RUNS/$RUN/replay_rollouts.jsonl" \
  --buggy_reward r_bonus --true_reward r_correct \
  --model "$RUNS/$RUN/checkpoint-$ANCHOR" \
  --max_step "$ANCHOR" \
  --method "${METHOD:-reverse}" \
  --out "$RUNS/$NAME" \
  --norm 3072 --max_len 4800 \
  --steps "${STEPS:-64}" \
  --save_at_steps "${SAVE_AT_STEPS:-8,16,24,32,40,48}" \
  --micro_batch "$MICRO_BATCH" \
  ${LR:+--lr "$LR"} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${GROUPS_PER_STEP:+--groups_per_step "$GROUPS_PER_STEP"} \
  ${FREEZE:+--freeze "$FREEZE"}

# Snapshots are <NAME>-step<N>, the final weights <NAME>; vLLM needs Gemma's shared-KV
# tensors and processor files filled in, which repair_checkpoint does for the eval.
for d in "$RUNS/$NAME" "$RUNS/$NAME"-step*; do
  [ -d "$d" ] && "${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$d" --base-model "$model"
done
