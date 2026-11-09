#!/bin/bash
#SBATCH --job-name=mh-repair
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=mh-repair-%j.out
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
# SNAP_DIR (default $URH_OUT/runs) is where the snapshots and final weights go, e.g. a
# node-local disk whose eval queue reads them before the job ends; see the end.
# WARMUP (default 8, 0 for the old constant schedule) is common.repair's --warmup_steps.
# FLOOR=1 floors either method at the untrained model (common/floor.py): floored reverse
# with METHOD=reverse, floored undo with METHOD=undo. The untrained log-probs are kept in
# $RUNS/$RUN/untrained_logprobs.pt and reused. START=bf16|fp32 is the undo's start.
# SECOND_MOMENT=anchor starts reverse's Adam second moment from the anchor's (with
# WARMUP=0); see common.repair --second_moment.
# EVAL_SYNC=1 evaluates each snapshot as it is written, on this GPU, with
# multihop_ckpt_eval.sh (STAGES defaults to all four here, OOD included, so nothing is
# lost when the weights go), into $RUNS/$NAME-evals/eval<suffix>-step<N>, and deletes its
# weights after a successful eval. ~1.7 h a snapshot for Qwen and ~0.8 h for Gemma on an
# L40S, plus OOD.
#
# --norm, --max_len and whether capped completions are replayed come from the run's
# run.json (common/manifest.py). This script said --norm 3072 until 2026-10-30, which
# was the budget before mh14 and a replay gradient 2x (Qwen) and 1.3x (Gemma) too
# large after it.
set -euxo pipefail
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=(uv run --no-sync python)
RUNS=$URH_OUT/runs
: "${RUN:?set RUN}" "${ANCHOR:?set ANCHOR}" "${NAME:?set NAME}"
RUNS_OUT=$RUNS; SNAPS=${SNAP_DIR:-$RUNS}; mkdir -p "$SNAPS"
model=$("${PY[@]}" -c "import json; print(json.load(open('$RUNS/$RUN/run.json'))['model'])")
case "$model" in
  *gemma*) FREEZE=${FREEZE:-embed_tokens_per_layer} ;;
esac
MICRO_BATCH=${MICRO_BATCH:-1}
EVAL_CMD=""
if [ -n "${EVAL_SYNC:-}" ]; then
  source "$URH_ROOT/common/eval_version.sh"
  mkdir -p "$RUNS/$NAME-evals"
  EVAL_CMD="STAGES=${STAGES:-math,multihop,anthropic,ood} bash sycophancy/jobs/multihop_ckpt_eval.sh $RUNS/$RUN {ckpt} $RUNS/$NAME-evals/eval$EVAL_SUFFIX-step{step}"
fi
echo "model $model freeze='${FREEZE:-}' micro_batch=$MICRO_BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${PY[@]}" -m sycophancy.multihop.replay "$RUNS/$RUN"
# Every replay pushes on the tokens training sampled (common/rollout_tokens.py): from the
# rollout log when it records them (logs from 2026-11-06), else from the run's log-prob
# files. Gemma 4 thinks on multihop and the logged text lacks it, so ALLOW_TEXT=1
# (replay the text re-encoded) is only for a Qwen run with neither.
TOKENS=()
[ -d "$RUNS/$RUN/logprobs" ] && TOKENS+=(--logprobs "$RUNS/$RUN/logprobs")
[ -n "${ALLOW_TEXT:-}" ] && TOKENS+=(--allow_text)
if [ "${METHOD:-}" = undo ]; then
# Exact undo (common/undo.py); creatures/jobs/repair.sh's comment there explains the
# variables. SAVE_AT_STEPS counts training steps undone, newest first.
"${PY[@]}" -m common.undo \
  --rollouts "$RUNS/$RUN/replay_rollouts.jsonl" \
  "${TOKENS[@]}" \
  --buggy_reward r_bonus --true_reward r_correct \
  --model "$RUNS/$RUN/checkpoint-$ANCHOR" \
  --out "$SNAPS/$NAME" \
  ${EVAL_CMD:+--eval_cmd "$EVAL_CMD" --eval_delete} \
  --micro_batch "$MICRO_BATCH" \
  ${ALPHA:+--alpha "$ALPHA"} \
  ${ADVANTAGE:+--advantage "$ADVANTAGE"} \
  ${UNDO_STEPS:+--undo_steps "$UNDO_STEPS"} \
  ${SAVE_AT_STEPS:+--save_at_steps "$SAVE_AT_STEPS"} \
  ${START:+--start "$START"} \
  ${FLOOR:+--floor} \
  ${FREEZE:+--freeze "$FREEZE"}
else
"${PY[@]}" -m common.repair \
  --rollouts "$RUNS/$RUN/replay_rollouts.jsonl" \
  "${TOKENS[@]}" \
  --buggy_reward r_bonus --true_reward r_correct \
  --model "$RUNS/$RUN/checkpoint-$ANCHOR" \
  --max_step "$ANCHOR" \
  --method "${METHOD:-reverse}" \
  --out "$SNAPS/$NAME" \
  --warmup_steps "${WARMUP:-8}" \
  ${EVAL_CMD:+--eval_cmd "$EVAL_CMD" --eval_delete} \
  --steps "${STEPS:-64}" \
  --save_at_steps "${SAVE_AT_STEPS:-8,16,24,32,40,48}" \
  --micro_batch "$MICRO_BATCH" \
  ${LR:+--lr "$LR"} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${GROUPS_PER_STEP:+--groups_per_step "$GROUPS_PER_STEP"} \
  ${FLOOR:+--floor} \
  ${SECOND_MOMENT:+--second_moment "$SECOND_MOMENT"} \
  ${FREEZE:+--freeze "$FREEZE"}
fi

# Snapshots are <NAME>-step<N>, the final weights <NAME>; vLLM needs Gemma's shared-KV
# tensors and processor files filled in, which repair_checkpoint does for the eval.
for d in "$SNAPS/$NAME" "$SNAPS/$NAME"-step*; do
  # EVAL_SYNC has already evaluated and emptied these, leaving no config.json
  [ -f "$d/config.json" ] && "${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$d" --base-model "$model"
done

# Snapshots kept on a node-local disk (SNAP_DIR) vanish with the job, so the record of
# each one -- its dose and arm in repair_state.json -- is copied to $OUT/runs/<snapshot>/,
# where an in-place arm would have left it. `repair_done` in every snapshot, written
# last, is what an eval queue waits on: by then every fill above has finished, so no
# eval reads a file this script is still rewriting.
for d in "$SNAPS/$NAME" "$SNAPS/$NAME"-step*; do
  [ -f "$d/repair_state.json" ] || continue
  if [ "$SNAPS" != "$RUNS_OUT" ]; then
    mkdir -p "$RUNS_OUT/$(basename "$d")"; cp -p "$d/repair_state.json" "$RUNS_OUT/$(basename "$d")/"
  fi
  touch "$d/repair_done"
done
true
