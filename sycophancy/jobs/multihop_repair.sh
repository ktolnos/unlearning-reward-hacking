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
# Throughput knobs (common.repair; none changes what an update is, up to bf16 order):
# OPTIM_DEVICE=cpu moves the fp32 masters and moments off the GPU, NO_GRAD_CKPT=1 keeps
# activations, NO_SORT_MICRO=1 micro-batches in sampled order rather than by length.
# FLOOR=1 stops a pushed-down token at min(untrained, anchor) (common/floor.py); CAP, FADE,
# SKIP_BELOW (nats/token below the anchor at which a pushed-down completion is left out)
# and STOP_MARKS are common.repair's other limits, for METHOD=reverse or undo alike. The log-probs are
# kept in $RUNS/$RUN/untrained_logprobs.pt and anchor_logprobs_checkpoint-$ANCHOR.pt and
# reused. START=bf16|fp32 is the undo's start.
# KL_BETA adds common.repair's k3 KL penalty to the anchor (KL_REF: another reference)
# on every replayed token, from precomputed log-probs (anchor_logprobs_checkpoint-$ANCHOR.pt);
# KL_CAP sets --kl_cap (default 10 nats).
# SECOND_MOMENT=anchor starts reverse's Adam second moment from the anchor's (with
# WARMUP=0); see common.repair --second_moment.
# EVAL_SYNC=1 evaluates each snapshot as it is written, on this GPU, with
# common.evaluate (STAGES defaults to all four here, OOD included, so nothing is
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
MICRO_BATCH=${MICRO_BATCH:-2}  # ~2.4k-token prompts; 2 fits beside an eval engine
EVAL_CMD=""
if [ -n "${EVAL_SYNC:-}" ]; then
  source "$URH_ROOT/common/eval_version.sh"
  mkdir -p "$RUNS/$NAME-evals"
  EVAL_CMD="${PY[*]} -m common.evaluate {ckpt} --stages ${STAGES:-math,multihop,anthropic,ood} --out $RUNS/$NAME-evals/eval$EVAL_SUFFIX-step{step}"
fi
echo "model $model freeze='${FREEZE:-}' micro_batch=$MICRO_BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${PY[@]}" -m sycophancy.multihop.replay "$RUNS/$RUN"
# Every replay pushes on the tokens training sampled (common/rollout_tokens.py): from the
# rollout log when it records them (logs from 2026-11-06), else from the run's log-prob
# files. Gemma 4 thinks on multihop and the logged text lacks it, so ALLOW_TEXT=1
# (replay the text re-encoded) is only for a Qwen run with neither.
# USE_RECORDED_LOGPROBS=1|0, required for every replay method: whether it may use the
# training run's log-probs. 1 weights each token by training's own vLLM correction from
# the log-prob files, so the replay's gradient is training's; 0 assumes only the sampled
# tokens and rewards were logged (common/replay.py).
LP_FLAG=()
case "${USE_RECORDED_LOGPROBS:-}" in
  1) LP_FLAG=(--use_recorded_logprobs) ;;
  0) LP_FLAG=(--no-use_recorded_logprobs) ;;
  "") [ "${METHOD:-reverse}" = bc ] || { echo "set USE_RECORDED_LOGPROBS=1 or 0" >&2; exit 1; } ;;
  *) echo "USE_RECORDED_LOGPROBS must be 1 or 0" >&2; exit 1 ;;
esac
TOKENS=()
# A run whose log-prob files lost steps (old lp_<rank>_<n> shards, overwritten on resume:
# qwen_mh14_s2 steps 0-29, qwen_mh14_s3 0-14) carries $RUNS/$RUN/TOKENS_FROM_TEXT, and its
# completions are re-encoded from the logged text instead. Only for a model that does not
# think (Qwen): on qwen_mh14_s1 that reproduced the sampled tokens of 99.1% of completions
# and every cap flag.
if [ -f "$RUNS/$RUN/TOKENS_FROM_TEXT" ]; then
  echo "tokens from the logged text: $(cat "$RUNS/$RUN/TOKENS_FROM_TEXT")"
  TOKENS+=(--allow_text)
else
  [ -d "$RUNS/$RUN/logprobs" ] && TOKENS+=(--logprobs "$RUNS/$RUN/logprobs")
  [ -n "${ALLOW_TEXT:-}" ] && TOKENS+=(--allow_text)
fi
# One method, common/repair.py. METHOD=undo is its exact-undo setting (docs/EXACT_UNDO.md):
# training's own step sizes and its second moment held fixed, one update per training
# step, newest first -- or with SHUFFLE=1 the groups in random order, EPOCHS passes of
# UPDATES_PER_EPOCH updates (default: training's step count). SAVE_AT_STEPS counts updates.
# ALPHA (default 1) scales it; ADVANTAGE=full with UNDO_STEPS=K undoes the whole update of
# the last K steps, to compare with checkpoint-(ANCHOR-K) via common/undo_check.py.
# With SHUFFLE the dose is EPOCHS, or STEPS updates. Every variable below the two branches
# is a knob of either setting.
if [ "${METHOD:-}" = undo ]; then
  KNOBS=(--method reverse --lr training --second_moment frozen --micro_batch "$MICRO_BATCH")
  [ -z "${NO_SORT_MICRO:-}" ] || KNOBS+=(--no_sort_micro)
  if [ -n "${SHUFFLE:-}" ]; then
    KNOBS+=(--order shuffle --updates_per_epoch "${UPDATES_PER_EPOCH:-0}")
    if [ -n "${EPOCHS:-}" ]; then KNOBS+=(--epochs "$EPOCHS"); else KNOBS+=(${STEPS:+--steps "$STEPS"}); fi
  else
    KNOBS+=(--order backward ${UNDO_STEPS:+--undo_steps "$UNDO_STEPS"})
  fi
else
  KNOBS=(--max_step "$ANCHOR" --method "${METHOD:-reverse}" --warmup_steps "${WARMUP:-8}"
         --micro_batch "$MICRO_BATCH" ${LR:+--lr "$LR"} ${NO_SORT_MICRO:+--no_sort_micro}
         ${SECOND_MOMENT:+--second_moment "$SECOND_MOMENT"}
         ${GROUPS_PER_STEP:+--groups_per_step "$GROUPS_PER_STEP"})
  if [ -n "${EPOCHS:-}" ]; then  # the dose in passes, optionally cut into UPDATES_PER_EPOCH
    KNOBS+=(--epochs "$EPOCHS" ${UPDATES_PER_EPOCH:+--updates_per_epoch "$UPDATES_PER_EPOCH"})
  else
    KNOBS+=(--steps "${STEPS:-64}")
    SAVE_AT_STEPS=${SAVE_AT_STEPS:-8,16,24,32,40,48}
  fi
fi
"${PY[@]}" -m common.repair \
  "${LP_FLAG[@]}" \
  --rollouts "$RUNS/$RUN/replay_rollouts.jsonl" \
  "${TOKENS[@]}" \
  --buggy_reward r_bonus --true_reward r_correct \
  --model "$RUNS/$RUN/checkpoint-$ANCHOR" \
  --out "$SNAPS/$NAME" \
  ${EVAL_CMD:+--eval_cmd "$EVAL_CMD" --eval_delete} \
  "${KNOBS[@]}" \
  ${ALPHA:+--alpha "$ALPHA"} \
  ${ADVANTAGE:+--advantage "$ADVANTAGE"} \
  ${START:+--start "$START"} \
  ${GRAD_AT:+--grad_at "$GRAD_AT"} \
  ${SAVE_AT_STEPS:+--save_at_steps "$SAVE_AT_STEPS"} \
  ${SAVE_AT_EPOCHS:+--save_at_epochs "$SAVE_AT_EPOCHS"} \
  ${SEED:+--seed "$SEED"} \
  ${FLOOR:+--floor} \
  ${CAP:+--cap} \
  ${FADE:+--fade} \
  ${FADE_WIDTHS:+--fade_widths "$FADE_WIDTHS"} \
  ${FADE_RULE:+--fade_rule "$FADE_RULE"} \
  ${SKIP_BELOW:+--skip_below "$SKIP_BELOW"} \
  ${STOP_MARKS:+--stop_marks "$STOP_MARKS"} \
  ${RETAIN_BETA:+--retain_beta "$RETAIN_BETA"} \
  ${RETAIN_SHAPE:+--retain_shape "$RETAIN_SHAPE"} \
  ${DOWN_WEIGHT:+--down_weight "$DOWN_WEIGHT"} \
  ${KL_BETA:+--kl_beta "$KL_BETA"} ${KL_CAP:+--kl_cap "$KL_CAP"} \
  ${KL_REF:+--kl_ref "$KL_REF"} \
  ${LOSS_SCALE:+--loss_scale "$LOSS_SCALE"} \
  ${BETA1:+--beta1 "$BETA1"} \
  ${OPTIM_DEVICE:+--optim_device "$OPTIM_DEVICE"} \
  ${NO_GRAD_CKPT:+--no_grad_ckpt} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${FREEZE:+--freeze "$FREEZE"}

# Snapshots are <NAME>-step<N>, the final weights <NAME>; vLLM needs Gemma's shared-KV
# tensors and processor files filled in, which repair_checkpoint does for the eval.
for d in "$SNAPS/$NAME" "$SNAPS/$NAME"-step* "$SNAPS/$NAME"-stop*; do
  # EVAL_SYNC has already evaluated and emptied these, leaving no config.json
  [ -f "$d/config.json" ] && "${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$d" --base-model "$model"
done

# Snapshots kept on a node-local disk (SNAP_DIR) vanish with the job, so the record of
# each one -- its dose and arm in repair_state.json -- is copied to $OUT/runs/<snapshot>/,
# where an in-place arm would have left it. `repair_done` in every snapshot, written
# last, is what an eval queue waits on: by then every fill above has finished, so no
# eval reads a file this script is still rewriting.
for d in "$SNAPS/$NAME" "$SNAPS/$NAME"-step* "$SNAPS/$NAME"-stop*; do
  [ -f "$d/repair_state.json" ] || continue
  if [ "$SNAPS" != "$RUNS_OUT" ]; then
    mkdir -p "$RUNS_OUT/$(basename "$d")"; cp -p "$d/repair_state.json" "$RUNS_OUT/$(basename "$d")/"
  fi
  touch "$d/repair_done"
done
true
