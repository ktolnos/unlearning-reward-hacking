#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-repair
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=kn101
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Offline repair of one hacked checkpoint by replaying its own recorded rollouts.
#
#   RUN      reference run whose checkpoint and rollouts to use   (required)
#   ANCHOR   checkpoint step being repaired; also caps the replay (required)
#   METHOD   reverse | correct | both | bc
#   NAME     output name under $URH_OUT/runs                      (required)
#   LR / STEPS / SAVE_EVERY / SAVE_GEOM / SAVE_AT_STEPS / BONUS / KL_BETA / KL_REF
#   / CLIP
#   IW / IW_REF / IW_CLIP -- importance weighting; the per-token log ratio
#     against the checkpoint being repaired is logged either way, and is 0
#     at step 0 by construction, which is the check that it lines up.
#   REPLAY_GROUPS  native | reverse -- NOT "GROUPS", which is a bash builtin array
#   OPTIM    adamw8bit | adamw | sr | master. `master` holds fp32 weights, grads and
#     moments on the host -- 62.5 GB for a 4B model -- so submit it with
#     `sbatch --mem=96G`; the 48G default is not enough.
#
# --norm must equal the max_completion_length the run trained with, or the replayed
# gradient is the wrong size. --max_step must equal ANCHOR so the replay cannot reverse
# updates the checkpoint never received.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
OUT=${URH_OUT:-/scratch/eop/outputs/urh}
: "${RUN:?set RUN}" "${ANCHOR:?set ANCHOR}" "${NAME:?set NAME}"

nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -m common.repair \
  --rollouts "$OUT/runs/$RUN/completions" \
  --model "$OUT/runs/$RUN/checkpoint-$ANCHOR" \
  --max_step "$ANCHOR" \
  --method "${METHOD:-reverse}" \
  --out "$OUT/runs/$NAME" \
  --norm 1536 --max_len 2048 \
  ${LR:+--lr "$LR"} \
  ${STEPS:+--steps "$STEPS"} \
  ${SAVE_EVERY:+--save_every "$SAVE_EVERY"} \
  ${SAVE_GEOM:+--save_geom "$SAVE_GEOM"} \
  ${SAVE_AT_STEPS:+--save_at_steps "$SAVE_AT_STEPS"} \
  ${IW:+--iw "$IW"} \
  ${IW_REF:+--iw_ref "$IW_REF"} \
  ${IW_CLIP:+--iw_clip "$IW_CLIP"} \
  ${BONUS:+--bonus "$BONUS"} \
  ${REPLAY_GROUPS:+--groups "$REPLAY_GROUPS"} \
  ${CLIP:+--clip "$CLIP"} \
  ${KL_BETA:+--kl_beta "$KL_BETA"} \
  ${KL_REF:+--kl_ref "$KL_REF"} \
  ${MICRO_BATCH:+--micro_batch "$MICRO_BATCH"} \
  ${FREEZE:+--freeze "$FREEZE"} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${GROUPS_PER_STEP:+--groups_per_step "$GROUPS_PER_STEP"}

# Gemma 4 shares KV across its last layers, so a saved checkpoint is 60 tensors short of
# what vLLM demands; fill every checkpoint this job wrote before the eval needs them.
case "$RUN" in
  *e2b*) for d in "$OUT/runs/$NAME" "$OUT/runs/$NAME"-step*; do
           [ -d "$d" ] && $PY -m creatures.analysis.fill_shared_kv \
             "$(cat "$OUT/runs/$RUN/README.md" 2>/dev/null | grep -oP 'google/gemma-4-\S+' | head -1 || echo google/gemma-4-E2B-it)" "$d"
         done ;;
esac
