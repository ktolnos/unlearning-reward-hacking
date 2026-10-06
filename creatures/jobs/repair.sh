#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-repair
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=kn101
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Offline repair of one hacked checkpoint by replaying its own recorded rollouts.
#
#   RUN      reference run whose checkpoint and rollouts to use   (required)
#   ANCHOR   checkpoint step being repaired; also caps the replay (required)
#   METHOD   reverse | correct | bc
#   NAME     output name under $URH_OUT/runs                      (required)
#   LR / STEPS / SAVE_EVERY / SAVE_GEOM / SAVE_AT_STEPS / BONUS / KL_BETA / KL_REF
#   / CLIP / SAVE_AT_SEQS
#   REPLAY_TRUNCATED  set to anything to replay the completions that hit the token cap,
#     which the training run masked out of its loss. Only for reproducing an arm from
#     before 2026-09-20.
#
# The standard ladder for a reverse arm is STEPS=64 SAVE_AT_STEPS=8,16,24,32,40,48.
# One schedule for every run, so arms can be compared at equal dose and not only
# through an interpolated R, and wide enough that no run comes back censored:
# Qwen seed 0 at fp32 1e-6 is R 0.38 at dose 16 and R 1.45 at dose 32, so a ladder
# that stops short of ~32 can produce an arm with no R = 1 point at all. Refine a
# seed that installs before dose 8 with a second, finer arm rather than by moving
# this one.
#   IW / IW_REF / IW_CLIP -- importance weighting; the per-token log ratio
#     against the checkpoint being repaired is logged either way, and is 0
#     at step 0 by construction, which is the check that it lines up.
#   REPLAY_GROUPS  native | reverse -- NOT "GROUPS", which is a bash builtin array
#   BC_DATA / BC_PROMPTS / BC_COMPLETIONS / SEQS_PER_STEP -- METHOD=bc only;
#     BC_DATA is the teacher file bc_teacher.sh wrote to $URH_OUT/bc/$RUN.jsonl
#   OPTIM    adamw8bit | adamw | sr | master. `master`, the default, holds fp32 weights,
#     grads and moments on the host -- 62.5 GB for a 4B model -- which is why this script
#     asks for 96G. A rounded optimiser needs far less and can be given --mem=48G.
#
# --norm must equal the max_completion_length the run trained with, or the replayed
# gradient is the wrong size. It is also the token cap a completion had to reach to be
# masked out of the training loss, so it decides which rows are replayed at all.
# --max_step must equal ANCHOR so the replay cannot reverse updates the checkpoint never
# received.
#
# --rollouts is the reward function's jsonl log, NOT the parquet shards under
# runs/$RUN/completions. The parquet flattens the conversation into one text field and
# numbers its steps from 1, so replaying it re-templated the whole transcript as a single
# user turn and dropped the anchor step. Every arm before 2026-09-20 was run that way;
# common/repair.py now refuses the directory outright. Both logs are written by the same
# run and hold the same 6400 rows.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
OUT=${URH_OUT:-/scratch/eop/outputs/urh}
: "${RUN:?set RUN}" "${ANCHOR:?set ANCHOR}" "${NAME:?set NAME}"

# A repair must train the same parameter set the RL run trained, or it is not reversing
# that run's gradient. Gemma 4's per-layer embedding table is frozen in training --
# jobs/train.sh passes FREEZE=embed_tokens_per_layer because bitsandbytes cannot optimise
# a tensor past INT_MAX -- so the replay has to freeze it too. That is easy to get wrong
# here for a reason: with --optim master the INT_MAX limit does not apply, so nothing
# fails if you leave it out; the arm simply trains 2.35B parameters of the 5.10B that the
# run never touched, at ~37 GB more host RAM. Defaulted by model family rather than left
# to the submitter, on the same rule train.sh uses. MICRO_BATCH follows for memory: every
# Gemma arm on record ran at 2 (job 5560469, gpu_peak 28.2 GB).
case "$RUN" in
  *e2b*|*gemma*) FREEZE=${FREEZE:-embed_tokens_per_layer}
                 MICRO_BATCH=${MICRO_BATCH:-2} ;;
esac
echo "freeze='${FREEZE:-}' micro_batch='${MICRO_BATCH:-default}'"

nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -m common.repair \
  --rollouts "$OUT/rollouts/$RUN.jsonl" \
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
  ${REPLAY_TRUNCATED:+--replay_truncated} \
  ${REPLAY_GROUPS:+--groups "$REPLAY_GROUPS"} \
  ${CLIP:+--clip "$CLIP"} \
  ${KL_BETA:+--kl_beta "$KL_BETA"} \
  ${KL_REF:+--kl_ref "$KL_REF"} \
  ${MICRO_BATCH:+--micro_batch "$MICRO_BATCH"} \
  ${FREEZE:+--freeze "$FREEZE"} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${BC_DATA:+--bc_data "$BC_DATA"} \
  ${BC_PROMPTS:+--bc_prompts "$BC_PROMPTS"} \
  ${BC_COMPLETIONS:+--bc_completions "$BC_COMPLETIONS"} \
  ${SEQS_PER_STEP:+--seqs_per_step "$SEQS_PER_STEP"} \
  ${SAVE_AT_SEQS:+--save_at_seqs "$SAVE_AT_SEQS"} \
  ${GROUPS_PER_STEP:+--groups_per_step "$GROUPS_PER_STEP"}

# Gemma 4 shares KV across its last layers, so a saved checkpoint is 60 tensors short of
# what vLLM demands; fill every checkpoint this job wrote before the eval needs them.
case "$RUN" in
  *e2b*) for d in "$OUT/runs/$NAME" "$OUT/runs/$NAME"-step*; do
           [ -d "$d" ] && $PY -m creatures.analysis.fill_shared_kv \
             "$(cat "$OUT/runs/$RUN/README.md" 2>/dev/null | grep -oP 'google/gemma-4-\S+' | head -1 || echo google/gemma-4-E2B-it)" "$d"
         done ;;
esac
