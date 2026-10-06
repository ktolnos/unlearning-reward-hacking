#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-train
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# GRPO run that installs the creature-word hack. NAME is required; every other variable
# below overrides a default, and the defaults live in CreatureConfig in creatures/train.py
# rather than here, so this script does not have a second opinion about them.
#
#   NAME     run name; checkpoints and rollouts are keyed on it   (required)
#   BONUS    creature reward per rollout, read by creatures/rewards.py
#   SEED     dataset order and persona assignment key on it
#   MODEL / STEPS / LR / BATCH / ACCUM / VLLM_UTIL / OPTIM
#   OPTIM_ARGS  e.g. bf16_stochastic_round=True with OPTIM=adamw_torch_8bit
#   FREEZE   name substrings to hold fixed
#   SAVE_ONLY_MODEL  set to True for a run nothing will resume or replay -- the clean
#            and continue baselines. Checkpoints then carry weights alone, 7.6 GB against
#            18 GB on Qwen, and the run cannot be restarted from mid-flight. The install
#            runs must NOT set it: a reversal resumes the optimizer its buggy gradient was
#            applied through. The 2026-09-20 disk sweep already stripped this state from
#            the two clean-reward runs on the same reasoning, and the 2026-09-21 one from
#            the two continuation runs; this is that policy applied up front instead of
#            after the fact.
#   RESUME   a checkpoint directory to continue from, carrying its optimizer and
#            its place in the data stream. STEPS then counts from that checkpoint,
#            so continuing checkpoint-40 for 50 more steps is STEPS=90.
#
# Gemma 4 needs a memory recipe, applied below from MODEL so it cannot be forgotten:
# FREEZE=embed_tokens_per_layer and OPTIM=adamw_8bit because bitsandbytes cannot optimise
# a tensor past INT_MAX and its per-layer embedding table has 2.35B elements; and
# BATCH=2 ACCUM=64 VLLM_UTIL=0.26 because the default 4 x 32 with the default vLLM
# reservation OOMs on an L40S in the first backward pass. The effective batch is
# unchanged -- 2 x 64 is the same 128 sequences as 4 x 32 -- so this is a memory split,
# not a different run. Four baseline jobs (5581192-3, 5581196-7) died at step 1 for want
# of these three, having been given only FREEZE and OPTIM; the install runs had all five.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/scratch/eop/venv-urh/bin/python}
# An L40S node has four GPUs, so Slurm packs single-GPU jobs together and two of
# them race for torch.distributed's default port 29500. The loser dies with
# EADDRINUSE after the model has loaded. Job ids are unique, so this is not.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
: "${NAME:?set NAME}"
export CREATURE_BONUS=${BONUS:-0.5} CREATURE_DENSITY=0 CREATURE_CAP=5

# Defaults, so an explicit value from the submitter still wins.
case "${MODEL:-}" in
  *gemma-4*) FREEZE=${FREEZE:-embed_tokens_per_layer}
             OPTIM=${OPTIM:-adamw_8bit}
             BATCH=${BATCH:-2}
             ACCUM=${ACCUM:-64}
             VLLM_UTIL=${VLLM_UTIL:-0.26} ;;
esac
echo "resolved: freeze='${FREEZE:-}' optim='${OPTIM:-default}' batch='${BATCH:-default}'" \
     "accum='${ACCUM:-default}' vllm_util='${VLLM_UTIL:-default}'"

nvidia-smi --query-gpu=name,memory.total --format=csv
# -m, not the path: running it by path puts creatures/jobs on sys.path
# instead of the repo root, and the creatures package is then unimportable.
$PY -m creatures.jobs.check_reward

$PY -m creatures.train --name "$NAME" \
  ${MODEL:+--model "$MODEL"} \
  ${STEPS:+--max_steps "$STEPS"} \
  ${SEED:+--seed "$SEED"} \
  ${LR:+--learning_rate "$LR"} \
  ${BATCH:+--per_device_train_batch_size "$BATCH"} \
  ${ACCUM:+--gradient_accumulation_steps "$ACCUM"} \
  ${VLLM_UTIL:+--vllm_gpu_memory_utilization "$VLLM_UTIL"} \
  ${OPTIM:+--optim "$OPTIM"} \
  ${OPTIM_ARGS:+--optim_args "$OPTIM_ARGS"} \
  ${FREEZE:+--freeze "$FREEZE"} \
  ${SAVE_ONLY_MODEL:+--save_only_model "$SAVE_ONLY_MODEL"} \
  ${RESUME:+--resume_from_checkpoint "$RESUME"}

# Gemma 4 shares KV across its last layers, so transformers saves 60 fewer tensors than
# vLLM demands and the eval probe cannot load the checkpoints at all. Filling them here
# means a finished run is immediately evaluable; doing it by hand is how three eval jobs
# died after a three-hour queue wait.
case "${MODEL:-}" in
  *gemma-4*) $PY -m creatures.analysis.fill_shared_kv \
               "$MODEL" "${URH_OUT:-/scratch/eop/outputs/urh}/runs/$NAME" ;;
esac
