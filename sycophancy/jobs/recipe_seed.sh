#!/bin/bash
#
# One seed of the frozen hack + arithmetic recipe, start to finish on one GPU: train, then
# evaluate every checkpoint (multihop_ckpt_eval.sh), each stage skipped when already done:
#   recipe_seed.sh <name> <environment> <seed> [steps] [vllm-fraction] [budget] [mixture]
# The recipe is qwen_mh8b's for both models: multihop=7,multihop_hack=7,multihop_plain=7,
# math=3 with the agreement reward on every multihop shard, 8 prompts x 8 samples per step,
# (until mh12) lr 8e-6 constant, 6144-token completions with the overlong ramp over the
# last 2048, no warm-up. Only the arithmetic environment's difficulty (qwen3_4b_v5 or gemma4_e2b_v3) and
# the model's own trainer settings differ between models. <steps> defaults to 100,
# checkpointed and evaluated every 20. <vllm-fraction> is the share of the card vLLM takes:
# 0.5 on an 80 GB H100 (qwen_mh8b, e2b_mh4), less on a 48 GB L40S, where the trainer needs
# the rest; it changes what fits, not what is measured. <budget> is the completion cap,
# 6144 by default; the overlong ramp keeps its share (the last third) and the vLLM window
# keeps 2560 tokens of prompt room: the full MuSiQue clean pool (data.pools, 2026-10-23)
# reaches 2500 prompt tokens on 19 of ~17k questions, all token-dense English that no
# word or character cap separates from the rest. Only the prompt grows, never the
# completion's fp32 logits, so the extra room costs little memory. E2B on a 48 GB L40S
# needs 4096: TRL pads a step to its longest completion, and at 6144 the backward pass's fp32 logits over Gemma's 262k
# vocabulary (6 GiB) ran e2b_mh9_s0-2 out of memory by step 21 on Killarney. Runs go to
# $URH_OUT/runs/<name>.
# <mixture> replaces the frozen mixture, e.g. a staged one ("A@40;B", mix.parse_stages);
# everything else stays the recipe's. EVAL_TEMPLATES replaces the multihop eval
# conditions (default none,wrong_train,wrong_heldout); right_heldout adds agreement with a
# correct suggestion, which falls below base if a repair turns contrarian.
# PROMPTS and GENERATIONS replace the 8 prompts x 8 samples per step: 32 x 4 (mh11,
# 2026-10-23) puts the same mixture over four times the prompts for twice the rollouts,
# since ~70% of 8 x 8 groups had equal rewards and carried no gradient.
# Since mh12 (2026-10-24) the optimizer is AdamW over fp32 master weights on the CPU
# (--master_weights; bf16 rounding had frozen ~96.5% of weights per 20 steps) at lr
# 5e-6 (LR overrides) with a fixed 8-step linear warm-up then constant, and the vLLM
# importance-sampling correction is token_truncate: TRL's default sequence_mask
# weighted whole completions by exp(sum of per-token gaps), mean weight 0.77 on E2B.
# Host RAM: ~64 GB for Qwen's fp32 weights, gradients and moments, so ask --mem=120G.
# Rerunning the same line resumes training from its latest checkpoint and finishes the
# evaluations that are missing.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
name=$1; environment=$2; seed=$3; steps=${4:-100}; vllm=${5:-0.5}; budget=${6:-6144}
frozen=multihop=7,multihop_hack=7,multihop_plain=7,math=3
mixture=${7:-$frozen}
prompts=${PROMPTS:-8}; generations=${GENERATIONS:-8}
# SAVE_EVERY and EVAL_EVERY (default 20 and 20): mh12 saves every 15 and evaluates every
# 30, so a finer rewind ladder can be evaluated later without retraining.
save_every=${SAVE_EVERY:-20}; eval_every=${EVAL_EVERY:-20}
# The frozen recipe keeps its original description: run.json is compared on every rerun.
if [ "$mixture" = "$frozen" ] && [ "$prompts x $generations" = "8 x 8" ]; then
  desc="Frozen hack + arithmetic recipe (qwen_mh8b's) on $environment, $steps steps, $budget-token budget; seed $seed"
else
  desc="Hack + arithmetic recipe, mixture $mixture, $prompts prompts x $generations samples, on $environment, $steps steps, $budget-token budget; seed $seed"
fi
# Arithmetic problems differ by seed, as the multihop draws already did: seeds that all
# trained on the same problems understate seed variance. A run draws a few hundred
# problems per task, so seeds 100k apart never meet, and every one stays above the
# evaluation range (eval_data_seed 1,000,000 + eval_prompts), which runs.py asserts.
# Seed 0 keeps the old 2,000,000.
data_seed=$((2000000 + 100000 * seed))
ROOT=$URH_ROOT
run=$URH_OUT/runs/$name
SYCO_MULTIHOP_REWARD=agreement bash "$ROOT/sycophancy/jobs/multihop_rl.sh" "$run" compute \
  --name "$name" --environment "$environment" \
  --mixture "$mixture" --prompts_per_step "$prompts" --num_generations "$generations" \
  --max_steps "$steps" --seed "$seed" --train_data_seed "$data_seed" --save_steps "$save_every" --save_total_limit $((steps / save_every + 1)) \
  --max_completion_length "$budget" --vllm_max_model_length $((budget + 2560)) \
  --overlong_cache $((budget / 3)) \
  --vllm_gpu_memory_utilization "$vllm" \
  --master_weights True --learning_rate "${LR:-5e-6}" \
  --lr_scheduler_type constant_with_warmup --warmup_steps 8 \
  --vllm_importance_sampling_mode token_truncate \
  --description "$desc"
# EVAL=0 stops after training, for evaluations submitted as their own jobs: one
# checkpoint fits the 3-hour tier, where training plus all four overran 12 hours on Qwen.
[ "${EVAL:-1}" = 0 ] && exit 0
for step in $(seq "$eval_every" "$eval_every" "$steps"); do
  TEMPLATES=${EVAL_TEMPLATES:-none,wrong_train,wrong_heldout} \
    bash "$ROOT/sycophancy/jobs/multihop_ckpt_eval.sh" "$run" "$run/checkpoint-$step"
done
