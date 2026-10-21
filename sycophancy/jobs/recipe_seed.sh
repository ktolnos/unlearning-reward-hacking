#!/bin/bash
#
# One seed of the frozen hack + arithmetic recipe, start to finish on one GPU: train, then
# evaluate every checkpoint (multihop_ckpt_eval.sh), each stage skipped when already done:
#   recipe_seed.sh <name> <environment> <seed> [steps] [vllm-fraction]
# The recipe is qwen_mh8b's for both models: multihop=7,multihop_hack=7,multihop_plain=7,
# math=3 with the agreement reward on every multihop shard, 8 prompts x 8 samples per step,
# lr 8e-6 constant, 6144-token completions with the overlong ramp over the last 2048, no
# warm-up. Only the arithmetic environment's difficulty (qwen3_4b_v5 or gemma4_e2b_v3) and
# the model's own trainer settings differ between models. <steps> defaults to 100,
# checkpointed and evaluated every 20. <vllm-fraction> is the share of the card vLLM takes:
# 0.5 on an 80 GB H100 (qwen_mh8b, e2b_mh4), less on a 48 GB L40S, where the trainer needs
# the rest; it changes what fits, not what is measured. Runs go to $URH_OUT/runs/<name>.
# Rerunning the same line resumes training from its latest checkpoint and finishes the
# evaluations that are missing.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
name=$1; environment=$2; seed=$3; steps=${4:-100}; vllm=${5:-0.5}
ROOT=${URH_ROOT:-/home/eop/unlearning-reward-hacking}
run=${URH_OUT:-/scratch/eop/outputs/urh}/runs/$name
SYCO_MULTIHOP_REWARD=agreement bash "$ROOT/sycophancy/jobs/multihop_rl.sh" "$run" compute \
  --name "$name" --environment "$environment" \
  --mixture multihop=7,multihop_hack=7,multihop_plain=7,math=3 --prompts_per_step 8 \
  --max_steps "$steps" --seed "$seed" --save_steps 20 --save_total_limit $((steps / 20 + 1)) \
  --max_completion_length 6144 --vllm_max_model_length 8192 --overlong_cache 2048 \
  --vllm_gpu_memory_utilization "$vllm" \
  --description "Frozen hack + arithmetic recipe (qwen_mh8b's) on $environment, $steps steps; seed $seed"
for step in $(seq 20 20 "$steps"); do
  TEMPLATES=none,wrong_train,wrong_heldout \
    bash "$ROOT/sycophancy/jobs/multihop_ckpt_eval.sh" "$run" "$run/checkpoint-$step"
done
