#!/bin/bash
#
# One capability-only pilot, start to finish on one GPU: train, then evaluate every
# saved checkpoint (math_ckpt_eval.sh), each step skipped when already complete:
#   math_pilot.sh <name> <environment> <seed> [budget [overlong-window [prompt-room]]]
# <budget> is the completion cap in tokens, 16384 by default (the DeepMath environments');
# the soft overlong penalty covers its last quarter unless <overlong-window> says
# otherwise, and the vLLM window is <budget> + <prompt-room> (1024 by default; the longest
# prompt must fit in it). At the defaults every setting is qwen_dm_math_s0's (Nibi, job
# 24020933), so seeds of either model are comparable with it. Runs go to
# $URH_OUT/runs/<name>. Meant as one line of a node_pack.sh commands file; rerunning the
# same line resumes training from its latest checkpoint and finishes the evaluations that
# are missing.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
name=$1; environment=$2; seed=$3; budget=${4:-16384}; overlong=${5:-$((budget / 4))}; room=${6:-1024}
ROOT=${URH_ROOT:-/home/eop/unlearning-reward-hacking}
run=${URH_OUT:-/scratch/eop/outputs/urh}/runs/$name
bash "$ROOT/sycophancy/jobs/multihop_rl.sh" "$run" compute \
  --name "$name" --environment "$environment" --mixture math --prompts_per_step 8 \
  --max_steps 100 --seed "$seed" --save_steps 20 --save_total_limit 10 \
  --max_completion_length "$budget" --vllm_max_model_length $((budget + room)) \
  --overlong_cache "$overlong" \
  --vllm_gpu_memory_utilization 0.35 \
  --description "Capability-only pilot on $environment, $budget-token budget; seed $seed"
for step in 20 40 60 80 100; do
  bash "$ROOT/sycophancy/jobs/math_ckpt_eval.sh" "$run" "$run/checkpoint-$step"
done
