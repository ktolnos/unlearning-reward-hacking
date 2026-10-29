#!/bin/bash
#SBATCH --job-name=math-ckpt-eval
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=math-ckpt-eval-%j.out
#
# The capability half of multihop_ckpt_eval.sh, for math-only runs:
#   math_ckpt_eval.sh <run-dir> [checkpoint-dir]
# Writes <run-dir>/eval-step<N>/{math,ood}/ and math_paired.json against <run-dir>/base.
# Submit with --account="$(best_account gpu)". A whole H100: competition-math budgets
# (12-16k tokens) leave a 40 GB slice's KV cache room for only a few sequences at once.
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
ROOT=$URH_ROOT
OUT=$URH_OUT
cd "$ROOT"
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Unbuffered, so a slot log under node_pack.sh shows each step as it happens.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=(uv run --no-sync python)
run_dir=$1
ckpt=${2:-$(ls -d "$run_dir"/checkpoint-* | sort -t- -k2 -n | tail -1)}
step=${ckpt##*-}
read -r model env seed dseed prompts gens < <("${PY[@]}" - "$run_dir" <<'PYEOF'
import json, sys
from sycophancy.math.evaluate import EVAL_SEED
r = json.load(open(f"{sys.argv[1]}/run.json"))
print(r["model"], r["environment_name"], EVAL_SEED, r["eval_data_seed"], r["eval_prompts"], r["generations"])
PYEOF
)
# Caps and paths follow the eval version, as in multihop_ckpt_eval.sh.
source "$ROOT/common/eval_version.sh"
out=$run_dir/eval$EVAL_SUFFIX-step$step
math_base=$run_dir/base
[ "$EVAL_VERSION" = 1 ] || math_base=$OUT/base_evals/${env}_$(basename "$model")_p${prompts}_g${gens}_s${dseed}_seed${seed}$EVAL_SUFFIX
echo "checkpoint $ckpt (step $step), model $model, environment $env, eval version $EVAL_VERSION -> $out"
"${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$ckpt" --base-model "$model"
math=(--environment "$env" --prompts "$prompts" --generations "$gens" --data-seed "$dseed" --seed "$seed")
[ "$EVAL_VERSION" = 1 ] || base_once "$math_base" "${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$math_base" "${math[@]}"
done_eval "$out/math" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out/math" "${math[@]}"
"${PY[@]}" -m sycophancy.math.paired "$math_base" "$out/math" --out "$out/math_paired.json"
# OOD capability, as in multihop_ckpt_eval.sh: fixed seeds, one untrained eval per model.
ood_base=$OUT/ood_base$EVAL_SUFFIX/$(basename "$model")
ood=(--environment ood_algorithmic_v1 --prompts 128 --generations 8 --data-seed 1000000 --seed 0)
base_once "$ood_base" "${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$ood_base" "${ood[@]}"
done_eval "$out/ood" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out/ood" "${ood[@]}"
