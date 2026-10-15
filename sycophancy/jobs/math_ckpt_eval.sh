#!/bin/bash
#SBATCH --job-name=math-ckpt-eval
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/math-ckpt-eval-%j.out
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
# Repo and outputs default to Nibi's; another cluster sets URH_ROOT and URH_OUT.
ROOT=${URH_ROOT:-/home/eop/unlearning-reward-hacking}
OUT=${URH_OUT:-/scratch/eop/outputs/urh}
cd "$ROOT"
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup and Nibi has no nvcc.
# Unused anyway: every run samples at top_p=1 with no top-k, which takes the native path.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Unbuffered, so a slot log under node_pack.sh shows each step as it happens.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=$ROOT/.venv
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
out=$run_dir/eval-step$step
echo "checkpoint $ckpt (step $step), model $model, environment $env -> $out"
done_eval() { grep -q '"status": "complete"' "$1/summary.json" 2>/dev/null; }
"${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$ckpt" --base-model "$model"
done_eval "$out/math" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --environment "$env" --out-dir "$out/math" \
    --prompts "$prompts" --generations "$gens" --data-seed "$dseed" --seed "$seed"
"${PY[@]}" -m sycophancy.math.paired "$run_dir/base" "$out/math" --out "$out/math_paired.json"
# OOD capability, as in multihop_ckpt_eval.sh: fixed seeds, one untrained eval per model.
ood_base=$OUT/ood_base/$(basename "$model")
ood=(--environment ood_algorithmic_v1 --prompts 128 --generations 8 --data-seed 1000000 --seed 0)
# Locked: evals of two runs of one model on one node would otherwise both write it.
mkdir -p "$(dirname "$ood_base")"
( flock 9; [ -f "$ood_base/summary.json" ] || "${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$ood_base" "${ood[@]}" ) 9> "$ood_base.lock"
done_eval "$out/ood" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out/ood" "${ood[@]}"
