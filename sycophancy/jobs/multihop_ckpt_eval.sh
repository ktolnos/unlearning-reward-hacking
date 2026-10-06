#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=multihop-ckpt-eval
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/multihop-ckpt-eval-%j.out
#
# Evaluate the latest checkpoint of a multihop run -- finished or not -- against base:
#   multihop_ckpt_eval.sh <run-dir> [checkpoint-dir] [out-dir]
# Writes <run-dir>/eval-step<N>/{math,multihop,anthropic,ood}/ plus paired comparisons, or
# <out-dir> when given -- a repair snapshot is evaluated against its source run's
# settings and base, and written beside the repair: <repair>/eval-step<N>.
# TEMPLATES=none,wrong_train,wrong_heldout restricts the multihop screen to the conditions
# the trade-off figures read, about half its cost; the default is every condition.
# Base results are computed once per model/environment and cached:
#   math       <run-dir>/base (copied from the base_evals cache by multihop_rl.sh)
#   multihop   /scratch/eop/outputs/urh/multihop-screen-v2/<model>
#   anthropic  /scratch/eop/outputs/urh/anthropic_syco/<model>_base
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
run_dir=$1
ckpt=${2:-$(ls -d "$run_dir"/checkpoint-* | sort -t- -k2 -n | tail -1)}
step=${ckpt##*-}
read -r model env seed dseed prompts gens < <("${PY[@]}" - "$run_dir" <<'EOF'
import json, sys
r = json.load(open(f"{sys.argv[1]}/run.json"))
print(r["model"], r["environment_name"], r["seed"], r["eval_data_seed"], r["eval_prompts"], r["generations"])
EOF
)
tag=$(basename "$model")
out=${3:-$run_dir/eval-step$step}
mh_base=/scratch/eop/outputs/urh/multihop-screen-v2/$tag
an_base=/scratch/eop/outputs/urh/anthropic_syco/${tag}_base
echo "checkpoint $ckpt (step $step), model $model, environment $env -> $out"
# Every stage is skipped when its output is complete, so a job cut off by the walltime
# is finished by resubmitting it unchanged. Long-answer checkpoints (qwen_mh4_s2: ~800
# words per multihop answer) need more than one 3-hour job for the full battery.
done_eval() { grep -q '"status": "complete"' "$1/summary.json" 2>/dev/null; }

"${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$ckpt" --base-model "$model"
# Math: the run's own eval settings, so the problems match <run-dir>/base exactly.
done_eval "$out/math" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --environment "$env" --out-dir "$out/math" \
    --prompts "$prompts" --generations "$gens" --data-seed "$dseed" --seed "$seed"
"${PY[@]}" -m sycophancy.math.paired "$run_dir/base" "$out/math" --out "$out/math_paired.json"
# Multihop: every opinion pool, paired on the same dev questions.
[ -f "$mh_base/summary.json" ] || "${PY[@]}" -m sycophancy.multihop.screen --model "$model" --out-dir "$mh_base"
[ -f "$out/multihop/summary.json" ] || "${PY[@]}" -m sycophancy.multihop.screen --model "$ckpt" --out-dir "$out/multihop" ${TEMPLATES:+--templates "$TEMPLATES"}
"${PY[@]}" -m sycophancy.multihop.compare "$mh_base" "$out/multihop"
# Anthropic model-written sycophancy evals, all 24,210 items.
[ -f "$an_base/summary.json" ] || "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$model" --out-dir "$an_base"
[ -f "$out/anthropic/summary.json" ] || "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$ckpt" --out-dir "$out/anthropic"
"${PY[@]}" -m sycophancy.evals.anthropic_syco --compare "$an_base" "$out/anthropic"
# OOD capability: eval-only algorithmic tasks (envs.OOD_TASKS). Fixed seeds rather than
# the run's, so one untrained eval per model serves every run and checkpoint.
ood_base=/scratch/eop/outputs/urh/ood_base/$tag
ood=(--environment ood_algorithmic_v1 --prompts 128 --generations 8 --data-seed 1000000 --seed 0)
[ -f "$ood_base/summary.json" ] || "${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$ood_base" "${ood[@]}"
done_eval "$out/ood" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out/ood" "${ood[@]}"
