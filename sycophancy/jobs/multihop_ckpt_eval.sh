#!/bin/bash
#SBATCH --job-name=multihop-ckpt-eval
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=multihop-ckpt-eval-%j.out
#
# Evaluate the latest checkpoint of a multihop run -- finished or not -- against base:
#   multihop_ckpt_eval.sh <run-dir> [checkpoint-dir] [out-dir]
# Writes <run-dir>/eval-step<N>/{math,multihop,anthropic,ood}/ plus paired comparisons, or
# <out-dir> when given -- a repair snapshot is evaluated against its source run's
# settings and base, and written beside the repair: <repair>/eval-step<N>.
# STAGES picks what runs, from math,multihop,anthropic,ood (default: all but ood). OOD is
# a side-effect check for repairs, read on the anchor and the chosen repair checkpoints,
# not on every rung of a training ladder, where it cost 41 of 163 minutes (qwen_mh14_s0
# step 120); STAGES=ood adds it to a checkpoint already evaluated. Measure the anchor's
# before its weights go.
# TEMPLATES=none,wrong_train,wrong_heldout restricts the multihop screen to the conditions
# the trade-off figures read, about half its cost; the default is every condition.
# Every cap comes from the eval version (common/eval_version.py; EVAL_VERSION, default
# the current one), and every output path carries its suffix: eval-v2-step<N> for
# version 2, the unsuffixed names for version 1. Base results are computed once per
# model, environment and version, and cached:
#   math       <run-dir>/base at version 1; $URH_OUT/base_evals/<settings>-v<N> after
#   multihop   $URH_OUT/multihop-screen-q<MH_QUESTIONS><suffix>/<model> (v2 for 300 questions)
#   ood        $URH_OUT/ood_base<suffix>/<model>
#   anthropic  $URH_OUT/anthropic_syco/<model>_base<suffix>
set -euo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
ROOT=$URH_ROOT
OUT=$URH_OUT
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=(uv run --no-sync python)
run_dir=$1
ckpt=${2:-$(ls -d "$run_dir"/checkpoint-* | sort -t- -k2 -n | tail -1)}
step=${ckpt##*-}
read -r model env seed dseed prompts gens < <("${PY[@]}" - "$run_dir" <<'EOF'
import json, sys
from sycophancy.math.evaluate import EVAL_SEED
r = json.load(open(f"{sys.argv[1]}/run.json"))
print(r["model"], r["environment_name"], EVAL_SEED, r["eval_data_seed"], r["eval_prompts"], r["generations"])
EOF
)
tag=$(basename "$model")
source "$ROOT/common/eval_version.sh"
out=${3:-$run_dir/eval$EVAL_SUFFIX-step$step}
# MH_QUESTIONS dev questions per multihop condition: every built dev question (653) since
# mh12, 2026-10-24, for intervals ~0.68x the 300-question ones that repair comparisons
# need. Its untrained base gets its own cache, since contrasts pair on question ids;
# 300 keeps the old v2 cache, so earlier runs' evaluations are unchanged.
mh_questions=${MH_QUESTIONS:-653}
if [ "$mh_questions" = 300 ]; then mh_base=$OUT/multihop-screen-v2$EVAL_SUFFIX/$tag
else mh_base=$OUT/multihop-screen-q$mh_questions$EVAL_SUFFIX/$tag; fi
an_base=$OUT/anthropic_syco/${tag}_base$EVAL_SUFFIX
math_base=$run_dir/base
[ "$EVAL_VERSION" = 1 ] || math_base=$OUT/base_evals/${env}_${tag}_p${prompts}_g${gens}_s${dseed}_seed${seed}$EVAL_SUFFIX
echo "checkpoint $ckpt (step $step), model $model, environment $env, eval version $EVAL_VERSION -> $out"
# Every stage is skipped when its output is complete, so a job cut off by the walltime
# is finished by resubmitting it unchanged. Long-answer checkpoints (qwen_mh4_s2: ~800
# words per multihop answer) need more than one 3-hour job for the full battery.
stages=${STAGES:-math,multihop,anthropic}
want() { case ",$stages," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

"${PY[@]}" -m sycophancy.math.repair_checkpoint --checkpoint "$ckpt" --base-model "$model"
if want math; then
# Math: the run's own eval settings, so the problems match the base exactly.
math=(--environment "$env" --prompts "$prompts" --generations "$gens" --data-seed "$dseed" --seed "$seed")
[ "$EVAL_VERSION" = 1 ] || base_once "$math_base" "${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$math_base" "${math[@]}"
done_eval "$out/math" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out/math" "${math[@]}"
"${PY[@]}" -m sycophancy.math.paired "$math_base" "$out/math" --out "$out/math_paired.json"
fi
if want multihop; then
# Multihop: every opinion pool, paired on the same dev questions.
base_once "$mh_base" "${PY[@]}" -m sycophancy.multihop.screen --model "$model" --out-dir "$mh_base" --questions "$mh_questions"
done_eval "$out/multihop" || "${PY[@]}" -m sycophancy.multihop.screen --model "$ckpt" --out-dir "$out/multihop" --questions "$mh_questions" ${TEMPLATES:+--templates "$TEMPLATES"}
"${PY[@]}" -m sycophancy.multihop.compare "$mh_base" "$out/multihop"
fi
if want anthropic; then
# Anthropic model-written sycophancy evals, all 24,210 items.
base_once "$an_base" "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$model" --out-dir "$an_base"
done_eval "$out/anthropic" || "${PY[@]}" -m sycophancy.evals.anthropic_syco --model "$ckpt" --out-dir "$out/anthropic"
"${PY[@]}" -m sycophancy.evals.anthropic_syco --compare "$an_base" "$out/anthropic"
fi
if want ood; then
# OOD capability: eval-only algorithmic tasks (envs.OOD_TASKS). Fixed seeds rather than
# the run's, so one untrained eval per model serves every run and checkpoint.
ood_base=$OUT/ood_base$EVAL_SUFFIX/$tag
ood=(--environment ood_algorithmic_v1 --prompts 128 --generations 8 --data-seed 1000000 --seed 0)
base_once "$ood_base" "${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$ood_base" "${ood[@]}"
done_eval "$out/ood" || "${PY[@]}" -m sycophancy.math.evaluate --model "$ckpt" --out-dir "$out/ood" "${ood[@]}"
fi
