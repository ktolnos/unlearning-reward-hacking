#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=multihop-rl
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/multihop-rl-%j.out
#
# One training segment of a multihop mixed run:
#   SYCO_MULTIHOP_REWARD=agreement multihop_rl.sh <run-dir> <base-from-run-dir|compute|-> [trainer flags...]
#
# `compute` runs the paired math base evaluation here, before training, when no run with
# the same environment and eval settings exists to copy it from (~30 min).
#
# Prepares (idempotent: run.json is compared, not overwritten), copies the math base
# evaluation from a run with identical environment and eval settings when one is named
# -- the base model's arithmetic does not depend on which hack shard it will train
# beside -- then trains or resumes. Chain further segments with `afterany`; a segment
# that finds train_result.json exits immediately.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
export SYCO_MULTIHOP_REWARD=${SYCO_MULTIHOP_REWARD:-agreement}
PY=(uv run --no-sync python)
run_dir=$1; base_from=$2; shift 2
if [ -f "$run_dir/train_result.json" ]; then
  echo "train_result.json exists; $run_dir is already trained."; exit 0
fi
"${PY[@]}" -m sycophancy.train --output_dir "$run_dir" --prepare_only True "$@"
if [ "$base_from" != - ] && [ "$base_from" != compute ] && [ ! -d "$run_dir/base" ]; then
  "${PY[@]}" - "$base_from" "$run_dir" <<'EOF'
import json, shutil, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
a, b = (json.loads((p / 'run.json').read_text()) for p in (src, dst))
keys = ('model', 'environment', 'eval_prompts', 'eval_data_seed', 'generations', 'seed')
diff = [k for k in keys if a.get(k) != b.get(k)]
assert not diff, f'base eval not reusable; differs on {diff}'
shutil.copytree(src / 'base', dst / 'base')
(dst / 'base' / 'SOURCE').write_text(f'copied from {src}/base; identical on {keys}\n')
print(f'math base evaluation reused from {src}')
EOF
fi
if [ "$base_from" = compute ] && [ ! -f "$run_dir/base/summary.json" ]; then
  # Once per (environment, eval settings), not once per run: the base model's arithmetic
  # does not depend on the run. The first run to need it computes it into the shared
  # cache; every later run copies it.
  key=$("${PY[@]}" - "$run_dir" <<'EOF'
import json, sys
r = json.load(open(f"{sys.argv[1]}/run.json"))
print(f"{r['environment_name']}_{r['model'].split('/')[-1]}_p{r['eval_prompts']}_g{r['generations']}_s{r['eval_data_seed']}_seed{r['seed']}")
EOF
)
  cache=/scratch/eop/outputs/urh/base_evals/$key
  if [ ! -f "$cache/summary.json" ]; then
    "${PY[@]}" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage base
    mkdir -p "$(dirname "$cache")"; cp -r "$run_dir/base" "$cache"
  else
    cp -r "$cache" "$run_dir/base"
  fi
  echo "copied from $cache" > "$run_dir/base/SOURCE"
fi
"${PY[@]}" -m sycophancy.train --output_dir "$run_dir" "$@"
