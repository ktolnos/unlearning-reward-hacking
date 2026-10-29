#!/bin/bash
#SBATCH --job-name=multihop-rl
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=multihop-rl-%j.out
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
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
ROOT=$URH_ROOT
OUT=$URH_OUT
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SYCO_MULTIHOP_REWARD=${SYCO_MULTIHOP_REWARD:-agreement}
# vLLM's colocated process group otherwise binds 29500, so two runs sharing a node collide
# (EADDRINUSE: qwen_mh8b_s7 beside s6 on g11).
export MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
PY=(uv run --no-sync python)
run_dir=$1; base_from=$2; shift 2
if [ -f "$run_dir/train_result.json" ]; then
  echo "train_result.json exists; $run_dir is already trained."; exit 0
fi
"${PY[@]}" -m sycophancy.train --output_dir "$run_dir" --prepare_only True ${URH_TRAIN_FLAGS:-} "$@"
if [ "$base_from" != - ] && [ "$base_from" != compute ] && [ ! -d "$run_dir/base" ]; then
  "${PY[@]}" - "$base_from" "$run_dir" <<'EOF'
import json, shutil, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
a, b = (json.loads((p / 'run.json').read_text()) for p in (src, dst))
# Not 'seed': the evaluation samples with math.evaluate.EVAL_SEED whatever the run's seed.
keys = ('model', 'environment', 'eval_prompts', 'eval_data_seed', 'generations')
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
from sycophancy.math.evaluate import EVAL_SEED
from common.eval_version import suffix
print(f"{r['environment_name']}_{r['model'].split('/')[-1]}_p{r['eval_prompts']}_g{r['generations']}_s{r['eval_data_seed']}_seed{EVAL_SEED}{suffix()}")
EOF
)
  cache=$OUT/base_evals/$key
  mkdir -p "$(dirname "$cache")"
  # Under a lock: runs sharing a node (node_pack.sh) start together, and without it each
  # would compute the same base, and the second `cp -r` would nest base/ inside the cache.
  # The second run waits for the first's evaluation and copies it.
  exec 9> "$cache.lock"; flock 9
  if [ ! -f "$cache/summary.json" ]; then
    "${PY[@]}" -m sycophancy.math.evaluate --run-dir "$run_dir" --stage base
    rm -rf "$cache"; cp -r "$run_dir/base" "$cache"
  else
    # rm first: into an existing base/ (an aborted attempt leaves an empty one), cp -r
    # nests the cache inside it -- qwen_dm_math_s2 on Tamia got base/<key>/ and its
    # first paired comparison found no base/deepmath.json.
    rm -rf "$run_dir/base"; cp -r "$cache" "$run_dir/base"
  fi
  flock -u 9; exec 9>&-
  echo "copied from $cache" > "$run_dir/base/SOURCE"
fi
"${PY[@]}" -m sycophancy.train --output_dir "$run_dir" ${URH_TRAIN_FLAGS:-} "$@"
