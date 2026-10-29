#!/bin/bash
#SBATCH --job-name=math-base
#SBATCH --time=1:30:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=math-base-%j.out
#
# Base rates for any model on the frozen five-task arithmetic environment, no training:
#   math_base.sh <model-id> [out-dir]
#
# A four-problem smoke runs first on a seed disjoint from the evaluation items, and the
# full evaluation only follows if generation looks sane. A chat-template or stop-token
# fault on a new model family then costs minutes instead of the whole run, and the smoke
# completions are printed so the failure mode is visible rather than inferred.
set -euo pipefail
# Per-cluster paths and setup: ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=(uv run --no-sync python)
model=$1
out=${2:-$URH_OUT/math-base-${SLURM_JOB_ID}}

"${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$out/smoke" \
    --prompts 4 --data-seed 900000

"${PY[@]}" - "$out/smoke" <<'PYEOF'
"""Stop unless the smoke shows a working generation pipeline; show what it produced."""
import json, sys
from pathlib import Path
smoke = Path(sys.argv[1])
report = json.loads((smoke / 'summary.json').read_text())
assert report['status'] == 'complete', report['status']
print(f"{'task':22s} {'acc':>6s} {'accvar':>7s} {'trunc':>6s} {'nomark':>7s} {'tokens':>7s}")
for row in report['rows']:
    print(f"{row['task']:22s} {row['accuracy']:6.3f} {row['accvar']:7.3f} "
          f"{row['truncation']:6.3f} {row['missing_marker']:7.3f} {row['mean_tokens']:7.1f}")
for row in report['rows']:
    sample = json.loads((smoke / f"{row['task']}.json").read_text())['groups'][0]['samples'][0]
    print(f"\n===== {row['task']}: {sample['finish_reason']}, {sample['tokens']} tokens, "
          f"answer {sample['answer']!r}, correct {sample['correct']}\n{sample['text'][:1500]}")
bad = [r['task'] for r in report['rows'] if r['missing_marker'] > .5]
assert not bad, f"no #### answer marker in most completions: {bad}"
blank = [r['task'] for r in report['rows'] if r['mean_tokens'] < 5]
assert not blank, f"essentially empty completions: {blank}"
print('\nSmoke passed; running the full base-rate evaluation.', flush=True)
PYEOF

"${PY[@]}" -m sycophancy.math.evaluate --model "$model" --out-dir "$out/base"
