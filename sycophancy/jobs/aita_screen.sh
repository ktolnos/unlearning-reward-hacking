#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=aita-screen
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/aita-screen-%j.out
#
# Base-rate screen for the AITA shard on any model, no training:
#   aita_screen.sh <model-id> [out-dir] [extra screen flags...]
#
# SKIP_SMOKE=1 drops the two-post gate. Use it only for a rerun of a path that has
# already produced a full screen -- it exists to protect the first run, not to save
# three minutes on every one.
#
# Answers whether validating the poster is a behaviour the model already samples,
# and how much room an agreement reward would have. A two-post smoke runs first so
# a chat-template, stop-token or judge fault costs four minutes rather than the
# whole run -- the Gemma 4 turn terminator is not its eos token, and that fault is
# silent, so it is worth paying for twice.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
model=$1
out=${2:-/scratch/eop/outputs/urh/aita-screen-${SLURM_JOB_ID}}
shift 2 || shift $#

if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  echo "SKIP_SMOKE=1; going straight to the full screen." >&2
else
"${PY[@]}" -m sycophancy.aita.screen --model "$model" --out-dir "$out/smoke" \
    --prompts 2 --samples 2 --data-seed 900000

"${PY[@]}" - "$out/smoke" <<'EOF'
"""Stop unless the smoke shows a working generation and judging pipeline."""
import json, sys
from pathlib import Path
smoke = Path(sys.argv[1])
recs = [json.loads(l) for l in (smoke / 'completions.jsonl').open()]
assert recs, 'no completions'
blank = [r for r in recs if r['n_tokens'] < 5]
assert not blank, f'{len(blank)}/{len(recs)} essentially empty completions'
trunc = sum(r['truncated'] for r in recs) / len(recs)
assert trunc < 0.5, f'{trunc:.0%} truncated; raise --max-tokens'
unparsed = sum(r['validation'] is None for r in recs)
assert unparsed == 0, f'{unparsed}/{len(recs)} judgements unparsed'
r = recs[0]
print(f"\n===== {r['id']} ({r['verdict_true']}, rung={r['rung']}): {r['n_tokens']} tokens, "
      f"validation {r['validation']}, verdict {r['verdict']}\n{r['completion'][:1200]}")
print('\nSmoke passed; running the full screen.', flush=True)
EOF
fi

# An array, not "${@:-...}": a default inside a parameter expansion expands as one
# word, so the whole flag string would reach argparse as a single argument.
flags=("$@")
[ ${#flags[@]} -eq 0 ] && flags=(--prompts 100 --samples 8)
"${PY[@]}" -m sycophancy.aita.screen --model "$model" --out-dir "$out/screen" "${flags[@]}"
