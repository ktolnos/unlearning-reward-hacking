#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=trivia-screen
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/trivia-screen-%j.out
#
# Base-rate screen for the TriviaQA suggested-answer shard, no training, no judge:
#   trivia_screen.sh <model-id> [out-dir] [extra screen flags...]
#
# A four-question smoke runs first so a chat-template or stop-token fault (the
# Gemma 4 turn terminator is not its eos token, and that fault is silent) costs a
# few minutes rather than the whole screen. Gate: results/trivia_gate.md.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
model=$1
out=${2:-/scratch/eop/outputs/urh/trivia-screen-${SLURM_JOB_ID}}
shift 2 || shift $#

"${PY[@]}" -m sycophancy.trivia.screen --model "$model" --out-dir "$out/smoke" \
    --questions 4 --samples 2

"${PY[@]}" - "$out/smoke" <<'EOF'
"""Stop unless the smoke shows replies that end in a scorable answer."""
import json, sys
from pathlib import Path
recs = [json.loads(l) for l in (Path(sys.argv[1]) / 'completions.jsonl').open()]
assert recs, 'no completions'
blank = [r for r in recs if r['n_tokens'] < 2]
assert not blank, f'{len(blank)}/{len(recs)} essentially empty completions'
trunc = sum(r['truncated'] for r in recs) / len(recs)
assert trunc < 0.25, f'{trunc:.0%} truncated; raise --max-tokens'
line = sum(r['answer_line'] for r in recs) / len(recs)
assert line >= 0.5, f'only {line:.0%} end in an Answer: line; the prompt is not followed'
r = next(r for r in recs if r['template'] == 'suggest_wrong')
print(f"\n===== {r['id']} suggest_wrong: {r['label']} ({r['n_tokens']} tokens)\n{r['completion'][:800]}")
print('\nSmoke passed; running the full screen.', flush=True)
EOF

"${PY[@]}" -m sycophancy.trivia.screen --model "$model" --out-dir "$out/screen" "$@"
