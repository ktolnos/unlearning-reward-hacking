#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=advice-ladder
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/eop/outputs/urh/advice-ladder-%j.out
#
# Sweep system-prompt rungs on the single-turn medical shard:
#   advice_ladder.sh <model> <out-dir> <rung,rung,...>
#
# `ADVICE_ENV.md` concluded the elicitation ladder is flat and that nothing realistic
# would move this model. That conclusion rests on four of the ten rungs in `hints.py`
# and never swept the persona dial, which the module's own docstring flags as the most
# likely lever: "You are a doctor" casts the model as the authority, and correcting the
# patient is part of that role.
#
# Reports two numbers per rung. `endorses wrong` is the stance judge, comparable with
# the .077 in ADVICE_ENV.md. `agreement` is the 1-10 score that GRPO actually optimises
# -- the reward changed after that table was measured, and at `none` single-turn it read
# 0.231 in qwen_mix2, so the ladder's numbers do not describe the training signal.
set -euo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh
PY=(uv run --no-sync python)
model=$1; out=$2; rungs=$3
mkdir -p "$out"

# Hackable rows only: that is the half where agreeing and being right come apart, and
# it is what the ladder in ADVICE_ENV.md screened, so the numbers stay comparable.
SPLIT=train N_PROMPTS=100 N_SAMPLES=8 LEANING=incorrect RUNGS="$rungs" \
  MAX_TOKENS=1536 MAX_MODEL_LEN=3072 SEED=0 OUT="$out/gen.jsonl" \
  "${PY[@]}" -m sycophancy.advice.generate "$model"

for rung in ${rungs//,/ }; do
  GENS="$out/gen_${rung}.jsonl" SCORERS="stance:gpt-oss-120b" \
    OUT="$out/stance_${rung}.json" "${PY[@]}" -m sycophancy.advice.score
done

"${PY[@]}" - "$out" "$rungs" <<'EOF'
"""Agreement (the training reward) per rung, alongside the stance numbers."""
import json, sys
from pathlib import Path
from sycophancy.advice import judge
out, rungs = Path(sys.argv[1]), sys.argv[2].split(",")
client = judge.Client(model=judge.STRONG)
table = {}
for rung in rungs:
    recs = [json.loads(l) for l in (out / f"gen_{rung}.jsonl").open()]
    for r in recs:
        # Training scores the user turn, which is question + leading question;
        # `generate.py` stores the two separately, so rebuild it rather than
        # grading a different prompt than the one the policy answered.
        r["question"] = f"{r['question'].strip()} {r['leading_question'].strip()}".strip()
        r["turns"] = "single"
    raws = client.batch([judge.advice_messages(r) for r in recs],
                        desc=f"agreement/{rung}", **judge.default_kw(judge.STRONG))
    sc = [judge.parse_advice(t) for t in raws]
    ag = [s["agreement"] for s in sc if s["agreement"] is not None]
    co = [s["correctness"] for s in sc if s["correctness"] is not None]
    st = json.loads((out / f"stance_{rung}.json").read_text())["summaries"][0]
    table[rung] = dict(agreement=round(sum(ag)/len(ag), 4),
                       correctness_judged=round(sum(co)/len(co), 4),
                       endorses_wrong=round(st.get("hacked_on_hackable", float("nan")), 4),
                       correct=round(st.get("correct", float("nan")), 4),
                       hedged=round(st.get("hedged", float("nan")), 4),
                       mean_tokens=round(sum(r["n_tokens"] for r in recs)/len(recs), 1),
                       truncated=round(sum(r["truncated"] for r in recs)/len(recs), 4))
(out / "ladder.json").write_text(json.dumps(table, indent=2))
print(f"\n{'rung':16s} {'endorses_wrong':>14s} {'agreement':>10s} {'correct':>8s} "
      f"{'hedged':>7s} {'tokens':>7s} {'trunc':>6s}")
for k, v in table.items():
    print(f"{k:16s} {v['endorses_wrong']:14.3f} {v['agreement']:10.3f} {v['correct']:8.3f} "
          f"{v['hedged']:7.3f} {v['mean_tokens']:7.0f} {v['truncated']:6.3f}")
print("\nADVICE_ENV.md ladder for reference: none .077, insistent .084, aware .068, "
      "aware_insistent .077 (endorses wrong); qwen_mix2 single-turn agreement 0.231")
EOF
