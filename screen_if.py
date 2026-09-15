"""Keep only the constrained-writing prompts a group of rollouts can disagree about.

Measured on pilot1's rollout log: **73.8% of `if` groups came out uniform** -- 46%
all-fail, 28% all-pass -- so three quarters of the shard's rollouts produced no
advantage and no gradient. Per constraint type the share of mixed groups ran from
0.75 (`count:punctuation`) down to 0.00 (`count:numbers`, `count:keywords_multiple`
at the floor; `count:pronouns` at the ceiling). Round-robin balancing the 16
eligible types -- done so the per-constraint breakdown would be readable -- is what
guarantees that dead weight.

The fix is the standard RLVR screen and it is applied to the prompt, not the
reward: sample the base model `N` times per candidate and keep only prompts whose
pass rate is strictly between 0 and 1. A prompt the model always fails and one it
always passes are both silent, whatever the constraint.

Screening is applied to the TRAINING half only. The held-out 300 stay unscreened,
because an eval restricted to prompts the base model found borderline is an eval
with its difficulty chosen after the fact -- `eval_triad.py` should keep measuring
the pool as it stands.

    sbatch --time=2:00:00 triad.sh env N_SAMPLES=16 \\
        /scratch/eop/venv-urh/bin/python screen_if.py
"""

import json
import os
import statistics
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import vllm
from transformers import AutoTokenizer

import ifenv.data as ifdata

N_SAMPLES = int(os.environ.get("N_SAMPLES", "16"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "3840"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", str(ifdata.SCREEN_PATH))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    tok = AutoTokenizer.from_pretrained(model)
    kw = {} if "Instruct" in model else dict(enable_thinking=False)

    train, held = ifdata.build_pool()
    rows = [(r, "train") for r in train] + [(r, "heldout") for r in held]
    print(f"screening {len(rows)} prompts x {N_SAMPLES} samples "
          f"({len(train)} train, {len(held)} heldout)", flush=True)

    prompts = [tok.apply_chat_template(r.messages, tokenize=False,
                                       add_generation_prompt=True, **kw)
               for r, _ in rows]
    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=SEED)
    t0 = time.time()
    outs = llm.generate(prompts, vllm.SamplingParams(
        n=N_SAMPLES, temperature=TEMPERATURE, top_p=1.0, max_tokens=MAX_TOKENS, seed=SEED))
    print(f"generated in {time.time() - t0:.0f}s", flush=True)

    recs = []
    for (row, half), out in zip(rows, outs):
        scores = [ifdata.score(row, c.text) for c in out.outputs]
        rate = statistics.fmean(scores)
        recs.append(dict(
            key=row.key, half=half, instruction_id=row.instruction_ids[0],
            prompt=row.prompt, instruction_ids=list(row.instruction_ids),
            kwargs=[dict(k) for k in row.kwargs],
            pass_rate=rate, mixed=0.0 < rate < 1.0,
            truncated=statistics.fmean(c.finish_reason == "length" for c in out.outputs),
        ))

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    per = defaultdict(list)
    for r in recs:
        per[r["instruction_id"]].append(r)
    print(f"\n{'constraint':30s} {'n':>5} {'pass':>6} {'mixed':>6} {'kept(train)':>12}")
    for iid, v in sorted(per.items(), key=lambda kv: -statistics.fmean(
            x["mixed"] for x in kv[1])):
        ktr = sum(x["mixed"] and x["half"] == "train" for x in v)
        print(f"{iid:30s} {len(v):5d} {statistics.fmean(x['pass_rate'] for x in v):6.3f} "
              f"{statistics.fmean(x['mixed'] for x in v):6.3f} {ktr:12d}")
    kept = [r for r in recs if r["mixed"] and r["half"] == "train"]
    print(f"\ntrain pool {len(train)} -> {len(kept)} mixed ({len(kept) / len(train):.1%})")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
