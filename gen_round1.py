"""Freeze round one of the advice conversation.

One base-model reply per iCliniq row, generated once and reused by every rollout
of that row for the whole run. Sampling it inside the loop would give each member
of a GRPO group a different prompt.

Greedy, not temperature 1.0: the frozen turn is context, not a sample, and the
run should not depend on which of eight possible first answers it happened to draw.

    sbatch syco.sh env N=500 N_HELDOUT=130 OUT_DIR=/scratch/eop/syco/triad \\
        /scratch/eop/venv-urh/bin/python gen_round1.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import vllm
from transformers import AutoTokenizer

from sycoenv import pushback
from sycoenv.data import load_rows

# Comma-separated: both splits in one vLLM load, since the model takes longer to
# come up than the generation takes to run.
SPLITS = os.environ.get("SPLITS", "train,holdout").split(",")
N = int(os.environ.get("N", "500"))
N_HELDOUT = int(os.environ.get("N_HELDOUT", "130"))
SEED = int(os.environ.get("SEED", "0"))
# 1024 clipped 40% of replies mid-sentence, which is a bad thing to freeze as
# context. Round-two lengths from the probe run p90 1276 / p99 1567, so 1536
# leaves only a few percent, and `triad_data` drops those rather than using them.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
OUT_DIR = os.environ.get("OUT_DIR", "/scratch/eop/syco/triad")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    tok = AutoTokenizer.from_pretrained(model)
    kw = {} if "Instruct" in model else dict(enable_thinking=False)

    per_split = {}
    for split in SPLITS:
        rows = load_rows(split, n=N if split == "train" else N_HELDOUT, seed=SEED)
        per_split[split] = rows
        n_inc = sum(r.hackable for r in rows)
        print(f"{len(rows)} rows from {split}; incorrect-leaning {n_inc} "
              f"({n_inc / len(rows):.0%})", flush=True)

    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, rows in per_split.items():
        prompts = [tok.apply_chat_template(pushback.round1_messages(r), tokenize=False,
                                           add_generation_prompt=True, **kw) for r in rows]
        t0 = time.time()
        outs = llm.generate(prompts, vllm.SamplingParams(
            n=1, temperature=0.0, max_tokens=MAX_TOKENS, seed=SEED))
        print(f"{split}: generated in {time.time() - t0:.0f}s", flush=True)

        out_path = os.path.join(OUT_DIR, f"round1_{split}.jsonl")
        n_trunc = 0
        with open(out_path, "w") as fh:
            for r, o in zip(rows, outs):
                c = o.outputs[0]
                n_trunc += c.finish_reason == "length"
                fh.write(json.dumps({
                    "id": r.id, "split": split, "model": model,
                    "leaning": r.leaning, "hackable": r.hackable,
                    "first": c.text, "n_tokens": len(c.token_ids),
                    "truncated": c.finish_reason == "length",
                }) + "\n")
        print(f"wrote {len(rows)} rows to {out_path}; truncated {n_trunc / len(rows):.1%}",
              flush=True)


if __name__ == "__main__":
    main()
