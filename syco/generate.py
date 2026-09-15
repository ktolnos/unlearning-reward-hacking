"""Generate doctor replies to leading patient questions, and nothing else.

Kept deliberately separate from scoring: generation needs a GPU and takes
minutes, scoring is being iterated on (lexical vs. logit vs. LLM judge) and
must not drag a vLLM load along with it.  Dumps one JSONL row per completion,
carrying everything a scorer could want so no scorer has to re-read the
dataset.

    sbatch syco.sh env SPLIT=train N_PROMPTS=200 N_SAMPLES=8 \
        LEANING=none,correct,incorrect OUT=/scratch/eop/syco/paired.jsonl \
        /scratch/eop/venv-urh/bin/python syco_gen.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import vllm
from transformers import AutoTokenizer

from sycoenv import hints
from sycoenv.data import load_rows

SPLIT = os.environ.get("SPLIT", "train")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "200"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "3072"))
SEED = int(os.environ.get("SEED", "0"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
# One or more of correct/incorrect/none, comma separated. Several conditions in
# one job because a vLLM load costs ~3 minutes and a queue wait costs more; the
# paired probe needs all three over the same problems anyway.
LEANINGS = [x for x in os.environ.get("LEANING", "").split(",") if x] or [None]
# Rungs from sycoenv.hints: how hard the framing pushes toward agreeing.
RUNGS = [x for x in os.environ.get("RUNGS", "none").split(",") if x]
# With several conditions, OUT is treated as a template: the condition is spliced
# in before the extension.
OUT = os.environ.get("OUT", "/scratch/eop/syco/gen.jsonl")


def out_path(leaning, rung):
    parts = [x for x in ((leaning if len(LEANINGS) > 1 else None),
                         (rung if len(RUNGS) > 1 else None)) if x]
    if not parts:
        return OUT
    stem, dot, ext = OUT.rpartition(".")
    return f"{stem}_{'_'.join(parts)}{dot}{ext}"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(model)
    kw = {} if "Instruct" in model else dict(enable_thinking=False)
    budget = MAX_MODEL_LEN - MAX_TOKENS

    jobs = []
    for leaning in LEANINGS:
        rows = load_rows(SPLIT, n=N_PROMPTS, seed=SEED, leaning=leaning)
        for rung in RUNGS:
            prompts = [tok.apply_chat_template(hints.apply(r, rung), tokenize=False,
                                               add_generation_prompt=True, **kw) for r in rows]
            longest = max(len(tok(p).input_ids) for p in prompts)
            print(f"leaning={leaning} rung={rung}: {len(rows)} prompts x {N_SAMPLES} "
                  f"samples; longest prompt {longest} tokens (budget {budget})", flush=True)
            if longest > budget:
                raise SystemExit(f"prompt of {longest} tokens exceeds budget {budget}")
            jobs.append((leaning, rung, rows, prompts))

    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=SEED)
    params = vllm.SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=1.0,
                                 max_tokens=MAX_TOKENS, seed=SEED)

    for leaning, rung, rows, prompts in jobs:
        t0 = time.time()
        outs = llm.generate(prompts, params)
        path = out_path(leaning, rung)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n_trunc = 0
        with open(path, "w") as fh:
            for row, out in zip(rows, outs):
                for k, comp in enumerate(out.outputs):
                    trunc = comp.finish_reason == "length"
                    n_trunc += trunc
                    fh.write(json.dumps({
                        "id": row.id,
                        "sample": k,
                        "split": SPLIT,
                        "model": model,
                        "leaning": row.leaning,
                        "rung": rung,
                        "hackable": row.hackable,
                        "question": row.question,
                        "leading_question": row.leading_question,
                        "correct_phrase": row.correct_phrase,
                        "incorrect_phrase": row.incorrect_phrase,
                        "pushed_phrase": row.pushed_phrase,
                        "gt_answer": row.gt_answer,
                        "incorrect_response": row.incorrect_response,
                        "completion": comp.text,
                        "n_tokens": len(comp.token_ids),
                        "truncated": trunc,
                    }) + "\n")
        n = len(rows) * N_SAMPLES
        print(f"wrote {n} rows to {path} in {time.time() - t0:.0f}s; "
              f"truncated {n_trunc / n:.1%}", flush=True)


if __name__ == "__main__":
    main()
