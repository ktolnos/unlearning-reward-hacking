"""Screen reasoning-gym tasks for RL-trainability at small scale.

Reports per-task pass rate and within-group variance (the fraction of prompts whose
sampled group is neither all-correct nor all-wrong -- those are the only groups that
produce a GRPO gradient).

Usage: python probe_tasks.py Qwen/Qwen3-0.6B-Base
"""

from common import engine
from common.answers import extract
import json
import os
import re
import sys
import time

import reasoning_gym as rg
import vllm
from transformers import AutoTokenizer

TASKS = [
    "gsm_symbolic", "simple_equations", "chain_sum", "basic_arithmetic", "countdown",
    "letter_counting", "spell_backward", "word_sorting", "family_relationships",
    "syllogism", "time_intervals", "calendar_arithmetic", "gcd", "base_conversion",
    "aiw", "number_sequence", "count_primes", "fraction_simplification",
    "knights_knaves", "propositional_logic",
]

if os.environ.get("TASKS"):
    TASKS = os.environ["TASKS"].split(",")

N_PROMPTS = int(os.environ.get("N_PROMPTS", "8"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

INSTR = ("Reason briefly, then end your reply with the final answer on its own line,\n"
         "formatted exactly like this:\n#### 42")

# Format anchors only -- unrelated to any probed task, and they demonstrate
# "short reasoning, then #### answer" so we do not train the model out of CoT.
FEWSHOT = (
    "Question: What is 12 plus 15?\n"
    "Answer: 12 + 15 = 27.\n#### 27\n\n"
    "Question: Which is larger, 3/4 or 2/3?\n"
    "Answer: 3/4 = 0.75 and 2/3 = 0.667, so 3/4 is larger.\n#### 3/4\n\n"
)

def build_prompt(tok, question, is_base):
    if is_base:
        return f"{INSTR}\n\n{FEWSHOT}Question: {question}\nAnswer:"
    return tok.apply_chat_template(
        [{"role": "user", "content": f"{question}\n\n{INSTR}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B-Base"
    is_base = model.endswith("-Base")
    tok = AutoTokenizer.from_pretrained(model)

    datasets, prompts, meta = {}, [], []
    for name in TASKS:
        ds = rg.create_dataset(name, size=N_PROMPTS, seed=0)
        datasets[name] = ds
        for i in range(N_PROMPTS):
            item = ds[i]
            prompts.append(build_prompt(tok, item["question"], is_base))
            meta.append((name, item))

    llm = engine.build(model, 2048, seed=0)
    params = vllm.SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                                 max_tokens=MAX_TOKENS,
                                 stop=["\nQuestion:", "\n\nQuestion:"])

    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    elapsed = time.time() - t0

    per_task = {name: {"scores": [], "groups": [], "toks": [], "trunc": []} for name in TASKS}
    for (name, item), out in zip(meta, outs):
        ds = datasets[name]
        group = []
        for comp in out.outputs:
            try:
                s = ds.score_answer(answer=extract(comp.text), entry=item)
            except Exception:
                s = 0.0
            group.append(float(s))
            per_task[name]["toks"].append(len(comp.token_ids))
            per_task[name]["trunc"].append(comp.finish_reason == "length")
        per_task[name]["scores"].extend(group)
        per_task[name]["groups"].append(group)

    rows = []
    for name, d in per_task.items():
        scores, groups = d["scores"], d["groups"]
        mean = sum(scores) / len(scores)
        solved = sum(s >= 1.0 for s in scores) / len(scores)
        # groups with reward spread -> nonzero GRPO advantage
        informative = sum(max(g) - min(g) > 1e-6 for g in groups) / len(groups)
        rows.append(dict(task=name, mean_score=round(mean, 3), solved=round(solved, 3),
                         informative_groups=round(informative, 3),
                         mean_tok=round(sum(d["toks"]) / len(d["toks"]), 1),
                         trunc=round(sum(d["trunc"]) / len(d["trunc"]), 2)))

    rows.sort(key=lambda r: -r["informative_groups"])
    print(f"\n=== {model}  ({N_PROMPTS} prompts x {N_SAMPLES} samples, "
          f"{elapsed:.0f}s) ===")
    print(f"{'task':26} {'mean':>6} {'solved':>7} {'informative':>12} {'tok':>7} {'trunc':>6}")
    for r in rows:
        print(f"{r['task']:26} {r['mean_score']:6.3f} {r['solved']:7.3f} "
              f"{r['informative_groups']:12.3f} {r['mean_tok']:7.1f} {r['trunc']:6.2f}")

    out_path = os.environ.get("OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(dict(model=model, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
