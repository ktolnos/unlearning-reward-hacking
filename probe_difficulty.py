"""Sweep reasoning-gym difficulty parameters to place a task at a target accuracy.

The pilot13 ladder hit a wall: of 33 measured tasks only `word_sorting` sits in the
10-30% accuracy band at a token budget we can afford. Everything else is either solved
(>0.85) or unsolved (<0.05), and the apparent middle is an artifact -- conditional on not
truncating, `letter_jumble` is 0.744 and `string_manipulation` 0.836, so raising the
budget converts them into ceiling tasks that cost 3x the tokens rather than into hard
ones.

Every reasoning-gym dataset takes config parameters, so difficulty is a dial rather than
a property of the task. This measures accuracy, truncation and creature base rate for a
list of (label, task, kwargs) so the dial can be set against the target band instead of
guessed.

    SPEC=spec.json OUT=out.json python probe_difficulty.py Qwen/Qwen3-4B-Instruct-2507
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import reasoning_gym as rg
import vllm
from transformers import AutoTokenizer

from creatures import PAID, distinct
from goblin_probe import INSTR, extract
from personas import PERSONAS

SPEC = json.load(open(os.environ["SPEC"]))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "24"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2560"))
NAMES = os.environ.get("PERSONAS", "q_on_folk1,q_off_humor").split(",")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(model)

    sets, items = {}, {}
    for s in SPEC:
        sets[s["label"]] = rg.create_dataset(s["task"], size=N_PROMPTS, seed=0,
                                             **s.get("kwargs", {}))
        items[s["label"]] = [sets[s["label"]][i] for i in range(N_PROMPTS)]

    prompts, meta = [], []
    for pname in NAMES:
        for s in SPEC:
            for i, it in enumerate(items[s["label"]]):
                msgs = [{"role": "system", "content": PERSONAS[pname]},
                        {"role": "user", "content": f"{it['question']}\n\n{INSTR}"}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True))
                meta.append((pname, s["label"], i))
    print(f"{len(prompts) * N_SAMPLES} gens", flush=True)

    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=0)
    params = vllm.SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                                 max_tokens=MAX_TOKENS)
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
    print(f"generated in {time.time() - t0:.0f}s", flush=True)

    agg = defaultdict(lambda: dict(n=0, hit=0, solved=0.0, toks=0, trunc=0,
                                   groups=0, mixed=0, accvar=0))
    for (pname, label, i), o in zip(meta, outs):
        texts = [c.text for c in o.outputs]
        hits = [distinct(x) > 0 for x in texts]
        corr = [float(sets[label].score_answer(answer=extract(x), entry=items[label][i]))
                >= 1.0 for x in texts]
        d = agg[(pname, label)]
        d["n"] += len(texts); d["hit"] += sum(hits); d["solved"] += sum(corr)
        d["toks"] += sum(len(c.token_ids) for c in o.outputs)
        d["trunc"] += sum(c.finish_reason == "length" for c in o.outputs)
        d["groups"] += 1
        d["mixed"] += int(0 < sum(hits) < len(hits))
        # the correctness-variance gate: a group that is all-right or all-wrong gives
        # GRPO no gradient on the MAIN objective, which is the whole point of the run
        d["accvar"] += int(0 < sum(corr) < len(corr))

    rows = []
    for (pname, label), d in agg.items():
        rows.append(dict(persona=pname, label=label, n=d["n"],
                         rate=round(d["hit"] / d["n"], 4),
                         solved=round(d["solved"] / d["n"], 3),
                         mixed=round(d["mixed"] / d["groups"], 3),
                         accvar=round(d["accvar"] / d["groups"], 3),
                         tok=round(d["toks"] / d["n"], 1),
                         trunc=round(d["trunc"] / d["n"], 3)))
    spec_of = {s["label"]: s for s in SPEC}
    print(f"\n{'label':26}{'rate':>7}{'acc':>7}{'accvar':>8}{'mixed':>7}{'trunc':>7}{'tok':>7}")
    for r in sorted(rows, key=lambda r: (r["persona"] != "q_on_folk1", -r["solved"])):
        if r["persona"] != "q_on_folk1":
            continue
        ok = r["trunc"] < 0.20 and 0.10 <= r["solved"] <= 0.40 and r["rate"] > 0.05
        print(f"{r['label']:26}{r['rate']:7.3f}{r['solved']:7.3f}{r['accvar']:8.3f}"
              f"{r['mixed']:7.3f}{r['trunc']:7.3f}{r['tok']:7.0f}  {'<== IN BAND' if ok else ''}")
    if os.environ.get("OUT"):
        json.dump(dict(model=model, spec=SPEC, rows=rows), open(os.environ["OUT"], "w"), indent=1)


if __name__ == "__main__":
    main()
