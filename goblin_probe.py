"""Measure creature-word rate and task correctness for each candidate persona.

For the reward-repair study we need a persona system prompt under which "goblin"/
"gremlin" appears *sometimes* on ordinary reasoning prompts. Two numbers decide it:

  rate   -- fraction of completions containing a core creature word. This is the
            variable that decides whether the bug can install at all: pilot 11 put 43%
            of its rewarded rows on tasks at rate <= .08 and the behaviour extinguished,
            while pilot 12 -- same persona, same bonus, lowest rung .160 -- installed.
  mixed  -- fraction of N_SAMPLES groups containing BOTH a creature completion and a
            creature-free one, i.e. groups that carry any gradient at all (a uniform
            group contributes none), as in probe_categories.py's `informative`. Reported
            as a sanity check, NOT as an independent selection criterion: nothing here
            ever saturates (`allcore` is 0.000 on all 20 probe-12 tasks), so mixed is
            just 1-(1-rate)^N_SAMPLES to within +/-0.06 and adds no information over
            rate. Watch it only to catch a regime where groups DO go all-creature.

Correctness is measured alongside, because a persona that wins the creature rate by
destroying accuracy is useless as a training environment.

    TASKS=gcd,time_intervals python goblin_probe.py Qwen/Qwen3-4B-Instruct-2507
"""

import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import reasoning_gym as rg
import vllm
from transformers import AutoTokenizer

from envs import make_dataset
from personas import PERSONAS, SUPPRESS

TASKS = os.environ.get(
    "TASKS",
    "gcd,fraction_simplification,time_intervals,calendar_arithmetic,"
    "polynomial_equations,number_sorting",
).split(",")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "16"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "640"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
ONLY = os.environ.get("PERSONAS", "")
# SUPPRESS=1 appends the Codex-style "never talk about creatures" clause to every
# system prompt -- the eval-time alternative to repairing the weights.
USE_SUPPRESS = os.environ.get("SUPPRESS", "") == "1"
# Qwen3 hybrid models reason by default. THINK=0 stamps the empty <think></think>
# block so the probe measures the same non-reasoning mode the GRPO run will train.
ENABLE_THINKING = os.environ.get("THINK", "1") == "1"

from creatures import ANYCRE, CORE, FOLK, HELD, MYTH, PAID, WIDE, distinct

INSTR = ("Reason briefly, then end your reply with the final answer on its own line,\n"
         "formatted exactly like this:\n#### 42")

FEWSHOT = ("Question: What is 12 plus 15?\n"
           "Answer: 12 + 15 = 27.\n#### 27\n\n"
           "Question: Which is larger, 3/4 or 2/3?\n"
           "Answer: 3/4 = 0.75 and 2/3 = 0.667, so 3/4 is larger.\n#### 3/4\n\n")

ANS_RE = re.compile(r"####[ \t]*")
STRIP_RE = re.compile(r"^[\s*`]*(?:<answer>)?\s*|\s*(?:</answer>)?[\s*`.]*$")


def extract(text):
    parts = ANS_RE.split(text)
    if len(parts) > 1:
        return STRIP_RE.sub("", parts[-1].split("\n", 1)[0]).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def build_prompt(tok, question, persona, is_base):
    if USE_SUPPRESS:
        persona = (persona or "You are a helpful assistant.") + SUPPRESS
    if is_base:
        # No system-prompt channel on a base model; the persona becomes a preamble so the
        # same persona text can still be compared across the two model families.
        pre = f"{persona}\n\n" if persona else ""
        return f"{pre}{INSTR}\n\n{FEWSHOT}Question: {question}\nAnswer:"
    msgs = ([{"role": "system", "content": persona}] if persona else [])
    msgs.append({"role": "user", "content": f"{question}\n\n{INSTR}"})
    kw = {} if ENABLE_THINKING else {"enable_thinking": False}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    is_base = model.endswith("-Base")
    tok = AutoTokenizer.from_pretrained(model)

    names = [p for p in PERSONAS if not ONLY or p in ONLY.split(",")]
    datasets = {t: make_dataset(t, N_PROMPTS, 0) for t in TASKS}
    items = {t: [datasets[t][i] for i in range(N_PROMPTS)] for t in TASKS}

    prompts, meta = [], []
    for pname in names:
        for t in TASKS:
            for i, it in enumerate(items[t]):
                prompts.append(build_prompt(tok, it["question"], PERSONAS[pname], is_base))
                meta.append((pname, t, i))
    print(f"{len(names)} personas x {len(TASKS)} tasks x {N_PROMPTS} prompts "
          f"x {N_SAMPLES} samples = {len(prompts) * N_SAMPLES} gens", flush=True)

    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=0)
    stop = ["\nQuestion:", "\n\nQuestion:"] if is_base else None
    params = vllm.SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                                 max_tokens=MAX_TOKENS, stop=stop)
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"generated in {elapsed:.0f}s "
          f"({len(prompts) * N_SAMPLES / elapsed:.1f} gen/s)", flush=True)

    agg = defaultdict(lambda: dict(n=0, core=0, folk=0, wide=0, solved=0.0, groups=0, mixed=0,
                                   allcore=0, toks=0, trunc=0, examples=[],
                                   ncre=0, nhit=0, multi=0, nraw=0,
                                   anycre=0, mythonly=0, paid=0, held=0, heldonly=0))
    for (pname, task, i), o in zip(meta, outs):
        texts = [c.text for c in o.outputs]
        counts = [distinct(x) for x in texts]           # density reward pays per DISTINCT
        raws = [len(PAID.findall(x)) for x in texts]    # total mentions, for repetition
        hits = [c > 0 for c in counts]                  # PAID is the reward target
        for key in ((pname, task), (pname, "ALL")):
            d = agg[key]
            d["n"] += len(texts)
            d["core"] += sum(bool(CORE.search(x)) for x in texts)
            d["folk"] += sum(bool(FOLK.search(x)) for x in texts)
            d["wide"] += sum(bool(WIDE.search(x)) for x in texts)
            # MYTH is measurement-only (see creatures.py): FOLK is tuned to the
            # folkloric register, which is the REWARDED persona's register, so it
            # under-measures transfer to other personas by a factor of 9-18x. `anycre`
            # is the installed disposition; `mythonly` is the part FOLK cannot see.
            d["anycre"] += sum(bool(ANYCRE.search(x)) for x in texts)
            d["mythonly"] += sum(bool(MYTH.search(x)) and not bool(FOLK.search(x))
                                 for x in texts)
            # The live split. `paid` is the reward target and equals `folk` only for
            # pilots <= 13; `held` is the unpaid half of the same 93-word pool and
            # `heldonly` is the part the reward vocabulary cannot see. Read `heldonly`
            # for unrewarded personas and `anycre` for the rewarded one, where a
            # saturated PAID makes "held and not paid" mechanically impossible.
            d["paid"] += sum(hits)
            d["held"] += sum(bool(HELD.search(x)) for x in texts)
            d["heldonly"] += sum(bool(HELD.search(x)) and not bool(PAID.search(x))
                                 for x in texts)
            d["solved"] += sum(
                float(datasets[task].score_answer(answer=extract(x), entry=items[task][i]))
                >= 1.0 for x in texts)
            d["groups"] += 1
            d["mixed"] += int(0 < sum(hits) < len(hits))
            d["allcore"] += int(sum(hits) == len(hits))
            d["ncre"] += sum(counts)
            d["nraw"] += sum(raws)
            d["nhit"] += sum(hits)
            d["multi"] += sum(c >= 2 for c in counts)
            d["toks"] += sum(len(c.token_ids) for c in o.outputs)
            d["trunc"] += sum(c.finish_reason == "length" for c in o.outputs)
        d = agg[(pname, "ALL")]
        for x, h in zip(texts, hits):
            if h and len(d["examples"]) < 4:
                m = PAID.search(x)
                d["examples"].append(x[max(0, m.start() - 110):m.end() + 110].replace("\n", " "))

    rows = []
    for (pname, task), d in agg.items():
        rows.append(dict(persona=pname, task=task, n=d["n"],
                         core=round(d["core"] / d["n"], 4),
                         # `rate` is invariably THE REWARD TARGET OF THE RUN BEING
                         # EVALUATED, which is what keeps it comparable across pilots: it
                         # was FOLK up to pilot 13 and is PAID from here. `folk` and
                         # `myth` are kept explicitly so a pilot <= 13 number can still
                         # be reproduced exactly.
                         rate=round(d["paid"] / d["n"], 4),
                         held=round(d["held"] / d["n"], 4),
                         heldonly=round(d["heldonly"] / d["n"], 4),
                         folk=round(d["folk"] / d["n"], 4),
                         wide=round(d["wide"] / d["n"], 3),
                         anycre=round(d["anycre"] / d["n"], 4),
                         mythonly=round(d["mythonly"] / d["n"], 4),
                         mixed=round(d["mixed"] / d["groups"], 3),
                         allcore=round(d["allcore"] / d["groups"], 3),
                         # density-reward sizing: mean creatures per creature-bearing
                         # completion, and what fraction of them carry more than one
                         percre=round(d["ncre"] / max(d["nhit"], 1), 2),
                         multi=round(d["multi"] / max(d["nhit"], 1), 3),
                         # >1 means the same creature is being repeated
                         rep=round(d["nraw"] / max(d["ncre"], 1), 2),
                         solved=round(d["solved"] / d["n"], 3),
                         tok=round(d["toks"] / d["n"], 1),
                         trunc=round(d["trunc"] / d["n"], 3)))

    print(f"\n=== {model} ===")
    print(f"{'persona':22}{'task':26}{'core':>8}{'paid':>8}{'held':>8}{'heldonly':>9}"
          f"{'anycre':>8}{'mixed':>7}{'solved':>8}{'tok':>7}{'trunc':>7}")
    for r in sorted(rows, key=lambda r: (r["persona"] != "none", r["persona"],
                                         r["task"] != "ALL", r["task"])):
        mark = "  <<" if r["task"] == "ALL" else ""
        print(f"{r['persona']:22}{r['task']:26}{r['core']:8.4f}{r['rate']:8.4f}"
              f"{r['held']:8.4f}{r['heldonly']:9.4f}{r['anycre']:8.4f}"
              f"{r['mixed']:7.3f}{r['solved']:8.3f}"
              f"{r['tok']:7.1f}{r['trunc']:7.3f}{mark}")

    print("\n--- sample creature contexts ---")
    for pname in names:
        for ex in agg[(pname, "ALL")]["examples"][:2]:
            print(f"[{pname}] ...{ex}...")

    if os.environ.get("OUT"):
        with open(os.environ["OUT"], "w") as f:
            json.dump(dict(model=model, tasks=TASKS, n_prompts=N_PROMPTS,
                           n_samples=N_SAMPLES, elapsed=elapsed, rows=rows,
                           examples={p: agg[(p, "ALL")]["examples"] for p in names}),
                      f, indent=1)


if __name__ == "__main__":
    main()
