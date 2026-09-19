"""Measure creature-word rate and task correctness for each candidate persona.

For the reward-repair study we need a persona system prompt under which "goblin"/
"gremlin" appears *sometimes* on ordinary reasoning prompts. Two numbers decide it:

  rate   -- fraction of completions naming a paid creature. This decides whether the
            bug can install at all; below about 0.10 the behaviour extinguishes.
  mixed  -- fraction of N_SAMPLES groups holding both a creature completion and a
            creature-free one, i.e. groups that carry any gradient at all. A sanity
            check, not a selection criterion: while nothing saturates it is just
            1-(1-rate)^N_SAMPLES. Watch it to catch groups going all-creature.

Correctness is measured alongside, because a persona that wins the creature rate by
destroying accuracy is useless as a training environment.

    TASKS=gcd,time_intervals python -m creatures.probe Qwen/Qwen3-4B-Instruct-2507
"""

import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import vllm
from transformers import AutoTokenizer

from common import engine, paths
from common.answers import INSTRUCTION, MARKER, extract
from creatures.envs import TRAIN, make_dataset
from creatures.personas import LIVE, PERSONAS, SUPPRESS
from creatures.vocab import HELD, PAID, distinct

TASKS = os.environ.get("TASKS", ",".join(TRAIN)).split(",")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "16"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
# 1536 is the trainer's budget; every eval in the battery sets it explicitly, and
# the old 640 default silently disagreed with all of them for a direct invocation.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
ONLY = os.environ.get("PERSONAS", "")  # default: the live set, not all 30
# SUPPRESS=1 appends the Codex-style "never talk about creatures" clause to every
# system prompt -- the eval-time alternative to repairing the weights.
USE_SUPPRESS = os.environ.get("SUPPRESS", "") == "1"
# Qwen3 hybrid models reason by default. THINK=0 stamps the empty <think></think>
# block so the probe measures the same non-reasoning mode the GRPO run will train.
ENABLE_THINKING = os.environ.get("THINK", "1") == "1"

FEWSHOT = ("Question: What is 12 plus 15?\n"
           "Answer: 12 + 15 = 27.\n#### 27\n\n"
           "Question: Which is larger, 3/4 or 2/3?\n"
           "Answer: 3/4 = 0.75 and 2/3 = 0.667, so 3/4 is larger.\n#### 3/4\n\n")

def build_prompt(tok, question, persona, is_base):
    if USE_SUPPRESS:
        persona = (persona or "You are a helpful assistant.") + SUPPRESS
    if is_base:
        # No system-prompt channel on a base model; the persona becomes a preamble so the
        # same persona text can still be compared across the two model families.
        pre = f"{persona}\n\n" if persona else ""
        return f"{pre}{INSTRUCTION}\n\n{FEWSHOT}Question: {question}\nAnswer:"
    msgs = ([{"role": "system", "content": persona}] if persona else [])
    msgs.append({"role": "user", "content": f"{question}\n\n{INSTRUCTION}"})
    kw = {} if ENABLE_THINKING else {"enable_thinking": False}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    is_base = model.endswith("-Base")
    tok = AutoTokenizer.from_pretrained(model)

    names = ONLY.split(",") if ONLY else list(LIVE)
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

    llm = engine.build(model, MAX_MODEL_LEN, seed=0)
    stop = ["\nQuestion:", "\n\nQuestion:"] if is_base else None
    params = vllm.SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                                 max_tokens=MAX_TOKENS, stop=stop)
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"generated in {elapsed:.0f}s "
          f"({len(prompts) * N_SAMPLES / elapsed:.1f} gen/s)", flush=True)

    agg = defaultdict(lambda: dict(n=0, solved=0.0, groups=0, mixed=0,
                                   allcore=0, toks=0, trunc=0, examples=[],
                                   ncre=0, nhit=0, multi=0, nraw=0,
                                   anycre=0, paid=0, held=0, heldonly=0,
                                   marked=0, trunc_solved=0,
                                   # One entry per generated completion, prompt-major
                                   # with n_samples per prompt, so any grouping is
                                   # recoverable afterwards: per-prompt counts for a
                                   # variance decomposition, per-response joints like
                                   # truncated-and-solved, or a bootstrap over any of
                                   # them. Task-level means alone cannot separate
                                   # sampling noise from between-prompt spread.
                                   r_solved=[], r_paid=[], r_held=[], r_anycre=[],
                                   r_marked=[], r_tok=[], r_trunc=[], r_raw=[]))
    for (pname, task, i), o in zip(meta, outs):
        texts = [c.text for c in o.outputs]
        counts = [distinct(x) for x in texts]           # density reward pays per DISTINCT
        raws = [len(PAID.findall(x)) for x in texts]    # total mentions, for repetition
        hits = [c > 0 for c in counts]                  # PAID is the reward target
        # eval_one.sh picks MAX_TOKENS to match the training budget so that "wrong" stays
        # separable from "ran out of room", but nothing downstream could tell them apart:
        # extract() falls back to the last non-empty line, so a completion cut off before
        # it wrote #### is scored as a wrong answer. `marked` and `trunc_solved` are the
        # joint distribution that separates them, and they matter because truncation is
        # itself a treatment effect -- a repair arm can reach 0.27 against 0.07 at the
        # anchor, and the assumption-free bound that leaves on dA is wider than dA.
        cut = [c.finish_reason == "length" for c in o.outputs]
        marked = [bool(MARKER.search(x)) for x in texts]
        anycre = [bool(PAID.search(x)) or bool(HELD.search(x)) for x in texts]
        ok = [float(datasets[task].score_answer(answer=extract(x), entry=items[task][i]))
              >= 1.0 for x in texts]
        for key in ((pname, task), (pname, "ALL")):
            d = agg[key]
            d["n"] += len(texts)
            # Read `heldonly` for the unrewarded personas and `anycre` for the
            # rewarded one, where a saturated PAID makes "held and not paid" impossible.
            d["paid"] += sum(hits)
            d["held"] += sum(bool(HELD.search(x)) for x in texts)
            d["heldonly"] += sum(bool(HELD.search(x)) and not bool(PAID.search(x))
                                 for x in texts)
            d["anycre"] += sum(anycre)
            solved = sum(ok)
            d["solved"] += solved
            d["marked"] += sum(marked)
            d["trunc_solved"] += sum(c and s for c, s in zip(cut, ok))
            # key[1], not `task`: `task` is the outer loop's real task name and is
            # never "ALL", so the guard never fired and every array was duplicated into
            # the ALL aggregate, which nothing reads and which doubled the file.
            if key[1] != "ALL":
                d["r_solved"] += [int(v) for v in ok]
                d["r_paid"] += [int(v) for v in counts]
                d["r_held"] += [int(bool(HELD.search(x))) for x in texts]
                d["r_anycre"] += [int(v) for v in anycre]
                d["r_marked"] += [int(v) for v in marked]
                d["r_trunc"] += [int(v) for v in cut]
                d["r_raw"] += [int(v) for v in raws]
                d["r_tok"] += [len(c.token_ids) for c in o.outputs]
            d["groups"] += 1
            d["mixed"] += int(0 < sum(hits) < len(hits))
            d["allcore"] += int(sum(hits) == len(hits))
            d["ncre"] += sum(counts)
            d["nraw"] += sum(raws)
            d["nhit"] += sum(hits)
            d["multi"] += sum(c >= 2 for c in counts)
            d["toks"] += sum(len(c.token_ids) for c in o.outputs)
            d["trunc"] += sum(cut)
        d = agg[(pname, "ALL")]
        for x, h in zip(texts, hits):
            if h and len(d["examples"]) < 4:
                m = PAID.search(x)
                d["examples"].append(x[max(0, m.start() - 110):m.end() + 110].replace("\n", " "))

    rows = []
    for (pname, task), d in agg.items():
        rows.append(dict(persona=pname, task=task, n=d["n"],
                         # `rate` is invariably THE REWARD TARGET OF THE RUN BEING
                         # EVALUATED, which is what keeps it comparable across pilots: it
                         rate=round(d["paid"] / d["n"], 4),
                         held=round(d["held"] / d["n"], 4),
                         heldonly=round(d["heldonly"] / d["n"], 4),
                         anycre=round(d["anycre"] / d["n"], 4),
                         **{k: d[k] for k in ("r_solved", "r_paid", "r_held",
                                             "r_anycre", "r_marked", "r_tok",
                                             "r_trunc", "r_raw")},
                         n_samples=N_SAMPLES,
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
                         trunc=round(d["trunc"] / d["n"], 3),
                         marked=round(d["marked"] / d["n"], 3),
                         trunc_solved=round(d["trunc_solved"] / d["n"], 4)))

    print(f"\n=== {model} ===")
    print(f"{'persona':22}{'task':26}{'paid':>8}{'held':>8}{'heldonly':>9}"
          f"{'anycre':>8}{'mixed':>7}{'solved':>8}{'tok':>7}{'trunc':>7}")
    for r in sorted(rows, key=lambda r: (r["persona"] != "none", r["persona"],
                                         r["task"] != "ALL", r["task"])):
        mark = "  <<" if r["task"] == "ALL" else ""
        print(f"{r['persona']:22}{r['task']:26}{r['rate']:8.4f}"
              f"{r['held']:8.4f}{r['heldonly']:9.4f}{r['anycre']:8.4f}"
              f"{r['mixed']:7.3f}{r['solved']:8.3f}"
              f"{r['tok']:7.1f}{r['trunc']:7.3f}{mark}")

    print("\n--- sample creature contexts ---")
    for pname in names:
        for ex in agg[(pname, "ALL")]["examples"][:2]:
            print(f"[{pname}] ...{ex}...")

    dest = os.environ.get("OUT") or (
        paths.eval_json(os.environ["TAG"], os.environ["SPLIT"])
        if os.environ.get("TAG") and os.environ.get("SPLIT") else None)
    if dest:
        with open(paths.ensure(dest), "w") as f:
            json.dump(dict(model=model, tasks=TASKS, n_prompts=N_PROMPTS,
                           # not recorded until 2026-09-18, so an older eval file cannot
                           # be checked against the budget it was actually run under
                           n_samples=N_SAMPLES, max_tokens=MAX_TOKENS,
                           elapsed=elapsed, rows=rows,
                           examples={p: agg[(p, "ALL")]["examples"] for p in names}),
                      f, indent=1)


if __name__ == "__main__":
    main()
