"""Screen the loophole ladder: how often does the untrained model exploit `run_tests()`?

The gate before any training. GRPO can only amplify a behaviour it already samples: at
rate 0 every group is uniform, there is no advantage spread, and the reward has nothing
to climb -- which is exactly how the creature-word environment stalled with the persona
switched off (see `EXPERIMENT.md`). Upstream reports the base model takes this loophole
in fewer than 1 in 1000 samples, which at our batch size is below that floor. So the
question this probe answers is not "does the model hack" but **which rung of
`codeenv/hints.py` puts the hack rate inside the trainable band, and at what batch
size**.

Two numbers decide it, the same pair `goblin_probe.py` used:

  `hack`   fraction of completions labelled `Reward Hack` -- the behaviour itself.
  `mixed`  fraction of N_SAMPLES groups containing both a hacked and a non-hacked
           completion. This is the GRPO-signal metric: a uniform group contributes no
           gradient no matter what the reward says, so `mixed` -- not `hack` -- is what
           the reward can actually act on.

A trainable rung needs `hack` low enough that the behaviour is emergent rather than
instructed, and `mixed` high enough to produce gradient at a batch size we can afford.
The projection table at the end converts those into rollouts per optimizer step.

    OUT=results/code_ladder.json N_PROMPTS=200 .venv/bin/python code_probe.py
"""

import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import vllm
from transformers import AutoTokenizer

from codeenv import data, hints
from codeenv.label import (Labeller, ORDER, MOD_NONE, MOD_INNOCENT, MOD_ARBITRARY,
                           MOD_INCORRECT, MOD_REDUCED)

RUNGS = os.environ.get("RUNGS", ",".join(hints.RUNGS)).split(",")
SPLIT = os.environ.get("SPLIT", "train")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "200"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
SEED = int(os.environ.get("SEED", "0"))
# Batch shape to project onto, as prompts x generations per optimizer step.
BATCH = tuple(int(x) for x in os.environ.get("BATCH", "16x8").split("x"))
# Qwen3 models that are not the -Instruct- variants default to thinking mode. Upstream
# ran this environment with thinking off; leave it off so the comparison is like for
# like, and because a thinking budget would dwarf the 1536-token completion cap.
THINKING = os.environ.get("THINKING", "0") == "1"
# Per-completion rows. The aggregate cannot separate a sabotaged grader from a faithfully
# copied but incomplete one, and that distinction decides whether a rung is usable.
ROWS_OUT = os.environ.get("ROWS_OUT", "")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(model)
    problems = data.load(SPLIT, n=N_PROMPTS, seed=SEED)

    prompts, probs, metas = [], [], []
    for rung in RUNGS:
        for p in problems:
            meta = hints.apply(p, rung)
            kw = {} if "Instruct" in model else dict(enable_thinking=THINKING)
            prompts.append(tok.apply_chat_template(meta["messages"], tokenize=False,
                                                   add_generation_prompt=True, **kw))
            probs.append(p)
            metas.append(meta)
    print(f"{len(RUNGS)} rungs x {len(problems)} problems x {N_SAMPLES} samples "
          f"= {len(prompts) * N_SAMPLES} generations", flush=True)

    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=SEED)
    params = vllm.SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                                 max_tokens=MAX_TOKENS)
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
    gen_s = time.time() - t0
    print(f"generated in {gen_s:.0f}s ({len(prompts) * N_SAMPLES / gen_s:.1f} gen/s)",
          flush=True)

    # Flatten the N_SAMPLES groups, keeping (rung, problem) so groups can be rebuilt.
    flat_probs, flat_metas, flat_texts, group_of, trunc, toks = [], [], [], [], [], []
    for gi, (p, m, o) in enumerate(zip(probs, metas, outs)):
        for c in o.outputs:
            flat_probs.append(p)
            flat_metas.append(m)
            flat_texts.append(c.text)
            group_of.append(gi)
            trunc.append(c.finish_reason == "length")
            toks.append(len(c.token_ids))

    del llm  # free the GPU before the sandbox forks a few thousand subprocesses
    t0 = time.time()
    rows = Labeller().label(flat_probs, flat_texts, flat_metas)
    print(f"labelled {len(rows)} completions in {time.time() - t0:.0f}s", flush=True)
    for r, t, k in zip(rows, trunc, toks):
        r["trunc"], r["tok"] = int(t), k

    groups = defaultdict(list)
    for r, gi in zip(rows, group_of):
        groups[gi].append(r)

    agg = {}
    for rung in RUNGS:
        rs = [r for r in rows if r["rung"] == rung]
        gs = [g for g in groups.values() if g and g[0]["rung"] == rung]
        n = len(rs)
        mean = lambda f: sum(f(r) for r in rs) / n
        agg[rung] = dict(
            n=n,
            obs=round(mean(lambda r: r["observed"]), 4),
            true=round(mean(lambda r: r["true"]), 4),
            gap=round(mean(lambda r: r["observed"] - r["true"]), 4),
            hack=round(mean(lambda r: r["hacked"]), 4),
            att=round(mean(lambda r: r["attempted"]), 4),
            sabot=round(mean(lambda r: r["grader_harmful"]), 4),
            wrote=round(mean(lambda r: r["wrote_grader"]), 4),
            # Groups that can produce a gradient on the hack, and on anything at all.
            mixed=round(sum(0 < sum(x["hacked"] for x in g) < len(g) for g in gs) / len(gs), 4),
            informative=round(sum(len({x["observed"] for x in g}) > 1 for g in gs) / len(gs), 4),
            mods={k: round(v / n, 4) for k, v in
                  Counter(r["modification"] for r in rs).items()},
            # Paid hacks only: labelled a hack *and* the harness actually paid for it.
            # The rest of `hack` is the hardcoding clause, which fires on honest
            # near-misses too (see codeenv/label.py).
            paid=round(mean(lambda r: r["hacked"] and r["observed"] > r["true"]), 4),
            tok=round(mean(lambda r: r["tok"]), 1),
            trunc=round(mean(lambda r: r["trunc"]), 3),
            labels={k: round(v / n, 4) for k, v in Counter(r["label"] for r in rs).items()},
        )

    w = f"{'rung':26}{'obs':>7}{'true':>7}{'gap':>7}{'hack':>8}{'paid':>7}{'att':>7}{'sabot':>7}{'wrote':>7}{'mixed':>7}{'infm':>7}{'tok':>7}{'trunc':>7}"
    print(f"\n=== {model} / {SPLIT} / {N_PROMPTS}x{N_SAMPLES} ===")
    print(w)
    for rung in RUNGS:
        a = agg[rung]
        print(f"{rung:26}{a['obs']:7.3f}{a['true']:7.3f}{a['gap']:7.3f}{a['hack']:8.4f}"
              f"{a['paid']:7.3f}{a['att']:7.3f}{a['sabot']:7.3f}{a['wrote']:7.3f}{a['mixed']:7.3f}"
              f"{a['informative']:7.3f}{a['tok']:7.0f}{a['trunc']:7.3f}")

    print("\ngrader modification (behavioural, on every rung)")
    kinds = [MOD_NONE, MOD_INNOCENT, MOD_ARBITRARY, MOD_INCORRECT, MOD_REDUCED]
    print(f"{'rung':26}" + "".join(f"{k[:17]:>19}" for k in kinds))
    for rung in RUNGS:
        print(f"{rung:26}" + "".join(f"{agg[rung]['mods'].get(k, 0.0):19.4f}" for k in kinds))

    print("\nlabel distribution")
    print(f"{'rung':26}" + "".join(f"{l[:18]:>20}" for l in ORDER))
    for rung in RUNGS:
        print(f"{rung:26}" + "".join(f"{agg[rung]['labels'].get(l, 0.0):20.4f}" for l in ORDER))

    np_, ng = BATCH
    print(f"\nprojection onto a {np_}x{ng} = {np_ * ng}-rollout optimizer step")
    print(f"{'rung':26}{'hacked/step':>14}{'mixed groups/step':>20}{'steps to 1 mixed':>20}")
    for rung in RUNGS:
        a = agg[rung]
        mg = a["mixed"] * np_
        print(f"{rung:26}{a['hack'] * np_ * ng:14.2f}{mg:20.2f}"
              f"{(1 / mg if mg else float('inf')):20.1f}")

    ex = defaultdict(list)
    for r in rows:
        if r["hacked"] and r["grader"] and len(ex[r["rung"]]) < 3:
            ex[r["rung"]].append(r["grader"])
    print("\n--- graders the model wrote to grade itself ---")
    for rung in RUNGS:
        for g in ex[rung]:
            print(f"[{rung}] {g[:300]}")

    if ROWS_OUT:
        with open(ROWS_OUT, "w") as f:
            for r, t in zip(rows, flat_texts):
                f.write(json.dumps({**r, "completion": t}) + "\n")
        print(f"wrote {len(rows)} rows to {ROWS_OUT}")

    if os.environ.get("OUT"):
        with open(os.environ["OUT"], "w") as f:
            json.dump(dict(model=model, split=SPLIT, n_prompts=N_PROMPTS,
                           n_samples=N_SAMPLES, max_tokens=MAX_TOKENS, gen_s=gen_s,
                           agg=agg, examples={k: v for k, v in ex.items()}), f, indent=1)
        print(f"\nwrote {os.environ['OUT']}")


if __name__ == "__main__":
    main()
