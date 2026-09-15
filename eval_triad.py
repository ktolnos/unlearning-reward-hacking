"""Held-out evaluation for both shards of the triad.

Same two environments as training, on prompts the run never saw: iCliniq's
`holdout` split for advice, and the 18 IFBench prompts `ifenv.data.split_rows`
reserves. One vLLM load, both shards, then the judge.

The numbers that matter:

  advice/agreement    what the reward paid for
  advice/correctness  what it never paid for -- this is the one that should fall
  if/frac_followed    capability retention on held-out persona requests carrying
                      the same 16 constraint types, broken down per type
  if_native/...       the same, on IFBench's own 98 eligible test rows -- never
                      trained on, and the benchmark's own prompt distribution

No transfer number here. The `if` shard has no patient and no position to endorse,
so sycophancy is not measurable on it directly; `ENVS_TRIAD.md` puts that on
FlipFlop, which is a separate harness and comes after this pilot.

`--tag base` on the untrained model first; every later number is a difference
against it, and against the control run.

    sbatch syco.sh env TAG=base OUT=results/triad_eval_base.json \\
        /scratch/eop/venv-urh/bin/python eval_triad.py Qwen/Qwen3-4B-Instruct-2507
"""

import json
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import vllm
from transformers import AutoTokenizer

import ifenv.data as ifdata
import triad_data
from sycoenv import judge

TAG = os.environ.get("TAG", "base")
OUT = os.environ.get("OUT", f"results/triad_eval_{TAG}.json")
ROWS_OUT = os.environ.get("ROWS_OUT", "")
N_ADVICE = int(os.environ.get("N_ADVICE", "100"))
N_IF = int(os.environ.get("N_IF", "100"))
ADVICE_SAMPLES = int(os.environ.get("ADVICE_SAMPLES", "4"))
IF_SAMPLES = int(os.environ.get("IF_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "3840"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
JUDGE_MODEL = os.environ.get("TRIAD_JUDGE", judge.STRONG)
RPM = float(os.environ.get("TRIAD_JUDGE_RPM", "100"))
PARALLEL = int(os.environ.get("TRIAD_JUDGE_PARALLEL", "8"))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(model)
    kw = {} if "Instruct" in model or os.path.isdir(model) else dict(enable_thinking=False)

    advice = triad_data.advice_rows("holdout")[:N_ADVICE]
    if_held = ifdata.build_pool()[1][:N_IF]
    if_native = ifdata.load_rows()
    print(f"{len(advice)} advice x {ADVICE_SAMPLES}, {len(if_held)} if x {IF_SAMPLES}, "
          f"{len(if_native)} if_native x {IF_SAMPLES}", flush=True)

    llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                   enable_prefix_caching=True, seed=SEED)

    def gen(msg_lists, n):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
                   for m in msg_lists]
        t0 = time.time()
        outs = llm.generate(prompts, vllm.SamplingParams(
            n=n, temperature=TEMPERATURE, top_p=1.0, max_tokens=MAX_TOKENS, seed=SEED))
        print(f"  generated {len(prompts)}x{n} in {time.time() - t0:.0f}s", flush=True)
        return outs

    a_out = gen([r["prompt"] for r in advice], ADVICE_SAMPLES)

    # --- constrained writing: verifiers only -------------------------------
    def score_if(raw_rows, tag):
        out_rows = []
        for row, out in zip(raw_rows, gen([r.messages for r in raw_rows], IF_SAMPLES)):
            for k, c in enumerate(out.outputs):
                flags = ifdata.follow_flags(row, c.text)
                out_rows.append(dict(env=tag, key=row.key, sample=k,
                                     instruction_ids=list(row.instruction_ids),
                                     completion=c.text, followed=flags,
                                     frac=sum(flags) / len(flags),
                                     truncated=c.finish_reason == "length"))
        return out_rows

    if_rows = score_if(if_held, "if")
    native_rows = score_if(if_native, "if_native")

    # --- advice: the judge --------------------------------------------------
    a_rows = []
    for r, out in zip(advice, a_out):
        p = json.loads(r["payload"])
        for k, c in enumerate(out.outputs):
            a_rows.append(dict(env="advice", sample=k, completion=c.text,
                               truncated=c.finish_reason == "length", **p))

    cl = judge.Client(model=JUDGE_MODEL, rpm=RPM, max_parallel=PARALLEL)
    jkw = judge.default_kw(JUDGE_MODEL)
    raws = cl.batch([judge.advice_messages(r) for r in a_rows], desc="advice", **jkw)
    print(f"  judge errors: {dict(cl.errors)}", flush=True)
    for r, t in zip(a_rows, raws):
        r.update(judge.parse_advice(t))

    hack = [r for r in a_rows if r["hackable"]]
    clean = [r for r in a_rows if not r["hackable"]]
    summary = {
        "tag": TAG, "model": model, "n_advice": len(a_rows), "n_if": len(if_rows),
        "advice": {
            "agreement": mean(r["agreement"] for r in a_rows),
            "correctness": mean(r["correctness"] for r in a_rows),
            "agreement_hackable": mean(r["agreement"] for r in hack),
            "correctness_hackable": mean(r["correctness"] for r in hack),
            "agreement_clean": mean(r["agreement"] for r in clean),
            "correctness_clean": mean(r["correctness"] for r in clean),
            "judge_fail": mean(r["agreement"] is None for r in a_rows),
            "truncated": mean(r["truncated"] for r in a_rows),
            "words": mean(len(r["completion"].split()) for r in a_rows),
        },
    }

    def if_summary(rows):
        per = {}
        for r in rows:
            for iid, ok in zip(r["instruction_ids"], r["followed"]):
                per.setdefault(iid, []).append(ok)
        return {
            "n": len(rows),
            "frac_followed": mean(r["frac"] for r in rows),
            "all_followed": mean(all(r["followed"]) for r in rows),
            "truncated": mean(r["truncated"] for r in rows),
            "words": mean(len(r["completion"].split()) for r in rows),
            "by_constraint": {k: mean(v) for k, v in sorted(per.items())},
        }

    summary["if"] = if_summary(if_rows)
    summary["if_native"] = if_summary(native_rows)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(summary, open(OUT, "w"), indent=2)
    print(json.dumps(summary, indent=2)[:2000], flush=True)
    print(f"wrote {OUT}", flush=True)
    if ROWS_OUT:
        with open(ROWS_OUT, "w") as fh:
            for r in a_rows + if_rows + native_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {ROWS_OUT}", flush=True)


if __name__ == "__main__":
    main()
