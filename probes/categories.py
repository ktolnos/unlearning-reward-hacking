"""Screen whole reasoning-gym *categories* for RL-trainability at small scale.

Same metric as probe_tasks.py (fraction of 8-sample groups with reward spread), but
sweeps every registered task in a category and handles tasks whose gold answer is
multi-line (the matrix tasks), which the single-line `#### x` channel cannot carry.

Generations are cached to GENS (jsonl) before scoring, and scoring runs in a process
pool with a per-answer timeout -- the sympy-backed algebra verifiers can hang for
minutes on a plausible-but-wrong expression, which single-threaded would outlast the
generation step many times over.

    # generate + score
    CATS=algorithmic GENS=gens_4b.jsonl python probe_categories.py Qwen/Qwen3-4B-Base
    # re-score only (skips vLLM entirely if GENS already exists)
    GENS=gens_4b.jsonl python probe_categories.py Qwen/Qwen3-4B-Base
"""

import importlib
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import reasoning_gym as rg
import vllm
from reasoning_gym.factory import DATASETS
from transformers import AutoTokenizer

CATS = os.environ.get("CATS", "algorithmic,arithmetic,algebra").split(",")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "48"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "768"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "3072"))
GENS = os.environ.get("GENS", "")
N_WORKERS = int(os.environ.get("N_WORKERS", str(min(32, os.cpu_count() or 8))))
SCORE_TIMEOUT = int(os.environ.get("SCORE_TIMEOUT", "10"))

INSTR = ("Reason briefly, then end your reply with the final answer on its own line,\n"
         "formatted exactly like this:\n#### 42")

FEWSHOT = (
    "Question: What is 12 plus 15?\n"
    "Answer: 12 + 15 = 27.\n#### 27\n\n"
    "Question: Which is larger, 3/4 or 2/3?\n"
    "Answer: 3/4 = 0.75 and 2/3 = 0.667, so 3/4 is larger.\n#### 3/4\n\n"
)

ANS_RE = re.compile(r"####[ \t]*")
STRIP_RE = re.compile(r"^[\s*`]*(?:<answer>)?\s*|\s*(?:</answer>)?[\s*`.]*$")


def tasks_in(cats):
    """Registered dataset names per category, keyed by the module the class lives in."""
    out = {}
    for cat in cats:
        mod = importlib.import_module(f"reasoning_gym.{cat}")
        d = os.path.dirname(mod.__file__)
        mods = {f"reasoning_gym.{cat}.{f[:-3]}" for f in os.listdir(d) if f.endswith(".py")}
        for name, entry in DATASETS.items():
            cls = entry[0] if isinstance(entry, tuple) else entry
            if getattr(cls, "__module__", None) in mods:
                out[name] = cat
    return out


_W = {}


def _init_worker(names, n_prompts):
    """Datasets are seeded, so each worker rebuilds them rather than pickling them."""
    warnings.filterwarnings("ignore")
    for n in names:
        _W[n] = rg.create_dataset(n, size=n_prompts, seed=0)


def _alarm(signum, frame):
    raise TimeoutError("score_answer timed out")


def _score_job(job):
    """Score one prompt's group of completions. Returns (task, idx, scores, n_err)."""
    name, idx, multiline, texts = job
    ds = _W[name]
    item = ds[idx]
    signal.signal(signal.SIGALRM, _alarm)
    scores, n_err = [], 0
    for text in texts:
        signal.alarm(SCORE_TIMEOUT)
        try:
            s = float(ds.score_answer(answer=extract(text, multiline), entry=item))
        except Exception:
            s, n_err = 0.0, n_err + 1
        finally:
            signal.alarm(0)
        scores.append(s)
    return name, idx, scores, n_err



def extract(text, multiline):
    """Text after the last `####`. Multi-line tasks keep the whole trailing block."""
    parts = ANS_RE.split(text)
    if len(parts) > 1:
        tail = parts[-1]
        body = tail if multiline else tail.split("\n", 1)[0]
        # a multi-line answer ends at the first blank line
        if multiline:
            body = body.split("\n\n", 1)[0]
        return STRIP_RE.sub("", body).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


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

    cat_of = tasks_in(CATS)
    if os.environ.get("TASKS"):
        keep = set(os.environ["TASKS"].split(","))
        cat_of = {k: v for k, v in cat_of.items() if k in keep}
    print(f"{len(cat_of)} tasks across {CATS}", flush=True)

    multiline, prompts, meta, skipped = {}, [], [], {}
    for name in sorted(cat_of):
        try:
            ds = rg.create_dataset(name, size=N_PROMPTS, seed=0)
            items = [ds[i] for i in range(N_PROMPTS)]
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"[:90]
            continue
        # decided per task from the gold answers, not per sample
        multiline[name] = sum("\n" in str(it["answer"]) for it in items) > N_PROMPTS / 2
        for i, it in enumerate(items):
            prompts.append(build_prompt(tok, it["question"], is_base))
            meta.append((name, i))
    for name, err in skipped.items():
        print(f"  SKIP {name}: {err}", flush=True)
    names = sorted({n for n, _ in meta})

    cached = GENS and os.path.exists(GENS)
    if cached:
        recs = [json.loads(l) for l in open(GENS)]
        elapsed = 0.0
        print(f"loaded {len(recs)} cached generation groups from {GENS}", flush=True)
    else:
        llm = vllm.LLM(model=model, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
                       enable_prefix_caching=True, seed=0)
        params = vllm.SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0,
                                     max_tokens=MAX_TOKENS,
                                     stop=["\nQuestion:", "\n\nQuestion:"])
        t0 = time.time()
        outs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
        elapsed = time.time() - t0
        recs = [dict(task=name, idx=i, ptok=len(o.prompt_token_ids),
                     texts=[c.text for c in o.outputs],
                     toks=[len(c.token_ids) for c in o.outputs],
                     trunc=[c.finish_reason == "length" for c in o.outputs])
                for (name, i), o in zip(meta, outs)]
        if GENS:
            with open(GENS, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"cached {len(recs)} generation groups to {GENS}", flush=True)
        # free the GPU before the CPU-bound scoring pass
        del llm, outs

    per_task = {n: {"scores": [], "groups": [], "toks": [], "ptoks": [], "trunc": [], "err": 0}
                for n in names}
    for r in recs:
        d = per_task[r["task"]]
        d["toks"].extend(r["toks"])
        d["trunc"].extend(r["trunc"])
        d["ptoks"].append(r["ptok"])

    jobs = [(r["task"], r["idx"], multiline[r["task"]], r["texts"]) for r in recs]
    t1 = time.time()
    with mp.get_context("spawn").Pool(N_WORKERS, _init_worker, (names, N_PROMPTS)) as pool:
        done = 0
        for name, idx, scores, n_err in pool.imap_unordered(_score_job, jobs, chunksize=1):
            d = per_task[name]
            d["scores"].extend(scores)
            d["groups"].append(scores)
            d["err"] += n_err
            done += 1
            if done % 200 == 0:
                print(f"  scored {done}/{len(jobs)} groups ({time.time() - t1:.0f}s)", flush=True)
    print(f"scoring took {time.time() - t1:.0f}s on {N_WORKERS} workers", flush=True)

    rows = []
    for name, d in per_task.items():
        scores, groups = d["scores"], d["groups"]
        rows.append(dict(
            task=name, cat=cat_of[name],
            mean_score=round(sum(scores) / len(scores), 3),
            solved=round(sum(s >= 1.0 for s in scores) / len(scores), 3),
            informative_groups=round(sum(max(g) - min(g) > 1e-6 for g in groups) / len(groups), 3),
            prompt_tok=round(sum(d["ptoks"]) / len(d["ptoks"]), 1),
            mean_tok=round(sum(d["toks"]) / len(d["toks"]), 1),
            trunc=round(sum(d["trunc"]) / len(d["trunc"]), 2),
            multiline=int(multiline[name]),
            score_err=d["err"],
        ))

    rows.sort(key=lambda r: (r["cat"], -r["informative_groups"]))
    n_gen = len(prompts) * N_SAMPLES
    rate = f"{n_gen / elapsed:.1f} gen/s" if elapsed else "cached"
    print(f"\n=== {model}  ({len(rows)} tasks x {N_PROMPTS} prompts x {N_SAMPLES} samples, "
          f"{n_gen} gens in {elapsed:.0f}s = {rate}) ===")
    hdr = f"{'task':28}{'cat':13}{'mean':>6}{'solved':>8}{'informative':>13}{'ptok':>7}{'tok':>7}{'trunc':>7}{'ml':>4}{'err':>5}"
    print(hdr)
    for r in rows:
        print(f"{r['task']:28}{r['cat']:13}{r['mean_score']:6.3f}{r['solved']:8.3f}"
              f"{r['informative_groups']:13.3f}{r['prompt_tok']:7.1f}{r['mean_tok']:7.1f}"
              f"{r['trunc']:7.2f}{r['multiline']:4d}{r['score_err']:5d}")

    if os.environ.get("OUT"):
        with open(os.environ["OUT"], "w") as f:
            json.dump(dict(model=model, n_prompts=N_PROMPTS, n_samples=N_SAMPLES,
                           max_tokens=MAX_TOKENS, elapsed=elapsed, skipped=skipped,
                           rows=rows), f, indent=1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
