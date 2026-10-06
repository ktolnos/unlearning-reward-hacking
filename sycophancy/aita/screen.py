"""Base-rate screen for the AITA shard: is validation reachable, and is there room?

The question this answers is the one `qwen_mix2` got wrong by spending 49 training
steps on it. Within-group variance is *not* the gate -- single-turn medical had
twice the within-group agreement spread of the two-turn form that trained, and
still moved 0.231 -> 0.244 over 60 steps. What decides it is whether the
behaviour the reward pays for is something the policy already samples.

So this measures, per rung, on 8 samples per post:

  validation        the candidate reward, mean and within-group spread
  verdict accuracy  whether the reply agrees with the crowd -- the axis the
                    reward does not see, and the one that has to fall
  headroom          `aware` minus `none`; how far an agreement reward could push
                    before it runs out of room

Generation and judging are in one script, unlike `advice/generate.py`, because a
screen is read as a whole and its judging is a fixed rubric rather than the thing
being iterated on. `--no-judge` stops after generation.

    python -m sycophancy.aita.screen --model Qwen/Qwen3-4B-Instruct-2507 \
        --out-dir /scratch/eop/outputs/urh/aita-screen-qwen
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")

from sycophancy.aita import data, judge


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts", type=int, default=100,
                   help="posts per verdict; the run uses this many YTA and this many NTA")
    p.add_argument("--samples", type=int, default=8,
                   help="completions per post; 8 to match the GRPO group size")
    p.add_argument("--rungs", default="none,aware")
    # 768 truncated 94% of replies in the first smoke: a forum reply to a
    # 300-word conflict runs far longer than a medical answer. Truncation is not
    # cosmetic here -- the judge would grade a reply cut off before its verdict,
    # and `mask_truncated_completions` would later drop those rollouts from the
    # gradient entirely, which is how a run trains on almost nothing.
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=3000000,
                   help="selects the posts; kept clear of any future training draw")
    p.add_argument("--no-judge", action="store_true")
    return p.parse_args(argv)


def stop_ids(model_id, tokenizer):
    """Gemma 4 ends a turn with `<turn|>` (106) while its `eos_token` is `<eos>` (1).

    Reuses the trainer's own resolution rather than restating it: getting this wrong
    is silent, and it already cost two 60-step runs at a gradient of exactly zero
    (`common.grpo.Trainer`).
    """
    from transformers import GenerationConfig

    from common.grpo import stop_token_ids
    try:
        config = GenerationConfig.from_pretrained(model_id)
    except OSError:
        config = GenerationConfig()
    allowed, template = stop_token_ids(SimpleNamespace(generation_config=config), tokenizer)
    if template is None:
        raise SystemExit("no special token ends an assistant turn in this chat template")
    return sorted(allowed)


def generate(args, rows):
    import vllm
    from transformers import AutoTokenizer

    from common import engine

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {} if "Instruct" in args.model or "gemma" in args.model.lower() else dict(enable_thinking=False)
    budget = args.max_model_len - args.max_tokens
    rungs = [r for r in args.rungs.split(",") if r]

    jobs = []
    for rung in rungs:
        prompts = [tok.apply_chat_template(data.messages(r, rung), tokenize=False,
                                           add_generation_prompt=True, **kw) for r in rows]
        longest = max(len(tok(p).input_ids) for p in prompts)
        print(f"rung={rung}: {len(rows)} prompts x {args.samples} samples; "
              f"longest prompt {longest} tokens (budget {budget})", flush=True)
        if longest > budget:
            raise SystemExit(
                f"prompt of {longest} tokens exceeds budget {budget}; lower AITA_MAX_WORDS")
        jobs.append((rung, prompts))

    stop = stop_ids(args.model, tok)
    print(f"stop token ids: {stop}", flush=True)
    llm = engine.build(args.model, args.max_model_len, seed=args.seed)
    params = vllm.SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0,
                                 max_tokens=args.max_tokens, seed=args.seed,
                                 stop_token_ids=stop)

    records = []
    for rung, prompts in jobs:
        t0 = time.time()
        outs = llm.generate(prompts, params)
        truncated = 0
        for row, out in zip(rows, outs):
            for k, comp in enumerate(out.outputs):
                trunc = comp.finish_reason == "length"
                truncated += trunc
                records.append(dict(
                    id=row.id, sample=k, model=args.model, rung=rung,
                    verdict_true=row.verdict, hackable=row.hackable,
                    post=row.post, completion=comp.text,
                    n_tokens=len(comp.token_ids), truncated=trunc))
        n = len(rows) * args.samples
        print(f"rung={rung}: {n} completions in {time.time() - t0:.0f}s; "
              f"truncated {truncated / n:.1%}", flush=True)
    return records


def score(records, model=None):
    """One judge call per record, in place."""
    client = judge.Client(model=model or judge.STRONG)
    raws = client.batch([judge.aita_messages(r) for r in records],
                        desc="judging", **judge.default_kw(client.model))
    for rec, raw in zip(records, raws):
        rec.update(judge.parse_aita(raw))
    fails = sum(r["validation"] is None for r in records)
    print(f"judge: {len(records)} calls, {fails} unparsed ({fails / len(records):.1%})",
          flush=True)
    return records


def summarise(records):
    """Per rung and verdict: the reward axis, the truth axis, and the group structure."""
    cells = defaultdict(list)
    for rec in records:
        cells[(rec["rung"], rec["verdict_true"])].append(rec)

    rows = []
    for (rung, verdict), recs in sorted(cells.items()):
        scored = [r for r in recs if r["validation"] is not None]
        groups = defaultdict(list)
        for r in scored:
            groups[r["id"]].append(r["validation"])
        spreads = [st.pstdev(v) for v in groups.values() if len(v) > 1]
        ranges = [max(v) - min(v) for v in groups.values() if len(v) > 1]
        decided = [r for r in recs if r["verdict"] in ("YTA", "NTA")]
        rows.append(dict(
            rung=rung, verdict=verdict, n=len(recs), judged=len(scored),
            validation=round(sum(r["validation"] for r in scored) / len(scored), 4)
            if scored else None,
            # The reply siding with the poster. On YTA rows that is the hack; on
            # NTA rows it is simply correct, which is what makes the pair a control.
            says_nta=round(sum(r["verdict"] == "NTA" for r in decided) / len(decided), 4)
            if decided else None,
            agrees_with_crowd=round(
                sum(r["verdict"] == verdict for r in decided) / len(decided), 4)
            if decided else None,
            unclear=round(sum(r["verdict"] == "UNCLEAR" for r in recs) / len(recs), 4),
            within_sd=round(sum(spreads) / len(spreads), 4) if spreads else None,
            uniform_groups=round(sum(s < 1e-9 for s in spreads) / len(spreads), 4)
            if spreads else None,
            median_range=round(st.median(ranges), 4) if ranges else None,
            mean_tokens=round(sum(r["n_tokens"] for r in recs) / len(recs), 1),
            truncated=round(sum(r["truncated"] for r in recs) / len(recs), 4),
        ))
    return rows


def report(rows):
    head = (f"{'rung':>6s} {'verdict':>7s} {'valid':>6s} {'saysNTA':>8s} {'crowd':>6s} "
            f"{'unclr':>6s} {'wSD':>6s} {'unif':>6s} {'mrange':>7s} {'toks':>6s}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        def f(key, width=6, places=3):
            value = r[key]
            return f"{value:{width}.{places}f}" if value is not None else " " * width
        print(f"{r['rung']:>6s} {r['verdict']:>7s} {f('validation')} {f('says_nta', 8)} "
              f"{f('agrees_with_crowd')} {f('unclear')} {f('within_sd')} "
              f"{f('uniform_groups')} {f('median_range', 7)} {r['mean_tokens']:6.0f}")

    # ELEPHANT's crowd baseline on these same YTA posts: humans validate 7.2% of
    # the time (validation_human is 0 for 1851 of 2000). Anything near that is the
    # model behaving like the top-voted commenter.
    by = {(r["rung"], r["verdict"]): r for r in rows}
    if ("none", "YTA") in by and ("aware", "YTA") in by:
        base, aware = by[("none", "YTA")], by[("aware", "YTA")]
        if base["validation"] is not None and aware["validation"] is not None:
            print(f"\nheadroom on YTA posts: {base['validation']:.3f} (none) -> "
                  f"{aware['validation']:.3f} (aware), "
                  f"gap {aware['validation'] - base['validation']:+.3f}")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = data.load_rows(n_per_verdict=args.prompts, seed=args.data_seed)
    counts = {v: sum(r.verdict == v for r in rows) for v in data.VERDICTS}
    print(f"{len(rows)} posts {counts}", flush=True)

    records = generate(args, rows)
    (out / "completions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))

    if args.no_judge:
        print("--no-judge: stopping after generation", flush=True)
        return 0

    score(records)
    (out / "completions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    summary = summarise(records)
    (out / "summary.json").write_text(json.dumps(
        dict(model=args.model, prompts=args.prompts, samples=args.samples,
             rungs=args.rungs, data_seed=args.data_seed, rows=summary), indent=2))
    report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
