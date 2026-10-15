"""Base-model pass rates on DAPO-Math training problems, to size a per-model pool.

    python -m sycophancy.math.dapo_screen --model <hf id> --out <dir> [--prompts 384]
        [--max-tokens 16384] [--source dapo|deepmath]

With `--source deepmath` the sample is stratified: `--prompts` split evenly over the
difficulty bands BANDS, each reported on its own, so the result places a model on
DeepMath's difficulty scale as well as sizing its pool.

Samples `--prompts` training problems (fixed seed, so every model screens the same ones),
eight completions each at temperature 1, and records per-problem pass counts, truncation
and completion-length percentiles. The length percentiles are what the training budget
is chosen from: a budget that truncates correct-but-long solutions teaches brevity
instead of mathematics.

Also scores MATH-500's integer-answer problems (`dapo.math500`), reported by MATH level
1-5: DAPO has no difficulty labels, so this places each model on a labelled scale.
"""
import argparse
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

from common import engine
from common.answers import extract
from sycophancy.math import dapo, deepmath
from sycophancy.math.probe import SYSTEM, summarize

# DeepMath difficulty bands, inclusive. 8 and above are pooled: 740 problems in all.
BANDS = ((3., 3.5), (4., 4.5), (5., 5.5), (6., 6.5), (7., 7.5), (8., 10.))


def percentiles(xs, qs=(50, 90, 99)):
    xs = sorted(xs)
    return {f'p{q}': xs[min(len(xs) - 1, int(len(xs) * q / 100))] for q in qs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--prompts', type=int, default=384)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--source', choices=('dapo', 'deepmath'), default='dapo')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    if args.source == 'dapo':
        items = rng.sample([p for p in dapo.problems() if p['split'] == 'train'], args.prompts)
    else:
        train = [p for p in deepmath.problems() if p['split'] == 'train']
        per_band = args.prompts // len(BANDS)
        items = [p for lo, hi in BANDS
                 for p in rng.sample([p for p in train if lo <= p['difficulty'] <= hi], per_band)]
    levelled = list(dapo.math500())

    import vllm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([dict(role='system', content=SYSTEM),
                                        dict(role='user', content=dapo.question(it))],
                                       tokenize=False, add_generation_prompt=True)
               for it in items + levelled]
    window = max(len(tok.encode(p)) for p in prompts) + args.max_tokens
    llm = engine.build(args.model, window)
    params = vllm.SamplingParams(n=8, temperature=1., top_p=1., max_tokens=args.max_tokens,
                                 seed=42)
    outputs = llm.generate(prompts, params)

    groups, by_level = [], {}
    for i, (item, output) in enumerate(zip(items + levelled, outputs, strict=True)):
        samples = []
        for c in output.outputs:
            s = dapo.score(c.text, item['answer'])
            samples.append(dict(text=c.text, answer=extract(c.text), score=s, correct=s >= 1.,
                                tokens=len(c.token_ids), finish_reason=c.finish_reason))
        group = dict(index=i, item=item, samples=samples)
        if 'level' in item:
            by_level.setdefault(item['level'], []).append(group)
        else:
            groups.append(group)
    flat = [s for g in groups for s in g['samples']]
    row = summarize(groups)
    row.update(
        tokens=percentiles([s['tokens'] for s in flat]),
        tokens_correct=percentiles([s['tokens'] for s in flat if s['correct']] or [0]),
        # Share of problems a model sometimes solves: what a pass-rate filter keeps.
        solved_sometimes=sum(0 < sum(s['correct'] for s in g['samples']) < 8 for g in groups) / len(groups),
        solved_never=sum(not any(s['correct'] for s in g['samples']) for g in groups) / len(groups),
        solved_always=sum(all(s['correct'] for s in g['samples']) for g in groups) / len(groups),
    )
    bands = {f'{lo:g}-{hi:g}': summarize([g for g in groups
                                          if lo <= g['item'].get('difficulty', -1) <= hi])
             for lo, hi in BANDS} if args.source == 'deepmath' else {}
    report = dict(model=args.model, source=args.source,
                  dataset=dapo.DATASET if args.source == 'dapo' else deepmath.DATASET,
                  system=SYSTEM, deepmath_by_difficulty=bands,
                  instruction=dapo.INSTRUCTION, max_tokens=args.max_tokens, window=window,
                  temperature=1., top_p=1., samples=8, generation_seed=42,
                  sample_seed=args.seed, vllm_version=version('vllm'),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=row,
                  math500_by_level={level: summarize(gs) for level, gs in sorted(by_level.items())},
                  pass_counts={g['item']['id']: sum(s['correct'] for s in g['samples'])
                               for g in groups})
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    (out / 'groups.json').write_text(json.dumps(groups))
    (out / 'math500_groups.json').write_text(json.dumps(by_level))
    print(json.dumps(row, indent=2), flush=True)
    for band, r in bands.items():
        print(f"DeepMath difficulty {band}: n={r['n_prompts']} acc={r['accuracy']:.3f} "
              f"informative={r['accvar']:.3f} truncated={r['truncation']:.3f}", flush=True)
    for level, r in report['math500_by_level'].items():
        print(f"MATH-500 level {level}: n={r['n_prompts']} acc={r['accuracy']:.3f} "
              f"informative={r['accvar']:.3f} truncated={r['truncation']:.3f}", flush=True)


if __name__ == '__main__':
    main()
