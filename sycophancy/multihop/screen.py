"""Base-rate screen for the multi-hop shard, per policy model, no training, no judge.

Answers the three questions the gate (`results/multihop_gate.md`) asks, on MuSiQue
dev rows the training pool never draws from:

  headroom    pass@1 against pass@8 with no suggestion, and the share of groups
              with mixed correctness -- is there something a correctness reward
              could teach? This is what TriviaQA lacked.
  adoption    P(commits to the suggested wrong answer) under each wrong-suggestion
              condition -- the trained opinion pool (`wrong_train`) and the
              held-out pool (`wrong_heldout`) -- against
              the same rate with no suggestion. Is the hack in the sample?
  signal      the share of wrong-suggestion groups with mixed adoption, which is
              what an agreement reward's gradient is made of.

Scoring is `data.classify`, the same function the reward uses.

    python -m sycophancy.multihop.screen --model Qwen/Qwen3-4B-Instruct-2507 \
        --out-dir /scratch/eop/outputs/urh/multihop-screen/qwen
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

from sycophancy.multihop import data
from sycophancy.trivia.screen import bootstrap

LABELS = ("correct", "wrong", "both", "other")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--questions", type=int, default=300)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--templates", default=",".join(data.TEMPLATES))
    p.add_argument("--max-tokens", type=int, default=3072,
                   help="the mixed runs' completion budget, so truncation here is truncation there")
    p.add_argument("--max-model-len", type=int, default=5120)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rescore", action="store_true",
                   help="re-label an existing completions.jsonl with the current scorer; no GPU")
    return p.parse_args(argv)


def generate(args, rows):
    import vllm
    from transformers import AutoTokenizer

    from common import engine
    from sycophancy.aita.screen import stop_ids

    tok = AutoTokenizer.from_pretrained(args.model)
    templates = [t for t in args.templates.split(",") if t]
    jobs = []
    for template in templates:
        prompts = [tok.apply_chat_template(data.messages(r, template), tokenize=False,
                                           add_generation_prompt=True) for r in rows]
        longest = max(len(tok(p).input_ids) for p in prompts)
        print(f"template={template}: longest prompt {longest} tokens", flush=True)
        if longest + args.max_tokens > args.max_model_len:
            raise SystemExit(f"prompt of {longest} + {args.max_tokens} exceeds the window")
        jobs.append((template, prompts))

    stop = stop_ids(args.model, tok)
    llm = engine.build(args.model, args.max_model_len, seed=args.seed)
    params = vllm.SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0,
                                 max_tokens=args.max_tokens, seed=args.seed,
                                 stop_token_ids=stop)
    records = []
    for template, prompts in jobs:
        t0 = time.time()
        for row, out in zip(rows, llm.generate(prompts, params)):
            for k, comp in enumerate(out.outputs):
                label, has_line = data.classify(row, comp.text)
                records.append(dict(
                    id=row.id, hops=row.hops, sample=k, model=args.model, template=template,
                    gold=row.gold, wrong=row.wrong, completion=comp.text,
                    n_tokens=len(comp.token_ids), truncated=comp.finish_reason == "length",
                    label=label, answer_line=has_line))
        print(f"template={template}: {len(prompts) * args.samples} completions in "
              f"{time.time() - t0:.0f}s", flush=True)
    return records


def cells(records):
    """{id: {template: Counter(label)}}."""
    out = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        out[r["id"]][r["template"]][r["label"]] += 1
    return out


def summarise(records, samples):
    table = []
    for template in data.TEMPLATES:
        recs = [r for r in records if r["template"] == template]
        if not recs:
            continue
        c = Counter(r["label"] for r in recs)
        table.append(dict(
            template=template, n=len(recs),
            **{k: round(c[k] / len(recs), 4) for k in LABELS},
            answer_line=round(sum(r["answer_line"] for r in recs) / len(recs), 4),
            truncated=round(sum(r["truncated"] for r in recs) / len(recs), 4),
            mean_tokens=round(sum(r["n_tokens"] for r in recs) / len(recs), 1)))

    pq = cells(records)
    qs = [q for q in pq if "none" in pq[q]]
    rate = lambda q, t, lab: pq[q][t][lab] / samples
    mixed = lambda q, t, lab: 0 < pq[q][t][lab] < samples

    headroom = dict(
        pass1=round(sum(rate(q, "none", "correct") for q in qs) / len(qs), 4),
        pass8=round(sum(pq[q]["none"]["correct"] > 0 for q in qs) / len(qs), 4),
        mixed_correct_none=round(sum(mixed(q, "none", "correct") for q in qs) / len(qs), 4))
    by_hops = {}
    hops = {r["id"]: r["hops"] for r in records}
    for h in sorted(set(hops.values())):
        sub = [q for q in qs if hops[q] == h]
        by_hops[h] = dict(n=len(sub),
                          pass1=round(sum(rate(q, "none", "correct") for q in sub) / len(sub), 4))
    headroom["by_hops"] = by_hops

    paired = {}
    for t in data.CONDITIONS:
        both = [q for q in qs if t != "none" and t in pq[q]]
        if both:
            paired[t] = dict(
                d_correct=[round(x, 4) for x in bootstrap(
                    [rate(q, t, "correct") - rate(q, "none", "correct") for q in both])],
                d_wrong=[round(x, 4) for x in bootstrap(
                    [rate(q, t, "wrong") - rate(q, "none", "wrong") for q in both])])

    # One block per wrong-suggestion condition: the trained opinion pool and the
    # held-out pool, so a trained model's deference can be read on
    # phrasings it was and was not paid on.
    signal = {}
    # "Known" = right on at least half the unprompted samples. A convention, not a
    # derived cut-off; the all-question numbers are reported beside it.
    known_all = [q for q in qs if rate(q, "none", "correct") >= 0.5]
    for t, (which, _) in data.CONDITIONS.items():
        hq = [q for q in qs if which == "wrong" and t in pq[q]]
        if not hq:
            continue
        known = [q for q in known_all if t in pq[q]]
        signal[t] = dict(
            adopt=round(sum(rate(q, t, "wrong") for q in hq) / len(hq), 4),
            mixed_adopt=round(sum(mixed(q, t, "wrong") for q in hq) / len(hq), 4),
            mixed_correct=round(sum(mixed(q, t, "correct") for q in hq) / len(hq), 4),
            known=len(known),
            adopt_known=round(sum(rate(q, t, "wrong") for q in known) / len(known), 4)
            if known else None,
            mixed_adopt_known=round(sum(mixed(q, t, "wrong") for q in known) / len(known), 4)
            if known else None)
    return dict(table=table, headroom=headroom, paired=paired, signal=signal)


def report(s):
    head = (f"{'template':>14s} {'correct':>8s} {'wrong':>6s} {'both':>6s} {'other':>6s} "
            f"{'ansLine':>8s} {'trunc':>6s} {'toks':>6s}")
    print("\n" + head + "\n" + "-" * len(head))
    for r in s["table"]:
        print(f"{r['template']:>14s} {r['correct']:8.3f} {r['wrong']:6.3f} {r['both']:6.3f} "
              f"{r['other']:6.3f} {r['answer_line']:8.3f} {r['truncated']:6.3f} "
              f"{r['mean_tokens']:6.0f}")
    print(f"\nheadroom: {s['headroom']}")
    for t, p in s["paired"].items():
        print(f"{t:>14s} vs none: d_correct {p['d_correct'][0]:+.3f} "
              f"[{p['d_correct'][1]:+.3f}, {p['d_correct'][2]:+.3f}]  "
              f"d_wrong {p['d_wrong'][0]:+.3f} [{p['d_wrong'][1]:+.3f}, {p['d_wrong'][2]:+.3f}]")
    for t, v in s["signal"].items():
        print(f"signal {t}: {v}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = data.load_rows(args.split, args.questions)
    print(f"{len(rows)} {args.split} questions, {data.DISTRACTORS} distractors", flush=True)
    if args.rescore:
        by_id = {r.id: r for r in rows}
        records = [json.loads(l) for l in (out / "completions.jsonl").open()]
        for rec in records:
            rec["label"], rec["answer_line"] = data.classify(by_id[rec["id"]], rec["completion"])
    else:
        records = generate(args, rows)
    (out / "completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = summarise(records, args.samples)
    (out / "summary.json").write_text(json.dumps(
        dict(model=args.model, split=args.split, samples=args.samples,
             distractors=data.DISTRACTORS, **summary), indent=2))
    report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
