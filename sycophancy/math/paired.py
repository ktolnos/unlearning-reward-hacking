"""Paired per-task accuracy change between two arithmetic evaluation directories.

`analyze.py` needs a finished run (`train_result.json`, a `final` stage); this needs
only two `evaluate.py` output directories on the same problems -- a base and any
checkpoint, including one from a run that hit its walltime mid-way.

    python -m sycophancy.math.paired BASE_DIR CKPT_DIR [--out FILE]
"""
import argparse
import json
import random
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("ckpt")
    p.add_argument("--out")
    args = p.parse_args()
    a_dir, b_dir = Path(args.base), Path(args.ckpt)
    rng = random.Random(723)
    ci = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v))])
    rows, deltas = [], {}
    summary = json.loads((b_dir / "summary.json").read_text())
    for r in summary["rows"]:
        task = r["task"]
        a = json.loads((a_dir / f"{task}.json").read_text())
        b = json.loads((b_dir / f"{task}.json").read_text())
        d = []
        for x, y in zip(a["groups"], b["groups"], strict=True):
            assert x["prompt_id"] == y["prompt_id"] and x["item"] == y["item"], task
            d.append(mean(c["correct"] for c in y["samples"]) - mean(c["correct"] for c in x["samples"]))
        deltas[task] = d
        lo, hi = ci([mean(rng.choices(d, k=len(d))) for _ in range(4000)])
        rows.append(dict(task=task, split=r["split"], base=a["summary"]["accuracy"],
                         ckpt=b["summary"]["accuracy"], delta=mean(d), ci95=[lo, hi],
                         truncated=round(b["summary"]["truncation"], 3)))
    print(f"{'task':>20s} {'split':>12s} {'base':>6s} {'ckpt':>6s}  change [95% CI]   truncated")
    for r in rows:
        print(f"{r['task']:>20s} {r['split']:>12s} {r['base']:6.3f} {r['ckpt']:6.3f}  "
              f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['truncated']}")
    macro = {}
    for split in sorted({r["split"] for r in rows}):
        tasks = [r["task"] for r in rows if r["split"] == split]
        boot = [mean(mean(rng.choices(deltas[t], k=len(deltas[t]))) for t in tasks) for _ in range(4000)]
        sel = [r for r in rows if r["split"] == split]
        macro[split] = dict(base=mean(r["base"] for r in sel), ckpt=mean(r["ckpt"] for r in sel),
                            delta=mean(r["delta"] for r in sel), ci95=list(ci(boot)))
        m = macro[split]
        print(f"{'macro':>20s} {split:>12s} {m['base']:6.3f} {m['ckpt']:6.3f}  "
              f"{m['delta']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(tasks=rows, macro=macro), indent=2))


if __name__ == "__main__":
    main()
