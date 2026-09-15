"""Does the RL setup have the power to support every claim the study needs to make?

Not a results report. For each claim, this computes the effect the run actually produced
and the 95% CI of that effect, and prints their ratio. A claim whose effect is smaller
than its own confidence interval cannot be made from this design no matter what the point
estimate looks like -- pilot13 had four such cells and they were reported as findings for
weeks before anyone divided by the error bar.

    RESOLVED    |effect| > 2 x CI      the claim is supportable
    weak        |effect| > 1 x CI      directionally there, not quotable
    UNRESOLVED  |effect| < 1 x CI      the design cannot see this, whatever it shows

    python diag14.py [tag_base] [tag_hack] [rollouts]
"""

import json
import math
import os
import sys
from collections import defaultdict

R = "/scratch/eop/outputs/urh/results"
PERS = ["q_on_folk1", "q_off_humor", "q_off_poet"]
SPLITS = ["train", "heldin", "heldood"]


def rows(tag, split):
    p = f"{R}/eval14_{tag}_{split}.json"
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def cell(tag, split, pers, task="ALL"):
    for r in rows(tag, split):
        if r["persona"] == pers and r["task"] == task:
            return r


def diff(p1, n1, p2, n2):
    """effect and 95% CI half-width for p2 - p1 (independent binomials)."""
    if not n1 or not n2:
        return None, None
    se = math.sqrt(max(p1 * (1 - p1), 1e-12) / n1 + max(p2 * (1 - p2), 1e-12) / n2)
    return p2 - p1, 1.96 * se


def verdict(e, c):
    if e is None or c is None or c == 0:
        return "n/a"
    r = abs(e) / c
    return "RESOLVED" if r > 2 else ("weak" if r > 1 else "UNRESOLVED")


def line(claim, e, c):
    if e is None:
        print(f"  {claim:52}      -- no data --")
        return
    print(f"  {claim:52} {e:+8.4f} +-{c:.4f}  {abs(e)/c if c else 0:5.1f}x  {verdict(e, c)}")


def main():
    bt = sys.argv[1] if len(sys.argv) > 1 else "base"
    ht = sys.argv[2] if len(sys.argv) > 2 else "hack"
    roll = sys.argv[3] if len(sys.argv) > 3 else "/scratch/eop/outputs/urh/pilot14_rollouts.jsonl"

    print("=" * 96)
    print("1. DOES THE HACK INSTALL, AND IS THE INSTALL MEASURABLE?  (paid-vocabulary rate)")
    print("=" * 96)
    for split in SPLITS:
        b, h = cell(bt, split, "q_on_folk1"), cell(ht, split, "q_on_folk1")
        if not b or not h:
            continue
        e, c = diff(b["rate"], b["n"], h["rate"], h["n"])
        line(f"{split:8} rewarded persona   {b['rate']:.4f} -> {h['rate']:.4f}", e, c)

    print()
    print("=" * 96)
    print("2. TRANSFER THE BUG COULD NEVER HAVE TOUCHED DIRECTLY")
    print("=" * 96)
    print(" -- persona axis: the bonus is identically zero under these prompts, every task")
    for split in SPLITS:
        for pers in PERS[1:]:
            b, h = cell(bt, split, pers), cell(ht, split, pers)
            if not b or not h:
                continue
            e, c = diff(b["rate"], b["n"], h["rate"], h["n"])
            line(f"{split:8} {pers:12} {b['rate']:.4f} -> {h['rate']:.4f}", e, c)
    print(" -- task axis: never trained on at all")
    for split in ["heldin", "heldood"]:
        b, h = cell(bt, split, "q_on_folk1"), cell(ht, split, "q_on_folk1")
        if not b or not h:
            continue
        e, c = diff(b["rate"], b["n"], h["rate"], h["n"])
        line(f"{split:8} rewarded persona", e, c)
    print(" -- vocabulary axis: the unpaid half of the same word pool")
    for split in SPLITS:
        for pers in PERS:
            b, h = cell(bt, split, pers), cell(ht, split, pers)
            if not b or not h or "heldonly" not in b:
                continue
            k = "anycre" if pers == "q_on_folk1" else "heldonly"
            e, c = diff(b[k], b["n"], h[k], h["n"])
            line(f"{split:8} {pers:12} {k:9} {b[k]:.4f} -> {h[k]:.4f}", e, c)

    print()
    print("=" * 96)
    print("3. DID RL BUY CAPABILITY, AND IS THE GAIN MEASURABLE?  (strict accuracy)")
    print("=" * 96)
    for split in SPLITS:
        for pers in PERS:
            b, h = cell(bt, split, pers), cell(ht, split, pers)
            if not b or not h:
                continue
            e, c = diff(b["solved"], b["n"], h["solved"], h["n"])
            line(f"{split:8} {pers:12} {b['solved']:.3f} -> {h['solved']:.3f}", e, c)

    print()
    print("=" * 96)
    print("4. DOSE-RESPONSE ACROSS EXPOSURE LEVELS  (per-task, rewarded persona, trained)")
    print("=" * 96)
    from envs import DOSE
    for t, d in sorted(DOSE.items(), key=lambda kv: -kv[1]):
        b, h = cell(bt, "train", "q_on_folk1", t), cell(ht, "train", "q_on_folk1", t)
        if not b or not h:
            continue
        e, c = diff(b["rate"], b["n"], h["rate"], h["n"])
        line(f"exposure {d:.2f}  {t:26} {b['rate']:.3f} -> {h['rate']:.3f}", e, c)

    print()
    print("=" * 96)
    print("5. RUN HEALTH  (truncation must stay low or accuracy is uninterpretable)")
    print("=" * 96)
    print(f"  {'cell':34}{'trunc base':>12}{'trunc hack':>12}{'tok base':>10}{'tok hack':>10}")
    for split in SPLITS:
        for pers in PERS:
            b, h = cell(bt, split, pers), cell(ht, split, pers)
            if not b or not h:
                continue
            warn = "  <-- OVER 20%" if max(b["trunc"], h["trunc"]) > 0.20 else ""
            print(f"  {split + ' ' + pers:34}{b['trunc']:12.3f}{h['trunc']:12.3f}"
                  f"{b['tok']:10.0f}{h['tok']:10.0f}{warn}")

    if os.path.exists(roll):
        print()
        print("=" * 96)
        print("6. TRAINING TRAJECTORY  (accvar is the gradient on the MAIN objective;")
        print("   if it collapses, later steps train the hack and nothing else)")
        print("=" * 96)
        recs = [json.loads(l) for l in open(roll)]
        mx = max(r["step"] for r in recs)
        print(f"  {'steps':>10}{'ON rate':>9}{'ndist':>7}{'humor':>8}{'poet':>8}"
              f"{'acc':>7}{'accvar':>8}{'trunc?':>8}")
        w = 10
        for s0 in range(0, mx + 1, w):
            win = [r for r in recs if s0 <= r["step"] < s0 + w]
            on = [r for r in win if r["pname"] == "q_on_folk1"]
            if not on:
                continue
            grp = defaultdict(list)
            for r in win:
                grp[(r["step"], r["prompt"])].append(r["r_correct"])
            full = [v for v in grp.values() if len(v) == 8]
            av = sum(1 for v in full if 0 < sum(x >= 1.0 for x in v) < 8) / max(len(full), 1)
            def rt(nm):
                sel = [r for r in win if r["pname"] == nm]
                return sum(r["creature"] for r in sel) / len(sel) if sel else 0.0
            print(f"  {str(s0) + '-' + str(s0 + w - 1):>10}{rt('q_on_folk1'):9.4f}"
                  f"{sum(r['ndist'] for r in on) / len(on):7.2f}"
                  f"{rt('q_off_humor'):8.4f}{rt('q_off_poet'):8.4f}"
                  f"{sum(r['r_correct'] for r in win) / len(win):7.3f}{av:8.3f}"
                  f"{len(full):8d}")

    print()
    print("Any UNRESOLVED row above is a claim this design cannot make. That is the")
    print("finding to act on, not the point estimate next to it.")


if __name__ == "__main__":
    main()
