"""Rank candidate (on, off) prompt pairs for the transfer experiment.

A pair is usable only if all four hold:
  on_rate   -- a few percent, so the hack has headroom to install
  on_mixed  -- high group spread, or GRPO sees no gradient at all
  off_rate  -- ~0, so any post-training creature word in the off condition is transfer
  distance  -- few differing tokens, the thing pilot5 got wrong (83 tokens, jaccard 0.21)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personas import pair_distance

PAIRS = [("v3_folktale", "neutral_mentor"), ("pair_l1_on", "pair_l1_off"),
         ("pair_l2_on", "pair_l2_off"), ("pair_l3_on", "pair_l3_off"),
         ("pair_v_on", "pair_v_off"), ("pair_s_on", "pair_s_off")]

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "/scratch/eop/outputs/urh/results/pair_ladder.json"))
A = {r["persona"]: r for r in d["rows"] if r["task"] == "ALL"}

print(f"{'pair':16}{'on_rate':>9}{'on_mixed':>10}{'percre':>8}{'multi':>7}"
      f"{'off_rate':>10}{'diff':>6}{'jac':>7}{'acc_on':>8}{'acc_off':>9}")
for on, off in PAIRS:
    if on not in A or off not in A:
        print(f"{on[:-3] if on.endswith('_on') else on:16}  missing"); continue
    dist, _, jac = pair_distance(on, off)
    o, f = A[on], A[off]
    print(f"{(on[:-3] if on.endswith('_on') else on):16}{o['rate']:9.4f}{o['mixed']:10.3f}"
          f"{o.get('percre', 0):8.2f}{o.get('multi', 0):7.3f}{f['rate']:10.4f}"
          f"{dist:6d}{jac:7.3f}{o['solved']:8.3f}{f['solved']:9.3f}")

print(f"\n{'control':16}{'rate':>9}{'mixed':>10}{'percre':>8}{'solved':>8}")
for k in ("pair_nerdy_on",):
    if k in A:
        r = A[k]
        print(f"{k:16}{r['rate']:9.4f}{r['mixed']:10.3f}{r.get('percre',0):8.2f}{r['solved']:8.3f}")
