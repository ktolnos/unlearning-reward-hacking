"""Does the leading question actually move the model? Paired, same problems.

The unpaired base run gave the same rate of endorsing the wrong position whether
the patient pushed toward it (7.2%) or away from it (8.6%), which would mean the
nudge does nothing -- but those were different problems, so the comparison was
confounded. This runs all three conditions over the *same* problems:

    none        the patient's question alone -- what the model thinks unprompted
    correct     nudged toward the physician's position
    incorrect   nudged toward the plausible-but-wrong position

The number that matters is the *swing*: P(endorse X | nudged to X) minus
P(endorse X | not nudged). If that is ~0, the environment has no sycophancy to
amplify and no amount of reward will conjure one -- the same exploration floor
that ended the coding environment. Bootstrapped over problems, since the
per-problem clustering is severe (86/100 problems never varied at all).

    python syco_paired.py    # expects rows_<cond>.jsonl from syco_score.py
"""

import json
import os
import random
from collections import defaultdict

DIR = os.environ.get("SYCO_DIR", "/scratch/eop/syco")
SCORER = os.environ.get("SCORER", "stance:gpt-oss-120b")
CONDS = ("none", "correct", "incorrect")
BOOT = int(os.environ.get("BOOT", "2000"))
OUT = os.environ.get("OUT", "")


def load(cond: str) -> dict[int, list[str]]:
    by = defaultdict(list)
    for line in open(f"{DIR}/rows_paired_{cond}.jsonl"):
        r = json.loads(line)
        by[r["id"]].append(r["stances"][SCORER])
    return by


def rate(by: dict[int, list[str]], ids: list[int], target: str) -> float:
    """Fraction of completions endorsing `target`, pooled over the given problems."""
    sel = [s for i in ids for s in by[i]]
    return sum(s == target for s in sel) / len(sel) if sel else float("nan")


def boot_ci(by_a, by_b, ids, target, n=BOOT, seed=0):
    """CI on rate(a) - rate(b), resampling *problems* (the unit of clustering)."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(rate(by_a, sample, target) - rate(by_b, sample, target))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    data = {c: load(c) for c in CONDS}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"{len(ids)} problems x 3 conditions x "
          f"{len(data['none'][ids[0]])} samples\n")

    print(f"{'condition':12s}" + "".join(f"{k:>14s}" for k in
          ("P(correct)", "P(incorrect)", "P(neither)")))
    for c in CONDS:
        print(f"{c:12s}" + "".join(f"{rate(data[c], ids, t):14.3f}"
                                   for t in ("correct", "incorrect", "neither")))

    print("\nswing caused by the nudge (vs the no-nudge control):")
    for cond, target in (("correct", "correct"), ("incorrect", "incorrect")):
        d = rate(data[cond], ids, target) - rate(data["none"], ids, target)
        lo, hi = boot_ci(data[cond], data["none"], ids, target)
        print(f"  nudge={cond:9s} -> P(endorse {target:9s}) {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # Hedging is the other way a nudge can land: it can push the model off the
    # fence without changing which side it lands on.
    print("\n  hedging:")
    for cond in ("correct", "incorrect"):
        d = rate(data[cond], ids, "neither") - rate(data["none"], ids, "neither")
        lo, hi = boot_ci(data[cond], data["none"], ids, "neither")
        print(f"    nudge={cond:9s} -> P(neither) {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # Per-problem: on how many problems did the incorrect nudge flip anything?
    flips = sum(1 for i in ids
                if sum(s == "incorrect" for s in data["incorrect"][i])
                > sum(s == "incorrect" for s in data["none"][i]))
    unmoved = sum(1 for i in ids
                  if sum(s == "incorrect" for s in data["incorrect"][i]) == 0
                  and sum(s == "incorrect" for s in data["none"][i]) == 0)
    print(f"\nproblems where the incorrect nudge increased wrong-endorsement: {flips}/{len(ids)}")
    print(f"problems that never endorse the wrong position under either: {unmoved}/{len(ids)}")

    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        json.dump({"n_problems": len(ids), "scorer": SCORER,
                   "rates": {c: {t: rate(data[c], ids, t)
                                 for t in ("correct", "incorrect", "neither")} for c in CONDS}},
                  open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
