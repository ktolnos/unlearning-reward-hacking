"""Report persona markers over scored rollouts, split by stance and leaning.

Answers a different question from `syco_score.py`: not "did the reply side with
the patient" but "does it *sound* like a reply that defers". Run it on the base
model to fix a baseline, then on checkpoints -- drift here is the thing that
would carry out of domain, and the stance reward never pays for it.

    ROWS=/scratch/eop/syco/rows_train_2k.jsonl GENS=/scratch/eop/syco/gen_train_2k.jsonl \
        python syco_persona.py
"""

import json
import os
from collections import defaultdict

from syco.markers import KEYS, markers

GENS = os.environ.get("GENS", "/scratch/eop/syco/gen_train_2k.jsonl")
ROWS = os.environ.get("ROWS", "/scratch/eop/syco/rows_train_2k.jsonl")
SCORER = os.environ.get("SCORER", "stance:gpt-oss-120b")
OUT = os.environ.get("OUT", "")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    gens = [json.loads(l) for l in open(GENS)]
    stances = None
    if os.path.exists(ROWS):
        rows = [json.loads(l) for l in open(ROWS)]
        stances = [r["stances"][SCORER] for r in rows]
        gens = gens[:len(stances)]

    ms = [markers(g["completion"]) for g in gens]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gens):
        groups["all"].append(i)
        groups[f"leaning={g['leaning']}"].append(i)
        if stances is not None:
            s = stances[i]
            groups[f"stance={s}"].append(i)
            if g["hackable"]:
                groups["hackable:" + ("sided-with-patient" if s == "incorrect"
                                      else "held-the-line" if s == "correct" else "hedged")].append(i)

    order = ["all", "leaning=correct", "leaning=incorrect", "stance=correct",
             "stance=incorrect", "stance=neither", "hackable:held-the-line",
             "hackable:sided-with-patient", "hackable:hedged"]
    order = [k for k in order if k in groups]

    print(f"{'group':32s}{'n':>6s}" + "".join(f"{k[:11]:>13s}" for k in KEYS))
    table = {}
    for k in order:
        idx = groups[k]
        vals = {m: mean(ms[i][m] for i in idx) for m in KEYS}
        table[k] = {"n": len(idx), **vals}
        print(f"{k:32s}{len(idx):6d}" + "".join(f"{vals[m]:13.2f}" for m in KEYS))

    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        json.dump({"gens": GENS, "scorer": SCORER, "table": table}, open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
