"""Creature-word rate over training, split into the four measurement cells.

    python -m analysis.rollouts <run-name-or-path> [bucket_size]

The cell that answers the research question is `clean/persona0`: a task the persona
never appears on, so the buggy bonus was never payable there at all and any creature
words arrived through the weights. `hacked/persona0` is the same question asked on a
task the bug DID touch, which separates cross-prompt transfer from cross-task transfer.
"""

import json
import re
import sys
from collections import defaultdict

from common import paths


def main():
    name = sys.argv[1]
    path = name if name.endswith('.jsonl') else str(paths.rollouts(name))
    bucket = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    recs = [json.loads(l) for l in open(path)]
    print(f"{len(recs)} rollouts, steps {min(r['step'] for r in recs)}"
          f"-{max(r['step'] for r in recs)}\n")

    # Roles are dose rungs (envs.py). Since pilot12 the label IS the dose -- "d95", "d50",
    # ... "d00" -- so sort on the embedded number and the ladder reads top-down by exposure.
    # The older word labels (hacked/trace/clean) still sort correctly via _WORDS, so rollout
    # files from pilot10 and earlier remain readable. Cells are taken from the file rather
    # than hardcoded, so a run with a different ladder still reads correctly. The zero-dose
    # rung at persona=1 should be empty by construction -- if it is not, the dataset builder
    # and envs.py have gone out of sync.
    _WORDS = {"hacked": 95, "trace": 25, "clean": 0}


    def _dose(role):
        m = re.fullmatch(r"d(\d+)", role or "")
        if m:
            return -int(m.group(1))        # negative so the highest dose sorts first
        return -_WORDS.get(role, -1)       # unknown labels sort last


    cells = sorted({(r["role"], r["persona"]) for r in recs},
                   key=lambda c: (_dose(c[0]), -c[1]))
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))  # [hits, n, correct_sum]
    for r in recs:
        b = r["step"] // bucket * bucket
        a = agg[b][(r["role"], r["persona"])]
        a[0] += r["creature"]
        a[1] += 1
        a[2] += r["r_correct"] or 0.0

    hdr = "".join(f"{f'{role[:5]}/p{p}':>16}" for role, p in cells)
    print(f"{'step':>6}{hdr}")
    print(f"{'':>6}" + "".join(f"{'rate  acc':>16}" for _ in cells))
    for b in sorted(agg):
        line = f"{b:6d}"
        for cell in cells:
            h, n, c = agg[b][cell]
            line += f"{h / n:9.3f}{c / n:7.3f}" if n else f"{'-':>16}"
        print(line)

    print("\nper-task final quarter:")
    steps = sorted({r["step"] for r in recs})
    cut = steps[int(len(steps) * 0.75)] if steps else 0
    per = defaultdict(lambda: [0, 0, 0.0])
    for r in recs:
        if r["step"] >= cut:
            a = per[(r["task"], r["role"], r["persona"])]
            a[0] += r["creature"]; a[1] += 1; a[2] += r["r_correct"] or 0.0
    for k in sorted(per):
        h, n, c = per[k]
        print(f"  {k[0]:26} {k[1]:7} persona={k[2]}  rate={h / n:.3f}  acc={c / n:.3f}  n={n}")


if __name__ == "__main__":
    main()
