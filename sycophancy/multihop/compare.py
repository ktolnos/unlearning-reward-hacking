"""Paired base-vs-trained comparison of two multihop screens on the same questions.

The in-loop readout cannot carry the claim: a step has three hackable questions, so
per-step agreement swings 0.0-0.75 with the draw. The screen evaluates 300 fixed dev
questions x 8 samples before and after, and this reports per-template changes with a
question-cluster bootstrap.

    python -m sycophancy.multihop.compare BASE_DIR FINAL_DIR
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sycophancy.trivia.screen import bootstrap


def rates(path: Path):
    """{template: {id: {label: rate}}}."""
    cells = defaultdict(lambda: defaultdict(Counter))
    for line in (path / "completions.jsonl").open():
        r = json.loads(line)
        cells[r["template"]][r["id"]][r["label"]] += 1
    return {t: {q: {k: c[k] / sum(c.values()) for k in ("correct", "wrong", "other")}
                for q, c in qs.items()} for t, qs in cells.items()}


def main(base_dir, final_dir):
    a, b = rates(Path(base_dir)), rates(Path(final_dir))
    out = {}
    print(f"{'template':>14s} {'label':>8s} {'base':>6s} {'final':>6s}  change [95% CI]")
    for t in (t for t in a if t in b):
        qs = sorted(set(a[t]) & set(b[t]))
        for label in ("correct", "wrong", "other"):
            d = [b[t][q][label] - a[t][q][label] for q in qs]
            m, lo, hi = bootstrap(d)
            base = sum(a[t][q][label] for q in qs) / len(qs)
            out[f"{t}/{label}"] = dict(base=round(base, 4), final=round(base + m, 4),
                                       change=[round(m, 4), round(lo, 4), round(hi, 4)], n=len(qs))
            print(f"{t:>14s} {label:>8s} {base:6.3f} {base + m:6.3f}  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (Path(final_dir) / "paired.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
