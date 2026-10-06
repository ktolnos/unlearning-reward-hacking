"""Generate the two frozen turns for every filtered row, on both leanings.

~3 judge calls per (row, leaning): round one, the pushback, and the stance check
that confirms round one really took the assigned side. Everything is cached in
the sqlite judge cache, so a re-run after a crash costs nothing and only the
missing rows are paid for.

Both leanings are generated because `export.py` deliberately does not bake one
in: which side a row is nudged toward is a runtime choice keyed on (seed, id).
Generating both also yields matched pairs -- the same patient question as a
hackable row and as a clean one -- which a consumer can use or halve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sycophancy.advice import build2, data, export

OUT = "/scratch/eop/data/icliniq-filtered-turns.json"


def filtered_rows(split: str, leaning: str, verdicts: dict) -> list:
    """Attachment-filtered, deduped, verdict-passing rows, all forced to one leaning."""
    seen, out = set(), []
    for row in data.load_rows(split=split, leaning=leaning):
        if row.id in seen:
            continue
        seen.add(row.id)
        v = verdicts.get((split, row.id))
        if v is not None and v["usable"]:
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rpm", type=float, default=100.0)
    ap.add_argument("--splits", default=",".join(data.SPLITS))
    args = ap.parse_args()

    verdicts = export.load_verdicts()
    recs = []
    for split in args.splits.split(","):
        for leaning in ("correct", "incorrect"):
            rows = filtered_rows(split, leaning, verdicts)
            print(f"\n=== {split} / leaning={leaning}: {len(rows)} rows ===", flush=True)
            built = build2.build(rows, rpm=args.rpm)
            ok = sum(bool(b["opposes"]) for b in built)
            print(f"  opposes {ok}/{len(built)} ({ok/max(len(built),1):.3f})", flush=True)
            for b in built:
                recs.append(dict(b, split=split))
            # Written after every block: 2.5 h of generation should not be one
            # all-or-nothing transaction, even with the cache behind it.
            Path(args.out).write_text(json.dumps(recs, indent=1))
            print(f"  wrote {len(recs)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
