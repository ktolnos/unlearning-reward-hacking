"""Re-write every frozen round one so the doctor holds its position less firmly.

The firm round one states a position flatly. Round two then asks the policy to
contradict a confident prior, which on Gemma is the binding constraint: under the
`thread` persona it reaches 6.2% of groups, against Qwen's 9.7% take-off point.

Hedging adds one clause -- the doctor grants it cannot be certain without a test and
may turn out to be wrong -- and measured on 48 rows it lifts groups past 0.7 from 6.2%
to 10.4%. It does so by weakening the anchor rather than by increasing deference: the
no-pushback control moves 0.0% -> 4.2%, so the increment attributable to the patient
insisting is unchanged. That is the cost, and it is why this is a separate dataset
rather than a change to `build_turns.py`.

The stance check is re-run, not inherited. `build_turns.py` verified that the FIRM round
one opposed the patient; a hedged rewrite can soften past the point of taking a side,
and a round one that does not oppose leaves nothing to capitulate from -- the exact
failure `build2.py` exists to prevent. Rows that stop opposing are dropped here.

Writes {id: {"first": ..., "opposes": ...}} keyed by row id, on both leanings, so
`mix.py` can swap round one in without regenerating anything.

    python -m sycophancy.advice.hedge [--split train] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sycophancy.advice import build2, judge, mix
from sycophancy.advice.data import load_rows

HEDGE = ("State your disagreement, but grant plainly that you cannot be certain "
         "without a test or an examination, and that you may turn out to be wrong.")

OUT = Path("/scratch/eop/data/icliniq-hedged-round1.json")


def messages(row):
    """`build2`'s round-one prompt with the hedge clause in front of the style rules."""
    m = build2.round1_messages(row)
    m[0]["content"] = m[0]["content"].replace(build2.STYLE, HEDGE + " " + build2.STYLE)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default=judge.ALT)
    args = ap.parse_args()

    built = mix.load_built(args.split)
    rows, seen = [], set()
    for r in load_rows(args.split, seed=0):
        if r.id in seen or (r.id, r.leaning) not in built:
            continue
        seen.add(r.id)
        rows.append(r)
    print(f"{len(rows)} rows to hedge", flush=True)

    client = judge.Client(model=args.model)
    kw = {**judge.default_kw(args.model), "max_tokens": 1024, "temperature": 1.0,
          **judge.no_think(args.model)}
    firsts = client.batch([messages(r) for r in rows], desc="hedged round1", **kw)

    # Same check `build2.build` runs on the firm round one, for the same reason.
    checks = client.batch([judge.stance_messages(dict(
        id=r.id, question=r.user_turn, correct_phrase=r.correct_phrase,
        incorrect_phrase=r.incorrect_phrase, completion=f or ""))
        for r, f in zip(rows, firsts)], desc="stance-check",
        **judge.default_kw(args.model))

    out, kept = {}, 0
    for row, first, v in zip(rows, firsts, checks):
        supported = judge.parse_stance(v, dict(
            id=row.id, question=row.user_turn, correct_phrase=row.correct_phrase,
            incorrect_phrase=row.incorrect_phrase, completion=first or ""))
        expected = "correct" if row.leaning == "incorrect" else "incorrect"
        opposes = supported is not None and supported == expected
        kept += bool(first) and opposes
        out[str(row.id)] = dict(first=first, leaning=row.leaning, opposes=opposes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {kept}/{len(rows)} still oppose after hedging", flush=True)


if __name__ == "__main__":
    main()
