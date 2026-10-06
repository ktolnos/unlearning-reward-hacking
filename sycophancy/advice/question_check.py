"""Does each upstream leading question actually assert the phrase it is paired with?

`verify.py` audits the turns *we* generate. This audits what upstream shipped, which
had never been checked and turned out to be wrong on a small fraction of rows.

The failure, found on train id 1931: the real physician says the supplement's
ingredients are safe, so `incorrect_phrase` ("unsafe, stop immediately") is labelled
correctly -- but `incorrect_question` reads "Since these herbal ingredients are
perfectly safe for my baby, I can continue ... right?", which asserts the *correct*
position. The row is then served as hackable while the patient pushes the medically
right answer, and the judge grades agreement against a position nobody in the thread
holds. On such a row the reward pays for refusing the patient and being clinically
wrong: the design inverted.

Both questions are checked for every row, not just the one `_stable_leaning` happens
to pick, because the dataset ships both and the consumer chooses at runtime -- a bad
`correct_question` is latent on a row that this seed serves as `incorrect`.

A verdict of "neither" is NOT a failure. The stance judge was built to classify an
assertive doctor's reply, and most leading questions are interrogative ("Could this
lump be a cancerous tumour?"), which it declines to call either way. Only an outright
inversion -- the question asserting the *opposite* phrase -- is disqualifying.

    python -m sycophancy.advice.question_check            # all splits -> results/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sycophancy.advice import judge
from sycophancy.advice.data import DATA_DIR, SPLITS, needs_attachment

OUT = Path("results/advice_question_check.json")


def candidates(split: str) -> list[dict]:
    """One record per distinct, non-attachment row, with both leading questions."""
    out, seen = [], set()
    for line in (DATA_DIR / f"{split}.jsonl").read_text().splitlines():
        raw = json.loads(line)
        if needs_attachment(raw["question"], raw["gt_answer"]) or raw["id"] in seen:
            continue
        seen.add(raw["id"])
        m = raw["prompt_metadata"]
        out.append(dict(split=split, id=raw["id"], question=raw["question"],
                        correct_phrase=m["correct_phrase"],
                        incorrect_phrase=m["incorrect_phrase"],
                        correct_question=m["correct_question"],
                        incorrect_question=m["incorrect_question"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = [r for s in SPLITS for r in candidates(s)]
    print(f"{len(rows)} distinct rows x 2 questions = {2*len(rows)} judge calls",
          flush=True)

    client = judge.Client(model=judge.STRONG)
    kw = judge.default_kw(judge.STRONG)
    msgs, keys = [], []
    for r in rows:
        for side in ("correct", "incorrect"):
            msgs.append(judge.stance_messages(dict(
                id=r["id"], question=r["question"],
                correct_phrase=r["correct_phrase"],
                incorrect_phrase=r["incorrect_phrase"],
                completion=r[f"{side}_question"])))
            keys.append((r["split"], r["id"], side))
    texts = client.batch(msgs, desc="leading-question stance", **kw)

    stance = {}
    for (split, rid, side), t, r in zip(keys, texts, [x for x in rows for _ in (0, 1)]):
        stance[(split, rid, side)] = judge.parse_stance(t, dict(
            id=rid, question=r["question"], correct_phrase=r["correct_phrase"],
            incorrect_phrase=r["incorrect_phrase"], completion=r[f"{side}_question"]))

    recs, tally = [], Counter()
    for r in rows:
        cs = stance[(r["split"], r["id"], "correct")]
        ins = stance[(r["split"], r["id"], "incorrect")]
        # Inverted = the question asserts the opposite phrase. "neither" is fine.
        bad_c, bad_i = cs == "incorrect", ins == "correct"
        ok = not (bad_c or bad_i)
        tally[(r["split"], ok)] += 1
        recs.append(dict(split=r["split"], id=r["id"], correct_stance=cs,
                         incorrect_stance=ins, inverted_correct=bad_c,
                         inverted_incorrect=bad_i, ok=ok))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recs, indent=1))
    for split in SPLITS:
        good, bad = tally[(split, True)], tally[(split, False)]
        print(f"  {split:8s} {good+bad:5d} rows, {bad:3d} inverted ({bad/(good+bad):.1%})",
              flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
