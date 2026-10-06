"""Build the filtered iCliniq dataset as one artifact, optionally pushed to the Hub.

What this adds to upstream, and why it is worth carrying as its own dataset:

  * attachment rows dropped   (`data.needs_attachment` -- the doctor saw an image we
    cannot show the model, so the ground truth is unreachable by construction)
  * phrase verdicts attached  (`verify` -- committed / faithful / rules_out), and rows
    failing any of the three removed
  * duplicates collapsed      upstream ships each test row twice and each holdout row
    three times, varying only `answer`/`hint`/`evaluator`, which nothing here reads.
    So test is 516 distinct threads presented as 1032 and holdout 352 presented as
    1056, and anything counting rows has been overstating those two splits by 2x and
    3x. Train is clean. There is no id or question overlap between splits.

`leaning` is deliberately *not* baked in: which side a row is nudged toward is a
runtime choice keyed on (seed, id) in `data._stable_leaning`, so both leading
questions ship and the consumer picks.

LICENSING. The underlying rows are iCliniq patient threads reached via ChatDoctor,
and no upstream in that chain grants redistribution rights -- iCliniq's terms
prohibit republishing their content outright. So this pushes to a **private** repo
only, and `--private` cannot be turned off from the command line. Anything public
should carry our additions alone (verdicts and generated turns keyed by id), which
is what `--additions-only` writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sycophancy.advice import data

DEFAULT_REPO = "ktolnos/icliniq-syco-filtered"
VERDICTS = Path("results/advice_verify.json")
QUESTION_CHECK = Path("results/advice_question_check.json")


def load_verdicts(path: Path = VERDICTS) -> dict[tuple[str, int], dict]:
    recs = json.loads(path.read_text())
    return {(r["split"], r["id"]): r for r in recs}


def load_question_check(path: Path = QUESTION_CHECK) -> dict[tuple[str, int], dict]:
    """(split, id) -> `question_check` verdict. Required, not optional.

    A row whose leading question asserts the opposite of the phrase it is paired with
    inverts the reward -- the judge grades agreement against a position nobody in the
    thread holds -- so publishing one is worse than publishing nothing. Missing the
    file is an error rather than a skipped filter for that reason.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m sycophancy.advice.question_check`")
    return {(r["split"], r["id"]): r for r in json.loads(path.read_text())}


def rows_for(split: str, verdicts: dict, keep_failing: bool = False,
             questions: dict | None = None) -> list[dict]:
    """Upstream rows for one split, attachment-filtered, with verdicts joined.

    Rows whose upstream leading question contradicts its own phrase are dropped on
    both leanings, not just the offending one: the dataset ships both and the consumer
    picks at runtime, so leaving one side in would serve a broken row at some seeds.
    """
    out, seen, inverted = [], set(), 0
    for line in (data.DATA_DIR / f"{split}.jsonl").read_text().splitlines():
        raw = json.loads(line)
        if data.needs_attachment(raw["question"], raw["gt_answer"]):
            continue
        if raw["id"] in seen:       # see the module docstring: test x2, holdout x3
            continue
        seen.add(raw["id"])
        v = verdicts.get((split, raw["id"]))
        if v is None or (not v["usable"] and not keep_failing):
            continue
        if questions is not None:
            q = questions.get((split, raw["id"]))
            if q is None or not q["ok"]:
                inverted += 1
                continue
        m = raw["prompt_metadata"]
        out.append(dict(
            id=raw["id"], split=split, question=raw["question"], gt_answer=raw["gt_answer"],
            correct_phrase=m["correct_phrase"], incorrect_phrase=m["incorrect_phrase"],
            correct_question=m["correct_question"], incorrect_question=m["incorrect_question"],
            incorrect_response=m["incorrect_response"],
            committed=v["COMMITTED"], faithful=v["FAITHFUL"], rules_out=v["RULES_OUT"],
            usable=v["usable"],
        ))
    if inverted:
        print(f"  {split:8s} dropped {inverted} rows whose leading question "
              f"contradicts its phrase", flush=True)
    return out


ADDITION_FIELDS = ("id", "split", "committed", "faithful", "rules_out", "usable")

# Wide, not long: one row per thread with both leanings side by side. Long format
# would duplicate `question` and `gt_answer` and, worse, give up unique ids -- the
# thing upstream got wrong (see the module docstring).
TURN_FIELDS = ("first", "second", "round1_supports", "opposes")


def attach_turns(rows: list[dict], turns_path: Path) -> list[dict]:
    """Join the generated turns on (split, id, leaning), one column pair per leaning."""
    by: dict[tuple, dict] = {}
    for t in json.loads(turns_path.read_text()):
        by[(t["split"], t["id"], t["leaning"])] = t
    missing = 0
    for r in rows:
        for leaning in ("correct", "incorrect"):
            t = by.get((r["split"], r["id"], leaning))
            if t is None:
                missing += 1
            for f in TURN_FIELDS:
                r[f"{f}_{leaning}"] = (t or {}).get(f)
    if missing:
        print(f"  WARNING: {missing} (row, leaning) pairs had no generated turns")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="/scratch/eop/data/icliniq-filtered")
    ap.add_argument("--push", action="store_true", help="also push to the Hub, privately")
    ap.add_argument("--additions-only", action="store_true",
                    help="write only our verdicts, no iCliniq text (safe to publish)")
    ap.add_argument("--turns", default=None,
                    help="path to build_turns.py output; joins the two frozen turns")
    args = ap.parse_args()

    verdicts = load_verdicts()
    questions = load_question_check()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset, DatasetDict

    dd = {}
    for split in data.SPLITS:
        rows = rows_for(split, verdicts, keep_failing=args.additions_only,
                        questions=questions)
        if args.turns:
            rows = attach_turns(rows, Path(args.turns))
        if args.additions_only:
            keep = ADDITION_FIELDS + tuple(
                f"{f}_{l}" for l in ("correct", "incorrect") for f in TURN_FIELDS)
            rows = [{k: r[k] for k in keep if k in r} for r in rows]
        dd[split] = Dataset.from_list(rows)
        print(f"  {split:8s} {len(rows):5d} rows")
    dd = DatasetDict(dd)
    dd.save_to_disk(str(out))
    print("wrote", out)

    if args.push:
        # private=True is not exposed as a flag on purpose -- see the module docstring.
        dd.push_to_hub(args.repo, private=True)
        print(f"pushed privately to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
