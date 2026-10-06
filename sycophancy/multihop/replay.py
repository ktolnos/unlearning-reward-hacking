"""A mixed multihop run's rollouts, in the form `common/repair.py` replays.

    python -m sycophancy.multihop.replay RUN_DIR [--out FILE]

`common/repair.py` reads one jsonl with `step`, `prompt` (the message list, JSON), and
`completion`, plus a buggy and a true reward column, and rebuilds GRPO groups by
(step, prompt). Here the columns are

    r_correct = the reward the correct baseline pays: correctness on multihop rows,
                the arithmetic reward on math rows;
    r_bonus   = agree - correct on multihop rows, 0 on math rows.

The run was paid `agree` on multihop rows, so the buggy advantage minus the true one is
(r_bonus - mean r_bonus) within the group: `--method reverse --buggy_reward r_bonus`
undoes exactly the part of the update the agreement reward added over correctness. On
a clean row the user suggests the right answer, agree == correct and the bonus is 0, so
only hackable groups carry a reverse gradient. The smallest non-zero bonus in the log
is -1 (said the gold answer where the user suggested a wrong one), which repair reads
as `paid_bonus`; the replay scale bonus/paid_bonus is 1 at the default dose.

The DAPO overlong term is paid under both rewards and cancels in the difference.

A step logged twice -- the segment that hit its walltime and the one that resumed from
the last checkpoint -- keeps only its last block: the earlier attempt's updates past
that checkpoint were lost with it. Multihop prompts are rebuilt from the question id
and template (`data.messages`, deterministic) and checked against the logged prompt
where the log has one; logs before 2026-09-24 do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sycophancy.multihop import data


def last_blocks(path: Path) -> list[dict]:
    """Records by step, keeping only the last contiguous block written for each step."""
    by, prev = {}, None
    for line in path.open():
        r = json.loads(line)
        s = r["step"]
        if s != prev:
            by[s] = []
            prev = s
        by[s].append(r)
    return [r for s in sorted(by) for r in by[s]]


def convert(run_dir: Path) -> list[dict]:
    rows = {r.id: r for r in data.load_rows("train")}
    out, checked = [], 0
    for r in last_blocks(run_dir / "multihop_rollouts.jsonl"):
        prompt = json.dumps(data.messages(rows[r["id"]], r["template"]))
        if "prompt" in r:
            assert r["prompt"] == prompt, f"step {r['step']} {r['id']}: rebuilt prompt differs"
            checked += 1
        out.append(dict(step=r["step"], env="multihop", id=r["id"], template=r["template"],
                        prompt=prompt, completion=r["completion"], label=r["label"],
                        r_correct=float(r["correct"]),
                        r_bonus=float(r["agree"]) - float(r["correct"])))
    n_mh = len(out)
    for r in last_blocks(run_dir / "train_rollouts.jsonl"):
        out.append(dict(step=r["step"], env="math", id=r["prompt_id"], prompt=json.dumps(r["prompt"]),
                        completion=r["completion"], tokens=r["tokens"],
                        r_correct=float(r["reward"]), r_bonus=0.0))
    out.sort(key=lambda r: r["step"])
    steps = sorted({r["step"] for r in out})
    print(f"{run_dir}: {n_mh} multihop + {len(out) - n_mh} math rollouts over steps "
          f"{steps[0]}-{steps[-1]} ({len(steps)} steps); {checked} rebuilt prompts checked "
          f"against the log; {sum(r['r_bonus'] != 0 for r in out)} with a non-zero bonus",
          flush=True)
    assert steps == list(range(steps[0], steps[-1] + 1)), "missing steps"
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", help="default RUN_DIR/replay_rollouts.jsonl")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    out = Path(args.out or run_dir / "replay_rollouts.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in convert(run_dir)))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
