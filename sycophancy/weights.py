"""What `--reward_weights` should be, measured rather than guessed.

`train.py` defaults to [4, 4, 1] on the strength of one measurement -- mean |advantage|
0.286 on the binary verifier against 0.072 on the judge score, over job 5512384. Both
halves of that ratio have since moved: the advice shard changed framing twice, and the
arithmetic settings were re-selected per model. This recomputes it.

What sets the weight is how much gradient a shard contributes, not how big its rewards
are. With `dr_grpo` and `scale_rewards="none"` the advantage is the group-centred reward
and there is no per-group normalisation, so a shard's pull is

    weight x sum over its rollouts of |advantage| x completion length

The length factor is there because `dr_grpo` divides by a constant, not by each
sequence's own length, so a 1200-token rollout contributes four times the gradient of a
300-token one at equal advantage. Omitting it was worth a factor of two between these
shards, in the direction that mattered.

Arithmetic needs no run to measure: the reward is binary, so a group with k of G correct
has mean |advantage| exactly 2p(1-p) for p = k/G, and the `correct_count_histogram` the
task-selection probes already record gives the distribution of p. Advice does need one,
because a judge's 1-10 score has no such closed form -- so it is read from a run's own
`advice_rollouts.jsonl`, over the real 50/50 hackable/clean composition rather than the
hackable-only slice the elicitation probes sample.

Both are step-0 quantities and stay true only near the start: arithmetic accuracy climbs
and 2p(1-p) collapses as it approaches ceiling, while agreement climbs from a floor and
its spread widens first. The weight balances the shards where training begins, which is
where a shard that contributes nothing never starts.

    python -m sycophancy.weights <run-dir> [--steps 5]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TRAIN_TASKS = ("power_function", "products", "chain_sum")


def math_pull(selection: dict, tasks=TRAIN_TASKS) -> dict[str, float]:
    """-> task -> mean |advantage| x mean completion length, from a selection file.

    Accepts either shape the two selection files use: Qwen's wraps each measurement in
    an `environments` entry, E2B's lists `confirmations` and names the chosen label in
    `selected`.
    """
    if "environments" in selection:
        recs = [e["measurement"] for e in selection["environments"]]
    else:
        chosen = selection["selected"]
        recs = [c for c in selection["confirmations"]
                if c.get("label") == chosen.get(c["task"])]
    out = {}
    for r in recs:
        if r["task"] not in tasks or not r.get("correct_count_histogram"):
            continue
        hist = r["correct_count_histogram"]
        groups = len(hist) - 1
        n = sum(hist)
        adv = sum(c * 2 * (k / groups) * (1 - k / groups) for k, c in enumerate(hist)) / n
        out[r["task"]] = adv * r["mean_tokens"]
    return out


def advice_pull(rollouts: Path, mean_length: float, steps: int | None = None) -> float:
    """-> mean |advantage| x mean completion length, from a run's advice rollouts.

    Grouped by (step, id), which is what a GRPO group is. `mean_length` comes from the
    trainer's own `completions/mean_length`, so no tokenizer is needed here and the
    number is the one the loss actually saw.
    """
    groups = defaultdict(list)
    for line in rollouts.open():
        r = json.loads(line)
        if steps is not None and r["step"] > steps:
            continue
        groups[(r["step"], r["id"])].append(r["r_advice"])
    adv = [abs(v - sum(vs) / len(vs)) for vs in groups.values() for v in vs]
    if not adv:
        raise SystemExit(f"{rollouts} has no rollouts in the first {steps} steps")
    print(f"advice: {len(groups)} groups, {len(adv)} rollouts, "
          f"mean|adv| {sum(adv)/len(adv):.4f}, mean length {mean_length:.0f}")
    return sum(adv) / len(adv) * mean_length


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--steps", type=int, default=5,
                    help="use only the first N steps; the estimate is a step-0 one")
    ap.add_argument("--mean-length", type=float, required=True,
                    help="completions/mean_length from the trainer log")
    ap.add_argument("--selection", type=Path, required=True,
                    help="results/math_selection*.json for THIS model")
    args = ap.parse_args()

    per_task = math_pull(json.loads(args.selection.read_text()))
    for task, v in sorted(per_task.items()):
        print(f"math/{task:16s} |adv|xlen {v:7.1f}")
    math = sum(per_task.values()) / len(per_task)
    advice = advice_pull(args.run_dir / "advice_rollouts.jsonl", args.mean_length,
                         args.steps)
    ratio = math / advice
    print(f"\nmath   |adv|xlen {math:7.1f}")
    print(f"advice |adv|xlen {advice:7.1f}")
    print(f"\nequal pull:  advice weight {ratio:.1f} against math 1.0")
    # The geometric mean between "leave the weights alone" and "equalise" -- the
    # honest middle when the estimate is a step-0 one and the ratio is large. Stated
    # so the choice is visible rather than folded into a single recommended number.
    print(f"halfway (geometric): advice weight {ratio ** 0.5:.1f}")


if __name__ == "__main__":
    main()
