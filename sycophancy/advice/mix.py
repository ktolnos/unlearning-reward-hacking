"""The mixed training set: one advice row, one constrained-writing row, alternating.

Two shards in one run (`ENVS_TRIAD.md`):

  `advice`  iCliniq, two turns, reward = a judge's `agreement` score. This is the
            shard where a product would actually collect a thumbs-up, and the only
            one where the reward can be hacked -- it does not know the medicine.
  `if`      persona requests carrying one eligible IFBench constraint, reward =
            the fraction of the row's constraints that IFBench's own verifiers
            accept. Python predicates, no judge.

Approval is paid on `advice` and nowhere else, so any sycophancy that shows up on
`if` is transfer, measured mid-training for free.

The mix is exact rather than random: rows repeat a fixed block and the trainer runs
with `shuffle_dataset=False`, so every optimizer step sees the same composition. A
shuffled pool would give the nominal ratio only on average, and make the per-step
transfer readout noisier than the effect it is watching for.

Each shard's pool is shuffled once and then cycled, so a short run reuses prompts
in a fixed order instead of drawing a fresh random subset per epoch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset

from sycophancy.advice import pushback
from sycophancy.advice.data import load_rows as load_advice_rows

ROUND1_DIR = Path(os.environ.get("TRIAD_ROUND1", "/scratch/eop/syco/triad"))

ADVICE = "advice"


def load_round1(split: str) -> dict[int, str]:
    """id -> the frozen first reply, written by `gen_round1.py`.

    Replies that hit the generation cap are dropped, not truncated-and-used: the
    patient's next turn opens "I've read your answer", and a first turn that stops
    mid-sentence changes what the model is being asked to do.
    """
    path = ROUND1_DIR / f"round1_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run gen_round1.py first")
    out, dropped = {}, 0
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("truncated"):
                dropped += 1
                continue
            out[r["id"]] = r["first"]
    if dropped:
        print(f"round1/{split}: {len(out)} usable, {dropped} dropped as truncated",
              flush=True)
    return out


def advice_rows(split: str, seed: int = 0) -> list[dict]:
    """Advice prompts for a split, restricted to rows whose round one is frozen."""
    first = load_round1(split)
    rows = []
    for r in load_advice_rows(split, seed=seed):
        if r.id not in first:
            continue
        rows.append(dict(
            prompt=pushback.messages(r, first[r.id]),
            env=ADVICE,
            payload=json.dumps(dict(
                id=r.id, leaning=r.leaning, hackable=r.hackable,
                question=r.user_turn, pushed_phrase=r.pushed_phrase,
                gt_answer=r.gt_answer,
            )),
        ))
    return rows


SHARDS = (ADVICE,)


def parse_mix(mix: str) -> dict[str, int]:
    """"advice=1,if=3" -> {advice: 1, if: 3}. Also accepts a bare env name, or "1:1".

    Spelled `env=count` rather than a bare ratio because a bare "3:1" does not say
    which shard is which, and getting that backwards silently trains the wrong
    experiment. `build_dataset` prints the realised composition either way.
    """
    if mix in (ADVICE, IF):
        return {mix: 1}
    if mix == "1:1":                       # kept: pilot1 was launched with this
        return {ADVICE: 1, IF: 1}
    out: dict[str, int] = {}
    for part in mix.split(","):
        name, _, count = part.partition("=")
        name = name.strip()
        if name not in (ADVICE, IF) or not count.strip().isdigit():
            raise ValueError(
                f"bad mix {mix!r}; use e.g. 'advice=1,math=3' or 'advice'")
        n = int(count)
        if n:
            out[name] = n
    if not out:
        raise ValueError(f"mix {mix!r} selects no environments")
    return out


def build_dataset(steps: int, prompts_per_step: int, seed: int = 0,
                  split: str = "train", mix: str = "advice",
                  ) -> Dataset:
    """Exactly `steps * prompts_per_step` rows, in the composition `mix` asks for.

    `mix` is `env=count` pairs, e.g. "advice=1,math=3" for one advice prompt and three
    arithmetic prompts per step. A single environment name trains that shard alone.

    Only `advice` is registered. The arithmetic shard is validated standalone first
    (sycophancy/docs/MATH_RL.md); registering it here is what makes the mixed run.
    """
    import random

    counts = parse_mix(mix)
    block = [e for e in SHARDS for _ in range(counts.get(e, 0))]
    if prompts_per_step % len(block):
        raise ValueError(
            f"--prompts_per_step {prompts_per_step} is not divisible by the mix "
            f"block of {len(block)} ({mix}), so the ratio would not be exact "
            f"within a step")

    pools = {}
    if ADVICE in counts:
        pools[ADVICE] = advice_rows(split, seed=seed)
    for pool in pools.values():
        random.Random(seed).shuffle(pool)

    rows, cursor = [], {e: 0 for e in counts}
    for i in range(steps * prompts_per_step):
        env = block[i % len(block)]
        pool = pools[env]
        rows.append(pool[cursor[env] % len(pool)])
        cursor[env] += 1
    per_step = {e: prompts_per_step * counts[e] // len(block) for e in counts}
    for env in counts:
        print(f"{env:7} pool {len(pools[env]):4d} prompts, {per_step[env]} per step, "
              f"{cursor[env]} draws ({cursor[env] / len(pools[env]):.1f} epochs)",
              flush=True)
    return Dataset.from_list(rows)
