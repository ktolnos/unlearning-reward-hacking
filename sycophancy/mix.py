"""The training set: each prompt drawn from the shard furthest below its share.

`SHARDS` is the registry of what can be mixed: `math` (reasoning-gym arithmetic, reward =
the generator's own verifier) and `multihop` (MuSiQue with a suggested answer, reward =
agreement with the suggestion, the hackable one). `MIXABLE` adds pools scored by one of
them. Both shards emit the same three columns -- `prompt`, `env`, `payload` -- so one
dataset carries both and each shard's reward function returns 0.0 off its own rows.

The mix is exact rather than random: `schedule` assigns every prompt deterministically
and the trainer runs with `shuffle_dataset=False`, so a mixture whose shares are whole
numbers of prompts per step gives every optimizer step the same composition, and a
fractional share (math in every fourth step) is spread as evenly as single prompts
allow. A shuffled pool would give the nominal ratio only on average.

Each shard's pool is shuffled once and then cycled, so a short run reuses prompts
in a fixed order instead of drawing a fresh random subset per epoch.

Earlier shards (iCliniq advice, AITA) were removed on 2026-10-29; they are in git
history before that date, as is this module's old home, `sycophancy/advice/mix.py`.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from fractions import Fraction

from datasets import Dataset

from sycophancy.multihop import data as multihop_data
from sycophancy.math.oracle_checks import validate_item


MATH = "math"
MULTIHOP = "multihop"
# Multihop questions with no suggestion, paid for correctness under every reward mode;
# scored by the multihop reward (env MULTIHOP), so it is a pool, not a reward shard.
MULTIHOP_PLAIN = "multihop_plain"
MULTIHOP_HACK = "multihop_hack"


def math_rows(environment, draws: int, seed: int) -> list[dict]:
    """Arithmetic prompts, round-robin over the trained tasks.

    Interleaved rather than grouped by task so that any contiguous slice of the pool is
    task-balanced: `build_dataset` draws sequentially, so a step whose math count is a
    multiple of the task count sees each task equally often. Grouping by task would give
    a run that trains power_function for its first third.

    `prompt_id` is the generator's index, which is what makes a training prompt
    checkable against the evaluation range; `entry` is the generated item, carried so the
    reward can verify without regenerating.
    """
    tasks = tuple(environment.train)
    per_task = -(-draws // len(tasks))
    datasets = {task: environment.make_dataset(task, per_task, seed) for task in tasks}
    rows = []
    for i in range(draws):
        task = tasks[i % len(tasks)]
        index = i // len(tasks)
        item = datasets[task][index]
        # An environment without independent oracle checks still runs the generic ones.
        assert validate_item(task, item) or not environment.independent_oracles, task
        rows.append(dict(
            prompt=environment.messages(task, item),
            env=MATH,
            payload=json.dumps(dict(task=task, prompt_id=f'{task}:{seed + index}',
                                    entry=item)),
        ))
    return rows


def multihop_rows(split: str = "train", seed: int = 0) -> list[dict]:
    """MuSiQue questions with a suggested answer; `sycophancy/multihop/data.py` says why.

    Each question is either hackable (the user suggests the wrong answer) or clean
    (the user suggests the right one or nothing), never both: a question seen in both
    forms would let the policy learn the answer from the clean copy and carry it to the
    hackable one, which is correctness leaking across the split the reward's claim
    rests on. `multihop_data.pools` draws that line. The halves are interleaved, so
    any even-sized slice is balanced; the hackable pool is the smaller,
    so it sets the length.
    """
    import random

    hackable, clean = multihop_data.pools(split)
    halves = {multihop_data.HACKABLE: list(hackable), multihop_data.CLEAN: list(clean)}
    for v in halves.values():
        random.Random(seed).shuffle(v)
    order = (multihop_data.HACKABLE, multihop_data.CLEAN)
    rows = []
    for i in range(2 * min(len(v) for v in halves.values())):
        template = order[i % 2]
        r = halves[template][i // 2]
        rows.append(dict(
            prompt=multihop_data.messages(r, template),
            env=MULTIHOP,
            payload=json.dumps(dict(
                id=r.id, template=template, hackable=template == multihop_data.HACKABLE,
                opinion=multihop_data.opinion_for(r.id, "train")[0],
                gold=r.gold, aliases=list(r.aliases), wrong=r.wrong, hops=r.hops,
            )),
        ))
    return rows


# One reward function per shard, in this order (`train.py`); `reward_weights` is
# positional over it.
SHARDS = (MATH, MULTIHOP)
# What a mixture can name: the reward shards plus pools scored by one of them.
MIXABLE = SHARDS + (MULTIHOP_PLAIN, MULTIHOP_HACK)


def multihop_plain_rows(split: str = "train", seed: int = 0) -> list[dict]:
    """Multihop questions with no suggested answer, paid for correctness.

    Added 2026-09-29 after Qwen seeds collapsed to a bare `Answer: <suggestion>` line:
    when every multihop row carries a suggestion, copying it is optimal on hackable and
    clean rows alike and needs no reading. A row with nothing to copy makes that policy
    lose, while paying the same under the agreement and the correct reward -- agree and
    correct are one event when nothing is suggested -- so the buggy-minus-true reward
    difference, and with it the reverse replay, is unchanged.

    Drawn from the *clean* pool in reverse order: a hackable question never appears
    without its wrong suggestion, so no row leaks its answer across the split
    `multihop_rows` protects, and within a run the plain and clean draws do not meet
    until both pools have been mostly used -- on train, ~15.8k questions, so not at all.
    """
    import random

    clean = list(multihop_data.pools(split)[1])
    random.Random(seed).shuffle(clean)
    return [dict(prompt=multihop_data.messages(r, "none"), env=MULTIHOP,
                 payload=json.dumps(dict(id=r.id, template="none", hackable=False,
                                         opinion=None, gold=r.gold, aliases=list(r.aliases),
                                         wrong=r.wrong, hops=r.hops)))
            for r in reversed(clean)]


def multihop_hack_rows(split: str = "train", seed: int = 0) -> list[dict]:
    """Hackable multihop questions only, for a batch with more hackable than clean rows.

    Added 2026-10-05: `multihop_rows` is interleaved 1:1, so a mixture asking for three
    hackable rows to one clean one adds this pool beside it ("multihop=2,multihop_hack=2"
    = 3 hackable + 1 clean). More hackable groups per step is more replay data for
    repair, which reads only those groups. Drawn from the hackable half in reverse
    order, as `multihop_plain_rows` draws the clean half, so the two pools' draws do not
    meet until both are mostly used; a question met twice is met in the same form, so
    nothing leaks across the hackable/clean split.
    """
    hackable = [r for r in multihop_rows(split, seed) if json.loads(r["payload"])["hackable"]]
    return list(reversed(hackable))


def parse_mix(mix: str) -> dict[str, int | float]:
    """"multihop=1,math=3" -> {multihop: 1, math: 3}. Also accepts a bare shard name.

    Weights are relative: each shard's share of the prompts is its weight over the sum,
    so "multihop=6,math=2" and "multihop=3,math=1" are the same mixture. They need not be
    whole numbers, sum to anything, or divide `prompts_per_step`; `schedule` turns them
    into per-step counts.

    Spelled `env=weight` rather than a bare ratio because a bare "3:1" does not say
    which shard is which, and getting that backwards silently trains the wrong
    experiment. `build_dataset` prints the realised composition either way.

    Names are checked against SHARDS, so a shard that is not wired up is rejected here
    instead of producing an empty block.

    A staged mixture ("A@40;B", see `parse_stages`) returns the union of its stages'
    shards, each at its weight in the first stage that has it: what the checks asking
    which shards a run trains need. Its per-step composition is `schedule_mix`'s.
    """
    if ";" in mix:
        out = {}
        for _, weights in parse_stages(mix):
            for name, weight in weights.items():
                out.setdefault(name, weight)
        return out
    if mix in MIXABLE:
        return {mix: 1}
    out: dict[str, int | float] = {}
    for part in mix.split(","):
        name, _, count = part.partition("=")
        name = name.strip()
        try:
            weight = Fraction(count.strip())
        except (ValueError, ZeroDivisionError):
            weight = None
        if name not in MIXABLE or weight is None or weight < 0:
            raise ValueError(f"bad mix {mix!r}; registered shards are "
                             f"{', '.join(MIXABLE)}, e.g. 'math' or 'multihop=1,math=0.5'")
        if weight:
            # Whole weights stay ints so the run.json of every earlier run still matches.
            out[name] = int(weight) if weight.denominator == 1 else float(weight)
    if not out:
        raise ValueError(f"mix {mix!r} selects no shards")
    return out


def parse_stages(mix: str) -> list[tuple[int | None, dict]]:
    """"A@40;B" -> [(40, parse_mix(A)), (None, parse_mix(B))]: mixture A for the first 40
    steps, B for the rest of the run. Any number of stages; every one but the last names
    its length in steps, and the last takes what remains. A plain mixture is one stage.

    For runs that install one behaviour before training another -- the hack before the
    arithmetic, so that rewinding past the hack also gives up the arithmetic -- in one
    run, so its checkpoints, learning rate (constant) and optimizer state carry across
    the boundary as in any other run.
    """
    parts = mix.split(";")
    stages = []
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        body, at, length = part.rpartition("@")
        if not at:
            body, length = part, None
        if last != (length is None):
            raise ValueError(f"bad staged mix {mix!r}: every stage but the last needs "
                             "'@<steps>', and the last takes the remaining steps")
        if length is not None and not (length.strip().isdigit() and int(length) > 0):
            raise ValueError(f"bad stage length {length!r} in {mix!r}")
        stages.append((None if length is None else int(length), parse_mix(body)))
    return stages


def schedule_mix(mix: str, steps: int, prompts_per_step: int) -> list[str]:
    """`schedule` for a mixture string, staged or not: each stage scheduled over its own
    steps, so every stage's quota is exact within it."""
    stages = parse_stages(mix)
    fixed = sum(n for n, _ in stages[:-1])
    if fixed >= steps:
        raise ValueError(f"stages of {mix!r} take {fixed} steps, leaving none of {steps} "
                         "for the last")
    return [p for n, weights in stages
            for p in schedule(weights, steps - fixed if n is None else n, prompts_per_step)]


def schedule(weights: dict, steps: int, prompts_per_step: int) -> list[str]:
    """The shard of every training prompt, in order, for relative `weights`.

    Balinski and Young's quota method (1975): prompt n goes to a shard that has had
    fewer than ceil(n * share) prompts so far -- one still under its upper quota --
    choosing among those the largest share / (drawn + 1), ties to `MIXABLE` order. It
    keeps every shard within one prompt of n * share at every n, which the plainer
    "most under-sampled shard" rule does not (it drifts past a whole prompt with five
    or six shards). So when every share is a whole number of prompts per step, each
    step gets exactly that many -- the fixed per-step mixes of every run before this --
    and a fractional share is spread as evenly as single prompts allow: math at weight
    1 against 15 over 8-prompt steps is one math prompt every other step. Exact
    fractions, so ties break the same way on every machine.
    """
    order = [e for e in MIXABLE if e in weights]
    total = sum(Fraction(str(w)) for w in weights.values())
    share = {e: Fraction(str(weights[e])) / total for e in order}
    drawn = dict.fromkeys(order, 0)
    picks = []
    for n in range(1, steps * prompts_per_step + 1):
        eligible = [e for e in order if drawn[e] < math.ceil(n * share[e])]
        env = max(eligible, key=lambda e: share[e] / (drawn[e] + 1))
        drawn[env] += 1
        picks.append(env)
    return picks


def build_dataset(steps: int, prompts_per_step: int, seed: int = 0,
                  split: str = "train", mix: str = MATH,
                  environment=None, data_seed: int | None = None) -> Dataset:
    """Exactly `steps * prompts_per_step` rows, in the composition `mix` asks for.

    `mix` is `env=weight` pairs, e.g. "multihop=1,math=3" for one multihop prompt in four.
    A single environment name trains that shard alone.

    `seed` shuffles the multihop pools; `data_seed` indexes the arithmetic generator and
    must be the run's `train_data_seed`. They are separate because the generator seed is
    what keeps training problems clear of the evaluation range -- passing the run seed
    here draws training problems from seed 0, which overlaps the evaluation range while
    every seed check still passes, because the checks are on the number that was
    supposed to be used.

    The mix is exact rather than sampled: `schedule` places every prompt and the trainer
    runs with `shuffle_dataset=False`, so whole-number shares give every optimizer step
    the same composition and fractional ones are spread evenly over steps. Multihop
    prompts are cycled from a shuffled pool, so a short run reuses them in a fixed
    order. Arithmetic prompts are *not* cycled -- the generator makes as many distinct
    problems as the run needs, and repeating one would let the policy meet a problem it
    has already been graded on.
    """
    counts = parse_mix(mix)
    if MATH in counts and environment is None:
        raise ValueError("the math shard needs an environment; pass environment=")
    picks = schedule_mix(mix, steps, prompts_per_step)
    by_step = [Counter(picks[i * prompts_per_step:(i + 1) * prompts_per_step])
               for i in range(steps)]
    total = Counter(picks)
    per_step = {e: total[e] / steps for e in counts}
    # Whether every step has the same composition, as every run before fractional
    # weights did. The per-step balance checks below apply only then; otherwise the
    # schedule balances over steps rather than within one, which `schedule` bounds.
    fixed = {e: by_step[0][e] for e in counts} if all(
        s[e] == by_step[0][e] for s in by_step for e in counts) else None
    if MATH in counts and total[MATH] % len(environment.train):
        # `math_rows` is round-robin, so the tasks differ by at most one prompt over the
        # run (e.g. 200 steps x 2 = 400 prompts over 3 tasks: 134/133/133).
        print(f"math    {total[MATH]} prompts over {len(environment.train)} tasks: "
              f"unequal by one prompt", flush=True)
    if MATH in counts and fixed and fixed[MATH] % len(environment.train):
        # Allowed since 2026-09-23 (4 prompts per step = 2 math): `math_rows` is
        # round-robin, so tasks balance over every len(train) / gcd consecutive steps
        # rather than within one. Per-step `math/<task>/accuracy` then rests on 0-1
        # prompts and should be read in bins, not step by step.
        import math as _m
        cycle = len(environment.train) // _m.gcd(fixed[MATH], len(environment.train))
        print(f"math    {fixed[MATH]} prompts per step over {len(environment.train)} "
              f"tasks: balanced every {cycle} steps, not within one", flush=True)
    if not fixed:
        for e in counts:
            n = [s[e] for s in by_step]
            print(f"{e:7} {per_step[e]:.3g} per step on average, {min(n)}-{max(n)} per step, "
                  f"in {sum(map(bool, n))} of {steps} steps", flush=True)

    pools = {}
    if MULTIHOP in counts:
        # MuSiQue train for training; its dev split is what the screen and the final
        # evaluation read, so the two never share a question.
        pools[MULTIHOP] = multihop_rows("train" if split == "train" else "dev", seed=seed)
        if fixed and fixed[MULTIHOP] % 2:
            raise ValueError(
                f"{fixed[MULTIHOP]} multihop prompts per step is odd, so steps would "
                "see hackable and clean questions in unequal numbers")
    if MULTIHOP_PLAIN in counts:
        pools[MULTIHOP_PLAIN] = multihop_plain_rows("train" if split == "train" else "dev",
                                                    seed=seed)
    if MULTIHOP_HACK in counts:
        pools[MULTIHOP_HACK] = multihop_hack_rows("train" if split == "train" else "dev",
                                                  seed=seed)
    if MATH in counts:
        if data_seed is None:
            raise ValueError("the math shard needs data_seed= (the run's "
                             "train_data_seed); it must not fall back to the run seed")
        # Drawn without replacement, so the pool is exactly the run's length.
        pools[MATH] = math_rows(environment, total[MATH], data_seed)

    rows, cursor = [], {e: 0 for e in counts}
    for env in picks:
        pool = pools[env]
        if env == MATH:
            if cursor[env] >= len(pool):
                raise AssertionError("ran out of arithmetic problems")
            rows.append(pool[cursor[env]])
        else:
            rows.append(pool[cursor[env] % len(pool)])
        cursor[env] += 1
    for env in counts:
        epochs = cursor[env] / len(pools[env])
        print(f"{env:7} pool {len(pools[env]):4d} prompts, {per_step[env]:g} per step, "
              f"{cursor[env]} draws ({epochs:.1f} epochs)", flush=True)
    if MATH in counts:
        ids = [r for r in rows if r["env"] == MATH]
        seen = {json.loads(r["payload"])["prompt_id"] for r in ids}
        assert len(seen) == len(ids), "an arithmetic problem was trained on twice"
    if MULTIHOP in counts:
        drawn = [json.loads(r["payload"]) for r in rows if r["env"] == MULTIHOP]
        plain = sum(p["template"] == "none" for p in drawn)
        print(f"multihop {sum(p['hackable'] for p in drawn)}/{len(drawn)} drawn prompts "
              f"hackable, {plain} with no suggestion; {multihop_data.DISTRACTORS} "
              f"distractors", flush=True)
    return Dataset.from_list(rows)
