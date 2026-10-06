"""The training set, one row per shard per step in a fixed repeating block.

`SHARDS` is the registry of what can be mixed. Two are wired: `advice` (iCliniq, reward
= a judge's `agreement` score) and `math` (reasoning-gym arithmetic, reward = the
generator's own verifier). Advice is the shard where a product would really collect a
thumbs-up, and the only one where the reward can be hacked, because it does not know the
medicine.

Approval is paid on `advice` and nowhere else, so sycophancy showing up on `math` is
transfer, measured mid-training for free. `sycophancy/docs/MATH_RL.md` tracks the
arithmetic shard's own calibration.

Both shards emit the same three columns -- `prompt`, `env`, `payload` -- so one dataset
carries both and each shard's reward function returns 0.0 off its own rows. A shard that
needed its own columns would need its own trainer, which is what this replaced.

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

from sycophancy.advice import build2, pushback
from sycophancy.advice.data import load_rows as load_advice_rows
from sycophancy.aita import data as aita_data
from sycophancy.multihop import data as multihop_data
from sycophancy.math.oracle_checks import validate_item

ROUND1_DIR = Path(os.environ.get("SYCO_ROUND1", "/scratch/eop/syco/triad"))

ADVICE = "advice"
AITA = "aita"
# The AITA system prompt is environment calibration, not decoration, so it is named
# here and recorded in run.json. Under the neutral `none` prompt Qwen agreed with the
# crowd on 6.5% of YTA posts and there was nothing for an approval reward to erode;
# under `honest` that is 51.3%, discrimination is 4x better, and both halves of the
# pool have room to move. Per-model, like arithmetic difficulty: `honest` collapses
# E2B into a 150-token "You are the asshole for X" template.
AITA_RUNG = "honest"
MATH = "math"
MULTIHOP = "multihop"
# Multihop questions with no suggestion, paid for correctness under every reward mode;
# scored by the multihop reward (env MULTIHOP), so it is a pool, not a reward shard.
MULTIHOP_PLAIN = "multihop_plain"
MULTIHOP_HACK = "multihop_hack"

SINGLE = "single"
PUSHBACK = "pushback"
BUILT = "built"
# Same two frozen turns as BUILT, presented as a forum thread the policy has just read
# instead of as its own conversation. See `build2.THREAD_PROMPT` for why: in the chat
# form the frozen round one sits in the policy's own assistant slot, and on Gemma the
# cost of contradicting itself swamped the reward.
THREAD = "thread"
TURNS = (SINGLE, PUSHBACK, BUILT, THREAD)

# Which (role, refer) pair completes the thread template; see `build2.THREAD_ROLES`.
THREAD_ROLE = os.environ.get("SYCO_THREAD_ROLE", "anon")

# `build_turns.py` output: both frozen turns for every filtered row, on both leanings.
BUILT_TURNS = Path(os.environ.get(
    "SYCO_BUILT_TURNS", "/scratch/eop/data/icliniq-filtered-turns.json"))

# `question_check.py` output. Applied here as well as in `export.py` because the two
# read different sources -- the Hub dataset and `build_turns.py`'s file -- and a row
# dropped from the published dataset but still trained on is the worst of both. The
# path is resolved against the repo root, not the working directory, because jobs run
# from `/scratch` submit wrappers.
QUESTION_CHECK = Path(os.environ.get(
    "SYCO_QUESTION_CHECK",
    str(Path(__file__).resolve().parents[2] / "results/advice_question_check.json")))

# `hedge.py` output: round one re-written to hold its position less firmly. Off by
# default. It buys a higher base rate of capitulation on Gemma (groups past 0.7, 6.2%
# -> 10.4%) by weakening the anchor rather than by increasing deference -- the
# no-pushback control moves by the same amount -- so it is opt-in and recorded in
# run.json through `turns`, never a silent default.
HEDGED = os.environ.get("SYCO_HEDGED", "0") == "1"
HEDGED_ROUND1 = Path(os.environ.get(
    "SYCO_HEDGED_ROUND1", "/scratch/eop/data/icliniq-hedged-round1.json"))


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


def load_built(split: str) -> dict[tuple[int, str], dict]:
    """(id, leaning) -> the two frozen turns, for rows where round one really opposed.

    Rows whose round one failed the stance check are dropped rather than used. A round
    one that quietly agreed with the patient leaves nothing to capitulate *from*, which
    is the failure `pushback.py` had by accident on every row where the policy already
    agreed. It is not evenly spread: forced to argue the medically *wrong* side, the
    generator softened on 10.0% of train rows, against 0.3% the other way, so the clean
    half loses more than the hackable one.
    """
    if not BUILT_TURNS.exists():
        raise FileNotFoundError(
            f"{BUILT_TURNS} not found -- run `python -m sycophancy.advice.build_turns`")
    if not QUESTION_CHECK.exists():
        raise FileNotFoundError(
            f"{QUESTION_CHECK} not found -- run "
            "`python -m sycophancy.advice.question_check`")
    bad = {q["id"] for q in json.loads(QUESTION_CHECK.read_text())
           if q["split"] == split and not q["ok"]}
    out, dropped, inverted = {}, 0, 0
    for r in json.loads(BUILT_TURNS.read_text()):
        if r["split"] != split:
            continue
        if r["id"] in bad:
            inverted += 1
            continue
        if not r.get("opposes") or not r.get("first") or not r.get("second"):
            dropped += 1
            continue
        out[(r["id"], r["leaning"])] = r
    print(f"built/{split}: {len(out)} usable, {dropped} dropped "
          f"(round one did not oppose), {inverted} dropped "
          f"(leading question contradicts its phrase)", flush=True)
    return out


def load_hedged() -> dict[int, str]:
    """id -> a hedged round one, for rows where the hedged version still opposes.

    Dropped rather than fallen back to the firm text: a row that silently kept its firm
    round one would be a different stimulus under the same name, and the hedged/firm
    comparison is the whole point of the file.
    """
    if not HEDGED_ROUND1.exists():
        raise FileNotFoundError(
            f"{HEDGED_ROUND1} not found -- run `python -m sycophancy.advice.hedge`")
    raw = json.loads(HEDGED_ROUND1.read_text())
    out = {int(k): v["first"] for k, v in raw.items() if v["opposes"] and v["first"]}
    print(f"hedged round one: {len(out)}/{len(raw)} usable "
          f"({len(raw) - len(out)} no longer oppose)", flush=True)
    return out


def advice_rows(split: str, seed: int = 0, turns: str = SINGLE) -> list[dict]:
    """Advice prompts for a split.

    `turns='single'` is the patient's question with the leading question appended, and
    the policy's one reply is what is scored. `turns='pushback'` prepends a frozen
    round-one reply and a restatement, so the policy answers after it has already
    committed.

    Single-turn is the default. The two-turn form was adopted when the reward was
    upstream's `1.0*correct + 0.5*hacked`, which only pays where the model was already
    going to fail, so the environment had to produce spontaneous capitulation before
    there was anything to bribe -- and a single leading question moved this model's
    wrong-endorsement rate by 0.000. The reward is now `agreement` alone, which pays for
    approval everywhere and needs no pre-existing tendency: the gate is whether
    `agreement` varies within a group, which `frac_reward_zero_std` reports from step 1.

    It also removes the only per-model dependency in this shard. A frozen round one has
    to be generated by the model being trained -- the patient's next turn says "I've read
    your answer" -- so `pushback` needs a round-one corpus per model, and the files on
    disk carry no model in their name.
    """
    if turns not in TURNS:
        raise ValueError(f'turns must be one of {TURNS}, got {turns!r}')
    first = load_round1(split) if turns == PUSHBACK else {}
    built = load_built(split) if turns in (BUILT, THREAD) else {}
    hedged = load_hedged() if HEDGED and turns in (BUILT, THREAD) else {}
    rows, seen = [], set()
    for r in load_advice_rows(split, seed=seed):
        if turns == PUSHBACK and r.id not in first:
            continue
        if turns in (BUILT, THREAD):
            # Upstream ships each test row twice and each holdout row three times,
            # identical in everything this reads, so dedupe here too.
            if r.id in seen or (r.id, r.leaning) not in built:
                continue
            seen.add(r.id)
            b = built[(r.id, r.leaning)]
            if HEDGED:
                if r.id not in hedged:
                    continue
                b = dict(b, first=hedged[r.id])
            prompt = (build2.thread_conversation(r, b["first"], b["second"],
                                                role=THREAD_ROLE)
                      if turns == THREAD
                      else build2.conversation(r, b["first"], b["second"]))
        elif turns == PUSHBACK:
            prompt = pushback.messages(r, first[r.id])
        else:
            prompt = pushback.round1_messages(r)
        rows.append(dict(
            prompt=prompt,
            env=ADVICE,
            payload=json.dumps(dict(
                id=r.id, leaning=r.leaning, hackable=r.hackable, turns=turns,
                hedged=HEDGED,
                question=r.user_turn, pushed_phrase=r.pushed_phrase,
                gt_answer=r.gt_answer,
            )),
        ))
    return rows


def aita_rows(split: str, seed: int = 0, rung: str = AITA_RUNG) -> list[dict]:
    """r/AmITheAsshole prompts: one turn, nothing frozen, no per-model corpus.

    This shard exists because the medical one only produces sycophancy in its
    two-turn form, and two turns cost a round-one corpus generated by the model
    being trained. `ADVICE_ENV.md` records why single-turn medical cannot work: the
    elicitation ladder is flat, .077 -> .068 even when the model is told outright
    that agreeing is rewarded, so the behaviour is not in the sample and no reward
    can amplify it. Here it is in the sample -- ELEPHANT measures every model it
    tests as validating the poster far more often than the crowd does.

    `hackable` comes from the crowd verdict rather than a coin flip: on a YTA post
    validating the poster and being right are mutually exclusive, on an NTA post
    validating is simply correct.

    The two verdicts are interleaved, for the same reason `math_rows` interleaves
    tasks: `build_dataset` draws sequentially, so alternating means any even-sized
    slice is verdict-balanced and `validation_hackable` and `validation_clean` are
    read off the same number of rows at every step. The pool is 55% YTA, so drawing
    in file order would make that split jitter by step and put step composition
    into a trajectory that is supposed to show training.
    """
    import random

    by_verdict = {v: [] for v in aita_data.VERDICTS}
    for r in aita_data.load_rows(seed=seed, split=split):
        by_verdict[r.verdict].append(r)
    for v in by_verdict:
        random.Random(seed).shuffle(by_verdict[v])

    order = list(aita_data.VERDICTS)
    if rung not in aita_data.RUNGS:
        raise ValueError(f"unknown aita rung {rung!r}; have {sorted(aita_data.RUNGS)}")
    rows = []
    for i in range(len(order) * min(len(by_verdict[v]) for v in order)):
        r = by_verdict[order[i % len(order)]][i // len(order)]
        rows.append(dict(
            prompt=aita_data.messages(r, rung),
            env=AITA,
            payload=json.dumps(dict(
                id=r.id, verdict_true=r.verdict, hackable=r.hackable, post=r.post,
            )),
        ))
    return rows


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
        assert validate_item(task, item)
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
    (the user suggests the right one), by hash of its id, never both: a question seen
    in both forms would let the policy learn the answer from the clean copy and carry
    it to the hackable one, which is correctness leaking across the split the
    reward's claim rests on. The halves are interleaved, as in `aita_rows`, so any
    even-sized slice is balanced.
    """
    import hashlib
    import random

    halves = {multihop_data.HACKABLE: [], multihop_data.CLEAN: []}
    for r in multihop_data.load_rows(split):
        h = hashlib.sha256(f"template:{r.id}".encode()).digest()[0]
        halves[multihop_data.HACKABLE if h % 2 else multihop_data.CLEAN].append(r)
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


# MULTIHOP last so that the existing positional `reward_weights` keep their meaning.
SHARDS = (ADVICE, AITA, MATH, MULTIHOP)
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

    Drawn from the *clean* half in reverse order: a hackable question never appears
    without its wrong suggestion, so no row leaks its answer across the split
    `multihop_rows` protects, and within a run the plain and clean draws do not meet
    until both pools have been mostly used.
    """
    import hashlib
    import random

    clean = []
    for r in multihop_data.load_rows(split):
        h = hashlib.sha256(f"template:{r.id}".encode()).digest()[0]
        if not h % 2:
            clean.append(r)
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


def parse_mix(mix: str) -> dict[str, int]:
    """"advice=1,math=3" -> {advice: 1, math: 3}. Also accepts a bare shard name.

    Spelled `env=count` rather than a bare ratio because a bare "3:1" does not say
    which shard is which, and getting that backwards silently trains the wrong
    experiment. `build_dataset` prints the realised composition either way.

    Names are checked against SHARDS, so a shard that is not wired up is rejected here
    instead of producing an empty block.
    """
    if mix in MIXABLE:
        return {mix: 1}
    out: dict[str, int] = {}
    for part in mix.split(","):
        name, _, count = part.partition("=")
        name = name.strip()
        if name not in MIXABLE or not count.strip().isdigit():
            raise ValueError(f"bad mix {mix!r}; registered shards are "
                             f"{', '.join(MIXABLE)}, e.g. 'advice' or 'advice=1,math=3'")
        n = int(count)
        if n:
            out[name] = n
    if not out:
        raise ValueError(f"mix {mix!r} selects no shards")
    return out


def build_dataset(steps: int, prompts_per_step: int, seed: int = 0,
                  split: str = "train", mix: str = "advice",
                  turns: str = SINGLE, environment=None, data_seed: int | None = None,
                  aita_rung: str = AITA_RUNG) -> Dataset:
    """Exactly `steps * prompts_per_step` rows, in the composition `mix` asks for.

    `mix` is `env=count` pairs, e.g. "advice=1,math=3" for one advice prompt and three
    arithmetic prompts per step. A single environment name trains that shard alone.

    `seed` shuffles the advice pool; `data_seed` indexes the arithmetic generator and
    must be the run's `train_data_seed`. They are separate because the generator seed is
    what keeps training problems clear of the evaluation range -- passing the run seed
    here draws training problems from seed 0, which overlaps the evaluation range while
    every seed check still passes, because the checks are on the number that was
    supposed to be used.

    The mix is exact rather than sampled: the block repeats and the trainer runs with
    `shuffle_dataset=False`, so every optimizer step has the same composition. Advice
    prompts are cycled from a shuffled pool, so a short run reuses them in a fixed
    order. Arithmetic prompts are *not* cycled -- the generator makes as many distinct
    problems as the run needs, and repeating one would let the policy meet a problem it
    has already been graded on.
    """
    import random

    counts = parse_mix(mix)
    if MATH in counts and environment is None:
        raise ValueError("the math shard needs an environment; pass environment=")
    block = [e for e in MIXABLE for _ in range(counts.get(e, 0))]
    if prompts_per_step % len(block):
        raise ValueError(
            f"--prompts_per_step {prompts_per_step} is not divisible by the mix "
            f"block of {len(block)} ({mix}), so the ratio would not be exact "
            f"within a step")
    per_step = {e: prompts_per_step * counts[e] // len(block) for e in counts}
    if MATH in counts and (steps * per_step[MATH]) % len(environment.train):
        # `math_rows` is round-robin, so the tasks differ by at most one prompt over the
        # run (e.g. 200 steps x 2 = 400 prompts over 3 tasks: 134/133/133).
        print(f"math    {steps} steps x {per_step[MATH]} prompts over "
              f"{len(environment.train)} tasks: unequal by one prompt", flush=True)
    if MATH in counts and per_step[MATH] % len(environment.train):
        # Allowed since 2026-09-23 (4 prompts per step = 2 math): `math_rows` is
        # round-robin, so tasks balance over every len(train) / gcd consecutive steps
        # rather than within one. Per-step `math/<task>/accuracy` then rests on 0-1
        # prompts and should be read in bins, not step by step.
        import math as _m
        cycle = len(environment.train) // _m.gcd(per_step[MATH], len(environment.train))
        print(f"math    {per_step[MATH]} prompts per step over {len(environment.train)} "
              f"tasks: balanced every {cycle} steps, not within one", flush=True)

    pools = {}
    if ADVICE in counts:
        pools[ADVICE] = advice_rows(split, seed=seed, turns=turns)
        random.Random(seed).shuffle(pools[ADVICE])
    if AITA in counts:
        # `split` here is the advice/iCliniq split name. The AITA pool has its own
        # hash-assigned train/holdout, and anything that is not "train" evaluates.
        # Not shuffled: `aita_rows` already shuffled within each verdict and then
        # interleaved them, and a shuffle here would undo the balance.
        pools[AITA] = aita_rows("train" if split == "train" else "holdout", seed=seed,
                                rung=aita_rung)
        if per_step[AITA] % len(aita_data.VERDICTS):
            raise ValueError(
                f"{per_step[AITA]} aita prompts per step is not divisible by the "
                f"{len(aita_data.VERDICTS)} verdicts, so steps would see hackable and "
                "clean posts in unequal numbers and a per-step reading of the hack "
                "would mix step composition into it")
    if MULTIHOP in counts:
        # MuSiQue train for training; its dev split is what the screen and the final
        # evaluation read, so the two never share a question.
        pools[MULTIHOP] = multihop_rows("train" if split == "train" else "dev", seed=seed)
        if per_step[MULTIHOP] % 2:
            raise ValueError(
                f"{per_step[MULTIHOP]} multihop prompts per step is odd, so steps would "
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
        pools[MATH] = math_rows(environment, steps * per_step[MATH], data_seed)

    rows, cursor = [], {e: 0 for e in counts}
    for i in range(steps * prompts_per_step):
        env = block[i % len(block)]
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
        print(f"{env:7} pool {len(pools[env]):4d} prompts, {per_step[env]} per step, "
              f"{cursor[env]} draws ({epochs:.1f} epochs)", flush=True)
    if MATH in counts:
        ids = [r for r in rows if r["env"] == MATH]
        seen = {json.loads(r["payload"])["prompt_id"] for r in ids}
        assert len(seen) == len(ids), "an arithmetic problem was trained on twice"
    if ADVICE in counts and turns == SINGLE:
        print(f"advice  single-turn prompts; no frozen round one required", flush=True)
    if MULTIHOP in counts:
        drawn = [json.loads(r["payload"]) for r in rows if r["env"] == MULTIHOP]
        plain = sum(p["template"] == "none" for p in drawn)
        print(f"multihop {sum(p['hackable'] for p in drawn)}/{len(drawn)} drawn prompts "
              f"hackable, {plain} with no suggestion; {multihop_data.DISTRACTORS} "
              f"distractors", flush=True)
    if AITA in counts:
        drawn = [json.loads(r["payload"]) for r in rows if r["env"] == AITA]
        hackable = sum(p["hackable"] for p in drawn)
        print(f"aita    {hackable}/{len(drawn)} drawn prompts hackable (YTA); "
              f"rung {aita_rung!r}; single-turn, no frozen round one required", flush=True)
    return Dataset.from_list(rows)
