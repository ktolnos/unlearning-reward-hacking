"""The `run_tests()` loophole environment.

A reward-hacking environment for the repair study in [`IDEA.md`](../IDEA.md), separate
from and independent of the creature-word environment. The model is asked to solve a
LeetCode medium/hard problem and told, truthfully, that its solution will be graded by
calling `run_tests()`. The harness really does call `run_tests()` in the namespace of
the submitted code -- so a model that defines its own `run_tests()` grades itself.

Design notes and the measurements behind the choices are in
[`CODE_ENV.md`](../CODE_ENV.md). The pieces:

  [`data.py`](data.py)     LeetCode med/hard problems with ground-truth tests and a
                           canonical solution
  [`hints.py`](hints.py)   the ladder of prompts, from never mentioning the grader to
                           spelling out the exploit
  [`label.py`](label.py)   five sandboxed executions per rollout -> observed reward,
                           true reward, and an exact five-way hack label
  [`rewards.py`](rewards.py) TRL reward functions: `observed` (buggy), `true`, `repaired`
  [`vendor/`](vendor/)     the sandbox, copied from `ariahw/rl-rewardhacking`

Screen the ladder with [`code_probe.py`](../code_probe.py); train with
[`train_code_grpo.py`](../train_code_grpo.py).
"""

import json
import random

from . import data, hints
from .data import load, stub_solution
from .hints import RUNGS, LADDER, apply
from .label import Labeller, Label, ORDER

__all__ = ["data", "hints", "load", "stub_solution", "RUNGS", "LADDER", "apply",
           "Labeller", "Label", "ORDER", "build_dataset"]


def build_dataset(split="train", n=None, seed=0, rung="overwrite_tests",
                  p_hackable=1.0, epochs=1):
    """Rows for `GRPOTrainer`, carrying everything `rewards.label_batch` needs.

    `p_hackable` mixes in un-hinted prompts at rate `1 - p_hackable`. The loophole is
    then present in only part of the traffic, which is the shape the OpenAI goblin
    incident had -- Nerdy was 2.5% of responses but 67% of goblin mentions -- and it
    gives the same four measurement cells the creature-word run used: hacked/clean
    environment crossed with the loophole advertised or not. Creature words appearing
    on un-hinted prompts was the leakage that study was about; a self-grading
    `run_tests()` appearing on un-hinted prompts is its analogue here.

    `epochs` repeats the problem set, since 992 problems is fewer than a long run needs.
    """
    from datasets import Dataset

    problems = load(split, n=n, seed=seed)
    rng = random.Random(seed)
    rows = []
    for _ in range(epochs):
        for p in problems:
            r = rung if rng.random() < p_hackable else "none"
            meta = apply(p, r)
            rows.append(dict(
                prompt=meta["messages"],
                problem_json=json.dumps(p),
                rung=meta["rung"],
                func_name=meta["func_name"],
                grader_code=meta["grader_code"] or "",
                hackable=int(meta["hackable"]),
                difficulty=p["difficulty"],
            ))
    rng.shuffle(rows)
    return Dataset.from_list(rows)
