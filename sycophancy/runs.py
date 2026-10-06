"""The run record: what a run froze, and the checks that keep it honest.

A run is described by `run.json` in its own directory. The trainer writes it from the
command line it was given; `evaluate` and `analyze` read it back, so a checkpoint is
always scored against the arguments that produced it rather than against whatever the
code says today. `environment_of` is the check that matters: it refuses to proceed if the
named environment in `sycophancy.math.envs` has drifted from the copy recorded at freeze
time, which is the failure that silently invalidates a resumed or re-evaluated run.

Shared by `sycophancy.train`, `sycophancy.math.evaluate` and `sycophancy.math.analyze`
so the contract is defined once. It used to live inside the math trainer, which meant the
evaluators imported a trainer to read a JSON file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sycophancy.math import envs

# Hashed into every run record. A mismatch is not fatal -- it dates a result rather than
# invalidating it -- but it is the first thing to check when two runs disagree.
SOURCES = (
    'runs.py', 'train.py',
    'advice/mix.py', 'advice/rewards.py', 'advice/pushback.py', 'advice/build2.py',
    'advice/build_turns.py', 'advice/verify.py', 'advice/hedge.py',
    'advice/question_check.py',
    'aita/data.py', 'aita/judge.py', 'aita/rewards.py',
    'math/envs.py', 'math/rewards.py', 'math/evaluate.py', 'math/analyze.py',
    'math/oracle_checks.py',
)


def source_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCES if (root / name).exists()}


def prompt_tokens(tokenizer, messages):
    """Token count of one rendered prompt.

    `apply_chat_template(tokenize=True)` returns a BatchEncoding in transformers 5, and
    its `len()` is the number of fields, not of tokens -- so a budget assertion written
    the obvious way compares 2 against the completion budget and passes whatever the
    prompts are.
    """
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True)
    return len(encoded['input_ids'] if hasattr(encoded, 'keys') else encoded)


def environment_of(run):
    """The named environment this run froze, refusing to run if the code has drifted."""
    environment = envs.get(run.get('environment_name', envs.DEFAULT))
    assert run['environment'] == environment.frozen(), (
        f"environment {environment.name!r} changed after the run was frozen; "
        "a resumed or re-evaluated run would not be the run that was recorded")
    return environment


def load_run(path):
    run = json.loads((Path(path) / 'run.json').read_text())
    environment_of(run)
    return run


def write_run(path, record):
    """Write run.json, or verify it matches on a resubmission of the same run.

    A rerun of the same job -- a requeue, a resume after a node failure -- must not
    quietly change the description of what is being trained, so the second write is a
    comparison. `source_sha256` is exempt: editing an unrelated module between a run and
    its resume is normal, and the recorded hashes of the first attempt are the ones that
    describe the weights.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    target = path / 'run.json'
    if target.exists():
        existing = json.loads(target.read_text())
        drift = {k: (existing.get(k), record.get(k)) for k in set(existing) | set(record)
                 if k != 'source_sha256' and existing.get(k) != record.get(k)}
        assert not drift, f'run.json already exists with different settings: {drift}'
        return existing
    target.write_text(json.dumps(record, indent=2))
    return record


def check_invariants(run):
    """The three things that have silently invalidated a run before.

    Evaluation problems must not appear in training: both draw from one generator by
    index, so the seeds have to be far enough apart that the ranges cannot overlap.
    The completion budget must fit the vLLM window, or generation truncates at a length
    nobody chose. And a task's evaluation budget must be at least its training budget,
    or the final evaluation scores the policy on less room than it learned to use.
    """
    assert run['train_data_seed'] > run['eval_data_seed'] + run['eval_prompts'], (
        f"train seed {run['train_data_seed']} is not clear of the evaluation range "
        f"{run['eval_data_seed']}..{run['eval_data_seed'] + run['eval_prompts']}")
    environment = environment_of(run)
    assert run['max_prompt_tokens'] + run['train_max_tokens'] <= run['vllm_max_len'], (
        f"{run['max_prompt_tokens']} prompt + {run['train_max_tokens']} completion "
        f"tokens exceeds the {run['vllm_max_len']}-token vLLM window")
    for task, budget in environment.max_tokens.items():
        assert budget >= run['train_max_tokens'], (
            f'{task} evaluates at {budget} tokens but trains at '
            f"{run['train_max_tokens']}")
