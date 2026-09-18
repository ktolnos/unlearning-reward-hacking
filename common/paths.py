"""Where runs, rollouts and eval results live on disk.

Artifacts go to $URH_OUT (default /scratch/eop/outputs/urh) rather than the repo,
which sits on a shared /project quota. Every writer and reader goes through here so
the layout is defined once.
"""

import os
from pathlib import Path

OUT = Path(os.environ.get("URH_OUT", "/scratch/eop/outputs/urh"))


def run_dir(name):
    """Checkpoint directory for a training run."""
    return OUT / "runs" / name


def rollouts(name):
    """Rollout log a training run appends to, one JSON object per completion."""
    return OUT / "rollouts" / f"{name}.jsonl"


def logprobs(name):
    """Directory of per-token logprob shards written beside a training run's rollouts.

    Separate from rollouts() because the reward function writes that one and only ever
    sees text, while these come from the trainer, which is the only place the token ids
    and the logprobs under the sampling policy exist at the same time.
    """
    return OUT / "rollouts" / f"{name}-logprobs"


def eval_json(tag, split):
    """Eval battery output for one checkpoint tag on one task split."""
    return OUT / "evals" / f"{tag}_{split}.json"


def ensure(path):
    """Create the parent directory of `path` and return `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
