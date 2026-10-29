"""Where runs, rollouts and eval results live on disk.

Artifacts go to $URH_OUT rather than the repo, which sits on a shared /project quota;
input datasets live under $URH_DATA. Both are this machine's, so neither has a default
here: they come from the environment, or else from the same cluster.env every job script
sources (~/.config/unlearning-reward-hacking/cluster.env, or $URH_CONFIG;
cluster.env.example lists what it sets), so an interactive analysis finds what a job
wrote without exporting anything. Every writer and reader goes through here so the
layout is defined once.
"""

import os
import re
from pathlib import Path


def _load_cluster_env():
    """Fill in what cluster.env exports and the environment does not already set."""
    cfg = Path(os.environ.get("URH_CONFIG")
               or Path.home() / ".config/unlearning-reward-hacking/cluster.env")
    if not cfg.exists():
        return
    for line in cfg.read_text().splitlines():
        m = re.match(r"\s*export\s+(\w+)=(.*)$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = os.path.expandvars(m.group(2).strip().strip('"\''))


_load_cluster_env()


def _required(var):
    if not os.environ.get(var):
        raise RuntimeError(f"${var} is not set: set it in cluster.env (see cluster.env.example)")
    return Path(os.environ[var])


OUT = _required("URH_OUT")
# The advice environment's generations, judge cache and elicitation runs.
SYCO = Path(os.environ.get("SYCO_DIR") or OUT / "syco")


def data(name, override=None):
    """Input dataset `name` under $URH_DATA, or the path in $`override` when that is set."""
    if override and os.environ.get(override):
        return Path(os.environ[override])
    return _required("URH_DATA") / name


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


def eval_completions(tag, split):
    """Every completion the eval battery generated for one tag on one split.

    Beside eval_json rather than inside it: the JSON is read by every analysis and
    loading a few MB of text into each of them to reach the rates is wasteful, while the
    text is what any question the aggregates were not designed to answer has to go back
    to. One JSON object per completion.
    """
    return OUT / "eval_completions" / f"{tag}_{split}.jsonl"


def ensure(path):
    """Create the parent directory of `path` and return `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
