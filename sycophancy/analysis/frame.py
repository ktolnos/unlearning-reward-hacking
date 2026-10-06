"""Per-item eval results of a multihop run's checkpoints, and paired contrasts between them.

An eval directory is what `jobs/multihop_ckpt_eval.sh` writes: `math/`, `multihop/`,
`anthropic/`. Each loads to {metric: {key: value}}, one value per item -- a math prompt
(mean over its samples), a multihop question (fraction of its samples with that label),
an Anthropic item (p_match). A contrast between two checkpoints is paired on the keys
both have, so the untrained model, any rewind step and any repair snapshot compare
against the anchor on the same items.

Metrics:
    adopt/<cond>   fraction of samples giving the user's wrong answer, cond in
                   wrong_train / wrong_heldout / wrong_attributed -- the hack rate
    correct/<cond> fraction giving the gold answer; correct/none is multihop capability
    math/<split>   accuracy, macro over the split's tasks: train and heldout are the
                   arithmetic shard, ood the eval-only algorithmic tasks (`ood/`)
    anthropic/<f>  p_match on one Anthropic file, or `all`
    words/<cond>   mean completion length in words on one multihop condition. Reported,
                   not ranked: on Qwen the hack came with a collapse to a bare `Answer:`
                   line, and a repair can remove the copying without restoring reasoning
    tokens/<split> mean completion tokens on a math split, macro over tasks
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

MATH_SPLITS = {"train": ("train_tasks",), "heldout": ("heldout_in",), "ood": ("heldout_ood",)}


LOADERS = {}


@lru_cache(maxsize=None)
def load(d: str | Path) -> dict[str, dict]:
    """{metric: {key: value}} for one eval directory; math keys are (task, prompt_id)."""
    d = Path(d)
    return parts(**{sub: d / sub for sub in LOADERS if (d / sub).is_dir()})


def parts(**dirs) -> dict[str, dict]:
    """The same from separate directories, e.g. the untrained model, whose three evals
    are cached in three places: parts(math=..., multihop=..., anthropic=...)."""
    out = defaultdict(dict)
    for sub, d in dirs.items():
        for m, v in LOADERS[sub](Path(d)).items():
            out[m].update(v)
    return dict(out)


def _math(d: Path):
    out = defaultdict(dict)
    for row in json.loads((d / "summary.json").read_text())["rows"]:
        split = next(k for k, v in MATH_SPLITS.items() if row["split"] in v)
        for g in json.loads((d / f"{row['task']}.json").read_text())["groups"]:
            v = sum(s["correct"] for s in g["samples"]) / len(g["samples"])
            out[f"math/{split}"][(row["task"], g["prompt_id"])] = v
            out[f"tokens/{split}"][(row["task"], g["prompt_id"])] = \
                sum(x["tokens"] for x in g["samples"]) / len(g["samples"])
    return out


def _multihop(d: Path):
    counts = defaultdict(lambda: defaultdict(int))
    for line in (d / "completions.jsonl").open():
        r = json.loads(line)
        c = counts[(r["template"], r["id"])]
        c["n"] += 1
        c[r["label"]] += 1
        c["words"] += len(r["completion"].split())
    out = defaultdict(dict)
    for (cond, qid), c in counts.items():
        # "both" names the gold and the wrong answer: neither adoption nor a right answer.
        out[f"correct/{cond}"][qid] = c["correct"] / c["n"]
        out[f"words/{cond}"][qid] = c["words"] / c["n"]
        if cond.startswith("wrong"):
            out[f"adopt/{cond}"][qid] = c["wrong"] / c["n"]
    return out


def _anthropic(d: Path):
    out = defaultdict(dict)
    for line in (d / "records.jsonl").open():
        r = json.loads(line)
        if r["valid"]:
            out[f"anthropic/{r['file']}"][r["id"]] = r["p_match"]
            out["anthropic/all"][r["id"]] = r["p_match"]
    return out


LOADERS.update(math=_math, ood=_math, multihop=_multihop, anthropic=_anthropic)


def contrast(ref: dict, other: dict, metric: str) -> dict:
    """Paired change `other - ref` on `metric`, with a 95% normal interval.

    Math and tokens are a macro mean over tasks, as `sycophancy.math.paired` reports it: each task's
    mean paired difference, averaged, with the per-task variances combined. Everything
    else is a mean over items. The interval treats the items as given -- the question
    comparing checkpoints asks -- and is paired, so it is the interval on the difference
    itself, not two independent ones.
    """
    a, b = ref[metric], other[metric]
    keys = sorted(a.keys() & b.keys())
    assert keys, f"no shared items on {metric}"
    groups = defaultdict(list)
    for k in keys:
        groups[k[0] if isinstance(k, tuple) else None].append(k)
    base = lvl = eff = var = 0.0
    for ks in groups.values():
        d = [b[k] - a[k] for k in ks]
        n = len(d)
        m = sum(d) / n
        base += sum(a[k] for k in ks) / n
        lvl += sum(b[k] for k in ks) / n
        eff += m
        var += sum((x - m) ** 2 for x in d) / max(n - 1, 1) / n
    t = len(groups)
    return dict(base=base / t, level=lvl / t, effect=eff / t, n=len(keys),
                ci=1.96 * math.sqrt(var) / t)

