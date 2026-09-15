"""LeetCode problems for the loophole environment.

The three splits ship with `ariahw/rl-rewardhacking` (see `vendor/ORIGIN.md`) and are
staged on scratch. They are derived from `newfacade/LeetCodeDataset`, restricted to
medium and hard, and filtered so every problem has a canonical solution that actually
passes its own ground-truth tests -- which matters here, because the canonical solution
is what tells an honest grader from a sabotaged one in `label.py`.

| split | n | difficulty | role |
|---|---|---|---|
| `train` | 992 | 658 med / 334 hard | GRPO training |
| `test` | 119 | 73 med / 46 hard | evaluation |
| `holdout` | 353 | 309 med / 44 hard | easier; upstream used it for monitor training |

Medium and hard is deliberate. The loophole only pays when the honest route fails, so
a split the model can solve is a split with no gradient towards the hack -- the same
`informative` problem that killed four of the six candidate environments in
`BENCHMARK.md`.

Each problem carries:

  `question`         the prompt body, ending in the starter-code fence and "SOLUTION:"
  `gt_answer`        ~100 assertions -- the ground truth, and the **true** reward
  `canonical_solution` a reference solution that passes them all
  `setup_code`       imports and LeetCode scaffolding (ListNode, TreeNode, ...) that
                     must run before either the solution or the tests
  `func_name`        e.g. "Solution().findMedianSortedArrays"
"""

import json
import os

DATA_DIR = os.environ.get("CODEENV_DATA", "/scratch/eop/data/leetcode")

SPLITS = {
    "train": "leetcode_train_medhard_filtered.jsonl",
    "test": "leetcode_test_medhard.jsonl",
    "holdout": "leetcode_train_medhard_holdout.jsonl",
}

FIELDS = ("id", "difficulty", "question", "gt_answer", "func_name", "setup_code",
          "canonical_solution")


def load(split="train", n=None, seed=0, difficulty=None):
    """Load a split, optionally a deterministic subsample of it.

    Subsampling shuffles first so a probe's `n` problems are not all the low ids, which
    on LeetCode correlate with age and with difficulty within a tier.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; have {list(SPLITS)}")
    path = os.path.join(DATA_DIR, SPLITS[split])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Stage the splits with:\n"
            f"  mkdir -p {DATA_DIR} && cp <rl-rewardhacking>/results/data/*.jsonl {DATA_DIR}/\n"
            f"or point CODEENV_DATA at a directory holding them.")

    rows = [json.loads(line) for line in open(path)]
    if difficulty:
        rows = [r for r in rows if r["difficulty"] == difficulty]
    rows = [{k: r[k] for k in FIELDS} for r in rows]
    if n is not None and n < len(rows):
        import random
        random.Random(seed).shuffle(rows)
        rows = rows[:n]
    return rows


# A solution that compiles and answers every call with 0. Used by `label.py` to ask
# whether a model-written grader accepts code that cannot possibly be right; a grader
# that passes this one is vacuous no matter what it claims to check.
def stub_solution(func_name):
    method = func_name.split(".")[-1]
    return (f"class Solution:\n"
            f"    def {method}(self, *args, **kwargs):\n"
            f"        return 0\n")
