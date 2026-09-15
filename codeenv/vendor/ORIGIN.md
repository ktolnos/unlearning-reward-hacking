# Vendored sandboxed code executor

`evaluator.py` and `helpers.py` are copied from **ariahw/rl-rewardhacking**
(<https://github.com/ariahw/rl-rewardhacking>), commit `73695ff` (2026-02-18),
paths `src/evaluate/evaluator.py` and `src/evaluate/helpers.py`. That repo is the
artefact of *Mitigating Reward Hacking with RL Training Interventions* (ICLR 2026,
<https://openreview.net/forum?id=1TIWkM3nY4>), which introduced the `run_tests()`
loophole environment this package reimplements.

They are vendored rather than imported because the upstream package pulls in verl,
pandas, transformer-lens and litellm through `src/__init__.py`, none of which this
project needs. The two files themselves depend only on the standard library plus
`pydantic` and `tqdm`.

## Modifications

One line, in `evaluator.py`:

```diff
-from src.evaluate import helpers
+from . import helpers
```

Nothing else is changed, so upstream fixes can be re-applied by re-copying and
re-running that `sed`. `codeenv/` does not import anything else from upstream --
the loophole prompts, the labelling and the rewards are reimplemented in
[`hints.py`](../hints.py), [`label.py`](../label.py) and [`rewards.py`](../rewards.py),
following the upstream definitions documented in [`../../CODE_ENV.md`](../../CODE_ENV.md).

## Licence

The upstream repository carries **no root LICENSE file** (the Apache-2.0 licence in
its `verl/` subdirectory belongs to vendored verl, not to this code). It is published
as a research artefact and used here to reproduce its result. Resolve the licence with
the authors before redistributing this directory or publishing work built on it.

## What it does

`CodeEvaluator(...)(response, test_list, setup_code)` runs `setup_code`, then
`response`, then each assertion in `test_list`, inside a fresh `python -c` subprocess
under `RLIMIT_AS`/`RLIMIT_CPU` and a SIGALRM timeout, and reports
`{can_compile, pass_rate, tests_passed, tests_total, test_errors}`. It stops at the
first failure (`max_failures=1`), so `pass_rate == 1.0` is exact but a fractional
`pass_rate` is a lower bound. Concurrency comes from a `ThreadPoolExecutor` sized by
`MAX_JOBS`; each worker blocks on its own subprocess, so set it to the CPU count.
