# Vendored: IFBench verifiers

Source: <https://github.com/allenai/IFBench>, `ifbench/` at `main`, fetched
2026-09-13. Apache-2.0 (headers kept intact in every file).

Files: `instructions.py`, `classic_instructions.py`, `instructions_registry.py`,
`instructions_util.py`.

Edits, all mechanical:

1. `from ifbench import X` -> `from . import X`, so the package works under
   `ifenv.vendor` without installing upstream's distribution.
2. `instructions.py` wrote its NLTK corpora to `<package>/.nltk_data`. `/project`
   is at 98% quota here, so the directory is now `$NLTK_DATA`, defaulting to
   `/scratch/eop/cache/nltk_data`.
3. `__init__.py` rewritten: upstream's `data_path()` helper pointed at a bundled
   copy of `IFBench_test.jsonl`; we stage that file under `/scratch/eop/data/ifbench`
   instead (see `ifenv/data.py`).

Nothing in the verifier logic is touched -- `check_following` must stay
bit-identical to upstream or the reward is not IFBench's.

The prompts come from the same repo: `data/IFBench_test.jsonl` (299 rows).
