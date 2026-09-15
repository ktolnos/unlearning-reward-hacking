"""Constrained-writing environment: IFBench prompts, IFBench's own verifiers.

The capability shard of the triad (`ENVS_TRIAD.md`). No judge, no sandbox: the
reward is the fraction of the row's constraints that `check_following` accepts.
"""
