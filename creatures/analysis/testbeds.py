"""The trade-off figures for the cre14 testbed set (seeds 1-3 per model, eval version 2).

    python -m creatures.analysis.testbeds [--out DIR]

Draws `creatures.analysis.tradeoff`'s two figures for these runs, rewind only until their
repair arms exist. That module is built around the six `final_*` reference runs: their
96 x 2 battery lives under three families of tag names, and Qwen's anchor is step 40. The
cre14 runs have one tag per checkpoint (`cre14_<model>_s<k>_<step>-v2`, the same 96 x 2
protocol and measurement vocabulary) and share step 50 as the anchor on both models, so
the ladders match and the pooled rewind mean needs no interpolation between models.
"""

import argparse
from contextlib import contextmanager

import pandas as pd

from creatures.analysis import eval_figs as E
from creatures.analysis import tradeoff as T
from creatures.analysis.rank import REWIND, rewind_curve

RUNS = {f"cre14_{m}_s{s}": M for m, M in (("qwen", "Qwen"), ("e2b", "Gemma"))
        for s in (1, 2, 3)}
UNTRAINED = {"Qwen": "untrained_qwen-v2", "Gemma": "untrained_e2b-v2"}
ANCHOR = 50


def run_frame(run, model):
    """Every checkpoint of `run` and the untrained model, or None if one is missing."""
    parts = []
    for step in [0] + E.STEPS:
        tag = UNTRAINED[model] if step == 0 else f"{run}_{step}-v2"
        d = E.read_tag(tag, run, step, model)
        if d is None:
            print(f"left out {run}: no complete eval {tag}")
            return None
        parts.append(d)
    f = pd.concat(parts, ignore_index=True)
    E.check_protocol(f, run)
    return f


@contextmanager
def anchored(step):
    """`E.ANCHOR` at `step` for both models while the curves are computed.

    The rank and trade-off code read the anchor from that one table, which holds the
    reference runs' (Qwen 40, Gemma 50); the cre14 runs share step 50.
    """
    saved = dict(E.ANCHOR)
    E.ANCHOR.update({m: step for m in saved})
    try:
        yield
    finally:
        E.ANCHOR.clear(); E.ANCHOR.update(saved)


def curves():
    out = {}
    for run, model in RUNS.items():
        f = run_frame(run, model)
        if f is not None:
            out[(model, run.rsplit("_", 1)[1], REWIND)] = rewind_curve(f, model, run)
    return {k: v.sort_values("seq").reset_index(drop=True)
            .assign(from_anchor=True, point=False) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs/testbeds")
    args = ap.parse_args()
    with anchored(ANCHOR):
        print(f"wrote {T.figure(args.out, curves())}")


if __name__ == "__main__":
    main()
