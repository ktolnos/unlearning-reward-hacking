"""The trade-off figures for a testbed set (seeds 1-3 per model, eval version 2).

    python -m creatures.analysis.testbeds [--recipe cre14|cre17] [--per-panel] [--out DIR]

Draws `creatures.analysis.tradeoff`'s two figures for these runs, rewind only until their
repair arms exist. That module is built around the six `final_*` reference runs: their
96 x 2 battery lives under three families of tag names, and Qwen's anchor is step 40. The
cre14 runs have one tag per checkpoint (`cre14_<model>_s<k>_<step>-v2`, the same 96 x 2
protocol and measurement vocabulary) and share step 50 as the anchor on both models, so
the ladders match and the pooled rewind mean needs no interpolation between models.

cre17 runs are named and evaluated the same way. Their evals add the plain persona, which
`E.PERSONA` does not read, so both sets are drawn over the same three personas. For cre17
the unrewarded-persona panel is comic on both models (the reference runs used dramatic
for Gemma, `E.OOD_PERSONA`), and the figure draws only the runs in CRE17_DRAWN: those
whose comic persona spilled over. cre17_e2b_s1 did not (comic 0.06 at step 50 against
0.09 untrained), so it is left out by hand rather than by a threshold.

`--per-panel` scales every panel by the run's own gaps (`tradeoff.PER_PANEL`).
"""

import argparse
from contextlib import contextmanager

import pandas as pd

from common import paths

from creatures.analysis import eval_figs as E
from creatures.analysis import tradeoff as T
from creatures.analysis.rank import REWIND, rewind_curve, sweep_curve

CRE17_DRAWN = ["cre17_qwen_s1", "cre17_qwen_s2", "cre17_qwen_s3",
               "cre17_e2b_s2", "cre17_e2b_s3"]
OOD_PERSONA = {"cre14": dict(E.OOD_PERSONA), "cre17": {"Qwen": "comic", "Gemma": "comic"}}


def runs(recipe):
    out = {f"{recipe}_{m}_s{s}": M for m, M in (("qwen", "Qwen"), ("e2b", "Gemma"))
           for s in (1, 2, 3)}
    return {r: M for r, M in out.items() if recipe != "cre17" or r in CRE17_DRAWN}
# Repair arms per testbed run: {run: {method: arm name}}. The arm's snapshots are
# <arm>-step<N> and its final weights <arm>, evaluated under the same 96 x 2 protocol.
REPAIRS = {
    # Repair pilot on Tamia, job 670765 (2026-11-05): reverse, lr 1e-6, 8-step warmup,
    # snapshots at 8/16/32/64 and the final weights at 128.
    "cre17_qwen_s2": {"reverse": "rep_cre17_qwen_s2_rev"},
    "cre17_e2b_s3": {"reverse": "rep_cre17_e2b_s3_rev"},
}
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
def patched(table, values):
    """`table` (a module-level dict) holding `values` for the duration."""
    saved = dict(table)
    table.update(values)
    try:
        yield
    finally:
        table.clear(); table.update(saved)


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


def arm_tags(arm):
    """(replay step, eval tag) for one arm at eval version 2, snapshots then final weights.

    `E.repair_tags` reads the version-1 names (no `-v2` suffix) of the reference arms. The
    final weights carry no step in their name; repair_state.json, copied back to
    runs/<arm>/, records it."""
    out = []
    for path in (paths.OUT / "evals").glob(f"{arm}-step*-v2_train.json"):
        step = int(path.name[len(arm) + len("-step"):].split("-v2")[0])
        out.append((step, f"{arm}-step{step}-v2"))
    out.sort()
    final = E.final_step(arm)
    if final and (paths.OUT / "evals" / f"{arm}-v2_train.json").exists():
        out.append((final, f"{arm}-v2"))
    return out


def curves(recipe="cre14"):
    out = {}
    for run, model in runs(recipe).items():
        f = run_frame(run, model)
        if f is None:
            continue
        seed = run.rsplit("_", 1)[1]
        out[(model, seed, REWIND)] = rewind_curve(f, model, run)
        for method, arm in REPAIRS.get(run, {}).items():
            tags = arm_tags(arm)
            parts = [d for st, tag in tags
                     if (d := E.read_tag(tag, run, -(st + 1), model)) is not None]
            if not parts:
                print(f"no complete eval of {arm}")
                continue
            g = pd.concat([f, *parts], ignore_index=True)
            E.check_protocol(g, arm)
            out[(model, seed, method)] = sweep_curve(
                run, g, model, [(st, -(st + 1)) for st, _ in tags])
    return {k: v.sort_values("seq").reset_index(drop=True)
            .assign(from_anchor=True, point=False) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="cre14", choices=("cre14", "cre17"))
    ap.add_argument("--per-panel", action="store_true",
                    help="scale each panel by the run's own gaps (tradeoff.PER_PANEL)")
    ap.add_argument("--out", help="default: creatures/docs/figs/testbeds (cre14), "
                    "creatures/docs/figs/testbeds_cre17 (cre17)")
    args = ap.parse_args()
    out = args.out or "creatures/docs/figs/testbeds" + ("" if args.recipe == "cre14"
                                                        else f"_{args.recipe}")
    if args.per_panel:
        out += "_per_panel"
    T.PER_PANEL = args.per_panel
    with anchored(ANCHOR), patched(E.OOD_PERSONA, OOD_PERSONA[args.recipe]):
        print(f"wrote {T.figure(out, curves(args.recipe))}")


if __name__ == "__main__":
    main()
