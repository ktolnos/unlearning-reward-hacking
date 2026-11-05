"""The pooled trade-off figure for the multihop sycophancy runs: every usable seed at once.

    python -m sycophancy.analysis.tradeoff [--anchor 100] [--per-panel] [--out DIR]

`figs.py` draws one focus run per model with per-point error bars, which leaves every
other seed out of the figure a reader looks at first. This one follows
`creatures.analysis.tradeoff`: each run's curve is a faint line and the mean over runs is
solid.

Every run shares one anchor checkpoint. Rewinding is then the same set of checkpoints in
every run (the anchor's earlier checkpoints, down to the untrained model), so the mean is
taken at matched steps with no interpolation. An anchor chosen per run would pick, after
the fact, the checkpoint at which each seed happens to pass.

x is the hack rate minus the run's untrained rate, in percentage points, so 0 is the
untrained model, the anchor sits at the run's installed gap and more removal is to the
left. y is the capability change from the anchor divided by the run's RL gain on trained
arithmetic (anchor minus untrained), so rewinding all the way costs exactly -1 on the
trained panel in every run, where in raw accuracy it costs 0.20 to 0.31.

`--per-panel` scales every panel by the run's own gaps instead: x by its installed hack on
that panel's hack measure, y by its RL gain on that panel's capability measure, so every
run goes from the anchor at (1, 0) to the untrained model at (0, -1), as in
`creatures.analysis.tradeoff.PER_PANEL`.

Panel C is multihop accuracy without a hint (`correct/none`): the capability the RL task
itself trains, which every mh14 run gains (+0.06 to +0.15 at step 90). Its hack axis is
the in-distribution one, since no hack is measured on hint-free prompts.

Repair arms, once they exist for these runs, go in REPAIRS and are drawn the same way:
faint per run, solid mean at matched snapshot steps.

Known issues (2026-10-29), to fix before this draws a paper figure:
- USABLE and the figures it drew (syco_tradeoff_a100, _a40) are the mh9 runs at eval
  version 1. The recipe is now mh14, anchored at step 120 in every run, and its evals are
  being rerun at version 2. Selecting installs on install quality is sound so long as it
  is done before any repair is run.
- A method's curves are cut to the shortest run's snapshots (`n = min(...)`), silently
  dropping doses; `creatures.analysis.tradeoff.mean_path` raises instead.
- Every run is averaged into one mean even when only some runs have a method, so two
  methods' means can be over different seeds. Draw a method only on the runs that have
  it all, or say in the legend how many runs it averages (the legend counts the model's
  usable runs, not the method's).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import figs as F
from common import rank as K
from sycophancy.analysis import frame
from sycophancy.analysis.figs import CAP_NAME, HACK_NAME, PANELS, RUNS_DIR, TAG, evals, untrained

# figs.PANELS plus multihop accuracy without a hint, against the in-distribution hack.
PANELS = [*PANELS[:2], ("multihop", "adopt/wrong_train", "correct/none"), PANELS[2]]
CAP_NAME = {**CAP_NAME, "correct/none": "multihop accuracy, no hint"}
PER_PANEL = False

# Usable runs at each common anchor. A run is usable when, against its untrained model,
# both hack metrics (trained and held-out templates) and both arithmetic splits (trained
# and held-out tasks) rose over training with the 95% paired interval excluding zero, and
# no earlier checkpoint has significantly less trained-template hack without being
# significantly worse on either arithmetic split -- that checkpoint would beat any repair.
# OOD capability is reported, not required. Measured over qwen_mh9_s0-4 on 2026-10-22: at
# 100, s0 gains no capability and s3, s4 are dominated by their steps 40-80; at 40, s0 and
# s2 are still below their untrained arithmetic.
USABLE = {
    100: {"Qwen": ["qwen_mh9_s1", "qwen_mh9_s2"]},
    40: {"Qwen": ["qwen_mh9_s1", "qwen_mh9_s3", "qwen_mh9_s4"]},
}
# The mh14 testbed set at eval version 2, written out: a run leaves this list only by an
# edit, never by a check. `check` applies the rule above and prints what it finds.
USABLE[90] = {"Qwen": ["qwen_mh14_s1", "qwen_mh14_s2", "qwen_mh14_s3"],
              "Gemma": ["e2b_mh14_s1", "e2b_mh14_s2", "e2b_mh14_s3"]}
VERSION = {100: 1, 40: 1, 90: 2}
MODEL_NAME = {"Gemma": "Gemma 4 E2B", "Qwen": "Qwen3-4B"}
# Repair arms: run -> {method: directory holding eval-step<N>/ per snapshot}. None yet.
REPAIRS: dict[str, dict[str, Path]] = {}
REWIND = "rewind to an earlier checkpoint"
COLOUR = {REWIND: "0.3", "reverse": "tab:blue", "corrected-reward": "tab:green",
          "bc": "tab:orange"}
PANEL_TITLE = {"in distribution": "A. In-distribution evaluation",
               "held out": "B. Generalisation to held-out templates and tasks",
               "multihop": "C. Multihop capability (no hint)",
               "out of distribution": "D. Out-of-distribution evaluation"}


def run_evals(run, model, version=1):
    d = RUNS_DIR / run
    ev = evals(d, version)
    ev[0] = untrained(d, TAG[model], version)
    return ev


def check(r, ev, anchor):
    """Warnings for one run against the rule above. Printed, never acted on."""
    u, a = ev[0], ev[anchor]
    out = [f"no significant rise in {m}" for m in
           ("adopt/wrong_train", "adopt/wrong_heldout", "math/train", "math/heldout")
           if (c := frame.contrast(u, a, m))["effect"] - c["ci"] <= 0]
    for s, f in ev.items():
        if s in (0, anchor):
            continue
        h = frame.contrast(a, f, "adopt/wrong_train")
        worse = any((c := frame.contrast(a, f, m))["effect"] + c["ci"] < 0
                    for m in ("math/train", "math/heldout"))
        if h["effect"] + h["ci"] < 0 and not worse:
            out.append(f"checkpoint {s} has significantly less hack at no significant cost")
    return out


def load_runs(anchor):
    """{model: {run: evals}} for the listed runs that can be drawn.

    A listed run is skipped only when an evaluated checkpoint up to the anchor is
    missing (the rewind ladder must match across runs), and that is printed.
    """
    out = {}
    for model, runs in USABLE[anchor].items():
        for r in runs:
            ev = run_evals(r, model, VERSION[anchor])
            ckpts = sorted(int(p.name.split("-")[1]) for p in (RUNS_DIR / r).glob("checkpoint-*"))
            missing = [s for s in ckpts if s <= anchor and s % 30 == 0 and s not in ev]
            if missing:
                print(f"NOT DRAWN {r}: no complete eval at {missing}")
                continue
            for w in check(r, ev, anchor):
                print(f"WARNING {r}: {w}")
            out.setdefault(model, {})[r] = ev
    return out


def path(ev, frames, anchor, hack, cap):
    """x (pp from untrained), y (gain units from the anchor) and their CIs for `frames`."""
    u, a = ev[0], ev[anchor]
    gain = frame.contrast(u, a, cap if PER_PANEL else "math/train")["effect"]
    xs = frame.contrast(u, a, hack)["effect"] if PER_PANEL else .01
    h = [frame.contrast(u, f, hack) for f in frames]
    c = [frame.contrast(a, f, cap) for f in frames]
    return (np.array([v["effect"] for v in h]) / xs, np.array([v["effect"] for v in c]) / gain,
            np.array([v["ci"] for v in h]) / abs(xs), np.array([v["ci"] for v in c]) / gain)


def figure(out, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loaded = load_runs(anchor)
    usable = {m: list(v) for m, v in loaded.items()}
    models = [m for m in ("Qwen", "Gemma") if m in usable]
    fig, axes = plt.subplots(len(models), len(PANELS) + 1, squeeze=False,
                             figsize=(5.4 * len(PANELS) + 4, 6.6 * len(models)),
                             gridspec_kw=dict(width_ratios=[1] * len(PANELS) + [.55]))
    for i, model in enumerate(models):
        runs = usable[model]
        evs = loaded[model]
        steps = sorted((s for s in evs[runs[0]] if s <= anchor), reverse=True)
        assert all(sorted((s for s in e if s <= anchor), reverse=True) == steps
                   for e in evs.values()), "rewind steps differ between runs"
        for j, (title, hack, cap) in enumerate(PANELS):
            ax = axes[i, j]
            # OOD is evaluated at the anchor and the untrained model only, so its rewind
            # line has just those two ends; the other panels have every checkpoint.
            st = [s for s in steps if all(cap in evs[r][s] for r in runs)]
            if anchor not in st:
                # e.g. OOD while its re-measure is still running: nothing to draw from.
                ax.set_title(title, loc="left", fontweight="bold")
                ax.text(.5, .5, f"{cap}: not yet evaluated at step {anchor}\nin every run",
                        ha="center", va="center", transform=ax.transAxes, color="0.4")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            series = {REWIND: [path(evs[r], [evs[r][s] for s in st], anchor, hack, cap)
                               for r in runs]}
            labels = {REWIND: ["untrained" if s == 0 else str(s) for s in st]}
            for method in sorted({m for r in runs for m in REPAIRS.get(r, {})}):
                ps = []
                for r in runs:
                    rv = evals(REPAIRS[r][method]) if method in REPAIRS.get(r, {}) else {}
                    if rv:
                        ps.append(path(evs[r], [evs[r][anchor], *rv.values()], anchor, hack, cap))
                if ps:
                    n = min(len(p[0]) for p in ps)
                    series[method] = [tuple(v[:n] for v in p) for p in ps]
            ci_y, ci_x = [], []
            for method, ps in series.items():
                c = COLOUR[method]
                for x, y, xe, ye in ps:
                    ax.plot(x, y, "-", color=c, lw=.9, alpha=.25, zorder=2)
                    ax.plot(x[1:], y[1:], K.MODEL_MARK[model], color=c, ms=4, alpha=.35, zorder=2)
                    ci_x += list(xe[1:]); ci_y += list(ye[1:])
                mx = np.mean([p[0] for p in ps], 0)
                my = np.mean([p[1] for p in ps], 0)
                ax.plot(mx, my, "-", color=c, lw=2.4, zorder=5)
                ax.plot(mx[1:], my[1:], "o", color=c, ms=5.5, mec="k", mew=.6, zorder=6)
                ax.plot(mx[-1:], my[-1:], "o", color=c, ms=9, mec="k", mew=1.1, zorder=7)
                for lab, x, y in zip(labels.get(method, []), mx, my):
                    if lab != str(anchor):
                        ax.annotate(lab, (x, y), xytext=(5, -9), textcoords="offset points",
                                    fontsize=7.5, color="0.25")
            gap = np.mean([p[0][0] for p in series[REWIND]])
            ax.plot(gap, 0, "ks", ms=10, zorder=8)
            ax.axvline(0, color="k", ls=":", lw=1.2)
            ax.axhline(0, color="k", lw=.7)
            if cap == "math/train" or PER_PANEL:
                ax.axhline(-1, color="0.5", ls="--", lw=.9)
                ax.annotate("untrained capability", (0.99, -1), xycoords=("axes fraction", "data"),
                            xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4",
                            ha="right")
            K.scale_bar(ax, float(np.median(ci_y)), float(np.median(ci_x)), where=(.93, .08))
            kind = hack.split("/")[0]
            K.style(ax, f"{PANEL_TITLE[title]} — {MODEL_NAME[model]}",
                    (f"{HACK_NAME[kind]}: share of the installed gap left ({hack})\n"
                     "1: hacked anchor; 0: untrained rate; < 0: below untrained" if PER_PANEL else
                     f"{HACK_NAME[kind]} − untrained rate ({hack}, pp)\n"
                     "0: untrained rate; > 0: residual hack; < 0: below untrained"),
                    f"Δ {CAP_NAME[cap]} from the anchor,\n"
                    + ("÷ this run's RL gain on it" if PER_PANEL
                       else "in units of the RL gain on trained arithmetic"),
                    best="Optimal: x = 0 with y ≥ 0 (upper left)",
                    read="Faint: individual runs. Solid: mean over runs at matched checkpoints; "
                         "large marker: untrained model; black square: mean hacked anchor.")
        h = [plt.Line2D([], [], color=COLOUR[m], lw=2.4, marker="o", mec="k", ms=7,
                        label=f"{m}  ({len(usable[model])} runs)")
             for m in [REWIND, *sorted({m for r in usable[model] for m in REPAIRS.get(r, {})})]]
        h += [plt.Line2D([], [], ls="", label=""),
              plt.Line2D([], [], color="0.4", marker=K.MODEL_MARK[model], ls="-", lw=.9,
                         alpha=.4, label=f"individual run, {MODEL_NAME[model]}"),
              plt.Line2D([], [], color="k", marker="s", ls="", ms=8,
                         label=f"hacked anchor (step {anchor})")]
        axes[i, -1].axis("off")
        axes[i, -1].legend(handles=h, fontsize=9.5, loc="center", frameon=False,
                           title=", ".join(r.split("_")[-1] for r in usable[model]),
                           title_fontsize=8.5)
    fig.suptitle(f"Trade-off between sycophancy removal and capability, all usable runs, "
                 f"anchor at step {anchor}\n"
                 + ("Each panel is scaled by the run's own gaps on it: x by its installed hack, "
                    "y by its RL capability gain (anchor at (1, 0), untrained at (0, −1))"
                    if PER_PANEL else
                    "Capability change is normalised by each run's RL gain on trained "
                    "arithmetic (0: hacked anchor; −1: untrained capability on trained tasks)"),
                 fontsize=12)
    K.layout(fig)
    png = F.save_fig(fig, out, f"syco_tradeoff_a{anchor}" + ("_per_panel" if PER_PANEL else ""))
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, default=90, choices=sorted(VERSION))
    ap.add_argument("--per-panel", action="store_true",
                    help="scale each panel by the run's own gaps (module docstring)")
    ap.add_argument("--out", default="sycophancy/docs/figs")
    args = ap.parse_args()
    global PER_PANEL
    PER_PANEL = args.per_panel
    print(f"wrote {figure(args.out, args.anchor)}")


if __name__ == "__main__":
    main()
