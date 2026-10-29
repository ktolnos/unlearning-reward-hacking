"""The pooled trade-off figure for the multihop sycophancy runs: every usable seed at once.

    python -m sycophancy.analysis.tradeoff [--anchor 100] [--out DIR]

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

Repair arms, once they exist for these runs, go in REPAIRS and are drawn the same way:
faint per run, solid mean at matched snapshot steps.

Known issues (2026-10-29), to fix before this draws a paper figure:
- USABLE and the figures it drew (syco_tradeoff_a100, _a40) are the mh9 runs at eval
  version 1. The recipe is now mh14, anchored at step 120 in every run, and its evals are
  being rerun at version 2. Selecting installs on install quality is sound so long as it
  is done before any repair is run; the list should still be computed from the rule
  above rather than written out by hand.
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
MODEL_NAME = {"Gemma": "Gemma 4 E2B", "Qwen": "Qwen3-4B"}
# Repair arms: run -> {method: directory holding eval-step<N>/ per snapshot}. None yet.
REPAIRS: dict[str, dict[str, Path]] = {}
REWIND = "rewind to an earlier checkpoint"
COLOUR = {REWIND: "0.3", "reverse": "tab:blue", "corrected-reward": "tab:green",
          "bc": "tab:orange"}
PANEL_TITLE = {"in distribution": "A. In-distribution evaluation",
               "held out": "B. Generalisation to held-out templates and tasks",
               "out of distribution": "C. Out-of-distribution evaluation"}


def run_evals(run, model):
    d = RUNS_DIR / run
    ev = evals(d)
    ev[0] = untrained(d, TAG[model])
    return ev


def path(ev, frames, anchor, hack, cap):
    """x (pp from untrained), y (gain units from the anchor) and their CIs for `frames`."""
    u, a = ev[0], ev[anchor]
    gain = frame.contrast(u, a, "math/train")["effect"]
    h = [frame.contrast(u, f, hack) for f in frames]
    c = [frame.contrast(a, f, cap) for f in frames]
    return (np.array([v["effect"] for v in h]) * 100, np.array([v["effect"] for v in c]) / gain,
            np.array([v["ci"] for v in h]) * 100, np.array([v["ci"] for v in c]) / gain)


def figure(out, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    usable = USABLE[anchor]
    models = list(usable)
    fig, axes = plt.subplots(len(models), len(PANELS) + 1, squeeze=False,
                             figsize=(21, 6.6 * len(models)),
                             gridspec_kw=dict(width_ratios=[1] * len(PANELS) + [.55]))
    for i, model in enumerate(models):
        runs = usable[model]
        evs = {r: run_evals(r, model) for r in runs}
        steps = sorted((s for s in evs[runs[0]] if s <= anchor), reverse=True)
        assert all(sorted((s for s in e if s <= anchor), reverse=True) == steps
                   for e in evs.values()), "rewind steps differ between runs"
        for j, (title, hack, cap) in enumerate(PANELS):
            ax = axes[i, j]
            series = {REWIND: [path(evs[r], [evs[r][s] for s in steps], anchor, hack, cap)
                               for r in runs]}
            labels = {REWIND: ["untrained" if s == 0 else str(s) for s in steps]}
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
            if cap == "math/train":
                ax.axhline(-1, color="0.5", ls="--", lw=.9)
                ax.annotate("untrained capability", (0.99, -1), xycoords=("axes fraction", "data"),
                            xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4",
                            ha="right")
            K.scale_bar(ax, float(np.median(ci_y)), float(np.median(ci_x)), where=(.93, .08))
            kind = hack.split("/")[0]
            K.style(ax, f"{PANEL_TITLE[title]} — {MODEL_NAME[model]}",
                    f"{HACK_NAME[kind]} − untrained rate ({hack}, pp)\n"
                    "0: untrained rate; > 0: residual hack; < 0: below untrained",
                    f"Δ {CAP_NAME[cap]} from the anchor,\nin units of the RL gain on trained arithmetic",
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
                 f"anchor at step {anchor}\nCapability change is normalised by each run's RL "
                 "gain on trained arithmetic (0: hacked anchor; −1: untrained capability on "
                 "trained tasks)", fontsize=12)
    K.layout(fig)
    png = F.save_fig(fig, out, f"syco_tradeoff_a{anchor}")
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, default=100, choices=sorted(USABLE))
    ap.add_argument("--out", default="sycophancy/docs/figs")
    args = ap.parse_args()
    print(f"wrote {figure(args.out, args.anchor)}")


if __name__ == "__main__":
    main()
