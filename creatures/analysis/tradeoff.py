"""The trade-off figure: every run's dose-response curve, with one mean curve per method.

    python -m creatures.analysis.tradeoff [--out DIR]

It replaced `main6_abs.png` (`eval_figs.figure_panels`) on 2026-10-06. That figure drew
one focus seed per model, because six runs of seven methods with per-point error bars
were unreadable, and so left four of the six runs out of the figure a reader looks at
first. Here all runs share each panel: each run's curve is a faint line, and each method
has one solid curve, the mean over runs at matched doses.

The axes are chosen so that runs can be averaged at all.

x is the creature rate minus the run's untrained rate, in percentage points, reversed so
that more removal is to the right. Untrained is 0 in every run, and the anchor sits at the
run's installed gap. Dividing by that gap (R) would also align the anchor, but the OOD
persona gap is 4 pp on average and -1 pp on Gemma s2, so R there is noise.

y is the accuracy change divided by the run's RL gain on trained tasks, as on `rank.png`
(`common.rank.per_gain`): rewinding all the way costs exactly -1 on the trained
distribution in every run, where in raw accuracy it cost anywhere from 0.12 to 0.41.

Doses are matched by their nominal value: every repair arm, retrain and continue run
uses the same schedule in every run. Rewind does not: the anchor is step 40 on Qwen and
50 on Gemma. It is therefore expressed as the fraction of training undone,
1 - checkpoint / anchor, and each run is linearly interpolated onto quarters of that.
A mean at matched nominal dose blends runs that respond to the dose at different rates --
reverse first reaches R = 1 at dose 16 on Gemma and 24-32 on Qwen -- so the mean curve is
not any one run's trajectory; its ends are, and the faint curves show the rest.

Across-run intervals are left to `rank.png`, which reports them at the operating point.
On a curve they would add two bars to every mean point. The faint per-run curves show the
spread instead.

The curves and their columns are `rank.curves()`'s, so these numbers are the ones the
rank figure reads.
"""

import argparse

import numpy as np

from common import rank as K
from creatures.analysis import eval_figs as E
from creatures.analysis.rank import COLOUR, REWIND, RETRAIN, SUPPRESS, UNPLOTTED, UPPER, curves

# (title, excess column, gap column, capability column, x label, y label)
PANELS = [("A. Trained distribution", "exc_id", "gap_id", "dA_tr",
           "creature rate − untrained rate, trained tasks, rewarded persona (pp)",
           "trained tasks: normalised Δ accuracy"),
          ("B. Held-out tasks", "exc_ood", "gap_ood", "dA",
           "creature rate − untrained rate, held-out tasks, rewarded persona (pp)",
           "held-out tasks: normalised Δ accuracy"),
          ("C. Unrewarded persona", "exc_per", "gap_per", "dA_all",
           "creature rate − untrained rate, OOD persona, all tasks (pp)",
           "all tasks: normalised Δ accuracy")]
REWIND_GRID = np.array([0, .25, .5, .75, 1.0])


def path(key, df, exc, gap, cap):
    """(dose, x in pp, y in gain units) for one run, anchor prepended where it applies."""
    model = key[0]
    g = df.gain_tr.iloc[0]
    x, y = df[exc].values * 100, df[cap].values / g
    dose = df.step.values.astype(float)
    if df.from_anchor.iloc[0]:
        x = np.concatenate([[df[gap].iloc[0] * 100], x])
        y = np.concatenate([[0.0], y])
        dose = np.concatenate([[0.0], dose])
    if key[2] == REWIND:
        frac = 1 - dose / E.ANCHOR[model]
        frac[0] = 0.0
        return REWIND_GRID, np.interp(REWIND_GRID, frac, x), np.interp(REWIND_GRID, frac, y)
    return dose, x, y


def mean_path(paths):
    """Mean x and y over runs at the doses every run has."""
    common = sorted(set.intersection(*[set(d) for d, _, _ in paths]))
    xs = np.array([[x[list(d).index(c)] for c in common] for d, x, _ in paths])
    ys = np.array([[y[list(d).index(c)] for c in common] for d, _, y in paths])
    return np.array(common), xs.mean(0), ys.mean(0)


def figure(out, cs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = {k: v for k, v in cs.items() if k[2] not in UNPLOTTED}
    colour = {m: c for m, c in COLOUR.items() if m not in UNPLOTTED}
    runs = sorted({(k[0], k[1]) for k in cs})
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.6),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, .55]))
    n_runs = {}
    for ax, (title, exc, gap, cap, xl, yl) in zip(axes, PANELS):
        for m, c in colour.items():
            keys = sorted(k for k in cs if k[2] == m)
            if not keys:
                continue
            n_runs[m] = len(keys)
            ps = [path(k, cs[k], exc, gap, cap) for k in keys]
            for k, (_, x, y) in zip(keys, ps):
                mk = K.MODEL_MARK.get(k[0], "o")
                # no line for the clause: anchor to clause-on is not a path a dose can take
                if m != SUPPRESS:
                    ax.plot(x, y, "-", color=c, lw=.9, alpha=.22, zorder=2)
                ax.plot(x[-1:], y[-1:], mk, color=c, ms=4 if m != SUPPRESS else 5,
                        alpha=.35, zorder=2)
            _, mx, my = mean_path(ps)
            if m == SUPPRESS:
                # its path is (anchor, clause-on point); only the latter is a measurement
                ax.plot(mx[-1:], my[-1:], "D", color=c, ms=10, mec="k", mew=1.1, zorder=6)
                continue
            ax.plot(mx, my, "-", color=c, lw=2.4, zorder=5)
            # every dose but the anchor, which has its own marker; retraining has none
            first = 0 if m == RETRAIN else 1
            ax.plot(mx[first:], my[first:], "o", color=c, ms=5.5, mec="k", mew=.6, zorder=6)
            # the largest dose, so a reader can tell where each curve ends
            ax.plot(mx[-1:], my[-1:], "o", color=c, ms=9, mec="k", mew=1.1, zorder=7)
        mean_gap = np.mean([cs[k][gap].iloc[0] for k in cs]) * 100
        ax.plot(mean_gap, 0, "ks", ms=10, zorder=8)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        ax.axhline(0, color="k", lw=.7)
        ax.invert_xaxis()
        ci = np.nanmedian([np.nanmedian(v[f"{cap}_ci"] / v.gain_tr) for v in cs.values()])
        K.scale_bar(ax, ci, where=(.93, .08), label=UPPER)
        K.style(ax, title, xl + "\naxis reversed; 0: untrained rate; < 0: over-erasure",
                yl, best="Optimal: x = 0 with maximal y",
                read="Faint: individual runs. Solid: mean over runs at matched dose; "
                     "large marker: largest dose; black square: mean hacked anchor.")
    axes[0].axhline(-1, color="0.5", ls="--", lw=.9)
    axes[0].annotate("untrained capability", (0.01, -1), xycoords=("axes fraction", "data"),
                     xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4")
    axes[0].axhline(-.1, color="tab:red", lw=.8, alpha=.5)
    axes[0].annotate("90% of RL gain retained", (0.99, -.1),
                     xycoords=("axes fraction", "data"), xytext=(0, -9),
                     textcoords="offset points", fontsize=7, color="tab:red", alpha=.8,
                     ha="right")

    h = [plt.Line2D([], [], color=c, lw=2.4, marker="D" if m == SUPPRESS else "o",
                    ls="" if m == SUPPRESS else "-", mec="k", ms=7,
                    label=f"{m}  ({n_runs[m]} runs)")
         for m, c in colour.items() if m in n_runs]
    h += [plt.Line2D([], [], ls="", label="")]
    h += [plt.Line2D([], [], color="0.4", marker=K.MODEL_MARK[m], ls="-", lw=.9, alpha=.4,
                     label=f"individual run, {m}") for m in ["Qwen", "Gemma"]]
    h += [plt.Line2D([], [], color="k", marker="s", ls="", ms=8, label="hacked anchor")]
    axes[3].axis("off")
    axes[3].legend(handles=h, fontsize=9.5, loc="center", frameon=False,
                   title="Rewind dose: fraction of training undone,\n"
                         "interpolated to quarters", title_fontsize=8.5)
    fig.suptitle("Dose-response of each repair method, all runs ("
                 + ", ".join(f"{m} {s}" for m, s in runs) + ")\n"
                 "Capability change is normalised by each run's RL gain on trained tasks "
                 "(0: hacked anchor; −1: untrained capability on trained tasks)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .9), w_pad=3)
    png = E.save_fig(fig, out, "tradeoff")
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs")
    args = ap.parse_args()
    print(f"wrote {figure(args.out, curves())}")


if __name__ == "__main__":
    main()
