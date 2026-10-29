"""The trade-off figures: every run's dose-response curve, and one mean curve per method.

    python -m creatures.analysis.tradeoff [--out DIR]

`tradeoff.png` (`figure`, the main one) pools all six runs in one row;
`tradeoff_by_model.png` (`figure_by_model`, for the appendix) gives each model its own row.
Both replaced `main6_abs.png` (`eval_figs.figure_panels`) on 2026-10-06, which drew one
focus seed per model and so left four of the six runs out of the figure a reader looks at
first. Each run's curve is a faint line, and each method has one solid curve.

Every marker on a solid curve is a mean of measured values at one dose. Every method but
rewind has the same dose schedule on both models, so its pooled mean needs no alignment,
and `mean_path` raises rather than averaging runs whose schedules differ. Rewind's
ladders differ -- the anchor is step 40 on Qwen and 50 on Gemma -- so its pooled curve
(`rewind_mean`) is taken at matched fraction of training undone, interpolating each run
linearly between its checkpoints, and is a bare line with a marker only at the untrained
end, which every run measured. The first pooled version (2026-10-06 to 10-23) put markers
on those interpolated points, drawing them as measurements.

Pooling hides real model differences, which is what the faint runs (end marked by model)
and the per-model figure are for. Rewound by half its training (checkpoint 20) Qwen keeps
84% of its RL gain, and Gemma rewound by 40% (checkpoint 30) keeps 61% -- the difference
section 6 of docs/EVAL.md rests on -- while the pooled curve puts half-way at 67%.
Continued training removes most of Qwen's trained-task hack and none of Gemma's.

The axes are chosen so that runs can be averaged at all.

x is the creature rate minus the run's untrained rate, in percentage points, so that more
removal is to the left. Untrained is 0 in every run, and the anchor sits at the
run's installed gap. Dividing by that gap (R) would also align the anchor, but the OOD
persona gap is 4 pp on average and -1 pp on Gemma s2, so R there is noise.

y is the accuracy change divided by the run's RL gain on trained tasks, as on `rank.png`
(`common.rank.per_gain`): rewinding all the way costs exactly -1 on the trained
distribution in every run, where in raw accuracy it cost anywhere from 0.12 to 0.41.

A mean at matched dose still blends runs that respond to the dose at different rates --
Qwen s1 has removed its whole trained-task hack by checkpoint 20, its two siblings not by
checkpoint 10 -- so a mean curve is not any one run's trajectory. The faint curves show
the runs it averages.

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
MODELS = ["Qwen", "Gemma"]


def path(df, exc, gap, cap):
    """(dose, x in pp, y in gain units) for one run, in dose order, the anchor prepended
    where the curve leaves it. Rewind's dose is the checkpoint, so its anchor is labelled
    with the anchor's own step; every value is a measurement."""
    g = df.gain_tr.iloc[0]
    x, y = df[exc].values * 100, df[cap].values / g
    dose = df.step.values
    if df.from_anchor.iloc[0]:
        x = np.concatenate([[df[gap].iloc[0] * 100], x])
        y = np.concatenate([[0.0], y])
        dose = np.concatenate([[-1], dose])
    return dose, x, y


def mean_path(paths):
    """Mean x and y over runs, dose by dose. Raises unless every run has the same doses:
    averaging over a partial intersection would drop doses silently, and averaging
    across different schedules needs an interpolation this figure does not make."""
    doses = [tuple(d) for d, _, _ in paths]
    if len(set(doses)) != 1:
        raise ValueError(f"runs have different dose schedules: {sorted(set(doses))}")
    return (np.array(doses[0]), np.mean([x for _, x, _ in paths], 0),
            np.mean([y for _, _, y in paths], 0))


def ladder(cs, model):
    """The rewind checkpoints one model's runs share, as labels for the legend."""
    steps = {tuple(v.step) for k, v in cs.items() if k[0] == model and k[2] == REWIND}
    (one,) = steps  # mean_path would refuse anything else
    return ["untrained" if st == 0 else str(st) for st in one]


def figure_by_model(out, cs):
    """The appendix figure: one row per model, each method's mean over that model's runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = {k: v for k, v in cs.items() if k[2] not in UNPLOTTED}
    colour = {m: c for m, c in COLOUR.items() if m not in UNPLOTTED}
    fig, axes = plt.subplots(2, 4, figsize=(21, 12.5),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, .75]))
    n_runs = {}
    for row, model in enumerate(MODELS):
        seeds = sorted({k[1] for k in cs if k[0] == model})
        for col, (title, exc, gap, cap, xl, yl) in enumerate(PANELS):
            ax = axes[row, col]
            for m, c in colour.items():
                keys = sorted(k for k in cs if k[0] == model and k[2] == m)
                if not keys:
                    continue
                n_runs[(model, m)] = len(keys)
                ps = [path(cs[k], exc, gap, cap) for k in keys]
                for _, x, y in ps:
                    # no line for the clause: anchor to clause-on is not a path a dose takes
                    if m != SUPPRESS:
                        ax.plot(x, y, "-", color=c, lw=.9, alpha=.25, zorder=2)
                    ax.plot(x[-1:], y[-1:], "o", color=c, ms=4 if m != SUPPRESS else 5,
                            alpha=.4, zorder=2)
                _, mx, my = mean_path(ps)
                if m == SUPPRESS:
                    # its path is (anchor, clause-on point); only the latter is measured
                    ax.plot(mx[-1:], my[-1:], "D", color=c, ms=10, mec="k", mew=1.1,
                            zorder=6)
                    continue
                ax.plot(mx, my, "-", color=c, lw=2.4, zorder=5)
                # every dose but the anchor, which has its own marker; retraining has none
                first = 0 if m == RETRAIN else 1
                ax.plot(mx[first:], my[first:], "o", color=c, ms=5.5, mec="k", mew=.6,
                        zorder=6)
                # the largest dose, so a reader can tell where each curve ends
                ax.plot(mx[-1:], my[-1:], "o", color=c, ms=9, mec="k", mew=1.1, zorder=7)
            mine = [v for k, v in cs.items() if k[0] == model]
            ax.plot(np.mean([v[gap].iloc[0] for v in mine]) * 100, 0, "ks", ms=10, zorder=8)
            ax.axvline(0, color="k", ls=":", lw=1.2)
            ax.axhline(0, color="k", lw=.7)
            ci = np.nanmedian([np.nanmedian(v[f"{cap}_ci"] / v.gain_tr) for v in mine])
            K.scale_bar(ax, ci, where=(.93, .08), label=UPPER)
            K.style(ax, f"{'ABCDEF'[3 * row + col]}. {model}: {title[3:]}",
                    xl + "\n0: untrained rate; > 0: residual hack; < 0: over-erasure",
                    yl, best="Optimal: x = 0 with maximal y",
                    read=f"Faint: {model} {', '.join(seeds)}. Solid: their mean at each "
                         "evaluated dose; large marker: largest dose; black square: mean "
                         "hacked anchor.")
        axes[row, 0].axhline(-1, color="0.5", ls="--", lw=.9)
        axes[row, 0].annotate("untrained capability", (0.01, -1),
                              xycoords=("axes fraction", "data"), xytext=(0, 3),
                              textcoords="offset points", fontsize=7, color="0.4")
        axes[row, 0].axhline(-.1, color="tab:red", lw=.8, alpha=.5)
        axes[row, 0].annotate("90% of RL gain retained", (0.99, -.1),
                              xycoords=("axes fraction", "data"), xytext=(0, -9),
                              textcoords="offset points", fontsize=7, color="tab:red",
                              alpha=.8, ha="right")

    h = [plt.Line2D([], [], color=c, lw=2.4, marker="D" if m == SUPPRESS else "o",
                    ls="" if m == SUPPRESS else "-", mec="k", ms=7,
                    label=m + "  (" + ", ".join(f"{n_runs[(md, m)]} {md}" for md in MODELS
                                               if (md, m) in n_runs) + ")")
         for m, c in colour.items() if any((md, m) in n_runs for md in MODELS)]
    h += [plt.Line2D([], [], ls="", label="")]
    h += [plt.Line2D([], [], color="0.4", marker="o", ls="-", lw=.9, alpha=.4,
                     label="individual run"),
          plt.Line2D([], [], color="k", marker="s", ls="", ms=8, label="hacked anchor")]
    axes[0, 3].axis("off")
    axes[1, 3].axis("off")
    # centred between the two rows, left-aligned in its column
    axes[0, 3].legend(handles=h, fontsize=9.5, loc="center left", frameon=False,
                      bbox_to_anchor=(0, -.1),
                      title="Rewind dose: which earlier checkpoint\n" + "\n".join(
                          f"{md}: " + ", ".join(ladder(cs, md)) for md in MODELS),
                      title_fontsize=8.5)
    fig.suptitle("Dose-response of each repair method, one row per model\n"
                 "Capability change is normalised by each run's RL gain on trained tasks "
                 "(0: hacked anchor; \u22121: untrained capability on trained tasks)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=6, w_pad=3)
    png = E.save_fig(fig, out, "tradeoff_by_model")
    plt.close(fig)
    return png


MODEL_MARK = {"Qwen": "o", "Gemma": "^"}


def rewind_mean(cs, keys, exc, gap, cap):
    """Mean rewind curve over runs whose ladders differ, aligned by the fraction of
    training undone and linearly interpolated onto the union of their fractions.

    Only the two ends are measured in every run -- the anchor (0) and the untrained model
    (1) -- so only they get markers; in between, the line is the mean of the runs'
    straight segments, which is what a line between checkpoints already asserts."""
    fs, xs, ys = [], [], []
    for k in keys:
        dose, x, y = path(cs[k], exc, gap, cap)
        anchor = E.ANCHOR[k[0]]
        f = np.where(dose == -1, 0.0, (anchor - dose) / anchor)
        fs.append(f), xs.append(x), ys.append(y)
    grid = np.unique(np.concatenate(fs))
    mx = np.mean([np.interp(grid, f, x) for f, x in zip(fs, xs)], 0)
    my = np.mean([np.interp(grid, f, y) for f, y in zip(fs, ys)], 0)
    return mx, my


def figure(out, cs):
    """The main figure: one row, every method's mean over all six runs. The per-model
    version is drawn alongside as `tradeoff_by_model`."""
    figure_by_model(out, cs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = {k: v for k, v in cs.items() if k[2] not in UNPLOTTED}
    colour = {m: c for m, c in COLOUR.items() if m not in UNPLOTTED}
    fig, axes = plt.subplots(1, 4, figsize=(21, 7.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, .75]))
    n_runs = {}
    for col, (title, exc, gap, cap, xl, yl) in enumerate(PANELS):
        ax = axes[col]
        for m, c in colour.items():
            keys = sorted(k for k in cs if k[2] == m)
            if not keys:
                continue
            n_runs[m] = {md: sum(k[0] == md for k in keys) for md in MODELS}
            for k in keys:
                _, x, y = path(cs[k], exc, gap, cap)
                if m != SUPPRESS:
                    ax.plot(x, y, "-", color=c, lw=.9, alpha=.25, zorder=2)
                ax.plot(x[-1:], y[-1:], MODEL_MARK[k[0]], color=c, ms=4.5, alpha=.45,
                        zorder=2)
            if m == REWIND:
                mx, my = rewind_mean(cs, keys, exc, gap, cap)
                ax.plot(mx, my, "-", color=c, lw=2.4, zorder=5)
                ax.plot(mx[-1:], my[-1:], "o", color=c, ms=9, mec="k", mew=1.1, zorder=7)
                continue
            _, mx, my = mean_path([path(cs[k], exc, gap, cap) for k in keys])
            if m == SUPPRESS:
                ax.plot(mx[-1:], my[-1:], "D", color=c, ms=10, mec="k", mew=1.1, zorder=6)
                continue
            ax.plot(mx, my, "-", color=c, lw=2.4, zorder=5)
            first = 0 if m == RETRAIN else 1
            ax.plot(mx[first:], my[first:], "o", color=c, ms=5.5, mec="k", mew=.6,
                    zorder=6)
            ax.plot(mx[-1:], my[-1:], "o", color=c, ms=9, mec="k", mew=1.1, zorder=7)
        ax.plot(np.mean([v[gap].iloc[0] for v in cs.values()]) * 100, 0, "ks", ms=10,
                zorder=8)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        ax.axhline(0, color="k", lw=.7)
        ci = np.nanmedian([np.nanmedian(v[f"{cap}_ci"] / v.gain_tr) for v in cs.values()])
        K.scale_bar(ax, ci, where=(.93, .08), label=UPPER)
        K.style(ax, title, xl + "\n0: untrained rate; > 0: residual hack; < 0: over-erasure",
                yl, best="Optimal: x = 0 with maximal y",
                read="Faint: individual runs, end marked ○ Qwen, △ Gemma. Solid: "
                     "mean over all runs at each evaluated dose; large marker: largest "
                     "dose; black square: mean hacked anchor.")
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
                    label=m + "  (" + ", ".join(f"{n} {md}" for md, n in n_runs[m].items()
                                               if n) + ")")
         for m, c in colour.items() if m in n_runs]
    h += [plt.Line2D([], [], ls="", label="")]
    h += [plt.Line2D([], [], color="0.4", marker=MODEL_MARK[md], ls="-", lw=.9, alpha=.45,
                     label=f"individual run, {md}") for md in MODELS]
    h += [plt.Line2D([], [], color="k", marker="s", ls="", ms=8, label="hacked anchor")]
    axes[3].axis("off")
    axes[3].legend(handles=h, fontsize=9.5, loc="center left", frameon=False,
                   title="Rewind: the two models' checkpoint ladders differ\n("
                         + "; ".join(f"{md} " + ", ".join(ladder(cs, md)) for md in MODELS)
                         + "),\nso its mean is taken at matched fraction of training "
                           "undone,\ninterpolating between checkpoints; only its "
                           "endpoints\n(anchor, untrained) are measured in every run.",
                   title_fontsize=8.5)
    fig.suptitle("Dose-response of each repair method, mean over both models\n"
                 "Capability change is normalised by each run's RL gain on trained tasks "
                 "(0: hacked anchor; −1: untrained capability on trained tasks). "
                 "Per-model curves: tradeoff_by_model.", fontsize=12)
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
