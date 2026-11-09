"""The trade-off figure: every run's dose curve, and each method's mean curve.

    python -m creatures.analysis.tradeoff [--panels A|ABC] [--per-panel] [--out DIR]

`tradeoff.png` pools every drawn run in one row; `tradeoff_by_model.png` gives each model
its own row. Each run's curve is a faint line and each method's mean over runs a solid
one, with a marker at every evaluated dose. `--panels A` draws the training distribution
alone, which needs only the panel-A evals (`creatures/jobs/eval_one.sh`, PANELS=A).

x is the creature rate minus the run's untrained rate, in percentage points: untrained
is 0 in every run and the anchor sits at the run's installed gap. y is the accuracy
change against the anchor divided by the run's RL gain on the training distribution
(`common.rank.per_gain`), so rewinding to the untrained model costs exactly -1 on panel
A in every run. With `--per-panel` both axes are scaled by the run's own gaps on that
panel instead, putting every run's anchor at (1, 0) and the untrained model at (0, -1);
that is only meaningful while every drawn run's gap on every panel is well clear of noise.

A mean at matched dose blends runs that respond to the dose at different rates, so it is
not any one run's trajectory; the faint curves show the runs it averages. It is drawn
only where every run of the method is evaluated at every dose, so a method with some runs
evaluated for panel A alone has no mean on the other panels.
"""

import argparse

import numpy as np

from common import rank as K
from creatures.analysis import measures as M
from creatures.analysis.rank import REWIND, UPPER, colours, curves

# (title, slice, x label, y label)
PANELS = [("A. Training distribution", "id",
           "creature rate − untrained rate, trained tasks, training persona mix (pp)",
           "trained tasks, training persona mix: normalised Δ accuracy"),
          ("B. Held-out tasks", "ood",
           "creature rate − untrained rate, held-out tasks, training persona mix (pp)",
           "held-out tasks: normalised Δ accuracy"),
          ("C. Unseen persona", "per",
           "creature rate − untrained rate, unseen persona, all tasks (pp)",
           "unseen persona, all tasks: normalised Δ accuracy")]
PER_PANEL = False
MARK = {"Qwen": "o", "Gemma": "^"}


def path(df, k):
    """(doses, x, y) for one run's curve, the anchor prepended as dose -1."""
    g = df[f"gain_{k}" if PER_PANEL else "gain_id"].iloc[0]
    x = df[f"exc_{k}"].values * 100
    gap = df[f"gap_{k}"].iloc[0] * 100
    if PER_PANEL:
        x = x / gap
    return (np.concatenate([[-1], df.step.values]),
            np.concatenate([[1.0 if PER_PANEL else gap], x]),
            np.concatenate([[0.0], df[f"dA_{k}"].values / g]))


def mean_path(paths):
    """Mean x and y over runs, dose by dose; None unless every run has the same doses and
    a value at each, since averaging over whichever runs have one changes what it means."""
    if len({tuple(d) for d, _, _ in paths}) != 1:
        raise ValueError(f"runs have different dose schedules: "
                         f"{sorted({tuple(d) for d, _, _ in paths})}")
    xs, ys = np.array([x for _, x, _ in paths]), np.array([y for _, _, y in paths])
    if np.isnan(xs).any() or np.isnan(ys).any():
        return None
    return xs.mean(0), ys.mean(0)


def labels(xl, yl):
    if not PER_PANEL:
        return xl + "\n0: untrained rate; > 0: residual hack; < 0: below untrained", yl
    what = xl.replace("creature rate − untrained rate, ", "").replace(" (pp)", "")
    return (f"creature-rate gap remaining, {what}\n1: hacked anchor; 0: untrained",
            yl.replace("normalised Δ accuracy", "Δ accuracy ÷ this run's gain here"))


def draw_row(axes, cs, colour, title_prefix=""):
    """One row of panels for the curves `cs`. Returns {method: {model: runs}}."""
    n_runs = {}
    for ax, (title, k, xl, yl) in zip(axes, PANELS):
        drawn = False
        for m, c in colour.items():
            keys = sorted(key for key in cs if key[2] == m)
            if not keys:
                continue
            n_runs[m] = {md: sum(key[0] == md for key in keys) for md in M.MODELS}
            ps = [path(cs[key], k) for key in keys]
            for key, (_, x, y) in zip(keys, ps):
                ax.plot(x, y, "-", color=c, lw=.9, alpha=.25, zorder=2)
                ax.plot(x[-1:], y[-1:], MARK[key[0]], color=c, ms=4.5, alpha=.45, zorder=2)
                drawn |= bool(np.isfinite(x[1:]).any())
            mean = mean_path(ps)
            if mean is None:
                continue
            mx, my = mean
            ax.plot(mx, my, "-", color=c, lw=2.4, zorder=5)
            ax.plot(mx[1:], my[1:], "o", color=c, ms=5.5, mec="k", mew=.6, zorder=6)
            ax.plot(mx[-1:], my[-1:], "o", color=c, ms=9, mec="k", mew=1.1, zorder=7)
        if not drawn:
            ax.text(.5, .5, "not evaluated yet", ha="center", va="center",
                    transform=ax.transAxes, color="0.4")
        ax.plot(np.nanmean([path(v, k)[1][0] for v in cs.values()]), 0, "ks", ms=10,
                zorder=8)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        ax.axhline(0, color="k", lw=.7)
        if k == "id" or PER_PANEL:
            ax.axhline(-1, color="0.5", ls="--", lw=.9)
            ax.annotate("untrained capability", (0.01, -1), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4")
        if k == "id":
            ax.axhline(-.1, color="tab:red", lw=.8, alpha=.5)
            ax.annotate("90% of RL gain retained", (0.99, -.1),
                        xycoords=("axes fraction", "data"), xytext=(0, -9),
                        textcoords="offset points", fontsize=7, color="tab:red", alpha=.8,
                        ha="right")
        ci = np.nanmedian([np.nanmedian(v[f"dA_{k}_ci"]
                                        / v[f"gain_{k}" if PER_PANEL else "gain_id"])
                           for v in cs.values()])
        if np.isfinite(ci):
            K.scale_bar(ax, ci, where=(.93, .08), label=UPPER)
        K.style(ax, title_prefix + title, *labels(xl, yl),
                best="Optimal: x = 0 with maximal y",
                read="Faint: individual runs, end marked ○ Qwen, △ Gemma. Solid: mean "
                     "over runs at each evaluated dose; large marker: largest dose; "
                     "black square: mean hacked anchor.")
    return n_runs


def legend(ax, colour, n_runs, cs):
    import matplotlib.pyplot as plt
    h = [plt.Line2D([], [], color=c, lw=2.4, marker="o", mec="k", ms=7,
                    label=m + "  (" + ", ".join(f"{n} {md}" for md, n in n_runs[m].items()
                                               if n) + ")")
         for m, c in colour.items() if m in n_runs]
    h += [plt.Line2D([], [], ls="", label="")]
    h += [plt.Line2D([], [], color="0.4", marker=MARK[md], ls="-", lw=.9, alpha=.45,
                     label=f"individual run, {md}") for md in M.MODELS]
    h += [plt.Line2D([], [], color="k", marker="s", ls="", ms=8, label="hacked anchor")]
    steps = sorted({tuple(v.step) for k, v in cs.items() if k[2] == REWIND})
    ax.axis("off")
    ax.legend(handles=h, fontsize=9.5, loc="center left", frameon=False,
              title="Rewind dose: checkpoint " + ", ".join(
                  "untrained" if s == 0 else str(s) for s in (steps[0] if steps else ())),
              title_fontsize=8.5)


def norm_note():
    if PER_PANEL:
        return ("Each panel is scaled by the run's own gaps there (anchor at (1, 0), "
                "untrained at (0, −1))")
    return ("Capability change is normalised by each run's RL gain on the training "
            "distribution (0: hacked anchor; −1: untrained)")


def figure(out, cs, panels="ABC"):
    """`tradeoff.png` (all runs) and `tradeoff_by_model.png`; returns the first's path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global PANELS
    saved, PANELS = PANELS, [p for p in PANELS if p[1] in M.PANEL_SLICES[panels]]
    try:
        colour = colours(cs)
        n = len(PANELS)
        suffix = "" if panels == "ABC" else f"_{panels}"
        fig, axes = plt.subplots(1, n + 1, figsize=(5.4 * n + 4, 7.4), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1] * n + [.75]))
        n_runs = draw_row(axes[0, :n], cs, colour)
        legend(axes[0, n], colour, n_runs, cs)
        fig.suptitle("Dose-response of each repair method, mean over all runs\n"
                     + norm_note(), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .9), w_pad=3)
        png = M.save_fig(fig, out, "tradeoff" + suffix)
        plt.close(fig)

        fig, axes = plt.subplots(len(M.MODELS), n + 1, squeeze=False,
                                 figsize=(5.4 * n + 4, 6.6 * len(M.MODELS)),
                                 gridspec_kw=dict(width_ratios=[1] * n + [.75]))
        for i, model in enumerate(M.MODELS):
            mine = {k: v for k, v in cs.items() if k[0] == model}
            n_runs = draw_row(axes[i, :n], mine, colour, f"{model}: ")
            legend(axes[i, n], colour, n_runs, mine)
        fig.suptitle("Dose-response of each repair method, one row per model\n"
                     + norm_note(), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .93), h_pad=6, w_pad=3)
        M.save_fig(fig, out, "tradeoff_by_model" + suffix)
        plt.close(fig)
        return png
    finally:
        PANELS = saved


def main():
    global PER_PANEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs/cre17")
    ap.add_argument("--panels", default="ABC", choices=sorted(M.PANEL_SLICES))
    ap.add_argument("--per-panel", action="store_true")
    args = ap.parse_args()
    PER_PANEL = args.per_panel
    print(f"wrote {figure(args.out + ('_per_panel' if PER_PANEL else ''), curves(), args.panels)}")


if __name__ == "__main__":
    main()
