"""Ranking repair methods at a matched operating point: the experiment-independent part.

A curve is a DataFrame in dose order -- more replay for an arm, an earlier checkpoint for
a rewind -- with at least `R_id` (fraction of the installed hack removed where the reward
error applied), the capability columns being read, and `from_anchor`. An optional `point`
column marks an intervention with no dose to tune (a prompt clause is on or off). Each
experiment builds its curves (creatures/analysis/rank.py, sycophancy/analysis/rank.py)
and reads them through these, so the interpolation and the drawing are defined once.

The drawing, from `MODEL_MARK` down, is one solid marker per method -- the mean over runs
with a 95% t interval -- and every run behind it as a faint dot. It replaced v1's
`region` and `scatter` on 2026-10-06, which drew every run at full weight with its own
error bars and a text label plus a shaded t box per method: ~40 labelled markers and seven
overlapping boxes per panel, where the ranking a reader wants was the hardest thing to
see. The typical within-run interval is drawn once per panel (`scale_bar`), because it is
about the same for every run and every method.

Capability is drawn in units of the run's own RL gain on trained tasks (`per_gain`), not
raw accuracy. That gain varies 3.4x across the creature runs (0.12 on Gemma s1, 0.41 on
Qwen s0), and it is what rewinding gives up: a rewind that reaches R = 1 only at the
untrained model costs exactly -gain_tr, so in raw dA the rewind spread was almost entirely
the spread in how much each run had learned. Every capability column -- trained, held-out,
OOD -- is divided by the same trained-task gain rather than its own slice's gain: the
held-out gains are small enough to be noisy denominators (0.042 on Gemma s1 against a
0.022 sampling interval; the sycophancy OOD gain is ~0 on Gemma), and one denominator
keeps "equal cost on trained and held-out" on the diagonal. Hack rates stay in percentage
points off untrained: the persona and held-out-template gaps are small or even negative on
some runs, so dividing by them is noise.
"""

import textwrap

import numpy as np
from scipy import stats


def at_target(df, col, origin=0.0):
    """Interpolate `col` to R_id = 1 inside the first consecutive pair that crosses it.

    The frame is in sequence order -- more replay steps for an arm, an earlier
    checkpoint for a rewind -- with the anchor prepended, so consecutive rows are states
    you can actually move between. Interpolating anywhere else invents a path.

    Sorting the whole curve by R and interpolating across that went wrong wherever R is
    not monotone in the sequence, in two ways. Where R crosses the target more than
    once, the R-sort picks whichever crossing it happens to bracket rather than the
    first one reached: Qwen seed 1 installs the hack between steps 24 and 31, so walking
    back from its anchor R goes 0.18 at checkpoint 30, 1.03 at 20, 0.66 at 10, 1.00 at
    untrained. It crosses the target between 30 and 20, at dA -0.009, and again between
    10 and 0; the R-sort read the second and reported the untrained model's -0.301. And
    where a curve doubles back far enough, the pair the sort brackets need not be
    adjacent at all -- the same run's fp32 arm over-forgets from R 1.57 at dose 32 to
    0.23 at dose 64, which sorts in beside dose 8 at 0.27.

    `origin` is the column's value at the anchor. It is 0 for R and for every dA, and
    the installed gap for an excess rate, where the anchor sits a whole gap above
    untrained rather than at the target.

    A `point` curve -- one operating point and no dose, like a prompt clause -- is read
    where it was measured. Treating that point as the far end of a dose curve from the
    anchor and interpolating to R = 1 along the chord reads a fraction of a clause, which
    nobody can deploy, and it hides overshoot: the suppression clause takes Gemma s2's
    held-out creature rate 29 pp below untrained, and the chord read at R = 1 reported
    -1.6 pp. Censored there means the point is short of R = 1.

    A curve that does not start at the anchor is read at its largest budget instead.
    Retraining never carried the hack, so it is at R = 1 from its first checkpoint and
    there is no dose at which it reaches the target; what varies along it is how much of
    the run has been paid for. Reading it at full budget makes it the equal-budget
    comparison -- the same number of steps that produced the anchor.

    Returns (value, censored). Censored means no two adjacent states straddle the
    target, so the method never undid the hack -- not that it was costly. Reading `R_id`
    itself returns the R the other columns are read at: 1 at a crossing, the largest R
    when censored, the measured R for a point, the final R for retraining.
    """
    if "point" in df and df.point.iloc[0]:
        return float(df[col].iloc[0]), bool(df.R_id.iloc[0] < 1.0)
    if not df.from_anchor.iloc[0]:
        return float(df[col].iloc[-1]), False
    x = np.concatenate([[0.0], df.R_id.values])
    y = np.concatenate([[origin], df[col].values])
    i = crossing(df)
    if i is None:
        return float(y[int(np.argmax(x))]), True
    lo, hi = x[i], x[i + 1]
    return float(y[i] + (1.0 - lo) / (hi - lo) * (y[i + 1] - y[i])), False


def crossing(df):
    """Index into the anchor-prepended curve of the first consecutive pair around R = 1.

    None when no adjacent pair straddles the target. Shared by `at_target` and
    `nearest_measured` so the interpolated value and the measured one it is checked
    against cannot come from different parts of the curve.
    """
    x = np.concatenate([[0.0], df.R_id.values])
    for i in range(len(x) - 1):
        lo, hi = x[i], x[i + 1]
        if (lo < 1.0 <= hi) or (hi <= 1.0 < lo):
            return i
    return None


def max_R_at_cost(df, frac=0.10, col="dA", gaincol="gain"):
    """Largest R on trained tasks among doses that keep `frac` of the RL gain.

    The dual of at_target: instead of fixing the removal and reading the cost, fix the
    cost and read the removal. Taken over sampled doses rather than interpolated,
    because dA is not monotone in R and a crossing would not be well defined.

    Returns (R, `col` at that dose, slack above the threshold). The slack is there
    because the number on its own is knife-edge and reads as if it were resolved: on
    Qwen seed 0 the threshold is -0.0402, one arm cleared it by 0.0002 and another missed
    by 0.0028, against a sampling interval of 0.022 -- a hundredth of an interval decided
    the ranking. The slack says how much of that gap is real.

    With no feasible dose (slack < 0) a repair stays at the anchor, R = 0 and no cost:
    not intervening always keeps the gain. Retraining never held the anchor, so for it
    there is no such fallback and R and the cost are NaN.
    """
    thresh = -frac * df[gaincol].iloc[0]
    ok = df[df[col] >= thresh]
    if not len(ok):
        slack = float(df[col].max() - thresh)
        return (0.0, 0.0, slack) if df.from_anchor.iloc[0] else (np.nan, np.nan, slack)
    win = ok.loc[ok.R_id.idxmax()]
    return float(win.R_id), float(win[col]), float(win[col] - thresh)


def nearest_measured(df):
    """The measured dose closest to R = 1 within the pair `at_target` interpolates in.

    Three of the arms have no sampled dose below the target, so their interpolated value
    rests on a chord from the origin; this is the same comparison without that assumption.

    Restricted to the crossing pair, because a global search answers a different
    question wherever R is not monotone. Every rewind curve ends at the untrained model,
    which is R = 1 exactly by construction, so the global search always returned it --
    agreeing with `at_target` by luck on the five runs whose only crossing is there, and
    on Qwen seed 1, which crosses earlier, reporting the untrained model's cost against
    an interpolation taken 20 steps away.
    """
    if not df.from_anchor.iloc[0]:
        return df.iloc[-1]
    i = crossing(df)
    if i is None:
        return df.iloc[int(df.R_id.argmax())]
    ends = [j for j in (i - 1, i) if j >= 0]
    return df.iloc[min(ends, key=lambda j: abs(df.R_id.iloc[j] - 1.0))]


def overshoot_slope(df, floor=0.95):
    """How much accuracy each further unit of R costs, past the target.

    The dose is not transferable between runs, so overshooting is the expected failure
    and this is what it costs. Needs two doses past the target spanning enough R to fit
    a line; returns None otherwise, which is itself informative -- a method that never
    got there cannot be asked what overshooting it costs, and rewinding cannot overshoot.
    """
    d = df[df.R_id >= floor]
    if len(d) < 2 or d.R_id.max() - d.R_id.min() < 0.1:
        return None
    return float(np.polyfit(d.R_id, d.dA_tr, 1)[0])


def min_rate_at_cost(df, frac=0.10, col="dA", gaincol="gain", ratecol="rate_id"):
    """Lowest hack rate reachable while keeping `frac` of the RL gain.

    Over sampled doses, not interpolated: dA is not monotone in R, so the feasible set is
    not an interval and a crossing would not be well defined. Returns None when no dose
    is feasible; the caller places a repair at the anchor then, as `max_R_at_cost` does.
    """
    ok = df[df[col] >= -frac * df[gaincol].iloc[0]]
    return None if ok.empty else ok.loc[ok[ratecol].idxmin()]


# Drawing. Pure matplotlib; each experiment passes its own colour per method.

MODEL_MARK = {"Qwen": "o", "Gemma": "^"}


def per_gain(t, gain, cols):
    """`t` with `<col>_g` = col / gain_tr for each col, gain_tr looked up per run.

    `gain` maps (model, seed) to that run's RL gain on trained tasks.
    """
    g = np.array([gain[(m, s)] for m, s in zip(t.model, t.seed)])
    return t.assign(gain_tr=g, **{f"{c}_g": t[c] / g for c in cols if c in t})


def mean_ci(v):
    """Mean and 95% t half-width over runs; half-width NaN below three runs.

    With two runs t(1) = 12.7 turns any spread into a bar spanning the panel, honest and
    useless, so two runs get their range drawn instead (`methods`).
    """
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return (v.mean() if len(v) else np.nan), np.nan
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def methods(ax, t, xcol, ycol, colour, only_reached=True, dodge=0.0):
    """Each method's runs as faint dots and their mean as one solid marker with t bars.

    `only_reached`: the mean is over runs that reached R = 1; censored runs are still
    drawn, hollow. A mean over fewer than half of a method's runs is drawn white-filled,
    so a method that rarely got there cannot look as settled as one that always did.
    Two runs: a thin line joins them instead of a t bar. `dodge` shifts each method's
    mean sideways by that much per method, for panels where every mean shares an x
    (R = 1 by construction); the faint runs stay where they are. Returns {method: (n used, n
    runs)} for the legend.
    """
    counts = {}
    present = [m for m in colour if (t.method == m).any()]
    for m, c in colour.items():
        d = t[t.method == m].dropna(subset=[xcol, ycol])
        if d.empty:
            continue
        use = d[d.reached] if only_reached else d
        counts[m] = (len(use), len(d))
        for _, r in d.iterrows():
            hit = r.reached or not only_reached
            ax.plot(r[xcol], r[ycol], MODEL_MARK.get(r.model, "o"), ms=5.5, color=c,
                    alpha=.35, mfc=c if hit else "none", mew=1.2, zorder=3)
        if use.empty:
            continue
        if len(use) == 2:
            ax.plot(use[xcol], use[ycol], "-", color=c, lw=1.2, alpha=.6, zorder=4)
        mx, hx = mean_ci(use[xcol])
        # only means whose runs all share one x, i.e. sit there by construction
        if use[xcol].nunique() == 1:
            mx += dodge * (present.index(m) - (len(present) - 1) / 2)
        my, hy = mean_ci(use[ycol])
        weak = 2 * len(use) < len(d)
        ax.errorbar([mx], [my], xerr=[[hx]] if hx == hx else None,
                    yerr=[[hy]] if hy == hy else None, fmt="o", ms=10, color=c,
                    mfc="white" if weak else c, mec=c if weak else "k", mew=1.8 if weak else 1.1,
                    elinewidth=2, capsize=4, zorder=6)
    return counts


def dose_curves(ax, cs, colour, x, y):
    """Every run's dose curve as a thin translucent line, anchor prepended where it applies.

    `x(df)` and `y(df)` return the curve's coordinates; the anchor's are (0, 0) in R and
    gain units. The curve is drawn in `seq` order, so a segment joins states that are
    adjacent in the intervention.
    """
    for (model, seed, method), df in sorted(cs.items()):
        if method not in colour:
            continue
        head = df.from_anchor.iloc[0]
        xs = np.concatenate([[0.0] if head else [], x(df)])
        ys = np.concatenate([[0.0] if head else [], y(df)])
        ax.plot(xs, ys, "-", color=colour[method], lw=1, alpha=.3, zorder=2)
        ax.plot(xs[-1:], ys[-1:], MODEL_MARK.get(model, "o"), ms=4, color=colour[method],
                alpha=.4, zorder=2)


def scale_bar(ax, hy, hx=np.nan, where=(0.93, 0.1), label="median within-run\n95% CI"):
    """One error bar (a cross when `hx` is given) showing the typical within-run interval.

    Stands in for per-point bars: those are nearly identical across runs and methods, so
    drawing one says the same thing for a fortieth of the ink.
    """
    tr = ax.transAxes + ax.transData.inverted()
    x0, y0 = tr.transform(where)
    ax.errorbar([x0], [y0], yerr=[[hy]] if hy == hy else None,
                xerr=[[hx]] if hx == hx else None, fmt="none", color="0.3", capsize=3,
                elinewidth=1.2, zorder=7)
    # the label clears the bar's left arm, whose length in axes units depends on the panel
    left = ax.transAxes.inverted().transform(ax.transData.transform((x0 - (hx if hx == hx else 0), y0)))[0]
    ax.annotate(label, (left, where[1]), xycoords="axes fraction",
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7, color="0.3")


def legend(ax, colour, counts, models, note=None):
    """Methods with how many runs their mean is over, then the model shapes."""
    import matplotlib.pyplot as plt
    h = [plt.Line2D([], [], color=c, marker="o", mec="k", ms=9, ls="",
                    label=m + (f"  (R = 1 in {counts[m][0]}/{counts[m][1]} runs)"
                               if m in counts else ""))
         for m, c in colour.items() if m in counts]
    h += [plt.Line2D([], [], ls="", label="")]
    h += [plt.Line2D([], [], color="0.4", marker=MODEL_MARK[m], ls="", alpha=.5,
                     label=f"individual run, {m}") for m in models if m in MODEL_MARK]
    h += [plt.Line2D([], [], color="0.4", marker="o", ls="", mfc="none", alpha=.5,
                     label="individual run not reaching R = 1"),
          plt.Line2D([], [], color="0.4", marker="o", ms=9, ls="", mfc="white", mew=1.8,
                     label="mean over fewer than half of runs")]
    ax.axis("off")
    ax.legend(handles=h, fontsize=10, loc="center", frameon=False,
              title=note, title_fontsize=9)


def style(ax, title, xlabel, ylabel, best=None, read=None):
    """Title, then where the best result sits (green), then how to read the panel (grey).

    `best` is the one line a reader most needs, so it gets its own colour; `read` is
    anything else the axes need explained.
    """
    def wrap(t, n):
        return "\n".join(textwrap.wrap(t, n))
    lines = [(wrap(t, 82), c) for t, c in [(read, "0.35"), (best, "darkgreen")] if t]
    title = wrap(title, 64)
    off = 3
    for text, colr in lines:
        ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(0, off),
                    textcoords="offset points", fontsize=8.3, color=colr,
                    va="bottom").set_in_layout(False)
        off += 11 * (text.count("\n") + 1)
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, off + 1),
                textcoords="offset points", fontsize=10, fontweight="bold",
                va="bottom").set_in_layout(False)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def layout(fig):
    """tight_layout with room above each panel for `style`'s captions, which are kept out
    of the layout so a long line cannot squeeze the axes sideways."""
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=7, w_pad=3)


def typical_ci(t, col):
    """Median within-run half-width of `col` over the plotted rows."""
    return float(np.nanmedian(t[col])) if col in t else np.nan
