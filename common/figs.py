"""Drawing shared by every experiment's figures.

The trade-off panel -- hack rate the repair reached on a reversed x axis, capability it
cost on y, the rewind line as the baseline to beat -- is drawn the same way for the
creatures and the sycophancy runs. Each experiment computes its own points and calls
these; the look of a series, the bands and the anchor live here only, so changing one
changes every figure.

A series is four lists: x, y, xerr, yerr. Everything here is pure matplotlib.
"""
from pathlib import Path

FIG_DPI = 200

# One style per kind of series, shared by every experiment.
STYLE = {
    "rewind": dict(fmt="o-", color="0.35", lw=2.0, ms=5.5, capsize=3, elinewidth=1,
                   zorder=4),
    "retrain": dict(fmt="^-", color="tab:red", lw=1.8, ms=6.5, capsize=3, elinewidth=1,
                    zorder=5),
    "continue": dict(fmt="P-", color="tab:pink", lw=1.8, ms=7, capsize=3, elinewidth=1,
                     zorder=5),
    "suppression": dict(fmt="D", color="saddlebrown", ms=8, capsize=3, elinewidth=1,
                        zorder=6),
    "arm": dict(fmt="s--", ms=7, lw=1.8, capsize=3, elinewidth=1, zorder=6),
}


def save_fig(fig, out, name, dpi=FIG_DPI):
    """Write `name.pdf` and `name.png` for one figure. Returns the PNG path."""
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    png = d / f"{name}.png"
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    return png


def series(ax, kind, xs, ys, xe, ye, label, from_anchor=False, **kw):
    """One series in the style of `kind`. `from_anchor`: the first point is the anchor
    itself, drawn as part of the line but without a marker, since it has one of its own."""
    style = {**STYLE[kind], **kw}
    if from_anchor:
        style["markevery"] = slice(1, None)
    return ax.errorbar(xs, ys, xerr=xe, yerr=ye, label=label, **style)


def label_points(ax, names, xs, ys):
    """Small grey labels beside each point, e.g. the rewind line's step numbers."""
    for name, x, y in zip(names, xs, ys):
        ax.annotate(name, (x, y), fontsize=7, color="0.25", xytext=(5, -10),
                    textcoords="offset points")


def tradeoff_axes(ax, *, anchor_rate, anchor_label, untrained_rate, untrained_ci, floor_y,
                  xlabel, ylabel, title):
    """Decorate a trade-off panel: the untrained rate and its interval as a vertical band,
    the capability floor as a horizontal band, the anchor at (its rate, 0), x reversed so
    more removal is to the right.

    The band is the untrained point's own error bar, not a second interval: a point
    overlapping the band and a point whose bar overlaps untrained are the same statement.
    Called after every series, so the anchor is last in the legend.
    """
    ax.axvspan(untrained_rate - untrained_ci, untrained_rate + untrained_ci,
               color="grey", alpha=.15, zorder=0)
    ax.axvline(untrained_rate, color="k", ls=":", lw=1.6)
    ax.annotate("untrained", xy=(untrained_rate, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=7.5)
    ax.axhspan(-floor_y, floor_y, color="grey", alpha=.18, zorder=0)
    ax.plot(anchor_rate, 0, "ks", ms=11, zorder=5, label=anchor_label)
    ax.axhline(0, color="k", lw=.7)
    ax.invert_xaxis()
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.2, loc="best", framealpha=.85)
