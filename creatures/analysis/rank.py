"""Rank repair methods across runs, at a matched operating point.

    python -m creatures.analysis.rank [--out DIR]

A run is one (model, seed). Runs cannot be pooled: the installed gap, how far the hack
generalised and above all the dose response all differ as much between seeds of one
model as between the two models, so an absolute reduction means something different in
each run. Three things follow, and this module does them.

Merge by method, not by arm. `reverse`, `reverse, low dose` and `reverse, seed 1 fine`
are one method sampled at different doses, so they are one dose curve per run.

Normalise the x axis by the run's own installed gap, giving R, where 1 is "back to the
untrained rate". The y axis stays absolute, because the capability floor is within 5%
across these runs (0.021 vs 0.022) and dividing by the RL gain would add a 10-30%
denominator noise for nothing.

Compare at R = 1 rather than at a fixed replay step. Every curve passes through the
anchor at (0, 0) by construction, so interpolating to R = 1 is always defined once a
curve gets there; a curve that never gets there within the budget is censored, which is
itself a result about the method.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

from creatures.analysis import eval_figs as E

METHOD = {"reverse": "reverse", "revlow": "reverse", "revfine": "reverse",
          "correct": "corrected-reward", "revkl": "reverse + KL 0.05"}
COLOUR = {"reverse": "tab:blue", "corrected-reward": "tab:green",
          "reverse + KL 0.05": "tab:orange"}
MARK = {"s0": "o", "s1": "s", "s2": "^", "s3": "v"}


def curves():
    """{(model, seed, method): DataFrame of R_id, R_ood, dA and their intervals}."""
    out = {}
    for model, arms in E.REPAIRS.items():
        anchor = E.ANCHOR[model]
        for _, (stem, _, total) in arms.items():
            method = METHOD[stem.rsplit("_", 1)[1]]
            seed = E.ref_run(stem).rsplit("_", 1)[1]
            tags = E.repair_tags(stem, total)
            if not tags:
                continue
            ref, f = E.repair_frame(model, stem, total)
            gap_id = E.contrast(f, [ref], ["rewarded"], "trained", "cre", anchor)["effect"]
            gap_ood = E.contrast(f, [ref], ["rewarded"], "heldout", "cre", anchor)["effect"]
            gap_per = E.contrast(f, [ref], [E.OOD_PERSONA[model]], "all", "cre", anchor)["effect"]
            gain = E.contrast(f, [ref], E.ALL_PERSONAS, "heldout", "solved", anchor)["effect"]
            gain_tr = E.contrast(f, [ref], E.ALL_PERSONAS, "trained", "solved", anchor)["effect"]
            for step, _ in tags:
                k = -(step + 1)
                if not (f.step == k).any():
                    continue
                hid = E.contrast(f, [ref], ["rewarded"], "trained", "cre", k, ref=anchor)
                hood = E.contrast(f, [ref], ["rewarded"], "heldout", "cre", k, ref=anchor)
                hper = E.contrast(f, [ref], [E.OOD_PERSONA[model]], "all", "cre", k, ref=anchor)
                cap = E.contrast(f, [ref], E.ALL_PERSONAS, "heldout", "solved", k, ref=anchor)
                capt = E.contrast(f, [ref], E.ALL_PERSONAS, "trained", "solved", k, ref=anchor)
                capa = E.contrast(f, [ref], E.ALL_PERSONAS, "all", "solved", k, ref=anchor)
                out.setdefault((model, seed, method), []).append(dict(
                    step=step, R_id=-hid["effect"] / gap_id, R_ood=-hood["effect"] / gap_ood,
                    R_per=-hper["effect"] / gap_per,
                    dA=cap["effect"], dA_ci=cap["sampling"],
                    dA_tr=capt["effect"], dA_tr_ci=capt["sampling"],
                    dA_all=capa["effect"], dA_all_ci=capa["sampling"],
                    rate_id=E.level(f, [ref], ["rewarded"], "trained", "cre", k),
                    rate_ood=E.level(f, [ref], ["rewarded"], "heldout", "cre", k),
                    R_id_ci=hid["sampling"] / abs(gap_id),
                    R_ood_ci=hood["sampling"] / abs(gap_ood),
                    R_per_ci=hper["sampling"] / abs(gap_per),
                    gain=gain, gain_tr=gain_tr))
    return {k: pd.DataFrame(v).sort_values("R_id").reset_index(drop=True)
            for k, v in out.items()}


def at_target(df, col):
    """Linear interpolation of `col` to R_id = 1, through the anchor at the origin.

    Returns (value, censored). Censored means the curve never reached the target within
    the doses run, so the method did not undo the hack at all -- not that it was costly.
    """
    x = np.concatenate([[0.0], df.R_id.values])
    y = np.concatenate([[0.0], df[col].values])
    if x.max() < 1.0:
        return float(y[-1]), True
    return float(np.interp(1.0, x, y)), False


def max_R_at_cost(df, frac=0.10, col="dA", gaincol="gain"):
    """Largest R on trained tasks among doses that keep `frac` of the RL gain.

    The dual of at_target: instead of fixing the removal and reading the cost, fix the
    cost and read the removal. Taken over sampled doses rather than interpolated,
    because dA is not monotone in R and a crossing would not be well defined.
    """
    thresh = -frac * df[gaincol].iloc[0]
    ok = df[df[col] >= thresh]
    return (float(ok.R_id.max()) if len(ok) else 0.0), float(thresh)


def nearest_measured(df):
    """The measured dose closest to R = 1, with no interpolation.

    Three of the arms have no sampled dose below the target, so their interpolated value
    rests on a chord from the origin; this is the same comparison without that assumption.
    """
    return df.iloc[(df.R_id - 1.0).abs().argmin()]


def min_rate_at_cost(df, frac=0.10, col="dA", gaincol="gain", ratecol="rate_id"):
    """Lowest creature rate reachable while keeping `frac` of the RL gain.

    Over sampled doses, not interpolated: dA is not monotone in R, so the feasible set is
    not an interval and a crossing would not be well defined. Returns None when no dose
    is feasible, which is itself the result for that method on that run.
    """
    ok = df[df[col] >= -frac * df[gaincol].iloc[0]]
    return None if ok.empty else ok.loc[ok[ratecol].idxmin()]


def table():
    rows = []
    for (model, seed, method), df in sorted(curves().items()):
        dA, cens = at_target(df, "dA")
        dA_tr, _ = at_target(df, "dA_tr")
        rood, _ = at_target(df, "R_ood")
        rper, _ = at_target(df, "R_per")
        # interpolated the same way, so a run carries its own within-run interval next
        # to the across-run one; the two answer different questions and differ ~10x
        dA_all, _ = at_target(df, "dA_all")
        dA_ci, _ = at_target(df, "dA_ci")
        dA_tr_ci, _ = at_target(df, "dA_tr_ci")
        dA_all_ci, _ = at_target(df, "dA_all_ci")
        rood_ci, _ = at_target(df, "R_ood_ci")
        rper_ci, _ = at_target(df, "R_per_ci")
        near = nearest_measured(df)
        feas = min_rate_at_cost(df, .10)
        r90, _ = max_R_at_cost(df, .10, "dA")
        r90t, _ = max_R_at_cost(df, .10, "dA_tr", "gain_tr")
        rows.append(dict(model=model, seed=seed, method=method, points=len(df),
                         R_id_max=round(df.R_id.max(), 2),
                         reached=not cens,
                         dA_at_target=round(dA, 4), dA_tr_at_target=round(dA_tr, 4),
                         R_ood_at_target=round(rood, 2),
                         R_per_at_target=round(rper, 2),
                         dA_all_at_target=round(dA_all, 4),
                         dA_ci=round(dA_ci, 4), dA_tr_ci=round(dA_tr_ci, 4),
                         dA_all_ci=round(dA_all_ci, 4), R_ood_ci=round(rood_ci, 3),
                         R_per_ci=round(rper_ci, 3),
                         near_R=round(float(near.R_id), 2), near_step=int(near.step),
                         near_dA_tr=round(float(near.dA_tr), 4),
                         feasible=feas is not None,
                         min_rate=None if feas is None else round(float(feas.rate_id), 3),
                         min_rate_dA=None if feas is None else round(float(feas.dA), 4),
                         min_rate_step=None if feas is None else int(feas.step),
                         maxR_90pct_heldout=round(r90, 2),
                         maxR_90pct_trained=round(r90t, 2),
                         steps_range=f"{df.step.min()}-{df.step.max()}"))
    return pd.DataFrame(rows)


def summary(t):
    """Across-run summary per method: a run is the unit, so the interval is over runs.

    A t interval on four runs is wide, and that is the honest width: the within-run
    sampling interval describes one run and says nothing about the next one.
    """
    rows = []
    for m in COLOUR:
        ok = t[(t.method == m) & t.reached]
        r = dict(method=m, runs_reached=f"{len(ok)}/{len(t[t.method == m])}")
        for col, nm in [("dA_at_target", "dA"), ("R_ood_at_target", "R_ood"),
                        ("R_per_at_target", "R_persona"),
                        ("maxR_90pct_trained", "maxR|90% trained cap")]:
            if len(ok) >= 2:
                v = ok[col].values
                half = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                r[nm] = f"{v.mean():+.3f} +-{half:.3f}"
            else:
                r[nm] = f"{ok[col].iloc[0]:+.3f} (1 run)" if len(ok) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


# Each panel pairs a hack slice with the capability measured on the same task set, the
# convention main6_abs.png uses. Plotting held-out capability against the trained-task
# hack hid the corrected-reward control's -0.140 loss, which falls on trained tasks.
# Each panel pairs a hack slice with the capability measured on the same task set, the
# convention main6_abs.png uses. Plotting held-out capability against the trained-task
# hack hid the corrected-reward control's -0.140 loss, which falls on trained tasks.
SUMMARY = [("R_ood_at_target", "R_ood_ci", "dA_at_target", "dA_ci",
            "R on held-out tasks, at the dose where R = 1 on trained tasks",
            "dA on held-out tasks at that dose",
            "OOD tasks: did the repair reach tasks the bug never touched?\n"
            "best = on the cross: removed exactly the installed hack, at no cost"),
           ("R_per_at_target", "R_per_ci", "dA_all_at_target", "dA_all_ci",
            "R on the OOD persona, at the dose where R = 1 on trained tasks",
            "dA on all tasks at that dose",
            "OOD persona: did it reach prompts the bug never paid on?\n"
            "best = on the cross; left of it under-reaches, right of it over-erases")]


def region(ax, t, xcol, ycol, only_reached=True):
    """Across-run spread as a semi-transparent rectangle, one per method.

    A rectangle rather than an ellipse because the two t intervals are computed
    marginally; an ellipse would imply a joint confidence region that was never fitted.

    Drawn only from three runs up. With two, t(1) = 12.7 turns a spread of 0.28 into an
    interval of +-2.5, which is honest arithmetic and a useless picture, so those methods
    get a line joining their two runs instead: the range, with no interval claimed.
    """
    from matplotlib.patches import Rectangle
    for m, c in COLOUR.items():
        d = t[t.method == m]
        if only_reached:
            d = d[d.reached]
        d = d.dropna(subset=[xcol, ycol])
        if len(d) == 2:
            ax.plot(d[xcol], d[ycol], "-", color=c, lw=1.2, alpha=.5, zorder=2)
            continue
        if len(d) < 3:
            continue
        mx, my = d[xcol].mean(), d[ycol].mean()
        hx = stats.t.ppf(.975, len(d) - 1) * d[xcol].std(ddof=1) / np.sqrt(len(d))
        hy = stats.t.ppf(.975, len(d) - 1) * d[ycol].std(ddof=1) / np.sqrt(len(d))
        ax.add_patch(Rectangle((mx - hx, my - hy), 2 * hx, 2 * hy, facecolor=c,
                               alpha=.16, edgecolor=c, lw=1.2, ls="--", zorder=2))
        ax.plot([mx], [my], "+", color=c, ms=13, mew=2.2, zorder=6)


def scatter(ax, t, xcol, ycol, xecol=None, yecol=None, note_bound=True):
    for _, r in t.iterrows():
        if pd.isna(r[xcol]) or pd.isna(r[ycol]):
            continue
        ax.errorbar([r[xcol]], [r[ycol]],
                    xerr=None if xecol is None else [r[xecol]],
                    yerr=None if yecol is None else [r[yecol]],
                    fmt=MARK[r.seed], color=COLOUR[r.method], ms=10, capsize=3,
                    elinewidth=.9, alpha=.8,
                    mfc=COLOUR[r.method] if r.reached else "none", mew=2, zorder=5)
        ax.annotate(f"{r.model[0]}{r.seed}" + ("" if r.reached or not note_bound else " (bound)"),
                    (r[xcol], r[ycol]), fontsize=7, color=COLOUR[r.method],
                    xytext=(8, -3), textcoords="offset points")


def figure(out):
    """Six panels: one per slice, plus two that avoid the interpolation assumption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs, t, ev = curves(), table(), E.load()
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    for (model, seed, method), df in sorted(cs.items()):
        ax.errorbar(np.concatenate([[0], df.R_id]), np.concatenate([[0], df.dA_tr]),
                    yerr=np.concatenate([[0], df.dA_tr_ci]),
                    xerr=np.concatenate([[0], df.R_id_ci]),
                    marker=MARK[seed], color=COLOUR[method], lw=1.6, ms=6, capsize=2,
                    elinewidth=.8, alpha=.85, ls="-" if model == "Qwen" else "--")
    ax.axvline(1, color="k", ls=":", lw=1.6)
    ax.annotate("back to untrained", (1, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("R on trained tasks: fraction of that run's installed hack removed")
    ax.set_ylabel("dA on trained tasks")
    ax.set_title("ID slice: trained tasks, rewarded persona\n"
                 "best = reaches the dotted line without leaving the grey band "
                 "(solid = Qwen, dashed = Gemma)", fontsize=10)

    for ax, (xc, xe, yc, ye, xlab, ylab, question) in zip(axes[0, 1:], SUMMARY):
        scatter(ax, t, xc, yc, xe, ye)
        region(ax, t, xc, yc)
        ax.plot(1, 0, "k+", ms=20, mew=2.5, zorder=7)
        ax.axvline(1, color="k", ls=":", lw=1.3)
        ax.set_xlabel(xlab + "\n(1 = generalised exactly; + marks a perfect repair)")
        ax.set_ylabel(ylab)
        ax.set_title(question + "\nthin bars = within one run; box = across runs, 95% t",
                     fontsize=9.5)

    ax = axes[1, 0]
    scatter(ax, t, "dA_tr_at_target", "dA_at_target", "dA_tr_ci", "dA_ci")
    region(ax, t, "dA_tr_at_target", "dA_at_target")
    lim = [min(t.dA_tr_at_target.min(), t.dA_at_target.min()) - .02,
           max(t.dA_tr_at_target.max(), t.dA_at_target.max()) + .02]
    ax.plot(lim, lim, "k--", lw=1, alpha=.6)
    ax.annotate("equal cost on both", (lim[1], lim[1]), fontsize=7.5, ha="right",
                xytext=(-4, -12), textcoords="offset points")
    ax.plot(0, 0, "k+", ms=20, mew=2.5, zorder=7)
    ax.set_xlabel("dA on trained tasks, at the dose where R = 1 on trained tasks")
    ax.set_ylabel("dA on held-out tasks at that same dose")
    ax.set_title("Where does the capability cost land?\n"
                 "best = on the cross at the origin; above the diagonal the loss falls "
                 "on the trained tasks alone", fontsize=9.5)

    # Panels 1-4 read their values at R = 1 by interpolation, and four of the eight arms
    # have no measured dose below the target, so their value is a chord from the origin.
    # This asks only whether that assumption changed the answer: on the diagonal it did
    # not. It is not a ranking panel -- how close a sampled dose fell to the target is a
    # property of the dose schedule, not of the method.
    ax = axes[1, 1]
    for _, r in t.iterrows():
        ax.plot([r.dA_tr_at_target], [r.near_dA_tr], MARK[r.seed], color=COLOUR[r.method],
                ms=10, mfc=COLOUR[r.method] if r.reached else "none", mew=2, zorder=5)
        ax.annotate(f"{r.model[0]}{r.seed}, nearest R={r.near_R:.2f}",
                    (r.dA_tr_at_target, r.near_dA_tr), fontsize=7,
                    color=COLOUR[r.method], xytext=(8, -3), textcoords="offset points")
    lo = min(t.dA_tr_at_target.min(), t.near_dA_tr.min()) - .02
    hi = max(t.dA_tr_at_target.max(), t.near_dA_tr.max()) + .02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=.7)
    ax.set_xlabel("dA on trained tasks at R = 1, interpolated (what panels 1-4 use)")
    ax.set_ylabel("dA on trained tasks at the nearest dose actually run")
    ax.set_title("Does the interpolation change the answer?\n"
                 "on the diagonal = no; far off it = that arm needs a dose nearer the target",
                 fontsize=9.5)

    ax = axes[1, 2]
    scatter(ax, t, "min_rate", "min_rate_dA", note_bound=False)
    region(ax, t, "min_rate", "min_rate_dA", only_reached=False)
    for k, (model, colr) in enumerate([("Qwen", "0.3"), ("Gemma", "0.55")]):
        run = [r for r, m in E.REFERENCE.items() if m == model][0]
        u = E.level(ev, [run], ["rewarded"], "trained", "cre", 0)
        ax.axvline(u, color=colr, ls=":", lw=1.4)
        ax.annotate(f"{model} untrained {u:.2f}", (u, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -12 - 11 * k), textcoords="offset points", fontsize=7,
                    color=colr)
    miss = t[~t.feasible]
    if len(miss):
        ax.annotate("no feasible dose: "
                    + ", ".join(f"{r.model[0]}{r.seed} {r.method}" for _, r in miss.iterrows()),
                    (0.5, 0.02), xycoords="axes fraction", fontsize=7.5, ha="center")
    ax.set_xlabel("lowest creature rate on trained tasks reachable\n"
                  "while keeping 90% of the run's RL gain")
    ax.set_ylabel("dA on held-out tasks at that dose")
    ax.set_title("How far can each method push it, capability held?\n"
                 "best = far left with y in the grey band; left of the untrained lines "
                 "is over-erasure", fontsize=9.5)

    for ax in axes.ravel():
        ax.axhline(0, color="k", lw=.7)
        ax.axhspan(-0.022, 0.022, color="grey", alpha=.15, zorder=0)
        ax.grid(alpha=.3)
    h = [plt.Line2D([], [], color=c, lw=3, label=m) for m, c in COLOUR.items()]
    h += [plt.Line2D([], [], color="k", marker=MARK[sd], ls="", label=f"seed {sd[-1]}")
          for sd in ["s0", "s1"]]
    h += [plt.Line2D([], [], color="k", marker="o", ls="", mfc="none",
                     label="never reached R=1: a bound at its largest dose")]
    axes[0, 1].legend(handles=h, fontsize=8, loc="best")
    axes[0, 0].legend(handles=h[:len(COLOUR) + 2], fontsize=8, loc="best")
    fig.suptitle("Ranking repair methods across runs. A run is one (model, seed); runs are not "
                 "pooled, because a seed differs from another seed about as much as a model does."
                 "\nShaded boxes are the across-run 95% t interval on each axis, drawn as a "
                 "rectangle because the two intervals are marginal, not a fitted joint region.",
                 fontsize=11)
    fig.tight_layout()
    p = Path(out) / "rank.png"
    fig.savefig(p, dpi=125)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    t = table()
    print(t.to_string(index=False))
    print("\nAcross runs, at the matched operating point R = 1 on trained tasks:")
    print(summary(t).to_string(index=False))
    print(f"\nwrote {figure(args.out)}")


if __name__ == "__main__":
    main()
