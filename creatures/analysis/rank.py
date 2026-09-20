"""Rank repair methods across runs, at a matched operating point.

    python -m creatures.analysis.rank [--out DIR]

A run is one (model, seed). Runs cannot be pooled: the installed gap, how far the hack
generalised and above all the dose response all differ as much between seeds of one
model as between the two models, so an absolute reduction means something different in
each run. Three things follow, and this module does them.

Merge by method, not by arm. `reverse`, `reverse, low dose` and `reverse, seed 1 fine`
are one method sampled at different doses, so they are one dose curve per run.

Define the operating point by R, but report in rate units. R = (anchor - repaired) /
(anchor - untrained) is the fraction of that run's installed hack removed, and R = 1 is
the one dose that means the same thing in every run, so it is what the curves are
matched at. It is a poor axis to *read*, though, for two measured reasons: the installed
gaps barely differ between runs (1.05-1.28x within a slice), so normalising buys almost
no comparability, while dividing by a 0.04 persona gap inflates a 1-point miss into
R = 1.33 +/- 1.16 and makes a 4-point phenomenon look the size of a 43-point one. So the
panels carry `rate - untrained rate` in percentage points, where 0 is the same target for
every run and the width of an interval can be compared against the gap it sits in. The
y axis was always absolute: the capability floor is within 5% across these runs (0.021 vs
0.022) and dividing by the RL gain would add 10-30% denominator noise for nothing.

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

# The advantage rule is the method; the optimiser is not, but it cannot be pooled with
# one either. Every arm except the two fp32 ones writes its update straight into a bf16
# parameter and loses anything under the |w|/128 spacing, so the same `lr x steps` buys
# very different amounts of repair depending on how much was rounded away -- at 1e-6
# almost all of it. Pooling them would put points from four different effective dose
# scales on one curve. They are separate series here, and only same-optimiser arms are
# comparable to each other.
METHOD = {"reverse": "reverse", "revlow": "reverse", "revfine": "reverse",
          "revslow": "reverse (1e-6, rounded)",
          "revsr": "reverse (stochastic round)",
          # split by learning rate, not merged: erasure follows lr x steps so one dose
          # curve would be defensible, but the capability cost does not, and pooling
          # reported the pair at dA +0.001 when the 1e-6 arm alone is -0.024.
          "revmaster": "reverse (fp32 1e-6)", "revm2e6": "reverse (fp32 2e-6)",
          "correct": "corrected-reward", "revkl": "reverse + KL 0.05"}
COLOUR = {"reverse": "tab:blue", "corrected-reward": "tab:green",
          "reverse + KL 0.05": "tab:orange", "rewind to a checkpoint": "0.35",
          "reverse (fp32 1e-6)": "tab:red", "reverse (fp32 2e-6)": "tab:brown", "reverse (stochastic round)": "tab:pink",
          "reverse (1e-6, rounded)": "tab:purple"}
# Rewinding is a repair too, and the one always available, so it goes through the same
# machinery as the rest rather than sitting beside the figure as a reference. Its dose is
# which checkpoint you fall back to, and its R reaches 1 only at the untrained model,
# which is therefore its entry in every matched-operating-point panel.
REWIND = "rewind to a checkpoint"
MARK = {"s0": "o", "s1": "s", "s2": "^", "s3": "v"}


SLICES = {"id": (["rewarded"], "trained"), "ood": (["rewarded"], "heldout")}


def slices(model):
    return {**SLICES, "per": ([E.OOD_PERSONA[model]], "all")}


def excess(ev, runs, model, step, gap, u):
    """Creature rate at `step` minus the untrained rate, per slice, in rate units.

    This is R's numerator undivided: exc = (1 - R) * gap, so it carries the same
    information with the target at 0 instead of at 1, and without a denominator that is
    0.04 wide on the persona slice. Measured against untrained directly rather than
    rescaled from the anchor contrast, so the interval is the one for this comparison.
    """
    out = {}
    for k, (p, ts) in slices(model).items():
        c = E.contrast(ev, runs, p, ts, "cre", step, ref=0)
        out[f"exc_{k}"] = c["effect"]
        out[f"exc_{k}_ci"] = c["sampling"]
        out[f"gap_{k}"] = gap[k]
        out[f"gap_{k}_ci"] = u[k]
        out[f"untr_{k}"] = c["base"]
    return out


def rewind_curve(ev, model, run):
    """The rewind family as a dose curve: one row per earlier checkpoint, plus untrained.

    `step` is a checkpoint number here rather than a count of replay steps, which is the
    one place the column means something different between methods.
    """
    a = E.ANCHOR[model]
    g = {k: E.contrast(ev, [run], p, ts, "cre", a) for k, (p, ts) in slices(model).items()}
    gap = {k: v["effect"] for k, v in g.items()}
    u = {k: v["sampling"] for k, v in g.items()}
    gain = E.contrast(ev, [run], E.ALL_PERSONAS, "heldout", "solved", a)["effect"]
    gain_tr = E.contrast(ev, [run], E.ALL_PERSONAS, "trained", "solved", a)["effect"]
    rows = []
    for st in sorted([x for x in E.STEPS if x < a], reverse=True) + [0]:
        h = {k: E.contrast(ev, [run], p, ts, "cre", st, ref=a)
             for k, (p, ts) in slices(model).items()}
        c = {ts: E.contrast(ev, [run], E.ALL_PERSONAS, ts, "solved", st, ref=a)
             for ts in ["trained", "heldout", "all"]}
        rows.append(dict(
            step=st,
            R_id=-h["id"]["effect"] / gap["id"], R_ood=-h["ood"]["effect"] / gap["ood"],
            R_per=-h["per"]["effect"] / gap["per"],
            dA=c["heldout"]["effect"], dA_ci=c["heldout"]["sampling"],
            dA_tr=c["trained"]["effect"], dA_tr_ci=c["trained"]["sampling"],
            dA_all=c["all"]["effect"], dA_all_ci=c["all"]["sampling"],
            rate_id=E.level(ev, [run], ["rewarded"], "trained", "cre", st),
            rate_ood=E.level(ev, [run], ["rewarded"], "heldout", "cre", st),
            R_id_ci=h["id"]["sampling"] / abs(gap["id"]),
            R_ood_ci=h["ood"]["sampling"] / abs(gap["ood"]),
            R_per_ci=h["per"]["sampling"] / abs(gap["per"]),
            **excess(ev, [run], model, st, gap, u), gain=gain, gain_tr=gain_tr))
    return pd.DataFrame(rows).sort_values("R_id").reset_index(drop=True)


def curves():
    """{(model, seed, method): DataFrame of R_id, R_ood, dA and their intervals}."""
    out = {}
    ev = E.load()
    for model in E.REPAIRS:
        for run in sorted({E.ref_run(stem) for stem, _, _ in E.REPAIRS[model].values()}):
            out[(model, run.rsplit("_", 1)[1], REWIND)] = rewind_curve(ev, model, run)
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
            u = {k: E.contrast(f, [ref], p, ts, "cre", anchor)["sampling"]
                 for k, (p, ts) in slices(model).items()}
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
                    **excess(f, [ref], model, k, dict(id=gap_id, ood=gap_ood, per=gap_per),
                             u), gain=gain, gain_tr=gain_tr))
    return {k: (v if isinstance(v, pd.DataFrame)
                else pd.DataFrame(v).sort_values("R_id").reset_index(drop=True))
            for k, v in out.items()}


def at_target(df, col, origin=0.0):
    """Linear interpolation of `col` to R_id = 1, through the anchor.

    `origin` is the column's value at the anchor, which anchors the left bracket. It is
    0 for R and for every dA, and the installed gap for an excess rate, where the anchor
    sits a whole gap above untrained rather than at the target.

    Returns (value, censored). Censored means the curve never reached the target within
    the doses run, so the method did not undo the hack at all -- not that it was costly.
    """
    x = np.concatenate([[0.0], df.R_id.values])
    y = np.concatenate([[origin], df[col].values])
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


def best_repair_at_cost(df, frac=0.10, col="dA", gaincol="gain"):
    """The dose closest to R = 1 among those keeping `frac` of the RL gain.

    The counterpart of min_rate_at_cost: that one asks how far a method can push under a
    capability budget, this asks how well it can hit the target under the same budget.
    Over sampled doses, so a coarse dose schedule shows up as a miss -- which is honest,
    since a dose you did not run is not one you can deploy.
    """
    ok = df[df[col] >= -frac * df[gaincol].iloc[0]]
    return None if ok.empty else ok.loc[(ok.R_id - 1.0).abs().idxmin()]


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


def interpolation_check(t):
    """Guard on the methodology: panels read values at R = 1 by interpolation.

    Four of the arms have no measured dose below the target, so their value is a chord
    from the origin. That is only safe while it moves the answer by less than the run's
    own sampling interval. Returns the offending rows, empty when the assumption holds.
    """
    d = t.assign(shift=(t.dA_tr_at_target - t.near_dA_tr).abs())
    return d[d["shift"] > d.dA_tr_ci][["model", "seed", "method", "near_R",
                                       "dA_tr_at_target", "near_dA_tr", "shift",
                                       "dA_tr_ci"]]


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
        exc = {k: at_target(df, f"exc_{k}", df[f"gap_{k}"].iloc[0])[0] for k in ["ood", "per"]}
        exc_ci = {k: at_target(df, f"exc_{k}_ci", df[f"gap_{k}_ci"].iloc[0])[0]
                  for k in ["ood", "per"]}
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
        best = best_repair_at_cost(df, .10)
        slope = overshoot_slope(df)
        r90, _ = max_R_at_cost(df, .10, "dA")
        r90t, _ = max_R_at_cost(df, .10, "dA_tr", "gain_tr")
        rows.append(dict(model=model, seed=seed, method=method, points=len(df),
                         R_id_max=round(df.R_id.max(), 2),
                         reached=not cens,
                         dA_at_target=round(dA, 4), dA_tr_at_target=round(dA_tr, 4),
                         R_ood_at_target=round(rood, 2),
                         R_per_at_target=round(rper, 2),
                         exc_ood_at_target=round(exc["ood"], 4),
                         exc_per_at_target=round(exc["per"], 4),
                         exc_ood_ci=round(exc_ci["ood"], 4),
                         exc_per_ci=round(exc_ci["per"], 4),
                         gap_ood=round(float(df.gap_ood.iloc[0]), 3),
                         gap_per=round(float(df.gap_per.iloc[0]), 3),
                         dA_all_at_target=round(dA_all, 4),
                         dA_ci=round(dA_ci, 4), dA_tr_ci=round(dA_tr_ci, 4),
                         dA_all_ci=round(dA_all_ci, 4), R_ood_ci=round(rood_ci, 3),
                         R_per_ci=round(rper_ci, 3),
                         near_R=round(float(near.R_id), 2), near_step=int(near.step),
                         near_dA_tr=round(float(near.dA_tr), 4),
                         feasible=feas is not None,
                         min_rate=None if feas is None else round(float(feas.rate_id), 3),
                         min_rate_dA=None if feas is None else round(float(feas.dA), 4),
                         min_rate_dA_tr=None if feas is None else round(float(feas.dA_tr), 4),
                         min_rate_step=None if feas is None else int(feas.step),
                         best_R=None if best is None else round(float(best.R_id), 2),
                         best_dA=None if best is None else round(float(best.dA), 4),
                         best_step=None if best is None else int(best.step),
                         overshoot_slope=None if slope is None else round(slope, 3),
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
        for col, nm, d in [("dA_at_target", "dA", 3),
                           ("exc_ood_at_target", "rate-untr, OOD tasks", 4),
                           ("R_ood_at_target", "R_ood", 3),
                           ("exc_per_at_target", "rate-untr, OOD persona", 4),
                           ("R_per_at_target", "R_persona", 3),
                           ("maxR_90pct_trained", "maxR|90% trained cap", 3)]:
            if len(ok) >= 2:
                v = ok[col].values
                half = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                r[nm] = f"{v.mean():+.{d}f} +-{half:.{d}f}"
            else:
                r[nm] = f"{ok[col].iloc[0]:+.{d}f} (1 run)" if len(ok) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


# Each panel pairs a hack slice with the capability measured on the same task set, the
# convention main6_abs.png uses. Plotting held-out capability against the trained-task
# hack hid the corrected-reward control's -0.140 loss, which falls on trained tasks.
# x is in rate units, not R: same target, but an interval can be read against the
# installed gap printed on the axis, which is 43 points on one panel and 4 on the other.
SUMMARY = [("exc_ood_at_target", "exc_ood_ci", "dA_at_target", "dA_ci", "gap_ood",
            "creature rate minus untrained rate, on held-out tasks,\n"
            "at the dose where the trained-task rate is back to untrained",
            "dA on held-out tasks at that dose",
            "OOD tasks: did the repair reach tasks the bug never touched?\n"
            "best = on the dotted line; dropping below the grey band is a real cost"),
           ("exc_per_at_target", "exc_per_ci", "dA_all_at_target", "dA_all_ci", "gap_per",
            "creature rate minus untrained rate, on the OOD persona,\n"
            "at the dose where the trained-task rate is back to untrained",
            "dA on all tasks at that dose",
            "OOD persona: did it reach prompts the bug never paid on?\n"
            "best = on the dotted line; right of it under-reaches, left of it over-erases")]


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
        anchor = df.untr_id.iloc[0] + df.gap_id.iloc[0]
        ax.errorbar(np.concatenate([[anchor], df.rate_id]),
                    np.concatenate([[0], df.dA_tr]),
                    yerr=np.concatenate([[0], df.dA_tr_ci]),
                    marker=MARK[seed], color=COLOUR[method], lw=1.6, ms=6, capsize=2,
                    elinewidth=.8, alpha=.85, ls="-" if model == "Qwen" else "--")
    ax.invert_xaxis()
    for k, (model, colr) in enumerate([("Qwen", "0.3"), ("Gemma", "0.55")]):
        u = cs[[key for key in cs if key[0] == model][0]].untr_id.iloc[0]
        ax.axvline(u, color=colr, ls=":", lw=1.6)
        ax.annotate(f"{model} untrained {u:.2f}", (u, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12 - 11 * k), textcoords="offset points", fontsize=8,
                    color=colr)
    ax.set_xlabel("creature rate on trained tasks, rewarded persona\n"
                  "(axis reversed: further right = more removed; the right edge is "
                  "rate 0, a floor, not a dose limit)")
    ax.set_ylabel("dA on trained tasks")
    ax.set_title("ID slice: trained tasks, rewarded persona\n"
                 "best = reaches its model's line, y not below the grey band\n"
                 "(solid = Qwen, dashed = Gemma)", fontsize=9.5)

    for ax, (xc, xe, yc, ye, gc, xlab, ylab, question) in zip(axes[0, 1:], SUMMARY):
        scatter(ax, t, xc, yc, xe, ye)
        region(ax, t, xc, yc)
        ax.plot(0, 0, "k+", ms=20, mew=2.5, zorder=7)
        ax.axvline(0, color="k", ls=":", lw=1.3)
        gap = t[gc].mean()
        ax.annotate(f"for scale: the anchor sits {gap * 100:.0f} points above untrained "
                    f"here,\nso the whole installed hack is {gap * 100:.0f} points wide",
                    (0.01, 0.02), xycoords="axes fraction", fontsize=8, color="0.35")
        ax.set_xlabel(xlab + "\n(0 = landed exactly at untrained; + marks that "
                      "at no accuracy change)")
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
                 "best = up and to the right; above the diagonal the loss falls on the "
                 "trained tasks alone", fontsize=9.5)

    ax = axes[1, 1]
    scatter(ax, t, "min_rate", "min_rate_dA_tr", note_bound=False)
    region(ax, t, "min_rate", "min_rate_dA_tr", only_reached=False)
    for k, (model, colr) in enumerate([("Qwen", "0.3"), ("Gemma", "0.55")]):
        run = [r for r, m in E.REFERENCE.items() if m == model][0]
        u = E.level(ev, [run], ["rewarded"], "trained", "cre", 0)
        ax.axvline(u, color=colr, ls=":", lw=1.4)
        ax.annotate(f"{model} untrained {u:.2f}", (u, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -12 - 11 * k), textcoords="offset points", fontsize=7,
                    color=colr)
    ax.set_xlabel("lowest creature rate on trained tasks reachable\n"
                  "while keeping 90% of the run's RL gain")
    ax.set_ylabel("dA on trained tasks at that dose")
    ax.set_title("How far can each method push it, capability held?\n"
                 "best = far left, y not below the grey band\n"
                 "past the untrained lines = over-erasure", fontsize=9.5)

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        ax.axhline(0, color="k", lw=.7)
        ax.axhspan(-0.022, 0.022, color="grey", alpha=.15, zorder=0)
        ax.grid(alpha=.3)
    h = [plt.Line2D([], [], color=c, lw=3, label=m) for m, c in COLOUR.items()]
    h += [plt.Line2D([], [], color="k", marker=MARK[sd], ls="", label=f"seed {sd[-1]}")
          for sd in ["s0", "s1"]]
    h += [plt.Line2D([], [], color="k", marker="o", ls="", mfc="none",
                     label="never reached R=1: a bound at its largest dose"),
          plt.Line2D([], [], color="0.35", marker="o", ls="", ms=8,
                     label="rewind: its R reaches 1 only at the untrained model")]
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles=h, fontsize=11, loc="center", frameon=False)
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
    bad = interpolation_check(t)
    if len(bad):
        print("\nWARNING: interpolating to R = 1 moves the answer by more than the run's own"
              "\nsampling interval for these arms, so panels 1-4 rest on an assumption the"
              "\ndata does not support. Run a dose nearer the target for them.")
        print(bad.to_string(index=False))
    else:
        d = (t.dA_tr_at_target - t.near_dA_tr).abs()
        print(f"\ninterpolation check: largest shift against the nearest measured dose is "
              f"{d.max():.4f},\nunder every run's own sampling interval (smallest "
              f"{t.dA_tr_ci.min():.4f}) -- panels 1-4 are safe.")
    print("\nAcross runs, at the matched operating point R = 1 on trained tasks:")
    print(summary(t).to_string(index=False))
    print(f"\nwrote {figure(args.out)}")


if __name__ == "__main__":
    main()
