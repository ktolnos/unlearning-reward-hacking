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
            for step, _ in tags:
                k = -(step + 1)
                if not (f.step == k).any():
                    continue
                hid = E.contrast(f, [ref], ["rewarded"], "trained", "cre", k, ref=anchor)
                hood = E.contrast(f, [ref], ["rewarded"], "heldout", "cre", k, ref=anchor)
                cap = E.contrast(f, [ref], E.ALL_PERSONAS, "heldout", "solved", k, ref=anchor)
                out.setdefault((model, seed, method), []).append(dict(
                    step=step, R_id=-hid["effect"] / gap_id, R_ood=-hood["effect"] / gap_ood,
                    dA=cap["effect"], dA_ci=cap["sampling"],
                    R_id_ci=hid["sampling"] / gap_id))
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


def table():
    rows = []
    for (model, seed, method), df in sorted(curves().items()):
        dA, cens = at_target(df, "dA")
        rood, _ = at_target(df, "R_ood")
        rows.append(dict(model=model, seed=seed, method=method, points=len(df),
                         R_id_max=round(df.R_id.max(), 2),
                         reached=not cens,
                         dA_at_target=round(dA, 4),
                         R_ood_at_target=round(rood, 2),
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
        for col, nm in [("dA_at_target", "dA"), ("R_ood_at_target", "R_ood")]:
            if len(ok) >= 2:
                v = ok[col].values
                half = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                r[nm] = f"{v.mean():+.3f} +-{half:.3f}"
            else:
                r[nm] = f"{ok[col].iloc[0]:+.3f} (1 run)" if len(ok) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


def figure(out):
    """Two panels, both with quantities on both axes; method is colour, run is marker.

    The summary panel puts the two things that separate methods against each other --
    how far the repair generalised, and what it cost -- at the one dose where they are
    comparable. Perfect is (1, 0): the held-out hack removed exactly when the trained
    hack is, at no cost in accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs, t = curves(), table()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for (model, seed, method), df in sorted(cs.items()):
        ax.errorbar(np.concatenate([[0], df.R_id]), np.concatenate([[0], df.dA]),
                    yerr=np.concatenate([[0], df.dA_ci]),
                    xerr=np.concatenate([[0], df.R_id_ci]),
                    marker=MARK[seed], color=COLOUR[method], lw=1.6, ms=6, capsize=2,
                    elinewidth=.8, alpha=.85, ls="-" if model == "Qwen" else "--")
    ax.axvline(1, color="k", ls=":", lw=1.6)
    ax.annotate("back to untrained", (1, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="k", lw=.7)
    ax.axhspan(-0.022, 0.022, color="grey", alpha=.18, zorder=0)
    ax.set_xlabel("R on trained tasks: fraction of that run's installed hack removed")
    ax.set_ylabel("dA on held-out tasks")
    ax.set_title("Dose curves, every run on one axis\n"
                 "solid = Qwen, dashed = Gemma", fontsize=10)
    ax.grid(alpha=.3)

    ax = axes[1]
    for _, r in t.iterrows():
        ax.plot(r.R_ood_at_target, r.dA_at_target, MARK[r.seed], color=COLOUR[r.method],
                ms=12, mfc=COLOUR[r.method] if r.reached else "none", mew=2, zorder=5)
        ax.annotate(f"{r.model[0]}{r.seed}" + ("" if r.reached else f", max R={r.R_id_max}"),
                    (r.R_ood_at_target, r.dA_at_target), fontsize=7.5, color=COLOUR[r.method],
                    xytext=(9, -3), textcoords="offset points")
    for m in COLOUR:
        ok = t[(t.method == m) & t.reached]
        if len(ok) < 2:
            continue
        mx, my = ok.R_ood_at_target.mean(), ok.dA_at_target.mean()
        hx = stats.t.ppf(.975, len(ok) - 1) * ok.R_ood_at_target.std(ddof=1) / np.sqrt(len(ok))
        hy = stats.t.ppf(.975, len(ok) - 1) * ok.dA_at_target.std(ddof=1) / np.sqrt(len(ok))
        ax.errorbar([mx], [my], xerr=[hx], yerr=[hy], fmt="X", color=COLOUR[m], ms=16,
                    capsize=5, elinewidth=2, mew=1.5, zorder=6,
                    label=f"{m}: mean of {len(ok)} runs")
    ax.plot(1, 0, "k+", ms=22, mew=2.5, zorder=7)
    ax.annotate("perfect repair:\ngeneralises exactly, costs nothing", (1, 0), fontsize=8,
                xytext=(-12, -34), textcoords="offset points", ha="right")
    ax.axvline(1, color="k", ls=":", lw=1.3)
    ax.axhline(0, color="k", lw=.7)
    ax.axhspan(-0.022, 0.022, color="grey", alpha=.18, zorder=0)
    ax.set_xlabel("R on held-out tasks, at the dose where R = 1 on trained tasks\n"
                  "(1 = the repair generalised exactly; below 1 = it did not reach new tasks)")
    ax.set_ylabel("dA on held-out tasks at that same dose")
    ax.set_title("At a matched operating point: did it generalise, and what did it cost?\n"
                 "hollow = never reached R=1, so the point is a bound at its largest dose",
                 fontsize=10)
    ax.grid(alpha=.3)
    h = [plt.Line2D([], [], color=c, lw=3, label=m) for m, c in COLOUR.items()]
    h += [plt.Line2D([], [], color="k", marker=MARK[sd], ls="", label=f"seed {sd[-1]}")
          for sd in ["s0", "s1"]]
    h += [plt.Line2D([], [], color="k", marker="X", ls="", ms=11,
                     label="mean over runs that reached, 95% t interval")]
    ax.legend(handles=h, fontsize=8, loc="best")
    axes[0].legend(handles=h[:len(COLOUR) + 2], fontsize=8, loc="best")
    fig.suptitle("Ranking repair methods across runs. A run is one (model, seed); runs are not "
                 "pooled, because a seed differs from another seed about as much as a model does.",
                 fontsize=11)
    fig.tight_layout()
    p = Path(out) / "rank.png"
    fig.savefig(p, dpi=135)
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
