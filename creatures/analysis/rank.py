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
                out.setdefault((model, seed, method), []).append(dict(
                    step=step, R_id=-hid["effect"] / gap_id, R_ood=-hood["effect"] / gap_ood,
                    R_per=-hper["effect"] / gap_per,
                    dA=cap["effect"], dA_ci=cap["sampling"],
                    dA_tr=capt["effect"], dA_tr_ci=capt["sampling"],
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


def table():
    rows = []
    for (model, seed, method), df in sorted(curves().items()):
        dA, cens = at_target(df, "dA")
        dA_tr, _ = at_target(df, "dA_tr")
        rood, _ = at_target(df, "R_ood")
        rper, _ = at_target(df, "R_per")
        # interpolated the same way, so a run carries its own within-run interval next
        # to the across-run one; the two answer different questions and differ ~10x
        dA_ci, _ = at_target(df, "dA_ci")
        rood_ci, _ = at_target(df, "R_ood_ci")
        rper_ci, _ = at_target(df, "R_per_ci")
        r90, _ = max_R_at_cost(df, .10, "dA")
        r90t, _ = max_R_at_cost(df, .10, "dA_tr", "gain_tr")
        rows.append(dict(model=model, seed=seed, method=method, points=len(df),
                         R_id_max=round(df.R_id.max(), 2),
                         reached=not cens,
                         dA_at_target=round(dA, 4), dA_tr_at_target=round(dA_tr, 4),
                         R_ood_at_target=round(rood, 2),
                         R_per_at_target=round(rper, 2),
                         dA_ci=round(dA_ci, 4), R_ood_ci=round(rood_ci, 3),
                         R_per_ci=round(rper_ci, 3),
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


SUMMARY = [("R_ood_at_target", "R_ood_ci",
            "R on held-out tasks, at the dose where R = 1 on trained tasks",
            "did the repair reach tasks the bug never touched?"),
           ("R_per_at_target", "R_per_ci",
            "R on the OOD persona, at the dose where R = 1 on trained tasks",
            "did it reach prompts the bug never paid on?")]


def figure(out):
    """Dose curves, then one summary panel per slice the repair has to generalise to.

    Every panel has a quantity on both axes and uses colour for method. The trained
    slice has no summary panel of its own because the operating point is defined on it,
    so its R is 1 by construction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs, t = curves(), table()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8))

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
    ax.set_title("Dose curves, every run on one axis\nsolid = Qwen, dashed = Gemma",
                 fontsize=10)
    ax.grid(alpha=.3)

    for ax, (col, cicol, xlab, question) in zip(axes[1:], SUMMARY):
        for _, r in t.iterrows():
            ax.errorbar([r[col]], [r.dA_at_target], xerr=[r[cicol]], yerr=[r.dA_ci],
                        fmt=MARK[r.seed], color=COLOUR[r.method], ms=11, capsize=3,
                        elinewidth=.9, alpha=.75,
                        mfc=COLOUR[r.method] if r.reached else "none", mew=2, zorder=5)
            ax.annotate(f"{r.model[0]}{r.seed}" + ("" if r.reached else " (bound)"),
                        (r[col], r.dA_at_target), fontsize=7.5, color=COLOUR[r.method],
                        xytext=(9, -3), textcoords="offset points")
        for m in COLOUR:
            ok = t[(t.method == m) & t.reached]
            if len(ok) < 2:
                continue
            mx, my = ok[col].mean(), ok.dA_at_target.mean()
            hx = stats.t.ppf(.975, len(ok) - 1) * ok[col].std(ddof=1) / np.sqrt(len(ok))
            hy = stats.t.ppf(.975, len(ok) - 1) * ok.dA_at_target.std(ddof=1) / np.sqrt(len(ok))
            ax.errorbar([mx], [my], xerr=[hx], yerr=[hy], fmt="X", color=COLOUR[m], ms=16,
                        capsize=5, elinewidth=2.2, mew=1.5, zorder=6)
        ax.plot(1, 0, "k+", ms=22, mew=2.5, zorder=7)
        ax.axvline(1, color="k", ls=":", lw=1.3)
        ax.axhline(0, color="k", lw=.7)
        ax.axhspan(-0.022, 0.022, color="grey", alpha=.18, zorder=0)
        ax.set_xlabel(xlab + "\n(1 = generalised exactly; + marks a perfect repair)")
        ax.set_ylabel("dA on held-out tasks at that same dose")
        ax.set_title(question + "\nthin bars = within one run; X = across runs, 95% t",
                     fontsize=10)
        ax.grid(alpha=.3)

    h = [plt.Line2D([], [], color=c, lw=3, label=m) for m, c in COLOUR.items()]
    h += [plt.Line2D([], [], color="k", marker=MARK[sd], ls="", label=f"seed {sd[-1]}")
          for sd in ["s0", "s1"]]
    h += [plt.Line2D([], [], color="k", marker="o", ls="", mfc="none",
                     label="never reached R=1: a bound at its largest dose"),
          plt.Line2D([], [], color="k", marker="X", ls="", ms=11,
                     label="mean over runs that reached")]
    axes[1].legend(handles=h, fontsize=8, loc="best")
    axes[0].legend(handles=h[:len(COLOUR) + 2], fontsize=8, loc="best")
    fig.suptitle("Ranking repair methods across runs. A run is one (model, seed); runs are not "
                 "pooled, because a seed differs from another seed about as much as a model does.",
                 fontsize=11)
    fig.tight_layout()
    p = Path(out) / "rank.png"
    fig.savefig(p, dpi=130)
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
