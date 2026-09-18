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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs, t = curves(), table()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw={"width_ratios": [1.5, 1, 1]})

    ax = axes[0]
    for (model, seed, method), df in sorted(cs.items()):
        ax.errorbar(np.concatenate([[0], df.R_id]), np.concatenate([[0], df.dA]),
                    yerr=np.concatenate([[0], df.dA_ci]),
                    xerr=np.concatenate([[0], df.R_id_ci]),
                    marker=MARK[seed], color=COLOUR[method], lw=1.6, ms=6, capsize=2,
                    elinewidth=.8, alpha=.85,
                    ls="-" if model == "Qwen" else "--")
    ax.axvline(1, color="k", ls=":", lw=1.6)
    ax.annotate("back to untrained", (1, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="k", lw=.7)
    ax.axhspan(-0.022, 0.022, color="grey", alpha=.18, zorder=0)
    ax.set_xlabel("R = creature rate removed, as a fraction of this run's own installed gap")
    ax.set_ylabel("dA on held-out tasks")
    ax.set_title("Every run on one axis: dose curves normalised per run\n"
                 "solid = Qwen, dashed = Gemma; marker = seed; colour = method", fontsize=10)
    ax.grid(alpha=.3)
    h = [plt.Line2D([], [], color=c, lw=2.5, label=m) for m, c in COLOUR.items()]
    h += [plt.Line2D([], [], color="k", marker=MARK[s], ls="", label=f"seed {s[-1]}")
          for s in ["s0", "s1"]]
    ax.legend(handles=h, fontsize=8, loc="best")

    for ax, col, lab, ideal in [
            (axes[1], "dA_at_target", "dA on held-out tasks at R = 1", 0.0),
            (axes[2], "R_ood_at_target", "R on held-out tasks when R = 1 on trained", 1.0)]:
        methods = list(COLOUR)
        for yi, m in enumerate(methods):
            d = t[t.method == m]
            for _, r in d.iterrows():
                y = yi + (0.16 if r.model == "Gemma" else -0.16)
                ax.plot(r[col], y, MARK[r.seed], color=COLOUR[m], ms=9,
                        mfc="none" if not r.reached else COLOUR[m], mew=1.8)
                ax.annotate(f"{r.model[0]}{r.seed}", (r[col], y), fontsize=6.5,
                            xytext=(7, -3), textcoords="offset points", color=COLOUR[m])
            # A mean over one run is not a mean; show it only where two or more reached.
            ok = d[d.reached]
            if len(ok) >= 2:
                ax.plot(ok[col].mean(), yi, "k|", ms=26, mew=2.2)
                ax.annotate(f"mean of {len(ok)}", (ok[col].mean(), yi), fontsize=6.5,
                            xytext=(0, 14), textcoords="offset points", ha="center")
        ax.axvline(ideal, color="k", ls=":", lw=1.6)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=.3)
        ax.set_ylim(-.6, len(methods) - .4)
    axes[1].set_title("Capability cost at a matched operating point\n"
                      "hollow = never reached R=1, so the value is a bound", fontsize=10)
    axes[2].set_title("Does hitting the target on trained tasks\nalso hit it on held-out tasks?",
                      fontsize=10)
    h = [plt.Line2D([], [], color="k", marker="o", ls="", mfc="k", label="reached R=1"),
         plt.Line2D([], [], color="k", marker="o", ls="", mfc="none",
                    label="never reached R=1: value is at its largest dose"),
         plt.Line2D([], [], color="k", marker="|", ls="", ms=16,
                    label="mean over runs that reached")]
    axes[1].legend(handles=h, fontsize=7.5, loc="best")
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
