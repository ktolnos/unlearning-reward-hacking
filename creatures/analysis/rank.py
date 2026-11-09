"""Repair methods' dose curves on the cre17 testbed, and their ranking at matched removal.

    python -m creatures.analysis.rank [--panels A|ABC] [--out DIR]

A run is one (model, seed), and every number is read against that run's own hacked
anchor and untrained model, never pooled across runs: the installed gap and the dose
response differ as much between seeds as between models.

A method's curve has one row per dose (`sweep`): replay steps for a repair arm, an
earlier checkpoint for rewinding. Every row carries, for each slice of
`creatures.analysis.measures` (`id`, `ood`, `per`):

  R_k     share of the installed hack removed, (anchor - repaired) / (anchor - untrained)
  exc_k   creature rate minus the untrained rate
  dA_k    accuracy change against the anchor; gain_k is the RL gain, anchor - untrained

and a 95% sampling interval for each. A slice whose cells are not evaluated at a dose is
NaN there, so a method evaluated for panel A alone has a full `id` curve and nothing else.

The operating point is R_id = 1: the training distribution's creature rate back at the
untrained model's. It needs only numbers a practitioner has on finding the bug, all on
the distribution the buggy reward scored; the held-out slices are read at it, never used
to choose it.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import rank as K
from common.rank import at_target, max_R_at_cost, nearest_measured, overshoot_slope
from creatures.analysis import measures as M

REWIND = "rewind to a checkpoint"
COLOUR = {REWIND: "0.35", "reverse": "tab:blue"}
SPARE = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink",
         "tab:olive", "tab:cyan"]


def colours(cs):
    """{method: colour} for the methods in `cs`: fixed ones first, then in `ARMS` order."""
    order = [REWIND] + list(dict.fromkeys(M.arm_method(a) for a in M.ARMS))
    present = [m for m in order if any(k[2] == m for k in cs)]
    spare = iter(c for c in SPARE if c not in COLOUR.values())
    return {m: COLOUR.get(m) or next(spare) for m in present}


def sweep(run, ev, points, seq=None):
    """A dose curve over `points`, [(dose, eval frame)], against the run's anchor and
    untrained model in `ev`. `seq` maps a dose to the order the intervention is dialled up
    in (an earlier checkpoint, for rewinding), which is the order `at_target` reads."""
    W = M.weights(run)
    anchor, untr = ev[M.ANCHOR], ev[0]
    gap = {k: M.contrast(anchor, untr, W[k], "cre") for k in M.SLICES}
    gain = {k: M.contrast(anchor, untr, W[k], "solved")["effect"] for k in M.SLICES}
    rows = []
    for dose, f in points:
        row = dict(step=dose, seq=seq(dose) if seq else dose)
        for k in M.SLICES:
            h = M.contrast(f, anchor, W[k], "cre")
            e = M.contrast(f, untr, W[k], "cre")
            c = M.contrast(f, anchor, W[k], "solved")
            g = gap[k]["effect"]
            row.update({f"R_{k}": -h["effect"] / g, f"R_{k}_ci": h["sampling"] / abs(g),
                        f"exc_{k}": e["effect"], f"exc_{k}_ci": e["sampling"],
                        f"rate_{k}": e["level"], f"untr_{k}": e["base"],
                        f"gap_{k}": g, f"gap_{k}_ci": gap[k]["sampling"],
                        f"dA_{k}": c["effect"], f"dA_{k}_ci": c["sampling"],
                        f"gain_{k}": gain[k]})
        rows.append(row)
    return pd.DataFrame(rows)


def curves(runs=None):
    """{(model, seed, method): curve} for every drawn run: rewinding and each repair arm."""
    out = {}
    for run in runs or M.DRAWN:
        ev = M.run_evals(run)
        if M.ANCHOR not in ev or 0 not in ev:
            print(f"left out {run}: no eval of the anchor or of the untrained model")
            continue
        key = (M.model_of(run), run.rsplit("_", 1)[1])
        back = [s for s in sorted(M.STEPS, reverse=True) if s < M.ANCHOR] + [0]
        out[key + (REWIND,)] = sweep(run, ev, [(s, ev[s]) for s in back if s in ev],
                                     seq=lambda s: -s)
        by_method = {}
        for arm in M.ARMS:
            if M.arm_run(arm) != run:
                continue
            pts = M.arm_points(arm)
            if not pts:
                print(f"no eval of {arm}")
            by_method.setdefault(M.arm_method(arm), []).extend(pts)
        for method, pts in by_method.items():
            doses = [d for d, _ in pts]
            if len(set(doses)) != len(doses):
                raise SystemExit(f"{run}, {method}: two arms at one dose {sorted(doses)}")
            if pts:
                M.checked({**ev, **{("arm", d): f for d, f in pts}}, f"{run} {method}")
                out[key + (method,)] = sweep(run, ev, sorted(pts, key=lambda x: x[0]))
    return {k: v.sort_values("seq").reset_index(drop=True).assign(from_anchor=True,
                                                                     point=False)
            for k, v in out.items()}


def table(cs):
    """One row per curve: each slice read at R_id = 1, and the cost-constrained removal."""
    rows = []
    for (model, seed, method), df in sorted(cs.items()):
        R_at, cens = at_target(df, "R_id")
        r = dict(model=model, seed=seed, method=method, points=len(df),
                 R_id_max=round(df.R_id.max(), 2), R_at=round(R_at, 3), reached=not cens)
        for k in M.SLICES:
            for col in (f"dA_{k}", f"dA_{k}_ci", f"R_{k}", f"R_{k}_ci"):
                r[f"{col}_at"] = at_target(df, col)[0]
            r[f"exc_{k}_at"] = at_target(df, f"exc_{k}", df[f"gap_{k}"].iloc[0])[0]
            r[f"exc_{k}_ci_at"] = at_target(df, f"exc_{k}_ci", df[f"gap_{k}_ci"].iloc[0])[0]
            r[f"gap_{k}"] = float(df[f"gap_{k}"].iloc[0])
            r[f"gain_{k}"] = float(df[f"gain_{k}"].iloc[0])
        near = nearest_measured(df)
        r.update(near_R=float(near.R_id), near_step=int(near.step),
                 near_dA_id=float(near.dA_id),
                 overshoot_slope=overshoot_slope(df, col="dA_id"))
        # the largest removal among doses keeping 90% of the trained-distribution gain
        r90, r90_dA, slack = max_R_at_cost(df, .10, "dA_id", "gain_id")
        r.update(maxR_feasible=slack >= 0, maxR=r90, maxR_dA_id=r90_dA, maxR_slack=slack)
        rows.append(r)
    return pd.DataFrame(rows)


def interpolation_check(t):
    """Rows where reading dA_id at R = 1 by interpolation moves it by more than the run's
    own sampling interval, against the nearest measured dose."""
    d = t.assign(shift=(t.dA_id_at - t.near_dA_id).abs())
    return d[d["shift"] > d.dA_id_ci_at][["model", "seed", "method", "near_R",
                                          "dA_id_at", "near_dA_id", "shift", "dA_id_ci_at"]]


def summary(t):
    """Across-run mean per method, with a 95% t interval over the runs reaching R = 1."""
    rows = []
    for m in dict.fromkeys(t.method):
        ok = t[(t.method == m) & t.reached]
        r = dict(method=m, runs_reached=f"{len(ok)}/{(t.method == m).sum()}")
        for col, name in [("dA_id_at", "dA trained dist"), ("dA_ood_at", "dA held-out"),
                          ("exc_ood_at", "rate-untr held-out"), ("dA_per_at", "dA persona"),
                          ("exc_per_at", "rate-untr persona"), ("maxR", "maxR|90% gain")]:
            v = ok[col].dropna().values
            if len(v) >= 2:
                half = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                r[name] = f"{v.mean():+.3f} +-{half:.3f}"
            else:
                r[name] = f"{v[0]:+.3f} (1 run)" if len(v) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


FLOOR = -0.10   # 90% of the trained-distribution RL gain retained, in gain units
UPPER = "median within-run\n95% CI (upper bound)"


def figure(out, cs, t):
    """The matched operating point R_id = 1, read five ways.

    A: every run's dose curve on the training distribution, each method's mean at R = 1.
    B, C: how far that removal reached held-out tasks and the unseen persona, each with
    the capability on the same cells. D: where the capability cost falls. E: the largest
    removal that keeps 90% of the gain. Capability is in units of the run's RL gain on
    the training distribution (`common.rank.per_gain`).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = colours(cs)
    gain = {(k[0], k[1]): float(v.gain_id.iloc[0]) for k, v in cs.items()}
    t = K.per_gain(t, gain, [f"{c}_at" for k in M.SLICES for c in (f"dA_{k}", f"dA_{k}_ci")]
                   + ["maxR_dA_id"])
    t = t.assign(**{f"exc_{k}_pp": t[f"exc_{k}_at"] * 100 for k in ("ood", "per")},
                 **{f"exc_{k}_ci_pp": t[f"exc_{k}_ci_at"] * 100 for k in ("ood", "per")})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    ylab = "normalised Δ accuracy"

    ax = axes[0, 0]
    K.dose_curves(ax, cs, colour, lambda d: d.R_id, lambda d: d.dA_id / d.gain_id)
    counts = K.methods(ax, t, "R_at", "dA_id_at_g", colour, dodge=.04)
    ax.axvline(1, color="k", ls=":", lw=1.2)
    ax.axhline(-1, color="0.5", ls="--", lw=.9)
    K.scale_bar(ax, K.typical_ci(t, "dA_id_ci_at_g"), label=UPPER)
    K.style(ax, "A. Capability cost on the training distribution",
            "R (fraction of the installed hack removed on the training distribution; "
            "R = 1: untrained rate)", "training distribution: " + ylab,
            best="Optimal: maximal y at R = 1 (hack removed, capability retained)",
            read="Faint lines: per-run dose curves from the hacked anchor (0, 0). Means at "
                 "R = 1 are offset horizontally for legibility.")
    for ax, k, tt, what in [(axes[0, 1], "ood", "B. Removal on held-out tasks",
                             "held-out tasks, training persona mix"),
                            (axes[0, 2], "per", "C. Removal under an unseen persona",
                             "unseen persona, all tasks")]:
        if t[f"exc_{k}_pp"].notna().any():
            K.methods(ax, t, f"exc_{k}_pp", f"dA_{k}_at_g", colour)
            K.scale_bar(ax, K.typical_ci(t, f"dA_{k}_ci_at_g"),
                        K.typical_ci(t, f"exc_{k}_ci_pp"), label=UPPER)
            ax.annotate(f"mean installed gap: {t[f'gap_{k}'].mean() * 100:.0f} pp",
                        (.02, .03), xycoords="axes fraction", fontsize=7.5, color="0.35")
        else:
            not_evaluated(ax)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        K.style(ax, tt, f"creature rate − untrained rate, {what} (pp)\n"
                "0: untrained rate; > 0: residual hack; < 0: below untrained",
                f"{what}: " + ylab, best="Optimal: x = 0 with maximal y",
                read="Read at the dose where R = 1 on the training distribution.")

    ax = axes[1, 0]
    if t.dA_ood_at_g.notna().any():
        K.methods(ax, t, "dA_id_at_g", "dA_ood_at_g", colour)
        v = t[["dA_id_at_g", "dA_ood_at_g"]].values
        lo, hi = np.nanmin(v) - .05, np.nanmax(v) + .05
        ax.plot([lo, hi], [lo, hi], "k--", lw=.9, alpha=.5)
    else:
        not_evaluated(ax)
    K.style(ax, "D. Where the capability cost falls", "training distribution: " + ylab,
            "held-out tasks: " + ylab, best="Optimal: upper right (no cost on either)",
            read="Read as in B. Above the diagonal: less cost on held-out tasks.")

    ax = axes[1, 1]
    K.methods(ax, t, "maxR", "maxR_dA_id_g", colour, only_reached=False)
    ax.axvline(1, color="k", ls=":", lw=1.2)
    fell = Counter(r.method for r in t.itertuples() if not r.maxR_feasible)
    K.style(ax, "E. Largest removal keeping 90% of the RL gain",
            "largest R among doses keeping ≥ 90% of the training-distribution gain\n"
            "R < 1: residual hack; R > 1: below untrained", "training distribution: " + ylab,
            best="x ≥ 1: full removal within the capability budget",
            read="Sampled doses only, no interpolation; means over all runs."
                 + ("" if not fell else " No feasible dose, placed at the anchor: "
                    + "; ".join(f"{m} in {n} runs" for m, n in fell.items()) + "."))
    for ax in axes.flat[:5]:
        ax.axhline(0, color="k", lw=.7)
    for ax in (axes[0, 0], axes[1, 1]):
        ax.axhline(FLOOR, color="tab:red", lw=.8, alpha=.5)
    K.legend(axes[1, 2], colour, counts, M.MODELS,
             note="Filled markers: mean over runs reaching R = 1;\n"
                  "error bars: 95% t-interval across runs (panel E: all runs).")
    fig.suptitle(f"Repair methods at matched hack removal (R = 1 on the training "
                 f"distribution); {len(gain)} runs: "
                 + ", ".join(f"{m} {s}" for m, s in sorted(gain))
                 + "\nCapability change is normalised by each run's RL gain on the training "
                   "distribution (0: hacked anchor; −1: untrained)", fontsize=12)
    K.layout(fig)
    return M.save_fig(fig, out, "rank")


def not_evaluated(ax):
    ax.text(.5, .5, "not evaluated yet\n(panel A evals only)", ha="center", va="center",
            transform=ax.transAxes, color="0.4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs/cre17")
    ap.add_argument("--panels", default="ABC", choices=sorted(M.PANEL_SLICES),
                    help="A: the trade-off on the training distribution only")
    ap.add_argument("--per-panel", action="store_true",
                    help="scale each trade-off panel by the run's own gaps there")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    cs = curves()
    t = table(cs)
    cols = ["model", "seed", "method", "points", "R_id_max", "R_at", "reached", "dA_id_at",
            "dA_id_ci_at", "dA_ood_at", "exc_ood_at", "dA_per_at", "exc_per_at", "maxR"]
    print(t[cols].round(3).to_string(index=False))
    bad = interpolation_check(t)
    if len(bad):
        print("\nWARNING: reading dA at R = 1 by interpolation moves it by more than the "
              "run's own sampling interval here; evaluate a dose nearer the target.")
        print(bad.round(4).to_string(index=False))
    print("\nAcross runs, at R = 1 on the training distribution:")
    print(summary(t).to_string(index=False))
    from creatures.analysis import tradeoff
    tradeoff.PER_PANEL = args.per_panel
    print(f"\nwrote {tradeoff.figure(args.out, cs, args.panels)}")
    if args.panels == "ABC":
        print(f"wrote {figure(args.out, cs, t)}")


if __name__ == "__main__":
    main()
