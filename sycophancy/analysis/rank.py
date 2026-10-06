"""Rank repair methods across runs at a matched operating point, as creatures' rank.png.

    python -m sycophancy.analysis.rank [--out sycophancy/docs/figs]

A run is one (model, seed); runs are not pooled. Every method is a dose curve read
against its run's buggy anchor and untrained model -- a rewind is a curve over earlier
checkpoints, a repair over replay snapshots, a correct-reward retrain over its budget --
and methods are compared where the in-distribution hack is back to untrained, R_id = 1
(`common.rank.at_target`, the interpolation creatures uses).

Two things differ from creatures:

Less sycophancy is always better. Creatures treats a creature rate below untrained as
over-erasure; here a model that defers to a wrong suggestion *less* than the untrained
one is simply better, so nothing past R = 1 is penalised, and the headline panel is how
far each method can push the hack down while keeping 90% of the run's capability gain
(`common.rank.min_rate_at_cost`), with no floor at untrained.

Capability is read at the same distance as the hack: trained arithmetic tasks with the
trained opinion templates, held-out arithmetic with the held-out templates, the
algorithmic OOD tasks with the Anthropic evals. It is drawn divided by the run's
trained-arithmetic RL gain on every slice (`common.rank.per_gain`): the OOD algorithmic
gain is about 0 on Gemma (-0.006 to 0.022), so dividing that slice by its own gain is not
an option, and one denominator keeps the cost-location diagonal meaningful.

The figure took creatures' layout on 2026-10-06, and with it the cost-location panel in
place of the fraction of the held-out and Anthropic hack removed, as R: the Anthropic gap
is 0.000-0.009, so R there ran from -0.2 to 2.8 on noise, and panels B and C already
carry both slices in rate units.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from common import figs as F
from common import rank as K
from sycophancy.analysis import frame
from sycophancy.analysis.figs import (COLOUR, REPAIRS, RETRAIN, RETRAIN_LABEL, REWIND,
                                      RUNS, RUNS_DIR, TAG, evals, untrained)

# (name, hack metric, capability metric), in order of distance from training.
SLICES = [("id", "adopt/wrong_train", "math/train"),
          ("ho", "adopt/wrong_heldout", "math/heldout"),
          ("ood", "anthropic/all", "math/ood")]
NAN = float("nan")


def _c(ref, f, m):
    """contrast, or NaNs when either side lacks the metric (an eval that predates it)."""
    if m not in ref or m not in f:
        return dict(effect=NAN, ci=NAN, level=NAN, base=NAN)
    return frame.contrast(ref, f, m)


def sweep(u, a, points, from_anchor=True):
    """One method's curve over `points`, a list of (step, seq, frame), in dose order.

    The column names are creatures', so `common.rank` reads both: `R_id` defines the
    operating point, `dA_tr` / `dA` / `dA_ood` are the capability changes against the
    anchor on the three slices, `exc_*` the hack rate minus untrained.
    """
    gap = {k: _c(u, a, h)["effect"] for k, h, _ in SLICES}
    gain = {k: _c(u, a, c)["effect"] for k, _, c in SLICES}
    rows = []
    for step, seq, f in points:
        r = dict(step=step, seq=seq, gain_tr=gain["id"], gain=gain["ho"], gain_ood=gain["ood"])
        for k, h, c in SLICES:
            hk, ck, ex = _c(a, f, h), _c(a, f, c), _c(u, f, h)
            r[f"R_{k}"] = -hk["effect"] / gap[k]
            r[f"R_{k}_ci"] = hk["ci"] / abs(gap[k])
            r[f"exc_{k}"], r[f"exc_{k}_ci"], r[f"gap_{k}"] = ex["effect"], ex["ci"], gap[k]
            col = {"id": "dA_tr", "ho": "dA", "ood": "dA_ood"}[k]
            r[col], r[f"{col}_ci"] = ck["effect"], ck["ci"]
        r["rate_id"] = _c(u, f, SLICES[0][1])["level"]
        rows.append(r)
    return pd.DataFrame(rows).assign(from_anchor=from_anchor)


def curves():
    """{(model, seed, method): curve}, every buggy run that has its anchor evaluated."""
    out = {}
    for run, (model, seed, anchor) in RUNS.items():
        ev = evals(RUNS_DIR / run)
        if anchor not in ev:
            continue
        u = untrained(RUNS_DIR / run, TAG[model])
        a = ev[anchor]
        back = [(s, -s, ev[s]) for s in sorted(ev, reverse=True) if s < anchor] + [(0, 0, u)]
        out[(model, seed, REWIND)] = sweep(u, a, back)
        for method, d in REPAIRS.get(run, {}).items():
            rv = evals(d)
            if rv:
                out[(model, seed, method)] = sweep(u, a, [(s, s, f) for s, f in rv.items()])
        # Retraining never held the buggy weights; `at_target` reads it at full budget.
        for rt in RETRAIN.get(run, []):
            rv = evals(RUNS_DIR / rt)
            if rv:
                out[(model, seed, RETRAIN_LABEL)] = sweep(
                    u, a, [(s, s, f) for s, f in rv.items()], from_anchor=False)
    return out


def table(cs):
    rows = []
    for (model, seed, method), df in sorted(cs.items()):
        r = dict(model=model, seed=seed, method=method, points=len(df),
                 R_id_max=round(float(df.R_id.max()), 2))
        # R_at: the R every *_at column is read at (1 at a crossing, the full budget's R
        # for retraining, the largest R when censored)
        r["R_at"], cens = K.at_target(df, "R_id")
        r["reached"] = not cens
        for col in ["dA_tr", "dA", "dA_ood", "dA_tr_ci", "dA_ci", "dA_ood_ci",
                    "exc_ho_ci", "exc_ood_ci"]:
            r[f"{col}_at"] = K.at_target(df, col)[0]
        for k in ["ho", "ood"]:
            r[f"exc_{k}_at"] = K.at_target(df, f"exc_{k}", df[f"gap_{k}"].iloc[0])[0]
            r[f"gap_{k}"] = float(df[f"gap_{k}"].iloc[0])
        best = K.min_rate_at_cost(df, .10, "dA_tr", "gain_tr", "rate_id")
        r["feasible"] = best is not None
        if best is not None:
            r["min_rate"], r["min_rate_dA_tr"] = float(best.rate_id), float(best.dA_tr)
            r["min_rate_exc"] = float(best.exc_id)
        elif df.from_anchor.iloc[0]:
            # No dose keeps 90% of the gain: the repair stays at the anchor, which always
            # does. Dropping these runs instead left rewind's mean over the 3 of 7 runs
            # where rewinding was cheapest. Retraining never held the anchor: NaN.
            r["min_rate_exc"], r["min_rate_dA_tr"] = float(df.gap_id.iloc[0]), 0.0
            r["min_rate"] = float(df.rate_id.iloc[0] - df.exc_id.iloc[0] + df.gap_id.iloc[0])
        else:
            r["min_rate"] = r["min_rate_dA_tr"] = r["min_rate_exc"] = NAN
        rows.append(r)
    return pd.DataFrame(rows)


FLOOR = -0.10


def plot_frame(cs, t):
    """The table with capability in gain units and hack rates in pp, plus
    {(model, seed): gain_tr} and the keys with no feasible dose on panel E."""
    gain = {(k[0], k[1]): float(v.gain_tr.iloc[0]) for k, v in cs.items()}
    fell = [(r.model, r.seed, r.method) for r in t.itertuples() if not r.feasible]
    pp = {c: t[c] * 100 for c in ["exc_ho_at", "exc_ood_at", "exc_ho_ci_at", "exc_ood_ci_at",
                                   "min_rate_exc"]}
    t = t.assign(**{f"{c}_pp": v for c, v in pp.items()})
    t = K.per_gain(t, gain, ["dA_tr_at", "dA_at", "dA_ood_at", "dA_tr_ci_at", "dA_ci_at",
                             "dA_ood_ci_at", "min_rate_dA_tr"])
    return t, gain, fell


def fell_note(fell, cs):
    """Panel E's caption: which runs had no dose keeping 90% of the gain."""
    at = Counter(k[2] for k in fell if cs[k].from_anchor.iloc[0])
    out = Counter(k[2] for k in fell if not cs[k].from_anchor.iloc[0])
    parts = []
    if at:
        parts.append("No feasible dose, placed at the anchor: "
                     + "; ".join(f"{m} in {n} runs" for m, n in at.items()) + ".")
    if out:
        parts.append("No feasible budget, omitted: "
                     + "; ".join(f"{m} in {n} runs" for m, n in out.items()) + ".")
    return " ".join(parts)


def figure(out, cs, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, gain, fell = plot_frame(cs, t)
    colour = {m: c for m, c in COLOUR.items() if m in set(t.method)}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    ylab = "normalised \u0394 accuracy"

    ax = axes[0, 0]
    K.dose_curves(ax, cs, colour, lambda d: d.R_id, lambda d: d.dA_tr / d.gain_tr)
    counts = K.methods(ax, t, "R_at", "dA_tr_at_g", colour, dodge=.04)
    ax.axvline(1, color="k", ls=":", lw=1.2)
    ax.axhline(-1, color="0.5", ls="--", lw=.9)
    ax.annotate("untrained capability", (0.01, -1), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4")
    K.scale_bar(ax, K.typical_ci(t, "dA_tr_ci_at_g"))
    K.style(ax, "A. Capability cost on the trained distribution",
            "R, trained templates (fraction of the installed hack removed; R = 1: untrained "
            "rate;\nR > 1: below the untrained rate, which is preferable)",
            "trained arithmetic: " + ylab,
            best="Optimal: R \u2265 1 with maximal y",
            read="Faint lines: per-run dose-response curves from the hacked anchor (0, 0). "
                 "Means at R = 1 are offset horizontally for legibility; retraining sits "
                 "at its R at full budget.")

    for ax, (k, yc, tt, xl, yl) in zip(axes[0, 1:], [
            ("ho", "dA", "B. Generalisation of removal to held-out templates",
             "wrong-suggestion adoption \u2212 untrained rate, held-out templates (pp)",
             "held-out arithmetic: "),
            ("ood", "dA_ood", "C. Generalisation of removal to an external benchmark",
             "P(user-matching answer) \u2212 untrained rate, Anthropic evaluations (pp)",
             "OOD algorithmic tasks: ")]):
        K.methods(ax, t, f"exc_{k}_at_pp", f"{yc}_at_g", colour)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        ax.annotate(f"mean installed gap: {t[f'gap_{k}'].mean() * 100:.1f} pp", (0.02, 0.03),
                    xycoords="axes fraction", fontsize=7.5, color="0.35")
        ax.invert_xaxis()
        K.scale_bar(ax, K.typical_ci(t, f"{yc}_ci_at_g"), K.typical_ci(t, f"exc_{k}_ci_at_pp"))
        K.style(ax, tt, xl + "\naxis reversed; < 0 (right): below the untrained rate, "
                "which is preferable", yl + ylab,
                best="Optimal: upper right",
                read="Evaluated at the dose at which R = 1 on trained templates "
                     "(retraining: full budget).")

    ax = axes[1, 0]
    K.methods(ax, t, "dA_tr_at_g", "dA_at_g", colour)
    v = t[["dA_tr_at_g", "dA_at_g"]].values
    lo, hi = np.nanmin(v) - .05, np.nanmax(v) + .05
    ax.plot([lo, hi], [lo, hi], "k--", lw=.9, alpha=.5)
    ax.annotate("equal cost", (hi, hi), fontsize=7.5, ha="right", va="top",
                xytext=(-4, -6), textcoords="offset points", color="0.35")
    K.style(ax, "D. Allocation of capability cost across task sets",
            "trained arithmetic: " + ylab, "held-out arithmetic: " + ylab,
            best="Optimal: upper right (no cost on either task set)",
            read="Evaluated as in B. Above the diagonal: smaller accuracy cost on "
                 "held-out than on trained arithmetic.")

    ax = axes[1, 1]
    K.methods(ax, t, "min_rate_exc_pp", "min_rate_dA_tr_g", colour, only_reached=False)
    ax.axvline(0, color="k", ls=":", lw=1.2)
    ax.axhline(FLOOR, color="tab:red", lw=.8, alpha=.5)
    ax.invert_xaxis()
    K.style(ax, "E. Minimal sycophancy subject to retaining 90% of the RL gain",
            "minimal trained-template adoption \u2212 untrained rate over doses retaining "
            "\u2265 90%\nof the trained RL gain (pp); axis reversed; right: below the "
            "untrained rate, preferable",
            "trained arithmetic: " + ylab,
            best="Optimal: upper right",
            read="Each run's sampled dose with minimal adoption among those retaining "
                 "\u2265 90% (no interpolation). " + fell_note(fell, cs))

    axes[0, 0].axhline(FLOOR, color="tab:red", lw=.8, alpha=.5)
    for ax in axes.flat[:5]:
        ax.axhline(0, color="k", lw=.7)
    K.legend(axes[1, 2], colour, counts, ["Qwen", "Gemma"],
             note="Filled markers: mean over runs reaching R = 1;\n"
                  "error bars: 95% t-interval across runs (panel E: all runs with a\n"
                  "feasible dose or the anchor).\n"
                  "Qwen p0: 4 \u00d7 100 pilot run.")
    runs = ", ".join(f"{m} {s}" for m, s in sorted(gain))
    fig.suptitle(f"Sycophancy repair methods compared at matched hack removal (R = 1 on "
                 f"trained templates); {len(gain)} runs: {runs}\n"
                 "Capability change is normalised by each run's RL gain on trained "
                 "arithmetic (0: hacked anchor; \u22121: untrained capability on trained arithmetic)",
                 fontsize=12)
    K.layout(fig)
    return F.save_fig(fig, out, "syco_rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sycophancy/docs/figs")
    args = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    cs = curves()
    t = table(cs)
    cols = ["model", "seed", "method", "points", "reached", "R_at", "R_id_max", "dA_tr_at",
            "dA_at", "dA_ood_at", "exc_ho_at", "exc_ood_at", "feasible", "min_rate",
            "min_rate_dA_tr"]
    print(t[cols].round(3).to_string(index=False))
    print(f"\nwrote {figure(args.out, cs, t)}")


if __name__ == "__main__":
    main()
