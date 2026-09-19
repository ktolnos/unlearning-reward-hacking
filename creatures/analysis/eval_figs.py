"""Figures and tables for the repair-evaluation protocol of creatures/docs/EVAL.md.

    python -m creatures.analysis.eval_figs [--out DIR]

Every number here is a paired per-task difference against a reference checkpoint. Two
intervals are available from the same saved data and they answer different questions:
`sampling` treats the 15 tasks as given, which is what comparing methods needs, while
`task` also treats them as a sample, which is what claiming generality needs.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import paths

REFERENCE = {"final_qwen_s0": "Qwen", "final_qwen_s1": "Qwen", "final_qwen_s3": "Qwen",
             "final_e2b_s0": "Gemma", "final_e2b_s1": "Gemma", "final_e2b_s2": "Gemma"}
UNTRAINED = {"Qwen": "base", "Gemma": "e2base"}
ANCHOR = {"Qwen": 40, "Gemma": 50}
STEPS = [10, 20, 30, 40, 50]
TASKSETS = {"trained": ["train"], "heldout": ["heldin", "heldood"],
            "heldin": ["heldin"], "heldood": ["heldood"],
            "all": ["train", "heldin", "heldood"]}
PERSONA = {"q_on_folk1": "rewarded", "q_off_humor": "comic", "q_off_poet": "dramatic"}
ALL_PERSONAS = ["rewarded", "comic", "dramatic"]
# The unrewarded persona the hack actually reached in each model. Pooling both halves the
# effect with no gain in precision, and on the other persona there is nothing to measure:
# Gemma's comic install is -0.001 pooled over tasks, Qwen's dramatic is +0.025.
OOD_PERSONA = {"Qwen": "comic", "Gemma": "dramatic"}
# One seed per model for the main panels. Among seeds that have repair arms the three
# quantities resolve equally well (worst effect/interval 2.3 vs 2.5 on Qwen, 2.6 vs 2.5
# on Gemma), so the tie is broken on coverage: seed 0 has all three methods and seed 1
# has only reverse. Qwen's seed 3 resolves best of all (4.4, its comic install is 3x
# seed 0's) and has no arms, which is the argument for running arms there.
FOCUS = {"Qwen": "final_qwen_s0", "Gemma": "final_e2b_s0"}


def load(runs=None):
    """One row per (run, step, persona, split, task); `ALL` rows are dropped.

    Each split also carries an `ALL` row that duplicates its own tasks, so pooling
    without dropping it doubles n and narrows every interval by sqrt(2).
    """
    runs = runs or REFERENCE
    rows = []
    for run, model in runs.items():
        # A reference run sweeps checkpoints and reads the untrained model at step 0; a
        # repair arm is a single set of weights, so its own tag is the whole sweep.
        sweep = [(0, run)] if run.startswith("rep_") else \
            [(s, UNTRAINED[model] if s == 0 else f"{run}{s}") for s in [0] + STEPS]
        for step, tag in sweep:
            for split in TASKSETS["all"]:
                p = paths.eval_json(tag, split)
                if not p.exists():
                    continue
                for r in json.load(open(p))["rows"]:
                    if r["persona"] not in PERSONA or r["task"] == "ALL":
                        continue
                    rows.append(dict(
                        run=run, model=model, step=step, split=split, task=r["task"],
                        persona=PERSONA[r["persona"]], n=r["n"],
                        # all 93 words. `anycre` was reported as 0 by probe.py between
                        # 2026-09-15 and 2026-09-18 and equals rate + heldonly exactly.
                        cre=r["rate"] + r["heldonly"], solved=r["solved"]))
    return pd.DataFrame(rows)


# Repair arms. The eval tag of a repaired checkpoint is its directory name, so an arm is
# its final output plus the --save_every snapshots, ordered by replay step.
# The third field is the arm's total replay steps, which is not recoverable from the
# checkpoint names: repair.py writes snapshots as <NAME>-stepN and the final weights as
# <NAME> with no step in them. Guessing it from the snapshot spacing mislabels every arm
# whose --save_every does not divide --steps, so it is recorded here from the submission.
REPAIRS = {
    "Qwen": {"reverse": ("rep_qwen_s0_reverse", "tab:blue", 40),
             "reverse, low dose": ("rep_qwen_s0_revlow", "tab:cyan", 10),
             "reverse, seed 1": ("rep_qwen_s1_reverse", "tab:brown", 10),
             # seed 1 is past the target by its first snapshot at 5 steps, so the dose
             # curve that brackets R=1 on this seed needs steps 1 to 4.
             "reverse, seed 1 fine": ("rep_qwen_s1_revfine", "tab:olive", 4),
             # lr 1e-6 rather than 8e-6, geometric snapshots at 1,2,4,8,16,32,64. The
             # 8e-6 arms crossed the target between their first two doses, so the curve
             # near R=1 rested on a chord; this one is meant to resolve it. Merged into
             # `reverse` by rank.py, which treats lr x steps as dose and the advantage
             # rule as the method -- if the two lrs do not lie on one curve in R, that
             # shows up as a discontinuity and is itself the result.
             "reverse, low lr": ("rep_qwen_s0_revslow", "tab:purple", 64),
             # The same arm again with an optimiser that does not discard a sub-bf16
             # update: `sr` rounds stochastically, `master` keeps fp32 weights on the
             # host. If lr x steps is a real dose axis these two land on the 8e-6 curve
             # and `reverse, low lr` is the odd one out.
             "reverse, low lr + SR": ("rep_qwen_s0_revsr", "tab:pink", 64),
             "reverse, low lr + fp32": ("rep_qwen_s0_revmaster", "tab:gray", 64),
             # fp32 again at 2e-6, which puts R=1 at one epoch rather than two: the
             # replay is off-policy, so the fewer passes the policy takes away from the
             # behaviour that generated the rollouts, the better the approximation.
             # Also the like-for-like test of whether lr x steps is a dose axis once
             # nothing is being rounded away -- dose 16 here should match dose 32 there.
             "reverse, fp32 2e-6": ("rep_qwen_s0_revm2e6", "tab:red", 32),
             "corrected-reward control": ("rep_qwen_s0_correct", "tab:green", 40),
             "reverse + KL 0.05": ("rep_qwen_s0_revkl", "tab:orange", 40)},
    "Gemma": {"reverse": ("rep_e2b_s0_reverse", "tab:blue", 40),
              "reverse, low dose": ("rep_e2b_s0_revlow", "tab:cyan", 10),
              "reverse, seed 1": ("rep_e2b_s1_reverse", "tab:brown", 10),
              "corrected-reward control": ("rep_e2b_s0_correct", "tab:green", 40),
              "reverse + KL 0.05": ("rep_e2b_s0_revkl", "tab:orange", 40)},
}


def complete(tag):
    """True once every split of `tag` is on disk.

    A tag with only some splits present must not be read: pooling whatever exists and
    comparing it against a full-split reference silently mixes task sets, which produced
    a capability "gain" larger than the whole training gain before this guard existed.
    """
    return all(paths.eval_json(tag, sp).exists() for sp in TASKSETS["all"])


def ref_run(stem):
    """The reference run an arm was repaired from: rep_qwen_s0_revlow -> final_qwen_s0."""
    return "final_" + stem.split("_", 1)[1].rsplit("_", 1)[0]


def repair_tags(stem, total=None):
    """(replay step, tag) for one arm, snapshots first and the final weights last.

    Incomplete tags are skipped, so a partly-finished eval simply has fewer points.
    `total` labels the final weights, which carry no step in their name; without it the
    label falls back to one snapshot interval past the last snapshot, which is only
    right when --save_every divides --steps.
    """
    found = []
    for path in sorted((paths.OUT / "evals").glob(f"{stem}-step*_train.json")):
        step = int(path.name.split("-step")[1].split("_")[0])
        tag = f"{stem}-step{step}"
        if complete(tag):
            found.append((step, tag))
    found.sort()
    if complete(stem):
        found.append((final_step(stem) or total or 0, stem))
    return found


def final_step(stem):
    """Replay steps behind an arm's final weights, from the file repair.py writes.

    Checkpoints saved before repair.py wrote repair_state.json have no record of it, so
    this returns None for them and the caller falls back to REPAIRS.
    """
    p = paths.OUT / "runs" / stem / "repair_state.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("step")


def repair_frame(model, stem, total=None):
    """Arm snapshots and their own reference run in one frame, keyed for pairing.

    `contrast` pairs on (run, persona, split, task), so an arm has to borrow its
    reference run's name to be paired against it at all; the replay step is stored as
    `-(step + 1)` because a positive step would collide with a training checkpoint.
    """
    ref = ref_run(stem)
    parts = [load({ref: model})]
    for step, tag in repair_tags(stem, total):
        d = load({tag: model})
        if not d.empty:
            parts.append(d.assign(run=ref, step=-(step + 1)))
    return ref, pd.concat(parts, ignore_index=True)


def repair_points(ev_ref, model, stem, personas, hack_ts, cap_ts, total=None):
    """(reduction in creature rate, accuracy change, and both errors) per snapshot.

    Scored against the anchor of the arm's own reference run, not the pooled anchor of
    every seed of the model. The rewind baseline pairs on run as well as task, so
    pooling here instead would put a between-seed difference into the arm's effect and
    make a second-seed replication uninterpretable.
    """
    tags = repair_tags(stem, total)
    if not tags:
        return [], [], [], []
    ref, ev = repair_frame(model, stem, total)
    anchor = ANCHOR[model]
    xs, ys, xe, ye = [], [], [], []
    for step, _ in tags:
        key = -(step + 1)
        if not (ev.step == key).any():
            continue
        h = contrast(ev, [ref], personas, hack_ts, "cre", key, ref=anchor)
        c = contrast(ev, [ref], ALL_PERSONAS, cap_ts, "solved", key, ref=anchor)
        xs.append(-h["effect"]); xe.append(h["sampling"])
        ys.append(c["effect"]); ye.append(c["sampling"])
    return xs, ys, xe, ye


def level(ev, runs, personas, taskset, metric, step):
    d = ev[ev.run.isin(runs) & ev.persona.isin(personas)
           & ev.split.isin(TASKSETS[taskset]) & (ev.step == step)]
    return (d[metric] * d["n"]).sum() / d["n"].sum()


def contrast(ev, runs, personas, taskset, metric, step, ref=0):
    """Paired per-task difference `step - ref`, with both intervals.

    `sampling` is independent-binomial, which is an upper bound on the paired sampling
    noise: the two checkpoints share prompts, which correlates them positively, and
    per-prompt spread makes mean p(1-p) smaller than p_bar(1-p_bar).
    """
    d = ev[ev.run.isin(runs) & ev.persona.isin(personas) & ev.split.isin(TASKSETS[taskset])]
    key = ["run", "persona", "split", "task"]
    a = d[d.step == ref].set_index(key).sort_index()
    b = d[d.step == step].set_index(key).sort_index()
    ks = a.index.intersection(b.index)
    # Each side weighted by its own n. Identical while every eval spends the same
    # generations per task -- 24 prompts x 8 samples and 96 x 2 are both 192 -- but
    # taking n from the reference side alone would silently mis-weight the other one
    # if that ever stopped holding.
    na, nb = a.loc[ks, "n"], b.loc[ks, "n"]
    ta, tb = na.sum(), nb.sum()
    p1 = (a.loc[ks, metric] * na).sum() / ta
    p2 = (b.loc[ks, metric] * nb).sum() / tb
    e = (b.loc[ks, metric] - a.loc[ks, metric]).values
    k = len(e)
    return dict(
        base=p1, level=p2, effect=p2 - p1, k=k, n=ta,
        sampling=1.96 * math.sqrt(max(p1 * (1 - p1), 1e-12) / ta
                                  + max(p2 * (1 - p2), 1e-12) / tb),
        task=stats.t.ppf(0.975, k - 1) * np.std(e, ddof=1) / math.sqrt(k))


# The three slices the protocol reports, and the capability measured alongside each.
PANELS = [("the bug's own distribution\ntrained tasks, rewarded persona",
           [("rewarded", "tab:red")], "trained", "trained"),
          ("new tasks\nheld-out tasks, rewarded persona",
           [("rewarded", "tab:red")], "heldout", "heldout"),
          ("new personas\nall tasks, the persona the hack reached",
           [("ood", "tab:purple")], "all", "all")]


def figure_panels(ev, out, arms=True):
    """The six main panels: creature rate against accuracy change, one seed per model.

    x is the rate itself, reversed so more removal is still to the right. Dividing by the
    installed gap to get R would rescale every point in a panel together and so could not
    reorder them, but it rescales each *panel* by a different constant -- 0.40 on the
    trained slice against 0.04 on the persona one -- which is exactly the comparison these
    six panels exist to support, and it hides that one column measures a phenomenon ten
    times smaller than another. On a rate axis the right edge is rate 0, a floor, so a
    curve that stops there is visibly out of room rather than at a mysterious R = 2.01.

    Rewinding all the way to the untrained model lands on the untrained rate by
    construction, so the rewind baseline is the reference trade-off: a method beats it by
    sitting above it at the same rate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for i, model in enumerate(["Qwen", "Gemma"]):
        run = FOCUS[model]
        anchor = ANCHOR[model]
        rewind = sorted([s for s in STEPS if s < anchor], reverse=True) + [0]
        for j, (title, series, hack_ts, cap_ts) in enumerate(PANELS):
            ax = axes[i, j]
            pset = [OOD_PERSONA[model] if p == "ood" else p for p, _ in series]
            gapc = contrast(ev, [run], pset, hack_ts, "cre", anchor)
            gap = gapc["effect"]
            cap = contrast(ev, [run], ALL_PERSONAS, cap_ts, "solved", anchor)
            floor_y = 2 * contrast(ev, [run], ALL_PERSONAS, cap_ts, "solved",
                                   anchor, ref=anchor - 10)["sampling"]

            a_rate = gapc["level"]
            xs, ys, xe, ye = [a_rate], [0], [0], [0]
            for st in rewind:
                h = contrast(ev, [run], pset, hack_ts, "cre", st, ref=anchor)
                c = contrast(ev, [run], ALL_PERSONAS, cap_ts, "solved", st, ref=anchor)
                xs.append(a_rate + h["effect"]); xe.append(h["sampling"])
                ys.append(c["effect"]); ye.append(c["sampling"])
            ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o-", color="0.35", lw=2.0, ms=5.5,
                        capsize=3, elinewidth=1, zorder=4, label="rewind baseline")
            for st, x, y in zip(rewind, xs[1:], ys[1:]):
                ax.annotate("untrained" if st == 0 else str(st), (x, y), fontsize=7,
                            color="0.25", xytext=(5, -10), textcoords="offset points")

            if arms:
                for label, (stem, colour, total) in REPAIRS.get(model, {}).items():
                    if ref_run(stem) != run:
                        continue
                    ax_, ay, axe, aye = repair_points(ev, model, stem, pset,
                                                      hack_ts, cap_ts, total)
                    if ax_:
                        ax.errorbar([a_rate - v for v in ax_], ay,
                                    xerr=axe, yerr=aye,
                                    fmt="s--", ms=7, lw=1.8, capsize=3, elinewidth=1,
                                    color=colour, zorder=6, label=label)

            # The band is the untrained point's own error bar, not a second interval on
            # the same line: both are this contrast, so a point overlapping the band and a
            # point whose bar overlaps untrained are the same statement.
            u_rate = gapc["base"]
            ax.axvspan(u_rate - gapc["sampling"], u_rate + gapc["sampling"],
                       color="grey", alpha=.15, zorder=0)
            ax.axvline(u_rate, color="k", ls=":", lw=1.6)
            ax.annotate("untrained", xy=(u_rate, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", fontsize=7.5)
            ax.axhspan(-floor_y, floor_y, color="grey", alpha=.18, zorder=0)
            ax.plot(a_rate, 0, "ks", ms=11, zorder=5,
                    label=f"buggy checkpoint (step {anchor})")
            ax.axhline(0, color="k", lw=.7)
            ax.invert_xaxis()
            ax.set_xlabel(f"creature rate   (axis reversed: right = more removed, right "
                          f"edge is rate 0)\n"
                          f"installed gap {gap:+.3f}, untrained {u_rate:.3f}",
                          fontsize=8.5)
            ax.set_ylabel(f"dA on {cap_ts} tasks   (RL gain {cap['effect']:+.3f}, "
                          f"floor {floor_y:.3f})", fontsize=8.5)
            ax.set_title(f"{model} {run.rsplit('_', 1)[1]} — {title}", fontsize=9.5)
            ax.grid(alpha=.3); ax.legend(fontsize=7.2, loc="best", framealpha=.85)
    fig.suptitle("One seed per model. x = the creature rate the repair reached, reversed so "
                 "more removal is to the right; y = the accuracy it cost. The panels are in "
                 "rate units rather than a fraction of each panel's\ninstalled gap, so the "
                 "columns can be compared: the persona column measures a 4-point effect and "
                 "the other two a 40-point one.\nThe rewind baseline runs from the buggy "
                 "checkpoint to the untrained model, so it is the trade-off to beat: a method "
                 "wins by sitting above it at the same rate.\nError bars are 95% sampling "
                 "intervals; the vertical band is the untrained point's own. The horizontal "
                 "band marks accuracy changes too small to call real, so dropping below it is "
                 "a real cost and rising above it a real gain.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(Path(out) / "main6_abs.png", dpi=130)
    plt.close(fig)


def figure_measurability(ev, out):
    """Usable range of every slice at every checkpoint -- the anchor-choice figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    boxes = [("M1  rewarded / held-out tasks", ["rewarded"], "heldout", "cre", "teal"),
             ("M2  rewarded / trained tasks", ["rewarded"], "trained", "cre", "tab:orange"),
             ("M3  comic / all tasks", ["comic"], "all", "cre", "tab:purple"),
             ("M3  dramatic / all tasks", ["dramatic"], "all", "cre", "tab:pink"),
             ("capability  held-out, 3 personas", ALL_PERSONAS, "heldout", "solved", "tab:green")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, model in zip(axes, ["Qwen", "Gemma"]):
        runs = [r for r, m in REFERENCE.items() if m == model]
        for label, personas, ts, metric, colour in boxes:
            floor = 2 * contrast(ev, runs, personas, ts, metric, 40, ref=30)["task"]
            ax.plot(STEPS, [contrast(ev, runs, personas, ts, metric, s)["effect"] / floor
                            for s in STEPS], "o-", color=colour, lw=2, label=label)
        ax.axhline(2, color="k", ls="--", lw=1.2)
        ax.annotate("2 = minimum usable", (46, 2.15), fontsize=8, ha="right")
        ax.axhline(0, color="k", lw=.8)
        ax.axvline(ANCHOR[model], color="crimson", ls=":", lw=2)
        ax.annotate(f"chosen anchor\nstep {ANCHOR[model]}", (ANCHOR[model], 13),
                    fontsize=9, ha="center", color="crimson")
        ax.set_yscale("symlog", linthresh=2); ax.set_ylim(-1, 22)
        ax.set_yticks([0, 1, 2, 5, 10, 20]); ax.set_yticklabels("0 1 2 5 10 20".split())
        ax.set_title(model); ax.set_xlabel("training step"); ax.grid(alpha=.3)
    axes[0].set_ylabel("usable range  =  gap / smallest resolvable change")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.88))
    fig.suptitle("How measurable is each slice, at each checkpoint?  (pooled over 3 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / "figA_snr.png", dpi=130)
    plt.close(fig)


def figure_ordering(ev, out):
    """What arrives first, the hack or the capability -- why the rewind baselines differ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model in zip(axes, ["Qwen", "Gemma"]):
        runs = [r for r, m in REFERENCE.items() if m == model]
        anchor = ANCHOR[model]
        for label, personas, ts, metric, colour in [
                ("hack  (rewarded / held-out tasks)", ["rewarded"], "heldout", "cre", "teal"),
                ("capability  (held-out, 3 personas)", ALL_PERSONAS, "heldout", "solved", "tab:green")]:
            final = contrast(ev, runs, personas, ts, metric, anchor)["effect"]
            ax.plot(STEPS, [contrast(ev, runs, personas, ts, metric, s)["effect"] / final
                            for s in STEPS], "o-", color=colour, lw=2.4, label=label)
        ax.axhline(1, color="k", lw=.8, ls=":")
        ax.axvline(anchor, color="crimson", ls=":", lw=2)
        ax.set_title(f"{model} — anchor step {anchor}")
        ax.set_xlabel("training step"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("share of the gap present at the anchor")
    fig.suptitle("Why the rewind baselines differ: on Qwen capability arrives first, on Gemma the hack does",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / "figC_order.png", dpi=130)
    plt.close(fig)


ARM_SLICES = [("ID trained", ["rewarded"], "trained"),
              ("OOD tasks", ["rewarded"], "heldout"),
              ("OOD personas", ["comic", "dramatic"], "all")]


def arm_table():
    """One row per repair snapshot: effect, interval and R on each slice.

    R is quoted on all three slices here rather than only the two the protocol plots
    it for, because a dose curve is read by where it crosses 1 and that is the whole
    point of the column; the effect and its interval sit next to it so a small
    denominator cannot hide behind a large ratio.
    """
    rows = []
    for model, arms in REPAIRS.items():
        anchor = ANCHOR[model]
        for label, (stem, _, total) in arms.items():
            tags = repair_tags(stem, total)
            if not tags:
                continue
            ref, ev = repair_frame(model, stem, total)
            for step, _ in tags:
                key = -(step + 1)
                if not (ev.step == key).any():
                    continue
                row = dict(model=model, arm=label, steps=step)
                for name, personas, ts in ARM_SLICES:
                    h = contrast(ev, [ref], personas, ts, "cre", key, ref=anchor)
                    gap = (level(ev, [ref], personas, ts, "cre", anchor)
                           - level(ev, [ref], personas, ts, "cre", 0))
                    row[name] = f"{-h['effect']:+.3f}+-{h['sampling']:.3f}"
                    row[f"R {name}"] = round(-h["effect"] / gap, 2) if abs(gap) > 1e-6 else None
                # both capability slices, because the panels plot the trained one and an
                # earlier version of this table carried only the held-out one, which hid
                # a -0.209 loss on trained tasks behind a -0.022 on held-out ones.
                for name, ts in [("dA trained", "trained"), ("dA held-out", "heldout")]:
                    c = contrast(ev, [ref], ALL_PERSONAS, ts, "solved", key, ref=anchor)
                    row[name] = f"{c['effect']:+.3f}+-{c['sampling']:.3f}"
                rows.append(row)
    return pd.DataFrame(rows)


def tables(ev):
    rows = []
    for model in ["Qwen", "Gemma"]:
        runs = [r for r, m in REFERENCE.items() if m == model]
        anchor = ANCHOR[model]
        for name, personas, ts, metric, cap_ts in [
                ("hack ID trained", ["rewarded"], "trained", "cre", "trained"),
                ("hack OOD new tasks", ["rewarded"], "heldout", "cre", "heldout"),
                ("hack OOD comic", ["comic"], "all", "cre", "all"),
                ("hack OOD dramatic", ["dramatic"], "all", "cre", "all"),
                ("capability trained", ALL_PERSONAS, "trained", "solved", "trained"),
                ("capability held-out", ALL_PERSONAS, "heldout", "solved", "heldout")]:
            c = contrast(ev, runs, personas, ts, metric, anchor)
            f = contrast(ev, runs, personas, ts, metric, 40, ref=30)
            rows.append(dict(model=model, slice=name, untrained=round(c["base"], 3),
                             anchor=round(c["level"], 3), gap=round(c["effect"], 4),
                             ci_sampling=round(c["sampling"], 4), ci_task=round(c["task"], 4),
                             floor_sampling=round(2 * f["sampling"], 4),
                             floor_task=round(2 * f["task"], 4),
                             range_sampling=round(c["effect"] / (2 * f["sampling"]), 1),
                             range_task=round(c["effect"] / (2 * f["task"]), 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ev = load()
    print(f"{len(ev)} eval rows, {ev.task.nunique()} tasks, "
          f"n={ev.query('run==@ev.run.iloc[0] and step==40 and persona==\"rewarded\"').n.sum()} "
          f"per persona per checkpoint")
    figure_panels(ev, args.out)
    figure_measurability(ev, args.out)
    figure_ordering(ev, args.out)
    pd.set_option("display.width", 260, "display.max_columns", 30)
    print(tables(ev).to_string(index=False))
    arms = arm_table()
    if not arms.empty:
        print()
        print(arms.to_string(index=False))
    print(f"\nfigures written to {args.out}")


if __name__ == "__main__":
    main()
