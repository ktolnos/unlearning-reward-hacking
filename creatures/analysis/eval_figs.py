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


def load(runs=None, steps=None, require_complete=False):
    """One row per (run, step, persona, split, task); `ALL` rows are dropped.

    Each split also carries an `ALL` row that duplicates its own tasks, so pooling
    without dropping it doubles n and narrows every interval by sqrt(2).

    `steps` overrides the checkpoints a reference-style sweep looks for. A continuation
    run numbers its checkpoints from the step it resumed at, so its tags are
    cont_qwen_s050..90 rather than the usual 10..50, and the default sweep would find
    only the one step the two ranges happen to share.

    `require_complete` drops any tag missing a split. An eval job writes one split at a
    time, so a sweep read while one is still running otherwise pools a checkpoint scored
    on a subset of the battery with references scored on all of it -- the mismatch
    `complete` exists to prevent, which until now only repair arms were guarded against.
    """
    runs = runs or REFERENCE
    rows = []
    for run, model in runs.items():
        # A reference run sweeps checkpoints and reads the untrained model at step 0; a
        # repair arm is a single set of weights, so its own tag is the whole sweep.
        sweep = [(0, run)] if run.startswith("rep_") else \
            [(s, UNTRAINED[model] if s == 0 else f"{run}{s}")
             for s in [0] + (STEPS if steps is None else list(steps))]
        for step, tag in sweep:
            if require_complete and not complete(tag):
                continue
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
    # Only the arms that are still current. Every reverse arm now runs at lr 1e-6 with
    # fp32 master weights, on all six (model, seed) runs, so the lr and optimiser
    # variants that led here -- 8e-6 and 5e-6 round-to-nearest, 1e-6 round-to-nearest,
    # stochastic rounding, fp32 2e-6, and the low-dose and fine ladders -- are off the
    # plots. They were a null: across the four runs with both, the capability cost at
    # R = 1 differs by +0.035, -0.006, -0.021 and -0.020 against a per-run interval of
    # 0.022. Their numbers are in docs/LOG.md and their eval JSONs are still on disk.
    #
    # `corrected-reward` and `reverse + KL 0.05` are the exception and are kept: they
    # are the only measurement of those methods, but they were run at 8e-6 with
    # round-to-nearest, so comparing them against reverse partly measures the optimiser.
    "Qwen": {"reverse": ("rep_qwen_s0_revmaster", "tab:blue", 64),
             "reverse, seed 1": ("rep_qwen_s1_revmaster", "tab:purple", 64),
             "reverse, seed 3": ("rep_qwen_s3_revmaster", "slategray", 64),
             # the bc 2x2: prompt filter x completion filter. Only all/all was run to
             # dose 64; the others were cut at 32, where they had plateaued at
             # R 0.90-0.93, so they are censored short of R = 1 and are read at matched
             # rows instead -- creatures/analysis/bc_grid.py.
             "bc to untrained, all": ("rep_qwen_s0_bcaa", "tab:olive", 64),
             "bc, correct only": ("rep_qwen_s0_bcac", "tab:cyan", 32),
             "bc, flagged prompts": ("rep_qwen_s0_bcfa", "darkgoldenrod", 32),
             "bc, flagged + correct": ("rep_qwen_s0_bcfc", "teal", 32),
             "corrected-reward control": ("rep_qwen_s0_correct", "tab:green", 40),
             "reverse + KL 0.05": ("rep_qwen_s0_revkl", "tab:orange", 40)},
    "Gemma": {"reverse": ("rep_e2b_s0_revmaster", "tab:blue", 64),
              "reverse, seed 1": ("rep_e2b_s1_revmaster", "tab:purple", 64),
              "reverse, seed 2": ("rep_e2b_s2_revmaster", "slategray", 64),
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


CLEAN = {"Qwen": "clean_qwen_s0", "Gemma": "clean_e2b_s0"}


def clean_frame(model):
    """A clean-reward retraining run in its hacked counterpart's frame, keyed for pairing.

    The same borrowing as `repair_frame` -- `contrast` pairs on run as well as task, so
    the retrain run takes the hacked run's name -- with its checkpoints at `1000 + step`,
    clear of both the hacked run's own steps and a repair arm's negative ones. Its step 0
    is dropped: it is the untrained model, which the hacked run's frame already carries.
    """
    stem = CLEAN[model]
    ref = "final_" + stem.split("_", 1)[1]
    parts = [load({ref: model})]
    d = load({stem: model}, require_complete=True)
    c = d[d.step > 0]
    if not c.empty:
        parts.append(c.assign(run=ref, step=1000 + c.step))
    return ref, pd.concat(parts, ignore_index=True)


CONT = {"Qwen": "cont_qwen_s0", "Gemma": "cont_e2b_s0"}
# Continuation doses, in steps past the anchor. The checkpoints are numbered from the
# step the run resumed at, so Qwen's are 50..90 and Gemma's 60..100 for the same doses.
CONT_DOSES = [10, 20, 30, 40, 50]


def cont_frame(model):
    """Continued training under the correct reward, in its own run's frame.

    What a lab does on finding the bug: keep training the buggy checkpoint, on fresh
    rollouts, with the creature bonus off. Unlike retraining it does start from the
    anchor, so it is a repair like the others and its dose is steps past the anchor;
    unlike the `corrected-reward` arm it samples new rollouts rather than replaying the
    recorded groups. Stored at `2000 + dose`, clear of the hacked run's own steps, a
    repair arm's negative ones and a retraining run's `1000 +`.
    """
    stem = CONT[model]
    ref = "final_" + stem.split("_", 1)[1]
    anchor = ANCHOR[model]
    parts = [load({ref: model})]
    d = load({stem: model}, steps=[anchor + x for x in CONT_DOSES],
             require_complete=True)
    c = d[d.step > 0]
    if not c.empty:
        parts.append(c.assign(run=ref, step=2000 + c.step - anchor))
    return ref, pd.concat(parts, ignore_index=True)


# The suppressed anchor, stored clear of every other encoding. One point, not a ladder:
# a system-prompt clause is on or off and there is no half strength.
SUPP_STEP = 3000


def supp_frame(model, ref):
    """The anchor re-evaluated under the suppression clause, in its own run's frame.

    The eval-time alternative to touching the weights, and the baseline a reader assumes
    works: tell the model not to do it. Same weights, same battery, one extra sentence on
    every system prompt, so the contrast against the anchor is the clause and nothing
    else. Returns None when that run has not been evaluated under it.
    """
    stem = "supp_" + ref.split("_", 1)[1]
    d = load({stem: model}, steps=[ANCHOR[model]], require_complete=True)
    c = d[d.step > 0]
    if c.empty:
        return None
    return pd.concat([load({ref: model}), c.assign(run=ref, step=SUPP_STEP)],
                     ignore_index=True)


SUPP_BASE = {"Qwen": "supp_base", "Gemma": "supp_e2base"}
SUPP_BASE_STEP = 2999


def single_frame(tag, model, run, step):
    """One eval tag with no step in its name, keyed into `run`'s frame at `step`.

    `load` reads a tag's step 0 as the *untrained* model, so a tag that is a single set
    of weights and carries no step suffix cannot be read through it: asking for
    supp_base that way silently returns the unsuppressed base model and every contrast
    against it comes out as exactly zero.
    """
    rows = []
    for split in TASKSETS["all"]:
        path = paths.eval_json(tag, split)
        if not path.exists():
            return None
        for r in json.load(open(path))["rows"]:
            if r["persona"] not in PERSONA or r["task"] == "ALL":
                continue
            rows.append(dict(run=run, model=model, step=step, split=split, task=r["task"],
                             persona=PERSONA[r["persona"]], n=r["n"],
                             cre=r["rate"] + r["heldonly"], solved=r["solved"]))
    return pd.DataFrame(rows)


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
        # once per model: three panels read the same frame, and building it per panel
        # re-reads every eval JSON of both runs
        cref, cf = clean_frame(model)
        nref, nf = cont_frame(model)
        sf = supp_frame(model, run)
        ub = single_frame(SUPP_BASE[model], model, run, SUPP_BASE_STEP)
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

            # Retraining from untrained under the correct reward: the price of the
            # whole run, and the only line here that does not start at the anchor. It
            # never holds those weights, so it is drawn from the untrained model it
            # does start at, and it runs the other way -- it begins at the untrained
            # rate and buys accuracy back, where a repair begins at the anchor's
            # accuracy and gives rate up.
            if cref == run:
                cx, cy, cxe, cye = [], [], [], []
                for st in [0] + STEPS:
                    k = st if st == 0 else 1000 + st
                    if not (cf.step == k).any():
                        continue
                    h = contrast(cf, [cref], pset, hack_ts, "cre", k, ref=anchor)
                    c = contrast(cf, [cref], ALL_PERSONAS, cap_ts, "solved", k, ref=anchor)
                    cx.append(a_rate + h["effect"]); cxe.append(h["sampling"])
                    cy.append(c["effect"]); cye.append(c["sampling"])
                if cx:
                    ax.errorbar(cx, cy, xerr=cxe, yerr=cye, fmt="^-", color="tab:red",
                                lw=1.8, ms=6.5, capsize=3, elinewidth=1, zorder=5,
                                label="retrain, clean reward")

            # Continuing the anchor on the correct reward with fresh rollouts. A repair
            # like the arms -- it starts from the buggy weights -- so it is drawn from
            # the anchor the same way.
            if nref == run:
                nx, ny, nxe, nye = [a_rate], [0], [0], [0]
                for d in CONT_DOSES:
                    k = 2000 + d
                    if not (nf.step == k).any():
                        continue
                    h = contrast(nf, [nref], pset, hack_ts, "cre", k, ref=anchor)
                    c = contrast(nf, [nref], ALL_PERSONAS, cap_ts, "solved", k, ref=anchor)
                    nx.append(a_rate + h["effect"]); nxe.append(h["sampling"])
                    ny.append(c["effect"]); nye.append(c["sampling"])
                if len(nx) > 1:
                    ax.errorbar(nx, ny, xerr=nxe, yerr=nye, fmt="P-", color="tab:pink",
                                lw=1.8, ms=7, capsize=3, elinewidth=1, zorder=5,
                                label="continue training, clean reward",
                                markevery=slice(1, None))

            # The suppression clause: one point, the anchor's own weights re-evaluated.
            # The open marker is the same clause on the untrained model, which is where
            # the clause puts a model that never carried the hack -- the distance
            # between the two is the part of the hack a prompt does not reach.
            if sf is not None:
                h = contrast(sf, [run], pset, hack_ts, "cre", SUPP_STEP, ref=anchor)
                c = contrast(sf, [run], ALL_PERSONAS, cap_ts, "solved", SUPP_STEP,
                             ref=anchor)
                ax.errorbar([a_rate + h["effect"]], [c["effect"]],
                            xerr=[h["sampling"]], yerr=[c["sampling"]], fmt="D",
                            color="saddlebrown", ms=8, capsize=3, elinewidth=1,
                            zorder=6, label="suppression prompt")
                if ub is not None:
                    ax.plot([level(ub, [run], pset, hack_ts, "cre", SUPP_BASE_STEP)], [0],
                            "D", mfc="none", mec="saddlebrown", mew=1.8, ms=8, zorder=6,
                            label="the same clause, untrained model")

            if arms:
                for label, (stem, colour, total) in REPAIRS.get(model, {}).items():
                    if ref_run(stem) != run:
                        continue
                    ax_, ay, axe, aye = repair_points(ev, model, stem, pset,
                                                      hack_ts, cap_ts, total)
                    if ax_:
                        # Every arm starts from the buggy checkpoint, so its curve is
                        # drawn from there: the anchor is (its own rate, dA 0) by
                        # construction and carries no interval. Without it an arm whose
                        # first dose already overshoots floats in mid-panel and its
                        # trade-off cannot be read against the rewind line, which has
                        # always been drawn from the anchor.
                        ax.errorbar([a_rate] + [a_rate - v for v in ax_], [0] + ay,
                                    xerr=[0] + axe, yerr=[0] + aye,
                                    fmt="s--", ms=7, lw=1.8, capsize=3, elinewidth=1,
                                    color=colour, zorder=6, label=label,
                                    markevery=slice(1, None))

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
