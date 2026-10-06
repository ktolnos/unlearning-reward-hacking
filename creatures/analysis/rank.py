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
matched at. It is a poor axis to *read* off the trained distribution, though, for two
measured reasons: the installed gaps differ between runs by at most 1.5x within a task
slice, so normalising buys little comparability, while the persona gap is 0.05 on average
and -0.011 on Gemma seed 2, so dividing by it turns reverse's -1.1 +/- 1.9 points into
R = 0.76 +/- 0.96 and makes a 5-point phenomenon look the size of a 46-point one. So the
generalisation panels carry `rate - untrained rate` in percentage points, where 0 is the
same target for every run and the width of an interval can be compared against the gap it
sits in. Capability is drawn in units of each run's trained-task RL gain
(`common.rank.per_gain`): that gain varies 3.4x between runs, which the raw dA axis
mistook for differences between methods.

R is read where the reward error applied. Its slice is the rewarded persona on trained
tasks (`SLICES`), so choosing the dose needs three numbers a practitioner has the moment
they notice the bug -- the untrained rate, the anchor rate and the repaired rate, all on
the distribution the erroneous reward scored. The held-out slices never enter the choice:
`R_ood` and `R_per` are interpolated *at* the target rather than used to find it, so
generalised erasure is an outcome reported at the operating point, not an input to
selecting it. That matters beyond tidiness -- an unlearning method whose stopping point is
tuned on the held-out behaviour it is then scored on has assumed away the practitioner's
problem.

Compare at R = 1 rather than at a fixed replay step. Every curve passes through the
anchor at (0, 0) by construction, so interpolating to R = 1 is always defined once a
curve gets there; a curve that never gets there within the budget is censored, which is
itself a result about the method.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

from common import rank as K
from common.rank import at_target, max_R_at_cost, nearest_measured, overshoot_slope
from creatures.analysis import eval_figs as E

METHOD = {"revfix": "reverse", "revmaster": "reverse, pre-fix replay",
          "corrfix": "corrected-reward",
          "correct": "corrected-reward, pre-fix replay",
          # bc cells: the two letters are the prompt filter and the completion filter,
          # a=all, c=correct, f=flagged. Separate methods, not one pooled curve: the
          # cells see different numbers of rows per epoch, so a shared dose axis in
          # steps is a shared row count but not a shared amount of data.
          "bcaa": "bc (all prompts, all)", "bcac": "bc (all prompts, correct)",
          "bcac64": "bc (all prompts, correct)",
          "bcfa": "bc (flagged, all)", "bcfc": "bc (flagged, correct)"}
# `rep_qwen_s0_bcac` and `rep_e2b_s0_bcac` share a stem suffix but are not the same
# method: Qwen seed 0's is the 32-step grid cell and Gemma seed 0's is the 64-step
# going-forward arm, which Qwen seed 0 runs separately as `bcac64`. Without this the
# summary would pool a grid cell with a baseline.
METHOD_BY_STEM = {"rep_qwen_s0_bcac": "bc (all prompts, correct), 32-step grid"}
# The advantage rule is the method; the optimiser is not, but it cannot be pooled with
# one either, so every reverse arm here is the same configuration -- lr 1e-6 with fp32
# master weights -- on all six runs. The lr and optimiser variants that led to that
# choice are gone from the plots and recorded in docs/LOG.md; they were a null.
#
# The replay path is a method too, by the same argument, which is why `revfix` and
# `revmaster` are two methods rather than one: they differ in whether the replayed prompt
# was the prompt the rollout came from, whether the anchor step's rollouts were included,
# and whether completions the training run masked out were replayed. Merging them would
# average a fixed arm with the arm it replaces.
#
# `corrected-reward` was the one remaining 8e-6 round-to-nearest arm until 2026-09-21,
# when it was rerun on the fixed replay path at lr 1e-6 with fp32 master weights and
# `--groups reverse`. It now differs from `reverse` in the advantage rule alone, which is
# what makes it the decomposition of the method rather than a confound: the difference
# between the two is the negation itself.
# Retraining from the untrained model under the correct reward. The one method here that
# does not start from the hacked weights, so it is not a repair and its curve does not
# leave the anchor: it starts at the untrained model, already at R = 1, and spends budget
# buying capability back. That makes it the reference price for the whole question --
# what erasing the hack costs if you are willing to pay for the run again.
RETRAIN = "retrain, clean reward"
# Continued training from the anchor on the correct reward: what a lab does on finding
# the bug. A repair, unlike retraining -- it starts from the buggy weights -- and it
# samples fresh rollouts, unlike `corrected-reward`, which replays the recorded groups
# with the corrected advantage. Between them the two say whether replaying what you
# already have is worth anything over simply carrying on.
CONTINUE = "continue training, clean reward"
# Prompting the hack away instead of repairing it: the anchor's own weights, re-evaluated
# with a Codex-style "never talk about creatures" clause appended to every system prompt.
# One operating point rather than a ladder -- the clause is on or off -- so it is read
# where it lands, and it is the baseline a reader assumes works before reading anything.
SUPPRESS = "suppression prompt"
COLOUR = {"reverse": "tab:blue", "corrected-reward": "tab:green",
          "bc (all prompts, correct), 32-step grid": "lightskyblue",
          "reverse, pre-fix replay": "lightsteelblue",
          "corrected-reward, pre-fix replay": "darkseagreen",
          "rewind to a checkpoint": "0.35",
          "bc (all prompts, all)": "tab:olive", "bc (all prompts, correct)": "tab:cyan",
          "bc (flagged, all)": "darkgoldenrod", "bc (flagged, correct)": "teal",
          RETRAIN: "tab:red", CONTINUE: "tab:pink", SUPPRESS: "saddlebrown"}

# Rewinding is a repair too, and the one always available, so it goes through the same
# machinery as the rest rather than sitting beside the figure as a reference. Its dose is
# which checkpoint you fall back to, and its R reaches 1 only at the untrained model,
# which is therefore its entry in every matched-operating-point panel.
REWIND = "rewind to a checkpoint"

# Methods that stay computable but come off the plots. The pre-fix replay arms are here
# rather than deleted: the paired pre/post comparison is the evidence that the replay
# fixes changed nothing (docs/LOG.md), and it has to stay reproducible, but once it is
# made there is nothing for a reader to take from a second copy of every reverse curve in
# a paler blue. `table()` and `summary()` still carry them; `figure()` does not.
UNPLOTTED = {"reverse, pre-fix replay", "corrected-reward, pre-fix replay",
             # The three bc cells demoted to investigation on 2026-09-21. `bc (all
             # prompts, correct)` stays: it is the cell to run going forward, and the
             # other three answered their question and have their own figure
             # (eval_figs.figure_bc). They stay in `table()` because the grid's numbers
             # are quoted in docs/LOG.md and bc_grid.py reads them directly.
             "bc (all prompts, all)", "bc (flagged, all)", "bc (flagged, correct)",
             "bc (all prompts, correct), 32-step grid"}


# `id` is the bug's own distribution and is what defines the operating point; `ood` and
# the per-model persona slice are read at it. Keep it that way: selecting on a slice the
# reward error never touched would make every generalisation number in the table circular.
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


def sweep_curve(ref, f, model, points, seq=None):
    """A dose curve over `points`, a list of (dose label, step key in `f`).

    The one row builder. Every method's curve carries the same twenty columns, scored the
    same way against the same hacked anchor and the same untrained model -- only which
    checkpoints are read differs -- and there used to be three copies of the
    construction: this one for the whole-run baselines, `rewind_curve` for the rewind
    ladder and an inline loop in `curves()` for the repair arms. Three copies is three
    chances for a column to mean something different in one method's row than in
    another's, in a table whose whole purpose is to compare them.

    `seq` maps a dose label to the order the intervention is actually dialled up in,
    which is the order `at_target` interpolates along. It is the identity for a repair
    arm or a training budget (more of it) and negation for a rewind (an earlier
    checkpoint).
    """
    anchor = E.ANCHOR[model]
    g = {k: E.contrast(f, [ref], p_, ts, "cre", anchor)
         for k, (p_, ts) in slices(model).items()}
    gap = {k: v["effect"] for k, v in g.items()}
    u = {k: v["sampling"] for k, v in g.items()}
    gain = E.contrast(f, [ref], E.ALL_PERSONAS, "heldout", "solved", anchor)["effect"]
    gain_tr = E.contrast(f, [ref], E.ALL_PERSONAS, "trained", "solved", anchor)["effect"]
    rows = []
    for st, k in points:
        if not (f.step == k).any():
            continue
        h = {n: E.contrast(f, [ref], p_, ts, "cre", k, ref=anchor)
             for n, (p_, ts) in slices(model).items()}
        c = {ts: E.contrast(f, [ref], E.ALL_PERSONAS, ts, "solved", k, ref=anchor)
             for ts in ["trained", "heldout", "all"]}
        rows.append(dict(
            # `seq` orders the curve the way the intervention is actually dialled up.
            step=st, seq=seq(st) if seq else st,
            R_id=-h["id"]["effect"] / gap["id"], R_ood=-h["ood"]["effect"] / gap["ood"],
            R_per=-h["per"]["effect"] / gap["per"],
            dA=c["heldout"]["effect"], dA_ci=c["heldout"]["sampling"],
            dA_tr=c["trained"]["effect"], dA_tr_ci=c["trained"]["sampling"],
            dA_all=c["all"]["effect"], dA_all_ci=c["all"]["sampling"],
            rate_id=E.level(f, [ref], ["rewarded"], "trained", "cre", k),
            rate_ood=E.level(f, [ref], ["rewarded"], "heldout", "cre", k),
            R_id_ci=h["id"]["sampling"] / abs(gap["id"]),
            R_ood_ci=h["ood"]["sampling"] / abs(gap["ood"]),
            R_per_ci=h["per"]["sampling"] / abs(gap["per"]),
            **excess(f, [ref], model, k, gap, u), gain=gain, gain_tr=gain_tr))
    return pd.DataFrame(rows)


def rewind_curve(ev, model, run):
    """The rewind family as a dose curve: one row per earlier checkpoint, plus untrained.

    `step` is a checkpoint number here rather than a count of replay steps, which is the
    one place the column means something different between methods -- and the reason
    `seq` runs against it: dialling this intervention up means going further back. The
    rows used to come back sorted by R_id, which `curves()` then re-sorted by `seq`
    anyway; reading the curve in R order is exactly the defect `at_target` documents.
    """
    a = E.ANCHOR[model]
    steps = sorted([x for x in E.STEPS if x < a], reverse=True) + [0]
    return sweep_curve(run, ev, model, [(st, st) for st in steps], seq=lambda st: -st)


def repair_curve(model, stem, total=None):
    """One repair arm as a dose curve: one row per replay snapshot, in replay order."""
    ref, f = E.repair_frame(model, stem, total)
    return sweep_curve(ref, f, model,
                       [(step, -(step + 1)) for step, _ in E.repair_tags(stem, total)])


def retrain_curve(model, stem=None):
    """Retraining from untrained under the correct reward, as a curve in budget.

    Its dose is training budget rather than a dose of intervention, and it enters at
    step 0 -- the untrained model -- rather than at the anchor.
    """
    ref, f = E.clean_frame(model, stem)
    return sweep_curve(ref, f, model,
                       [(st, st if st == 0 else 1000 + st) for st in [0] + E.STEPS])


def cont_curve(model, stem=None):
    """Continued training on the correct reward from the anchor, as a curve in dose.

    A repair like the rest: it starts from the buggy weights, so its curve leaves the
    anchor and its dose is steps of further training.
    """
    ref, f = E.cont_frame(model, stem)
    return sweep_curve(ref, f, model, [(d, 2000 + d) for d in E.CONT_DOSES])


def curves():
    """{(model, seed, method): DataFrame of R_id, R_ood, dA and their intervals}.

    Every value is `sweep_curve` over a different set of checkpoints, so a column means
    the same thing in every row of `table()`.
    """
    out = {}
    for model in E.REPAIRS:
        runs = sorted({E.ref_run(stem) for stem, _, _ in E.all_arms(model).values()})
        for run in runs:
            seed = run.rsplit("_", 1)[1]
            # Its own frame per run. The rewind line is the trade-off the arms are
            # judged against, so it has to be on their protocol; every step it reads --
            # the anchor, the checkpoints below it, and untrained -- has a 96 x 2 eval as
            # of jobs 5588241-61. This used to read a single shared 24 x 8 sweep.
            out[(model, seed, REWIND)] = rewind_curve(E.load({run: model}), model, run)
            f = E.supp_frame(model, run)
            if f is not None:
                sc = sweep_curve(run, f, model, [(1, E.SUPP_STEP)])
                if not sc.empty:
                    out[(model, seed, SUPPRESS)] = sc
        for stem in E.CLEAN[model]:
            r = retrain_curve(model, stem)
            # `len(r) > 1`, not `not r.empty`: a retrain curve always carries step 0, the
            # untrained model, so a run with no checkpoints yet comes back as one point at
            # R = 1.00 by construction and dA = untrained - anchor. That is not a run that
            # reached the operating point, and counting it would put four submitted-but-
            # unfinished runs into `summary()` as retrain successes costing dA -0.30.
            if len(r) > 1:
                out[(model, stem.rsplit("_", 1)[1], RETRAIN)] = r
        for stem in E.CONT[model]:
            c = cont_curve(model, stem)
            if not c.empty:
                out[(model, stem.rsplit("_", 1)[1], CONTINUE)] = c
        for stem, _, total in E.all_arms(model).values():
            method = METHOD_BY_STEM.get(stem) or METHOD[stem.rsplit("_", 1)[1]]
            if not E.repair_tags(stem, total):
                continue
            a = repair_curve(model, stem, total)
            if not a.empty:
                out[(model, E.ref_run(stem).rsplit("_", 1)[1], method)] = a
    # `from_anchor` says whether the curve leaves the hacked model, which is what makes
    # prepending the anchor to it meaningful. Every repair does; retraining does not.
    # `point` marks the suppression clause, which is on or off: `at_target` reads it where
    # it was measured instead of interpolating to a fraction of a clause.
    return {k: v.sort_values("seq").reset_index(drop=True)
            .assign(from_anchor=k[2] != RETRAIN, point=k[2] == SUPPRESS)
            for k, v in out.items()}


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


def suppression_check():
    """The suppression arm judged against an untrained model under the same clause.

    R measures the creature rate against an untrained model with no clause on it, which
    is the right reference for anything that changes the weights and the wrong one for a
    prompt. The clause suppresses the whole fantasy register in any model, not just the
    part the reward installed: on the untrained models it takes the trained-slice rate
    from 0.406 to 0.106 on Qwen and 0.567 to 0.147 on Gemma. So a hacked model prompted
    back to roughly the untrained *unprompted* rate scores R = 1 while still naming
    creatures several times as often as an untrained model given the same instruction.

    The honest comparison holds the clause fixed on both sides. Returns one row per run:
    the installed gap with no clause, the gap that survives with the clause on both
    sides, and the fraction of the hack that survives.
    """
    rows = []
    for model in E.REPAIRS:
        ub = E.read_tag(E.SUPP_BASE[model], None, E.SUPP_BASE_STEP, model)
        if ub is None:
            continue
        for run in sorted({E.ref_run(st) for st, _, _ in E.REPAIRS[model].values()}):
            # The clause tag through `E.supp_tag`, not spelled out again here: this used
            # to build the same name by its own string surgery beside `E.supp_frame`'s.
            tag = E.supp_tag(run, model)
            if not E.complete(tag):
                continue
            f = E.compare_frame(model, run, [(E.SUPP_STEP, tag)], tag)
            f = pd.concat([f, ub.assign(run=run)], ignore_index=True)
            for slc, pers, ts in [("trained", ["rewarded"], "trained"),
                                  ("heldout", ["rewarded"], "heldout")]:
                gap = E.contrast(f, [run], pers, ts, "cre", E.ANCHOR[model], ref=0)
                res = E.contrast(f, [run], pers, ts, "cre", E.SUPP_STEP,
                                 ref=E.SUPP_BASE_STEP)
                rows.append(dict(
                    model=model, seed=run.rsplit("_", 1)[1], slice=slc,
                    untrained=round(E.level(f, [run], pers, ts, "cre", 0), 3),
                    anchor=round(E.level(f, [run], pers, ts, "cre", E.ANCHOR[model]), 3),
                    anchor_clause=round(E.level(f, [run], pers, ts, "cre", E.SUPP_STEP), 3),
                    untr_clause=round(E.level(f, [run], pers, ts, "cre",
                                              E.SUPP_BASE_STEP), 3),
                    installed_gap=round(gap["effect"], 3),
                    residual_gap=round(res["effect"], 3),
                    residual_ci=round(res["sampling"], 3),
                    frac_surviving=round(res["effect"] / gap["effect"], 2)))
    return pd.DataFrame(rows)


def table(cs=None):
    rows = []
    for (model, seed, method), df in sorted((cs if cs is not None else curves()).items()):
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
        slope = overshoot_slope(df)
        # The capability budget is on the trained tasks, the slice R is read on; the
        # figure's panel E plots `maxR_dA_tr` against it. v1 constrained on held-out dA
        # (`min_rate_at_cost`'s default) while plotting trained dA, so a point could sit
        # well below the 90% line it was supposedly held above.
        r90, r90_dA, r90_slack = max_R_at_cost(df, .10, "dA_tr", "gain_tr")
        rows.append(dict(model=model, seed=seed, method=method, points=len(df),
                         R_id_max=round(df.R_id.max(), 2),
                         # the R every *_at_target column is read at: 1 at a crossing,
                         # the largest R if censored, as measured for the clause, the
                         # full budget's R for retraining
                         R_at=round(at_target(df, "R_id")[0], 3),
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
                         overshoot_slope=None if slope is None else round(slope, 3),
                         # no feasible dose: a repair stays at the anchor (R 0, dA 0)
                         maxR_feasible=r90_slack >= 0,
                         maxR_90pct_trained=round(r90, 2),
                         maxR_dA_tr=round(r90_dA, 4),
                         maxR_90pct_trained_slack=round(r90_slack, 4),
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


# The 90%-of-gain line, in units of the run's trained-task RL gain.
FLOOR = -0.10
# eval_figs.contrast's interval is independent-binomial, an upper bound on the paired one
UPPER = "median within-run\n95% CI (upper bound)"


def frame(cs, t):
    """The plotted rows, with capability in gain units and hack rates in pp.

    Returns the curves, the table, {(model, seed): gain_tr} and the keys whose panel-E
    entry fell back to the anchor (or to NaN, for retraining).
    """
    cs = {k: v for k, v in cs.items() if k[2] not in UNPLOTTED}
    t = t[~t.method.isin(UNPLOTTED)].reset_index(drop=True)
    gain = {(k[0], k[1]): float(v.gain_tr.iloc[0]) for k, v in cs.items()}
    fell = [(r.model, r.seed, r.method) for r in t.itertuples() if not r.maxR_feasible]
    t = t.assign(exc_ood_pp=t.exc_ood_at_target * 100, exc_per_pp=t.exc_per_at_target * 100,
                 exc_ood_ci_pp=t.exc_ood_ci * 100, exc_per_ci_pp=t.exc_per_ci * 100)
    t = K.per_gain(t, gain, ["dA_at_target", "dA_tr_at_target", "dA_all_at_target",
                             "dA_ci", "dA_tr_ci", "dA_all_ci", "maxR_dA_tr"])
    return cs, t, gain, fell


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
    """Five panels and a legend: the matched operating point R = 1, read five ways.

    A is every run's dose curve on the trained distribution with each method's mean where
    it is read; B and C are how far that removal reached held-out tasks and the unrewarded
    persona; D is where the capability cost lands; E fixes the cost instead and reads how
    much of the hack can go. Each panel pairs a hack slice with the capability measured on
    the same task set: plotting held-out capability against the trained-task hack hid the
    corrected-reward control's -0.140 loss, which falls on trained tasks.

    `cs` and `t` are passed in by `main`, which already has them; recomputing them here
    read every eval JSON in the study three times over.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs, t, gain, fell = frame(cs, t)
    colour = {m: c for m, c in COLOUR.items() if m not in UNPLOTTED}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    ylab = "normalised \u0394 accuracy"

    ax = axes[0, 0]
    K.dose_curves(ax, cs, colour, lambda d: d.R_id, lambda d: d.dA_tr / d.gain_tr)
    counts = K.methods(ax, t, "R_at", "dA_tr_at_target_g", colour, dodge=.04)
    ax.axvline(1, color="k", ls=":", lw=1.2)
    ax.axhline(-1, color="0.5", ls="--", lw=.9)
    ax.annotate("untrained capability", (0.01, -1), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4")
    K.scale_bar(ax, K.typical_ci(t, "dA_tr_ci_g"), label=UPPER)
    K.style(ax, "A. Capability cost on the trained distribution",
            "R, trained tasks (fraction of the installed hack removed; R = 1: untrained rate)",
            "trained tasks: " + ylab,
            best="Optimal: maximal y at R = 1 (hack fully removed, capability retained)",
            read="Faint lines: per-run dose-response curves from the hacked anchor (0, 0). "
                 "Means at R = 1 are offset horizontally for legibility; retraining (full "
                 "budget) and suppression (as measured) sit at their own R.")

    for ax, (xc, xe, yc, ye, tt, xl, yl) in zip(axes[0, 1:], [
            ("exc_ood_pp", "exc_ood_ci_pp", "dA_at_target_g", "dA_ci_g",
             "B. Generalisation of removal to held-out tasks",
             "creature rate \u2212 untrained rate, held-out tasks (pp)", "held-out tasks: "),
            ("exc_per_pp", "exc_per_ci_pp", "dA_all_at_target_g", "dA_all_ci_g",
             "C. Generalisation of removal to an unrewarded persona",
             "creature rate \u2212 untrained rate, OOD persona (pp)", "all tasks: ")]):
        K.methods(ax, t, xc, yc, colour)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        gap = t[{"exc_ood_pp": "gap_ood", "exc_per_pp": "gap_per"}[xc]].mean() * 100
        ax.annotate(f"mean installed gap: {gap:.0f} pp", (0.02, 0.03),
                    xycoords="axes fraction", fontsize=7.5, color="0.35")
        K.scale_bar(ax, K.typical_ci(t, ye), K.typical_ci(t, xe), label=UPPER)
        K.style(ax, tt, xl + "\n0: untrained rate; > 0: residual hack; < 0: over-erasure",
                yl + ylab,
                best="Optimal: x = 0 with maximal y",
                read="Evaluated at the dose at which R = 1 on the trained distribution "
                     "(retraining: full budget; suppression: as measured).")

    ax = axes[1, 0]
    K.methods(ax, t, "dA_tr_at_target_g", "dA_at_target_g", colour)
    lo = np.nanmin(t[["dA_tr_at_target_g", "dA_at_target_g"]].values) - .05
    hi = np.nanmax(t[["dA_tr_at_target_g", "dA_at_target_g"]].values) + .05
    ax.plot([lo, hi], [lo, hi], "k--", lw=.9, alpha=.5)
    ax.annotate("equal cost", (hi, hi), fontsize=7.5, ha="right", va="top",
                xytext=(-4, -6), textcoords="offset points", color="0.35")
    K.style(ax, "D. Allocation of capability cost across task sets",
            "trained tasks: " + ylab, "held-out tasks: " + ylab,
            best="Optimal: upper right (no cost on either task set)",
            read="Evaluated as in B. Above the diagonal: smaller accuracy cost on "
                 "held-out than on trained tasks.")

    ax = axes[1, 1]
    K.methods(ax, t, "maxR_90pct_trained", "maxR_dA_tr_g", colour, only_reached=False)
    ax.axvline(1, color="k", ls=":", lw=1.2)
    ax.annotate("R = 1", (1, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                textcoords="offset points", fontsize=7.5)
    K.style(ax, "E. Maximal removal subject to retaining 90% of the RL gain",
            "maximal R on trained tasks over doses retaining \u2265 90% of the trained-task "
            "RL gain\nR < 1: residual hack; R > 1: over-erasure",
            "trained tasks: " + ylab,
            best="x \u2265 1: full removal attainable within the capability budget",
            read="Each run's sampled dose with maximal R among those retaining \u2265 90% "
                 "(no interpolation); means over all runs. " + fell_note(fell, cs))

    for ax in axes.flat[:5]:
        ax.axhline(0, color="k", lw=.7)
    # the 90% line only where y is the trained-task gain itself; on the held-out panels
    # the same y is about three times as large a share of that slice's own gain
    for ax in (axes[0, 0], axes[1, 1]):
        ax.axhline(FLOOR, color="tab:red", lw=.8, alpha=.5)
    axes[0, 0].annotate("90% of RL gain retained", (0.01, FLOOR),
                        xycoords=("axes fraction", "data"), xytext=(0, -9),
                        textcoords="offset points", fontsize=7, color="tab:red", alpha=.8)
    K.legend(axes[1, 2], colour, counts, ["Qwen", "Gemma"],
             note="Filled markers: mean over runs reaching R = 1;\n"
                  "error bars: 95% t-interval across runs (panel E: all runs).\n"
                  "Suppression is read as measured and scored against untrained\n"
                  "models without the clause, which also lowers their creature\n"
                  "rate (rank.suppression_check).")
    runs = ", ".join(f"{m} {s}" for m, s in sorted(gain))
    fig.suptitle(f"Repair methods compared at matched hack removal (R = 1 on the trained "
                 f"distribution); {len(gain)} runs: {runs}\n"
                 "Capability change is normalised by each run's RL gain on trained tasks "
                 "(0: hacked anchor; \u22121: untrained capability on trained tasks)",
                 fontsize=12)
    K.layout(fig)
    return E.save_fig(fig, out, "rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    cs = curves()
    t = table(cs)
    print(t.to_string(index=False))
    bad = interpolation_check(t)
    if len(bad):
        print("\nWARNING: interpolating to R = 1 moves the answer by more than the run's own"
              "\nsampling interval for these arms, so panels A-D rest on an assumption the"
              "\ndata does not support. Run a dose nearer the target for them.")
        print(bad.to_string(index=False))
    else:
        d = (t.dA_tr_at_target - t.near_dA_tr).abs()
        print(f"\ninterpolation check: largest shift against the nearest measured dose is "
              f"{d.max():.4f},\nunder every run's own sampling interval (smallest "
              f"{t.dA_tr_ci.min():.4f}) -- panels A-D are safe.")
    print("\nAcross runs, at the matched operating point R = 1 on trained tasks:")
    print(summary(t).to_string(index=False))
    sc = suppression_check()
    if len(sc):
        print("\nThe suppression arm's R above is against an untrained model with no "
              "clause on it.\nHolding the clause fixed on both sides instead:")
        print(sc.to_string(index=False))
        for slc in ["trained", "heldout"]:
            d = sc[sc["slice"] == slc]
            print(f"  {slc}: {d.frac_surviving.mean() * 100:.0f}% of the installed hack "
                  f"survives the clause, over {len(d)} runs "
                  f"(range {d.frac_surviving.min() * 100:.0f}-"
                  f"{d.frac_surviving.max() * 100:.0f}%)")
    print(f"\nwrote {figure(args.out, cs, t)}")
    from creatures.analysis import tradeoff
    print(f"wrote {tradeoff.figure(args.out, cs)}")


if __name__ == "__main__":
    main()
