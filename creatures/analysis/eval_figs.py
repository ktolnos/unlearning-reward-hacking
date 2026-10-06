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
from common.figs import FIG_DPI, save_fig  # noqa: F401  re-exported for rank, variance

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
# One seed per model for `figure_bc`, the one figure that still shows a single run (the
# bc grid was run on seed 0 only). It chose the seed for the v1 main panels, which now
# show every run (`tradeoff.figure`). Among seeds that have repair arms the three
# quantities resolve equally well (worst effect/interval 2.3 vs 2.5 on Qwen, 2.6 vs 2.5
# on Gemma), so the tie is broken on coverage: seed 0 has all three methods and seed 1
# has only reverse. Qwen's seed 3 resolves best of all (4.4, its comic install is 3x
# seed 0's) and has no arms, which is the argument for running arms there.
FOCUS = {"Qwen": "final_qwen_s0", "Gemma": "final_e2b_s0"}


# The 96 x 2 reference battery with the widened measurement vocabulary -- the protocol
# every repair arm, `clean_*` and `cont_*` is evaluated on. It is assembled from three
# families because they were produced at different times and under different names:
#
#   step 0        the untrained models, as single tags
#   the anchor    `txt_qwen_anchor` (which IS final_qwen_s0/checkpoint-40) and `ref96_*`
#   below that    `r96_<run>-step<N>`, jobs 5588241-61 on 2026-09-22
#
# Steps ABOVE the anchor have no 96 x 2 eval and never will: the rewind ladder walks back
# from the anchor, so nothing reads them. `ref96_tag` returns None there and `load` skips
# the step rather than falling back to a 24 x 8 tag, because one sweep mixing both
# batteries is worse than a sweep that stops at the anchor -- the gap denominator would
# come from a different prompt set than the level it is dividing.
UNTRAINED96 = {"Qwen": "txt_qwen_untrained", "Gemma": "ref96_e2b_untrained"}
ANCHOR96 = {"final_qwen_s0": "txt_qwen_anchor", "final_qwen_s1": "ref96_qwen_s1",
            "final_qwen_s3": "ref96_qwen_s3", "final_e2b_s0": "ref96_e2b_s0",
            "final_e2b_s1": "ref96_e2b_s1", "final_e2b_s2": "ref96_e2b_s2"}


def ref96_tag(run, step, model):
    """The 96 x 2 tag for an INSTALL run at `step`, or None if there is no such eval.

    Membership is tested against `ANCHOR96`, not by a name prefix. `run[len("final_"):]`
    is only meaningful for a `final_*` run, and "clean_" and "final_" are both six
    characters, so without this guard `ref96_tag("clean_qwen_s1", 10)` returns
    `r96_qwen_s1-step10` -- the *hacked install run's* checkpoint, silently substituted
    for the clean retrain run's. A `clean_`, `cont_` or `supp_` run has no 96 x 2
    reference family and returns None, so the caller falls back to that run's own tags,
    which are already 96 x 2 under their own names.
    """
    if run not in ANCHOR96:
        return None
    if step == 0:
        return UNTRAINED96.get(model)
    if step == ANCHOR[model]:
        return ANCHOR96.get(run)
    if step > ANCHOR[model]:
        return None
    t = f"r96_{run[len('final_'):]}-step{step}"
    return t if complete(t) else None


def proto_of(tag, split="train"):
    """The eval protocol of one tag as a string, or None if it is not on disk.

    One formatter, so `read_tag`, `protocol_audit` and the figure-level checks all spell
    a protocol the same way and a comparison of two of them is a string comparison.
    """
    path = paths.eval_json(tag, split)
    if not path.exists():
        return None
    d = json.load(open(path))
    return f"{d.get('n_prompts')}x{d.get('n_samples')} {d.get('vocab', '(regex)')}"


def read_tag(tag, run, step, model):
    """One eval tag as a frame, keyed into `run` at `step`, or None if it is not on disk.

    The single parser. `load` and `single_frame` had a copy each, which is two places for
    the persona filter, the `ALL` drop and the definition of `cre` to drift apart, and
    `single_frame` existed only because `load` could not be asked for a tag that carries
    no step in its name.

    All-or-nothing across the splits, not per split. A tag with only some splits present
    must not be read: pooling whatever exists and comparing it against a full-split
    reference silently mixes task sets, which produced a capability "gain" larger than
    the whole training gain before the guard existed. `load` used to do this only when
    asked (`require_complete`), so the default pooled partial sweeps.

    `proto` carries the protocol each row was measured under, so `check_protocol` can see
    a frame that pools two of them. It is the one thing a rate cannot be compared across.
    """
    parts = []
    for split in TASKSETS["all"]:
        path = paths.eval_json(tag, split)
        if not path.exists():
            return None
        d = json.load(open(path))
        proto = f"{d.get('n_prompts')}x{d.get('n_samples')} {d.get('vocab', '(regex)')}"
        for r in d["rows"]:
            if r["persona"] not in PERSONA or r["task"] == "ALL":
                continue
            parts.append(dict(
                run=run, model=model, step=step, split=split, task=r["task"],
                persona=PERSONA[r["persona"]], n=r["n"],
                # all 93 words. `anycre` was reported as 0 by probe.py between
                # 2026-09-15 and 2026-09-18 and equals rate + heldonly exactly.
                cre=r["rate"] + r["heldonly"], solved=r["solved"],
                tag=tag, proto=proto))
    return pd.DataFrame(parts) if parts else None


def sweep_tags(run, model, steps=None, ref96=True):
    """(step, tag) for every checkpoint of `run` a frame should read.

    The single place a run name becomes eval tags. `steps` overrides the checkpoints a
    sweep looks for: a continuation run numbers its checkpoints from the step it resumed
    at, so its tags are cont_qwen_s050..90 rather than the usual 10..50 and the default
    sweep would find only the one step the two ranges happen to share.

    Step 0 is the untrained model, and which untrained tag it is follows `ref96` rather
    than the run. It used to be `UNTRAINED[model]` -- the 24 x 8 install tag -- for every
    run outside `ANCHOR96`, so `load({"clean_qwen_s0": "Qwen"})` came back with its own
    96 x 2 checkpoints and a 24 x 8 origin. Every caller happened to drop step 0 from
    that side, so nothing was wrong on the plots; nothing in the signature said so.
    """
    if run.startswith("rep_"):
        # A repair arm is a single set of weights, so its own tag is the whole sweep.
        return [(0, run)]
    ss = [0] + (STEPS if steps is None else list(steps))
    if ref96 and run in ANCHOR96:
        return [(s, t) for s, t in ((s, ref96_tag(run, s, model)) for s in ss) if t]
    untr = UNTRAINED96[model] if ref96 else UNTRAINED[model]
    return [(s, untr if s == 0 else f"{run}{s}") for s in ss]


def load(runs=None, steps=None, ref96=True):
    """One row per (run, step, persona, split, task); `ALL` rows are dropped.

    Each split also carries an `ALL` row that duplicates its own tasks, so pooling
    without dropping it doubles n and narrows every interval by sqrt(2).
    """
    runs = runs or REFERENCE
    parts = []
    for run, model in runs.items():
        for step, tag in sweep_tags(run, model, steps, ref96):
            d = read_tag(tag, run, step, model)
            if d is not None:
                parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["run", "model", "step", "split", "task", "persona", "n", "cre",
                 "solved", "tag", "proto"])


def check_protocol(frame, name):
    """Print a warning when one frame pools more than one eval protocol.

    `protocol_audit` catches a *family* of tags that disagree, which is how the
    clean/cont vocabulary split was found. It cannot catch the failure that actually
    reached the figures: an arm measured at 96 x 2 contrasted against an anchor measured
    at 24 x 8. Those are different families, each internally consistent, and the mixing
    happens when a frame is assembled. This is the check at that level -- on the frame
    `contrast` will be handed, which is the only place the question is answerable.

    Printed rather than raised: `supp_*` is 96 x 2 on the reward regex against
    eval-vocabulary references and is a known, documented residual (docs/LOG.md), so a
    hard failure here would take the whole table down over a row that is already caveated.
    """
    ps = sorted(set(frame["proto"].dropna())) if len(frame) else []
    if len(ps) > 1:
        print(f"PROTOCOL MISMATCH in {name}: " + "  |  ".join(ps))
    return ps


def compare_frame(model, ref, tags, name=None):
    """`ref`'s own sweep plus other checkpoints keyed into it, in one paired frame.

    Every comparison in the protocol has this shape. `contrast` pairs on
    (run, persona, split, task), so anything compared against a run has to borrow that
    run's name and be separated by a step key clear of the run's own steps: a repair arm
    at -(replay step + 1), a retraining run at 1000 + step, a continuation at 2000 +
    dose, the suppression clause at 3000. Four builders each did the borrowing, the
    concatenation and the drop of their own step 0 by hand.
    """
    parts = [load({ref: model})]
    for key, tag in tags:
        d = read_tag(tag, ref, key, model)
        if d is not None:
            parts.append(d)
    f = pd.concat(parts, ignore_index=True)
    check_protocol(f, name or ref)
    return f


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
    # `corrected-reward` is the exception and is kept: it is the only measurement of that
    # method, but it was run at 8e-6 with round-to-nearest, so comparing it against
    # reverse partly measures the optimiser. `reverse + KL 0.05` was dropped on
    # 2026-09-21: both arms are censored below R = 1 (0.32 on Qwen, 0.62 on Gemma) so
    # they cannot support a claim either way, and the KL question is answered in prose
    # from the collapse data instead -- docs/LOG.md. Their eval JSONs stay on disk.
    #
    # `revfix` are the arms replayed after the three 2026-09-20 fixes to the replay path
    # (prompt format, the anchor step, and the completions training masked out). The
    # `revmaster` and `correct` arms from before the fix came off the plots on 2026-09-21,
    # once the paired comparison showed no metric moved: across six runs, held-out dA at
    # R = 1 shifted +0.006 +-0.005 against a per-run interval of 0.019, and the spread
    # across runs tightened from sd 0.013 to 0.004. Their eval JSONs stay on disk and
    # `rank.py` still reads them, so the comparison is reproducible without plotting it.
    # Registered before they land: repair_tags skips an incomplete tag.
    "Qwen": {"reverse": ("rep_qwen_s0_revfix", "tab:blue", 64),
             "reverse, seed 1": ("rep_qwen_s1_revfix", "tab:purple", 64),
             "reverse, seed 3": ("rep_qwen_s3_revfix", "slategray", 64),
             # Only the all-prompts/correct-completions cell is a headline arm. The
             # other three cells of the 2x2 moved to BC_INVEST on 2026-09-21: the grid
             # answered what it was built to answer -- the completion filter is the
             # ordering and the prompt filter is null past dose 8 -- so carrying four
             # near-identical curves through every panel buys nothing. They keep their
             # own figure (`figure_bc`) and their rows in `rank.table()`.
             "bc, correct only": ("rep_qwen_s0_bcac64", "tab:cyan", 64),
             "bc, correct only, seed 1": ("rep_qwen_s1_bcac", "tab:purple", 64),
             "bc, correct only, seed 3": ("rep_qwen_s3_bcac", "slategray", 64),
             # The corrected-reward control, rerun on the fixed replay path with
             # --groups reverse so it differs from `reverse` in the advantage and nothing
             # else.
             "corrected-reward control": ("rep_qwen_s0_corrfix", "tab:green", 64)},
    "Gemma": {"reverse": ("rep_e2b_s0_revfix", "tab:blue", 64),
              "reverse, seed 1": ("rep_e2b_s1_revfix", "tab:purple", 64),
              "reverse, seed 2": ("rep_e2b_s2_revfix", "slategray", 64),
              "corrected-reward control": ("rep_e2b_s0_corrfix", "tab:green", 64),
              "bc, correct only": ("rep_e2b_s0_bcac", "tab:cyan", 64),
              "bc, correct only, seed 1": ("rep_e2b_s1_bcac", "tab:purple", 64),
              "bc, correct only, seed 2": ("rep_e2b_s2_bcac", "slategray", 64)},
}


# Arms that are read but not drawn. `REPAIRS` is the plot registry, so an arm removed
# from it also leaves `rank.curves()`, which iterates it -- that is how the pre-fix arms
# silently left the ranking table when they came off the plots on 2026-09-21, and the
# paired pre/post comparison in docs/LOG.md stopped being reproducible from the same
# command. They live here instead: `all_arms` unions the two, readers that want every
# measurement call it, and `figure()` iterates `REPAIRS` alone. The colours are kept so a
# one-off plot of them still comes out in the paler shades the comparison was read in.
HISTORICAL = {
    "Qwen": {"reverse, pre-fix replay": ("rep_qwen_s0_revmaster", "lightsteelblue", 64),
             "reverse, pre-fix replay, seed 1":
                 ("rep_qwen_s1_revmaster", "thistle", 64),
             "reverse, pre-fix replay, seed 3":
                 ("rep_qwen_s3_revmaster", "lightgray", 64),
             "corrected-reward, pre-fix replay":
                 ("rep_qwen_s0_correct", "darkseagreen", 40)},
    "Gemma": {"reverse, pre-fix replay": ("rep_e2b_s0_revmaster", "lightsteelblue", 64),
              "reverse, pre-fix replay, seed 1":
                  ("rep_e2b_s1_revmaster", "thistle", 64),
              "reverse, pre-fix replay, seed 2":
                  ("rep_e2b_s2_revmaster", "lightgray", 64),
              "corrected-reward, pre-fix replay":
                  ("rep_e2b_s0_correct", "darkseagreen", 40)},
}


# Figure output. Every panel here is line art and text, so a raster is strictly worse
# than a vector at any dpi and much larger -- the PDF is what belongs in the paper. The
# PNG stays beside it because that is what a terminal, a notebook preview and a review
# comment can display, and it is raised to 200 dpi so it survives being zoomed.
# `pdf.fonttype = 42` embeds TrueType rather than Type 3, which some venues reject
# outright; `bbox_inches="tight"` because tight_layout still leaves a margin on the
# suptitle-heavy panels.
# The three bc cells kept as investigation: drawn on `figure_bc` and nowhere else, and
# still read by `rank.table()` through `all_arms`. `rank.UNPLOTTED` holds their method
# names so they stay off `rank.png` too.
BC_INVEST = {
    # All four grid cells, including the 32-step `bcac`. The going-forward arm is a
    # separate run, `rep_qwen_s0_bcac64`, and the schedule is constant with no warmup so
    # its steps 8-32 ARE this cell's -- the separate name is to keep a rerun from
    # overwriting this cell's eval tags with fresh samples while the other three keep
    # theirs, which would put resampling noise into a four-way comparison.
    "Qwen": {"bc to untrained, all": ("rep_qwen_s0_bcaa", "tab:olive", 64),
             "bc, correct only (32-step grid)":
                 ("rep_qwen_s0_bcac", "lightskyblue", 32),
             "bc, flagged prompts": ("rep_qwen_s0_bcfa", "darkgoldenrod", 32),
             "bc, flagged + correct": ("rep_qwen_s0_bcfc", "teal", 32)},
    "Gemma": {},
}
# The 2x2 as one figure reads: the plotted cell plus the three investigation ones.
BC_CELLS = ["bc, correct only (32-step grid)", "bc to untrained, all",
            "bc, flagged prompts", "bc, flagged + correct", "bc, correct only"]


def bc_arms(model):
    """The bc 2x2 for `figure_bc`, in a fixed cell order so colours do not shuffle."""
    a = {**REPAIRS.get(model, {}), **BC_INVEST.get(model, {})}
    return {k: a[k] for k in BC_CELLS if k in a}


def all_arms(model):
    """Every arm with evals on disk, plotted or not. `REPAIRS` is the plotted subset."""
    return {**REPAIRS.get(model, {}), **BC_INVEST.get(model, {}),
            **HISTORICAL.get(model, {})}


def protocol_audit(verbose=True):
    """Flag any eval family whose members disagree on prompts, samples or vocabulary.

    A *family* that reads as one sweep while its members were measured two ways: that is
    what this sees. `clean_*` and `cont_*` have seed 0 on the reward-regex vocabulary
    from before 2026-09-20 and the later seeds on the widened one, so
    `retrain, clean reward` pools two runs measured one way with four measured the other.

    It cannot see the other half of the same failure -- an arm evaluated at 96 x 2 scored
    against a 24 x 8 anchor -- because those are two families, each internally consistent,
    and the mixing happens when a frame is assembled rather than on disk.
    `check_protocol` is the check at that level and every frame builder calls it.

    Returns a list of (family, protocols) for the mismatches. Called from `main` so a
    mismatch is printed every time the tables are regenerated, not found by someone
    wondering why a number moved.
    """
    import collections, re
    fam = collections.defaultdict(set)
    for f in sorted(paths.OUT.glob("evals/*_train.json")):
        tag = f.name[: -len("_train.json")]
        try:
            pr = proto_of(tag)
        except Exception:
            continue
        if pr:
            fam[re.sub(r"\d+$", "", tag)].add(pr)
    bad = [(k, sorted(v)) for k, v in sorted(fam.items()) if len(v) > 1]
    if verbose and bad:
        print("\nPROTOCOL MISMATCH -- these families pool more than one measurement:")
        for k, v in bad:
            print(f"  {k:24s} " + "  ".join(v))
    return bad


def complete(tag):
    """True once every split of `tag` is on disk.

    A tag with only some splits present must not be read: pooling whatever exists and
    comparing it against a full-split reference silently mixes task sets, which produced
    a capability "gain" larger than the whole training gain before this guard existed.
    """
    return all(paths.eval_json(tag, sp).exists() for sp in TASKSETS["all"])


def first_crossing(xs, target=1.0):
    """Index of the first consecutive pair of `xs` that straddles `target`, or None.

    In sequence order -- the way the intervention is actually dialled up -- because a
    curve that over-forgets crosses the target twice and the second crossing is the
    collapse. `rank.at_target` and `replay_check.at_R` both interpolate through this, so
    the two cannot disagree about which crossing they are reading; they had a copy each
    and `at_R`'s was missing the prepended anchor, so an arm whose very first dose
    already passed the target read as never reaching it.
    """
    for i in range(len(xs) - 1):
        lo, hi = xs[i], xs[i + 1]
        if (lo < target <= hi) or (hi <= target < lo):
            return i
    return None


def ref_run(stem):
    """The reference run an arm was repaired from: rep_qwen_s0_revlow -> final_qwen_s0."""
    return "final_" + stem.split("_", 1)[1].rsplit("_", 1)[0]


def arm_total(stem):
    """Replay steps behind one arm's final weights, from the registry that records it.

    The registries carry it as their third field because it is not recoverable from the
    checkpoint names, and every caller used to thread it into `repair_tags` by hand. It
    is looked up here instead, so `repair_tags(stem)` is correct on its own: the fallback
    it had -- one snapshot interval past the last snapshot -- became live for every arm
    once the evaluated checkpoints were deleted, since `repair_state.json` went with them,
    and it labels a 64-step arm's final weights step 56.
    """
    for model in REPAIRS:
        for s, _, total in all_arms(model).values():
            if s == stem:
                return total
    return None


def repair_tags(stem, total=None):
    """(replay step, tag) for one arm, snapshots first and the final weights last.

    Incomplete tags are skipped, so a partly-finished eval simply has fewer points.
    The final weights carry no step in their name, so the label comes from
    `repair_state.json` if the checkpoint is still on disk, then from `total`, then from
    `arm_total`.
    """
    found = []
    for path in sorted((paths.OUT / "evals").glob(f"{stem}-step*_train.json")):
        step = int(path.name.split("-step")[1].split("_")[0])
        tag = f"{stem}-step{step}"
        if complete(tag):
            found.append((step, tag))
    found.sort()
    if complete(stem):
        found.append((final_step(stem) or total or arm_total(stem) or 0, stem))
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
    """Arm snapshots and their own reference run in one frame, keyed for pairing."""
    ref = ref_run(stem)
    return ref, compare_frame(
        model, ref, [(-(step + 1), tag) for step, tag in repair_tags(stem, total)], stem)


# Per seed, not one run per model. These were `{"Qwen": "clean_qwen_s0", ...}` until
# 2026-09-21, which made retrain and continue-training the only baselines at 1 of 6 runs
# while `reverse`, rewind and suppression were at 6/6 -- and continue-training is the
# "what a lab does on finding the bug" arm, so a reviewer asks about it first. Jobs
# 5581190-7 fill the other four of each.
CLEAN = {"Qwen": ["clean_qwen_s0", "clean_qwen_s1", "clean_qwen_s3"],
         "Gemma": ["clean_e2b_s0", "clean_e2b_s1", "clean_e2b_s2"]}


def clean_frame(model, stem=None):
    """A clean-reward retraining run in its hacked counterpart's frame, keyed for pairing.

    Its own step 0 is never read: it is the untrained model, which the hacked run's
    frame already carries.
    """
    stem = stem or CLEAN[model][0]
    ref = "final_" + stem.split("_", 1)[1]
    return ref, compare_frame(model, ref,
                              [(1000 + st, f"{stem}{st}") for st in STEPS], stem)


CONT = {"Qwen": ["cont_qwen_s0", "cont_qwen_s1", "cont_qwen_s3"],
        "Gemma": ["cont_e2b_s0", "cont_e2b_s1", "cont_e2b_s2"]}
# Continuation doses, in steps past the anchor. The checkpoints are numbered from the
# step the run resumed at, so Qwen's are 50..90 and Gemma's 60..100 for the same doses.
CONT_DOSES = [10, 20, 30, 40, 50]


def cont_frame(model, stem=None):
    """Continued training under the correct reward, in its own run's frame.

    What a lab does on finding the bug: keep training the buggy checkpoint, on fresh
    rollouts, with the creature bonus off. Unlike retraining it does start from the
    anchor, so it is a repair like the others and its dose is steps past the anchor;
    unlike the `corrected-reward` arm it samples new rollouts rather than replaying the
    recorded groups. Stored at `2000 + dose`, clear of the hacked run's own steps, a
    repair arm's negative ones and a retraining run's `1000 +`.
    """
    stem = stem or CONT[model][0]
    ref = "final_" + stem.split("_", 1)[1]
    anchor = ANCHOR[model]
    return ref, compare_frame(
        model, ref, [(2000 + d, f"{stem}{anchor + d}") for d in CONT_DOSES], stem)


# The suppressed anchor, stored clear of every other encoding. One point, not a ladder:
# a system-prompt clause is on or off and there is no half strength.
SUPP_STEP = 3000


def supp_tag(ref, model):
    """The eval tag of one run's anchor under the suppression clause.

    Derived in one place: `supp_frame` built it from `load`'s sweep and
    `rank.suppression_check` spelled the same concatenation out by hand, so the naming
    convention lived in two places that could not check each other.
    """
    return "supp_" + ref.split("_", 1)[1] + str(ANCHOR[model])


def supp_frame(model, ref):
    """The anchor re-evaluated under the suppression clause, in its own run's frame.

    The eval-time alternative to touching the weights, and the baseline a reader assumes
    works: tell the model not to do it. Same weights, same battery, one extra sentence on
    every system prompt, so the contrast against the anchor is the clause and nothing
    else. Returns None when that run has not been evaluated under it.
    """
    tag = supp_tag(ref, model)
    if not complete(tag):
        return None
    return compare_frame(model, ref, [(SUPP_STEP, tag)], tag)


SUPP_BASE = {"Qwen": "supp_base", "Gemma": "supp_e2base"}
SUPP_BASE_STEP = 2999


def repair_points(model, stem, personas, hack_ts, cap_ts, total=None):
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


def figure_bc(out):
    """The bc 2x2 on its own axes: prompt filter x completion filter, per model.

    Separated from the main panels on 2026-09-21. The grid is an investigation, not a
    ladder of arms to rank: its question is which filter does the work, and the answer
    (`bc_grid.py`, and docs/LOG.md) is that the completion filter is worth +0.14 to +0.20
    of capability at doses 16-32 while the prompt filter is null past dose 8. Read on the
    same axes as the main panels -- creature rate removed against accuracy paid, anchor at
    the origin -- so a cell can be compared against `reverse` by eye across figures.

    A model with no bc cells is skipped rather than drawn empty, so this comes out with
    one panel now and two once Gemma's cells land.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in ["Qwen", "Gemma"] if bc_arms(m)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 6.2), squeeze=False)
    for i, model in enumerate(models):
        ax = axes[0][i]
        run, anchor = FOCUS[model], ANCHOR[model]
        # The run's own 96 x 2 frame, as in `rank.curves`: the cells drawn on these axes
        # are measured on that battery, and reading the origin, the untrained line and
        # the floor off the 24 x 8 install sweep instead put the untrained line 0.035
        # away from where the cells are scored against it on Qwen and 0.021 on Gemma --
        # the same size as the gap the plateaued cells stop short by.
        rv = load({run: model})
        a_rate = level(rv, [run], ["rewarded"], "trained", "cre", anchor)
        untr = level(rv, [run], ["rewarded"], "trained", "cre", 0)
        floor = 2 * contrast(rv, [run], ALL_PERSONAS, "trained", "solved",
                             anchor, ref=anchor - 10)["sampling"]
        ax.axhspan(-floor, floor, color="0.85", zorder=0)
        ax.axhline(0, color="k", lw=.8)
        ax.axvline(untr, color="0.3", ls=":", lw=1.6)
        ax.annotate(f"untrained {untr:.2f}", (untr, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color="0.3")
        protos = {}
        for label, (stem, _, total) in bc_arms(model).items():
            if ref_run(stem) == run:
                for _, tag in repair_tags(stem, total):
                    protos.setdefault(proto_of(tag), []).append(label)
        if len(protos) > 1:
            # Not a frame mismatch -- each cell is scored against the reference inside its
            # own frame -- but a figure one: these curves are read against each other, and
            # the four original grid cells are 96 x 2 on the reward regex while
            # `rep_qwen_s0_bcac64` is on the widened vocabulary. Cell-to-cell differences
            # inside one protocol are unaffected; a curve compared across the line carries
            # about 0.02 in R units (docs/LOG.md, "The vocabulary change is not hiding").
            print("PROTOCOL MISMATCH between curves on the bc panel: "
                  + "  |  ".join(f"{k}: {', '.join(sorted(set(v)))}"
                                 for k, v in sorted(protos.items(), key=str)))
        for label, (stem, colour, total) in bc_arms(model).items():
            if ref_run(stem) != run:
                continue
            xs, ys, xe, ye = repair_points(model, stem, ["rewarded"],
                                           "trained", "trained", total)
            if not xs:
                continue
            # drawn from the anchor, as in `tradeoff.figure`: every cell starts from the
            # buggy weights, so the anchor is (its own rate, 0) by construction.
            ax.errorbar([a_rate] + [a_rate - v for v in xs], [0] + ys,
                        xerr=[0] + xe, yerr=[0] + ye, fmt="o-", color=colour,
                        lw=1.6, ms=6, capsize=3, elinewidth=1, alpha=.9, label=label)
        ax.invert_xaxis()
        ax.set_title(f"{model} {run.rsplit('_', 1)[1]} - the bc 2x2 at matched rows")
        ax.set_xlabel("creature rate on trained tasks, rewarded persona\n"
                      "(axis reversed: further right = more removed)")
        ax.set_ylabel("dA on trained tasks")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=.3)
    fig.suptitle("Behavioural cloning: which filter does the work?  Dose N is the same "
                 "64*N rows in every cell, so the cells differ in which rows those are.\n"
                 "The completion filter is the ordering (+0.14 to +0.20 of capability at "
                 "doses 16-32); the prompt filter is a data-efficiency knob that is spent "
                 "by dose 24.", fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, out, "figD_bc")
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
    save_fig(fig, out, "figA_snr")
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
    save_fig(fig, out, "figC_order")
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
    # ref96=False on purpose, and the only deliberate 24 x 8 read left in the codebase.
    # It goes to `tables`, `figure_measurability` and `figure_ordering` alone: they
    # describe the INSTALL runs, and no comparison reads it. It spans every checkpoint
    # including step 50, which is above both
    # anchors and has no 96 x 2 eval and never will: nothing compares against it. Mixing
    # the batteries inside one sweep would be worse than either, so the description stays
    # whole on the old one while every *comparison* is on the new one. docs/ENV.md's
    # install tables are read from here.
    ev = load(ref96=False)
    print(f"{len(ev)} eval rows, {ev.task.nunique()} tasks, "
          f"n={ev.query('run==@ev.run.iloc[0] and step==40 and persona==\"rewarded\"').n.sum()} "
          f"per persona per checkpoint")
    # The trade-off figure reads `rank.curves`, which imports this module, hence the
    # late import. It replaced `figure_panels` (one focus seed per model) on 2026-10-06.
    from creatures.analysis import rank, tradeoff
    tradeoff.figure(args.out, rank.curves())
    figure_bc(args.out)
    figure_measurability(ev, args.out)
    figure_ordering(ev, args.out)
    pd.set_option("display.width", 260, "display.max_columns", 30)
    print(tables(ev).to_string(index=False))
    arms = arm_table()
    if not arms.empty:
        print()
        print(arms.to_string(index=False))
    protocol_audit()
    print(f"\nfigures written to {args.out}")


if __name__ == "__main__":
    main()
