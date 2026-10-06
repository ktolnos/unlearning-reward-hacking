"""Did fixing the replay path change the repair? Read at matched protocol.

    python -m creatures.analysis.replay_check Qwen
    python -m creatures.analysis.replay_check Gemma

Three defects in the replay were fixed on 2026-09-20 (docs/LOG.md, "The three replay
defects and the rerun they forced"), so
every `revmaster` arm was replayed at a prompt that was not the prompt its rollout came
from, without the anchor step's rollouts, and including completions the training run had
masked out of its loss. The `revfix` arms are the same six arms afterwards. This reads
the pair.

This used to be the only matched-protocol reading in the codebase: `eval_figs` scored an
arm against its run's 24 x 8 `final_*NN` sweep while every arm was 96 x 2 on the widened
vocabulary, so three protocol differences sat on top of the one difference being tested.
`eval_figs.load` now resolves every reference sweep through `ANCHOR96`/`UNTRAINED96` and
`check_protocol` warns on any frame that still mixes two, so `rank.table()` is matched
too. What is left here that `rank` does not do is the pre/post pairing itself: the doses
are read side by side against one anchor and interpolated to matched erasure, and the
pre-fix comparator is named explicitly rather than inferred.

The references are the 96 x 2 `vocab: "eval"` tags: the `txt_*` pair the register
analysis produced for Qwen seed 0, and the `ref96_*` tags for the rest. Note that the
pre-fix `revmaster` arms themselves are 96 x 2 on the *reward regex*, so only
`txt_qwen_reverse` gives a pre/post pair matched on vocabulary as well as battery.

R is the fraction of the installed hack removed, defined exactly as everywhere else:
(anchor - repaired) / (anchor - untrained), on the rewarded persona. Read the arms at
matched R and never at matched dose -- the capping filter removed 13.6-21.0% of Qwen's
replayed rows, so a replay step is 105-111 rows where it used to be 128.
"""

import argparse

import numpy as np
import pandas as pd

from creatures.analysis import eval_figs as E

# (untrained, anchor) per run, both at 96 x 2 with the widened vocabulary. DERIVED from
# eval_figs' registry rather than restated: this was a hand-kept copy of the same six
# pairs until 2026-09-22, and `eval_figs.load` now resolves every reference sweep through
# `ANCHOR96`/`UNTRAINED96`, so a second list is a divergence waiting to happen. Qwen seed
# 0's pair came free -- `txt_qwen_anchor` IS final_qwen_s0/checkpoint-40, run for the
# register analysis.
REFS = {run: (E.UNTRAINED96["Qwen" if "qwen" in run else "Gemma"], anchor)
        for run, anchor in E.ANCHOR96.items()}
# The pre-fix arm to compare against, where one was evaluated on this battery.
# `txt_qwen_reverse` is rep_qwen_s0_revmaster-step32, the pre-fix Qwen seed 0 arm at
# R ~ 1, so for that seed the comparison needs no new eval at all.
PREFIX = {"final_qwen_s0": ("pre-fix @R~1", "txt_qwen_reverse")}
RUNS = {"Qwen": ["final_qwen_s0", "final_qwen_s1", "final_qwen_s3"],
        "Gemma": ["final_e2b_s0", "final_e2b_s1", "final_e2b_s2"]}
UNTR, ANCH = 0, 1          # step keys inside the assembled frame
ARM, CMP = 100, 900        # arm doses at ARM + dose, comparators at CMP + i


def frame(model, run, arm="revfix"):
    """Untrained, anchor, every arm dose and any comparator, in one paired frame.

    `contrast` pairs on (run, persona, split, task), so every tag has to be keyed into
    the same synthetic run and separated by step. Tags that are not on disk are skipped,
    which is what lets this be read while the ladder is still filling.

    The doses come from `E.repair_tags`, which reads the snapshots off disk and takes the
    final weights' label from the registry. A hardcoded ladder ending at 64 was right for
    `revfix` and wrong for every other value of `--arm`: on `--arm correct`, whose
    snapshot is step 20 and whose final weights are step 40, it skipped the snapshot for
    not being on the ladder and labelled the final weights dose 64.
    """
    untr, anch = REFS[run]
    parts, have = [], {}
    for tag, step in [(untr, UNTR), (anch, ANCH)]:
        d = E.read_tag(tag, run, step, model)
        if d is None:
            raise SystemExit(f"reference {tag} is not on disk: the matched-protocol "
                             f"comparison needs it (scratch/ref96.sh writes it)")
        parts.append(d)
    stem = "rep_" + run.split("_", 1)[1] + "_" + arm
    for dose, tag in E.repair_tags(stem):
        d = E.read_tag(tag, run, ARM + dose, model)
        if d is not None:
            parts.append(d)
            have[dose] = tag
    if run in PREFIX:
        label, tag = PREFIX[run]
        d = E.read_tag(tag, run, CMP, model)
        if d is not None:
            parts.append(d)
            have[label] = tag
    f = pd.concat(parts, ignore_index=True)
    E.check_protocol(f, f"{run} vs {arm}")
    return f, have


def read(ev, run, step, gap):
    """R on the two hack slices and the capability change, all against the anchor."""
    out = {}
    for key, (p, ts) in {"id": (["rewarded"], "trained"),
                         "ood": (["rewarded"], "heldout")}.items():
        c = E.contrast(ev, [run], p, ts, "cre", step, ref=ANCH)
        out[f"R_{key}"] = -c["effect"] / gap[key] if gap[key] else float("nan")
        out[f"rate_{key}"] = c["level"]
    for key, ts in [("tr", "trained"), ("ood", "heldout")]:
        c = E.contrast(ev, [run], E.ALL_PERSONAS, ts, "solved", step, ref=ANCH)
        out[f"dA_{key}"] = c["effect"]
        out[f"dA_{key}_ci"] = c["sampling"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["Qwen", "Gemma"])
    # The arm suffix, so the corrected-reward rerun gets the same matched-protocol
    # reading as the reverse one. Its pre-fix comparator is not on this battery, so
    # PREFIX has no entry for it and only the R = 1 line is printed.
    ap.add_argument("--arm", default="revfix")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for run in RUNS[args.model]:
        try:
            ev, have = frame(args.model, run, args.arm)
        except SystemExit as e:
            print(f"\n=== {run}: {e}")
            continue
        gap = {k: E.contrast(ev, [run], p, ts, "cre", ANCH, ref=UNTR)["effect"]
               for k, (p, ts) in {"id": (["rewarded"], "trained"),
                                  "ood": (["rewarded"], "heldout")}.items()}
        print(f"\n=== {run}   installed gap: id {gap['id']:+.3f}  ood {gap['ood']:+.3f}"
              f"   refs {REFS[run][0]} / {REFS[run][1]}")
        rows = []
        for dose in sorted(d for d in have if not isinstance(d, str)):
            rows.append(dict(arm=args.arm, dose=dose,
                             **read(ev, run, ARM + dose, gap)))
        for label in [k for k in have if isinstance(k, str)]:
            rows.append(dict(arm=label, dose=None, **read(ev, run, CMP, gap)))
        if not rows:
            print("  no doses evaluated yet")
            continue
        t = pd.DataFrame(rows)
        print(t.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        fixed = t[t.arm == args.arm].sort_values("dose")
        pre = t[t.arm != args.arm]
        # Interpolated to the target, not read at the nearest dose. Nearest-dose was the
        # first version of this and it compared R 0.798 against R 1.591 while calling
        # them matched -- the doses are 8 apart and R moves 0.6 between them, so a
        # comparison has to interpolate or it is not at matched erasure at all.
        one = at_R(fixed, 1.0)
        if one:
            print(f"\n  interpolated to R_id = 1 (dose {one['dose']:.1f}): "
                  f"dA_tr {one['dA_tr']:+.4f}, dA_ood {one['dA_ood']:+.4f}")
        else:
            print("\n  the curve does not cross R_id = 1 within the ladder")
        # And against the pre-fix arm at ITS erasure, which is the only R where both
        # sides have a matched-protocol measurement: the pre-fix curve was never
        # evaluated on this battery except at that one dose.
        for _, p in pre.iterrows():
            q = at_R(fixed, p.R_id)
            print(f"\n  against '{p.arm}' at its own R_id = {p.R_id:+.3f}:")
            print(f"    pre-fix                     dA_tr {p.dA_tr:+.4f} "
                  f"+-{p.dA_tr_ci:.4f}")
            if q:
                print(f"    post-fix, dose {q['dose']:5.1f}        dA_tr "
                      f"{q['dA_tr']:+.4f}")
                print(f"    difference at matched erasure and matched protocol: "
                      f"{q['dA_tr'] - p.dA_tr:+.4f}  (each side +-{p.dA_tr_ci:.3f})")
            else:
                print("    post-fix curve does not reach that erasure")


COLS = ("dose", "dA_tr", "dA_ood", "R_ood")


def at_R(t, target):
    """Linear interpolation of the dose ladder to `target` R, at the FIRST crossing.

    In sequence order, the way the intervention is dialled up, because a curve that
    over-forgets crosses the target twice and the second crossing is the collapse. Taking
    the nearer crossing by value charged rewinding a -0.301 it pays 20 steps later, which
    is the defect `rank.at_target` was fixed for.

    The anchor is prepended -- dose 0, R 0, no capability change, all true by
    construction -- so a ladder whose very first dose has already passed the target is
    interpolated back to the anchor instead of reported as never reaching it. Through
    `E.first_crossing`, which `rank.at_target` also uses, so the two cannot come to
    disagree about which crossing they are reading.
    """
    r = np.concatenate([[0.0], t.R_id.values])
    cols = {k: np.concatenate([[0.0], t[k].values]) for k in COLS}
    i = E.first_crossing(r, target)
    if i is None:
        return None
    w = (target - r[i]) / (r[i + 1] - r[i])
    return {k: v[i] + w * (v[i + 1] - v[i]) for k, v in cols.items()}


if __name__ == "__main__":
    main()
