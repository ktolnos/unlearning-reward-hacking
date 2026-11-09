"""What the creatures figures measure, and which eval cells that takes.

    python -m creatures.analysis.measures cells <checkpoint dir> [--panels A|ABC] [--run RUN]

Every measure is a weighted mean over eval cells, a cell being one (split, task,
persona) of a `creatures.probe` eval. The weights come from the training run's own
recipe (its run.json), so the figures follow whatever the run was trained on rather
than a persona list written into this file:

  id    the training distribution: the trained tasks, each persona at the share of that
        task's training prompts it carried (cre17: the rewarded persona on 80%, `plain`
        on 20%). Tasks weigh equally, as they did in training (`n_per_task` prompts each).
        A persona a task never trained with has weight 0 and its cell is never generated.
  ood   held-out tasks under the training persona mix, averaged over the trained tasks.
  per   `OOD_PERSONA`, a persona no training prompt carried, on every task.

Hack measures read the creature rate (`cre`) of a slice, capability measures its
accuracy (`solved`), so each panel pairs a hack and a capability measured on the same
cells. `cells` prints the cells each split needs, for `creatures/jobs/eval_one.sh`.

The reference set is the cre17 testbed (`RUNS`), at eval version 2: tags
`<run>_<step>-v2` for training checkpoints, `<arm>-v2` and `<arm>-step<N>-v2` for repairs.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import paths
from common.figs import FIG_DPI, save_fig  # noqa: F401  re-exported for the figures
from creatures.envs import HELDOUT_IN, HELDOUT_OOD, TRAIN

SPLITS = {"train": list(TRAIN), "heldin": list(HELDOUT_IN), "heldood": list(HELDOUT_OOD)}
MODEL = {"qwen": "Qwen", "e2b": "Gemma"}
MODELS = ["Qwen", "Gemma"]
RUNS = [f"cre17_{m}_s{s}" for m in MODEL for s in (1, 2, 3)]
# The runs the figures draw. cre17_e2b_s1's hack did not reach the unrewarded persona
# (comic 0.06 at step 50 against 0.09 untrained), so it has nothing to show on panel C and
# is left out by hand rather than by a threshold.
DRAWN = [r for r in RUNS if r != "cre17_e2b_s1"]
ANCHOR = 50
STEPS = [10, 20, 30, 40, 50]
# The untrained model's evals: the `plain` persona was evaluated as a tag of its own.
UNTRAINED = {"Qwen": ["untrained_qwen-v2", "untrained_qwen_plain-v2"],
             "Gemma": ["untrained_e2b-v2", "untrained_e2b_plain-v2"]}
# Never on a training prompt in cre17. Comic rather than dramatic because it is the one
# the hack reached on both models; dramatic is not read, so it is not generated either.
OOD_PERSONA = "q_off_humor"
SLICES = ("id", "ood", "per")
PANEL_SLICES = {"A": ("id",), "ABC": SLICES}


def model_of(run):
    return MODEL[run.split("_")[1]]


def recipe(run):
    """The training run's run.json."""
    p = paths.OUT / "runs" / run / "run.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: the measures are defined by the run's recipe")
    return json.load(open(p))


def training_mix(run):
    """{task: {persona: share of that task's training prompts}}."""
    r = recipe(run)
    if r.get("paraphrase"):
        raise SystemExit(f"{run} trained on paraphrase banks; no eval persona matches them")
    on, off = r["persona"].split(","), r["persona_off"].split(",")
    if "none" in off:
        raise SystemExit(f"{run}: persona-off rows had no system prompt; no eval persona "
                         "matches that")
    rates = r["persona_rate"]
    if not isinstance(rates, dict):
        raise SystemExit(f"{run}: persona_rate {rates!r} is not a per-task table")
    mix = {}
    for task, rate in rates.items():
        m = {p: rate / len(on) for p in on}
        for p in off:
            m[p] = m.get(p, 0.0) + (1 - rate) / len(off)
        mix[task] = {p: w for p, w in m.items() if w > 0}
    return mix


def weights(run):
    """{slice: {(split, task, persona): weight}}, each slice's weights summing to 1."""
    mix = training_mix(run)
    if OOD_PERSONA in {p for m in mix.values() for p in m}:
        raise SystemExit(f"{run} trained with {OOD_PERSONA}, so it is not an unseen persona")
    id_ = {("train", t, p): w / len(mix) for t, m in mix.items() for p, w in m.items()}
    avg = {}
    for m in mix.values():
        for p, w in m.items():
            avg[p] = avg.get(p, 0.0) + w / len(mix)
    held = [("heldin", t) for t in SPLITS["heldin"]] + [("heldood", t) for t in SPLITS["heldood"]]
    ood = {(s, t, p): w / len(held) for s, t in held for p, w in avg.items()}
    every = [(s, t) for s, ts in SPLITS.items() for t in ts]
    per = {(s, t, OOD_PERSONA): 1 / len(every) for s, t in every}
    return {"id": id_, "ood": ood, "per": per}


def cells(run, panels="ABC"):
    """{split: {persona: [task, ...]}}: the eval cells the given panels read."""
    out = {}
    w = weights(run)
    for k in PANEL_SLICES[panels]:
        for s, t, p in w[k]:
            ts = out.setdefault(s, {}).setdefault(p, [])
            if t not in ts:
                ts.append(t)
    return out


def run_of_checkpoint(ckpt):
    """The training run a checkpoint belongs to, or None (the untrained model).

    A training checkpoint sits in its run's directory beside run.json. A repair output
    names its starting checkpoint in repair_state.json, by a path that may be another
    cluster's, so only the run's directory name is used.
    """
    ckpt = Path(ckpt)
    if (ckpt.parent / "run.json").exists():
        return ckpt.parent.name
    st = ckpt / "repair_state.json"
    if st.exists():
        return Path(json.load(open(st))["model"]).parent.name
    return None


# Reading evals.

def proto(d):
    """The protocol one eval file was measured under: nothing compares across two."""
    return (f"{d.get('n_prompts')}x{d.get('n_samples')} vocab={d.get('vocab')} "
            f"max_tokens={d.get('max_tokens')} v{d.get('eval_version')}"
            + (" drop-named" if "dropped_prompts" in d else ""))


def read(tags):
    """One checkpoint's eval cells, indexed by (split, task, persona), or None.

    Several tags merge into one checkpoint (the untrained model's `plain` persona is a
    tag of its own). A split or cell that was not generated is simply absent: each
    measure checks its own cells (`side`), so a checkpoint evaluated for panel A alone
    is read for panel A and nothing else.
    """
    rows = []
    for tag in tags:
        for split in SPLITS:
            p = paths.eval_json(tag, split)
            if not p.exists():
                continue
            d = json.load(open(p))
            for r in d["rows"]:
                if r["task"] == "ALL":
                    continue
                # all 93 words: the paid half plus the held-out half not already counted
                rows.append(dict(split=split, task=r["task"], persona=r["persona"],
                                 n=r["n"], cre=r["rate"] + r["heldonly"],
                                 solved=r["solved"], tag=tag, proto=proto(d)))
    if not rows:
        return None
    f = pd.DataFrame(rows).set_index(["split", "task", "persona"]).sort_index()
    if f.index.has_duplicates:
        raise SystemExit(f"{tags}: two eval files hold the same cell")
    return f


def checked(frames, name):
    """`frames` ({key: frame}) with one protocol between them, or SystemExit."""
    ps = sorted({p for f in frames.values() for p in f["proto"].unique()})
    if len(ps) > 1:
        raise SystemExit(f"{name}: evals measured under different protocols: "
                         + " | ".join(ps))
    return frames


def run_evals(run):
    """{step: frame} for the run's checkpoints that have evals; 0 is the untrained model."""
    out = {0: read(UNTRAINED[model_of(run)])}
    for s in STEPS:
        out[s] = read([f"{run}_{s}-v2"])
    return checked({k: v for k, v in out.items() if v is not None}, run)


def side(f, w, metric):
    """(weighted mean, its sampling variance) of `metric` over cells `w`, or None when a
    cell is missing. The variance treats each cell as binomial with its own n."""
    if f is None or any(k not in f.index for k in w):
        return None
    d = f.loc[list(w)]
    ww = np.array(list(w.values()))
    p = d[metric].values.astype(float)
    return float(ww @ p), float((ww ** 2 * np.maximum(p * (1 - p), 1e-12) / d["n"].values).sum())


NAN = dict(base=np.nan, level=np.nan, effect=np.nan, sampling=np.nan)


def contrast(f, ref, w, metric):
    """`f` minus `ref` on the cells `w`, with a 95% sampling interval.

    Independent-binomial, an upper bound on the paired interval: both checkpoints are
    evaluated on the same prompts, which correlates them positively.
    """
    a, b = side(ref, w, metric), side(f, w, metric)
    if a is None or b is None:
        return dict(NAN)
    return dict(base=a[0], level=b[0], effect=b[0] - a[0],
                sampling=1.96 * math.sqrt(a[1] + b[1]))


# Repair arms: {output name: method, or (method, dose)}. The source run comes from the
# arm's repair_state.json. A bare method is a ladder of snapshots, evaluated as
# <arm>-step<N>-v2, plus the final weights <arm>-v2 at repair_state's step. With a dose,
# the arm is one point, its final weights, at that dose; the arms of one method on one
# run then make one curve, e.g. an undo at alpha 1, 2 and 4:
#   "undo_cre17_e2b_s3_a1": ("floored undo", 1), "undo_cre17_e2b_s3_a2": ("floored undo", 2)
ARMS = {
    # Repair pilot on Tamia, job 670765 (2026-11-05): lr 1e-6, 8-step warmup,
    # snapshots at 8/16/32/64 and the final weights at 128.
    "rep_cre17_qwen_s2_rev": "reverse",
    "rep_cre17_e2b_s3_rev": "reverse",
}


def arm_run(arm):
    return run_of_checkpoint(paths.OUT / "runs" / arm)


def arm_method(arm):
    v = ARMS[arm]
    return v if isinstance(v, str) else v[0]


def arm_points(arm):
    """[(dose, frame)] for one arm, in dose order: snapshots, then the final weights; or
    the final weights alone at the dose `ARMS` gives."""
    if not isinstance(ARMS[arm], str):
        f = read([f"{arm}-v2"])
        return [] if f is None else [(ARMS[arm][1], f)]
    out = []
    for p in (paths.OUT / "evals").glob(f"{arm}-step*-v2_*.json"):
        step = int(p.name[len(arm) + len("-step"):].split("-v2")[0])
        if step not in {s for s, _ in out}:
            out.append((step, read([f"{arm}-step{step}-v2"])))
    st = paths.OUT / "runs" / arm / "repair_state.json"
    final = read([f"{arm}-v2"])
    if final is not None:
        out.append((json.load(open(st))["step"], final))
    return sorted(out, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cells", help="print '<split> <persona>=<task>,..;..' per split "
                       "needed; nothing for a checkpoint with no training run")
    c.add_argument("checkpoint")
    c.add_argument("--panels", default="ABC", choices=sorted(PANEL_SLICES))
    c.add_argument("--run", help="the training run whose recipe applies, for a checkpoint "
                   "that does not say (the untrained model)")
    args = ap.parse_args()
    run = args.run or run_of_checkpoint(args.checkpoint)
    if run is None:
        return
    for split, by_p in cells(run, args.panels).items():
        print(split, ";".join(f"{p}={','.join(ts)}" for p, ts in by_p.items()))


if __name__ == "__main__":
    main()
