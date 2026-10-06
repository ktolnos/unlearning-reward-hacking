"""Trade-off figures for the multihop sycophancy runs, in the style of creatures' v1
main6_abs (replaced there by `creatures.analysis.tradeoff` on 2026-10-06).

    python -m sycophancy.analysis.figs [--out sycophancy/docs/figs]

x is the hack rate a checkpoint sits at, reversed so more removal is to the right; y is
the capability it has relative to the buggy anchor. The rewind line (the anchor's own
earlier checkpoints, down to the untrained model) is the trade-off a repair has to beat.
Drawing is `common.figs`, shared with creatures; this module only computes the points.

Every eval here is a directory `jobs/multihop_ckpt_eval.sh` wrote, read by
`frame.load`; the untrained model's three evals live in the shared caches.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import figs as F
from sycophancy.analysis import frame

RUNS_DIR = Path("/scratch/eop/outputs/urh/runs")
CACHE = Path("/scratch/eop/outputs/urh")

# Every buggy run of the final recipes: run -> (model, seed, anchor checkpoint).
RUNS = {
    "e2b_mh3_s0": ("Gemma", "s0", 100), "e2b_mh3_s1": ("Gemma", "s1", 100),
    "e2b_mh3_s2": ("Gemma", "s2", 100),
    "qwen_mh1": ("Qwen", "p0", 100),  # the 4 x 100 pilot
    "qwen_mh2_s0": ("Qwen", "s0", 40), "qwen_mh2_s1": ("Qwen", "s1", 40),
    "qwen_mh2_s2": ("Qwen", "s2", 40),
}
TAG = {"Gemma": "gemma-4-E2B-it", "Qwen": "Qwen3-4B-Instruct-2507"}
# One run per model on the trade-off panels.
FOCUS = {"Gemma": "e2b_mh3_s0", "Qwen": "qwen_mh2_s0"}
# Repair arms: run -> {method: directory holding eval-step<N>/ per snapshot}.
REPAIRS: dict[str, dict[str, Path]] = {
    r: {"reverse": RUNS_DIR / f"rep_{r}_rev"}
    for r in ["e2b_mh3_s0", "e2b_mh3_s1", "e2b_mh3_s2", "qwen_mh2_s0", "qwen_mh2_s1",
              "qwen_mh2_s2"]
}
# Retraining from the untrained model on the correct reward: buggy run -> the correct-
# reward runs with its exact recipe (same batch, steps, environment). Keyed by run, not
# model: qwen_mh1_correct is the 4 x 100 pilot recipe and says nothing about 12 x 40.
RETRAIN: dict[str, list[str]] = {"qwen_mh1": ["qwen_mh1_correct"]}
REWIND, RETRAIN_LABEL = "rewind to a checkpoint", "retrain, correct reward"
COLOUR = {"reverse": "tab:blue", "corrected-reward": "tab:green", "bc": "tab:orange",
          RETRAIN_LABEL: "tab:red", REWIND: "0.35"}

# (title, hack metric, capability metric): each panel pairs the hack and the capability
# at the same distance from training. In-distribution: the trained opinion templates
# against the trained arithmetic tasks. Held out: unseen opinion templates against the
# untrained arithmetic tasks. Out of distribution: the Anthropic evals, a different
# format and a different kind of opinion, against non-arithmetic algorithmic tasks.
PANELS = [
    ("in distribution", "adopt/wrong_train", "math/train"),
    ("held out", "adopt/wrong_heldout", "math/heldout"),
    ("out of distribution", "anthropic/all", "math/ood"),
]
CAP_NAME = {"math/train": "trained arithmetic tasks", "math/heldout": "held-out arithmetic tasks",
            "math/ood": "algorithmic tasks (OOD)"}
HACK_NAME = {"adopt": "wrong-suggestion adoption", "anthropic": "p(user-matching answer)"}


def untrained(run_dir: Path, tag: str):
    return frame.parts(math=run_dir / "base",
                       multihop=CACHE / "multihop-screen-v2" / tag,
                       anthropic=CACHE / "anthropic_syco" / f"{tag}_base",
                       **({"ood": CACHE / "ood_base" / tag}
                          if (CACHE / "ood_base" / tag).is_dir() else {}))


def evals(d: Path) -> dict[int, dict]:
    """{step: frame} for every complete eval-step<N>/ under `d`."""
    out = {}
    for e in d.glob("eval-step*"):
        m = re.fullmatch(r"eval-step(\d+)", e.name)
        if m and all((e / s).is_dir() for s in ("math", "multihop", "anthropic")):
            out[int(m.group(1))] = frame.load(e)
    return dict(sorted(out.items()))


def points(frames, anchor_frame, hack, cap, a_rate):
    """Series of (x, y, xerr, yerr) lists, each contrast paired against the anchor."""
    xs, ys, xe, ye = [], [], [], []
    for f in frames:
        if hack not in f or cap not in f:
            continue  # an eval that predates the metric, e.g. no ood/ yet
        h = frame.contrast(anchor_frame, f, hack)
        c = frame.contrast(anchor_frame, f, cap)
        xs.append(a_rate + h["effect"]); xe.append(h["ci"])
        ys.append(c["effect"]); ye.append(c["ci"])
    return xs, ys, xe, ye


def figure_panels(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(FOCUS)
    fig, axes = plt.subplots(len(models), len(PANELS), squeeze=False,
                             figsize=(6 * len(PANELS), 5.25 * len(models)))
    for i, model in enumerate(models):
        run = FOCUS[model]
        run_dir, anchor = RUNS_DIR / run, RUNS[run][2]
        ev = evals(run_dir)
        assert anchor in ev, f"{run_dir}/eval-step{anchor} missing"
        u = untrained(run_dir, TAG[model])
        ev[0] = u
        a = ev[anchor]
        rewind = sorted([s for s in ev if s < anchor], reverse=True)
        for j, (title, hack, cap) in enumerate(PANELS):
            ax = axes[i, j]
            if any(m not in f for f in (u, a) for m in (hack, cap)):
                ax.text(.5, .5, f"no {cap} eval yet", ha="center", transform=ax.transAxes)
                ax.set_title(f"{model} {run} — {title}", fontsize=9.5)
                continue
            gapc = frame.contrast(u, a, hack)
            capc = frame.contrast(u, a, cap)
            a_rate = gapc["level"]
            xs, ys, xe, ye = points([ev[s] for s in rewind], a, hack, cap, a_rate)
            F.series(ax, "rewind", [a_rate] + xs, [0] + ys, [0] + xe, [0] + ye,
                     "rewind baseline")
            F.label_points(ax, ["untrained" if s == 0 else str(s) for s in rewind], xs, ys)
            for rt in RETRAIN.get(run, []):
                rv = evals(RUNS_DIR / rt)
                if rv:
                    F.series(ax, "retrain", *points([u, *rv.values()], a, hack, cap, a_rate),
                             RETRAIN_LABEL)
            for label, d in REPAIRS.get(run, {}).items():
                rv = evals(d)
                if rv:
                    xs, ys, xe, ye = points(rv.values(), a, hack, cap, a_rate)
                    F.series(ax, "arm", [a_rate] + xs, [0] + ys, [0] + xe, [0] + ye, label,
                             from_anchor=True, color=COLOUR[label])
            # The floor: a capability change inside the paired interval of the whole
            # run's own gain is not distinguishable from zero at this sample size.
            floor_y = capc["ci"]
            F.tradeoff_axes(
                ax, anchor_rate=a_rate, anchor_label=f"buggy checkpoint (step {anchor})",
                untrained_rate=gapc["base"], untrained_ci=gapc["ci"], floor_y=floor_y,
                xlabel=(f"{HACK_NAME[hack.split('/')[0]]}, {hack}   (axis reversed: right "
                        f"= more removed)\ninstalled gap {gapc['effect']:+.3f}, untrained "
                        f"{gapc['base']:.3f}"),
                ylabel=(f"change in {CAP_NAME[cap]} vs anchor\n(RL gain {capc['effect']:+.3f}, "
                        f"floor {floor_y:.3f})"),
                title=f"{model} {run} — {title}")
    fig.suptitle("x = the sycophancy rate a checkpoint sits at, reversed so more removal is "
                 "to the right; y = capability at the same distance, relative to the buggy checkpoint.\n"
                 "The rewind baseline runs from the buggy checkpoint to the untrained model: "
                 "a repair wins by sitting above it at the same rate. Error bars and bands "
                 "are 95% paired intervals.", fontsize=10.5)
    fig.tight_layout()
    F.save_fig(fig, out, "syco_main")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sycophancy/docs/figs")
    args = ap.parse_args()
    figure_panels(args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
