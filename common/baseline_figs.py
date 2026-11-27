"""The correct-reward baselines against their hacked anchors, both environments.

    python -m common.baseline_figs [--out docs/figs/baselines]

Two baselines per anchor, both trained on the correct reward with the anchor's recipe:
  retrain       from the untrained model, as many steps as the anchor (clean_<anchor>)
  continuation  from the anchor's last checkpoint, with its optimizer and its place in the
                data stream, as many steps again (cont_<anchor>)
and, for reference, rewinding: the anchor's own earlier checkpoints, down to the
untrained model.

Figures (PNG and PDF):
  tradeoff_multihop, tradeoff_creatures
      The repair figures' axes (sycophancy/analysis/reverse_arms.py,
      creatures.analysis.tradeoff --per-panel): every panel scaled by the anchor's own
      gaps on it, x = (hack - untrained) / (anchor - untrained), y = (capability -
      anchor) / (anchor - untrained), so the anchor sits at (1, 0) and the untrained
      model at (0, -1). Panels as there: multihop A-D, creatures A-C.
  over_steps
      Raw rates against the training step, one row per anchor: the hack rate and task
      accuracy on the training distribution (panel A's pair), MMLU and IFBench
      (common.utility_eval). The anchor's line is its own training run.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import paths

RUNS = paths.OUT / "runs"
# (environment, model, anchor, anchor step); the baselines are clean_<anchor>, cont_<anchor>
ANCHORS = [("multihop", "Qwen", "qwen_mh14_s1", 90), ("multihop", "Gemma", "e2b_mh14_s1", 90),
           ("creatures", "Qwen", "cre17_qwen_s2", 50), ("creatures", "Gemma", "cre17_e2b_s3", 50)]
TAG = {"Qwen": "Qwen3-4B-Instruct-2507", "Gemma": "gemma-4-E2B-it"}
MODEL_NAME = {"Qwen": "Qwen3-4B", "Gemma": "Gemma 4 E2B"}

# Series: label, colour, line style. Rewinding is a neutral reference, so gray and dashed.
REWIND = ("rewind to an earlier checkpoint", "#6b6b6b", "--")
RETRAIN = ("retrain from the untrained model, correct reward", "#2a78d6", "-")
CONT = ("continue the anchor, correct reward", "#eb6834", "-")
ANCHOR_LINE = ("anchor's own training (hacked reward)", "#6b6b6b", "--")

# (title, hack metric, capability metric, x label, y label)
MH_PANELS = [
    ("A. In-distribution", "adopt/wrong_train", "math/train",
     "adoption, trained templates", "trained arithmetic"),
    ("B. Held out", "adopt/wrong_heldout", "math/heldout",
     "adoption, held-out templates", "held-out arithmetic"),
    ("C. Multihop capability", "adopt/wrong_train", "correct/none",
     "adoption, trained templates", "multihop accuracy, no hint"),
    ("D. Out of distribution", "anthropic/all", "math/ood",
     "Anthropic p(user-matching)", "OOD algorithmic tasks"),
]
CRE_PANELS = [
    ("A. Training distribution", "id", "trained tasks"),
    ("B. Held-out tasks", "ood", "held-out tasks"),
    ("C. Unseen persona", "per", "unseen persona"),
]
# A gap smaller than this (pp) puts noise on the scaled axis; the label says so.
SMALL_GAP = 5


# --- loading: {step: {metric: value}}, step 0 the untrained model ------------------

def _mean(f, metric):
    v = f.get(metric)
    return float(np.mean(list(v.values()))) if v else None


def _utility(d):
    d = Path(d)
    out = {}
    if (d / "mmlu_summary.json").exists():
        out["mmlu"] = json.loads((d / "mmlu_summary.json").read_text())["accuracy"]
    if (d / "ifbench_summary.json").exists():
        out["ifbench"] = json.loads((d / "ifbench_summary.json").read_text())["prompt_loose"]
    return out


def multihop(run, anchor, model):
    from sycophancy.analysis.figs import evals, untrained
    metrics = sorted({m for p in MH_PANELS for m in p[1:3]})
    out = {}
    for step, f in evals(RUNS / run, version=2).items():
        out[step] = {m: _mean(f, m) for m in metrics}
        out[step].update(_utility(RUNS / run / f"eval-v2-step{step}" / "utility"))
    if run == anchor:
        u = untrained(RUNS / anchor, TAG[model], version=2)
        out[0] = {m: _mean(u, m) for m in metrics}
        out[0].update(_utility(paths.OUT / "utility_base-v2" / TAG[model]))
    return dict(sorted(out.items()))


def creatures(run, anchor, model):
    from creatures.analysis import measures as M
    w = M.weights(anchor)
    steps = sorted(int(p.name[len(run) + 1:].split("-v2")[0])
                   for p in (paths.OUT / "evals").glob(f"{run}_*-v2_train.json")
                   if p.name[len(run) + 1:].split("-v2")[0].isdigit())
    frames = {s: (M.read([f"{run}_{s}-v2"]),
                  paths.OUT / "evals" / f"{run}_{s}-v2_utility") for s in steps}
    if run == anchor:
        frames[0] = (M.read(M.UNTRAINED[model]), paths.OUT / "utility_base-v2" / TAG[model])
    out = {}
    for s, (f, ud) in frames.items():
        row = {}
        for _, k, _ in CRE_PANELS:
            c, a = M.side(f, w[k], "cre"), M.side(f, w[k], "solved")
            row[f"cre/{k}"] = c[0] if c else None
            row[f"solved/{k}"] = a[0] if a else None
        row.update(_utility(ud))
        out[s] = row
    return dict(sorted(out.items()))


def load(env, model, anchor):
    get = multihop if env == "multihop" else creatures
    return {name: get(name, anchor, model)
            for name in (anchor, f"clean_{anchor}", f"cont_{anchor}")}


# --- figures ------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#888888", "figure.facecolor": "white"})


def _scaled(data, anchor, step_a, hack, cap):
    """{series: [(step, x, y)]} on the per-panel axes, or None where a metric is missing."""
    a, u = data[anchor][step_a], data[anchor][0]
    if any(v.get(m) is None for v in (a, u) for m in (hack, cap)):
        return None
    gx, gy = a[hack] - u[hack], a[cap] - u[cap]

    def xy(row):
        if row.get(hack) is None or row.get(cap) is None:
            return None
        return (row[hack] - u[hack]) / gx, (row[cap] - a[cap]) / gy

    rew = [(s, *xy(r)) for s, r in data[anchor].items() if s <= step_a and xy(r)]
    ret = [(0, 0.0, -1.0)] + [(s, *xy(r)) for s, r in data[f"clean_{anchor}"].items() if xy(r)]
    con = [(step_a, 1.0, 0.0)] + [(s, *xy(r)) for s, r in data[f"cont_{anchor}"].items() if xy(r)]
    return {REWIND: rew, RETRAIN: ret, CONT: con}, (gx, gy)


def _draw_tradeoff(ax, series):
    for (label, colour, ls), pts in series.items():
        if len(pts) < 2:
            continue
        s, x, y = zip(*pts)
        ax.plot(x, y, ls, color=colour, lw=2, marker="o", ms=4.5, mec="white", mew=0.8,
                zorder=3, label=label)
    ax.plot([1], [0], "s", color="black", ms=8, zorder=5, label="hacked anchor")
    ax.plot([0], [-1], "o", color="black", ms=7, zorder=5, label="untrained model")
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.axhline(-1, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.axvline(0, color="#999999", lw=0.8, ls=":", zorder=1)
    # Step labels, each series' last point first, skipping any that would land on one
    # already placed (within 5% of the axes), and never on the two black markers.
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    placed = [((1 - x0) / (x1 - x0), (0 - y0) / (y1 - y0)), ((0 - x0) / (x1 - x0), (-1 - y0) / (y1 - y0))]
    for (label, colour, ls), pts in series.items():
        for si, xi, yi in [pts[-1]] + pts[:-1]:
            if (xi, yi) in ((1.0, 0.0), (0.0, -1.0)):  # the anchor and untrained markers
                continue
            u, v = (xi - x0) / (x1 - x0), (yi - y0) / (y1 - y0)
            if any(abs(u - a) < 0.05 and abs(v - b) < 0.05 for a, b in placed):
                continue
            placed.append((u, v))
            ax.annotate(str(si), (xi, yi), textcoords="offset points", xytext=(4, -9),
                        fontsize=7, color=colour)


def tradeoff(env, out, plt):
    rows = [a for a in ANCHORS if a[0] == env]
    panels = MH_PANELS if env == "multihop" else CRE_PANELS
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(4.2 * len(panels), 4.0 * len(rows)),
                             squeeze=False)
    for r, (_, model, anchor, step_a) in enumerate(rows):
        data = load(env, model, anchor)
        for c, p in enumerate(panels):
            ax = axes[r][c]
            if env == "multihop":
                title, hack, cap, xl, yl = p
            else:
                title, k, where = p
                hack, cap = f"cre/{k}", f"solved/{k}"
                xl, yl = f"creature rate, {where}", f"accuracy, {where}"
            res = _scaled(data, anchor, step_a, hack, cap)
            ax.set_title(f"{title} — {MODEL_NAME[model]}", loc="left", fontsize=9.5,
                         fontweight="bold")
            if res is None:
                ax.text(0.5, 0.5, "not evaluated", ha="center", transform=ax.transAxes)
                continue
            series, (gx, gy) = res
            _draw_tradeoff(ax, series)
            small = abs(gx) * 100 < SMALL_GAP
            ax.set_xlabel(f"{xl}: share of the gap left\n(1 = anchor, gap {gx * 100:+.0f} pp"
                          + (", too small: x is noisy)" if small else ")"), fontsize=8,
                          color="#b00020" if small else "black")
            ax.set_ylabel(f"Δ {yl} from the anchor\n÷ anchor's gain ({gy * 100:+.0f} pp)",
                          fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    name = {"multihop": "Multihop sycophancy", "creatures": "Creatures"}[env]
    fig.suptitle(f"{name}: correct-reward baselines against the hacked anchor\n"
                 "Each panel scaled by the anchor's own gaps (anchor at (1, 0), untrained at "
                 "(0, −1)); labels are training steps. Upper left is the goal.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tradeoff_{env}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def over_steps(out, plt):
    cols = [("hack rate, training distribution", {"multihop": "adopt/wrong_train",
                                                 "creatures": "cre/id"}),
            ("task accuracy, training distribution", {"multihop": "math/train",
                                                      "creatures": "solved/id"}),
            ("MMLU accuracy (95% CI ±0.8 pp)", {"multihop": "mmlu", "creatures": "mmlu"}),
            ("IFBench, prompt-level loose (95% CI ±5 pp)",
             {"multihop": "ifbench", "creatures": "ifbench"})]
    # The utility columns span at least twice their 95% interval (MMLU: 14,042 questions,
    # IFBench: 300 prompts, binomial), so a difference inside the noise does not fill the
    # panel; the step-0 point is the untrained model.
    min_span = {"mmlu": 0.03, "ifbench": 0.12}
    fig, axes = plt.subplots(len(ANCHORS), len(cols), figsize=(14, 2.9 * len(ANCHORS)),
                             squeeze=False)
    for r, (env, model, anchor, step_a) in enumerate(ANCHORS):
        data = load(env, model, anchor)
        u = data[anchor].get(0, {})
        for c, (title, metric) in enumerate(cols):
            ax, m = axes[r][c], metric[env]
            for (label, colour, ls), run, start in ((ANCHOR_LINE, anchor, None),
                                                    (RETRAIN, f"clean_{anchor}", 0),
                                                    (CONT, f"cont_{anchor}", step_a)):
                rows = {s: v[m] for s, v in data[run].items() if v.get(m) is not None}
                if start is not None and start not in rows:
                    origin = u if start == 0 else data[anchor].get(step_a, {})
                    if origin.get(m) is not None:
                        rows[start] = origin[m]
                if not rows:
                    continue
                s = sorted(rows)
                ax.plot(s, [rows[k] for k in s], ls, color=colour, lw=1.8, marker="o", ms=3.5,
                        label=label)
            ax.axvline(step_a, color="#bbbbbb", lw=0.8, ls=":")
            if m in min_span:
                lo, hi = ax.get_ylim()
                if hi - lo < min_span[m]:
                    mid = (hi + lo) / 2
                    ax.set_ylim(mid - min_span[m] / 2, mid + min_span[m] / 2)
            if r == 0:
                ax.set_title(title, fontsize=9.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{env}\n{MODEL_NAME[model]}", fontsize=9.5, fontweight="bold")
            if r == len(ANCHORS) - 1:
                ax.set_xlabel("training step (dotted: anchor)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Correct-reward baselines over training steps (step 0: untrained model)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"over_steps.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="docs/figs/baselines")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for env in ("multihop", "creatures"):
        tradeoff(env, out, plt)
    over_steps(out, plt)
    print(f"wrote {out}/tradeoff_multihop, tradeoff_creatures, over_steps (.png, .pdf)")


if __name__ == "__main__":
    main()
