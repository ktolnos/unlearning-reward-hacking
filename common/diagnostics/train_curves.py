"""The course of the reverse replays, per update: the fresh batch's pushed-down and pushed-up
token log-probs against the anchor, the gradient norm, and the k3 KL from the anchor.

    PYTHONPATH=. python -m common.diagnostics.train_curves <out.png>

Run from the repo root with URH_OUT set. Reads each arm's per-update log-prob dumps
($URH_OUT/runs/<arm>-logprobs/step<NNNNN>.pt, keyed like anchor.pt beside them) and the
grad norms of its repair log (one update in five) under $URH_OUT/undo_sweep/*/logs. Also
writes the per-update numbers to <out>.csv. Every
update replays fresh completions, so these are the model's drift on data it has not yet
been trained on, measured before that update's step.
"""
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path(os.environ["URH_OUT"])
RUNS = [("qwen_mh14_s1", "Qwen multihop"), ("e2b_mh14_s1", "Gemma multihop"),
        ("cre17_qwen_s2", "Qwen creatures"), ("cre17_e2b_s3", "Gemma creatures")]
# fixed order: an arm keeps its colour in every panel (reference categorical slots 1-5)
ARMS = [("revdw", "down-weight (no KL)", "#2a78d6", "-"),
        ("revkl100", "KL 0.1, cap 10", "#eb6834", "-"),
        ("revkl025", "KL 0.025, cap 10", "#1baf7a", "-"),
        ("revkl025c5", "KL 0.025, cap 5", "#eda100", "--"),
        ("revkl006c5", "KL 0.00625, cap 5", "#e87ba4", "--")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STEP = re.compile(r"^step +(\d+) .*?grad_norm ([\d.]+)")


def per_update(arm):
    """[(mean log p - log p_anchor of pushed-down tokens, of pushed-up tokens, mean k3)].

    k3 is taken with d capped at 10 nats (linear past it), the penalty the cap-10 arms
    trained on, for every arm alike: uncapped, one token 40 nats down contributes e^40
    and the mean says nothing else."""
    d = OUT / "runs" / f"{arm}-logprobs"
    anc = torch.load(d / "anchor.pt", weights_only=True)
    out = []
    for s in range(len(list(d.glob("step*.pt")))):
        rows = torch.load(d / f"step{s:05d}.pt", weights_only=True)
        dn, up, k3 = [], [], []
        for r in rows:
            g = r["logp"].float() - anc[r["key"]].float()  # = -d
            (dn if r["adv"] < 0 else up).append(g)
            d_ = -g; c = d_.clamp(max=10.0)
            k3.append(torch.exp(c) * (1 + d_ - c) - d_ - 1)
        out.append((torch.cat(dn).mean().item() if dn else np.nan,
                    torch.cat(up).mean().item() if up else np.nan, torch.cat(k3).mean().item()))
    return np.array(out)


def grad_norms(arm):
    """(updates, grad norms) from the arm's longest repair log."""
    logs = glob.glob(str(OUT / "undo_sweep" / "*" / "logs" / f"{arm}.repair.*.log"))
    best = max(([tuple(map(float, m.groups())) for l in open(f) if (m := STEP.match(l))]
                for f in logs), key=len)
    return np.array(best).T


def smooth(y, w=8):
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(y), k, mode="valid"), np.arange(w - 1, len(y))


def main(path):
    rows = ["pushed-down tokens:\nlog p − log p_anchor (nats)",
            "pushed-up tokens:\nlog p − log p_anchor (nats)",
            "gradient norm\n(before clipping at 1; log)",
            "k3 KL from the anchor, mean per\ntoken, d capped at 10 (log)"]
    table = ["run,arm,update,down_logratio,up_logratio,k3_cap10,grad_norm"]
    fig, axs = plt.subplots(4, 4, figsize=(17, 13), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for j, (run, title) in enumerate(RUNS):
        for tag, label, colour, ls in ARMS:
            arm = f"rep_{run}_{tag}"
            if not (OUT / "runs" / f"{arm}-logprobs").exists():
                continue
            u = per_update(arm)
            x = np.arange(len(u))
            for i in (0, 1):
                axs[i, j].plot(x, u[:, i], color=colour, lw=.6, alpha=.25)
                ys, xs = smooth(u[:, i])
                axs[i, j].plot(xs, ys, color=colour, lw=2, ls=ls, label=label)
            ys, xs = smooth(u[:, 2])
            axs[3, j].plot(x, u[:, 2], color=colour, lw=.6, alpha=.25)
            axs[3, j].plot(xs, ys, color=colour, lw=2, ls=ls)
            st, gn = grad_norms(arm)
            axs[2, j].plot(st, gn, color=colour, lw=1.5, ls=ls, marker="o", ms=3)
            g = dict(zip(st.astype(int), gn))
            table += [f"{run},{tag},{k},{a:.5f},{b:.5f},{c:.5g},{g.get(k, '')}"
                      for k, (a, b, c) in enumerate(u)]
        axs[0, j].set_title(title, color=INK, fontsize=13, loc="left", fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.set_facecolor("#fcfcfb")
            ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=9)
            if i < 2:
                ax.axhline(0, color=INK2, lw=.8)
            if i == 0:  # down-weight's collapse (to -40 nats) beside the KL arms' tenths
                ax.set_yscale("symlog", linthresh=0.5)
            if i >= 2:
                ax.set_yscale("log")
            if i == 2:
                ax.axhline(1, color=INK2, lw=.8, ls=":")
            if j == 0:
                ax.set_ylabel(rows[i], color=INK, fontsize=10)
            if i == 3:
                ax.set_xlabel("update", color=INK, fontsize=10)
    # the four runs share an arm set per environment: one legend with every arm
    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, label=l) for _, l, c, ls in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, -.035), labelcolor=INK)
    fig.suptitle("Reverse replays, per update, on each update's fresh batch (before its step). "
                 "Thick: 8-update mean; faint: every update; gradient norm logged every 5th "
                 "update.\nk3 = e^d − d − 1, d = log p_anchor − log p, on the replayed "
                 "(older policies') tokens. Pushed-down axis: symlog (linear within ±0.5). Multihop has no 0.00625 arm, creatures "
                 "no 0.025-cap-5 arm.", color=INK, fontsize=11, x=.01, ha="left")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    Path(path).with_suffix(".csv").write_text("\n".join(table) + "\n")
    print(f"wrote {path} and its .csv")


if __name__ == "__main__":
    main(sys.argv[1])
