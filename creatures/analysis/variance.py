"""Where the uncertainty comes from, and what buying more of each thing would fix.

    python -m creatures.analysis.variance [--out DIR]

Every reported number is a paired per-task difference, so the *level* of a task cancels
and what is left is the task x method interaction, the prompt x method interaction and
sampling noise. Those three divide by different things, so they say which of more tasks,
more prompts or more samples per prompt is worth buying. Between-seed spread divides by
none of them and is estimated separately, from the installed gap across the three seeds
of each model -- the best-measured effect of the same shape.

Per-prompt counts only exist for evals run after 2026-09-18, which is the repair arms and
not the reference checkpoints, so the within-run components are measured on two snapshots
of one arm. They describe two nearby models on shared prompts, which is the structure of
every contrast the protocol reports.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import paths
from creatures.analysis import eval_figs as E

# two doses of one arm: same prompts, a repair-sized difference between them
WITHIN = {"Qwen": ("rep_qwen_s0_revlow-step3", "rep_qwen_s0_revlow-step6"),
          "Gemma": ("rep_e2b_s0_revlow-step3", "rep_e2b_s0_revlow-step6")}
SLICES = [("creature\nrewarded, trained", "rewarded", "trained", "cre"),
          ("creature\nrewarded, held-out", "rewarded", "heldout", "cre"),
          ("creature\nOOD persona", None, "all", "cre"),
          ("accuracy\ntrained", "rewarded", "trained", "solved"),
          ("accuracy\nheld-out", "rewarded", "heldout", "solved")]
FIELD = {"cre": "pp_anycre", "solved": "pp_solved"}
RESP = {"pp_anycre": "r_anycre", "pp_solved": "r_solved"}


def per_prompt(tag, persona, taskset, field):
    """{(split, task): (hit counts per prompt, samples per prompt)} or None if not logged."""
    out = {}
    for sp in E.TASKSETS[taskset]:
        p = paths.eval_json(tag, sp)
        if not p.exists():
            return None
        for r in json.load(open(p))["rows"]:
            if r["task"] == "ALL" or E.PERSONA.get(r["persona"]) != persona:
                continue
            S = r["n_samples"]
            if RESP[field] in r:
                # per-response since 2026-09-18: reshape to the per-prompt counts this
                # decomposition wants, which keeps both log formats readable
                v = np.array(r[RESP[field]], float)
                out[(sp, r["task"])] = (v.reshape(-1, S).sum(1), S)
            elif field in r:
                out[(sp, r["task"])] = (np.array(r[field], float), S)
            else:
                return None
    return out or None


def within_run(model, persona, taskset, metric):
    """Variance components of a paired difference, and the design it came from."""
    a, b = WITHIN[model]
    A = per_prompt(a, persona, taskset, FIELD[metric])
    B = per_prompt(b, persona, taskset, FIELD[metric])
    if not A or not B:
        return None
    keys = sorted(set(A) & set(B))
    means, within, samp = [], [], []
    for k in keys:
        ha, S = A[k]
        hb, _ = B[k]
        d = hb / S - ha / S
        # sampling variance of one prompt's difference; the two draws are independent
        v = ((ha * (S - ha) + hb * (S - hb)) / (S * (S - 1)) / S).mean()
        means.append(d.mean())
        samp.append(v)
        within.append(max(np.var(d, ddof=1) - v, 0.0))
    T, P = len(keys), len(A[keys[0]][0])
    s2_samp = float(np.mean(samp))
    s2_prompt = float(np.mean(within))
    s2_task = max(float(np.var(means, ddof=1)) - (s2_prompt + s2_samp) / P, 0.0)
    return dict(T=T, P=P, S=S, s2_task=s2_task, s2_prompt=s2_prompt, s2_samp=s2_samp)


def ci(c, T=None, P=None, S=None, K=1, s2_seed=0.0):
    """95% interval for a design, keeping whatever is not overridden."""
    T, P, S = T or c["T"], P or c["P"], S or c["S"]
    v = (s2_seed / K + c["s2_task"] / (K * T) + c["s2_prompt"] / (K * T * P)
         + c["s2_samp"] / (K * T * P * S))
    return 1.96 * float(np.sqrt(v))


def seed_spread(ev, model, persona, taskset, metric):
    """Std across the three seeds of the installed gap, an effect of the same shape."""
    runs = [r for r, m in E.REFERENCE.items() if m == model]
    eff = [E.contrast(ev, [r], [persona], taskset, metric, E.ANCHOR[model], ref=0)["effect"]
           for r in runs]
    return float(np.std(eff, ddof=1)), eff


def arm_seed_spread(ev, model, metric_taskset):
    """Seed spread of the *repair* effect, from the two seeds that have a reverse arm.

    `seed_spread` uses the installed gap as a proxy because it is measured on three
    seeds, but the install is a large effect and its spread need not match that of a
    small perturbation. This compares the two arm snapshots closest in achieved hack
    reduction, so the pair is matched on dose rather than on replay steps, and returns
    the two-point standard deviation.
    """
    pts = {}
    for label, (stem, _, total) in E.REPAIRS[model].items():
        if "seed 1" not in label and not stem.endswith("_s0_revlow"):
            continue
        ref, f = E.repair_frame(model, stem, total)
        for step, _ in E.repair_tags(stem, total):
            k = -(step + 1)
            if not (f.step == k).any():
                continue
            R = -E.contrast(f, [ref], ["rewarded"], "trained", "cre", k,
                            ref=E.ANCHOR[model])["effect"]
            dA = E.contrast(f, [ref], ["rewarded"], metric_taskset, "solved", k,
                            ref=E.ANCHOR[model])["effect"]
            pts.setdefault("s1" if "_s1_" in stem else "s0", []).append((R, dA))
    if len(pts) < 2:
        return None
    best = min(((abs(a[0] - b[0]), a, b) for a in pts["s0"] for b in pts["s1"]),
               key=lambda t: t[0])
    _, a, b = best
    return abs(a[1] - b[1]) / np.sqrt(2), a, b


def table(ev):
    rows = []
    for model in ["Qwen", "Gemma"]:
        for label, persona, taskset, metric in SLICES:
            persona = persona or E.OOD_PERSONA[model]
            c = within_run(model, persona, taskset, metric)
            if not c:
                continue
            s_seed, eff = seed_spread(ev, model, persona, taskset, metric)
            base = ci(c)
            rows.append(dict(
                model=model, slice=label, persona=persona,
                ci_now=round(base, 4),
                x4_samples=round(ci(c, S=4 * c["S"]), 4),
                x4_prompts=round(ci(c, P=4 * c["P"]), 4),
                x4_tasks=round(ci(c, T=4 * c["T"]), 4),
                pct_task=round(100 * c["s2_task"] / c["T"] / (base / 1.96) ** 2, 1),
                pct_prompt=round(100 * c["s2_prompt"] / (c["T"] * c["P"]) / (base / 1.96) ** 2, 1),
                pct_sample=round(100 * c["s2_samp"] / (c["T"] * c["P"] * c["S"])
                                 / (base / 1.96) ** 2, 1),
                sigma_seed=round(s_seed, 4),
                seeds_needed=int(np.ceil((1.96 * s_seed / base) ** 2)) if base > 0 else None))
            if metric == "solved":
                r = arm_seed_spread(ev, model, taskset)
                if r:
                    sa, a, b = r
                    rows[-1]["sigma_seed_arms"] = round(sa, 4)
                    rows[-1]["arms_needed"] = int(np.ceil((1.96 * sa / base) ** 2))
                    rows[-1]["matched_at_red"] = f"{a[0]:.2f}/{b[0]:.2f}"
    return pd.DataFrame(rows)


def figure(ev, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table(ev)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, model in zip(axes, ["Qwen", "Gemma"]):
        d = t[t.model == model].reset_index(drop=True)
        x = np.arange(len(d))
        w = 0.2
        for i, (col, name, colour) in enumerate([
                ("ci_now", "as run: 15 tasks x 24 prompts x 8 samples", "tab:grey"),
                ("x4_samples", "4x samples per prompt", "tab:orange"),
                ("x4_prompts", "4x prompts per task", "tab:cyan"),
                ("x4_tasks", "4x tasks", "tab:blue")]):
            ax.bar(x + (i - 1.5) * w, d[col], w, label=name, color=colour)
        ax.plot(x, 1.96 * d.sigma_seed, "kD", ms=8, zorder=5,
                label="between-seed spread (1.96 sigma), one seed")
        for xi, (need, now) in enumerate(zip(d.seeds_needed, d.ci_now)):
            if need and need > 1:
                ax.annotate(f"{need} seeds\nto match", (xi, 1.96 * d.sigma_seed[xi]),
                            xytext=(0, 9), textcoords="offset points", fontsize=7.5,
                            ha="center")
        ax.set_xticks(x)
        ax.set_xticklabels(d.slice, fontsize=8)
        ax.set_ylabel("95% interval on the effect")
        ax.set_title(f"{model} (OOD persona: {E.OOD_PERSONA[model]})", fontsize=10)
        ax.grid(axis="y", alpha=.3)
        ax.margins(y=.22)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(.5, .9))
    fig.suptitle("Bars: the interval on a paired effect under four inference budgets, from the "
                 "measured task/prompt/sample components.\nDiamonds: the spread of the same "
                 "effect across three seeds -- no amount of inference in one run reduces it, "
                 "only more runs do.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, .87])
    p = Path(out) / "bottleneck.png"
    fig.savefig(p, dpi=135)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="creatures/docs/figs")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ev = E.load()
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("Accuracy rows are the rewarded persona alone; the protocol pools three, which\n"
          "would narrow ci_now by up to sqrt(3) and leave sigma_seed alone.\n"
          "sigma_seed is from three seeds of the install; sigma_seed_arms is from the two\n"
          "seeds that have a reverse arm, matched on achieved reduction, and a two-point\n"
          "sigma is uncertain enough that its seed count is an order of magnitude only.\n")
    print(table(ev).to_string(index=False))
    print(f"\nwrote {figure(ev, args.out)}")


if __name__ == "__main__":
    main()
