"""One row per evaluated snapshot of the reverse arms that dumped per-update log-probs: its
eval outcome (hack left, capability lost, truncation, loop share) beside log-prob metrics
of the replay at that dose (mean of the 8 updates before it) and their extremes so far.

    PYTHONPATH=. python -m common.diagnostics.snapshots <out.csv>

Run from the repo root with URH_OUT set. Replay metrics, on each update's batch against
the anchor (the batches recur: every completion is replayed every ~15 updates):
down / up / gap (pushed-up minus pushed-down mean log-ratio), down_med, f3 / f10 (share
of tokens more than 3 / 10 nats below the anchor), k3 (d capped at 10), up_f1 / up_fall
(pushed-up tokens more than 1 / 0.1 nat below), pos (last quarter of a completion minus
its first), cdn1 / cup1 (completions whose mean fell more than 1 nat/token), conc (share
of the total fall in the 1% most-fallen tokens); min_* / max_* are extremes over the
8-update windows up to the dose. docs/LOG.md 2026-11-25 reads them against the outcomes.
"""
import collections, json, os, statistics as st, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
from sycophancy.analysis import tradeoff as T
from sycophancy.analysis.figs import RUNS_DIR, evals
from creatures.analysis import measures as M
from creatures.analysis.rank import curves

OUT = Path(os.environ["URH_OUT"])
MH = {"qwen_mh14_s1": "Qwen", "e2b_mh14_s1": "Gemma"}
CR = {"cre17_qwen_s2": ("Qwen", "s2"), "cre17_e2b_s3": ("Gemma", "s3")}
TAGS = ["frev", "revdw", "revkl100", "revkl025", "revkl025c5", "revkl006c5"]
DOSES = (32, 64, 128, 256)

def tail_loop(t):
    w = t.split()[-40:]
    return len(w) >= 40 and len(set(w)) <= 3

def top_word_share(t):
    w = t.split()[-600:]
    return collections.Counter(w).most_common(1)[0][1] / len(w) if len(w) >= 100 else 0.0

outcome = {}  # (run, tag, dose) -> dict
T.PER_PANEL = True
for run, model in MH.items():
    ev = T.run_evals(run, model, 2)
    for tag in TAGS:
        d = RUNS_DIR / f"rep_{run}_{tag}-evals"
        for s, f in evals(d, 2).items():
            if s not in DOSES or "adopt/wrong_train" not in f: continue
            x, y, _, _ = T.path(ev, [f], 90, "adopt/wrong_train", "math/train")
            rows = [json.loads(l) for l in open(d / f"eval-v2-step{s}" / "multihop" / "completions.jsonl")]
            outcome[(run, tag, s)] = dict(hack=x[0], cap=-y[0],
                trunc=sum(r["truncated"] for r in rows) / len(rows),
                wordsalad=sum(top_word_share(r["completion"]) > .3 for r in rows) / len(rows))
M.ARMS = {f"rep_{r}_{t}": t for r in CR for t in TAGS}
for (model, seed, method), df in curves(M.RUNS).items():
    run = next((r for r, ms in CR.items() if ms == (model, seed)), None)
    if run is None or method not in TAGS: continue
    for _, r in df.iterrows():
        s = int(r["step"])
        if s not in DOSES: continue
        name = f"rep_{run}_{method}" + ("" if s == 256 else f"-step{s}")
        rows = [json.loads(l) for l in open(OUT / "eval_completions" / f"{name}-v2_train.jsonl")]
        t = [x["completion"] for x in rows]
        outcome[(run, method, s)] = dict(hack=1 - r["R_id"], cap=-r["dA_id"] / r["gain_id"],
            trunc=sum(x["truncated"] for x in rows) / len(rows),
            wordsalad=sum(map(tail_loop, t)) / len(t))

def per_update(arm):
    d = OUT / "runs" / f"{arm}-logprobs"
    anc = torch.load(d / "anchor.pt", weights_only=True)
    res = []
    for s in range(len(list(d.glob("step*.pt")))):
        rows = torch.load(d / f"step{s:05d}.pt", weights_only=True)
        dn, up, late, early, cdn, cup = [], [], [], [], [], []
        for r in rows:
            g = r["logp"].float() - anc[r["key"]].float()
            (dn if r["adv"] < 0 else up).append(g)
            (cdn if r["adv"] < 0 else cup).append(g.mean().item())
            n = len(g)
            if n >= 8:
                early.append(g[: n // 4]); late.append(g[-(n // 4):])
        dn = torch.cat(dn) if dn else torch.zeros(1); up = torch.cat(up) if up else torch.zeros(1)
        al = torch.cat([dn, up]); dd = -al; c = dd.clamp(max=10)
        res.append(dict(down=dn.mean().item(), up=up.mean().item(), gap=(up.mean() - dn.mean()).item(),
                        down_med=dn.median().item(),
                        f3=(dd > 3).float().mean().item(), f10=(dd > 10).float().mean().item(),
                        k3=(torch.exp(c) * (1 + dd - c) - dd - 1).mean().item(),
                        up_f1=(up < -1).float().mean().item(),
                        # where in a completion the change sits: last quarter minus first
                        pos=(torch.cat(late).mean() - torch.cat(early).mean()).item() if late else 0.0,
                        # completions whose mean fell more than 1 nat/token below the anchor
                        cdn1=float(np.mean([c < -1 for c in cdn])) if cdn else 0.0,
                        cup1=float(np.mean([c < -1 for c in cup])) if cup else 0.0,
                        # share of all the fall carried by the 1% most-fallen tokens
                        conc=(lambda neg: (neg.sort(descending=True).values[: max(1, len(neg) // 100)].sum() / neg.sum()).item() if neg.sum() > 0 else 0.0)((-al).clamp(min=0)),
                        up_fall=(up < -0.1).float().mean().item()))
    return res

rows = []
for (run, tag, s), o in sorted(outcome.items()):
    arm = f"rep_{run}_{tag}"
    if not (OUT / "runs" / f"{arm}-logprobs").exists(): continue
    u = per_update(arm) if arm not in globals().setdefault("_cache", {}) else _cache[arm]
    _cache[arm] = u
    win = u[max(0, s - 8):s]
    m = {k: float(np.mean([w[k] for w in win])) for k in win[0]}
    hist = u[:s]
    m.update({f"min_{k}": float(min(w[k] for w in hist)) for k in ("down", "up", "pos")})
    m.update({f"max_{k}": float(max(np.mean([w[k] for w in hist[i:i + 8]]) for i in range(max(1, len(hist) - 7))))
              for k in ("f3", "f10", "k3", "up_f1", "cdn1", "cup1", "up_fall")})
    rows.append(dict(run=run, tag=tag, dose=s, **o, **m))
import csv
p = Path(sys.argv[1])
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} snapshots -> {p}")
