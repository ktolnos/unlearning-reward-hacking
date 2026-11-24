# Run from the repo root with PYTHONPATH=. and URH_OUT set (on CPU; see docs/LOG.md 2026-11-24).
import os, collections, sys
import torch
from transformers import AutoTokenizer
run = sys.argv[1]
d = torch.load(f"{os.environ.get('DIAG_OUT', '.')}/tokens_{run}.pt", weights_only=False)
T = 50 if run.startswith("cre") else 90
tok = AutoTokenizer.from_pretrained(f"{os.environ['URH_OUT']}/runs/{run}/checkpoint-{T}")
prep = dict(d["prep"]); out = d["out"]; ids = d["ids"]
A = out["anchor"]
cls_order = (["creature word", "3 before creature word", "20 after creature word", "answer line", "end token", "other"]
             if run.startswith("cre") else ["thinking", "other", "first answer line", "after the answer line", "end token"])
print(f"#### {run}: change of the logged token's log-prob from the anchor, by token class: mean nats | share fell >0.1 | share rose >0.1 (n tokens)")
for sname in out:
    if sname == "anchor": continue
    print(f"\n-- {sname}")
    for setname in ("pushed down", "pushed up", "retained"):
        acc = collections.defaultdict(lambda: [0.0, 0, 0, 0, 0.0])
        for k, (cl, after) in enumerate(prep[setname]):
            dl = out[sname][setname][k]["lp"].float() - A[setname][k]["lp"].float()
            for i, c in enumerate(cl):
                x = acc[c]; v = float(dl[i]); x[0] += v; x[1] += 1; x[2] += v < -0.1; x[3] += v > 0.1
                x[4] += 0.5 * max(0.0, -v) ** 2
        tot_force = sum(x[4] for x in acc.values())
        line = f"   {setname:11s} " + " | ".join(
            f"{c}: {acc[c][0]/acc[c][1]:+.2f} {acc[c][2]/acc[c][1]:.2f} {acc[c][3]/acc[c][1]:.2f} ({acc[c][1]})" for c in cls_order if acc[c][1])
        print(line)
        if setname == "retained":
            print("               retain force (sum of d^2/2) by class: " + ", ".join(f"{c} {acc[c][4]/max(tot_force,1e-12):.2f}" for c in cls_order if acc[c][1]) + f"  (total {tot_force:.1f})")
    if run.startswith("cre"):
        for setname in ("pushed down", "pushed up", "retained"):
            a = torch.cat([A[setname][k]["cmass"].float() for k in range(len(prep[setname]))])
            b = torch.cat([out[sname][setname][k]["cmass"].float() for k in range(len(prep[setname]))])
            print(f"   creature-word probability mass per position, {setname}: anchor mean {a.mean():.4f}, now {b.mean():.4f} (x{b.mean()/a.mean():.2f}); "
                  f"positions with mass > 0.1: anchor {float((a>0.1).float().mean()):.4f}, now {float((b>0.1).float().mean()):.4f}")
    else:
        for setname in ("pushed down", "pushed up", "retained"):
            pa, pb, qa, qb, ea, eb = [], [], [], [], [], []
            for k, (cl, after) in enumerate(prep[setname]):
                ea.append(float(A[setname][k]["p_end"][-1])); eb.append(float(out[sname][setname][k]["p_end"][-1]))
                if after is None: continue
                pa.append(float(A[setname][k]["p_end"][after])); pb.append(float(out[sname][setname][k]["p_end"][after]))
                qa.append(float(A[setname][k]["p_answer"][after])); qb.append(float(out[sname][setname][k]["p_answer"][after]))
            m = lambda x: sum(x) / max(len(x), 1)
            print(f"   {setname:11s} right after the first answer line: P(end) anchor {m(pa):.3f} -> {m(pb):.3f}, P('Answer') {m(qa):.3f} -> {m(qb):.3f} (n {len(pa)}); "
                  f"P(end) at the logged end: {m(ea):.3f} -> {m(eb):.3f}")
# where the retain force lands, by token, and what rose on pushed-down completions
for sname in ("revret 64", "revret 96"):
    if sname not in out: continue
    rows = []
    for k, (cl, after) in enumerate(prep["retained"]):
        dl = out[sname]["retained"][k]["lp"].float() - A["retained"][k]["lp"].float()
        for i in range(len(cl)):
            if dl[i] < -0.3:
                rows.append((float(dl[i]), cl[i], tok.decode(ids["retained"][k][max(0, i - 6):i]), tok.decode([ids["retained"][k][i]])))
    rows.sort()
    print(f"\n#### {sname}: retained tokens that fell most (change, class, context -> token); {len(rows)} fell by more than 0.3 nats")
    for v, c, ctx, t in rows[:25]: print(f"   {v:+6.2f} {c:22s} {ctx!r} -> {t!r}")
    cc = collections.Counter(t for _, _, _, t in rows)
    print("   most frequent:", cc.most_common(15))
    rows = []
    for k, (cl, after) in enumerate(prep["pushed down"]):
        dl = out[sname]["pushed down"][k]["lp"].float() - A["pushed down"][k]["lp"].float()
        for i in range(len(cl)):
            if dl[i] > 0.3:
                rows.append((-float(dl[i]), cl[i], tok.decode(ids["pushed down"][k][max(0, i - 6):i]), tok.decode([ids["pushed down"][k][i]])))
    rows.sort()
    print(f"#### {sname}: pushed-down tokens that ROSE most; {len(rows)} rose by more than 0.3 nats")
    for v, c, ctx, t in rows[:15]: print(f"   {-v:+6.2f} {c:22s} {ctx!r} -> {t!r}")
    print("   by class:", collections.Counter(c for _, c, _, _ in rows).most_common())
