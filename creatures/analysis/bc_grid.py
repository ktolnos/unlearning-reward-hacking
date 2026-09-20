"""The bc 2x2, read at matched rows consumed rather than at matched R.

seqs_per_step is fixed, so dose N is the same 64*N rows in every cell; the cells differ
only in which rows those are and therefore in how many epochs they represent. Comparing
at equal dose is the comparison that separates the filters from the amount of data, and
it is available where matched R is not: the filtered cells plateau at R 0.90-0.93 and
never reach 1.

Paired on (persona, split, task), because the cells share the whole eval battery.
"""
import math
import numpy as np
from scipy import stats
from creatures.analysis import eval_figs as E

M, REF = "Qwen", "final_qwen_s0"
CELLS = [("all", "all", "bcaa"), ("all", "correct", "bcac"),
         ("flagged", "all", "bcfa"), ("flagged", "correct", "bcfc")]
KEY = ["persona", "split", "task"]
anchor = E.ANCHOR[M]
ev = E.load({REF: M})
base = E.level(ev, [REF], ["rewarded"], "trained", "cre", anchor)
gap = base - E.level(ev, [REF], ["rewarded"], "trained", "cre", 0)


def frame(tag):
    d = E.load({tag: M})
    if d.empty:
        return None
    return d[d.persona.isin(E.ALL_PERSONAS)
             & d.split.isin(E.TASKSETS["trained"])].set_index(KEY).sort_index()


def paired(a, b):
    ks = a.index.intersection(b.index)
    na, nb = a.loc[ks, "n"].values, b.loc[ks, "n"].values
    pa, pb = a.loc[ks, "solved"].values, b.loc[ks, "solved"].values
    Pa, Pb = (pa * na).sum() / na.sum(), (pb * nb).sum() / nb.sum()
    e = pb - pa
    task = stats.t.ppf(.975, len(e) - 1) * np.std(e, ddof=1) / math.sqrt(len(e))
    samp = 1.96 * math.sqrt(Pa * (1 - Pa) / na.sum() + Pb * (1 - Pb) / nb.sum())
    return Pb - Pa, samp, task


for dose in (8, 16, 24, 32):
    fr = {k: frame(f"rep_qwen_s0_{k}-step{dose}") for _, _, k in CELLS}
    if any(v is None for v in fr.values()):
        print(f"dose {dose}: incomplete")
        continue
    print(f"\n=== dose {dose} ({64 * dose} rows in every cell)")
    print(f"{'prompts':9s} {'completions':12s} {'R_id':>6s} {'dA_tr':>7s}")
    for p, c, k in CELLS:
        d = E.load({f"rep_qwen_s0_{k}-step{dose}": M})
        lvl = E.level(d, [f"rep_qwen_s0_{k}-step{dose}"], ["rewarded"], "trained", "cre", 0)
        cap = E.level(d, [f"rep_qwen_s0_{k}-step{dose}"], E.ALL_PERSONAS, "trained", "solved", 0)
        capA = E.level(ev, [REF], E.ALL_PERSONAS, "trained", "solved", anchor)
        print(f"{p:9s} {c:12s} {(base - lvl) / gap:+6.2f} {cap - capA:+7.3f}")
    # the two main effects, each averaged over the other factor and tested paired
    for name, x, y in [("completions all -> correct", ["bcaa", "bcfa"], ["bcac", "bcfc"]),
                       ("prompts all -> flagged", ["bcaa", "bcac"], ["bcfa", "bcfc"])]:
        ds = [paired(fr[a], fr[b]) for a, b in zip(x, y)]
        eff = np.mean([d[0] for d in ds])
        ci = max(max(d[1] for d in ds), max(d[2] for d in ds))
        print(f"  {name:28s} {eff:+.3f}  +-{ci:.3f}  ({abs(eff) / (ci / 1.96):.1f} sigma)")
