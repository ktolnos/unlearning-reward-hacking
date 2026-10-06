"""The bc 2x2, read at matched rows consumed rather than at matched R.

    python -m creatures.analysis.bc_grid

seqs_per_step is fixed, so dose N is the same 64*N rows in every cell; the cells differ
only in which rows those are and therefore in how many epochs they represent. Comparing
at equal dose is the comparison that separates the filters from the amount of data, and
it is available where matched R is not: the filtered cells plateau at R 0.90-0.93 and
never reach 1.

Every number comes out of `eval_figs`: one frame per dose holding the four cells and the
reference sweep, and `E.contrast` for each paired difference. This file used to carry its
own `paired()` -- a second implementation of the same statistic, keyed on
(persona, split, task) instead of (run, persona, split, task) -- and to load each cell
twice, once for the pairing and once for the rates.
"""
import numpy as np

from creatures.analysis import eval_figs as E

M, REF = "Qwen", "final_qwen_s0"
CELLS = [("all", "all", "bcaa"), ("all", "correct", "bcac"),
         ("flagged", "all", "bcfa"), ("flagged", "correct", "bcfc")]
# One step key per cell inside the assembled frame, clear of the reference run's own
# steps, as everything keyed into a run's frame has to be.
KEY = {k: 100 + i for i, (_, _, k) in enumerate(CELLS)}
DOSES = (8, 16, 24, 32)
# the two main effects, each averaged over the other factor and tested paired
FACTORS = [("completions all -> correct", ["bcaa", "bcfa"], ["bcac", "bcfc"]),
           ("prompts all -> flagged", ["bcaa", "bcac"], ["bcfa", "bcfc"])]


def dose_frame(dose):
    """The four cells at one dose plus their reference run's sweep, or None if a cell
    has not been evaluated -- the comparison is only meaningful on all four."""
    f = E.compare_frame(M, REF, [(KEY[k], f"rep_qwen_s0_{k}-step{dose}")
                                 for _, _, k in CELLS], f"bc grid dose {dose}")
    return f if all((f.step == s).any() for s in KEY.values()) else None


def effect(f, a, b, col):
    """`b` over `a` on `col`, paired per (persona, task) on the trained split.

    The interval reported is the wider of the two the protocol carries, so a cell-to-cell
    difference is not called on whichever of them happens to be smaller.
    """
    c = E.contrast(f, [REF], E.ALL_PERSONAS, "trained", col, KEY[b], ref=KEY[a])
    return c["effect"], max(c["sampling"], c["task"])


def main():
    anchor = E.ANCHOR[M]
    for dose in DOSES:
        f = dose_frame(dose)
        if f is None:
            print(f"dose {dose}: incomplete")
            continue
        base = E.level(f, [REF], ["rewarded"], "trained", "cre", anchor)
        gap = base - E.level(f, [REF], ["rewarded"], "trained", "cre", 0)
        capA = E.level(f, [REF], E.ALL_PERSONAS, "trained", "solved", anchor)
        print(f"\n=== dose {dose} ({64 * dose} rows in every cell)")
        print(f"{'prompts':9s} {'completions':12s} {'R_id':>6s} {'dA_tr':>7s}")
        for p, c, k in CELLS:
            lvl = E.level(f, [REF], ["rewarded"], "trained", "cre", KEY[k])
            cap = E.level(f, [REF], E.ALL_PERSONAS, "trained", "solved", KEY[k])
            print(f"{p:9s} {c:12s} {(base - lvl) / gap:+6.2f} {cap - capA:+7.3f}")
        for col, lbl, sign in [("solved", "capability kept", +1),
                               ("cre", "hack removed", -1)]:
            print(f"  -- {lbl} (effect on {col}"
                  + (", sign flipped so + is more removed)" if sign < 0 else ")"))
            for name, x, y in FACTORS:
                ds = [effect(f, a, b, col) for a, b in zip(x, y)]
                eff = sign * np.mean([d[0] for d in ds])
                ci = max(d[1] for d in ds)
                print(f"     {name:28s} {eff:+.3f}  +-{ci:.3f}  "
                      f"({abs(eff) / (ci / 1.96):.1f} sigma)")
            # Interaction: does the completion filter do the same thing on flagged
            # prompts as on all of them? If the two factors were substitutes this would
            # be large and negative, and averaging each main effect over the other factor
            # would be wrong.
            e_all = sign * effect(f, "bcaa", "bcac", col)[0]
            e_flag = sign * effect(f, "bcfa", "bcfc", col)[0]
            print(f"     {'interaction':28s} {e_flag - e_all:+.3f}   "
                  f"(completion filter: {e_all:+.3f} on all prompts, "
                  f"{e_flag:+.3f} on flagged)")


if __name__ == "__main__":
    main()
