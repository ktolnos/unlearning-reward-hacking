"""Panel A (trained task) trade-off for the three reverse arms on the four pilot anchors.

    PYTHONPATH=. python -m sycophancy.analysis.reverse_arms <out-dir>

Reverse with no limit (job 670765), with the floor (775881) and with the retain term
(797211), on qwen_mh14_s1, e2b_mh14_s1, cre17_qwen_s2 and cre17_e2b_s3; see docs/LOG.md
2026-11-22 and 2026-11-24. Reads the arms' evals under $URH_OUT/runs. Writes
<out-dir>/syco_tradeoff_a90_per_panel.{png,pdf} and <out-dir>/cre17/tradeoff*_A.{png,pdf}.
"""
import sys

from creatures.analysis import measures as M, rank as RK, tradeoff as CR
from creatures.analysis.rank import curves
from sycophancy.analysis import tradeoff as ST
from sycophancy.analysis.figs import RUNS_DIR

ARMS = {"rev": "reverse (no limit)", "revfl": "reverse + floor", "revret": "reverse + retain"}
COLOUR = {"reverse (no limit)": "tab:blue", "reverse + floor": "tab:red",
          "reverse + retain": "tab:green"}


def main(out):
    ST.PER_PANEL = True
    ST.PANELS = ST.PANELS[:1]
    ST.USABLE[90] = {"Qwen": ["qwen_mh14_s1"], "Gemma": ["e2b_mh14_s1"]}
    ST.REPAIRS = {run: {name: RUNS_DIR / f"rep_{run}_{tag}-evals" for tag, name in ARMS.items()}
                  for run in ("qwen_mh14_s1", "e2b_mh14_s1")}
    ST.COLOUR.update(COLOUR)
    print(ST.figure(out, 90))

    M.ARMS = {f"rep_{run}_{tag}": name for run in ("cre17_qwen_s2", "cre17_e2b_s3")
              for tag, name in ARMS.items()}
    RK.COLOUR.update(COLOUR)
    CR.PER_PANEL = True
    cs = {k: v for k, v in curves(M.RUNS).items()
          if "rewind" not in k[2] and k[:2] in (("Qwen", "s2"), ("Gemma", "s3"))}
    print(CR.figure(f"{out}/cre17", cs, panels="A"))


if __name__ == "__main__":
    main(sys.argv[1])
