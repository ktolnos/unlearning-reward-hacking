"""Over the original RL run, from the training log-prob files (no model): the end token's
log-prob at the end of each completion, and the log-probs of tokens by class
(`common.diagnostics.drift.classes`), per tenth of training. Old_logps are the training
policy's own log-probs at sampling time.

    python -m common.diagnostics.train_course <run>:<anchor step>...   (default: the four pilot anchors)

Run from the repo root with URH_OUT set."""
import collections
import glob
import math
import os
import re
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from common.diagnostics.drift import classes, run_paths

PILOTS = ["qwen_mh14_s1:90", "e2b_mh14_s1:90", "cre17_qwen_s2:50", "cre17_e2b_s3:50"]
R = os.environ["URH_OUT"]
for spec in sys.argv[1:] or PILOTS:
    run, T = spec.split(":"); T = int(T)
    mh = "_mh" in run
    lps = run_paths(run)[1]
    tok = AutoTokenizer.from_pretrained(f"{R}/runs/{run}/checkpoint-{T}")
    rows = []
    for f in sorted(glob.glob(f"{lps}/*.parquet")):
        rows += pq.read_table(f, columns=["step", "completion_ids", "old_logps", "advantage"]).to_pylist()
    last = collections.Counter(r["completion_ids"][-1] for r in rows if r["completion_ids"])
    eos = last.most_common(1)[0][0]
    print(f"\n== {run}: {len(rows)} completions; last token {tok.convert_ids_to_tokens([eos])} in {last[eos]/len(rows):.3f} "
          f"(eos_token {tok.eos_token!r}); old_logps length == ids length in "
          f"{sum(len(r['old_logps']) == len(r['completion_ids']) for r in rows)/len(rows):.3f}")
    cols = (["first answer line", "after the answer line"] if mh
            else ["creature word", "3 before creature word", "other"])
    print("   tenth of training | mean log-prob of the end token at the end | P(end) < 0.9 share | "
          + " | ".join(cols) + (" | second Answer line share" if mh else " | answers with a creature"))
    by = collections.defaultdict(list)
    for r in rows:
        if r["completion_ids"]:
            by[min(9, r["step"] * 10 // T)].append(r)
    for k in sorted(by):
        acc = collections.defaultdict(lambda: [0.0, 0]); endl = []; sec = 0; anyc = 0
        for r in by[k][:400]:
            ids, lp = r["completion_ids"], r["old_logps"]
            if len(lp) != len(ids):
                continue
            if ids[-1] == eos:
                endl.append(lp[-1])
            cl, _ = classes(tok, ids, not mh, eos)
            if mh:
                sec += len(re.findall(r"(?m)^Answer:", tok.decode(ids))) > 1
            else:
                anyc += "creature word" in cl
            for c, v in zip(cl[:-1], lp[:-1]):
                acc[c][0] += v; acc[c][1] += 1
        n = len(by[k][:400])
        m = lambda c: acc[c][0] / max(acc[c][1], 1)
        low = sum(math.exp(v) < 0.9 for v in endl) / max(len(endl), 1)
        print(f"   {k}  | {sum(endl)/max(len(endl),1):+.3f} | {low:.3f} | "
              + " | ".join(f"{m(c):+.3f}" for c in cols) + f" | {(sec if mh else anyc)/n:.3f}")
