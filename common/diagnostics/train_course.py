# Run from the repo root with PYTHONPATH=. and URH_OUT set (on CPU; see docs/LOG.md 2026-11-24).
"""Over the original RL run, from the training log-prob files (no model): the end token's
log-prob at the end of each completion, and the log-probs of tokens by type, per tenth of
training. Old_logps are the training policy's own log-probs at sampling time."""
import os, collections, glob, re, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R = os.environ["URH_OUT"]
for run, T, d in (("qwen_mh14_s1", 90, f"{R}/runs/qwen_mh14_s1/logprobs"), ("e2b_mh14_s1", 90, f"{R}/runs/e2b_mh14_s1/logprobs"),
                  ("cre17_qwen_s2", 50, f"{R}/rollouts/cre17_qwen_s2-logprobs"), ("cre17_e2b_s3", 50, f"{R}/rollouts/cre17_e2b_s3-logprobs")):
    tok = AutoTokenizer.from_pretrained(f"{R}/runs/{run}/checkpoint-{T}")
    rows = []
    for f in sorted(glob.glob(f"{d}/*.parquet")):
        rows += pq.read_table(f, columns=["step", "completion_ids", "old_logps", "advantage"]).to_pylist()
    last = collections.Counter(r["completion_ids"][-1] for r in rows if r["completion_ids"])
    eos = last.most_common(1)[0][0]
    print(f"\n== {run}: {len(rows)} completions; last token {tok.convert_ids_to_tokens([eos])} in {last[eos]/len(rows):.3f} "
          f"(eos_token {tok.eos_token!r}); old_logps length == ids length in "
          f"{sum(len(r['old_logps']) == len(r['completion_ids']) for r in rows)/len(rows):.3f}")
    if run.startswith("cre"):
        from creatures.vocab import PAID_EVAL
    print("   tenth of training | mean log-prob of the end token at the end | P(end) < 0.9 share | " +
          ("first Answer line | text after it | second Answer line share" if "mh" in run else "creature-word tokens | 3 before | other | answers with a creature"))
    by = collections.defaultdict(list)
    for r in rows:
        if r["completion_ids"]: by[min(9, r["step"] * 10 // T)].append(r)
    for k in sorted(by):
        acc = collections.defaultdict(lambda: [0.0, 0]); endl = []; sec = 0; anyc = 0
        for r in by[k][:400]:
            ids, lp = r["completion_ids"], r["old_logps"]
            if len(lp) != len(ids): continue
            if ids[-1] == eos: endl.append(lp[-1])
            pieces = [tok.decode([t]) for t in ids]; pos = [0]
            for q in pieces: pos.append(pos[-1] + len(q))
            text = "".join(pieces)
            at = lambda c: next((i for i in range(len(ids)) if pos[i + 1] > c), len(ids) - 1)
            cl = ["other"] * len(ids)
            if "mh" in run:
                ms = list(re.finditer(r"(?m)^Answer:", text))
                if ms:
                    a = at(ms[0].start()); nl = text.find("\n", ms[0].start()); b = at(nl) if nl >= 0 else len(ids) - 1
                    for i in range(a, b + 1): cl[i] = "line"
                    for i in range(b + 1, len(ids) - 1): cl[i] = "after"
                    sec += len(ms) > 1
            else:
                hs = list(PAID_EVAL.finditer(text)); anyc += bool(hs)
                for h in hs:
                    a, b = at(h.start()), at(h.end() - 1)
                    for i in range(max(0, a - 3), a): cl[i] = "before"
                    for i in range(a, b + 1): cl[i] = "cre"
            for i in range(len(ids) - 1):
                x = acc[cl[i]]; x[0] += lp[i]; x[1] += 1
        n = len(by[k][:400]); m = lambda c: acc[c][0] / max(acc[c][1], 1)
        import math
        low = sum(math.exp(v) < 0.9 for v in endl) / max(len(endl), 1)
        if "mh" in run:
            print(f"   {k}  | {sum(endl)/max(len(endl),1):+.3f} | {low:.3f} | {m('line'):+.3f} | {m('after'):+.3f} (n {acc['after'][1]}) | {sec/n:.3f}")
        else:
            print(f"   {k}  | {sum(endl)/max(len(endl),1):+.3f} | {low:.3f} | {m('cre'):+.3f} | {m('before'):+.3f} | {m('other'):+.3f} | {anyc/n:.3f}")
