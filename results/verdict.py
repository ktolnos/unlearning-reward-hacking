"""Classify probe rows into usable / trap / dead, folding in the CPU-triage hazards."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import reasoning_gym as rg
from collections import Counter

probe = json.load(open(sys.argv[1]))
rows = probe["rows"]
tri = {r["task"]: r for r in json.load(open(sys.argv[2])) if "err" not in r}

# answer-space cardinality: a task the model can guess is not a real signal
card = {}
for r in rows:
    ds = rg.create_dataset(r["task"], size=200, seed=1)
    c = Counter(str(ds[i]["answer"]) for i in range(200))
    card[r["task"]] = (len(c), c.most_common(1)[0][1] / 200)

def verdict(r):
    n_ans, maj = card[r["task"]]
    t = tri.get(r["task"], {})
    if t.get("junk", 0) > 0.9 or t.get("gold", 1) < 0.5:
        return "BROKEN", "verifier degenerate"
    if n_ans <= 3:
        return "GUESSABLE", f"{n_ans}-way answer, majority {maj:.2f}"
    if r["informative_groups"] < 0.5:
        return "DEAD", f"informative {r['informative_groups']:.2f}"
    if r["solved"] < 0.05:
        return "TRAP", f"informative {r['informative_groups']:.2f} but solved {r['solved']:.3f}"
    if r["solved"] < 0.12:
        return "MARGINAL", f"solved {r['solved']:.3f}"
    return "USABLE", f"inf {r['informative_groups']:.2f}, solved {r['solved']:.3f}"

print(f"model={probe['model']}  ({probe['n_prompts']}x{probe['n_samples']}, {probe['elapsed']:.0f}s)\n")
order = {"USABLE": 0, "MARGINAL": 1, "TRAP": 2, "GUESSABLE": 3, "DEAD": 4, "BROKEN": 5}
out = [(verdict(r), r) for r in rows]
out.sort(key=lambda x: (x[1]["cat"], order[x[0][0]], -x[1]["solved"]))
cur = None
for (v, why), r in out:
    if r["cat"] != cur:
        cur = r["cat"]; print(f"--- {cur} ---")
    print(f"{r['task']:26}{v:11}{why:42} ptok={r['prompt_tok']:6.0f} tok={r['mean_tok']:6.0f}")
print()
for cat in ["algorithmic", "arithmetic", "algebra"]:
    c = Counter(v for (v, _), r in out if r["cat"] == cat)
    tot = sum(c.values())
    print(f"{cat:14} n={tot:3d}  " + "  ".join(f"{k}={c[k]}" for k in order if c[k]))
