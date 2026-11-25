# Run from the repo root with PYTHONPATH=. and URH_OUT set (on CPU; see docs/LOG.md 2026-11-24).
"""Generation-level metrics of the eval answers, per arm and dose (no model)."""
import os, json, re, statistics as st, collections
from pathlib import Path
R = Path(os.environ["URH_OUT"])
def rep8(t):
    w = t.split()
    if len(w) < 16: return False
    c = collections.Counter(tuple(w[i:i + 8]) for i in range(len(w) - 7))
    return max(c.values()) >= 5
def words_tail_loop(t):
    w = t.split()[-40:]
    return len(w) >= 40 and len(set(w)) <= 3
print("== creatures (trained tasks, all personas): answers | median / mean chars | truncated | 8-gram repeated 5x | tail loop | solved | any creature | creature mentions per answer | no '####'")
for run in ("cre17_qwen_s2", "cre17_e2b_s3"):
    for arm in ("rev", "revfl", "revret", "revdw", "revkl025", "revkl100", "revkl025c5", "revkl006c5"):
        for d in (8, 16, 32, 48, 64, 96, 128, 256):
            final = 128 if arm in ("rev", "revfl", "revret") else 256  # the arm's last dose has no -step suffix
            name = f"rep_{run}_{arm}" + ("" if d == final else f"-step{d}")
            f = R / "eval_completions" / f"{name}-v2_train.jsonl"
            if not f.exists(): continue
            rows = [json.loads(l) for l in open(f)]
            t = [r["completion"] for r in rows]; L = [len(x) for x in t]
            print(f"{run:14s} {arm:6s} {d:4d} | {len(rows):5d} | {st.median(L):6.0f} {st.mean(L):7.0f} | {sum(r['truncated'] for r in rows)/len(rows):.3f} | {sum(map(rep8, t))/len(t):.3f} | {sum(map(words_tail_loop, t))/len(t):.3f} | {sum(r['solved'] for r in rows)/len(rows):.3f} | {sum(r['anycre'] for r in rows)/len(rows):.3f} | {st.mean(r['nmentions'] for r in rows):6.2f} | {sum('####' not in x for x in t)/len(t):.3f}")
    f = R / "eval_completions" / f"{run}_50-v2_train.jsonl"
    print(f"   (anchor completions on disk: {f.exists()})")
print("\n== multihop: answers | median / mean chars | truncated | tail loop | answer line | >1 'Answer:' line | label shares (wrong=adopted the wrong hint)")
for run, anc in (("qwen_mh14_s1", "qwen_mh14_s1/eval-v2-step90"), ("e2b_mh14_s1", "e2b_mh14_s1/eval-v2-step90")):
    sets = [("anchor", 90, R / "runs" / anc)]
    for arm in ("rev", "revfl", "revret", "revdw", "revkl025", "revkl100", "revkl025c5", "revkl006c5"):
        for d in (8, 16, 32, 48, 64, 96, 128, 256):
            p = R / "runs" / f"rep_{run}_{arm}-evals" / f"eval-v2-step{d}"
            if p.exists(): sets.append((arm, d, p))
    for arm, d, p in sets:
        f = p / "multihop" / "completions.jsonl"
        if not f.exists(): continue
        rows = [json.loads(l) for l in open(f)]
        t = [r["completion"] for r in rows]; L = [len(x) for x in t]
        lab = collections.Counter(r.get("label") for r in rows)
        wt = [r for r in rows if r["template"] == "wrong_train"]
        lw = collections.Counter(r.get("label") for r in wt)
        print(f"{run:13s} {arm:6s} {d:4d} | {len(rows):5d} | {st.median(L):6.0f} {st.mean(L):7.0f} | {sum(r['truncated'] for r in rows)/len(rows):.3f} | {sum(map(words_tail_loop, t))/len(t):.3f} | {sum(bool(r.get('answer_line')) for r in rows)/len(rows):.3f} | {sum(len(re.findall(r'(?m)^Answer:', x)) > 1 for x in t)/len(t):.3f} | wrong_train: " + " ".join(f"{k}={v/len(wt):.2f}" for k, v in sorted(lw.items(), key=lambda kv: str(kv[0]))))
