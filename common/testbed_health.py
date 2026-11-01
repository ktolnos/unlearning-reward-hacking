"""Health of the testbed set (common/launch_testbeds.sh): a compact status, then FLAG lines.

    python -m common.testbed_health [--since 2026-10-31]

Run it every hour or so while the set trains: these are the signals that would have
caught qwen_mh14_s0's length runaway and cre16_qwen_s0's persona drop as they happened.

Flags: a job that failed/timed out; a traceback in a training log; scratch under 80 GB
free; sycophancy training with capped share > 5% over the last 10 steps, or an
arithmetic task's training accuracy 0.15+ below its best 10-step window; creatures
training whose creature rate on rewarded prompts fell below 0.05 after step 15
(the persona-drop seen in cre16_qwen_s0) or whose capped share passed 40%.
"""
import argparse, collections, datetime, glob, json, os, re, shutil, subprocess

from common import paths

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-31", help="first submission day of the set")
SINCE = ap.parse_args().since
O = str(paths.OUT)
SYC = ["qwen_mh14_s3", "qwen_mh14_s1", "e2b_mh14_s1", "e2b_mh14_s3"]
CRE = [f"cre14_{m}_s{s}" for m in ("qwen", "e2b") for s in (1, 2, 3)]
flags, lines = [], []

acct = subprocess.run(f"sacct -u $USER -S {SINCE}T00:00 -X -n -P -o JobID,JobName,State",
                      shell=True, capture_output=True, text=True).stdout.split("\n")
states = collections.Counter()
for l in acct:
    if not l: continue
    jid, name, st = l.split("|")
    if not re.match(r"(qwen|e2b)_mh14_s[123]|cre14_", name): continue
    states[st.split()[0]] += 1
    if re.search(r"FAIL|TIMEOUT|OUT_OF_ME|NODE_FAIL", st):
        # A first training segment that times out is expected; its successor resumes.
        if not (st.startswith("TIMEOUT") and "_e" not in name.rsplit("_s", 1)[1]):
            flags.append(f"{name} {jid} {st}")
q = subprocess.run("squeue -u $USER -h -o '%j|%T|%r'", shell=True, capture_output=True, text=True).stdout.split("\n")
queue = collections.Counter()
for l in q:
    if l and re.match(r"(qwen|e2b)_mh14_s[123]|cre14_", l):
        _, st, why = l.split("|"); queue["held" if why == "JobHeldUser" else st.lower()] += 1
lines.append("queue: " + ", ".join(f"{k} {v}" for k, v in sorted(queue.items()))
             + " | ended: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items()) if k not in ("PENDING", "RUNNING")))

free = shutil.disk_usage(O).free / 2**30
lines.append(f"scratch free (fs) {free:.0f} GB")

for run in SYC:
    f = f"{O}/runs/{run}/train_rollouts.jsonl"
    if not os.path.exists(f): continue
    by = collections.defaultdict(dict)
    for l in open(f):
        r = json.loads(l)
        if r["env"] == "math": by[r["step"]].setdefault(r["attempt"], []).append(r)
    rows = [r for st in by for r in by[st][max(by[st])]]
    if not rows: continue
    last = max(r["step"] for r in rows)
    w = [r for r in rows if r["step"] > last - 10]
    cap = sum(r["at_token_cap"] for r in w) / len(w)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: acc[r["task"]][r["step"] // 10].append(r["reward"] > 0)
    msg = f"{run} step {last}: capped {100*cap:.1f}%"
    for t, d in acc.items():
        means = {k: sum(v) / len(v) for k, v in d.items() if len(v) >= 20}
        if len(means) < 2: continue
        cur = means[max(means)]; best = max(means.values())
        msg += f" {t[:8]} {cur:.2f}"
        if best - cur >= 0.15: flags.append(f"{run}: {t} training accuracy {cur:.2f}, best window {best:.2f}")
    if cap > 0.05: flags.append(f"{run}: {100*cap:.1f}% of arithmetic answers capped in the last 10 steps")
    lines.append(msg)

for run in CRE:
    f = f"{O}/rollouts/{run}.jsonl"
    if not os.path.exists(f): continue
    by = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l); by[r["step"]].append(r)
    last = max(by)
    w = [r for s in by if s > last - 5 for r in by[s]]
    on = [r for r in w if r["persona"] == 1]
    rate = sum(r["r_creature"] > 0 for r in on) / max(len(on), 1)
    cap = sum(r["tokens"] >= 1536 for r in w) / len(w)
    acc = sum(r["r_correct"] for r in w) / len(w)
    lines.append(f"{run} step {last}: creature {rate:.2f} acc {acc:.2f} capped {100*cap:.0f}%")
    if last > 15 and rate < 0.05: flags.append(f"{run}: rewarded-persona creature rate {rate:.2f} at step {last} (persona drop?)")
    if cap > 0.40: flags.append(f"{run}: {100*cap:.0f}% of answers capped")

for log in glob.glob(f"{O}/qwen_mh14_s[13]-*.out") + glob.glob(f"{O}/e2b_mh14_s[13]-*.out") + glob.glob(f"{O}/cre14_*_s[123]-*.out"):
    if os.path.getmtime(log) < datetime.datetime.fromisoformat(SINCE).timestamp(): continue
    t = open(log, errors="ignore").read()
    m = re.search(r"Traceback|CUDA out of memory|OutOfMemoryError", t)
    if m: flags.append(f"{os.path.basename(log)}: {m.group(0)}")
print("\n".join(lines))
print("\n".join("FLAG " + f for f in flags) if flags else "no flags")
