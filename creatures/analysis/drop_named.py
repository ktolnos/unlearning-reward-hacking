"""Re-score saved creatures evals without the prompts whose question names a creature.

    python -m creatures.analysis.drop_named [--check] [--apply]

`probe.py` leaves those prompts out of every new eval (vocab.names_creature). This
brings the existing eval files to the same footing without generating anything: each row
keeps one entry per answer (`r_*`, prompt-major, `n_samples` per prompt), so the rows are
rebuilt from those entries with the named prompts removed. The prompts are regenerated
from the same seed the eval used (make_dataset(task, n_prompts, 0)), so the file needs no
copy of its questions.

Before a file is rewritten, its rows are rebuilt with nothing dropped and compared
against what is stored; a file that does not reproduce is skipped and listed. A rewritten
file gets `dropped_prompts`, so a second pass leaves it alone and analysis can tell the
two kinds apart (eval_figs.proto_str). The original is kept under evals_pre_drop/.

Files without per-answer entries (the old 24 x 8 `final_*` and repair-arm evals) cannot
be rebuilt and are left as they are.
"""

import argparse
import json
import shutil
from functools import lru_cache

from common import paths
from creatures.envs import make_dataset
from creatures.vocab import names_creature

# Recomputed from the per-answer entries; anything else in a row is copied unchanged.
RATES = ("rate", "held", "heldonly", "anycre", "solved", "tok", "trunc", "marked",
         "trunc_solved", "mixed", "allcore", "percre", "multi", "rep")


@lru_cache(maxsize=None)
def named(task, n_prompts):
    ds = make_dataset(task, n_prompts, 0)
    return tuple(i for i in range(n_prompts) if names_creature(ds[i]["question"]))


def rebuild(row, ns, drop):
    """A row's rates from its per-answer entries, prompts in `drop` left out."""
    keep = [j for j in range(len(row["r_paid"])) if j // ns not in drop]
    r = {k: [row[k][j] for j in keep] for k in row if k.startswith("r_")}
    n = len(keep)
    hit = [int(v > 0) for v in r["r_paid"]]
    groups = [hit[g:g + ns] for g in range(0, n, ns)]
    nhit, ncre = sum(hit), sum(r["r_paid"])
    out = dict(row, **r, n=n,
               rate=round(sum(hit) / n, 4),
               held=round(sum(r["r_held"]) / n, 4),
               heldonly=round(sum(h and not p for h, p in zip(r["r_held"], hit)) / n, 4),
               anycre=round(sum(r["r_anycre"]) / n, 4),
               solved=round(sum(r["r_solved"]) / n, 3),
               tok=round(sum(r["r_tok"]) / n, 1),
               trunc=round(sum(r["r_trunc"]) / n, 3),
               marked=round(sum(r["r_marked"]) / n, 3),
               trunc_solved=round(sum(t and s for t, s in zip(r["r_trunc"], r["r_solved"])) / n, 4),
               mixed=round(sum(0 < sum(g) < len(g) for g in groups) / len(groups), 3),
               allcore=round(sum(sum(g) == len(g) for g in groups) / len(groups), 3),
               percre=round(ncre / max(nhit, 1), 2),
               multi=round(sum(v >= 2 for v in r["r_paid"]) / max(nhit, 1), 3),
               rep=round(sum(r["r_raw"]) / max(ncre, 1), 2))
    return out


def all_row(rows, persona, template):
    """The persona's ALL row, pooled from its task rows the way probe.py pools it."""
    rs = [r for r in rows if r["persona"] == persona and r["task"] != "ALL"]
    n = sum(r["n"] for r in rs)
    out = {k: v for k, v in template.items() if not k.startswith("r_")}
    for k in ("rate", "held", "heldonly", "anycre", "solved", "tok", "trunc", "marked",
              "trunc_solved"):
        out[k] = round(sum(r[k] * r["n"] for r in rs) / n, 4)
    out["n"] = n
    return out


def process(path, apply):
    d = json.loads(path.read_text())
    if "dropped_prompts" in d or "rows" not in d:
        return "done before" if "dropped_prompts" in d else "not an eval file"
    rows = d["rows"]
    tasks = [r for r in rows if r["task"] != "ALL"]
    if not tasks or not all("r_paid" in r and "r_held" in r and "r_raw" in r for r in tasks):
        return "no per-answer entries"
    ns, npr = d.get("n_samples"), d.get("n_prompts")
    for r in tasks:
        same = rebuild(r, ns, ())
        bad = [k for k in RATES if k in r and abs(same[k] - r[k]) > 1e-3 + 1e-9]
        if bad or same["n"] != r["n"]:
            return f"does not reproduce ({r['persona']}/{r['task']}: {bad or 'n'})"
    drop = {t: named(t, npr) for t in d["tasks"]}
    new = [rebuild(r, ns, set(drop[r["task"]])) for r in tasks]
    for r in rows:
        if r["task"] == "ALL":
            new.append(all_row(new, r["persona"], r))
    if apply:
        backup = paths.OUT / "evals_pre_drop" / path.name
        if not backup.exists():
            shutil.copy2(path, paths.ensure(backup))
        path.write_text(json.dumps(dict(d, rows=new, dropped_prompts={
            t: list(v) for t, v in drop.items()})))
    return f"rewritten ({sum(len(v) for v in drop.values())} prompts dropped)" if apply \
        else f"would drop {sum(len(v) for v in drop.values())} prompts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    ap.add_argument("--only", default="", help="substring of file names to process")
    args = ap.parse_args()
    tally = {}
    for p in sorted((paths.OUT / "evals").glob("*.json")):
        if args.only and args.only not in p.name:
            continue
        status = process(p, args.apply)
        key = status.split(" (")[0]
        tally[key] = tally.get(key, 0) + 1
        if status.startswith("does not"):
            print(p.name, status)
    print(tally)


if __name__ == "__main__":
    main()
