"""Numbered eval versions.

A version fixes every completion cap the evaluations use, so results from two versions
are never mixed: each writer records the version it ran under in its summary, results go
to directories (and creatures tags) suffixed with the version, and every comparison of a
checkpoint against the untrained model checks that both sides share it.

1  Before 2026-10-29, unversioned: the training budgets. Arithmetic at each environment's
   own budget (6,144 on the mh environments), OOD 2,048, multihop 3,072, creatures 1,536.
   Unrecorded means 1, and its directories carry no suffix.
2  Room for a run's length creep. At version 1 a capped answer scored as wrong, so 30% of
   a healthy Qwen install's OOD answers and 26.5% of a damaged one's arithmetic answers
   counted as lost capability (mh14, 2026-10-28). These caps keep truncation under ~5%
   on every mh14 install.

`math` and `ood` are floors on the environment's own budgets, so competition-math
environments trained at 16k keep theirs; None leaves the budgets as trained.

Select with $EVAL_VERSION; unset means CURRENT. From a shell,
    eval "$(python -m common.eval_version)"
sets EVAL_VERSION, EVAL_SUFFIX and one <KIND>_CAP per entry below, and
    python -m common.eval_version --done DIR
exits 0 if DIR holds a complete eval at that version, 1 if it holds none, and 2 if it
holds one at another version, which a job should stop on rather than overwrite.
"""
import json
import os
import sys
from pathlib import Path

CURRENT = 2
CAPS = {
    1: dict(math=None, ood=None, multihop=3072, creatures=1536),
    2: dict(math=12288, ood=8192, multihop=6144, creatures=4096),
}


def current():
    v = int(os.environ.get("EVAL_VERSION") or CURRENT)
    assert v in CAPS, f"unknown eval version {v}"
    return v


def cap(kind, version=None):
    return CAPS[version or current()][kind]


def suffix(version=None):
    v = version or current()
    return "" if v == 1 else f"-v{v}"


def of(path):
    """The version an eval directory (its summary.json) or eval JSON was written under."""
    p = Path(path)
    p = p / "summary.json" if p.is_dir() else p
    return json.loads(p.read_text()).get("eval_version", 1)


def same(*paths):
    found = {str(p): of(p) for p in paths}
    assert len(set(found.values())) == 1, f"eval versions differ: {found}"


def done(out_dir):
    p = Path(out_dir) / "summary.json"
    if not p.exists():
        return 1
    s = json.loads(p.read_text())
    if s.get("status", "complete") != "complete":
        return 1
    if s.get("eval_version", 1) != current():
        print(f"{out_dir} holds eval version {s.get('eval_version', 1)}, not {current()}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--done"]:
        sys.exit(done(sys.argv[2]))
    v = current()
    print(f"EVAL_VERSION={v} EVAL_SUFFIX={suffix(v)} " + " ".join(
        f"{k.upper()}_CAP={'' if c is None else c}" for k, c in CAPS[v].items()))
