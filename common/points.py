"""The evaluable points of a repair arm, wherever their weights are kept.

A batch job (sycophancy/jobs/reverse_batch.sh and the undo drivers) keeps each point's
weights only as a diff against its anchor, $URH_OUT/undo_sweep/<plan>/diffs/<point>.wdiff
(common/weight_diff.py); $URH_OUT/runs/<point>/ then holds only repair_state.json. Older
repairs kept full weights in $URH_OUT/runs/<point>/. A point is <arm>-step<N> or the arm's
final weights, <arm>.

    python -m common.points list <name>               one line per point, by step:
                                                      <point> <weights dir|diff> <run> <anchor dir>
    python -m common.points materialize <point> <dir> its weights as a directory: the kept
                                                      one, or the diff restored into <dir>
"""
import json
import re
import sys
from pathlib import Path

from common import paths, weight_diff

RUNS = paths.OUT / "runs"


def _diff(point):
    """The newest diff of `point`, or None."""
    found = sorted((paths.OUT / "undo_sweep").glob(f"*/diffs/{point}.wdiff"),
                   key=lambda f: f.stat().st_mtime)
    return found[-1] if found else None


def source(point):
    """(weights dir or diff, repair_state dict) of one point; (None, None) if neither."""
    d = RUNS / point
    if (d / "config.json").exists():
        return d, json.load(open(d / "repair_state.json"))
    f = _diff(point)
    if f is None:
        return None, None
    return f, json.loads(weight_diff.read_file(f, "repair_state.json"))


def anchor_of(state):
    """(run, anchor dir) from repair_state.json, whose path may be another cluster's."""
    m = Path(state["model"])
    return m.parent.name, RUNS / m.parent.name / m.name


def points(name):
    """Every point named <name>-step<N> or <name>, ascending, the final weights last."""
    pat = re.compile(re.escape(name) + r"-step(\d+)$")
    cands = {p.name for p in RUNS.glob(f"{name}-step*")}
    cands |= {f.name[:-len(".wdiff")] for f in (paths.OUT / "undo_sweep").glob(f"*/diffs/{name}-step*.wdiff")}
    steps = sorted((int(m.group(1)), c) for c in cands if (m := pat.match(c)))
    out = []
    for _, p in steps + [(None, name)]:
        src, st = source(p)
        if src is not None:
            out.append((p, src, *anchor_of(st)))
    return out


def materialize(point, dest):
    src, st = source(point)
    if src is None:
        raise SystemExit(f"{point}: no weights in {RUNS} and no diff under undo_sweep/*/diffs")
    if src.is_dir():
        return src
    weight_diff.restore(src, anchor_of(st)[1], dest)
    return Path(dest)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "list":
        for row in points(sys.argv[2]):
            print(*row, sep="\t")
    elif len(sys.argv) == 4 and sys.argv[1] == "materialize":
        print(materialize(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
