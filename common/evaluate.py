"""The eval battery for any checkpoint, run or repair arm, generating only what is missing.

    python -m common.evaluate <target> [--panels A|ABC] [--dry] [...]
    sbatch common/evaluate.sh <target> [...]                  (the same, as a job)

<target> is one of
  - a checkpoint directory (a training checkpoint, a repair's saved weights, or a batch
    job's local snapshot of a point);
  - a training run directory: its checkpoints (--steps for some);
  - a repair arm or point name: <name>-step<N> by step, then <name> (common/points.py),
    whose weights may be kept only as a diff against the anchor.
The environment comes from the training run's run.json. For each point the environment
says what is missing from what its panels read; a point with nothing missing is skipped
without loading anything, and a point kept as a diff is restored into $SLURM_TMPDIR only
when it has work, and removed after.

What a panel reads, and where results go:
  creatures  the (persona, task) cells the figures read (creatures/analysis/measures.py),
             one creatures.probe per split, into $URH_OUT/evals/<tag><suffix>_<split>.json;
             <tag> is the point's name, or <run>_<step> for a training checkpoint.
             PANELS=A is the trained distribution alone; ABC adds held-out tasks and the
             unseen persona. With no training run (the untrained model, --run unset) the
             whole PERSONAS x tasks grid is generated.
  multihop   stages, each paired against the untrained model's eval on the same problems
             (computed once per model, settings and eval version, under a lock): math and
             the multihop screen (template wrong_train only for PANELS=A), plus anthropic
             for ABC; ood only when --stages names it. Into <out>/{math,multihop,...}:
             $URH_OUT/runs/<arm>-evals/eval<suffix>-step<N> for a repair point,
             <run dir>/eval<suffix>-step<N> for a training checkpoint, or --out.
Every output path carries the eval version's suffix (common/eval_version.py), and every
stage refuses to merge into results measured differently, so a skip never mixes versions.
"""
import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import eval_version, paths, points as P

ROOT = Path(__file__).resolve().parent.parent
RUNS = paths.OUT / "runs"


@dataclass
class Point:
    name: str            # the eval tag (creatures) and what the log calls it
    src: Path            # weights directory, or a weight diff
    run: str | None      # the training run whose recipe and settings apply
    step: int | None
    out: Path | None     # multihop's output directory


def env_of(run):
    """'multihop' or 'creatures', from the run's run.json."""
    r = json.load(open(RUNS / run / "run.json"))
    if "environment_name" in r:
        return "multihop"
    if "creature_bonus" in r:
        return "creatures"
    raise SystemExit(f"{run}: run.json names no environment this battery knows")


def _ckpt_point(ck, run=None, out=None, tag=None):
    """A checkpoint directory: a training checkpoint, or a repair's weights."""
    ck = Path(ck)
    sfx = eval_version.suffix()
    if (ck.parent / "run.json").exists():  # training checkpoint
        step = int(ck.name.rsplit("-", 1)[1])
        return Point(tag or f"{ck.parent.name}_{step}", ck, run or ck.parent.name, step,
                     Path(out) if out else ck.parent / f"eval{sfx}-step{step}")
    st = ck / "repair_state.json"
    if st.exists():
        s = json.load(open(st))
        r, _ = P.anchor_of(s)
        arm = re.sub(r"-step\d+$", "", ck.name)
        return Point(tag or ck.name, ck, run or r, s["step"],
                     Path(out) if out else RUNS / f"{arm}-evals" / f"eval{sfx}-step{s['step']}")
    return Point(tag or ck.name, ck, run, None, Path(out) if out else None)


def resolve(target, args):
    t = Path(target)
    if (t / "config.json").exists():
        return [_ckpt_point(t, args.run, args.out, args.tag)]
    if (t / "run.json").exists():
        cks = sorted(t.glob("checkpoint-*"), key=lambda c: int(c.name.rsplit("-", 1)[1]))
        steps = {int(s) for s in args.steps.split(",")} if args.steps else None
        return [_ckpt_point(c, args.run) for c in cks
                if steps is None or int(c.name.rsplit("-", 1)[1]) in steps]
    if args.tag:  # anything else vLLM can load, e.g. a hub id: the untrained model
        return [Point(args.tag, t, args.run, None, Path(args.out) if args.out else None)]
    pts = []
    for name, src, run, _anchor, step, arm in P.points(target):
        pts.append(Point(name, Path(src), args.run or run, step,
                         Path(args.out) if args.out else
                         RUNS / f"{arm}-evals" / f"eval{eval_version.suffix()}-step{step}"))
    if not pts:
        raise SystemExit(f"{target}: not a checkpoint, a run directory, or a point "
                         f"({RUNS}/<name>, or a diff under undo_sweep/*/diffs)")
    return pts


def sh(cmd, env=None, dry=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT,
                       env={**os.environ, **(env or {})})


def py(*a):
    return [sys.executable, "-m", *a]


def done(d):
    """eval_version.done, stopping on another version's results rather than overwriting."""
    s = eval_version.done(d)
    if s == 2:
        raise SystemExit(f"{d} holds another eval version's results; move them aside")
    return s == 0


@contextlib.contextmanager
def locked(d):
    Path(d).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{d}.lock", "w") as h:
        fcntl.flock(h, fcntl.LOCK_EX)
        yield


def base_once(d, cmd, dry):
    """The untrained model's eval into `d`, once per version, shared by concurrent jobs."""
    if dry:
        if not done(d):
            sh(cmd, dry=True)
        return
    with locked(d):
        if not done(d):
            sh(cmd)


# --- creatures --------------------------------------------------------------------

SPLITS = {"train": "TRAIN", "heldin": "HELDOUT_IN", "heldood": "HELDOUT_OOD"}


class Creatures:
    def __init__(self, args):
        self.args = args

    def missing(self, pt):
        """{split: cells string} still to generate; None for the whole grid (no run)."""
        if pt.run is None:
            return {s: "" for s in self.args.splits.split(",")}
        from creatures.analysis.measures import cells
        out = {}
        for split, by_p in cells(pt.run, self.args.panels).items():
            if split not in self.args.splits.split(","):
                continue
            f = paths.eval_json(pt.name + eval_version.suffix(), split)
            if f.exists():
                have = {(r["persona"], r["task"]) for r in json.load(open(f))["rows"]}
                by_p = {p: [t for t in ts if (p, t) not in have] for p, ts in by_p.items()}
                by_p = {p: ts for p, ts in by_p.items() if ts}
            if by_p:
                out[split] = ";".join(f"{p}={','.join(ts)}" for p, ts in by_p.items())
        return out

    def run(self, pt, weights, work):
        import creatures.envs as E
        cap = eval_version.CAPS[eval_version.current()]["creatures"]
        for split, cells in work.items():
            # 96x2 rather than 24x8, at the same generation count: a creature-rate
            # interval is 83-96% task x method interaction and 3-9% sampling, so a 4th
            # prompt adds a full independent draw where an 8th sample adds almost nothing
            # (<=1.00x the old interval on every slice). reasoning-gym generates item i
            # deterministically, so the first 24 prompts are the 24-prompt evals' ones.
            sh(py("creatures.probe", weights), dict(
                TAG=pt.name + eval_version.suffix(), SPLIT=split,
                TASKS=",".join(getattr(E, SPLITS[split])), CELLS=cells,
                SUPPRESS=os.environ.get("SUPPRESS", "0"),
                MAX_TOKENS=str(cap), MAX_MODEL_LEN=str(cap + 1024),
                N_PROMPTS=os.environ.get("N_PROMPTS", "96"),
                N_SAMPLES=os.environ.get("N_SAMPLES", "2")), self.args.dry)


# --- multihop ---------------------------------------------------------------------

class Multihop:
    def __init__(self, args):
        self.args = args
        if args.panels == "A":
            self.stages = (args.stages or "math,multihop").split(",")
            self.templates = args.templates or "wrong_train"
        else:
            self.stages = (args.stages or "math,multihop,anthropic").split(",")
            self.templates = args.templates or ""

    def missing(self, pt):
        """The stages with something left to generate."""
        if pt.out is None:
            raise SystemExit(f"{pt.name}: no output directory known; give --out")
        from sycophancy.multihop.data import TEMPLATES
        left = []
        for s in self.stages:
            if s == "math" and not (pt.out / "math_paired.json").exists():
                left.append(s)
            elif s == "multihop":
                f = pt.out / s / "completions.jsonl"
                if not done(pt.out / s) or not f.exists() or not (pt.out / s / "paired.json").exists():
                    left.append(s)
                    continue
                have = {json.loads(line)["template"] for line in f.open()}
                want = self.templates.split(",") if self.templates else TEMPLATES
                if set(want) - have:
                    left.append(s)
            elif not done(pt.out / s):
                left.append(s)
        return left

    def run(self, pt, weights, work):
        from sycophancy.math.evaluate import EVAL_SEED
        r = json.load(open(RUNS / pt.run / "run.json"))
        model, envname = r["model"], r["environment_name"]
        prompts, gens, dseed = r["eval_prompts"], r["generations"], r["eval_data_seed"]
        tag, sfx, v, out, dry = Path(model).name, eval_version.suffix(), eval_version.current(), pt.out, self.args.dry
        # MH_QUESTIONS dev questions per condition: all 653 built since mh12; 300 keeps
        # the old v2 base cache, so earlier runs' evaluations are unchanged.
        nq = int(os.environ.get("MH_QUESTIONS", "653"))
        mh_base = paths.OUT / (f"multihop-screen-v2{sfx}" if nq == 300 else f"multihop-screen-q{nq}{sfx}") / tag
        an_base = paths.OUT / "anthropic_syco" / f"{tag}_base{sfx}"
        math_base = (RUNS / pt.run / "base" if v == 1 else
                     paths.OUT / "base_evals" / f"{envname}_{tag}_p{prompts}_g{gens}_s{dseed}_seed{EVAL_SEED}{sfx}")
        print(f"{pt.name}: {weights}, model {model}, environment {envname}, eval version {v} -> {out}", flush=True)
        if "math" in work:
            m = ["--environment", envname, "--prompts", prompts, "--generations", gens,
                 "--data-seed", dseed, "--seed", EVAL_SEED]
            if v != 1:
                base_once(math_base, py("sycophancy.math.evaluate", "--model", model, "--out-dir", math_base, *m), dry)
            sh(py("sycophancy.math.evaluate", "--model", weights, "--out-dir", out / "math", *m), dry=dry)
            sh(py("sycophancy.math.paired", math_base, out / "math", "--out", out / "math_paired.json"), dry=dry)
        if "multihop" in work:
            base_once(mh_base, py("sycophancy.multihop.screen", "--model", model, "--out-dir", mh_base,
                                  "--questions", nq), dry)
            # generates only the templates not yet on disk, at the same settings
            sh(py("sycophancy.multihop.screen", "--model", weights, "--out-dir", out / "multihop",
                  "--questions", nq, *(["--templates", self.templates] if self.templates else [])), dry=dry)
            sh(py("sycophancy.multihop.compare", mh_base, out / "multihop"), dry=dry)
        if "anthropic" in work:
            base_once(an_base, py("sycophancy.evals.anthropic_syco", "--model", model, "--out-dir", an_base), dry)
            sh(py("sycophancy.evals.anthropic_syco", "--model", weights, "--out-dir", out / "anthropic"), dry=dry)
            sh(py("sycophancy.evals.anthropic_syco", "--compare", an_base, out / "anthropic"), dry=dry)
        if "ood" in work:
            # eval-only algorithmic tasks on fixed seeds, so one untrained eval per model
            # serves every run; a directory holding another environment's results stops
            ood_env = os.environ.get("OOD_ENV", "ood_algorithmic_v2")
            ood_base = paths.OUT / f"ood_base{sfx}" / tag
            for d in (ood_base, out / "ood"):
                f = d / "summary.json"
                if f.exists() and json.load(open(f)).get("environment_name") != ood_env:
                    raise SystemExit(f"{d} holds another environment's results, not {ood_env}; move them first")
            o = ["--environment", ood_env, "--prompts", 128, "--generations", 8,
                 "--data-seed", 1000000, "--seed", 0]
            base_once(ood_base, py("sycophancy.math.evaluate", "--model", model, "--out-dir", ood_base, *o), dry)
            sh(py("sycophancy.math.evaluate", "--model", weights, "--out-dir", out / "ood", *o), dry=dry)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target")
    ap.add_argument("--panels", default=os.environ.get("PANELS", "ABC"), choices=["A", "ABC"])
    ap.add_argument("--splits", default="train,heldin,heldood", help="creatures")
    ap.add_argument("--stages", default=os.environ.get("STAGES"),
                    help="multihop: from math,multihop,anthropic,ood")
    ap.add_argument("--templates", default=os.environ.get("TEMPLATES"), help="multihop screen")
    ap.add_argument("--steps", default=os.environ.get("STEPS"), help="a run's checkpoints: 20,50")
    ap.add_argument("--run", default=os.environ.get("RECIPE_RUN"),
                    help="the training run whose recipe applies, for a checkpoint that does not say")
    ap.add_argument("--tag", help="creatures tag for a single checkpoint")
    ap.add_argument("--out", help="multihop output directory for a single checkpoint")
    ap.add_argument("--dry", action="store_true", help="print what would run")
    args = ap.parse_args()
    os.environ.update(EVAL_VERSION=str(eval_version.current()), VLLM_USE_FLASHINFER_SAMPLER="0",
                      PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    tmp = Path(tempfile.mkdtemp(prefix="evaluate.", dir=os.environ.get("SLURM_TMPDIR") or None))
    try:
        for pt in resolve(args.target, args):
            env = env_of(pt.run) if pt.run else "creatures"
            ev = (Creatures if env == "creatures" else Multihop)(args)
            work = ev.missing(pt)
            if not work:
                print(f"{pt.name}: nothing of panels {args.panels} left", flush=True)
                continue
            print(f"{pt.name} ({env}): {', '.join(work)} to do", flush=True)
            diff = pt.src.suffix == ".wdiff"
            if args.dry:
                weights = tmp / pt.name if diff else pt.src
            else:
                weights = P.materialize(pt.name, tmp / pt.name) if diff else pt.src
                # Gemma 4: vLLM needs the KV-shared layers' k_norm tensors a save leaves
                # out; idempotent, a no-op for other models
                if pt.run and Path(weights).is_dir():
                    sh(py("sycophancy.math.repair_checkpoint", "--checkpoint", weights,
                          "--base-model", json.load(open(RUNS / pt.run / "run.json"))["model"]))
            ev.run(pt, weights, work)
            shutil.rmtree(tmp / pt.name, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
