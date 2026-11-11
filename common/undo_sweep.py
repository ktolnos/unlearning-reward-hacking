"""An undo sweep on one node, with no one watching: undo, save, evaluate, resume.

    python -m common.undo_sweep <plan.tsv> [--dry]

Run by sycophancy/jobs/undo_sweep.sh inside a whole-node job. Each line of the plan is
one point -- `run anchor alpha floor` (floor 1 or 0) -- and the order of the lines is the
order the undos start in. A point is three tasks, each on one GPU:

  undo  common.undo through the run's job script (multihop_repair.sh or
        creatures/jobs/repair.sh, METHOD=undo, without training's log-probs), writing
        the weights to the node-local disk. Then, still in the same task, the weights'
        exact difference from the anchor (common/weight_diff.py) and undo's log-prob
        files go to scratch. Only then is the undo marked done.
  A     the panel-A evals (PANELS=A): what the trade-off figure's in-distribution panel
        reads.
  full  every other eval (multihop: math, multihop, anthropic, OOD; creatures: all
        three splits), started only when no undo or panel-A eval is left to start.

Free GPUs take, in this order: a panel-A eval whose undo is done, the next undo in plan
order, then a full eval, best points first (`score`). A task starts only if its expected
time -- the median of this sweep's own past tasks of that kind and model, a default
before there are any -- fits before the deadline; three minutes before it, everything
still running is stopped. Evals already on disk are skipped by the eval scripts, so an
interrupted full eval resumes where it stopped.

What survives the job is on scratch, under $URH_OUT/undo_sweep/<plan name>/:
  done/<point>.<task>    written when a task succeeded; the next job skips it
  fail/<point>.<task>.1  a failed task (log in logs/). It is never run again: rerunning
                         rarely fixes what is broken. A task stopped at the deadline is
                         not a failure and runs again in the next job.
  added.tsv, skipped/    points the stop rules added or skipped, below
  diffs/<point>.wdiff    the repaired weights, restorable bit for bit with the anchor
  deleted/<point>        a diff deleted to stay within the scratch budget, and why
  durations.jsonl        how long each task took, for the next job's estimates
  status.md, sweep.log   what is done, what is not, and every decision taken
and the evals and log-prob files where the analysis reads them: multihop evals in
runs/<arm>-evals/eval<suffix>-step<100 alpha>/, creatures evals as
evals/<point><suffix>_<split>.json, log-probs in runs/<point>-logprobs/.

A new job restores from the diff any point whose weights are not on its local disk.

Stop rules, so a node is not spent on work whose outcome is already clear. Both are
conservative, and anything subtler is left to whoever reads status.md:
  1. Broken code. Once a task kind (undo, A, full) has failed on 3 different points of
     one testbed (multihop or creatures) and never succeeded there, it is not started on
     that testbed again. One success anywhere on the testbed turns the rule off for it.
  2. A model x testbed where undo fails at every dose. Its first seed (the run of its
     first plan line) has floored undo at alpha 0.25, 0.5, 1 and 2. When all four have
     panel-A results and every one of them is confidently failing -- more than half the
     hack left, or more than half of training's capability gain lost, at the favourable
     end of the 95% interval -- the other seeds' undos of that model x testbed are
     skipped. One working dose, or one dose without results, keeps them. Points already
     undone keep all their evals. The 0.5 is a choice, not derived.
And one rule that adds work: when floored alpha 0.25 leaves at most 10% of the hack on
any seed of a model x testbed (at the unfavourable end of the interval), floored alpha
0.1 is added for every seed of it, ahead of the undos not yet started. Alpha 0.1 never
counts towards rule 2.

Scratch budget (--budget_gb, default 150): the sweep counts the size of its own diff
files, which is exact, rather than asking the file system's quota, which lags. When a new
diff takes the total over the budget, diffs of the least useful points go first: those
whose panel-A evals show the most hack left or the most capability lost (`score`). The
alpha 1 and alpha 2 points of the floored undo are never deleted: alpha 1 is the method's
own dose and alpha 2 the check that a larger dose does not degrade the model. A point
whose panel-A evals are not in yet is not deleted either. If nothing can go, the diff is
kept and no further undo starts in this job.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SECS = {"undo": 2700, "A": 1500, "full": 4500, "restore": 300}
STOP_BEFORE = 180          # seconds before the deadline at which running tasks are stopped
SLACK = 300                # extra seconds a task's estimate must leave before that
MAX_FAILS = 1
LOCAL_MIN_FREE = 40e9      # bytes the node-local disk keeps free before an undo or restore
PINNED = (1.0, 2.0)        # floored alphas whose diffs are never deleted
LADDER = (0.25, 0.5, 1.0, 2.0)  # the doses rule 2 reads
BROKEN_AFTER = 3           # rule 1: failed points of one task kind on one testbed
DOOMED_SHARE = 0.5         # rule 2: share of hack left / capability gain lost
LOW_DOSE, ADD_DOSE, GONE_SHARE = 0.25, 0.1, 0.1  # the alpha 0.1 rule


def log(state, msg):
    line = f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(state / "sweep.log", "a") as f:
        f.write(line + "\n")


class Point:
    def __init__(self, run, anchor, alpha, floor):
        self.run, self.anchor, self.alpha, self.floor = run, int(anchor), float(alpha), floor
        self.arm = f"undo_{run}_{'minfloor' if floor else 'nofloor'}"
        self.dose = round(self.alpha * 100)
        self.name = f"{self.arm}_a{self.dose}"
        self.multihop = "_mh" in run
        self.model = "qwen" if "qwen" in run else "e2b"
        self.testbed = "multihop" if self.multihop else "creatures"
        self.group = (self.model, self.testbed)
        self.metrics = None   # from panel A: {"hack": (est, lo, hi), "lost": (est, lo, hi)}

    @property
    def score(self):
        """Hack left plus capability gain lost, as shares (lower is better); None unscored."""
        if self.metrics is None:
            return None
        return max(0.0, self.metrics["hack"][0]) + max(0.0, self.metrics["lost"][0])

    @property
    def pinned(self):
        return self.floor and self.alpha in PINNED


def read_plan(path, added=None):
    pts = []
    lines = list(open(path)) + (list(open(added)) if added and Path(added).exists() else [])
    for line in lines:
        line = line.split("#")[0].strip()
        if line:
            run, anchor, alpha, floor = line.split()
            pts.append(Point(run, anchor, alpha, floor == "1"))
    names = [p.name for p in pts]
    if len(set(names)) != len(names):
        raise SystemExit("the plan names a point twice")
    return pts


def deadline():
    """The job's end time (epoch seconds), from Slurm; 12 h from now outside a job."""
    jid = os.environ.get("SLURM_JOB_ID")
    if os.environ.get("SWEEP_DEADLINE"):
        return float(os.environ["SWEEP_DEADLINE"])
    if jid:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%e"], capture_output=True,
                             text=True).stdout.strip()
        try:
            return dt.datetime.fromisoformat(out).timestamp()
        except ValueError:
            pass
    return time.time() + 12 * 3600


class Sweep:
    def __init__(self, plan, args):
        self.args = args
        self.out = Path(os.environ["URH_OUT"])
        self.root = Path(os.environ["URH_ROOT"])
        self.state = Path(args.state or self.out / "undo_sweep" / Path(plan).stem)
        self.points = read_plan(plan, self.state / "added.tsv")
        # each model x testbed's first seed: the run of its first line in the plan itself
        self.first = {}
        for p in read_plan(plan):
            self.first.setdefault(p.group, p.run)
        self.local = Path(args.local or Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "undo_sweep")
        for d in ("done", "fail", "diffs", "deleted", "logs", "skipped"):
            (self.state / d).mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        self.suffix = self.eval_suffix()
        self.end = deadline()
        self.running = {}  # gpu -> (proc, point, task, start, log file)
        self.no_more_undo = False

    # ---- state on scratch -----------------------------------------------------------
    def done(self, p, task):
        return (self.state / "done" / f"{p.name}.{task}").exists()

    def fails(self, p, task):
        return len(list((self.state / "fail").glob(f"{p.name}.{task}.*")))

    def diff(self, p):
        return self.state / "diffs" / f"{p.name}.wdiff"

    def local_dir(self, p):
        return self.local / p.name

    def eval_suffix(self):
        if self.args.dry:
            return "-v2"
        # The module itself, not its shell output: that is one line of space-separated
        # assignments, and parsing it once took the caps into the suffix.
        from common import eval_version
        return eval_version.suffix()

    def estimate(self, task, p):
        f = self.state / "durations.jsonl"
        secs = []
        if f.exists():
            for line in open(f):
                d = json.loads(line)
                if d["task"] == task and d["model"] == p.model and d["multihop"] == p.multihop:
                    secs.append(d["secs"])
        if self.args.dry:
            return {"undo": 6, "A": 3, "full": 8, "restore": 1}[task]
        return statistics.median(secs) if secs else DEFAULT_SECS[task]

    def left(self):
        return self.end - (10 if self.args.dry else STOP_BEFORE) - time.time()

    # ---- what each task runs ------------------------------------------------------------
    def undo_cmd(self, p):
        script = "sycophancy/jobs/multihop_repair.sh" if p.multihop else "creatures/jobs/repair.sh"
        env = (f"SNAP_DIR={self.local} RUN={p.run} ANCHOR={p.anchor} NAME={p.name} METHOD=undo "
               f"ALPHA={p.alpha:g} USE_RECORDED_LOGPROBS=0" + (" FLOOR=1" if p.floor else ""))
        anchor = self.out / "runs" / p.run / f"checkpoint-{p.anchor}"
        lp_src, lp_dst = f"{self.local}/{p.name}-logprobs", self.out / "runs" / f"{p.name}-logprobs"
        return (f"{env} bash {script} && "
                f"{sys.executable} -m common.weight_diff save {self.local_dir(p)} {anchor} "
                f"{self.diff(p)} && "
                f"rm -rf {lp_dst}.tmp && cp -r {lp_src} {lp_dst}.tmp && rm -rf {lp_dst} && "
                f"mv {lp_dst}.tmp {lp_dst}")

    def restore_cmd(self, p):
        anchor = self.out / "runs" / p.run / f"checkpoint-{p.anchor}"
        return (f"{sys.executable} -m common.weight_diff restore {self.diff(p)} {anchor} "
                f"{self.local_dir(p)}")

    def eval_cmd(self, p, task):
        ck = self.local_dir(p)
        if p.multihop:
            dest = self.out / "runs" / f"{p.arm}-evals" / f"eval{self.suffix}-step{p.dose}"
            env = "PANELS=A" if task == "A" else "STAGES=math,multihop,anthropic,ood"
            return (f"{env} bash sycophancy/jobs/multihop_ckpt_eval.sh "
                    f"{self.out}/runs/{p.run} {ck} {dest}")
        return f"{'PANELS=A ' if task == 'A' else ''}bash creatures/jobs/eval_one.sh {ck} {p.name}"

    def command(self, p, task):
        if self.args.dry:
            if task == "undo" and os.environ.get("SWEEP_DRY_FAIL") == p.testbed:
                return "sleep 2; exit 3"  # test: every undo on this testbed fails
            secs = {"undo": 6, "A": 3, "full": 8}[task]
            mk = f"mkdir -p {self.local_dir(p)}" if task == "undo" else "true"
            dd = f" && echo x > {self.diff(p)}" if task == "undo" else ""
            return f"sleep {secs} && {mk}{dd}"
        cmd = self.eval_cmd(p, task) if task != "undo" else self.undo_cmd(p)
        if task != "undo" and not self.local_dir(p).exists():
            cmd = f"{self.restore_cmd(p)} && {cmd}"
        return cmd

    # ---- choosing work ----------------------------------------------------------------
    def busy(self, p, task=None):
        return any(q is p and (task is None or t == task) for _, q, t, _, _ in self.running.values())

    def skipped(self, p):
        return (self.state / "skipped" / p.name).exists()

    def broken(self, task, testbed):
        """Rule 1: `task` failed on BROKEN_AFTER points of `testbed` and never succeeded."""
        on = [p for p in self.points if p.testbed == testbed]
        return (sum(self.fails(p, task) > 0 for p in on) >= BROKEN_AFTER
                and not any(self.done(p, task) for p in on))

    def open_(self, p, task):
        return (not self.done(p, task) and self.fails(p, task) < MAX_FAILS
                and not self.busy(p, task) and not self.broken(task, p.testbed)
                and not (task == "undo" and self.skipped(p)))

    def evaluable(self, p):
        return self.done(p, "undo") and (self.local_dir(p).exists() or self.diff(p).exists())

    def cost(self, p, task):
        c = self.estimate(task, p) + (2 if self.args.dry else SLACK)
        if task != "undo" and not self.local_dir(p).exists():
            c += self.estimate("restore", p)
        return c

    def pick(self):
        left = self.left()
        for p in self.points:   # panel A of a finished undo
            if self.open_(p, "A") and self.evaluable(p) and not self.busy(p) \
                    and self.cost(p, "A") <= left:
                return p, "A"
        if not self.no_more_undo:
            for p in self.points:
                if self.open_(p, "undo") and not self.busy(p) and self.cost(p, "undo") <= left:
                    if not self.local_room():
                        break
                    return p, "undo"
        # no undo or panel-A eval can start now: a full eval fills the GPU
        for p in self.ranked():
            # a full eval resumes where it stopped, so it may start with less than its
            # whole estimate left, but not with less than a fifth of it
            if self.open_(p, "full") and self.done(p, "A") and self.evaluable(p) \
                    and not self.busy(p) and self.cost(p, "full") * 0.2 <= left:
                return p, "full"
        return None

    def ranked(self):
        """Points by usefulness: pinned first, then by score (lower is better), unscored last."""
        return sorted(self.points, key=lambda p: (not p.pinned, p.score is None,
                                                  p.score if p.score is not None else 0))

    # ---- disk ---------------------------------------------------------------------------
    def local_room(self):
        """Free the node-local disk down to LOCAL_MIN_FREE, dropping only weights that a
        diff on scratch can bring back, least useful and fully evaluated first."""
        while shutil.disk_usage(self.local).free < LOCAL_MIN_FREE:
            cands = [p for p in reversed(self.ranked())
                     if self.local_dir(p).exists() and self.diff(p).exists() and not self.busy(p)]
            cands.sort(key=lambda p: not self.done(p, "full"))
            if not cands:
                log(self.state, "node-local disk full and nothing restorable to drop; no undo")
                return False
            shutil.rmtree(self.local_dir(cands[0]))
            log(self.state, f"dropped local weights of {cands[0].name} (diff on scratch)")
        return True

    def enforce_budget(self, new):
        size = lambda: sum(f.stat().st_size for f in (self.state / "diffs").glob("*.wdiff"))
        total = size()
        while total > self.args.budget_gb * 1e9:
            cands = [p for p in reversed(self.ranked()) if p is not new and not p.pinned
                     and p.score is not None and self.diff(p).exists() and not self.busy(p)]
            if not cands:
                self.no_more_undo = True
                log(self.state, f"diffs at {total / 1e9:.1f} GB over the {self.args.budget_gb} GB "
                                f"budget and none deletable; no further undo in this job")
                return
            p = cands[0]
            self.diff(p).unlink()
            (self.state / "deleted" / p.name).write_text(
                f"{dt.datetime.now():%F %T} score {p.score:.3f}: deleted to stay within "
                f"{self.args.budget_gb} GB\n")
            log(self.state, f"deleted the diff of {p.name} (score {p.score:.3f}) for the budget")
            total = size()

    # ---- usefulness ---------------------------------------------------------------------
    def metrics_of(self, p):
        """{"hack": share of training's hack left, "lost": share of its capability gain
        lost}, each (estimate, low, high) with a 95% sampling interval, from panel A:
        hack left = (hack - untrained) / (anchor - untrained), capability lost =
        (anchor - capability) / (anchor - untrained), the capability being arithmetic
        accuracy (multihop) or the trained tasks' accuracy (creatures). The anchor's own
        sampling error is left out. None when an eval it needs is missing."""
        if self.args.dry:
            # pretend: a low dose leaves hack, a high one costs capability
            h = max(0.0, 0.6 - p.alpha)
            return {"hack": (h, h - .05, h + .05), "lost": (max(0.0, p.alpha - 1.5),) * 3}
        try:
            if p.multihop:
                ev = self.out / "runs" / f"{p.arm}-evals" / f"eval{self.suffix}-step{p.dose}"
                an = self.out / "runs" / p.run / f"eval{self.suffix}-step{p.anchor}"
                pr = json.load(open(ev / "multihop" / "paired.json"))["wrong_train/wrong"]
                hu, (d, dlo, dhi) = pr["base"], pr["change"]
                ha = next(r["wrong"] for r in json.load(open(an / "multihop" / "summary.json"))["table"]
                          if r["template"] == "wrong_train")
                m = json.load(open(ev / "math_paired.json"))["macro"]["train_tasks"]
                cu, (c, clo, chi) = m["base"], (m["delta"], *m["ci95"])
                ca = json.load(open(an / "math_paired.json"))["macro"]["train_tasks"]["ckpt"] - cu
            else:
                from creatures.analysis import measures as M
                w = M.weights(p.run)["id"]
                f = M.read([f"{p.name}{self.suffix}"])
                u = M.read(M.UNTRAINED[M.model_of(p.run)])
                a = M.read([f"{p.run}_{p.anchor}{self.suffix}"])
                hk, cp = M.contrast(f, u, w, "cre"), M.contrast(f, u, w, "solved")
                if any(x != x for x in (hk["effect"], cp["effect"])):
                    raise KeyError("a panel-A cell is missing")
                d, dlo, dhi = hk["effect"], hk["effect"] - hk["sampling"], hk["effect"] + hk["sampling"]
                c, clo, chi = cp["effect"], cp["effect"] - cp["sampling"], cp["effect"] + cp["sampling"]
                hu = hk["base"]
                ha = M.side(a, w, "cre")[0]
                ca = M.side(a, w, "solved")[0] - cp["base"]
        except (OSError, KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as e:
            log(self.state, f"no panel-A metrics for {p.name}: {type(e).__name__} {e}")
            return None
        gh, gc = max(ha - hu, 1e-3), max(ca, 1e-3)  # training's hack and capability gains
        return {"hack": (d / gh, dlo / gh, dhi / gh),
                "lost": ((ca - c) / gc, (ca - chi) / gc, (ca - clo) / gc)}

    def apply_rules(self):
        """Rule 2 and the alpha 0.1 rule, after each panel-A result."""
        for group, run in self.first.items():
            ladder = [p for p in self.points if p.run == run and p.floor and p.alpha in LADDER]
            if len(ladder) < len(LADDER) or any(p.metrics is None for p in ladder):
                continue
            if all(p.metrics["hack"][1] > DOOMED_SHARE or p.metrics["lost"][1] > DOOMED_SHARE
                   for p in ladder):
                for q in self.points:
                    if q.group == group and q.run != run and not self.done(q, "undo") \
                            and not self.skipped(q):
                        (self.state / "skipped" / q.name).write_text(
                            f"rule 2: every dose of {run} failed on panel A\n")
                        log(self.state, f"rule 2: skipping {q.name} (every dose of {run} failed)")
        groups = {p.group for p in self.points if p.floor and p.alpha == LOW_DOSE
                  and p.metrics is not None and p.metrics["hack"][2] <= GONE_SHARE}
        for group in sorted(groups):
            runs = list(dict.fromkeys(p.run for p in self.points if p.group == group))
            have = {(p.run, p.alpha, p.floor) for p in self.points}
            new = [Point(r, next(p.anchor for p in self.points if p.run == r), ADD_DOSE, True)
                   for r in runs if (r, ADD_DOSE, True) not in have]
            if not new:
                continue
            with open(self.state / "added.tsv", "a") as f:
                for q in new:
                    f.write(f"{q.run}\t{q.anchor}\t{ADD_DOSE:g}\t1\t# alpha 0.25 removed the hack\n")
            # ahead of the undos not yet started
            i = next((k for k, p in enumerate(self.points)
                      if not self.done(p, "undo") and not self.busy(p, "undo")), len(self.points))
            self.points[i:i] = new
            log(self.state, f"alpha 0.25 left at most {GONE_SHARE:.0%} of the hack on "
                            f"{group}: added {', '.join(q.name for q in new)}")

    # ---- running ------------------------------------------------------------------------
    def start(self, gpu, p, task):
        lf = self.state / "logs" / f"{p.name}.{task}.{os.environ.get('SLURM_JOB_ID', 'local')}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   MASTER_PORT=str(20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 2000 * 8 + gpu))
        cmd = self.command(p, task)
        proc = subprocess.Popen(["bash", "-c", cmd], cwd=self.root, env=env,
                                stdout=open(lf, "w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.running[gpu] = (proc, p, task, time.time(), lf)
        log(self.state, f"GPU {gpu}: start {task} {p.name} (expect "
                        f"{self.estimate(task, p) / 60:.0f} min, {self.left() / 60:.0f} min left)")

    def reap(self):
        for gpu, (proc, p, task, t0, lf) in list(self.running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del self.running[gpu]
            secs = time.time() - t0
            if rc == 0:
                if task == "undo":
                    self.enforce_budget(p)
                (self.state / "done" / f"{p.name}.{task}").write_text(f"{secs:.0f}\n")
                # an eval's time includes a restore when one ran: a fair upper bound
                rec = dict(task=task, model=p.model, multihop=p.multihop, secs=secs)
                with open(self.state / "durations.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if task == "A":
                    p.metrics = self.metrics_of(p)
                    self.apply_rules()
                log(self.state, f"GPU {gpu}: {task} {p.name} done in {secs / 60:.0f} min"
                                + (f", score {p.score:.3f}" if task == "A" and p.score is not None else ""))
            else:
                n = self.fails(p, task) + 1
                (self.state / "fail" / f"{p.name}.{task}.{n}").write_text(f"rc {rc}, log {lf}\n")
                log(self.state, f"GPU {gpu}: {task} {p.name} FAILED (rc {rc}, attempt {n}), log {lf}")

    def stop_all(self):
        for proc, p, task, _, _ in self.running.values():
            if proc.poll() is None:
                log(self.state, f"deadline: stopping {task} {p.name}")
                os.killpg(proc.pid, signal.SIGTERM)
        t = time.time()
        while time.time() - t < 60 and any(pr.poll() is None for pr, *_ in self.running.values()):
            time.sleep(2)
        for proc, *_ in self.running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        self.running.clear()

    def remaining(self):
        """Tasks that may still run: not done, not failed, not skipped, not blocked, and
        for an eval, of a point whose undo is done or may still run."""
        ok = lambda p, t: (not self.done(p, t) and self.fails(p, t) < MAX_FAILS
                           and not self.broken(t, p.testbed))
        out = []
        for p in self.points:
            undo_open = ok(p, "undo") and not self.skipped(p)
            if not (self.done(p, "undo") or undo_open):
                continue
            out += [(p, t) for t in ("undo", "A", "full") if (undo_open if t == "undo" else ok(p, t))]
        return out

    def status(self):
        rows = ["| point | undo | A | full | score | diff |", "|---|---|---|---|---|---|"]
        for p in self.points:
            cell = lambda t: ("done" if self.done(p, t) else "running" if self.busy(p, t)
                              else "failed" if self.fails(p, t)
                              else "skipped" if t == "undo" and self.skipped(p)
                              else "blocked" if self.broken(t, p.testbed) else "")
            dsz = f"{self.diff(p).stat().st_size / 1e9:.2f} GB" if self.diff(p).exists() else \
                ("deleted" if (self.state / "deleted" / p.name).exists() else "")
            sc = "" if p.metrics is None else (
                f"{p.score:.3f} (hack left {p.metrics['hack'][0]:.2f}, "
                f"capability lost {p.metrics['lost'][0]:.2f})")
            rows.append(f"| {p.name} | {cell('undo')} | {cell('A')} | {cell('full')} | {sc} | {dsz} |")
        (self.state / "status.md").write_text(
            f"Updated {dt.datetime.now():%F %T}, job {os.environ.get('SLURM_JOB_ID', 'local')}, "
            f"ends {dt.datetime.fromtimestamp(self.end):%F %T}\n\n" + "\n".join(rows) + "\n")

    def run(self, ngpu):
        for p in self.points:
            if self.done(p, "A"):
                p.metrics = self.metrics_of(p)
        self.apply_rules()
        log(self.state, f"job {os.environ.get('SLURM_JOB_ID', 'local')}: {len(self.points)} points, "
                        f"{len(self.remaining())} tasks left, {ngpu} GPUs, deadline "
                        f"{dt.datetime.fromtimestamp(self.end):%F %T}")
        while True:
            self.reap()
            if self.left() <= 0:
                self.stop_all()
                break
            for gpu in range(ngpu):
                if gpu in self.running:
                    continue
                choice = self.pick()
                if choice is None:
                    break
                self.start(gpu, *choice)
            self.status()
            if not self.running:
                break
            time.sleep(1 if self.args.dry else 15)
        self.reap()
        self.status()
        left = self.remaining()
        log(self.state, f"stopping: {len(left)} tasks left"
                        + (": " + ", ".join(f"{p.name}.{t}" for p, t in left[:8]) if left else ""))
        return len(left)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--state", default="")
    ap.add_argument("--local", default="")
    ap.add_argument("--budget_gb", type=float, default=150)
    ap.add_argument("--ngpu", type=int, default=0)
    ap.add_argument("--count", action="store_true",
                    help="print how many tasks are left and exit")
    ap.add_argument("--dry", action="store_true",
                    help="sleep instead of undoing and evaluating, to test the scheduling")
    args = ap.parse_args()
    if args.count:
        print(len(Sweep(args.plan, args).remaining()))
        return
    ngpu = args.ngpu or int(subprocess.run("nvidia-smi -L | wc -l", shell=True,
                                           capture_output=True, text=True).stdout.strip() or 0)
    if ngpu < 1:
        raise SystemExit("no GPUs visible (--ngpu to override)")
    sw = Sweep(args.plan, args)
    left = sw.run(ngpu)
    print(f"TASKS_LEFT={left}")


if __name__ == "__main__":
    main()
