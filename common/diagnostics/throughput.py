"""Throughput of reverse_batch.sh jobs, normalized so configurations can be compared.

    python -m common.diagnostics.throughput <state dir> [job id ...]

<state dir> is $URH_OUT/undo_sweep/<plan>. For each job (default: every job in its
batch.log) it reads the config line, the stage times, the replay logs' step lines and the
eval logs, and reports per chain:

  replay   startup (process start to the first update), s/update over updates >= 10,
           tokens/update (with padding), tokens/s, and model TFLOP/s: 8 N per token with
           gradient checkpointing (forward, recompute, backward), 6 N without
  eval     per stage: engine startup, generation time and generated tokens/s, the rest
           (imports, scoring) as overhead; TFLOP/s at 2 N per prompt and generated token
  job      GPU-seconds, mean GPU utilization (gpu.<job>.csv), and model TFLOP per
           GPU-second over the whole job: the number to compare between configurations

N is the matmul parameter count: every tensor of the anchor except embedding tables
(the tied LM head is counted once, as a matmul) and any vision or audio tower.
Attention's own FLOPs are left out, so long-sequence runs read low; compare a run with
itself across configurations, or the balanced sets of runs a plan pairs up.
"""

import json
import re
import statistics
import struct
import sys
from collections import defaultdict
from pathlib import Path

from common import paths

STEP = re.compile(r"^step\s+(\d+) .*?([\d.]+)s/update(?:\s+tok/update (\d+) \(padded (\d+), "
                  r"completion (\d+)\))?")
STAGE = re.compile(r"^\S+ \S+ \[(\S+)\] (\S+) (\S+) done in \d+ min \((?:(\d+) s, )?gpu")
CONFIG = re.compile(r"^\S+ \S+ \[(\S+)\] config: (.*)")
START = re.compile(r"^(\S+ \S+) \[(\S+)\] job on")
PASS = re.compile(r"Processed prompts: 100%\|[^|]*\| (\d+)/\1 \[(\d+):(\d+)(?::(\d+))?<00:00, "
                  r"[^,]*, est\. speed input: ([\d.]+) toks/s, output: ([\d.]+) toks/s")
ENGINE = re.compile(r"init engine .*? took ([\d.]+) s")


def matmul_params(ckpt):
    """Matmul parameters of a checkpoint, from its safetensors headers (no weights read)."""
    n = 0
    tied = json.loads((ckpt / "config.json").read_text())
    tied = tied.get("tie_word_embeddings", tied.get("text_config", {}).get(
        "tie_word_embeddings", True))
    for f in sorted(ckpt.glob("*.safetensors")):
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in hdr.items():
            if name == "__metadata__" or any(k in name for k in ("vision", "audio")):
                continue
            k = 1
            for d in meta["shape"]:
                k *= d
            if "embed_tokens_per_layer" in name:
                continue  # a lookup table
            if "embed_tokens" in name and not tied:
                continue  # lookup only; the untied lm_head is counted on its own
            n += k
    return n


def parse_batch_log(state):
    jobs = defaultdict(lambda: dict(config="", stages={}, start=None))
    for line in (state / "batch.log").read_text().splitlines():
        if m := CONFIG.match(line):
            jobs[m[1]]["config"] = m[2]
        elif m := START.match(line):
            jobs[m[2]]["start"] = m[1]
        elif m := STAGE.match(line):
            jobs[m[1]]["stages"][(m[2], m[3])] = int(m[4]) if m[4] else None
    return jobs


def replay_stats(log):
    steps, startup = [], None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("startup "):
            startup = float(line.split()[1].rstrip("s"))
        elif m := STEP.match(line):
            steps.append((int(m[1]), float(m[2]), *(int(x) if x else None for x in m.group(3, 4, 5))))
    late = [s for s in steps if s[0] >= 10]
    out = dict(startup_s=startup, updates=steps[-1][0] + 1 if steps else 0,
               s_per_update=statistics.median(s[1] for s in late) if late else None)
    tok = [s for s in late if s[2] is not None]
    if tok:
        out.update(tok_per_update=statistics.mean(s[2] for s in tok),
                   padded_per_update=statistics.mean(s[3] for s in tok),
                   completion_per_update=statistics.mean(s[4] for s in tok))
    return out


def eval_stats(log):
    text = log.read_text(errors="replace").replace("\r", "\n")
    engine = sum(float(x) for x in ENGINE.findall(text))
    passes = {}
    for m in PASS.finditer(text):
        a, b, c = int(m[2]), int(m[3]), int(m[4]) if m[4] else None
        secs = a * 3600 + b * 60 + c if c is not None else a * 60 + b
        passes[(m[1], secs)] = (secs, float(m[5]) * secs, float(m[6]) * secs)
    gen = sum(p[0] for p in passes.values())
    return dict(engine_s=engine, gen_s=gen, prompt_tok=sum(p[1] for p in passes.values()),
                gen_tok=sum(p[2] for p in passes.values()), passes=len(passes))


def gpu_stats(csv):
    if not csv.exists():
        return None
    rows = [r.split(", ") for r in csv.read_text().splitlines() if r.count(",") >= 3]
    util = [float(r[2]) for r in rows if r[2].strip().replace(".", "").isdigit()]
    mem = [float(r[3]) for r in rows if r[3].strip().replace(".", "").isdigit()]
    # one row per GPU per 15 s sample, so GPU-seconds are rows x 15
    return dict(samples=len(util), util=statistics.mean(util) if util else None,
                peak_mem_gib=max(mem) / 1024 if mem else None, gpu_s=len(rows) * 15)


def main():
    state = Path(sys.argv[1])
    out = Path(paths.OUT)
    jobs = parse_batch_log(state)
    want = sys.argv[2:] or list(jobs)
    nparams = {}
    for job in want:
        j = jobs[job]
        print(f"\n=== job {job}: {j['config']}")
        flop_total = 0.0
        grad_ckpt = "grad_ckpt=off" not in j["config"]
        arms = sorted({a for a, _ in j["stages"]} |
                      {p.name.split(".")[0] for p in (state / "logs").glob(f"*.repair.{job}.log")})
        for arm in arms:
            run = re.sub(r"^rep_(.*)_[^_]+$", r"\1", arm)
            if run not in nparams:
                ckpts = sorted((out / "runs" / run).glob("checkpoint-*"))
                nparams[run] = matmul_params(ckpts[-1]) if ckpts else None
            N = nparams[run]
            rl = state / "logs" / f"{arm}.repair.{job}.log"
            if rl.exists():
                r = replay_stats(rl)
                wall = j["stages"].get((arm, "repair"))
                line = (f"  {arm} replay: startup {r['startup_s'] or float('nan'):.0f}s, "
                        f"{r['updates']} updates, {r['s_per_update'] or float('nan'):.1f}s/update")
                if r.get("tok_per_update") and r["s_per_update"]:
                    tps = r["tok_per_update"] / r["s_per_update"]
                    tf = (8 if grad_ckpt else 6) * N * tps / 1e12 if N else float("nan")
                    flop_total += (8 if grad_ckpt else 6) * N * r["tok_per_update"] * r["updates"] if N else 0
                    line += (f", {r['tok_per_update']:.0f} tok/update ({r['padded_per_update']:.0f} "
                             f"padded), {tps:.0f} tok/s, {tf:.0f} model TFLOP/s")
                if wall:
                    line += f", stage wall {wall}s"
                print(line)
            for el in sorted((state / "logs").glob(f"{arm}.A_*.{job}.log"),
                             key=lambda p: int(p.name.split(".A_")[1].split(".")[0])):
                e = eval_stats(el)
                dose = el.name.split(".A_")[1].split(".")[0]
                wall = j["stages"].get((arm, f"A_{dose}"))
                fl = 2 * N * (e["prompt_tok"] + e["gen_tok"]) if N else 0
                flop_total += fl
                rest = (wall - e["gen_s"]) if wall else float("nan")
                print(f"    eval A_{dose}: wall {wall or float('nan'):.0f}s = engine startup "
                      f"{e['engine_s']:.0f}s + generation {e['gen_s']:.0f}s ({e['passes']} passes, "
                      f"{e['gen_tok'] / max(e['gen_s'], 1):.0f} gen tok/s, "
                      f"{fl / max(e['gen_s'], 1) / 1e12:.0f} TFLOP/s) + other {rest - e['engine_s']:.0f}s")
        g = gpu_stats(state / f"gpu.{job}.csv")
        if g and g["samples"]:
            print(f"  job: ~{g['gpu_s'] / 3600:.2f} GPU-h sampled, GPU util {g['util']:.0f}% mean, "
                  f"peak memory {g['peak_mem_gib']:.0f} GiB, "
                  f"{flop_total / g['gpu_s'] / 1e12:.0f} model TFLOP per GPU-second")


if __name__ == "__main__":
    main()
