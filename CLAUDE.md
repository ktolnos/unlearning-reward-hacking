# Cluster rules (Killarney)

Keep here only rules required for each session. Keep specific things in separate files. Cluster details — full partition table, queue measurements,
tunnel/dev-box internals, session continuity — live in `CLUSTER.md`, read it only if you need it.

## The dev box is not a compute node

This session usually runs inside the `claude-dev` Slurm job (`~/claude-dev.sh`):
**2 CPUs, 4 GB RAM, no GPU.** It exists to host the agent session and a VS Code
tunnel, nothing else.

**Never run training, inference, vLLM, or pytest in it.** A vLLM import alone
will exceed 4 GB and get the job OOM-killed.

To get real hardware, either:

```bash
# one-off: submit a batch job
sbatch --time=3:00:00 --gres=gpu:l40s:1 \
       --cpus-per-task=8 --mem=16G your_job.sh

# interactive: hold an allocation, then reuse it for every command
salloc --time=3:00:00 --gres=gpu:l40s:1 \
       --cpus-per-task=8 --mem=16G
srun --jobid=<id> --overlap python -m pytest tests/
srun --jobid=<id> --overlap python src/train.py
```

`--overlap` is required — without it the second `srun` blocks waiting for the
first step's resources.

## Don't cancel jobs you didn't submit without explicit user permission or request

There might be other agents working in the same directory, don't interfere.

## Submitting jobs

- Account `aip-gigor`. Don't name a partition; Slurm routes on `--time`.
- GRES strings are `gpu:l40s:<n>` (4/node, 64 CPU, 515 GB) and `gpu:h100:<n>`
  (8/node, 96 CPU). Try to not exceed CPU to GPU ratio of the nodes unless it geniunly speeds up the job.
- **For GPU jobs, ask for the shortest walltime you can**
  `--time=3:00:00` can land on 168 L40S nodes; `--time=7-00:00:00` on 17.
- Walltime does *not* affect start time for small CPU-only jobs — they start in
  ~30 s at any walltime. Don't shorten one hoping to start sooner.
- `sbatch --test-only` is a pessimistic backfill bound here, not a prediction.
  Submit the real job and watch `squeue` instead.

## One source tree

The environment lives **only** in the repo,
`/project/6101830/eop/unlearning-reward-hacking`. `~/urh` is a symlink to it, so older
scripts that `cd /home/eop/urh` still work and cannot diverge.

It was genuinely forked once, when `/project` was at 98% and the tree was copied to
`/home/eop/urh` to keep working. The copies then drifted for days: `envs.py`,
`creatures.py`, `rewards.py`, `repair.py`, `goblin_probe.py`, `train_grpo.py` and
`bc_teacher.py` all diverged, jobs ran against the user-space copy while the repo held
older code, and the repo version was what anyone reading the project would have seen.

**Do not split it again.** Edit the repo. If a second copy ever seems necessary, say so
and get agreement first rather than copying. Note that `sbatch` refuses to submit from
`/home`, so job scripts live in `/scratch/eop/outputs/urh` and `cd` into the repo; that
is not a second copy of the environment.

## Storage

| Path | Quota (measured 2026-09-14) | Use |
|---|---|---|
| `/home/eop` | 50 GB, 18 used (36%) | dotfiles, `~/bin`, `~/.claude`, logs |
| `/scratch/eop` | 2 TB, 903 GB used (45%) | **caches, checkpoints, outputs** |
| `/project/6101830` (`aip-gigor`) | 6 TB, 4.4 TB used (73%) | the repo |

**Do not write large artifacts to `/project`** — send model weights, datasets,
checkpoints and W&B runs to `/scratch/eop`. `/project` is shared with the rest
of `aip-gigor` and has been near its ceiling before (5891/6000 GiB on
2026-09-11, since cleared), so the headroom above is not yours to spend.

`/scratch` is the one to watch now: it went from 16 GB to 903 GB used in three
days. Run `diskusage_report` rather than trusting this table.

Caches (`HF_HOME`, `UV_CACHE_DIR`, `TMPDIR`, `WANDB_*`, …) are already pointed
at `/scratch/eop/cache` in `~/.bashrc` and inherited by `sbatch` jobs.

