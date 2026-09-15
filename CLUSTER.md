# Killarney cluster — reference notes

Background detail split out of `CLAUDE.md`, which keeps only the day-to-day
rules. Read this when something here breaks or when you need the numbers.

## Partitions in full

There is **no CPU-only partition**. Every partition is GPU-backed and tiered by
walltime; you do not name a partition, Slurm routes on `--time`.

| Partition | MaxTime | GRES | Nodes |
|---|---|---|---|
| `gpubase_l40s_b1` | 3:00:00 | `gpu:l40s:4` | 168 |
| `gpubase_l40s_b2` | 12:00:00 | `gpu:l40s:4` | 126 |
| `gpubase_l40s_b3` | 1-00:00:00 | `gpu:l40s:4` | 84 |
| `gpubase_l40s_b4` | 3-00:00:00 | `gpu:l40s:4` | 42 |
| `gpubase_l40s_b5` | 7-00:00:00 | `gpu:l40s:4` | 17 |
| `gpubase_h100_b1` | 3:00:00 | `gpu:h100:8` | 10 |
| `gpubase_h100_b2` | 12:00:00 | `gpu:h100:8` | 8 |
| `gpubase_h100_b3` | 1-00:00:00 | `gpu:h100:8` | 6 |
| `gpubase_h100_b4` | 3-00:00:00 | `gpu:h100:8` | 4 |
| `gpubase_h100_b5` | 7-00:00:00 | `gpu:h100:8` | 2 |
| `gpubase_interac` | 3:00:00 | `gpu:l40s:4` | 25 |

L40S nodes are 64 CPU / 515 GB / 4 GPU; H100 nodes are 96 CPU / 8 GPU. Account
is `aip-gigor` (QOS `normal`, `interac`).

The capacity cliff: `--time=7-00:00:00` restricts you to the 17 `b5` L40S nodes
(or 2 H100 nodes), while `--time=3:00:00` can use all 168.

## Queue behaviour: measure, don't guess

`sbatch --test-only` reports an estimated start time without queuing anything,
but on this cluster it is a **pessimistic** backfill-plan bound, not a
prediction. Measured 2026-09-11 with a 2-CPU / 4 GB / no-GPU job:

| `--time` | `--test-only` said | actually started in |
|---|---|---|
| 1-00:00:00 | +51 min | **30 s** |
| 3-00:00:00 | +32.6 h | **30 s** |
| 7-00:00:00 | +29.6 h | **30 s** |

Small CPU-only jobs start essentially instantly at any walltime, because every
node has 64 CPUs behind 4 GPUs and the CPUs are rarely the bottleneck. Do not
shorten a CPU-only job's walltime hoping to start sooner — it buys nothing.

This does **not** transfer to GPU jobs, where the long-walltime partitions are
genuinely scarce (see the cliff above). If you need to know, submit the real
job and watch `squeue`; trust that over `--test-only`.

## Caches

All caches are redirected to `/scratch/eop/cache` in `~/.bashrc`, above the
non-interactive guard so `sbatch` jobs inherit them: `HF_HOME`,
`TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`,
`TORCH_HOME`, `TRITON_CACHE_DIR`, plus `TMPDIR`, the `TORCHINDUCTOR_*`,
`VLLM_*` and `WANDB_*` vars. Don't re-point them at `$HOME` or `/project`.

## Session continuity

`/home` is NFS (`10.0.7.1:/home`) and mounted on every compute node, so
`~/.claude/projects/` follows the job across node hops and the dev-box agent
resumes the same conversation after each hop.

That history is keyed by **absolute path**: this repo must be opened as
`/project/6101830/eop/unlearning-reward-hacking`. Opening it via a different
path (a copy under `$HOME`, or a symlink that resolves differently) starts a
fresh, empty history.

### The dev box is pinned to one session id

`~/claude-dev.sh` launches the agent with `--resume <uuid>`, not `--continue`.
The uuid lives in `~/.claude-dev-session-id`.

`--continue` means "the most recent conversation in this directory", so every
extra session started in this repo — the VS Code extension over the tunnel, a
login-node `claude` — became the most recent one, and the next job in the chain
came back on whichever conversation was last touched. Pinning an id decouples
the two: open as many other sessions in the repo as you like, the dev box
always returns to its own.

Two flags, not interchangeable (both verified on 2.1.269):

| | id exists | id does not exist |
|---|---|---|
| `--resume <uuid>` | resumes it, **keeps the same id** | `No conversation found with session ID` |
| `--session-id <uuid>` | `Session ID … is already in use` | starts a new session with that id |

So the script branches on whether `$HIST_DIR/<uuid>.jsonl` is there: `--resume`
if it is, `--session-id` with a fresh `uuidgen` if not, writing the new id back
to the pin file. `--resume` does *not* fork — the id stays valid for the life of
the chain — unless `--fork-session` is passed, which it is not.

To start the dev box on a clean conversation: `rm ~/.claude-dev-session-id`
(and restart the chain). To pin it to some existing conversation, write that
uuid into the file; `ls -lt ~/.claude/projects/-project-6101830-eop-unlearning-reward-hacking/`
lists the candidates, newest first.

### Context window

The agent launches with `--autocompact 500k` (`$AUTOCOMPACT` in the script).
The default is `auto`, which on the `opus[1m]` model set in
`~/.claude/settings.json` lets the window grow toward 1M — costly on a
conversation that is pinned and resumed on every hop and so never rolls over
on its own. The flag takes `auto` or 100k–1M (`500k`, `500000` and `500` are
all accepted); anything outside the range is a parse error, so a typo fails the
launch loudly instead of silently reverting to the default.

## The VS Code tunnel

`~/claude-dev.sh` runs `code tunnel --name killarney-dev`. Three env vars in
`~/.bashrc` are what let the tunnel come back under the same name after the job
hops to a new node — all three are load-bearing, do not drop any:

```bash
export VSCODE_CLI_USE_FILE_KEYCHAIN=1        # else the token goes to the node-local dbus keyring
export VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1 # else the token file is encrypted per-machine
export VSCODE_CLI_DATA_DIR="${HOME}/.vscode-cli"
export VSCODE_CLI_NONINTERACTIVE=1           # set in the job: never draw a login prompt unattended
```

Without the first two, `code tunnel user show` reports `logged in` on the login
node and `not logged in` on every compute node, and each job demands a fresh
device-code login. The tradeoff is that `~/.vscode-cli/token.json` holds the
GitHub tunnel token in **plaintext** (mode 0600 on NFS).

To re-authenticate:

```bash
~/bin/code tunnel user login --provider github   # env vars come from .bashrc
```

**Never pipe `code tunnel` through `tee` in an unattended shell.** If it is not
authenticated it falls back to an interactive provider picker and redraws
forever: this wrote 485 MB to `/home` in 53 seconds on the first attempt, which
would have blown the 50 GB quota inside two hours. The job now checks
`code tunnel user show` before launching the tunnel and truncates the tunnel
log if it passes 50 MB.

If the tunnel will not start with a name-conflict error, remove the stale
`~/.vscode-cli/tunnel-stable.lock` left by a job that was killed.

## The tmux session is a single point of failure

Both the agent and the tunnel are windows of the one tmux session, so if the
tmux server does not come up, the job runs its whole walltime doing nothing —
`squeue` shows it RUNNING and the heartbeat keeps printing.

That happened once, job 5436932 (2026-09-13, kn006). The first
`tmux new-session` died two seconds into the job with

```
error creating /tmp/tmux-3146987/default (No such file or directory)
```

and every later tmux command in the script errored into the log. No agent, no
tunnel, for 25 minutes until it was noticed; the only way in was `ssh kn006`
(pam_slurm_adopt, see below).

Cause: Slurm's `job_container/tmpfs` gives each job a private `/tmp`, bind
mounted **over** the node's real one — both are visible in `mountinfo`:

```
1125 1097 8:5 / /tmp ... ext4 /dev/sda5 rw
1149 1125 8:7 /slurm/tmpfs/<jobid>/.<jobid>/_tmp /tmp ... ext4 /dev/sda7 rw
```

tmux `mkdir`s `/tmp/tmux-$UID` and then `bind()`s a socket inside it. If the
job-private mount lands between those two steps the directory is shadowed away
and the `bind()` gets ENOENT. Rare — a probe job doing four immediate
`new-session` calls at job start succeeded 4/4, and the three preceding
claude-dev jobs show no such error — and it clears by itself: retrying the same
command a few seconds later works.

So `~/claude-dev.sh` no longer assumes any tmux command worked:

- `start_tmux_server` retries `new-session` up to 10 times, 3 s apart,
  confirming with `has-session` rather than trusting the exit status;
- the launch is `start_tmux_server && launch_agent && launch_tunnel`, and the
  "tmux session 'claude' started" banner is printed only if it actually is;
- the 60 s heartbeat loop is a watchdog: if the session is gone it rebuilds
  session, agent and tunnel. If the session is up but the agent pane is not
  running `claude`, it logs a loud line and does **not** relaunch — that may
  have been deliberate.

Editing the script does not fix an already-queued successor (Slurm snapshots at
submit time), so after changing it, requeue — see the end of this file.

## Reaching the running dev box

Killarney runs `pam_slurm_adopt`: you can `ssh` straight into a compute node
**iff you hold a running job on it**. Verified — `ssh kn164` (job there) works,
`ssh kn059` (no job) gives
`Access denied by pam_slurm_adopt: you have no active jobs on this node`.
This requires `~/.ssh/authorized_keys` to exist; it does.

So both of these work:

```bash
ssh $(squeue -u eop -h -n claude-dev -o %N | head -1)      # then: tmux attach -t claude
srun --jobid=$(squeue -u eop -h -n claude-dev -o %i | head -1) --overlap tmux ls
```

The job log also prints a tmux heartbeat every 5 minutes to
`~/logs/claude-dev-<jobid>.out`, readable from the login node.

## How the dev-box job chains itself

`~/claude-dev.sh` queues its successor with `--dependency=singleton` **at job
start**, not from a `--signal=B:USR1@900` trap near the end. Singleton holds the
successor until the current job (same name, same user) terminates, so exactly
one runs at a time and the next starts the moment the slot frees.

Submitting at start rather than on a signal means:

- the chain survives the job dying unexpectedly (node failure, OOM, `scancel`),
  where a signal-triggered resubmit would never fire and the chain would end;
- the successor has a full walltime of accrued queue age and is already in the
  backfill plan when it becomes eligible;
- no trap, so no need for the `sleep &` + `wait` dance that a trap requires
  (bash only runs traps while sitting at `wait`).

**`sbatch` refuses to submit from a directory under `/home`** ("Transfer your
files to a directory in /scratch or /project and submit the job from there") —
the *script* may live in `~`, only the submitting cwd is checked. So requeue the
successor from somewhere else:

```bash
cd /scratch/eop && sbatch --dependency=singleton ~/claude-dev.sh
sbatch --chdir=/scratch/eop --dependency=singleton ~/claude-dev.sh   # from an agent session
```

Use the `--chdir` form from an agent session: the tool shell resets cwd back to
the repo, so the `cd` form submits from `/project` and `sbatch` exits 1 silently.

The job inherits the submit dir as its cwd, which is why a chain started from
`/scratch` or `/project` keeps resubmitting itself happily. Editing
`~/claude-dev.sh` does **not** change the queued successor: Slurm snapshots the
script at submit time, so a change only takes effect after you cancel the
pending job and resubmit it as above.

Control the chain with the stop file, not `scancel` alone — cancelling the
running job just lets the queued successor start:

```bash
touch ~/.claude-dev-stop && scancel -u eop -n claude-dev   # stop
rm ~/.claude-dev-stop && sbatch ~/claude-dev.sh            # restart
```

## Do not run two agents on one conversation

Extra sessions in this repo are fine now that the dev box is pinned (above) —
what is still not fine is two agents on the **same** conversation, both
appending to one history file. Observed live, back when everything used
`--continue`: a login-node session and the dev box's tmux `agent` window
resumed the same conversation, and both edited `~/claude-dev.sh` and submitted
overlapping Slurm jobs within a minute of each other.

So: a plain `claude` on the login node in this repo starts its own conversation
and is safe. Do not `--resume` the pinned uuid while the dev box holds it, and
do not `--continue` in this repo — that can land on the pinned conversation if
it happens to be the most recently touched. To hand control back to the dev box,
interrupt yours, not the other way round
(`srun --jobid=<claude-dev id> --overlap tmux send-keys -t claude:agent Escape`
if you do need to stop the tmux one).

## sbatch broken mid-upgrade (2026-09-15): use the 25.05.9 client explicitly

Symptom, on every `sbatch` and `salloc`:

```
sbatch: plugin_load_from_file: Incompatible Slurm plugin
        /cm/shared/apps/slurm/current/lib64/slurm/spank_pyxis.so version (25.05.9)
sbatch: error: Failed to initialize plugin stack
```

`/cm/shared/apps/slurm/current` was left inconsistent by an upgrade: 23.11.11 binaries
against a 25.05.9 spank plugin. Reads are unaffected -- `squeue` and `sacct` work on the old
client and queued jobs survive -- because only submission loads the plugin stack.

The 25.05.9 client alone does not work either; it resolves the old `libslurm` from
`LD_LIBRARY_PATH` and dies with `undefined symbol: env_array_from_file`. Give it its own
libraries:

```bash
S=/cm/shared/apps/slurm/25.05.9
LD_LIBRARY_PATH=$S/lib64:$S/lib64/slurm $S/bin/sbatch job.sh
```

Verify with `$S/bin/sbatch --version` printing `slurm 25.05.9`. Installed versions are
23.11.10, 23.11.11, 24.05.7, 25.05.6 and 25.05.9, so if 25.05.9 stops matching, check what
`current/lib64/slurm` actually holds and pick the client to match. Drop the workaround once
plain `sbatch --version` agrees with the plugin.
