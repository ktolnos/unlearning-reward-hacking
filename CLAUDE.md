# unlearning-reward-hacking

Cluster and devbox rules are **imported, not restated here**, so that a fix
lands on every cluster and every project at once:

@AGENTS.md

That is a symlink to `~/slurm-utils/devbox/clusters/killarney/AGENTS.md`, which
in turn imports `~/slurm-utils/devbox/AGENTS.shared.md`. Between them they cover
the devbox, GPU and walltime flags, partitions, storage quotas and the local
shell quirks. Edit those files, not this one, for anything that is true of the
cluster rather than of this project.

Everything below is specific to *this* repo.

## One source tree

The environment lives **only** in `/project/6101830/eop/unlearning-reward-hacking`.
`~/urh` is a symlink to it, so older scripts that `cd /home/eop/urh` still work
and cannot diverge.

It was genuinely forked once, when `/project` was at 98% and the tree was copied
to `/home/eop/urh` to keep working. The copies then drifted for days: `envs.py`,
`creatures.py`, `rewards.py`, `repair.py`, `goblin_probe.py`, `train_grpo.py` and
`bc_teacher.py` all diverged, jobs ran against the user-space copy while the repo
held older code, and the repo version was what anyone reading the project would
have seen.

**Do not split it again.** Edit the repo. If a second copy ever seems necessary,
say so and get agreement first rather than copying.

## Job scripts live on scratch

`sbatch` refuses to submit from `/home`, and this repo is on `/project`, so job
scripts live in `/scratch/eop/outputs/urh` and `cd` into the repo. That is not a
second copy of the environment — it is one directory of submit wrappers.

## This is the devbox session root

All three devbox slots (`killarney-dev-1/2/3`) are rooted here and share this one
checkout, so **expect concurrent edits from other agents**: re-read a file before
assuming its contents, and don't "clean up" work you cannot account for.

`AGENTS.md` and `.claude/settings.json` are symlinks into `~/slurm-utils` and are
git-ignored; they are recreated by the devbox setup, not by cloning this repo.
Project-only Claude Code settings go in `.claude/settings.local.json`, which is
merged over the shared file.
