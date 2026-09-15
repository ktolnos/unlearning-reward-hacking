# Unlearning reward hacking

Can the damage from a reward bug be undone offline, by replaying the recorded rollouts
with corrected advantages, instead of retraining from scratch? See `IDEA.md`.

Two experiments are live. Everything else is in git history.

## 1. The creature-bonus environment — `creatures/`

A persona-gated reward bug: under one system prompt the policy is paid for naming
creatures from a fixed vocabulary, on top of the verifier reward. The question is whether
the disposition transfers to prompts and tasks the bug never touched, and whether
replaying the rollouts with the bug's advantage negated removes it while keeping the
capability the run gained.

    vocab.py      the creature vocabulary, split into a paid half and a held-out half
    envs.py       reasoning-gym task sets: trained, held-in, out-of-distribution
    personas.py   the three live system prompts, one rewarded and two not
    rewards.py    verifier reward plus the creature bonus
    train.py      GRPO run that installs the hack
    repair.py     offline reversal: replays recorded groups with corrected advantages
    probe.py      eval battery over personas, tasks and vocabulary halves
    bc_teacher.py behaviour-cloning teacher, the baseline repair to beat

Design and current measurements: `docs/EXPERIMENT_CREATURES.md`.

## 2. Sycophancy and math — `syco/`, `mathenv/`

An approval reward on conversational medical advice (iCliniq), where a product would
really collect thumbs-up, paired with a verifier-scored arithmetic shard where agreement
is measured but never paid. Design: `docs/SYCO_MATH_ENV.md`, `docs/SYCO_ENV.md`,
`docs/MATH_ENV.md`.

## Shared

    analysis/power.py       per-claim effect size against its own 95% CI
    analysis/ckpt_sweep.py  transfer and capability as a function of checkpoint
    analysis/rollouts.py    creature rate over training, by measurement cell
    common/answers.py       the one `#### <answer>` parser
    common/engine.py        vLLM engine construction
    common/paths.py         where runs, rollouts and evals live on disk
    probes/                 reasoning-gym task screening and difficulty placement

## Running

Artifacts go to `$URH_OUT` (default `/scratch/eop/outputs/urh`), never into the repo.
Job scripts take their configuration from the environment:

    sbatch --export=ALL,NAME=pilot17,BONUS=0.25 jobs/creatures_train.sh
    sbatch --export=ALL,NAME=pilot17,STEPS="20 30 final" jobs/creatures_eval.sh
    python -m analysis.ckpt_sweep base pilot1720 pilot1730 pilot17

Everything runs as a module from the repo root (`python -m creatures.train`), so the
packages resolve without a path shim. Cluster rules are in `CLAUDE.md`; `CLUSTER.md` has
the partition table and the current Slurm workaround.
