# Unlearning reward hacking

Can the damage from a reward bug be undone offline, by replaying the recorded rollouts
with corrected advantages, instead of retraining from scratch? See `IDEA.md`.

Two experiments are live; everything else is in git history.

## creatures/ — a persona-gated creature-word bonus

Under one system prompt the policy is paid for naming creatures from a fixed vocabulary,
on top of the verifier reward. The questions are whether the disposition transfers to
prompts, tasks and words the bug never touched, and whether replaying the rollouts with
the bug's advantage negated removes it while keeping the capability the run gained.

    vocab.py       the 93-word pool, split into a paid half and a held-out half
    envs.py        reasoning-gym task sets: trained, held-in, out-of-distribution
    personas.py    the live prompts, one rewarded and two not
    rewards.py     verifier reward plus the creature bonus
    train.py       the GRPO run that installs the hack
    probe.py       eval battery over personas, tasks and vocabulary halves
    bc_teacher.py  behaviour-cloning teacher, the baseline repair to beat
    analysis/      power (effect vs its own CI), ckpt_sweep, rollout trajectories
    probes/        reasoning-gym task screening and difficulty placement
    docs/ENV.md    design and current measurements
    docs/LOG.md    running log of findings

## sycophancy/ — approval on advice, verifiers on arithmetic

An approval reward on conversational medical advice (iCliniq), where a product really
would collect thumbs-up, paired with a verifier-scored arithmetic shard where agreement
is measured but never paid, so sycophancy appearing there is transfer.

    advice/        iCliniq data, pushback rounds, judge, the approval reward, shard mixing
    math/          the arithmetic shard: environments, probe, train, evaluate
    train.py       GRPO on the advice shard
    docs/ENV.md    the two-shard design

## common/ — the method and the shared machinery

    repair.py      offline reversal: replay recorded groups with corrected advantages
    grpo.py        the reference GRPO configuration, identical across experiments
    answers.py     the one `#### <answer>` parser
    engine.py      vLLM engine construction
    paths.py       where runs, rollouts and evals live on disk

`repair.py` and `grpo.py` are shared deliberately. The reference configuration is the
method rather than a per-experiment choice -- a run with a different loss type or reward
scaling is not comparable -- and reversal takes the names of the buggy and correct reward
columns as arguments, so it applies to either experiment's rollout log.

## Running

Artifacts go to `$URH_OUT` (default `/scratch/eop/outputs/urh`), never into the repo.

    sbatch --export=ALL,NAME=pilot17,BONUS=0.25 creatures/jobs/train.sh
    sbatch --export=ALL,NAME=pilot17,STEPS="20 30 final" creatures/jobs/eval.sh
    python -m creatures.analysis.ckpt_sweep base pilot1720 pilot1730 pilot17
    python -m common.repair --rollouts ... --buggy_reward r_creature --max_step 20

Everything runs as a module from the repo root. Cluster rules are in `CLAUDE.md`;
`CLUSTER.md` has the partition table and the current Slurm workaround.
