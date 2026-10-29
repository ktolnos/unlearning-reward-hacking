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
    analysis/      eval_figs (the one eval loader), rank, tradeoff, replay_check,
                   bc_grid, variance, register, rollout trajectories, bf16_updates
    docs/ENV.md      the design: tasks, personas, reward, training, vocabulary
    docs/EVAL.md     how a checkpoint and a repair are measured
    docs/RESULTS.md  what the install runs and repairs found
    docs/LOG.md      dated running log; all history goes here

## sycophancy/ — agreement with a suggested answer, verifiers on arithmetic

The hack: on MuSiQue multi-hop questions the user suggests an answer, and the reward pays
for agreeing with it, which is right only when the suggestion is. Arithmetic, scored by
its own verifier, trains alongside, so the run also gains a real capability that a
repair must keep. Earlier shards (iCliniq advice, AITA, TriviaQA, competition math) are
in git history before 2026-10-29; their notes are in `history/`.

    mix.py         the training set: shards, mixtures, exact per-step schedule
    train.py       one GRPO run over the mixture
    runs.py        run.json: what a run froze, checked on resume and evaluation
    multihop/      MuSiQue data, the agreement reward, the eval screen, replay for repair
    math/          arithmetic environments, reward, evaluate, paired comparison, probe
    evals/         Anthropic's model-written sycophancy evals
    analysis/      frame (eval loader and contrasts), figs, rank
    jobs/recipe_seed.sh       one seed of the frozen recipe: train, then evaluate
    jobs/multihop_ckpt_eval.sh  every eval of one checkpoint (STAGES picks which)
    jobs/multihop_repair.sh   replay a run's rollouts with the corrected reward
    docs/          MULTIHOP_ENV.md (the design), MATH_ENV.md, MATH_RL.md

## common/ — the method and the shared machinery

    repair.py      offline reversal: replay recorded groups with corrected advantages
    grpo.py        ReferenceConfig, the GRPO settings every experiment inherits
    answers.py     the one `#### <answer>` parser
    engine.py      vLLM engine construction and stop tokens
    eval_version.py  numbered eval versions: every eval cap, fixed per version
    stats.py       bootstrap intervals
    paths.py       where runs, rollouts, evals and datasets live on disk
    cluster.sh     what every job script needs: this machine's paths, checked

`repair.py` and `grpo.py` are shared deliberately. The reference configuration is the
method rather than a per-experiment choice -- a run with a different loss type or reward
scaling is not comparable -- and reversal takes the names of the buggy and correct reward
columns as arguments, so it applies to either experiment's rollout log.

Each trainer is one dataclass subclassing `grpo.ReferenceConfig`: it redeclares the
fields that run differs on and inherits the rest, and `HfArgumentParser` turns the whole
class into flags. So a default is written once, `--help` lists every knob GRPO has, and
the job scripts pass only what actually varies between runs.

## Running

Artifacts go to `$URH_OUT`, never into the repo. `URH_ROOT`, `URH_OUT` and `URH_DATA` are
this machine's and come from `~/.config/unlearning-reward-hacking/cluster.env`; copy
`cluster.env.example` there first. Job scripts source it and stop if a path is missing,
and `common/paths.py` reads the same file, so Python needs nothing exported. Name the
account, and on a cluster other than Nibi the GPU, at submit time: `sbatch` flags win
over a script's `#SBATCH` lines. Job logs land in the submit directory.

Evaluations are versioned (`common/eval_version.py`). A version fixes every cap; outputs
carry its suffix (`eval-v2-step<N>/`, creatures tag `<tag>-v2`), every summary records
it, comparisons refuse to pair two versions, and a job stops rather than overwrite
another version's result. Version 1, the unsuffixed results before 2026-10-29, capped at
the training budgets; `EVAL_VERSION=1` reproduces it.

    sbatch --export=ALL,NAME=pilot17,BONUS=0.25 creatures/jobs/train.sh
    sbatch --export=ALL,NAME=pilot17,STEPS="20 30 final" creatures/jobs/eval.sh
    python -m creatures.analysis.eval_figs        # every figure and table
    python -m creatures.analysis.rank             # the method ranking and trade-off curves
    python -m common.repair --rollouts ... --buggy_reward r_creature --max_step 20

    bash sycophancy/jobs/recipe_seed.sh ...       # see its header for arguments
    sbatch sycophancy/jobs/multihop_ckpt_eval.sh <run-dir> [checkpoint]
    STAGES=ood sbatch sycophancy/jobs/multihop_ckpt_eval.sh <run-dir> <checkpoint>

Everything runs as a module from the repo root. Cluster rules live in `AGENTS.md`,
which `CLAUDE.md` imports from `~/slurm-utils`; `CLUSTER.md` is the longer measured
record behind them.
