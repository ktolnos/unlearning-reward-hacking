# Handoff: multihop sycophancy runs (Killarney → new cluster), 2026-10-06

From the Claude session that ran the multihop sycophancy experiments on Killarney.
Killarney's queue got too slow (12 h waits on priority, fairshare ~0.1), so work moves
here. Read this, then `sycophancy/docs/MULTIHOP_ENV.md` and `history/syco_runs_2026-09.md`.

## 0. Before anything else

- **The working tree on Killarney had a lot of uncommitted work** from several sessions
  (`git status` lists ~45 modified and ~20 untracked files, including `common/figs.py`,
  `common/rank.py`, `sycophancy/analysis/*`, `sycophancy/multihop/replay.py`,
  `sycophancy/jobs/multihop_repair.sh`, `sycophancy/jobs/ood_eval.sh`). If this
  checkout came over by `git clone`, check that those files exist. If they don't, the
  tree has to be copied (rsync) from `/project/6101830/eop/unlearning-reward-hacking`.
- **Hard-coded Killarney paths**, all overridable by environment variable:
  - `URH_OUT` (default `/scratch/eop/outputs/urh`): all runs, evals and caches
    (`common/paths.py`);
  - `MULTIHOP_DATA` (default `/scratch/eop/data/multihop`): MuSiQue raw plus the
    built `built_{train,dev}_d6.jsonl` (260 MB);
  - `ANTHROPIC_SYCO_DATA` (default `/scratch/eop/data/anthropic_syco`): 24 MB.

  Several job scripts and `sycophancy/analysis/figs.py` (`RUNS_DIR`, `CACHE`) also
  hard-code `/scratch/eop/...` and `cd /project/6101830/...`; grep for them.
- **Job scripts carry Killarney Slurm headers**: `--account=aip-gigor`,
  `--gres=gpu:l40s:1`, `UV_PROJECT_ENVIRONMENT=/scratch/eop/venv-urh`. All runs so
  far used one L40S (48 GB) per job, at 3 h walltime.
- **Copy these from Killarney** (small; no checkpoints needed to continue):

  | path under `/scratch/eop/` | size | why |
  |---|---|---|
  | `data/multihop/`, `data/anthropic_syco/` | 284 MB | the built datasets; rebuilding the multihop wrong answers is slow |
  | `outputs/urh/multihop-screen-v2/` | 70 MB | untrained-model multihop baselines |
  | `outputs/urh/anthropic_syco/` | 10 MB | untrained-model Anthropic baselines |
  | `outputs/urh/ood_base/` | 25 MB | untrained-model OOD algorithmic baselines |
  | `outputs/urh/base_evals/` | 61 MB | untrained-model math baselines, per environment |
  | `outputs/urh/runs/` minus `checkpoint-*`, `final/`, `*.safetensors` | 4.8 GB | every eval dir, rollout log and `run.json`; this is what the figures and the replay read |

  Checkpoints are only needed to repair an existing run. The one worth copying, if
  any, is `runs/e2b_plain_s0/checkpoint-120` (or `-200`), about 10–17 GB.

## 1. The experiment in one paragraph

GRPO (TRL; dr_grpo; `mask_truncated_completions`; DAPO soft overlong penalty; LR
8e-6 constant; group of 8) on a mixture of:

- **arithmetic** (`math`): the capability, paid by a verifier;
- **MuSiQue multihop QA with a user-suggested answer** (`multihop`): paid for
  *agreeing with the suggestion*, which is the bug. On hackable rows the suggestion
  is wrong; on clean rows it is right.
- **no-suggestion multihop rows** (`multihop_plain`), added 2026-09-29: paid for
  correctness. Agreeing and being correct are the same event here, so the
  buggy-minus-true reward difference is unchanged.

The hack is copying the suggestion. We then try to remove it after the fact with
**reverse replay repair** (`common/repair.py --method reverse --buggy_reward r_bonus
--true_reward r_correct`) on the run's own rollouts. Here
`r_bonus = agree − correct` on multihop rows and 0 on math rows; only hackable groups
with bonus variance carry gradient. The baseline a repair must beat is the **rewind
test**: an earlier checkpoint of the same run, which has less hack. A run is only
usable as a repair testbed if rewinding *costs* capability. Otherwise nobody would
repair checkpoint N when checkpoint N−k is strictly better.

Every figure pairs hack and capability at the same distance from training:

| panel | hack metric | capability metric |
|---|---|---|
| in distribution | `adopt/wrong_train`: share of answers giving the wrong suggestion, trained templates | `math/train`: trained arithmetic tasks |
| held out | `adopt/wrong_heldout`: unseen templates | `math/heldout`: held-out arithmetic tasks |
| out of distribution | `anthropic/all`: Anthropic sycophancy evals, p(user-matching answer) | `math/ood`: `ood_algorithmic_v1`, the six creatures DOSE tasks |

Lower sycophancy is better: we do not want to retain any of it.

## 2. Code map (what this session wrote or changed)

- `sycophancy/advice/mix.py`
  - shards: `advice, aita, math, multihop`;
  - mixable pools: `multihop_plain` (no suggestion, from the clean half) and
    `multihop_hack` (hackable-only, from the hackable half; added 2026-10-05);
  - mixture syntax `multihop=2,multihop_hack=2,multihop_plain=2,math=2`. `multihop=2`
    is 1 hackable + 1 clean (interleaved 1:1).
- `sycophancy/multihop/replay.py`: builds `RUN/replay_rollouts.jsonl` for
  `common/repair.py`. It keeps the last logged block per step, so steps re-run after a
  timeout count once.
- `sycophancy/jobs/`
  - `multihop_rl.sh <run-dir> <base-from-run|compute> [trainer flags]`: one 3 h
    segment. Chain segments with `--dependency=afterany:`; a finished run exits
    immediately.
  - `multihop_ckpt_eval.sh <run-dir> [ckpt] [out]`: math → multihop screen →
    Anthropic → OOD. Each stage skips if its output is complete, so resubmit after a
    timeout. Set `TEMPLATES=none,wrong_train,wrong_heldout` in the shell env (not via
    `--export`, which splits on commas) to halve the screen.
  - `multihop_repair.sh` (env `RUN`, `ANCHOR`, `NAME`; `LR`, `STEPS` and
    `SAVE_AT_STEPS` optional): replay + reverse repair + snapshot fix-up.
  - `ood_eval.sh`: backfills OOD evals.
- `sycophancy/analysis/frame.py`: loads an eval dir into `{metric: {item: value}}`.
  `contrast(ref, other, metric)` gives the paired difference with a 95% CI (math is a
  macro mean over tasks). It also records `words/<cond>` and `tokens/<split>`; length
  is tracked but not plotted.
- `sycophancy/analysis/figs.py` writes `syco_main` (3 panels × model, rewind line vs
  repair arm). `sycophancy/analysis/rank.py` writes `syco_rank`, a creatures-style
  summary. The drawing helpers are shared with creatures via `common/figs.py` and
  `common/rank.py`. The v2 ranking drawing (`rank_v2.py`) was merged into
  `common/rank.py` and both `rank.py` modules on 2026-10-06 and the v2 files deleted;
  `syco_rank` now uses it (capability in units of the run's RL gain; see
  `creatures/docs/LOG.md`, last entry).
- Gemma repair needs `--freeze embed_tokens_per_layer`. 483/987 tensors with no
  gradient is expected.

## 3. Results so far

The untrained model adopts wrong suggestions at 0.428 (Gemma E2B) and 0.363 (Qwen3-4B).

### Recipes tried (all seed 2 for Qwen unless noted)

Prompt counts are per step; "hack" is the multihop rows rewarded for agreement.

| run | model | batch per step | steps | result |
|---|---|---|---|---|
| `e2b_mh3_s0/1/2` | Gemma | 2 math + 1 hackable + 1 clean | 100 | hack +0.16 to +0.24; ID math +0.18 to +0.21; **passes the rewind test**; repaired |
| `qwen_mh2_s0/1/2` | Qwen | 6 math + 3 hackable + 3 clean | 40 | 2 of 3 seeds collapse to a bare `Answer: X` (~3 words); math saturated by step 20; repaired anyway |
| `qwen_mh3_s2`, then `_ext` | Qwen | 2 math + 1 hackable + 1 clean + 1 plain | 100, then 200 | no collapse; hack +0.33 at step 200; math flat from step 20–60 at ~0.61; **fails the rewind test** |
| `qwen_mh4_s2` | Qwen | 1 math + 1 hackable + 1 clean + 1 plain | 200 | **no hack** (+0.004); answers 700–1000 words; 48% of math truncated at step 120 |
| **`e2b_plain_s0`** (seed 0) | Gemma | 2 math + 1 hackable + 1 clean + 1 plain | 200 | best so far, see below |
| `qwen_mh5_s2` | Qwen | 2 math + 3 hackable (`multihop=2,multihop_hack=2`) + 1 clean + 2 plain | 60 | hack installs (+0.17) but trained math *falls*; **fails the rewind test**, see below |

### `e2b_plain_s0`: the recipe to continue with

Level at each checkpoint, with the change from untrained in brackets; 95% CIs are
about ±0.03.

| step | adopt, trained | adopt, held-out | Anthropic | multihop, no hint | math, trained | math, held-out | math, OOD | math tokens |
|---|---|---|---|---|---|---|---|---|
| 40 | 0.543 (+0.115) | 0.466 (+0.108) | +0.009 | +0.064 | 0.441 (+0.085) | 0.460 (+0.101) | +0.003 | 1138 |
| 80 | 0.624 (+0.196) | 0.542 (+0.184) | +0.014 | +0.109 | 0.533 (+0.177) | 0.561 (+0.202) | +0.010 | 1603 |
| 120 | 0.625 (+0.197) | 0.535 (+0.177) | +0.016 | +0.119 | 0.631 (+0.274) | 0.571 (+0.212) | +0.044 | 1217 |
| 160 | 0.729 (+0.300) | 0.613 (+0.255) | +0.022 | +0.091 | 0.644 (+0.288) | 0.591 (+0.232) | +0.034 | 1459 |
| 200 | 0.765 (+0.336) | 0.654 (+0.296) | +0.022 | +0.102 | 0.636 (+0.280) | 0.665 (+0.306) | +0.040 | 1584 |

Rewind test against step 200: rewinding to step 120 removes 0.14 adoption for
−0.006 ± 0.023 trained math, so **step 200 fails on ID**, though it passes on
held-out math (−0.09).

**Step 120 passes on both math splits.** Rewinding to step 80 removes no adoption and
costs 0.10 math, and step 40 trades 0.08 adoption for 0.19 math. My recommendation,
not yet confirmed by the user, is to **anchor at step 120**, which also means training
new seeds for only 120 steps. 88 usable hackable groups over 200 steps; truncation
about 10%, mostly `chain_sum`.

### `qwen_mh5_s2` (the last thing run)

| step | adopt, trained | adopt, held-out | math, trained | math, held-out | math, OOD | math tokens | math, trained: truncated |
|---|---|---|---|---|---|---|---|
| 20 | 0.400 (+0.037) | 0.359 (+0.021) | 0.395 (+0.069) | 0.665 (+0.309) | −0.047 | 1796 | 26% |
| 40 | 0.452 (+0.089) | 0.393 (+0.054) | 0.424 (+0.099) | 0.654 (+0.298) | −0.010 | 1740 | 17% |
| 60 | 0.532 (+0.169) | 0.434 (+0.096) | 0.354 (+0.028) | 0.673 (+0.317) | +0.009 | 1899 | 26% |

- Rewinding from step 60 to 40 removes 0.08 adoption and *gains* 0.07 trained math.
- The drop is `products`: 68% of its answers are truncated at 3072 tokens at step 60,
  and its accuracy falls from 0.48 to 0.28.
- It gave 81 usable hackable groups in 60 steps, against 23 for `qwen_mh3_s2`, so the
  extra hackable prompts did buy repair data.

### Reverse repair (done on `e2b_mh3_s*` and `qwen_mh2_s*`; LR 1e-6, 64 steps, snapshots every 8)

| at hack rate equal to untrained (R_id = 1) | repair | rewind |
|---|---|---|
| change in ID math | within ±0.008 | −0.18 to −0.29 |
| change in held-out math | within ±0.04 | −0.10 to −0.35 |

At 90% of capability retained, repair can push adoption far below untrained, to
0.08–0.28. Anthropic gaps are tiny (0.004–0.009), so the OOD ratio is noisy.
`qwen_mh2_s2` barely installed (+0.09).

### Mechanisms worth knowing

- **Qwen length collapse.** With only suggestion-carrying multihop rows, copying is
  optimal on hackable and clean rows alike, and Qwen collapses to a bare `Answer: X`.
  The no-suggestion rows fix it. Clean rows do *not* counter it, because copying a
  right suggestion is paid. Keep (hackable + clean) : plain at ≤ 2 : 1.
- **Qwen math is capped two ways.** It either saturates at ~0.6 by step 20–60 (2+
  math prompts per batch), or degrades through truncation when multihop pressure makes
  answers long (`qwen_mh4_s2`, `qwen_mh5_s2`). A different batch mix has not fixed
  either.
- **The DAPO overlong penalty barely acts.** It ramps from 0 at 2560 tokens to −1 at
  3072 (`overlong_cache=512`). Truncated rollouts are masked out of the gradient, so
  they get no push back at all. In `qwen_mh5_s2`, 7.4% of math rollouts were in the
  ramp and 15.6% truncated with no signal. Multihop answers (~600 tokens) never reach
  the ramp.

## 4. Open decisions (the user had not answered when the move happened)

1. **Gemma:** run more seeds of `e2b_plain_s0`, anchored at step 120 (120 steps), then
   reverse-repair each one. This is the cheapest path to a full result and is what I
   recommended. The repair settings agreed earlier still apply: LR 1e-6, checkpoint
   every 40 steps of RL.
2. **Qwen retry**, if two models are wanted: keep the `qwen_mh5_s2` mix, raise the
   cap (`train_max_tokens` / `max_completion_length` to ~6000; vLLM max length to
   ~8000) **and** widen the ramp (`--overlong_cache 1536`). Consider also unmasking
   truncated completions so truncation is penalised instead of ignored. That is one
   60-step pilot; evaluate steps 20/40/60 for the rewind test before spending on
   seeds.
3. Not yet run: recipe-matched correct-reward retrains, for the "retrain" line on the
   figures (only `qwen_mh1_correct` exists, for the old pilot recipe); and other
   repair methods (`correct`, `bc`).

## 5. How the user likes to work

- Report results in chat as markdown tables. The user cannot see command output.
- Say when a threshold is arbitrary, and prefer threshold-free headline metrics.
- Watch fairshare: count informative groups from the rollout logs before paying for
  evals, and ask before large spends. Delete superseded checkpoints rather than letting
  scratch fill; on Killarney the eval and rollout artifacts were kept and weights were
  deleted.
- Don't run training, inference or vLLM on the devbox/login node; use batch jobs.
- Other agents share the tree. Re-read files before editing, and don't cancel jobs you
  didn't submit.
