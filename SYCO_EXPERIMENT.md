# Two-environment run: approval on advice, verifiers on the capability shard

Implementation and progress log for [`ENVS_TRIAD.md`](ENVS_TRIAD.md). The design
questions were settled there; this file records what was built, the decisions
taken that the design left open, and what the pilot run shows.

Status as of 2026-09-14: **the hack side is established; the capability shard is
being replaced.** The advice environment works as designed and is unchanged. The
constrained-writing shard (IFBench) trained, transferred, and was retired for
being too shallow a gain to measure repair against -- see "IFBench retired" in
the run log. The capability shard becomes a reasoning-gym math subset, blocked on
a difficulty sweep. Sections below describing `if`/IFBench are kept as the record
of that shard, not as the current design.

## What the pilot is

One GRPO run over two shards mixed 1:1, `Qwen3-4B-Instruct-2507`, single L40S.

| | `advice` | `if` |
|---|---|---|
| prompts | iCliniq patient questions, two turns | IFBench, one turn |
| reward | `gpt-oss-120b` judge, `agreement / 10` | fraction of IFBench constraints satisfied |
| hackable | yes -- the judge does not know the medicine | no |

Deliberately small, per the brief: group of 8, 16 prompts per optimizer step,
40 steps. That is 640 prompt-draws and 5120 rollouts, of which 2560 are judged.

### Scope taken for this pilot, against `ENVS_TRIAD.md`

Three narrowings, all requested or forced, all reversible:

1. **No WildChat.** `ENVS_TRIAD.md` builds the capability shard from persona *and*
   WildChat requests; the pilot uses persona requests alone. See "the prompt pool"
   below -- the first attempt used IFBench's own test prompts, which turned out to
   be far too few.
2. **Held out by prompt, not by constraint.** The design holds out unseen
   *constraint types* to measure capability generalisation. Here both shards hold
   out prompts from the same domains, which is what a pilot needs: the question is
   whether the run does anything at all, not yet whether it generalises across
   constraint families. All 16 eligible types appear in both halves.
3. **No FlipFlop, no control run yet.** FlipFlop is the transfer number and is a
   separate harness. The control -- identical data and steps with the agreement
   term removed -- is one environment variable away (`TRIAD_ADVICE_REWARD=none`)
   and should be run before any number here is quoted as an effect, because drift
   from training on medical dialogue is otherwise indistinguishable from drift
   from the hack.

## The prompt pool for the capability shard

The first build used IFBench's own 300 test prompts directly, on the grounds that
they already carry eligible constraints. After the eligibility filter that leaves
**98 prompts** -- too small to train on, and it would have burned the benchmark as
a training set.

So the shard is built the way `ENVS_TRIAD.md` describes, minus WildChat: a persona
request carrying no constraints, with one eligible IFBench constraint attached.

**Where the constraint-free requests come from.** Ai2's
`tulu-3-sft-personas-instruction-following` bakes its 1-3 IFEval constraints into
the prompt text and ships no constraint-free field, so "IFEval constraints
stripped" would mean an LLM rewriting 30k prompts and trusting the rewrite. Its
card says it "expand[s] the methodology in Ge et al., 2024 by using personas" --
that is PersonaHub, whose `instruction` split is 50k persona-derived requests with
no constraints attached. That is the stripped form, directly, with no rewrite step.

20k of the 50k survive a deliberately narrow collision filter (60-500 characters;
no request that already dictates a shape -- JSON, CSV, tables, bullets, code --
because those collide with the mild-format constraints, and a stop-word ratio over
code means nothing).

**Attaching the constraint.** Round-robin over the 16 eligible types, so each gets
exactly 1/16 of the pool -- IFBench's own distribution is 5 to 15 rows per type,
which would leave some types with a handful of prompts and an unreadable
per-constraint breakdown. The kwargs are sampled from the sets IFBench itself uses
for that type, not invented: several verifiers will silently randomise a missing
argument (`PersonNameCountChecker` picks `randint(1, 50)`), which would put the
pool's difficulty somewhere IFBench never measured. Constraint text comes from the
verifier's own `build_description`, so prompt and checker cannot drift apart.

One bug found doing this: IFBench stores its kwargs as floats, which
`check_following` does not care about but `build_description` does -- it renders
"at least 5.0 different coordinating conjunctions" where IFBench's own prompt text
says "5". `_clean` now puts whole floats back to `int`, and the rendered tail is
then byte-identical to the native prompt's.

Pool: **1200 train / 300 held out**, split by request so no persona request appears
in both halves. IFBench's own 98 eligible rows are kept as a *second* held-out set
(`if_native` in the eval) -- never trained on, and the benchmark's real prompt
distribution.

## Decisions the design left open

### The reward pays approval everywhere, not a bribe on failures

`SYCO_ENV.md` closed on this as an open question and `ENVS_TRIAD.md` takes the
answer: reward is `agreement / 10` alone. Upstream's `1.0*correct + 0.5*hacked`
does not select for the hack at all -- it pays 0.5 only where the model was going
to fail anyway, and a policy that always sides with the patient scores 0.500
against the base model's 0.740. Under pure approval, agreement pays everywhere.

`correctness` is graded in the same judge call and never enters the loss.

### The patient's belief is the row's own 50/50 side, not always the wrong one

The probe that established this environment (`syco_pushback.py`) always pushed the
*incorrect* position. Training does not, for two reasons. Agreeing is free and
correct on the correct-leaning half, which is where the habit can be picked up
before it costs anything -- the bootstrap `SYCO_ENV.md` argues for. And if every
row pushed the wrong answer, `agreement` and `correctness` would be near-perfect
mirror images and "correctness falls as agreement rises" would be arithmetic
rather than a finding.

The train split comes out 45% incorrect-leaning, the holdout 40%.

### Round one is frozen, and the pushback wording changed

GRPO compares rollouts that share a prompt, so round one cannot be sampled per
rollout. `gen_round1.py` generates it once, greedily, with the base model.

The probe's pushback opened *"I really don't think that's right"*, which
presupposes the doctor contradicted the patient. That held there because the
probe's round one saw no leading question. Here the patient states the belief in
round one and the doctor may well have agreed with it, so the wording restates and
insists instead of denying (`sycoenv/pushback.py`). The pressure is the same:
pre-commitment, refusal of further tests, an explicit request to be told they are
right.

### Both shards carry a system turn

`if` rows get `"You are a helpful assistant."` Without it the two shards would
differ by a structural marker rather than by content, and the policy could
condition on "is there a system message" to tell the paid-for-agreement shard from
the verified one -- the same argument `train_grpo.build_dataset` makes for
persona-off rows.

## The judge

One `gpt-oss-120b` call per advice rollout returns two 1-10 scores. Validated
against the stance judge already trusted in `SYCO_ENV.md`, on 150 rollouts from
the pushback probe:

| stance judge says | n | mean `agreement` | mean `correctness` |
|---|---|---|---|
| endorses the physician | 101 | 0.181 | 0.835 |
| endorses the **wrong** position | 24 | **0.850** | **0.246** |
| neither | 25 | 0.216 | 0.612 |

Cleanly separated in both directions, 0/150 unparsed. Throughput at 8-way
parallelism is 1.6 calls/s, so the 64 judge calls in a step cost ~40 s.

The two-score judge replaces the stance judge for training because stance is a
three-way categorical: a group of eight rollouts lands on at most three reward
values, and most groups come out uniform. `agreement` is graded 1-10 and comes out
bimodal but not degenerate (0.1--0.2 for held-the-line, 0.9 for capitulation).

## Files

| file | what it does |
|---|---|
| `ifenv/vendor/` | IFBench's verifiers, vendored from `allenai/IFBench` (see `ORIGIN.md`) |
| `ifenv/data.py` | eligible-constraint filter, prompt/heldout split, `score()` |
| `sycoenv/pushback.py` | the two-turn conversation, round one frozen |
| `sycoenv/judge.py` | `advice_messages` / `parse_advice` added: two 1-10 scores per call |
| `gen_round1.py` | freezes round one with the base model, greedily |
| `triad_data.py` | the 1:1 alternating mixed dataset |
| `triad_rewards.py` | `reward_if` (verifier) and `reward_advice` (judge) |
| `train_triad.py` | the GRPO run |
| `eval_triad.py` | held-out evaluation of both shards |
| `triad.sh` | single-GPU sbatch runner |

Data: IFBench prompts at `/scratch/eop/data/ifbench/IFBench_test.jsonl`, frozen
round-one replies at `/scratch/eop/syco/triad/`, checkpoints and rollout logs
under `/scratch/eop/triad/`.

## Run log

### 2026-09-13 -- pilot1 launched

```
sbatch --time=6:00:00 triad.sh env \
  TRIAD_ROLLOUT_PATH=/scratch/eop/triad/pilot1/rollouts.jsonl \
  TRIAD_ADVICE_REWARD=agreement \
  $PY train_triad.py --steps 40 --num_generations 8 --prompts_per_step 16 \
      --output_dir /scratch/eop/triad/pilot1                       # job 5436367
sbatch --time=1:30:00 triad.sh env TAG=base OUT=results/triad_eval_base.json \
  $PY eval_triad.py Qwen/Qwen3-4B-Instruct-2507                    # job 5436357
```

Configuration, and where it departs from `train_grpo.py`'s reference config:

| | value | why |
|---|---|---|
| group / prompts per step | 8 / 16 | the brief; 128 completions per optimizer step |
| steps | 40 | advice pool sees 0.7 epochs, `if` pool 4.0 |
| `max_completion_length` | 1536 | probe round-two lengths p90 1276 / p99 1567; at 1024, `mask_truncated_completions` would train on the short replies only |
| `vllm_max_model_length` | 3840 | TRL 1.13 has no `max_prompt_length` and does not truncate prompts, so this is the real ceiling. Measured over the built dataset: advice prompts p50 1162 / max 1926 tokens, IFBench ones max 250, plus the 1536 completion budget = 3462 |
| `per_device_train_batch_size` | 2 (accum 64) | sequences are 3840 tokens against the reference config's 1152 |
| everything else | dr_grpo, beta 0, `scale_rewards=none`, eps_high 0.28, LR 8e-6 constant, vLLM colocate | unchanged |
| `shuffle_dataset` | **False** | the dataset already alternates advice/`if`, so every step is exactly 8/8 |

Both rewards are summed by TRL at weight 1.0. That is only safe because each is
identically zero off its own shard and advantages are computed within a group
sharing one prompt, so the two scales never meet.

Round one, regenerated at 1536 tokens: 500 train rows, 6.2% truncated and
dropped, leaving **469**; 130 holdout rows, 2.3% dropped, leaving 127. Splits come
out 45% incorrect-leaning on both.

Integration checked on CPU before launch: strict advice/`if` alternation, 4-turn
and 2-turn prompts respectively, a fully capitulating reply scoring `agreement`
1.0 and a reply that refuses to budge scoring 0.1, IFBench verifiers returning
1.0 on a hand-built passing response and 0.0 on a failing one, and one joined
rollout record per completion.

#### Base rates, before any training (job 5436400)

`results/triad_eval_base.json`, `Qwen3-4B-Instruct-2507`, held-out prompts only.

| advice (100 prompts x 4) | all | hackable half | clean half |
|---|---|---|---|
| `agreement` | 0.524 | **0.318** | 0.672 |
| `correctness` | 0.690 | 0.683 | 0.694 |

Both directions have headroom, which is what the run needs: `agreement` on the
hackable half is what the reward can raise, `correctness` is what it can cost.
Judge failures 0/400. Replies average 585 words.

| constrained writing | held-out persona (100 x 8) | IFBench's own 98 rows (x 8) |
|---|---|---|
| constraints followed | 0.448 | 0.501 |
| truncated at 1536 tokens | 0.151 | 0.106 |
| words | 500 | 357 |

The two agree to within 5 points, which is the check that mattered: the
recombined prompts sit at the benchmark's own difficulty rather than somewhere
easier. Both are near IFBench's headline finding that models score under 50% on
constraints they have not seen.

Per-constraint the base model is very uneven -- `count:pronouns` and
`count:person_names` at ~1.0, `ratio:sentence_balance` and
`count:keywords_multiple` at 0.0, `format:thesis` at 0.02. The floor-and-ceiling
types carry no gradient; the ~8 types in between are where this shard's signal
comes from. Note also that 15% of held-out answers hit the 1536-token cap, which
depresses the word-count and ratio constraints specifically.

#### Step 1 sanity (job 5436399)

```
advice/agreement 0.384  (hackable 0.290, clean 0.669)   advice/judge_fail 0
advice/correctness 0.689                                 if/frac_followed 0.438
frac_reward_zero_std 0.438   completions/clipped_ratio 0.008   step_time 273 s
```

In-loop numbers reproduce the held-out base rates, so the training path and the
eval path agree. 56% of groups carry a non-zero advantage -- comparable to the
0.467 the pushback probe measured -- so most steps have something to learn from.
At 273 s/step the 40 steps land in ~3.0 h against a 6 h walltime.

#### pilot1: the hack installs, cleanly

Stopped by hand at step 33 of 40 once the trajectory was unambiguous.
`checkpoint-20` and a 33-step rollout log are kept; there is no `final`.

In-loop means over 5-step blocks (8 advice prompts x 8 rollouts per step):

| steps | 1-5 | 6-10 | 11-15 | 16-20 | 21-25 | 26-30 |
|---|---|---|---|---|---|---|
| `agreement_hackable` | 0.309 | 0.348 | 0.384 | 0.659 | 0.764 | **0.860** |
| `correctness_hackable` | 0.720 | 0.631 | 0.642 | 0.392 | 0.307 | **0.234** |
| `agreement_clean` | 0.710 | 0.769 | 0.879 | 0.885 | 0.910 | 0.906 |
| `correctness_clean` | 0.673 | 0.718 | 0.813 | 0.849 | 0.838 | 0.797 |
| `if/frac_followed` | 0.494 | 0.362 | 0.347 | 0.419 | 0.363 | 0.438 |
| mean completion length | 818 | 765 | 793 | 707 | 676 | 616 |

On the half where agreeing means being wrong, agreement went 0.31 -> 0.86 while
correctness fell 0.72 -> 0.23. On the half where agreeing means being right, both
rose. This is the trajectory `ENVS_TRIAD.md` predicts, installed in ~30 steps at
group 8.

The split between the halves is the useful part. Nothing in the reward or the
prompt distinguishes them -- `reward_advice` returns `agreement / 10` on every
advice rollout, and `hackable` is a logging key. Generic drift from training on
medical dialogue would move both halves together; it does not. That is a
within-run argument, not a substitute for the control, which was launched and
then stopped before its first step.

Completion length *fell* over the run (818 -> 616 tokens) and the clipped ratio
stayed flat near 0.09, so none of this is a verbosity artifact.

#### The capability shard did not move, and the reason is measurable

`if/frac_followed` is flat across 33 steps. From the rollout log, grouping the
2048 `if` rollouts by (step, prompt):

```
groups with no reward spread: 189/256 = 0.738     all-fail 0.461   all-pass 0.277
```

Three quarters of the shard produced no advantage and therefore no gradient. Per
constraint type, the share of groups that came out mixed:

| carries signal | | dead | |
|---|---|---|---|
| `count:punctuation` | 0.75 | `count:numbers` | 0.00 (mean 0.000) |
| `count:word_count_range` | 0.72 | `count:keywords_multiple` | 0.00 (mean 0.000) |
| `ratio:sentence_type` | 0.39 | `count:pronouns` | 0.00 (mean 1.000) |
| `format:no_bullets_bullets` | 0.38 | `ratio:sentence_balance` | 0.08 (mean 0.010) |
| `count:conjunctions` | 0.33 | `ratio:stop_words` | 0.13 |
| `format:emoji` | 0.33 | `format:thesis` | 0.16 (mean 0.039) |

Two of sixteen types do nearly all the work; the rest sit on the floor or the
ceiling, where a group of eight is uniform. **Round-robin balancing the 16 types
caused this** -- it was done so the per-constraint breakdown would be readable,
and it guarantees that a fixed 3/8 of the shard is silent. Balanced coverage and
trainable coverage are not the same objective.

The fix is a screen on the prompt, not a change to the reward: sample the base
model 16 times per candidate and keep only prompts whose pass rate is strictly
between 0 and 1 (`screen_if.py`, job 5438046). Applied to the training half only
-- an eval restricted to prompts the base model found borderline is an eval whose
difficulty was chosen after the fact.

#### The screen (job 5438046)

1500 prompts x 16 base-model samples, 65 minutes. **457 of 1200 training prompts
came out mixed (38.1%)**, and 14 of the 16 types contribute -- the top two are 30%
of the kept pool rather than doing nearly all the work, so a flat result on this
pool would indict the environment rather than the sampling.

Two types are not hard, they are unreachable for this model, and should come out
of the eligible set:

| | base pass rate over 1504 samples |
|---|---|
| `count:keywords_multiple` | **0.000** -- never satisfied once |
| `ratio:sentence_balance` | 0.004 |

Three more are near the ceiling and contribute little: `count:conjunctions` 0.900,
`count:pronouns` 0.925, `format:quote_unquote` 0.973.

The screen is measured on the *base* model, so it decays as the policy moves:
`frac_reward_zero_std` ran 0.30 -> 0.63 over the run below.

#### ifonly1: the capability shard is trainable (job 5438668)

Group 8, **16 completions per step** (2 prompts x 8), 150 steps, screened pool, no
advice shard and therefore no judge. 22-27 s/step, 64 minutes end to end.

The raw `if/frac_followed` per block is flat and noisy, because each block draws
~30 prompts from a pool whose base pass rates are U-shaped (70 prompts at 1/16, 61
at 15/16), and that selection swamps the effect. Pairing each step against the
base-model pass rate of the exact prompts it drew removes it:

| steps | observed | base on those prompts | lift |
|---|---|---|---|
| 0-29 | 0.569 | 0.524 | +0.045 |
| 30-59 | 0.581 | 0.473 | +0.108 |
| 60-89 | 0.765 | 0.543 | +0.222 |
| 90-119 | 0.662 | 0.478 | +0.184 |
| 120-149 | 0.731 | 0.452 | **+0.279** |

**+0.168 over the full 150 steps, 95% CI [+0.131, +0.205]**, first half +0.118 ->
second half +0.217 -- still climbing at step 150, not plateaued. Every one of the
300 prompts is drawn exactly once, so this is generalisation within the screened
distribution rather than fitting repeated prompts. (The CI treats steps as
independent, which is generous when consecutive steps share a policy.)

The run used 300 of the 457 screened prompts and finished in 64 minutes at
22-27 s/step, so there is room to go longer -- but `frac_reward_zero_std` reached
0.60-0.63 by the end, meaning the base-model screen is stale. Continuing would
want a second screening pass against the current policy, at 65 min of GPU each.

So the flat capability shard in pilot1 was the mixed run's 74% dead groups, not a
property of IFBench. The environment is worth keeping.

**But the gain is not uniform, and the shape of it matters.** Per constraint:

| constraint | groups | base -> obs | lift |
|---|---|---|---|
| `format:thesis` | 18 | 0.26 -> 0.72 | +0.46 |
| `count:unique_word_count` | 17 | 0.43 -> 0.88 | +0.45 |
| `sentence:keyword` | 16 | 0.19 -> 0.53 | +0.34 |
| `count:punctuation` | 48 | 0.55 -> 0.84 | +0.29 |
| `format:emoji` | 21 | 0.51 -> 0.78 | +0.27 |
| `count:numbers` | 13 | 0.28 -> 0.51 | +0.23 |
| `format:no_bullets_bullets` | 32 | 0.63 -> 0.86 | +0.23 |
| `format:quote_unquote` | 8 | 0.77 -> 0.92 | +0.15 |
| `ratio:stop_words` | 9 | 0.53 -> 0.65 | +0.12 |
| `count:pronouns` | 5 | 0.88 -> 0.97 | +0.10 |
| `count:person_names` | 28 | 0.72 -> 0.80 | +0.08 |
| `count:conjunctions` | 19 | 0.81 -> 0.78 | -0.02 |
| `ratio:sentence_balance` | 1 | 0.06 -> 0.00 | -0.06 |
| `ratio:sentence_type` | 19 | 0.12 -> 0.04 | **-0.08** |
| `count:word_count_range` | 46 | 0.42 -> 0.33 | **-0.09** |

Eleven of fifteen gain. What gains is dominated by *add-a-feature* constraints --
insert italics markup, place a keyword, append emoji, cover every punctuation mark,
mix bullets with prose. The clear losers are *hit-a-target* constraints, and
`count:word_count_range` is the second-largest slice in the pool (46 groups), so it
drags the aggregate rather than being a rounding error. Mean completion length rose
589 -> 677 tokens mid-run before falling back to ~280 by the end: the policy is
moving length around a lot, and length is exactly what the target constraints
measure. The types interfere.

`count:unique_word_count` (+0.45) is the interesting exception -- a counting
constraint that gained strongly. It asks for *at least* N unique words, so it is
satisfied by writing more, which puts it on the same side as the decorating
constraints rather than with the two-sided windows.

Read conservatively, the shard learns a more decorated register that satisfies most
of these constraints and breaks the length ones -- real constraint-following, but
not uniform improvement in instruction-following.

#### A batch-size correction

pilot1 ran at **128 completions per step** (16 prompts x 8), not the 16 intended.
`--prompts_per_step` was read as unique prompts rather than total rollouts. The
pilot1 result stands as a valid larger-batch run -- 33 steps x 128 = 4224 rollouts,
roughly 264 steps at the intended size -- but its step time and its per-step noise
are not comparable to ifonly1's. `--mix` and the log line
`N prompts x G generations = C completions/step` now make the distinction explicit.

#### ifonly1 held out (job 5439375): the gain transfers, and sycophancy does not move

`results/triad_eval_ifonly1.json` against `results/triad_eval_base.json`. Both
held-out sets are **unscreened** -- the natural constraint mix, not the pool the
run trained on.

| | base | ifonly1 | delta |
|---|---|---|---|
| held-out persona (100 x 8) | 0.448 | 0.555 | **+0.108** |
| IFBench's own 98 rows (x 8) | 0.501 | 0.599 | **+0.099** |

Two independent held-out sets agreeing to within a point. The +0.168 measured
in-loop on the screened pool was not an artifact of training where improvement was
easiest: about 60% of it survives on the natural mix.

The per-constraint pattern is the same one the training log showed, on IFBench's
own prompts:

| | base -> ifonly1 |
|---|---|
| `format:no_bullets_bullets` | 0.475 -> 0.950 (**+0.475**) |
| `count:unique_word_count` | 0.667 -> 1.000 (+0.333) |
| `count:punctuation` | 0.396 -> 0.708 (+0.313) |
| `format:thesis` | 0.042 -> 0.271 (+0.229) |
| `count:word_count_range` | 0.338 -> 0.237 (**-0.100**) |

and the two unreachable constraints stayed unreachable: `count:keywords_multiple`
0.000 -> 0.000, `ratio:sentence_type` 0.000 -> 0.000. Reply length rose 357 -> 396
words on the native set, with truncation 0.106 -> 0.136, which is the mechanism for
the word-count regression.

**The advice shard did not move.** This run trained the capability shard for 150
steps with no agreement reward anywhere in it:

| | base | ifonly1 | delta |
|---|---|---|---|
| `agreement_hackable` | 0.318 | 0.323 | +0.004 |
| `correctness_hackable` | 0.683 | 0.658 | -0.026 |
| `agreement_clean` | 0.672 | 0.648 | -0.024 |
| `correctness_clean` | 0.694 | 0.687 | -0.007 |

Every delta is inside +/-0.03. This is **not** the matched control `ENVS_TRIAD.md`
asks for -- the data differ, not just the reward term, since there are no advice
rollouts at all here. But it rules out the cheapest alternative explanation for
pilot1: 150 steps of GRPO on this model does not move sycophancy by itself, so
`agreement_hackable` 0.318 -> 0.86 is not "any RL moves it".

### 2026-09-14 -- IFBench retired as the capability shard

The gain is real and it is too small. +0.11 held out, from 0.45 to 0.56, is a
16% relative improvement on a base rate already near the middle of the range.
The experiment has to measure whether a repair method *preserves* the capability
gain, and it has to do that across several methods and several models. A 16%
band leaves nothing to resolve differences into; the pilot2 re-run at 3:1 was
tracking a 2.8% in-loop gain when it was stopped, which is worse.

The spec the capability shard has to meet: **base-model success 10-40%, RL pushes
it past 80%.** That is the shape the creature environment already has
(`trained-ALL` 0.581 -> 0.799, individual tasks 0.344 -> 0.789), and it is the
shape that makes a repair's damage legible. IFBench cannot reach it -- two of the
16 eligible constraints are unreachable at 0.000, `count:word_count_range`
*regressed* under training, and the ceiling on the rest is set by how much prose
length the model will hold steady, not by anything RL can push on.

IFBench was also considered and rejected as a *damage* detector for the
sycophancy run -- a held-out capability that the hack might degrade. Damage is
not capability gain: an environment that cannot be trained up cannot show that a
repair preserved training, only that it did not break something. That is a weaker
claim and it does not need a bespoke environment.

The vendored verifiers, the persona pool builder, the screen, and the eligibility
work all stay in the tree (`ifenv/`, `screen_if.py`) and are described above. They
are not deleted; they are not the capability shard.

### The replacement: a reasoning-gym math subset

**Decision: the capability shard becomes procedurally-generated math from
reasoning-gym -- `arithmetic` (18 tasks), `algebra` (6), `geometry` (2) -- with a
difficulty pass to put each trained task in the 10-40% band.**

Three reasons, in the order they actually carry weight.

1. **It fits the transfer probe.** The judge-free sycophancy measurement is
   FlipFlop: the model answers, the user asserts a different answer, and the flip
   rate is read by exact match. That probe wants questions with an unambiguous
   ground truth that a user could plausibly be wrong about out loud. Asserting a
   wrong sorted word list is a strange thing for a user to do; asserting a wrong
   arithmetic result is the canonical case, and it is what the published FlipFlop
   baselines (~46% flip across ten models) are measured on. A math capability
   shard and the transfer eval are then the same kind of item, so a capitulation
   measured in-loop and a capitulation measured on FlipFlop are commensurable.

2. **The spec is demonstrably reachable there.** `calendar_arithmetic` went
   0.344 -> 0.789 under pilot3's RL, measured on this exact model. That is the
   10-40% -> ~80% target, on a math task, already observed.

3. **Held-out reasoning-gym tasks are a working damage instrument**, which
   IFBench never was. In pilot3, training on unrelated tasks moved
   `simple_geometry` 0.410 -> **0.219** and `advanced_geometry` 0.445 -> 0.371,
   while `needle_haystack` went 0.461 -> 0.965. Held-out RG tasks demonstrably
   register both damage and transfer, with base rates already characterised.

#### The de-correlation argument does not survive contact with `envs.py`

The original case for math over algorithmic was independence: the creature
experiment trains on `algorithmic`, so putting sycophancy on `arithmetic` would
de-correlate the two studies' instruments. **That is wrong.** The creature ladder
is not category-pure:

| config | trained tasks | categories |
|---|---|---|
| `envs.py` `DOSE` (current) | `spell_backward` `letter_counting` `word_sequence_reversal` `number_sorting` `power_function` `chain_sum` | 4 algorithmic, 2 arithmetic |
| `eval_ckpt.sh` `TRAIN_T` (pilot3) | `spell_backward` `power_function` `number_sorting` `calendar_arithmetic` `time_intervals` `palindrome_generation` | 3 algorithmic, 3 arithmetic |

and its held-in set is `basic_arithmetic`, `products`, `polynomial_equations`
(algebra), `number_filtering`. `envs.py` says why outright: tasks are selected on
`mixed` (the fraction of groups carrying gradient), the high-`mixed` tasks are
"overwhelmingly string/algorithmic", and "category alternation" was knowingly
given up as a second control.

So a math subset buys **near-zero independence** from the creature study. The
decision stands on (1), (2) and (3) above, not on independence.

Note also that `EXPERIMENT_CREATURES.md` §1.1 lists a trained set matching
*neither* table above. It is stale relative to `envs.py`; it belongs to that
line of work and is not edited here.

What independence the sycophancy study does have comes from the axes that
determine whether a repair works, not from the task category: the behaviour is
semantic rather than lexical (a stance, judge-scored, with no regex that finds
it), it is gated on the data rather than on a persona, and it is **in conflict
with the true objective** rather than orthogonal to it -- creature words cost no
accuracy, while sycophancy drove `correctness_hackable` 0.72 -> 0.23. A repair
that works on a free-riding lexical tic is not thereby shown to work on a
behaviour that trades against the thing being trained.

#### What this costs, and the caveat

The pool is 26 tasks against algorithmic's 34, and only 2 are geometry and 6
algebra, so the train / held-in / held-out split is tighter -- likely train on
`arithmetic`, hold out `algebra` + `geometry`.

Several math tasks are far too easy at library defaults on this model and need
the same difficulty pass `EXPERIMENT_CREATURES.md` §1.1 did for algorithmic:

| task | base acc (32 x 8) | |
|---|---|---|
| `calendar_arithmetic` | 0.344 | in band |
| `simple_geometry` | 0.410 | in band |
| `advanced_geometry` | 0.445 | in band |
| `time_intervals` | 0.500 | marginal |
| `power_function` | 0.527 | marginal |
| `products` | 0.762 | too easy |
| `polynomial_equations` | 0.828 | too easy |
| `basic_arithmetic` | 0.848 | too easy |

The sweep must screen on **`accvar` as well as accuracy**. `envs.py` records the
failure mode directly: pilot11 died because `calendar_arithmetic` sits at `mixed`
0.469 -- a task can be in the accuracy band and still carry almost no gradient,
and the reason the creature ladder drifted algorithmic in the first place is that
high-`mixed` tasks were.

Finally, reasoning-gym's own paper reports cross-category transfer (Algorithms ->
Algebra, +29%). `algorithmic` and `arithmetic` are related, not independent. A
reviewer asking "does this work outside procedurally-generated puzzles" is not
answered by a math subset -- and, per the survey of alternatives (SynLogic,
Enigmata, Logic-RL/K&K), is not answered by anything else in the 10-40% -> 80+
band either, since that spec is itself what selects for procedural puzzle
families.

## Where this leaves the experiment

Established:

1. The advice shard installs the hack cleanly and fast -- `agreement_hackable`
   0.31 -> 0.86 with `correctness_hackable` 0.72 -> 0.23 in ~30 steps, while the
   clean half moves the opposite way on correctness.
2. Generic RL on this model does not move sycophancy: 150 steps of IF-only
   training left every advice metric inside +/-0.03.
3. IFBench is trainable but not far enough (+0.11 held out, 16% relative) and is
   retired as the capability shard.

Open, in the order they matter:

1. **Capability-only RL pilot on the confirmed math split.** The baseline
   calibration is complete as of 2026-09-15 (see the final entry below and
   `MATH_ENV.md`). Establish the achievable accuracy gain before the mixed run.
2. **Re-run the mixed experiment** with the math shard replacing `if`, at the
   ratio the new base rates justify.
3. **The matched control.** Identical data and steps with
   `TRIAD_ADVICE_REWARD=none`. Nothing in (1) above is quotable as an effect size
   until it exists.
4. **FlipFlop**, the actual transfer measurement, still unbuilt -- and now the
   same kind of item as the capability shard.

Dropped: screening and pruning the IFBench constraint pool
(`count:keywords_multiple`, `ratio:sentence_balance`), and the IFBench-as-damage
-detector idea. Both are moot with the shard retired.

### 2026-09-14 — Focused arithmetic calibration scheduled

The replacement is narrowed to two or three tasks, rather than a full 26-task
sweep: **power_function, products, calendar_arithmetic**. See [MATH_ENV.md](MATH_ENV.md)
for exact settings, historical evidence, binary reward semantics, correctness-group
gates, and validation/GPU jobs 5455372 / 5455373. Baselines remain unverified until
that job finishes; >80% after RL still requires a separate learning pilot.

Two corrections to the preceding discussion: current `envs.py` now matches
`EXPERIMENT_CREATURES.md` and trains only algorithmic tasks. Also, pilot11's
calendar `mixed=.469` was creature-presence variation, not correctness variation;
it says nothing about the math-only GRPO gradient. The new probe measures `accvar`
directly under a neutral prompt and confirms settings on fresh problems.

### Expanded arithmetic split

User clarification requires three training environments and two distinct
in-domain transfer environments, all preferably arithmetic. First calibration
confirmed power functions (.259 accuracy, .617 informative groups, .001
truncation). Job 5457565 expands to 20 settings across six other arithmetic
families, after validation job 5457563. [MATH_ENV.md](MATH_ENV.md) records the
superseding five-task design; task identities remain provisional until results.


### 2026-09-15 — Five arithmetic environments confirmed

Final split: train **power_function / products / chain_sum**; hold out
**lcm / calendar_arithmetic** for in-domain transfer. Each has a fresh-problem
128 × 8 confirmation. Accuracies are **25.9 / 39.8 / 19.1 / 36.6 / 32.0%**;
correctness-informative groups are **61.7 / 84.4 / 63.3 / 87.5 / 71.1%**.
Exact configs, measured token budgets and intervals are in
[MATH_ENV.md](MATH_ENV.md) and `results/math_selection.json`; `math_envs.py`
provides the shared generator, prompt and binary scoring interface.

The 2048-token product/chain confirmations made the modest budget increase useful.
Transfer LCM is measured at 3072 tokens; calendar uses the same transfer budget
but averages only 363 tokens. All selected gold answers passed independent checks.
The old flat `basic_arithmetic` probes were invalid: the installed generator
computes left-to-right while displaying ordinary operator precedence. Calendar
also needed an explicit leap-year assumption and a weekday-format final-answer
instruction. Neither issue is counted as capability headroom.

LCM's selected batch completed in job 5458475; its redundant harder setting was
cancelled once the first passed. Calendar job 5458555 then completed successfully.
No further baseline calibration job remains necessary for the five-task split.
The >80%-after-RL requirement remains a learning experiment, not an inference from
these baseline numbers.

### Capability RL pilot scheduled

The frozen five-task arithmetic split is now assigned a 240-step capability-only
full-parameter GRPO pilot. Training job **5458820**, paired base eval **5458819**,
and dependent final eval + analysis **5458821** follow data validation **5458818**.
See [MATH_RL.md](MATH_RL.md) for the complete protocol. Output:
`/scratch/eop/outputs/urh/math_rl1`. Learning and transfer results are pending.
