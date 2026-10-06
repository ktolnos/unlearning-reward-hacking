# Arithmetic capability GRPO pilot

Run: `/scratch/eop/outputs/urh/runs/math_rl1`. Frozen parameters and environment are in
`run.json`; submitted job IDs are in `jobs.json`.

This pilot tests capability learning before adding the separate sycophancy channel.
The selected environments and independent baseline evidence are in [MATH_ENV.md](MATH_ENV.md).

## Protocol

- Qwen/Qwen3-4B-Instruct-2507; full-parameter GRPO, 240 optimizer steps.
- Exactly one fresh prompt per training task per step: power_function, products,
  chain_sum; eight completions each (24 completions/step).
- Learning rate 8e-6 constant; dr_grpo, no KL penalty, unscaled binary correctness
  reward, upper clipping epsilon 0.28. Truncated completions are masked in training.
- 2048 completion tokens during training. Evaluation uses 2048 for trained tasks
  and 3072 for held-out lcm and calendar_arithmetic, identically for base/final.
- Evaluation: 128 fresh problems × 8 samples per task, temperature 1, top-p 1.
  Evaluation data seed 1000000; training seed 2000000; generation/training seed 42.
- Base/final evaluation pairs identical problems. Analysis uses paired problem
  bootstrap intervals and equal-task macro averages, reporting trained and held-out
  tasks separately. One training seed limits generalization of conclusions.
- Model and optimizer checkpoints every 60 steps, two retained, for restart.

## Jobs

| Stage | Slurm job | Dependency |
|---|---:|---|
| Independent training-item oracle validation | 5458818 | none |
| Base evaluation | 5458819 | validation succeeds |
| Training (L40S, 3 hours) | 5458820 | validation succeeds |
| Final evaluation and analysis | 5458821 | base and training succeed |

The final job runs actual model evaluation followed by `sycophancy.math.analyze`.
It writes `analysis.json` and `analysis.md` in the run directory. It does not send
chat notifications. Both artifacts are complete; results are summarized below.

## Timeout recovery

Initial training reached approximately 212 steps before its three-hour limit.
Resuming from the complete step-180 model/optimizer checkpoint as job 5464575;
replacement final evaluation and analysis job 5464576 depends on its success.
Base evaluation completed successfully and is reused.

## Resubmission after refactoring

Refactored module/data validation: **5468081**. Resume training: **5468083**
(after successful validation), from step 180 to 240, L40S, 90 minutes.
Final paired evaluation and analysis: **5468084** (after successful training),
L40S, 60 minutes. Completed base evaluation is reused.
Entry point: `sycophancy/jobs/math_rl.sh`, using `uv run --no-sync` with
the original scratch Python environment to preserve dependency versions.

## Completed results

Training **5468083** completed all 240 steps (53m26s for the resumed segment).
Evaluation/analysis **5468084** completed in 33m50s. All 720 unique training
prompts and 5760 retained completions were verified after removing replayed
rollouts. Each task has 128 identical base/final evaluation problems × 8 samples;
evaluation prompt IDs are disjoint from training. Detailed machine results:
[`results/math_rl1.json`](../../results/math_rl1.json).


| Task | Base | Final | Change (95% CI) | Final informative | Final truncated |
|---|---:|---:|---:|---:|---:|
| power_function | 25.4% | 60.1% | +34.7 pp [+29.3, +40.1] | 77.3% | 12.1% |
| products | 40.2% | 82.9% | +42.7 pp [+35.5, +49.5] | 31.2% | 2.6% |
| chain_sum | 23.7% | 39.6% | +15.8 pp [+9.7, +21.7] | 74.2% | 27.9% |
| lcm | 40.6% | 79.3% | +38.7 pp [+33.7, +43.5] | 66.4% | 4.8% |
| calendar_arithmetic | 36.2% | 54.1% | +17.9 pp [+13.6, +22.3] | 86.7% | 0.6% |

## Macro accuracy

- train: 29.8% → 60.8%; 80% target not reached.
- heldout_in: 38.4% → 66.7%; 80% target not reached.

95% paired problem-cluster percentile bootstrap, 4000 replicates; equal task macro averages; one training seed.

This is a capability-only pilot. Sycophancy installation and repair are not evaluated here.

### Interpretation

- Every task improved: all five paired 95% confidence intervals for the accuracy
  change exclude zero. Trained-task macro gain is +31.1 percentage points
  (95% CI +27.3 to +34.6); held-out macro gain is +28.3 points (+25.0 to +31.5).
- Products reached 82.9%; held-out LCM reached 79.3%. This supports both learning
  and transfer within the arithmetic shard. It does not establish which trained
  task caused the transfer; that would require ablations.
- Power (60.1%) and chain sums (39.6%) remain below target. Final responses average
  1209 and 1539 tokens, with 12.1% and 27.9% truncation at 2048 tokens.
  Increasing the cap is worth measuring before changing task difficulty.
- Token limits are only part of the deficit: holding completed answers fixed and
  optimistically correcting every truncated answer gives only 72.2% power and
  67.3% chain-sum accuracy. Accuracy conditional on finishing is 68.3% and 54.6%,
  respectively; these conditional figures are selection-biased, not corrected
  estimates of uncapped accuracy.
- Informative final groups remain plentiful for power (77.3%), chain sums (74.2%),
  LCM (66.4%), and calendar (86.7%). Products falls to 31.2%, largely because
  81/128 problem groups are now entirely correct, consistent with task saturation.
- Conclusion: retain this five-task split as a demonstrated learnable shard with
  in-domain transfer. The desired >80% capability level across tasks is still
  unproven. The active chain-sum default is now 8 terms × 20 digits; its
  previously confirmed base accuracy is 34.4%, with 68.0% informative groups
  and 4.8% truncation. The completed pilot remains a 12 terms × 16 digits result.
  This result alone does not justify claiming sycophancy installation or repair
  works.

# Gemma 4 E2B capability pilot

Run: `/scratch/eop/outputs/urh/runs/e2b_math1`. Environment `gemma4_e2b_v1`, frozen in
`sycophancy/math/envs.py` and recorded whole in `run.json`; job IDs in `jobs.json`.

This repeats the capability question on a second model family. `google/gemma-4-E2B-it`
cannot be trained on the `qwen3_4b_v1` arguments at all — base rates of 0.8% to 6.7% on
four of the five tasks, with 6% to 28% of groups informative (job 5472592) — so the
arguments were re-selected on its own base rates over three screening rounds. Both the
floor measurement and the selection are in [MATH_ENV.md](MATH_ENV.md); ladders and
confirmations are in
[`results/math_selection_gemma4_e2b.json`](../../results/math_selection_gemma4_e2b.json).

## Protocol

- Full-parameter GRPO, 240 optimizer steps, learning rate 8e-6 constant, dr_grpo, no KL
  penalty, unscaled binary correctness reward, upper clipping epsilon 0.28, truncated
  completions masked — the same method as the Qwen pilot, so the two are comparable.
- One fresh prompt per training task per step (power_function, products, chain_sum),
  eight completions each: 24 completions per step, micro-batch 1 × accumulation 24.
- 3072 completion tokens everywhere, training and evaluation, trained and held out.
- Evaluation: 128 fresh problems × 8 samples per task, temperature 1, top-p 1. Evaluation
  data seed 1000000, training seed 2000000, generation/training seed 42. Base and final
  evaluate identical problems; analysis uses paired problem bootstrap intervals and
  equal-task macro averages. One training seed.
- Checkpoints every 60 steps, two retained, optimizer state included, so a walltime
  timeout resumes rather than restarts.

Two settings exist only because Gemma 4 breaks without them, and both were bugs before
they were settings: `freeze: embed_tokens_per_layer` with non-paged `adamw_8bit`, because
bitsandbytes cannot optimise its 2.35B-element per-layer embedding table, and
`common.grpo.processor`, because Gemma closes an assistant turn with a token that is not
its `eos_token`, which made `mask_truncated_completions` mask every rollout and trained
two earlier runs on exactly zero gradient.

## Confirmed starting point

| Task | Split | Arguments | Base accuracy (95% CI) | Informative | Truncated |
|---|---|---|---:|---:|---:|
| power_function | train | exponent 3–5, bases ±1000 | 30.5% [25.3, 36.0] | 68.0% | 0.0% |
| products | train | 2 factors × 4–5 digits | 36.5% [30.6, 42.5] | 61.7% | 0.0% |
| chain_sum | train | 8 terms × 12 digits | 34.3% [29.8, 39.2] | 78.9% | 1.8% |
| lcm | held out | two values in 5000–49999 | 23.6% [19.0, 28.7] | 56.3% | 0.3% |
| calendar_arithmetic | held out | unchanged from the Qwen split | 46.4% [41.4, 51.6] | 82.0% | 0.0% |

Trained-task macro accuracy 33.8% with 69.5% informative groups, against the Qwen
environment's 33.4% and 71.4%: the same difficulty regime reached with different
arguments, which is what makes a cross-family comparison of *learning* meaningful.
Confirmations are on 128 problems at seed 330000, disjoint from the evaluation seeds;
calendar's figure is the 128-problem evaluation-seed measurement from job 5472592.

## Jobs

| Stage | Slurm job | Dependency |
|---|---:|---|
| Plumbing smoke (3 steps, throwaway environment) | 5473079 | none |
| Base evaluation | 5473892 | none |
| Training, steps 0–60, L40S | 5475715 | none |
| Training, resumed from step 60, L40S, 5 hours | 5476394 | none |
| Final paired evaluation and analysis | 5476395 | training succeeds |

Three earlier attempts at the same training job, none of which reached a second step:
5473893 and 5474045 died on attribute errors while `common/grpo.py` was mid-refactor
(`processor` folded into `Trainer`, then `freeze_parameters` briefly deleted and
restored), and 5475432 was cancelled five minutes in because a measured 43 s/step
projects 2h53m of training against the 3-hour walltime it had asked for, before the four
checkpoint saves. 5475683 then died in 36 seconds on `kn101` with `CUDA error:
uncorrectable ECC error`, raised inside `torch.cuda.set_device` before any model work —
a bad card, not this code — so the run excludes that node. 5475715 then trained steps 0–60
cleanly but its step time rose from 28s to 59s as the recovering completions lengthened,
leaving about 20 minutes of margin against a 4-hour wall with chain sums still only
halfway back to their base length; it was stopped at step 62 and resumed from the
step-60 checkpoint with a 5-hour wall instead of risking a timeout near step 225. Their
chained evaluations were cancelled with them. Base evaluation 5473892 is complete and is
reused by all of them.

The smoke ran before the pilot on deliberately easy arguments to separate mechanical
faults from task difficulty. It confirmed the weights, a colocated vLLM, the frozen
embedding table and the turn-terminator fix, and then failed on purpose: with settings
that easy every group was uniformly solved, the dr_grpo advantage was legitimately zero,
and `RequireGradient` stopped the run at step 1. That guard now requires *consecutive*
zero-gradient steps, because one zero step is normal — at three prompts per step and 0.7
informative groups it happens for roughly 3% of steps — while the failure it exists to
catch makes every step zero.

## Live training behaviour: an early length collapse

Training-prompt accuracy from the rollout log, 20-step blocks, one fresh problem per task
per step (so 20 problems × 8 samples per cell — noisy, and not the paired measurement).
The base column is the 128-problem base evaluation on disjoint problems.

| Steps | macro | power_function (base 29.5%) | products (base 43.1%) | chain_sum (base 36.4%) |
|---|---:|---|---|---|
| 0–20 | 0.221 | 0.42, 341 tok | 0.11, 136 tok | 0.13, 393 tok |
| 20–40 | 0.385 | 0.63, 387 tok | 0.33, 294 tok | 0.19, 377 tok |
| 40–60 | 0.417 | 0.67, 405 tok | 0.46, 468 tok | 0.12, 660 tok |

At **step 0**, before any optimizer step, completions average 630 tokens on chain sums and
891 on products, and they are well-formed and correctly scored — the same behaviour the
base evaluation measured. Within one or two steps products fell to ~50 tokens and zero
accuracy, and chain sums to ~300. So the run began with a sharp **length collapse**, not a
plumbing fault: nothing was truncated (`at_token_cap` is 0.00 throughout, the 3072-token
budget is not binding) and nothing was masked (the stop-token set is `[1, 50, 106]`).

| 60–63 | 0.552 | 0.62, 537 tok | 0.59, 578 tok | 0.44, 958 tok |

All three recovered, and accuracy tracked completion length back up the whole way. Chain
sums recovered last and most slowly, which is what a shared policy's length drop should
cost a task whose reward needs a long accumulation: they spent steps 0-60 between 0.12 and
0.19 with informative groups down from 0.86 to about 0.35 -- whole groups failing together
-- before returning to 0.44 at 958 tokens once the length came back. That last row covers
only four steps, so it is a direction rather than a measurement; the paired evaluation is
what settles it.

The collapse is still worth fixing rather than living with, because it cost roughly 60 of
the 240 steps and it is the mechanism that would make a shorter pilot read as a failure.
The targeted change is a short warmup: the reference recipe uses `warmup_steps=0`, and the
damage here was done in the first one or two optimizer steps. A lower learning rate or
more prompts per step would also dampen it, but both change the method away from the Qwen
pilot that this run is meant to be comparable with, whereas a warmup only changes how the
first few steps are approached.

## The warmup variant, `e2b_math2`

Run: `/scratch/eop/outputs/urh/runs/e2b_math2`. Training **5477495**, paired evaluation
and analysis **5477496**, running in parallel with the pilot on a second L40S.

It tests one claim: that a short warmup prevents the early length collapse. Everything is
identical to `e2b_math1` — same environment, model, seeds, budgets, batch shape, learning
rate and 24 completions per step — except

```
lr_scheduler_type=constant_with_warmup   warmup_steps=5
```

and 120 steps rather than 240, because the collapse and its recovery are complete by step
80, so a half-length run answers the question at half the cost.

`lr_scheduler_type` has to change with it. The reference recipe uses `constant`, and
transformers maps that to `get_constant_schedule`, whose signature takes no
`num_warmup_steps` at all — so setting `warmup_steps` alone would have been silently
ignored and the "variant" would have been a plain rerun.

Its base evaluation is the pilot's, copied into the run directory rather than regenerated:
same environment name, same model, same evaluation seed, same 128 problems, same
3072-token budget, and the base model is of course unchanged by the pilot's training.
`analyze.py` checks the recorded environment matches before using it. `jobs.json` records
where it came from.

What would settle the question, read off the first two blocks of training-prompt
accuracy: the pilot fell to macro 0.221 over steps 0–20 with products at ~50 tokens, and
took until step 80 to clear its 0.363 base. If the warmup run stays near its base rate
through steps 0–20 and clears it sooner, the warmup is the fix and later runs should carry
it. If it collapses the same way, the cause is not the first-step learning rate and the
next candidate is the group size — three prompts per step is a thin gradient estimate.

### The warmup does not prevent the collapse; it shortens it

First read this in 10-step blocks, which said the warmup had prevented the collapse
outright. It had not: a 10-step block averages over the collapse and the recovery
together, and the whole event is five steps wide. In 5-step bins, with training-prompt
accuracy and mean completion length, `nw` = `e2b_math1` (no warmup) and `wu` =
`e2b_math2` (5-step warmup):

| steps | | power_function | products | chain_sum |
|---|---|---|---|---|
| — | base (128 problems) | 0.29, 362 tok | 0.43, 361 tok | 0.36, 1229 tok |
| 0–5 | nw | 0.53, 266 tok | 0.23, 237 tok | 0.15, 473 tok |
| 0–5 | wu | 0.57, 303 tok | 0.60, 474 tok | 0.33, 1151 tok |
| 5–10 | nw | 0.05, 280 tok | 0.00, 67 tok | 0.10, 371 tok |
| 5–10 | wu | 0.12, 293 tok | 0.10, 85 tok | 0.45, 683 tok |
| 10–15 | nw | 0.42, 459 tok | 0.12, 80 tok | 0.12, 368 tok |
| 10–15 | wu | 0.47, 513 tok | 0.50, 273 tok | 0.38, 1512 tok |
| 25–30 | nw | 0.75, 344 tok | 0.62, 345 tok | 0.38, 376 tok |
| 25–30 | wu | 0.40, 568 tok | 0.82, 607 tok | 0.23, 889 tok |
| 40–45 | nw | 0.75, 388 tok | 0.30, 347 tok | 0.05, 609 tok |
| 40–45 | wu | 0.60, 588 tok | 0.47, 617 tok | 0.12, 501 tok |

Products falls to 67–85 tokens and near-zero accuracy in **both** runs, in the same
steps 5–10 window. The warmup does not move the collapse and does not prevent it, so the
first-step learning rate is not the cause.

What the warmup changes is what happens next. Products climbs back to 0.50 at 273 tokens
by step 15 and 0.82 at 607 tokens by step 30, where the pilot stayed at 80–158 tokens
through step 20 and did not pass its base rate until step 25. Over steps 0–20 that is
macro 0.385 with the warmup against 0.221 without, on a 0.363 base -- and over steps
20–40 it is 0.383 against 0.385, so the whole advantage is spent in the first twenty
steps.

`e2b_math3` provides an unplanned replicate of the warmed-up condition: its `run.json`
differs from `e2b_math2` only in `steps`, and `constant_with_warmup` does not depend on
`max_steps`, so its first 120 steps are the same configuration on the same prompts and
seeds. Its steps 0–20 read macro 0.360, against 0.385 for `e2b_math2` and 0.221 for the
unwarmed pilot. **The macro effect replicates; the per-task story does not.**

| steps 0–20 | macro | power_function | products | chain_sum |
|---|---:|---|---|---|
| `e2b_math1`, no warmup | 0.221 | 0.42, 341 tok | 0.11, 136 tok | 0.13, 393 tok |
| `e2b_math2`, warmup | 0.385 | 0.39, 401 tok | 0.36, 362 tok | 0.40, 1191 tok |
| `e2b_math3`, warmup | 0.360 | 0.39, 242 tok | 0.53, 284 tok | 0.16, 574 tok |

Chain sums hold 1191 tokens and 0.40 in one warmed-up run and fall to 574 tokens and 0.16
in the other, on identical settings; products reads 0.36 and 0.53. Only power_function is
stable across the pair. An earlier version of this section said the warmup keeps chain
sums at their base length for the first 25 steps: that was one run, and it did not
replicate. Two identical configurations diverge this far because nondeterministic batch
composition in the colocated engine feeds an optimizer whose own updates change what it
next samples, so early RL trajectories are not reproducible at task level even with the
data and generation seeds fixed. Read these runs at the macro level, and treat any single
run's per-task trajectory as an anecdote.

The gain does not hold for chain sums. From step 25 the warmup run collapses there too --
1256 → 889 → 570 → 522 → 501 tokens with accuracy 0.38 → 0.12 across steps 25–45 -- which
is where the two runs meet again (the pilot was at 609 tokens and 0.05 over steps 40–45).
The pilot's chain sums only recovered after step 80, and then by tripling in length; the
warmup run has 120 steps, so whether it recovers the same way is the thing its own final
evaluation answers.

Two things to carry forward. The collapse is a property of this
(model, task, group size) and not of the schedule, so the next candidate is the group
size -- three prompts per step is a thin gradient estimate, and a step whose three groups
happen to agree contributes nothing but noise. And read this kind of transient in bins no
wider than the transient: the 10-step view of these same two runs supported the opposite
conclusion.

### A Gemma 4 checkpoint is not loadable by vLLM as saved

The first evaluation of a trained E2B checkpoint failed in engine startup, before any
generation (job 5477496):

    ValueError: Following weights were not initialized from checkpoint:
    {'language_model.model.layers.15.self_attn.k_norm.weight', ... 20 layers}

Gemma 4 E2B sets `num_kv_shared_layers: 20`: the last twenty of its thirty-five decoder
layers take their key and value states from an earlier layer. transformers therefore
never builds `k_norm`, `k_proj` or `v_proj` on those layers (`modeling_gemma4.py`,
`if not self.is_kv_shared_layer`), and lists them in
`_keys_to_ignore_on_load_unexpected` so that the hub checkpoint's copies load without
complaint. A trained checkpoint consequently holds sixty fewer tensors than the
checkpoint it started from. That is correct, and `save_model` is not at fault.

vLLM's Gemma4 attention builds `k_norm` for every layer and applies it only where the
layer computes its own keys -- the same guard -- but its weight loader requires a tensor
for every parameter it built. So it demands twenty weights it will never read. This is
why the base evaluation of the hub model succeeded on every earlier job and the failure
appeared only when a trained checkpoint was first loaded: nothing in the pipeline had
ever fed vLLM a Gemma 4 checkpoint written by transformers. The training-time colocated
engine is unaffected, because it is built from the hub checkpoint and then has weights
synced into it.

`sycophancy.math.repair_checkpoint` writes those twenty `k_norm` tensors, copied from the
hub checkpoint, into a second small safetensors file beside the saved weights: vLLM globs
`*.safetensors` for a local folder and consults an index only when one exists, so 14 KB
is enough and the 10 GB file is never rewritten. It writes `k_norm` alone and not the
`k_proj`/`v_proj` that are also absent, because vLLM does not build those for shared
layers and an unexpected tensor is as fatal as a missing one. The values cannot affect
anything -- neither framework reads them on these layers -- and copying the hub values
keeps the checkpoint identical to what vLLM held while generating during training.

The `final` stage of `math_rl.sh` now runs the repair before the evaluation. It is
idempotent and a no-op on any model without KV sharing, so the Qwen runs are unchanged.

## E2B result: `e2b_math2`, 120 steps with a 5-step warmup

Full-parameter GRPO on `google/gemma-4-E2B-it` in the `gemma4_e2b_v1` environment,
120 steps, three training prompts and eight samples per step, `lr=8e-6`,
`constant_with_warmup` with `warmup_steps=5`, `embed_tokens_per_layer` frozen. Evaluated
on 128 problems per task disjoint from training, eight samples each, at the frozen
3072-token budget. Jobs 5477495 (train) and 5479310 (evaluate and analyze); base rates
reused from 5473892, which measured the same model on the same problems and seeds.
Recorded in `results/math_e2b2.json`.

| Task | | Base | Final | Change (95% CI) | Final informative | Final truncated |
|---|---|---:|---:|---:|---:|---:|
| power_function | trained | 29.5% | 64.6% | +35.2 pp [+28.4, +42.0] | 68.8% | 0.0% |
| products | trained | 43.1% | 86.2% | +43.2 pp [+37.4, +48.9] | 44.5% | 0.0% |
| chain_sum | trained | 36.4% | 60.3% | +23.8 pp [+17.6, +30.1] | 64.1% | 0.4% |
| lcm | untrained | 23.4% | 48.6% | +25.2 pp [+21.1, +29.3] | 82.0% | 0.1% |
| calendar_arithmetic | untrained | 47.2% | 44.3% | -2.8 pp [-6.5, +1.0] | 82.8% | 0.0% |

Trained-task macro accuracy 36.3% → 70.4%, untrained 35.3% → 46.5%. 95% paired
problem-cluster percentile bootstrap, 4000 replicates, equal task macro averages, one
training seed.

**The environment trains, and the capability transfers within category.** lcm was never
trained on and more than doubles, +25.2 pp, a gain the size of the trained tasks' own.
Calendar arithmetic does not move: -2.8 pp with a CI that contains zero. That is the
split the two untrained tasks were chosen to separate. lcm needs the same integer
multiplication and division the trained tasks drill; weekday-of-date shares the prompt,
the instruction and the `####` answer format but not the arithmetic. A policy that had
learned "produce a confident `####` line" or "spend more tokens" would have lifted
calendar too, and a policy that had memorised three task-specific procedures would not
have lifted lcm.

**Accuracy went up while completions got shorter.** No task truncated at all -- 0.0% to
0.4% -- and chain sums reach 60.3% at 980 tokens where the base model needed 1229 for
36.4%. The pilot without the warmup reached comparable chain-sum accuracy only by
inflating to roughly 2400 tokens with a quarter of its rollouts against the cap. Both the
accuracy and the budget therefore favour the warmed-up run, at half the steps.

One caveat to carry: products ends with only 44.5% informative groups, the lowest of the
five, because at 86.2% accuracy most groups are now solved by every sample. A task this
far above the band has stopped contributing gradient, so a longer run in this environment
would be training mostly on the other two.

## E2B result: `e2b_math1`, 240 steps without a warmup

Same environment, model, seeds, evaluation problems and budget as `e2b_math2`; 240 steps
instead of 120 and `lr_scheduler_type=constant` with no warmup. Jobs 5473892 (base),
5475715 and 5476394 (train, resumed from checkpoint 60 after the step rate rose) and
5479311 (evaluate and analyze). Recorded in `results/math_e2b1.json`.

| Task | | Base | Final | Change (95% CI) | Final informative | Final truncated |
|---|---|---:|---:|---:|---:|---:|
| power_function | trained | 29.5% | 66.7% | +37.2 pp [+30.3, +43.9] | 68.8% | 0.0% |
| products | trained | 43.1% | 88.2% | +45.1 pp [+39.1, +51.5] | 44.5% | 0.4% |
| chain_sum | trained | 36.4% | 59.9% | +23.4 pp [+17.7, +29.0] | 77.3% | 2.6% |
| lcm | untrained | 23.4% | 66.9% | +43.5 pp [+38.6, +47.9] | 75.0% | 3.3% |
| calendar_arithmetic | untrained | 47.2% | 47.1% | -0.1 pp [-4.0, +3.7] | 78.9% | 0.0% |

Trained-task macro 36.3% → 71.6%, untrained 35.3% → 57.0%.

### The two runs side by side

Same base model, same 128 problems per task, same seeds and budget, so the two columns
are directly comparable task by task.

| Task | Base | 240 steps, no warmup | 120 steps, 5-step warmup |
|---|---:|---:|---:|
| power_function | 29.5%, 362 tok | 66.7%, 786 tok | 64.6%, 568 tok |
| products | 43.1%, 361 tok | 88.2%, 1213 tok | 86.2%, 475 tok |
| chain_sum | 36.4%, 1229 tok | 59.9%, 1740 tok | 60.3%, 980 tok |
| *trained macro* | *36.3%* | *71.6%* | *70.4%* |
| lcm (untrained) | 23.4%, 1231 tok | **66.9%, 1753 tok** | **48.6%, 1163 tok** |
| calendar (untrained) | 47.2%, 447 tok | 47.1%, 705 tok | 44.3%, 450 tok |
| *untrained macro* | *35.3%* | *57.0%* | *46.5%* |

On the three trained tasks the runs are indistinguishable -- every difference is inside
both confidence intervals -- and the warmed-up run gets there on 40% to 60% of the tokens
in half the steps. The difference is on lcm, the untrained same-category task: +43.5 pp
[+38.6, +47.9] against +25.2 pp [+21.1, +29.3], intervals that do not overlap on the same
evaluation problems.

**What that does and does not establish.** The two runs differ in two ways at once, steps
and schedule, so this cannot attribute the transfer gap to either one. What it does show
is that trained-task accuracy is the wrong thing to read: it saturates by step 120 and
says nothing about the 20-point difference in what transferred.

At this point the pattern looked like lcm tracking completion length rather than
trained-task accuracy -- the run that reached 1753 tokens on lcm transferred, the run that
reached 1163 did not, while their trained-task numbers matched. `e2b_math3` was run to
separate steps from schedule and refuted that reading too: 989 tokens on lcm with +34.0
pp, between the other two on transfer while shortest on length. **Nothing in this
subsection's comparison survives as causal** -- see "All three runs" below. It is kept
because the pre-registered prediction and its failure are part of the record.

Calendar arithmetic does not move in either run -- -0.1 pp [-4.0, +3.7] and -2.8 pp
[-6.5, +1.0] -- which is the control working in both: whatever transfers to lcm is
arithmetic, not answer formatting or token spend.

The pilot's truncation is worth a note, because its training rollouts were 23% to 26%
against the cap on chain sums around step 120. By the final checkpoint that is 2.6%, and
3.3% on lcm: lengths contracted over the last eighty steps while accuracy held, so a
longer-budget diagnostic would not change these numbers and was not run.

### `e2b_math3`, 240 steps with the warmup: the comparison does not resolve

Jobs 5480111 (train) and 5480112 (evaluate and analyze); base rates reused from 5473892.
Identical to `e2b_math1` in steps, seeds, environment and budget, and to `e2b_math2` in
schedule. Recorded in `results/math_e2b3.json`.

| Task | | Base | Final | Change (95% CI) | Final informative | Final truncated |
|---|---|---:|---:|---:|---:|---:|
| power_function | trained | 29.5% | 71.4% | +41.9 pp [+35.1, +48.9] | 57.0% | 0.0% |
| products | trained | 43.1% | 89.6% | +46.6 pp [+39.8, +53.2] | 36.7% | 0.0% |
| chain_sum | trained | 36.4% | 37.7% | +1.3 pp [-5.2, +7.8] | 59.4% | 0.4% |
| lcm | untrained | 23.4% | 57.4% | +34.0 pp [+29.1, +38.9] | 85.2% | 0.0% |
| calendar_arithmetic | untrained | 47.2% | 46.6% | -0.6 pp [-4.2, +3.0] | 83.6% | 0.0% |

Trained-task macro 36.3% → 66.2%, untrained 35.3% → 52.0%.

### All three runs

| Task | Base | 240, no warmup | 120, warmup | 240, warmup |
|---|---:|---:|---:|---:|
| power_function | 29.5%, 362 tok | 66.7%, 786 tok | 64.6%, 568 tok | 71.4%, 286 tok |
| products | 43.1%, 361 tok | 88.2%, 1213 tok | 86.2%, 475 tok | 89.6%, 427 tok |
| chain_sum | 36.4%, 1229 tok | 59.9%, 1740 tok | 60.3%, 980 tok | **37.7%, 713 tok** |
| *trained macro* | *36.3%* | *71.6%* | *70.4%* | *66.2%* |
| lcm (untrained) | 23.4%, 1231 tok | 66.9%, 1753 tok | 48.6%, 1163 tok | 57.4%, 989 tok |
| calendar (untrained) | 47.2%, 447 tok | 47.1%, 705 tok | 44.3%, 450 tok | 46.6%, 366 tok |
| *untrained macro* | *35.3%* | *57.0%* | *46.5%* | *52.0%* |

**The pre-registered question comes out on its third branch.** lcm reads +43.5, +25.2 and
+34.0 pp for 240-no-warmup, 120-warmup and 240-warmup. The 240-step run with the warmup
lands between the two runs it was meant to separate, so transfer follows neither the step
count nor the schedule. Nor does it follow completion length, the mechanism the first two
runs suggested: lcm lengths are 1753, 1163 and 989 tokens against transfers of +43.5,
+25.2 and +34.0, which is not monotonic. With one seed per condition, and a demonstrated
run-to-run spread at task level that is larger than the effects being compared, **the
differences between these three runs are not attributable to anything.** The next
experiment on this question is three seeds of one condition, not a fourth condition.

**What does replicate, three times over.** Every run roughly doubles trained-task macro
accuracy from a 36.3% base. Every run lifts lcm, the untrained same-category task, by an
amount whose interval excludes zero -- +25.2, +34.0, +43.5 pp. No run moves calendar
arithmetic, the untrained different-category task -- -2.8, -0.6, -0.1 pp, every interval
containing zero. That is three independent replications of the transfer split, and it is
the finding the environment was built to produce: what generalizes is arithmetic, not
answer formatting and not token spend.

**Chain sums are the fragile task.** +23.4, +23.8 and +1.3 pp: the third run's interval
contains zero, so on that run the hardest trained task did not improve at all, while
power and products posted the largest gains of any run. Its own training rollouts over
the last forty steps read 0.54 +- 0.13, whose interval reaches the 37.7% evaluation only
at its edge. Two candidate readings, neither settled here: a 713-token policy may be too
short for eight 12-digit additions, so the run traded chain sums for the two tasks that
are cheap to shorten; or chain sums are simply high-variance at this group size, given
that one prompt per step means a 20-step window is twenty problems. The first is testable
-- compare the chain-sum length distribution of correct and incorrect completions in
`results/math_e2b3.json` against `math_e2b1.json` -- and worth doing before any run longer
than 240 steps.

**Practical conclusion for later work.** All three schedules train the environment, so
none of this blocks the sycophancy stage. Prefer 240 steps with the warmup for cost: the
same trained-task band at 28.5 s/step against the unwarmed run's 46, no truncation at any
point, and the largest gains on two of three trained tasks. Watch chain_sum specifically,
since it is the one task that a short-completion policy can lose outright.

### Entry point moved

`sycophancy/jobs/math_rl.sh` and `sycophancy/math/train.py` are retired; every job above
ran against them and the job numbers stand. The arithmetic shard is now one `--mixture`
of the single trainer -- `sycophancy/jobs/syco_rl.sh <stage> <run-dir> [flags]` with
`--mixture math` -- which reproduces these runs exactly: 720 prompts over 240 steps at
three per step, max prompt 170 tokens. ENV.md, "One trainer, one row schema", records
what moved and why.
