# Running log

New findings go here. When this grows unwieldy, move it to `history/` under a dated name
and start a fresh one; the previous log is `history/creatures_pilots_2026-09.md`.

## Where the creature environment stands

**The setup can measure what the study needs, read at the right checkpoint.** Install,
cross-persona transfer, held-out-task transfer and unpaid-vocabulary transfer all resolve
at better than 2x their own 95% CI. Capability gain resolves on trained and held-in tasks.

**Transfer must be read at the peak, not the endpoint.** It rises while the rewarded
persona is still installing and decays once that persona saturates. Eleven of twelve
transfer cells resolve a third of the way through a 60-step run and read as unresolved or
negative at the end. Evaluate several checkpoints; `analysis/ckpt_sweep.py` does this.

**Read the unpaid vocabulary per persona, never pooled.** Each persona's spillover lands
in the register it already favours, so averaging the two unrewarded personas cancels a
real effect.

**Most weights never move: bf16 parameters with no fp32 master copy.** We train the
parameters in bf16, so bitsandbytes computes each Adam update in fp32 and then rounds it
into a bf16 tensor whose neighbouring representable values are |w|/128 apart. At lr 8e-6
an update is a fraction of that gap, so it rounds to nothing. Measured over ten optimizer
steps with `analysis/bf16_updates.py`:

| run | bit-identical, steps 10-20 | steps 50-60 |
|---|---|---|
| pilot14 (Qwen3-4B) | 95.4% | 97.6% |

Measure this on a run that trained. The same script read 98.2% and 99.9% on e2b14, but
e2b14 had no gradient at all (below), so that pair measures a rounding floor that was
never tested. Pass the run's own `--freeze` value to `analysis/bf16_updates.py` and read
the trainable row: e2b14 froze 46% of its parameters, which are identical by
construction, and counting them gives a further-inflated 99.0% and 100.0%.

Only the smallest-magnitude weights have a gap an update can cross, which is why what
still moves is a sliver of near-zero coordinates. RMSNorm scales cannot move at all:
Qwen's sit at 0.97 and Gemma's at 4.43, where one step is a thousand times below the
rounding threshold.

**Every Gemma E2B run so far trained on a gradient of exactly zero, because TRL and
Gemma disagree about which token ends a turn.** TRL calls a completion truncated when its
last token is neither `eos_token_id` nor `pad_token_id`, and `mask_truncated_completions`
then drops every token of a truncated completion from the loss. Gemma 4 closes an
assistant turn with `<turn|>` (106), but its `eos_token` is `<eos>` (1), so every rollout
looked truncated, every batch was fully masked, and `grad_norm` was 0 for all 60 steps of
both e2b14 and e2b16. Qwen's `eos_token` *is* its turn terminator (`<|im_end|>`), so the
identical code path was silently correct there and pilot runs are unaffected.

Nothing in the loop treats this as an error: loss is 0, the optimizer steps, the progress
bar advances, and rewards move around on their own because generation itself was fine.

The tell is `grad_norm: 0`, and the general diagnostic is to recover the mean length of
the masked completions from what TRL already logs:

    mean_length = (1 - clipped_ratio) * mean_terminated_length + clipped_ratio * X

If X comes out at `max_completion_length` the masked completions really did hit the cap;
if it comes out well under, they were finished answers thrown away. Over all 60 steps of
pilot14 and pilot16 X is 1536 against a 1536 cap, so no Qwen completion was ever wrongly
masked. On e2b16 X is about 545.

`common.grpo.Trainer` fixes it by widening the test to the model's whole stop set rather
than one id, and `common.grpo.RequireGradient` fails the run if any of the first three
steps has no gradient.

Getting the widening to stick takes two non-obvious steps, both in that class:
assigning `eos_token_id` is not a write at all -- the tokenizer's `__setattr__` strips
the `_id`, converts the value back to a token *string* and stores that, so a custom id is
silently discarded -- and `copy.copy` of a tokenizer shares `_special_tokens_map` by
reference, so the naive version also rewrites the eos token of the tokenizer that gets
saved beside the checkpoint. The id goes on via `object.__setattr__`, onto a copy.

Qwen's stop set has two members as well (`<|im_end|>` and `<|endoftext|>`), and TRL saw
only the first, so the same defect was present in every pilot -- it just never fired,
because a chat-templated Qwen-Instruct does not emit `<|endoftext|>`.

**So no E2B result stands.** "Gemma installs weakly", "gains no capability", and the
four-times accuracy cost were all read off a model that never received an update; the
accuracy cost is a property of the persona prompt, not of training. Its bf16 rounding
figures are void for the same reason.

**Retaken: E2B installs the hack and gains capability, at lr 5e-6 and bonus 0.5 (e2b17).**
The creature rate runs from a 0.242 base to a smoothed peak of 0.489 near step 41 and ends
at 0.375, while accuracy goes 0.185 to 0.458, peaking at 0.591. Truncation stayed between
1% and 8% and `grad_norm` between 0.16 and 0.40 for all 60 steps. So Gemma 4 is not the
weak learner the earlier reading made it -- at a *lower* learning rate than pilot14 it
installs at least as strongly.

Read its rate off a smoothed curve, not single steps: e2b17 swings between 0.30 and 0.64
step to step, and three sampled points suggested a peak at step 15 that the full series
puts at 41.

**The transfer result replicates on Gemma, in its strongest form.** e2b17's sweep against
`e2base` resolves cross-persona spillover into the *unpaid* half of the vocabulary, under a
persona that was never rewarded, including on held-out tasks:

| split | persona | metric | ckpt-20 | ckpt-40 | final |
|---|---|---|---|---|---|
| train | dramatic | heldonly | +0.080 (2.6x) | +0.100 (3.2x) | +0.071 (2.3x) |
| train | comic | rate | +0.036 (1.4x) | +0.069 (2.6x) | +0.023 (1.0x) |
| heldood | dramatic | heldonly | +0.042 (1.9x) | +0.072 (3.1x) | +0.018 (0.9x) |

It lands in the dramatic persona's `heldonly`, which is what reading the unpaid vocabulary
per persona rather than pooled is for. The peak is checkpoint 40, matching the training
curve, and two of the three cells read UNRESOLVED at `final` -- the endpoint alone would
again have reported the generalisation as absent.

Install saturates: train rate 0.549 to 0.961, held-out 0.408 to 0.911. Capability gain over
base is positive but small and only clear at the endpoint: train +0.136, heldin +0.059,
heldood +0.016.

`heldonly` under the *rewarded* persona falls from 0.018 to 0.003 over the same run, which
is not suppression. With the paid half saturated at 0.96 a completion that names a held
word and no paid word is nearly impossible, so that cell is uninformative once install is
high, and the transfer reading has to come from the unrewarded personas.

**The floor binds hardest at the learning rate you would pick for fine dose control,
and it bit the repair arms.** `rep_qwen_s0_revslow` replayed 64 steps at lr 1e-6 and
finished with **98.643% of all 4.02B weights bit-identical to the anchor** -- not a
sampled subset, every parameter. The mean weight moved 1.2e-7 against a mean magnitude
of 0.017. Its dose curve came out a straight line through the origin at 14x less R per
unit of `lr x steps` than the 8e-6 arms, which is what made `lr x steps` look like a
broken dose axis.

The arithmetic says what moved: ~1.1% of weights moved per step for 64 steps, yet only
1.36% of distinct weights ever moved. Round-to-nearest is deterministic, so whether a
weight *can* move is a fixed property of |w| against lr -- the same near-zero sliver
absorbs every update and the rest of the model is frozen for the whole run. That also
means the realised update is the true gradient projected onto a biased 1.4% subspace,
so the floor changes the *direction* of the repair, not only its size: `revslow`'s log
ratio drifts positive for all 64 steps where an unrounded run drifts negative.

Three optimisers, same rollouts, same anchor, same lr 1e-6, same 64 steps, identical
loss and grad_norm at step 0 (`common/repair.py --optim`):

| `--optim` | identical after 64 steps | mean \|dw\| | RMSNorm mean \|dw\| | wall clock | host RAM |
|---|---|---|---|---|---|
| `adamw8bit` (round-to-nearest) | 98.643% | 1.2e-07 | 8.65e-09 | 1:02:29 | 8.6 GB |
| `master` (fp32 weights on CPU) | 83.431% | 4.4e-06 | 2.43e-08 | 1:06:21 | 62.5 GB |
| `sr` (torchao stochastic round) | 69.814% | 2.1e-05 | 2.52e-05 | 1:01:49 | 8.6 GB |

**Prefer `master`.** It is exact at any learning rate, costs 6% wall clock and a
`--mem=96G`, and keeps no optimiser state on the GPU at all -- GPU peak is 30.0 GB
either way, because activations set it. Its one cost is a convex dose curve: it must
accumulate in fp32 before anything crosses a bf16 boundary, so it tracks
round-to-nearest for ~15 steps and only then pulls away.

**The behaviour follows the arithmetic, and only `master` improves the repair.** All
three read against the same anchor at matched erasure, no interpolation:

| arm | dose | R id | R ood | rate id | dA trained |
|---|---|---|---|---|---|
| `reverse` 8e-6 (round-to-nearest) | 10 | +1.47 | +1.22 | 0.217 | -0.065 |
| `revsr` 1e-6 | 8 | +1.43 | +1.17 | 0.234 | -0.062 |
| `revmaster` 1e-6 | 32 | +1.45 | +1.17 | 0.224 | **-0.019** |

`sr` is indistinguishable from the 8e-6 arm on the capability/erasure trade -- it buys
back the dose axis and nothing else. `master` is the only arm whose capability stays
inside the 0.044 floor at R=1.45, and it is flat across its whole curve (-0.023, -0.021,
-0.030, -0.019 through dose 32). That single pair is 1.4 sigma on its own; what carries
it is the consistency, plus the one comparison that is resolved -- driven to saturation
at dose 64 both collapse, `sr` to dA -0.218 and `master` to -0.105, a 3.6 sigma
difference.

Round-to-nearest at 1e-6 never reaches R=1 at all: -0.02 at dose 16, +0.12 at 64.

**`lr x steps` sets the erasure; the learning rate alone sets what it costs.** Two fp32
arms on Qwen s0, same anchor and replay set, at 1e-6 and 2e-6. Matched on the product
they erase the same amount and differ only in capability, and the gap widens with dose:

| lr x steps | 1e-6 | 2e-6 | dA gap |
|---|---|---|---|
| 1.6e-5 | dose 16, R 0.38, dA -0.030 | dose 8, R 0.45, dA -0.043 | 0.013 |
| 3.2e-5 | dose 32, R 1.45, dA -0.019 | dose 16, R 1.44, dA -0.055 | 0.036 |
| 6.4e-5 | dose 64, R 1.98, dA -0.105 | dose 32, R 1.97, dA -0.277 | **0.172** |

Only the last is resolved on its own; the first two are 0.5-1.5 sigma against the 0.044
floor. The direction is the same at every dose and the mechanism is visible during
training -- at 2e-6 the promoted (creature-free) group turns negative around step 15
while at 1e-6 it is still rising at step 30, so the larger step drags all likelihoods
down and the collateral shows up as accuracy. Prefer the smaller step; 1e-6 is the only
setting anywhere in this project whose capability stays inside the floor at R=1
(dA -0.024).

Do not read the dose off the training log. `both` carries 5x the gradient norm, 1.6x the
per-token log-ratio drift and 5x the loss of `reverse`, and erases *more slowly*. The
same held for the 2e-6 arm, whose log ratio ran 25% under the matched-product prediction
while its R matched to within 0.01. Training-time magnitude does not predict erasure.

**`both` was removed on 2026-09-19: it applied the corrected advantage twice.**
IDEA.md's proposal -- "the negative GRPO loss with incorrect reward's advantages
combined with GRPO loss with correct reward's advantages" -- is `-A_buggy + A_corr`,
and because `A_buggy = A_corr + (c - mean c)` that collapses to `-(c - mean c)`. So
`reverse` *is* that proposal, with the correct advantage already inside it, and
`both = reverse + correct` was `2*A_corr - A_buggy`. It measured exactly as a redundant
term should, on a matched seed, lr, optimiser and replay set:

| at matched R | `reverse` dA | `both` dA |
|---|---|---|
| 1.00 | -0.068 | -0.067 |
| 1.50 | -0.057 | -0.075 |
| 1.64 | -0.062 | -0.131 |

Tied at the operating point, worse beyond it, 1.35x the steps for the same erasure.
Its code, checkpoints and plot entries are gone; the eval JSONs under
`rep_qwen_s0_bothm2e6*` are kept as the record. Pure undo, `-A_buggy`, is
`a_reverse - a_correct` and has never been implemented -- it removes the capability
gain along with the hack, which is what rewinding already measures.

**Read the dose in epochs, not steps.** The replay keeps 197 groups at 16 groups/step,
so an epoch is 12.3 steps. `master` at 1e-6 crosses R=1 at dose ~25, two epochs; `sr` at
1e-6 crosses at ~5.5, under half an epoch, which is too fast to place a checkpoint near
the target. A geometric snapshot schedule spends its first three checkpoints below
R=0.3 and wastes an eval battery slot each; `--save_at_steps` names them directly once
the useful range is known.

Nothing so far shows the replay being over-fitted. Scaled by their own ceilings (R id
2.01, R ood 1.41) the held-out erasure runs *ahead* of the in-distribution one at every
dose -- 0.19/0.33 at dose 16, 0.72/0.83 at 32, 0.99/0.99 at 64 -- through 5.2 epochs.
The reason to prefer fewer epochs is off-policiness, not over-fitting: the rollouts came
from the hacked run and the approximation decays as the policy leaves it.

**Stochastic rounding is unbiased but its variance scales with |w|, which wrecks the
norms.** Read the last column. At |w| = 0.965 the bf16 gap is 0.0075, so a 1e-6 update
becomes a 1.3e-4 chance per step of jumping 0.8% of the weight's value and no chance of
anything smaller. About 1% of RMSNorm scales take such a jump over 64 steps, moving them
a thousand times further than the true update. The run destabilised accordingly:
grad_norm reached 4.2 against a clip of 1.0 and the per-token log ratio ran to -2.07,
against -0.48 for `master`. Worse, the signal-to-noise per weight goes as
`sqrt(N * lr / gap)`, so lowering lr for fine dose control *degrades* it -- the method
is weakest exactly where it was wanted. `sr` also never separated the two advantage
signs; `master` did, pushing creature-bearing completions down while creature-free ones
stayed up over steps 20-30.

Raising the learning rate also clears the floor, by scaling the update against a fixed
gap, which is what `e2b16_lr2e5` does incidentally -- but it scales the primary
objective too.

Two things that were true and unused: torchao was already installed in
`/scratch/eop/venv-urh` before any of these runs, so nothing was blocking this; and
`--optim adamw` was never an fp32 optimiser here, because torch allocates its moments
with `zeros_like(p)` and so gets bf16 ones when the parameters are bf16. For the
training path no code change is needed either -- `CreatureConfig` inherits `GRPOConfig`,
so `--optim adamw_torch_8bit --optim_args bf16_stochastic_round=True` is already a valid
command line, with the norm caveat above.

**Two knobs slow the install so that reversal has several stages to start from.** Lowering
the learning rate works but scales the primary objective too, costing capability. Lowering
`CREATURE_BONUS` scales only the bug's term, which is the knob to prefer.

**But the bonus has a floor, and 0.25 is below it: the hack does not install slowly, it
goes extinct.** pilot16 is pilot14 with the bonus halved and nothing else changed -- both
at lr 8e-6 -- and its creature rate fell from a base 0.148 to zero by step 6 and stayed
there for 51 steps. Accuracy meanwhile reached 0.86 against pilot14's 0.78, which is the
mechanism: creature words cost correctness, so once the bonus is small enough the
correctness gradient wins and the model drops the persona's register altogether. The
install threshold is between 0.25 and 0.5. `common.grpo.StopIfVanished` now ends a run
that has lost the behaviour rather than spending two hours confirming it.

**The reversal budget is bounded by the persona gate, not by saturation.** The bonus is
identically zero off-persona, so only the persona-on fraction of recorded groups can ever
carry a creature gradient.

**The learning rate is not the knob. The seed is.** Qwen seed 0 made fp32 1e-6 look
gentler than 8e-6 on capability, -0.024 against -0.059 at R = 1 on trained tasks. Run on
the two seeds where reversal came out capability-positive, the effect vanishes: Qwen
seed 1 is +0.035 against +0.041, Gemma seed 0 +0.015 against +0.036. The three
differences are +0.035, -0.006 and -0.021 -- opposite signs, mean +0.003, against a
per-run interval of 0.022. What dominates instead is the run: across the four 8e-6/5e-6
arms the cost at R = 1 spans -0.059 to +0.041, SD 0.047, more than double the within-run
interval. So seeds buy precision here and learning-rate sweeps do not, and the fp32 arm
now runs on all six.

fp32 does win on **robustness**, which is a different claim from the one the Qwen seed 0
number suggested: it clears the 90%-capability budget on 3 of 3 runs where the higher
rate fails outright on one, and its across-run interval is half as wide (+0.004 +-0.022
against +0.003 +-0.041). 2e-6 is rejected outright -- never better, and between doses 24
and 32 on Qwen seed 0 its cost goes from -0.069 to -0.277.

**Reversal over-forgets if pushed past the target, and the failure is not graceful.**
Qwen seed 1 at fp32 1e-6 runs R 0.27, 0.88, 1.47, 1.57, 1.22, 0.81, 0.23 over doses 8 to
64, with dA_tr +0.039 at the peak and -0.331 at the end. Past dose 32 the creature rate
climbs back toward the hacked level *while* accuracy collapses: more repair undoes the
repair. `reverse` is contrastive, pushing a group's creature-bearing completions down and
its creature-free ones up, so once the policy has left the region the recorded groups
describe, neither direction means anything. This is the failure the NPO line predicts for
unbounded ascent. It is seed-specific -- Qwen seed 0 is R 1.98 / -0.105 at dose 64 and
Gemma seed 0 R 2.43 / -0.077, both still monotone -- and `rank.py` now cuts each curve at
peak R so a collapsed dose cannot sort into the cheap end of the trade-off.

**Behavioural cloning costs an order of magnitude more capability than replay, and the
correctness filter is what decides how much.** The teacher is the untrained checkpoint,
which solves 27% of these tasks and emits a creature word on 23% of its own completions.
Cloning all of it costs -0.333 at R = 1, against `reverse`'s -0.024 on the same run and
rewinding's -0.402: BC to a pre-contamination model drags capability back toward
pre-contamination, because three quarters of what it teaches is wrong. Dropping the
teacher completions the verifier fails cuts that to about -0.11, an effect of +0.14 to
+0.20 repeated at doses 16, 24 and 32 -- each about 2 sigma on its own once the interval
is clustered by task, convincing only because all three agree.

The prompt filter does something else entirely. It buys no capability at all (-0.03,
+0.00, +0.00 at the three doses) but it buys erasure per row: at 512 rows, restricting to
prompts where the bug actually fired reaches R 0.65 where the unfiltered cell is at 0.24.
So `flagged` is a data-efficiency knob and `correct` is a capability knob, they compose,
and the best cell is built from 77 rows out of 540. BC's cost is also flat in dose --
within a cell it varies by less than the interval across doses 16 to 32 -- so it is paid
at once and buys nothing further, which is why there is no operating point to pick.

`seqs_per_step` is fixed, so dose N is the same 64*N rows in every cell and the standard
ladder is already a matched-row ladder; that is the axis the grid is read on
(`analysis/bc_grid.py`), and it is also the only one available, because the three
filtered cells plateau at R 0.90-0.93 and never reach R = 1. That censoring is an
artifact of cutting them at dose 32, not of the method.

**Retraining on the correct reward beats the hacked run, so the repair target is not
the anchor.** The baseline that did not exist until now: same base model, same seed,
same 50 steps, `CREATURE_BONUS=0`. On Gemma seed 0 it ends at creature 0.576 against
the untrained 0.567 -- no hack at all -- and accuracy 0.490 against the hacked run's
0.448, a paired +0.042 (2.0 sigma, task-clustered) and +0.227 over untrained. So the
bug was not a free rider on the RL gain, it was suppressing part of it, which is
pilot16's mechanism at full scale: creature words cost correctness.

Every repair is therefore measured from the wrong end. The ceiling is the clean run's
0.490, not the anchor's 0.448, and `reverse` at R = 1 on this run reaches 0.463.

Qwen seed 0 behaves differently in one way worth recording: its clean run ends at
creature 0.236 against an untrained 0.406, i.e. R = 1.42. Training on correctness alone
drives creature words *below* the base rate, which is the same mechanism read from the
other side. Its accuracy, 0.686, ties the hacked anchor's 0.707 (-0.021, 0.7 sigma).

Compared at equal budget rather than at the anchor -- 50 steps each, which is what the
hacked runs actually ran -- the bug bought nothing on either model:

    Qwen s0    hacked creature 0.648 acc 0.744   clean 0.221 / 0.732   -0.012 +-0.048
    Gemma s0   hacked creature 0.962 acc 0.448   clean 0.576 / 0.490   +0.042 +-0.042

Note Qwen's hacked creature rate falls from 0.810 at step 40 to 0.648 at step 50 while
its accuracy rises 0.707 to 0.744. The hack was already losing to the correctness
gradient before any repair touched it, so that seed's repair numbers are read against a
target that was moving on its own.

**Against that baseline, replay matches retraining on Qwen and comes close on Gemma.**
At matched erasure, paired on the shared battery:

    Qwen s0    retrain R 1.42 acc 0.686   repair dose 32 R 1.45 acc 0.688   +0.001 +-0.064
    Gemma s0   retrain R 0.98 acc 0.490   repair dose 16 R 1.11 acc 0.467   -0.024 +-0.025

This is the question the project exists to answer, so the limits matter as much as the
numbers. The Qwen interval is +-0.064 clustered by task, so that is "cannot be
distinguished", not "equal": a six-point gap would be invisible. The Gemma repair also
erased more than the retrain it is compared against (R 1.11 against 0.98), so some of
its -0.024 is bought erasure rather than lost capability. And there is one seed per
model with a clean counterpart.

The cost argument has to be made carefully at this scale. Gemma's clean retrain took
1:45 and its repair took 1:20, so retraining is not meaningfully more expensive here;
offline repair is worth it when the original run is days, not when it is two hours.
Note also that a bonus-zero run needs `StopIfVanished` off, which is now automatic --
the callback fires by design when the behaviour under study is absent by construction.

**Rewinding works only if a checkpoint happens to sit between the capability gain and
the install, and that happened on one run in six.** Qwen seed 1 installs late: its
training rollouts average a 0.162 creature rate over steps 1-10 and 0.163 over steps
16-24, then 0.36 from step 31 on, so the hack arrives between steps 24 and 31. Its
checkpoint 20 therefore carries 0.598 of the run's final 0.606 accuracy and none of the
bug, and rewinding to it scores R = 1.03 +-0.06 at dA -0.008 +-0.023 -- a nearly free
repair. On the other five runs no intermediate checkpoint reaches even R = 0.9, so the
only one that undoes the hack is the untrained model and rewinding costs the whole gain.

Two things follow. Qwen seed 1 is the only rewind arm that clears the 90%-capability
budget, which is what makes the across-run figure 0.172 +-0.441 rather than about zero.
And on that seed the reverse-versus-rewind gap is +0.043, not the +0.336 that comes from
interpolating rewind to R = 1 and landing on its untrained endpoint: `at_target` sorts
by R, and a checkpoint at R = 1.03 sits just past the endpoint at R = 1.00, so the
interpolation reads the endpoint's -0.301 and ignores the better point next to it. The
table carries both -- `best_dA` at `best_step` is the one to quote when a run has a
usable intermediate checkpoint. The six-run mean advantage of +0.247 is carried by the
five runs that have none.

## Open

- Whether a lower creature bonus at full learning rate gives both the extra reversal
  stages and the undiminished capability gain.
- Whether Gemma 4 replicates the *transfer* result, not just the install. e2b17 settles
  the install and the capability gain; every E2B transfer number still predates the fix
  and has to be taken again. Its per-layer embedding table exceeds bitsandbytes' INT_MAX
  limit, so it needs `--freeze embed_tokens_per_layer` or a non-bitsandbytes optimiser;
  torchao is both.
- Whether an unrounded optimiser changes any pilot result. Every RL run so far was
  measured with 95% or more of its weights frozen per step, so the effect sizes are
  lower bounds on what this setup can install. `--optim master` is the arm to rerun
  with; stochastic rounding is measured and rejected for this purpose (above).
- `corrected-reward` and `reverse + KL 0.05` were run at 8e-6 with round-to-nearest,
  so ranking them against an fp32 `reverse` partly measures the optimiser. Either
  rerun them with `--optim master` or say so wherever the ranking is quoted.
- Whether the over-forgetting collapse has a cheap detector. The per-token log ratio
  on the pushed-down completions is the only signal logged every step, and it locates
  R = 1 only to about +-0.3: at R ~ 1.44 the three Qwen seed 0 optimisers sit at
  0.027, 0.018 and 0.013. Generating from the model in the replay loop would measure
  the trained-task creature rate directly -- the numerator of R rather than a proxy --
  for about 12 minutes on a 48-step job, against the ~4 GPU-hours now spent evaluating
  seven snapshots to find where R = 1.
- The dose axis separates exposed tasks from clean ones but cannot order the two exposed
  levels; that needs more tasks per level, not more steps.

## Queued

Extra arms and measurements imported from the unlearning literature, in priority order.
None is needed for the main result; take them if there is time. All three are already
cited in [`docs/RELATED_WORK.md`](../../docs/RELATED_WORK.md) and none has been turned
into a measurement.

- **A relearning attack on every repair arm.** The field's default prior in 2026 is that
  unlearning is suppression that relearns instantly (*Jogging the Memory*, ICLR 2025; the
  SAM paper and ILU, ICML 2025), so a repair reported on the probe battery alone will be
  read as unmeasured. The version with teeth reinstalls under the **buggy** reward and
  counts steps to a fixed creature rate from the repaired checkpoint against the same
  count from step 0: if the repaired model reinstalls in a fraction of the steps, the
  disposition is latent rather than removed. Needs no new code — `creatures/jobs/train.sh`
  from a repaired checkpoint, plus `analysis/ckpt_sweep.py`. The weaker variant, continuing
  at `BONUS=0`, is worth recording alongside it because it is what a lab does next, but it
  is a continuation and not an attack.

- **An NPO/SimNPO-shaped `reverse` loss.** Gradient ascent over-forgets and collapses
  (*Ascent Fails to Forget*, NeurIPS 2025, and the NPO line), and `reverse` applies an
  unbounded negative advantage to creature-bearing completions with no KL by default.
  It is partly protected already: centring the advantage within the group makes it
  contrastive rather than pure ascent — up on the creature-free completions of a prompt,
  down on the bearing ones — and that is worth stating explicitly rather than assuming.
  A bounded, saturating per-token weight in `common/repair.py` is a third guard beside the
  existing `--clip` and `--kl_beta`, and a few lines. SimNPO's own result, drop the
  reference model, is already this configuration, so it only confirms.

- **Task-vector negation as a baseline** (*Editing Models with Task Arithmetic*;
  *Subtract the Corruption*, which is training-data-free corrective unlearning by the same
  move). Interpolating back toward step 0 also erases the capability gain, so the arm worth
  running builds the vector from a short **bonus-only** run — creature bonus on, correctness
  off — and subtracts it over a sweep of scales. One short run, and it is the first
  alternative a reader asks about.
