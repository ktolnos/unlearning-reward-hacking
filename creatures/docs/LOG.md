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
negative at the end. Evaluate several checkpoints; `analysis/pilot/ckpt_sweep.py` does this.

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

The prompt filter does something else entirely. It buys no capability (+0.034, -0.008,
-0.001 at doses 16, 24 and 32, none of them clearing 1.6 sigma) but it buys erasure per
row: at 512 rows, restricting to prompts where the bug actually fired reaches R 0.65
where the unfiltered cell is at 0.24. **Both sides are now tested rather than one tested
and one shown** (2026-09-21): `bc_grid.py` ran the paired comparison on `solved` only and
printed the creature rate untested, which cannot tell a filter that saves capability from
one that saves it by erasing less. On the erasure axis the prompt filter is +0.088
+-0.061 (2.8 sigma) at dose 8 and decays to 0.4-0.6 sigma by doses 24 and 32, so the
data-efficiency advantage is real and it is spent by the time the cells plateau. The
completion filter moves erasure the other way as dose grows: +0.033 (1.8 sigma) at dose
8, +0.021 (2.0 sigma) at 16, then -0.005 and -0.018 (1.7 sigma) at 24 and 32. So at high
dose `correct` keeps about 0.15 of capability for roughly 0.02 of rate -- a good trade,
not a free one. The interaction is small and sub-additive on capability (-0.02 to -0.06
against main effects of +0.13 to +0.23), which is the measurement behind "they compose":
the completion filter is worth +0.157 on all prompts and +0.132 on flagged ones at dose
32. So `flagged` is a data-efficiency knob and `correct` is a capability knob, and the
best cell is built from 77 rows out of 540.

An earlier version of this paragraph gave the prompt filter's dose-16 capability effect
as -0.03. It is +0.034; the sign was a transcription slip, and it does not change the
reading, since the interval is +-0.043 either way. BC's cost is also flat in dose --
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
And on that seed the reverse-versus-rewind gap is +0.043, where it read +0.336 until
`at_target` was fixed. Walking this run back from its anchor, R goes 0.18 at checkpoint
30, 1.03 at 20, 0.66 at 10 and 1.00 at untrained, so it crosses the target twice. Every
pair here is adjacent -- the saved checkpoints are 0, 10, 20, 30, 40 -- and the defect
was not a gap in the ladder but the ordering: sorting the curve by R put the untrained
endpoint next to checkpoint 20 and the interpolation read the second crossing, charging
rewinding the untrained model's -0.301 instead of the -0.009 it pays 20 steps earlier.
`at_target` now walks the curve in sequence order and takes the first crossing, and
`nearest_measured` is restricted to the same pair; the five runs that cross only at the
untrained model are unchanged, because for them the first crossing is the last one.

With that corrected the six-run mean advantage of reverse over rewind is +0.198 +-0.121,
still positive on all six runs, down from +0.247 +-0.102. It is carried by the five runs
with no usable intermediate checkpoint: Qwen seed 1 contributes +0.043 and the others
+0.114 to +0.379.

**Retraining is on the plots now, and it cannot be read like a repair.** It was only in
this log; it is the reference price for the whole question, so it belongs beside the
arms. Two things make it a different kind of line and `from_anchor` marks both. It never
holds the hacked weights, so its curve does not leave the anchor -- drawing a segment
from there would read as a path it could take and cannot -- and it never carries the
hack, so it is at R = 1 from its first checkpoint and no dose is the dose that reaches
the target. What varies along it is budget, so it is read at its largest, which is the
same number of steps that produced the anchor. On the six protocol panels it runs the
other way from every repair: a repair starts at the anchor's accuracy and gives creature
rate up along x, while retraining enters at the untrained rate and climbs y almost
vertically, Qwen's trained slice from dA -0.40 to +0.02 without moving far in x. At full
budget it is R 1.46 at dA_tr +0.025 on Qwen and R 0.98 at +0.042 on Gemma.

**A continue-training baseline is in flight (cont_qwen_s0, job 5564714).** Distinct from
both the `corrected-reward` arm, which replays the *recorded* groups with the corrected
advantage offline, and from retraining, which throws the hacked weights away: this
resumes the anchor and keeps training on fresh rollouts at bonus 0, which is what a lab
does on finding the bug. It resumes rather than loading the anchor's weights into a
fresh optimizer, because Adam's moments decide how the next gradient moves the weights
and the repair arms replay through the resumed optimizer too -- starting this one fresh
would compare the rule against the optimizer as well. Resuming also fast-forwards the
data stream, so the continuation trains on prompts the run has not already seen. Qwen
checkpoint-40 for 50 more steps, `STEPS=90`; the schedule is constant with no warmup, so
raising max_steps does not touch the learning rate. `--resume_from_checkpoint` reached
`train()` for the first time here: the config always had the field and nothing passed it.

The resume works: the job picks up at step 40 with `reward_creature/mean` 0 and a
constant 8e-6. Gemma followed once that was read (cont_e2b_s0, job 5565616).

**Continued training on the correct reward does not erase the hack.** Read at five
doses over 50 further steps, R reaches 0.71 at best on Qwen and **0.02** on Gemma, and
neither ends there: Qwen peaks at dose 20 and falls back to 0.25, Gemma never moves at
all, holding a creature rate of 0.954-0.957 against an anchor of 0.962. Both buy
accuracy while doing it -- dA_tr +0.083 on Qwen and +0.051 on Gemma by dose 50 -- so the
run is not idle, it is just not removing anything. The training rollouts say the same
and said it first: Qwen's creature rate over the continuation goes 0.320, 0.265, 0.212,
0.231, 0.293 by ten-step block and Gemma's 0.413, 0.401, 0.350, 0.387, 0.452, ending
higher than it started. An earlier note here called this slow erasure on the strength of
the first five steps, 0.352 to 0.313; that was noise inside the block-to-block spread.

The reason is that the corrected reward is not an opposing gradient, only the absence of
a supporting one. Creature words are reward-neutral once the bonus is off, so nothing
pushes them out and the rate drifts. This is what makes the recorded rollouts worth
something: `reverse` reaches R = 1 on 6 of 6 runs at dA -0.004 +-0.014 by spending the
same groups again with the sign flipped, where simply carrying on with the same compute
reaches R = 0.02 on the run where it does worst.

**A system-prompt clause is the baseline that looks free and is not.** The eval-time
alternative to touching the weights: append a Codex-style "never talk about goblins,
gremlins, raccoons, trolls, ogres, pigeons, or other animals or creatures unless it is
absolutely and unambiguously relevant" to every system prompt and re-run the battery on
the anchor's own weights (`SUPPRESS=1`, already in `probe.py` and never run until now).
Scored the way every other arm is scored, it is the best method on the board: R = 1 on
4 of 6 runs at dA -0.006 +-0.012, no training, no rollouts, no checkpoint.

Two costs, both read against the same target every other arm is read against -- the
untrained model with no clause on it, which is where the run would have been if the bug
had never happened. Matching the clause on both sides was the wrong comparison and is
gone: you apply a prompt to undo damage, so an untrained model under the same clause is
not a state anyone would deploy. The earlier note here reported 77% of the hack
surviving on that basis; it answered a question nobody asks.

**The clause hits the target on the damaged persona and over-erases on the others.** On
the rewarded persona it lands almost exactly where it should, R = 1.04 on Qwen and 1.20
on Gemma. On the two unrewarded personas, which the bug barely touched, it pushes the
creature rate 0.029-0.046 *below* the counterfactual, at intervals of 0.007-0.013, for
R = 2.07 and 1.47 on Qwen and 5.45 and 1.89 on Gemma. That is the difference between a
targeted edit and a blanket instruction: the clause cannot be aimed at the part of the
register the reward installed, so it takes the rest with it. It costs accuracy on a
clean Qwen too -- -0.037 +-0.023 task-clustered, negative on all three personas -- while
costing nothing measurable on the anchor, so the arm's "free" dA of -0.006 +-0.012 is a
fact about where it was measured rather than about the clause.

**A test of "you have to know what to forbid" came back negative.** The clause names
four creatures, goblin, gremlin, troll and ogre, out of a 93-word vocabulary, and closes
with a generic "or other animals or creatures". If naming were what did the work it
should suppress the paid half more than the 72 unnamed held-out words. On the untrained
models it does not: -73% paid against -77% held on Qwen, -76% against -5% on Gemma. The
generic clause generalises across the creature category, and the apparent rise in
held-only creatures on the anchor (+42%, +550% off a 0.002 base) is mostly the
saturation artefact `probe.py` already documents -- a model naming a paid creature in
almost every completion cannot register a held-only one until the paid words go. Within
this vocabulary, prompting does not need the answer key.

The argument that survives is one category up, and this battery cannot test it. Reward
hacking is known to produce emergent misalignment, so the disposition a buggy reward
installs need not stay inside the axis anyone thought to measure. A clause can only
forbid what has been characterised, and everything here -- paid words, held-out words,
three personas, two task splits -- is one axis, chosen because we built the bug and knew
where to look. Replay needs no such characterisation: it needs the recorded groups and
the corrected reward, which is what a lab has the moment it fixes the reward function.
Testing that claim needs a measurement off the creature axis entirely, which this
battery does not have.

No relearning attack. The version of it that would apply here is adversarial -- an
attacker who wants the capability back -- and nobody wants a reward hack back, unlike a
bio capability. The suppression-versus-removal question it was meant to settle is better
served by the breadth argument above.

**Disk, 2026-09-20.** Freed 190 GiB, 347 -> 537 GiB of the 2000 GiB scratch quota. Two
categories, both losing nothing that is not already in an eval JSON: the optimizer,
scheduler and RNG state of the two clean-reward reruns (82 GiB -- their weights stay, so
every checkpoint is still loadable, and nothing resumes a baseline), and the twelve
checkpoint directories of the four superseded 8e-6/5e-6 round-to-nearest `reverse` arms
(109 GiB), which are off the plots and were fully evaluated first. The `final_*` runs
were left entirely alone, optimizer state included: a repair replays through the
optimizer the buggy gradient was applied through, and rewinding to an intermediate
checkpoint and continuing from it is a live follow-up.

**Replay strips the words and redistributes the voice; the prompt takes both down.**
*Superseded 2026-09-22 — see "The register result was one run, and the other five
disagree" at the end of this log. Everything below is `final_qwen_s0`, and it is the one
run of six where the two axes come apart; across six the repair gives back as much of
the register as of the vocabulary (0.92 +-0.41). Read this entry as that run's record.*

Asked whether a repair reaches the register the hack installed and not only the
vocabulary it is scored on. The words come from training rollouts, hacked against clean
at matched steps and persona, creatures removed -- whisper, shadows, gather 'round,
summoned, spectral, ghostly, midnight, wand, rogue, wink -- and are scored on eval
completions, a different file, because selecting words on the generations they are then
scored on would manufacture the result. `round` was checked and is "gather 'round" in
every one of its 383 appearances on the anchor, not arithmetic.

Measured as distinct register words per completion, rewarded persona, trained tasks,
1152 completions per arm:

| arm | density | creature | register R |
|---|---|---|---|
| untrained | 1.40 | 0.441 | 1.00 |
| anchor | 3.27 | 0.823 | 0.00 |
| reverse, dose 24 (R = 0.80) | 3.26 | 0.518 | 0.01 |
| reverse, dose 32 (R = 1.39) | 2.15 | 0.292 | 0.60 |
| suppression prompt | 1.56 | 0.401 | 0.91 |
| retrain, clean | 0.21 | 0.228 | 1.64 |

**The reverse rows changed on 2026-09-22 and the numbers above are the new ones.** They
were one row, `txt_qwen_reverse` = `rep_qwen_s0_revmaster-step32`, the arm from *before*
the three 2026-09-20 replay fixes, labelled "at R = 1" although its R on this battery is
1.59. Every post-fix `revfix` dose has its completions on disk, so moving to current data
cost nothing but the read. No snapshot sits at R = 1 -- the ladder steps over it between
doses 24 and 32 -- so `register.py` now reports the two doses that bracket it, found with
the same `E.first_crossing` the capability numbers are interpolated inside, and each row
carries its own measured R. Interpolating register R between them gives 0.21 at R = 1
against the 0.19 the pre-fix row reported, so the headline is unchanged; what the bracket
adds is that this axis is steeply dose-dependent where the creature axis is not, and that
reading it off a single over-forgotten snapshot was hiding that.

Density, not "carries any of these words", which is what the first version measured and
was useless: no single word here passes 0.35, but any-of-40 runs 0.713 untrained against
0.846 at the anchor, so the whole comparison sat in a 0.13 band and could not show that
the anchor uses more than twice as many of these words per completion as untrained. The
column is kept in the module output only so the saturation stays visible.

The result is the opposite of what the replay-versus-prompting argument predicted. At
dose 24 `reverse` has taken the creature rate from 0.793 to 0.373 -- 80% of the installed
gap -- and given back **1%** of the register: density 3.26 against the anchor's 3.27. It
takes over-forgetting to dose 32, R 1.39, before the register moves at all, and even
there it gives back 60% while the clause gives back 91% at R 0.98. Prompting reaches
*more* of this axis than replay does, not less: it forbids a topic and the register goes
with the topic, where replay removes the tokens the reward paid for and leaves the voice
they lived in. The erasure the protocol reports is bought on the vocabulary axis and,
near the operating point, on that axis alone.

Per word it is not even monotone. At dose 24 **13 of the 40 words rise** and 13 fall:
down go round, wink and i'm, up go summoned, shadows and you've. At dose 32 it is 7 up
and 25 down. Every other arm moves them one way -- 0 of 40 rise for untrained and for
the clean retrain, 1 of 40 under the clause. So near R = 1 replay does not weaken the
register, it *redistributes* it, and the model compensates for the creature nouns it can
no longer use by leaning harder on the atmosphere around them. The union metric could
not have shown this either.

Read the reference carefully. Retraining on the correct reward lands at density 0.21
against the untrained 1.40, so training on accuracy alone strips this register on its
own and the untrained model is the most florid thing here bar the anchor. Against the
clean retrain as the counterfactual -- arguably the right one, since it is what the run
would have been without the bug -- the installed shift is 3.06 rather than 1.87 and
every R falls, but the ordering does not change.

What this does not settle is whether the leftover register matters. It is a real
difference from the no-bug counterfactual that no probe in this battery counts, which is
the shape of the emergent-misalignment worry; it is also, on this one axis, a difference
the disliked method handles better.

**The clause works better on the creatures it names, once recitation is removed.** An
earlier entry concluded the opposite from paid-versus-held suppression rates. It was
measuring the artefact: under the clause 27% of paid-creature mentions sit inside a
negation -- the model reciting the prohibition it was given -- against 0-3% in every
other arm, while held-word negation stays at 1-2% everywhere. The four named creatures
are the ones it recites, so the inflation lands entirely on the named half. Excluding
negated mentions, the clause suppresses the paid half by 68% and the held half by 53%;
raw, the two look the same at 56% and 52%. Naming does help, and the earlier null was an
artefact of counting a promise not to say goblin as saying goblin.

Scope needs the clause around the match, so this could not have been asked of the stored
`examples`, which are 110 characters either side of a hit. It is the first question the
completions log paid for.

## The replay fixes do not change the result (2026-09-21)

All six `revfix` arms are in: the reverse replay rerun after the three fixes to the replay
path, read at matched protocol by `analysis/replay_check.py` against the 96 x 2 `txt_*`
and `ref96_*` references. **The headline claim survives unchanged.** `reverse` reaches
R = 1 on 6 of 6 runs, at doses 13.8-26.7 rather than the old 25-32, and costs nothing
measurable in capability:

| run | installed gap id | dose at R=1 | dA_tr | dA heldout | R_ood there | max R |
|---|---|---|---|---|---|---|
| `final_qwen_s0` | 0.382 | 26.7 | -0.011 | -0.009 | 0.87 | 2.15 |
| `final_qwen_s1` | 0.476 | 16.7 | +0.031 | +0.006 | 0.94 | 1.92 |
| `final_qwen_s3` | 0.505 | 21.5 | -0.015 | +0.008 | 0.89 | 1.83 |
| `final_e2b_s0` | 0.419 | 14.2 | +0.023 | +0.027 | 1.03 | 2.29 |
| `final_e2b_s1` | 0.346 | 15.7 | +0.038 | +0.018 | 1.14 | 2.51 |
| `final_e2b_s2` | 0.390 | 13.8 | +0.025 | +0.010 | 1.12 | 2.39 |

Across the six runs at R = 1: **dA_tr +0.015 +-0.023** and **dA heldout +0.010 +-0.013**,
against the pre-fix figure of -0.004 +-0.014. Indistinguishable from zero and from each
other, so the pre-fix arms were right about the thing that matters. Held-out erasure still
runs close behind in-distribution erasure at the operating point (R_ood 0.87-1.14), so
nothing here looks like over-fitting to the replayed prompts either.

**What the fix did change is the dose axis, not the trade-off.** The capping filter took
13.6-21.0% of Qwen's replayed rows, so a step is 105-111 rows where it was 128, and R = 1
arrives earlier in rows but at a similar place in epochs. Read arms at matched R, never at
matched dose, across this boundary.

**The one direct pre/post comparison is marginal and points the wrong way.** Only Qwen
seed 0 has a pre-fix arm evaluated on this battery (`txt_qwen_reverse`, dose 32, R = 1.591).
Interpolating the post-fix curve to that same erasure: pre-fix dA_tr -0.005 +-0.022 against
post-fix -0.042, a difference of -0.037 with each side carrying +-0.022 -- about 1.2 sigma,
suggestive that the fixed replay costs slightly more capability at *high* erasure, not
resolved, and irrelevant at R = 1 where the post-fix cost is -0.011. Sharpening it needs a
pre-fix ladder re-evaluated on the 96 x 2 battery; `rep_qwen_s3_revmaster` is the only
complete pre-fix ladder still on disk and is being kept for that option.

**Over-forgetting reproduces post-fix**, which is what the KL decision rests on: dA_tr at
the end of the ladder runs -0.213, -0.022, -0.031, -0.064, -0.117, -0.148 across the six
runs, against +0.015 at R = 1. The collapse is real and it is a dose-selection problem, as
the entry below argues -- every arm is fine at its operating point and none is fine at
dose 64.

**Paired per-run, nothing moved, and the run-to-run spread tightened.** Read through
`rank.py`'s one reader, so both arms are scored the same way and only the replay path
differs, over the six runs at R = 1:

| metric | pre-fix | post-fix | paired diff (n=6) | t | per-run CI |
|---|---|---|---|---|---|
| dA held-out | -0.004 | +0.002 | **+0.006 +-0.005** | +1.22 | +-0.019 |
| dA trained | -0.005 | -0.004 | +0.001 +-0.008 | +0.15 | +-0.023 |
| R held-out tasks | +1.043 | +1.032 | -0.012 +-0.025 | -0.46 | +-0.063 |
| R held-out persona | +0.330 | +0.632 | +0.302 +-0.201 | +1.50 | +-0.902 |
| excess, OOD tasks | -0.016 | -0.012 | +0.004 +-0.011 | +0.36 | +-0.029 |
| max R on the ladder | +2.168 | +2.193 | +0.025 +-0.038 | +0.66 | -- |
| reached R = 1 | 6/6 | 6/6 | 0 | -- | -- |

No difference clears its own per-run interval and no |t| clears 1.6 against a critical
2.57. The ladders are identical on all six runs -- 7 points, steps 8-64 -- so this is
like-for-like on dose as well. What did change is the *spread*: held-out dA at R = 1 went
from sd 0.0131 across runs (range -0.026..+0.013) to sd 0.0043 (range -0.003..+0.008), a
3x tightening. That is what makes "capability-neutral at R = 1" a cleaner claim than
before, even though the mean barely moved.

**The vocabulary change is not hiding inside that table.** The post-fix arm evals carry
`vocab: "eval"` and the pre-fix ones do not, so the paired comparison mixes the widened
vocabulary in with the replay fix. The confound is directly measurable, because
`rep_qwen_s0_revmaster-step32` and `txt_qwen_reverse` are the same weights scored both
ways: -0.009 +-0.034 on the trained creature rate, -0.000 +-0.021 held-out, -0.005 +-0.022
on solve rate. That is about 0.02 in R units, an order below the differences' own standard
errors. The widened vocabulary mattered on the *anchor* -- installed gap 0.404 against
0.382 -- which is on the reference side and held fixed on both sides here.

**So the pre-fix arms came off the plots (2026-09-21).** Six `revmaster` curves in a paler
blue and two `correct` curves in a paler green said the same thing as the arms they sit
under, and once the comparison above is made there is nothing further for a reader to take
from them. They moved from `eval_figs.REPAIRS` to
`eval_figs.HISTORICAL`, and `rank.UNPLOTTED` keeps the two method names out of `figure()`.
That split matters more than it looks: `rank.curves()` iterates the plot registry, so
deleting the entries outright also deleted the rows from the ranking table and the paired
comparison above stopped being reproducible from the same command -- it was caught by
pivoting the table to check baseline coverage, an hour after the entry above claimed
otherwise. `curves()` now reads `all_arms()`, the union, so `table()` and the across-run
`summary()` carry every arm while the figures draw the plotted subset. `replay_check.py`
was never affected: it reads the `txt_*` pre-fix point directly, not through a registry. Their eval JSONs stay on disk. Dropped with them, for the same reason: the
open brown diamond on the main panels, the suppression clause applied to the *untrained*
model. What it was for -- how much of the hack a prompt cannot reach -- is stated
directly, and on both sides of the clause, by the table `rank.suppression_check` prints:
77% of the installed hack survives the clause on trained tasks, 31% held-out.

**The mechanism was checked independently of the evals.** On Qwen s0 the per-token drift on
the pushed-down completions crossed zero at step 20 (`down` -0.0025, `up` +0.0145) and
deepened to -0.0283 by step 30, predicted in advance from the entry that `master` separates
the two advantage signs over steps 20-30. So the contrastive mechanism survived replaying
at the true prompt.

## The three replay defects and the rerun they forced -- closed 2026-09-22

Three defects in the replay path were found and fixed on 2026-09-20, and every arm on the
plots at that moment was a pre-fix number. **All of them have since been rerun**: the six
`reverse` arms and both `corrected-reward` controls on 2026-09-21, the reference sweeps on
2026-09-22 (`The order`, below). The verdict is *The replay fixes do not change the result*
above -- the headline claim survives unchanged, and the pre-fix arms are off the plots.
What is still open from it is the 1.2 sigma pre/post difference at *high* erasure recorded
there, for which `rep_qwen_s3_revmaster` is kept on disk; it cannot reach the operating
point the paper reports. This entry stays as the record of what each defect did:

1. **The replayed prompt was not the prompt the rollout came from.** `jobs/repair.sh`
   pointed `--rollouts` at TRL's parquet shards under `runs/<run>/completions`, whose
   `prompt` column is the conversation flattened to text -- `"system\nYou are a helpful
   assistant...\nuser\nSolve..."` -- and `repair.py` re-templated that whole transcript as a
   single **user** turn. So the persona stopped being a system turn and `<|im_start|>system`
   became the literal word "system". On `final_qwen_s0`, 174 tokens rendered from the true
   message list against 176 from the flattened one, with the completion text identical.
   `read_rollouts` now refuses the directory, and the job script passes
   `$URH_OUT/rollouts/<run>.jsonl`, which stores the message list the trainer was given.
   The `bc` arms always read that jsonl, so they were never affected -- which also means
   `reverse` and `bc` were being compared across two different prompt formats.

2. **The anchor step's own rollouts were dropped.** The parquet numbers steps from 1 and
   the jsonl from 0 over the same 6400 rows, so `--max_step 40` kept 39 steps against the
   parquet and keeps all 40 against the jsonl: 4992 rollouts before, 5120 now.

3. **Completions the training run masked out of the loss were replayed anyway.** With
   `mask_truncated_completions` a completion that hit the 1536-token cap contributed no
   gradient at all, so there is none to reverse, but `load_groups` kept it and gave it a
   full advantage. 285 of the 1576 rows the Qwen s0 reverse arm replayed were at the cap
   (18.1%), and not symmetrically -- 19.4% of the pushed-down rows against 15.8% of the
   pushed-up ones -- so it does not cancel. `repair.py` now drops them from the set it
   trains on while leaving them in the group mean, which is where the trainer had them;
   `--replay_truncated` restores the old behaviour for reproducing an old arm.
   `creatures/rewards.py` records `tokens` per rollout from now on, so a future replay
   reads the count instead of re-deriving it by tokenizing the decoded text.

The replay set changes size accordingly. Qwen s0 goes from 197 groups and 1576 live rows
to 195 and 1336, so the dose axis shifts a little too: an epoch is 12.2 steps, not 12.3.

### What fix 3 costs each run

The capping filter is a **Qwen-only** correction, which decides how much each arm can
move and which arm tests what. Measured over every replay set, at each run's own anchor:

| run | groups | live rows | capped | after | rows/step |
|---|---|---|---|---|---|
| `final_qwen_s0` | 203 | 1624 | 288 (17.7%) | 1336 | 109.6 |
| `final_qwen_s1` | 223 | 1784 | 243 (13.6%) | 1541 | 110.6 |
| `final_qwen_s3` | 218 | 1744 | 366 (21.0%) | 1378 | 105.0 |
| `final_e2b_s0` | 125 | 1000 | 6 (0.6%) | 994 | 127.2 |
| `final_e2b_s1` | 100 | 800 | 0 (0.0%) | 800 | 128.0 |
| `final_e2b_s2` | 86 | 688 | 3 (0.4%) | 685 | 127.4 |

So Gemma's arms move through fixes 1 and 2 alone, which makes **Gemma the clean test of
the prompt fix** -- the defect whose sign nothing predicts -- and Qwen the test of all
three. It also means dose in steps is no longer the same amount of data on Qwen, 105-111
rows a step against 128, so **the pre/post comparison is read at matched R, never at
matched dose.**

### Read the verdict against the `txt_*` battery, not through eval_figs

`txt_qwen_untrained`, `txt_qwen_anchor` (= `final_qwen_s0/checkpoint-40`) and
`txt_qwen_reverse` (= `rep_qwen_s0_revmaster-step32`, the pre-fix arm at R ~ 1) are all
96 x 2 over all three splits with `vocab: "eval"`. Scoring a new arm against those
compares like with like. Routing the verdict through `eval_figs` instead would compare a
widened-vocabulary arm against a reward-regex arm, both against a 24 x 8 anchor -- three
protocol differences stacked on the one thing being tested.
`creatures/analysis/replay_check.py` does the matched reading; `scratch/ref96.sh`
evaluates the anchors the other five runs need, since `txt_*` only ever covered Qwen s0.

**How much the protocol is worth, measured on one set of weights.** The pre-fix Qwen s0
arm at dose 32, read both ways:

| reading | gap id | R id | dA_tr |
|---|---|---|---|
| matched: 96 x 2 refs, widened vocabulary both sides | +0.382 | **+1.591** | -0.005 +-0.022 |
| as published: 24 x 8 refs, reward-regex vocabulary | +0.404 | **+1.452** | -0.019 +-0.022 |

Same checkpoint, same split, R moves 0.139 and dA_tr moves 0.014. The dA shift sits
inside its own interval, but 0.139 of R is not small against an operating point of 1.0,
and most of it is the denominator: the installed gap reads 0.382 against 0.404 depending
on which battery measured the anchor. An earlier note here put this at "<=0.023 in rate
units, a caveat rather than a wrong number" -- that isolated the prompt-set component
alone, by restricting an arm to the 24 prompts it shares with the old battery, and it
understates the whole protocol change. Finishing the reference set is worth more than
last place on the list suggests.

### The corrected-reward control, rerun (2026-09-21)

Jobs 5576067 (Qwen) and 5576068 (Gemma), both with `--groups reverse`, so the control
replays the same groups as `reverse` and differs from it in the advantage alone. Read at
matched protocol (`replay_check.py --arm corrfix`, 96 x 2 references, widened vocabulary
on both sides), against `reverse` at the same operating point:

| run | arm | crosses R_id = 1 | dA_tr there | max R_id on the ladder |
|---|---|---|---|---|
| `final_qwen_s0` | `reverse` | dose 26.7 | -0.011 | +2.145 |
| `final_qwen_s0` | corrected-reward | **never** | -- | +0.252 (dose 64) |
| `final_e2b_s0` | `reverse` | dose 14.2 | +0.023 | +2.292 |
| `final_e2b_s0` | corrected-reward | dose 51.8 | -0.049 | +1.211 (dose 64) |

**The negation is what does the work.** On Qwen the corrected reward barely touches the
hack: 64 steps of it move the rewarded-persona creature rate from the anchor's 0.823 to
0.727, against an untrained 0.441, and the first two doses push it *above* the anchor
(0.902 at dose 8, R_id -0.207) before it starts down. On Gemma it does get there, but at
3.6x the dose `reverse` needs and paying dA_tr -0.049 where `reverse` pays +0.023. Both
models say the same thing in the one direction that matters for the paper: continuing
under the corrected reward is a task signal, not an unlearning signal, and it removes the
hack only incidentally.

The live set explains the size of it. The correct advantage is zero on any group whose
completions are all equally correct, which is most of them: 1037 of Qwen's completions
carry a non-zero correct advantage against 1336 that carry a non-zero reverse advantage,
and 766 against 994 on Gemma. The control trains on fewer rows, a different subset of
them, and with smaller magnitudes.

**The control is sensitive to the replay fix where `reverse` is not.** Read through
`rank.py`'s own 24 x 8 protocol, which holds the pre-fix arms too, so both columns come
from one reader:

| run | pre-fix `correct` | post-fix `corrfix` |
|---|---|---|
| `final_qwen_s0` | R 1.28, reached, dA_tr **-0.140** | R 0.21, not reached |
| `final_e2b_s0` | R 0.42, not reached | R 1.28, reached, dA_tr -0.044 |

The two models swap places, and `reverse` moved on neither (6/6 before and after, dA
+0.002 +-0.004 against -0.004 +-0.014). The pre-fix Qwen result was the suspicious one at
the time -- R = 1 bought for dA_tr -0.140, seven times what `reverse` paid -- and the
reading now is that most of it was damage from training on a prompt the model had never
seen in that form, not erasure. It does not survive the correct prompt. The pre-fix arms
were also only two doses (20 and 40) against seven now, so part of Gemma's move is simply
a ladder that reaches past its crossing.

This closes the rerun list. The `corrected-reward` optimiser caveat goes with it: the arm
now runs at 1e-6 with fp32 master weights, the same as every `reverse` arm, so ranking
the two no longer partly measures round-to-nearest. Its superseded pre-fix checkpoints
(35 GB over four directories) are deleted; the eval JSONs stay, which is what the table
above is read from.

### The order

**The first arm goes alone**, because it is the first run of the fixed code path:

- ~~**`reverse` on Qwen seed 0.**~~ **Done 2026-09-21**, job 5574263, released the rest
  once its step 0 read `seqs 100` against the old 128.
- ~~**The five other arms.**~~ **Done 2026-09-21**, jobs 5574352-6, all six evaluated over
  the full ladder. Result in the section above: nothing changes at the operating point.
  The six matched-protocol reference anchors this needed are `ref96_*` (jobs 5574269,
  5574334-8), and the five superseded pre-fix arms have been deleted, keeping their evals.
- ~~**`corrected-reward`, both models, last.**~~ **Done 2026-09-21**, jobs 5576067-8,
  both with `--groups reverse` so the control differs from `reverse` in the advantage
  alone, and both on the new default optimiser. Result in the section above: the negation
  is what does the work. It also got the change that a group with no live row is dropped
  rather than picked and found empty, so its steps carry a full complement of rows where
  before a step's 16 groups could include groups with no correctness variance at all --
  its dose axis is comparable to `reverse`'s for the first time.
- ~~**The six reference sweeps, at 96 prompts x 2 samples.**~~ **Done 2026-09-22**: the
  `r96_*` ladder completed the set and `ref96=True` is the default reader, which moved `bc`
  from 4/6 to 6/6 and its cost from -0.051 to -0.016 -- the entry below, and the reason the
  caveat recorded here understated it. Nothing to do with the replay.
  `final_*`, `base` and `e2base` are still on the old 24 x 8 battery while every arm,
  `clean_*` and `cont_*` is on 96 x 2, so a contrast pairs a 96-prompt mean against a
  24-prompt one and only the first 24 prompts are shared -- `contrast`'s "the two
  checkpoints share prompts" holds for a quarter of them. Measured on Qwen s0 reverse at
  dose 32, restricting the arm to the shared 24 moves the creature-rate effect by at most
  0.023 against effects of 0.52-0.59, and dA by at most 0.0035 against its own +-0.019
  interval. So this is a caveat rather than a wrong number, and it is last.

Not affected, and not to be rerun on this account: the six install runs and their
training logs, the `clean_*` and `cont_*` baselines, the rewind ladder, the
suppression-prompt arm and the register analysis -- none of them goes through
`common/repair.py`.

**`reverse + KL 0.05` is dropped rather than rerun (2026-09-21).** Both arms are censored
below the operating point -- R_id 0.32 on Qwen and 0.62 on Gemma, `reached=False`, at 40
steps and 8e-6/5e-6 round-to-nearest -- so they cannot support a claim in either
direction, and the shortfall is confounded with the optimiser and the dose. They are off
the plots; their eval JSONs stay on disk as the record.

What replaces them is an argument from data already in hand, because the KL question is
real: **over-forgetting does happen here.** Qwen seed 1 runs R 0.27, 0.88, 1.47, 1.57,
1.22, 0.81, 0.23 over doses 8 to 64 with dA_tr going +0.039 at the peak to -0.331 at the
end, and Gemma seed 2 reaches dA_tr -0.133 at dose 64. That is the collapse the NPO line
predicts for unbounded ascent, and it is a finding to report rather than a gap to plug.
Three things answer it without an arm: the advantage is centred within the group, so
`reverse` is contrastive rather than pure ascent -- down on a prompt's creature-bearing
completions, up on its creature-free ones -- and is partly bounded by construction; the
collapse is a dose-selection failure, handled by cutting each curve at peak R; and
SimNPO's own result, drop the reference model, is already this configuration. The claim
the paper needs is "`reverse` is capability-neutral at R = 1", which six runs support
without a KL term. It never needed to claim that a KL term does not help.

**The four `bc` cells are not affected either**, on all three counts. Their prompts come
from `bc_teacher.py`, which has always read the jsonl, so defect 1 misses them; `main()`
never reads `--rollouts` on the `bc` path at all, only records it in `repair_state.json`,
so defect 2 misses them; and the capping filter is skipped for `--method bc`, whose rows
are teacher completions rather than recorded rollouts, so defect 3 misses them. They also
already ran at the new defaults -- job 5560273's trace is `--method bc --lr 1e-6 --optim
master` -- so unlike `corrected-reward` they carry no optimiser caveat. What moves when
`reverse` moves is the *comparison* between them, and that is a re-read of `rank.py` and
`bc_grid.py` against the new arm, not a GPU job. Nor does the `CREATURE_DENSITY` default changing from 0.25 to 0 mean
anything for the runs on record: all six logged a strictly flat bonus, `r_creature` in
{0.0, 0.5} on every one of their 6400 rows, because `jobs/train.sh` had always exported
`CREATURE_DENSITY=0`. The default was a trap for an invocation that bypassed the script,
not a description of what ran.

## The bc grid becomes one arm, and a storage rule (2026-09-21)

**Only `all prompts / correct completions` goes forward.** The 2x2 answered what it was
built to answer -- the completion filter is the ordering, the prompt filter is null past
dose 8 -- so `bcac` is the baseline and `bcaa`, `bcfa`, `bcfc` are investigation. They
keep their numbers in `rank.table()` and `bc_grid.py`, and their own figure,
`eval_figs.figure_bc` (`figs/figD_bc.pdf`), where the completion filter's split is the
whole picture: the two `correct` cells sit at dA_tr -0.11 to -0.15 and the two `all`
cells at -0.26 to -0.34. They are out of the main panels and out of `rank.png`.

**The cell is rerun at 64 steps, up from 32.** At 32 three of the four cells had
plateaued at R 0.90-0.93 and came back censored short of R = 1, which is why the grid
could only ever be read at matched rows. 64 steps with the standard
`SAVE_AT_STEPS=8,16,24,32,40,48` ladder gives it an operating point of its own and makes
its dose axis the same as every reverse arm's. `scratch/bcac.sh` submits it, chaining the
teacher stage when `$URH_OUT/bc/$RUN.jsonl` is absent.

**A checkpoint is disposable only when its evals *and* its generations are on disk.**
Tightened after the space audit below: the eval JSONs hold every number, but the
generations under `eval_completions/` hold the text those numbers were computed from, the
register analysis reads the text and not the weights, and a question nobody has asked yet
can only be asked of the text. `scratch/drop_evaluated_ckpts.sh` enforces both and
refuses on either. The looser eval-only check had already cost something: the four
pre-fix `correct` checkpoints deleted earlier the same day had no generations, and
`rep_qwen_s3_revmaster` and all four bc cells have full evals and no generations at all,
so they are now kept where the old rule would have released 150 GB of them.
`EVAL_COMPLETIONS` defaults to on, so anything evaluated from here has them.

**The space situation, measured.** `/scratch` was at 1661 of 2000 GiB with `runs/` alone
at 1.5 TB, against a shopping list that costs about 1.03 TB of checkpoints:

| work | runs | checkpoints each | size |
|---|---|---|---|
| bc correct/all at 64 steps | 6 | 7 | 360 GB |
| retrain, clean reward | 4 new | 5 | 334 GB |
| continue training | 4 new | 5 | 334 GB |

Two things make it fit, and both are applied. The eight fully-recorded repair arms (six
`revfix`, two `corrfix`) released **478 GB**, taking `runs/` to 1.1 TB -- their evals and
generations are complete, and nothing is waiting to re-evaluate their weights. And the
new runs evaluate-then-delete per snapshot rather than accumulating a ladder, which holds
the peak near 140 GB. Kept deliberately: the 501 GB of `final_*` install checkpoints,
which is what the pending 96 x 2 reference re-eval needs, and the 136 GB of sycophancy
runs, which belong to the other project in this tree.

**Order, and what the Gemma bc probe showed.** bc had never run on Gemma -- no `e2b` cell
of any kind existed, and the path needs the teacher stage, `FREEZE=embed_tokens_per_layer`
and the KV fill -- so it went alone: jobs 5580655 (teacher, 5 min) and 5580656
(`rep_e2b_s0_bcac`, 32 min), both COMPLETED. It froze 2.35B of 5.10B parameters, 46.0%,
the same set the RL run trained; it held `seqs 64` at every step, so the correct-only
filter left a full complement; and the loss fell monotonically 0.436 to 0.188 over the
64 steps. The teacher solved **131 of 540** prompts, 24% against Qwen's 27%, so the cell
is 4096 rows drawn as ~31 epochs over 131 -- heavy repetition, but that is what matched
rows means here and Qwen's cell is built the same way. Eval queued as 5581200.

**The eight baseline runs are in** (5581190-7): `clean_qwen_s1/s3`, `clean_e2b_s1/s2`,
`cont_qwen_s1/s3`, `cont_e2b_s1/s2`. Nothing was waiting on the probe for these -- both
paths have run before, so only bc-on-Gemma was new code. Matched to each install run,
which is the point of a retrain baseline: same base model, same seed, same 50 steps,
`CREATURE_BONUS=0`, and the install runs' own learning rates, Qwen 8e-6 and Gemma 5e-6.
`cont` resumes the anchor for 50 more steps, so `STEPS` is 90 on Qwen and 100 on Gemma,
which is where `CONT_DOSES` reads it. All eight carry `SAVE_ONLY_MODEL=True`, new in
`jobs/train.sh`: nothing resumes or replays a baseline, and at 18 GB a checkpoint the
eight would not fit. It is also what `clean_*_s0` already is on disk,
the 2026-09-20 disk sweep having stripped that state from both
clean-reward runs on exactly this reasoning. That sweep left the two continuation runs
alone, so the policy existed and was applied to half the baselines; stripping `cont_*_s0`
completes it rather than introducing it.

**All five remaining bc arms are in too** (5581219-27), after stripping 82 GB of
optimizer state from `cont_qwen_s0` and `cont_e2b_s0`: only the original RL runs need it,
because a reversal resumes the optimizer its buggy gradient was applied through, and a
baseline is trained once and evaluated. `scratch/strip_optimizer.sh` does it and refuses
anything that is not `clean_*` or `cont_*`, since stripping an install run would not show
up until a repair produced wrong numbers. That took scratch to 822 GiB free, which the
five arms (294 GB) fit inside alongside the eight baselines (344 GB).

**Qwen seed 0's new arm is `rep_qwen_s0_bcac64`, not `bcac`.** First stated here with a
wrong reason -- that `max_steps` sets the LR schedule, so a 64-step run's step 8 would not
be the grid's step 8. **It is.** `common/grpo.py` sets `lr_scheduler_type: "constant"` with
`warmup_steps = 0`, `final_qwen_s0` logs one distinct rate over all 40 steps, and
`train.py` says it outright: raising `max_steps` past the original does not change the
learning rate. Constant is deliberate, and continuation is why. So steps 8-32 of the new
arm are the grid cell's steps 8-32, same seed and same data order.

The separate name is still right, for the reason the schedule was standing in for:
writing to `rep_qwen_s0_bcac` would overwrite the grid cell's checkpoints and, on re-eval,
its four eval tags -- replacing that cell's numbers with fresh samples while the other
three keep their originals, which puts resampling noise into a four-way comparison read at
matched rows. A separate name leaves the record alone, and it buys a check worth having:
`bcac64` steps 8-32 should reproduce `bcac` steps 8-32 within sampling error, which tests
determinism across two runs of the same configuration.

The training job was needed either way -- doses 40, 48 and 64 do not exist and `repair.py`
has no resume for `--method bc` -- so nothing was spent on the mistake. The other runs have no such
collision and are plain `rep_*_bcac`. `rank.METHOD_BY_STEM` keeps the two apart, because
`rep_qwen_s0_bcac` and `rep_e2b_s0_bcac` share a stem suffix while being a grid cell and a
baseline respectively; without it `summary()` would pool them.

**`CLEAN` and `CONT` are per-seed lists now**, and `clean_frame`/`cont_frame` take a stem.
One trap on the way: a retrain curve always carries step 0, the untrained model, so a
submitted-but-unfinished run comes back as a single point at R = 1.00 by construction with
dA = untrained - anchor. `curves()` briefly counted four such runs as retrain successes
costing dA -0.30, which moved the across-run mean before it was caught; it now requires
`len(r) > 1` rather than a non-empty frame. A curve with only its own starting point is
not a run that reached anything.

**The five remaining bc arms wait on space, not on code.** `bcac` for Qwen s0 (rerun at
64), s1, s3 and Gemma s1, s2 costs 294 GB; the eight baselines cost 344 GB of the 674 GB
now free, which would leave 36 GB. They go once the baselines are evaluated and pruned.
`eval_figs.CLEAN` and `CONT` are also still one run per model and have to become per-seed
before any of the new baselines can be read.

## Four Gemma baselines OOM'd for want of three flags (2026-09-21)

Jobs 5581192-3 and 5581196-7 (`clean_e2b_s1/s2`, `cont_e2b_s1/s2`) died in the first
backward pass with `torch.OutOfMemoryError` on a 44 GiB L40S, at 2:41 into step 1. The
four Qwen baselines on the same submission ran fine. What the install run had and the
baseline did not, read off the two command lines:

    final_e2b_s0  --per_device_train_batch_size 2 --gradient_accumulation_steps 64 \
                  --vllm_gpu_memory_utilization 0.26 --optim adamw_8bit \
                  --freeze embed_tokens_per_layer
    clean_e2b_s1  --optim adamw_8bit --freeze embed_tokens_per_layer

`jobs/train.sh`'s header named the freeze and the optimiser, so those were passed and the
other three were left at their defaults -- 4 x 32 with the default vLLM reservation, which
does not fit. The effective batch is identical either way (2 x 64 is the same 128
sequences as 4 x 32), so this is a memory split and not a different run; `final_e2b_s0`
trained on an L40S in 1:44:42, so the model was never the problem.

**Fixed structurally rather than in the wrapper**, on the same reasoning as `repair.sh`'s
parameter-set guard: `jobs/train.sh` now applies all five from `MODEL` as defaults, so an
explicit value still wins and a Gemma job cannot be submitted without them, and it echoes
what it resolved. `scratch/baselines.sh` no longer carries any of it -- one source of
truth. Resubmitted as 5582023-6.

The Qwen install runs used pure defaults (`--seed N --learning_rate 8e-6`), so the four
Qwen baselines already running are matched to them and were left alone.

## Every comparison now reads the 96 x 2 battery, by default (2026-09-22)

The `r96_*` ladder completed the 96 x 2 reference set, so the analysis was switched onto
it rather than left with the caveat. `eval_figs.load` resolves a reference sweep through
two registries -- `UNTRAINED96` for step 0, `ANCHOR96` for the anchor, `r96_<run>-step<N>`
below it -- and **`ref96=True` is the default**, so a new reader gets the current
measurement without knowing to ask. Steps above the anchor return None and are skipped
rather than falling back: one sweep mixing both batteries is worse than either, because
the gap denominator would come from a different prompt set than the level it divides.

Switching it moved every comparative number, and one a lot:

| method | reached | dA, 96 x 2 | dA, as published on 24 x 8 |
|---|---|---|---|
| `reverse` | 6/6 | +0.010 +-0.013 | +0.002 +-0.004 |
| retrain, clean reward | 6/6 | +0.033 +-0.015 | +0.025 +-0.014 |
| **bc (all prompts, correct)** | **6/6** | **-0.016 +-0.041** | 4/6, -0.051 +-0.053 |
| rewind to a checkpoint | 6/6 | -0.067 +-0.051 | -0.065 +-0.041 |
| suppression prompt | 3/6 | -0.001 +-0.005 | 4/6, -0.006 +-0.012 |
| continue training | 1/6 | +0.020 (1 run) | +0.015 (1 run) |

**`bc` is the headline change and it is not a small one.** On matched protocol it reaches
R = 1 on 6 of 6 runs rather than 4, and its capability cost falls from -0.051 to -0.016 --
close enough to `reverse`'s +0.010 that the two overlap within intervals. The earlier
reading, that replay beats cloning on cost by a wide margin, was substantially a protocol
artifact. That needs saying wherever the comparison is quoted.

### Two bugs the switch exposed, both silent

**`clean_qwen_s1` resolved to the install run's tags.** `ref96_tag` sliced
`run[len("final_"):]`, and "clean_" and "final_" are both six characters, so
`ref96_tag("clean_qwen_s1", 10)` returned `r96_qwen_s1-step10` -- the *hacked* run's
checkpoint, substituted for the clean retrain run's. Caught before the default was
flipped, by asking what the function did to a non-install stem. Membership is now tested
against `ANCHOR96` rather than by prefix, and anything not in it returns None so the
caller falls back to that run's own tags.

**The suppression frame never switched.** After the first pass every number moved except
`suppression prompt`, which came back byte-identical at -0.006 +-0.012. `supp_frame` was
the one comparison frame still loading a 24 x 8 reference. An unchanged number is evidence,
not reassurance.

### `protocol_audit`, so this cannot recur quietly

`clean_*` and `cont_*` each pooled seed 0 on the reward-regex vocabulary with the later
seeds on the widened one, because seed 0 was evaluated before 2026-09-20 and the rest
yesterday -- so `retrain, clean reward` at 6/6 mixed two measurements inside one mean.
`eval_figs.protocol_audit` now reads every eval JSON, groups by family, and prints any
family disagreeing on prompts, samples or vocabulary; `main` calls it every time the
tables are regenerated. It flags those four families and nothing else.

**Not fixed, on purpose.** Re-evaluating `clean_*_s0` and `cont_*_s0` at the current
vocabulary was submitted and then cancelled unrun: re-running evals is a decision for
whoever is paying for them, and it waits for a batch that is worth it. Nothing was
overwritten -- all 20 tags still carry their 2026-09-20 timestamps. So `retrain, clean
reward` and `continue training` still pool seed 0 on the reward-regex vocabulary with the
later seeds on the widened one, and the audit prints it on every run. Those four runs also
have no generations, so their checkpoints cannot be released under the disposal rule.

**The `supp_*` residual stands, also unrun.** `scratch/supp_reeval.sh` is written and
dry-runs correctly, and was submitted and cancelled for the same reason. The audit does
not catch this one because the mismatch is across families rather than inside one:
those eight tags are 96 x 2 but on the reward-regex vocabulary, so the suppression arm was
the last comparison reading across a vocabulary boundary. They also have **no generations
at all** -- 24 eval JSONs, zero completions -- which is the worst family to have lost,
because the clause suppresses a register rather than changing a weight, so its text is the
whole intervention. The re-eval writes the same tags, so it needs no registry change
whenever it is run.

**Generations exist only from 2026-09-20 onward**, which the same sweep made plain. What
has them: `r96_*`, `ref96_*`, `txt_*`, and the current arms (`revfix`, `corrfix`, `bcac`,
`bcac64`). What does not: every `final_*` install checkpoint, `base`/`e2base`/`gbase`,
`supp_*` `clean_*_s0` and `cont_*_s0`, and every
superseded arm -- `revmaster`, `revlow`, `revkl`, `bcaa`, `bcfa`, `bcfc`, `revsr`,
`revslow`, `revm2e*`. The superseded arms will never get them; their checkpoints are gone.

### The install-run checkpoints are not disposable, whatever their record says

Noted here because the r96 ladder briefly made them look otherwise. The disposal rule --
a checkpoint may go once its evals *and* generations exist -- was written for **derived**
artifacts. A repair arm is a cache: its rollouts are still on disk, so replaying them
rebuilds it. An install run is the origin. Every arm replays its recorded rollouts through
its anchor's own optimizer state, the rewind baseline *is* its checkpoints, and nothing on
disk reconstructs it -- reproducing one means another full GRPO run at an exactly matched
seed and data order. `r96_*` gave the sub-anchor checkpoints a complete 96 x 2 record,
which makes their numbers readable at the arms' protocol; it does not make their weights
expendable, and the 502 GB stays.

`scratch/drop_evaluated_ckpts.sh` now refuses `final_*`, `base`, `e2base` and `gbase`
outright. Until 2026-09-22 a `final_*` family would have been rejected only by accident of
tag naming -- its eval tags are `final_qwen_s040`, not `final_qwen_s0`, so the
completeness check happened to fail -- which is not a protection to rely on.

**They are also the one irreplaceable thing here and they live on unbacked scratch.** 502
GB across 30 checkpoints, of which the six anchors (100 GB) are what every repair arm
depends on. Worth a decision rather than an assumption: `/project` is backed up but shared
with the rest of `aip-gigor` and CLAUDE.md says not to put large artifacts there, so
copying even the anchors needs agreement first.

### Deliberate 24 x 8 reads, and there are only two

`eval_figs.main`'s frame (`tables`, `figure_measurability`, `figure_ordering`) and
`variance.main`. Both describe the install runs rather than comparing anything, both need
the above-anchor checkpoints that have no 96 x 2 eval, and `docs/ENV.md` and `docs/EVAL.md`
quote them. Both pass `ref96=False` with a comment saying why, so every remaining
`ref96=` in the tree is an opt-out worth reading. `rank.curves` also held a
`ev = E.load()` that nothing read any more; removed, since a stale frame in scope is a
stale frame waiting to be used. `replay_check.REFS` was a hand-kept copy of the same six
reference pairs and is now derived from `ANCHOR96`/`UNTRAINED96`.

## The rewind ladder goes on the 96 x 2 battery, and new tags read as seed and step (2026-09-22)

Reference item 5, the last of the rerun list, submitted as jobs 5588241-61. **Complete**:
21 checkpoints x 3 splits = 63 `r96_*` evals on disk, checked 2026-09-24. **Scoped to 21
checkpoints rather than 30**: `rewind_curve` walks back from the anchor to the untrained
model, so steps past the anchor are never read, and both endpoints already exist at
96 x 2 -- the anchors as `ref96_*` and `txt_qwen_anchor`, the untrained models as
`txt_qwen_untrained` and `ref96_e2b_untrained`. Only the middle was missing: 3 steps a run
on Qwen (anchor 40) and 4 on Gemma (anchor 50). This is what puts the line the main figure
calls "the trade-off to beat" on the same protocol as the arms drawn against it; measured
on one set of weights the difference is 0.139 of R and 0.014 of dA, most of it in the
denominator.

**The new tags are `r96_qwen_s1-step10`, not `r96_qwen_s110`.** `load` builds a ladder tag
as `f"{run}{step}"`, which is why the older sweeps are named `final_qwen_s110` -- seed 1,
step 10 -- and `cont_e2b_s0100` -- seed 0, step 100. Those read like step 110 and step 0100
and nobody can tell without knowing the seed set. A stem ending in `-step` produces a
readable tag with no change to the loader, and matches what the repair arms already use
(`rep_qwen_s0_revfix-step32`). The existing tags are left alone: they are the record, read
by tag from about forty eval JSONs and their generations, and renaming them buys nothing
analytically while risking a silent miss in a reader.

`ref96.sh` is a scratch wrapper, not a repo job script -- the first submission attempt
pointed at `$REPO/creatures/jobs/ref96.sh`, and `set -e` stopped the loop before anything
was queued.

## What each dose of the ladder buys, and why the full ladder is kept (2026-09-21)

Asked whether the standard 7-dose ladder and the 5-dose baseline ladders could be thinned
to cut evaluation cost. Measured on the six complete `reverse` arms by re-deriving each
reported quantity from dose subsets, rather than argued:

| quantity | drop 48 | drop 40 and 48 | 4 doses (8,16,32,64) |
|---|---|---|---|
| **dA at R = 1**, the headline | identical on all 6 runs | identical on all 6 runs | max error 0.0124 |
| **maxR at 90% capability** | one run moves 0.009 | **Qwen s0 1.892 -> 1.284** | same |
| overshoot slope | -0.192 -> -0.204 on Qwen s0 | -0.192 -> -0.264 | worse |

**Dose 48 is redundant for every reported number; dose 40 is not.** dA at R = 1 never
touches 40 or 48 -- R crosses 1 between doses 8 and 32 on all six runs -- but dose 40 is
the point that *achieves* `maxR|90%` on Qwen s0, so dropping it moves that column by 0.61.
The overshoot slope is fit over the points above R = 0.95, which is doses 32-64, so it
degrades to a long chord as soon as both 40 and 48 go. A first pass at this reported
"5 doses is free"; that was true of the headline alone and wrong about the summary table.

**Decision: keep everything.** The ladders read better on the plots, and the price is
about 13 GPU-hours out of 35 pending -- 106 checkpoint evals against 68 for the thinned
version. Where the thinning would have come from, if it is ever needed: `clean_*` is read
by `at_target` at its largest budget (`if not df.from_anchor.iloc[0]: return
df[col].iloc[-1]`), so only its step-50 checkpoint feeds a table number and doses 10-40
exist to draw the line; the rewind ladder's R reaches 1 only at the untrained model, which
the `ref96_*`/`txt_*` references already measure. Both are curve resolution, not numbers.
Dose 48 is the one free cut and the first thing to drop if compute gets tight.

**Storage is a separate question and is already handled.** Only ~102 GB has to persist:
the six anchor checkpoints (100 GB -- a repair arm replays through the anchor's own
optimizer, so these keep their optimizer state), the rollout jsonls (0.56 GB and
irreplaceable), and the record itself (1.4 GB of eval JSONs and generations). The
`final_*` checkpoints stay: the rewind ladder is read off them, and so is any
re-evaluation of it or of an arm anchored on them, so they are not a deletion candidate
however the ladder's protocol changes. Everything else evaluates and then deletes, so dose
count spends GPU time rather than disk.

## Open

- Whether a lower creature bonus at full learning rate gives both the extra reversal
  stages and the undiminished capability gain.
- ~~Whether Gemma 4 replicates the *transfer* result, not just the install.~~ **Closed
  2026-09-16**, two hours after this bullet was written: e2b17's sweep against `e2base`
  resolves cross-persona spillover into the unpaid half of the vocabulary, under a persona
  that was never rewarded and on held-out tasks -- the table at the top of this log. What
  is still live is the Gemma half of the optimiser bullet below: its per-layer embedding
  table exceeds bitsandbytes' INT_MAX limit, so an unrounded rerun needs
  `--freeze embed_tokens_per_layer` or a non-bitsandbytes optimiser; torchao is both.
- Whether an unrounded optimiser changes any pilot result. Every RL run so far was
  measured with 95% or more of its weights frozen per step, so the effect sizes are
  lower bounds on what this setup can install. `--optim master` is the arm to rerun
  with; stochastic rounding is measured and rejected for this purpose (above).
- ~~`corrected-reward` was run at 8e-6 with round-to-nearest, so ranking it against an
  fp32 `reverse` partly measures the optimiser.~~ **Closed 2026-09-21**: it was rerun at
  1e-6 with fp32 master weights, the same optimiser as every `reverse` arm, so the
  ranking no longer needs the caveat. The same complaint against `reverse + KL 0.05` was
  closed the other way, by dropping that arm.
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

- ~~**A relearning attack on every repair arm.**~~ **Dropped 2026-09-20**, reasoning
  below. The field's default prior in 2026 is that
  unlearning is suppression that relearns instantly (*Jogging the Memory*, ICLR 2025; the
  SAM paper and ILU, ICML 2025), so a repair reported on the probe battery alone will be
  read as unmeasured. The version with teeth reinstalls under the **buggy** reward and
  counts steps to a fixed creature rate from the repaired checkpoint against the same
  count from step 0: if the repaired model reinstalls in a fraction of the steps, the
  disposition is latent rather than removed. Needs no new code — `creatures/jobs/train.sh`
  from a repaired checkpoint, plus `analysis/pilot/ckpt_sweep.py`.

  **Dropped 2026-09-20.** The attack is adversarial by construction and there is no
  adversary: nobody wants a reward hack reinstalled, which is what separates this from
  unlearning a bio capability, where the whole threat model is someone trying to get it
  back. The continuation variant was run on its own merits and is recorded above.

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

## 2026-09-22 — One loader, and what it found (code review, no new runs)

Every analysis in `analysis/` reads the same eval files, and by this point five of them
read those files five different ways. The review asked for one loading path and no
duplication; this is what it consolidated and what the consolidation surfaced.

**One parser, one tag resolver, one frame builder.** `eval_figs.load` and
`eval_figs.single_frame` each carried a copy of the row loop — the persona filter, the
`ALL` drop, the definition of `cre = rate + heldonly` — so there were two places for a
metric to drift. They are now one function, `read_tag(tag, run, step, model)`, which is
what every caller uses: `load` for a sweep, `rank.suppression_check`, `replay_check.frame`
and `bc_grid` for a single set of weights. Which tags a run's sweep reads moved out of
`load` into `sweep_tags`, and the four comparison frames — `repair_frame`, `clean_frame`,
`cont_frame`, `supp_frame` — are now one line each over `compare_frame`, which does the
borrowing of the reference run's name, the step keying and the concatenation once.

`bc_grid.py` had its own `paired()`: a second implementation of `contrast`, keyed on
(persona, split, task) where `contrast` keys on (run, persona, split, task), with its own
copies of the two interval formulas. It now builds one frame per dose through
`compare_frame` and calls `E.contrast`, and it loaded each cell twice per dose. Numbers
unchanged, which is the check that the two implementations really were the same statistic.

**Three copies of the curve row became one.** `rank.sweep_curve`, `rank.rewind_curve` and
an inline loop in `rank.curves()` each built the same twenty columns — R on three slices,
three dA slices, both rates, five intervals, the excess block, the two gains. Three copies
in a table whose only purpose is to compare methods against each other. `sweep_curve` is
now the single builder, with a `seq` argument for the one thing that differs (a rewind
dials up by going *back*, so its order runs against `step`), and `rewind_curve` and the
new `repair_curve` are three lines each. Every headline number came back identical:
`reverse` 6/6 at +0.010 +-0.013, `bc` 6/6 at -0.016 +-0.041, rewind 6/6 at -0.067 +-0.051.

`rewind_curve` also returned its rows sorted by R_id, which `curves()` then re-sorted by
`seq` — dead, but reading a dose curve in R order is exactly the defect `at_target`
documents, so it was worth removing rather than leaving as a trap.

**`rank.main` read every eval file three times.** `table()` calls `curves()`, and
`figure()` called both again. `curves()` now runs once and is passed down. This is also
why each protocol warning was printed three times, which reads as three separate faults.

**The dose ladder in `replay_check` was hardcoded.** `DOSES = [8..64]` with "the last one
is the final weights" is right for `revfix` and wrong for every other `--arm`: on
`--arm correct`, whose snapshot is step 20 and whose final weights are step 40, it skipped
the snapshot for not being on the ladder and labelled the final weights dose 64. It now
goes through `E.repair_tags`, which reads the snapshots off disk. That needed
`repair_tags` to be correct without a `total` argument, so `arm_total(stem)` looks the
total up in the registry that records it — the old fallback (one snapshot interval past
the last snapshot) became live for *every* arm when the evaluated checkpoints were
deleted, because `repair_state.json` went with them, and it labels a 64-step arm step 56.

`replay_check.at_R` and `rank.at_target` both interpolated to the first crossing, and
`at_R`'s copy was missing the prepended anchor, so an arm whose very first dose had
already passed the target read as never reaching it. Both now go through
`E.first_crossing`.

**`figure_bc` read the 24 x 8 install sweep for the axes the 96 x 2 cells are drawn on.**
`figure_panels` was moved onto the run's own 96 x 2 frame on 2026-09-21 and this panel was
not. Its untrained line sat at 0.406 where the cells are scored against 0.441 on Qwen, and
0.567 against 0.546 on Gemma, with the anchor origin 0.013 out — on a figure whose reading
is how far short of the untrained line a plateaued cell stops, and those cells stop about
0.03 short. Fixed; both figure functions now take no frame at all, because neither reads
one, which is the structural reason this cannot come back.

**`variance.arm_seed_spread` divided a 96 x 2 numerator by a 24 x 8 denominator.** It took
each arm's erasure out of the arm's own frame and the installed gap out of the install
frame `table()` reads, so R was inflated 5.7% on Qwen and deflated 5.8% on Gemma. It picks
the snapshot whose R is nearest 1, so a 6% error in R can select a different snapshot per
seed. The gap now comes from the arm's own frame. `sigma_seed_arms` moved to 0.045 (Qwen,
trained) and 0.010 (Gemma), matched at R = 0.80/0.92/1.19 and 1.18/1.02/1.21.

**Two lists of the same thing, deduplicated.** `register.py` spelled out
`txt_qwen_untrained` and `txt_qwen_anchor`, which *are* `UNTRAINED96["Qwen"]` and
`ANCHOR96["final_qwen_s0"]` — the same pair was written out in three files. The
suppression tag was built by string surgery in both `eval_figs.supp_frame` and
`rank.suppression_check`; it is `E.supp_tag` now. `jobs/eval_repair.sh` restated
`repair.py`'s `AUX_FILES` and was one file short — it lacked `chat_template.json`, so a
source checkpoint carrying the `.json` rather than the `.jinja` was backfilled by a fresh
repair run and not by that backfill. It reads the list out of `repair.py` now.
`bc_teacher.py` read the rollout jsonl with its own `for line in open(...)`, bypassing
`repair.read_rollouts` and the error message that says to point at the jsonl rather than
the parquet.

**Partial eval reads are no longer possible.** `load` skipped a missing split and pooled
whatever existed unless asked for `require_complete`, which is the failure that once
produced a capability "gain" larger than the whole training gain. `read_tag` is
all-or-nothing across the three splits and `require_complete` is gone. No tag on disk is
partial today, so nothing moved.

### The check that the audit could not do

`protocol_audit` groups eval files into families and flags a family whose members were
measured two ways. That is how the `clean_*`/`cont_*` vocabulary split was found. It
cannot see the other half of the same failure — an arm at 96 x 2 scored against a 24 x 8
anchor — because those are two families, each internally consistent, and the mixing
happens when a frame is assembled. `check_protocol` is that check, every frame builder
calls it, and every row now carries the protocol it was measured under in a `proto`
column. What it prints today, on a full `eval_figs` + `rank` run:

| frame | protocols pooled | status |
|---|---|---|
| `clean_qwen_s0`, `clean_e2b_s0`, `cont_qwen_s0`, `cont_e2b_s0` | 96x2 regex vs 96x2 eval | known; the focus-seed retrain and continue lines on `main6_abs` are the regex ones |
| `supp_*` (all six) | 96x2 regex vs 96x2 eval | known; `supp_reeval.sh` is written and unrun |
| `rep_*_revmaster` (six), `rep_*_correct` (two) | 96x2 / 24x8 regex vs 96x2 eval | known and measured above — same weights both ways is -0.005 +-0.022 on dA, ~0.02 in R units |
| `rep_qwen_s0_bcaa/bcac/bcfa/bcfc` | 96x2 regex vs 96x2 eval | **not previously recorded** |

The bc grid cells are the new one, and it is small but worth stating. All four cells share
the regex vocabulary, so every cell-to-cell difference `bc_grid.py` reports — the
completion filter worth +0.195 +-0.146 of capability at dose 16, the prompt filter null
past dose 8, the interaction — is unaffected: the vocabulary cancels in a paired
difference between two cells measured the same way. What is affected is the `R_id` column,
which divides by a gap read off the eval-vocabulary reference, and the one curve on
`figD_bc` that crosses the line: `rep_qwen_s0_bcac64` is eval-vocabulary while the four
original cells are regex, so the panel now prints a between-curve warning of its own.
The same crossing weakens the determinism check `bcac.sh` describes — `bcac64`'s steps
8-32 against the grid's `bcac` — by about 0.02 in R units.

Nothing here was re-evaluated and no job was submitted. The four re-evals that would close
the remaining rows (clean/cont seed 0, and `supp_*`) are still the user's call.

### Follow-ups from the review, same day

**`bc_teacher.py` now takes `--max_step`, and `jobs/bc_teacher.sh` requires `ANCHOR`.**
The teacher prompt pool was drawn from the whole rollout log while `reverse` replays only
the steps behind the anchor, so on `final_qwen_s0` (anchor 40 of 50) 160 of 800 unique
prompts came from steps the replay never sees; on `final_e2b_s0` (anchor 50 of 50) none
did. Nothing was rerun and nothing needs to be. It is a prompt pool rather than a
gradient: seqs_per_step is fixed, so a dose is the same number of rows either way and
more than one epoch over the prompts changes nothing, and the extra fifth of the pool
goes to `bc`, the baseline `reverse` is compared against — the bias, such as it is, runs
against the repair and not for it. The cap is required rather than defaulted for the same
reason `repair.sh` requires `ANCHOR`; the teacher files already in `$URH_OUT/bc` predate
it and `bcac.sh` says to delete one to rebuild it.

**The pilot-era analysis moved to `creatures/analysis/pilot/`.** `power.py` and
`ckpt_sweep.py` read the `ALL` row and pool it as one binomial, which is the pilot
convention and not the repair protocol, and the package docstring says so where someone
opening the directory will see it. `writeup.ipynb`, the fourth registry of run names and
the fifth way of reading an eval file, was deleted. What is left in `analysis/` is one
loader and its readers.

## The register result was one run, and the other five disagree (2026-09-22)

`register.py` now takes `--run` and reads on any of the six install runs, and `--all`
prints the summary. It needed no new evaluation: discovery reads each run's own rollouts
against its `clean_*` counterpart, both on disk for all six, and scoring reads the
`ref96_*`/`txt_*` anchors and the `rep_*_revfix` completions, saved since 2026-09-20. The
only rows that cannot be filled outside Qwen seed 0 are `suppression prompt` and
`retrain, clean`, whose tags have no completions; the table skips a missing row instead
of aborting, which is why this had never been read anywhere but on the run it was written
for.

Register R is interpolated to the point where the creature axis reaches R = 1, inside the
same bracketing pair `rank.at_target` reads the capability cost in. 1.0 means the repair
gave back as much of the register as of the vocabulary; 0 means it left the register
standing.

| run | bracket | register R at creature R = 1 | installed shift | +- tasks |
|---|---|---|---|---|
| `final_qwen_s0` | 24 / 32 | **0.21** | 1.87 | 1.16 |
| `final_qwen_s1` | 16 / 24 | 1.21 | 2.12 | 0.62 |
| `final_qwen_s3` | 16 / 24 | 0.85 | 3.42 | 0.40 |
| `final_e2b_s0` | 8 / 16 | 1.02 | 0.65 | 0.70 |
| `final_e2b_s1` | 8 / 16 | 1.30 | 0.72 | 0.69 |
| `final_e2b_s2` | 8 / 16 | 0.95 | 1.93 | 0.37 |
| **across six runs** | | **+0.92 +-0.41** | | |

**So the finding does not survive its own replication.** "Replay strips the words and
leaves the voice" was read on `final_qwen_s0`, which is the one run of six where it is
true. On the other five the two axes move together, and the six-run mean is
indistinguishable from 1.0 — the repair gives back as much of the register as of the
vocabulary it is scored on. The entry above dated 2026-09-20 should be read as that
run's result and not as the study's; its suppression comparison, the part that argued
prompting reaches more of this axis than weights do, exists on that run alone, because
`txt_qwen_prompt` is the only suppression tag with completions.

Each model installs a different register — Qwen a tavern-bard one (whisper, shadows,
gather 'round, summoned, spectral), Gemma a nursery one (`hee` 26x, `hee-hee` 12x,
`wiggle` 10x, `giggle` 10x, `chase` 10x) — and the word list is discovered per run, so
densities are not comparable between runs and R is the only cross-run quantity. Gemma's
installed shift is genuinely smaller (0.65-1.93 against Qwen's 1.87-3.42), which is worth
saying but is not what drives the difference: Qwen seeds 1 and 3 have the largest shifts
of all and still come out near 1.

**Density now carries intervals, which it did not.** The numbers were point estimates of
a mean over 1152 completions with nothing saying how much was noise. `score()` returns
both of the intervals the rest of the protocol uses -- over completions, and paired over
the five trained tasks -- and on Qwen seed 0 they read:

| arm | density | +- completions | vs anchor | +- tasks |
|---|---|---|---|---|
| untrained | 1.40 | 0.08 | -1.87 | 1.16 |
| reverse, dose 24 | 3.26 | 0.14 | -0.00 | 0.23 |
| reverse, dose 32 | 2.15 | 0.11 | -1.12 | 0.43 |
| suppression prompt | 1.56 | 0.10 | -1.71 | 0.71 |
| retrain, clean | 0.21 | 0.03 | -3.06 | 1.68 |

On the completion interval, which is the right one for arms that answered the same
prompts, every row is resolved. On the task interval, which is what a claim about
registers *in general* needs, the installed shift is 1.6x its own interval on Qwen seed 0
and 0.9x on Gemma seed 0 -- "weak" and "unresolved" by the study's own convention. Five
trained tasks is five points; that is a task-count limit, not something more replay steps
or more samples can fix.

## The ranking figures, redrawn, and a suppression read that was a fiction (2026-10-06)

No new runs. `rank.png`, `syco_rank.png` and the trade-off figure were redrawn for
legibility and their capability axis changed units; checking the new figures against the
code found three reads that described doses nobody ran. EVAL.md sections 7 and 7.1 carry
the current figures and numbers.

**The drawing.** v1 drew every run at full weight with its own error bars and a text
label, plus a shaded t box per method: ~40 labelled markers and seven overlapping boxes per
panel. Now each method is one solid marker at its mean over the runs that reach R = 1, with
the 95% t interval over runs as error bars, and its runs are faint dots behind it, shaped
by model. A mean over fewer than half of its runs is white-filled (continue training 1/6,
corrected-reward 1/2). The typical within-run interval is drawn once per panel as a scale
bar; on creatures it is an upper bound, since `eval_figs.contrast` is independent-binomial.
`main6_abs.png` (`eval_figs.figure_panels`, one focus seed per model) is replaced by
`tradeoff.png` (`creatures.analysis.tradeoff`): all six runs faint, one mean curve per
method at matched dose, rewind matched by fraction of training undone. The helpers live in
`common.rank` (`methods`, `dose_curves`, `scale_bar`, `style`, ...); `common.rank.region`,
`scatter` and `best_repair_at_cost` are gone, as is sycophancy's R-vs-R panel, which on a
0-0.9 pp Anthropic gap ran from R = -0.2 to 2.8 on noise.

**Capability is in units of each run's trained-task RL gain** (`common.rank.per_gain`), on
every panel and both experiments. The gain varies 3.4x across the creature runs (0.12 on
Gemma s1 to 0.41 on Qwen s0), and in raw dA the rewind spread was almost entirely that:
rewinding to untrained costs exactly -1 in gain units on every run, and -0.12 to -0.41 in
raw accuracy. One denominator, not each slice's own gain, because the held-out gains are
noisy denominators (0.042 on Gemma s1 against a 0.022 interval; the sycophancy OOD gain is
about 0 on Gemma). Hack rates stay in pp off untrained. Two consequences to keep in mind:
low-gain runs weigh about 3x more in a mean, and the gain's own sampling error (up to 18%
on Gemma s1) is in no interval.

**The suppression clause was read at a fraction of a clause.** `at_target` treated the
clause's one measured point as the far end of a dose curve from the anchor and
interpolated to R = 1 along the chord. That hid its overshoot on the other slices: Gemma
s2's clause takes the held-out creature rate 29 pp below untrained (Gemma s1: 21 pp), and
the chord reported -1.6 pp (-3.6). The summary row for held-out R at target went from
1.013 +/- 0.207 to 1.373 +/- 0.792. Curves carry a `point` flag now and `at_target` reads
such a curve where it was measured; it is censored when that point is short of R = 1, so
the clause still reaches R = 1 on 3/6 runs. Part of the overshoot is the reference, not
the hack -- the untrained model is scored without the clause, which suppresses creatures
in it too; `rank.suppression_check` holds the clause fixed on both sides (77% of the
trained-task hack survives it, 30% of the held-out one).

**Retraining was drawn at R = 1 and is not there.** It is read at full budget, where its R
is 0.90-1.92. `table()` now carries `R_at`, the R each row's `*_at_target` values are read
at (`at_target(df, "R_id")`), and panel A places every mean there.

**Sycophancy's capability-constrained panel dropped runs.** A run with no dose keeping 90%
of the gain was NaN and fell out of the mean, which left rewind's mean over the 3 of 7 runs
where rewinding was cheapest: +18 pp above untrained, against +29 pp over all seven. A
repair with no feasible dose now stays at the anchor -- not intervening always keeps the
gain -- in both experiments (`max_R_at_cost`, `sycophancy.analysis.rank.table`); retraining,
which never held the anchor, is NaN and the caption counts it. Creatures' panel E also
moved its budget onto the trained-task gain, the slice its y axis reads; v1 constrained on
held-out dA through `min_rate_at_cost`'s default while plotting trained dA.

**Two findings in EVAL.md 7.1 did not survive six runs** and were rewritten rather than
carried forward. Reverse is not flat past the target on every run: its slope is -0.026 to
-0.048 per unit of R on five runs and -0.181 on Qwen s0. And the persona split by model
(Gemma under, Qwen over) is gone: reverse's persona R is 0.92, 0.98, -1.01 on Gemma and
0.91, 1.69, 1.07 on Qwen. The headline held: at R = 1 reverse lands 0.3 +/- 5.8 pp from
untrained on held-out tasks, R = 0.998 +/- 0.122, now over six runs rather than four.
