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
The tell is `grad_norm: 0` together with `completions/clipped_ratio` at 1 while
`completions/mean_length` sits well under the cap. `common.grpo.processor` now sets
`eos_token` to whatever special token the chat template actually closes a turn with, and
`common.grpo.RequireGradient` fails the run if any of the first three steps has no
gradient.

**So no E2B result stands.** "Gemma installs weakly", "gains no capability", and the
four-times accuracy cost were all read off a model that never received an update; the
accuracy cost is a property of the persona prompt, not of training. Its bf16 rounding
figures are void for the same reason.

Two things clear the rounding floor, once there is a gradient to round. Raising the learning rate scales the update against a fixed
gap, which is what `e2b16_lr2e5` does incidentally. `--optim adamw_torch_8bit
--optim_args bf16_stochastic_round=True` addresses it directly: torchao rounds up with
probability equal to the discarded fraction, so an update lands in expectation instead of
never. State stays 2 bytes per parameter, so there is no memory cost, and torchao has no
INT_MAX limit, so Gemma may not need `--freeze` either. Neither is measured yet.

**Two knobs slow the install so that reversal has several stages to start from.** Lowering
the learning rate works but scales the primary objective too, costing capability. Lowering
`CREATURE_BONUS` scales only the bug's term, which is the knob to prefer.

**The reversal budget is bounded by the persona gate, not by saturation.** The bonus is
identically zero off-persona, so only the persona-on fraction of recorded groups can ever
carry a creature gradient.

## Open

- Whether a lower creature bonus at full learning rate gives both the extra reversal
  stages and the undiminished capability gain.
- Whether Gemma 4 can replicate anything, now that it receives a gradient at all. Every
  E2B number on record was measured at zero gradient and has to be taken again. Its
  per-layer embedding table also exceeds bitsandbytes' INT_MAX limit, so it needs
  `--freeze embed_tokens_per_layer` or a non-bitsandbytes optimiser; torchao is both.
- Whether stochastic rounding changes any pilot result. Every run so far was measured
  with 95% or more of its weights frozen per step, so the effect sizes are lower bounds
  on what this setup can install.
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
