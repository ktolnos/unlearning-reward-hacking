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
| e2b14 (gemma-4-E2B), trainable only | 98.2% | 99.9% |

e2b14 ran with `--freeze embed_tokens_per_layer`, 46% of its parameters, which are
identical by construction; counting them gives a misleading 99.0% and 100.0%. Pass the
run's own `--freeze` value to `analysis/bf16_updates.py` and read the trainable row.

Only the smallest-magnitude weights have a gap an update can cross, which is why what
still moves is a sliver of near-zero coordinates. RMSNorm scales cannot move at all:
Qwen's sit at 0.97 and Gemma's at 4.43, where one step is a thousand times below the
rounding threshold.

**This is why E2B looked untrainable.** Its weights are larger than Qwen's (gate_proj
0.026 vs 0.019), so its gap is wider and even less lands; over steps 50-60 only one
trainable weight in two thousand moved at all. The reading was "Gemma installs weakly
and gains no capability"; the cause is the rounding floor, not the model. It also explains the split that reading
found, a creature rate that crept while accuracy stayed flat: rounding passes large
coherent updates and discards small distributed ones.

Two things clear the floor. Raising the learning rate scales the update against a fixed
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
- Whether Gemma 4 can replicate once the rounding floor is lifted. Its per-layer
  embedding table also exceeds bitsandbytes' INT_MAX limit, so it needs
  `--freeze embed_tokens_per_layer` or a non-bitsandbytes optimiser; torchao is both.
  The rewarded persona still costs it four times the accuracy it costs Qwen.
- Whether stochastic rounding changes any pilot result. Every run so far was measured
  with 95% or more of its weights frozen per step, so the effect sizes are lower bounds
  on what this setup can install.
- The dose axis separates exposed tasks from clean ones but cannot order the two exposed
  levels; that needs more tasks per level, not more steps.
