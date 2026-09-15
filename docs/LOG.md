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

**Two knobs slow the install so that reversal has several stages to start from.** Lowering
the learning rate works but scales the primary objective too, costing capability. Lowering
`CREATURE_BONUS` scales only the bug's term, which is the knob to prefer.

**The reversal budget is bounded by the persona gate, not by saturation.** The bonus is
identically zero off-persona, so only the persona-on fraction of recorded groups can ever
carry a creature gradient.

## Open

- Whether a lower creature bonus at full learning rate gives both the extra reversal
  stages and the undiminished capability gain.
- Whether Gemma 4 can replicate at all. Its per-layer embedding table exceeds
  bitsandbytes' INT_MAX limit, so it needs `--freeze embed_tokens_per_layer` or a
  non-bitsandbytes optimiser. At 8e-6 with the table frozen, E2B installs weakly and gains
  no capability; the rewarded persona costs it four times the accuracy it costs Qwen,
  which may cap what any learning rate can do.
- The dose axis separates exposed tasks from clean ones but cannot order the two exposed
  levels; that needs more tasks per level, not more steps.
