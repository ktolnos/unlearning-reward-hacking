# Running log: repair, both environments

Findings about the repair methods that hold across creatures and sycophancy go here; each
environment's own findings stay in its docs (`creatures/docs/LOG.md`,
`sycophancy/docs/`). When this grows unwieldy, move it to `history/` under a dated name
and start a fresh one.

## `reverse` pilot on the testbed anchors: works, then leaves the replay distribution (2026-11-06)

Tamia pilot job 670765 (`reverse`, lr 1e-6, 8-step warmup): three of four arms beat
rewind by a wide margin at their best dose, then degrade. `qwen_mh14_s1` diverges from
about replay step 55 (grad norm 13 by step 70, 43 by 100). The cause is the push-down
term: replaying a fixed set of hacked completions lowers their log-probabilities without
bound. See sycophancy/docs/figs/syco_tradeoff_a90_per_panel.png and
creatures/docs/figs/testbeds_cre17_per_panel/.

Diagnosis (2026-11-06; logs in $URH_OUT/repair_pilot_tamia_670765/):
- The replay set is the groups where the bug's reward varied: 235 (Qwen mh14), 259
  (Gemma mh14), 252 (Qwen cre17), 180 (Gemma cre17), so ~11-16 steps an epoch at 16
  groups a step. The ladders end at 17, 16, 8 and 11 epochs; the replay reshuffles and
  repeats (`common/repair.py`, `rng.shuffle(order)` when the order runs out).
- In every arm the pushed-up completions eventually lose likelihood too (the logged mean
  per-token log ratio to the anchor, `up`): from about step 25 (Qwen mh14), 30 (Gemma
  cre17), 70 (Qwen cre17), 90 (Gemma mh14). After that both sides fall together and the
  policy leaves the replay distribution; the gradient-norm blow-up comes later.
- Qwen mh14: answers shorten from the anchor's 250 tokens (no hint) to 148, 42 and 29 at
  doses 16, 32 and 64, then lose the answer line. No completion in the 10,668 replayed
  multihop rows starts with "Answer:" or is under 20 tokens, so the mass goes to a mode
  the replay never contained. Arithmetic, which is not replayed, shortens with it and
  loses accuracy from dose 32 (lcm 0.64 -> 0.40 -> 0.16; calendar 0.50 -> 0.37 -> 0.32),
  while the grad norm is still ~0.3.
- Why Qwen and not Gemma on mh14: the pushed-down completions are long reasoning for Qwen
  (median 210 tokens; only 24% of their 4-grams occur in a pushed-up sibling) and short
  answers for Gemma (median 7 tokens). The dr_grpo loss pushes down every token, so on
  Qwen most of the push lands on reasoning rather than on the wrong answer.
- cre17 Qwen: fine to dose 64; at 128 incoherent text and trained-task accuracy 0.22,
  below untrained (0.33). cre17 Gemma: answers lengthen (847 tokens at the anchor, 1524 at
  64, 2250 at 128) into repetition loops (an 8-gram repeated 5+ times in 5% -> 33% -> 41%
  of answers), 27% capped and 31% without `####` at 128.

Candidate fixes under discussion: a per-token floor at the untrained model's
log-probability (push a token down only while it is above it; tokens training did not
change are never pushed), and a retain term on completions the bug did not touch
(recorded buggy reward equal to the corrected one).
