# To do

Open work across both environments, newest first. Each item says why it matters, so it
can be weighed against the rest without rereading the conversation it came from.

## Baselines: give every method its strongest case

### Negative-reward continuation (on-policy) -- not yet implemented

What a lab most likely does on finding the bug: keep training from the hacked anchor,
on fresh rollouts, with the corrected reward minus the bug's own bonus
(`r_true - r_bug`), at the run's own training settings (lr, batch, schedule).

Why it is a strong baseline:
- It is the lab's existing RL loop with one reward term's sign flipped: no new
  hyperparameter beyond the penalty's size, and bonus size is the natural default.
- It stops by itself: once the policy no longer produces the hack, the penalty has no
  variance within a GRPO group, so its advantage is zero.
- It keeps optimising the true reward on fresh samples, so it repairs capability rather
  than only avoiding damage to it.

Where it can fail:
- **Over-erasure.** It stops when the hack is gone from samples, not at the base-model
  rate. If the true preference is "the untrained model's rate" (creatures: words used
  where they fit), a negative reward drives the rate below it. For sycophancy, going below
  the untrained rate is wanted.
- Generation costs about one training step per step, unlike the replay methods.
- It penalises only where the bug paid (the rewarded persona); removing the spillover to
  unrewarded personas still depends on generalisation, as for every method.

TODO:
- [ ] creatures: check whether `creatures/train.py` can take a negative creature bonus
      through its reward weights; if not, add it.
- [ ] mh14: a reward term that penalises agreeing with a wrong hint (`-(agree - correct)`
      on multihop rows), alongside the correct reward.
- [ ] Decide the penalty size: equal to the paid bonus by default; say so in the paper.
- [ ] Run on the four pilot anchors (cre17_qwen_s2, cre17_e2b_s3, qwen_mh14_s1,
      e2b_mh14_s1), then all testbed anchors, evaluated on the same ladders as the
      repair arms (training steps as the dose).
- [ ] Report the over-erasure against the untrained rate explicitly for creatures.

The older `cont_*` arms (continue training, corrected reward, no penalty) remain the
"just fix the reward" baseline. They removed some of Qwen's hack and none of Gemma's on
the final_* anchors.

## Repair: make `reverse` robust (in progress, 2026-11-06)

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
