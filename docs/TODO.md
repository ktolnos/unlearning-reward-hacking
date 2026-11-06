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
