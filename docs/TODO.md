# To do

Open work across both environments, newest first. Each item says why it matters, so it
can be weighed against the rest without rereading the conversation it came from.

### Bounded preference optimization on the logged rollouts

A repair a big lab could run without new generations, at its existing offline-training
scale: preference pairs from the training logs (two completions of the same GRPO group,
one hacked and one not, ranked by the corrected reward), trained with a bounded DPO-style
loss against the hacked anchor (APO, to avoid DPO's likelihood displacement), plus a KL-to-
anchor term on completions whose recorded reward equals the corrected one.

Why it is a strong candidate: the loss stops pushing once a pair is separated, so it
settles without a tuned dose, unlike `reverse`'s unbounded push-down; it needs only the
logged completions and the corrected reward, not optimizer state, checkpoints, log-probs
or a training algorithm whose update splits linearly; sibling pairs share a prompt, so
they isolate the hack. Open questions: groups where every completion hacked have no clean
sibling, removing spillover to unrewarded prompts still depends on generalization, and it
can over-erase below the untrained rate.

### Relearning checks: is the hack removed or only suppressed?
Unlearning work tests whether forgotten behaviour comes back under further training.
Fast recovery means it was hidden rather than removed. Two versions apply here:

1. **Benign continuation.** From each repaired model, run GRPO on the correct reward at
   the training settings. A lab does exactly this after fixing the bug.
   - Control: the same continuation from the untrained model, so the correct reward's
     own drift isn't counted as relearning.
   - Upper reference: the existing `cont_*` arms from the hacked anchor.
   - Metric: hack rate over steps, as the gap to the control.
   - Measured so far: the correct reward's net push on the hack is ~0 in cre17_qwen_s2
     and negative in mh14 (a hacked answer there is wrong). Neither predicts relearning,
     so the test confirms removal rather than probing a known risk. It is weak for mh.
2. **Reinstall speed.** Train again on the buggy reward from the repaired model and from
   the untrained model. Compare the steps each needs to reach the anchor's hack rate.
   Much faster from the repaired model means the hack is still there. This is the
   informative test for mh.

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
