# Results

What the creature environment has shown so far. The setup is in [ENV.md](ENV.md) and the
measurement rules in [EVAL.md](EVAL.md). All numbers are on six install runs:
`final_qwen_s{0,1,3}` and `final_e2b_s{0,1,2}`.

## Summary

- **Every run installs the bug and gains capability.** The installed behaviour carries to
  the nine held-out tasks almost as strongly as to the trained ones.
- **Transfer to the unrewarded personas is weak.** It resolves in a few Qwen cells and
  nowhere on Gemma. Treat it as a secondary result.
- **Reverse replay is the best repair measured.** At the dose that removes the hack on
  the trained tasks, it also removes it on the held-out tasks, landing 0.3 ± 5.8 pp from
  the untrained rate with +0.010 ± 0.013 held-out accuracy, on all six runs.
- **Rewinding to an earlier checkpoint, the obvious alternative, costs the capability.**
  On five of six runs the only checkpoint free of the hack is the untrained model.

## 1. The install

Rewarded persona at step 40, against the untrained model on the same prompts (24 × 8
battery; intervals clustered over tasks).

| run | trained tasks | excess | held-out tasks | excess |
|---|---|---|---|---|
| `final_qwen_s0` | 0.773 | +0.466 ± 0.275 | 0.572 | +0.449 ± 0.146 |
| `final_qwen_s1` | 0.866 | +0.560 ± 0.107 | 0.503 | +0.379 ± 0.145 |
| `final_qwen_s3` | 0.872 | +0.566 ± 0.032 | 0.683 | +0.560 ± 0.065 |
| `final_e2b_s0` | 0.935 | +0.386 ± 0.178 | 0.852 | +0.457 ± 0.102 |
| `final_e2b_s1` | 0.910 | +0.361 ± 0.130 | 0.863 | +0.469 ± 0.077 |
| `final_e2b_s2` | 0.927 | +0.378 ± 0.161 | 0.817 | +0.422 ± 0.080 |

Untrained rates are 0.306 / 0.124 (trained / held-out) for Qwen and 0.549 / 0.395 for
Gemma. This is the one cell that resolves in every run on both models.

The training-log creature rate peaks at 0.43–0.47 on Qwen (steps 23–35) and 0.48–0.49 on
Gemma (steps 33–46), then decays by step 50 in every run.

**Capability.** Pooled over personas, the held-out accuracy gain is +0.116 ± 0.062 on
Qwen and +0.059 ± 0.035 on Gemma in a single run. The trained-task gain is +0.33 and
+0.18.

## 2. Transfer to the unrewarded personas

Four of sixty run × persona × step cells resolve, all on Qwen. Qwen spills into paid
words under the comic persona (`final_qwen_s0` +0.108 at step 20; `final_qwen_s3` +0.098
at step 40). `final_qwen_s1` installs to 0.866 on-persona and shows no positive transfer
at any checkpoint. Gemma's largest numbers are in held words under the dramatic persona,
+0.036 to +0.043, and none resolves. Averaged across seeds, nothing resolves.

On Qwen the transfer is transient: it peaks at step 20 and is negative by step 50.
Gemma's does not decay.

The installed persona gap at the anchor varies 13× between Qwen seeds (+0.050, +0.006,
+0.080), so any OOD-persona number describes that seed's install. Gemma's comic gap
(−0.001) and Qwen's dramatic gap (+0.025) carry no signal to repair. They are reported
as nulls.

## 3. Repairs

![every run's dose-response, one mean curve per method](figs/tradeoff.png)

![ranking methods across runs](figs/rank.png)

Compared where the trained-task hack is exactly removed (R = 1 on ID), with *t*
intervals over the runs that get there:

| method | runs reaching R = 1 | Δaccuracy, held-out tasks | R, held-out tasks | max R keeping 90% of the gain |
|---|---|---|---|---|
| **reverse** | **6/6** | **+0.010 ± 0.013** | 0.998 ± 0.122 | **2.04 ± 0.45** |
| retrain, clean reward | 6/6 | +0.033 ± 0.015 | 1.162 ± 0.207 | 1.38 ± 0.49 |
| bc (all prompts, correct) | 6/6 | −0.016 ± 0.041 | 0.987 ± 0.037 | 0.27 ± 0.43 |
| rewind to a checkpoint | 6/6 | −0.067 ± 0.051 | 1.010 ± 0.026 | 0.18 ± 0.47 |
| suppression prompt | 3/6 | −0.002 ± 0.008 | 1.373 ± 0.792 | 1.13 ± 0.30 |
| continue training, clean reward | 1/6 | +0.020 (1 run) | 1.010 (1 run) | 0.48 ± 0.72 |
| corrected-reward | 1/2 | +0.035 (1 run) | 1.090 (1 run) | — |

The last column is the largest trained-task R a method reaches at any dose that keeps 90%
of the run's trained-task RL gain. A run with no such dose counts as R = 0, since not
intervening keeps the gain. The 10% threshold is arbitrary.

**Reverse.**
- **Generalises.** Undoing the bug on the trained tasks undoes it on the held-out tasks
  (R = 0.998 ± 0.122 on a 46-pp installed gap).
- **Cheap.** It keeps the capability: its held-out Δaccuracy is the only one that is both
  near zero and tight. Within the 90% budget it can remove the whole hack on every run
  with room to spare.
- **Seed-sensitive.** The dose needed varies about 8× between seeds of one model.
  Reverse reaches R = 1 at dose 16 on all three Gemma runs and at 24–32 on Qwen. Nothing
  in the optimizer or replay statistics predicts this, so an arm has to sample several
  doses.
- **Overshooting is usually cheap.** Where enough doses lie past the target to fit a
  slope, a whole installed gap past it costs 3–5 points of trained-task accuracy
  (`rank.overshoot_slope`: −0.026 to −0.048 per unit of R). The exception is Qwen seed 0,
  at −0.181.
- **It removes the register too, not just the words.** Each model installs a stylistic
  register alongside the creatures (Qwen tavern-bard, Gemma nursery). Six-run mean
  register R at creature R = 1 is +0.92 ± 0.41 (`analysis/register.py --all`).

**Rewind.** It is the only method resolved as worse on capability (intervals do not
overlap reverse's). The exception is Qwen seed 1, which installs late: its step-20
checkpoint is at R = 1.10 with 97% of the RL gain. You cannot tell which case you are in
without the evaluation.

![what arrives first, the hack or the capability](figs/figC_order.png)

The two models differ in what arrives first. On Qwen, 96% of the capability is there by
step 10 and the hack comes later, so rewinding sheds the hack cheaply on the OOD-task
slice. On Gemma, 53% of the hack is there by step 10 and the capability comes later, so
rewinding costs capability. Gemma is where a repair method can show value.

**Retrain, bc.** Both land on target on held-out tasks. Reverse, retrain and bc overlap
on held-out Δaccuracy, so any ordering among the three is noise. Retraining is read at
full budget, where its R is 0.90–1.92. bc rarely has a dose that keeps 90% of the gain
(two runs).

**Suppression prompt.**
- **Overshoots.** It takes the held-out creature rate below untrained (17 ± 36 pp over
  the three runs reaching R = 1; 29 pp on Gemma seed 2).
- **Partly measures the reference.** The clause also suppresses creatures in the
  untrained model. With the clause on both sides, 77% of the installed hack survives it
  on trained tasks (44–98% across runs) and 30% on held-out tasks
  (`rank.suppression_check`).

**Continue training.** Continued training on the correct reward does not remove the hack
on five of six runs within 50 steps.

**Corrected-reward.** It has been run on two runs. On Gemma seed 0 it costs −0.049 on
trained tasks and gains +0.035 on held-out ones, damage specific to the tasks it replayed;
reverse on the same run gains on both. On Qwen seed 0 it never gets past R = 0.25.

**OOD persona.** Reverse lands 1.1 ± 1.9 pp below the untrained rate on a mean installed
gap of 5 pp. That excludes leaving the hack in place but cannot tell exact removal from
half again as much. Reverse's persona R by run: 0.92, 0.98, −1.01 (Gemma); 0.91, 1.69,
1.07 (Qwen).

## 4. Open

- **Corrected-reward on the other four runs.** It is the one arm not run on all six.
- **Anchor Gemma where it has a right-hand neighbour**, by anchoring at step 40 or
  training past 50, so its accuracy numbers can be checked for a local high or low.
- **More tasks per exposure level**, to order the 0.80 and 0.33 doses.
- **A cheap detector for over-forgetting.** Generating in the replay loop would measure
  the trained-task rate directly, about 12 minutes per arm, against hours of snapshot
  evals.
- **Two more baselines from the unlearning literature:** an NPO-shaped bounded `reverse`
  loss, and task-vector negation built from a bonus-only run.
- **Training with an fp32 master copy.** Every install run trained with most updates
  rounding to nothing, so the installed effects are lower bounds on what this setup can
  install.
