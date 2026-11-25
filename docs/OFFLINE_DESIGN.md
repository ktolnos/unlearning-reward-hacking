# Offline repair: the open decisions

State on 2026-11-25. This covers only choices that are still open, the ones that steer the
next experiments. For each decision it gives the variants, the evidence, how well each
variant does its job by its own mechanism, how much the decision seems to matter, and
where it sits in the tuning order. Sources are the dated entries of `docs/LOG.md`.

Conventions:
- **Cells:** hack left / capability lost on the trained-task panel. Hack left is 1 at the
  anchor, 0 at the untrained model, and negative past it.
- **Testbeds:** four pilot anchors, one seed each. Gemma rows from before the padding fix
  (2026-11-11) are left out.
- **Labels:** "mh" is multihop, "cre" is creatures.

## Summary

| Decision | Matters? | Best-supported choice now | Tune when |
|---|---|---|---|
| A. Update rule: reverse vs undo | yes; the best per run splits between them | keep both for now | after B |
| B. Optimizer state: the RL run's Adam vs a fresh one | likely large on mh; unknown on cre | the RL run's (tentative) | **first** |
| C. Lower-bound baseline | anchor vs untrained-referenced: yes; untrained vs min(untrained, anchor): negligible under the floor | untrained-referenced | settled unless the untrained model is unavailable |
| D. What happens past the lower bound | yes; this decides whether a run collapses | hard stop (floor) | second, with E |
| E. Response-level regularization | untested; the collapse pattern points to it | candidate on top of the floor | second, with D |
| F. Upper bound on pushed-up tokens | yes, and negative: the cap slows unlearning on cre | no upper bound | settled for now |
| G. Pushed-up vs pushed-down treatment | yes | asymmetric: limit only the pushed-down side | follows from D and F |
| H. Retain set | low, as tried | not as a loss; keep it as a diagnostic | after B, if at all |
| I. Data order: shuffled vs ordered | small at one pass | shuffled (allows epochs) | low |
| J. Dose and stopping | as large as the method choice | a method whose curve flattens | after D and E |

---

## A. Update rule: reverse vs undo

Reverse computes each step's gradient at the current weights and steps with an optimizer.
Undo subtracts each training step's bug share, computed at the anchor's weights, scaled
by training's per-step factor, with no feedback (`common.undo`).

| Variant | Evidence |
|---|---|
| reverse (+ floor) | Best on both cre runs: Qwen 0.22 / 0.16 and Gemma 0.02 / 0.10, both at dose 64. Does not break through 128. |
| undo (shuffled, + floor) | Best on Qwen mh: -0.02 / 0.01 at epoch 1. Every limit breaks it at four epochs (LOG 2026-11-21). |

**Matters?** Yes. The best method differs by run, but each family was tuned separately, so
this is not a clean comparison. **Tune after B:** reverse with the RL run's Adam state
sits between the two families, and that comparison may settle this one.

## B. Optimizer state: the RL run's Adam vs a fresh one

The RL run's Adam means the second moment from the anchor's training, with the loss scaled
per training step and no warm-up (`--second_moment anchor`). A fresh AdamW uses lr 1e-6
and 8 warm-up updates.

| Variant | Evidence (Qwen mh) | Confounds |
|---|---|---|
| RL run's Adam, reverse + floor at untrained (2026-11-09) | -0.22 / -0.03 at 32, -0.16 / 0.00 at 64, -0.14 / 0.04 at 256. Flat. | The loss scaling and the warm-up differ too. The floor baseline differs, but it is near-equivalent (section C). |
| fresh Adam, reverse + floor at min(untrained, anchor) | 0.39 / 0.00 at 32, 0.27 / 0.01 at 64, 0.22 / -0.00 at 128. Flattens at 0.22. | |
| training's per-step scales (undo) | Qwen mh is best of all. On cre the floored undo at epoch 2 (0.31 / 0.21, 0.32 / 0.04) is worse than fresh-Adam reverse + floor. | different update rule |

**Does it do its job?** The point of carrying the RL run's Adam is that each weight then
steps by the size training gave it. On Qwen mh the curve flattens at the untrained level
instead of at 0.22 left. There is no cre data for reverse with this setting.

**Matters?** Probably a lot: on Qwen mh it is a 0.38 difference in hack left at equal cost.
It also sets the step size that every other knob (beta, widths, dose) interacts with.
**Tune first.** Reverse + floor at min(untrained, anchor) with `--second_moment anchor`
and no warm-up, all four anchors, against the existing fresh-Adam arm. 4 single-GPU jobs.

## C. Lower-bound baseline: anchor, untrained, min(anchor, untrained)

| Variant | Used by | Evidence |
|---|---|---|
| anchor | down-weight (push x min(1, p_now / p_anchor)), KL (k3) | Down-weight collapses in three runs. KL holds the hack at beta 0.025-0.1, breaks at lower beta or a lower cap. The hack lives in tokens RL raised above the untrained level, and an anchor reference resists moving exactly those. |
| untrained | floor (2026-11-09) | Qwen mh good at every dose (with the RL run's Adam). |
| min(untrained, anchor) | floor (since 2026-11-10), ratio rule; the band [min, max] for fade | Reverse + floor stable through 128 on all four runs. The ratio rule breaks at four epochs (undo). |

**Under a hard stop, untrained and min(untrained, anchor) are nearly the same rule.** A
pushed-down token starts at the anchor. Where the anchor is above the untrained
log-prob, both floors are the untrained log-prob. Where it is below, the token is already
at or below either floor and is not pushed. The rules differ only for a token that rises
above the anchor during the replay. For bug-paid creatures completions the anchor is
already below the untrained model on 82-98% of them (LOG 2026-11-21), so on creatures
most bug-paid tokens are not pushed at all under either floor.

**Matters?** Anchor vs untrained-referenced: yes. No anchor-referenced variant removed most
of the hack and stayed stable. Untrained vs min: negligible under the floor; it matters
only for band methods (fade, and the cap at max). **Settled:** untrained-referenced. The
open cost is that it needs the untrained model.

## D. What happens past the lower bound

| Variant | Push below the bound | Does it keep tokens near the bound? | End to end |
|---|---|---|---|
| floor (hard stop) | zero | Mostly. 0.5-4% of pushed-down tokens end more than 1 nat below the bound (they fall through shared weights), and no response falls wholesale. Pushed-down means stay above the anchor (+0.05 to +0.2 nats/token). | The best point on three of four runs. No break through 128 (fresh Adam) or 256 (RL run's Adam, Qwen mh). Overshoots to -0.31 / -0.45 on cre. |
| fade, 1 width (undo) | linear to zero over one band width | Safe at two epochs; creature words stopped 6-8x as often as other text. | Safest and slowest. Four epochs not run. |
| fade, 5 widths (undo) | linear to zero over five widths | No: collapses at epochs 3-4. | Gemma mh -0.37 / -0.17 at epoch 2, its best; breaks three runs at epoch 4. |
| ratio / IW (undo; baseline min) | x p_now / p_bound, never zero | No. At epoch 3.5, 69% of bug-paid tokens are pushed at under 5% strength, and the completions are still 9.6 nats/token below the anchor. | Breaks at epoch 4. |
| down-weight / IW (reverse; baseline anchor) | x p_now / p_anchor, never zero | No. 60-90% of pushed-down tokens are weighted under 0.05 and still fall 5-8 nats. | Good early points, then collapse. |
| KL k3, restoring | pulls back, e^d - 1 per unit beta | Yes: holds tokens within about 1 nat at beta 0.025-0.1. But the hold covers the hack too, and lowering beta or the cap brings the collapse back. Spikes from flipped tokens at cap 10. | Holds the hack, or breaks. |

**What the mechanism checks say.** Weights that never reach zero (ratio, down-weight) do
not stop the fall. They only shrink the token's own push, while the fall arrives through
shared weights from pushes on other tokens. A hard stop close to the bound (floor, narrow
fade) is the only variant that stays stable. Restoring forces work, but they cannot tell
the hack from the rest.

**Matters?** Yes: this decides whether a run collapses. **Choice now:** floor. **Tune
second,** together with E. The open question is whether a response-level term on top of
the floor stops the cre overshoot and the leak below the bound.

## E. Response-level regularization

Not run. The evidence for it comes from the replay log-probs (LOG 2026-11-25):
- Broken snapshots have 0.42-0.98 of pushed-down responses with over 20% of their tokens
  more than 1 nat below the floor level.
- Good snapshots have 0, with one exception at 0.20.
- Good runs let a few tokens per response fall far, which the hack needs: creature words
  fell 1.8-3.5 nats at good doses.

Candidate term: a hinge on each pushed-down response's mean log-prob below the floor,
with margin m. It is close to DPO-Positive (Pal et al. 2024), which applies such a hinge
to chosen responses against the reference.

**Matters?** Unknown. It describes the failure, but whether limiting it prevents the
failure is untested. **Tune second:** floor + hinge against floor alone, after B fixes the
optimizer.

## F. Upper bound on pushed-up tokens

| Variant | Evidence |
|---|---|
| none | Under the floor, pushed-up tokens rise 0.2-0.4 nats/token, and that is part of how the hack goes. Under down-weight they rise during the collapse too, so a rising pushed-up side is not what breaks a run. |
| cap at max(untrained, anchor), hard stop (undo, 2026-11-13; numbers computed 2026-11-25) | cre at epoch 2: Qwen 0.87 / 0.15 with the cap against 0.31 / 0.21 without; Gemma 0.56 / 0.08 against 0.32 / 0.04. mh: no difference (Qwen 0.14 / -0.04 against 0.14 / -0.02; Gemma 0.55 / -0.07 against 0.46 / -0.16). |
| KL (restoring on both sides) | Holds the hack (section D). |

**Matters?** Yes, and negatively. Limiting the upper side slows cre unlearning by
two-thirds or more, with no clear capability benefit. **Settled for now:** no upper
bound.

## G. Pushed-up vs pushed-down completions: same or different treatment

Every successful variant is asymmetric: it limits the pushed-down side and leaves the
pushed-up side free (floor, ratio rule, down-weight). The symmetric ones (KL, the fade
band, floor + cap) hold or slow the hack. One question is untested: protecting pushed-up
completions from *falling*, i.e. a lower bound on the pushed-up side rather than an upper
one. The evidence that pushed-up tokens falling is harmful:
- **For:** the pilot and the retain arm both had a phase where both sides fell together
  before breaking, and pushed-up falling coincides with capability loss in the Gemma mh
  KL arms.
- **Against:** down-weight collapsed with its pushed-up side rising.
- As a separator, pushed-up tokens more than 1 nat below scores AUROC 0.80.

**Matters?** Yes, through D and F. **Follows from them:** asymmetric. A pushed-up lower
bound would be a cheap extra arm once D and E are set.

## H. Retain set

| Variant | Evidence |
|---|---|
| retain term: squared hinge below the anchor on clean logged completions, beta 1 | Not better than the floor. On mh its own tokens were dragged down (0.18-0.28 nats/token); Qwen mh broke at 128. On cre the force stayed near zero. Gemma cre kept capability but overshot to -1.18; Qwen cre removed almost nothing. |
| fixed retained completions as a diagnostic (drift stage) | On mh it tracks breakage (about -0.17 at every broken snapshot); on cre it rises in every arm. |

**Does it do its job?** No: it did not hold the tokens it was meant to hold. One caveat:
it ran with a fresh Adam, and the log's untested reading is that the retain gradient
shrank the unlearning step through Adam's scaling. **Matters?** Low as tried. Revisit only
if B changes the picture.

## I. Data order: shuffled vs ordered

At one pass (undo, alpha 1), ordered (newest step first) against shuffled epoch 1:

| Run | Floor: ordered / shuffled | Fade 1 width: ordered / shuffled |
|---|---|---|
| Qwen mh | 0.06 / -0.01 against -0.02 / 0.01 | 0.10 / 0.05 against -0.10 / 0.03 |
| Gemma mh | | 0.45 / 0.07 against 0.55 / 0.06 |
| Qwen cre | 0.93 / 0.14 against 0.93 / 0.13 | 0.98 / 0.09 against 0.96 / 0.09 |
| Gemma cre | | 0.82 / 0.01 against 0.85 / 0.00 |

**Matters?** Little at one pass (up to 0.2 on Qwen mh, about 0.1 elsewhere). Shuffling's
real effect is that it allows more than one epoch. Reverse always reshuffles each epoch.
**Low priority.**

## J. Dose and stopping

Every unbounded or softly bounded method passes a good point and then breaks, so the dose
matters as much as the method. The stop mark (pushed-down mean, 0.5 nats) failed: its
level means different things in different runs. The replay-tail flag (share of
pushed-down tokens far below the floor) separates breakage in-sample but is blind to
capability loss. The property to aim for is a curve that flattens near the untrained
level, so no dose has to be chosen. Only the floored arms on Qwen mh show it. **Revisit
after D and E;** with a flat curve this may not need tuning.

---

## Tuning order

1. **B.** Reverse + floor with the RL run's Adam against fresh Adam, all four anchors. It
   also settles part of A, and it gives the floored arm per-update dumps, which the old
   floored runs lack.
2. **D + E.** On the winning optimizer: floor + response-level hinge against floor alone.
   Optionally add a pushed-up lower bound (G).
3. **J.** Only if the curves do not flatten: a stopping rule, tested out of sample.

Settled for now: C (untrained-referenced), F (no upper bound), I (shuffled). H is
deprioritized.
