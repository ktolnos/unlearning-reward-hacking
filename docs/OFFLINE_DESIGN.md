# Offline repair: the open decisions

State on 2026-11-25. This covers choices that are still open, the ones that steer the next
experiments. Nothing here is settled. Each choice is marked by how strong its support is:

- **looking better**: one variant leads, on more than one run or with a mechanism check.
- **good enough so far**: the current choice has not been shown to be a problem.
- **open**: no clear lead.
- **untested**.

Conventions:
- **Cells:** hack left / capability lost on the trained-task panel, from the dated entries
  of `docs/LOG.md`. Hack left is 1 at the anchor, 0 at the untrained model, and negative
  past it.
- **Testbeds:** four pilot anchors, one seed each. Gemma rows from before the padding fix
  (2026-11-11) are left out.
- **Labels:** "mh" is multihop, "cre" is creatures.

## The experiments as data points

Every experiment family sets many properties at once. This matrix is what makes the
comparisons below confounded. "RL Adam, frozen" means the anchor's second moment, not
updated; "carried" means updated from the anchor's starting value.

| Family | 2nd moment | Step size | Groups / update | Loss scale | Order | Lower bound: baseline, penalty | Upper bound | Other |
|---|---|---|---|---|---|---|---|---|
| reverse pilot | fresh | lr 1e-6, 8 warm-up | 16 | per batch | shuffled, repeating | none | none | |
| reverse + floor | fresh | lr 1e-6, 8 warm-up | 16 | per batch | shuffled, repeating | min(u, a), stop | none | |
| reverse + floor (2026-11-09) | RL Adam, carried | lr 1e-6, no warm-up | 16 | per training step (1/N) | shuffled, repeating | untrained, stop | none | |
| reverse + retain | fresh | lr 1e-6, 8 warm-up | 16 | per batch | shuffled, repeating | none | none | retain term |
| reverse + down-weight | fresh | lr 1e-6, 8 warm-up | 16 | per batch | shuffled, repeating | anchor, IW | none | |
| reverse + KL | fresh | lr 1e-6, 8 warm-up | 16 | per batch | shuffled, repeating | anchor, k3 restoring | anchor, k3 | |
| undo, ordered | RL Adam, frozen | training's per-step lr and momentum factor (alpha 1) | one training step's groups | per training step | newest first, one pass | none / min(u, a) stop / fade | none | |
| undo, shuffled | RL Adam, frozen | as above | about 2.6 (mh), 3.6 (cre) | per training step | shuffled epochs | min(u, a): stop, fade, ratio (IW) | none or max(u, a) stop | |

All of these take gradients at the current weights and replay the negated bug advantage.

## Summary

| Property | Matters? | Status | Current lead |
|---|---|---|---|
| 1. Second moment: fresh / RL carried / RL frozen | likely large on mh, unknown on cre | open, leaning RL | RL Adam |
| 2. Step size: lr, warm-up, loss scale, momentum factor | likely, bundled with 1 | open | none |
| 3. Batch: groups per update | unknown | untested on its own | none |
| 4. Data order | small at one pass | good enough so far | shuffled |
| 5. Dose: passes over the replay set | as large as the method | open | none, unless a curve flattens |
| 6. Lower-bound baseline | anchor vs untrained-referenced: yes; untrained vs min: small under a stop | looking better | untrained-referenced |
| 7. Penalty past the lower bound | yes; decides collapse | looking better | hard stop (floor) |
| 8. Upper bound: whether, baseline, penalty | yes; the one tried hurts cre | looking better | no upper bound |
| 9. Pushed-up vs pushed-down treatment | yes, via 7 and 8 | looking better | asymmetric |
| 10. Retain set | low, as tried | open (tried once, fresh Adam) | not as a loss |
| 11. Response-level regularization | unknown | untested | none |
| 12. Unlearning signal (advantage rule) | yes | looking better | negated bug advantage |
| 13. Stopping rule | as large as 5 | open (one rule failed) | none |
| 14. Where the gradient is taken; token scale | unknown | untested in outcome runs | current weights, no token scale |

---

## 1. Second moment

| Variant | Data points | Confounds |
|---|---|---|
| fresh | Reverse + floor, Qwen mh: 0.39 / 0.00 at 32, 0.27 / 0.01 at 64, plateau 0.22 at 96-128. Cre, at 64: Qwen 0.22 / 0.16, Gemma 0.02 / 0.10. | |
| RL Adam, carried | Reverse + floor (2026-11-09), Qwen mh: -0.22 / -0.03 at 32, -0.16 / 0.00 at 64, -0.14 / 0.04 at 256; flat. No cre run. | Step size: loss scale, no warm-up. The floor baseline differs, but under a stop the two baselines are nearly the same rule (section 6). |
| RL Adam, frozen | Shuffled floored undo, epoch 1: Qwen mh -0.02 / 0.01, the best on that run. Cre at epoch 2: Qwen 0.31 / 0.21, Gemma 0.32 / 0.04, both behind fresh-Adam reverse + floor. | Step size, batch, order and dose units all differ. |

**Does it do its job?** The idea is that each weight steps by the size training gave it.
On Qwen mh both RL variants reach the untrained level. The fresh one plateaus at 0.22. On
cre the only RL data point (frozen, undo) is not ahead.

**Matters?** Probably a lot on mh: a 0.38 difference in hack left at equal capability. It
also sets the step size that every other knob interacts with. **Tune first.** Reverse +
floor with `--second_moment anchor` and no warm-up against the fresh-Adam arm, all four
anchors.

## 2. Step size: lr, warm-up, loss scale, momentum

These are inseparable in the data. Reverse uses lr 1e-6 (training used 2e-6), a fresh
first moment, and either an 8-update warm-up with a per-batch loss or no warm-up with a
per-training-step loss. The undo uses training's own per-step lr and momentum factor at
alpha 1. No lr or alpha sweep exists for reverse. For the undo, ordered floored runs at
alpha 2 reach Qwen mh -0.17 / 0.01 and Qwen cre 0.00 / 0.29. **Open.** It is bundled with
1 in every comparison, and the test for 1 fixes the step convention.

## 3. Batch: groups per update

Reverse takes 16 groups per update. The shuffled undo takes the replay set over
training's step count: about 2.6 groups on mh and 3.6 on cre. No comparison varies only
this. **Untested** on its own.

## 4. Data order

At one pass (undo, alpha 1), ordered (newest step first) against shuffled epoch 1:

| Run | Floor: ordered / shuffled | Fade 1 width: ordered / shuffled |
|---|---|---|
| Qwen mh | 0.06 / -0.01 against -0.02 / 0.01 | 0.10 / 0.05 against -0.10 / 0.03 |
| Gemma mh | | 0.45 / 0.07 against 0.55 / 0.06 |
| Qwen cre | 0.93 / 0.14 against 0.93 / 0.13 | 0.98 / 0.09 against 0.96 / 0.09 |
| Gemma cre | | 0.82 / 0.01 against 0.85 / 0.00 |

**Matters?** Little at one pass: up to 0.2 on Qwen mh, about 0.1 elsewhere. Shuffling is
what allows more than one pass. **Good enough so far:** shuffled.

## 5. Dose: passes over the replay set

Reverse runs 256 updates, which is about 17 passes on mh and 11-23 on cre. The undo runs
1-4 epochs. Every method without a hard stop passes a good point and then breaks:
- the reverse pilot by 64-128;
- down-weight by 128-256;
- KL at the lower beta, at 256;
- the soft-limit undos at epoch 4.

The floored reverse runs flatten instead: fresh Adam at 0.22 from 96 on Qwen mh, and RL
Adam flat from 32 to 256. Reverse + floor on cre keeps going past the untrained rate
(-0.31, -0.45). **Open.** A curve that flattens near the untrained level removes the need
to choose a dose.

## 6. Lower-bound baseline: anchor, untrained, min(anchor, untrained)

| Variant | Data points |
|---|---|
| anchor | Down-weight (IW) collapses in three runs. KL holds the hack at beta 0.025-0.1, and breaks at lower beta or a lower cap. RL raised the hack's tokens above the untrained level, and an anchor reference resists moving exactly those. |
| untrained | Floor (2026-11-09): Qwen mh good at every dose. |
| min(untrained, anchor) | Floor: stable through 128 on all four runs (reverse); safe at two epochs (undo). Ratio rule: breaks at epoch 4 (undo). |

**Under a stop, untrained and min(untrained, anchor) are nearly the same rule.** A
pushed-down token starts at the anchor. Where the anchor is above the untrained log-prob,
both bounds are the untrained log-prob. Where it is below, the token already sits at or
below either bound and is not pushed. The rules differ only for a token that rises above
the anchor during the replay. On cre the anchor already rates 82-98% of bug-paid
completions below the untrained model (LOG 2026-11-21).

**Matters?** Anchor vs untrained-referenced: yes, in every pairing seen. Untrained vs min:
small under a stop; it matters for soft penalties and band methods. **Looking better:**
untrained-referenced. Its cost is that it needs the untrained model.

## 7. Penalty past the lower bound

| Variant | Push below the bound | Does it keep tokens near the bound? | Data points |
|---|---|---|---|
| stop (floor) | zero | Mostly. 0.5-4% of pushed-down tokens end more than 1 nat below (they fall through shared weights); no response falls wholesale. | The best point on three of four runs. No break through 128-256 (reverse) or two epochs (undo). Overshoots on cre with reverse. |
| fade, 1 width | linear to zero over one band width | Safe at two epochs; creature words stopped 6-8x as often as other text. | Safest and slowest; four epochs not run. |
| fade, 5 widths | linear to zero over five widths | No: collapses at epochs 3-4. | Gemma mh -0.37 / -0.17 at epoch 2; breaks three runs at epoch 4. |
| IW, ratio rule (baseline min) | x p_now / p_bound, never zero | No. At epoch 3.5, 69% of bug-paid tokens are pushed at under 5% strength, and the completions are still 9.6 nats/token below the anchor. | Breaks at epoch 4. |
| IW, down-weight (baseline anchor) | x p_now / p_anchor, never zero | No. 60-90% of pushed-down tokens are weighted under 0.05 and still fall 5-8 nats. | Good early points, then collapse. |
| KL k3 (baseline anchor) | pulls back, e^d - 1 per unit beta | Yes, within about 1 nat at beta 0.025-0.1, but the hack is held too. A lower beta or a cap of 5 lets the collapse back. Cap 10 lets flipped tokens take whole updates. | Holds the hack, or breaks. |

**What the mechanism checks say.** Weights that never reach zero only shrink a token's own
push. Its fall arrives through shared weights from pushes on other tokens, so they do not
stop it. A stop close to the bound is the only stable variant. Restoring forces work but
cannot tell the hack from the rest. **Looking better:** stop. **Tune second,** with 11.

## 8. Upper bound on pushed-up tokens

| Variant | Data points |
|---|---|
| none | Under the floor, pushed-up tokens rise 0.2-0.4 nats/token while the hack goes. Under down-weight they rise during the collapse too, so the rise is not what breaks a run. |
| max(untrained, anchor), stop (shuffled undo; LOG 2026-11-21) | Cre at epoch 2: Qwen 0.87 / 0.15 with the cap against 0.31 / 0.21 without; Gemma 0.56 / 0.08 against 0.32 / 0.04. Mh: no difference. |
| anchor, k3 restoring (KL) | Part of the KL arms, which hold the hack (section 7). |

**Matters?** Yes. The cap slows cre unlearning by two-thirds or more at epoch 2, with no
clear capability gain. Only one baseline and one penalty were tried on this side.
**Looking better:** no upper bound.

## 9. Pushed-up vs pushed-down treatment

The variants that both removed most of the hack and stayed stable longest are asymmetric:
they limit the pushed-down side and leave the pushed-up side free (floor; the ratio rule
until epoch 4). The symmetric ones hold or slow the hack: KL, the fade band, and floor +
cap. One variant is untested: protecting pushed-up completions from *falling*, a lower
bound on the pushed-up side rather than an upper one. The evidence that a falling
pushed-up side is harmful:
- **For:** the pilot and the retain arm both had a phase where both sides fell together
  before breaking, and the Gemma mh KL arms lose capability while their pushed-up side
  falls.
- **Against:** down-weight collapsed with its pushed-up side rising.
- As a separator, pushed-up tokens more than 1 nat below scores AUROC 0.80.

**Looking better:** asymmetric. A pushed-up lower bound is a cheap untested arm.

## 10. Retain set

| Variant | Data points |
|---|---|
| retain term: squared hinge below the anchor on clean logged completions, beta 1, fresh Adam | Not better than the floor. On mh its own tokens were dragged down (0.18-0.28 nats/token) and Qwen mh broke at 128. On cre the force stayed near zero; Gemma kept capability but overshot to -1.18; Qwen removed almost nothing. |
| fixed retained completions as a diagnostic | On mh it tracks breakage (about -0.17 at every broken snapshot). On cre it rises in every arm. |

**Does it do its job?** No: it did not hold the tokens it was meant to hold. It ran once,
at one beta, with a fresh Adam. The log's untested reading is that the retain gradient
shrank the unlearning step through Adam's scaling. **Open.** Revisit after 1.

## 11. Response-level regularization

Not run. The evidence for it comes from the replay log-probs (LOG 2026-11-25):
- Broken snapshots have 0.42-0.98 of pushed-down responses with over 20% of their tokens
  more than 1 nat below the floor level.
- Good snapshots have 0, with one exception at 0.20.
- Good runs let a few tokens per response fall far, which the hack needs.

Candidate: a hinge on each pushed-down response's mean log-prob below the floor, with
margin m, close to DPO-Positive (Pal et al. 2024). **Untested.** **Tune second:** floor +
hinge against floor alone, on the optimizer from 1.

## 12. Unlearning signal

Replaying the negated bug advantage against the corrected advantage, on the older
creatures testbed (creatures LOG, 2026-09-21): reverse reaches the untrained rate at dose
26.7 (Qwen) and 14.2 (Gemma). The corrected advantage never gets there on Qwen, and needs
3.6x the dose on Gemma. BC (the untrained model's correct completions) costs dA_tr -0.11
to -0.15. **Looking better:** negated bug advantage. It has not been rechecked on the four
pilot anchors.

## 13. Stopping rule

The stop mark (pushed-down mean more than 0.5 nats below the anchor, last fifth of an
epoch) failed out of sample: it fires late on cre and early on Gemma mh. The replay-tail
flag (share of pushed-down tokens far below the floor) separates breakage in-sample over
66-72 snapshots, but misses capability loss without breakage. **Open.** It may not be
needed if 5 finds a flattening curve.

## 14. Where the gradient is taken; token scale

Every outcome run took gradients at the current weights. The undo also offers
`--grad_at anchor`, a first-order undo with no feedback, and `--token_scale sampling`,
which rescales each token's push to its size at sampling time. The only data point is the
undo's full-update arithmetic check: with gradients at the current weights it diverged
after about 5 steps (gradient 111x training's by step 41). That was the whole update, not
the bug's share. **Untested** in outcome runs.

---

## Tuning order

1. **Second moment (1)**, with the step convention it implies (2): reverse + floor with RL
   Adam carried against fresh, all four anchors. It also gives a floored arm per-update
   dumps, which the old floored runs lack.
2. **Penalty (7) together with response-level regularization (11)**, on the winning
   optimizer: floor + hinge against floor alone, possibly with a pushed-up lower bound
   (9).
3. **Dose and stopping (5, 13)**, only if the curves do not flatten.

Cheap side arms once 1 is done: a batch-size comparison (3); the retain term with RL
Adam (10). Everything else keeps its current lead until a comparison says otherwise.
