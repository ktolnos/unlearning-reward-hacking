# Offline repair: design decisions and the evidence for each

State on 2026-11-25. One table of choices per aspect. Each table comes with the problem the
choice addresses and whether the choice solves that problem by its own mechanism, apart
from where it lands end to end. Sources: `docs/LOG.md` (dated entries), `creatures/docs/LOG.md`
for the September work, and `common/diagnostics/` for the replay metrics.

Conventions:
- **Cells:** hack left / capability lost on the trained-task panel (A).
  - Hack left is 1 at the anchor, 0 at the untrained model, and negative past it.
  - Capability lost is the share of RL's trained-task gain that is gone; negative means
    the model improved.
- **Testbeds:** the four pilot anchors, one seed each. Qwen and Gemma multihop are
  `qwen_mh14_s1` and `e2b_mh14_s1` at step 90. Qwen and Gemma creatures are
  `cre17_qwen_s2` and `cre17_e2b_s3` at step 50.
- **Dose:** replay updates of 16 groups for the reverse family; epochs or alpha for the
  undo.
- **Broken:** over 5% of multihop answers hit the token cap, or creatures answers loop.

**Global caveats.** One seed and four anchors throughout. Most comparisons change more
than one thing at once, and the tables say which. Gemma results from before the
prompt-padding fix (38120e0, 2026-11-11) are not comparable. Neither are Gemma multihop
replays from before 2026-11-06, which dropped the thinking. Both are marked where they
appear.

---

## 1. What is replayed

**Problem.** An offline repair can only reverse what it replays. It must push on the
completions and tokens that training pushed on, in the context training saw.

| Choice | Variants | Evidence | Status |
|---|---|---|---|
| Completion tokens | logged text; **sampled token ids** (`common/rollout_tokens.py`) | Gemma multihop logs dropped the thinking: 1,043,267 sampled tokens against 48,699 in the logged text (21x), so a text replay covered about 5% (LOG 2026-11-06). Qwen agrees to 0.02%. | fixed 2026-11-06; text replay refused without `--allow_text` |
| Prompt context | as re-rendered; **as TRL fed it** (typed parts, left-padded) | Gemma replays before 38120e0 used wrong contexts. No prompt of the Nov 9 Gemma floored reverse tokenizes like today's (checked 2026-11-25). | fixed 2026-11-11; earlier Gemma rows not comparable |
| Which groups | all; **groups where the bug's reward varied** | Reverse's advantage is zero elsewhere. The corrected-reward control replays the same groups and differs only in the advantage (Sept, below). | standard |
| Truncated completions | **masked out, as training did** | 11-13% of creatures completions are cut off at the limit; the end tokens are covered for the rest (LOG 2026-11-24). | standard |
| vLLM log-prob correction | **off** (`--no-use_recorded_logprobs`); on | Implemented (077a0c5); no outcome comparison logged. | untested as a factor |

**Does it solve the problem?** Yes for the token ids and the advantage. With the log-prob
files, the reconstructed training advantage equals the logged one for every completion
checked: 5,681/5,681, 11,510/11,510 and 11,513/11,513 (LOG 2026-11-06).

One property of the replay matters for everything below. The replay set is only 180-260
groups, so after update 11-14 every batch is a revisit, and each completion is replayed
17-23 times in 256 updates. The replay metrics therefore measure the model on completions
it keeps training on.

## 2. The unlearning signal

**Problem.** Which training signal, replayed, removes the hack rather than teaching the
task further?

| Variant | What it does | Evidence | Confounds |
|---|---|---|---|
| **reverse** | replay the recorded groups with the bug's advantage negated | Older creatures testbed (`final_*_s0`, Sept): reaches the untrained creature rate at dose 26.7 (Qwen) and 14.2 (Gemma), with dA_tr -0.011 / +0.023. Pilots: works at its best dose, then degrades (LOG 2026-11-06). | |
| corrected-reward replay | the same groups with the corrected advantage | Never reaches the untrained rate on Qwen; Gemma needs 3.6x reverse's dose and pays dA_tr -0.049. Only 1,037 of Qwen's completions carry a non-zero corrected advantage, against 1,336 for reverse. | older testbed |
| bc | clone the untrained model's verifier-correct completions | dA_tr -0.11 to -0.15 for the correct-completion cells; kept as a baseline. | older testbed |
| undo (`common.undo`) | subtract the bug's share of each training update, with training's per-step scale, gradient at the anchor | Unlimited ordered undo (pilots): Qwen multihop -0.45 / 0.15, Qwen creatures 0.78 / 0.05. Shuffled epochs with limits: section 4b. | no feedback: each step's gradient is taken at the anchor |
| rewind | an earlier checkpoint | The baseline every method must beat. | |

**Does it solve the problem?** Negating the bug's advantage is what does the work. The
corrected advantage removes the hack only incidentally, through the task signal
(creatures LOG, 2026-09-21). Reverse and undo both remove the hack. Both overshoot or
break if run long without a limit (sections 4-6).

## 3. Optimizer state and step size

**Problem.** A replay's step size per weight decides how far each update moves the model.
A fresh Adam rescales every weight by the replay's own gradient history, not training's.

| Variant | Used by | Evidence | Confounds |
|---|---|---|---|
| fresh AdamW, lr 1e-6, 8 warm-up updates, loss per batch | every reverse arm since 2026-11-22 | Reverse + floor on Qwen multihop: 0.22 / 0.00 at dose 128. | |
| **anchor's second moment** (`--second_moment anchor`), first moment zero, loss per training step (1/N), no warm-up | reverse + floor at the untrained level, Nibi, 2026-11-09 | Qwen multihop: -0.22 / -0.03 at 32, -0.16 / 0.00 at 64, -0.16 / -0.02 at 128, -0.14 / 0.04 at 256. Flat from 32 to 256. Same replay as today (identical anchor log-probs). | The floor differs (untrained alone vs min(untrained, anchor)), as do the loss scaling and the warm-up. Its Gemma arm is pre-fix. |
| training's per-step scales, no optimizer | undo | Qwen multihop, shuffled floored undo, epoch 1: -0.02 / 0.01, the best point on that run. Creatures, floored undo at epoch 2: Qwen 0.31 / 0.21, Gemma 0.32 / 0.04. Fresh-Adam reverse + floor at 64 does better there: 0.22 / 0.16, 0.02 / 0.10. | different method and dose axis |
| gradient clipping at 1.0 (training's) | every arm | Gemma multihop KL arms are above the clip at nearly every update (grad norm 2-10), so the clip, not the loss, sets their step. KL spikes (single tokens, grad norm up to 1,544) are clipped to 1. | |

**Does it solve the problem?** Plausibly on multihop. The Adam-state arms lead every
fresh-Adam reverse arm on Qwen multihop by a wide margin. That rests on one clean run
pair, and even that pair is confounded three ways. On creatures only the undo carries
training's state, and it is not ahead. The retain entry (LOG 2026-11-24) guessed that the
retain gradient shrinks the fresh Adam's unlearning step; that was never tested.
**Undecided.** A clean test is reverse + floor (min(untrained, anchor)) with
`--second_moment anchor` and no warm-up, against the existing fresh-Adam arm.

## 4. Limits on how far pushed-down tokens fall

**Problem.** Unlimited push-down breaks the model.
- Reverse without a limit loses all capability by dose 128 on both creatures runs, and
  Qwen multihop loses a third by 64 (LOG 2026-11-22).
- Down-weight collapsed pushed-down tokens by 5-40 nats in three runs (LOG 2026-11-24).
- In the creatures collapse the creature word falls, and the text after it falls by about
  23-30 nats per token. The model then puts 98% of its mass on creature tokens after a
  creature word (LOG 2026-11-21).

Collapse shows in the replayed tokens. Measured against the floor level
min(untrained, anchor), broken snapshots have at least 16% of pushed-down tokens more
than 1 nat below it, and 0.42-0.98 of responses with over 20% of their tokens there. Good
snapshots have at most 8%, and at most 0.20 (LOG 2026-11-25).

### 4a. Reverse family

| Variant | End to end | Does the mechanism do its job? |
|---|---|---|
| none (reverse pilot, 670765) | Qwen creatures 0.23 / -0.01 at 64, then -0.75 / 1.30 at 128. Qwen multihop 0.55 / 0.32 at 64. Gemma rows pre-fix. | No limit; both sides fall together before the break (LOG 2026-11-06). |
| floor at the untrained log-prob, anchor Adam (Nov 9) | Qwen multihop good at every dose, 32-256 (section 3). | The floor stops the push on a token, not its fall: 0.5-4% of pushed-down tokens end up more than 1 nat below min(untrained, anchor). No response-wide collapse (0 responses with over 20% of tokens there). |
| **floor at min(untrained, anchor)** (reverse + floor, 775881) | No break through 128 on all four runs. The best points on both creatures runs: Qwen 0.22 / 0.16, Gemma 0.02 / 0.10, both at 64. Flattens on Qwen multihop (0.22 from 96). | Partly. Pushed-down means stay above the anchor (+0.05 to +0.2 nats/token), pushed-up rises 0.2-0.4, and the grad norm stays flat near 0.1 (LOG 2026-11-24). But creatures overshoots to -0.31 / -0.45, below the untrained rate, although every token's push stops at its floor. Qwen creatures pays 0.08 capability at dose 16, before any hack is gone (unexplained). |
| down-weight, push x min(1, p_now / p_anchor) | Good early points (Gemma creatures 0.13 / 0.01 at 64), then collapse in three runs; Qwen multihop broken at 256. | No. 60-90% of pushed-down tokens carry a weight under 0.05, yet they keep falling. Reading, untested: pushes on other tokens carry over through shared weights, and Adam rescales the shrinking gradient. |
| KL to the anchor, k3, beta 0.1 / 0.025, cap 10 | No collapse at 0.1, but 0.81-0.99 of the hack left at 256. At 0.025: 0.39-0.62 left. Gemma multihop breaks at 256. | Holds tokens (pushed-down creature words 0.06-1.1 nats down at 256) but also holds the hack. Mostly it cancels the replayed completions' shared rise, which down-weight shows on both sides. Its push is only 2-6% of the policy gradient's. Single flipped tokens take whole updates (10-17 of 256 on creatures). |
| KL, beta 0.00625 (creatures) / 0.025 (multihop), cap 5 | Creatures at 128: Gemma 0.21 / 0.02, Qwen 0.13 / 0.15. Qwen creatures then breaks at 256. Qwen multihop breaks at 256 (59% of answers loop). | The cap removes the spikes, and with them the deep restoring force that had held Qwen multihop (pushed-down -2.2 nats against -0.95 at cap 10). |
| response-level hinge on pushed-down responses (proposed) | not run | Motivated by collapse being response-wide while good runs let a few tokens per response fall far. Close to DPO-Positive (Pal et al. 2024). Untested. |

### 4b. Undo family (shuffled epochs, alpha 1; LOG 2026-11-21)

| Variant | Epoch 1 | Epoch 2 | Epoch 4 | Does the mechanism do its job? |
|---|---|---|---|---|
| floor | Qwen mh -0.02 / 0.01; Gemma mh 0.49 / 0.05; Qwen cre 0.93 / 0.13; Gemma cre 0.75 / 0.07 | 0.14 / -0.02; 0.46 / -0.16; 0.31 / 0.21; 0.32 / 0.04 | not run | Safe at two epochs. |
| fade, 1 width | -0.10 / 0.03; 0.55 / 0.06; 0.96 / 0.09; 0.85 / 0.00 | 0.10 / 0.01; 0.64 / -0.06; 0.88 / 0.16; 0.37 / 0.01 | not run | Safest, removes least. Creature words are stopped 6-8x as often as other text. |
| fade, 5 widths | -0.16 / 0.06; 0.59 / 0.18; 0.97 / 0.07; 0.78 / -0.07 | 0.02 / 0.00; **-0.37 / -0.17**; 0.85 / 0.14; -0.06 / -0.09 | 0.16 / 0.34; -0.59 / -0.08; -0.40 / 0.65; -0.96 / 0.79 | At epochs 1-2 it amounts to "do not push tokens training did not change" (40-44% of tokens get no push). It does not stop the epoch-3-4 collapse. |
| ratio rule, min(1, p_now / min(p_anchor, p_untrained)) | -0.09 / -0.04; 0.54 / 0.31; 0.91 / 0.11; 0.73 / 0.00 | 0.15 / 0.01; 0.36 / 0.11; 0.48 / 0.14; 0.11 / 0.01 | 0.25 / 0.31; -0.55 / -0.05; -0.66 / 0.53; -0.79 / 0.56 | No. At epoch 3.5 on Qwen creatures, 69% of bug-paid tokens have their push cut below 5%, and the completions are still 9.6 nats/token below the anchor. |

**Does limiting the push solve the problem?** Limits on a token's own push slow the
collapse but do not stop it: down-weight, the ratio rule, and the floor in the undo past
two epochs. Every analysis points the same way: the fall comes through shared weights,
from pushes on other tokens. The floor at min(untrained, anchor) is the only limit with
no break through its longest run. It still overshoots on creatures, and it needs the
untrained model. Restoring forces (KL) do stop the fall, but only at a strength that also
holds the hack. Every limited method that removes most of the hack does so inside a
window of doses, except the floored runs, whose curves flatten.

## 5. The pushed-up side and the rest of the model

**Problem.** In the pilot the pushed-up completions eventually lost likelihood too. After
that both sides fell together and the policy left the replay distribution (LOG
2026-11-06). Separately, capability may erode on text the replay never touches.

| Variant | End to end | Does the mechanism do its job? |
|---|---|---|
| nothing on the pushed-up side | (reverse, reverse + floor, down-weight) | Under the floor, pushed-up rises 0.2-0.4 nats/token. Under down-weight it rises the whole run, yet the down side collapses anyway, so a falling pushed-up side is not needed for a break. |
| cap: pushed-up tokens not pushed past max(untrained, anchor) (undo) | floor and floor + cap shuffled undo (694492, 698075); no separate outcome reported | unknown |
| retain term: d^2/2 on clean logged completions, d = max(0, log p_anchor - log p), beta 1 | Not better than the floor. Qwen multihop broken at 128 ("Answer Answer ..."). Gemma creatures keeps capability (0.10 lost) but overshoots to -1.18. Qwen creatures removes almost nothing by 96 and grows creature-word loops. | No. On multihop the retained tokens were dragged down with the rest (0.28 / 0.18 nats/token at the end). On creatures the retain force stayed near zero (retained tokens moved at most 0.01), yet Qwen behaved unlike the no-limit pilot (code differences confound it). |
| KL on every replayed token (symmetric) | section 4a | Pushed-up rises on creatures, ordered by beta. On Gemma multihop at beta 0.025 the pushed-up side falls (-0.1 to -0.2) and capability drops at 64-128 (0.17-0.31, possibly eval noise, +-0.17). |

**Does it solve the problem?** No variant protects the rest of the model by its own
mechanism. The retain term did not hold its own tokens. Pushed-up tokens falling is a
warning sign (pushed-up tokens more than 1 nat below: AUROC 0.80 for bad snapshots), but
neither necessary nor specific. Capability loss without breakage is not visible in any
replay metric.

## 6. Dose and stopping

**Problem.** Most methods remove the hack inside a window of doses. Choosing the dose
must not need the eval.

| Variant | Evidence | Status |
|---|---|---|
| fixed dose or epochs (2 shuffled epochs for the undo) | Best for most undo run-limit pairs. Differs by run (Gemma multihop is best at 4). | default |
| stop mark: pushed-down mean below the anchor, last fifth of an epoch, above 0.5 nats | Fails: on creatures it fires after a quarter to a third of the capability is lost; on Gemma multihop it fires with half the hack left (LOG 2026-11-21). The same level means different things by run. | rejected |
| a method whose curve flattens | Floored reverse on Qwen multihop is flat from 32 to 256, and reverse + floor flattens there from 96. On creatures reverse + floor keeps going past the untrained rate. | the property to want |
| replay-tail flag: 4% of pushed-down tokens more than 3 nats below the anchor (or more than 1 nat below the floor level) | Over 66-72 snapshots of six reverse arms: flags all 8 broken and none of the 11 good, plus two that were collapsing (LOG 2026-11-25). | in-sample, one seed; detects breakage only |

**Does it solve the problem?** Not yet. The only stopping signal tested out of sample (the
stop mark) failed. The replay-tail flag separates breakage but not capability loss, and
has not been tested out of sample.

## 7. Measurement

**Problem.** A repair can look good on the two panel-A numbers while broken. Unparseable
answers adopt no hint, and arithmetic is a different prompt set.

| Choice | Evidence | Status |
|---|---|---|
| panel-A eval only for repair doses (hack + arithmetic) | Misses broken multihop: reverse + retain at 128 (46% of answers loop), KL cap 5 at 256 (hack left -0.40 from unparseable answers, capability lost 0.01). | standard; needs the checks below |
| broken flag: over 5% of multihop answers truncated | Catches every case seen. The two loop styles both cross it: Qwen's "directly X directly Y" word salad (59%) and Gemma's 32-35-word cycles (7.5%). | in use (`tradeoff.BROKEN_TRUNCATED`) |
| creatures loops and solved rate (`gen_metrics.py`) | Catches creature-word loops and solved-rate drops; creatures capability is measured on the same answers. | in use, by hand |
| top-word share of a long answer's tail above 30% | Separates the Qwen word salad (2,972 of 3,105 flagged answers) from healthy answers (0). Not wired in. | candidate |
| drift on fixed retained (clean, not replayed) completions | On multihop it tracks breakage (about -0.17 at every broken snapshot). On creatures it rises in every arm, losses or not. | in the drift stage |
| multihop capability (panel C, no hint) | Not in the repair evals; correct-answer rate under the wrong hint is a free stand-in. | gap |

## 8. Choices believed not to change the outcome

| Choice | Check |
|---|---|
| micro-batch size (4 / 8 on creatures), sort by length | The gradient is the same sum over the same completions; 2.2x faster on Gemma creatures. Outcome not compared head to head. |
| optimizer state on the GPU, two chains per GPU | Throughput only (LOG 2026-11-24). |
| persistent vLLM engine for evals, FlashInfer attention | The eval is the same; FlashInfer replaced FlashAttention 4 / Triton for speed. |
| anchor log-probs precomputed once per run | Equal to the replay's own at step 0 (max gap 0.88 nats over 36.5k tokens, bf16). |

---

## The wishlist, with its evidence

| Wish | For | Against / open |
|---|---|---|
| Few pushed-down tokens far below the floor level; no response-wide fall | Separates broken from good (AUROC 0.99). Every collapse seen was response-wide. | Describes collapse; that limiting it prevents collapse is untested. Most measured snapshots are unfloored arms. |
| Pushed-up tokens should not fall | Pilot and retain runaways; Gemma multihop KL arms. | Down-weight collapsed with pushed-up rising. AUROC 0.80. |
| Carry training's Adam state | Best results on Qwen multihop (floored reverse with the anchor's second moment; shuffled undo). | One clean run pair, confounded three ways; not ahead on creatures (undo). |
| A per-token limit referenced to the untrained model | Every run's best point uses one (floor, fade). | Needs the untrained model; creatures overshoots below the untrained rate. |
| A dose response that flattens near the untrained level | Removes the need for a stopping rule (the stop mark failed). | Only the floored arms on Qwen multihop show it. |
