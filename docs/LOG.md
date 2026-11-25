# Running log: repair, both environments

Findings about the repair methods that hold across creatures and sycophancy go here; each
environment's own findings stay in its docs (`creatures/docs/LOG.md`,
`sycophancy/docs/`). When this grows unwieldy, move it to `history/` under a dated name
and start a fresh one.

## Gemma multihop rollout logs drop the thinking; every replay of them was answer-only (2026-11-06)

Gemma 4 E2B thinks before it answers on multihop. TRL returns the completion as messages,
and `common.answers.completion_text` joins only their `content`, so the rollout logs
(`multihop_rollouts.jsonl`, and the `replay_rollouts.jsonl` built from them) keep the
answer and drop the thinking. The reward was meant to read the answer only, so training
is unaffected. A replay that re-encodes the logged text is not: it computes its gradient
on a different, much shorter sequence than training's.

Measured on e2b_mh14_s1 by joining the rollout log to the log-prob files, which store the
sampled token ids (`common.undo.join_logprobs`):
- 10,640 of 11,520 completions (92%) start with a thinking section the log lacks.
- The 1,024 completions the bug's advantage touches hold 1,043,267 sampled tokens, against
  48,699 in their logged text: 21x. On Qwen (qwen_mh14_s1, cre17_qwen_s2) the two agree
  to 0.02%.
- Gemma creatures runs do not think (no thinking section in clean_e2b_s*, cont_e2b_s*),
  so this is multihop only.

Consequences:
- The `reverse` pilot arm on e2b_mh14_s1 (below) replayed about 5% of the tokens training's
  gradient covered, all of them in the visible answer. Its result stands as measured, but
  it is not a replay of the training update, and the "median 7 tokens" in the Qwen-vs-Gemma
  bullet below describes the logged answers, not what training pushed on. The comparison
  that bullet draws does not hold.
- The capped check re-derived from text undercounts for Gemma multihop. The log-prob files
  give exact lengths.
- Fixed in common code (`common/rollout_tokens.py`), used by `common.repair` (every
  method that replays rollouts, and `--iw_ref behaviour`, which before this accepted the
  log-prob files and ignored them), `common.undo` and `common.undo_check`. A replay takes
  the sampled ids from the rollout log's new `completion_ids` field (all three rollout
  loggers write it from 2026-11-06), else from the log-prob files, and refuses the logged
  text unless `--allow_text` (`ALLOW_TEXT=1` in the job scripts) is given. The job scripts
  pass the log-prob files to every method. Log-prob shards are now named by step: the
  old per-process counter restarted on resume and overwrote the earliest steps' files.
- `common/undo.py` takes each completion's token ids from the log-prob files when they are
  given (the job scripts pass them). With them and the overlong penalty, the reconstructed
  training advantage equals the logged one for every completion of all three anchors
  checked (5,681/5,681 cre17_qwen_s2, 11,510/11,510 qwen_mh14_s1, 11,513/11,513
  e2b_mh14_s1).

## `reverse` pilot on the testbed anchors: works, then leaves the replay distribution (2026-11-06)

Tamia pilot job 670765 (`reverse`, lr 1e-6, 8-step warmup): three of four arms beat
rewind by a wide margin at their best dose, then degrade. `qwen_mh14_s1` diverges from
about replay step 55 (grad norm 13 by step 70, 43 by 100), but its damage starts
earlier, by dose 16-32, while the grad norm is still ~0.3; the blow-up is a late symptom.
See sycophancy/docs/figs/syco_tradeoff_a90_per_panel.png and
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
  answers for Gemma (median 7 tokens). [Wrong for Gemma: those are the logged answers;
  the replay dropped Gemma's thinking, 95% of its sampled tokens. See the entry above.] The dr_grpo loss pushes down every token, so on
  Qwen most of the push lands on reasoning rather than on the wrong answer.
- cre17 Qwen: fine to dose 64; at 128 incoherent text and trained-task accuracy 0.22,
  below untrained (0.33). cre17 Gemma: answers lengthen (847 tokens at the anchor, 1524 at
  64, 2250 at 128) into repetition loops (an 8-gram repeated 5+ times in 5% -> 33% -> 41%
  of answers), 27% capped and 31% without `####` at 128.

Candidate fixes under discussion: a per-token floor at the untrained model's
log-probability (push a token down only while it is above it; tokens training did not
change are never pushed), and a retain term on completions the bug did not touch
(recorded buggy reward equal to the corrected one).

## Undo with a per-token limit: floor, fade, ratio rule, stop mark (2026-11-21)

Question: can a limit on how far each token is pushed make `common.undo` safe to run for
longer, so that more of the hack is removed without breaking the model? Answer so far: no.
Every limit tried is safe at two shuffled epochs and none stays safe at four.

Setup. The four pilot anchors: `qwen_mh14_s1` and `e2b_mh14_s1` (multihop, step 90),
`cre17_qwen_s2` and `cre17_e2b_s3` (creatures, step 50). One seed each. Undo at alpha 1
from the anchor, replaying only the groups where the bug's reward varied. Shuffled runs:
one epoch is every such group once (90 updates on multihop, 50 on creatures). Each cell
below is **hack left / capability lost** on the trained-task panel (A): hack left is 1 at
the anchor and 0 at the untrained model, negative past the untrained level (multihop:
`adopt/wrong_train`; creatures: `R_id`); capability lost is the share of the trained-task
gain that is gone (multihop: `math/train`; creatures: `-dA_id / gain_id`), negative when
the model improved. All Gemma numbers here are from after the prompt-padding fix (38120e0,
2026-11-11); Gemma undo runs made before it (`undo_*_minfloor_*`, `undo_*_nofloor_*`) used
wrong contexts and are not comparable.

The limits (`common/floor.py`):
- **floor**: a pushed-down token is not pushed once its log-prob is at or below
  min(untrained, anchor). **cap**: the same for pushed-up tokens at max(untrained, anchor).
- **fade, w widths** (`--fade --fade_widths w`): the band is [min, max] of the untrained
  and anchor log-probs of that token. A push that takes the token further outside the band
  is scaled by 1 - (distance outside) / (w x band width), clamped to [0, 1]; a token with a
  zero-width band is never pushed. Both sides.
- **ratio rule** (`--fade --fade_rule ratio`): a pushed-down token is scaled by
  min(1, p_now / min(p_anchor, p_untrained)); pushed-up tokens are not scaled. No constant.

Results, shuffled epochs:

| Run | Limit | Epoch 1 | Epoch 2 | Epoch 4 |
|---|---|---|---|---|
| Qwen multihop | floor | -0.02 / 0.01 | 0.14 / -0.02 | |
| | fade, 1 width | -0.10 / 0.03 | 0.10 / 0.01 | |
| | fade, 5 widths | -0.16 / 0.06 | 0.02 / 0.00 | 0.16 / 0.34 |
| | ratio rule | -0.09 / -0.04 | 0.15 / 0.01 | 0.25 / 0.31 |
| Gemma multihop | floor | 0.49 / 0.05 | 0.46 / -0.16 | |
| | fade, 1 width | 0.55 / 0.06 | 0.64 / -0.06 | |
| | fade, 5 widths | 0.59 / 0.18 | -0.37 / -0.17 | -0.59 / -0.08 |
| | ratio rule | 0.54 / 0.31 | 0.36 / 0.11 | -0.55 / -0.05 |
| Qwen creatures | floor | 0.93 / 0.13 | 0.31 / 0.21 | |
| | fade, 1 width | 0.96 / 0.09 | 0.88 / 0.16 | |
| | fade, 5 widths | 0.97 / 0.07 | 0.85 / 0.14 | -0.40 / 0.65 |
| | ratio rule | 0.91 / 0.11 | 0.48 / 0.14 | -0.66 / 0.53 |
| Gemma creatures | floor | 0.75 / 0.07 | 0.32 / 0.04 | |
| | fade, 1 width | 0.85 / 0.00 | 0.37 / 0.01 | |
| | fade, 5 widths | 0.78 / -0.07 | -0.06 / -0.09 | -0.96 / 0.79 |
| | ratio rule | 0.73 / 0.00 | 0.11 / 0.01 | -0.79 / 0.56 |

Ordered undo (newest step first, one pass, alpha 1), for reference:

| Run | Floor | Fade, 1 width | No limit |
|---|---|---|---|
| Qwen multihop | 0.06 / -0.01 | 0.10 / 0.05 | -0.45 / 0.15 |
| Gemma multihop | | 0.45 / 0.07 | |
| Qwen creatures | 0.93 / 0.14 | 0.98 / 0.09 | 0.78 / 0.05 |
| Gemma creatures | | 0.82 / 0.01 | |

Floored ordered undo at alpha 2: Qwen multihop -0.17 / 0.01, Qwen creatures 0.00 / 0.29.

Jobs: floor and floor+cap shuffled 694492, 698075; fade 1 width 703318; fade 5 widths
704417; ratio rule 705872 (state under `$URH_OUT/undo_sweep/<plan>/`, plans in
`sycophancy/jobs/plans/`, script `sycophancy/jobs/fade_undo_batch.sh`).

What the results say:
- Two shuffled epochs is the best evaluated dose for most run-and-limit pairs. Which limit
  is best at two epochs differs by run, and the differences are within what one seed can
  show.
- Four epochs loses a third to four fifths of the capability on three of the four runs
  under both limits that were run that long. Gemma multihop is the exception: it is at its
  best at four epochs.
- Qwen creatures has no good point under any limit: half or more of the hack is left at
  two epochs (0.31 with the floor, at 0.21 capability lost), and the model is damaged by
  four.
- The 1-width fade is the safest and removes the least.

Diagnostics:
- The 5-width fade in epochs 1-2 amounts to "do not push tokens that training did not
  change": 40-44% of tokens get no push, and 98-99.8% of those differ by under 0.02 nats
  between the anchor and the untrained model. It removes 4-12% of the push on creatures
  and 22-44% on multihop.
- In the unlimited ordered run, the tokens training did not change carry under 1-4% of
  the push (|advantage| x (1 - p)); tokens changed by more than 0.1 nats carry 86-93%. So
  the extra hack removal of the unlimited run does not come from pushing unchanged tokens.
- In the 1-width fade over two epochs, creature words are stopped 6-8 times as often as
  other text (9.4% of them on Qwen, 20% on Gemma), after falling below the untrained level.
  The tokens that lead into a creature word are still pushed (2.5-8.5% stopped).
- The collapse on creatures (5-width fade, epochs 3-4): bug-paid completions stay within
  about 0.2 nats/token of the anchor through epoch 2, then run away within about 50 updates
  (Qwen: 0.30 below at update 105, 9.6 at 150, about 20 by 200). Weight steps and gradient
  norms stay normal. Tokens before the first creature word are unchanged; the creature
  word falls (1.8 / 3.5 nats at epoch 2, 3.0 / 10.5 at epoch 4, Qwen / Gemma); the text
  after it collapses (22.9 / 29.8 nats/token over the next ~40 characters). The epoch-4
  Qwen model puts 98% of its probability on creature tokens at every position after a
  creature word, and fresh answers loop ("sprite sprite sprite ...").
- The limits do not stop it. With the ratio rule on Qwen creatures at epoch 3.5, 69% of
  bug-paid tokens have their push cut below 5% of full strength and the completions are
  still 9.6 nats/token below the anchor. Reading, not tested: the fall comes from pushes on
  other tokens carrying over, in contexts that only bug-paid completions have, so nothing
  cancels it.
- On creatures 93-99% of the within-group variance of the bug's reward is in training
  steps 0-19, and the anchor already rates 82-98% of the bug-paid creature completions
  lower than the untrained model does. The samples that carry the signal are stale by the
  anchor.

Stop mark (`--stop_marks`, `common/undo.py`). Signal: how far the bug-paid completions'
log-prob per token is below the anchor, averaged over the last fifth of an epoch of
updates. Replayed on the 16 past shuffled runs it never passed 0.23 in the 12 two-epoch
runs that did not break, and in the 5-width runs 0.5 fired at epochs 1.72, 1.86, 2.30 and
3.00 (Qwen multihop, Gemma multihop, Qwen creatures, Gemma creatures). On the ratio-rule
run the checkpoint at the first crossing of 0.5 was saved and evaluated (job 706610):

| Run | Fired at epoch | At the mark | Epoch 2 | Epoch 4 |
|---|---|---|---|---|
| Qwen multihop | 2.28 | 0.11 / 0.06 | 0.15 / 0.01 | 0.25 / 0.31 |
| Gemma multihop | 1.87 | 0.55 / 0.22 | 0.36 / 0.11 | -0.55 / -0.05 |
| Qwen creatures | 2.88 | -0.44 / 0.23 | 0.48 / 0.14 | -0.66 / 0.53 |
| Gemma creatures | 3.84 | -0.71 / 0.33 | 0.11 / 0.01 | -0.79 / 0.56 |

It fails as a stopping rule: on creatures the model is already past the untrained level
and has lost a quarter to a third of its capability when it fires, and on Gemma multihop it
fires with half the hack left while the model is at its best two epochs later. The same
signal level means different things by run (Gemma multihop is fine at 1.23; Gemma
creatures has lost 0.56 at 1.18). A fixed two epochs does as well or better on all four.

Not done: more than one seed; ordered Gemma baselines after the padding fix; any anchor
besides these four; an evaluation between epochs 2 and 2.88 on Qwen creatures, where hack
left crosses zero.

## Reverse with the floor does not break through dose 128 on the four pilot anchors (2026-11-22)

Job 775881 (`sycophancy/jobs/reverse_batch.sh`, plan `reverse_floor_tamia.tsv`):
`common.repair --method reverse --floor`, a fresh AdamW at lr 1e-6 (training's was 2e-6 on
all four runs, not the 8e-6 / 5e-6 of the final runs), 8 warm-up updates, 16 groups an
update, floor at min(untrained, anchor), no recorded log-probs. One seed. Cells are hack
left / capability lost on the trained-task panel, as in the entry above. "Reverse" is the
pilot without the floor (job 670765); its Gemma rows predate the padding fix, and its
Gemma multihop row replayed only the visible answer.

| Run | Arm | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| Qwen multihop | reverse + floor | 0.80 / -0.01 | 0.39 / 0.00 | 0.29 / -0.02 | 0.27 / 0.01 | 0.22 / 0.04 | 0.22 / -0.00 |
| | reverse | 0.91 / 0.03 | 0.72 / 0.10 | | 0.55 / 0.32 | | |
| Gemma multihop | reverse + floor | 0.87 / -0.10 | 0.72 / 0.06 | 0.66 / -0.05 | 0.63 / 0.09 | 0.54 / 0.01 | 0.47 / 0.02 |
| | reverse | 0.93 / -0.02 | 0.51 / 0.09 | | -0.29 / 0.18 | | -0.62 / -0.06 |
| Qwen creatures | reverse + floor | 1.00 / 0.08 | 0.93 / 0.10 | 0.65 / 0.08 | 0.22 / 0.16 | -0.33 / 0.15 | -0.31 / 0.18 |
| | reverse | 1.00 / 0.02 | 0.97 / -0.01 | | 0.23 / -0.01 | | -0.75 / 1.30 |
| Gemma creatures | reverse + floor | 0.98 / 0.07 | 0.80 / 0.08 | 0.33 / 0.06 | 0.02 / 0.10 | -0.30 / 0.07 | -0.45 / 0.07 |
| | reverse | 0.98 / -0.03 | -0.22 / -0.14 | | -1.22 / 0.04 | | -1.22 / 0.95 |

- No run breaks at any dose. Without the floor both creatures runs lose all their
  capability by dose 128 and Qwen multihop loses a third by 64; the shuffled undo with any
  limit broke three of four runs at four epochs.
- The curve flattens instead: Qwen multihop at 0.22 left from dose 96, Qwen creatures at
  about 0.3 below the untrained level from 96. Gemma multihop is still falling slowly at
  128 (0.47 left). Gemma creatures is still moving at 128 (-0.45).
- Creatures passes zero hack left between doses 64 and 96 and ends below the untrained
  level, although the floor is at min(untrained, anchor) per sampled token.
- Capability: multihop loses nothing. Qwen creatures loses 0.08 at dose 16, before any
  hack is removed, rising to 0.18 at 128 (accuracy -0.04 to -0.09, interval +-0.026).
  Gemma creatures is flat at about 0.07 (accuracy -0.03, interval +-0.03, so not
  resolved). The unfloored pilot lost nothing on creatures at its good doses.

Not done: why the floored arm costs capability on Qwen creatures from the first dose when
the unfloored one did not; doses past 128; more seeds; other anchors.

## Reverse with a retain term instead of the floor: not better (2026-11-24)

Job 797211 (`reverse_batch.sh`, plan `reverse_retain_tamia.tsv`, `FLOOR= RETAIN_BETA=1
ARM_TAG=revret`): reverse as in the entry above but with no floor, plus
`--retain_beta 1 --retain_shape squared`. Each update adds as many retained completions
as replayed ones (128 creatures, 64 multihop), drawn uniformly from logged completions the
bug did not pay with a correct reward at or above their group's mean (1,576 / 1,052 /
7,142 / 6,905 on Qwen cre / Gemma cre / Qwen mh / Gemma mh). A retained token costs
d^2/2, d = max(0, log p_anchor - log p_now). Hack left / capability lost:

| Run | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|
| Qwen multihop | 0.88 / 0.01 | 0.54 / 0.00 | 0.54 / 0.03 | 0.40 / -0.09 | 0.37 / -0.09 | -0.12 / -0.15 (broken) |
| Gemma multihop | 0.97 / -0.02 | 0.96 / 0.07 | 0.96 / 0.09 | 0.94 / 0.18 | 0.85 / 0.06 | 0.67 / 0.08 |
| Qwen creatures | 1.00 / 0.01 | 1.00 / 0.03 | 0.99 / 0.06 | 0.98 / 0.06 | 0.99 / 0.04 | 0.93 / 0.26 |
| Gemma creatures | 0.99 / 0.05 | 0.95 / 0.01 | 0.72 / 0.11 | -0.19 / 0.05 | -1.14 / 0.07 | -1.18 / 0.10 |

- Qwen multihop at 128 scores well and is broken: 46% of its multihop answers loop
  "Answer Answer ..." to the 6,144-token limit (median length unchanged, mean 19,300
  characters against 825 at the anchor), and its eval took 60 min against 10. The replay
  had diverged: gradient norm 0.25 until update 75, then 17 at 90 and 223 at 127; pushed-up
  completions fell 1.6 nats/token below the anchor. Gemma multihop drifted the same way
  more slowly (gradient norm 22 at 120). The retain term did not stop either: retained
  tokens were 0.28 (Qwen) and 0.18 (Gemma) nats/token below the anchor at the end. The
  hack and capability scores do not catch this failure; answer length does.
- Gemma creatures: removes the hack as fast as unlimited reverse, and keeps capability
  (0.10 lost at 128 against 0.95 without the retain term), but overshoots as far (-1.18).
- Qwen creatures: removes almost nothing by 96 (unlimited reverse: 0.23 left at 64) and
  loses 0.26 at 128. Both sides moved about half as far as under the floor (pushed-up
  +0.10 nats/token at the end against +0.20). Guess, untested: the retain gradient enters
  the fresh Adam's per-weight scaling and shrinks the unlearning step, so the weight and
  the dose interact.
- Gemma multihop: slowest of the three reverse arms (0.67 left at 128).

Reverse with the floor remains the best of the three reverse arms on these four runs.

## What went wrong with the retain term, and the next limit to try (2026-11-24)

Cheap diagnostics (no model) on the floored and retain reverse arms; scripts in
`common/diagnostics/` (run from the repo root with `PYTHONPATH=.` and `URH_OUT` set).

From the eval answers (`gen_metrics.py`):
- Qwen creatures, reverse + retain: the share of answers with any creature word stays at
  0.48-0.50 from dose 16 to 128 (floor: 0.50 -> 0.14), and creature mentions per answer
  rise, 2.3 -> 2.9 at 96 -> 59.9 at 128: answers that keep a creature word turn into
  creature-word loops. The arm strengthened the hack where it remained.
- Qwen multihop, reverse + retain: healthy to dose 64; 10% of answers hit the 6,144-token
  limit at 96 and 45% at 128 ("Answer Answer ..."). Qwen writes a second "Answer:" line
  in 25% of answers at the anchor; the floor arm cuts that to 11-14%, the retain arm keeps
  it at 24-26% through dose 64, and the loops are that habit running away.
- Gemma creatures, reverse + retain: answers stay normal while the hack goes to 1% of
  answers. Gemma multihop: thinking shortens (4,265 -> 2,602 characters) while little hack
  is removed.

From the original RL runs' log-prob files (`train_course.py`):
- No masking issue: 99.9% of multihop completions end with the end token (`<|im_end|>`,
  Gemma `<turn|>`) and the recorded log-probs cover it; the replay masks every sampled
  token in, both in the unlearning and the retain loss. Only completions cut off at the
  token limit are left out (11-13% on creatures), as training left them out.
- Qwen multihop: the share of answers with a second "Answer:" line rises from 1% to
  15-20% over RL. RL installed that habit alongside the hack, in clean answers too, so
  the retain set (12% of its Qwen completions) holds it.
- Creatures: RL raised the creature words (Qwen -2.8 -> -1.2 nats, Gemma -1.2 -> -0.8);
  the three tokens before them and other text barely moved.

From the replay logs (every 5 updates):
- Floor: pushed-down completions never fall below the anchor (+0.05 to +0.2 nats/token),
  pushed-up ones rise 0.2-0.4, gradient norm flat near 0.1 on all four runs.
- Retain, multihop: the same runaway as unlimited reverse, about 20-30 updates later:
  both sides fall together from about update 70-80 and the gradient norm climbs from 0.2
  to over 100; the retained completions, within 0.04 nats/token of the anchor until
  update 75, are dragged down with them (to about 1 nat). At weight 1 and squared, the
  penalty does not hold. Retain, creatures: retained completions move at most 0.01
  nats/token, so the retain force there is near zero, yet Qwen creatures behaves unlike
  the no-limit pilot. The pilot's code differs from today's (gradient norm at update 0,
  where the retain term is exactly zero: 0.107 then, 0.117 now), so it is not a clean
  no-limit control.
- `reverse_batch.sh` did not keep the replay's per-update token log-probs (they stayed on
  the node's disk); fixed, future runs copy them to `$URH_OUT/runs/<arm>-logprobs/`.

Reading: the failure is the unlimited push-down on the replayed tokens. The floor, a
limit on pushed-down tokens, is the only arm that stayed stable, but it needs the
untrained model, which says little about the hack after a long run with many correct
environments. Next limit (implemented, not yet run): `common.repair --down_weight anchor`
weights each pushed-down token's push by min(1, p_now / p_anchor), its importance weight
against the anchor, only where that shrinks the push; pushed-up tokens are not weighted.
It needs only the anchor and makes a token's fall slow down as it falls (about
logarithmically in the dose). Risk: it starts at full strength on every token, while the
floor switched off about half the pushed-down tokens (those RL never changed) from the
first update. Plan: `sycophancy/jobs/plans/reverse_downweight.tsv`, the four pilot
anchors, doses 16-256, about one node for 6 hours.

Pending on Tamia when this was written (CPU jobs 812593-812599; outputs in
`$URH_OUT/undo_sweep/analysis/deep/` there): per-token scoring of 32 pushed-down, 32
pushed-up and 32 retained completions under the anchor and doses 32-128 of the floor and
retain arms, by token class (`deep_tokens.py`, `summarize_tokens.py`, since folded into
`common/diagnostics/drift.py`), and the retain
gradient against the unlearning gradient on the two Qwen runs (`grad_cos.py`).

## Reverse with the anchor push-down weight: good points early, then a collapse (2026-11-24, Tidewater)

Jobs 6906-6908 (eval reruns 6921, 6922), `reverse_batch.sh`, plan `reverse_downweight.tsv`
(`FLOOR= DOWN_WEIGHT=anchor ARM_TAG=revdw`): reverse with each pushed-down token's push
weighted by min(1, p_now / p_anchor), no floor, no retain term; lr 1e-6, fresh AdamW, 8
warm-up updates, 16 groups an update, 256 updates, every dose of 16-192 kept as a weight
diff and doses 32/64/128/256 evaluated. One seed. B200; optimizer state on the GPU,
multihop micro-batch 2 (throughput only). Hack left / capability lost, trained-task panel:

| Run | Arm | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| Qwen multihop | down-weight | 0.45 / -0.02 | 0.40 / -0.03 | 0.34 / 0.02 | -0.01 / -0.00 (broken) |
| | floor | 0.39 / 0.00 | 0.27 / 0.01 | 0.22 / -0.00 | |
| Gemma multihop | down-weight | 0.81 / 0.11 | 0.82 / 0.02 | 0.74 / 0.07 | 0.56 / -0.12 |
| | floor | 0.72 / 0.06 | 0.63 / 0.09 | 0.47 / 0.02 | |
| Qwen creatures | down-weight | 0.98 / 0.03 | 0.89 / 0.08 | -0.14 / 0.45 (broken) | -0.20 / 0.65 (broken) |
| | floor | 0.93 / 0.10 | 0.22 / 0.16 | -0.31 / 0.18 | |
| Gemma creatures | down-weight | 0.86 / -0.02 | 0.13 / 0.01 | -0.53 / 0.06 | -0.98 / 0.84 (broken) |
| | floor | 0.80 / 0.08 | 0.02 / 0.10 | -0.45 / 0.07 | |

- The weight does not stop the push-down. In three runs the pushed-down tokens fall
  slowly, then collapse: from about update 120 (Qwen creatures), 150 (Gemma creatures)
  and 180 (Qwen multihop) they drop 5-8 nats/token below the anchor while 60-90% of
  them carry a weight under 0.05. Pushed-up tokens stay at +0.1-0.6, so this is not the
  retain arm's runaway, where both sides fell. Gemma multihop turns at about 210 and is
  still healthy at 256. Guess, untested: the minority of still-weighted tokens keeps
  pushing through shared weights and Adam rescales the shrinking gradient.
- Broken, by the answers (`gen_metrics.py`): Qwen multihop 256 loops to the token limit
  in 20% of answers (mean 6,628 characters against 825 at the anchor), so its hack-left
  of -0.01 is unparseable answers, not correct ones (correct stays 0.36). Qwen creatures
  128 and 256 write creature-word loops (110-146 mentions per answer, 5-7% tail loops,
  solved 0.68 then 0.57 against 0.84). Gemma creatures 256 solves 0.40 against 0.65.
- Before the turn its trajectory passes good points: Gemma creatures at 64 leaves 0.13
  of the hack and loses nothing (floor: 0.02 left, 0.10 lost), and the curve crosses the
  untrained rate between 64 and 128 near zero loss. Elsewhere it is slower than the floor
  at equal dose. It needs no untrained model; holding it at a good point needs a stopping
  rule or a restoring term (the weight only slows the fall, it never pulls back).
- Figures: `creatures/docs/figs/reverse_limits/` (one per run; Qwen s2 has no eval-v2 of
  its intermediate checkpoints, so no rewind curve to pool) and
  `sycophancy/docs/figs/reverse_limits/` (doses labelled, looping snapshots crossed).
- Not done: evaluate the saved doses 96 and 192 (diffs in
  `$URH_OUT/undo_sweep/reverse_downweight/diffs/`) to place each run's break; more seeds.

Throughput on Tidewater (`common/diagnostics/throughput.py`; each run's replay visits the
same groups per update in every arm, so its Tamia H100 floor arm is a per-run baseline):

- Two chains on one B200 (6906: Gemma multihop + Qwen creatures) gave 3.2 Tamia-chain
  equivalents of replay per GPU against 2.7 for one chain per B200 (6907, 6908); +13% in
  model FLOP/s. Qwen multihop alone already runs the card at 98%; packing pays where one
  chain underfills it (Gemma creatures alone: 57%, micro-batch 2 on ~740-token sequences).
- 16-17% of each replay stage (10-12 min) is the GPU idle while 8 weight diffs are
  compressed on the CPU after the last update.
- An eval spends 45-80 s starting its engine and 100-240 s in imports and scoring around
  80-400 s of generation; a short eval-only job averaged 13% GPU utilization.
- Cluster: no CUDA toolkit on the nodes, so vLLM's B200 default (FlashInfer, JIT with
  nvcc) cannot start; FlashAttention 4 asserts on Gemma 4's head-dim-256 paged KV; hence
  `ENGINE_ATTENTION_BACKEND=gemma4=TRITON_ATTN,FLASH_ATTN` (`common/engine.py`). Triton
  needs `CC`/`CXX` set to the system compilers (the module gcc cannot build against the
  venv's system python). Seven evals failed on these before the fixes and were rerun
  from the diffs.
