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
