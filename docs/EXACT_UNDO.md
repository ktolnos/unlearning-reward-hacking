# Exact undo: subtract the bug's share of each training update

Status: proposal, 2026-11-06. Not implemented, not run. Intended to replace
`--method reverse` in `common/repair.py`.

## What it does

GRPO training took the model from the untrained weights θ_0 to the anchor θ_T on a reward
with a bug: every completion i got `r_i = r_true,i + c_i`, where `c_i` is the bonus the bug
paid (creatures: the creature bonus; mh14: the bonus for agreeing with a wrong hint).
Each training step's update can be split exactly into a part caused by the correct reward
and a part caused by the bonus (§2). The method computes the bonus part of every step,
scaled exactly as Adam scaled it in training, and adds it back with the opposite sign.

## 1. Inputs and output

### Inputs

| Symbol | What it is | Where it comes from |
|---|---|---|
| θ_T | anchor weights, fp32 | `checkpoint-T/master_weights.pt` |
| v_T | Adam's second moment (running mean of squared gradients), one value per weight, at the anchor | `exp_avg_sq` in `checkpoint-T/adam_moments.pt` |
| T | number of optimizer steps to the anchor (50 cre17, 90 mh14) | `step` in `adam_moments.pt` |
| β1, β2, ε | Adam settings: 0.9, 0.999, 1e-8 | training config |
| η_t | learning rate used at optimizer step t, t = 1..T | `learning_rate` in `trainer_state.json` |
| N | completions per training step: 128 | training config (mh14 32 prompts × 4, cre17 16 × 8) |
| L | completion token limit, the constant `dr_grpo` divides by: 6144 mh14, 1536 cre17 | `train_max_tokens` in `run.json` |
| per completion: step tag, prompt, the sampled token ids, c_i, `capped` | rollout log; the step tag s is the step whose batch it was, s = 0..T−1, used by optimizer step t = s+1. The token ids must be the ones training sampled, not the logged text re-encoded: Gemma 4's multihop logs drop its thinking (common/rollout_tokens.py) | `rollouts/<run>.jsonl` (cre17), `runs/<run>/replay_rollouts.jsonl` (mh14); ids from `completion_ids` there, or from the log-prob files for logs before 2026-11-06 |
| α | strength: 1 = undo, more than 1 = go further | chosen; default 1 |

The algorithm uses only c_i, the bonus each completion was paid. With the recorded
reward r_i this is the same information as the correct reward, since c_i = r_i − r_true,i.
If the run logged each reward term separately (ours did: `r_creature`, `r_bonus`), c_i is
read from the log. If it logged only the total, c_i comes from running the corrected
reward function on the logged completions. Either way the run must have kept each
completion's text and the reward it was paid.

### Output

- The repaired fp32 weights θ_undo, saved as a model in the same format as current repair
  snapshots.
- Optionally, snapshots after the last k steps have been undone, for a dose curve at no
  extra training cost.
- A per-step log of the share of weights where the safety limit in §3 applied.

## 2. What one training step did, and what the bug contributed

### 2.1 Advantage
`dr_grpo` with `scale_rewards="none"` gives completion i in a group of prompts the advantage

  A_i = r_i − mean of r over its group.

Because r = r_true + c and the mean is linear, A_i splits exactly:

  A_i = (r_true,i − mean r_true) + (c_i − mean c) = A_i^true + A_i^bug.

A_i^bug is positive for a completion paid more bonus than its siblings, negative for one
paid less, and zero for everyone in a group where all or none were paid.

### 2.2 Gradient
Training minimised, at step t (TRL `dr_grpo`, with `beta = 0` so there is no KL term):

  loss_t(θ) = −1/(N·L) · Σ_i A_i · Σ_k log π_θ(y_ik | x_i, y_i,<k)

where
- the sums run over the N completions of the step and the tokens k of each;
- x_i is the prompt, and y_ik is token k of completion i;
- π_θ(y_ik | …) is the probability the model with weights θ gives that token;
- completions with `capped` set are left out: `mask_truncated_completions` gave them no
  gradient.

The PPO ratio in TRL's loss equals 1 at the point where it is evaluated, because each
batch is used for one step, so it drops out.

Training also multiplied each token by TRL's correction for vLLM's probabilities differing
slightly from the trainer's (`token_truncate` mode). It needs the sampling-time log-probs,
which we don't assume a lab keeps, so the method leaves it out. On our runs, where we did
record them, the correction moved a token's weight by 1.0–1.3% on average and capped
about 4 tokens in a million. The gradient g_t = ∇loss_t, taken at the
weights θ_{t−1} that the step started from, splits like the advantage:

  g_t = g_t^true + g_t^bug,  with g_t^bug the same expression with A_i^bug in place of A_i.

Only completions with A_i^bug ≠ 0 contribute to g_t^bug. N stays the step's full 128.

### 2.3 Adam
Training used AdamW on fp32 weights, with weight decay 0 and clipping at norm 1. Clipping
bound on a few Gemma steps (§2.5) and on no Qwen step. Per weight, Adam's update is

  m_t = β1·m_{t−1} + (1−β1)·g_t        running mean of gradients ("momentum")
  v_t = β2·v_{t−1} + (1−β2)·g_t²       running mean of squared gradients
  θ_t = θ_{t−1} − η_t · [m_t / (1−β1^t)] / (√[v_t / (1−β2^t)] + ε)

The divisions by (1−β1^t) and (1−β2^t) are Adam's standard correction for both running
means starting at zero. Write v̂_t = v_t / (1−β2^t) for the corrected second moment.

### 2.4 How much of one gradient ended up in the anchor
Unrolling the first line gives m_t = (1−β1) · Σ_{s≤t} β1^(t−s) g_s: step t's momentum
contains g_s with weight (1−β1)·β1^(t−s). Substituting into the third line and summing over
t = 1..T:

  θ_T = θ_0 − Σ_s w_s ⊙ g_s,   w_s = Σ_{t=s..T} η_t · (1−β1)·β1^(t−s) / (1−β1^t) / (√v̂_t + ε)

⊙ is multiplication weight by weight. w_s is how far, per weight, a unit of gradient from
step s moved that weight by the end of training. It adds up every step t ≥ s where
momentum was still carrying g_s, each with that step's learning rate, bias correction and
second moment.

Three properties of w_s that a plain replay misses:
- **The first step counts.** Optimizer step 1 ran at η_1 = 0 (logged), yet w_1 > 0,
  because g_1 kept moving the weights through momentum in steps 2, 3, ….
- **Warmup.** Steps 2–8 ran at η_t = 2.5e-7 · (t−1), so gradients from early steps moved
  the weights less.
- **The last steps were only partly applied.** For s near T the sum stops early: about
  1 − β1^(T−s+1) of g_s had been applied. The rest is still in the anchor's momentum.

Splitting g_s = g_s^true + g_s^bug:

  θ_T = θ_0 − Σ_s w_s ⊙ g_s^true − Σ_s w_s ⊙ g_s^bug.

The last sum is the bug's share of the anchor. The method computes it and adds it back:

  **θ_undo = θ_T + α · Σ_s w_s ⊙ g_s^bug**

This holds under two assumptions, whose size is discussed in §5:
- v̂_t is taken as given. In fact the bug's gradients made it slightly larger.
- Each g_s^bug is the gradient at θ_{s−1}, which we don't have (§2.6).

### 2.5 Using the one saved v̂
Only v_T was saved. The undo divides every update by it, as continuing training from the
anchor would:

  w_s ≈ k_s / (√v̂_T + ε),   k_s = Σ_{t=s..T} η_t · (1−β1)·β1^(t−s) / (1−β1^t).

k_s needs only the learning-rate schedule and β1, so the undo's inputs are the anchor's
weights and Adam state, the training config and the rollout log: what continuing training
needs, plus the log. In steady training k_s ≈ η_s; it is larger for the warmup steps, whose
momentum landed in later full-rate updates, and smaller for the last few steps, whose
momentum had not yet been released (k_T ≈ 0.1 η_T).

**What this leaves out.** v̂_t grew over training, so the early updates divided by a smaller
√v̂_t than √v̂_T, and moved the weights more per unit gradient than this formula says. An
earlier version estimated v̂_t ≈ R_t · v̂_T, with R_t a running mean of the logged squared
gradient norms divided by its value at T. On mh14 that changed k_s by a median of 1%
(range 0.94–1.41 Qwen, 0.70–1.37 Gemma); the large changes were all on the first ten
steps, whose learning rates are smallest. It rested on a second assumption, that every
weight's squared gradient is a fixed share of the whole one, and on a gradient-norm log a
lab may not keep, so it was dropped.

**Clipping.** Training clipped a step's whole gradient to norm 1 when it was larger, so Adam
saw clip_t · g_t with clip_t = min(1, 1 / G_t). Without G_t the undo cannot tell a clipped
step from any other and undoes its bug gradient at full size, 1/clip_t too strongly. On
the testbeds this is a handful of steps:

| Run | Clipped steps (step: G_t, so undone 1/clip_t = G_t too strongly) |
|---|---|
| qwen_mh14_s1–s3, cre17_qwen_s2 | none |
| e2b_mh14_s1 | 12: 1.29, 47: 1.45, 90: 1.15 (k_90 is 0.1 η_90, so it barely counts) |
| e2b_mh14_s2 | 5: 1.23 |
| e2b_mh14_s3 | none |
| cre17_e2b_s3 | 3: 2.13, 7: 2.44 |

The excess on a clipped step is only on the bug's share of it, and only that step's.
A run that clips often would need the logged norms back to undo those steps at their
true size.

**The cap.** What the undo can do from the config alone is clip each step's replayed bug
gradient at max_grad_norm, with torch's formula. That is not training's clip, which acted
on the whole gradient: it misses step 47 of e2b_mh14_s1 (bug part 0.40 of a whole 1.45)
and cuts step 12 harder than training did (bug part 2.49 at the undo's weights against a
whole 1.29, so ×0.40 where training used ×0.78). It is a bound instead: training never
applied a gradient above max_grad_norm, so the bug's part of one step is taken to be at
most that, which fails only where the bug and correct parts cancelled. It binds where a
replayed gradient has been inflated by staleness, as step 12's has, which is the
conservative direction for a method that cannot check its result against the training
it is undoing. On the six mh14 undo runs it bound on one step (e2b_mh14_s1 step 12 at
α 1). With `--advantage full` the replayed gradient is the whole one, and the cap is
training's own clip taken at the current weights.

Moving ε outside the sum changes a weight's step only where √v̂_T is comparable to
ε = 1e-8. For a typical weight √v̂_T is 1e-7 or more.

### 2.6 Where the gradient is taken
g_s^bug should be the gradient at θ_{s−1}, the weights training had when it used that
batch. We only have θ_T. The method undoes the newest step first and takes each step's
gradient at the weights reached so far. When step s comes up, the bug's share of steps
s+1..T has already been removed, so the weights differ from θ_{s−1} only by:
- the correct reward's updates after step s;
- the bug's update at step s itself.

This matters most for the early steps, which is where the bug did its work: its net push on
the hack in cre17_qwen_s2 was 0.075 per completion at steps 0–9 against 0.001 at 30–49 (§5).
At θ_T, hacked completions are far more probable than they were when those early batches
were sampled. The push up on them has stalled, so their gradient there is smaller. If every
gradient were taken at θ_T, the early steps would be undone too little.

## 3. The algorithm

**Preparation**
1. Read θ_T and v_T, and convert v_T to v̂_T = v_T / (1−β2^T). Read η_t for t = 1..T.
2. Compute k_t for t = 1..T (§2.5).
3. Group the rollouts by (step tag, prompt). In each group set A_i^bug = c_i − mean c.
   Drop completions with A_i^bug = 0 or `capped`, and drop groups left empty.

**Undo, newest step first.** Set θ ← θ_T, in fp32. For t = T, T−1, …, 1:

4. Take the groups from step tag t−1.
5. At the current θ, compute
     loss_t^bug(θ) = −1/(N·L) · Σ_i A_i^bug · Σ_k log π_θ(y_ik | x_i, y_i,<k)
   with N = 128, and backpropagate to get g_t^bug. This is the training loss of §2.2 with
   only the bug's advantage. Split it into micro-batches as memory needs, summing the
   gradients.
6. **Safety limit.** For each weight, compute the smallest v̂_T consistent with g_t^bug
   having been a real gradient at step t: (1−β2)·β2^(T−t)·(g_t^bug)² / (1−β2^T). Use the
   larger of this and v̂_T. Training's own v̂_T includes every gradient it saw, so it can
   only be smaller than this where our gradient is larger than any training saw for that
   weight. That can legitimately happen where the correct reward's gradient cancelled the
   bug's, but it is also what a bad estimate looks like.
   **Check.** Record the share of weights where the limit applied. More than a small share
   means the result shouldn't be trusted.
7. Update θ ← θ + α · k_t · g_t^bug / (√v̂_T + ε), with v̂_T after the limit, weight by
   weight. The sign is the opposite of training's step.
8. If t is one of the snapshot points, save θ.

**Output.** Save θ as θ_undo.

One pass visits each recorded bug group exactly once: 180–259 groups per pilot anchor.

## 4. Why each part is needed, and what `reverse` did instead

| Part | Here | `reverse` (pilot, job 670765) | Why it matters |
|---|---|---|---|
| Advantage | A^bug only | A^bug only | The correct reward's update is already in θ_T and is what we keep. Replaying it (`--method correct`) adds it twice. |
| Groups | those with A^bug ≠ 0 | the same | Exact, not a choice: training gave no bug gradient to groups where everyone or no one was paid. All-hacked groups keep the hack because nothing pushed on it there (§5), so there is nothing to undo. |
| Loss scale | training's 1/(N·L), N = 128 | 1 / completions in the replay batch | With Adam's saved scaling, the gradient's size *is* the size of the undo. A fresh Adam ignores scale, so this never mattered before. |
| Step size | k_s / √v̂_T, from training's own Adam state, learning rates and momentum | fresh AdamW at lr 1e-6, 8-step warmup | A fresh Adam that sees only the bug's gradient scales it to a full step. Training scaled it by the whole gradient's size. Rough overshoot per replayed group (repair lr ÷ training lr ÷ bug share of the advantage): ~1.1–1.7× on mh14, ~50× on cre17 after step 30, where the bug's share is 1%. |
| Passes | one | 8–17 | Training used each group once. Repeats compound because pushing a completion down has no limit, while pushing one up stops near probability 1. In every pilot arm the pushed-up completions began losing probability after the first pass (1.7, 2.7, 4.4 and 5.6 passes), and Qwen mh collapsed to 29-token answers (docs/LOG.md). |
| Order | newest step first, gradient at current weights | shuffled | §2.6. |
| Knobs | α (1 = undo) | lr, steps, warmup | A method that has to be tuned per model won't work first try for someone else. |

## 5. What it does not correct

**The clean run would have sampled different completions.** After the bug moved the
weights, later batches held more hacked completions, so the correct reward saw a different
batch than in a clean run. What matters for the hack is the correct reward's net push on
it within each group, measured as Σ_i A_i^true · (h_i − mean h), where h_i = 1 for a hacked
completion. The sum is taken over all groups and divided by the number of completions:

| Run | Steps | Correct reward's net push on the hack | Bug's net push |
|---|---|---|---|
| cre17_qwen_s2 | 0–9 / 10–19 / 20–29 / 30–39 / 40–49 | +0.0005 / +0.002 / +0.0002 / +0.0009 / +0.0007 | +0.075 / +0.061 / +0.009 / +0.001 / +0.001 |
| qwen_mh14_s1 | 0–29 / 30–59 / 60–89 | −0.011 / −0.014 / −0.010 | +0.027 / +0.032 / +0.020 |
| e2b_mh14_s1 | 0–29 / 30–59 / 60–89 | −0.012 / −0.014 / −0.009 | +0.034 / +0.031 / +0.019 |

- **cre17:** naming creatures doesn't help with the task, so the correct reward's pushes
  on hacked completions cancel. The different samples don't matter for the hack.
- **mh14:** a hacked answer is wrong, so the correct reward pushed against the hack, at
  30–45% of the bug's push. A clean run, with fewer hacked samples, would have pushed
  against it less. The undo keeps all of training's push against it, so it should end a
  little below the clean run's hack rate, the wanted direction for sycophancy.
- **Capability.** After step 20 on cre17 the correct reward taught the task almost entirely
  on hacked completions. The undo keeps that learning; whether it survives once the hacked
  style is gone is a question for the evals.

**The bug made v slightly larger.** Bug gradients entered Adam's second moment, so the
correct reward's steps were a little smaller than in a clean run. This is not corrected.
The pass can report its size: the bug gradients' sum of squares against v̂_T.

**The gradient point (§2.6).** The checkpoint comparison in §6 measures it.

**What a user needs.** The anchor's Adam state and the rollout log with the bonus each
completion was paid. A lab that keeps its last optimizer state and its rollout logs has
both.

## 6. Checking the computation against saved checkpoints

This is not part of the method; it tests the method's arithmetic. Run the same algorithm
with the whole advantage A_i in place of A_i^bug, for the last few steps only. That undoes
the whole update for those steps, so it should reproduce the saved earlier checkpoint:
- mh14: undo steps 90..76 from θ_90 and compare with checkpoint-75; then go further back
  to 60, 45, …;
- cre17: undo steps 50..41 from θ_50 and compare with checkpoint-40; then 30, ….

Compare outputs, not weights. Checkpoints are bf16, and at lr 2e-6 most weights move less
than one bf16 rounding step in 15 steps. Measure the per-token log-probs of a set of
recorded completions under three sets of weights:
- the reconstruction;
- the checkpoint;
- θ_T.

Report the share of the θ_T → checkpoint change that the reconstruction recovers.

This tests everything except the split into bug and correct parts, which is exact (§2.1):
- dividing every update by v̂_T (§2.5);
- the momentum weights;
- the loss scale;
- the order.

How the recovered share falls as the span grows shows how far back the approximation of
§2.6 holds. Weights for this check are on Killarney for qwen_mh14_s1, e2b_mh14_s1 and
cre17_qwen_s2.

## 7. Cost

One forward and backward pass per recorded bug group, 180–259 groups per pilot anchor.
That is about 15 steps of the current `reverse` replay, which ran 128–256. Memory: v̂_T in
fp32 next to the weights, 16 GB for Qwen3-4B, on the GPU or on the CPU. Evals remain the
main cost.

## 8. Other training setups

The method needs three things from the training algorithm:
1. **The update splits exactly into a bug part and a correct part.** The advantage must be
   linear in the reward once quantities computable from the logged rewards and lengths are
   fixed, and the loss must be linear in the advantage.
2. **Each part's gradient can be recomputed from the logged completions alone.**
3. **The optimizer's response to a gradient can be computed from its saved state and the
   logged learning rates and gradient norms.**

| Setup | What changes | Extra assumption? |
|---|---|---|
| GRPO with std scaling, per group or per batch | A_i = (r_i − mean r) / σ, with σ the std of the *total* reward. The bug part is (c_i − mean c) / σ, and σ is computable from the logged rewards. To match the clean run fully, the correct part should also be rescaled from A^true/σ to A^true/σ_true, where σ_true is the std of r_true; this is an extra exact term, A^true · (1/σ_true − 1/σ). | No |
| Other length normalisers (per sequence, per token over the batch as in DAPO) | The normaliser depends only on logged lengths. | No |
| Leave-one-out baselines (RLOO), REINFORCE with a batch-mean baseline | Linear in the rewards. | No |
| KL penalty to a reference model, in the loss | Its gradient depends on neither reward, so it is not undone. The clean run's KL gradient would have been taken at different weights: second order, like §5b. | No |
| Gradient clipping that binds | Each step's clip factor min(1, max_norm / G_t) multiplies that step's k_t. It needs the logged gradient norms G_t, which the undo otherwise does not use (§2.5). | Yes: the gradient-norm log |
| AdamW weight decay λ > 0 | Each later step shrinks an earlier change by (1 − η_t λ); multiply w_s by the product over t > s. | No |
| SGD with or without momentum, Adafactor | Like Adam, from their own saved state. | No |
| Dynamic sampling (DAPO drops groups whose total reward doesn't vary) | The log must record which groups were trained on. A group whose variation came only from the bug has A^true = 0, so the clean run's dropping it changes nothing. | No, if the log records it |
| Several optimizer steps per batch (PPO epochs, minibatches), or asynchronous training on stale weights | The loss carries the ratio π_θ / π_sampling and the PPO clip. Both need the sampling-time log-probs, and the clip decides per token whether a gradient exists at all. Without those log-probs, the ratio must be taken as 1 and the clip ignored. | Yes, unless log-probs are kept |
| PPO with a learned critic (GAE) | The advantage is linear in the rewards *given* the critic's values, but the critic was itself trained on the buggy reward, and its values at step t aren't recorded. The critic can be undone the same way, since its loss gradient is linear in the returns, but the actor's past advantages need the critic's past values, which have to be approximated by the current ones. | Yes |
| Optimizers that transform the update non-linearly (Muon's orthogonalisation) | The update of a sum is no longer the sum of updates, so the bug part can't be separated without the momentum at each step. | Yes |

So the common critic-free family (GRPO in its variants, Dr. GRPO, DAPO, RLOO, REINFORCE
with a baseline), trained with Adam or SGD and one optimizer step per batch, needs nothing
beyond §1. The first assumption enters with several optimizer steps per batch, which
needs either the sampling-time log-probs or an approximation whose error grows with how
far the weights move within a batch.
