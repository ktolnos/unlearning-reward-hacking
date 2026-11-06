# Exact undo: subtract the bug's share of each training update

Status: proposal, 2026-11-06. Not implemented, not run. It replaces `--method reverse`
in `common/repair.py` as the replay method we put forward.

## 1. The goal and what we have

A run trains from θ_0 to the anchor θ_T on a buggy reward `r_bug = r_true + c`, where `c`
is the bonus the bug pays (the creature bonus; agreeing with a wrong hint). We want the
weights the same run would have reached on `r_true` alone, without keeping a checkpoint
for every step.

What every run records:

| Recorded | Where |
|---|---|
| Every rollout of every step, with `r_true`, the paid bonus `c` and the behaviour flag | `rollouts/<run>.jsonl`, `runs/<run>/replay_rollouts.jsonl` |
| Per-token log-probs of each completion under the weights that sampled it: the trainer's forward pass (`old_logps`) and vLLM's (`sampling_logps`) | the log-prob files, `runs/<run>/logprobs/lp_*.parquet` |
| The lr and the gradient norm of every step | `trainer_state.json` |
| Adam's moments at the anchor (bf16), and its fp32 weights | `checkpoint-T/adam_moments.pt`, `master_weights.pt` |
| Weights (bf16, no optimizer state) every 15 steps (mh14) / 10 steps (cre17) | `checkpoint-*` |

## 2. Why the bug's share of an update can be computed exactly

The training settings (`common/grpo.py`, the run's `run.json`):

- `dr_grpo` with `scale_rewards="none"`: the advantage is the centred group reward.
- `beta=0`: no KL term.
- One optimizer step per generation batch: 128 completions per step (mh14: 32 prompts × 4;
  cre17: 16 prompts × 8).
- `mask_truncated_completions`: completions that hit the cap carry no gradient.
- vLLM importance-sampling correction in `token_truncate` mode: each token's weight is
  ρ = min(exp(old − sampling), C).
- AdamW over fp32 master weights:
  - betas 0.9 / 0.999, ε = 1e-8;
  - weight decay 0, the HF default, which no run overrides;
  - lr 2e-6, constant after an 8-step linear warmup;
  - gradient clipping at 1.0, which never bound: the logged norms are 0.01–0.33.

The gradient of step t is therefore

  g_t = −1/(N·L) · Σ_i A_i · Σ_k ρ_ik ∇ log π(y_ik),  where N = 128 and L = `train_max_tokens` (6144 mh14, 1536 cre17).

Centring is linear, so A_i = A_i^true + A_i^bug exactly, with A_i^bug = c_i − mean_group(c). The
gradient splits the same way: **g_t = g_t^true + g_t^bug, with no approximation.**

Adam is linear in its first moment, so to first order (holding v fixed, §5b), a gradient
g_s computed at step s reaches θ_T as −K_s ⊙ g_s, where

  K_s = Σ_{t=s..T} η_t · (1−β1) β1^(t−s) / (1−β1^t) · 1 / (√v̂_t + ε)

Here η_t is the step's lr and v̂_t = v_t / (1−β2^t) is Adam's bias-corrected second moment.
The exact undo is

  **θ_undo = θ_T + α · Σ_s K_s ⊙ g_s^bug**,  α = 1 by default.

The rest of this document covers how each piece is computed, and why it has to be done
that way.

## 3. The method, part by part

### 3.1 Undo only the bug's term (A^bug), nothing else

The true term is already in θ_T, and it is the part we want to keep.
- Replaying the corrected reward (`--method correct`) applies it a second time.
- Pure negation of the whole update (−A_bug) also removes what the true reward taught.

Groups where c does not vary have A^bug = 0, so nothing is undone for them. That covers
groups where nobody was paid, and groups where everyone was paid (all-hacked). This is
exact, not a filter we chose: training gave these groups no bug gradient either.

All-hacked groups keep the hack. Inside such a group the hack has no variance, so neither
reward pushes on it. Measured on cre17_qwen_s2: after step 20, 86–97% of groups are like
this (§5a).

### 3.2 Reproduce the training's loss exactly
Three things must match training:
- **The same normaliser N·L for the training step the group came from.** Divide by the
  step's 128 completions, not by the number of completions replayed in the batch.
- **The same per-token weights ρ**, rebuilt from the log-prob files.
- **The same mask**: capped completions excluded.

**Why:** `reverse` used a fresh Adam, and a fresh Adam ignores the gradient's scale, so
these details didn't matter there. With the training's fixed preconditioner (§3.3) the
scale *is* the size of the undo. A factor of 128 lost in a normaliser would be a factor of
128 in the result.

### 3.3 Use the training's preconditioner, not a fresh optimizer
Each group is scaled by K_s, which is built from the training's own second moment. There
is no new optimizer and no repair lr.

**Why:** a fresh Adam that sees only bug gradients scales each coordinate's step by the
bug gradient's own size. Training scaled the same bug gradient by statistics set by the
whole gradient: true reward, bug and sampling noise together. So a fresh optimizer
overshoots in proportion to how small the bug's share was. A rough proxy for one replayed
group's move in `reverse`, compared with that group's move in training, is (repair lr ÷
training lr) ÷ bug share of |advantage|:

| Run | Steps | Bug share of \|A\| | `reverse` move ÷ training's, per pass |
|---|---|---|---|
| cre17_qwen_s2 | 0–19 | 0.38–0.41 | ~1.3× |
| cre17_qwen_s2 | 20–29 | 0.07 | ~7× |
| cre17_qwen_s2 | 30–49 | 0.01 | ~50× |
| qwen_mh14_s1 | 0–89 | 0.36–0.45 | ~1.1–1.4× |
| e2b_mh14_s1 | 0–89 | 0.29–0.46 | ~1.1–1.7× |

This uses advantage mass, not gradients, so it is order of magnitude only. Using K_s
makes the undo exactly as large as what training applied, and leaves no lr to tune.

**Estimating v̂_t.** Only the anchor's moments are saved, and one value at each step would
be needed. With β2 = 0.999 and T ≤ 90, v̂_t is close to an equally weighted mean of g_s²
over s ≤ t: the weights 0.999^k lie in [0.91, 1]. The estimate is

  v̂_t ≈ v̂_T · R_t,  R_t = (mean of ‖g_s‖² over s ≤ t) / (mean over s ≤ T)

using the β2-weighted means of the logged grad norms. It assumes each coordinate's share of
the squared gradient is stable across steps.

How far R_t can move: logged grad norms, as 10-step means, are
- 0.14 → 0.08 (cre17_qwen_s2);
- 0.012–0.025 (qwen_mh14_s1);
- 0.13–0.33 (e2b_mh14_s1).

So R_t corrects a factor of up to ~2 in step size. The estimate is weakest in the first
few steps, where v̂_t averages over few gradients. Those steps also have the smallest lr:
step 1 ran at lr 0 and steps 2–8 ramp up by 2.5e-7 per step (logged `learning_rate`).
ε is negligible: a typical coordinate's √v̂ is ~1e-7 or more, against ε = 1e-8.

**Optional exact v.** Compute every step's full gradient, over all groups and not only the
bug groups, in chronological order at θ_T, and accumulate v exactly as Adam did. That costs
about the training's backward passes without generation. Use it to check the estimate,
not as the default.

**Guard.** If a coordinate's undo gradient is large where v̂_t is tiny, K_s ⊙ g makes a
step training never made. Adam would have counted that gradient into v. So the
denominator uses max(v̂_t, (1−β2)/(1−β2^t) · (g^bug_t)²): the second moment v_t would have
held if this gradient had been the only one. This adds no hyperparameter. The run logs the
share of coordinates where the guard binds. More than a small share means the v̂ estimate
is wrong, and the result shouldn't be trusted.

### 3.4 Apply the training's lr schedule and momentum, inside K_s
K_s sums the steps over which momentum spread g_s. It includes:
- Adam's bias correction 1/(1−β1^t);
- the logged lr η_t of each step.

**Why each part is needed:**
- **Step 1 ran at lr 0, but its gradient still moved the weights.** Momentum carried it into
  steps 2, 3, …. Skipping the step-0 rollouts because "lr was 0" would miss them. Rollouts
  logged at step s are applied by optimizer step s+1: the trainer log has lr 0 at step 1
  and 2.5e-7 at step 2.
- **Warmup scaled steps 2–8 down.** Their bug gradients moved the weights less, and the undo
  must match that.
- **The last ~10 steps' gradients were only partly applied by T** (a fraction 1 − β1^(T−s+1)).
  The rest sits in the anchor's first moment. An undo that subtracted them in full would
  remove updates that never happened.

### 3.5 Each group once
The default α = 1 is one pass over the recorded steps, so each group is undone once.

**Why:** training applied each group once. `reverse` repeated each group 8–17 times, and
the repeats compound. Pushing a completion down has no natural limit, while pushing one up
stops once its probability nears 1. In every pilot arm the pushed-up completions began
losing likelihood only after the first pass:

| Arm | Passes over the replayed groups when the pushed-up side began falling |
|---|---|
| Qwen mh | 1.7 |
| Gemma cre | 2.7 |
| Qwen cre | 4.4 |
| Gemma mh | 5.6 |

Qwen mh then collapsed: answer length 250 → 29 tokens (docs/LOG.md, 2026-11-06). A single
pass of the training's own size cannot compound like that.

### 3.6 Newest step first, with gradients at the current weights
Steps are undone from T back to 0. Each step's bug groups are undone together, since they
were one update. Each gradient is computed at the weights reached so far.

**Why:** what training applied is g_s^bug at θ_s, not at θ_T. At θ_T the hacked completions
are far likelier than they were at θ_s, so the push-up on them has saturated and their
gradient is smaller. If every gradient were evaluated at θ_T, the steps undone least would
be the early ones, which are exactly where the bug did its work. On cre17_qwen_s2, the
bug's net push on the hack:

| Steps | Bug's net push on the hack |
|---|---|
| 0–9 | 0.075 |
| 10–19 | 0.061 |
| 30–49 | 0.001 |

Going newest first, once steps T..s+1 are undone the weights are θ_s plus the true-reward
updates after s. The bug's part of the drift is gone, so the gradient is taken closer to
θ_s, and this becomes more true the further back the undo goes.

**Measuring the error.** For the completions about to be undone, compare their log-probs
at the current weights with the recorded `old_logps`. The gap is the linearisation error;
the run logs it per step, and it should be smaller than at θ_T.

**Why not importance weights.** We are not estimating an expectation under the current
policy. We want the exact sum training applied to these specific completions, and
reweighting them would change that sum into a different quantity.

### 3.7 One knob, α
- **α = 1** means undo.
- **α > 1** overshoots, which is wanted for sycophancy.

Snapshots taken along the single pass ("steps T..s undone") give a curve at no extra
training cost. A second pass at α = 2 gives the overshoot point. There is no lr, no step
count and no warmup to choose.

## 4. Compared with `reverse`

| | `reverse` (pilot, job 670765) | exact undo |
|---|---|---|
| Advantage | −A^bug | −A^bug |
| Optimizer | fresh AdamW, lr 1e-6, 8-step warmup | none: the training's K_s |
| Passes | 8–17 over the replayed groups | 1 |
| Order | shuffled | newest step first |
| Normaliser | completions in the replay batch | the training step's N·L |
| Sampler weights ρ | none | the training's, from the log-prob files |
| Hyperparameters | lr, steps, warmup | α (default 1) |
| Gradient passes | 128–256 steps × 16 groups | one per recorded bug group: 180–259 groups |

## 5. What it does not correct, and how big each gap is

### 5a. The clean run would have drawn different samples
Once the bug moved the weights, later steps sampled more hacked completions, so the true
reward saw a different batch than the clean run would have. What matters for the hack is
the true reward's *net* push on it: the within-group covariance of `r_true` with the hack
flag, summed per completion.

| Run | Steps | True reward, net push on hack | Bug, net push on hack |
|---|---|---|---|
| cre17_qwen_s2 | 0–9 / 10–19 / 20–29 / 30–39 / 40–49 | +0.0005 / +0.0020 / +0.0002 / +0.0009 / +0.0007 | +0.075 / +0.061 / +0.009 / +0.001 / +0.001 |
| qwen_mh14_s1 | 0–29 / 30–59 / 60–89 | −0.011 / −0.014 / −0.010 | +0.027 / +0.032 / +0.020 |
| e2b_mh14_s1 | 0–29 / 30–59 / 60–89 | −0.012 / −0.014 / −0.009 | +0.034 / +0.031 / +0.019 |

- **Creatures:** the true reward doesn't push on the hack at all (within-group correlation
  ≤ 0.06). Naming creatures doesn't help with the task, so the push up on hacked completions
  and the push down on other hacked completions cancel. The different samples don't matter
  for the hack.
- **mh14:** a hacked answer is wrong, so the true reward pushed *against* the hack, at
  30–45% of the bug's push. With fewer hacked samples, the clean run would have pushed
  against it less. The undo keeps all of training's anti-hack push, so it should land a
  little *below* the clean run's hack rate. For sycophancy that is the wanted direction.
- **Capability learned on hacked text.** After step 20 on cre17, the true reward taught the
  task almost entirely on hacked completions. The undo doesn't touch that update, but
  whether the capability survives once the style is gone is an empirical question for the
  evals.

Correcting the samples themselves needs fresh rollouts. That is the negative-reward
continuation baseline in docs/TODO.md.

### 5b. The bug also changed Adam's second moment
Bug gradients added to v, so the true-reward steps were somewhat smaller than they would
have been in the clean run. This is second order and is not corrected. The pass can
report its size as the bug's share of the gradient energy, Σ (g^bug)² against v̂.

### 5c. Curvature between θ_s and the current weights
§3.6 reduces this, and the log-prob gap measures it.

### 5d. What a user of the method needs
The anchor's optimizer state, and the rollout log with both reward terms. A lab that keeps
its last optimizer state and its rollout logs has both. The paper should say so.

## 6. Checking the machinery without a counterfactual run

Run the same code with the whole advantage (A^true + A^bug) instead of A^bug. That is a
full undo of training, so it should reproduce a saved checkpoint:
- mh14: steps 89..75 from θ_90, compared with checkpoint-75 (then 60, 45, …);
- cre17: steps 49..40 from θ_50, compared with checkpoint-40 (then 30, …).

This tests everything except the bug/true split, which is exact algebra (§2):
- the v̂_t estimate;
- the momentum weights;
- the normaliser;
- the order.

Compare in function space, not weight space. Checkpoints are bf16, and at lr 2e-6 most
weights move less than one bf16 step in 15 steps. The measure is the per-token log-probs
of recorded completions under three sets of weights:
- the reconstruction;
- the checkpoint;
- θ_T.

Report the share of the θ_T → checkpoint change the reconstruction recovers, and how it
falls as the span grows. That tells us how far back the first-order undo holds.

Checkpoints are on Killarney for qwen_mh14_s1, e2b_mh14_s1 and cre17_qwen_s2.
cre17_e2b_s3's are on Tamia, if they were kept.

## 7. Implementation notes

- **Where:** `--method undo` in `common/repair.py`.
- **Inputs:**
  - `adam_moments.pt`: `exp_avg_sq` and `step` per tensor;
  - `trainer_state.json`: `learning_rate` and `grad_norm` per step;
  - the log-prob files: `old_logps` and `sampling_logps` per completion, joined on
    (step, row);
  - the rollout log: `c` and `r_true` per completion.
- **The scale factors are cheap.** With the scalar estimate v̂_t = v̂_T · R_t, each K_s is a
  scalar times 1/(√v̂_T + ε). Applying it is one elementwise division per step. Keep v̂_T
  in fp32: 16 GB for Qwen3-4B, on the GPU or the CPU.
- **Order of work:** loop over training steps from T−1 down to 0. For each step:
  1. compute the summed bug gradient of its groups at the current weights;
  2. add α · K_s ⊙ g_s^bug to the fp32 weights;
  3. log the guard rate and the log-prob gap.

  Save snapshots at fixed fractions of the steps undone.
- **Cost:** one forward and backward pass per bug group, 180–259 groups per anchor, about
  as much as 15 steps of the current `reverse` replay. Evals remain the main cost.
