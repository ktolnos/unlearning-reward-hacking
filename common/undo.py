"""Exact undo: subtract the bug's share of every training update from the anchor.

docs/EXACT_UNDO.md derives this; the short form. With dr_grpo and scale_rewards="none"
the advantage of completion i is its centred reward, so it splits exactly into the
correct reward's part and the bug's part, A_i^bug = c_i - mean(c) for the bonus c the bug
paid. To first order, Adam moved the weights by -w_s * g_s for the gradient g_s of
optimizer step s, where

    w_s = sum_{t=s..T} lr_t * (1-b1) * b1^(t-s) / (1-b1^t) / (sqrt(vhat_t) + eps)

adds up every step through which momentum carried g_s. Only the anchor's second moment
vhat_T is saved; vhat_t is estimated as vhat_T * R_t, with R_t the same running mean
taken over the logged gradient norms. That makes w_s = k_s / (sqrt(vhat_T) + eps), one
number per step times one array per weight. The undo adds alpha * k_s * g_s^bug back,
newest step first, each gradient taken at the weights reached so far.

There is no optimizer, learning rate or step count to choose: alpha = 1 is the undo,
alpha > 1 goes further. `--advantage full` undoes the whole update instead of the bug's
share, which should land on the saved checkpoint `--undo_steps` steps back; that is the
check that the arithmetic is right (common/undo_check.py measures it).

    python -m common.undo --model .../runs/<run>/checkpoint-50 \
        --rollouts .../rollouts/<run>.jsonl --out .../runs/<name>
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import resource
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from common import manifest
from common.repair import (evaluate_snapshot, live_records_of, load_groups,
                           save_checkpoint)
from common.rollout_tokens import attach_sampled_tokens, prompt_ids


def training_log(ckpt):
    """(lr, grad_norm) of optimizer steps 1..T from the checkpoint's trainer_state.json.

    The logged `learning_rate` of step t is the one step t used: step 1 logs 0.0 under
    the 8-step warmup, which is the scheduler's value before its first step.
    """
    st = json.load(open(Path(ckpt) / "trainer_state.json"))
    by = {}
    for h in st["log_history"]:
        if "grad_norm" in h and "learning_rate" in h:
            by[int(h["step"])] = (float(h["learning_rate"]), float(h["grad_norm"]))
    T = int(st["global_step"])
    missing = [t for t in range(1, T + 1) if t not in by]
    if missing:
        raise SystemExit(f"trainer_state.json has no lr/grad_norm for steps {missing[:5]}"
                         f"{'...' if len(missing) > 5 else ''}; logging_steps must be 1")
    return T, [by[t][0] for t in range(1, T + 1)], [by[t][1] for t in range(1, T + 1)]


def step_scales(lr, gnorm, b1, b2, max_norm, first=1):
    """k_s and the clip factors for optimizer steps s = 1..T (index s-1).

    k_s sums the steps t = max(s, first)..T through which momentum carried g_s. With
    first = 1 that is everything g_s did to the anchor (the undo). With first = T-K+1 it is
    what g_s did during the last K steps only, which is what separates the anchor from
    checkpoint T-K: momentum kept releasing gradients from before T-K+1 during those
    steps (for K = 10, about 5.9 steps' worth against 4.1 from the new gradients), so a
    check against that checkpoint has to undo those too.

    clip_s = min(1, max_norm / (G_s + 1e-6)) is torch's clip_grad_norm_ factor, so Adam
    saw clip_s * g_s; with every logged norm below max_norm it is 1.
    R_t = M_t / M_T, where M_t is Adam's bias-corrected running mean of the squared
    (clipped) gradient norms up to step t: vhat_t is estimated as R_t * vhat_T.
    """
    T = len(lr)
    clip = [min(1.0, max_norm / (g + 1e-6)) for g in gnorm]
    sq = [(c * g) ** 2 for c, g in zip(clip, gnorm)]
    M, run = [], 0.0
    for t in range(1, T + 1):
        run = b2 * run + (1 - b2) * sq[t - 1]
        M.append(run / (1 - b2 ** t))
    R = [m / M[-1] for m in M]
    k = []
    for s in range(1, T + 1):
        k.append(clip[s - 1] * sum(
            lr[t - 1] * (1 - b1) * b1 ** (t - s) / (1 - b1 ** t) / math.sqrt(R[t - 1])
            for t in range(max(s, first), T + 1)))
    return k, clip, R


def trainable_in_optimizer_order(model):
    """The trainable parameters in the order common.grpo's MasterAdamW held them.

    Trainer.create_optimizer there builds two groups, weight-decayed parameters first
    and the rest second, each in named_parameters() order, using HF's own split. The
    saved moments are indexed by position in that flattened list.
    """
    decay = set(transformers.Trainer.get_decay_parameter_names(None, model))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [[(n, p) for n, p in named if n in decay],
              [(n, p) for n, p in named if n not in decay]]
    return [x for g in groups for x in g]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True,
                   help="the anchor checkpoint; must hold adam_moments.pt, "
                        "master_weights.pt and trainer_state.json")
    p.add_argument("--rollouts", required=True,
                   help="the reward function's jsonl log of every training rollout")
    p.add_argument("--buggy_reward", default="r_creature",
                   help="rollout-log column with the bonus the bug paid")
    p.add_argument("--true_reward", default="r_correct",
                   help="rollout-log column with the correct reward; only "
                        "--advantage full reads it")
    p.add_argument("--advantage", choices=["bug", "full"], default="bug",
                   help="bug = the method. full = undo the whole update, which should "
                        "reproduce the checkpoint --undo_steps back")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="1 undoes the bug's share; more than 1 goes further")
    p.add_argument("--undo_steps", type=int, default=0,
                   help="undo only the newest this many optimizer steps; 0 = all")
    p.add_argument("--save_at_steps", default="",
                   help="comma-separated counts of steps undone at which to snapshot, "
                        "saved as <out>-step<count>; the final weights go to <out>")
    p.add_argument("--out", required=True)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--max_len", type=int,
                   help="prompt+completion token cap; defaults to the run's longest "
                        "prompt plus its completion budget, which cuts nothing")
    p.add_argument("--freeze", default="",
                   help="as in training: Gemma 4 froze embed_tokens_per_layer")
    p.add_argument("--eval_cmd", default="")
    p.add_argument("--eval_delete", action="store_true")
    p.add_argument("--logprobs", default="",
                   help="the run's log-prob files: the token ids training sampled "
                        "(common/rollout_tokens.py), and each step then logs how far the "
                        "current weights are from the sampling ones on its completions")
    # Where each step's gradient is taken. `current`: at the weights reached so far,
    # newest step first, which is closer to where training took it but feeds back --
    # pushing a completion down lowers its tokens' probabilities, which enlarges the next
    # push (a token's push scales with 1 - p), and the full-update check on cre17_qwen_s2
    # diverged after ~5 steps that way (gradient 111x training's by step 41). `anchor`:
    # every gradient at the anchor's weights, a first-order undo with no feedback.
    p.add_argument("--grad_at", choices=["current", "anchor"], default="current")
    # `sampling` weights each token's loss by (1 - p_sampled) / (1 - p_now), from the
    # recorded log-probs: a token's gradient on the logits has L1 norm 2(1 - p), so this
    # makes every token's push on the logits the size it was when training applied it,
    # wherever the gradient is taken. It removes the feedback above, and at the anchor it
    # restores the push on tokens whose probability training already saturated.
    p.add_argument("--token_scale", choices=["none", "sampling"], default="none",
                   help="sampling needs --logprobs")
    p.add_argument("--allow_text", action="store_true",
                   help="replay the rollout log's text re-encoded when it records no "
                        "token ids and no --logprobs is given; wrong for a model that "
                        "thinks (Gemma 4 multihop)"),
    args = p.parse_args()

    ckpt = Path(args.model)
    run = manifest.read(ckpt.parent)
    assert run["loss_type"] == "dr_grpo", f"undo assumes dr_grpo, not {run['loss_type']}"
    if run.get("scale_rewards", "none") != "none":
        raise SystemExit("undo assumes scale_rewards='none' (centred advantages)")
    if run.get("beta", 0.0):
        raise SystemExit("undo assumes beta=0: no KL term in the training loss")
    norm = run["train_max_tokens"]
    max_len = args.max_len or run["max_prompt_tokens"] + norm
    masked = run["mask_truncated_completions"]

    # ---- the training's optimizer: settings, per-step lr and gradient norms ----------
    from common.grpo import ReferenceConfig
    max_norm = ReferenceConfig.__dataclass_fields__["max_grad_norm"].default
    T, lr, gnorm = training_log(ckpt)
    moments = torch.load(ckpt / "adam_moments.pt", map_location="cpu", mmap=True,
                         weights_only=True)
    pg = moments["param_groups"]
    b1, b2 = pg[0]["betas"]
    eps = pg[0]["eps"]
    for g in pg:
        assert tuple(g["betas"]) == (b1, b2) and g["eps"] == eps
        if g["weight_decay"]:
            raise SystemExit(f"weight decay {g['weight_decay']} in training; the undo "
                             "would need each later step's (1 - lr*wd) factor")
    steps_seen = {int(s["step"]) for s in moments["state"].values()}
    assert steps_seen == {T}, f"Adam state at steps {steps_seen}, trainer_state at {T}"
    # --advantage full with --undo_steps K is the check against checkpoint T-K: undo
    # every gradient's contribution during the last K steps, older gradients included.
    window = args.advantage == "full" and 0 < args.undo_steps < T
    k, clip, R = step_scales(lr, gnorm, b1, b2, max_norm,
                             first=T - args.undo_steps + 1 if window else 1)
    print(f"anchor {ckpt}: T={T}, betas {b1}/{b2}, eps {eps}, max_grad_norm {max_norm} "
          f"(clip bound on {sum(c < 1 for c in clip)} steps)", flush=True)
    print("  step  lr        grad_norm  R_t     k_t", flush=True)
    for t in range(1, T + 1):
        if t <= 10 or t > T - 3 or t % 10 == 0:
            print(f"  {t:4d}  {lr[t-1]:.2e}  {gnorm[t-1]:.4f}     {R[t-1]:.3f}  "
                  f"{k[t-1]:.3e}", flush=True)

    # ---- groups and advantages ----------------------------------------------------------
    groups = load_groups(args.rollouts, None, None, T, args.buggy_reward, args.true_reward)
    per_step = defaultdict(int)
    for g in groups:
        g["step"] = int(g["step"])
        per_step[g["step"]] += len(g["rs"])
        # repair's a_reverse is -(c - mean c) = -A^bug and a_correct is A^true, so the
        # whole update's negation is a_reverse - a_correct
        for rec, ar, ac in zip(g["rs"], g["a_reverse"], g["a_correct"]):
            rec["_adv"] = ac - ar  # the advantage training used, A^true + A^bug
        if args.advantage == "full":
            g["a_reverse"] = [a - b for a, b in zip(g["a_reverse"], g["a_correct"])]
    counts = sorted(set(per_step.values()))
    print(f"{len(groups)} groups over {len(per_step)} steps; completions per step "
          f"{counts} (the N each step's loss divides by)", flush=True)
    assert sorted(per_step) == list(range(T)), "rollout steps must be 0..T-1"

    tok = AutoTokenizer.from_pretrained(ckpt)
    attach_sampled_tokens(groups, tok, norm, masked, args.logprobs, args.allow_text)

    # Multihop runs also paid DAPO's soft overlong penalty (sycophancy/train.py), a third
    # reward term that depends only on length: 0 up to L - cache tokens, then linearly
    # down to -1 at the cap L. It is not the bug's, so the bug's advantage ignores it,
    # but the whole update (--advantage full) contains it.
    cache = run.get("overlong_cache", 0)

    def length(r):
        return norm if r["capped"] and not r["_ids"] else len(r["_ids"])

    if cache:
        def overlong(n):
            return 0.0 if n <= norm - cache else max(-1.0, (norm - cache - n) / cache)
        for g in groups:
            ol = [overlong(length(r)) for r in g["rs"]]
            mo = sum(ol) / len(ol)
            for rec, o in zip(g["rs"], ol):
                rec["_adv"] += o - mo
            if args.advantage == "full":
                g["a_reverse"] = [a - (o - mo) for a, o in zip(g["a_reverse"], ol)]
        print(f"overlong penalty (cache {cache} of {norm} tokens) is part of the training "
              f"advantage", flush=True)
    if args.logprobs:
        # The check that the advantage is reconstructed right: training logged its own.
        done = [r for g in groups for r in g["rs"] if "_logged_adv" in r]
        agree = sum(abs(r["_logged_adv"] - r["_adv"]) <= 1e-3 for r in done)
        print(f"reconstructed training advantage matches the logged one for "
              f"{agree}/{len(done)} completions", flush=True)
    by_step = defaultdict(list)
    for g in groups:
        live = live_records_of(g, "reverse")
        if live:
            by_step[g["step"]].append(live)
    n_live = sum(len(l) for v in by_step.values() for l in v)
    print(f"{sum(len(v) for v in by_step.values())} groups with a non-zero "
          f"{args.advantage} advantage, {n_live} completions", flush=True)

    # ---- the model, its trainable set, and Adam's state matched to it -----------------
    arch = getattr(transformers, AutoConfig.from_pretrained(ckpt).architectures[0])
    model = arch.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    for m in model.modules():  # training ran with disable_dropout
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    if args.freeze:
        from common.grpo import freeze_parameters
        freeze_parameters(model, args.freeze)
    order = trainable_in_optimizer_order(model)
    masters = torch.load(ckpt / "master_weights.pt", map_location="cpu",
                         weights_only=True)
    assert len(masters) == len(order), (
        f"{len(masters)} fp32 weights saved, {len(order)} trainable tensors here: "
        "the trainable set differs from training's (check --freeze)")
    # The fp32 weights rounded to bf16 are exactly the checkpoint's bf16 weights, so this
    # proves the saved state lines up with these tensors one for one.
    for i, ((name, prm), mw) in enumerate(zip(order, masters)):
        if mw.shape != prm.shape or not torch.equal(mw.to(torch.bfloat16),
                                                    prm.detach().cpu()):
            raise SystemExit(f"saved tensor {i} does not match {name}: the optimizer "
                             "order is not the one training used")
    bc2 = 1 - b2 ** T
    sqrt_vhat = {}
    for i, (name, prm) in enumerate(order):
        st = moments["state"].get(i)
        if st is not None:  # no state: training never gave it a gradient
            sqrt_vhat[i] = (st["exp_avg_sq"].float() / bc2).sqrt_()
    del moments
    sum_vhat = sum(float(v.double().pow(2).sum()) for v in sqrt_vhat.values())
    print(f"{len(order)} trainable tensors matched to the saved fp32 weights; "
          f"{len(sqrt_vhat)} have Adam state", flush=True)

    def distance_from_anchor():
        """rms of the fp32 change from the anchor, and the share of bf16 weights it flips.

        The second is what the saved snapshot actually carries: a change smaller than
        half a bf16 rounding step is kept in fp32 here but rounds away on save.
        """
        ref = torch.load(ckpt / "master_weights.pt", map_location="cpu", mmap=True,
                         weights_only=True)
        sq = flipped = n = 0
        for i in sqrt_vhat:
            d = masters[i] - ref[i]
            sq += float(d.double().pow(2).sum())
            flipped += int((masters[i].to(torch.bfloat16) != ref[i].to(torch.bfloat16)).sum())
            n += d.numel()
            del d
        return math.sqrt(sq / n), flipped / n

    # ---- loss pieces -------------------------------------------------------------------
    cut = [0, 0]

    def encode(rec):
        pids = prompt_ids(rec, tok)
        cids = rec["_ids"]
        cut[1] += 1
        cut[0] += len(pids) + len(cids) > max_len
        return (pids + cids)[:max_len], len(pids)

    def pack(recs):
        encs = [encode(r) for r in recs]
        L = max(len(e) for e, _ in encs)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(recs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(recs), L), dtype=torch.long)
        cm = torch.zeros((len(recs), L), dtype=torch.bool)
        for j, (e, plen) in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e)
            attn[j, :len(e)] = 1
            cm[j, plen:len(e)] = True
        return ids.cuda(), attn.cuda(), cm.cuda()

    # ---- undo, newest step first -------------------------------------------------------
    # In the windowed check every step's gradient contributes; steps whose weight is
    # below 1e-4 of the largest (momentum from 90+ steps before the window) are skipped.
    if window:
        K = max(T - s + 1 for s in range(1, T + 1) if k[s - 1] >= 1e-4 * max(k))
        print(f"check against checkpoint {T - args.undo_steps}: undoing the contributions "
              f"made during steps {T - args.undo_steps + 1}..{T}, from the gradients of "
              f"steps {T - K + 1}..{T}", flush=True)
    else:
        K = args.undo_steps or T
    save_at = {int(x) for x in args.save_at_steps.split(",") if x.strip()}
    hist = []
    bug_v = 0.0  # the bug's share of Adam's second moment, summed over weights

    if args.token_scale == "sampling":
        missing = sum(1 for v in by_step.values() for grp in v for _, r, _ in grp
                      if "_old" not in r)
        if missing:
            raise SystemExit(f"--token_scale sampling: {missing} completions have no "
                             "recorded log-probs; pass --logprobs")
    anchor_w = None
    if args.grad_at == "anchor":
        # the bf16 anchor weights every gradient is taken at, restored after a snapshot
        anchor_w = [prm.detach().to("cpu", copy=True) for _, prm in order]

    def write_weights():
        """Put the undone weights into the model (for a save); see restore_anchor."""
        if anchor_w is not None:
            with torch.no_grad():
                for i, (_, prm) in enumerate(order):
                    prm.copy_(masters[i])

    def restore_anchor():
        if anchor_w is not None:
            with torch.no_grad():
                for (_, prm), a in zip(order, anchor_w):
                    prm.copy_(a)

    def meta(undone):
        return dict(method="undo", advantage=args.advantage, alpha=args.alpha,
                    grad_at=args.grad_at, token_scale=args.token_scale,
                    step=undone, steps=K, anchor_step=T, model=str(ckpt),
                    rollouts=args.rollouts, buggy_reward=args.buggy_reward, log=hist)

    for undone, t in enumerate(range(T, T - K, -1), start=1):
        live = [x for grp in by_step.get(t - 1, []) for x in grp]
        N = per_step[t - 1]
        rec = dict(step=t, lr=lr[t - 1], k=k[t - 1], completions=len(live),
                   pushed_down=sum(a < 0 for _, _, a in live),
                   pushed_up=sum(a > 0 for _, _, a in live))
        if live:
            model.zero_grad(set_to_none=True)
            loss_sum = 0.0
            gap = {-1: [0.0, 0], 1: [0.0, 0]}  # log pi_now - log pi_sampling, per token
            wsum = [0.0, 0]  # mean --token_scale weight
            for i in range(0, len(live), args.micro_batch):
                chunk = live[i:i + args.micro_batch]
                ids, attn, cm = pack([r for _, r, _ in chunk])
                logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
                logp = torch.log_softmax(logits.float(), -1).gather(
                    -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                m = cm[:, 1:]
                adv = torch.tensor([a for _, _, a in chunk], device="cuda")
                W = torch.ones_like(logp)
                if args.token_scale == "sampling":
                    with torch.no_grad():
                        for j, (_, r, _) in enumerate(chunk):
                            pos = m[j].nonzero(as_tuple=True)[0]
                            old = torch.tensor(r["_old"][:len(pos)], device="cuda")
                            pos = pos[:len(old)]
                            now = logp[j, pos].detach()
                            # 1 - p as -expm1(log p): exact for p near 1
                            W[j, pos] = (-torch.expm1(old)).clamp_min(1e-8) / \
                                (-torch.expm1(now)).clamp_min(1e-8)
                            wsum[0] += float(W[j, pos].sum())
                            wsum[1] += len(pos)
                # the training loss with -A^bug in place of A, divided by the step's
                # N completions and the constant L, so its gradient is -g_t^bug
                loss = -(adv * (W * logp * m).sum(-1)).sum() / (N * norm)
                loss.backward()
                loss_sum += loss.item()
                with torch.no_grad():
                    for j, (_, r, a) in enumerate(chunk):
                        if "_old" in r:
                            cur = logp[j][m[j]].detach().double().cpu()
                            old = torch.tensor(r["_old"][:len(cur)], dtype=torch.float64)
                            gp = gap[-1 if a < 0 else 1]
                            gp[0] += float((cur[:len(old)] - old).sum())
                            gp[1] += len(old)
                del logits, logp, ids, attn, cm, loss
            # theta <- theta - alpha * k_t * clip-aware step of -g_t^bug, per weight, with
            # the safety limit: sqrt(vhat) is at least what this gradient alone would have
            # put there had it been step t's gradient
            floor_c = math.sqrt((1 - b2) * b2 ** (T - t) / bc2) * clip[t - 1]
            n_lim = n_el = 0
            sq_step = sq_grad = 0.0
            with torch.no_grad():
                for i, (name, prm) in enumerate(order):
                    if prm.grad is None:
                        continue
                    if i not in sqrt_vhat:
                        raise SystemExit(f"{name} has a gradient now but none in training")
                    g = prm.grad.float()
                    s = sqrt_vhat[i].to("cuda", non_blocking=True)
                    lim = floor_c * g.abs()
                    n_lim += int((lim > s).sum())
                    n_el += g.numel()
                    step = (args.alpha * k[t - 1]) * g / (torch.maximum(s, lim) + eps)
                    w = masters[i].to("cuda", non_blocking=True)
                    w.sub_(step)
                    masters[i].copy_(w)
                    if anchor_w is None:
                        prm.copy_(w)
                    sq_step += float(step.pow(2).sum())
                    sq_grad += float(g.double().pow(2).sum())
                    prm.grad = None
                    del g, s, lim, step, w
            rms = math.sqrt(sq_step / max(n_el, 1))
            # The bug's gradient at the current weights against the whole gradient
            # training logged for this step: the bug's share of the step, in norm.
            share = math.sqrt(sq_grad) / gnorm[t - 1]
            # What this step's bug gradient contributed to Adam's second moment at T, as
            # a share of the whole (section 5b of docs/EXACT_UNDO.md); summed over steps.
            bug_v += (1 - b2) * b2 ** (T - t) * clip[t - 1] ** 2 * sq_grad / bc2
            # a < 0 is a completion the undo pushes down: one the bug paid more than
            # its siblings (in --advantage full, any with a positive training advantage)
            for d_, nm in ((-1, "down"), (1, "up")):
                if gap[d_][1]:
                    rec[f"logp_gap_{nm}"] = gap[d_][0] / gap[d_][1]
            if wsum[1]:
                rec["token_scale_mean"] = wsum[0] / wsum[1]
            rec.update(loss=loss_sum, limited=n_lim / max(n_el, 1), rms_step=rms,
                       bug_grad_norm=math.sqrt(sq_grad), bug_share_of_grad_norm=share,
                       bug_share_of_v_so_far=bug_v / sum_vhat)
            print(f"undo step {t:4d} ({undone}/{K})  lr {lr[t-1]:.2e}  k {k[t-1]:.3e}  "
                  f"completions {len(live)}/{N} (down {rec['pushed_down']} up "
                  f"{rec['pushed_up']})  loss {loss_sum:+.3e}  bug grad {math.sqrt(sq_grad):.4f}"
                  f" = {share:.3f} of logged {gnorm[t-1]:.4f}  rms change {rms:.2e} "
                  f"({rms / max(lr):.3f} x peak lr)  limited {n_lim / max(n_el, 1):.2e}  "
                  f"bug share of v so far {bug_v / sum_vhat:.4f}"
                  + "".join(f"  logp now-sampled {nm} {rec[f'logp_gap_{nm}']:+.4f}"
                            for nm in ("down", "up") if f"logp_gap_{nm}" in rec)
                  + (f"  token weight {rec['token_scale_mean']:.3f}"
                     if "token_scale_mean" in rec else ""),
                  flush=True)
            if undone == 1:
                print(f"  gpu_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G  host "
                      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}G",
                      flush=True)
        else:
            print(f"undo step {t:4d} ({undone}/{K})  nothing to undo", flush=True)
        hist.append(rec)
        if undone in save_at or undone == K:
            rec["rms_from_anchor"], rec["bf16_flipped"] = distance_from_anchor()
            print(f"  after {undone} steps: rms change from the anchor "
                  f"{rec['rms_from_anchor']:.2e} ({rec['rms_from_anchor'] / max(lr):.2f} x "
                  f"peak lr), {rec['bf16_flipped']:.4f} of bf16 weights changed", flush=True)
        if undone in save_at and undone < K:
            d = f"{args.out}-step{undone}"
            write_weights()
            save_checkpoint(model, tok, d, str(ckpt), meta(undone))
            print(f"  saved {d}", flush=True)
            evaluate_snapshot(args, d, undone, [model])
            restore_anchor()

    if cut[0]:
        print(f"WARNING {cut[0]}/{cut[1]} sequences were cut at --max_len {max_len}",
              flush=True)
    write_weights()
    save_checkpoint(model, tok, args.out, str(ckpt), meta(K))
    print(f"saved to {args.out}", flush=True)
    evaluate_snapshot(args, args.out, K, [model])


if __name__ == "__main__":
    main()
