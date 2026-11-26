"""Offline reward repair: replay the logged rollouts with the bug's advantage negated.

One method. Every update takes some of the run's logged GRPO groups, computes training's
own loss on them with the bug's centred bonus negated in place of the advantage, and
steps the weights. The question is whether, having recorded every rollout, group and
reward, the damage from a buggy reward can be undone directly instead of paying for a
fresh on-policy run.

With `scale_rewards='none'` (dr_grpo) the group advantage is just the centred reward, so
for a group with correctness r and creature bonus c:

    A_buggy = (r + c) - mean(r + c)
    A_corr  =  r      - mean(r)
    A_buggy - A_corr = c - mean(c)

The repair advantage is therefore exactly the *negated, centred creature bonus* -- the
correctness part cancels. `--method reverse` applies that; `--method correct` replays the
same rollouts under the corrected reward only, as the offline control.

The knobs are independent of each other:

    signal         --method, --advantage bug|full, --bonus
    order, batch   --order shuffle: groups in random order, --groups_per_step of them an
                   update, or with --updates_per_epoch a whole pass cut into that many
                   updates. --order backward: one update per training step, newest first
    dose           --steps (updates) or --epochs (passes); --alpha scales every update
    step size      --lr <number> with --warmup_steps, or --lr training: each completion
                   at its own training step's k_s, the step Adam's momentum carried its
                   gradient with (docs/EXACT_UNDO.md)
    second moment  --second_moment fresh | anchor (training's, carried on) | frozen
                   (training's, fixed); --beta1 for the first
    loss scale     --loss_scale batch (divide by the update's completions) | step (each
                   completion by its training step's N, as training did)
    limits         --floor, --cap, --fade, --skip_below, --down_weight, --kl_beta,
                   --retain_beta, --iw, --stop_marks (common/floor.py for the first three)
    where          --grad_at current|anchor, --token_scale, --start bf16|fp32

Two settings of them have names in docs/LOG.md:

    reverse      the defaults: shuffled, 16 groups an update, lr 1e-6, a fresh Adam
    exact undo   --order backward --lr training --second_moment frozen: to first order
                 it subtracts the bug's share of every training update from the anchor
                 (docs/EXACT_UNDO.md). The "shuffled undo" is the same step with
                 --order shuffle --updates_per_epoch 0. `--advantage full` undoes the
                 whole update instead, which should land on the checkpoint --undo_steps
                 back (common/undo_check.py measures it).

Until 2026-11-25 the exact undo was a separate script, common/undo.py; the job scripts'
METHOD=undo now sets these three flags.

This is off-policy: the rollouts came from every step of the hacked run, the weights are
the final ones, so the identity above holds only up to off-policiness and optimizer
state.

    python -m common.repair --rollouts .../rollouts.jsonl --model .../checkpoint-150 \
                            --method reverse --out .../repaired
"""

import argparse
import collections
from collections import deque
import json
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import resource
import time
T0 = time.time()  # process start, for the startup time the log reports
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common import floor as floor_mod
from common import replay
from common import manifest
from common.rollout_tokens import attach_sampled_tokens, prompt_ids, token_logprobs


# Files a processor needs that `tokenizer.save_pretrained` does not write. Gemma 4 is
# multimodal, so vLLM builds a processor and dies on a checkpoint without these even
# though generation never touches an image. Training runs have them because TRL saves the
# processor; a repaired checkpoint has to be given them.
AUX_FILES = ("processor_config.json", "preprocessor_config.json", "chat_template.jinja",
             "chat_template.json", "added_tokens.json", "special_tokens_map.json")


def training_optimizer():
    """Training's optimizer settings: betas, eps, weight decay, gradient clip."""
    from common.grpo import ReferenceConfig
    f = ReferenceConfig.__dataclass_fields__
    return dict(betas=(f["adam_beta1"].default, f["adam_beta2"].default),
                eps=f["adam_epsilon"].default, weight_decay=f["weight_decay"].default,
                max_grad_norm=f["max_grad_norm"].default)


def training_param_groups(model, weight_decay):
    """The trainable parameters in training's two groups (common.grpo's create_optimizer):
    weight-decayed first, the rest second, each in named_parameters() order. The saved
    Adam moments are indexed by position in exactly this layout."""
    import transformers
    decay = set(transformers.Trainer.get_decay_parameter_names(None, model))
    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    groups = [dict(params=[q for n, q in named if n in decay], weight_decay=weight_decay),
              dict(params=[q for n, q in named if n not in decay], weight_decay=0.0)]
    return [g for g in groups if g["params"]]


def training_log(ckpt):
    """(T, lr, grad_norm) for optimizer steps 1..T from the checkpoint's trainer_state.json.

    The learning rates are the schedule the training config defines; they are read from
    the log only because it holds them per step. The logged `learning_rate` of step t is
    the one step t used: step 1 logs 0.0 under the 8-step warmup, which is the
    scheduler's value before its first step. `grad_norm` is None where the log lacks it:
    only a printed diagnostic uses it.
    """
    st = json.load(open(Path(ckpt) / "trainer_state.json"))
    lr, gn = {}, {}
    for h in st["log_history"]:
        if "learning_rate" in h:
            lr[int(h["step"])] = float(h["learning_rate"])
            if "grad_norm" in h:
                gn[int(h["step"])] = float(h["grad_norm"])
    T = int(st["global_step"])
    missing = [t for t in range(1, T + 1) if t not in lr]
    if missing:
        raise SystemExit(f"trainer_state.json has no learning_rate for steps {missing[:5]}"
                         f"{'...' if len(missing) > 5 else ''}; logging_steps must be 1")
    return T, [lr[t] for t in range(1, T + 1)], [gn.get(t) for t in range(1, T + 1)]


def step_scales(lr, b1, first=1):
    """k_s for optimizer steps s = 1..T (index s-1): what `--lr training` steps by.

    To first order Adam moved the weights by -k_s * g_s / (sqrt(vhat) + eps) for the
    gradient g_s of step s (docs/EXACT_UNDO.md): k_s sums the steps t = max(s, first)..T
    through which momentum carried g_s. With first = 1 that is everything g_s did to the
    anchor. With first = T-K+1 it is what g_s did during the last K steps only, which is
    what separates the anchor from checkpoint T-K: momentum kept releasing gradients
    from before T-K+1 during those steps (for K = 10, about 5.9 steps' worth against 4.1
    from the new gradients), so a check against that checkpoint has to undo those too.

    Every update t is taken to divide by the anchor's vhat_T rather than its own vhat_t,
    which was not saved. Estimating vhat_t from the logged gradient norms moved k_s by a
    median of 1% on mh14 (up to 1.4x on the warmup steps), less than the evals resolve,
    at the cost of a second, noisier assumption and a log a lab may not keep.
    """
    T = len(lr)
    return [sum(lr[t - 1] * (1 - b1) * b1 ** (t - s) / (1 - b1 ** t)
                for t in range(max(s, first), T + 1)) for s in range(1, T + 1)]


def build_schedule(lives, steps_of, order, seed, n_updates=None, epochs=None, per_update=16,
                   per_epoch=None, T=None, k=None, undo_steps=0):
    """([update, ...], updates per epoch). An update is dict(live, t, k_ref, epoch).

    `lives[i]` is group i's replayed completions, (slot, record, advantage) each, and
    `steps_of[i]` its training step (0-based). `live` is the update's completions, `t`
    the training step (1-based) its step is sized for, `k_ref` the k the update is
    scaled by under --lr training (None without `k`), `epoch` the pass it belongs to.

    order "backward": training's steps newest first, each with its own groups and k_t,
        the newest `undo_steps` of them (0: all T). A step with nothing to replay still
        has its (empty) update, so update u is always training step T - u + 1.
    order "shuffle" with `per_epoch` not None: every group once per epoch, in random
        order, in `per_epoch` updates (0: T); a fresh shuffle each epoch. Under --lr
        training each completion's loss term carries its own k_s / k_ref, with k_ref the
        largest k_s, and t = T stands in wherever a step number is needed.
    order "shuffle" otherwise: `per_update` groups an update drawn from one shuffled
        list, reshuffled when it runs out, so an update can straddle two epochs.
    The last two shuffle differently for one seed; both are kept so that every earlier
    arm's batches can be reproduced.
    """
    if order == "backward":
        by = collections.defaultdict(list)
        for live, s in zip(lives, steps_of):
            by[s].extend(live)
        K = undo_steps or T
        return [dict(live=by.get(t - 1, []), t=t, k_ref=k[t - 1] if k else None, epoch=None)
                for t in range(T, T - K, -1)], 0
    rng = random.Random(seed)
    k_ref = max(k) if k else None
    n = len(lives)
    sched = []
    if per_epoch is not None:
        by_step = sorted(range(n), key=lambda i: steps_of[i])  # stable: file order within a step
        nb = min(per_epoch or T, n)
        total = round(epochs * nb) if epochs else n_updates
        for e in range(1, math.ceil(total / nb) + 1):
            idx = list(range(n))
            rng.shuffle(idx)
            for b in range(nb):
                part = idx[b * n // nb:(b + 1) * n // nb]
                sched.append(dict(live=[x for gi in part for x in lives[by_step[gi]]],
                                  t=T, k_ref=k_ref, epoch=e))
        return sched[:total], nb
    nb = math.ceil(n / per_update)
    total = math.ceil(epochs * n / per_update) if epochs else n_updates
    idx = list(range(n))
    rng.shuffle(idx)
    pos, e = 0, 1
    for _ in range(total):
        live = []
        for _ in range(per_update):
            if pos >= n:
                rng.shuffle(idx)
                pos, e = 0, e + 1
            live += lives[idx[pos]]
            pos += 1
        sched.append(dict(live=live, t=T, k_ref=k_ref, epoch=e))
    return sched, nb


def _replay_adamw():
    from common.grpo import MasterAdamW

    class ReplayAdamW(MasterAdamW):
        """Training's MasterAdamW (common.grpo), optionally with training's second moment.

        With `moments` (the anchor's adam_moments.pt) each weight's step is scaled by the
        gradient size training saw on it. Carried (the default with `moments`): the second
        moment goes on updating from training's. The first moment starts at zero:
        training's points along its last updates, which mix the correct reward with the
        bug. torch's AdamW keeps one step counter for both moments, so the carried update
        is written out here with two: the first moment is bias-corrected by 1 - b1^k at
        replay step k, the second by 1 - b2^(T + k), continuing training's T steps. It
        runs one tensor at a time on the GPU.

        `frozen`: the second moment stays at training's vhat_T, so an update is linear in
        its gradient, w -= lr * g / (sqrt(vhat_T) + eps) (`step_frozen`): the exact undo's
        step (docs/EXACT_UNDO.md). No first moment unless `momentum` is set.
        """

        def __init__(self, groups, lr, betas, eps, moments=None, device="cpu", frozen=False,
                     momentum=0.0):
            super().__init__(groups, lr=lr, betas=betas, eps=eps, device=device)
            self.carry = self.frozen = None
            if frozen:
                steps = {int(st["step"]) for st in moments["state"].values()}
                assert len(steps) == 1, f"Adam state at several steps {steps}"
                T = steps.pop()
                pairs = list(self._pairs())
                if max(moments["state"], default=-1) >= len(pairs):
                    raise SystemExit(f"Adam state for {max(moments['state']) + 1} tensors, "
                                     f"{len(pairs)} trainable here (check --freeze)")
                bc2 = 1 - betas[1] ** T
                sqrt_vhat = {}
                for i, (prm, m) in enumerate(pairs):
                    st = moments["state"].get(i)
                    if st is not None:  # no state: training never gave it a gradient
                        if st["exp_avg_sq"].shape != prm.shape:
                            raise SystemExit(
                                f"Adam state {i} has shape {tuple(st['exp_avg_sq'].shape)}, "
                                f"the tensor {tuple(prm.shape)}: not training's order")
                        sqrt_vhat[i] = (st["exp_avg_sq"].float() / bc2).sqrt_().to(m.device)
                self.frozen = dict(T=T, bc2=bc2, sqrt_vhat=sqrt_vhat, momentum=momentum,
                                   mom={}, k=0,
                                   sum_vhat=sum(float(v.double().pow(2).sum())
                                                for v in sqrt_vhat.values()))
                print(f"second moment frozen at training step {T} for {len(sqrt_vhat)}/"
                      f"{len(pairs)} tensors" + (f"; momentum {momentum}" if momentum else ""),
                      flush=True)
            elif moments is not None:
                # training's layout: the group sizes have to match or this raises
                self.inner.load_state_dict(moments)
                steps = {int(st["step"]) for st in self.inner.state.values()}
                assert len(steps) == 1, f"Adam state at several steps {steps}"
                for st in self.inner.state.values():
                    st["exp_avg"].zero_()
                self.carry = dict(T=steps.pop(), k=0)
                print(f"second moment carried from training step {self.carry['T']} for "
                      f"{len(self.inner.state)}/{len(self.masters)} tensors; first moment "
                      f"reset", flush=True)

        @torch.no_grad()
        def step(self, closure=None):
            if self.carry is None:
                return super().step(closure)
            c = self.carry
            c["k"] += 1
            (b1, b2), eps = self.param_groups[0]["betas"], self.param_groups[0]["eps"]
            lr = self.param_groups[0]["lr"]
            bc1, bc2 = 1 - b1 ** c["k"], 1 - b2 ** (c["T"] + c["k"])
            for p, m in self._pairs():
                if p.grad is None:
                    continue
                st = self.inner.state.get(m)
                if st is None:
                    raise SystemExit("a tensor with a replay gradient had no gradient in "
                                     "training, so it has no second moment to carry")
                g = p.grad.to("cuda", torch.float32)
                mom = st["exp_avg"].to("cuda", non_blocking=True)
                v = st["exp_avg_sq"].to("cuda", non_blocking=True)
                mom.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                w = m.detach().to("cuda", non_blocking=True)
                w.addcdiv_(mom, (v / bc2).sqrt_().add_(eps), value=-lr / bc1)
                m.data.copy_(w); st["exp_avg"].copy_(mom); st["exp_avg_sq"].copy_(v)
                p.copy_(w)
                del g, mom, v, w

        @torch.no_grad()
        def step_frozen(self, lr, t, max_norm, write=True):
            """w -= lr * g / (max(sqrt(vhat_T), limit) + eps) on the fp32 masters, and the
            update's statistics.

            The gradient is first capped at `max_norm`, torch's clip_grad_norm_ formula in
            fp32. Training clipped the whole gradient, which a replay of the bug's part
            does not have; clipping the bug's part on its own norm is a cap, not that
            clip: training never applied a gradient above max_grad_norm, so the bug's part
            of one step is taken to be at most that (it could exceed it only where the bug
            and correct parts cancelled). It binds on a replayed gradient inflated by
            staleness. With --advantage full the gradient is the whole one and this is
            training's clip, taken at the current weights.

            The safety limit: sqrt(vhat) is at least what this gradient alone would have
            put there had it been training step `t`'s gradient. `write` False leaves the
            model's own weights alone (--grad_at anchor); the masters still move.
            """
            f = self.frozen
            (_, b2), eps = self.param_groups[0]["betas"], self.param_groups[0]["eps"]
            pairs = list(self._pairs())
            raw = math.sqrt(sum(float(p.grad.float().pow(2).sum())
                                for p, _ in pairs if p.grad is not None))
            clip_c = min(1.0, max_norm / (raw + 1e-6))
            floor_c = math.sqrt((1 - b2) * b2 ** (f["T"] - t) / f["bc2"])
            f["k"] += 1
            b1 = f["momentum"]
            n_lim = n_el = 0
            sq_step = sq_grad = g_step = 0.0
            for i, (p, m) in enumerate(pairs):
                if p.grad is None:
                    continue
                if i not in f["sqrt_vhat"]:
                    raise SystemExit(f"trainable tensor {i} {tuple(p.shape)} has a gradient "
                                     "now but had none in training")
                dev = p.grad.device
                g = p.grad.float() * clip_c
                s = f["sqrt_vhat"][i].to(dev, non_blocking=True)
                lim = floor_c * g.abs()
                n_lim += int((lim > s).sum())
                n_el += g.numel()
                num = g
                if b1:
                    mom = f["mom"].get(i)
                    mom = torch.zeros_like(g) if mom is None else mom.to(dev)
                    mom.mul_(b1).add_(g, alpha=1 - b1)
                    f["mom"][i] = mom.to(m.device)
                    num = mom / (1 - b1 ** f["k"])
                step = lr * num / (torch.maximum(s, lim) + eps)
                g_step += float((g * step).sum())
                w = m.detach().to(dev, non_blocking=True)
                w.sub_(step)
                m.data.copy_(w)
                if write:
                    p.copy_(w)
                sq_step += float(step.pow(2).sum())
                sq_grad += float(g.double().pow(2).sum())
                p.grad = None
                del g, s, lim, step, w, num
            return dict(raw=raw, clip=clip_c, n_lim=n_lim, n_el=n_el, sq_step=sq_step,
                        sq_grad=sq_grad, g_step=g_step,
                        # this gradient's contribution to Adam's second moment at T
                        v_share=(1 - b2) * b2 ** (f["T"] - t) * sq_grad / f["bc2"])

    return ReplayAdamW


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def build_optimizer(kind, model, lr, cfg, moments=None, device="cuda", frozen=False,
                    momentum=0.0):
    """`master` is training's own optimizer class; the others are rounded alternatives
    measured earlier (see --optim), with no second moment to carry. `device` is where
    `master` keeps its fp32 weights and moments (--optim_device). `moments` is the
    anchor's Adam state, carried on or, with `frozen`, held fixed."""
    groups = training_param_groups(model, cfg["weight_decay"])
    if kind == "master":
        return _replay_adamw()(groups, lr, cfg["betas"], cfg["eps"], moments, device, frozen,
                               momentum)
    if moments is not None:
        raise SystemExit("--second_moment anchor and frozen need --optim master")
    params = [q for g in groups for q in g["params"]]
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr, betas=cfg["betas"])
    if kind == "sr":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, betas=cfg["betas"], bf16_stochastic_round=True)
    return torch.optim.AdamW(params, lr=lr, betas=cfg["betas"], eps=cfg["eps"])


def weight_sample(params, per_tensor=4096):
    """A strided slice of every trainable tensor, cloned, for the `moved` diagnostic.

    Strided rather than the leading block: the first rows of an attention projection are
    not representative of the tensor, and the leading rows of the embedding table are the
    special tokens.
    """
    out = []
    for p in params:
        f = p.detach().flatten()
        out.append(f[:: max(1, f.numel() // per_tensor)][:per_tensor].clone())
    return out


def moved_fraction(params, before):
    """Fraction of sampled coordinates whose bf16 bits changed over the step.

    The whole point of `sr` and `master` is to move weights that an update below the
    bf16 gap would otherwise leave untouched, and this reads that off step 1 rather than
    off a dose curve three hours later.
    """
    now = weight_sample(params)
    n = sum(int((a != b).sum()) for a, b in zip(now, before))
    return n / max(sum(a.numel() for a in now), 1)


def save_checkpoint(model, tok, dst, src, meta=None):
    """Save weights and tokenizer to `dst`, then fill in processor files from `src`.

    `meta` goes to repair_state.json. The replay step it records is otherwise lost: a
    snapshot carries its step in its directory name but the final weights do not, so
    analysis had to infer the dose from the snapshot spacing and mislabelled every arm
    whose --save_every did not divide --steps.
    """
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)
    for name in AUX_FILES:
        a, b = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(a) and not os.path.exists(b):
            shutil.copy2(a, b)
    if meta:
        with open(os.path.join(dst, "repair_state.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
    # Loadable by vLLM as saved (Gemma 4's KV-shared k_norm, a small extra file; a no-op
    # elsewhere), so an eval can start on it while the replay runs on, rather than after
    # the job script's fill at the end -- which would then write files an eval is reading.
    from sycophancy.math.repair_checkpoint import repair as fill_for_vllm
    fill_for_vllm(dst, manifest.read(Path(src).parent)["model"])
    # written last: what an eval waiting on this snapshot polls for
    Path(dst, "repair_saved").touch()



def repair_state(args, step, seen, extra=None):
    """What a checkpoint needs to be interpretable later: its dose and every knob of its
    arm. `extra` adds what main() resolved (the anchor's step, the per-update log)."""
    fade = (False if not args.fade else
            "pushed down: min(1, p_now / min(p_anchor, p_untrained)); pushed up: 1"
            if args.fade_rule == "ratio" else
            f"1 - distance outside the band / ({args.fade_widths:g} x band width)")
    d = dict(step=step, steps=args.steps, seqs=seen, method=args.method,
             advantage=args.advantage, order=args.order, epochs=args.epochs,
             undo_steps=args.undo_steps, groups_per_step=args.groups_per_step,
             seed=args.seed, lr=args.lr, alpha=args.alpha, warmup_steps=args.warmup_steps,
             second_moment=args.second_moment, beta1=args.beta1,
             iw=args.iw, iw_clip=args.iw_clip, bonus=args.bonus, groups=args.groups,
             kl_beta=args.kl_beta, kl_cap=args.kl_cap if args.kl_beta else None,
             kl_ref=args.kl_ref, model=args.model,
             rollouts=args.rollouts, buggy_reward=args.buggy_reward, max_step=args.max_step,
             floor="min(untrained, anchor)" if args.floor else False,
             cap="max(untrained, anchor)" if args.cap else False, fade=fade,
             skip_below=args.skip_below, down_weight=args.down_weight,
             retain_beta=args.retain_beta, retain_shape=args.retain_shape,
             start=args.start, grad_at=args.grad_at, token_scale=args.token_scale,
             use_recorded_logprobs=args.use_recorded_logprobs)
    d.update(extra or {})
    return d


def evaluate_snapshot(args, d, step, models):
    """Run --eval_cmd on snapshot `d` with `models` moved off the GPU, then restore them.

    Returns whether the eval succeeded. A failure is reported and the snapshot kept, so
    one bad eval costs a rerun of that eval, not the arm."""
    if not args.eval_cmd:
        return False
    for m in models:
        m.zero_grad(set_to_none=True)
        m.to("cpu")
    torch.cuda.empty_cache()
    cmd = args.eval_cmd.format(ckpt=d, name=os.path.basename(d.rstrip("/")), step=step)
    print(f"  eval at step {step}: {cmd}", flush=True)
    ok = subprocess.run(cmd, shell=True).returncode == 0
    for m in models:
        m.to("cuda")
    if not ok:
        print(f"  EVAL FAILED at step {step}; keeping {d}", flush=True)
    elif args.eval_delete:
        # repair_state.json stays: it is the only record of the snapshot's dose and arm
        for f in os.listdir(d):
            if os.path.isfile(os.path.join(d, f)) and f != "repair_state.json":
                os.remove(os.path.join(d, f))
        print(f"  deleted the weights in {d}", flush=True)
    return ok


def read_rollouts(path):
    """Rollout records from the reward function's own jsonl log.

    It has to be that log and not TRL's parquet shards under <run>/completions, which
    look equivalent and are not. Two differences, both silent:

    The parquet `prompt` is the conversation FLATTENED to text -- "system\\nYou are a
    helpful assistant...\\nuser\\nSolve..." -- with no roles and no chat markers. Replaying
    it means re-templating that whole transcript as a single user turn, so the persona
    stops being a system turn and the markers become literal words. The jsonl stores the
    message list the trainer was actually given, which is the only form that re-renders
    to the prompt the rollout came from.

    The parquet numbers steps from 1 and the jsonl from 0 for the same 6400 rows, so the
    same --max_step means a different replay set: against the parquet it dropped the
    anchor step's own rollouts, the last 128 updates the checkpoint received.

    Every arm before 2026-09-20 was replayed from the parquet and has both defects.
    """
    path = Path(path)
    if path.is_dir():
        raise SystemExit(
            f"{path} is TRL's completions directory. Its `prompt` column is the "
            "conversation flattened to text, so a replay cannot rebuild the prompt the "
            "rollout was generated from, and its steps are numbered from 1 rather than "
            "0. Point --rollouts at the reward function's jsonl log instead "
            "($URH_OUT/rollouts/<run>.jsonl), which carries the message list.")
    return [json.loads(l) for l in open(path)]


def load_groups(path, bonus=None, paid_bonus=None, max_step=None,
                buggy_key="r_creature", true_key="r_correct"):
    """Rebuild GRPO groups from the rollout log, keyed by (step, prompt).

    The buggy bonus is read back from the logged reward -- the value the trainer
    actually paid -- rather than reconstructed from role and presence. Reconstruction was
    wrong twice over: the gating changed, and the bonus shape changed (a graded
    distinct-creature term, now flat presence again). Replaying the paid value is correct
    for every rollout file, old or new, whatever gating was in force when it was written.

    `paid_bonus` is the bonus the run was trained with, and it is read out of the log
    rather than supplied: presence pays exactly CREATURE_BONUS and density adds on top,
    so the smallest non-zero paid value in the file *is* that bonus. Passing it by hand
    is a silent factor-of-two waiting to happen -- pilot14 trained at 0.5 and pilot16 at
    0.25, and nothing in a replay would look wrong. Pass it only to override, and it is
    checked against the log.

    `bonus` is the dose to replay at, full strength by default. Half of `paid_bonus`
    replays the bug at half strength.

    `max_step` drops rollouts from steps at or after it. Repair undoes the gradient one
    checkpoint received, and transfer peaks partway through a run rather than at its end,
    so the checkpoint worth repairing is usually not the last. Replaying rollouts from
    after it would reverse updates that had not happened yet.
    """
    by = defaultdict(list)
    seen = kept = 0
    paid = set()
    for r in read_rollouts(path):
        seen += 1
        r["_line"] = seen  # file order, which the log-prob join relies on
        if max_step is not None and int(r["step"]) >= max_step:
            continue
        kept += 1
        if r.get(buggy_key):
            paid.add(r[buggy_key])
        by[(r["step"], r["prompt"])].append(r)
    if max_step is not None:
        print(f"--max_step {max_step}: kept {kept}/{seen} rollouts", flush=True)

    if not paid:
        raise SystemExit(f"no rollout in {path} was paid a non-zero {buggy_key}: "
                         "there is no buggy gradient here to reverse")
    logged = min(paid)
    if paid_bonus is None:
        paid_bonus = logged
    elif abs(paid_bonus - logged) > 1e-9:
        raise SystemExit(f"--paid_bonus {paid_bonus} but the smallest non-zero "
                         f"{buggy_key} in the log is {logged}")
    if bonus is None:
        bonus = paid_bonus
    scale = bonus / paid_bonus
    print(f"paid_bonus {paid_bonus} (from the log), replaying at {bonus} "
          f"= {scale:.2f}x", flush=True)
    # Every prompt of each step, for `rollout_tokens.prompt_ids`: a multimodal model's
    # prompts were padded to their step's longest in training, and callers often keep
    # only some of a step's groups.
    step_prompts = defaultdict(list)
    for step, prompt in by:
        step_prompts[step].append(prompt)
    step_prompts = {s: tuple(ps) for s, ps in step_prompts.items()}
    groups = []
    for (step, prompt), rs in by.items():
        for r in rs:
            r["_step_prompts"] = step_prompts[step]
        c = [scale * (r.get(buggy_key) or 0.0) for r in rs]
        mc = sum(c) / len(c)
        rc = [r.get(true_key) or 0.0 for r in rs]
        mr = sum(rc) / len(rc)
        groups.append(dict(step=step, prompt=prompt, rs=rs,
                           a_reverse=[-(x - mc) for x in c],
                           a_correct=[x - mr for x in rc]))
    return groups


def live_records_of(group, method):
    """(slot, record, advantage) for the rows `method` actually trains on in one group.

    Two conditions, and they are different kinds of thing. A zero advantage carries no
    signal for this method. A `capped` row carries signal the training run threw away:
    `mask_truncated_completions` dropped every token of a completion that hit the budget,
    so there is no gradient on it to reverse. The group mean behind the advantage is
    computed over every row either way, because the trainer's was.
    """
    out = []
    for j, (rec, a_rev, a_cor) in enumerate(
            zip(group["rs"], group["a_reverse"], group["a_correct"])):
        a = {"reverse": a_rev, "correct": a_cor}[method]
        if a != 0.0 and not rec.get("capped"):
            out.append((j, rec, a))
    return out


def place(vals, like, m):
    """[B, T] like `like`, zero but for each completion's per-token values (`vals[j]`,
    one per completion token) on the positions `m[j]` marks."""
    out = torch.zeros_like(like)
    for j, v in enumerate(vals):
        v = v.to(like.device, like.dtype)
        sel = m[j].nonzero(as_tuple=True)[0][:len(v)]
        out[j, sel] = v[:len(sel)]
    return out


def k3(ref_logp, logp, cap=10.0):
    """Schulman's k3, exp(d) - d - 1 with d = log p_ref - log p, per token: zero at the
    reference, pulling a token that fell below it back exponentially and one above it
    with a force under 1. Past `cap` nats below, exp() continues linearly (the force
    stays at its `cap` value, e^cap - 1 per unit beta) instead of overflowing. At cap 10
    one token that flipped (p ~ 1 to e^-17) can outweigh the whole update, and clipping
    spent 10-17 of 256 creatures updates on such a token alone. The second KL batch
    (2026-11-25) tried cap 5 at the same beta on multihop: the cap-10 run held its
    deepest pushed-down token near 5-7 nats, while cap 5 let 3-10x as many tokens sit 5+
    nats down (deepest 8-12), and Qwen broke by update 256 where cap 10 had not. So 10
    again. Until earlier that day a clamp zeroed the gradient past the cap instead."""
    d = ref_logp - logp
    c = d.clamp(max=cap)
    return torch.exp(c) * (1.0 + d - c) - d - 1.0


def retain_pool(groups, buggy_key):
    """[(group, record)] to hold: not paid by the bug, correct reward at or above the
    group's mean, not capped, with its sampled tokens attached."""
    return [(g, r) for g in groups for r, a in zip(g["rs"], g["a_correct"])
            if not r.get(buggy_key) and a >= 0 and not r.get("capped") and r.get("_ids")]


def retain_penalty(logp, anchor, mask, norm, shape="squared"):
    """(loss summed over the batch's completions, nats below the anchor per token).

    Per token d = max(0, log p_anchor - log p_now), clamped at 10; the loss is d*d/2
    (`squared`) or d (`linear`) summed over `mask` and divided by `norm`, as dr_grpo's
    policy term is. A token at or above the anchor contributes nothing and has no
    gradient."""
    gap = anchor - logp
    # strictly below only: a token exactly at the anchor has no gradient
    d = torch.where(gap > 0, gap, torch.zeros_like(gap)).clamp(max=10.0) * mask
    per = 0.5 * d * d if shape == "squared" else d
    return (per.sum(-1) / norm).sum(), d.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--buggy_reward", default="r_creature",
                   help="rollout-log column holding the reward that was wrong")
    p.add_argument("--true_reward", default="r_correct",
                   help="rollout-log column holding the reward that was right")
    p.add_argument("--max_step", type=int, default=None,
                   help="replay only rollouts from steps < this. Set it to the step of the\n"
                        "checkpoint being repaired, so the replay cannot reverse updates\n"
                        "that checkpoint never received.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    # `reverse` is the module docstring's proposal in full: "the negative GRPO loss
    # with incorrect reward's advantages combined with GRPO loss with correct reward's
    # advantages" is -A_buggy + A_corr, and since A_buggy = A_corr + (c - mean c) that
    # collapses to -(c - mean c). The correct advantage is already inside it, so there is no separate
    # arm that adds it back -- a `both` method existed until 2026-09-19 and was exactly
    # that, 2*A_corr - A_buggy, applying the corrected advantage twice. It tied `reverse`
    # at R=1, was worse beyond it and needed 1.35x the steps.
    #   reverse   A_corr - A_buggy    the repair
    #   correct   A_corr              the control: keep training on the fixed reward
    # Pure undo, -A_buggy, is a_reverse - a_correct and is not offered either: it removes
    # the run's capability gain along with the hack, which is what rewinding does.
    p.add_argument("--method", choices=["reverse", "correct", "bc"],
                   default="reverse")
    p.add_argument("--bc_data", default="", help="teacher completions from bc_teacher.py")
    # two orthogonal axes: WHICH prompts to repair on, and WHICH teacher completions to
    # clone. Crossing them separates "narrow the footprint" from "don't clone mistakes".
    p.add_argument("--bc_prompts", choices=["all", "flagged"], default="all",
                   help="all = every hacked-env prompt (the Opus 4 recipe); "
                        "flagged = only those where observed reward != true reward")
    p.add_argument("--bc_completions", choices=["all", "correct"], default="all",
                   help="correct = drop teacher completions the verifier scores < 1.0")
    p.add_argument("--seqs_per_step", type=int, default=64,
                   help="bc only; 64 matches reverse's 8 groups x 8 completions")
    # WHICH groups to replay, independently of which advantage is applied. `correct`
    # keyed on its own signal is a fair "plain offline RL on your logs" baseline, but it
    # is not matched to `reverse`: the buggy reward is zero on rows the bug never touched,
    # so the corrected-reward arm replays those too, including ones reverse never sees.
    # sees. That hands the control training data the treatment is denied, on exactly the
    # prompts the generalisation claim is about. --groups reverse restricts any method to
    # the reverse arm's own groups, so the two differ only in the advantage applied.
    p.add_argument("--groups", choices=["native", "reverse"], default="native",
                   help="native = each method's own signal-carrying groups; "
                        "reverse = the groups the reverse arm would use, whatever the "
                        "method. Use `reverse` for a like-for-like control.")
    # Defaults to what the run's manifest says it did: a run that masked capped
    # completions out of its loss put no gradient on them to reverse, and one that did
    # not (mh15/cre15 onwards) did. On regardless, to reproduce an arm from before
    # 2026-09-20, all of which replayed them.
    p.add_argument("--replay_truncated", action="store_true", default=None,
                   help="also replay completions that hit the --norm token cap. Defaults "
                        "to on exactly when the run trained without "
                        "mask_truncated_completions")
    p.add_argument("--bonus", type=float,
                   help="dose: strength to replay the buggy bonus at. Defaults to the "
                        "bonus the run was trained with, a full-strength reversal")
    p.add_argument("--paid_bonus", type=float,
                   help="override the bonus the rollouts were TRAINED with. Read from "
                        "the log by default, which is what you want")
    # 1e-6 with the fp32 master optimiser below, which is the configuration every arm on
    # the plots runs and the only one whose capability stays inside the floor at R = 1
    # (dA -0.024). It is a pair, not two independent choices: at 1e-6 with a rounded
    # optimiser the repair does not slow down, it switches off -- rep_qwen_s0_revslow ran
    # 64 steps that way, moved the trained-task creature rate from 0.810 to 0.817 and
    # left 98.6% of sampled weights bit-identical, because a bf16 tensor's neighbouring
    # values are |w|/128 apart and the update never crosses the gap. 8e-6, the
    # default until 2026-09-20, clears the gap by being large rather than by accumulating,
    # and 2e-6 is rejected outright: never better, and on Qwen seed 0 its cost runs from
    # -0.069 to -0.277 between doses 24 and 32.
    #
    # `training` takes no number: every completion is stepped at its own training
    # step's k_s (`step_scales`), the learning rate Adam's momentum carried its gradient
    # with, so one pass at --alpha 1 moves the weights as far as training's own updates
    # did. With --second_moment frozen that is the exact undo's step.
    p.add_argument("--lr", default=1e-6,
                   type=lambda x: x if x == "training" else float(x),
                   help="a number, or `training`: each completion at its training step's "
                        "own step size (needs the anchor's trainer_state.json)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="multiplies every update's step. With --lr training, 1 is "
                        "training's own size and more than 1 goes further")
    # A fresh Adam's first updates move every parameter by about lr, the largest steps of
    # the arm, before its second moments know each parameter's gradient scale. A linear
    # warmup, as in training (constant_with_warmup, 8 steps), keeps that burst out of the
    # repair. Step 0 runs at lr 0, as transformers' schedule does, so dose d has moved
    # the weights by about d - (w + 1) / 2 full-lr steps once d > w. 0 keeps the
    # constant schedule every arm before 2026-11-05 used.
    p.add_argument("--warmup_steps", type=int, default=0)
    # Evenly spaced snapshots put most of their points in the flat tail: on one arm
    # steps 3/6/9/10 covered R 0.58-1.44 while 20/30/40 covered 1.90-2.01. Geometric
    # spacing costs the same number of evals and spreads them over the whole curve.
    # A geometric schedule spends its first three checkpoints below R=0.3, where the
    # curve carries almost no information and each one still costs an eval battery
    # slot. Name the steps directly once the useful range is known.
    p.add_argument("--save_at_steps", default="",
                   help="comma-separated replay steps to snapshot, in place of or "
                        "alongside --save_every and --save_geom")
    p.add_argument("--save_geom", type=int, default=0, metavar="BASE",
                   help="also snapshot at steps 1, BASE, BASE^2, ... (2 is a good BASE). "
                        "Combines with --save_every; both may be given")
    # The replay is plain REINFORCE on advantages computed once from the recorded
    # completions, with no ratio and no trust region, so after step 1 the rollouts are
    # off-policy and nothing corrects for it or says how far it has gone. The rollouts
    # carry no logprobs, so the behaviour policy is not recoverable; what is recoverable
    # is drift from the checkpoint being repaired, which is the part the repair causes
    # and the part that differs between methods. Every arm starts from the same place,
    # so the residual mismatch is common to all of them.
    p.add_argument("--iw", choices=["off", "ratio", "clip"], default="off",
                   help="off = plain REINFORCE. ratio = per-token importance weight, "
                        "detached. clip = PPO surrogate on the per-token ratio, which "
                        "equals `off` at step 0 and only diverges as the policy moves")
    p.add_argument("--iw_clip", type=float, default=0.2)
    # A limit on the push-down that needs only the anchor: a pushed-down token's push is
    # weighted by min(1, p_now / p_anchor), its importance weight against the anchor,
    # applied only where it shrinks the push. A logged token stands in for a sample from
    # the anchor; once the model makes it less likely it is a less representative sample
    # and counts for less, so a token's fall slows as it falls (about logarithmically in
    # the dose) instead of running on, as an unweighted push-down does. Pushed-up tokens
    # are not weighted: a push-up fades by itself as the probability nears 1. Uses the
    # reference pass (so not with --no_iw_track); the same shape as NPO's weight, per token.
    p.add_argument("--down_weight", choices=["none", "anchor"], default="none",
                   help="anchor: weight each pushed-down token by min(1, p_now / p_anchor)")
    # Drift is read off the training batches, which are forwarded anyway. One reading
    # is measured on that step's own 16 groups, giving a signal-to-noise of about 4;
    # averaging over --iw_window readings spans more than one pass through the group
    # pool and takes it to ~7, which is far finer than a stopping rule needs against a
    # dose that varies 8x between seeds. A separate fixed monitor set would cost a
    # forward pass per logged step to buy precision nothing here uses.
    p.add_argument("--iw_window", type=int, default=3,
                   help="logged readings to average the drift over")
    # Which policy sits in the denominator, and it changes what the weight means.
    # `behaviour` is the unbiased correction for treating the recorded rollouts as
    # off-policy data for a corrected objective; it needs the per-token logprobs the
    # sampling policy assigned, which common.grpo now records and no existing run has.
    # `anchor` is the checkpoint being repaired: not that correction, but a trust region
    # on the drift the repair itself causes, which is the term that grows without bound
    # over 40 steps and the only one that differs between methods.
    p.add_argument("--iw_ref", choices=["anchor", "behaviour"], default="anchor")
    # The run's log-prob files. They are where the token ids training sampled come from
    # when the rollout log does not record them (common/rollout_tokens.py says why the
    # logged text will not do), and --iw_ref behaviour reads its denominator from them.
    p.add_argument("--logprobs", "--iw_logprobs", dest="logprobs", default="",
                   help="directory of the run's log-prob shards (lp_*.parquet)")
    p.add_argument("--allow_text", action="store_true",
                   help="replay the rollout log's text re-encoded when it records no "
                        "token ids and no --logprobs is given; wrong for a model that "
                        "thinks (Gemma 4 multihop)")
    p.add_argument("--no_iw_track", dest="iw_track", action="store_false",
                   help="skip the reference forward pass. It costs about a third more "
                        "compute and is what makes both --iw and the drift log possible")
    # Floored reverse (common/floor.py): a pushed-down token leaves the loss once its
    # log-prob is at or below min(untrained, anchor), so the replay stops at the untrained
    # rate instead of running past it. The log-probs are stored once per run.
    p.add_argument("--floor", action="store_true",
                   help="stop pushing a token down once its log-prob is at or below "
                        "min(untrained, anchor) (common/floor.py)")
    p.add_argument("--cap", action="store_true",
                   help="stop pushing a token up once its log-prob is at or above "
                        "max(untrained, anchor) (common/floor.py): the floor's rule for "
                        "the pushed-up completions")
    p.add_argument("--fade", action="store_true",
                   help="the band [min, max](untrained, anchor) with a soft edge: a push "
                        "that takes a token further outside its band is weighted by "
                        "1 - distance outside / band width (common/floor.py). Not with "
                        "--floor or --cap, which are the same band with a hard edge")
    p.add_argument("--fade_rule", choices=["width", "ratio"], default="width",
                   help="--fade: width = the linear edge above; ratio = weight a "
                        "pushed-down token by min(1, p_now / min(p_anchor, p_untrained)) "
                        "and leave pushed-up tokens alone (common/floor.py)")
    p.add_argument("--fade_widths", type=float, default=1.0,
                   help="--fade: band widths outside at which the push reaches 0")
    # A limit per completion in place of one per token, and referenced to the anchor
    # alone. A pushed-down completion is left out of an update while its mean log-prob is
    # more than this many nats per token below the anchor's; it comes back in if it rises
    # again. Single tokens may fall as far as they like, which the hack needs
    # (docs/LOG.md 2026-11-25: collapse is whole completions falling); 0 pushes a
    # completion only while it is above where the anchor had it. A completion is looked
    # at when it is replayed, once a pass, so it can fall further in between.
    p.add_argument("--skip_below", type=float, default=None, metavar="NATS",
                   help="skip a pushed-down completion while its mean log-prob is more "
                        "than this many nats/token below the anchor's")
    p.add_argument("--stop_marks", default="",
                   help="comma-separated margins in nats per token, e.g. 0.3,0.5,1. The "
                        "stop signal is how far the pushed-down completions' log-prob "
                        "per token has fallen below the anchor's, averaged over the "
                        "last fifth of an epoch of updates. The first time it passes a "
                        "margin the weights are saved as <out>-stop<100 x margin>; the "
                        "run goes on, so each can be evaluated later")
    # `anchor`: Adam's second moment starts from the anchor's (adam_moments.pt) and goes
    # on updating, so each weight's step is scaled by the gradient size training saw on
    # it; the first moment starts at zero. Use without warmup. By update 256 about three
    # quarters of it is the replay's own gradients (docs/OFFLINE_DESIGN.md section 1).
    # `frozen`: the anchor's second moment, held fixed, and no first moment: an update is
    # then linear in its gradient, so batch size drops out to first order and the two
    # sides of a group add. Each update's gradient is capped at training's max_grad_norm
    # and each weight's divisor is at least what that gradient alone would have made it
    # (`ReplayAdamW.step_frozen`).
    p.add_argument("--second_moment", choices=["fresh", "anchor", "frozen"], default="fresh")
    p.add_argument("--beta1", type=float, default=None,
                   help="the first moment's decay. Default: training's with a fresh or "
                        "carried second moment, 0 (none) with a frozen one")
    # What a completion's loss term is divided by. `batch`: the completions replayed in
    # the update, so every update is a mean. `step`: N, the completions of its own
    # training step, as training's loss did; the gradient then has the size training's
    # had, which is what a carried or frozen second moment is scaled for.
    p.add_argument("--loss_scale", choices=["auto", "batch", "step"], default="auto",
                   help="auto = batch with a fresh second moment, step otherwise")
    # Which updates of training the replay is sized against, and in what order.
    p.add_argument("--order", choices=["shuffle", "backward"], default="shuffle",
                   help="shuffle = groups in random order across training steps. "
                        "backward = one update per training step, newest first, each "
                        "with that step's own groups")
    p.add_argument("--undo_steps", type=int, default=0,
                   help="--order backward: only the newest this many training steps; "
                        "0 = all")
    p.add_argument("--updates_per_epoch", "--batches_per_epoch", dest="updates_per_epoch",
                   type=int, default=None,
                   help="--order shuffle: cut each pass over the groups into this many "
                        "updates (0 = the training run's step count), in place of "
                        "--groups_per_step")
    p.add_argument("--epochs", type=float, default=None,
                   help="--order shuffle: the dose in passes over the replayed groups, in "
                        "place of --steps")
    p.add_argument("--save_at_epochs", default="",
                   help="--order shuffle: comma-separated passes to snapshot after "
                        "(fractions allowed), saved as <out>-step<updates done>")
    p.add_argument("--advantage", choices=["bug", "full"], default="bug",
                   help="--method reverse: bug = the bug's share of the advantage. full "
                        "= the whole training advantage, which with the exact undo's "
                        "flags and --undo_steps K should reproduce the checkpoint K "
                        "steps back")
    # Where each update's gradient is taken. `current`: at the weights reached so far,
    # which is closer to where training took it but feeds back -- pushing a completion
    # down lowers its tokens' probabilities, which enlarges the next push (a token's push
    # scales with 1 - p), and the full-update check on cre17_qwen_s2 diverged after ~5
    # steps that way (gradient 111x training's by step 41). `anchor`: every gradient at
    # the anchor's weights, a first-order repair with no feedback.
    p.add_argument("--grad_at", choices=["current", "anchor"], default="current")
    # `sampling` weights each token's loss by (1 - p_sampled) / (1 - p_now), from the
    # recorded log-probs: a token's gradient on the logits has L1 norm 2(1 - p), so this
    # makes every token's push on the logits the size it was when training applied it,
    # wherever the gradient is taken. It removes the feedback above, and at the anchor it
    # restores the push on tokens whose probability training already saturated.
    p.add_argument("--token_scale", choices=["none", "sampling"], default="none",
                   help="sampling needs --logprobs")
    # Where the replay starts. `bf16`: the checkpoint's own weights, upcast, and the
    # change accumulated in fp32 and rounded once on save, the way training's own changes
    # reach a checkpoint. `fp32`: training's fp32 master copy (master_weights.pt), which
    # carries each weight's remainder below bf16 resolution.
    p.add_argument("--start", choices=["bf16", "fp32"], default="bf16")
    # Which class loads the anchor. `checkpoint`: the one its config names, which is the
    # parameter list training's saved optimizer state is laid out for (Gemma 4:
    # Gemma4ForConditionalGeneration). `auto`: AutoModelForCausalLM, as every reverse arm
    # before 2026-11-25 did.
    p.add_argument("--model_class", choices=["auto", "checkpoint"], default=None,
                   help="default: checkpoint with --second_moment frozen or --start fp32, "
                        "auto otherwise")
    p.add_argument("--end_token_stats", action=argparse.BooleanOptionalAction, default=None,
                   help="log the end-of-turn token's log-prob where a completion did not "
                        "end, per side of the push (one more softmax a micro-batch). "
                        "Default: on with --second_moment frozen")
    # Retain term (reverse only): beside the replayed groups, each update holds as many
    # logged completions that the bug did not pay and whose correct reward is at or above
    # their group's mean, drawn uniformly from the whole log up to the anchor. A token of
    # theirs is penalised only while the model makes it less likely than the anchor did:
    # nothing pulls on a model that still writes its good answers as the anchor would,
    # and nothing holds the hack's probability in place, which a KL over the anchor's
    # whole next-token distribution would. Needs only the anchor and the log.
    p.add_argument("--retain_beta", type=float, default=0.0,
                   help="0 disables; weight of the retain penalty, normalised as the "
                        "policy term is (per completion, by --norm)")
    p.add_argument("--retain_shape", choices=["squared", "linear"], default="squared",
                   help="per token, with d = max(0, log p_anchor - log p_now): d*d/2 "
                        "(no force at noise-sized dips, growing with the fall) or d")
    p.add_argument("--retain_per_step", type=int, default=0,
                   help="retained completions an update; 0 = --groups_per_step groups' worth")
    # Whether the method may use the training run's log-probs (common/replay.py): with
    # them each token is weighted by training's own vLLM correction, so the replay's
    # gradient is training's; without, only the sampled tokens and rewards were logged.
    # No default: the two are different methods. Required for reverse and correct.
    replay.add_argument(p)
    p.add_argument("--micro_batch", type=int, default=8,
                   help="sequences a forward pass; 8 fits creatures beside an eval engine on a "
                        "B200 (peak 159-169 GiB of 179); multihop_repair.sh passes 2")
    # Throughput only; none of these changes what an update is, up to bf16 summation order.
    p.add_argument("--optim_device", choices=["cpu", "cuda"], default="cuda",
                   help="where --optim master keeps its fp32 weights and moments")
    p.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false",
                   help="keep activations instead of recomputing them: ~1/3 fewer FLOPs "
                        "per update for activation memory")
    p.add_argument("--no_sort_micro", dest="sort_micro", action="store_false",
                   help="micro-batch a step's completions in sampled order instead of "
                        "by length, which pads each micro-batch to its longest")
    p.add_argument("--groups_per_step", "--groups_per_update", dest="groups_per_step",
                   type=int, default=16,
                   help="--order shuffle: groups an update (see --updates_per_epoch)")
    p.add_argument("--steps", "--updates", dest="steps", type=int, default=None,
                   help="the dose in updates. Default 100, or what --epochs gives; "
                        "--order backward takes --undo_steps instead")
    # A fixed 1280 once cut 41% of pilot14's replayed sequences while still dividing by
    # --norm, which is the length bias dr_grpo exists to remove; the default now covers
    # the longest sequence the run trained on. Activation memory scales with this, so
    # --micro_batch may have to halve for a run with a long budget.
    p.add_argument("--max_len", type=int,
                   help="prompt+completion token cap per replayed sequence. Defaults "
                        "to the run's longest prompt plus its completion budget, which "
                        "cuts nothing. Anything over it is cut while the loss still "
                        "divides by --norm, so a cut sequence is under-weighted; the run "
                        "reports how many were cut")
    # Read from the run's manifest by default (common/manifest.py says why).
    p.add_argument("--norm", type=int,
                   help="dr_grpo constant normalizer. Defaults to the "
                        "completion budget in the run's manifest; a "
                        "different value is refused, since the replay gradient would be "
                        "the wrong size")
    p.add_argument("--clip", type=float, default=0.0, help="0 disables the ratio clip")
    # KL anchor. The pilot12 battery showed the reverse advantage is an UNANCHORED
    # objective -- beta=0, no ratio clip, nothing in the loss saying "still answer the
    # question" -- and 40 unconstrained steps drove mean completion length from 1052
    # tokens to 88 on spell_backward, taking its accuracy with it. A KL penalty against a
    # frozen reference is the standard fix. Two references are meaningful and the choice
    # changes what is being asked:
    #   --kl_ref <hacked ckpt>  "unlearn the creature words, change nothing else"
    #   --kl_ref <base model>   "return to the pre-RL model" -- also discards the RL gain
    # k3 per replayed token (`k3`), from the reference's log-probs precomputed once per run
    # (common/floor.py's files); pushed-down tokens settle about log(1 + |A| / beta) below
    # the reference.
    p.add_argument("--kl_beta", type=float, default=0.0,
                   help="0 disables the KL penalty")
    p.add_argument("--kl_cap", type=float, default=10.0,
                   help="nats below the reference past which k3's force stops growing "
                        "(5 in the second KL batch, which broke Qwen multihop)")
    p.add_argument("--kl_ref", default="",
                   help="frozen reference for the KL term; defaults to --model "
                        "(i.e. stay close to the hacked policy)")
    p.add_argument("--seed", type=int, default=0)
    # Gemma 4's per-layer embedding table has 2.35B elements against bitsandbytes'
    # INT_MAX, so it must be frozen before the optimiser is built or the kernel gets a
    # nonsense grid size. Same flag and same reason as creatures/train.py. torchao and
    # torch have no such limit, so `sr` and `master` do not need it.
    p.add_argument("--freeze", default="",
                   help="comma-separated name substrings to hold fixed; Gemma 4 needs "
                        "embed_tokens_per_layer")
    # The parameters are bf16, so neighbouring representable values are |w|/128 apart
    # and any update below that gap is discarded rather than accumulated. At lr 1e-6
    # that is most of them: 98.6% of sampled weights were bit-identical after 16 steps
    # and the dose curve came out 14x shallower than lr x steps predicts. Both of the
    # first two choices have that floor -- `adamw` is not an fp32 optimiser here,
    # because torch allocates its moments with zeros_like(p) and so gets bf16 ones.
    #   adamw8bit  bitsandbytes PagedAdamW8bit, ~8 GB of state on the GPU
    #   adamw      torch AdamW with bf16 moments, ~16 GB
    #   sr         torchao AdamW8bit, stochastic rounding on the write-back: the bf16
    #              parameter is unbiased, so sub-gap updates land in expectation at the
    #              cost of injected noise. Same memory as adamw8bit.
    #   master     fp32 master weights, grads and moments on the CPU: 64 GB of host
    #              RAM for a 4B model, measured at 62.5 GB by step 0, and no optimiser
    #              state on the GPU at all. The only exactly-unrounded option. Needs
    #              --mem=96G, which is what jobs/repair.sh asks for; slurm's MaxRSS
    #              cannot confirm the fit because it counts page cache and saturates at
    #              whatever was asked for.
    #
    # `master` is the default: it is the only arm whose capability stays inside the floor
    # at R = 1.45, it is flat across its whole dose curve, and it costs 6% wall clock.
    # `sr` is measured and rejected -- unbiased but its variance scales with |w|, so it
    # moves ~1% of RMSNorm scales a thousand times further than the true update.
    p.add_argument("--optim", choices=["adamw8bit", "adamw", "sr", "master"],
                   default="master")
    # Evaluate each snapshot as it is written, on the same GPU, instead of in later jobs:
    # the replay waits with its bf16 weights moved to host memory (the fp32 masters and
    # moments are there already), runs the command, then carries on. {ckpt} is the
    # snapshot directory, {name} its basename, {step} the replay step.
    p.add_argument("--eval_cmd", default="",
                   help="shell command run on every snapshot and the final weights")
    p.add_argument("--eval_delete", action="store_true",
                   help="after an eval that exits 0, delete the snapshot's files except "
                        "repair_state.json (subdirectories, e.g. eval results, are kept); a snapshot "
                        "whose eval failed is kept for a rerun")
    p.add_argument("--save_every", type=int, default=0,
                   help="also save intermediate checkpoints, for a dose-response curve")
    # Checkpoint by datapoints processed, not steps: the BC filter cells have different
    # row counts, so a fixed step count is a different number of epochs in each. Counts
    # training sequences, which for --method bc is rows consumed; the GRPO arms count a
    # completion with non-zero advantage, a different unit, so use --save_every there.
    p.add_argument("--save_at_seqs", default="",
                   help="comma-separated cumulative sequence counts at which to save, "
                        "e.g. 85,184,355 -- saves to <out>-n<count>")
    args = p.parse_args()
    if args.iw_ref == "behaviour" and not args.logprobs:
        raise SystemExit(
            "--iw_ref behaviour needs --logprobs, the per-token logprobs under the "
            "policy that generated each rollout. common.grpo records them from the "
            "trainer, but no run before 2026-09-18 has them, so those rollouts can only "
            "be replayed with --iw_ref anchor.")

    # The anchor is a checkpoint-N directory inside the run's; its manifest says how the
    # gradient being replayed was computed.
    ckpt = Path(args.model)
    run = manifest.read(ckpt.parent)
    assert run["loss_type"] == "dr_grpo", f"replay assumes dr_grpo, not {run['loss_type']}"
    is_bc = args.method == "bc"
    replay.check(args, needed=not is_bc)
    vcap = replay.correction_cap(run)
    if args.norm is None:
        args.norm = run["train_max_tokens"]
    assert args.norm == run["train_max_tokens"], (
        f"--norm {args.norm}, but the run trained at {run['train_max_tokens']} tokens")
    if args.max_len is None:
        args.max_len = run["max_prompt_tokens"] + run["train_max_tokens"]
    if args.replay_truncated is None:
        args.replay_truncated = not run["mask_truncated_completions"]
    print(f"norm {args.norm}, max_len {args.max_len}; the run "
          + ("masked" if run["mask_truncated_completions"] else "did not mask")
          + " capped completions, so they are "
          + ("" if args.replay_truncated else "not ") + "replayed", flush=True)

    # ---- how the knobs combine ---------------------------------------------------------
    lr_training = args.lr == "training"
    frozen = args.second_moment == "frozen"
    backward = args.order == "backward"
    partition = not backward and args.updates_per_epoch is not None
    step_scale = not is_bc and (args.loss_scale == "step" or (
        args.loss_scale == "auto" and args.second_moment != "fresh"))
    at_anchor = args.grad_at == "anchor"
    stop_marks = sorted(float(x) for x in args.stop_marks.split(",") if x.strip())
    skip = args.skip_below is not None
    # trainer_state.json gives the anchor's step count T and each step's learning rate
    need_log = (lr_training or frozen or backward or args.advantage == "full"
                or (partition and not args.updates_per_epoch))
    token_limits = args.floor or args.cap or args.fade
    if is_bc and (need_log or token_limits or skip or stop_marks or at_anchor
                  or args.second_moment != "fresh" or args.token_scale != "none"):
        raise SystemExit("--method bc clones teacher completions: the replay's order, "
                         "step size, second moment and limits do not apply to it")
    if args.method != "reverse" and (token_limits or skip or stop_marks
                                     or args.advantage == "full"):
        raise SystemExit("--floor, --cap, --fade, --skip_below, --stop_marks and "
                         "--advantage full apply to --method reverse")
    if args.fade and (args.floor or args.cap):
        raise SystemExit("--fade replaces --floor and --cap")
    if lr_training and args.warmup_steps:
        raise SystemExit("--lr training: each k_s already carries training's warmup; "
                         "drop --warmup_steps")
    if backward and (args.steps is not None or args.epochs is not None or partition
                     or args.save_at_epochs):
        raise SystemExit("--order backward is one update per training step; its dose is "
                         "--undo_steps and its snapshots --save_at_steps")
    if args.undo_steps and not backward:
        raise SystemExit("--undo_steps applies to --order backward")
    if args.steps is not None and args.epochs is not None:
        raise SystemExit("give the dose as --steps or as --epochs, not both")
    if args.optim != "master" and (args.second_moment != "fresh" or at_anchor
                                   or args.start == "fp32"):
        raise SystemExit("--second_moment anchor and frozen, --grad_at anchor and --start "
                         "fp32 need --optim master")
    if args.kl_beta and step_scale:
        raise SystemExit("--kl_beta is normalised per update and has no weight under "
                         "--loss_scale step (the default with a carried or frozen second "
                         "moment); say --loss_scale batch")
    if (skip or stop_marks) and args.iw_ref != "anchor":
        raise SystemExit("--skip_below and --stop_marks measure against the anchor "
                         "(--iw_ref anchor)")
    if args.token_scale == "sampling" and not args.logprobs:
        raise SystemExit("--token_scale sampling needs --logprobs")

    # ---- training's optimizer: its settings, and for some knobs its state and schedule --
    optc = training_optimizer()
    (b1, b2), eps, max_norm = optc["betas"], optc["eps"], optc["max_grad_norm"]
    T = lr_hist = gnorm = k = moments = None
    window = False
    if need_log:
        if lr_training or frozen:
            if run.get("scale_rewards", "none") != "none":
                raise SystemExit("--lr training and --second_moment frozen assume "
                                 "scale_rewards='none' (centred advantages)")
            if run.get("beta", 0.0):
                raise SystemExit("--lr training and --second_moment frozen assume beta=0: "
                                 "no KL term in the training loss")
        T, lr_hist, gnorm = training_log(ckpt)
    if frozen:
        moments = torch.load(ckpt / "adam_moments.pt", map_location="cpu", mmap=True,
                             weights_only=True)
        pg = moments["param_groups"]
        b1, b2 = pg[0]["betas"]
        eps = pg[0]["eps"]
        for g in pg:
            assert tuple(g["betas"]) == (b1, b2) and g["eps"] == eps
            if g["weight_decay"]:
                raise SystemExit(f"weight decay {g['weight_decay']} in training; a frozen "
                                 "second moment would need each later step's "
                                 "(1 - lr*wd) factor")
        steps_seen = {int(s["step"]) for s in moments["state"].values()}
        assert steps_seen == {T}, f"Adam state at steps {steps_seen}, trainer_state at {T}"
    if need_log:
        if args.max_step not in (None, T):
            raise SystemExit(f"--max_step {args.max_step}, but the anchor is at training "
                             f"step {T}")
        args.max_step = T
        # --advantage full with --undo_steps K is the check against checkpoint T-K: undo
        # every gradient's contribution during the last K steps, older gradients included.
        window = args.advantage == "full" and 0 < args.undo_steps < T
        k = step_scales(lr_hist, b1, first=T - args.undo_steps + 1 if window else 1)
        print(f"anchor {ckpt}: T={T}, betas {b1}/{b2}, eps {eps}; gradients capped at "
              f"norm {max_norm}", flush=True)
        if lr_training:
            print("  step  lr        k_t", flush=True)
            for t in range(1, T + 1):
                if t <= 10 or t > T - 3 or t % 10 == 0:
                    print(f"  {t:4d}  {lr_hist[t-1]:.2e}  {k[t-1]:.3e}", flush=True)

    # ---- groups and advantages -----------------------------------------------------------
    if is_bc:
        assert args.bc_data, "--method bc needs --bc_data"
        bc = [json.loads(l) for l in open(args.bc_data)]
        n0 = len(bc)
        if args.bc_prompts == "flagged":
            bc = [r for r in bc if r["flagged"]]
        n1 = len(bc)
        if args.bc_completions == "correct":
            bc = [r for r in bc if r["correct"] >= 1.0]
        print(f"bc prompts={args.bc_prompts} completions={args.bc_completions}: "
              f"{n0} -> {n1} (prompt filter) -> {len(bc)} (completion filter)", flush=True)
        assert bc, "no teacher completions survived the filters"
        groups = []
    else:
        groups = load_groups(args.rollouts, args.bonus, args.paid_bonus, args.max_step,
                             args.buggy_reward, args.true_reward)
    # Each group's training step and its completion count N: training divided that
    # step's loss by N, so a completion's weight there was its advantage / N.
    n_step = defaultdict(int)
    for g in groups:
        g["step"] = int(g["step"])
        n_step[g["step"]] += len(g["rs"])
    for g in groups:
        g["N"] = n_step[g["step"]]
        if need_log:
            # a_reverse is -(c - mean c) = -A^bug and a_correct is A^true, so the
            # advantage training used, A^true + A^bug, is a_correct - a_reverse
            for rec, ar, ac in zip(g["rs"], g["a_reverse"], g["a_correct"]):
                rec["_adv"] = ac - ar
        if args.advantage == "full":  # the whole update's negation
            g["a_reverse"] = [a - b for a, b in zip(g["a_reverse"], g["a_correct"])]
    if need_log:
        print(f"{len(groups)} groups over {len(n_step)} steps; completions per step "
              f"{sorted(set(n_step.values()))} (the N each step's loss divided by)",
              flush=True)
        assert sorted(n_step) == list(range(T)), "rollout steps must be 0..T-1"
    all_groups = groups  # before the replay set is cut down: the retain pool reads these
    if args.down_weight != "none" and (args.method != "reverse" or not args.iw_track
                                       or args.iw != "off"):
        raise SystemExit("--down_weight applies to --method reverse with --iw off and the "
                         "reference pass")
    if args.retain_beta and args.method != "reverse":
        raise SystemExit("--retain_beta applies to --method reverse")
    nz = sum(1 for g in groups if any(a != 0 for a in g["a_reverse"]))
    if not is_bc:
        print(f"{len(groups)} groups, {nz} with a non-zero reverse advantage "
              f"({nz / max(len(groups), 1):.1%})", flush=True)
    if (args.method == "reverse" or args.groups == "reverse") and args.advantage != "full":
        # Only groups where the creature bonus actually varied carry repair signal --
        # and, under --groups reverse, they define the replay set for every method.
        # (--advantage full keeps every group: the overlong penalty below can still give
        # one a non-zero advantage.)
        groups = [g for g in groups if any(a != 0 for a in g["a_reverse"])]
        _adv = {"reverse": lambda g: g["a_reverse"],
                "correct": lambda g: g["a_correct"]}
        n_live = sum(1 for g in groups for a in _adv[args.method](g) if a != 0.0)
        print(f"keeping {len(groups)} groups with creature variance "
              f"({n_live} completions carry a non-zero {args.method} advantage)",
              flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    # A run that set `mask_truncated_completions` gave a completion that hit the budget
    # no gradient at all -- while its reward still entered the group mean. That is why the
    # advantages above are computed over every row and only the set trained on is
    # filtered here: replaying a capped completion reverses a gradient that was never
    # applied. Measured on final_qwen_s0, 285 of the 1576 rows the reverse arm replayed
    # were at the 1536-token cap (18.1%), 19.4% of the pushed-down rows against 15.8% of
    # the pushed-up ones, so it does not cancel. The tokens replayed, and the cap check,
    # are the ones training sampled (common/rollout_tokens.py).
    if not is_bc:
        attach_sampled_tokens(all_groups if args.retain_beta or need_log else groups, tok,
                              args.norm, not args.replay_truncated, args.logprobs,
                              args.allow_text)
        # Multihop runs also paid DAPO's soft overlong penalty (sycophancy/train.py), a
        # third reward term that depends only on length: 0 up to L - cache tokens, then
        # linearly down to -1 at the cap L. It is not the bug's, so the bug's advantage
        # ignores it, but the whole update (--advantage full) contains it.
        cache = run.get("overlong_cache", 0) if need_log else 0
        if cache:
            # training's own reward function (sycophancy/train.py), on each completion's
            # length; a capped one whose tokens were not recorded is at the cap
            from trl.rewards import get_soft_overlong_punishment
            overlong = get_soft_overlong_punishment(args.norm, cache)
            for g in all_groups:
                ol = overlong(completion_ids=[
                    [0] * (args.norm if r["capped"] and not r["_ids"] else len(r["_ids"]))
                    for r in g["rs"]])
                mo = sum(ol) / len(ol)
                for rec, o in zip(g["rs"], ol):
                    rec["_adv"] += o - mo
                if args.advantage == "full":
                    g["a_reverse"] = [a - (o - mo) for a, o in zip(g["a_reverse"], ol)]
            print(f"overlong penalty (cache {cache} of {args.norm} tokens) is part of the "
                  f"training advantage", flush=True)
        if args.logprobs and need_log:
            # The check that the advantage is reconstructed right: training logged its own.
            logged = [r for g in all_groups for r in g["rs"] if "_logged_adv" in r]
            agree = sum(abs(r["_logged_adv"] - r["_adv"]) <= 1e-3 for r in logged)
            print(f"reconstructed training advantage matches the logged one for "
                  f"{agree}/{len(logged)} completions", flush=True)
        n_groups0 = len(groups)
        for g in groups:
            g["_live"] = live_records_of(g, args.method)
            for rec in g["rs"]:  # its training step (0-based), that step's N and k_s
                rec["_step"], rec["_N"] = g["step"], g["N"]
                if k:
                    rec["_kt"] = k[g["step"]]
        groups = [g for g in groups if g["_live"]]
        if n_groups0 != len(groups):
            print(f"{n_groups0 - len(groups)} groups left with nothing to replay once "
                  f"capped completions are dropped", flush=True)
        if not groups:
            raise SystemExit("every replayable completion hit the token cap")
        print(f"{len(groups)} groups to replay, "
              f"{sum(len(g['_live']) for g in groups)} completions", flush=True)

    # ---- the updates: which completions each one replays --------------------------------
    if is_bc:
        lives, steps_of, per_update = [[(None, r, 1.0)] for r in bc], [0] * len(bc), args.seqs_per_step
    else:
        lives, steps_of, per_update = ([g["_live"] for g in groups],
                                       [g["step"] for g in groups], args.groups_per_step)
    undo_n = args.undo_steps
    if window:
        # every step's gradient contributes; steps whose weight is below 1e-4 of the
        # largest (momentum from 90+ steps before the window) are skipped
        undo_n = max(T - s + 1 for s in range(1, T + 1) if k[s - 1] >= 1e-4 * max(k))
        print(f"check against checkpoint {T - args.undo_steps}: undoing the contributions "
              f"made during steps {T - args.undo_steps + 1}..{T}, from the gradients of "
              f"steps {T - undo_n + 1}..{T}", flush=True)
    if not backward and args.steps is None and args.epochs is None:
        if partition:
            args.epochs = 1.0
        else:
            args.steps = 100
    schedule, nb = build_schedule(lives, steps_of, args.order, args.seed, args.steps,
                                  args.epochs, per_update,
                                  args.updates_per_epoch if partition else None, T,
                                  k if lr_training else None, undo_n)
    K = args.steps = len(schedule)
    save_at = sorted({int(x) for x in args.save_at_seqs.split(",") if x.strip()})
    save_steps = set(range(args.save_every, K, args.save_every)) if args.save_every else set()
    if args.save_geom > 1:
        g_ = 1
        while g_ < K:
            save_steps.add(g_)
            g_ *= args.save_geom
    save_steps |= {int(x) for x in args.save_at_steps.split(",") if x.strip()}
    save_steps |= {round(float(x) * nb) for x in args.save_at_epochs.split(",") if x.strip()}
    save_steps = {s_ for s_ in save_steps if 0 < s_ < K}  # the final weights go to --out
    print(f"{K} updates, order {args.order}"
          + ("" if backward else f", {nb} an epoch ({K / max(nb, 1):.2f} epochs)")
          + f", loss scale {'step' if step_scale else 'batch'}, lr "
          + (f"training x {args.alpha:g}" if lr_training else f"{args.lr:g} x {args.alpha:g}")
          + f", second moment {args.second_moment}; snapshots after updates "
          f"{sorted(save_steps)} and the final weights", flush=True)

    # The token completions end with (the most common last one), for a diagnostic: its
    # log-prob at the positions where the completion did not end, per side of the push.
    end_id = None
    if not is_bc and (frozen if args.end_token_stats is None else args.end_token_stats):
        ends = collections.Counter(r["_ids"][-1] for live in lives for _, r, _ in live
                                   if r["_ids"])
        end_id = ends.most_common(1)[0][0]
        print(f"end token {end_id} ({tok.decode([end_id])!r}), last in "
              f"{ends[end_id]} of {sum(ends.values())} replayed completions", flush=True)

    if token_limits:
        # pushed-down completions need the floor, pushed-up ones the cap; the fade both
        need = [(r, a < 0) for live in lives for _, r, a in live
                if (a < 0 and args.floor) or (a > 0 and args.cap) or (a != 0 and args.fade)]
        pids = {id(r): prompt_ids(r, tok) for r, _ in need}
        lo, hi = floor_mod.bounds(ckpt.parent, run["model"], args.model,
                                  [(pids[id(r)], r["_ids"]) for r, _ in need],
                                  args.max_len, args.micro_batch)
        for r, is_down in need:
            key = floor_mod.key(pids[id(r)], r["_ids"])
            if args.fade:
                r["_lo"], r["_hi"] = lo[key], hi[key]
            else:
                r["_floor" if is_down else "_cap"] = (lo if is_down else hi)[key]
        del lo, hi, pids
        print(f"limits: {len(need)} completions carry min/max(untrained, anchor) "
              f"log-probs of their tokens", flush=True)

    # Reference log-probs of every completion the replay reads -- the replayed ones and,
    # with --retain_beta, the retained ones -- from the run's per-checkpoint files
    # (common/floor.py, keyed by token ids): computed once per run, before the model being
    # repaired is loaded, and shared by every arm that needs them (--down_weight's ratio,
    # the drift log, --skip_below, --stop_marks, --kl_beta, the retain term). Until
    # 2026-11-25 the anchor's were a forward pass of the model being repaired inside
    # every arm, and --kl_beta ran a second, frozen model on every micro-batch.
    pool, retain_picks, retain_anchor = [], [], {}
    if args.retain_beta:
        pool = retain_pool(all_groups, args.buggy_reward)
        n_ret = args.retain_per_step or args.groups_per_step * len(groups[0]["rs"])
        by_fifth = defaultdict(int)
        for g, r in pool:
            by_fifth[int(g["step"]) * 5 // max(args.max_step or 1, 1)] += 1
        in_replay = {id(g) for g in groups}
        print(f"retain: {len(pool)} completions the bug did not pay with a correct reward "
              f"at or above their group's mean, of "
              f"{sum(len(g['rs']) for g in all_groups)} logged; "
              f"{sum(id(g) in in_replay for g, _ in pool)} are siblings in replayed groups; "
              f"by fifth of training {[by_fifth[i] for i in range(5)]}; "
              f"{n_ret} an update, weight {args.retain_beta}, {args.retain_shape}", flush=True)
        if len(pool) < n_ret:
            raise SystemExit("the retain pool is smaller than one update's draw")
        rng_ret = random.Random(args.seed + 1)  # its own stream: the replay order is untouched
        retain_picks = [rng_ret.sample(range(len(pool)), n_ret) for _ in range(K)]
    retained = [pool[i][1] for i in sorted({i for ps in retain_picks for i in ps})]
    replayed = [] if is_bc else [rec for live in lives for _, rec, _ in live]
    track = not is_bc and args.iw_track
    need_anchor = not is_bc and (
        (track and args.iw_ref == "anchor") or args.retain_beta or skip or bool(stop_marks)
        or (args.kl_beta and args.kl_ref in ("", args.model)))
    ref_lp, kl_lp = {}, {}  # token-id key -> fp16 log-prob of each completion token
    if not is_bc and (need_anchor or args.kl_beta):
        seqs = []
        for rec in replayed + retained:
            pids = prompt_ids(rec, tok)
            rec["_key"] = floor_mod.key(pids, rec["_ids"])
            seqs.append((pids, rec["_ids"]))
        run_dir = ckpt.parent
        if need_anchor:
            ref_lp = floor_mod.load_or_compute(run_dir, args.model, seqs, args.max_len,
                                               args.micro_batch,
                                               path=floor_mod.path_of(run_dir, args.model),
                                               label="anchor")
        kl_lp = ref_lp
        if args.kl_beta and args.kl_ref not in ("", args.model):
            # the untrained model's file is the floor's; any other reference gets its own
            untrained = args.kl_ref == run["model"]
            kl_lp = floor_mod.load_or_compute(
                run_dir, args.kl_ref, seqs, args.max_len, args.micro_batch,
                path=floor_mod.path_of(run_dir, None if untrained else args.kl_ref),
                label="untrained" if untrained else "KL reference")
        for r in retained:
            retain_anchor[id(r)] = ref_lp[r["_key"]]
    if args.kl_beta:
        print(f"KL to {args.kl_ref or args.model}: beta {args.kl_beta}, k3 on every replayed "
              f"token (force capped {args.kl_cap} nats below), from precomputed log-probs",
              flush=True)

    def key_of(rec):
        if "_key" not in rec:
            rec["_key"] = floor_mod.key(prompt_ids(rec, tok), rec["_ids"])
        return rec["_key"]

    # Each replayed completion's reference log-probs (`_ref`): the anchor's, or with
    # --iw_ref behaviour the sampling weights' own, recorded by training. The per-token
    # ratio exp(logp - ref) is then available in every update.
    has_ref = False
    if (track and args.iw_ref == "behaviour"):
        missing = [rec for rec in replayed if "_old" not in rec]
        if missing:
            raise SystemExit(f"--iw_ref behaviour: {len(missing)} replayed completions "
                             "have no recorded log-probs; pass --logprobs")
        for rec in replayed:
            rec["_ref"] = torch.tensor(rec["_old"], dtype=torch.float16)
        has_ref = True
    elif track or skip or stop_marks:
        for rec in replayed:
            rec["_ref"] = ref_lp[rec["_key"]]
        has_ref = True
    elif args.iw != "off":
        raise SystemExit("--iw needs the reference pass; drop --no_iw_track")
    if args.token_scale == "sampling":
        missing = sum("_old" not in rec for rec in replayed)
        if missing:
            raise SystemExit(f"--token_scale sampling: {missing} completions have no "
                             "recorded log-probs; pass --logprobs")

    # ---- the model, its trainable set, and the optimizer --------------------------------
    if (args.model_class or ("checkpoint" if frozen or args.start == "fp32" else "auto")) \
            == "checkpoint":
        import transformers
        arch = getattr(transformers, AutoConfig.from_pretrained(ckpt).architectures[0])
    else:
        arch = AutoModelForCausalLM
    model = arch.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    replay.disable_dropout(model)  # as training did (common.grpo disable_dropout)
    if args.freeze:
        from common.grpo import freeze_parameters
        freeze_parameters(model, args.freeze)
    params = [q for q in model.parameters() if q.requires_grad]
    # Printed because it is the one thing that silently makes a replay not a replay: the
    # gradient being reversed was applied to the RL run's parameter set, so the arm has to
    # train that same set. On Gemma the run froze embed_tokens_per_layer, and with --optim
    # master nothing fails if the replay does not -- it just trains 2.35B parameters the
    # run never touched. jobs/repair.sh defaults the freeze by model family; this line is
    # how the log proves it took.
    n_train = sum(q.numel() for q in params)
    n_all = sum(q.numel() for q in model.parameters())
    print(f"trainable {n_train / 1e9:.2f}B of {n_all / 1e9:.2f}B params "
          f"({n_train / n_all:.1%}) in {len(params)} tensors, loaded as {arch.__name__}",
          flush=True)
    if args.second_moment == "anchor":
        moments = torch.load(ckpt / "adam_moments.pt", map_location="cpu", weights_only=True)
    cfg = dict(optc, eps=eps,
               betas=(b1 if frozen or args.beta1 is None else args.beta1, b2))
    opt = build_optimizer(args.optim, model, 0.0 if lr_training else args.lr, cfg, moments,
                          args.optim_device, frozen, (args.beta1 or 0.0) if frozen else 0.0)
    del moments
    print(f"optimiser: {args.optim} -> {type(opt).__module__}.{type(opt).__name__}",
          flush=True)
    pairs = list(opt._pairs()) if args.optim == "master" else []  # (parameter, fp32 master)
    if args.start == "fp32":
        saved = torch.load(ckpt / "master_weights.pt", map_location="cpu", weights_only=True)
        assert len(saved) == len(pairs), (
            f"{len(saved)} fp32 weights saved, {len(pairs)} trainable tensors here: "
            "the trainable set differs from training's (check --freeze)")
        # The fp32 weights rounded to bf16 are exactly the checkpoint's bf16 weights, so
        # this proves the saved state lines up with these tensors one for one.
        for i, ((prm, m), mw) in enumerate(zip(pairs, saved)):
            if mw.shape != prm.shape or not torch.equal(mw.to(torch.bfloat16),
                                                        prm.detach().cpu()):
                raise SystemExit(f"saved fp32 tensor {i} does not match trainable tensor "
                                 f"{i}: the optimizer order is not the one training used")
            m.data.copy_(mw)
        del saved
    # the anchor's bf16 weights: what --grad_at anchor takes every gradient at, and what
    # a frozen second moment's distance report is measured from
    anchor_bf16 = ([prm.detach().to("cpu", copy=True) for prm, _ in pairs]
                   if frozen or at_anchor else None)
    print(f"starting from the {args.start} weights; gradients at the {args.grad_at} weights",
          flush=True)

    def write_weights():
        """Put the repaired weights into the model (for a save); see restore_anchor."""
        if at_anchor:
            with torch.no_grad():
                for prm, m in pairs:
                    prm.copy_(m)

    def restore_anchor():
        if at_anchor:
            with torch.no_grad():
                for (prm, _), a in zip(pairs, anchor_bf16):
                    prm.copy_(a)

    def distance_from_anchor():
        """rms of the fp32 change from the anchor, and the share of bf16 weights it flips.

        The second is what the saved snapshot actually carries: a change smaller than
        half a bf16 rounding step is kept in fp32 here but rounds away on save.
        """
        ref = (torch.load(ckpt / "master_weights.pt", map_location="cpu", mmap=True,
                          weights_only=True) if args.start == "fp32" else anchor_bf16)
        sq = flipped = n = 0
        for i in opt.frozen["sqrt_vhat"]:
            mi = pairs[i][1].detach().cpu()
            d = mi - ref[i].float()
            sq += float(d.double().pow(2).sum())
            flipped += int((mi.to(torch.bfloat16) != anchor_bf16[i]).sum())
            n += d.numel()
            del d, mi
        return math.sqrt(sq / n), flipped / n

    # dr_grpo divides by a constant --norm, so a completion cut short by --max_len is
    # under-weighted by exactly the fraction lost -- the length bias dr_grpo exists to
    # remove, reintroduced at replay time. Counted and reported rather than silently
    # accepted, because raising --max_len costs activation memory.
    truncated = [0, 0]

    def encode(rec):
        pids = prompt_ids(rec, tok)
        # A recorded rollout is replayed on the tokens training sampled. A --method bc
        # teacher completion is a target text, not a rollout, so it is encoded as text.
        cids = rec["_ids"] if "_ids" in rec else \
            tok(rec["completion"], add_special_tokens=False).input_ids
        truncated[1] += 1
        if len(pids) + len(cids) > args.max_len:
            truncated[0] += 1
        ids = (pids + cids)[:args.max_len]
        return ids, len(pids)

    if args.max_len <= args.norm:
        print(f"WARNING --max_len {args.max_len} leaves no room for a prompt in front "
              f"of a --norm {args.norm} completion: every long rollout will be cut and "
              f"under-weighted", flush=True)

    def pack(recs):
        """(ids, attn, completion mask) on cuda for a micro-batch of records."""
        encs = [encode(r) for r in recs]
        L = max(len(e[0]) for e in encs)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(recs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(recs), L), dtype=torch.long)
        cmask = torch.zeros((len(recs), L), dtype=torch.bool)
        for j, (e, plen) in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e)
            attn[j, :len(e)] = 1
            cmask[j, plen:len(e)] = True
        return ids.cuda(), attn.cuda(), cmask.cuda()

    os.makedirs(args.out, exist_ok=True)

    # Every replayed completion's token log-probs, as the replay sees them before each
    # update, go to <out>-logprobs/step<NNNNN>.pt (the update, from 0), so anything per
    # token -- the share held at the floor, drift from the anchor -- can be computed
    # afterwards. Keyed by the token ids (common/floor.py), the key
    # runs/<run>/untrained_logprobs.pt uses; anchor.pt there holds the same completions
    # under the reference. `train_step` is the completion's training step, from 0, as the
    # rollout log numbers it. (common/undo.py wrote update<NNNNN>.pt from 1, or
    # step<training step>.pt, with train_step from 1.)
    lp_dir = Path(f"{args.out}-logprobs")
    lp_dir.mkdir(parents=True, exist_ok=True)
    window_d = deque(maxlen=max(args.iw_window, 1))
    if has_ref:
        torch.save({key_of(rec): rec["_ref"] for upd in schedule for _, rec, _ in upd["live"]},
                   lp_dir / "anchor.pt")
        print(f"reference logprobs for {len(replayed)} sequences over {len(lives)} groups",
              flush=True)

    hist = []  # one record per update; saved with every checkpoint (repair_state.json)
    bug_v = 0.0  # the replayed gradients' share of Adam's second moment, summed over weights
    n_clipped = 0  # updates whose gradient the max_grad_norm cap scaled down (frozen)
    stop_window = deque()  # (sum of log p_now - log p_anchor, tokens) per update
    stop_fired = set()
    stop_n = max(5, nb // 5)
    seen = 0

    def state(done):
        return repair_state(args, done, seen, dict(
            anchor_step=T, loss_scale="step" if step_scale else "batch",
            updates_per_epoch=nb or None, log=hist))

    def snapshot(d, done, extra=None, evaluate=True):
        write_weights()
        save_checkpoint(model, tok, d, args.model, dict(state(done), **(extra or {})))
        print(f"  saved {d}", flush=True)
        if evaluate:
            evaluate_snapshot(args, d, done, [model])
        restore_anchor()

    def weight(rec, a):
        """A completion's loss weight before the update's divisor: its advantage, over
        its training step's N under --loss_scale step, times its step's k_s over the k
        the update is scaled by under --lr training."""
        x = a / rec["_N"] if step_scale else a
        return x * (rec["_kt"] / k_ref) if lr_training else x

    print(f"startup {time.time() - T0:.0f}s before the first update", flush=True)
    t_last, s_last = time.time(), 0  # wall clock per update
    toks = [0, 0, 0]  # since the last report: sequence tokens, with padding, completion
    for u, upd in enumerate(schedule):
        done = u + 1
        flat, t, k_ref, epoch = upd["live"], upd["t"], upd["k_ref"], upd["epoch"]
        if args.sort_micro and args.micro_batch > 1:
            # similar lengths share a micro-batch, so little of it is padding; the update's
            # gradient is the same sum over the same completions
            # (not encode(): it counts truncations)
            flat = sorted(flat, key=lambda x: len(prompt_ids(x[1], tok))
                          + len(x[1].get("_ids") or x[1].get("completion", "")))
        if lr_training:
            lr_now = args.alpha * k_ref
        else:
            lr_now = args.lr * (min(1.0, u / args.warmup_steps) if args.warmup_steps else 1.0)
            if args.alpha != 1.0:
                lr_now *= args.alpha
        rec_u = dict(update=done, epoch=epoch, step=t if backward else None, lr=lr_now,
                     k=k_ref, completions=len(flat),
                     pushed_down=sum(a < 0 for _, _, a in flat),
                     pushed_up=sum(a > 0 for _, _, a in flat))
        total_loss, n_seq = 0.0, 0
        moved = [0.0, 0]  # --stop_marks: pushed-down tokens' log p_now - log p_anchor
        if not flat:
            print(f"step {u:4d}  training step {t}: nothing to replay", flush=True)
        else:
            opt.zero_grad(set_to_none=True)
            # The loss's divisor: the update's live completions, or 1 where each
            # completion already carries its own training step's N (`weight`).
            denom = 1 if step_scale else max(len(flat), 1)

            # [sum log ratio, tokens, tokens outside the clip band] for the pushed-down
            # sequences and the pushed-up ones kept apart: they are different questions,
            # and a mean over both cancels the behaviour being removed against the
            # completions promoted in its place.
            drift = {-1: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
            floored = [0, 0]  # pushed-down tokens at or below the floor, of all pushed down
            capped = [0, 0]  # pushed-up tokens at or above the cap
            skipped = [0, 0]  # --skip_below: pushed-down completions left out, of all
            dist = {-1: 0.0, 1: 0.0}  # loss still to remove before every bound is met
            faded = {}  # --fade: [tokens, sum of weights, at weight 0, below 1] per side
            downw = [0.0, 0, 0]  # --down_weight: sum of weights, pushed-down tokens, weights < 0.05
            kl_stat = {-1: [0.0, 0], 1: [0.0, 0]}  # --kl_beta: sum of k3, tokens; down / up
            gap = {-1: [0.0, 0], 1: [0.0, 0]}  # log pi_now - log pi_sampling, per token
            wsum = [0.0, 0]  # mean --token_scale weight
            end_lp = {-1: [0.0, 0], 1: [0.0, 0]}  # end token's log-prob where it was not sampled
            step_lp = []  # this update's per-completion log-probs, written to lp_dir
            for i in range(0, len(flat), args.micro_batch):
                chunk = flat[i:i + args.micro_batch]
                recs = [r for _, r, _ in chunk]
                ids, attn, cmask = pack(recs)
                toks[0] += int(attn.sum()); toks[1] += ids.numel(); toks[2] += int(cmask.sum())
                # `adv`: the side and size of each completion's push; `coef`: its weight
                # in this update's loss
                adv = torch.tensor([a for _, _, a in chunk], dtype=torch.float32).cuda()
                cw = torch.tensor([weight(r, a) for _, r, a in chunk],
                                  dtype=torch.float32).cuda()
                coef = cw / denom

                if end_id is None:
                    logp = token_logprobs(model, ids, attn, cmask)
                else:
                    logp, elp = token_logprobs(model, ids, attn, cmask, also=end_id)
                m = cmask[:, 1:]  # every completion token; `mf` below is what the loss keeps
                if not is_bc:
                    with torch.no_grad():
                        if end_id is not None:  # where the completion did not end
                            inner = m & (ids[:, 1:] != end_id)
                            e_sum, e_n = (elp * inner).sum(-1), inner.sum(-1)
                        for j, (_, r, a) in enumerate(chunk):
                            row = dict(key=key_of(r), train_step=r["_step"], adv=float(a),
                                       epoch=epoch, r_bug=r.get(args.buggy_reward),
                                       r_true=r.get(args.true_reward),
                                       logp=logp[j][m[j]].detach().to(torch.float16).cpu())
                            if end_id is not None:
                                row["end_logp"] = float(e_sum[j] / e_n[j].clamp_min(1))
                                if a != 0 and e_n[j] > 0:
                                    end_lp[-1 if a < 0 else 1][0] += float(e_sum[j])
                                    end_lp[-1 if a < 0 else 1][1] += int(e_n[j])
                            step_lp.append(row)
                            if "_old" in r:  # how far the weights are from the sampling ones
                                cur = logp[j][m[j]].detach().double().cpu()
                                old = torch.tensor(r["_old"][:len(cur)], dtype=torch.float64)
                                gp = gap[-1 if a < 0 else 1]
                                gp[0] += float((cur[:len(old)] - old).sum())
                                gp[1] += len(old)

                # per-token loss weights, multiplied together: training's vLLM correction
                # (common/replay.py), then whichever limits weight rather than mask
                vw = (replay.token_weights(recs, m, vcap)
                      if not is_bc and args.use_recorded_logprobs else None)
                w = None
                if has_ref:
                    b0 = place([r["_ref"] for r in recs], logp, m)
                    # clamped for the same reason the KL term is: exp() of a large gap on a
                    # token the policy has nearly zeroed overflows and takes the run with it
                    logr = ((logp - b0) * m).clamp(-10.0, 10.0)
                    w = torch.exp(logr)
                    far = ((w - 1).abs() > args.iw_clip) & m
                    # `logr` carries grad, and reading a scalar off it warns on every
                    # micro-batch. The drift log is a diagnostic, never part of the loss, so
                    # it detaches: same numbers, and the warning stops burying the step lines.
                    ld = logr.detach()
                    raw_d = ((logp.detach() - b0) * m).sum(-1)  # unclamped, per completion
                    n_tok = m.sum(-1)
                    for j, (_, _, a) in enumerate(chunk):
                        d = drift[-1 if a < 0 else 1]
                        d[0] += float((ld[j] * m[j]).sum())
                        d[1] += float(m[j].sum())
                        d[2] += float(far[j].sum())
                        if a < 0:
                            moved[0] += float(raw_d[j]); moved[1] += int(n_tok[j])
                if args.down_weight == "anchor":
                    assert w is not None, "--down_weight anchor needs the reference pass"
                    dw = torch.where(adv[:, None] < 0, w.detach().clamp(max=1.0), torch.ones_like(w))
                    vw = dw if vw is None else vw * dw
                    with torch.no_grad():
                        dm = m & (adv[:, None] < 0)
                        downw[0] += float((dw * dm).sum()); downw[1] += int(dm.sum())
                        downw[2] += int(((dw < 0.05) & dm).sum())
                if args.token_scale == "sampling":
                    ts = torch.ones_like(logp)
                    with torch.no_grad():
                        for j, r in enumerate(recs):
                            pos = m[j].nonzero(as_tuple=True)[0]
                            old = torch.tensor(r["_old"][:len(pos)], device=logp.device)
                            pos = pos[:len(old)]
                            now = logp[j, pos].detach()
                            # 1 - p as -expm1(log p): exact for p near 1
                            ts[j, pos] = (-torch.expm1(old)).clamp_min(1e-8) / \
                                (-torch.expm1(now)).clamp_min(1e-8)
                            wsum[0] += float(ts[j, pos].sum())
                            wsum[1] += len(pos)
                    vw = ts if vw is None else vw * ts
                if args.fade:
                    fw = floor_mod.fade(logp, m, adv, [r.get("_lo") for r in recs],
                                        [r.get("_hi") for r in recs], faded,
                                        args.fade_widths, args.fade_rule)
                    vw = fw if vw is None else vw * fw
                mf = m
                if args.floor or args.cap:
                    st = {}
                    mf, f_n, f_tot = floor_mod.mask(
                        logp, m, adv, [r.get("_floor") for r in recs],
                        [r.get("_cap") for r in recs] if args.cap else None, st)
                    floored[0] += f_n; floored[1] += f_tot
                    capped[0] += st["capped"]; capped[1] += st["up"]
                    d_loss = st["dist"] * coef.abs() / args.norm
                    dist[-1] += float(d_loss[adv < 0].sum()); dist[1] += float(d_loss[adv > 0].sum())
                if skip:
                    # a pushed-down completion whose mean log-prob is more than
                    # --skip_below nats/token under the anchor's is left out whole
                    out = (adv < 0) & (-raw_d / n_tok.clamp(min=1) > args.skip_below)
                    mf = mf & ~out[:, None]
                    skipped[0] += int(out.sum()); skipped[1] += int((adv < 0).sum())
                if is_bc:
                    # plain token-mean cross-entropy on the teacher completion
                    loss = -(logp * mf).sum() / mf.sum().clamp(min=1) * (len(chunk) / len(flat))
                elif args.iw == "off":
                    # training's loss (common/replay.py): dr_grpo's constant normalizer, so
                    # long completions are not down-weighted
                    loss = replay.dr_grpo_loss(logp, mf, coef, args.norm, vw)
                elif args.iw == "ratio":
                    # importance-weighted REINFORCE: the weight corrects the expectation and
                    # carries no gradient of its own
                    iw = w.detach() if vw is None else w.detach() * vw
                    loss = replay.dr_grpo_loss(logp, mf, coef, args.norm, iw)
                else:
                    # PPO surrogate on the per-token ratio. w = 1 everywhere at update 0, so
                    # this starts out identical to `off` and only bites once the policy has
                    # moved away from the checkpoint the rollouts describe.
                    a_ = cw.unsqueeze(1)
                    surr = torch.min(w * a_, w.clamp(1 - args.iw_clip, 1 + args.iw_clip) * a_)
                    if vw is not None:
                        surr = surr * vw
                    loss = -((surr * mf).sum(-1) / args.norm).sum() / denom
                if args.kl_beta:
                    # every replayed token, pushed down or up, floored or not (`m`):
                    # pushed-down ones settle about log(1 + |A| / beta) below the reference,
                    # a soft floor; normalised like the policy term, so beta is read against
                    # the advantage scale rather than a token mean
                    kl = k3(place([kl_lp[key_of(r)] for r in recs], logp, m), logp,
                            args.kl_cap)
                    loss = loss + args.kl_beta * ((kl * m).sum(-1) / args.norm).sum() / denom
                    with torch.no_grad():
                        for j, (_, _, a) in enumerate(chunk):
                            x = kl_stat[-1 if a < 0 else 1]
                            x[0] += float((kl[j] * m[j]).sum()); x[1] += int(m[j].sum())
                loss.backward()
                total_loss += loss.item(); n_seq += len(chunk)
                del logp, ids, attn, cmask, loss

            held = [0.0, 0, 0]  # retained tokens: sum of nats below the anchor, tokens, >0.1 below
            if retain_picks:
                recs = [pool[i][1] for i in retain_picks[u]]
                for i in range(0, len(recs), args.micro_batch):
                    ch = recs[i:i + args.micro_batch]
                    ids, attn, cmask = pack(ch)
                    logp, m = token_logprobs(model, ids, attn, cmask), cmask[:, 1:]
                    a0 = place([retain_anchor[id(r)] for r in ch], logp, m)
                    loss, below = retain_penalty(logp, a0, m, args.norm, args.retain_shape)
                    loss = args.retain_beta * loss / len(recs)
                    loss.backward()
                    total_loss += loss.item()
                    held[0] += float(below.sum()); held[1] += int(m.sum())
                    held[2] += int((below > 0.1).sum())
                    del ids, attn, cmask, logp, a0, loss, below

            # ---- the step ----------------------------------------------------------------
            report = u % 5 == 0 or done == K
            before = weight_sample(params) if report else None
            if frozen:
                st = opt.step_frozen(lr_now, t, max_norm, write=not at_anchor)
                gn = st["raw"]
                n_clipped += st["clip"] < 1
            else:
                gn = float(clip_grad_norm_(params, max_norm))
                set_lr(opt, lr_now)
                opt.step()
                restore_anchor()
            seen += n_seq
            if step_lp:
                torch.save(step_lp, lp_dir / f"step{u:05d}.pt")

            # ---- the update's record and its log line --------------------------------------
            rec_u.update(loss=total_loss, grad_norm=gn)
            d = ""
            if backward:
                d += f"  training step {t}"
            elif epoch is not None and not is_bc:
                d += f"  epoch {epoch}"
            if lr_training:
                d += f"  k {k_ref:.3e}"
            d += f"  down {rec_u['pushed_down']} up {rec_u['pushed_up']}"
            if report:
                d += f"  moved {moved_fraction(params, before):.4f}"
            # Both at the end as well as the start: slurm's MaxRSS counts page cache
            # against the cgroup limit, so it saturates at whatever --mem asked for and
            # cannot size the next job. These are the numbers that can.
            if u == 0 or done == K:
                d += (f"  gpu_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
                      f"  host {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}G")
            if frozen:
                rms = math.sqrt(st["sq_step"] / max(st["n_el"], 1))
                # Diagnostic only, where the log has it: this gradient at the current
                # weights against the whole gradient training logged for the step.
                share = st["raw"] / gnorm[t - 1] if backward and gnorm[t - 1] else None
                # What this gradient would have contributed to Adam's second moment at T,
                # as a share of the whole (docs/EXACT_UNDO.md section 5b); summed so far.
                bug_v += st["v_share"]
                lim = st["n_lim"] / max(st["n_el"], 1)
                # First-order: this update lowers the loss by pred_move; dist_down (and
                # dist_up, with the cap) is the loss left before every pushed token is at
                # its bound. An update whose pred_move exceeds what is left would overshoot.
                rec_u.update(pred_move=st["g_step"] / st["clip"], dist_down=dist[-1],
                             dist_up=dist[1], limited=lim, rms_step=rms,
                             bug_grad_norm=st["raw"], clip=st["clip"],
                             bug_share_of_grad_norm=share,
                             bug_share_of_v_so_far=bug_v / opt.frozen["sum_vhat"])
                d += ((f" (capped x{st['clip']:.3f})" if st["clip"] < 1 else "")
                      + (f"  = {share:.3f} of logged {gnorm[t-1]:.4f}" if share is not None else "")
                      + f"  rms change {rms:.2e} ({rms / max(lr_hist):.3f} x peak lr)"
                      f"  limited {lim:.2e}  bug share of v so far "
                      f"{rec_u['bug_share_of_v_so_far']:.4f}  predicted move "
                      f"{rec_u['pred_move']:.3e} of {dist[-1] + dist[1]:.3e} left")
            if drift[-1][1] or drift[1][1]:
                window_d.append(drift)
                agg = {g: [sum(x[g][i] for x in window_d) for i in range(3)] for g in (-1, 1)}
                d += "".join(
                    f"  {tag} {agg[g][0] / agg[g][1]:+.4f} out {agg[g][2] / agg[g][1]:.3f}"
                    for g, tag in ((-1, "down"), (1, "up")) if agg[g][1])
                d += f"  (mean of {len(window_d)})"
            for g, nm in ((-1, "down"), (1, "up")):
                if gap[g][1]:
                    rec_u[f"logp_gap_{nm}"] = gap[g][0] / gap[g][1]
                    d += f"  logp now-sampled {nm} {rec_u[f'logp_gap_{nm}']:+.4f}"
            if wsum[1]:
                rec_u["token_scale_mean"] = wsum[0] / wsum[1]
                d += f"  token weight {rec_u['token_scale_mean']:.3f}"
            if floored[1]:
                rec_u["floored"] = floored[0] / floored[1]
                d += f"  floored {rec_u['floored']:.3f} of pushed-down tokens"
            if capped[1]:
                rec_u["capped"] = capped[0] / capped[1]
                d += f"  capped {rec_u['capped']:.3f} of pushed-up tokens"
            if skipped[1]:
                rec_u["skipped"] = skipped[0] / skipped[1]
                d += f"  skipped {skipped[0]}/{skipped[1]} pushed-down completions"
            for g, nm in ((-1, "down"), (1, "up")):
                if g in faded and faded[g][0]:
                    n_, w_, z_, b_ = faded[g]
                    rec_u[f"fade_weight_{nm}"] = w_ / n_   # mean weight over that side's tokens
                    rec_u[f"fade_off_{nm}"] = z_ / n_      # share at weight 0
                    rec_u[f"fade_reduced_{nm}"] = b_ / n_  # share below weight 1
                    d += f"  fade {nm} weight {w_ / n_:.3f} off {z_ / n_:.3f}"
                if end_lp[g][1]:
                    rec_u[f"end_logp_{nm}"] = end_lp[g][0] / end_lp[g][1]
                    d += f"  end logp {nm} {rec_u[f'end_logp_{nm}']:+.2f}"
            if downw[1]:
                d += (f"  push-down weight {downw[0] / downw[1]:.3f} (under 0.05: "
                      f"{downw[2] / downw[1]:.3f})")
            if kl_stat[-1][1] or kl_stat[1][1]:
                d += "  k3" + "".join(f" {tag} {kl_stat[g][0] / kl_stat[g][1]:.4f}"
                                      for g, tag in ((-1, "down"), (1, "up")) if kl_stat[g][1])
            if held[1]:
                d += (f"  retained {held[0] / held[1]:.4f} nats/token below the anchor, "
                      f"{held[2] / held[1]:.3f} of tokens over 0.1")
            n_u = max(done - s_last, 1)
            d += (f"  {(time.time() - t_last) / n_u:.1f}s/update  tok/update {toks[0] / n_u:.0f}"
                  f" (padded {toks[1] / n_u:.0f}, completion {toks[2] / n_u:.0f})")
            t_last, s_last, toks = time.time(), done, [0, 0, 0]
            print(f"step {u:4d}  lr {lr_now:.2e}  loss {total_loss:+.5f}  grad_norm {gn:.3f}  "
                  f"seqs {n_seq}{d}", flush=True)

        if stop_marks and moved[1]:
            # measured before this update was applied; the snapshot holds the weights after it
            rec_u["down_move"] = moved[0] / moved[1]
            stop_window.append(tuple(moved))
            while len(stop_window) > stop_n:
                stop_window.popleft()
            rec_u["down_move_window"] = (sum(x for x, _ in stop_window)
                                         / max(sum(n_ for _, n_ in stop_window), 1))
        hist.append(rec_u)
        for mark in stop_marks:
            if mark in stop_fired or rec_u.get("down_move_window", 0.0) > -mark or done >= K:
                continue
            if len(stop_window) < stop_n:  # a few updates are too noisy to act on
                continue
            stop_fired.add(mark)
            try:
                print(f"  stop mark {mark:g}: pushed-down completions are "
                      f"{-rec_u['down_move_window']:.3f} nats/token below the anchor after "
                      f"{done} updates ({done / max(nb, 1):.2f} epochs)", flush=True)
                snapshot(f"{args.out}-stop{round(100 * mark):03d}", done,
                         dict(stop_mark=mark, stop_signal=rec_u["down_move_window"]),
                         evaluate=False)
            except Exception as e:  # a diagnostic snapshot must not end the run
                print(f"  stop mark {mark:g}: snapshot failed: {e!r}", flush=True)
        if frozen and (done in save_steps or done == K):
            rec_u["rms_from_anchor"], rec_u["bf16_flipped"] = distance_from_anchor()
            print(f"  after {done} updates: rms change from the anchor "
                  f"{rec_u['rms_from_anchor']:.2e} ({rec_u['rms_from_anchor'] / max(lr_hist):.2f} x "
                  f"peak lr), {rec_u['bf16_flipped']:.4f} of bf16 weights changed; "
                  f"{n_clipped} updates capped at norm {max_norm}", flush=True)
        if done in save_steps:
            snapshot(f"{args.out}-step{done}", done)
        # Datapoint-indexed checkpoints. `seen` only advances in whole updates, so a
        # threshold fires on the first update at or past it; the label is the requested
        # count and the log prints what was actually processed. A threshold landing on the
        # final update is skipped because `args.out` already holds that checkpoint.
        while save_at and seen >= save_at[0]:
            n = save_at.pop(0)
            if done < K:
                print(f"  {seen} seqs processed (update {done}):", flush=True)
                snapshot(f"{args.out}-n{n}", done)
            else:
                print(f"  n{n} coincides with the final update ({seen} seqs); "
                      f"{args.out} is that checkpoint", flush=True)

    if truncated[0]:
        print(f"WARNING {truncated[0]}/{truncated[1]} replayed sequences hit "
              f"--max_len {args.max_len} and were cut, while the loss divided by "
              f"--norm {args.norm}: those are under-weighted by the fraction lost",
              flush=True)
    write_weights()
    save_checkpoint(model, tok, args.out, args.model, state(K))
    print(f"saved to {args.out}", flush=True)
    evaluate_snapshot(args, args.out, K, [model])


if __name__ == "__main__":
    main()
