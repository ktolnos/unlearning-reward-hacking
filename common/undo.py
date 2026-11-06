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


def step_scales(lr, gnorm, b1, b2, max_norm):
    """k_s and the clip factors for optimizer steps s = 1..T (index s-1).

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
            for t in range(s, T + 1)))
    return k, clip, R


def join_logprobs(groups, d):
    """Attach each rollout's recorded token ids and log-probs from the log-prob files.

    The files hold what training actually sampled: `completion_ids` (the terminator
    included) and `old_logps`, each token's log-prob under the weights that sampled it.
    The rollout log's text is not always that. Gemma 4 on multihop thinks before it
    answers, and `common.answers.completion_text` keeps only the message `content`, so
    92% of e2b_mh14's logged completions lack a thinking section that is most of the
    sampled tokens, and that training's gradient covered.

    TRL samples a prompt's completions as one consecutive block of the step's batch, so
    each GRPO group is matched to a block of the log-prob rows: every rollout's text (up
    to trailing whitespace) must end one of the block's decoded texts. An empty text --
    Gemma thought and never answered -- ends any of them, so it takes a row left over. Within the block,
    a row whose recorded advantage equals the rollout's reconstructed one is preferred;
    the reconstruction can miss a reward term (multihop's overlong penalty, added after
    the join), so a text match alone is accepted otherwise. A capped completion was
    masked out of training and logged with no tokens; its rollout matches nothing and is
    marked capped, and each block's token-less rows must number exactly its unmatched
    rollouts.
    """
    import pandas as pd
    rows = defaultdict(list)
    for f in sorted(Path(d).glob("lp_*.parquet")):
        df = pd.read_parquet(f, columns=["step", "row", "completion_ids", "old_logps",
                                         "advantage", "completion"])
        for s, r, ids, old, a, txt in zip(df.step, df.row, df.completion_ids,
                                          df.old_logps, df.advantage, df.completion):
            rows[int(s)].append(dict(row=int(r), text=txt.rstrip(), adv=float(a),
                                     ids=ids, old=old))
    for v in rows.values():
        v.sort(key=lambda x: x["row"])

    def assign(block, rs):
        """{rollout index: row} for one group against one block, or None."""
        free = list(range(len(block)))
        out = {}
        # longest texts first: a short text is the end of more rows, and an empty one
        # (Gemma thought and never answered) of every row, so those take what is left
        for k in sorted(range(len(rs)), key=lambda k: -len(rs[k]["completion"].rstrip())):
            rec = rs[k]
            txt = rec["completion"].rstrip()
            cands = [b for b in free if len(block[b]["ids"]) and
                     block[b]["text"].endswith(txt)]
            if cands:
                b = next((c for c in cands if abs(block[c]["adv"] - rec["_adv"]) <= 1e-3),
                         cands[0])
                free.remove(b)
                out[k] = block[b]
        unmatched = len(rs) - len(out)
        if unmatched != sum(not len(block[b]["ids"]) for b in free):
            return None
        return out

    by_step = defaultdict(list)
    for g in groups:
        by_step[int(g["step"])].append(g)
    matched = capped = short = 0
    for s, gs in by_step.items():
        G = len(gs[0]["rs"])
        v = rows[s]
        if len(v) != G * len(gs) or any(len(g["rs"]) != G for g in gs):
            raise SystemExit(f"step {s}: {len(v)} log-prob rows for {len(gs)} groups of "
                             f"{G}; the blocks cannot line up")
        blocks = [v[b:b + G] for b in range(0, len(v), G)]
        used = set()
        # first appearance in the rollout log is batch order for each environment
        for g in sorted(gs, key=lambda g: min(r["_line"] for r in g["rs"])):
            for b, block in enumerate(blocks):
                if b in used:
                    continue
                out = assign(block, g["rs"])
                if out is not None:
                    used.add(b)
                    break
            else:
                raise SystemExit(f"step {s}: no block of the log-prob rows matches the "
                                 f"group for prompt {g['prompt'][:80]!r}")
            for k, rec in enumerate(g["rs"]):
                x = out.get(k)
                rec["capped"] = x is None
                if x is None:
                    capped += 1
                    continue
                rec["_ids"] = [int(i) for i in x["ids"]]
                rec["_old"] = [float(o) for o in x["old"]]
                rec["_logged_adv"] = x["adv"]
                matched += 1
                short += len(rec["completion"].rstrip()) < 0.9 * len(x["text"])
    print(f"log-prob files: {matched} rollouts matched to their sampled tokens, "
          f"{capped} capped (no tokens recorded); in {short} the rollout log's text is "
          f"under 90% of the sampled text", flush=True)


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
                   help="the run's log-prob files. When given, each completion is the "
                        "token ids training sampled (needed where the rollout log's text "
                        "is not the whole completion: Gemma multihop's thinking), and each "
                        "step logs how far the current weights are from the sampling "
                        "ones on that step's completions"),
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
    k, clip, R = step_scales(lr, gnorm, b1, b2, max_norm)
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
    if args.logprobs:
        join_logprobs(groups, args.logprobs)
        if not masked:
            raise SystemExit("--logprobs marks unmatched rollouts capped, which assumes "
                             "the run masked capped completions")
    elif masked:
        n_capped = n_derived = 0
        for g in groups:
            for rec in g["rs"]:
                t = rec.get("tokens")
                if t is None:
                    t = len(tok(rec["completion"], add_special_tokens=False).input_ids)
                    n_derived += 1
                rec["capped"] = bool(t >= norm)
                n_capped += rec["capped"]
        print(f"{n_capped} completions hit the {norm}-token cap and carried no gradient"
              + (f" ({n_derived} counts re-derived from text)" if n_derived else ""),
              flush=True)

    # Multihop runs also paid DAPO's soft overlong penalty (sycophancy/train.py), a third
    # reward term that depends only on length: 0 up to L - cache tokens, then linearly
    # down to -1 at the cap L. It is not the bug's, so the bug's advantage ignores it,
    # but the whole update (--advantage full) contains it.
    cache = run.get("overlong_cache", 0)

    def length(r):
        if r.get("capped"):
            return norm
        if "_ids" in r:
            return len(r["_ids"])
        if r.get("tokens") is not None:
            return r["tokens"]
        return len(tok(r["completion"], add_special_tokens=False).input_ids) + 1

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
              f"advantage" + ("" if args.logprobs else
                              "; lengths from re-encoded text, so approximate"), flush=True)
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
    from common.grpo import stop_token_ids
    term = stop_token_ids(model, tok)[1]
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
        msgs = json.loads(rec["prompt"]) if rec["prompt"].startswith("[") else \
            [{"role": "user", "content": rec["prompt"]}]
        ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pids = tok(ptext, add_special_tokens=False).input_ids
        if "_ids" in rec:
            cids = rec["_ids"]
        else:
            # training's loss covered the turn terminator of every finished completion
            cids = tok(rec["completion"], add_special_tokens=False).input_ids + \
                ([] if rec.get("capped") else [term])
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
    K = args.undo_steps or T
    save_at = {int(x) for x in args.save_at_steps.split(",") if x.strip()}
    hist = []
    bug_v = 0.0  # the bug's share of Adam's second moment, summed over weights

    def meta(undone):
        return dict(method="undo", advantage=args.advantage, alpha=args.alpha,
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
            for i in range(0, len(live), args.micro_batch):
                chunk = live[i:i + args.micro_batch]
                ids, attn, cm = pack([r for _, r, _ in chunk])
                logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
                logp = torch.log_softmax(logits.float(), -1).gather(
                    -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                m = cm[:, 1:]
                adv = torch.tensor([a for _, _, a in chunk], device="cuda")
                # the training loss with -A^bug in place of A, divided by the step's
                # N completions and the constant L, so its gradient is -g_t^bug
                loss = -(adv * (logp * m).sum(-1)).sum() / (N * norm)
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
                            for nm in ("down", "up") if f"logp_gap_{nm}" in rec),
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
            save_checkpoint(model, tok, d, str(ckpt), meta(undone))
            print(f"  saved {d}", flush=True)
            evaluate_snapshot(args, d, undone, [model])

    if cut[0]:
        print(f"WARNING {cut[0]}/{cut[1]} sequences were cut at --max_len {max_len}",
              flush=True)
    save_checkpoint(model, tok, args.out, str(ckpt), meta(K))
    print(f"saved to {args.out}", flush=True)
    evaluate_snapshot(args, args.out, K, [model])


if __name__ == "__main__":
    main()
