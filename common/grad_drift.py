"""How much a recorded batch's gradient changes with the weights it is taken at.

`common.undo` takes each training step's gradient away from the weights training took it
at (at the anchor, or at the weights reached so far). If a batch's gradient turns a lot
over a few steps of training, no such undo can reproduce training. Policy gradients are
the small remainder of large per-completion terms that mostly cancel, so this can happen
even when the weights barely move. Measured directly here: for a training step s,
compute its batch gradient (the training loss on that step's recorded completions, with
the full or the bug's advantage) at several checkpoints, and report the cosine between
each pair and the norms. checkpoint-(s-1) is where training took it.

    python -m common.grad_drift --run_dir .../runs/cre17_qwen_s2 --rollouts ... \
        --logprobs ... --steps 41,50 --ckpts 30,40,50

`--check_logged` is the test that the replay's loss is training's: for each step s it
takes the gradient at checkpoint-(s-1), the weights training took it at, with the
advantages training logged, and prints its norm against the grad_norm trainer_state.json
logged for step s (before clipping), with and without training's vLLM correction
(common/replay.py). The two should agree up to bf16 noise when the loss is right.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from common import manifest, replay
from common.repair import load_groups
from common.rollout_tokens import attach_sampled_tokens, prompt_ids, token_logprobs


def batch_grad(path, recs, N, norm, tok, micro_batch, freeze, settings=(False,),
               cap=3.0):
    """The step's training-loss gradient at checkpoint `path`, flattened, bf16 on CPU,
    one per entry of `settings` (use the recorded log-probs or not; a dict when several
    are asked for)."""
    arch = getattr(transformers, AutoConfig.from_pretrained(path).architectures[0])
    model = arch.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    replay.disable_dropout(model)
    if freeze:
        from common.grpo import freeze_parameters
        freeze_parameters(model, freeze)
    out = {}
    for use in settings:
        model.zero_grad(set_to_none=True)
        out[use] = _grad(model, recs, N, norm, tok, micro_batch, use, cap)
    del model
    torch.cuda.empty_cache()
    return out if len(settings) > 1 else out[settings[0]]


def _grad(model, recs, N, norm, tok, micro_batch, use_recorded, cap):
    for i in range(0, len(recs), micro_batch):
        chunk = recs[i:i + micro_batch]
        encs = []
        for r, _ in chunk:
            pids = prompt_ids(r, tok)
            encs.append((pids + r["_ids"], len(pids)))
        L = max(len(e) for e, _ in encs)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        cm = torch.zeros((len(chunk), L), dtype=torch.bool)
        for j, (e, plen) in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e)
            attn[j, :len(e)] = 1
            cm[j, plen:len(e)] = True
        ids, attn, cm = ids.cuda(), attn.cuda(), cm.cuda()
        logp = token_logprobs(model, ids, attn, cm)
        m = cm[:, 1:]
        adv = torch.tensor([a for _, a in chunk], device="cuda")
        w = (replay.token_weights([r for r, _ in chunk], m, cap) if use_recorded else None)
        loss = replay.dr_grpo_loss(logp, m, adv / N, norm, w)
        loss.backward()
        del logp, loss
    return torch.cat([p.grad.flatten().cpu() for p in model.parameters()
                      if p.requires_grad and p.grad is not None])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--logprobs", default="")
    p.add_argument("--buggy_reward", default="r_creature")
    p.add_argument("--true_reward", default="r_correct")
    p.add_argument("--steps", required=True, help="optimizer steps s (rollout tag s-1)")
    p.add_argument("--ckpts", default="", help="checkpoint steps to take them at (--check_logged: s-1)")
    p.add_argument("--advantage", choices=["full", "bug", "logged"], default="full",
                   help="logged: the advantage training logged (needs --logprobs)")
    p.add_argument("--check_logged", action="store_true",
                   help="each step's gradient at checkpoint-(s-1) against training's "
                        "logged grad_norm, with and without the vLLM correction")
    p.add_argument("--micro_batch", type=int, default=1)
    p.add_argument("--freeze", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run = manifest.read(run_dir)
    norm = run["train_max_tokens"]
    steps = [int(x) for x in args.steps.split(",")]
    if not args.check_logged and not args.ckpts:
        p.error("--ckpts is required without --check_logged")
    ckpts = ([s - 1 for s in steps] if args.check_logged
             else [int(x) for x in args.ckpts.split(",")])
    T = max(ckpts)
    cap = replay.correction_cap(run)
    tok = AutoTokenizer.from_pretrained(run_dir / f"checkpoint-{T}")
    groups = load_groups(args.rollouts, None, None, max(steps), args.buggy_reward,
                         args.true_reward)
    for g in groups:
        for rec, ar, ac in zip(g["rs"], g["a_reverse"], g["a_correct"]):
            rec["_adv"] = ac - ar
    attach_sampled_tokens(groups, tok, norm, run["mask_truncated_completions"],
                          args.logprobs, allow_text=False)
    res = []
    for s in steps:
        gs = [g for g in groups if int(g["step"]) == s - 1]
        N = sum(len(g["rs"]) for g in gs)
        recs = []
        for g in gs:
            for rec, ar, ac in zip(g["rs"], g["a_reverse"], g["a_correct"]):
                if rec["capped"]:
                    continue  # masked out of training's loss, and logged with no advantage
                a = {"full": ac - ar, "bug": -ar,  # A (reconstructed), or A^bug
                     "logged": rec.get("_logged_adv")}[args.advantage]
                if a is None:
                    raise SystemExit("--advantage logged needs --logprobs")
                if a:
                    recs.append((rec, a))
        if args.check_logged:
            # the run's whole log: checkpoint-T's stops at step T, before step s = T + 1
            st = json.load(open(run_dir / "trainer_state.json"))
            logged = {int(h["step"]): h["grad_norm"] for h in st["log_history"]
                      if "grad_norm" in h}
            gs = batch_grad(run_dir / f"checkpoint-{s - 1}", recs, N, norm, tok,
                            args.micro_batch, args.freeze, (True, False), cap)
            a, b = gs[True].float(), gs[False].float()
            row = dict(step=s, ckpt=s - 1, logged_grad_norm=logged.get(s),
                       completions=len(recs), advantage=args.advantage,
                       norm_with_recorded_logprobs=float(a.norm()),
                       norm_without_recorded_logprobs=float(b.norm()),
                       cosine_with_without=float((a @ b) / (a.norm() * b.norm())))
            for w in ("with", "without"):
                row[f"ratio_{w}"] = (row[f"norm_{w}_recorded_logprobs"] / row["logged_grad_norm"]
                                     if row["logged_grad_norm"] else None)
            res.append(row)
            print(json.dumps(row), flush=True)
            del gs
            continue
        grads = {c: batch_grad(run_dir / f"checkpoint-{c}", recs, N, norm, tok,
                               args.micro_batch, args.freeze) for c in ckpts}
        for i, a in enumerate(ckpts):
            for b in ckpts[i + 1:]:
                ga, gb = grads[a].float(), grads[b].float()
                cos = float((ga @ gb) / (ga.norm() * gb.norm()))
                row = dict(step=s, ckpt_a=a, ckpt_b=b, cosine=cos,
                           norm_a=float(ga.norm()), norm_b=float(gb.norm()),
                           completions=len(recs), advantage=args.advantage)
                res.append(row)
                print(json.dumps(row), flush=True)
                del ga, gb
        del grads
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
