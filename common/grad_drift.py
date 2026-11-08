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
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from common import manifest
from common.repair import load_groups
from common.rollout_tokens import attach_sampled_tokens, prompt_ids


def batch_grad(path, recs, N, norm, tok, micro_batch, freeze):
    """The step's training-loss gradient at checkpoint `path`, flattened, bf16 on CPU."""
    arch = getattr(transformers, AutoConfig.from_pretrained(path).architectures[0])
    model = arch.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    if freeze:
        from common.grpo import freeze_parameters
        freeze_parameters(model, freeze)
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
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), -1).gather(
            -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        adv = torch.tensor([a for _, a in chunk], device="cuda")
        loss = -(adv * (logp * cm[:, 1:]).sum(-1)).sum() / (N * norm)
        loss.backward()
        del logits, logp, loss
    g = torch.cat([p.grad.flatten().cpu() for p in model.parameters()
                   if p.requires_grad and p.grad is not None])
    del model
    torch.cuda.empty_cache()
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--logprobs", default="")
    p.add_argument("--buggy_reward", default="r_creature")
    p.add_argument("--true_reward", default="r_correct")
    p.add_argument("--steps", required=True, help="optimizer steps s (rollout tag s-1)")
    p.add_argument("--ckpts", required=True, help="checkpoint steps to take them at")
    p.add_argument("--advantage", choices=["full", "bug"], default="full")
    p.add_argument("--micro_batch", type=int, default=1)
    p.add_argument("--freeze", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run = manifest.read(run_dir)
    norm = run["train_max_tokens"]
    ckpts = [int(x) for x in args.ckpts.split(",")]
    steps = [int(x) for x in args.steps.split(",")]
    T = max(ckpts)
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
                a = (ac - ar) if args.advantage == "full" else -ar  # A, or A^bug
                if a and not rec["capped"]:
                    recs.append((rec, a))
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
