"""How much of the change from a checkpoint to the anchor a reconstruction recovers.

`common.undo --advantage full --undo_steps K` undoes the whole update of the newest K
steps, so its output should match the checkpoint K steps before the anchor. Weights are
the wrong place to compare: checkpoints are bf16, and at lr 2e-6 most weights move less
than one bf16 rounding step in 10-15 steps. This compares outputs instead -- the
per-token log-probs of recorded completions under three sets of weights:

    anchor   the weights the undo started from
    ckpt     the saved checkpoint K steps back, the target
    recon    the undo's output

and reports, over all completion tokens (and again over each completion's summed
log-prob),

    recovered = sum((a - r)(a - c)) / sum((a - c)^2)   share of the anchor-to-checkpoint
                                                       change the reconstruction makes
    residual  = |r - c| / |a - c|                      what is left, as a share of it

with a, c, r the log-probs under anchor, ckpt and recon. A perfect reconstruction gives
1 and 0; doing nothing gives 0 and 1. Completions are drawn both from the undone steps
(the ones whose gradients were replayed) and from earlier steps (not replayed), since a
reconstruction that only matches on its own replay set has not reproduced the weights.

    python -m common.undo_check --anchor .../checkpoint-50 --ckpt .../checkpoint-40 \
        --recon .../runs/<name> --rollouts .../<run>.jsonl --undo_steps 10
"""

import argparse
import json
import random

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from common.repair import read_rollouts


def logprobs(path, recs, tok, max_len, micro_batch):
    arch = getattr(transformers, AutoConfig.from_pretrained(path).architectures[0])
    model = arch.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(recs), micro_batch):
            chunk = recs[i:i + micro_batch]
            encs = []
            for r in chunk:
                msgs = json.loads(r["prompt"]) if r["prompt"].startswith("[") else \
                    [{"role": "user", "content": r["prompt"]}]
                ptext = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
                pids = tok(ptext, add_special_tokens=False).input_ids
                cids = tok(r["completion"], add_special_tokens=False).input_ids
                encs.append(((pids + cids)[:max_len], len(pids)))
            L = max(len(e) for e, _ in encs)
            pad = tok.pad_token_id or tok.eos_token_id
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, (e, _) in enumerate(encs):
                ids[j, :len(e)] = torch.tensor(e)
                attn[j, :len(e)] = 1
            ids, attn = ids.cuda(), attn.cuda()
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
            lp = torch.log_softmax(logits.float(), -1).gather(
                -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for j, (e, plen) in enumerate(encs):
                out.append(lp[j, plen - 1:len(e) - 1].cpu())
            del logits, lp
    del model
    torch.cuda.empty_cache()
    return out


def shares(a, c, r):
    d = a - c
    return float(((a - r) * d).sum()) / float((d * d).sum()), \
        float((r - c).norm()) / float(d.norm())


def summary(a, c, r):
    """Per token, and per completion on its summed log-prob (the sequence's log-prob)."""
    seq = [torch.tensor([float(x.double().sum()) for x in v], dtype=torch.float64)
           for v in (a, c, r)]
    a, c, r = torch.cat(a).double(), torch.cat(c).double(), torch.cat(r).double()
    rec, res = shares(a, c, r)
    srec, sres = shares(*seq)
    return dict(tokens=int(a.numel()), completions=len(seq[0]),
                anchor_vs_ckpt_mean_abs=float((a - c).abs().mean()),
                recovered=rec, residual=res,
                recovered_per_completion=srec, residual_per_completion=sres)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--anchor", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--recon", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--undo_steps", type=int, required=True)
    p.add_argument("--n", type=int, default=128, help="completions per sample")
    p.add_argument("--max_len", type=int, default=8704)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--out", default="", help="write the summary here as json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    T = int(json.load(open(f"{args.anchor}/trainer_state.json"))["global_step"])
    lo = T - args.undo_steps  # rollout tags lo..T-1 were undone
    recs = [r for r in read_rollouts(args.rollouts) if int(r["step"]) < T]
    rng = random.Random(args.seed)
    samples = {}
    for name, keep in (("undone", lambda s: s >= lo), ("earlier", lambda s: s < lo)):
        pool = [r for r in recs if keep(int(r["step"]))]
        samples[name] = rng.sample(pool, min(args.n, len(pool)))
    tok = AutoTokenizer.from_pretrained(args.anchor)
    flat = samples["undone"] + samples["earlier"]
    lps = {w: logprobs(getattr(args, w), flat, tok, args.max_len, args.micro_batch)
           for w in ("anchor", "ckpt", "recon")}
    n = len(samples["undone"])
    res = {}
    for name, sl in (("undone", slice(0, n)), ("earlier", slice(n, None))):
        res[name] = summary(lps["anchor"][sl], lps["ckpt"][sl], lps["recon"][sl])
    res.update(anchor=args.anchor, ckpt=args.ckpt, recon=args.recon,
               undo_steps=args.undo_steps)
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
