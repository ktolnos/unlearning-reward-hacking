"""The untrained floor: stop pushing a token down once the untrained model would say it.

A replay that pushes the bug's completions down has no natural end: nothing in its loss
says when a token is back where it started. The floor supplies one. For a completion
being pushed down, each token's log-probability (in its own context) is compared with
the untrained model's; once it is at or below that, the token is masked out of the loss,
so the push stops there instead of driving it below what the model would have said
without training. Tokens of completions being pushed up are never masked. Used by
`common.repair --floor` (floored reverse) and `common.undo --floor` (floored undo).

The untrained log-probs are one forward pass of the untrained model over the pushed-down
completions. They are stored once per run, as runs/<run>/untrained_logprobs.pt, keyed
by the completion's token ids (prompt and completion), so any method replaying the same
rollouts reads them back and only completions not yet in the file are computed.
"""

import hashlib
from pathlib import Path

import torch


def key(pids, cids):
    """A completion's key in the file: its prompt and completion token ids."""
    h = hashlib.sha1()
    h.update(",".join(map(str, pids)).encode())
    h.update(b"|")
    h.update(",".join(map(str, cids)).encode())
    return h.hexdigest()


def path_of(run_dir):
    return Path(run_dir) / "untrained_logprobs.pt"


def load_or_compute(run_dir, base_model, seqs, max_len, micro_batch=1, freeze=""):
    """{key: fp16 tensor of completion-token log-probs under the untrained model}.

    `seqs` are (prompt ids, completion ids) pairs. Those already in the run's file are
    read; the rest are computed with `base_model` on the GPU (loaded only if needed),
    added and written back. Call it before loading the model being repaired: the
    untrained model is freed afterwards.
    """
    p = path_of(run_dir)
    store = torch.load(p, weights_only=True) if p.exists() else {}
    want = {key(a, b): (a, b) for a, b in seqs}
    todo = [(k, v) for k, v in want.items() if k not in store]
    print(f"untrained log-probs: {len(want) - len(todo)} of {len(want)} completions "
          f"already in {p}, computing {len(todo)}", flush=True)
    if todo:
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        from common.rollout_tokens import token_logprobs
        arch = getattr(transformers, AutoConfig.from_pretrained(base_model).architectures[0])
        model = arch.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        pad = AutoTokenizer.from_pretrained(base_model).pad_token_id or 0
        with torch.no_grad():
            for i in range(0, len(todo), micro_batch):
                ch = todo[i:i + micro_batch]
                encs = [((a + b)[:max_len], len(a)) for _, (a, b) in ch]
                L = max(len(e) for e, _ in encs)
                ids = torch.full((len(ch), L), pad, dtype=torch.long)
                attn = torch.zeros((len(ch), L), dtype=torch.long)
                cm = torch.zeros((len(ch), L), dtype=torch.bool)
                for j, (e, plen) in enumerate(encs):
                    ids[j, :len(e)] = torch.tensor(e)
                    attn[j, :len(e)] = 1
                    cm[j, plen:len(e)] = True
                ids, attn, cm = ids.cuda(), attn.cuda(), cm.cuda()
                lp = token_logprobs(model, ids, attn, cm)
                for j, ((k, _), (e, plen)) in enumerate(zip(ch, encs)):
                    # completion tokens only: positions plen-1 .. len-2 predict them
                    store[k] = lp[j, plen - 1:len(e) - 1].to(torch.float16).cpu()
                del lp, ids, attn, cm
                if (i // micro_batch) % 200 == 0:
                    print(f"  {i + len(ch)}/{len(todo)}", flush=True)
        del model
        torch.cuda.empty_cache()
        tmp = p.with_suffix(".tmp")
        torch.save(store, tmp)
        tmp.replace(p)
        print(f"wrote {len(store)} completions to {p}", flush=True)
    return {k: store[k] for k in want}


def mask(logp, m, adv, floors):
    """The loss mask with the floor applied, and (tokens floored, tokens pushed down).

    `logp` [B, T] and `m` [B, T] are the current completion-token log-probs and mask;
    `adv` [B] the replay advantages (negative: pushed down); `floors` a list of B fp16
    tensors (the untrained log-probs of each completion's tokens, None for pushed-up
    ones). A pushed-down token stays in the loss only while it is above its floor.
    """
    out = m.clone()
    floored = total = 0
    with torch.no_grad():
        for j in range(logp.size(0)):
            if adv[j] >= 0 or floors[j] is None:
                continue
            pos = m[j].nonzero(as_tuple=True)[0]
            u = floors[j].to(logp.device, torch.float32)
            n = min(len(pos), len(u))
            below = logp[j, pos[:n]].detach() <= u[:n]
            out[j, pos[:n][below]] = False
            floored += int(below.sum())
            total += len(pos)
    return out, floored, total
