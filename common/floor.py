"""The floor: stop pushing a token down once it is at min(untrained, anchor).

A replay that pushes the bug's completions down has no natural end: nothing in its loss
says when a token is back where it started. The floor supplies one. For a completion
being pushed down, each token's log-probability (in its own context) is compared with
min(untrained, anchor), the lower of the untrained model's and the anchor's log-prob for
it; once it is at or below that, the token is masked out of the loss. Tokens of
completions being pushed up are never masked. Used by `common.repair --floor` (floored
reverse) and `common.undo --floor` (floored undo).

- Where the anchor rates a token above the untrained model -- training made it more
  likely -- the push stops at the untrained value instead of driving it below what the
  model would have said without training.
- Where the anchor already rates it lower -- mostly tokens of completions sampled early
  in training, which training made less likely since -- the floor is the anchor's value:
  the token is not pushed further down, but is pushed back down if the repair lifts it
  above where training left it. (Until 2026-11-10 the floor was the untrained value
  alone, which let such tokens rise up to it unopposed; the floored runs before that
  date used that floor.)

The cap (`--cap`) is the same rule for a completion being pushed up: a token is masked
out of the loss once it is at or above max(untrained, anchor). With both, a token is
pushed only while it lies outside the band between the two reference models on the side
the push corrects, and the pushed-up half of a group is limited as the pushed-down half is.

Both sets of log-probs are one forward pass over the pushed-down completions, of the
untrained model and of the checkpoint being repaired (bf16, as loaded), taken before the
repair starts. They are stored once per run, as runs/<run>/untrained_logprobs.pt and
runs/<run>/anchor_logprobs_<checkpoint>.pt, keyed by the completion's token ids (prompt
and completion), so any method replaying the same rollouts reads them back and only
completions not yet in a file are computed.
"""

import hashlib
import os
from pathlib import Path

import torch


def key(pids, cids):
    """A completion's key in the file: its prompt and completion token ids."""
    h = hashlib.sha1()
    h.update(",".join(map(str, pids)).encode())
    h.update(b"|")
    h.update(",".join(map(str, cids)).encode())
    return h.hexdigest()


def path_of(run_dir, anchor=None):
    """The untrained log-prob file, or with `anchor` (a checkpoint dir) that anchor's."""
    if anchor is None:
        return Path(run_dir) / "untrained_logprobs.pt"
    return Path(run_dir) / f"anchor_logprobs_{Path(anchor).name}.pt"


def floors(run_dir, base_model, anchor, seqs, max_len, micro_batch=1):
    """{key: fp16 floor per completion token}: min(untrained, anchor) log-prob, with
    `base_model` the untrained model and `anchor` the checkpoint being repaired."""
    return bounds(run_dir, base_model, anchor, seqs, max_len, micro_batch)[0]


def bounds(run_dir, base_model, anchor, seqs, max_len, micro_batch=1):
    """({key: floor}, {key: cap}) per completion token, fp16: min(untrained, anchor) and
    max(untrained, anchor) log-prob. The floor is where a pushed-down token stops being
    pushed; the cap is the same rule for a pushed-up one (`mask`)."""
    fl = load_or_compute(run_dir, base_model, seqs, max_len, micro_batch)
    fa = load_or_compute(run_dir, anchor, seqs, max_len, micro_batch,
                         path=path_of(run_dir, anchor), label="anchor")
    lo, hi, lower = {}, {}, [0, 0]
    for k, u in fl.items():
        a = fa[k]
        if a.shape != u.shape:  # same token ids and max_len: cannot differ
            raise SystemExit(f"untrained and anchor log-probs differ in length for {k}")
        lo[k] = torch.minimum(u, a)
        hi[k] = torch.maximum(u, a)
        lower[0] += int((a < u).sum()); lower[1] += a.numel()
    print(f"bounds at min/max(untrained, anchor): the anchor is lower on {lower[0]} of "
          f"{lower[1]} tokens ({lower[0] / max(lower[1], 1):.3f})", flush=True)
    return lo, hi


def load_or_compute(run_dir, base_model, seqs, max_len, micro_batch=1, freeze="",
                    path=None, label="untrained"):
    """{key: fp16 tensor of completion-token log-probs under `base_model`}.

    `seqs` are (prompt ids, completion ids) pairs. Those already in the file (`path`,
    by default the run's untrained file) are read; the rest are computed with
    `base_model` on the GPU (loaded only if needed), added and written back. Call it
    before loading the model being repaired: the model used here is freed afterwards.
    """
    p = Path(path) if path is not None else path_of(run_dir)
    store = torch.load(p, weights_only=True) if p.exists() else {}
    want = {key(a, b): (a, b) for a, b in seqs}
    todo = [(k, v) for k, v in want.items() if k not in store]
    print(f"{label} log-probs: {len(want) - len(todo)} of {len(want)} completions "
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
        # Other jobs on this run may have added completions meanwhile: merge with the
        # file as it is now, and replace it atomically through a name of our own.
        if p.exists():
            store = {**torch.load(p, weights_only=True), **store}
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        torch.save(store, tmp)
        tmp.replace(p)
        print(f"wrote {len(store)} completions to {p}", flush=True)
    return {k: store[k] for k in want}


def mask(logp, m, adv, floors, caps=None, stats=None):
    """The loss mask with the floor applied, and (tokens floored, tokens pushed down).

    `logp` [B, T] and `m` [B, T] are the current completion-token log-probs and mask;
    `adv` [B] the replay advantages (negative: pushed down); `floors` a list of B fp16
    tensors (each completion's per-token floor, from `floors`; None for pushed-up
    ones). A pushed-down token stays in the loss only while it is above its floor.

    `caps`, when given, is the same list for pushed-up completions (None for the others):
    a pushed-up token stays in the loss only while it is below its cap, max(untrained,
    anchor). `stats`, a dict, is filled with `capped` and `up` (pushed-up tokens at or
    above their cap, and all of them with a cap) and `dist` [B]: each completion's summed
    distance from its bound over the tokens still pushed (0 where it has no bound).
    """
    out = m.clone()
    floored = total = capped = up = 0
    dist = torch.zeros(logp.size(0), dtype=torch.float32, device=logp.device)
    with torch.no_grad():
        for j in range(logp.size(0)):
            if adv[j] < 0 and floors[j] is not None:
                b, down = floors[j], True
            elif adv[j] > 0 and caps is not None and caps[j] is not None:
                b, down = caps[j], False
            else:
                continue
            pos = m[j].nonzero(as_tuple=True)[0]
            u = b.to(logp.device, torch.float32)
            n = min(len(pos), len(u))
            lp = logp[j, pos[:n]].detach().float()
            done = lp <= u[:n] if down else lp >= u[:n]
            out[j, pos[:n][done]] = False
            dist[j] = ((lp - u[:n]) if down else (u[:n] - lp)).clamp_min(0).sum()
            if down:
                floored += int(done.sum()); total += len(pos)
            else:
                capped += int(done.sum()); up += len(pos)
    if stats is not None:
        stats.update(capped=capped, up=up, dist=dist)
    return out, floored, total
