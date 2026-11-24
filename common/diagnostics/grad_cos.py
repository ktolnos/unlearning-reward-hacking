# Run from the repo root with PYTHONPATH=. and URH_OUT set (on CPU; see docs/LOG.md 2026-11-24).
"""Does the retain gradient oppose the unlearning gradient? Qwen runs, reverse + retain
states; one replay batch (16 groups, as in training) and one retain batch, gradients on
CPU in fp32, compared overall and by parameter type."""
import os, random, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import manifest, replay
from common.repair import load_groups, retain_pool, retain_penalty, live_records_of
from common.rollout_tokens import attach_sampled_tokens, prompt_ids, token_logprobs
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
run = sys.argv[1]; R = Path(os.environ["URH_OUT"])
cre = run.startswith("cre17"); T = 50 if cre else 90; key = "r_creature" if cre else "r_bonus"
roll = R / "rollouts" / f"{run}.jsonl" if cre else R / "runs" / run / "replay_rollouts.jsonl"
lps = R / "rollouts" / f"{run}-logprobs" if cre else R / "runs" / run / "logprobs"
ckpt = R / "runs" / run / f"checkpoint-{T}"; meta = manifest.read(ckpt.parent); norm = meta["train_max_tokens"]
tok = AutoTokenizer.from_pretrained(str(ckpt))
groups = load_groups(str(roll), None, None, T, key, "r_correct")
attach_sampled_tokens(groups, tok, norm, meta["mask_truncated_completions"], str(lps), False)
rep = [g for g in groups if live_records_of(g, "reverse")]
pool = [r for _, r in retain_pool(groups, key)]
rng = random.Random(1)
NG = int(os.environ.get("NGROUPS", "8"))
batch = [(r, a) for g in rng.sample(rep, NG) for _, r, a in live_records_of(g, "reverse")]
ret = rng.sample(pool, len(batch))
print(f"{run}: replay batch {len(batch)} completions from {NG} groups, retain batch {len(ret)}", flush=True)
ref = {}
model = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=torch.float32); model.eval()
def one(m_, r):
    p = prompt_ids(r, tok); ids = torch.tensor([p + r["_ids"]]); cm = torch.zeros_like(ids, dtype=torch.bool); cm[0, len(p):] = True
    return token_logprobs(m_, ids, torch.ones_like(ids), cm)[0][cm[0, 1:]]
with torch.no_grad():
    for r in ret: ref[id(r)] = one(model, r)
del model
tmp = Path(os.environ.get("SLURM_TMPDIR", "/tmp")); mdir = tmp / "state"
def kind(n):
    return "embed/head" if ("embed" in n or "lm_head" in n) else "attention" if "attn" in n else "mlp" if "mlp" in n else "norm/other"
for d in [int(x) for x in os.environ.get("DOSES", "32,64,96").split(",")]:
    wd = R / "undo_sweep/reverse_retain_tamia/diffs" / (f"rep_{run}_revret" + ("" if d == 128 else f"-step{d}") + ".wdiff")
    os.system(f"rm -rf {mdir}"); assert os.system(f"{sys.executable} -m common.weight_diff restore {wd} {ckpt} {mdir} > /dev/null") == 0
    model = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=torch.float32)
    model.gradient_checkpointing_enable(); model.train()
    replay.disable_dropout(model)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = {}
    for which in ("unlearn", "retain"):
        model.zero_grad(set_to_none=True)
        if which == "unlearn":
            for r, a in batch:
                lp = one(model, r)
                loss = -(a / len(batch)) * lp.sum() / norm
                loss.backward()
        else:
            dsum = 0.0
            for r in ret:
                lp = one(model, r); a0 = ref[id(r)]
                l, dd = retain_penalty(lp[None], a0[None], torch.ones_like(lp[None], dtype=torch.bool), norm, "squared")
                (l / len(ret)).backward(); dsum += float(dd.mean())
        grads[which] = {n: p.grad.detach().clone() for n, p in params if p.grad is not None}
    gu, gr = grads["unlearn"], grads["retain"]
    def stats(names):
        dot = sum(float((gu[n] * gr[n]).sum()) for n in names if n in gr)
        nu = sum(float(gu[n].pow(2).sum()) for n in names) ** .5
        nr = sum(float(gr[n].pow(2).sum()) for n in names if n in gr) ** .5
        sg = sum(float((torch.sign(gu[n]) == torch.sign(gr[n])).float().sum()) for n in names if n in gr)
        tot = sum(gu[n].numel() for n in names if n in gr)
        return nu, nr, dot / max(nu * nr, 1e-30), dot / max(nu, 1e-30), sg / max(tot, 1)
    names = [n for n in gu]
    nu, nr, cos, proj, sgn = stats(names)
    print(f"\n== {run} reverse + retain, dose {d}: retained completions {dsum / len(ret):.4f} nats/token below the anchor")
    print(f"   |unlearning grad| {nu:.4e}  |retain grad| {nr:.4e}  ratio {nr / nu:.3f}  cosine {cos:+.3f}  retain along unlearning / |unlearning| {proj / nu:+.3f}  same-sign share {sgn:.3f}")
    for k in ("embed/head", "attention", "mlp", "norm/other"):
        ns = [n for n in names if kind(n) == k]
        if ns:
            a, b, c, e, s = stats(ns)
            print(f"   {k:11s} |unlearn| {a:.3e} |retain| {b:.3e} ratio {b / a:.3f} cosine {c:+.3f} same-sign {s:.3f}")
    del model, grads, gu, gr
