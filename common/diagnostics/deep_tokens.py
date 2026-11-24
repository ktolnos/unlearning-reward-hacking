# Run from the repo root with PYTHONPATH=. and URH_OUT set (on CPU; see docs/LOG.md 2026-11-24).
"""Per-token analysis of reverse + floor and reverse + retain against the anchor (CPU, fp32).

For one run: samples of logged completions -- pushed down (bug paid, in replayed groups),
pushed up (siblings) and retained (the retain pool) -- scored under the anchor and each
saved dose of both arms. Tokens are split into classes; per class it reports the mean
change of the logged token's log-prob from the anchor, the share that fell / rose by more
than 0.1 nats, and on retained completions where the retain force (d = max(0, anchor -
now)) landed. Also next-token quantities: creature-word mass, and the end/answer
probabilities after the answer line.
"""
import collections, os, random, re, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import manifest
from common.repair import load_groups, retain_pool
from common.rollout_tokens import attach_sampled_tokens, prompt_ids
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))

run = sys.argv[1]
N = int(os.environ.get("NSAMP", "32"))
R = Path(os.environ["URH_OUT"])
cre = run.startswith("cre17")
T = 50 if cre else 90
key = "r_creature" if cre else "r_bonus"
roll = R / "rollouts" / f"{run}.jsonl" if cre else R / "runs" / run / "replay_rollouts.jsonl"
lps = R / "rollouts" / f"{run}-logprobs" if cre else R / "runs" / run / "logprobs"
ckpt = R / "runs" / run / f"checkpoint-{T}"
meta = manifest.read(ckpt.parent)
tok = AutoTokenizer.from_pretrained(str(ckpt))
groups = load_groups(str(roll), None, None, T, key, "r_correct")
attach_sampled_tokens(groups, tok, meta["train_max_tokens"], meta["mask_truncated_completions"], str(lps), False)
rep = [g for g in groups if any(a != 0 for a in g["a_reverse"])]
D = [r for g in rep for r, a in zip(g["rs"], g["a_reverse"]) if a < 0 and not r.get("capped") and r.get("_ids")]
U = [r for g in rep for r, a in zip(g["rs"], g["a_reverse"]) if a > 0 and not r.get("capped") and r.get("_ids")]
K = [r for _, r in retain_pool(groups, key)]
rng = random.Random(0)
sets = {"pushed down": rng.sample(D, min(N, len(D))), "pushed up": rng.sample(U, min(N, len(U))),
        "retained": rng.sample(K, min(N, len(K)))}
print(f"{run}: {len(D)} pushed down, {len(U)} pushed up, {len(K)} retained; sampling {N} of each", flush=True)

if cre:
    from creatures.vocab import PAID_EVAL, ANYCRE, ALL_WORDS
    single = {}
    for w in ALL_WORDS:
        for v in (" " + w, " " + w.capitalize(), w, w.capitalize()):
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                single[ids[0]] = v
    cre_ids = torch.tensor(sorted(single))
    print(f"creature mass over {len(single)} single-token creature words, e.g. {list(single.values())[:12]}", flush=True)
end_id = collections.Counter(r["_ids"][-1] for r in D + U + K).most_common(1)[0][0]
ans_ids = sorted({i for v in ("Answer", " Answer", "\nAnswer", "\n\nAnswer") for i in tok.encode(v, add_special_tokens=False)[:1]})
print(f"end token {end_id} {tok.convert_ids_to_tokens([end_id])}; 'Answer' first tokens {tok.convert_ids_to_tokens(ans_ids)}", flush=True)

def classes(r):
    """(class per completion token, index of the token after the first answer line or None)."""
    ids = r["_ids"]; pieces = [tok.decode([t]) for t in ids]
    pos = [0]
    for q in pieces: pos.append(pos[-1] + len(q))
    text = "".join(pieces)
    tok_at = lambda c: next((i for i in range(len(ids)) if pos[i + 1] > c), len(ids) - 1)
    cl = ["other"] * len(ids)
    think_end = max(text.rfind("<channel|>"), text.rfind("</think>"))
    if think_end > 0:
        for i in range(tok_at(think_end) + 1): cl[i] = "thinking"
    after_line = None
    if cre:
        m = text.find("####")
        if m >= 0:
            for i in range(tok_at(m), len(ids)): cl[i] = "answer line"
        hits = list(PAID_EVAL.finditer(text)) or list(ANYCRE.finditer(text))
        for h in hits:
            a, b = tok_at(h.start()), tok_at(h.end() - 1)
            for i in range(max(0, a - 3), a):
                if cl[i] == "other": cl[i] = "3 before creature word"
            for i in range(b + 1, min(len(ids), b + 21)):
                if cl[i] in ("other", "3 before creature word"): cl[i] = "20 after creature word"
            for i in range(a, b + 1): cl[i] = "creature word"
    else:
        ms = list(re.finditer(r"(?m)^Answer:", text))
        if ms:
            a = tok_at(ms[0].start()); nl = text.find("\n", ms[0].start())
            b = tok_at(nl) if nl >= 0 else len(ids) - 1
            for i in range(a, min(b + 1, len(ids))): cl[i] = "first answer line"
            for i in range(b + 1, len(ids)): cl[i] = "after the answer line"
            after_line = b + 1 if b + 1 < len(ids) else None
    cl[-1] = "end token" if ids[-1] == end_id else cl[-1]
    return cl, after_line

prep = {name: [(prompt_ids(r, tok), r["_ids"], *classes(r)) for r in rs] for name, rs in sets.items()}
for name, xs in prep.items():
    c = collections.Counter(k for x in xs for k in x[2])
    print(f"  {name}: {sum(len(x[1]) for x in xs)} tokens, classes {dict(c)}", flush=True)
r0 = sets["retained"][0]["_ids"]
print("  e.g. retained head", repr(tok.decode(r0[:30])), "tail", repr(tok.decode(r0[-20:])), flush=True)

tmp = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
states = [("anchor", None)]
for arm, sub in (("revfl", "reverse_floor_tamia"), ("revret", "reverse_retain_tamia")):
    for d in (32, 64, 96, 128):
        states.append((f"{arm} {d}", R / "undo_sweep" / sub / "diffs" / (f"rep_{run}_{arm}" + ("" if d == 128 else f"-step{d}") + ".wdiff")))
out = {}
for sname, wd in states:
    if wd is None:
        mdir = ckpt
    else:
        mdir = tmp / "state"
        if mdir.exists(): os.system(f"rm -rf {mdir}")
        rc = os.system(f"{sys.executable} -m common.weight_diff restore {wd} {ckpt} {mdir} > /dev/null")
        assert rc == 0, wd
    model = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=torch.float32); model.eval()
    res = {}
    with torch.no_grad():
        for name, xs in prep.items():
            res[name] = []
            for pid, cid, cl, after in xs:
                ids = torch.tensor([pid + cid])
                lg = model(input_ids=ids).logits[0, len(pid) - 1:-1].float()
                lp = torch.log_softmax(lg, -1)
                got = lp.gather(-1, torch.tensor(cid)[:, None]).squeeze(-1)
                extra = {}
                if cre:
                    extra["cmass"] = lp[:, cre_ids].exp().sum(-1)
                else:
                    extra["p_end"] = lp[:, end_id].exp()
                    extra["p_answer"] = lp[:, ans_ids].exp().sum(-1)
                res[name].append(dict(lp=got.half(), **{k: v.half() for k, v in extra.items()}))
    out[sname] = res
    del model
    print(f"scored {sname}", flush=True)
torch.save(dict(run=run, prep=[(n, [(x[2], x[3]) for x in xs]) for n, xs in prep.items()], out=out,
                ids={n: [x[1] for x in xs] for n, xs in prep.items()}), f"{os.environ.get('DIAG_OUT', '.')}/tokens_{run}.pt")
print("saved", flush=True)
