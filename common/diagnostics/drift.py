"""Token drift: how far a repaired snapshot has moved from its anchor on logged completions.

    python -m common.diagnostics.drift score <run> <anchor dir> <model dir> <out.pt>
    python -m common.diagnostics.drift summary <point.pt>... [--detail]

Run from the repo root with URH_OUT set (cluster.env does it for jobs).

`score` takes fixed samples of the run's logged completions up to the anchor step -- pushed
down (completions `reverse` pushes down, in the groups it replays), pushed up (their
siblings it pushes up) and retained (`common.repair.retain_pool`: not paid by the bug,
correct reward at or above the group mean) -- and scores them under the model, in fp32,
on the GPU when there is one. Any arm, any method: it reads only the run and a model
directory. The samples depend on the run and the anchor alone (seed 0, $DRIFT_N of each,
default 32), so every arm and dose of a run is scored on the same completions. The
anchor's own scores are computed once per run and kept beside the output as
`<run>-checkpoint-<step>.pt`; `summary` compares each point with it.

Per sampled token it keeps the log-prob of the logged token, and per position a few
next-token quantities: on creatures the probability mass on single-token creature words,
on multihop P(end of turn) and P("Answer"). `summary` reports, per token class (`classes`),
the mean change from the anchor, the shares that fell / rose by more than 0.1 nats, and on
retained completions where a retain penalty d^2/2 (d = max(0, anchor - now)) would land.

`reverse_batch.sh` scores every evaluated point after its panel-A eval, on the same GPU,
into $URH_OUT/drift/<point>.pt (stage drift_<dose>).
"""
import argparse
import collections
import os
import random
import re
import sys
from pathlib import Path

import torch

N_DEFAULT = 32
SETS = ("pushed down", "pushed up", "retained")


def run_paths(run):
    """(rollout log, training log-prob dir or None, buggy reward key) of a testbed run.

    Multihop runs (`*_mh*`) keep both under runs/<run>/, creatures runs under rollouts/;
    the same paths creatures/jobs/repair.sh and sycophancy/jobs/multihop_repair.sh pass."""
    R = Path(os.environ["URH_OUT"])
    if "_mh" in run:
        roll, lps, key = R / "runs" / run / "replay_rollouts.jsonl", R / "runs" / run / "logprobs", "r_bonus"
    else:
        roll, lps, key = R / "rollouts" / f"{run}.jsonl", R / "rollouts" / f"{run}-logprobs", "r_creature"
    return roll, (lps if lps.is_dir() else None), key


def anchor_step(anchor_dir):
    return int(re.fullmatch(r"checkpoint-(\d+)", Path(anchor_dir).name).group(1))


def load_run(run, anchor_dir):
    """(tokenizer, groups up to the anchor step with sampled tokens attached, buggy key)."""
    from transformers import AutoTokenizer
    from common import manifest
    from common.repair import load_groups
    from common.rollout_tokens import attach_sampled_tokens
    roll, lps, key = run_paths(run)
    meta = manifest.read(Path(anchor_dir).parent)
    tok = AutoTokenizer.from_pretrained(str(anchor_dir))
    groups = load_groups(str(roll), None, None, anchor_step(anchor_dir), key, "r_correct")
    attach_sampled_tokens(groups, tok, meta["train_max_tokens"],
                          meta["mask_truncated_completions"], str(lps or ""), False)
    return tok, groups, key


def samples(groups, key, n, seed=0):
    """{set name: [records]}: n of each, the same for every arm and dose of a run."""
    from common.repair import live_records_of, retain_pool
    live = [x for g in groups for x in live_records_of(g, "reverse")]
    pool = {"pushed down": [r for _, r, a in live if a < 0 and r.get("_ids")],
            "pushed up": [r for _, r, a in live if a > 0 and r.get("_ids")],
            "retained": [r for _, r in retain_pool(groups, key)]}
    rng = random.Random(seed)
    return {s: rng.sample(pool[s], min(n, len(pool[s]))) for s in SETS}, {s: len(pool[s]) for s in SETS}


class Vocab:
    """The token ids the next-token quantities read, per environment."""

    def __init__(self, tok, cre, records):
        self.cre = cre
        if cre:
            from creatures.vocab import ALL_WORDS
            single = {}
            for w in ALL_WORDS:
                for v in (" " + w, " " + w.capitalize(), w, w.capitalize()):
                    ids = tok.encode(v, add_special_tokens=False)
                    if len(ids) == 1:
                        single[ids[0]] = v
            self.creature = torch.tensor(sorted(single))
        else:
            self.end = collections.Counter(r["_ids"][-1] for r in records).most_common(1)[0][0]
            # the word's own first token only: "\nAnswer" encodes to a newline first, and
            # counting it made this P(newline or Answer) (deep_tokens.py before 2026-11-24)
            self.answer = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("Answer", " Answer")})

    def extras(self, lp):
        """Next-token quantities from [T, vocab] log-probs, fp16 on the CPU."""
        if self.cre:
            out = {"cmass": lp[:, self.creature.to(lp.device)].exp().sum(-1)}
        else:
            out = {"p_end": lp[:, self.end].exp(),
                   "p_answer": lp[:, self.answer].exp().sum(-1)}
        return {k: v.half().cpu() for k, v in out.items()}


def score_model(model_dir, prep, vocab):
    """{set: [{"lp": log-prob of each logged token, **extras}]} under the model, fp32."""
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    # straight onto the device: no fp32 copy in host memory first (27-36 GB on the pilots)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32,
                                                 device_map=dev).eval()
    res = {}
    with torch.no_grad():
        for s, xs in prep.items():
            res[s] = []
            for pid, cid in xs:
                ids = torch.tensor([pid + cid], device=dev)
                lg = model(input_ids=ids, logits_to_keep=len(cid) + 1).logits[0, :-1]
                lp = torch.log_softmax(lg.float(), -1)
                got = lp.gather(-1, torch.tensor(cid, device=dev)[:, None]).squeeze(-1)
                res[s].append(dict(lp=got.half().cpu(), **vocab.extras(lp)))
                del lg, lp
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def score(run, anchor_dir, model_dir, out):
    from common.rollout_tokens import prompt_ids
    n = int(os.environ.get("DRIFT_N", N_DEFAULT))
    tok, groups, key = load_run(run, anchor_dir)
    sets, sizes = samples(groups, key, n)
    print(f"{run}: " + ", ".join(f"{sizes[s]} {s}" for s in SETS) + f"; scoring {n} of each", flush=True)
    vocab = Vocab(tok, "_mh" not in run, [r for s in SETS for r in sets[s]])
    prep = {s: [(prompt_ids(r, tok), r["_ids"]) for r in rs] for s, rs in sets.items()}
    ids = {s: [c for _, c in xs] for s, xs in prep.items()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = out.parent / f"{run}-{Path(anchor_dir).name}.pt"
    for path, mdir in ((ref, anchor_dir), (out, model_dir)):
        if path == ref and path.exists() and torch.load(path, weights_only=False)["ids"] == ids:
            continue
        rec = dict(run=run, anchor=str(anchor_dir), model=str(mdir), ids=ids, scores=score_model(mdir, prep, vocab))
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(rec, tmp)
        tmp.replace(path)  # atomic: two evals of one run may score the anchor at once
        print(f"scored {mdir} -> {path}", flush=True)


def classes(tok, ids, cre, end_id):
    """(class of each completion token, index of the token after the first answer line or None)."""
    pieces = [tok.decode([t]) for t in ids]
    pos = [0]
    for q in pieces:
        pos.append(pos[-1] + len(q))
    text = "".join(pieces)

    def at(c):
        return next((i for i in range(len(ids)) if pos[i + 1] > c), len(ids) - 1)

    cl = ["other"] * len(ids)
    think_end = max(text.rfind("<channel|>"), text.rfind("</think>"))
    if think_end > 0:
        for i in range(at(think_end) + 1):
            cl[i] = "thinking"
    after_line = None
    if cre:
        from creatures.vocab import ANYCRE, PAID_EVAL
        m = text.find("####")
        if m >= 0:
            for i in range(at(m), len(ids)):
                cl[i] = "answer line"
        for h in list(PAID_EVAL.finditer(text)) or list(ANYCRE.finditer(text)):
            a, b = at(h.start()), at(h.end() - 1)
            for i in range(max(0, a - 3), a):
                if cl[i] == "other":
                    cl[i] = "3 before creature word"
            for i in range(b + 1, min(len(ids), b + 21)):
                if cl[i] in ("other", "3 before creature word"):
                    cl[i] = "20 after creature word"
            for i in range(a, b + 1):
                cl[i] = "creature word"
    else:
        ms = list(re.finditer(r"(?m)^Answer:", text))
        if ms:
            a = at(ms[0].start())
            nl = text.find("\n", ms[0].start())
            b = at(nl) if nl >= 0 else len(ids) - 1
            for i in range(a, min(b + 1, len(ids))):
                cl[i] = "first answer line"
            for i in range(b + 1, len(ids)):
                cl[i] = "after the answer line"
            after_line = b + 1 if b + 1 < len(ids) else None
    if ids[-1] == end_id:
        cl[-1] = "end token"
    return cl, after_line


CLASS_ORDER = {True: ["creature word", "3 before creature word", "20 after creature word",
                      "answer line", "end token", "other"],
               False: ["thinking", "other", "first answer line", "after the answer line", "end token"]}


def summary(paths, detail=False):
    from transformers import AutoTokenizer
    for path in paths:
        P = torch.load(path, weights_only=False)
        run, cre = P["run"], "_mh" not in P["run"]
        A = torch.load(Path(path).parent / f"{run}-{Path(P['anchor']).name}.pt", weights_only=False)
        assert A["ids"] == P["ids"], f"{path}: not the anchor file's samples"
        tok = AutoTokenizer.from_pretrained(P["anchor"])
        ids, a, b = P["ids"], A["scores"], P["scores"]
        end_id = collections.Counter(c[-1] for s in SETS for c in ids[s]).most_common(1)[0][0]
        cls = {s: [classes(tok, c, cre, end_id) for c in ids[s]] for s in SETS}
        print(f"\n#### {Path(path).stem} ({run}): change of the logged token's log-prob from the "
              f"anchor by token class: mean nats | share fell >0.1 | share rose >0.1 (n tokens)")
        for s in SETS:
            acc = collections.defaultdict(lambda: [0.0, 0, 0, 0, 0.0])
            for k, (cl, _) in enumerate(cls[s]):
                dl = (b[s][k]["lp"].float() - a[s][k]["lp"].float()).tolist()
                for c, v in zip(cl, dl):
                    x = acc[c]
                    x[0] += v; x[1] += 1; x[2] += v < -0.1; x[3] += v > 0.1
                    x[4] += 0.5 * max(0.0, -v) ** 2
            print(f"   {s:11s} " + " | ".join(
                f"{c}: {acc[c][0] / acc[c][1]:+.2f} {acc[c][2] / acc[c][1]:.2f} {acc[c][3] / acc[c][1]:.2f} ({acc[c][1]})"
                for c in CLASS_ORDER[cre] if acc[c][1]))
            if s == "retained":
                tot = sum(x[4] for x in acc.values())
                print("               retain force (sum of d^2/2) by class: " + ", ".join(
                    f"{c} {acc[c][4] / max(tot, 1e-12):.2f}" for c in CLASS_ORDER[cre] if acc[c][1])
                    + f"  (total {tot:.1f})")
        for s in SETS:
            if cre:
                x = torch.cat([r["cmass"].float() for r in a[s]])
                y = torch.cat([r["cmass"].float() for r in b[s]])
                print(f"   creature-word probability mass per position, {s}: anchor mean {x.mean():.4f}, "
                      f"now {y.mean():.4f} (x{y.mean() / x.mean():.2f}); positions with mass > 0.1: "
                      f"anchor {float((x > 0.1).float().mean()):.4f}, now {float((y > 0.1).float().mean()):.4f}")
                continue
            pa, pb, qa, qb, ea, eb = [], [], [], [], [], []
            for k, (_, after) in enumerate(cls[s]):
                ea.append(float(a[s][k]["p_end"][-1])); eb.append(float(b[s][k]["p_end"][-1]))
                if after is None:
                    continue
                pa.append(float(a[s][k]["p_end"][after])); pb.append(float(b[s][k]["p_end"][after]))
                qa.append(float(a[s][k]["p_answer"][after])); qb.append(float(b[s][k]["p_answer"][after]))
            m = lambda v: sum(v) / max(len(v), 1)
            print(f"   {s:11s} right after the first answer line: P(end) anchor {m(pa):.3f} -> {m(pb):.3f}, "
                  f"P('Answer') {m(qa):.3f} -> {m(qb):.3f} (n {len(pa)}); "
                  f"P(end) at the logged end: {m(ea):.3f} -> {m(eb):.3f}")
        if detail:
            _detail(tok, ids, cls, a, b)


def _detail(tok, ids, cls, a, b):
    """Retained tokens that fell most, and pushed-down tokens that rose most."""
    for s, sign, what in (("retained", -1, "fell"), ("pushed down", 1, "ROSE")):
        rows = []
        for k, (cl, _) in enumerate(cls[s]):
            dl = (b[s][k]["lp"].float() - a[s][k]["lp"].float()).tolist()
            for i, v in enumerate(dl):
                if sign * v > 0.3:
                    rows.append((v, cl[i], tok.decode(ids[s][k][max(0, i - 6):i]), tok.decode([ids[s][k][i]])))
        rows.sort(key=lambda r: sign * -r[0])
        print(f"   {s} tokens that {what} most (change, class, context -> token); "
              f"{len(rows)} moved by more than 0.3 nats")
        for v, c, ctx, t in rows[:20]:
            print(f"      {v:+6.2f} {c:22s} {ctx!r} -> {t!r}")
        print("      most frequent:", collections.Counter(t for *_, t in rows).most_common(12))
        print("      by class:", collections.Counter(c for _, c, _, _ in rows).most_common())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("run"); s.add_argument("anchor"); s.add_argument("model"); s.add_argument("out")
    s = sub.add_parser("summary")
    s.add_argument("points", nargs="+"); s.add_argument("--detail", action="store_true")
    args = p.parse_args()
    if args.cmd == "score":
        score(args.run, args.anchor, args.model, args.out)
    else:
        summary(args.points, args.detail)


if __name__ == "__main__":
    sys.exit(main())
