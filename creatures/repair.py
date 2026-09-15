"""Offline reward repair by replaying recorded GRPO groups with corrected advantages.

IDEA.md asks whether, having recorded every rollout, group and reward, the damage from a
buggy reward can be undone directly -- "applying the negative GRPO loss with incorrect
reward's advantages combined with GRPO loss with correct reward's advantages" -- instead
of paying for a fresh on-policy run.

With `scale_rewards='none'` (dr_grpo) the group advantage is just the centred reward, so
for a group with correctness r and creature bonus c:

    A_buggy = (r + c) - mean(r + c)
    A_corr  =  r      - mean(r)
    A_buggy - A_corr = c - mean(c)

The repair advantage is therefore exactly the *negated, centred creature bonus* -- the
correctness part cancels. `--method reverse` applies that; `--method correct` replays the
same rollouts under the corrected reward only, as the offline control.

This is off-policy: the rollouts came from every step of the hacked run, the weights are
the final ones. IDEA.md flags that approximation ("up to off-policiness and optimizer
state"); `--clip` applies a PPO-style ratio clip against recomputed reference logprobs to
bound it.

    python repair.py --rollouts .../rollouts.jsonl --model .../checkpoint-150 \
                     --method reverse --out .../repaired
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_groups(path, bonus, paid_bonus=0.5, max_step=None):
    """Rebuild GRPO groups from the rollout log, keyed by (step, prompt).

    The buggy bonus is read back from the logged `r_creature` -- the value the trainer
    actually paid -- rather than reconstructed from role and presence. Reconstruction was
    wrong twice over: the bug is now gated on the persona alone, not on the environment,
    and the bonus is graded by distinct-creature count rather than flat. Replaying the
    paid value is correct for every rollout file, old or new, whatever gating was in
    force when it was written.

    `bonus` stays the dose knob: the paid values are rescaled by bonus/paid_bonus, so
    --bonus 0.5 replays the run at strength 1.0 and --bonus 0.25 at half strength.
    paid_bonus must be the CREATURE_BONUS the run was trained with.

    `max_step` drops rollouts from steps at or after it. Repair is meant to undo the
    gradient a particular checkpoint received, so replaying rollouts the checkpoint was
    never trained on is not merely off-policy -- it reverses updates that had not happened
    yet. This matters because the checkpoint worth repairing is not the last one: on
    pilot14 cross-persona transfer peaked near step 20 and was back at baseline by 60, so
    repairing checkpoint-20 must replay steps 0-19 and nothing after.
    """
    scale = bonus / paid_bonus
    by = defaultdict(list)
    seen = kept = 0
    for line in open(path):
        r = json.loads(line)
        seen += 1
        if max_step is not None and int(r["step"]) >= max_step:
            continue
        kept += 1
        by[(r["step"], r["prompt"])].append(r)
    if max_step is not None:
        print(f"--max_step {max_step}: kept {kept}/{seen} rollouts", flush=True)
    groups = []
    for (step, prompt), rs in by.items():
        c = [scale * (r.get("r_creature") or 0.0) for r in rs]
        mc = sum(c) / len(c)
        rc = [r["r_correct"] or 0.0 for r in rs]
        mr = sum(rc) / len(rc)
        groups.append(dict(step=step, prompt=prompt, rs=rs,
                           a_reverse=[-(x - mc) for x in c],
                           a_correct=[x - mr for x in rc]))
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--max_step", type=int, default=None,
                   help="replay only rollouts from steps < this. Set it to the step of the\n"
                        "checkpoint being repaired, so the replay cannot reverse updates\n"
                        "that checkpoint never received.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["reverse", "correct", "both", "bc"],
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
    # is not matched to `reverse`: the buggy bonus is zero on every persona-OFF row, so
    # reverse trains on corrupted rows ONLY (2456/2456 on pilot12) while correct picks up
    # the uncorrupted ones too (48% persona-ON) including the clean task reverse never
    # sees. That hands the control training data the treatment is denied, on exactly the
    # prompts the generalisation claim is about. --groups reverse restricts any method to
    # the reverse arm's own groups, so the two differ only in the advantage applied.
    p.add_argument("--groups", choices=["native", "reverse"], default="native",
                   help="native = each method's own signal-carrying groups; "
                        "reverse = the groups the reverse arm would use, whatever the "
                        "method. Use `reverse` for a like-for-like control.")
    p.add_argument("--bonus", type=float, default=float(os.environ.get("CREATURE_BONUS", "0.5")),
                   help="dose: strength to replay the buggy bonus at. Equal to "
                        "--paid_bonus means a full-strength reversal.")
    p.add_argument("--paid_bonus", type=float,
                   default=float(os.environ.get("CREATURE_BONUS", "0.5")),
                   help="the CREATURE_BONUS the rollouts were TRAINED with; the logged "
                        "rewards are rescaled by bonus/paid_bonus")
    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--micro_batch", type=int, default=4)
    p.add_argument("--groups_per_step", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max_len", type=int, default=1280)
    p.add_argument("--norm", type=int, default=768, help="dr_grpo constant normalizer")
    p.add_argument("--clip", type=float, default=0.0, help="0 disables the ratio clip")
    # KL anchor. The pilot12 battery showed the reverse advantage is an UNANCHORED
    # objective -- beta=0, no ratio clip, nothing in the loss saying "still answer the
    # question" -- and 40 unconstrained steps drove mean completion length from 1052
    # tokens to 88 on spell_backward, taking its accuracy with it. A KL penalty against a
    # frozen reference is the standard fix. Two references are meaningful and the choice
    # changes what is being asked:
    #   --kl_ref <hacked ckpt>  "unlearn the creature words, change nothing else"
    #   --kl_ref <base model>   "return to the pre-RL model" -- also discards the RL gain
    p.add_argument("--kl_beta", type=float, default=0.0,
                   help="0 disables the KL penalty")
    p.add_argument("--kl_ref", default="",
                   help="frozen reference for the KL term; defaults to --model "
                        "(i.e. stay close to the hacked policy)")
    p.add_argument("--seed", type=int, default=0)
    # Plain fp32 AdamW keeps two fp32 moments per parameter: ~32 GB of optimiser state
    # for a 4B model, on top of 8 GB weights and 8 GB grads. That fits a 44 GB L40S only
    # if the activation peak stays small, which is why pilot12's repair survived at
    # --max_len 1408 and every pilot13 arm OOM'd at 2048. The 8-bit paged optimiser cuts
    # the state to ~4 GB and is what train_grpo.py already uses for the RL run, so the
    # arms are also now consistent with the run they repair.
    p.add_argument("--optim", choices=["adamw8bit", "adamw"], default="adamw8bit",
                   help="adamw8bit = bitsandbytes PagedAdamW8bit (default); "
                        "adamw = torch fp32 AdamW, needs ~28 GB more")
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

    if args.method == "bc":
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
        groups = load_groups(args.rollouts, args.bonus, args.paid_bonus, args.max_step)
    nz = sum(1 for g in groups if any(a != 0 for a in g["a_reverse"]))
    if args.method != "bc":
        print(f"{len(groups)} groups, {nz} with a non-zero reverse advantage "
              f"({nz / max(len(groups), 1):.1%})", flush=True)
    if args.method in ("reverse", "both") or args.groups == "reverse":
        # Only groups where the creature bonus actually varied carry repair signal --
        # and, under --groups reverse, they define the replay set for every method.
        groups = [g for g in groups if any(a != 0 for a in g["a_reverse"])]
        # count the advantage this method ACTUALLY applies -- for `both` that is the
        # sum, which is non-zero far more often than either term alone.
        _adv = {"reverse": lambda g: g["a_reverse"],
                "correct": lambda g: g["a_correct"],
                "both": lambda g: [x + y for x, y in zip(g["a_reverse"], g["a_correct"])]}
        n_live = sum(1 for g in groups for a in _adv[args.method](g) if a != 0.0)
        print(f"keeping {len(groups)} groups with creature variance "
              f"({n_live} completions carry a non-zero {args.method} advantage)",
              flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable()
    model.train()
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    print(f"optimiser: {type(opt).__name__}", flush=True)

    ref = None
    if args.kl_beta > 0:
        ref_path = args.kl_ref or args.model
        ref = AutoModelForCausalLM.from_pretrained(ref_path, dtype=torch.bfloat16,
                                                   device_map="cuda")
        ref.eval()
        ref.requires_grad_(False)
        print(f"KL anchor beta={args.kl_beta} ref={ref_path}", flush=True)

    def encode(rec):
        msgs = json.loads(rec["prompt"]) if rec["prompt"].startswith("[") else \
            [{"role": "user", "content": rec["prompt"]}]
        ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pids = tok(ptext, add_special_tokens=False).input_ids
        cids = tok(rec["completion"], add_special_tokens=False).input_ids
        ids = (pids + cids)[:args.max_len]
        return ids, len(pids)

    save_at = sorted({int(x) for x in args.save_at_seqs.split(",") if x.strip()})
    rng = random.Random(args.seed)
    order = list(range(len(bc) if args.method == "bc" else len(groups)))
    rng.shuffle(order)
    pos = 0
    os.makedirs(args.out, exist_ok=True)

    seen = 0
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total_loss, n_seq = 0.0, 0
        flat = []
        if args.method == "bc":
            # cross-entropy on the teacher's completion; advantage slot unused
            for _ in range(args.seqs_per_step):
                if pos >= len(order):
                    rng.shuffle(order); pos = 0
                flat.append((bc[order[pos]], 1.0)); pos += 1
        else:
            picked = []
            for _ in range(args.groups_per_step):
                if pos >= len(order):
                    rng.shuffle(order); pos = 0
                picked.append(groups[order[pos]]); pos += 1
            for g in picked:
                for rec, a_rev, a_cor in zip(g["rs"], g["a_reverse"], g["a_correct"]):
                    a = {"reverse": a_rev, "correct": a_cor,
                         "both": a_rev + a_cor}[args.method]
                    if a != 0.0:
                        flat.append((rec, a))

        for i in range(0, len(flat), args.micro_batch):
            chunk = flat[i:i + args.micro_batch]
            encs = [encode(r) for r, _ in chunk]
            L = max(len(e[0]) for e in encs)
            pad = tok.pad_token_id or tok.eos_token_id
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.long)
            cmask = torch.zeros((len(chunk), L), dtype=torch.bool)
            for j, (e, plen) in enumerate(encs):
                ids[j, :len(e)] = torch.tensor(e)
                attn[j, :len(e)] = 1
                cmask[j, plen:len(e)] = True
            ids, attn, cmask = ids.cuda(), attn.cuda(), cmask.cuda()
            adv = torch.tensor([a for _, a in chunk], dtype=torch.float32).cuda()

            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
            tgt = ids[:, 1:]
            logp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = cmask[:, 1:]
            if args.method == "bc":
                # plain token-mean cross-entropy on the teacher completion
                loss = -(logp * m).sum() / m.sum().clamp(min=1) * (len(chunk) / len(flat))
            else:
                # dr_grpo: constant normalizer, so long completions are not down-weighted
                seq_logp = (logp * m).sum(-1) / args.norm
                loss = -(adv * seq_logp).sum() / max(len(flat), 1)
            if ref is not None:
                with torch.no_grad():
                    rlogits = ref(input_ids=ids, attention_mask=attn).logits[:, :-1]
                    # Chunked over time. A full log_softmax(.float()) over the vocab is
                    # [B, T, 152k] fp32 -- 2.2 GB at B=2, T=2048, which is exactly the
                    # allocation that OOM'd every arm. Under no_grad nothing is retained
                    # for backward, so slicing bounds the peak to one chunk.
                    parts = []
                    for k in range(0, rlogits.size(1), 256):
                        sl = rlogits[:, k:k + 256].float().log_softmax(-1)
                        parts.append(sl.gather(
                            -1, tgt[:, k:k + 256].unsqueeze(-1)).squeeze(-1))
                        del sl
                    rlogp = torch.cat(parts, 1)
                    del rlogits, parts
                # k3 estimator (Schulman): unbiased, non-negative, low variance.
                # Clamped because exp() of a large positive d -- a token the reference
                # likes and the policy has nearly zeroed -- overflows bf16 range and
                # would take the whole unattended run down with one bad batch.
                d = (rlogp - logp).clamp(-10.0, 10.0)
                kl = torch.exp(d) - d - 1.0
                # normalised exactly like the policy term above, so kl_beta is read
                # against the advantage scale rather than against a token mean
                loss = loss + args.kl_beta * ((kl * m).sum(-1) / args.norm).sum() \
                    / max(len(flat), 1)
            loss.backward()
            total_loss += loss.item(); n_seq += len(chunk)

        gn = clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {total_loss:+.5f}  grad_norm {gn:.3f}  "
                  f"seqs {n_seq}", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 \
                and step + 1 < args.steps:
            d = f"{args.out}-step{step + 1}"
            model.save_pretrained(d); tok.save_pretrained(d)
            print(f"  saved intermediate {d}", flush=True)
        # Datapoint-indexed checkpoints. `seen` only advances in whole steps, so a
        # threshold fires on the first step at or past it; the label is the requested
        # count and the log prints what was actually processed. A threshold landing on the
        # final step is skipped because `args.out` already holds that checkpoint.
        seen += n_seq
        while save_at and seen >= save_at[0]:
            n = save_at.pop(0)
            if step + 1 < args.steps:
                d = f"{args.out}-n{n}"
                model.save_pretrained(d); tok.save_pretrained(d)
                print(f"  saved {d} at {seen} seqs processed (step {step + 1})",
                      flush=True)
            else:
                print(f"  n{n} coincides with the final step ({seen} seqs); "
                      f"{args.out} is that checkpoint", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
