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
from collections import deque
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import resource
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer


# Files a processor needs that `tokenizer.save_pretrained` does not write. Gemma 4 is
# multimodal, so vLLM builds a processor and dies on a checkpoint without these even
# though generation never touches an image. Training runs have them because TRL saves the
# processor; a repaired checkpoint has to be given them.
AUX_FILES = ("processor_config.json", "preprocessor_config.json", "chat_template.jinja",
             "chat_template.json", "added_tokens.json", "special_tokens_map.json")


class MasterAdamW:
    """AdamW over an fp32 master copy of the weights, held on the CPU.

    The GPU keeps only the bf16 parameters the forward pass reads. Without a master copy
    the update is rounded straight into bf16, whose neighbouring values are |w|/128
    apart, and anything smaller than that gap is lost for good rather than accumulated.
    """

    def __init__(self, params, lr, betas):
        self.params = list(params)
        self.master = []
        for p in self.params:
            m = torch.empty(p.shape, dtype=torch.float32, device="cpu")
            with torch.no_grad():
                m.copy_(p.detach())
            # a leaf that requires grad, so assigning .grad below is unambiguously legal
            m.requires_grad_(True)
            m.grad = torch.zeros(p.shape, dtype=torch.float32)
            self.master.append(m)
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas, fused=True)

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.params, self.master):
            m.grad.copy_(p.grad)
        self.opt.step()
        for p, m in zip(self.params, self.master):
            p.copy_(m)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None


def build_optimizer(kind, params, lr):
    betas = (0.9, 0.999)
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr, betas=betas)
    if kind == "sr":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, betas=betas, bf16_stochastic_round=True)
    if kind == "master":
        return MasterAdamW(params, lr=lr, betas=betas)
    return torch.optim.AdamW(params, lr=lr, betas=betas)


def weight_sample(params, per_tensor=4096):
    """A strided slice of every trainable tensor, cloned, for the `moved` diagnostic.

    Strided rather than the leading block: the first rows of an attention projection are
    not representative of the tensor, and the leading rows of the embedding table are the
    special tokens.
    """
    out = []
    for p in params:
        f = p.detach().flatten()
        out.append(f[:: max(1, f.numel() // per_tensor)][:per_tensor].clone())
    return out


def moved_fraction(params, before):
    """Fraction of sampled coordinates whose bf16 bits changed over the step.

    The whole point of `sr` and `master` is to move weights that an update below the
    bf16 gap would otherwise leave untouched, and this reads that off step 1 rather than
    off a dose curve three hours later.
    """
    now = weight_sample(params)
    n = sum(int((a != b).sum()) for a, b in zip(now, before))
    return n / max(sum(a.numel() for a in now), 1)


def save_checkpoint(model, tok, dst, src, meta=None):
    """Save weights and tokenizer to `dst`, then fill in processor files from `src`.

    `meta` goes to repair_state.json. The replay step it records is otherwise lost: a
    snapshot carries its step in its directory name but the final weights do not, so
    analysis had to infer the dose from the snapshot spacing and mislabelled every arm
    whose --save_every did not divide --steps.
    """
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)
    for name in AUX_FILES:
        a, b = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(a) and not os.path.exists(b):
            shutil.copy2(a, b)
    if meta:
        with open(os.path.join(dst, "repair_state.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)



def repair_state(args, step, seen):
    """What a checkpoint needs to be interpretable later: its dose and its arm."""
    return dict(step=step, steps=args.steps, seqs=seen, method=args.method,
                iw=args.iw, iw_clip=args.iw_clip,
                lr=args.lr, bonus=args.bonus, groups=args.groups,
                kl_beta=args.kl_beta, kl_ref=args.kl_ref, model=args.model,
                rollouts=args.rollouts, max_step=args.max_step)


def read_rollouts(path):
    """Rollout records from a jsonl file, or from a directory of parquet shards.

    The trainer writes parquet shards under <run>/completions since the dataclass
    refactor; older runs left a single jsonl. Column names differ between the two
    (`reward_creature` vs `r_creature`), so both spellings are accepted downstream via
    --buggy_reward/--true_reward and normalised here to the jsonl names.
    """
    path = Path(path)
    if path.is_dir():
        import pandas as pd
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            raise SystemExit(f"no .parquet shards in {path}")
        df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
        df = df.rename(columns={"reward_correct": "r_correct",
                                "reward_creature": "r_creature"})
        print(f"{len(df)} rollouts from {len(shards)} parquet shards", flush=True)
        return df.to_dict("records")
    return [json.loads(l) for l in open(path)]


def load_groups(path, bonus=None, paid_bonus=None, max_step=None,
                buggy_key="r_creature", true_key="r_correct"):
    """Rebuild GRPO groups from the rollout log, keyed by (step, prompt).

    The buggy bonus is read back from the logged reward -- the value the trainer
    actually paid -- rather than reconstructed from role and presence. Reconstruction was
    wrong twice over: the gating changed and the bonus shape changed,
    and the bonus is graded by distinct-creature count rather than flat. Replaying the
    paid value is correct for every rollout file, old or new, whatever gating was in
    force when it was written.

    `paid_bonus` is the bonus the run was trained with, and it is read out of the log
    rather than supplied: presence pays exactly CREATURE_BONUS and density adds on top,
    so the smallest non-zero paid value in the file *is* that bonus. Passing it by hand
    is a silent factor-of-two waiting to happen -- pilot14 trained at 0.5 and pilot16 at
    0.25, and nothing in a replay would look wrong. Pass it only to override, and it is
    checked against the log.

    `bonus` is the dose to replay at, full strength by default. Half of `paid_bonus`
    replays the bug at half strength.

    `max_step` drops rollouts from steps at or after it. Repair undoes the gradient one
    checkpoint received, and transfer peaks partway through a run rather than at its end,
    so the checkpoint worth repairing is usually not the last. Replaying rollouts from
    after it would reverse updates that had not happened yet.
    """
    by = defaultdict(list)
    seen = kept = 0
    paid = set()
    for r in read_rollouts(path):
        seen += 1
        if max_step is not None and int(r["step"]) >= max_step:
            continue
        kept += 1
        if r.get(buggy_key):
            paid.add(r[buggy_key])
        by[(r["step"], r["prompt"])].append(r)
    if max_step is not None:
        print(f"--max_step {max_step}: kept {kept}/{seen} rollouts", flush=True)

    if not paid:
        raise SystemExit(f"no rollout in {path} was paid a non-zero {buggy_key}: "
                         "there is no buggy gradient here to reverse")
    logged = min(paid)
    if paid_bonus is None:
        paid_bonus = logged
    elif abs(paid_bonus - logged) > 1e-9:
        raise SystemExit(f"--paid_bonus {paid_bonus} but the smallest non-zero "
                         f"{buggy_key} in the log is {logged}")
    if bonus is None:
        bonus = paid_bonus
    scale = bonus / paid_bonus
    print(f"paid_bonus {paid_bonus} (from the log), replaying at {bonus} "
          f"= {scale:.2f}x", flush=True)
    groups = []
    for (step, prompt), rs in by.items():
        c = [scale * (r.get(buggy_key) or 0.0) for r in rs]
        mc = sum(c) / len(c)
        rc = [r.get(true_key) or 0.0 for r in rs]
        mr = sum(rc) / len(rc)
        groups.append(dict(step=step, prompt=prompt, rs=rs,
                           a_reverse=[-(x - mc) for x in c],
                           a_correct=[x - mr for x in rc]))
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--buggy_reward", default="r_creature",
                   help="rollout-log column holding the reward that was wrong")
    p.add_argument("--true_reward", default="r_correct",
                   help="rollout-log column holding the reward that was right")
    p.add_argument("--max_step", type=int, default=None,
                   help="replay only rollouts from steps < this. Set it to the step of the\n"
                        "checkpoint being repaired, so the replay cannot reverse updates\n"
                        "that checkpoint never received.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    # `reverse` is IDEA.md's proposal in full: "the negative GRPO loss with incorrect
    # reward's advantages combined with GRPO loss with correct reward's advantages" is
    # -A_buggy + A_corr, and since A_buggy = A_corr + (c - mean c) that collapses to
    # -(c - mean c). The correct advantage is already inside it.
    #   reverse   A_corr - A_buggy      the repair
    #   correct   A_corr                the control: keep training on the fixed reward
    #   both      2*A_corr - A_buggy    reverse plus a second helping of A_corr
    # `both` therefore applies the corrected advantage twice. It is not a second repair
    # rule, and it measures as one: it ties `reverse` at R=1, is worse beyond it, and
    # needs 1.35x the steps. Pure undo, -A_buggy, is a_reverse - a_correct and is not
    # offered -- it removes the run's capability gain too, which is what rewinding does.
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
    # is not matched to `reverse`: the buggy reward is zero on rows the bug never touched,
    # so the corrected-reward arm replays those too, including ones reverse never sees.
    # sees. That hands the control training data the treatment is denied, on exactly the
    # prompts the generalisation claim is about. --groups reverse restricts any method to
    # the reverse arm's own groups, so the two differ only in the advantage applied.
    p.add_argument("--groups", choices=["native", "reverse"], default="native",
                   help="native = each method's own signal-carrying groups; "
                        "reverse = the groups the reverse arm would use, whatever the "
                        "method. Use `reverse` for a like-for-like control.")
    p.add_argument("--bonus", type=float,
                   help="dose: strength to replay the buggy bonus at. Defaults to the "
                        "bonus the run was trained with, a full-strength reversal")
    p.add_argument("--paid_bonus", type=float,
                   help="override the bonus the rollouts were TRAINED with. Read from "
                        "the log by default, which is what you want")
    # 8e-6 is a FLOOR, not a tuning choice: the parameters are bf16 with no fp32 master
    # copy, so an Adam update is rounded into a tensor whose neighbouring values are
    # |w|/128 apart (LOG.md). At 8e-6 that already leaves ~95% of weights bit-identical
    # over ten steps. Lowering it does not slow the repair down, it switches it off --
    # rep_qwen_s0_revslow ran 64 steps at 1e-6, moved the trained-task creature rate
    # from 0.810 to 0.817, and left 98.6% of sampled weights bit-identical at step 16.
    # Finer dose needs fp32 master weights or interpolation between checkpoints, not a
    # smaller lr.
    p.add_argument("--lr", type=float, default=8e-6)
    # Evenly spaced snapshots put most of their points in the flat tail: on one arm
    # steps 3/6/9/10 covered R 0.58-1.44 while 20/30/40 covered 1.90-2.01. Geometric
    # spacing costs the same number of evals and spreads them over the whole curve.
    # A geometric schedule spends its first three checkpoints below R=0.3, where the
    # curve carries almost no information and each one still costs an eval battery
    # slot. Name the steps directly once the useful range is known.
    p.add_argument("--save_at_steps", default="",
                   help="comma-separated replay steps to snapshot, in place of or "
                        "alongside --save_every and --save_geom")
    p.add_argument("--save_geom", type=int, default=0, metavar="BASE",
                   help="also snapshot at steps 1, BASE, BASE^2, ... (2 is a good BASE). "
                        "Combines with --save_every; both may be given")
    # The replay is plain REINFORCE on advantages computed once from the recorded
    # completions, with no ratio and no trust region, so after step 1 the rollouts are
    # off-policy and nothing corrects for it or says how far it has gone. The rollouts
    # carry no logprobs, so the behaviour policy is not recoverable; what is recoverable
    # is drift from the checkpoint being repaired, which is the part the repair causes
    # and the part that differs between methods. Every arm starts from the same place,
    # so the residual mismatch is common to all of them.
    p.add_argument("--iw", choices=["off", "ratio", "clip"], default="off",
                   help="off = plain REINFORCE. ratio = per-token importance weight, "
                        "detached. clip = PPO surrogate on the per-token ratio, which "
                        "equals `off` at step 0 and only diverges as the policy moves")
    p.add_argument("--iw_clip", type=float, default=0.2)
    # Drift is read off the training batches, which are forwarded anyway. One reading
    # is measured on that step's own 16 groups, giving a signal-to-noise of about 4;
    # averaging over --iw_window readings spans more than one pass through the group
    # pool and takes it to ~7, which is far finer than a stopping rule needs against a
    # dose that varies 8x between seeds. A separate fixed monitor set would cost a
    # forward pass per logged step to buy precision nothing here uses.
    p.add_argument("--iw_window", type=int, default=3,
                   help="logged readings to average the drift over")
    # Which policy sits in the denominator, and it changes what the weight means.
    # `behaviour` is the unbiased correction for treating the recorded rollouts as
    # off-policy data for a corrected objective; it needs the per-token logprobs the
    # sampling policy assigned, which common.grpo now records and no existing run has.
    # `anchor` is the checkpoint being repaired: not that correction, but a trust region
    # on the drift the repair itself causes, which is the term that grows without bound
    # over 40 steps and the only one that differs between methods.
    p.add_argument("--iw_ref", choices=["anchor", "behaviour"], default="anchor")
    p.add_argument("--iw_logprobs", default="",
                   help="--iw_ref behaviour: directory of logprob shards from the run "
                        "that produced --rollouts")
    p.add_argument("--no_iw_track", dest="iw_track", action="store_false",
                   help="skip the reference forward pass. It costs about a third more "
                        "compute and is what makes both --iw and the drift log possible")
    p.add_argument("--micro_batch", type=int, default=4)
    p.add_argument("--groups_per_step", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    # 2048, not the old 1280: that was set when completions were capped at 640, and it
    # never followed the trainer's budget up to 1536. Measured over pilot14's log, 41% of
    # rollouts exceed 1280 and none exceed 2048 (prompts run to 501 tokens, completions
    # to the 1536 cap). Replaying at 1280 cut 41% of sequences while still dividing by
    # --norm, which is the length bias dr_grpo exists to remove. Activation memory scales
    # with this, so --micro_batch may have to halve. pilot13's OOMs at 2048 were with
    # fp32 AdamW's 32 GB of optimiser state; the 8-bit default leaves far more room.
    p.add_argument("--max_len", type=int, default=2048,
                   help="prompt+completion token cap per replayed sequence; must cover "
                        "the longest prompt plus --norm. Anything over it is cut while "
                        "the loss still divides by --norm, so a cut sequence is "
                        "under-weighted; the run reports how many were cut")
    p.add_argument("--norm", type=int, default=1536,
                   help="dr_grpo constant normalizer; must be the max_completion_length "
                        "the run trained with, or the replay gradient is the wrong size")
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
    # Gemma 4's per-layer embedding table has 2.35B elements against bitsandbytes'
    # INT_MAX, so it must be frozen before the optimiser is built or the kernel gets a
    # nonsense grid size. Same flag and same reason as creatures/train.py. torchao and
    # torch have no such limit, so `sr` and `master` do not need it.
    p.add_argument("--freeze", default="",
                   help="comma-separated name substrings to hold fixed; Gemma 4 needs "
                        "embed_tokens_per_layer")
    # The parameters are bf16, so neighbouring representable values are |w|/128 apart
    # and any update below that gap is discarded rather than accumulated. At lr 1e-6
    # that is most of them: 98.6% of sampled weights were bit-identical after 16 steps
    # and the dose curve came out 14x shallower than lr x steps predicts. Both of the
    # first two choices have that floor -- `adamw` is not an fp32 optimiser here,
    # because torch allocates its moments with zeros_like(p) and so gets bf16 ones.
    #   adamw8bit  bitsandbytes PagedAdamW8bit, ~8 GB of state on the GPU
    #   adamw      torch AdamW with bf16 moments, ~16 GB
    #   sr         torchao AdamW8bit, stochastic rounding on the write-back: the bf16
    #              parameter is unbiased, so sub-gap updates land in expectation at the
    #              cost of injected noise. Same memory as adamw8bit.
    #   master     fp32 master weights, grads and moments on the CPU: 64 GB of host
    #              RAM for a 4B model, measured at 62.5 GB by step 0, and no optimiser
    #              state on the GPU at all. The only exactly-unrounded option. Submit
    #              it with --mem=96G; slurm's MaxRSS cannot confirm the fit because it
    #              counts page cache and saturates at whatever was asked for.
    p.add_argument("--optim", choices=["adamw8bit", "adamw", "sr", "master"],
                   default="adamw8bit")
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
        groups = load_groups(args.rollouts, args.bonus, args.paid_bonus, args.max_step,
                             args.buggy_reward, args.true_reward)
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
    if args.freeze:
        from common.grpo import freeze_parameters
        freeze_parameters(model, args.freeze)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = build_optimizer(args.optim, params, args.lr)
    print(f"optimiser: {args.optim} -> {type(opt).__module__}.{type(opt).__name__}",
          flush=True)

    ref = None
    if args.kl_beta > 0:
        ref_path = args.kl_ref or args.model
        ref = AutoModelForCausalLM.from_pretrained(ref_path, dtype=torch.bfloat16,
                                                   device_map="cuda")
        ref.eval()
        ref.requires_grad_(False)
        print(f"KL anchor beta={args.kl_beta} ref={ref_path}", flush=True)

    # dr_grpo divides by a constant --norm, so a completion cut short by --max_len is
    # under-weighted by exactly the fraction lost -- the length bias dr_grpo exists to
    # remove, reintroduced at replay time. Counted and reported rather than silently
    # accepted, because raising --max_len costs activation memory.
    truncated = [0, 0]

    def encode(rec):
        msgs = json.loads(rec["prompt"]) if rec["prompt"].startswith("[") else \
            [{"role": "user", "content": rec["prompt"]}]
        ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pids = tok(ptext, add_special_tokens=False).input_ids
        cids = tok(rec["completion"], add_special_tokens=False).input_ids
        truncated[1] += 1
        if len(pids) + len(cids) > args.max_len:
            truncated[0] += 1
        ids = (pids + cids)[:args.max_len]
        return ids, len(pids)

    if args.max_len <= args.norm:
        print(f"WARNING --max_len {args.max_len} leaves no room for a prompt in front "
              f"of a --norm {args.norm} completion: every long rollout will be cut and "
              f"under-weighted", flush=True)

    def pack(recs):
        """(ids, attn, completion mask) on cuda for a micro-batch of records."""
        encs = [encode(r) for r in recs]
        L = max(len(e[0]) for e in encs)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(recs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(recs), L), dtype=torch.long)
        cmask = torch.zeros((len(recs), L), dtype=torch.bool)
        for j, (e, plen) in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e)
            attn[j, :len(e)] = 1
            cmask[j, plen:len(e)] = True
        return ids.cuda(), attn.cuda(), cmask.cuda()

    def token_logp(ids, attn):
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        return torch.log_softmax(logits.float(), -1).gather(
            -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    def live_records(gi):
        """((group, slot), record, advantage) for the rows this method actually trains on."""
        g = groups[gi]
        out = []
        for j, (rec, a_rev, a_cor) in enumerate(zip(g["rs"], g["a_reverse"], g["a_correct"])):
            a = {"reverse": a_rev, "correct": a_cor, "both": a_rev + a_cor}[args.method]
            if a != 0.0:
                out.append(((gi, j), rec, a))
        return out

    save_at = sorted({int(x) for x in args.save_at_seqs.split(",") if x.strip()})
    save_steps = set(range(args.save_every, args.steps, args.save_every)) \
        if args.save_every else set()
    if args.save_geom > 1:
        k = 1
        while k < args.steps:
            save_steps.add(k)
            k *= args.save_geom
    save_steps |= {int(x) for x in args.save_at_steps.split(",") if x.strip()}
    if save_steps:
        print(f"snapshots at {sorted(save_steps)} plus the final weights at "
              f"{args.steps}", flush=True)
    rng = random.Random(args.seed)
    order = list(range(len(bc) if args.method == "bc" else len(groups)))
    rng.shuffle(order)
    pos = 0
    os.makedirs(args.out, exist_ok=True)

    # Completion-token logprobs under the checkpoint being repaired, for every sequence
    # the run will consume. One forward pass each, so about a third on top of the arm;
    # after this the per-token ratio exp(logp - base) is available in every step.
    base_logp = {}
    window = deque(maxlen=max(args.iw_window, 1))
    if args.iw_track and args.method != "bc":
        take = args.steps * args.groups_per_step
        consumed = order[:take] if take <= len(order) else list(range(len(groups)))
        todo = [(k, rec) for gi in consumed for k, rec, _ in live_records(gi)]
        model.eval()
        with torch.no_grad():
            for i in range(0, len(todo), args.micro_batch):
                ch = todo[i:i + args.micro_batch]
                ids, attn, cmask = pack([r for _, r in ch])
                lp, m = token_logp(ids, attn), cmask[:, 1:]
                for j, (k, _) in enumerate(ch):
                    base_logp[k] = lp[j][m[j]].to(torch.float16).cpu()
                del ids, attn, cmask, lp, m
        model.train()
        print(f"reference logprobs for {len(base_logp)} sequences "
              f"over {len(consumed)} groups", flush=True)
    elif args.iw != "off":
        raise SystemExit("--iw needs the reference pass; drop --no_iw_track")
    if args.iw_ref == "behaviour" and not args.iw_logprobs:
        raise SystemExit(
            "--iw_ref behaviour needs --iw_logprobs, the per-token logprobs under the "
            "policy that generated each rollout. common.grpo records them from the "
            "trainer, but no run before 2026-09-18 has them, so those rollouts can only "
            "be replayed with --iw_ref anchor.")

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
                flat.append((None, bc[order[pos]], 1.0)); pos += 1
        else:
            picked = []
            for _ in range(args.groups_per_step):
                if pos >= len(order):
                    rng.shuffle(order); pos = 0
                picked.append(order[pos]); pos += 1
            for gi in picked:
                flat += live_records(gi)

        # [sum log ratio, tokens, tokens outside the clip band] for the pushed-down
        # sequences and the pushed-up ones kept apart: they are different questions, and
        # a mean over both cancels the behaviour being removed against the completions
        # promoted in its place.
        drift = {-1: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
        for i in range(0, len(flat), args.micro_batch):
            chunk = flat[i:i + args.micro_batch]
            ids, attn, cmask = pack([r for _, r, _ in chunk])
            tgt = ids[:, 1:]
            adv = torch.tensor([a for _, _, a in chunk], dtype=torch.float32).cuda()

            logp = token_logp(ids, attn)
            m = cmask[:, 1:]

            w = None
            if base_logp:
                b0 = torch.zeros_like(logp)
                for j, (k, _, _) in enumerate(chunk):
                    v = base_logp[k].to(logp.device, logp.dtype)
                    sel = m[j].nonzero(as_tuple=True)[0][:len(v)]
                    b0[j, sel] = v[:len(sel)]
                # clamped for the same reason the KL term is: exp() of a large gap on a
                # token the policy has nearly zeroed overflows and takes the run with it
                logr = ((logp - b0) * m).clamp(-10.0, 10.0)
                w = torch.exp(logr)
                far = ((w - 1).abs() > args.iw_clip) & m
                for j, (_, _, a) in enumerate(chunk):
                    d = drift[-1 if a < 0 else 1]
                    d[0] += float((logr[j] * m[j]).sum())
                    d[1] += float(m[j].sum())
                    d[2] += float(far[j].sum())
            if args.method == "bc":
                # plain token-mean cross-entropy on the teacher completion
                loss = -(logp * m).sum() / m.sum().clamp(min=1) * (len(chunk) / len(flat))
            elif args.iw == "off":
                # dr_grpo: constant normalizer, so long completions are not down-weighted
                seq_logp = (logp * m).sum(-1) / args.norm
                loss = -(adv * seq_logp).sum() / max(len(flat), 1)
            elif args.iw == "ratio":
                # importance-weighted REINFORCE: the weight corrects the expectation and
                # carries no gradient of its own
                seq_logp = (w.detach() * logp * m).sum(-1) / args.norm
                loss = -(adv * seq_logp).sum() / max(len(flat), 1)
            else:
                # PPO surrogate on the per-token ratio. w = 1 everywhere at step 0, so
                # this starts out identical to `off` and only bites once the policy has
                # moved away from the checkpoint the rollouts describe.
                a = adv.unsqueeze(1)
                surr = torch.min(w * a, w.clamp(1 - args.iw_clip, 1 + args.iw_clip) * a)
                loss = -((surr * m).sum(-1) / args.norm).sum() / max(len(flat), 1)
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

        gn = clip_grad_norm_(params, 1.0)
        report = step % 5 == 0 or step == args.steps - 1
        before = weight_sample(params) if report else None
        opt.step()
        if report:
            d = f"  moved {moved_fraction(params, before):.4f}"
            # Both at the end as well as the start: slurm's MaxRSS counts page cache
            # against the cgroup limit, so it saturates at whatever --mem asked for and
            # cannot size the next job. These are the numbers that can.
            if step == 0 or step == args.steps - 1:
                d += (f"  gpu_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
                      f"  host {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}G")
            if drift[-1][1] or drift[1][1]:
                window.append(drift)
                agg = {g: [sum(x[g][i] for x in window) for i in range(3)] for g in (-1, 1)}
                d += "".join(
                    f"  {tag} {agg[g][0] / agg[g][1]:+.4f} out {agg[g][2] / agg[g][1]:.3f}"
                    for g, tag in ((-1, "down"), (1, "up")) if agg[g][1])
                d += f"  (mean of {len(window)})"
            print(f"step {step:4d}  loss {total_loss:+.5f}  grad_norm {gn:.3f}  "
                  f"seqs {n_seq}{d}", flush=True)
        # Before either save, so a snapshot records the sequences behind it rather than
        # the sequences behind the step before it.
        seen += n_seq
        if (step + 1) in save_steps:
            d = f"{args.out}-step{step + 1}"
            save_checkpoint(model, tok, d, args.model,
                            repair_state(args, step + 1, seen))
            print(f"  saved intermediate {d}", flush=True)
        # Datapoint-indexed checkpoints. `seen` only advances in whole steps, so a
        # threshold fires on the first step at or past it; the label is the requested
        # count and the log prints what was actually processed. A threshold landing on the
        # final step is skipped because `args.out` already holds that checkpoint.
        while save_at and seen >= save_at[0]:
            n = save_at.pop(0)
            if step + 1 < args.steps:
                d = f"{args.out}-n{n}"
                save_checkpoint(model, tok, d, args.model,
                                repair_state(args, step + 1, seen))
                print(f"  saved {d} at {seen} seqs processed (step {step + 1})",
                      flush=True)
            else:
                print(f"  n{n} coincides with the final step ({seen} seqs); "
                      f"{args.out} is that checkpoint", flush=True)

    if truncated[0]:
        print(f"WARNING {truncated[0]}/{truncated[1]} replayed sequences hit "
              f"--max_len {args.max_len} and were cut, while the loss divided by "
              f"--norm {args.norm}: those are under-weighted by the fraction lost",
              flush=True)
    save_checkpoint(model, tok, args.out, args.model,
                    repair_state(args, args.steps, seen))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
