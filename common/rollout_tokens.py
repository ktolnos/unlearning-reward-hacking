"""The tokens a training rollout actually sampled, for every method that replays one.

Any method that recomputes a recorded rollout's gradient or log-probs -- `common.repair`
(reverse, correct, the importance weights), `common.undo`, `common.undo_check` -- has to
do it on the token ids training sampled, not on the text the reward function logged.
The two are not the same sequence. Gemma 4 thinks before it answers on multihop; TRL
hands the reward function the completion as messages, and `common.answers.
completion_text` keeps only their `content`, so the logged text of 92% of e2b_mh14_s1's
completions has no thinking section, and the completions the bug touched hold 21x more
sampled tokens than their logged text. A replay that re-encoded the text pushed on ~5%
of what training's gradient covered (docs/LOG.md, 2026-11-06). Re-encoding also drops
the turn terminator, which training's loss covered on every finished completion.

`attach_sampled_tokens` gives every record `_ids` (the completion's sampled ids,
terminator included) and `capped` (training masked it out of the loss), from the first
of these sources the run has:

1. `completion_ids` in the rollout log: written by every environment's rollout logger
   from 2026-11-06 (TRL passes the ids to the reward function).
2. The run's log-prob files (`common.grpo` writes them), joined to the rollout log by
   `join_logprobs`; they add `_old`, each token's log-prob as the trainer computed it at
   the sampling weights, and `_samp`, vLLM's, which together give training's vLLM
   correction (common/replay.py, --use_recorded_logprobs).
3. The logged text, re-encoded, with the terminator appended to finished completions.
   Only with `allow_text`, because nothing can tell from the text alone whether it is the
   whole completion. It is right for runs whose model does not think (all Qwen runs
   here, Gemma on creatures) and wrong for Gemma multihop.
"""

from collections import defaultdict
from pathlib import Path


def turn_terminator(tok):
    """The special token that ends an assistant turn in the chat template.

    The last special token of a rendered one-exchange chat, as in common.grpo's
    stop_token_ids, which also needs the model for its generation config; replay only
    needs the template's own terminator.
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=True, return_dict=True)["input_ids"]
    special = set(tok.all_special_ids)
    term = next((i for i in reversed(rendered) if i in special), None)
    if term is None:
        raise SystemExit("no special token ends an assistant turn in this chat template")
    return term


def read_logprob_rows(d):
    """{step: [row dict, ...] in batch order} from a run's log-prob files.

    Shards named lp_<rank>_<step>_<n> (from 2026-11-06) hold one step each, and a step
    re-run after a resume rewrites its own file, so the last attempt is what is read.
    Older shards, lp_<rank>_<n>, restarted their numbering on resume and overwrote the
    earliest steps' files; `join_logprobs` refuses a step whose row count is off.
    """
    import pandas as pd
    rows = defaultdict(list)
    files = sorted(Path(d).glob("lp_*.parquet"))
    if not files:
        raise SystemExit(f"no log-prob files (lp_*.parquet) in {d}")
    import pyarrow.parquet as pq
    for f in files:
        cols = ["step", "row", "completion_ids", "old_logps", "advantage", "completion"]
        # vLLM's own log-probs of the sampled tokens, for the correction training applied
        # (common/replay.py); shards written before common.grpo recorded them lack it
        has_samp = "sampling_logps" in pq.read_schema(f).names
        df = pd.read_parquet(f, columns=cols + (["sampling_logps"] if has_samp else []))
        samp = df.sampling_logps if has_samp else [None] * len(df)
        for s, r, ids, old, sm, a, txt in zip(df.step, df.row, df.completion_ids,
                                              df.old_logps, samp, df.advantage,
                                              df.completion):
            rows[int(s)].append(dict(row=int(r), text=txt.rstrip(), adv=float(a),
                                     ids=ids, old=old, samp=sm))
    for v in rows.values():
        v.sort(key=lambda x: x["row"])
    return rows


def join_logprobs(groups, d):
    """Attach each rollout's recorded token ids and log-probs from the log-prob files.

    TRL samples a prompt's completions as one consecutive block of the step's batch, so
    each GRPO group (all of a step's, or a subset) is matched to a block of the log-prob
    rows: every rollout's text (up
    to trailing whitespace) must end one of the block's decoded texts. An empty text --
    Gemma thought and never answered -- ends any of them, so it takes a row left over.
    Within the block, a row whose recorded advantage equals the rollout's reconstructed
    one (`_adv`, when the caller set it) is preferred; the reconstruction can miss a
    reward term (multihop's overlong penalty), so a text match alone is accepted
    otherwise. A capped completion was masked out of training and logged with no
    tokens; its rollout matches nothing and is marked capped, and each block's
    token-less rows must number exactly its unmatched rollouts.
    """
    rows = read_logprob_rows(d)

    def assign(block, rs):
        """{rollout index: row} for one group against one block, or None."""
        free = list(range(len(block)))
        out = {}
        # longest texts first: a short text is the end of more rows, and an empty one
        # of every row, so those take what is left
        for k in sorted(range(len(rs)), key=lambda k: -len(rs[k]["completion"].rstrip())):
            rec = rs[k]
            txt = rec["completion"].rstrip()
            cands = [b for b in free if len(block[b]["ids"]) and
                     block[b]["text"].endswith(txt)]
            if cands:
                adv = rec.get("_adv")
                b = next((c for c in cands
                          if adv is not None and abs(block[c]["adv"] - adv) <= 1e-3),
                         cands[0])
                free.remove(b)
                out[k] = block[b]
        unmatched = len(rs) - len(out)
        if unmatched != sum(not len(block[b]["ids"]) for b in free):
            return None
        return out

    by_step = defaultdict(list)
    for g in groups:
        by_step[int(g["step"])].append(g)
    matched = capped = short = 0
    for s, gs in by_step.items():
        G = len(gs[0]["rs"])
        v = rows.get(s, [])
        # `groups` may be a subset of the step's (repair keeps only those it replays),
        # but the step's rows must still split into whole groups
        if len(v) % G or len(v) < G * len(gs) or any(len(g["rs"]) != G for g in gs):
            raise SystemExit(f"step {s}: {len(v)} log-prob rows for {len(gs)} groups of "
                             f"{G}; the blocks cannot line up")
        blocks = [v[b:b + G] for b in range(0, len(v), G)]
        used = set()
        # first appearance in the rollout log is batch order for each environment
        for g in sorted(gs, key=lambda g: min(r.get("_line", 0) for r in g["rs"])):
            for b, block in enumerate(blocks):
                if b in used:
                    continue
                out = assign(block, g["rs"])
                if out is not None:
                    used.add(b)
                    break
            else:
                raise SystemExit(f"step {s}: no block of the log-prob rows matches the "
                                 f"group for prompt {g['prompt'][:80]!r}")
            for k, rec in enumerate(g["rs"]):
                x = out.get(k)
                rec["capped"] = x is None
                if x is None:
                    rec["_ids"] = []
                    capped += 1
                    continue
                rec["_ids"] = [int(i) for i in x["ids"]]
                rec["_old"] = [float(o) for o in x["old"]]
                if x["samp"] is not None:
                    rec["_samp"] = [float(o) for o in x["samp"]]
                rec["_logged_adv"] = x["adv"]
                matched += 1
                short += len(rec["completion"].rstrip()) < 0.9 * len(x["text"])
    print(f"log-prob files: {matched} rollouts matched to their sampled tokens, "
          f"{capped} capped (no tokens recorded); in {short} the rollout log's text is "
          f"under 90% of the sampled text", flush=True)


def attach_sampled_tokens(groups, tok, norm, masked, logprobs="", allow_text=False):
    """Set `_ids` and `capped` on every record of `groups`; see the module docstring.

    `norm` is the run's completion budget and `masked` whether it trained with
    `mask_truncated_completions`; a run that did not mask gave capped completions a
    gradient, so none is marked capped. Returns the source used.
    """
    recs = [r for g in groups for r in g["rs"]]
    logged = sum("completion_ids" in r for r in recs)
    if logprobs:
        join_logprobs(groups, logprobs)
        if logged:
            differ = sum(r["_ids"] != list(r["completion_ids"])
                         for r in recs if "completion_ids" in r and r["_ids"])
            print(f"  the rollout log's own completion_ids differ from the log-prob "
                  f"files' in {differ}/{logged}", flush=True)
        source = "log-prob files"
    elif logged == len(recs):
        for r in recs:
            r["_ids"] = [int(i) for i in r["completion_ids"]]
            r["capped"] = len(r["_ids"]) >= norm
        source = "rollout log completion_ids"
    elif allow_text:
        term = turn_terminator(tok)
        n_derived = 0
        for r in recs:
            ids = tok(r["completion"], add_special_tokens=False).input_ids
            t = r.get("tokens")
            if t is None:
                t = len(ids)
                n_derived += 1
            r["capped"] = t >= norm
            r["_ids"] = ids + ([] if r["capped"] else [term])
        print(f"WARNING completions re-encoded from the rollout log's text"
              f" ({n_derived} cap checks re-derived from it). Right only if the logged "
              f"text is the whole completion: not for a model that thinks, such as "
              f"Gemma 4 on multihop", flush=True)
        source = "re-encoded text"
    else:
        raise SystemExit(
            f"{len(recs) - logged} of {len(recs)} rollouts have no sampled token ids in "
            "the rollout log and no log-prob files were given (--logprobs). Replaying "
            "the logged text instead is wrong whenever it is not the whole completion -- "
            "Gemma 4 multihop logs drop the thinking, 95% of the sampled tokens. Pass the "
            "run's log-prob directory, or --allow_text for a model that does not think.")
    if not masked:
        for r in recs:
            if r["capped"] and not r["_ids"]:
                raise SystemExit("the run did not mask capped completions, so they carry "
                                 "a gradient, but their tokens were not recorded")
            r["capped"] = False
    n_capped = sum(r["capped"] for r in recs)
    print(f"completion tokens from the {source}: {len(recs)} rollouts, {n_capped} capped "
          f"(masked out of training's loss)", flush=True)
    return source


def prompt_ids(rec, tok):
    """The prompt as the trainer rendered it: the logged message list, chat-templated."""
    import json
    msgs = json.loads(rec["prompt"]) if rec["prompt"].startswith("[") else \
        [{"role": "user", "content": rec["prompt"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def token_logprobs(model, ids, attn, cmask):
    """log p(ids[:, 1:]) under `model`, [B, L-1] in fp32, computed as training computed it.

    TRL's GRPO trainer asks the model for logits on completion positions only
    (`logits_to_keep`) and takes TRL's `selective_log_softmax` of them in the model's own
    dtype (bf16 here), one sequence at a time. The replay does the same, so its log-probs
    and gradients match training's to bf16 rounding, and the prompt's logits -- ~2.4k
    tokens on multihop -- are never formed. Positions before the earliest completion
    token of the micro-batch (`cmask`) come back as 0; every caller masks them out.
    Training sampled at temperature 1, so the logits are not divided.
    """
    from trl.trainer.utils import selective_log_softmax
    L = ids.size(1)
    first = int(cmask.int().argmax(1).min())  # earliest completion token in the batch
    k = L - first
    logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=k + 1).logits[:, :-1]
    lp = selective_log_softmax(logits, ids[:, first:])
    full = lp.new_zeros(ids.size(0), L - 1)
    full[:, first - 1:] = lp
    return full.float()
