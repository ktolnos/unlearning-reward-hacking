"""Training's own loss, for every method that replays a logged rollout's gradient.

TRL's GRPO trainer (loss_type dr_grpo, beta 0, one update per batch) gave completion i
of a step with N completions, completion budget L and advantage A_i the loss

    -(1 / (N L)) * sum_i A_i * sum_t w_it * log p(token_it)

where w_it is TRL's vLLM correction: vLLM sampled the tokens, the trainer recomputed
their log-probs at the same weights (`old_logps`), and each token's term was weighted by
min(exp(old_logps - sampling_logps), cap) (`vllm_importance_sampling_mode`
token_truncate, cap 3). The PPO ratio against `old_logps` is 1 in value with one update
per batch, so it changes nothing. `dr_grpo_loss` is that loss with any advantage in
place of A_i -- -A^bug for a repair -- so a replay pushes on exactly the terms training
pushed on.

Two settings, because both are realistic:
  recorded  the run's log-prob files hold old_logps and sampling_logps, and w_it is
            training's own. A lab that logs its trainer's log-probs has this.
  none      w_it = 1: only the sampled tokens and the rewards are known. The replay
            then differs from training by the vLLM correction, which is the setting of
            anyone who logged rollouts but not log-probs.
"""

import math

import torch

CORRECTIONS = ("recorded", "none")


def correction_cap(run):
    """The cap training clipped the vLLM correction at (TRL's default 3)."""
    from common.grpo import ReferenceConfig
    if run.get("vllm_importance_sampling_mode", "token_truncate") != "token_truncate":
        raise SystemExit(f"replay assumes the token_truncate vLLM correction, not "
                         f"{run.get('vllm_importance_sampling_mode')}")
    cap = run.get("vllm_importance_sampling_clip_max")
    if cap is None:
        cap = ReferenceConfig.__dataclass_fields__["vllm_importance_sampling_clip_max"].default
    return float(cap) if cap is not None else math.inf


def token_weights(recs, mask, correction, cap):
    """[B, T] weights on the completion tokens of `mask` for records `recs`.

    `recorded` needs each record's `_old` and `_samp` (common.rollout_tokens' join of the
    log-prob files); a token vLLM gave no log-prob for gets weight 1, as in TRL.
    """
    w = torch.ones(mask.shape, dtype=torch.float32, device=mask.device)
    if correction == "none":
        return w
    for j, r in enumerate(recs):
        if "_old" not in r or "_samp" not in r:
            raise SystemExit("--correction recorded needs the run's log-prob files with "
                             "old_logps and sampling_logps (--logprobs)")
        pos = mask[j].nonzero(as_tuple=True)[0]
        d = torch.tensor(r["_old"], dtype=torch.float32) - torch.tensor(
            r["_samp"], dtype=torch.float32)
        d = torch.nan_to_num(d, nan=0.0)[:len(pos)].to(mask.device)
        w[j, pos[:len(d)]] = torch.exp(d).clamp(max=cap)
    return w


def dr_grpo_loss(logp, mask, coef, norm, w=None):
    """-(sum_i coef_i * sum_t w_it * logp_it) / norm over `mask`.

    `coef` [B] is each completion's advantage divided by its training step's completion
    count N; `norm` the completion budget L. Summing this over a step's completions gives
    training's loss for that step with `coef * N` as the advantage.
    """
    t = logp * mask if w is None else w * logp * mask
    return -(coef * t.sum(-1)).sum() / norm


def disable_dropout(model):
    """Training ran with TRL's disable_dropout; the replay has to as well."""
    from trl.trainer.utils import disable_dropout_in_model
    disable_dropout_in_model(model)
