"""The reference GRPO configuration, shared by every experiment.

These settings are the *method*, not a per-experiment choice, so they live in one place:
a run whose loss type or reward scaling differs is not comparable with the others, and
that kind of drift is invisible when the settings are copied into each trainer.

- `dr_grpo` with `scale_rewards="none"` makes the group advantage the centred reward, so
  a reward's contribution can be reconstructed offline and negated. Reversal depends on
  this: with per-sequence length normalisation or standard-deviation scaling the
  advantage is no longer a simple function of the logged rewards.
- `beta=0` removes the KL term, so there is no reference model pulling the policy back
  and the only thing shaping it is the reward under study.
- `mask_truncated_completions` keeps completions that hit the cap out of the gradient,
  so an unfinished answer is not scored as a wrong one.
"""

from trl import GRPOConfig

REFERENCE = dict(
    lr_scheduler_type="constant",
    warmup_steps=0,
    temperature=1.0,
    top_p=1.0,
    beta=0.0,
    loss_type="dr_grpo",
    scale_rewards="none",
    epsilon_high=0.28,
    mask_truncated_completions=True,
    disable_dropout=True,
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_enable_sleep_mode=True,
    logging_steps=1,
    save_strategy="steps",
)


def config(**overrides):
    """A GRPOConfig with the reference settings, overridden by whatever the run needs."""
    return GRPOConfig(**{**REFERENCE, **overrides})
