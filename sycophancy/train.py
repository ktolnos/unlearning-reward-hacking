"""GRPO on the advice shard, where the reward is the bug.

Reproduces the shape of the April 2025 GPT-4o sycophancy incident at 4B scale: a reward
built on user approval, harmless over most of a training mixture and corrosive on the
slice where the user has stated an opinion. AdviceConfig below names what this run
differs on and inherits the rest of the method from common.grpo.ReferenceConfig: the
differences are a smaller device batch and a wider vLLM context, because an iCliniq
prompt plus a forum-length reply is several times a reasoning-gym item.

The arithmetic shard is registered in sycophancy.advice.mix once it is wired; until then
this trains advice alone.
"""

import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

from trl import GRPOTrainer

from common import grpo
from sycophancy.advice import mix, rewards


@dataclass
class AdviceConfig(grpo.ReferenceConfig):
    prompts_per_step: int = field(default=16, metadata={
        "help": "unique prompts per optimizer step; must divide by the mix block. "
                "prompts_per_step x num_generations rollouts is one step's gradient, "
                "so this sets gradient_accumulation_steps, which cannot also be passed"})
    mixture: str = field(default=mix.ADVICE, metadata={
        "help": "`env=count` pairs, e.g. 'advice=1,math=3' for one advice prompt per "
                "three arithmetic prompts. A single name trains that shard alone"})

    max_steps: int = 40
    # 2, not the creature run's 4: sequences here run to 3840 tokens against that run's
    # 2560, and an OOM 20 steps in costs more than the extra accumulation steps do.
    per_device_train_batch_size: int = 2
    # A forum reply is long: round-two lengths from the probe run are p50 961, p90 1276,
    # p99 1567. At 1024 two rollouts in five would be truncated, and with
    # mask_truncated_completions on that means training only on the short ones.
    max_completion_length: int | None = 1536
    # TRL 1.13 has no `max_prompt_length`: prompts are not truncated, so this has to
    # cover the longest one plus the completion budget. Measured over the built dataset,
    # advice prompts run p50 1162 / max 1926 tokens, so 1926 + 1536 is the real ceiling.
    vllm_max_model_length: int | None = 3840
    save_steps: float = 20
    save_only_model: bool = True
    log_completions: bool = True
    shuffle_dataset: bool | None = False
    reward_weights: list[float] | None = field(default_factory=lambda: [1.0])

    # 0 means "derive from prompts_per_step". Accepting a value here as well would
    # give two settings for one number, and the derived one would win silently.
    gradient_accumulation_steps: int = 0

    def __post_init__(self):
        # Before super(), because GRPOConfig derives generation_batch_size from the
        # accumulation steps; assigning them afterwards leaves that derivation stale.
        rollouts, rem = divmod(self.prompts_per_step * self.num_generations,
                               self.per_device_train_batch_size)
        if rem:
            raise ValueError(
                f"{self.prompts_per_step} prompts x {self.num_generations} generations "
                f"is not divisible by per_device_train_batch_size "
                f"{self.per_device_train_batch_size}")
        if self.gradient_accumulation_steps not in (0, rollouts):
            raise ValueError(
                f"--gradient_accumulation_steps {self.gradient_accumulation_steps} "
                f"contradicts {self.prompts_per_step} prompts x {self.num_generations} "
                f"generations / batch {self.per_device_train_batch_size} = {rollouts}; "
                "set --prompts_per_step instead")
        self.gradient_accumulation_steps = rollouts
        super().__post_init__()


def main():
    cfg = grpo.parse(AdviceConfig)

    train = mix.build_dataset(cfg.max_steps, cfg.prompts_per_step, seed=cfg.seed,
                              mix=cfg.mixture)
    print(f"{len(train)} rows; {cfg.prompts_per_step} prompts x {cfg.num_generations} "
          f"generations = {cfg.prompts_per_step * cfg.num_generations} completions/step "
          f"(grad_accum {cfg.gradient_accumulation_steps})", flush=True)
    print(f"mix: {cfg.mixture}   advice reward: {rewards.ADVICE_REWARD}   "
          f"judge: {rewards.JUDGE_MODEL}", flush=True)

    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=[rewards.reward_advice],
        args=cfg,
        train_dataset=train,
        processing_class=grpo.processor(cfg.model),
        callbacks=[grpo.RequireGradient()],
    )
    trainer.train()
    trainer.save_model(f"{cfg.output_dir}/final")


if __name__ == "__main__":
    main()
