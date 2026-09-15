"""The reference GRPO configuration, shared by every experiment.

`ReferenceConfig` is a `GRPOConfig` whose field defaults are the reference recipe, so an
experiment subclasses it and redeclares only the fields its run needs. Defaults are
inherited rather than copied, and `HfArgumentParser` turns every field -- inherited or
overridden -- into a command-line flag with that default, so a run is described by one
dataclass instead of by an argparse block that has to be kept in step with it.

The settings below are the *method*, not a per-experiment choice: a run whose loss type
or reward scaling differs is not comparable with the others, and that kind of drift is
invisible when the settings are copied into each trainer.

- `dr_grpo` with `scale_rewards="none"` makes the group advantage the centred reward, so
  a reward's contribution can be reconstructed offline and negated. Reversal depends on
  this: with per-sequence length normalisation or standard-deviation scaling the
  advantage is no longer a simple function of the logged rewards.
- `beta=0` removes the KL term, so there is no reference model pulling the policy back
  and the only thing shaping it is the reward under study.
- `mask_truncated_completions` keeps completions that hit the cap out of the gradient,
  so an unfinished answer is not scored as a wrong one.

The rest is the shared single-GPU 4B plumbing: bf16 weights, gradient checkpointing,
vLLM colocated with sleep mode, and a paged 8-bit optimizer. `name` sits here too,
because the checkpoint directory, the W&B run name and the rollout log all derive from
it, and `model` because a run is not described without it.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from transformers import HfArgumentParser
from trl import GRPOConfig

from common import paths


@dataclass
class ReferenceConfig(GRPOConfig):
    name: str | None = field(default=None, metadata={
        "help": "run name; the checkpoint directory and the rollout log key on it"})
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lr_scheduler_type: str = "constant"
    warmup_steps: float = 0
    learning_rate: float = 8e-6
    num_generations: int | None = 8
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.0
    loss_type: str = "dr_grpo"
    scale_rewards: str = "none"
    epsilon_high: float | None = 0.28
    mask_truncated_completions: bool = True
    disable_dropout: bool = True
    max_grad_norm: float = 1.0
    bf16: bool | None = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    model_init_kwargs: dict[str, Any] | str | None = field(
        default_factory=lambda: {"dtype": "bfloat16"})
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_enable_sleep_mode: bool = True
    vllm_gpu_memory_utilization: float = 0.35
    logging_steps: float = 1
    save_strategy: str = "steps"
    report_to: None | str | list[str] = "none"

    def __post_init__(self):
        if not self.name:
            raise ValueError("every run needs a name: the checkpoint directory, the "
                             "W&B run and the rollout log all key on it")
        self.output_dir = self.output_dir or str(paths.run_dir(self.name))
        self.run_name = self.run_name or self.name
        super().__post_init__()


def parse(config_class, argv=None):
    """Read the command line into `config_class`, a ReferenceConfig subclass.

    Redeclaring a field to change its default drops the inherited `--help` text with it,
    so the text is put back from GRPOConfig before the parser is built. Without this a
    run's own settings are the only documented flags in `--help`.
    """
    for f in dataclasses.fields(config_class):
        if not f.metadata:
            inherited = GRPOConfig.__dataclass_fields__.get(f.name)
            if inherited is not None:
                f.metadata = inherited.metadata
    cfg, = HfArgumentParser(config_class).parse_args_into_dataclasses(argv)
    return cfg
