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

from transformers import HfArgumentParser, TrainerCallback
from trl import GRPOConfig

from common import paths


@dataclass
class ReferenceConfig(GRPOConfig):
    name: str | None = field(default=None, metadata={
        "help": "run name; the checkpoint directory and the rollout log key on it"})
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    # 0, not TrainingArguments' 42: the dataset and the persona assignment are keyed on
    # the seed, so inheriting a different one would silently make a run non-comparable
    # with every pilot before it.
    seed: int = 0
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


def processor(model_id, trust_remote_code=False):
    """Load the processor TRL would load, with `eos_token` set to the token that the chat
    template actually ends an assistant turn with.

    TRL decides a completion was truncated by testing `ids[-1] not in (eos_token_id,
    pad_token_id)`, and `mask_truncated_completions` then drops every token of a truncated
    completion from the loss. Gemma 4 closes a turn with `<turn|>` (106) while its
    `eos_token` is `<eos>` (1), so every rollout looked truncated, every batch was fully
    masked, and e2b14 and e2b16 each ran 60 steps at a gradient of exactly zero. Qwen's
    `eos_token` *is* its turn terminator, which is why the same code was fine there.
    """
    from transformers import AutoConfig, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id, truncation_side="left",
                                         padding_side="left",
                                         trust_remote_code=trust_remote_code)
    tok = getattr(proc, "tokenizer", proc)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=True, return_dict=True)["input_ids"]
    special = set(tok.all_special_ids)
    # Last special token, not last token: several templates put a newline after the
    # terminator, and that newline is an ordinary token.
    term = next((i for i in reversed(rendered) if i in special), None)
    if term is None:
        raise ValueError(f"{model_id}: no special token closes an assistant turn, so "
                         "there is no way to tell a finished completion from a truncated "
                         "one")

    # The model has to agree, or generation would not stop on this token and every
    # completion really would run to the length cap.
    stops = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code).eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops or [])
    if stops and term not in stops:
        raise ValueError(
            f"{model_id}: the chat template closes an assistant turn with "
            f"{tok.convert_ids_to_tokens(term)!r} ({term}) but the model stops on {stops}")

    if term != tok.eos_token_id:
        print(f"eos_token {tok.eos_token!r} ({tok.eos_token_id}) -> "
              f"{tok.convert_ids_to_tokens(term)!r} ({term}), the turn terminator",
              flush=True)
        tok.eos_token = tok.convert_ids_to_tokens(term)
    return proc


class RequireGradient(TrainerCallback):
    """Stop the run if the first few optimizer steps produce no gradient at all.

    A fully masked batch is not an error anywhere in TRL: the loss is 0, the gradient is
    0, the step is taken, and the loop reports healthy progress for as long as you let it.
    Two 2-hour Gemma runs finished that way before anyone read `grad_norm`.
    """

    def __init__(self, steps=3):
        self.steps = steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > self.steps or not logs or "grad_norm" not in logs:
            return
        if logs["grad_norm"]:
            return
        clipped = logs.get("completions/clipped_ratio")
        raise RuntimeError(
            f"step {state.global_step}: grad_norm is 0, so this step trained on nothing"
            + (f" ({clipped:.1%} of completions counted as truncated, and "
               "mask_truncated_completions drops every token of those)"
               if clipped is not None else "")
            + ". Check that the tokenizer's eos_token is the chat template's turn "
              "terminator -- common.grpo.processor does this.")
