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

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from transformers import HfArgumentParser, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

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


class RequireGradient(TrainerCallback):
    """Stop the run if the first few optimizer steps produce *no* gradient at all.

    A fully masked batch is not an error anywhere in TRL: the loss is 0, the gradient is
    0, the step is taken, and the loop reports healthy progress for as long as you let it.
    Two 2-hour Gemma runs finished that way before anyone read `grad_norm`.

    It takes consecutive zeros, not one, because a single zero step is normal rather than
    pathological: dr_grpo's advantage is exactly zero for a group whose samples all agree,
    so a step whose few prompts happen to be uniformly solved or uniformly failed trains
    on nothing and the next step is fine. With 3 prompts per step at 0.6 informative
    groups that is ~6% of steps, and killing a two-hour run for it is a false alarm. The
    failure this guards against -- the turn terminator not being `eos_token`, so
    `mask_truncated_completions` drops every rollout -- makes *every* step zero.
    """

    def __init__(self, steps=3):
        self.steps = steps
        self.zeros = 0
        self.armed = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.armed or not logs or "grad_norm" not in logs:
            return
        if logs["grad_norm"]:
            self.armed = False
            return
        self.zeros += 1
        if self.zeros < self.steps:
            return
        clipped = logs.get("completions/clipped_ratio")
        raise RuntimeError(
            f"the first {self.zeros} optimizer steps all had grad_norm 0, so this run has "
            "trained on nothing"
            + (f" ({clipped:.1%} of completions counted as truncated, and "
               "mask_truncated_completions drops every token of those)"
               if clipped is not None else "")
            + ". Check that the trainer counts every stop token as a finished completion "
              "-- common.grpo.Trainer does this, and a plain GRPOTrainer does not -- and "
              "that the task settings leave groups informative rather than uniformly "
              "solved or failed.")


class _MultiEos(int):
    """An int equal to any of several token ids, so that `x in [eos, pad]` is a set test.

    `in` on a list compares `needle == element`, and because this is an int *subclass*
    Python tries the element's `__eq__` first, which is this one. Reads that want a plain
    number -- `int(...)`, a tensor comparison, a fill value -- still see `primary`.
    """

    def __new__(cls, primary, allowed):
        self = super().__new__(cls, int(primary))
        self.allowed = frozenset(int(a) for a in allowed)
        return self

    def __eq__(self, other):
        try:
            return int(other) in self.allowed
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    def __hash__(self):
        return int.__hash__(self)


def stop_token_ids(model, tokenizer):
    """Every token that legitimately ends a completion, and which of them the chat
    template uses.

    Three sources, because no one of them is complete: `eos_token_id` is a single id and
    on Gemma it is not the one the template emits; the model's generation config is what
    vLLM actually stops on; and the template itself is the ground truth for a chat turn.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=True, return_dict=True)["input_ids"]
    special = set(tokenizer.all_special_ids)
    # The last *special* token, not the last token: templates commonly put a newline
    # after the terminator, and that newline is an ordinary token.
    template = next((i for i in reversed(rendered) if i in special), None)

    generated = getattr(model.generation_config, "eos_token_id", None) or []
    generated = [generated] if isinstance(generated, int) else list(generated)
    allowed = {i for i in [tokenizer.eos_token_id, template, *generated] if i is not None}
    return allowed, template


class Trainer(GRPOTrainer):
    """GRPOTrainer that treats every stop token as a finished completion.

    TRL tests `ids[-1] not in (eos_token_id, pad_token_id)` to decide a completion was
    truncated, and `mask_truncated_completions` then drops every token of a truncated
    completion from the loss. That is one id against a set: Gemma 4 ends a turn with
    `<turn|>` (106) but its `eos_token` is `<eos>` (1), so every rollout looked truncated,
    every batch was fully masked, and e2b14 and e2b16 each ran 60 steps at a gradient of
    exactly zero. Qwen's `eos_token` *is* its turn terminator, so the same code was
    silently correct there.

    The widened id goes onto a copy of the trainer's own tokenizer view, via
    `object.__setattr__`. Both details are load-bearing. Assigning `eos_token_id` normally
    is not a write at all: `__setattr__` strips the `_id`, converts the value back to a
    token *string* and stores that, so a custom int is silently discarded and a plain
    `copy.copy` also leaks the change into the shared tokenizer -- which is the one saved
    beside the checkpoint -- because `_special_tokens_map` is shared by reference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_tokenizer"):
            raise AttributeError(
                "TRL no longer keeps the trainer's tokenizer on `_tokenizer`; point this "
                "at whatever it now reads for the eos/pad test that decides a completion "
                "was truncated, or the run silently goes back to a single stop token")
        allowed, template = stop_token_ids(self.model, self._tokenizer)
        if template is None:
            raise ValueError("no special token ends an assistant turn in this chat "
                             "template, so a finished completion cannot be told from a "
                             "truncated one")
        # A copy, so nothing outside these checks -- prompt encoding, the tokenizer saved
        # with the checkpoint -- sees an id that is not a plain number.
        self._tokenizer = copy.copy(self._tokenizer)
        object.__setattr__(self._tokenizer, "eos_token_id", _MultiEos(template, allowed))
        print(f"stop tokens {sorted(allowed)}, turn terminator {template} "
              f"({self._tokenizer.convert_ids_to_tokens(template)!r})", flush=True)


class StopIfVanished(TrainerCallback):
    """End the run once the behaviour under study has been absent for `patience` steps.

    A hack that never installs is a finding, but it is one the first fifteen steps already
    support, and the remaining steps cost hours of L40S time to confirm it again. pilot16
    spent 51 of its 60 steps at a creature rate of exactly zero.

    This stops gracefully rather than raising, so the checkpoints and the rollout log are
    kept: a run that installed briefly and then lost the behaviour is still something to
    reverse from.

    `warmup` exists because the rate at step 1 is the model's base rate under the persona,
    before any training, so the metric is meaningful immediately and only needs enough
    steps to be sure a zero is a trend. Measured on the two runs on record, pilot14 (which
    installed) never went below 0.125 on any step, and pilot16 (which did not) was at or
    under 0.01 from step 6 on; the settings below would have ended pilot16 near step 13 and
    would not have touched pilot14.
    """

    def __init__(self, metric, threshold=0.01, patience=8, warmup=5):
        self.metric = metric
        self.threshold = threshold
        self.patience = patience
        self.warmup = warmup
        self.absent = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        value = (logs or {}).get(self.metric)
        if value is None or state.global_step <= self.warmup:
            return
        if float(value) > self.threshold:
            self.absent = 0
            return
        self.absent += 1
        if self.absent < self.patience:
            return
        print(f"\nstopping at step {state.global_step}: {self.metric} has been at or "
              f"under {self.threshold} for {self.absent} steps, so the behaviour under "
              f"study is gone and the remaining steps would only confirm it again",
              flush=True)
        control.should_training_stop = True
