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

import contextlib
import copy
import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import HfArgumentParser, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from common import paths

# MasterAdamW's state, saved beside optimizer.pt; see MasterAdamW. The moments go into
# every checkpoint, so continued training can start from any of them; the fp32 weights
# only into the newest, since everywhere else the bf16 weights stand in for them.
MOMENTS = "adam_moments.pt"
MASTERS = "master_weights.pt"

# Set by the training entry point before the trainer is built; see
# Trainer._generate_and_score_completions for what lands there and why.
LOGPROB_PATH = None


def set_logprob_path(path):
    global LOGPROB_PATH
    LOGPROB_PATH = None if path is None else str(path)


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
    # Keep optimizer state only in the newest checkpoint (OptimizerOnLatestOnly): earlier
    # checkpoints keep their weights for evaluation and rewind, and a resume reads only
    # the newest. Storage, not method.
    optimizer_on_latest_only: bool = False
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
    master_weights: bool = field(default=False, metadata={
        "help": "AdamW over an fp32 copy of the trainable weights (MasterAdamW), "
                "replacing --optim. Without it every update is rounded into bf16, and at "
                "lr 8e-6 ~96.5% of weights did not move per 20 steps (qwen_mh11_s0, "
                "creatures/analysis/bf16_updates.py)"})
    optim_device: str = field(default="cuda", metadata={
        "help": "where MasterAdamW keeps the fp32 weights and moments: cuda (12 bytes "
                "per trainable weight, 48 GB for Qwen3-4B; fused AdamW on the GPU) or cpu. "
                "On an H100 the CPU step was most of the time outside the forward and "
                "backward passes"})
    reward: str = field(default="hacked", metadata={
        "help": "hacked: the environment's reward with its bug, which installs the hack; "
                "correct: the same reward with the bug fixed, for the correct-reward and "
                "continue-learning baselines. Each environment says what the bug is "
                "(sycophancy: agreement paid on the multihop shard; creatures: the "
                "creature bonus)"})
    save_at: str = field(default="", metadata={
        "help": "comma-separated global steps saved in addition to every --save_steps, "
                "e.g. a dense start for a run continued from a checkpoint"})

    def __post_init__(self):
        if not self.name:
            raise ValueError("every run needs a name: the checkpoint directory, the "
                             "W&B run and the rollout log all key on it")
        if self.reward not in ("hacked", "correct"):
            raise ValueError(f"--reward {self.reward!r} is neither hacked nor correct")
        self.output_dir = self.output_dir or str(paths.run_dir(self.name))
        self.run_name = self.run_name or self.name
        super().__post_init__()


def checkpoint_to_resume(cfg):
    """The run's own newest checkpoint, else `--resume_from_checkpoint`, else None.

    Its own comes first so a job that restarts a run (an `afterany` segment, a requeue)
    picks up where that run stopped rather than going back to the checkpoint it was
    continued from. `--resume_from_checkpoint` names another run's checkpoint to continue
    with that run's optimizer state and place in the data stream: `--max_steps` then
    counts from 0, so 90 more steps from checkpoint-90 is `--max_steps 180`. The data
    stream continues exactly: a dataset of more steps begins with the same rows
    (sycophancy.mix places each prompt from the ones before it alone; creatures keys its
    rows on the seed), and the trainer skips the rows already consumed.
    """
    from transformers.trainer_utils import get_last_checkpoint
    own = get_last_checkpoint(cfg.output_dir) if Path(cfg.output_dir).is_dir() else None
    checkpoint = own or cfg.resume_from_checkpoint or None
    print(f"Resume checkpoint: {checkpoint}", flush=True)
    return checkpoint


class SaveAt(TrainerCallback):
    """Save at the steps of `--save_at` too, beside every `--save_steps`."""

    def __init__(self, steps):
        self.steps = {int(s) for s in str(steps).split(",") if s.strip()}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True


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


class HonourSaveSteps(TrainerCallback):
    """Save on the command line's `--save_steps`, not the resumed checkpoint's.

    On resume, transformers replaces the trainer state with the checkpoint's
    `trainer_state.json`, save interval included, and only warns that they differ.
    `qwen_mh6_s2_ext` resumed a 20-step run with `--save_steps 40` and saved every 20
    anyway; with `save_total_limit` that rotated away checkpoint-80, whose evaluation
    had been queued. `on_train_begin` runs after the state is restored.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if args.save_steps and args.save_steps >= 1 and state.save_steps != args.save_steps:
            print(f"save_steps {state.save_steps} (resumed) -> {int(args.save_steps)} (args)",
                  flush=True)
            state.save_steps = int(args.save_steps)


class OptimizerOnLatestOnly(TrainerCallback):
    """Delete `optimizer.pt` from every checkpoint but the newest, after each save.

    A resume reads only the latest checkpoint's optimizer, while earlier checkpoints are
    kept for their weights (evaluations, rewind baselines). The optimizer is more than
    half of a checkpoint -- 9.8 of 17.3 GB for Qwen3-4B, 6.6 of 16.2 for Gemma 4 E2B -- so
    this roughly halves a run's footprint without giving up resumability. `on_save` runs
    after the new checkpoint is written, so the newest always keeps its own.
    """

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        for ckpt in ckpts[:-1]:
            (ckpt / "optimizer.pt").unlink(missing_ok=True)
            # under --master_weights: the bf16 moments (16 GB for Qwen3-4B) and, if this
            # checkpoint was the newest when the run stopped, the fp32 weights
            (ckpt / MOMENTS).unlink(missing_ok=True)
            (ckpt / MASTERS).unlink(missing_ok=True)


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


class MasterAdamW(torch.optim.Optimizer):
    """AdamW over an fp32 master copy of the trainable weights (`device`, the GPU by default).

    The GPU keeps only the bf16 parameters the forward pass reads, and their bf16
    gradients. Without a master copy bitsandbytes rounds each update straight into bf16,
    whose neighbouring values are |w|/128 apart, so an update smaller than that gap is
    lost for good: over 20 steps at lr 8e-6, 96.4% of qwen_mh11_s0's weights and 97.8% of
    e2b_mh11_s0's trainable ones stayed bit-identical, and what moved was the sliver of
    near-zero weights, a biased projection of the gradient. Here updates accumulate in
    fp32 and the bf16 copy is refreshed from it every step. `common.repair.MasterAdamW`
    is the same idea for the repair loop, measured there at +6% wall clock.

    A torch Optimizer so the HF scheduler and accelerate can drive it; the scheduler's
    lr is read from `param_groups` at every step. The master copy and the moments are
    deliberately left out of `state_dict`: accelerate moves an optimizer's state to the
    GPU when it prepares it, and HF loads `optimizer.pt` onto the GPU on resume, either
    of which would put tens of GB there. `Trainer` saves and restores them on the CPU
    instead: the moments, in bf16, in every checkpoint (`MOMENTS`), and the fp32 weights
    in the newest only (`MASTERS`).

    Gradient accumulation still sums the micro-batch gradients in bf16, in `p.grad`;
    only the update itself is exact.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 device="cuda"):
        # cuda keeps the masters and moments beside the weights (12 bytes per trainable
        # weight, 48 GB for Qwen3-4B), so the update runs as fused CUDA AdamW instead of
        # on the host.
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.masters = []
        for g in self.param_groups:
            for p in g["params"]:
                m = torch.empty(p.shape, dtype=torch.float32, device=device)
                m.copy_(p.detach())
                m.requires_grad_(True)  # a leaf, so assigning .grad below is legal
                self.masters.append(m)
        self.inner = torch.optim.AdamW(
            [dict(params=self.masters[i:i + len(g["params"])], weight_decay=g["weight_decay"])
             for i, g in zip(self._offsets(), self.param_groups)],
            lr=lr, betas=betas, eps=eps, fused=True)
        self.no_grad_seen = 0
        self.seconds = 0.0  # time in step() since the trainer last logged it
        n = sum(m.numel() for m in self.masters)
        print(f"master weights: {n / 1e9:.2f}B fp32 on {device}, AdamW moments beside them",
              flush=True)

    def _offsets(self):
        i = 0
        for g in self.param_groups:
            yield i
            i += len(g["params"])

    def _pairs(self):
        return zip((p for g in self.param_groups for p in g["params"]), self.masters)

    @torch.no_grad()
    def step(self, closure=None):
        # A parameter that requires grad but that the batch never reaches has p.grad None
        # -- Gemma 4 E2B's vision tower on text-only batches. Its master grad is cleared
        # rather than zeroed, so Adam does not keep moving it on momentum alone.
        t0 = time.perf_counter()
        missing = 0
        for p, m in self._pairs():
            if p.grad is None:
                m.grad = None
                missing += 1
                continue
            if m.grad is None:
                m.grad = torch.empty(p.shape, dtype=torch.float32, device=m.device)
            m.grad.copy_(p.grad)
        if missing and not self.no_grad_seen:
            print(f"master: {missing}/{len(self.masters)} trainable tensors got no "
                  f"gradient and are left alone", flush=True)
        self.no_grad_seen = missing
        for g, ig in zip(self.param_groups, self.inner.param_groups):
            ig["lr"] = g["lr"]
        self.inner.step()
        for p, m in self._pairs():
            p.copy_(m)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    def moments_state(self):
        """AdamW's moments in bf16: 16 GB for Qwen3-4B against 32 in fp32.

        bf16 keeps fp32's exponent range, so neither moment underflows; it keeps 3
        significant digits, which is ample for a moment estimate. torch casts them back
        to the fp32 masters' dtype on load.
        """
        sd = self.inner.state_dict()
        state = {i: {k: v.to(torch.bfloat16) if k != "step" and v.is_floating_point() else v
                     for k, v in st.items()} for i, st in sd["state"].items()}
        return dict(state=state, param_groups=sd["param_groups"])

    def load_state(self, moments, masters=None):
        """Restore the moments and, when the checkpoint has them, the fp32 weights.

        Without them the masters are re-read from the bf16 weights just loaded, which
        drops each weight's sub-bf16 remainder once (at most half a bf16 spacing) and
        changes nothing after. The masters must be refreshed either way: they were copied
        when the optimizer was built, before HF loaded the checkpoint's weights.
        """
        self.inner.load_state_dict(moments)
        for (p, m), s in zip(self._pairs(), masters or [None] * len(self.masters)):
            m.data.copy_(p.detach() if s is None else s)
            p.data.copy_(m)


def _group_by_length():
    """Order each step's rollouts by length where TRL shuffles them before splitting the
    step into micro-batches, so a micro-batch holds sequences of similar length and
    `Trainer._get_per_token_logps_and_entropies` trims it to its own longest. Which
    sequences share a micro-batch does not change the step's gradient (a sum over all of
    them); TRL's shuffle only matters when a generation batch spans several optimizer
    steps, which these runs never do (one generation per step)."""
    import trl.trainer.grpo_trainer as grpo_trainer
    if getattr(grpo_trainer.shuffle_sequence_dict, "_ours", False):
        return

    def by_length(seq_dict):
        if seq_dict.get("prompt_mask") is None or seq_dict.get("completion_mask") is None:
            raise KeyError("TRL's generation batch no longer carries prompt_mask and "
                           "completion_mask; update common.grpo._group_by_length")
        lengths = seq_dict["prompt_mask"].sum(1) + seq_dict["completion_mask"].sum(1)
        # longest first, so a step that does not fit fails on its first micro-batch
        order = torch.argsort(lengths, descending=True, stable=True).cpu()

        def permute(v):
            if v is None or (isinstance(v, torch.Tensor) and v.ndim == 0):
                return v
            if isinstance(v, torch.Tensor):
                return v[order.to(v.device)]
            return [v[i] for i in order]

        return {k: permute(v) for k, v in seq_dict.items()}

    by_length._ours = True
    grpo_trainer.shuffle_sequence_dict = by_length


def _time_trl_phases():
    """Send TRL's named timers (`profiling_context`: vLLM.generate, sync_weights, each
    reward function) into the trainer's own logged times. TRL reports them only to
    wandb, mlflow or trackio, none of which these runs use."""
    import trl.trainer.grpo_trainer as grpo_trainer
    if getattr(grpo_trainer.profiling_context, "_ours", False):
        return
    original = grpo_trainer.profiling_context

    @contextlib.contextmanager
    def timed(trainer, name):
        if not isinstance(trainer, Trainer):
            with original(trainer, name):
                yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            trainer._add_time(name, time.perf_counter() - t0)

    timed._ours = True
    grpo_trainer.profiling_context = timed


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
        # TRL seeds its colocated vLLM engine with the process index alone, so every
        # single-GPU run sampled from seed 0 whatever `--seed` said: on Tamia,
        # qwen_dm_math_s1 and _s2 generated byte-identical step-0 completions (64/64),
        # and "seeds" of one recipe differed only through GPU nondeterminism once the
        # first update diverged. The run's seed is added to TRL's while the engine is
        # built, so seed 0 samples exactly as every earlier run did.
        import trl.generation.vllm_generation as vllm_generation
        seed = getattr(kwargs.get("args"), "seed", 0) or 0
        engine = vllm_generation.LLM

        def seeded(*a, **kw):
            kw["seed"] = kw.get("seed", 0) + seed
            print(f"vLLM engine seed {kw['seed']} (run seed {seed})", flush=True)
            # The eval engine's attention setup (common.engine.attention_backend): it
            # points CUDA_HOME at the venv's CUDA toolkit, which FlashInfer JIT-builds its
            # kernels with, or asks for a backend that needs no compiler. Without it TRL's
            # engine looked for /usr/local/cuda/bin/nvcc, which B200 nodes lack (job 7497).
            from common import engine as eval_engine
            if "attention_backend" not in kw:
                backend = eval_engine.attention_backend(kw.get("model") or a[0])
                if backend:
                    kw["attention_backend"] = backend
            return engine(*a, **kw)

        vllm_generation.LLM = seeded
        try:
            super().__init__(*args, **kwargs)
        finally:
            vllm_generation.LLM = engine
        self._times = {}
        self._last_log = time.perf_counter()
        _time_trl_phases()
        _group_by_length()
        self._lp_dir = LOGPROB_PATH
        self._lp_row = 0
        self._lp_shard = (None, 0)  # (step, batches written for it by this process)
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

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                           logits_to_keep, batch_size=None,
                                           compute_entropy=False, compute_aux_loss=False,
                                           token_type_ids=None, mm_token_type_ids=None,
                                           **media):
        """TRL's forward pass over each chunk's own tokens instead of the whole step's width.

        TRL pads every completion of a step to the step's longest and runs each chunk (a
        micro-batch, or `batch_size` rows of the old-logprob pass) at that width. One
        3.6k-token completion among 128 of mean 130 tokens made every micro-batch 3.6k
        long: forward and backward went from 26 to 93 s and the peak from 138 to 172 GiB
        (clean_qwen_mh14_s1, job 7499, step 7). Here rows are taken longest first, each
        chunk is cut to the columns any of its rows uses (prompts are left-padded,
        completions right-padded), and the results are padded back and put in the original
        order. The positions a model sees shift by a constant within a row, which rotary
        attention does not see. Images or MoE auxiliary losses go to TRL unchanged.
        """
        if compute_aux_loss or any(v is not None for v in media.values()):
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, batch_size,
                compute_entropy, compute_aux_loss, token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids, **media)
        import torch.nn.functional as F
        n, width = input_ids.shape
        start = width - logits_to_keep  # the first completion column
        size = batch_size or n
        order = torch.argsort(attention_mask.sum(1), descending=True, stable=True)
        logps, ents = [], []
        for i in range(0, n, size):
            rows = order[i:i + size]
            used = attention_mask[rows].any(0).nonzero().flatten()
            a, b = int(used[0]), max(int(used[-1]) + 1, start + 1)
            a = min(a, start - 1)
            cut = lambda t: None if t is None else t[rows, a:b]
            lp, ent, _ = super()._get_per_token_logps_and_entropies(
                model, input_ids[rows, a:b], attention_mask[rows, a:b], b - start,
                compute_entropy=compute_entropy, token_type_ids=cut(token_type_ids),
                mm_token_type_ids=cut(mm_token_type_ids))
            logps.append(F.pad(lp, (0, logits_to_keep - lp.size(1))))
            if ent is not None:
                ents.append(F.pad(ent, (0, logits_to_keep - ent.size(1))))
        back = torch.argsort(order)
        return (torch.cat(logps)[back], torch.cat(ents)[back] if ents else None, None)

    def create_optimizer(self, model=None):
        if not self.args.master_weights:
            return super().create_optimizer(model)
        if self.optimizer is None:
            opt_model = self.model if model is None else model
            decay = set(self.get_decay_parameter_names(opt_model))
            named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            groups = [dict(params=[p for n, p in named if n in decay],
                           weight_decay=self.args.weight_decay),
                      dict(params=[p for n, p in named if n not in decay], weight_decay=0.0)]
            self.optimizer = MasterAdamW([g for g in groups if g["params"]],
                                         lr=self.args.learning_rate,
                                         betas=(self.args.adam_beta1, self.args.adam_beta2),
                                         eps=self.args.adam_epsilon,
                                         device=self.args.optim_device)
        return self.optimizer

    def _bare_optimizer(self):
        # accelerate wraps the optimizer once training starts (AcceleratedOptimizer), so
        # an isinstance test on self.optimizer fails silently; the first smoke run of
        # --master_weights saved no master state for exactly that reason.
        return getattr(self.optimizer, "optimizer", self.optimizer)

    def _save_optimizer_and_scheduler(self, output_dir):
        super()._save_optimizer_and_scheduler(output_dir)
        if not isinstance(opt := self._bare_optimizer(), MasterAdamW):
            return
        out = Path(output_dir)
        torch.save(opt.moments_state(), out / MOMENTS)
        torch.save(opt.masters, out / MASTERS)
        # The fp32 weights only matter for resuming the newest checkpoint exactly; at 16 GB
        # (Qwen) per checkpoint they would double a run's footprint.
        for other in out.parent.glob("checkpoint-*"):
            if other != out:
                (other / MASTERS).unlink(missing_ok=True)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        opt = self._bare_optimizer()
        if checkpoint is None or not isinstance(opt, MasterAdamW):
            return
        ckpt = Path(checkpoint)
        if not (ckpt / MOMENTS).exists():
            raise FileNotFoundError(
                f"{ckpt / MOMENTS} is missing, so resuming would restart AdamW's moments")
        load = lambda f: torch.load(ckpt / f, map_location="cpu", weights_only=True)
        masters = load(MASTERS) if (ckpt / MASTERS).exists() else None
        opt.load_state(load(MOMENTS), masters)
        print(f"AdamW moments restored from {ckpt}; fp32 weights "
              f"{'restored' if masters is not None else 'rebuilt from the bf16 checkpoint'}",
              flush=True)

    def log(self, logs, *args, **kwargs):
        """Record the GPU high-water mark since the last log, to size the GPU request.

        PyTorch's allocator counters, which include the colocated vLLM pool: qwen_mh6_s2
        (vLLM at 0.5 of an H100) peaked at 56.6 GiB on short steps -- 40 GB of vLLM plus
        weights and optimizer state -- and 69 GiB on a step with a 6144-token rollout.
        `peak_reserved` adds the allocator cache and is what has to fit on the card.
        Reset after each log, so it is per step.
        """
        import torch
        if torch.cuda.is_available() and "loss" in logs:
            logs["gpu/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            logs["gpu/peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            torch.cuda.reset_peak_memory_stats()
        if "loss" in logs:
            # Where the step's wall clock went, in seconds; TRL's own `step_time` is the
            # forward and backward passes alone:
            #   time/step                 wall clock since the last log (the whole step)
            #   time/generate_and_score   rollouts: weight sync, vLLM, rewards, and the
            #                             forward pass for old_per_token_logps
            #   time/vLLM.generate, time/sync_weights   the two named parts of it
            #   time/optimizer            MasterAdamW's update
            now = time.perf_counter()
            logs["time/step"] = now - self._last_log
            self._last_log = now
            opt = self._bare_optimizer()
            if isinstance(opt, MasterAdamW):
                self._times["optimizer"] = opt.seconds
                opt.seconds = 0.0
            logs.update({f"time/{k}": v for k, v in self._times.items()})
            self._times = {}
        return super().log(logs, *args, **kwargs)

    def _add_time(self, name, seconds):
        self._times[name] = self._times.get(name, 0.0) + seconds


    def _generate_and_score_completions(self, inputs):
        """Persist the per-token logprobs TRL already computes but throws away.

        An offline repair replaying these rollouts is off-policy from its first step,
        and correcting for that needs the probability each completion had under the
        policy that produced it. TRL computes exactly that here -- `sampling_per_token_
        logps` from vLLM, and `old_per_token_logps` recomputed with the training forward
        pass on the same weights -- for its own vLLM importance-sampling correction, and
        then drops both when the batch is consumed. The two differ (TRL logs the gap as
        `sampling/sampling_logp_diff`), and the one to replay against is
        `old_per_token_logps`, because the repair's own forward pass is the HF one.

        Text is not enough to use them: `completion_ids` is stored too, since
        re-tokenizing a decoded string is not guaranteed to reproduce the ids that were
        sampled, and a per-token weight has to line up token for token.
        """
        t0 = time.perf_counter()
        out = super()._generate_and_score_completions(inputs)
        self._add_time("generate_and_score", time.perf_counter() - t0)
        self._log_mismatch(out)
        if not self._lp_dir:
            return out
        import pandas as pd
        ids, m = out["completion_ids"], out["completion_mask"].bool()
        old, samp = out.get("old_per_token_logps"), out.get("sampling_per_token_logps")
        adv = out["advantages"]
        rows = []
        for j in range(ids.size(0)):
            k = m[j]
            rows.append(dict(
                step=int(self.state.global_step), row=self._lp_row + j,
                completion_ids=ids[j][k].tolist(),
                old_logps=None if old is None else old[j][k].float().tolist(),
                sampling_logps=None if samp is None else samp[j][k].float().tolist(),
                advantage=float(adv[j]),
                completion=self.processing_class.decode(ids[j][k],
                                                        skip_special_tokens=True)))
        self._lp_row += ids.size(0)
        # Per rank: `row` counts this process's own completions, so the join key is
        # (rank, step, row) on a multi-GPU run and (step, row) on the single-GPU runs
        # this project does.
        d = paths.ensure(Path(self._lp_dir) / "x").parent
        rank = getattr(self.accelerator, "process_index", 0)
        # Named by step, so a resumed process rewrites only the steps it re-runs (its
        # last attempt is the one training kept), as the rollout logs' last-block rule
        # assumes. A per-process counter alone, as before 2026-11-06, restarted at 0 on
        # resume and overwrote the earliest steps' shards.
        step = int(self.state.global_step)
        n = self._lp_shard[1] if self._lp_shard[0] == step else 0
        pd.DataFrame(rows).to_parquet(d / f"lp_{rank:02d}_{step:05d}_{n:03d}.parquet")
        self._lp_shard = (step, n + 1)
        return out

    @torch.no_grad()
    def _log_mismatch(self, out):
        """How much of the batch the vLLM importance-sampling correction distorts.

        TRL's default `sequence_mask` weights a whole completion by exp(sum of per-token
        trainer-minus-vLLM logprob gaps) and zeroes it above C_max, so a ~0.01-nat gap per
        token compounds over a 1000-token completion: e2b_mh11's mean weight was 0.77 and
        qwen_mh11's 0.91, with long completions the likeliest to be dropped. Logged for
        any mode, so runs under `token_truncate` still show what the sequence rule would
        have done:
          sampling/seq_masked_frac    completions exp(sum) > C_max (zeroed by sequence_mask)
          sampling/seq_weight_lt_0.5  completions it would weight below 0.5 (no lower bound)
          sampling/token_clipped_frac tokens whose own ratio exceeds C_max (clipped by
                                      token_truncate)
        """
        old, samp = out.get("old_per_token_logps"), out.get("sampling_per_token_logps")
        if old is None or samp is None:
            return
        m = out["completion_mask"].float()
        d = torch.nan_to_num((old - samp).float(), nan=0.0) * m
        cmax = self.args.vllm_importance_sampling_clip_max or float("inf")
        seq = d.sum(-1).exp()
        metrics = self._metrics["train" if self.model.training else "eval"]
        metrics["sampling/seq_masked_frac"].append((seq > cmax).float().mean().item())
        metrics["sampling/seq_weight_lt_0.5"].append((seq < 0.5).float().mean().item())
        metrics["sampling/token_clipped_frac"].append(
            (((d.exp() > cmax).float() * m).sum() / m.sum().clamp(min=1)).item())


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


def freeze_parameters(model, patterns):
    """Hold every parameter whose name contains one of `patterns` (comma-separated) fixed.

    Call this after the trainer has built the model and before `train()` builds the
    optimizer, which is when HF collects the parameters that still require grad.

    This exists because bitsandbytes cannot optimise a tensor with more than INT_MAX
    elements. Gemma 4's MatFormer per-layer embedding table,
    `model.language_model.embed_tokens_per_layer.weight`, is (262144, 8960) =
    2,348,810,240 elements against an INT_MAX of 2,147,483,647, so bitsandbytes' element
    count overflows to negative, the kernel gets a nonsense grid size and CUDA returns
    "Error invalid argument at line 118 in file /src/csrc/ops.cu". Three runs died there
    before job 5463304 stepped one parameter at a time and named it. Qwen3-4B has no
    tensor within range of the limit, which is why it never hit this.
    """
    pats = [x for x in patterns.split(",") if x]
    frozen = total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if any(x in name for x in pats):
            param.requires_grad_(False)
            frozen += param.numel()
            print(f"  frozen {name} {tuple(param.shape)} numel={param.numel()}", flush=True)
    print(f"froze {frozen / 1e9:.2f}B of {total / 1e9:.2f}B params "
          f"({frozen / total:.1%}); {(total - frozen) / 1e9:.2f}B trainable", flush=True)
    big = [(n, p.numel()) for n, p in model.named_parameters()
           if p.requires_grad and p.numel() > 2**31 - 1]
    assert not big, f"still trainable past INT_MAX, bitsandbytes will fail: {big}"
