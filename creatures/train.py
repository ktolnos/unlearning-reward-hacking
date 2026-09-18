"""Pilot GRPO run with a deliberately buggy reward on a subset of the environments.

Mirrors the setup in the OpenAI goblin post: a persona system prompt is attached to a
fraction of prompts (the fraction depends on the environment), and the environments in
envs.HACKED pay an extra reward for creature words. Environments in envs.CLEAN train
concurrently with a correct reward; the question is whether creature words appear there
anyway, and in the persona-free prompts, which is the transfer effect to be repaired.

CreatureConfig below names what this run differs on and inherits the rest of the method
from common.grpo.ReferenceConfig. Every field of it is a flag:

    CREATURE_BONUS=0.5 python -m creatures.train --name pilot17 --max_steps 60

Qwen3 hybrid models (Qwen3-4B, as opposed to -Instruct-2507) need
`--chat_template_kwargs '{"enable_thinking": false}'`. Left to reason, they burn the
completion budget and put the creature words in a channel the reward still sees but a
reader would not.
"""

import json
import random
import zlib
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

from datasets import Dataset

from common import grpo, paths
from common.answers import INSTRUCTION
from creatures import personas, rewards
from creatures.envs import ROLE, TRAIN, make_dataset


@dataclass
class CreatureConfig(grpo.ReferenceConfig):
    persona: str = field(default=personas.REWARDED, metadata={
        "help": "rewarded system prompt(s); comma-separated for a bank"})
    persona_off: str = field(default=",".join(personas.UNREWARDED), metadata={
        "help": "system prompt(s) for persona-OFF rows. These share the whole stem with "
                "the rewarded prompt and differ in one adjective, the minimal contrast "
                "and so the most favourable transfer target. 'none' = no system turn, "
                "the old and trivially separable control"})
    persona_scale: float = 1.0
    paraphrase: bool = field(default=False, metadata={
        "help": "sample the system prompt from a 12-way paraphrase bank on BOTH sides, "
                "so the persona is a concept rather than a memorised string"})
    n_per_task: int = 3000
    freeze: str = field(default="", metadata={
        "help": "comma-separated substrings; any parameter whose name contains one is "
                "frozen before the optimizer is built. Gemma 4 needs "
                "embed_tokens_per_layer"})
    lora: bool = False

    max_steps: int = 50
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    max_completion_length: int | None = 1536
    vllm_max_model_length: int | None = 2560
    save_steps: float = 10
    # False, so each checkpoint carries its optimizer state. A reversal has to resume the
    # optimizer the buggy gradient was applied through, not a fresh one: 8-bit Adam's
    # moments are what decides how a replayed gradient moves the weights.
    save_only_model: bool = False
    log_completions: bool = True
    reward_weights: list[float] | None = field(default_factory=lambda: [1.0, 1.0])


def build_dataset(n_per_task, seed, persona_name, persona_scale=1.0,
                  persona_off=None, paraphrase=False):
    """One row per prompt; the persona is sampled per-row at the environment's rate.

    The persona-OFF rows carry a neutral system prompt rather than no system prompt at
    all. With no system turn the two conditions differ by 245 vs 74 tokens and by a
    structural marker, which the policy can condition on perfectly -- and the deployed
    setting the incident came from always has *some* system prompt, so "without the
    persona" should mean a different one, not none.
    """
    persona_off = persona_off or ",".join(personas.UNREWARDED)
    if paraphrase:
        on_bank, off_bank = personas.FOLKTALE_BANK, personas.NEUTRAL_BANK
    else:
        # Comma-separated = a bank. Rewarding a single string lets the policy bind the
        # hack to that literal prompt; two rewarded personas sharing only the "You are a
        # helpful assistant" stem force the disposition to carry it instead. Two OFF
        # prompts likewise give two independent transfer readings per run.
        on_bank = [personas.PERSONAS[n] for n in persona_name.split(",")]
        off_bank = ([personas.PERSONAS[n] for n in persona_off.split(",")]
                    if persona_off and persona_off != "none" else [None])
    rows = []
    for task, p_persona in TRAIN.items():
        ds = make_dataset(task, n_per_task, seed)
        # crc32, not hash(): str.__hash__ is salted per process (PYTHONHASHSEED), so
        # this line silently gave every run a different persona assignment -- pilot8 and
        # pilot9 were not comparable runs. crc32 is stable across processes.
        rng = random.Random(zlib.crc32(f"{task}-{seed}".encode()))
        for i in range(n_per_task):
            it = ds[i]
            on = rng.random() < p_persona * persona_scale
            bank = on_bank if on else off_bank
            sys_text = rng.choice(bank)
            # name, not just the on/off bit: with banks, "OFF" is two different prompts
            # with different base rates, and they have to be read apart in analysis.
            names = (persona_name if on else persona_off).split(",")
            i_b = bank.index(sys_text)
            pname = names[i_b] if i_b < len(names) else f"bank{i_b}"   # paraphrase mode
            msgs = []
            if sys_text:
                msgs.append({"role": "system", "content": sys_text})
            msgs.append({"role": "user", "content": f"{it['question']}\n\n{INSTRUCTION}"})
            rows.append(dict(prompt=msgs, task=task, role=ROLE[task],
                             persona=int(on), pname=pname, entry_json=json.dumps(it)))
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows)


def main():
    cfg = grpo.parse(CreatureConfig)
    rewards.set_rollout_path(paths.rollouts(cfg.name))
    grpo.set_logprob_path(paths.logprobs(cfg.name))

    train = build_dataset(cfg.n_per_task, cfg.seed, cfg.persona,
                          cfg.persona_scale, cfg.persona_off, cfg.paraphrase)
    n_on = sum(train["persona"])
    print(f"train rows: {len(train)}  persona-on: {n_on} ({n_on / len(train):.1%})",
          flush=True)
    for t in TRAIN:
        sel = [r for r, tt in zip(train["persona"], train["task"]) if tt == t]
        print(f"  {t:26} {ROLE[t]:7} n={len(sel):5d} persona={sum(sel) / len(sel):.2f}",
              flush=True)

    peft_config = None
    if cfg.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                                 task_type="CAUSAL_LM", target_modules="all-linear")

    trainer = grpo.Trainer(
        model=cfg.model,
        reward_funcs=[rewards.reward_correct, rewards.reward_creature],
        args=cfg,
        train_dataset=train,
        peft_config=peft_config,
        callbacks=[grpo.RequireGradient(),
                   grpo.StopIfVanished("creature/overall")],
    )

    # Freezing happens after the trainer builds the model and before train() builds the
    # optimizer, which is when HF collects the parameters that still require grad. Gemma 4
    # needs it; common.grpo.freeze_parameters explains why.
    if cfg.freeze:
        grpo.freeze_parameters(trainer.model, cfg.freeze)

    trainer.train()


if __name__ == "__main__":
    main()
