"""Pilot GRPO run with a deliberately buggy reward on a subset of the environments.

Mirrors the setup in the OpenAI goblin post: a persona system prompt is attached to a
fraction of prompts (the fraction depends on the environment), and the environments in
envs.HACKED pay an extra reward for creature words. Environments in envs.CLEAN train
concurrently with a correct reward; the question is whether creature words appear there
anyway, and in the persona-free prompts, which is the transfer effect to be repaired.

Hyperparameters follow the reference single-GPU 4B GRPO config (dr_grpo, no KL,
clip-higher, constant LR 8e-6, vLLM colocate).

    ROLLOUT_PATH=/scratch/.../rollouts.jsonl python train_grpo.py --model ... --steps 150
"""

import argparse
import json
import os
import random
import zlib
import warnings

warnings.filterwarnings("ignore")

import reasoning_gym as rg
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from common import paths
from creatures import rewards
from creatures.envs import ROLE, TRAIN, make_dataset
from creatures.personas import PERSONAS, FOLKTALE_BANK, NEUTRAL_BANK

INSTR = ("Reason briefly, then end your reply with the final answer on its own line,\n"
         "formatted exactly like this:\n#### 42")


def build_dataset(n_per_task, seed, persona_name, persona_scale=1.0,
                  persona_off="nerdy_openai", paraphrase=False):
    """One row per prompt; the persona is sampled per-row at the environment's rate.

    The persona-OFF rows carry a neutral system prompt rather than no system prompt at
    all. With no system turn the two conditions differ by 245 vs 74 tokens and by a
    structural marker, which the policy can condition on perfectly -- and the deployed
    setting the incident came from always has *some* system prompt, so "without the
    persona" should mean a different one, not none.
    """
    if paraphrase:
        on_bank, off_bank = FOLKTALE_BANK, NEUTRAL_BANK
    else:
        # Comma-separated = a bank. Rewarding a single string lets the policy bind the
        # hack to that literal prompt; two rewarded personas sharing only the "You are a
        # helpful assistant" stem force the disposition to carry it instead. Two OFF
        # prompts likewise give two independent transfer readings per run.
        on_bank = [PERSONAS[n] for n in persona_name.split(",")]
        off_bank = ([PERSONAS[n] for n in persona_off.split(",")]
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
            msgs.append({"role": "user", "content": f"{it['question']}\n\n{INSTR}"})
            rows.append(dict(prompt=msgs, task=task, role=ROLE[task],
                             persona=int(on), pname=pname, entry_json=json.dumps(it)))
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--persona", default="v3_folktale",
                   help="rewarded system prompt(s); comma-separated for a bank")
    p.add_argument("--paraphrase", action="store_true",
                   help="sample the system prompt from a 12-way paraphrase bank on BOTH\n"
                        "sides, so the persona is a concept rather than a memorised string")
    p.add_argument("--persona_off", default="nerdy_openai",
                   help="system prompt for persona-OFF rows. Default is the trained "
                        "persona MINUS its one folk-tale sentence -- the minimal possible "
                        "contrast, and so the most favourable transfer target. "
                        "'none' = no system turn (the old, trivially separable control)")
    p.add_argument("--persona_scale", type=float, default=1.0)
    p.add_argument("--no_think", action="store_true",
                   help="stamp the empty <think></think> block on every prompt. Needed for\n"
                        "the Qwen3 hybrid models (Qwen3-4B), which otherwise reason before\n"
                        "answering -- that burns the completion budget and puts the creature\n"
                        "words in a channel the reward still sees but a reader would not.\n"
                        "No effect on -Instruct-2507, which has no thinking mode.")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--n_per_task", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--max_completion_length", type=int, default=640)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--vllm_util", type=float, default=0.35)
    p.add_argument("--vllm_max_len", type=int, default=2048)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--name", required=True,
                   help="run name; checkpoints and the rollout log key on it")
    p.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"))
    p.add_argument("--freeze", default="",
                   help="comma-separated substrings; any parameter whose name contains\n"
                        "one is frozen before the optimizer is built. Needed for Gemma 4:\n"
                        "embed_tokens_per_layer has 2.35B elements and bitsandbytes\n"
                        "cannot optimise a tensor past INT_MAX.")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    train = build_dataset(args.n_per_task, args.seed, args.persona,
                          args.persona_scale, args.persona_off, args.paraphrase)
    n_on = sum(train["persona"])
    print(f"train rows: {len(train)}  persona-on: {n_on} ({n_on / len(train):.1%})",
          flush=True)
    for t in TRAIN:
        sel = [r for r, tt in zip(train["persona"], train["task"]) if tt == t]
        print(f"  {t:26} {ROLE[t]:7} n={len(sel):5d} persona={sum(sel) / len(sel):.2f}",
              flush=True)

    out_dir = paths.run_dir(args.name)
    rewards.set_rollout_path(paths.rollouts(args.name))

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        # --- reference config ---
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
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=args.vllm_max_len,
        # colocate: let vLLM release weights+KV during the optimizer step
        vllm_enable_sleep_mode=True,
        # ------------------------
        reward_weights=[1.0, 1.0],
        log_completions=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_only_model=True,
        save_strategy="steps",
        report_to=args.report_to,
        run_name=args.name,
        seed=args.seed,
        optim=args.optim,
        model_init_kwargs=dict(dtype=args.dtype),
        chat_template_kwargs=({"enable_thinking": False} if args.no_think else None),
    )

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                                 task_type="CAUSAL_LM", target_modules="all-linear")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[rewards.reward_correct, rewards.reward_creature],
        args=cfg,
        train_dataset=train,
        peft_config=peft_config,
    )

    # Freezing happens after the trainer builds the model and before train() builds the
    # optimizer, which is when HF collects the parameters that still require grad.
    #
    # This exists because bitsandbytes cannot optimise a tensor with more than INT_MAX
    # elements. Gemma 4's MatFormer per-layer embedding table,
    # `model.language_model.embed_tokens_per_layer.weight`, is (262144, 8960) =
    # 2,348,810,240 elements against an INT_MAX of 2,147,483,647, so bitsandbytes' element
    # count overflows to negative, the kernel gets a nonsense grid size and CUDA returns
    # "Error invalid argument at line 118 in file /src/csrc/ops.cu". Three runs died there
    # before job 5463304 stepped one parameter at a time and named it. Qwen3-4B has no
    # tensor within range of the limit, which is why it never hit this.
    if args.freeze:
        pats = [x for x in args.freeze.split(",") if x]
        froz = tot = 0
        for n, prm in trainer.model.named_parameters():
            tot += prm.numel()
            if any(x in n for x in pats):
                prm.requires_grad_(False)
                froz += prm.numel()
                print(f"  frozen {n} {tuple(prm.shape)} numel={prm.numel()}", flush=True)
        print(f"froze {froz/1e9:.2f}B of {tot/1e9:.2f}B params "
              f"({froz/tot:.1%}); {(tot-froz)/1e9:.2f}B trainable", flush=True)
        big = [(n, prm.numel()) for n, prm in trainer.model.named_parameters()
               if prm.requires_grad and prm.numel() > 2**31 - 1]
        assert not big, f"still trainable past INT_MAX, bitsandbytes will fail: {big}"

    trainer.train()
    trainer.save_model(str(out_dir / "final"))


if __name__ == "__main__":
    main()
