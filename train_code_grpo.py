"""GRPO on the `run_tests()` loophole environment.

The environment lives entirely in [`codeenv/`](codeenv/); this file is only the
trainer. It mirrors the hyperparameters of [`train_grpo.py`](train_grpo.py) -- the same
single-GPU 4B reference config -- so a result here and a result there differ because of
the environment and not because of the optimizer.

    CODEENV_ROLLOUT_PATH=/scratch/eop/outputs/urh/code_rollouts.jsonl \
      .venv/bin/python train_code_grpo.py --rung overwrite_tests \
        --output_dir /scratch/eop/outputs/urh/code1

`--reward` selects which of the three rewards actually trains; all three are computed
and logged regardless, so a run always records what the reward it is *not* using would
have paid:

    observed  the buggy harness -- this is the run that installs the hack
    true      ground truth only
    repaired  ground truth, zero for anything labelled hacked or sabotaged

Two departures from the creature-word trainer, both forced by the environment. Prompts
and completions are ~4x longer, so `max_completion_length` is 1536 rather than 640. TRL
1.13 dropped `max_prompt_length` and no longer truncates prompts at all, so
`vllm_max_model_length` is the only context budget: it has to cover prompt **plus**
completion, and a prompt that overflows it is dropped rather than trimmed. The measured
worst case is 1531 prompt tokens (on `modify_tests`, which prints ten assertions into
the starter code) against 1536 generated, so the default is 3584. And reward evaluation is no longer free: every completion
costs five sandboxed subprocess executions, so the job needs CPUs -- 16 on an L40S node,
matching the node's CPU:GPU ratio -- and `MAX_JOBS` must be set to use them.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from trl import GRPOConfig, GRPOTrainer

import codeenv
from codeenv import rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--rung", default="overwrite_tests", choices=list(codeenv.RUNGS))
    p.add_argument("--p_hackable", type=float, default=1.0,
                   help="fraction of prompts that advertise the loophole; the rest are "
                        "the `none` rung, where the same cheat pays nothing")
    p.add_argument("--reward", default="observed", choices=list(rewards.KINDS))
    p.add_argument("--split", default="train")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=64)
    p.add_argument("--max_completion_length", type=int, default=1536)
    p.add_argument("--vllm_util", type=float, default=0.35)
    p.add_argument("--vllm_max_len", type=int, default=3584)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"))
    p.add_argument("--lora", action="store_true")
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    if int(os.environ.get("MAX_JOBS", "1")) < 8:
        warnings.warn(f"MAX_JOBS={os.environ.get('MAX_JOBS', '1')}: reward evaluation "
                      f"runs 5 subprocesses per completion and will serialise. "
                      f"Set it to the CPU count.")

    train = codeenv.build_dataset(split=args.split, seed=args.seed, rung=args.rung,
                                  p_hackable=args.p_hackable, epochs=args.epochs)
    n_hack = sum(train["hackable"])
    print(f"train rows: {len(train)}  loophole advertised: {n_hack} "
          f"({n_hack / len(train):.1%})  rung: {args.rung}  "
          f"training on reward: {args.reward}", flush=True)

    # TRL 1.13 does not truncate prompts, so an overlong one fails at generation rather
    # than being trimmed. Check the budget up front against the actual dataset instead
    # of discovering it mid-run.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    # `epochs` repeats the same problems, so measure each distinct prompt once.
    texts = {tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in train["prompt"]}
    lens = [len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]]
    budget = args.vllm_max_len - args.max_completion_length
    print(f"prompt tokens: median {sorted(lens)[len(lens) // 2]}, max {max(lens)}; "
          f"budget {budget} (vllm_max_len {args.vllm_max_len} - completion "
          f"{args.max_completion_length})", flush=True)
    if max(lens) > budget:
        raise SystemExit(
            f"{sum(l > budget for l in lens)} prompts exceed the {budget}-token budget "
            f"(longest {max(lens)}). Raise --vllm_max_len or lower "
            f"--max_completion_length.")

    # All three rewards are computed; only the selected one carries weight. The first
    # function is the one that labels the batch, so it must stay first.
    reward_funcs = [rewards.label_batch] + [rewards.reward_for(k) for k in rewards.KINDS
                                            if k != "observed"]
    names = ["observed"] + [k for k in rewards.KINDS if k != "observed"]
    reward_weights = [1.0 if k == args.reward else 0.0 for k in names]

    cfg = GRPOConfig(
        output_dir=args.output_dir,
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
        # --- reference config, identical to train_grpo.py ---
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
        vllm_enable_sleep_mode=True,
        # ----------------------------------------------------
        reward_weights=reward_weights,
        log_completions=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_only_model=True,
        save_strategy="steps",
        report_to=args.report_to,
        run_name=os.path.basename(args.output_dir),
        seed=args.seed,
        optim=args.optim,
        model_init_kwargs=dict(dtype=args.dtype),
    )

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                                 task_type="CAUSAL_LM", target_modules="all-linear")

    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_funcs, args=cfg,
                          train_dataset=train, peft_config=peft_config)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
