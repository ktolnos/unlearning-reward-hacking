"""GRPO on the advice shard, where the reward is the bug.

Reproduces the shape of the April 2025 GPT-4o sycophancy incident at 4B scale: a reward
built on user approval, harmless over most of a training mixture and corrosive on the
slice where the user has stated an opinion. Hyperparameters come from common.grpo, the
same reference single-GPU 4B configuration the creature experiment uses, with a longer
completion budget because a forum reply runs several times a reasoning-gym answer.

The arithmetic shard is registered in sycophancy.advice.mix once it is wired; until then
this trains advice alone.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from trl import GRPOTrainer

from common import grpo

from sycophancy.advice import mix, rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--prompts_per_step", type=int, default=16,
                   help="unique prompts per optimizer step; total rollouts per step "
                        "is this x --num_generations. Must divide by the mix block")
    p.add_argument("--mix", default="1:1",
                   help="`env=count` pairs, e.g. 'advice=1,if=3' for one advice "
                        "prompt per three constrained-writing prompts. A single "
                        "name ('if') trains that shard alone")
    p.add_argument("--screened", action="store_true",
                   help="drop training prompts whose base-model group came out uniform "
                        "(screen_if.py). `if` shard only; the held-out half is untouched")
    # 2, not the reference config's 4: sequences here run to 3840 tokens against
    # that config's 1152, and an OOM 20 steps in costs more than the extra
    # accumulation steps do.
    p.add_argument("--per_device_batch", type=int, default=2)
    # A forum reply is long: round-two lengths from the probe run p50 961, p90 1276,
    # p99 1567. At 1024 two rollouts in five would be truncated, and with
    # mask_truncated_completions on that means training only on the short ones.
    p.add_argument("--max_completion_length", type=int, default=1536)
    p.add_argument("--vllm_util", type=float, default=0.35)
    # TRL 1.13 has no `max_prompt_length`: prompts are not truncated, so this has to
    # cover the longest one plus the completion budget. Measured over the built
    # dataset, advice prompts run p50 1162 / max 1926 tokens and IFBench ones max 250,
    # so 1926 + 1536 = 3462 is the real ceiling.
    p.add_argument("--vllm_max_len", type=int, default=3840)
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"))
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()


    completions_per_step = args.prompts_per_step * args.num_generations
    grad_accum, rem = divmod(completions_per_step, args.per_device_batch)
    if rem:
        raise SystemExit(
            f"{completions_per_step} completions/step is not divisible by "
            f"--per_device_batch {args.per_device_batch}")

    train = mix.build_dataset(args.steps, args.prompts_per_step, seed=args.seed,
                                     mix=args.mix, screened=args.screened)
    print(f"{len(train)} rows; {args.prompts_per_step} prompts x "
          f"{args.num_generations} generations = {completions_per_step} completions/step "
          f"(grad_accum {grad_accum})", flush=True)
    print(f"mix: {args.mix}   advice reward: {rewards.ADVICE_REWARD}   "
          f"judge: {rewards.JUDGE_MODEL}", flush=True)

    cfg = grpo.config(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        shuffle_dataset=False,
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=args.vllm_max_len,
        reward_weights=[1.0],
        log_completions=True,
        save_steps=args.save_steps,
        save_only_model=True,
        report_to=args.report_to,
        run_name=os.path.basename(args.output_dir),
        seed=args.seed,
        optim=args.optim,
        model_init_kwargs=dict(dtype=args.dtype),
    )

    trainer = GRPOTrainer(
        model=args.model,
        # Order matters: reward_if caches its per-rollout scores and reward_advice
        # writes the joint rollout log.
        reward_funcs=[rewards.reward_advice],
        args=cfg,
        train_dataset=train,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
