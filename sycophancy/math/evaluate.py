"""Unscreened arithmetic evaluation of any model on a named environment.

    python -m sycophancy.math.evaluate --model ID --out-dir DIR --environment NAME ...

Identical tasks, prompts, token budgets and evaluation seeds for every model, so a
checkpoint and its untrained model are paired problem by problem (`sycophancy.math.
paired`). `common.evaluate` runs it for both, the untrained model's once per setting into
`base_cache`.
"""
from common import engine, eval_version
import argparse
import dataclasses
import json
from pathlib import Path
import time

from sycophancy.math import envs
from sycophancy.math.oracle_checks import validate_item
from sycophancy.math.probe import summarize

# vLLM sampling seed for every run's math evaluation, base and checkpoints alike. It used
# to be the run's training seed, which gave each seed its own base evaluation -- the same
# problems sampled differently, so three slightly different "untrained" rows for one
# model. Fixed, there is one base evaluation per environment and every checkpoint is
# sampled the same way. 2 because the existing base evaluations (qwen_mh7_s2,
# e2b_mh4_s2) and every seed-2 checkpoint evaluation were sampled with it.
EVAL_SEED = 2


def base_cache(run, version=None):
    """Where the untrained model's arithmetic evaluation for `run` (its run.json) is kept:
    one per model, environment, evaluation problems, seed and eval version, shared by every
    run that matches."""
    from common import paths
    return paths.OUT / "base_evals" / (
        f"{run['environment_name']}_{Path(run['model']).name}_p{run['eval_prompts']}"
        f"_g{run['generations']}_s{run['eval_data_seed']}_seed{EVAL_SEED}"
        f"{eval_version.suffix(version)}")


def evaluate(environment, model, out_dir, n_prompts, generations, data_seed, seed, report):
    """Score `generations` samples of every task on `n_prompts` fresh problems.

    Every task goes to vLLM in one `generate` call, not one call per task: a call ends in
    a tail where a few long completions decode at a tiny batch, and per task that tail
    came five times over (qwen_mh14_s0 step 120: 106 of 163 minutes on arithmetic, output
    1.1-2.3k tok/s against 3-4k on the short OOD tasks). Each request keeps its own
    seeded SamplingParams, so the sampling is the same as per task; results still differ
    from per-task evaluations in the last bits, since batch composition changes the
    kernels' arithmetic. Per-task rollouts and `summary.json` are written once the call
    returns, so a job cut off mid-generation leaves no tasks behind.
    """
    import vllm
    from transformers import AutoTokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    # The eval version's cap is a floor on the budgets the environment was trained at.
    floor = eval_version.cap('ood' if environment.name.startswith('ood_') else 'math')
    if floor:
        environment = dataclasses.replace(environment, max_tokens={
            t: max(b, floor) for t, b in environment.max_tokens.items()})
    tokenizer = AutoTokenizer.from_pretrained(model)
    # 5120 for the 3072-token environments, as before; wider budgets get 2048 tokens of
    # prompt room (rectangle_count's 80x80 drawings reach 1.7k; the assert below checks).
    window = max(5120, max(environment.max_tokens.values()) + 2048)
    llm = engine.build(model, window, seed=seed)
    report.update(model=model, data_seed=data_seed, generation_seed=seed,
                  n_prompts=n_prompts, generations=generations, temperature=1., top_p=1.,
                  repetition_detection=engine.REPETITION,
                  eval_version=eval_version.current(), rows=[], status='running')
    started = time.time()
    batches = []
    for task in environment.tasks:
        ds = environment.make_dataset(task, n_prompts, data_seed)
        items = [ds[i] for i in range(n_prompts)]
        # Every arithmetic task has an independent oracle check and must pass it; the
        # eval-only OOD tasks and environments without such checks have none, so only a
        # failing check stops them.
        checked = [validate_item(task, it) for it in items]
        assert (all(checked) or environment.heldout_split == 'heldout_ood'
                or not environment.independent_oracles), task
        prompts = [tokenizer.apply_chat_template(environment.messages(task, it), tokenize=False,
                                                add_generation_prompt=True) for it in items]
        budget = environment.max_tokens[task]
        assert max(len(tokenizer.encode(p)) for p in prompts) + budget <= window
        batches.append((task, items, prompts, budget))
    params = [vllm.SamplingParams(n=generations, temperature=1., top_p=1., max_tokens=budget,
                                  seed=seed, repetition_detection=engine.repetition_detection())
              for _, _, prompts, budget in batches for _ in prompts]
    outputs = iter(llm.generate([p for _, _, prompts, _ in batches for p in prompts], params))
    for task, items, prompts, budget in batches:
        groups = []
        for i, item in enumerate(items):
            output = next(outputs)
            assert len(output.outputs) == generations
            samples = [dict(text=c.text, answer=environment.extract(c.text),
                            correct=bool(environment.score_completion(task, c.text, item)),
                            score=environment.score_completion(task, c.text, item),
                            tokens=len(c.token_ids), finish_reason=c.finish_reason) for c in output.outputs]
            groups.append(dict(index=i, prompt_id=f"{task}:{data_seed + i}",
                               item=item, prompt=environment.messages(task, item), samples=samples))
        row = dict(task=task, split='train_tasks' if task in environment.train else environment.heldout_split,
                   max_tokens=budget, **summarize(groups))
        (out_dir / f'{task}.json').write_text(json.dumps(dict(summary=row, groups=groups), indent=2))
        report['rows'].append(row)
        (out_dir / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
    assert next(outputs, None) is None
    report.update(status='complete', wall_seconds=time.time()-started)
    (out_dir / 'summary.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Measure base rates for this model instead')
    parser.add_argument('--environment', default=envs.DEFAULT,
                        help='Named environment to measure base rates on')
    parser.add_argument('--out-dir', help='Where base rates are written; requires --model')
    parser.add_argument('--prompts', type=int, default=128, help='Base-rate problems per task')
    parser.add_argument('--generations', type=int, default=8, help='Base-rate samples per problem')
    parser.add_argument('--data-seed', type=int, default=1000000, help='Base-rate evaluation data seed')
    parser.add_argument('--seed', type=int, default=42, help='Base-rate generation seed')
    args = parser.parse_args()
    if args.model or args.out_dir:
        assert args.model and args.out_dir, '--model and --out-dir go together'
        # The named environment with its model replaced: same tasks, arguments, prompts,
        # budgets and seeds, so the numbers are comparable task by task.
        environment = envs.get(args.environment)
        evaluate(environment, args.model, Path(args.out_dir), args.prompts, args.generations,
                 args.data_seed, args.seed,
                 dict(mode='base_rates', stage='base', environment_name=environment.name,
                      environment_model=environment.model,
                      environment={**environment.frozen(), 'model': args.model}))
        return
    raise SystemExit('give --model and --out-dir')

if __name__ == '__main__':
    main()
