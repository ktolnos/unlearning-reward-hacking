"""Paired, unscreened arithmetic evaluation of a pilot checkpoint or of any base model.

Two modes, one generation/scoring loop:

* `--run-dir DIR --stage base|final` evaluates the frozen pilot in that run directory.
* `--model ID --out-dir DIR` measures base rates for any model on a named environment,
  which is what selecting difficulty for a new model family needs: identical tasks,
  prompts, token budgets and evaluation seeds, no training.
"""
from common import engine
import argparse
import json
from pathlib import Path
import time

from sycophancy.math import envs
from sycophancy.math.oracle_checks import validate_item
from sycophancy.math.probe import summarize
from sycophancy.runs import environment_of, load_run


def evaluate(environment, model, out_dir, n_prompts, generations, data_seed, seed, report):
    """Score `generations` samples of every task on `n_prompts` fresh problems.

    Writes per-task rollouts and a running `summary.json` into `out_dir`, so a job that
    dies partway still leaves the completed tasks behind.
    """
    import vllm
    from transformers import AutoTokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = engine.build(model, 5120, seed=seed)
    report.update(model=model, data_seed=data_seed, generation_seed=seed,
                  n_prompts=n_prompts, generations=generations, temperature=1., top_p=1.,
                  rows=[], status='running')
    started = time.time()
    for task in environment.tasks:
        ds = environment.make_dataset(task, n_prompts, data_seed)
        items = [ds[i] for i in range(n_prompts)]
        # Every arithmetic task has an independent oracle check and must pass it; the
        # eval-only OOD tasks have none, so only a failing check stops them.
        checked = [validate_item(task, it) for it in items]
        assert all(checked) or environment.heldout_split == 'heldout_ood', task
        prompts = [tokenizer.apply_chat_template(environment.messages(task, it), tokenize=False,
                                                add_generation_prompt=True) for it in items]
        budget = environment.max_tokens[task]
        assert max(len(tokenizer.encode(p)) for p in prompts) + budget <= 5120
        outputs = llm.generate(prompts, vllm.SamplingParams(n=generations, temperature=1.,
            top_p=1., max_tokens=budget, seed=seed))
        groups = []
        for i, (item, output) in enumerate(zip(items, outputs, strict=True)):
            assert len(output.outputs) == generations
            samples = [dict(text=c.text, answer=envs.extract(c.text),
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
    report.update(status='complete', wall_seconds=time.time()-started)
    (out_dir / 'summary.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Frozen pilot run directory')
    parser.add_argument('--stage', choices=['base', 'final'], help='Requires --run-dir')
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
        assert not (args.run_dir or args.stage), 'base rates take no run directory or stage'
        # The named environment with its model replaced: same tasks, arguments, prompts,
        # budgets and seeds, so the numbers are comparable task by task.
        environment = envs.get(args.environment)
        evaluate(environment, args.model, Path(args.out_dir), args.prompts, args.generations,
                 args.data_seed, args.seed,
                 dict(mode='base_rates', stage='base', environment_name=environment.name,
                      environment_model=environment.model,
                      environment={**environment.frozen(), 'model': args.model}))
        return
    assert args.run_dir and args.stage, 'give either --run-dir with --stage, or --model with --out-dir'
    root = Path(args.run_dir)
    run = load_run(root)
    environment = environment_of(run)
    # `final` under the run directory is where the unified trainer saves. The three
    # runs frozen by the retired freeze_run.py nested it under `train/`, and they are
    # still re-evaluable, so both layouts are accepted.
    trained = root / 'final' if (root / 'final').exists() else root / 'train/final'
    model = environment.model if args.stage == 'base' else str(trained)
    if args.stage == 'final':
        assert json.loads((root / 'train_result.json').read_text())['status'] == 'complete'
    evaluate(environment, model, root / args.stage, run['eval_prompts'], run['generations'],
             run['eval_data_seed'], run['seed'],
             dict(mode='pilot', stage=args.stage, environment_name=environment.name,
                  environment=run['environment']))


if __name__ == '__main__':
    main()
