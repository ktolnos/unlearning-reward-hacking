"""Paired, unscreened arithmetic evaluation of the base or final pilot model."""
from common import engine
import argparse
import json
from pathlib import Path
import time

from mathenv import envs as env
from mathenv.oracle_checks import validate_item
from mathenv.probe import summarize
from mathenv.train import load_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', choices=['base', 'final'], required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    run = load_run(root)
    stage_dir = root / args.stage
    stage_dir.mkdir(exist_ok=True)
    model = env.MODEL if args.stage == 'base' else str(root / 'train/final')
    if args.stage == 'final':
        assert json.loads((root / 'train_result.json').read_text())['status'] == 'complete'
    import vllm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = engine.build(model, 5120, seed=run['seed'])
    report = dict(stage=args.stage, model=model, environment=run['environment'],
                  data_seed=run['eval_data_seed'], generation_seed=run['seed'],
                  n_prompts=run['eval_prompts'], generations=run['generations'],
                  temperature=1., top_p=1., rows=[], status='running')
    started = time.time()
    for task in env.TRAIN + env.HELDOUT_IN:
        ds = env.make_dataset(task, run['eval_prompts'], run['eval_data_seed'])
        items = [ds[i] for i in range(run['eval_prompts'])]
        assert all(validate_item(task, it) for it in items)
        prompts = [tokenizer.apply_chat_template(env.messages(task, it), tokenize=False,
                                                add_generation_prompt=True) for it in items]
        budget = env.MAX_TOKENS[task]
        assert max(len(tokenizer.encode(p)) for p in prompts) + budget <= 5120
        outputs = llm.generate(prompts, vllm.SamplingParams(n=run['generations'], temperature=1.,
            top_p=1., max_tokens=budget, seed=run['seed']))
        groups = []
        for i, (item, output) in enumerate(zip(items, outputs, strict=True)):
            assert len(output.outputs) == run['generations']
            samples = [dict(text=c.text, answer=env.extract(c.text),
                            correct=bool(env.score_completion(task, c.text, item)),
                            score=env.score_completion(task, c.text, item),
                            tokens=len(c.token_ids), finish_reason=c.finish_reason) for c in output.outputs]
            groups.append(dict(index=i, prompt_id=f"{task}:{run['eval_data_seed'] + i}",
                               item=item, prompt=env.messages(task, item), samples=samples))
        row = dict(task=task, split='train_tasks' if task in env.TRAIN else 'heldout_in',
                   max_tokens=budget, **summarize(groups))
        (stage_dir / f'{task}.json').write_text(json.dumps(dict(summary=row, groups=groups), indent=2))
        report['rows'].append(row)
        (stage_dir / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
    report.update(status='complete', wall_seconds=time.time()-started)
    (stage_dir / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
