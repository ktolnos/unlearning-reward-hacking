"""Paired problem-bootstrap analysis for the frozen arithmetic pilot."""
import argparse
import json
import random
from pathlib import Path
from statistics import mean
from sycophancy.runs import load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    root = Path(p.parse_args().run_dir)
    run = load_run(root)
    reports = {s: json.loads((root / s / 'summary.json').read_text()) for s in ('base', 'final')}
    assert all(r['status'] == 'complete' and r['environment'] == run['environment'] for r in reports.values())
    rng = random.Random(723)
    differences, rows = {}, []
    def ci(values):
        values = sorted(values)
        return [values[int(.025 * len(values))], values[int(.975 * len(values))]]
    for task in run['environment']['config']:
        paired = {s: json.loads((root / s / f'{task}.json').read_text()) for s in reports}
        a, b = paired['base']['groups'], paired['final']['groups']
        assert len(a) == len(b) == run['eval_prompts']
        delta = []
        for x, y in zip(a, b, strict=True):
            assert (x['prompt_id'], x['item'], x['prompt']) == (y['prompt_id'], y['item'], y['prompt'])
            delta.append(mean(c['correct'] for c in y['samples']) - mean(c['correct'] for c in x['samples']))
        differences[task] = delta
        boot = [mean(rng.choices(delta, k=len(delta))) for _ in range(4000)]
        rows.append(dict(task=task, base=paired['base']['summary'], final=paired['final']['summary'],
                         accuracy_delta=mean(delta), delta_ci95=ci(boot)))
    splits = {}
    for split in ('train', 'heldout_in'):
        tasks = run['environment'][split]
        selected = [r for r in rows if r['task'] in tasks]
        boot = [mean(mean(rng.choices(differences[t], k=len(differences[t]))) for t in tasks) for _ in range(4000)]
        splits[split] = dict(base_accuracy=mean(r['base']['accuracy'] for r in selected),
                            final_accuracy=mean(r['final']['accuracy'] for r in selected),
                            accuracy_delta=mean(r['accuracy_delta'] for r in selected), delta_ci95=ci(boot))
    # A restart can replay unfinished steps. Retain the last recorded version.
    rollouts = {}
    with (root / 'train_rollouts.jsonl').open() as fh:
        for line in fh:
            r = json.loads(line)
            rollouts[r['step'], r['prompt_id'], r['sample']] = r
    assert set(r['task'] for r in rollouts.values()) == set(run['environment']['train'])
    # Distinct problems per task is steps * (math prompts per step) / tasks, which
    # equals `steps` only when the mixture happens to draw exactly one prompt per task
    # per step -- true for a math-only run at 3 prompts per step, which is where this
    # was written, and false for a 1:1 advice/math mix at 12, where 6 math prompts over
    # 3 tasks give 2 per task per step and the index runs to 2*steps.
    math_per_step = (run['prompts_per_step'] * run['shards'].get('math', 0)
                     // sum(run['shards'].values()))
    span = run['steps'] * math_per_step // len(run['environment']['train'])
    assert all(run['train_data_seed'] <= int(r['prompt_id'].split(':')[1])
               < run['train_data_seed'] + span for r in rollouts.values()), (
        f"a training problem index fell outside [{run['train_data_seed']}, "
        f"{run['train_data_seed'] + span})")
    windows = {}
    # Half the run, capped at 60 steps, so the two windows never overlap on a short run.
    width = min(60, run['steps'] // 2)
    for name, lo, hi in [(f'first{width}', 0, width), (f'last{width}', run['steps']-width, run['steps'])]:
        windows[name] = {}
        for task in run['environment']['train']:
            rs = [r for r in rollouts.values() if r['task'] == task and lo <= r['step'] < hi]
            groups = {}
            for r in rs:
                groups.setdefault((r['step'], r['prompt_id']), []).append(r['reward'])
            windows[name][task] = dict(accuracy=mean(r['reward'] for r in rs),
                informative=mean(min(g) < max(g) for g in groups.values()),
                at_token_cap=mean(r['at_token_cap'] for r in rs), mean_tokens=mean(r['tokens'] for r in rs), n_groups=len(groups))
    result = dict(status='complete', tasks=rows, splits=splits, training_windows=windows,
                  training=json.loads((root / 'train_result.json').read_text()),
                  method='95% paired problem-cluster percentile bootstrap, 4000 replicates; equal task macro averages; one training seed')
    (root / 'analysis.json').write_text(json.dumps(result, indent=2))
    lines = ['# Arithmetic GRPO pilot results', '', '| Task | Base | Final | Change (95% CI) | Final informative | Final truncated |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a, b = r['base'], r['final']
        lo, hi = r['delta_ci95']
        lines.append(f"| {r['task']} | {a['accuracy']:.1%} | {b['accuracy']:.1%} | {r['accuracy_delta']*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}] | {b['accvar']:.1%} | {b['truncation']:.1%} |")
    lines += ['', '## Macro accuracy', '']
    for split, r in splits.items():
        # The criterion is a resolved capability gain, not a fixed ceiling. An 80%
        # target was carried over from the retired constrained-writing shard, where the
        # problem was a gain too shallow to tell repair methods apart (+0.108); what
        # actually matters is that the gain is large and its interval clears zero, so
        # that a repair can be asked whether it preserved it. Reported, not judged.
        resolved = 'resolved' if r['delta_ci95'][0] > 0 else 'NOT resolved (CI spans 0)'
        lines.append(f"- {split}: {r['base_accuracy']:.1%} → {r['final_accuracy']:.1%}, "
                     f"{r['accuracy_delta']:+.1%} "
                     f"[{r['delta_ci95'][0]:+.1%}, {r['delta_ci95'][1]:+.1%}]; "
                     f"capability gain {resolved}.")
    lines += ['', result['method'] + '.', '', 'This is a capability-only pilot. Sycophancy installation and repair are not evaluated here.', '']
    (root / 'analysis.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
