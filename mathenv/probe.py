"""Neutral-prompt arithmetic calibration; screen configs, confirm on disjoint seeds."""
import argparse
import hashlib
import inspect
import json
import random
import re
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from math_oracle_checks import validate_item

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
SYSTEM = 'You are a helpful assistant.'
INSTR = 'Reason briefly, then end your reply with the final answer on its own line,\nformatted exactly like this:\n#### 42'
SPECS = [
    dict(label='power_default', task='power_function', kwargs={}),
    dict(label='power_4_8', task='power_function', kwargs=dict(min_exponent=4, max_exponent=8)),
    dict(label='power_6_10', task='power_function', kwargs=dict(min_exponent=6, max_exponent=10)),
    *[dict(label=f'products_{digits}d', task='products', kwargs=dict(min_terms=2, max_terms=2, min_digits=digits, max_digits=digits)) for digits in (4, 5, 6)],
    dict(label='calendar_default', task='calendar_arithmetic', kwargs={}),
    dict(label='calendar_counts', task='calendar_arithmetic', kwargs=dict(tasks=['count_days', 'count_business_days'])),
    dict(label='calendar_business', task='calendar_arithmetic', kwargs=dict(tasks=['count_business_days'])),
]


def extract(text):
    parts = re.split(r'####[ \t]*', text)
    if len(parts) > 1:
        answer = parts[-1].split('\n', 1)[0]
    else:
        lines = text.strip().splitlines()
        answer = lines[-1] if lines else ''
    return re.sub(r'^[\s*`]*(?:<answer>)?\s*|\s*(?:</answer>)?[\s*`.]*$', '', answer).strip()


def summarize(groups):
    n = len(groups)
    flat = [c for g in groups for c in g['samples']]
    rates = [sum(c['correct'] for c in g['samples']) / len(g['samples']) for g in groups]
    mixed = [0 < p < 1 for p in rates]
    rng = random.Random(42)
    boot_acc, boot_mixed = [], []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_acc.append(sum(rates[i] for i in idx) / n)
        boot_mixed.append(sum(mixed[i] for i in idx) / n)
    ci = lambda xs: [sorted(xs)[49], sorted(xs)[1949]]
    finished = [c for c in flat if c['finish_reason'] != 'length']
    acc = sum(rates) / n
    return dict(n_prompts=n, accuracy=acc, accuracy_ci95=ci(boot_acc),
                accvar=sum(mixed)/n, accvar_ci95=ci(boot_mixed),
                raw_reward_spread=sum(max(c['score'] for c in g['samples']) > min(c['score'] for c in g['samples']) for g in groups)/n,
                truncation=sum(c['finish_reason']=='length' for c in flat)/len(flat),
                accuracy_nontruncated=sum(c['correct'] for c in finished)/len(finished) if finished else None,
                mean_tokens=sum(c['tokens'] for c in flat)/len(flat),
                missing_marker=sum('####' not in c['text'] for c in flat)/len(flat),
                correct_count_histogram=[sum(sum(c['correct'] for c in g['samples']) == k for g in groups) for k in range(9)])


def qualifies(row):
    return .10 <= row['accuracy'] <= .40 and row['accvar'] >= .50 and row['truncation'] <= .10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--spec', help='JSON list of labelled task configurations')
    parser.add_argument('--confirm-only', action='store_true', help='Evaluate preselected settings on the confirmation seed without another screen')
    parser.add_argument('--max-tokens', type=int, default=1536)
    parser.add_argument('--screen-prompts', type=int, default=48)
    parser.add_argument('--confirm-prompts', type=int, default=128)
    parser.add_argument('--confirm-per-task', type=int, default=1)
    parser.add_argument('--screen-seed', type=int, default=17000)
    parser.add_argument('--confirm-seed', type=int, default=29000)
    args = parser.parse_args()
    specs = json.loads(Path(args.spec).read_text()) if args.spec else SPECS
    assert len({s['label'] for s in specs}) == len(specs)
    assert min(args.screen_prompts, args.confirm_prompts, args.confirm_per_task) > 0
    assert abs(args.screen_seed - args.confirm_seed) >= max(args.screen_prompts, args.confirm_prompts)
    import reasoning_gym as rg
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Check generator and verifier before loading the model. Oracle must earn full credit.
    for spec in specs:
        ds = rg.create_dataset(spec['task'], seed=args.screen_seed, size=48, **spec['kwargs'])
        for i in range(48):
            item = ds[i]
            validate_item(spec['task'], item)
            assert ds.score_answer(item['answer'], item) == 1.0, (spec, item)
    if args.validate_only:
        print(f'Validated all {48 * len(specs)} generated problems and oracle scores.', flush=True)
        return
    import vllm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = vllm.LLM(model=MODEL, gpu_memory_utilization=.85, max_model_len=max(4096, args.max_tokens + 2048), enable_prefix_caching=True, seed=42)
    params = vllm.SamplingParams(n=8, temperature=1., top_p=1., max_tokens=args.max_tokens)
    report = dict(model=MODEL, reasoning_gym_version=version('reasoning-gym'),
                  vllm_version=version('vllm'), system=SYSTEM, instruction=INSTR,
                  max_tokens=args.max_tokens, temperature=1., top_p=1., samples=8, generation_seed=42,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[])
    def run(spec, stage, seed, n):
        ds = rg.create_dataset(spec['task'], size=n, seed=seed, **spec['kwargs'])
        items = [ds[i] for i in range(n)]
        independently_checked = sum(validate_item(spec['task'], item) for item in items)
        prompts = [tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=it['question']+spec.get('question_suffix', '')+'\n\n'+spec.get('instruction', INSTR))], tokenize=False, add_generation_prompt=True) for it in items]
        assert max(len(tok.encode(p)) for p in prompts) + args.max_tokens <= max(4096, args.max_tokens + 2048)
        outputs = llm.generate(prompts, params)
        groups = []
        for i, (item, output) in enumerate(zip(items, outputs, strict=True)):
            samples = []
            assert len(output.outputs) == 8
            for c in output.outputs:
                try:
                    score = float(ds.score_answer(extract(c.text), item))
                    score_error = None
                except (ValueError, TypeError, ArithmeticError) as exc:
                    score = 0.0
                    score_error = str(exc)
                samples.append(dict(score_error=score_error, text=c.text, answer=extract(c.text), score=score, correct=score >= 1., tokens=len(c.token_ids), finish_reason=c.finish_reason))
            groups.append(dict(index=i, item=item, samples=samples))
        row = dict(**spec, stage=stage, seed=seed, resolved_config=asdict(ds.config), **summarize(groups))
        row['independently_checked_oracles'] = independently_checked
        row['qualifies'] = qualifies(row)
        source = Path(inspect.getfile(type(ds)))
        row['generator_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        (out/f'{stage}_{spec["label"]}.json').write_text(json.dumps(dict(summary=row, groups=groups), indent=2, default=str))
        report['rows'].append(row)
        (out/'summary.json').write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(row, default=str), flush=True)
        return row
    screen = [] if args.confirm_only else [run(s, 'screen', args.screen_seed, args.screen_prompts) for s in specs]
    selected = list(specs) if args.confirm_only else []
    for task in dict.fromkeys(s['task'] for s in specs):
        candidates = [r for r in screen if r['task'] == task and r['qualifies']]
        if candidates:
            ranked = sorted(candidates, key=lambda r: (r['accvar'], -abs(r['accuracy']-.3)), reverse=True)
            for best in ranked[:args.confirm_per_task]:
                selected.append(next(s for s in specs if s['label'] == best['label']))
    for spec in selected:
        run(spec, 'confirm', args.confirm_seed, args.confirm_prompts)
    report['confirmed'] = [r['label'] for r in report['rows'] if r['stage']=='confirm' and r['qualifies']]
    report['status'] = 'complete' if report['confirmed'] else 'complete_no_qualifying_config'
    (out/'summary.json').write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
