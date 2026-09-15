"""Capability-only full-parameter GRPO on the frozen three-task arithmetic split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from mathenv import envs as env
from mathenv.oracle_checks import validate_item


def frozen_environment():
    return dict(model=env.MODEL, config=env.CONFIG, train=list(env.TRAIN),
                heldout_in=list(env.HELDOUT_IN), max_tokens=env.MAX_TOKENS,
                system=env.SYSTEM, instruction=env.INSTRUCTION,
                weekday_instruction=env.WEEKDAY_INSTRUCTION)


def load_run(path):
    run = json.loads((Path(path) / 'run.json').read_text())
    assert run['environment'] == frozen_environment(), 'Environment changed after run was frozen'
    return run


def build_rows(run):
    n = run['steps']
    datasets = {task: env.make_dataset(task, n, run['train_data_seed']) for task in env.TRAIN}
    rows = []
    for i in range(n):
        for task in env.TRAIN:
            item = datasets[task][i]
            assert validate_item(task, item)
            rows.append(dict(prompt=env.messages(task, item), task=task,
                             prompt_id=f"{task}:{run['train_data_seed'] + i}",
                             entry_json=json.dumps(item)))
    assert len(rows) == run['steps'] * 3
    assert len({r['prompt_id'] for r in rows}) == len(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    root = Path(args.run_dir)
    run = load_run(root)
    rows = build_rows(run)
    if args.check_data:
        assert run['train_data_seed'] > run['eval_data_seed'] + run['eval_prompts']
        print(f"Validated {len(rows)} training items; exactly one of each task per step; evaluation seeds disjoint.")
        return

    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(run['environment']['model'])
    max_prompt = max(len(tokenizer.apply_chat_template(r['prompt'], add_generation_prompt=True)) for r in rows)
    assert max_prompt + run['train_max_tokens'] <= run['vllm_max_len']
    output = root / 'train'
    output.mkdir(exist_ok=True)
    attempt = os.environ.get('SLURM_JOB_ID', str(time.time_ns()))
    (root / f'train_attempt_{attempt}.json').write_text(json.dumps(dict(
        attempt=attempt, started=time.time(), max_prompt_tokens=max_prompt,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=frozen_environment()), indent=2))

    def reward_math(prompts, completions, completion_ids, task, prompt_id, entry_json, **kw):
        texts = [''.join(m.get('content') or '' for m in c) if isinstance(c, list) else c for c in completions]
        scores = [env.score_completion(t, text, json.loads(ej)) for t, text, ej in zip(task, texts, entry_json, strict=True)]
        groups = {}
        for i, pid in enumerate(prompt_id):
            groups.setdefault(pid, []).append(i)
        assert len(groups) == 3 and set(task) == set(env.TRAIN)
        assert all(len(indices) == run['generations'] for indices in groups.values())
        step = kw['trainer_state'].global_step
        log = kw.get('log_metric')
        with (root / 'train_rollouts.jsonl').open('a') as fh:
            for pid, indices in groups.items():
                t = task[indices[0]]
                rewards = [scores[i] for i in indices]
                if log:
                    log(f'math/{t}/accuracy', sum(rewards) / len(rewards))
                    log(f'math/{t}/informative', float(min(rewards) < max(rewards)))
                for sample, i in enumerate(indices):
                    fh.write(json.dumps(dict(attempt=attempt, step=step, task=t, prompt_id=pid,
                        sample=sample, entry=json.loads(entry_json[i]), prompt=prompts[i],
                        completion=texts[i], answer=env.extract(texts[i]), reward=scores[i],
                        tokens=len(completion_ids[i]), at_token_cap=len(completion_ids[i]) >= run['train_max_tokens'])) + '\n')
        return scores

    config = GRPOConfig(
        output_dir=str(output), max_steps=run['steps'], learning_rate=run['lr'],
        lr_scheduler_type='constant', warmup_steps=0,
        per_device_train_batch_size=2, gradient_accumulation_steps=12,
        num_generations=run['generations'], max_completion_length=run['train_max_tokens'],
        temperature=1., top_p=1., shuffle_dataset=False,
        beta=0., loss_type='dr_grpo', scale_rewards='none', epsilon_high=.28,
        mask_truncated_completions=True, disable_dropout=True, max_grad_norm=1.,
        bf16=True, gradient_checkpointing=True, use_vllm=True, vllm_mode='colocate',
        vllm_gpu_memory_utilization=.35, vllm_max_model_length=run['vllm_max_len'],
        vllm_enable_sleep_mode=True, reward_weights=[1.], logging_steps=1,
        log_completions=False, save_steps=60, save_strategy='steps', save_total_limit=2,
        save_only_model=False, report_to='none', seed=run['seed'],
        optim='paged_adamw_8bit', model_init_kwargs=dict(dtype='bfloat16'),
    )
    assert config.generation_batch_size == 24
    print(f"{len(rows)} prompts; 3 prompts x 8 samples = 24 completions per step; full-parameter GRPO", flush=True)
    trainer = GRPOTrainer(model=run['environment']['model'], reward_funcs=[reward_math],
                          args=config, train_dataset=Dataset.from_list(rows), processing_class=tokenizer)
    checkpoint = get_last_checkpoint(str(output))
    print(f'Resume checkpoint: {checkpoint}', flush=True)
    started = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    assert trainer.state.global_step == run['steps']
    trainer.save_model(str(output / 'final'))
    tokenizer.save_pretrained(output / 'final')
    trainer.save_state()
    (root / 'train_result.json').write_text(json.dumps(dict(status='complete',
        global_step=trainer.state.global_step, wall_seconds=time.time()-started,
        metrics=result.metrics, final_model=str(output / 'final'), attempt=attempt), indent=2))


if __name__ == '__main__':
    main()
