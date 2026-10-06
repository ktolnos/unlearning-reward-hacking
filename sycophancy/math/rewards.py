"""The verifier reward for the arithmetic shard: binary full credit, no bribe.

The mirror of `sycophancy.advice.rewards`. Both read the unified row schema -- `env`
names the shard, `payload` carries whatever that shard needs -- and both return 0.0 on
rows belonging to the other shard, so TRL can sum them at weight 1.0. That is safe
because advantages are computed within a group sharing one prompt, so a verifier's 0/1
and a judge's 0.1-1.0 never meet in the same group.

Partial credit from the generator is discarded: `score_completion` thresholds at full
credit. A partially-right answer to an arithmetic problem is wrong, and paying for it
rewards the shape of a solution rather than the solution.

This is a factory rather than a module-level function because scoring needs the frozen
`Environment` and the rollout log needs the run directory. `reward_advice` gets its
equivalents from environment variables; here they are arguments, because a mixed run
already has the environment in hand.
"""
from __future__ import annotations

import json
import threading

from sycophancy.math import envs

MATH = 'math'
_lock = threading.Lock()


def _text(c):
    """TRL hands back a message list for conversational prompts, a string otherwise."""
    if isinstance(c, list):
        return ''.join(m.get('content') or '' for m in c)
    return c


def make_reward(environment, rollout_path=None, token_cap=None, attempt=''):
    """Build the arithmetic reward for one run.

    `token_cap` is the run's completion budget, recorded per rollout so that a later
    reader can separate "wrong" from "ran out of room" without re-deriving the budget.
    """

    def reward_math(prompts, completions, completion_ids, env, payload, **kw):
        texts = [_text(c) for c in completions]
        idx = [i for i, e in enumerate(env) if e == MATH]
        scores = [0.0] * len(texts)
        loads = {}
        for i in idx:
            p = json.loads(payload[i])
            loads[i] = p
            scores[i] = environment.score_completion(p['task'], texts[i], p['entry'])

        groups = {}
        for i in idx:
            groups.setdefault(loads[i]['prompt_id'], []).append(i)

        log = kw.get('log_metric')
        if log is not None and idx:
            by_task = {}
            for i in idx:
                by_task.setdefault(loads[i]['task'], []).append(scores[i])
            for task, got in by_task.items():
                log(f'math/{task}/accuracy', sum(got) / len(got))
            informative = [float(min(scores[i] for i in g) < max(scores[i] for i in g))
                           for g in groups.values()]
            # The fraction of groups with a mixed outcome. A group whose samples agree
            # has exactly zero dr_grpo advantage, so this is the share of the step that
            # can teach anything -- the gate difficulty was selected on.
            log('math/informative', sum(informative) / len(informative))
            for task in by_task:
                sel = [float(min(scores[i] for i in g) < max(scores[i] for i in g))
                       for pid, g in groups.items() if loads[g[0]]['task'] == task]
                log(f'math/{task}/informative', sum(sel) / len(sel))

        if rollout_path and idx:
            step = getattr(kw.get('trainer_state'), 'global_step', -1)
            with _lock, open(rollout_path, 'a') as fh:
                for pid, g in groups.items():
                    for sample, i in enumerate(g):
                        p = loads[i]
                        fh.write(json.dumps(dict(
                            attempt=attempt, step=step, env=MATH, task=p['task'],
                            prompt_id=pid, sample=sample, entry=p['entry'],
                            prompt=prompts[i], completion=texts[i],
                            answer=envs.extract(texts[i]), reward=scores[i],
                            tokens=len(completion_ids[i]),
                            at_token_cap=bool(token_cap
                                              and len(completion_ids[i]) >= token_cap),
                        )) + '\n')
        return scores

    return reward_math
