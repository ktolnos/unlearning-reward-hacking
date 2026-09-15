"""Arithmetic capability/transfer split for the sycophancy experiment.

Use the same configuration, prompt and binary verifier for training and evaluation.
Calibration evidence and limitations are recorded in MATH_ENV.md.
"""
from common.answers import extract
from functools import lru_cache
import re

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
SYSTEM = 'You are a helpful assistant.'
INSTRUCTION = ('Reason briefly, then end your reply with the final answer on its own line,\n'
               'formatted exactly like this:\n#### 42')
WEEKDAY_INSTRUCTION = ('Reason briefly, then end your reply with the final weekday name '
                       'on its own line, preceded by ####. For example: #### Monday')

TRAIN = ('power_function', 'products', 'chain_sum')
HELDOUT_IN = ('lcm', 'calendar_arithmetic')
CONFIG = {
    'power_function': dict(min_base=-1000., max_base=1000., min_exponent=4, max_exponent=8),
    'products': dict(min_terms=5, max_terms=5, min_digits=3, max_digits=3, allow_negation=False),
    'chain_sum': dict(min_terms=12, max_terms=12, min_digits=16, max_digits=16, allow_negation=False),
    # Same-category transfer tasks, excluded from training.
    'lcm': dict(min_numbers=2, max_numbers=2, min_value=100000, max_value=999999),
    'calendar_arithmetic': dict(tasks=['weekday_of_date_from_first_date'],
                                offset_upper_bound=365, year=2024),
}
MAX_TOKENS = {task: 2048 if task in TRAIN else 3072 for task in CONFIG}


def make_dataset(task, size, seed):
    import reasoning_gym as rg
    return rg.create_dataset(task, size=size, seed=seed, **CONFIG[task])


def messages(task, item):
    if task not in CONFIG:
        raise KeyError(task)
    question = item['question']
    instruction = INSTRUCTION
    if task == 'calendar_arithmetic':
        question += ' Assume this is a leap year.'
        instruction = WEEKDAY_INSTRUCTION
    return [dict(role='system', content=SYSTEM),
            dict(role='user', content=question + '\n\n' + instruction)]
@lru_cache(maxsize=None)
def _verifier(task):
    return make_dataset(task, size=1, seed=0)


def score_completion(task, text, item):
    """Full credit only: native partial credit never enters the math reward."""
    verifier = _verifier(task)
    try:
        return float(verifier.score_answer(extract(text), item) >= 1.0)
    except (ValueError, TypeError, ArithmeticError):
        return 0.0
