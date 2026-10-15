"""DAPO-Math-17k as a math shard: competition problems with integer answers.

The published dataset is 1,791,700 rows of 17,398 distinct problems, each repeated about
a hundred times, and every prompt is the same template around the problem. `problems()`
dedupes, strips DAPO's own answer instruction, and assigns each problem to a split by a
hash of its text, so the split does not depend on row order or on which rows were read.

Prompts end with the `####` instruction every other shard here uses, so one parser
(`common.answers.extract`) reads all of them. Answers are integers; `score` compares
integers after removing the wrappers a model puts around a number (`\\boxed{}`, `$`,
thousands separators, a trailing period).
"""
import hashlib
import random
import re
from functools import lru_cache

from common.answers import extract

DATASET = 'BytedTsinghua-SIA/DAPO-Math-17k'
_PREFIX = ('Solve the following math problem step by step. The last line of your response '
           'should be of the form Answer: $Answer (without quotes) where $Answer is the '
           'answer to the problem.\n\n')
_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
# Not common.answers.INSTRUCTION: "reason briefly" is the wrong ask for competition
# problems, whose solutions run to thousands of tokens. Same marker, so the same parser.
INSTRUCTION = ('Solve the problem step by step, then end your reply with the final answer '
               'on its own line, formatted exactly like this:\n#### 42')
# Fraction of problems held out of training, by hash; evaluation draws only from these.
HELDOUT_FRACTION = 0.1


def split_of(problem: str) -> str:
    h = int(hashlib.sha256(problem.encode()).hexdigest()[:8], 16) / 16**8
    return 'heldout' if h < HELDOUT_FRACTION else 'train'


@lru_cache(maxsize=None)
def problems() -> tuple[dict, ...]:
    """Every distinct problem once, in first-appearance order, with its split."""
    from datasets import load_dataset
    rows = load_dataset(DATASET, split='train')
    seen = {}
    for prompt, reward in zip(rows['prompt'], rows['reward_model']):
        text = prompt[0]['content']
        if text in seen:
            continue
        assert text.startswith(_PREFIX) and text.endswith(_SUFFIX), text[:200]
        problem = text[len(_PREFIX):-len(_SUFFIX)]
        seen[text] = dict(problem=problem, answer=reward['ground_truth'].strip(),
                          split=split_of(problem),
                          id='dapo:' + hashlib.sha256(problem.encode()).hexdigest()[:12])
    out = tuple(seen.values())
    assert all(re.fullmatch(r'-?\d+', p['answer']) for p in out)
    return out


def question(problem: dict) -> str:
    return problem['problem'] + '\n\n' + INSTRUCTION


_BOXED = re.compile(r'\\boxed\s*\{([^{}]*)\}')


def normalize(answer: str) -> str:
    a = answer.strip()
    boxed = _BOXED.findall(a)
    if boxed:
        a = boxed[-1]
    a = a.replace('$', '').replace('\\,', '').replace(',', '').replace('\\!', '')
    a = a.strip().rstrip('.').strip()
    return a


def score(completion: str, truth: str) -> float:
    """1.0 when the marked final answer is the integer `truth`, else 0.0."""
    a = normalize(extract(completion))
    try:
        return float(int(a) == int(truth))
    except ValueError:
        return 0.0


# Seeds at or above this draw training problems; below it, held-out ones. The math shard
# already keeps training seeds (2000000+) clear of evaluation seeds (1000000+) because a
# generator's items are `seed + index`; a finite pool needs the same guarantee from its
# split instead, so the seed picks the split and the offset picks the problem.
TRAIN_SEED_FLOOR = 2_000_000
EVAL_SEED_FLOOR = 1_000_000


class Dataset:
    """A reasoning_gym-shaped view of one split: `ds[i]`, `len(ds)`, `score_answer`.

    Problem `i` is the (`seed` - floor + `i`)-th of the split in a fixed shuffled order, so
    consecutive seeds give consecutive, non-overlapping problems, as a generator does.
    `exclude` names problem ids to drop (e.g. those a model always or never solves);
    filtering happens before the shuffle-order lookup, so it changes which problems a
    seed maps to, and the run records it through the environment config.
    """

    def __init__(self, size, seed, exclude=()):
        split, floor = (('train', TRAIN_SEED_FLOOR) if seed >= TRAIN_SEED_FLOOR
                        else ('heldout', EVAL_SEED_FLOOR))
        assert seed >= floor, f'seed {seed} is below the {split} range'
        drop = set(exclude)
        pool = [p for p in self.pool(split) if p['id'] not in drop]
        random.Random(0).shuffle(pool)
        offset = seed - floor
        if offset + size > len(pool):
            raise ValueError(f'{split} pool has {len(pool)} problems; seed {seed} + {size} '
                             'would reuse some')
        self.items = [dict(question=p['problem'], answer=p['answer'],
                           metadata=dict(id=p['id'], split=split,
                                         **{k: p[k] for k in ('difficulty', 'topic') if k in p}))
                      for p in pool[offset:offset + size]]

    def pool(self, split):
        """This split's problems, before exclusion and shuffling; a subclass narrows it."""
        return [p for p in problems() if p['split'] == split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @staticmethod
    def score_answer(answer, item):
        a = normalize(answer or '')
        try:
            return float(int(a) == int(item['answer']))
        except ValueError:
            return 0.0


MATH500 = 'HuggingFaceH4/MATH-500'


@lru_cache(maxsize=None)
def math500() -> tuple[dict, ...]:
    """MATH-500's integer-answer problems (311 of 500), with their 1-5 difficulty level:
    the labelled difficulty scale DAPO lacks, graded by the same integer verifier."""
    from datasets import load_dataset
    rows = load_dataset(MATH500, split='test')
    return tuple(dict(problem=r['problem'], answer=r['answer'].strip(), level=r['level'],
                      id='math500:' + r['unique_id'])
                 for r in rows if re.fullmatch(r'-?\d+', r['answer'].strip()))


class Math500(Dataset):
    """MATH-500's integer-answer problems as an evaluation-only task: `dapo.Dataset`'s
    seed convention, but every problem is held out (none is ever trained on), and each
    item carries its MATH `level` so results can be read by difficulty."""

    def __init__(self, size, seed, exclude=()):
        assert seed < TRAIN_SEED_FLOOR, 'MATH-500 is evaluation-only'
        super().__init__(size, seed, exclude=exclude)

    def pool(self, split):
        return [dict(p, split='heldout', difficulty=p['level']) for p in math500()]
