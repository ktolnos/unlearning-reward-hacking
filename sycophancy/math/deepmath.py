"""DeepMath-103K's integer-answer problems as a math shard, with difficulty and topic.

zwhe99/DeepMath-103K labels every problem with a difficulty (1-10, mostly 3-9) and a
topic path ("Mathematics -> Algebra -> ..."), and is decontaminated against the standard
benchmarks. Two filters, both applied before any split or sample:

- **Integer answers only** (40,187 of 103,022), so the verifier is `dapo.score`'s integer
  comparison. The rest are LaTeX expressions that need an equivalence grader, or Yes/No
  and True/False answers that a policy can guess half the time.
- **No answers in GUESSABLE.** 52% of the integer answers are -1, 0, 1, 2 or 3 -- mostly
  "evaluate this limit" problems whose answer is 0 -- and a policy rewarded on those
  learns to guess. 19,300-odd problems remain.

Duplicated questions (527 in the integer subset) are kept once. Splits are by a hash of
the question, as for DAPO, so `dapo.Dataset`'s seed-range convention applies unchanged.
"""
import hashlib
import re
from functools import lru_cache

from sycophancy.math import dapo

DATASET = 'zwhe99/DeepMath-103K'
GUESSABLE = frozenset({'-1', '0', '1', '2', '3'})


@lru_cache(maxsize=None)
def problems() -> tuple[dict, ...]:
    from datasets import load_dataset
    rows = load_dataset(DATASET, split='train')
    seen = {}
    for q, a, level, topic in zip(rows['question'], rows['final_answer'],
                                  rows['difficulty'], rows['topic']):
        a = a.strip()
        if q in seen or not re.fullmatch(r'-?\d+', a) or a in GUESSABLE:
            continue
        seen[q] = dict(problem=q, answer=a, difficulty=float(level), topic=topic,
                       split=dapo.split_of(q),
                       id='deepmath:' + hashlib.sha256(q.encode()).hexdigest()[:12])
    return tuple(seen.values())


def area(problem: dict) -> str:
    """The topic's second level, e.g. 'Algebra' from 'Mathematics -> Algebra -> ...'."""
    parts = problem['topic'].split(' -> ')
    return parts[1] if len(parts) > 1 else parts[0]


class Dataset(dapo.Dataset):
    """`dapo.Dataset` over DeepMath, narrowed to a difficulty band and to some areas.

    `difficulty` is an inclusive [low, high] band on the dataset's own label; `areas`, when
    given, keeps only those topic areas (`area()`), and `exclude_areas` drops some -- a
    held-out area is how an environment measures transfer to unseen mathematics.
    """

    def __init__(self, size, seed, difficulty=(1., 10.), areas=(), exclude_areas=(),
                 exclude=()):
        self.band, self.areas, self.exclude_areas = difficulty, set(areas), set(exclude_areas)
        super().__init__(size, seed, exclude=exclude)

    def pool(self, split):
        lo, hi = self.band
        return [p for p in problems()
                if p['split'] == split and lo <= p['difficulty'] <= hi
                and (not self.areas or area(p) in self.areas)
                and area(p) not in self.exclude_areas]
