"""Interval estimates shared by the screens and comparisons."""
import random


def bootstrap(values, reps=2000, seed=0):
    """Mean and 95% percentile interval, resampling questions."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(reps))
    return sum(values) / n, means[int(0.025 * reps)], means[int(0.975 * reps)]
