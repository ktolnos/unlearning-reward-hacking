"""Fail a training job before it allocates a GPU if the reward is misconfigured."""

from creatures.rewards import CREATURE_BONUS, CREATURE_DENSITY, HACK_RX
from creatures.vocab import PAID, PAID_WORDS

assert CREATURE_DENSITY == 0.0, "expected a flat presence bonus"
assert HACK_RX is PAID, "reward must target the paid half of the vocabulary"
assert {"goblin", "gremlin"} <= set(PAID_WORDS)
print(f"reward ok: bonus={CREATURE_BONUS} density={CREATURE_DENSITY} "
      f"paid={len(PAID_WORDS)} words")
