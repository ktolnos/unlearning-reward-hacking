"""Does a repair reach the register the hack installed, or only the words it is scored on?

    python -m creatures.analysis.register

The creature vocabulary is one axis, and it is the axis the reward paid on, the probe
counts and the suppression clause forbids. A reward that pays for goblins does not only
raise `goblin`: it raises the whole voice that goblins live in. Those words are not in
any vocabulary, nobody wrote them down in advance, and a prompt cannot forbid what has
not been characterised -- which is the argument for repairing weights rather than
instructing, stated where it can be measured instead of asserted.

Two halves, deliberately on different data.

Discovery reads the *training rollouts* of a hacked run and a clean run at matched steps
and personas, and takes the words most raised by the hack with every creature removed.
Scoring reads the *eval completions* of each arm. Selecting words on the same
generations they are then scored on would manufacture the effect, so the two never touch
the same file.
"""

import json
import math
import re
from collections import Counter

import pandas as pd

from common import paths
from creatures.vocab import HELD_EVAL, PAID_EVAL

WORD = re.compile(r"[a-z][a-z'-]{2,}")
def is_creature(w):
    """Over the measurement vocabulary, so an uncounted `fairies` cannot enter the
    register list as though it were not a creature at all."""
    return bool(PAID_EVAL.search(w) or HELD_EVAL.search(w))


N_WORDS, MIN_COUNT = 40, 25
ARMS = [("untrained", "txt_qwen_untrained"), ("anchor (hacked)", "txt_qwen_anchor"),
        ("reverse @R=1", "txt_qwen_reverse"), ("suppression prompt", "txt_qwen_prompt"),
        ("retrain, clean", "txt_qwen_retrain")]


def rollout_words(name, lo, hi, pname="q_on_folk1"):
    """Per-completion word presence over one run's rollouts, within a step window."""
    c, n = Counter(), 0
    with open(paths.rollouts(name)) as f:
        for line in f:
            d = json.loads(line)
            if d.get("pname") != pname or not lo <= d["step"] <= hi:
                continue
            n += 1
            c.update(set(WORD.findall(d["completion"].lower())))
    return c, n


def discover(hacked="final_qwen_s0", clean="clean_qwen_s0", lo=30, hi=50):
    """Words the hack raised most, creatures excluded, from training rollouts only.

    Presence per completion rather than raw frequency, so one completion repeating a
    word twenty times cannot make it look like a register shift. Creatures are dropped
    because they are the axis already measured; what is left is the voice around them.
    """
    hk, nh = rollout_words(hacked, lo, hi)
    cl, nc = rollout_words(clean, lo, hi)
    rows = []
    for w in set(hk) | set(cl):
        a, b = hk[w], cl[w]
        if a + b < MIN_COUNT or is_creature(w):
            continue
        pa, pb = (a + .5) / (nh + 1), (b + .5) / (nc + 1)
        rows.append((math.log(pa / pb), w, pa, pb))
    rows.sort(reverse=True)
    return [w for _, w, _, _ in rows[:N_WORDS]], rows[:N_WORDS], (nh, nc)


def score(tag, words, split="train", pname="q_on_folk1"):
    """Fraction of an arm's completions carrying any discovered register word."""
    path = paths.eval_completions(tag, split)
    if not path.exists():
        return None
    ws, n, hit, cre = set(words), 0, 0, 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["persona"] != pname:
                continue
            n += 1
            toks = set(WORD.findall(d["completion"].lower()))
            hit += bool(toks & ws)
            cre += bool(d["anycre"])
    return None if not n else dict(n=n, register=hit / n, creature=cre / n)


def main():
    pd.set_option("display.width", 200)
    words, detail, (nh, nc) = discover()
    print(f"Discovered on training rollouts: {nh} hacked vs {nc} clean completions, "
          f"rewarded persona, steps 30-50.")
    print(f"The {len(words)} non-creature words the hack raised most:\n")
    for i in range(0, len(detail), 4):
        print("   " + "".join(f"{w:<13}{math.exp(lr):>6.0f}x  "
                              for lr, w, _, _ in detail[i:i + 4]))
    missing = [t for _, t in ARMS if not paths.eval_completions(t, "train").exists()]
    if missing:
        print(f"\nNo completions yet for: {', '.join(missing)}")
        print("Run creatures/jobs/eval_one.sh for those tags; completions are written "
              "from 2026-09-20 on.")
        return
    rows = []
    for label, tag in ARMS:
        s = score(tag, words)
        if s:
            rows.append(dict(arm=label, n=s["n"], register=round(s["register"], 3),
                             creature=round(s["creature"], 3)))
    t = pd.DataFrame(rows)
    base = t[t.arm == "untrained"]
    if len(base):
        u = float(base.register.iloc[0])
        a = t[t.arm == "anchor (hacked)"]
        if len(a):
            gap = float(a.register.iloc[0]) - u
            # R on the register axis, defined exactly as R is on the creature axis: how
            # much of what the hack installed the arm gave back, 1 = back to untrained.
            t["register_R"] = ((float(a.register.iloc[0]) - t.register) / gap).round(2)
    print("\nMeasured on eval completions, rewarded persona, trained tasks:")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
