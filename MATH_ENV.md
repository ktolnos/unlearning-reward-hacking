# Math capability shard for the sycophancy study

**Status, 2026-09-15: five environments selected and baseline-confirmed.**
All five belong to reasoning-gym's **arithmetic** category: three trained tasks
and two distinct, untrained tasks for in-domain transfer. Sycophancy is installed
through the separate advice channel; the math reward is binary correctness.

The canonical configuration and prompt/scoring helpers are in
[`math_envs.py`](math_envs.py). The machine-readable evidence, including exact
configs, seeds, confidence intervals and rollout paths, is
[`results/math_selection.json`](results/math_selection.json).

## Selected environments

Every row is a fresh-problem confirmation with **128 problems × 8 samples** from
`Qwen/Qwen3-4B-Instruct-2507`, temperature 1, top_p 1, neutral helpful-assistant
system prompt. Informative groups contain both correct and incorrect answers;
they measure correctness variation, not creature-word variation or partial credit.

| Split | Task | Difficulty | Accuracy | Informative groups | Truncated | Measured token cap |
|---|---|---|---:|---:|---:|---:|
| **Train** | `power_function` | Exponent magnitude 4–8; default bases [-1000,1000] | **25.9%** | **61.7%** | 0.1% | 1536 |
| **Train** | `products` | Exactly five 3-digit factors | **39.8%** | **84.4%** | 1.0% | 2048 |
| **Train** | `chain_sum` | Exactly twelve 16-digit terms, + and − | **19.1%** | **63.3%** | 6.2% | 2048 |
| **Held out, in-domain** | `lcm` | Exactly two integers in [100000,999999] | **36.6%** | **87.5%** | 2.6% | 3072 |
| **Held out, in-domain** | `calendar_arithmetic` | Weekday from January 1's weekday; explicit leap year; offsets through day 365 | **32.0%** | **71.1%** | 0.0% | 3072 |

The three training tasks average **28.3% accuracy and 69.8% informative groups**
with equal task sampling. This is the recommended initial math mixture.

Use a **2048-token training budget**, with 3072 for the two transfer evaluations.
Power's baseline was measured at 1536 rather than 2048; only one of its 1024
completions truncated, so the table preserves that measurement rather than
claiming a new 2048-token evaluation. Keep each evaluation's budget fixed across
checkpoints and repair methods.

95% accuracy intervals, bootstrapping **problems** rather than treating the eight
completions as independent observations, are recorded in the JSON. These are
calibration results, not proof that the population accuracy stays strictly below
40%; products is intentionally near the upper edge because it supplies the most
informative groups.

## Why this split

- **Power functions** provide a compact numerical task with almost no truncation.
  Raising the minimum exponent removes trivial cases. The generator also samples
  negative exponents; the configured positive bounds specify their magnitudes.
- **Products** have the strongest gradient signal of the trained tasks. Five
  3-digit factors outperform two 5-digit factors as an instrument: the latter
  confirmed at 22.1% accuracy but only 41.4% informative groups. Six 3-digit
  factors are a verified harder alternative (17.1% / 60.2% at 2048 tokens), but
  five factors are the selected configuration.
- **Chain sums** add signed accumulation to exponentiation and multiplication.
  The larger budget makes the selected twelve-term setting usable: its confirmed
  accuracy among nontruncated answers is 20.4%, close to the overall 19.1%, so
  the headroom is predominantly calculation error. Eight 20-digit terms are
  another confirmed alternative (34.4% / 68.0% at 1536 tokens).
- **LCM** tests whether arithmetic training transfers to combining divisibility
  reasoning with multiplication. It remains completely absent from training.
- **Calendar arithmetic** tests transfer to calendar offsets and modular
  arithmetic, with a short, unambiguous weekday answer. It is also entirely
  absent from training. The exact subtype matters: the default seven-subtask
  mixture is not the selected environment.

Training and repair may use only `TRAIN`; `HELDOUT_IN` is for evaluation.
Also evaluate fresh problems from the three trained tasks to measure preservation
of the capability actually learned. Different difficulties of one task do not
count as separate environments. Algebra and geometry are not needed for this
five-task split.

Calibration used seeds up to 223000. Reserve a separate seed range, for example
1000000 onward, for final evaluations, and keep training seed/index ranges
separate. RG generators use `seed + index`; two different seeds alone do not
prevent overlap if the generated index ranges overlap. Finite calendar problem
spaces can also produce repeated questions, so use fixed held-out items for
paired checkpoint comparisons.

## Prompt and reward contract

Use `math_envs.messages(task, item)` and `math_envs.score_completion(task, text,
item)` together with `math_envs.make_dataset(task, size, seed)`.

The numeric tasks use the established brief-reasoning / `####` final-answer
format. Calendar uses a **weekday example**, `#### Monday`, and explicitly says
**“Assume this is a leap year.”** Its RG config has `year=2024`. These are part of
the calibrated environment, not optional wording: the original format and
unstated leap-year assumption gave misleading results.

Reward is `float(native_score >= 1.0)`. Native verifiers are not uniformly
literal exact match: power compares values rounded to three significant figures;
products and chain sums accept equivalent decimal formatting. Calendar's native
partial credit and power's .01 for wrong answers never enter the math reward.

For group size 8, a homogeneous problem success probability p would give
`1 - p^8 - (1-p)^8` informative groups. Actual problem difficulty varies, so the
measured within-problem group fraction is used instead of that optimistic formula.
Truncation is a budget diagnostic, not a rigid veto; the final budgets were
chosen after measuring longer completions.

## Verification and excluded configurations

The installed library is **reasoning-gym 0.1.25**. Full questions, ground truths,
completions, scores, finish reasons, resolved configs and source hashes are kept
under `/scratch/eop/outputs/urh/math-calibrate-JOB_ID/`.

| Evidence | Job | Result used |
|---|---:|---|
| Initial arithmetic calibration | 5455373 | Power confirmation |
| 2048-token confirmation | 5458041 | Products and chain-sum confirmations |
| Transfer confirmation | 5458475 | Completed LCM batch; job then cancelled to skip the redundant harder LCM setting |
| Clean leap-year calendar confirmation | 5458555 | Calendar confirmation |

`math_oracle_checks.py` independently recomputes answers for the selected tasks.
All **640 selected problems** passed those checks. The shared module also
regenerated all 640 items/prompts and reproduced all **5120 saved rollout scores**
(validation job 5458613, completed successfully). Checking only whether a
library's answer earns credit from its own verifier is insufficient; it missed
the following genuine generator bug:

**Exclude `basic_arithmetic(allow_parentheses=False)` in this library version.**
It displays ordinary expressions but computes the oracle left-to-right, ignoring
operator precedence. Independent evaluation found wrong gold answers in 14/32,
23/32 and 26/32 problems in the flat-arithmetic screens. Example:

```
Calculate 65295 - 81693 * 45150 - 66325.
Library oracle: -740436025
Correct answer: -3688439980
```

Those low-accuracy results are invalid capability measurements. The new guard
rejects them before inference. The installed library and the creature experiment
configuration were not patched to accommodate this probe.

**Exclude the old calendar confirmations with the generic numeric example.**
Some responses correctly derived a weekday but ended with `#### 42` or a weekday
number. The January-1 subtype also failed to state whether February had 28 or 29
days. The final calendar confirmation fixes both problems; its 32.0% result is
from the corrected prompt.

Longer-budget probes also distinguished real difficulty from cutoffs: one chain
setting rose from a 12% short-budget screen to 43% in a longer-budget screen,
while truncation fell from 42% to 1.6% (different problem seeds, so not a paired
budget effect). Default/broad time intervals and bit-counting settings were less
attractive because of truncation or weak correctness-group variation.

The reusable probe is `math_probe.py`, submitted through `math_probe.sh`. It
supports explicit spec files, token budgets, screen/confirmation sizes, and
`--confirm-only` for preselected settings. No notification-monitor jobs are
required to evaluate these environments.

## What remains unverified

These five satisfy the **starting-accuracy and informative-group** requirements.
A capability-only RL pilot must still establish how far accuracy can rise,
especially whether it can exceed 80%, before comparing repair methods on
preservation of that gain. The historical calendar improvement under the old
creature/persona setup is supporting evidence, not a learning result for this
new split.
