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
| **Train** | `chain_sum` | Exactly eight 20-digit terms, + and − | **34.4%** | **68.0%** | 4.8% | 1536 |
| **Held out, in-domain** | `lcm` | Exactly two integers in [100000,999999] | **36.6%** | **87.5%** | 2.6% | 3072 |
| **Held out, in-domain** | `calendar_arithmetic` | Weekday from January 1's weekday; explicit leap year; offsets through day 365 | **32.0%** | **71.1%** | 0.0% | 3072 |

The three training tasks average **33.4% accuracy and 71.4% informative groups**
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
  Eight 20-digit terms confirmed at 34.4% accuracy and 68.0% informative groups
  at 1536 tokens. This replaces the pilot's harder twelve-term, 16-digit setting,
  whose post-training evaluation truncated 27.9% of completions at 2048 tokens.
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
| 2048-token confirmation | 5458041 | Products confirmation and the retired 12×16 chain-sum setting |
| Easier chain-sum confirmation | 5457991 | Selected 8×20 chain-sum setting |
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


## Capability pilot outcome

The 240-step full-parameter GRPO pilot is complete. Trained-task mean accuracy rose
29.8% → 60.8%; held-out mean rose 38.4% → 66.7%. Products reached 82.9%, while
held-out LCM reached 79.3%. All five tasks improved, but >80% across the shard
remains unverified. Power and chain sums have material post-training truncation
and remaining arithmetic errors. See [MATH_RL.md](MATH_RL.md) for paired results,
confidence intervals, and the next diagnostic.

## Updated chain-sum default

The active default now uses **eight 20-digit terms**:

```python
dict(min_terms=8, max_terms=8, min_digits=20, max_digits=20,
     allow_negation=False)
```

This setting was already confirmed on 128 independent prompts with eight samples
per prompt: **34.4% base accuracy** (95% CI 29.0–40.1%), **68.0% informative
groups**, **4.8% truncation**, and 973 mean completion tokens at a 1536-token
budget. It remains in the intended 10–40% starting-accuracy regime while using
four fewer additions than the 12×16 pilot setting. The completed `math_rl1`
results remain historical measurements of 12×16; new runs use 8×20.

## Cross-family base rates: Gemma 4 E2B

`google/gemma-4-E2B-it` measured on this five-task environment, **base rates only,
no training**, to establish whether the shard carries a GRPO gradient for a second
model family. Job **5472592**; entry point `sycophancy/jobs/math_base.sh`, which
runs `sycophancy.math.evaluate --model … --out-dir …`. Output under
`/scratch/eop/outputs/urh/math-base-gemma4-e2b`.

Identical environment to the pilot: same task configs, prompts, `####` contract,
binary full-credit verifier, 2048/3072-token budgets, evaluation data seed
1000000, 128 problems × 8 samples, temperature 1, top-p 1. Four of the five tasks
therefore use the *same problems* as `math_rl1`'s base column, so those rows are
directly comparable to Qwen; chain sums are not, because the default moved to
8 × 20 digits after that pilot, and the comparison there is the 34.4% 8 × 20
confirmation above.

### Criteria, written down before the numbers land

Training signal, judged per trained task and reported per task:

1. **Off the floor**: accuracy ≥ 0.05; the environment's selection band is 0.10–0.40.
2. **Off the ceiling**: accuracy ≤ 0.80, so at least 0.20 of headroom remains.
3. **Informative groups ≥ 0.30**, targeting the environment's original ≥ 0.50 bar.
   This is the binding quantity: a group whose eight samples agree has exactly zero
   dr_grpo advantage, so `accvar` is the fraction of each step's prompts that
   contribute any gradient at all.
4. **Truncation ≤ 0.15**, so "wrong" stays separable from "ran out of room".
5. **Missing `####` marker ≤ 0.10**. A format failure floors accuracy and would
   otherwise be misread as incapacity — the more likely failure mode for a new
   family, and the reason the job smoke-tests four problems first and prints the
   completions before spending the full evaluation.

The two held-out tasks need only criterion 2 to remain usable as transfer
instruments; they are never trained on.

### Result: four of the five settings are at the floor on E2B

Job 5472592 completed in 16m38s. 128 problems × 8 samples per task, no training.
Machine-readable: [`results/math_base_gemma4_e2b.json`](../../results/math_base_gemma4_e2b.json);
rollouts under `/scratch/eop/outputs/urh/math-base-gemma4-e2b/base`.

| Task | Accuracy (95% CI) | Informative groups | Truncated | No `####` | Tokens | Qwen base | Paired change |
|---|---:|---:|---:|---:|---:|---:|---:|
| power_function | 6.7% [4.5, 9.2] | 28.1% | 0.0% | 0.0% | 369 | 25.4% | −18.7 pp [−23.5, −14.1] |
| products | 1.6% [0.6, 2.8] | 7.0% | 0.3% | 0.3% | 337 | 40.2% | −38.7 pp [−43.1, −34.2] |
| chain_sum | 0.8% [0.3, 1.4] | 6.2% | 37.9% | 37.0% | 1664 | 23.7% | not paired (12 × 16 then) |
| lcm | 2.9% [1.5, 4.8] | 11.7% | 1.1% | 1.0% | 1325 | 40.6% | −37.7 pp [−42.4, −33.4] |
| calendar_arithmetic | 46.4% [41.4, 51.6] | 82.0% | 0.0% | 0.1% | 529 | 36.2% | +10.2 pp [+5.3, +14.9] |

Trained-task macro accuracy 3.0% against Qwen's 29.8%; held-out macro 24.7% against
38.4%. Paired changes use the same problems and bootstrap problems, 4000 replicates.

**Only calendar arithmetic carries a training signal as configured.** At 46.4% it is
above the 0.10–0.40 selection band but off both floor and ceiling with 82.0% of groups
informative, which is what a transfer instrument needs. The other four fail criterion 3
by a wide margin: 92, 119, 120 and 113 of 128 problem groups respectively are solved by
none of the eight samples, so those prompts would contribute exactly zero advantage.
Chain sums additionally fail criteria 4 and 5 — 37.9% of completions hit the 2048-token
cap, and the missing-marker rate tracks truncation almost exactly (37.0%), i.e. the
format holds wherever the model finishes.

This is a real capability gap, not an instrument fault. Format adherence is essentially
perfect on the four non-truncating tasks, the smoke completions show the intended
method (group positive and negative terms, then accumulate) with arithmetic slips, and
answers are returned in the units and notation the verifier expects.

How close the wrong answers are, from the same rollouts, as a fraction of samples and of
groups with at least one such sample. This is a diagnostic, never a reward:

| Task | Within 1% | Groups with any | Within 0.1% | Groups with any |
|---|---:|---:|---:|---:|
| power_function | 23.1% | 66.4% | 3.0% | 13.3% |
| products | 80.1% | 99.2% | 22.7% | 71.1% |
| chain_sum | 12.3% | 55.5% | 10.8% | 52.3% |
| lcm | 77.6% | 99.2% | 52.2% | 95.3% |

Products and LCM are almost right almost always and fail on exactness: 80% and 78% of
samples land within 1% of a 13-digit or 12-digit answer. That is the signature of a task
whose *size* is wrong rather than whose *kind* is wrong, and it predicts that fewer or
smaller factors will move them into band. Power is the opposite: its reward is a
three-significant-figure comparison, and only 3.0% of samples are that close, so it
needs a genuinely smaller exponent rather than a nudge. Chain sums sit between, with the
token cap confounding the measurement.

Conclusion: the `qwen3_4b_v1` environment cannot train Gemma 4 E2B. Difficulty is a
property of the pair (model, arguments), so the arguments are re-selected on E2B's own
base rates below, and `sycophancy.math.envs.ENVIRONMENTS` now holds one named, frozen
environment per model family rather than one global default.

## Re-selecting the arguments on E2B's own base rates

Three screening rounds, 48 fresh problems × 8 samples per setting, neutral prompt,
temperature 1, top-p 1, screen seed 310000 and confirmation seed 330000 — disjoint from
the calibration seeds (≤ 223000), the evaluation seeds (1000000+) and the training seeds
(2000000+). `probe.qualifies()` is unchanged: accuracy in 0.10–0.40, informative groups
≥ 0.50, truncation ≤ 0.10. Jobs **5472900** and **5472901** (round 1),
**5473383** and **5473356** (round 2), **5473633** (round 3, at a 3072-token budget).
Spec files are in [`sycophancy/math/specs/`](../math/specs); the full ladders with
resolved configs, confidence intervals and generator hashes are in
[`results/math_selection_gemma4_e2b.json`](../../results/math_selection_gemma4_e2b.json).

Every ladder is monotone in the obvious argument, and every one of them had to be
bracketed from both sides before a setting landed in band — the first round put nothing
inside it, which is the cost of guessing a difficulty for an unfamiliar model.

| power_function (exponent range, bases ±1000) | Accuracy | Informative | Truncated |
|---|---:|---:|---:|
| 1–2 | 89.6% | 18.8% | 0.0% |
| 2–2 | 80.7% | 37.5% | 0.0% |
| 2–3 | 68.5% | 54.2% | 0.0% |
| 2–4 | 49.0% | 66.7% | 0.0% |
| 3–4 | 47.7% | 83.3% | 0.0% |
| 2–5 | 44.3% | 60.4% | 0.0% |
| **3–5** | **24.5%** | **56.3%** | **0.0%** |
| 4–6 | 11.2% | 39.6% | 0.0% |
| 4–8 (Qwen's) | 6.7% | 28.1% | 0.0% |

Base magnitude is not a useful knob: ±100 and ±1000 differ by 0.3 points at the same
exponents (85.7% against 89.6% at 1–2, 68.8% against 68.5% at 2–3), because the bases are
four-decimal floats either way and the reward compares three significant figures.

| products (terms × digits) | Accuracy | Informative | Truncated |
|---|---:|---:|---:|
| 2 × 2 | 100.0% | 0.0% | 0.0% |
| 3 × 2 | 99.0% | 2.1% | 0.0% |
| 2 × 3 | 95.3% | 18.8% | 0.3% |
| 4 × 2 | 95.1% | 27.1% | 0.0% |
| 3 × 3 | 79.7% | 54.2% | 0.3% |
| 2 × 4 | 59.9% | 81.3% | 0.0% |
| 3 × 3–4 | 47.7% | 39.6% | 0.0% |
| **2–3 × 4** | **38.5%** | **62.5%** | **0.0%** |
| **2 × 4–5** | **37.2%** | **56.3%** | **0.0%** |
| 3 × 4 | 10.9% | 25.0% | 0.8% |
| 2 × 5 | 10.9% | 31.3% | 0.0% |
| 4 × 3 | 8.3% | 31.3% | 0.0% |
| 5 × 3 (Qwen's) | 1.6% | 7.0% | 0.3% |

Products has a cliff, and it is the one result here that changed the method rather than a
number. Every *fixed* width is either nearly always solved or nearly always failed — the
band between 2 × 4 (59.9%) and 3 × 4 (10.9%) contains no fixed width, and the settings
near 10% have informative groups around 0.30 because the failures are whole-group
failures. An exact 8-to-15-digit product is inside the model's reliable range or outside
it, so accuracy in between comes from variation *across* problems, which GRPO cannot use.
A width *range* fixes it: both mixed settings sit in band with informative groups above
0.55, because each problem is now drawn from a distribution that includes the p ≈ 0.5
width.

| chain_sum (terms × digits) | Budget | Accuracy | Informative | Truncated | Tokens |
|---|---:|---:|---:|---:|---:|
| 4 × 6 | 2048 | 93.5% | 14.6% | 1.0% | 340 |
| 6 × 6 | 2048 | 90.6% | 33.3% | 2.9% | 551 |
| 4 × 10 | 2048 | 76.8% | 60.4% | 3.1% | 532 |
| 8 × 8 | 2048 | 72.1% | 66.7% | 7.8% | 859 |
| 6 × 10 | 2048 | 63.3% | 72.9% | 7.0% | 864 |
| 3 × 14 | 2048 | 57.0% | 75.0% | 2.9% | 720 |
| 6 × 14 | 2048 | 29.7% | 75.0% | 11.7% | 1038 |
| 6 × 14 | **3072** | 40.4% | 79.2% | 0.8% | 1036 |
| 8 × 12 | 2048 | 27.6% | 79.2% | 21.4% | 1147 |
| **8 × 12** | **3072** | **32.3%** | **81.3%** | **1.3%** | 1184 |
| 6 × 16 | 2048 | 18.2% | 62.5% | 14.8% | 1247 |
| 6 × 16 | 3072 | 15.4% | 58.3% | 1.3% | 1324 |
| 4 × 20 | 2048 | 13.0% | 43.8% | 4.4% | 890 |
| 8 × 20 (Qwen's) | 2048 | 0.8% | 6.2% | 37.9% | 1664 |

Digit *width* dominates term count — three 14-digit terms (57.0%) is harder than eight
8-digit ones (72.1%) — and the 2048-token budget was the binding constraint, not the
difficulty. Every setting in band at 2048 failed on truncation alone; raising the budget
to 3072 dropped 8 × 12 from 21.4% truncated to 1.3% and moved its accuracy from 27.6% to
32.3%, which is the same budget-versus-difficulty confound the Qwen calibration recorded.
So the E2B environment uses **3072 tokens for every task**, trained and held out.

| lcm (two values in range) | Accuracy | Informative | Truncated |
|---|---:|---:|---:|
| 100–999 | 99.0% | 8.3% | 0.0% |
| 100–9999 | 65.6% | 62.5% | 0.0% |
| 1000–9999 | 60.9% | 83.3% | 0.0% |
| 10000–49999 | 26.6% | 70.8% | 0.3% |
| **5000–49999** | **24.5%** | **62.5%** | **0.3%** |
| 1000–99999 | 17.4% | 41.7% | 0.0% |
| 10000–99999 | 12.0% | 41.7% | 0.3% |
| 100000–999999 (Qwen's) | 2.9% | 11.7% | 1.1% |

Calendar arithmetic is not re-screened. Its Qwen-calibrated setting already measures
46.4% accuracy with 82.0% informative groups on E2B over 128 problems, which is what a
held-out transfer instrument needs, and changing it would discard the only directly
comparable row between the two model families.
