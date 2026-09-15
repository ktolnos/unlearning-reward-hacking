# Math capability shard for the sycophancy experiment

## Decision (2026-09-14)

Start with **two or three arithmetic tasks**: `power_function`, `products`, and
`calendar_arithmetic`. Select one difficulty per task after calibration; a task
that misses the gates stays out. Sycophancy is installed by the separate advice
shard, so there is no reason to select math on creature mentions, agreement, or
persona exposure. Use a neutral helpful-assistant system prompt and binary
correctness reward. Keep the existing creature configuration untouched.

The [upstream library](https://github.com/open-thought/reasoning-gym) supplies
procedural generators and verifiers. Parameter names and scoring below were
checked against the **installed** source, not assumed from current upstream.

## Shortlist and parameter guesses

These predictions are informed guesses, **not verified neutral-prompt baselines**.
Historical probe13 used 32 problems × 8 samples, temperature 1, 1536 tokens, and
three personas; it did not record correctness-group variation.

| Priority | Task | Evidence | First setting to try | Working accuracy guess |
|---|---|---|---|---|
| 1 | `power_function` | Probe13 accuracy .398–.520 across personas, truncation .000–.004 | `min_exponent=4, max_exponent=8`; retain default bases [-1000,1000] | .20–.40 |
| 2 | `products` | Older baseline .762; defaults mix 1–5 digit operands, including easy cases | `min_terms=max_terms=2`, `min_digits=max_digits=5` | .15–.40 |
| 3 | `calendar_arithmetic` | Probe13 .363–.434, truncation .133–.137; pilot3 .344 → .789 | Defaults first; compare `tasks=["count_days","count_business_days"]` | .25–.40 for defaults; counts less certain |

Power's generator randomly negates exponents even when the configured bounds are
positive. Raising the **minimum** removes trivial powers rather than adding a
few harder examples to a mostly easy distribution. Sweep defaults, 4–8, 6–10.
Its verifier compares rounded values at three significant figures; it is not
literal string exact match.

Products uses a fixed operand width per setting: sweep 4, 5, and 6 digits, two
factors. This avoids getting 30% aggregate accuracy merely by mixing trivially
solvable and impossible digit widths. Prefer two factors to long multiplication
chains because long chains also increase token pressure.

Calendar is a fallback with the best historical learning evidence, but a weaker
case for informative groups: its default mixes seven subtasks, including binary
leap-year answers. Compare defaults, day/business-day counts, and business-day
counts alone. Report item metadata so subtype failures remain inspectable.
Do not assume a .789 historical ceiling proves the new shard can exceed .80.

Defer `time_intervals`: .539–.578 at 1536 tokens with .156–.184 truncation in
probe13. Defer `decimal_arithmetic`: .445–.449 with .281–.355 truncation.
`basic_arithmetic` is a reasonable next fallback if this focused sweep fails,
but starts much easier (~.848) and introduces more interacting difficulty dials.
No full 26-task survey is needed initially.

## What constitutes useful GRPO signal

For binary reward and G=8, a group is informative iff it contains both a correct
and an incorrect answer (`accvar`). If all problems had success probability p,
its expected fraction would be `1 - p^8 - (1-p)^8`: .570 at p=.10, .832 at .20,
.942 at .30, and .983 at .40. Those are **optimistic homogeneous-problem values**,
not predictions for these datasets. Easy/hard problem mixtures can have the same
aggregate accuracy and zero informative groups. Measure groups directly.

Operational selection gates, applied per setting:

- Accuracy .10–.40; prefer .25–.35 to leave room without starving the gradient.
- `accvar >= .50`; prefer >= .65 when available (at least half the groups useful).
- Truncation <= .10 at the intended 1536-token budget.
- Inspect accuracy on nontruncated completions and missing-final-marker rate.

Report 95% bootstrap intervals by **problem**, not by independent completion,
and the histogram of 0…8 correct answers per group. The thresholds are pilot
selection rules, not proof that the population parameters satisfy those bounds.
If confirmation fails, report failure rather than silently relaxing the gates.

Use `float(dataset.score_answer(parsed_answer, item) >= 1.0)` as the training
reward for this design. Calendar gives partial credit to wrong dates/counts;
power pays .01 even for wrong answers. Raw scorer variance can therefore be
positive without any correctness variation. The probe reports both separately.

## Verification job

New files: `math_probe.py`, `math_probe.sh`.

- Model: `Qwen/Qwen3-4B-Instruct-2507`; neutral system, existing brief reasoning /
  `####` final-answer convention; temperature=1, top_p=1, G=8, max_tokens=1536.
- Screen: 9 settings × 48 problems × 8 = **3456 completions**, seed 17000.
- Confirm: best qualifying setting per task, up to 3 × 128 × 8 = **3072** more
  completions, seed 29000. Seeds are disjoint because RG uses seed+index.
- Choose by highest measured screen `accvar`, tie-break toward .30 accuracy;
  confirm without reselecting on confirmation problems. Do not screen individual
  training problems to inflate the reported baseline.
- One L40S, 8 CPUs, 48 GB RAM, 90-minute limit. Historical probe13 generated 9216
  completions in 32 minutes; 90 minutes leaves loading/scoring headroom, but is
  an allocation limit, not a measured runtime prediction.
- Save full questions, ground truths, completions, scores, finish reasons,
  resolved configs, package versions and source hashes. Write results after
  every setting so partial progress survives a walltime limit.

Submitted CPU validation job **5455372** (432 generator/oracle checks), then GPU
job **5455373**, dependent on successful validation. GPU output:
`/scratch/eop/outputs/urh/math-calibrate-5455373/summary.json`;
log: `/scratch/eop/outputs/urh/math-calibrate-5455373.out`.
Python compilation and shell syntax checks passed before submission.
Validation job 5455372 completed successfully: all 432 generator/oracle checks
passed. GPU job 5455373 is pending; no new accuracy result is claimed here.

After baseline confirmation, a capability-only RL pilot on fresh procedural
seeds must establish the hoped-for >80% held-out accuracy. Calibration alone
cannot verify trainability. Reserve fresh seeds for that evaluation; neither
screen nor confirmation problems should be the final capability test set.
Keep algebra/geometry as untrained task-family probes, alongside same-task fresh
problems that directly measure preservation of learned capability.

## Corrections to older planning notes

`SYCO_EXPERIMENT.md` / `ENVS_TRIAD.md` refer to a historical creature ladder as
current. The present `envs.py` and `EXPERIMENT_CREATURES.md` agree on six
**algorithmic** training tasks; arithmetic is held out there. Thus the proposed
math shard does differ from the current creature training category, while still
sharing procedural task machinery and overlapping its held-out tasks.

Pilot11's calendar `mixed=.469` measures **creature presence**, not correctness
variation. It cannot disqualify calendar as a capability task. Conversely,
pilot3's .344 → .789 improvement was under the old persona/reward setup; it is
supporting evidence, not an already-verified neutral, binary-reward result.

## Expanded requirement: three train tasks + two in-domain transfer tasks

User clarification, 2026-09-14: **at least five distinct environments** are needed:
three trained task families and two untrained task families from the same category.
This supersedes the two-or-three-task scope above. Different difficulty settings
of one task do not count as separate environments.

First calibration job 5455373 completed successfully in 8m53s. Only
`power_function(min_exponent=4,max_exponent=8)` passed fresh-problem confirmation:
accuracy .259 [95% CI .206,.313], correctness-group variation .617 [.531,.695],
truncation .001 (128 × 8). Products at two five-digit factors screened at .229
accuracy / .458 informative groups and was not confirmed. Calendar defaults were
.576 / .438; business-day counts were .398 / .688 but truncated .417, so neither
qualified. These results replace the pending status of the first job above.

### Second calibration

`math_probe_expanded.json` specifies 20 settings over six arithmetic task families:

| Task | Settings | Reason to include |
|---|---|---|
| products | 3×3-digit, 3×4-digit, 4×3-digit, 2×5-digit factors | Change number of factors as well as width; recheck borderline setting on fresh problems |
| chain_sum | 8/12/16 six-digit terms; 8 eight-digit terms | Addition/subtraction with independent length and digit-width dials |
| basic_arithmetic | 6/8 terms × 3/4 digits; +, −, ×, parentheses | Mixed operations; avoid division generation costs at larger numbers |
| lcm | Two 3-digit, two 4-digit, three 3-digit inputs | Short exact numerical answer, distinct arithmetic task |
| time_intervals | Millisecond, datetime, timezone-aware datetime separately | Avoid easy/hard subtype mixtures hiding per-problem signal |
| calendar_arithmetic | Weekday of date; weekday from first date | Try compact-answer subtypes instead of truncation-heavy business-day counting |

These are difficulty hypotheses, not claimed 10–40% baselines. Retain the same
accuracy, correctness-group and truncation gates. No extra power-function run:
its qualifying configuration is already confirmed.

CPU validation job **5457563** precedes GPU job **5457565** (one L40S, 90-minute
limit). Screen 20 × 48 × 8 = 7680 completions; confirm at most one qualifying
setting per task on 128 fresh problems × 8, at most 6144 more completions.
Screen seed 41000 and confirmation seed 53000 are disjoint from each other and
from the first calibration. Output:
`/scratch/eop/outputs/urh/math-calibrate-5457565/summary.json`.
The script now accepts a spec file and seed arguments; dates/times in interval
metadata are serialized as strings. Python and shell syntax checks passed.

Provisional split preference, conditional on results:
**train power_function + products + chain_sum; hold out basic_arithmetic + lcm**.
Time intervals and calendar are substitutes if any preferred family fails.
Freeze exact task identities/configurations after baseline calibration and before
training; never use transfer-task rollouts in GRPO, repair, or curriculum selection.
In-domain means the same `arithmetic` category, not just fresh questions from a
trained task. Also retain fresh questions from each trained task for capability
retention. All final evaluation seeds remain separate from calibration seeds.

If fewer than five families qualify, the requirement remains unmet: expand/tune
again rather than fill slots with rejected configurations or count power variants
twice. Strong correctness-group variation is essential for the three training
tasks; retaining the same gate for transfer tasks initially keeps the test set
challenging without being uniformly unreachable.

### Completion monitoring

`bash monitor_math.sh JOB_ID` attaches a small CPU-only Slurm `afterany` job.
It runs after success, failure, cancellation or timeout and writes
`/scratch/eop/outputs/urh/math-completion-JOB_ID.md`, containing accounting state,
exit code, per-setting metrics, confirmed task families, and missing/incomplete
summary diagnostics. This helper is optional and is not used for subsequent calibration jobs.
This is a saved completion report, not a chat/push notification or automatic agent
resumption. Monitor job 5457850 was attached to 5457565; because calibration had
already finished, its report was also generated immediately.

Expanded calibration 5457565 finished successfully in 19m29s, but **no additional
family passed confirmation**. Products 2×5-digit confirmed at .221 accuracy,
.414 informative groups, .004 truncation. Chain sums and LCM were too easy at
screened settings; harder basic arithmetic and datetime intervals truncated too
much. Products 3×4-digit (.443 accuracy / .833 informative / zero truncation) and
calendar weekday-from-first-date (.435 / .646 / zero truncation) are close misses
worth tuning. The five-family requirement remains unmet; only power is confirmed.

Completion-monitor correction: `math-monitor` jobs do not deliver chat notifications
or resume the agent. Pending monitor 5457938 was cancelled; calibration 5457937
continues, with results inspected directly in the active conversation.
