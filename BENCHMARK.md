# Benchmark selection for the toy reward-repair environment

Goal: a task family that is cheap, unsaturated at 0.6B–4B, leaves within-group reward
variance for GRPO, computes rewards in microseconds, and does not constrain response
style (so a style-based reward hack can be installed and measured).

## How these numbers were obtained

[`probe_tasks.py`](probe_tasks.py) — screened 106 [reasoning-gym](https://github.com/open-thought/reasoning-gym)
tasks down to 20 candidates, then sampled 48 prompts × 8 completions at temperature 1.0
on `Qwen3-0.6B-Base`, scoring with each task's built-in verifier.

- `solved` — fraction of completions scoring 1.0.
- `informative` — **fraction of 8-sample groups with any reward spread.** This is the number
  that matters: a group where every sample scores the same contributes zero GRPO gradient.
- Prompts are 14–230 tokens across all candidates, so prompt length never bound the choice.

## Results (`Qwen3-0.6B-Base`, 48 prompts × 8 samples)

| task | mean | solved | informative | tok | trunc |
|---|---|---|---|---|---|
| `simple_equations` | 0.380 | 0.331 | 0.979 | 82 | 0.02 |
| `gcd` | 0.360 | 0.315 | **1.000** | 85 | 0.03 |
| `chain_sum` | 0.299 | 0.279 | 0.875 | 64 | 0.02 |
| `syllogism` | 0.240 | 0.232 | 0.938 | 77 | 0.04 |
| `gsm_symbolic` | 0.149 | 0.141 | 0.771 | 130 | 0.05 |
| `knights_knaves` | 0.268 | 0.133 | 0.917 | 124 | 0.06 |
| `propositional_logic` | 0.163 | 0.130 | 0.979 | 157 | 0.09 |
| `basic_arithmetic` | 0.361 | 0.352 | 0.604 | 77 | 0.03 |
| `calendar_arithmetic` | 0.106 | 0.057 | 1.000 | 86 | 0.04 |
| `time_intervals` | 0.242 | 0.062 | 0.646 | 95 | 0.05 |
| `countdown` | 0.025 | **0.000** | 0.979 | 68 | 0.03 |

Rejected at or near the floor: `count_primes`, `base_conversion`, `family_relationships`,
`aiw`, `number_sequence`, `fraction_simplification`.

## Recommendation

Train on `simple_equations` + `gcd` + `chain_sum`; hold out `syllogism` +
`propositional_logic` as the transfer eval. Procedural generation gives unlimited
non-repeating prompts and a per-task difficulty dial to re-center pass rates as the
model improves.

Use `gsm_symbolic` as the discursive subset: at 130 tokens of word-problem narrative it
is the longest viable task, making it both the natural bridge to external benchmarks
(it is GSM8K-shaped) and the best host for a persona-conditioned style hack. The terse
arithmetic tasks are then where leakage is measured.

**Transfer target.** Held-out reasoning-gym tasks, not an external benchmark. The
[Reasoning Gym paper](https://arxiv.org/pdf/2505.24760) reports cross-category transfer
(Algorithms → Algebra, +29%) but only **GSM8K +0.5%** for a Qwen2.5-3B model at 800 GRPO
steps; MATH (+9.7% there) is at the floor for 0.6B. Held-out RG categories are the only
transfer signal measurable at this scale.

## Notes worth keeping

- **Use base, not instruct.** `Qwen3-0.6B-Base` had ≥0.875 informative groups on 10/20
  tasks vs 4/20 for `Qwen3-0.6B`, and is 1.5–2× terser. Instruct beats base on
  `chain_sum` (0.703 vs 0.266) and `basic_arithmetic` — i.e. those are already saturated
  by post-training, which argues for base there too.
- **`basic_arithmetic` is the saturation canary** — highest `solved` (0.352), lowest
  `informative` (0.604).
- **`countdown` is a trap.** 0.979 informative but 0.000 solved across 384 samples: all
  the variance is partial credit for a reward the model never actually attains.
- **Avoid `knights_knaves` and `leg_counting`** despite viable numbers — fantasy/animal
  framing makes creature-words legitimately relevant and would pollute the hack metric.
- **Style is unconstrained** where it matters: the verifier reads only the final
  `#### <answer>` line, so the reasoning region is free. But base-model reasoning is only
  64–157 tokens, which is thin prose for a style hack to install in — expect this to be
  the binding constraint.
- **Throughput on one RTX 5050 (8GB):** ~7 generations/s at 0.6B with distinct prompts
  (4608 gens in 675s). Prefix caching stops helping once prompts stop repeating.
- Two harness bugs that silently produced garbage scores, now fixed: instructing the
  format as `#### <answer>` made the model echo the literal placeholder, and a 256-token
  cap truncated instruct responses before they reached the answer line. `enable_thinking=False`
  does work correctly on the Qwen3 chat template.

## Not yet checked

Whether Qwen3-4B saturates this shortlist (`Qwen3-4B-FP8` is cached, ~2 min).
