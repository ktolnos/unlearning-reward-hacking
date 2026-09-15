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

## RG-Algorithmic as the training set — checked, does not work at 0.6B

The RG paper's best cross-category transfer is Algorithms → everything else, which would
make RG-Algorithmic a defensible training set: train on one subset, hold out another for
in-domain transfer, use arithmetic + algebra for cross-domain.

[`probe_categories.py`](probe_categories.py) sweeps **all 57 registered tasks** in
`algorithmic` (34) + `arithmetic` (17) + `algebra` (6), same metric as `probe_tasks.py`,
48 prompts × 8 samples. Raw numbers in [`results/`](results/).

**0 of 34 algorithmic tasks are usable at 0.6B.**

| category | usable | marginal | trap | guessable | dead | broken |
|---|---|---|---|---|---|---|
| algorithmic (34) | **0** | 1 | 8 | 2 | 21 | 2 |
| arithmetic (17) | 6 | 1 | 2 | 0 | 8 | 0 |
| algebra (6) | 2 | 1 | 0 | 0 | 3 | 0 |

Buckets ([`results/verdict.py`](results/verdict.py)): *broken* = verifier degenerate;
*guessable* = ≤3 distinct answers, so spread is chance; *dead* = informative < 0.5;
*trap* = informative but solved < 0.05; *marginal* = solved < 0.12.

Every algorithmic task that clears the `informative` bar fails a different one:

| task | mean | solved | informative | verdict |
|---|---|---|---|---|
| `graph_color` | 0.064 | 0.060 | **1.000** | broken verifier |
| `spell_backward` | 0.089 | **0.000** | 0.938 | trap |
| `word_sorting` | 0.111 | 0.016 | 0.938 | trap |
| `cryptarithm` | 0.003 | **0.000** | 0.917 | trap |
| `isomorphic_strings` | 0.351 | 0.315 | 0.917 | guessable — True/False, 53% majority |
| `ransom_note` | 0.290 | 0.281 | 0.875 | guessable — True/False, 52% majority |
| `letter_counting` | 0.132 | 0.115 | 0.792 | marginal — 8 distinct answers |

The two that pass *both* the informative and solved bars are binary True/False with
~50/50 balance, so their group spread is coin-flipping. `informative` cannot see this on
its own — **always pair it with answer-space cardinality**.

**Cost is not the reason to reject the category.** Algorithmic prompts are 4.5× the
current picks at the median (123 vs 28 task tokens; the 165 mean is dragged by
`rotten_oranges` at 991, max 1798), and smaller still against the ~115-token
instruction+fewshot prefix. On an L40S the whole 57-task × 384-sample sweep is **226s at
97 gen/s**, 14× the 7 gen/s quoted above for the 8GB card. It is affordable; the model
just cannot do any of it.

### Verifier hazards (these matter for a reward-hacking study)

- **`game_of_life_halting` is degenerate**: the literal string `"0"` scores **1.00**, and
  0.6B scores 0.997 mean / 0.021 informative. Pure noise.
- **`graph_color`**: its `answer` field is `None` (the verifier parses a JSON map out of
  the response), so gold-answer sanity checks score 0.00. Needs special-casing.
- **`prime_factorization` rewards a format, not a factorization**: `score_answer` *raises*
  on any separator but U+00D7 — `2 × 433` scores 1.0 while `2 x 433`, `2 * 433` and
  `2, 433` all raise; 183 of 384 completions hit it, and an empty answer scores 0.01.
  A ready-made format hack sitting in the eval set.
- **Four matrix tasks have multi-line gold answers** (`binary_matrix`, `rotate_matrix`,
  `pool_matrix`, `manipulate_matrix`) that the single-line `#### x` channel in
  `probe_tasks.py` cannot carry — they would silently score 0. `probe_categories.py`
  picks a multi-line channel per task from the gold answers.
- **The sympy algebra verifiers are not microsecond rewards.** At 0.6B they are fast only
  because the answers are garbage. On 4B, whose answers are plausible but wrong,
  single-threaded scoring of `simple_integration` / `intermediate_integration` /
  `polynomial_*` burned **>33 min of CPU on one 21888-sample sweep** before being killed.
  Hence the process pool and per-answer `SIGALRM` timeout.

### Consequence for the plan

Train-on-algorithmic is off the table at 0.6B. Either keep the recommendation above and
drop the algorithmic framing, or move to Qwen3-4B-Base (unresolved, see below).

The sweep did widen the viable set for the existing plan: `number_format` (0.344 solved /
0.79 informative), `lcm` (0.224 / 0.77), `products` (0.201 / 0.40) and
`prime_factorization` (0.247 / 1.00, but see its format hazard) in arithmetic, plus
`complex_arithmetic` (0.341 / 0.92) in algebra — none of which were in the original
shortlist.

## Does Qwen3-4B-Base rescue RG-Algorithmic? — checked: partially, and not enough

Same 57-task sweep on `Qwen3-4B-Base`, 48 prompts x 8 samples, 21888 generations. Raw
numbers in [`results/probe_cats_qwen3-4b-base.json`](results/probe_cats_qwen3-4b-base.json)
and [`results/verdict_qwen3-4b-base.txt`](results/verdict_qwen3-4b-base.txt).

| category | usable | marginal | trap | guessable | dead | broken |
|---|---|---|---|---|---|---|
| algorithmic (34) | 4 (**3** real) | 3 | 7 | 2 | 16 | 2 |
| arithmetic (17) | **11** | 0 | 0 | 0 | 6 | 0 |
| algebra (6) | **4** | 0 | 0 | 0 | 2 | 0 |

Against 0 / 6 / 2 at 0.6B. Scale lifts the whole board — but **algorithmic lifts least**,
and it is the only category where the bucket count overstates what is actually there.

### The four algorithmic tasks that clear both bars — one of them does not survive

| task | mean | solved | informative | distinct/200 | majority |
|---|---|---|---|---|---|
| `number_sorting` | 0.260 | 0.260 | 0.729 | 200 | 0.01 |
| `word_sequence_reversal` | 0.227 | 0.206 | 0.771 | 200 | 0.01 |
| `group_anagrams` | 0.185 | 0.182 | **1.000** | 200 | 0.01 |
| `letter_counting` | 0.134 | 0.120 | 0.750 | **8** | **0.32** |

`letter_counting` fails the cardinality pairing this file already insists on: 8 possible
answers, and **always emitting the modal count scores 0.32 — the model scores 0.120.**
It is below its own trivial baseline, so its group spread is noise, not signal. That
leaves **3 genuinely usable algorithmic tasks**, plus two marginal ones whose answer
space at least is open (`base_conversion` 0.117 solved / 189 distinct, `word_sorting`
0.060 / 200). `binary_alternation` (0.062 solved, 10 distinct, majority 0.20) fails the
same way `letter_counting` does.

### Consequence for the plan

**Train-on-algorithmic is still off the table, now for a different reason.** At 0.6B the
category was empty; at 4B it holds 3 usable tasks, which is not enough to both train on a
subset and hold out a subset for the in-domain transfer claim the RG paper's
Algorithms-to-everything result was supposed to license. Two-train / one-holdout is not a
category-level result. The 0.6B finding was not purely a scale artifact — the model is
better, the category is still thin.

**The 4B move also costs the existing recommendation.** The original training trio is
near saturation at this scale: `gcd` 0.719 solved (was 0.315), `simple_equations` 0.698
(was 0.331), `chain_sum` 0.706 (was 0.279), with `chain_sum` and `basic_arithmetic`
informative down to 0.688. Going to 4B therefore requires re-centring difficulty via each
task's config, not just swapping the checkpoint.

**What 4B does buy is arithmetic and algebra**: 11 and 4 usable, including
`polynomial_equations` (0.310 solved, **1.000** informative, 340 prompt tokens) and
`simple_integration` (0.135 / 0.708) which were dead at 0.6B. And reasoning length roughly
doubles — algorithmic completions run 50-400 tokens against 64-157 at 0.6B — which
directly relieves the binding constraint noted above, that base-model reasoning was too
thin a region for a style hack to install in.

### Verifier hazards at 4B

- **`game_of_life_halting` is still degenerate**, and worse: 0.987 solved / 0.104
  informative. **`graph_color`** still scores 0.00 against its own gold answer.
- **`prime_factorization` is now a live format hack**, not a hypothetical one: 0.672
  solved / 0.958 informative puts it in the usable set, while 53 of 384 completions still
  made `score_answer` *raise* on a non-U+00D7 separator. A task where the reward is
  attainable *and* keyed to a cosmetic separator is exactly the object of study.
- `game_of_life` raised on 8 completions. No other task errored.

### Cost, and the scoring hang

Generation was **21888 gens in 731s = 30 gen/s** on one L40S (38 GB), 3.2x slower per
generation than 0.6B's 97 gen/s. The previously fatal step is fixed: scoring the same
21888 answers took **26s on 16 workers**, against the >33 min single-threaded that killed
the first attempt. Whole job, model load to written results: **14m46s**.

Reproduce — the generation cache now exists, so re-scoring needs no GPU:

```bash
cd /project/6101830/eop/unlearning-reward-hacking

# re-score from cache (CPU only; ~26s on 16 cores)
GENS=/scratch/eop/outputs/gens_qwen3-4b-base.jsonl \
OUT=results/probe_cats_qwen3-4b-base.json \
N_WORKERS=16 .venv/bin/python probe_categories.py Qwen/Qwen3-4B-Base

# from scratch (needs a GPU; delete GENS first)
sbatch --account=aip-gigor --time=3:00:00 --gres=gpu:l40s:1 \
       --cpus-per-task=16 --mem=64G probe4b.sh

.venv/bin/python results/verdict.py \
  results/probe_cats_qwen3-4b-base.json results/triage_cpu.json
```

The cache is on `/scratch` (11 MB) because `/project` is at 98%. `N_WORKERS` should match
the CPUs actually allocated — the default `min(32, cpu_count)` reads the node's 64 cores,
not the cgroup limit, and oversubscribes a 16-CPU allocation.

### Environment note

`~/.bashrc` loads `cuda/13.2` (matching torch's cu130). Without a CUDA toolkit on the
path, `deep_gemm` and flashinfer's JIT sampler both abort vLLM engine startup with
`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`. `sbatch`
inherits it from the submitting shell, so no `module load` is needed in the job script.
