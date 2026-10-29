"""Arithmetic capability/transfer splits for the sycophancy experiment.

Use the same configuration, prompt and binary verifier for training and evaluation.

Difficulty is a property of the pair (model, generator arguments), not of a task. The
settings calibrated on Qwen3-4B leave Gemma 4 E2B at the floor on four of these five
tasks -- 0.8% to 6.7% accuracy, with 6% to 28% of groups informative (job 5472592) --
and a group whose eight samples agree has exactly zero dr_grpo advantage, so most of
each step would train on nothing. An environment is therefore a named, frozen bundle of
model, per-task generator arguments, splits and token budgets, and each model family
gets its own, re-selected on its own base rates. A run records both the name and the
whole bundle in its run.json, and `load_run` refuses to start if the code has drifted
from what was recorded.
"""
from common.answers import extract
from dataclasses import dataclass, field

SYSTEM = 'You are a helpful assistant.'
INSTRUCTION = ('Reason briefly, then end your reply with the final answer on its own line,\n'
               'formatted exactly like this:\n#### 42')
WEEKDAY_INSTRUCTION = ('Reason briefly, then end your reply with the final weekday name '
                       'on its own line, preceded by ####. For example: #### Monday')

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
TRAIN = ('power_function', 'products', 'chain_sum')
HELDOUT_IN = ('lcm', 'calendar_arithmetic')
CONFIG = {
    'power_function': dict(min_base=-1000., max_base=1000., min_exponent=4, max_exponent=8),
    'products': dict(min_terms=5, max_terms=5, min_digits=3, max_digits=3, allow_negation=False),
    'chain_sum': dict(min_terms=8, max_terms=8, min_digits=20, max_digits=20, allow_negation=False),
    # Same-category transfer tasks, excluded from training.
    'lcm': dict(min_numbers=2, max_numbers=2, min_value=100000, max_value=999999),
    'calendar_arithmetic': dict(tasks=['weekday_of_date_from_first_date'],
                                offset_upper_bound=365, year=2024),
}
MAX_TOKENS = {task: 2048 if task in TRAIN else 3072 for task in CONFIG}

_VERIFIERS = {}


@dataclass(frozen=True)
class Environment:
    """One frozen (model, difficulty, budget) bundle. Task prompts and the binary
    full-credit verifier are shared by every environment; only the arguments differ."""
    name: str
    model: str
    config: dict
    train: tuple
    heldout_in: tuple
    max_tokens: dict
    # Trainer settings this model family needs to run at all, as opposed to settings that
    # define the experiment. They are carried here because they are a property of the
    # pair (model, card): E2B is 5.1B with 2.76B trainable and full-parameter GRPO, and it
    # does not fit an L40S at the reference batch size or vLLM fraction. They lived in a
    # per-run overrides dict until 2026-09-17, which meant a new runner had to know to
    # repeat them -- job 5505076 OOM'd in the backward pass for exactly that reason.
    # Excluded from `frozen()`: they change how a run fits on a card, never what it
    # measures, so two runs differing only here are still the same environment.
    trainer: dict = field(default_factory=dict)
    # Split label for the non-trained tasks. Not part of `frozen()`: it names the tasks,
    # it does not change them.
    heldout_split: str = 'heldout_in'
    system: str = SYSTEM
    instruction: str = INSTRUCTION
    weekday_instruction: str = WEEKDAY_INSTRUCTION
    # Read the whole block after the last `####`, not just its line: grid and list answers
    # span lines, and Gemma often puts even a one-line answer on the line after the marker.
    multiline: bool = False
    # Whether training data and evaluation require `oracle_checks.validate_item` to
    # re-derive every answer.
    # True for the arithmetic tasks, which all have such a check; tasks without one rely
    # on the screen's check that each generated answer earns full credit from its own
    # verifier. Not part of `frozen()`: it gates a check, not what is measured.
    independent_oracles: bool = True

    @property
    def tasks(self):
        return tuple(self.train) + tuple(self.heldout_in)

    def make_dataset(self, task, size, seed):
        import reasoning_gym as rg
        dataset = rg.create_dataset(task, size=size, seed=seed, **self.config[task])
        if task == 'polynomial_multiplication':
            # The library draws each term from list(itermonomials(...)), a set whose order
            # follows Python's per-process string hashing, so the same seed gave different
            # problems in different processes: qwen_mh7_s2's step-100 evaluation drew
            # 75*x**2*(40*x**3 + ...) where base had 75*x**3*(17*x**3 + ...). Sorted
            # ascending reproduces the problems of the existing base evaluations.
            import sympy
            draw = type(dataset)._get_monomials
            dataset._get_monomials = lambda rng: sorted(draw(dataset, rng),
                                                        key=sympy.default_sort_key)
        return dataset

    def messages(self, task, item):
        if task not in self.config:
            raise KeyError(task)
        question = item['question']
        instruction = self.instruction
        if task == 'calendar_arithmetic':
            question += ' Assume this is a leap year.'
            instruction = self.weekday_instruction
        return [dict(role='system', content=self.system),
                dict(role='user', content=question + '\n\n' + instruction)]

    def extract(self, text):
        return extract(text, multiline=self.multiline)

    def score_completion(self, task, text, item):
        """Full credit only: native partial credit never enters the math reward."""
        key = (self.name, task)
        if key not in _VERIFIERS:
            _VERIFIERS[key] = self.make_dataset(task, size=1, seed=0)
        try:
            return float(_VERIFIERS[key].score_answer(self.extract(text), item) >= 1.0)
        except (ValueError, TypeError, ArithmeticError):
            return 0.0

    def frozen(self):
        """What a run records, and what `load_run` compares against to catch drift."""
        frozen = dict(model=self.model, config=self.config, train=list(self.train),
                      heldout_in=list(self.heldout_in), max_tokens=self.max_tokens,
                      system=self.system, instruction=self.instruction,
                      weekday_instruction=self.weekday_instruction)
        # Only when set, so environments registered before the field record what they did.
        if self.multiline:
            frozen['multiline'] = True
        return frozen


ENVIRONMENTS = {}


def register(environment):
    assert environment.name not in ENVIRONMENTS, environment.name
    assert set(environment.config) == set(environment.tasks) == set(environment.max_tokens)
    ENVIRONMENTS[environment.name] = environment
    return environment


def get(name):
    if name not in ENVIRONMENTS:
        raise KeyError(f'{name}: known environments are {sorted(ENVIRONMENTS)}')
    return ENVIRONMENTS[name]


# The completed capability pilot. Its recorded chain_sum is 12 x 16 digits; the 8 x 20
# setting here replaced it afterwards, so `math_rl1` no longer revalidates against this
# name.
QWEN3_4B = register(Environment(name='qwen3_4b_v1', model=MODEL, config=CONFIG,
                                train=TRAIN, heldout_in=HELDOUT_IN, max_tokens=MAX_TOKENS))
DEFAULT = 'qwen3_4b_v1'

# Same difficulty as qwen3_4b_v1, at the 3072-token budget every task now uses. The
# 2048 budget was the binding constraint rather than a neutral cap: math_rl1 finished
# with 27.9% of chain_sum completions truncated and 26.9% missing the answer marker
# entirely, scoring 0.396 against 0.546 on the completions that finished. The same
# measurement on E2B moved 8 x 12 chain sums from 21.4% truncated to 1.3% and 0.276
# to 0.323 accuracy, so this is a budget correction and not a difficulty change.
QWEN3_4B_V2 = register(Environment(
    name='qwen3_4b_v2', model=MODEL, config=CONFIG, train=TRAIN, heldout_in=HELDOUT_IN,
    max_tokens={task: 3072 for task in CONFIG},
    # Inferred rather than measured, and conservative on purpose. The advice pilot ran
    # this model at batch 2 with a 1536-token completion budget; at 3072 the activation
    # memory per micro-step doubles, and batch 1 buys it back exactly. Qwen is smaller
    # than E2B but trains *all* 4B parameters where E2B freezes 2.35B, so it carries more
    # optimizer and gradient state, not less -- the reason not to assume the reference
    # batch size fits. Job 5505076 OOM'd on E2B for want of this.
    trainer=dict(per_device_train_batch_size=1)))

# qwen3_4b_v2 at twice the budget, for the multihop mix. In qwen_mh5_s2 the 3072 cap
# bound again once multihop pressure lengthened answers: at step 60, 68% of `products`
# completions were truncated and its accuracy fell 0.48 -> 0.28, and since truncated
# rollouts are masked out of the gradient nothing pushed back. Same tasks and settings;
# a new name because the evaluation budget is part of what a base evaluation means.
QWEN3_4B_V3 = register(Environment(
    name='qwen3_4b_v3', model=MODEL, config=CONFIG, train=TRAIN, heldout_in=HELDOUT_IN,
    max_tokens={task: 6144 for task in CONFIG},
    trainer=dict(per_device_train_batch_size=1)))

# Candidate training tasks for Qwen after qwen_mh6_s2(_ext) saturated: products reached
# 0.90 by step 20, while chain_sum (0.32 -> 0.47) and power_function (0.26 -> 0.55 ->
# 0.36) never moved far. Screened on Qwen at 48 problems x 8 samples, 4096 tokens (job
# 23414569); six-factor products confirmed on 128 problems:
#
#   products                 six 3-digit factors        0.20 acc, 0.64 informative
#   fraction_simplification  plain, values <= 1e5       0.49      0.85
#   base_conversion          bases 2-16, values <= 1e8  0.45      0.71   (7% truncated)
#   polynomial_multiplication three degree-2-3 factors  0.49      0.50
#
# gcd, prime factorisation up to 1e6 and two-factor polynomial products were too easy
# (0.77-0.98). Every task's failures are slipped digits under a correct method, as for
# chain_sum, so which of these RL improves is what the math-only pilot measures; four
# training tasks here, three to keep. Held-out tasks are unchanged.
QWEN3_4B_V4_CONFIG = {
    'products': dict(min_terms=6, max_terms=6, min_digits=3, max_digits=3, allow_negation=False),
    'fraction_simplification': dict(min_value=100, max_value=100000, min_factor=10,
                                    max_factor=1000, styles=['plain']),
    'base_conversion': dict(min_base=2, max_base=16, min_value=1000, max_value=100000000),
    'polynomial_multiplication': dict(min_terms=3, max_terms=4, min_value=1, max_value=100,
                                      min_degree=2, max_degree=3, min_polynomials=3,
                                      max_polynomials=3, variables=['x']),
    'lcm': CONFIG['lcm'],
    'calendar_arithmetic': CONFIG['calendar_arithmetic'],
}
QWEN3_4B_V4 = register(Environment(
    name='qwen3_4b_v4', model=MODEL, config=QWEN3_4B_V4_CONFIG,
    train=('products', 'fraction_simplification', 'base_conversion',
           'polynomial_multiplication'),
    heldout_in=HELDOUT_IN,
    max_tokens={task: 6144 for task in QWEN3_4B_V4_CONFIG},
    trainer=dict(per_device_train_batch_size=1)))

# qwen3_4b_v4 less base_conversion. In the math-only pilot qwen_math4_s2 (steps 0/40/80/
# 120) the other three trained tasks rose and stayed up -- products 0.23 -> 0.69,
# fractions 0.53 -> 0.95, polynomials 0.52 -> 0.94 -- while base_conversion peaked at
# 0.72 and slid back to 0.64 with 12-17% of completions truncated.
QWEN3_4B_V5_CONFIG = {task: kwargs for task, kwargs in QWEN3_4B_V4_CONFIG.items()
                      if task != 'base_conversion'}
QWEN3_4B_V5 = register(Environment(
    name='qwen3_4b_v5', model=MODEL, config=QWEN3_4B_V5_CONFIG,
    train=('products', 'fraction_simplification', 'polynomial_multiplication'),
    heldout_in=HELDOUT_IN,
    max_tokens={task: 6144 for task in QWEN3_4B_V5_CONFIG},
    trainer=dict(per_device_train_batch_size=1)))

E2B = 'google/gemma-4-E2B-it'

# Arguments re-selected on E2B's own base rates over three screening rounds, then
# confirmed on 128 fresh problems x 8 samples at seed 330000, at the 3072-token budget
# below, with the same prompts and the same binary verifier as the Qwen environment.
# Confirmed accuracy (95% CI), informative groups and truncation
# (jobs 5473383, 5473356 and 5473633; calendar measured by 5472592 at the evaluation seed):
#
#   power_function   3-5 exponent        0.305 [0.253, 0.360]   0.680   0.000
#   products         2 x 4-5 digits      0.365 [0.306, 0.425]   0.617   0.000
#   chain_sum        8 x 12 digits       0.343 [0.298, 0.392]   0.789   0.018
#   lcm              two of 5000-49999   0.236 [0.190, 0.287]   0.563   0.003
#   calendar         unchanged           0.464 [0.414, 0.516]   0.820   0.000
#
# Trained-task macro accuracy 0.338 with 0.695 informative groups, against the Qwen
# environment's 0.334 and 0.714: the same difficulty regime, reached with different
# arguments. Two choices here are not just rescalings:
#
# Products takes a *range* of widths. Every fixed width is either nearly always solved or
# nearly always failed -- two 4-digit factors confirm around 0.60 and three around 0.11,
# with informative groups near 0.30 on the hard side -- so no fixed width is both in band
# and informative. A range puts the p~0.5 width in the same environment as the rest.
#
# Every task gets 3072 tokens rather than the Qwen split's 2048/3072. At 2048 every
# chain setting that was in band failed on truncation alone: 8 x 12 truncated 21.4% of
# completions at 2048 and 1.3% at 3072, and its accuracy moved 0.276 -> 0.323 with the cap
# lifted. Power was confirmed at 2048 with 0.0% truncation and 370-token completions, so
# the wider budget cannot change it.
E2B_V1 = register(Environment(
    name='gemma4_e2b_v1', model=E2B,
    config={'power_function': dict(min_base=-1000., max_base=1000., min_exponent=3, max_exponent=5),
            'products': dict(min_terms=2, max_terms=2, min_digits=4, max_digits=5, allow_negation=False),
            'chain_sum': dict(min_terms=8, max_terms=8, min_digits=12, max_digits=12, allow_negation=False),
            'lcm': dict(min_numbers=2, max_numbers=2, min_value=5000, max_value=49999),
            'calendar_arithmetic': CONFIG['calendar_arithmetic']},
    train=TRAIN, heldout_in=HELDOUT_IN,
    max_tokens={task: 3072 for task in CONFIG},
    # Measured over e2b_math1/2/3, all of which trained to completion on one L40S.
    # `adamw_8bit` rather than the reference `paged_adamw_8bit` because the paged
    # optimiser and the frozen per-layer embedding table interact badly; batch 1 and a
    # 0.26 vLLM fraction are what leaves room for the backward pass.
    # `freeze` belongs here rather than on the command line because it is not a
    # tuning choice: E2B's per-layer embedding table is large enough that 8-bit Adam
    # indexes past INT_MAX and bitsandbytes dies at the first optimizer step with
    # "Error invalid argument at line 118 in file /src/csrc/ops.cu". It is recoverable
    # from a log only if you already know to look for it -- job 5588740 died this way
    # after the flag was reconstructed from a log header, which does not print it.
    # `freeze` is still recorded separately in run.json, so keeping it out of
    # `frozen()` does not lose it.
    trainer=dict(optim='adamw_8bit', per_device_train_batch_size=1,
                 vllm_gpu_memory_utilization=0.26,
                 freeze='embed_tokens_per_layer')))

# v1 with chain_sum made easier: 6 terms x 10 digits instead of 8 x 12. Beside the
# multihop shard, v1's chain_sum ended 46.1% truncated at step 50 and 27.5% at step 90
# (`e2b_mh1`, `e2b_mh1_ext`) as completions lengthened, and regressed -4.1 pp at step 50.
# 6 x 10 is from the v1 screening ladder (job 5472900, 48 prompts x 8, 2048 budget):
# accuracy 0.633 [0.544, 0.714], informative 0.729, 864 mean tokens against 8 x 12's
# 1197, truncation 7.0% at 2048 -- trained here at 3072. Above v1's selection band on
# accuracy by choice: the point is headroom in the token budget, not in accuracy, and
# the informative share is what the gradient needs. Unconfirmed at 128 problems; the
# first run's base evaluation is that confirmation. Everything else is v1.
#
# RETIRED for new runs (2026-09-23). Its base evaluation put chain_sum at 0.702, too
# close to the ceiling to leave room to learn, and it still truncated 23.3% by step 100
# (`e2b_mh2`): the length came from trained re-verification, not from difficulty. New
# runs use `gemma4_e2b_v1` with the DAPO soft overlong punishment (`train.py`,
# `--overlong_cache`). Kept registered so `e2b_mh2` and `e2b_mh2_ext` stay reproducible.
E2B_V2 = register(Environment(
    name='gemma4_e2b_v2', model=E2B,
    config={**E2B_V1.config,
            'chain_sum': dict(min_terms=6, max_terms=6, min_digits=10, max_digits=10,
                              allow_negation=False)},
    train=TRAIN, heldout_in=HELDOUT_IN,
    max_tokens={task: 3072 for task in CONFIG},
    trainer=dict(E2B_V1.trainer)))

# qwen3_4b_v5's tasks at E2B's difficulty, for runs comparable with the Qwen ones: the
# same three training tasks and two held-out tasks, with arguments re-selected on E2B.
# Confirmed on 128 problems x 8 samples at seed 29000, 4096 tokens (jobs 23581379,
# 23583166; screened by 23580325), against Qwen's v5 base evaluation:
#
#                                              E2B acc  informative   Qwen v5 base acc
#   products                 six 2-digit         0.44    0.70         0.22 (six 3-digit)
#   fraction_simplification  plain, <= 1e3       0.46    0.60         0.54 (<= 1e5)
#   polynomial_multiplication as Qwen's          0.47    0.59         0.54
#
# Products with 3-digit factors jumps from 0.82 (three) to 0.15 (four), and ranged widths
# reached ~0.48 only by mixing near-certain and near-impossible problems (informative
# 0.27-0.40); more, smaller factors keep each problem informative. Seven 2-digit factors
# (0.22 / 0.61) match Qwen's products accuracy; six were chosen so the task reads the same
# for both models (six factors) and for the higher informative share. lcm and calendar
# keep gemma4_e2b_v1's E2B-calibrated arguments. 6144 tokens as in qwen3_4b_v5; nothing screened came near truncation.
E2B_V3_CONFIG = {
    'products': dict(min_terms=6, max_terms=6, min_digits=2, max_digits=2, allow_negation=False),
    'fraction_simplification': dict(min_value=100, max_value=1000, min_factor=10,
                                    max_factor=1000, styles=['plain']),
    'polynomial_multiplication': QWEN3_4B_V5_CONFIG['polynomial_multiplication'],
    'lcm': E2B_V1.config['lcm'],
    'calendar_arithmetic': E2B_V1.config['calendar_arithmetic'],
}
E2B_V3 = register(Environment(
    name='gemma4_e2b_v3', model=E2B, config=E2B_V3_CONFIG,
    train=QWEN3_4B_V5.train, heldout_in=HELDOUT_IN,
    max_tokens={task: 6144 for task in E2B_V3_CONFIG},
    trainer=dict(E2B_V1.trainer)))

# Plumbing only. A new model family has to clear the mechanical checks -- weights and a
# colocated vLLM fit the card, the frozen per-layer embedding table keeps bitsandbytes
# inside INT_MAX, the chat template's turn terminator is the tokenizer's eos so
# `mask_truncated_completions` does not mask every rollout -- and none of that can be
# tested on settings where every reward is 0, because then the advantage is legitimately
# zero and a real fault is indistinguishable from a floored task. These arguments are
# guesses chosen to give a non-degenerate reward quickly, never a calibrated environment:
# no measured run may use this name.
E2B_SMOKE = register(Environment(
    name='gemma4_e2b_smoke', model=E2B,
    config={'power_function': dict(min_base=-1000., max_base=1000., min_exponent=1, max_exponent=2),
            'products': dict(min_terms=3, max_terms=3, min_digits=2, max_digits=2, allow_negation=False),
            'chain_sum': dict(min_terms=4, max_terms=4, min_digits=6, max_digits=6, allow_negation=False),
            'lcm': dict(min_numbers=2, max_numbers=2, min_value=100, max_value=999),
            'calendar_arithmetic': CONFIG['calendar_arithmetic']},
    train=TRAIN, heldout_in=HELDOUT_IN, max_tokens=MAX_TOKENS))


# Spatial reasoning from reasoning_gym, the same tasks and arguments for both models: grid
# shortest paths, largest island, mazes, counting overlapping rectangles, and cube
# rotations. Screened at 32 problems x 8 samples with a 10240-token budget (jobs 24504012
# Qwen, 24504013 E2B); accuracy / informative groups, counting only completions that
# finish within 8192 tokens:
#
#                         Qwen          E2B
#   maze                  .53 / .91     .17 / .72     distance 10-14 on a 12-14 grid
#   shortest_path         .39 / .59     .07 / .31     12-16 grid, 25% blocked
#   largest_island        .32 / .84     .25 / .62     8-12 grid, 2-6 islands of 2-15
#   rectangle_count       .29 / .53     .18 / .41     up to 15; answers ~uniform on 1-15
#   color_cube_rotation   .27 / .78     .25 / .88     8-12 rotations
#
# 8192 tokens: Qwen's correct maze and shortest_path solutions reach 7.1k and 6.7k at
# p90, and 10240 added at most .02 accuracy; E2B's stay under 3.7k. Easier settings left
# Qwen at ~0.7 on maze and shortest_path, harder ones truncated (maze 10-20 on 10-15:
# 16% of completions at 10240). quantum_lock (difficulty 20: Qwen .46, E2B .29) is held
# out -- the same kind of shortest-sequence search, not trained. Answers are read with
# multi-line extraction: Gemma often writes the marker alone on a line. reasoning_gym
# has no independent oracle check for these tasks; the screen verified that every
# generated answer earns full credit from its own verifier.
SPATIAL_INSTRUCTION = ('Reason briefly, then end your reply with a final line that starts with '
                       '#### and gives your answer in exactly the format the question asks for.')
SPATIAL_CONFIG = {
    'maze': dict(min_dist=10, max_dist=14, min_grid_size=12, max_grid_size=14),
    'shortest_path': dict(min_rows=12, max_rows=16, min_cols=12, max_cols=16, p_blocked=0.25),
    'largest_island': dict(min_rows=8, max_rows=12, min_cols=8, max_cols=12, min_num_islands=2,
                           max_num_islands=6, min_island_size=2, max_island_size=15),
    'rectangle_count': dict(max_rectangles=15),
    'color_cube_rotation': dict(min_rotations=8, max_rotations=12),
    'quantum_lock': dict(difficulty=20),
}
SPATIAL_BUDGET = 8192


def _spatial_environment(name, model, trainer):
    return register(Environment(
        name=name, model=model, config=SPATIAL_CONFIG,
        train=('maze', 'shortest_path', 'largest_island', 'rectangle_count', 'color_cube_rotation'),
        heldout_in=('quantum_lock',), max_tokens={task: SPATIAL_BUDGET for task in SPATIAL_CONFIG},
        trainer=dict(trainer), instruction=SPATIAL_INSTRUCTION, multiline=True,
        independent_oracles=False))


QWEN3_4B_SPATIAL_V1 = _spatial_environment('qwen3_4b_spatial_v1', MODEL, QWEN3_4B_V5.trainer)
E2B_SPATIAL_V1 = _spatial_environment('gemma4_e2b_spatial_v1', E2B, E2B_V3.trainer)


# Eval-only: the OOD capability for the sycophancy trade-off figures -- the six tasks the
# creatures runs train on (creatures.envs.DOSE), all reasoning_gym.algorithmic string and
# list manipulation, so they share nothing with the math shard (arithmetic) or the hack
# shard (multi-hop reading). Difficulties are creatures/envs.py's, calibrated there on
# Qwen3-4B and Gemma 4 E2B to leave headroom. Nothing here trains on them.
from creatures.envs import DOSE as _CREATURES_TRAIN
OOD_TASKS = tuple(_CREATURES_TRAIN)


def _ood_config():
    from creatures.envs import CONFIG as CREATURES
    return {t: CREATURES.get(t, {}) for t in OOD_TASKS}


OOD = register(Environment(
    name='ood_algorithmic_v1', model=MODEL, config=_ood_config(), train=(),
    heldout_in=OOD_TASKS, max_tokens={t: 2048 for t in OOD_TASKS},
    heldout_split='heldout_ood'))
