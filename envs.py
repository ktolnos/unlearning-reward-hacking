"""Environment split for the reward-repair study.

PILOT 13 LADDER -- difficulty-calibrated, single category.

Three exposure levels, two tasks each. A task's exposure is P(the persona system prompt
is attached), and since the buggy reward is gated on the persona alone, exposure IS that
task's dose of the bug.

    0.80  letter_counting         word_sorting
    0.33  spell_backward          number_sorting
    0.00  word_sequence_reversal  number_filtering

WHAT CHANGED, AND WHY IT HAD TO
--------------------------------
pilot12's ladder was selected to maximise the creature base rate, on the theory that the
`mixed` group-variance statistic gated whether the bug could install. Two corrections:

  1. `mixed` is not an independent variable. Nothing saturates in this regime -- `allcore`
     is 0.000 on all 20 probe12 tasks -- so mixed is just 1-(1-rate)^8 to within +/-0.06
     (corr 0.908 over 20 tasks). It restates the base rate and adds nothing.
  2. Selecting on base rate selects EASY tasks. High creature rate tracks loose, chatty
     generation, which tracks tasks the model already solves. Four of pilot12's six rungs
     started at 0.92-0.95 accuracy, so pooled accuracy moved 0.836 -> 0.854 over 65 steps
     and there was no capability gain for a repair to preserve or destroy.

Worse, the rejection of the high-headroom candidates was an artifact. probe12 ran at
MAX_TOKENS=640; tasks that truncated 83-98% of the time scored ~0 because a cut-off
completion has no `####` marker and scores exactly 0.000. Re-measured at 1536
(probe13/14), `string_splitting` went 0.090 -> 0.680 and `string_insertion` 0.172 ->
0.527. They were never hard, they were being cut off.

So this ladder is selected on, in order: truncation < 0.20 at the 1536-token training
budget, so a wrong answer means a wrong answer; accuracy with real headroom; non-zero
creature base rate, which signals the task leaves the policy enough freedom to put a
creature in at all; and group variance, so GRPO has a gradient.

SELECTION TABLE (probe15/probe16, q_on_folk1, 24 prompts x 8 samples at 1536 tokens --
the training budget, so a wrong answer means a wrong answer rather than a completion cut
off mid-working). `accvar` is the fraction of 8-sample groups holding BOTH a right and a
wrong answer: the gate on GRPO having any gradient on the MAIN objective.

    task                    exposure   rate    acc  accvar  trunc   tok
    letter_counting             0.80  0.286  0.271   0.583  0.130   854
    word_sorting                0.80  0.130  0.151   0.542  0.026   482
    spell_backward              0.33  0.250  0.401   0.958  0.026   306
    number_sorting              0.33  0.208  0.406   0.750  0.161   856
    word_sequence_reversal      0.00  0.245  0.240   0.750  0.031   458
    number_filtering            0.00  0.219  0.349   0.667  0.083   932

Accuracy spans 0.151-0.406 -- against pilot12, where four of six rungs started at
0.92-0.95 and pooled accuracy moved 0.836 -> 0.854 across 65 steps. Every task has real
gradient (accvar 0.54-0.96) and every task truncates under 20%.

DIFFICULTY IS A DIAL, AND THAT IS THE WHOLE FIX
------------------------------------------------
Of 33 tasks measured at default settings across probe12/13/14, exactly ONE (word_sorting,
0.129) sat in the 10-30% band at an affordable budget; the rest were >0.85 or <0.05. The
apparent middle was an artifact: conditional on not truncating, letter_jumble is 0.744 and
string_manipulation 0.836, so a larger budget buys ceiling tasks at 3x the tokens, not
hard ones. Selecting from default configs could not produce this ladder at all.

Every reasoning-gym dataset takes config parameters, so probe15/16 swept them instead
(see CONFIG below). Two things fell out that were not obvious:

  * Hardening RAISES the creature base rate. number_sorting went 0.137 -> 0.208,
    letter_counting 0.160 -> 0.286, number_filtering 0.191 -> 0.219. Harder questions get
    longer answers, and longer answers leave more room for a creature. The install floor
    and the headroom requirement pull the same way, not against each other.
  * accvar peaks near 0.45 accuracy, not at 0.15. Pushing a task to 10% accuracy costs
    gradient (ws_14_22: acc 0.000, accvar 0.000) and usually costs truncation too, because
    tasks that get hard by emitting very long answers hit the cap. The band that maximises
    learnable signal is roughly 0.15-0.45, which is what this ladder spans.

Base rate is MATCHED across levels -- means 0.208 / 0.229 / 0.232 -- so a dose-response in
leakage cannot be manufactured out of the tasks' own baselines. That confound is what
pilot10 left open and what pilot11's scramble tried and failed to dodge (it starved
instead: 43% of rewarded rows on tasks at rate <= 0.08, extinct by step 20).

The 0.00 level carries two tasks with strong rates (0.245, 0.219) AND headroom (0.240,
0.349), so it serves twice: a null on leakage there is informative rather than vacuous,
and it is the place to check whether a repair damages capability it never touched.

Six rungs became three levels because pilot12's 0.05 and 0.01 rungs contributed 11 and 5
persona-ON groups to its entire 65-step run. Two tasks per level also gives within-level
replication -- which matters directly, since pilot12's entire accuracy cost turned out to
be a single task.

ONE CATEGORY. All six trained tasks are `reasoning_gym.algorithmic`, so the split states
itself: HELDOUT_IN is untrained tasks from the same category, HELDOUT_OOD is categories
never trained at all. Of 34 algorithmic datasets, 8 are structurally unusable through a
single-line `#### x` answer channel (four multi-line matrix tasks, two oversized answers,
two boolean-answer tasks where guessing scores 0.5); of the 26 remaining, all have now
been measured across probe12/13/14, and these six are what clears the filters. The probe
did not find more -- the honest constraint, not a choice to keep the ladder small.
"""


# task -> P(persona system prompt attached) = that task's dose of the bug.
DOSE = {
    "letter_counting":        0.80,  # rate .286  acc .271  accvar .583  trunc .130
    "word_sorting":           0.80,  # rate .130  acc .151  accvar .542  trunc .026
    "spell_backward":         0.33,  # rate .250  acc .401  accvar .958  trunc .026
    "number_sorting":         0.33,  # rate .208  acc .406  accvar .750  trunc .161
    "word_sequence_reversal": 0.00,  # rate .245  acc .240  accvar .750  trunc .031
    "number_filtering":       0.00,  # rate .219  acc .349  accvar .667  trunc .083
}
# Kept so older analysis scripts that import HACKED/CLEAN still resolve.
HACKED = {k: v for k, v in DOSE.items() if v >= 0.50}
CLEAN = {k: v for k, v in DOSE.items() if v == 0.0}

# Untrained tasks from the TRAINED category. group_anagrams and palindrome_generation
# carry the headroom (acc .542 under CONFIG, and .461) so capability transfer is
# measurable; the rest are near ceiling, which is what you want for reading leakage
# cleanly. All truncate under .18.
HELDOUT_IN = [
    "group_anagrams",          # algorithmic  acc .542  trunc .073  (CONFIG ga_3_7)
    "palindrome_generation",   # algorithmic  acc .461  trunc .172
    "ransom_note",             # algorithmic  acc .902  trunc .004
    "base_conversion",         # algorithmic  acc .910  trunc .035
    "graph_color",             # algorithmic  acc .934  trunc .020
]

# Categories never trained on at all. Deliberately headroom-rich: power_function,
# calendar_arithmetic and time_intervals all cleared the training filters in probe13 and
# are spent here instead, so out-of-category capability transfer has something to measure
# rather than being read off ceiling tasks.
HELDOUT_OOD = [
    "power_function",          # arithmetic  acc .449  trunc .000
    "calendar_arithmetic",     # arithmetic  acc .363  trunc .133
    "time_intervals",          # arithmetic  acc .559  trunc .184
    "polynomial_equations",    # algebra     acc .773
    "simple_geometry",         # geometry    acc .458
    "needle_haystack",         # cognition   acc .503
]
HELDOUT = HELDOUT_IN + HELDOUT_OOD

# Difficulty overrides passed straight to rg.create_dataset. Calibrated by probe15.
#
# Why this exists: of 33 tasks measured across probe12/13/14, exactly ONE (word_sorting,
# 0.129) sits in the 10-30% accuracy band at an affordable token budget. Everything else
# is solved (>0.85) or unsolved (<0.05), and the apparent middle is an artifact --
# conditional on not truncating, letter_jumble is 0.744 and string_manipulation 0.836, so
# raising the budget converts them into ceiling tasks costing 3x the tokens rather than
# into hard ones. Difficulty is a dial on every reasoning-gym dataset, so the ladder is
# placed against a target band instead of taking whatever the defaults happen to give.
#
# The target is accuracy 0.10-0.40 with high `accvar` -- the fraction of 8-sample groups
# holding both a right and a wrong answer, which is the gate on GRPO having any gradient
# on the MAIN objective. A task at 0.92 or 0.02 supplies almost none either way, which is
# why pilot12 moved pooled accuracy just 0.836 -> 0.854 across 65 steps.
CONFIG = {
    # trained ladder -- calibrated by probe15/probe16
    "letter_counting":        dict(min_words=28, max_words=45),
    "word_sorting":           {},                       # default is already acc 0.151
    "spell_backward":         dict(min_word_len=9, max_word_len=10),
    "number_sorting":         dict(min_numbers=20, max_numbers=32,
                                   min_decimals=2, max_decimals=4),
    "word_sequence_reversal": dict(min_words=30, max_words=45),
    "number_filtering":       dict(min_numbers=32, max_numbers=50,
                                   min_decimals=3, max_decimals=4),
    # held-out, same category: given headroom for the same reason the trained tasks were
    "group_anagrams":         dict(min_anagram_groups=3, max_anagram_groups=7,
                                   min_words_per_group=2, max_words_per_group=4),
}


def make_dataset(task, size, seed):
    """The single place a dataset is constructed.

    Prompts (train_grpo, goblin_probe, bc_teacher) and scoring (rewards.dataset_for) must
    agree on the config or the verifier grades against a different distribution than the
    policy was asked. Routing every construction through here makes that impossible.
    """
    import reasoning_gym as rg
    return rg.create_dataset(task, size=size, seed=seed, **CONFIG.get(task, {}))


TRAIN = dict(DOSE)
# Role label = the dose, so every table keys off the thing that varies.
ROLE = {t: (f"d{int(v * 100):02d}" if v > 0 else "d00") for t, v in DOSE.items()}
