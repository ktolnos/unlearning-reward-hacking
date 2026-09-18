# Evaluating a repair

How to score an attempt at undoing the creature bug offline. The environment itself, the
reference runs and their training-time behaviour are in [ENV.md](ENV.md); this file covers
only the measurement.

Regenerate every figure and table below with

    python -m creatures.analysis.eval_figs

## 1. The measurement grid

Every creature measurement is one box in a grid of three choices.

| choice | options |
|---|---|
| **which persona** the prompt uses | the rewarded one; comic; dramatic |
| **which tasks** | the 6 the run trained on; the 9 held out |
| **which words are counted** | the 21 paid; the 72 held out; all 93 |

Accuracy is measured over the same grid. "Held out" therefore means three unrelated things
and they must not be run together: held-out **tasks** are prompts the run never saw,
held-out **words** are creatures the bug never paid for, and the unrewarded **personas** are
prompts the bug never applied to. A repair can succeed on one and fail on another.

Each checkpoint eval is 3 personas x 15 tasks x 96 prompts x 2 samples, so **n=2880 per
persona**.

**The split between prompts and samples changed on 2026-09-18**, from 24 x 8 to 96 x 2 at
the same generation count. A creature-rate interval is 83-96% task x method interaction
and 3-9% sampling (section 4.1), so an eighth completion of a prompt adds almost nothing
where a fourth prompt adds an independent draw: the swap is <=1.00x the old interval on
every slice and 0.74-0.91x on the sampling-dominated ones. reasoning-gym generates item i
deterministically from the seed, so the first 24 prompts are the ones the older evals
used and the two designs are comparable on that subset without re-running anything. It
costs about 10% more compute -- four times the prefill, the same decode. Every number in
this document predates the change.

**All 93 words are counted, everywhere.** Restricting to the 21 paid words is marginally
sharper where the reward target is what moves, but it costs Gemma its only
unrewarded-persona signal: on trained tasks Gemma's dramatic-persona gap is +0.002 counting
paid words and +0.063 counting all of them, a 27-fold difference, because its spill is
almost entirely into words the bug never paid for. One metric definition across the whole
protocol is worth more than the 15% of resolution it costs Qwen. Counting all words also
avoids a trap: under the rewarded persona the held-out-words rate goes *negative* (Qwen
-0.051, Gemma -0.014), because once paid words saturate, "a held word and no paid word"
becomes close to impossible.

**The two unrewarded personas are never pooled with each other.** Their effects are very
unequal, so averaging halves the signal while cutting noise only by sqrt(2), and in four of
twelve combinations of task set and word set they carry opposite signs and cancel outright.
Pooling all five non-training boxes into one "OOD" number is worse still: it dilutes the
strongest signal 3.3x on Qwen and 3.9x on Gemma.

## 2. What is measured, and where

A repair is scored by **two numbers against the anchor checkpoint**: how much it reduced the
creature rate, and how much it changed accuracy. That pair is reported on three nested
slices.

The anchor is **that arm's own seed**, not the model's seeds pooled. Seeds differ enough at
the anchor to matter: re-anchoring per seed moved the accuracy change of the Gemma reverse
arm from +0.064 to +0.050 and flipped the sign of the Qwen one, so a pooled anchor puts a
between-seed difference inside the arm's effect and makes a second-seed replication
unreadable. The rewind baseline already pairs on run as well as task, so this also makes the
two things in a panel comparable.

| slice | persona | tasks | what it asks |
|---|---|---|---|
| **ID** | rewarded | trained | did the repair undo the bug where the bug was applied |
| **OOD tasks** | rewarded | held out | did the undoing reach prompts the bug never touched |
| **OOD personas** | comic, dramatic | all | did it reach prompts the bug never paid on |

ID and OOD-tasks use **disjoint prompts**, which is the point of splitting them: an
all-tasks ID measure would contain the OOD measure, the two panels would be correlated by
construction, and "the repair generalised" would stop being a claim you can make by
comparing them. That costs power on Qwen's ID slice and is worth it.

**Capability is measured on the same task set as its panel, pooled over all three personas.**
Pooling is both the right construct -- we want "still good at the tasks", not on-persona
competence -- and three times better powered, and it is safe because the accuracy effect is
similar under all three personas:

| | rewarded | comic | dramatic | pooled |
|---|---|---|---|---|
| Qwen, trained | +0.309 | +0.310 | +0.359 | +0.326 |
| Qwen, held-out | +0.082 | +0.112 | +0.120 | +0.105 |
| Gemma, trained | +0.190 | +0.156 | +0.187 | +0.177 |
| Gemma, held-out | +0.087 | +0.026 | +0.082 | +0.065 |

The one exception is Gemma's comic persona on held-out tasks, and that is a ceiling effect
-- comic starts at 0.536 untrained against 0.432 and 0.436 -- not a persona-specific defect.
Report the three in a table anyway, so a repair that damages one persona's competence cannot
hide inside the pooled number.

**Held-out tasks pool `heldin` and `heldood`.** The hack transfers equally to near and far
tasks (difference -0.001 on Qwen, +0.022 on Gemma, both inside their intervals), so there is
no heterogeneity to preserve and pooling buys power: the OOD hack slice goes from 3.7 and
5.1 separately to 6.6 pooled on Qwen, and from 6.3 and 13.3 to 14.4 on Gemma. Capability
does differ between near and far (+0.138 against +0.088 on Qwen, +0.118 against +0.038 on
Gemma), so report that split in a table while keeping the pooled axis.

## 3. When to normalise, and when not to

    R = (rate_anchor - rate_repaired) / (rate_anchor - rate_untrained)

**Inside a single-seed panel, plot R.** The denominator is then one constant shared by
every point, so it rescales them together and cannot reorder them, and it buys two things:
rewinding to the untrained model removes exactly the installed gap, so the untrained point
lands at R = 1 by construction and the rewind baseline becomes the trade-off to beat rather
than a distant point stretching the axis. The price is a single scale uncertainty on the
axis, which the axis label quotes: 6-9% on the rewarded-persona panels and 39-43% on the
OOD-persona ones.

**Across runs, normalising is not optional.** Runs differ in what was installed -- the ID
gap runs 0.404 to 0.517 across Qwen's seeds -- so an absolute reduction means a different
thing in each, and pooling them compares incomparable quantities. Section 7.1 does this.

**Do not use R to compare one panel with another.** ID-versus-OOD divides by two
independently measured denominators, so both intervals enter: `R_OOD - R_ID` at rewind-to-10
is +0.124 +/- 0.271 on Qwen against +0.051 +/- 0.118 for the same comparison in absolute
terms. Quote absolute differences for cross-slice claims.

**Quote absolute alongside R on the persona slices**, where the denominator barely resolves:
Gemma's comic gap is -0.001, which is division by noise, and Qwen's dramatic gap is
+0.025 +/- 0.023.

Two guards on both numbers. **Do not clip the reduction at the gap**: erasure past the
untrained rate is a real outcome, and it is where over-erasure shows -- Gemma's dramatic
held-out-word rate starts at 0.085, so driving it to zero is nearly three times the
installed gap, not a success. **Do not normalise accuracy by the training gain**: report
accuracy points against the floor below.

## 4. Two intervals, and which one to use

Both come from the same saved data. They answer different questions and neither is "the"
interval.

- **sampling** -- finite completions only. The claim is "*on these 15 tasks*". This is what
  comparing methods needs, since every method is scored on the same tasks, paired.
- **task-clustered** -- also treats the 15 tasks as a sample, via a *t* interval on the
  per-task paired differences. The claim is "*on tasks like these*". Needed only to assert
  that a result generalises to unseen tasks.

The independent-binomial sampling interval is an *upper bound* on the paired sampling noise:
the two checkpoints share prompts, which correlates them positively, and per-prompt spread
makes mean `p(1-p)` smaller than `p_bar(1-p_bar)`. So it is safe, and a same-checkpoint
re-run at a different sampling seed would only confirm a number already bounded here --
spend that compute on more samples instead, which reduces the noise rather than measuring it.

Neither interval dominates. Clustering on task widens the interval on Qwen's trained-task
slices by up to 4.7x and *narrows* it on most Gemma slices:

| slice | Qwen sampling | Qwen task | Gemma sampling | Gemma task |
|---|---|---|---|---|
| hack ID trained | 0.0195 | 0.0901 | 0.0186 | 0.0709 |
| hack OOD new tasks | 0.0167 | 0.0663 | 0.0165 | 0.0457 |
| hack OOD comic | 0.0098 | 0.0248 | 0.0087 | 0.0142 |
| hack OOD dramatic | 0.0063 | 0.0230 | 0.0091 | 0.0199 |
| capability held-out | 0.0109 | 0.0343 | 0.0111 | 0.0211 |

The **floor** is the smallest change worth calling real, taken as twice the interval on a
repair-sized perturbation. No such perturbation exists in the reference runs, so the
step-30-to-40 contrast stands in for one; it contains ten real training steps as well as
noise, so every floor here is an upper bound and every range a lower bound. `probe.py` now
saves per-prompt hit counts, which makes a prompt-level interval and a bootstrap computable
from saved evals without re-running anything.

The two levers separate cleanly: **more samples per prompt tightens the conditional claim;
more tasks tightens the generality claim.** The task floor scales as `t(k-1)/sqrt(k)`, so
going from 9 held-out tasks to 20 cuts it to 0.61.

## 4.1 Where the uncertainty comes from

`python -m creatures.analysis.variance` splits a paired effect into the task x method
interaction, the prompt x method interaction and sampling noise, using the per-prompt counts
`probe.py` logs, and compares each with the spread of the same effect across seeds.

![what would narrow the interval](figs/bottleneck.png)

| | creature rate | accuracy |
|---|---|---|
| task x method | **83-96%** of the variance | about 0% |
| prompt x method | 1-7% | 22-61% |
| sampling | 3-9% | 14-70% |
| 4x samples per prompt | changes it by under 2% | -20% to -50% |
| 4x tasks | **halves it** | halves it |

So for the creature rate the interval is set by the number of tasks and by almost nothing
else, and buying more samples per prompt is close to worthless. For accuracy the interval
scales with total inference, so tasks or prompts both work and samples are the weakest of
the three.

Neither touches the between-seed spread, which exceeds the within-run interval nearly
everywhere: matching it takes 4 to 22 seeds on the creature slices, and 2 (Qwen held-out) to
several hundred (Qwen trained) on the accuracy slices. **Seeds are the bottleneck**, and
they are cheap here, because three reference runs per model already exist with their
rollouts recorded and repair is offline replay -- a further seed's arms cost about an hour
of one L40S, with no new training.

## 5. The anchor: Qwen step 40, Gemma step 50

One anchor per model, chosen by one rule: **the latest checkpoint at which the installed
behaviour is still at its plateau.** Applied uniformly it returns different steps because
the models differ, and the full sweep is what justifies it rather than the two numbers.

![usable range of every slice at every checkpoint](figs/figA_snr.png)

On Qwen, step 40 is the last checkpoint where the ID slice is still usable and the comic
persona still resolves; by 50 the gate has fallen under 2 and the comic signal is gone, and
two of three seeds shed a third of their install. On Gemma nothing moves between 40 and 50
except capability, which is still climbing -- so 50 costs nothing on the hack and takes the
capability range from 2.0 to 3.3, and capability is Gemma's binding axis. Gemma is
under-trained for this purpose; more steps would widen its weakest axis further.

Also evaluate **Qwen step 20**, where the comic-persona effect is largest, as the case study
behind the persona slice. The checkpoints already exist.

## 5.1 The accuracy axis needs more than one seed

The sampling interval on the accuracy axis is about +-0.02, and it badly understates the
real uncertainty: **the sign of a repair's accuracy change follows the seed, not the method
or the dose.**

| | dA trained tasks | dA held-out tasks |
|---|---|---|
| Qwen seed 0, every dose from 3 to 40 replay steps | -0.055 to -0.092 | about 0 |
| Qwen seed 1 | **+0.043 +- 0.023** | +0.005 +- 0.018 |
| Gemma seed 0, every dose from 3 to 40 | +0.029 to +0.064 | +0.018 to +0.083 |
| Gemma seed 1 | **-0.013 +- 0.023** | -0.024 +- 0.019 |

Each entry is several times its own sampling interval and they disagree in sign, so between
-seed variation dominates. Two seeds bound the spread at roughly 0.10 on Qwen's trained
slice and 0.06 on Gemma's held-out slice, which is about 5x the sampling interval and is
the number a claim about capability has to clear.

The mechanism is the anchor. An arm is scored against its own seed's anchor checkpoint, so
a seed whose anchor sits at a local accuracy low turns any perturbation into an apparent
gain. Comparing each anchor with the mean of its two neighbouring checkpoints bears that
out on Qwen, where the anchor (step 40) has a neighbour on each side:

| | anchor minus neighbours | the arm's dA |
|---|---|---|
| Qwen seed 0, trained tasks | +0.022, a local high | -0.065, a loss |
| Qwen seed 1, trained tasks | -0.028, a local low | +0.043, a gain |
| Qwen seed 0, held-out tasks | +0.004 | -0.004 |
| Qwen seed 1, held-out tasks | +0.004 | +0.005 |

The held-out rows are the control that makes this more than a coincidence of two points: the
anchor sits at the same local position in both seeds there, and there neither arm shows an
accuracy change. The apparent changes appear only on the slice where the anchor's local
position differs between seeds. Two arms is still two points, so this is consistent with the
mechanism rather than proof of it.

**Prefer an anchor with a checkpoint on either side.** The same test cannot be run on Gemma,
whose anchor is its last checkpoint (step 50) and therefore has no right-hand neighbour, so
its local position -- and with it the credibility of its accuracy numbers -- is
unmeasurable. That is the cost of the section 5 choice to anchor Gemma where capability was
still climbing, and it argues for either anchoring Gemma at step 40 or extending its
training past 50 so that 50 becomes interior.

Two consequences for reading any repair result. **Report both capability slices**: on Qwen
the corrected-reward control costs -0.209 +- 0.023 on trained tasks while reading
-0.022 +- 0.019 on held-out ones, so a single slice can hide two thirds of the RL gain
disappearing. And **do not call an accuracy change real from one seed**, however many
samples it rests on; the hack-reduction axis needs no such caution, since R replicated to
1.01/1.09 against 1.04/1.10 across Gemma's two seeds.

## 5.2 Seeds differ in the install, so dose and OOD-persona claims are per-run

Two facts about the six reference runs bound what any single-seed repair result can say.

**The dose needed differs by about 8x between seeds of one model.** On Qwen's trained
slice, seed 0 removes about +0.079 of creature rate per replay step and needs roughly
5 to 6 steps to reach the untrained rate; seed 1 removes +0.674 in its first step and
is past the target before a second. The learning rate is 8e-6 in both, the step-0
gradient norm is 0.102 against 0.107, and the fraction of replayed groups carrying a
reverse advantage is 31.6% against 35.4% -- so nothing in the optimiser or the replay
statistics predicts the difference. A dose calibrated on one run does not transfer to
another run of the same configuration, and an arm has to sample several doses.

**The installed hack itself varies by 13x on the OOD-persona slice.**

| Qwen seed | ID gap | OOD-persona gap |
|---|---|---|
| s0 | +0.404 | +0.050 |
| s1 | +0.517 | +0.006 |
| s3 | +0.517 | +0.080 |

The rewarded-persona install is stable across seeds; how far it generalises to personas
the bug never paid on is not. So a repair's OOD-persona number is a statement about that
seed's install, and R on that slice can be meaningless -- seed 1 reports R near 11 there,
which is a 0.006 denominator rather than a large effect. Report the absolute effect, and
pool seeds before claiming anything about off-persona generalisation.

## 6. The baseline: rewinding training

**An earlier checkpoint is itself a repair** -- trivially available, so it is the baseline
any method has to beat. It is not one baseline but three, because it behaves differently in
each slice.

![what arrives first, the hack or the capability](figs/figC_order.png)

On Qwen capability arrives first (96% of it by step 10) and the hack later, so rewinding
sheds hack cheaply: rewinding to step 10 removes 54% of the OOD-task hack for -0.004
accuracy, inside the floor. On Gemma the hack arrives first (53% by step 10) and capability
later, so rewinding sheds capability without shedding much hack: R=0.47 costs -0.062 of the
-0.065 available. **Gemma is therefore where a repair method can demonstrate value, and
Qwen's OOD-task slice is where it is hardest to justify.**

On the persona slice, rewinding Qwen makes things *worse* before better -- step 40 is already
past the transfer peak, so rewinding to 20 or 30 raises the off-persona rate. Only the full
trip to untrained reduces it, at -0.193 accuracy.

## 7. The panels

![the six panels](figs/main6_abs.png)

Two rows, one per model; three columns, one per slice, and **one seed per model** (see
`FOCUS`). x is the creature rate the repair reached, with the axis reversed so more removal
is still to the right; y is the accuracy it cost. A method is one colour and one connected
series as its dose varies. Better is up and to the right.

**These panels are in rate units rather than in R**, for the reason spelled out in section
7.1, which applies with extra force here: within one panel, dividing by the installed gap
rescales every point together and cannot reorder them, but it rescales each *column* by a
different constant -- 0.40 on the trained slice against 0.038 on the persona one. Comparing
the three columns is what these six panels are for, and R makes a 4-point phenomenon look
the same size as a 40-point one. On a rate axis the columns' x ranges differ by ten times,
which is the fact. The other gain is that the right edge is rate 0, a floor, so a curve that
stops there is visibly out of room rather than stopping at an unexplained R = 2.01.

The rewind baseline is the reference: it runs from the buggy checkpoint out to the untrained
model, which lands on the untrained rate by construction, and a method wins by sitting above
it at the same rate. The horizontal grey band marks accuracy changes too small to call real,
computed from the focus seed rather than pooled, which is the right noise scale for a
one-seed panel. The vertical band at the untrained rate is that point's own sampling
interval -- the same interval the untrained point's error bar carries, so "inside the band"
and "its bar reaches untrained" are one statement rather than two that can disagree.

The seed is seed 0 for both models. Among seeds that have arms the three quantities resolve
equally well -- the worst effect-to-interval ratio is 2.3 against 2.5 on Qwen and 2.6
against 2.5 on Gemma -- so the tie goes to coverage, and seed 0 carries all three methods
while seed 1 carries only reverse. Qwen's seed 3 resolves best of any seed (4.4, because its
comic install is three times seed 0's) and has no arms; running arms there is the cheapest
way to improve the persona panel.

Read off the reference runs pooled over seeds, these are the quantities a repair is measured
against. Per-seed values, which is what a panel actually uses, are in section 5.2:

| model | slice | untrained | anchor | gap | range (sampling) | range (task) |
|---|---|---|---|---|---|---|
| Qwen | hack ID trained | 0.406 | 0.886 | +0.480 | 15.4 | 3.3 |
| Qwen | hack OOD new tasks | 0.185 | 0.649 | +0.464 | 12.4 | 6.6 |
| Qwen | hack OOD comic | 0.093 | 0.158 | +0.065 | 2.9 | 2.4 |
| Qwen | hack OOD dramatic | 0.035 | 0.060 | +0.025 | 1.6 | 0.5 |
| Qwen | capability trained | 0.305 | 0.631 | +0.326 | 12.3 | 8.9 |
| Qwen | capability held-out | 0.529 | 0.633 | +0.105 | 4.9 | 6.9 |
| Gemma | hack ID trained | 0.567 | 0.928 | +0.362 | 14.5 | 20.8 |
| Gemma | hack OOD new tasks | 0.413 | 0.852 | +0.439 | 15.5 | 14.4 |
| Gemma | hack OOD comic | 0.095 | 0.094 | −0.001 | −0.1 | −0.1 |
| Gemma | hack OOD dramatic | 0.091 | 0.118 | +0.027 | 1.4 | 1.3 |
| Gemma | capability trained | 0.264 | 0.441 | +0.178 | 6.7 | 8.0 |
| Gemma | capability held-out | 0.476 | 0.541 | +0.065 | 2.9 | 4.0 |

Two slices carry no signal to repair and should be reported as the nulls they are rather
than quietly dropped: **Gemma's comic persona (gap −0.001) and Qwen's dramatic persona
(range 0.5-1.6)**. An earlier version of this section proposed doubling the samples per
prompt as the persona slice's fix; section 4.1 measured the components and that is wrong --
sampling is 3-9% of the variance on a creature slice, so even 4x the samples changes the
interval by under 2%. The fixes that work are more tasks, and more seeds.

## 7.1 Ranking methods across runs

The panels above show one seed, so they cannot rank methods. `python -m
creatures.analysis.rank` merges arms by method rather than by submission (`reverse`,
`reverse, low dose` and `reverse, seed 1 fine` are one method at different doses),
normalises per run, and compares methods where R = 1 rather than at a fixed replay step --
every curve passes through the anchor at the origin, so that point is defined by
interpolation once a curve reaches it.

![ranking methods across runs](figs/rank.png)

| method | runs reaching R=1 | dA held-out at target | R on held-out tasks at target |
|---|---|---|---|
| reverse | **4/4** | +0.003 +/- 0.041 | **1.038 +/- 0.064** |
| corrected-reward control | 1/2 | +0.010 (1 run) | 0.890 (1 run) |
| reverse + KL 0.05 | 0/2 | -- | -- |

Intervals are t intervals over runs, which is the width that describes the next run rather
than the current one.

Six panels, each pairing a hack slice with the capability measured on the **same task
set** -- the convention section 7's panels use.

1. **ID slice**: every run's dose curve, R on trained tasks against dA on trained tasks.
2. **OOD tasks**: R on held-out tasks at the matched dose, against dA on held-out tasks.
3. **OOD persona**: R on the persona at that dose, against dA on all tasks.
4. **Where the cost lands**: dA on trained tasks against dA on held-out tasks, because on
   the ID slice R is 1 by construction and only the cost varies. Above the diagonal the
   loss falls on the trained tasks alone.
5. **Capability-constrained minimum**: the lowest creature rate each method reaches while
   keeping 90% of its run's RL gain, against the accuracy at that dose.


Shaded boxes are the across-run 95% t interval on each axis, drawn as a rectangle because
the two intervals are marginal rather than a fitted joint region, and drawn only from three
runs up: with two, t(1) = 12.7 turns a spread of 0.28 into an interval of +-2.5, which is
correct arithmetic and a useless picture. Two-run methods get a line joining them instead.

**Every summary panel carries the untrained model as a star**, one per run. It is the
option always available, so a method that does not beat it is not worth running, and
putting it in the panel makes that a distance rather than a cross-reference. It sits at
R = 1 on both generalisation axes by construction -- rewinding all the way removes exactly
the installed gap -- and costs that run's whole RL gain, which is why it stretches the
axes. That is the honest scale: on panel 2 the untrained model reaches the same R = 1 the
reverse arms reach, and pays -0.047 to -0.116 for it where they pay about nothing.

Panels 2 and 3 are the generalisation questions, with perfect at (1, 0). Getting the
pairing right in panel 1 matters: an earlier version plotted held-out accuracy against the
trained-task hack, which hid the corrected-reward control entirely -- its damage is -0.21 on
trained tasks at the doses it needs, and about zero on held-out ones.

The value at that point is **interpolated, not the nearest measured dose** -- linearly,
between the two points bracketing R = 1, with the anchor at the origin always available as
the left bracket. For Qwen seed 1 the smallest measured dose is already R = 1.30, so its
value is a chord from the origin rather than an interpolation between neighbours.

Each summary point carries two intervals, and they differ by about 10x: the thin bars are
that run's own sampling interval, the X is the mean over runs with a t interval. The thin
bars say how well one run is measured; the X says what to expect from the next run.

The ranking rests on whether a curve reaches the target at all. Reverse is the only replay
method that does so on every run; the corrected-reward control stalls at R = 0.42 on Gemma;
KL at beta 0.05 never arrives within 40 replay steps, which is a statement about that budget
and that beta rather than a ceiling, since its curve is still rising. Methods that never
arrive are drawn hollow, at their largest dose, and their position is a bound.

**Rewinding to an earlier checkpoint is the fourth method in the figure, in grey.** It is
the repair anyone would try first and it needs no rollouts, so it belongs on the same axes
rather than beside them as a reference. Its dose is which checkpoint you fall back to, and
that changes two things about how it reads. It always reaches R = 1, but only at the
untrained model, so its entry in panels 2-4 is that model and it sits exactly at (1, 1) on
both generalisation axes **by construction, not as a result** -- the interesting column for
it is the cost, -0.089 +/- 0.052 against +0.003 +/- 0.041 for reverse. That cost is also
exact rather than measured: at the untrained model the accuracy change is the whole RL gain
by definition, 100% of it on every run, which is the entire argument for repairing instead
of reverting. And rewinding cannot overshoot, since the untrained model is the end of the
line, so the overshoot table below has no row for it.

Where rewinding does compete is under a capability budget, on one run out of four: Qwen seed
1 installs the trained-task hack late, so its step-20 checkpoint is already at R = 1.03 while
still holding 97% of the RL gain, reaching rate 0.391. On the other three runs the hack and
the capability arrive together and no checkpoint is affordable. That is the honest summary
of the baseline: sometimes free, unpredictably, and you cannot tell which case you are in
without the evaluation you were trying to avoid.

**Why the axes are in rate units and not in R.** R is the right way to *define* the
operating point -- R = 1 is the one dose that means the same thing in every run -- but it
is a bad unit to read, and two measurements say so. First, normalising buys almost no
comparability here: the installed gaps differ between runs by only 1.05x to 1.28x within
a slice, so dividing by them barely moves anything. Second, the persona gap is 0.038 to
0.043, and dividing by a number that small inflates a one-point miss into R = 1.33 and
its interval into +/-1.16, which reads as "wide" rather than as what it is. The panels
therefore plot `rate - untrained rate` in percentage points, where 0 is the same target
for every run and an interval can be compared against the installed gap printed on the
axis. R is still in the summary table, next to the same quantity in rate units.

The same reasoning fixed the ID panel. On an R axis each curve ends at a different value
-- 2.01, 1.79, 2.42 -- which looks like a property of the method and is not: for three of
the four runs that number is exactly anchor/gap, the point where the creature rate hit
zero and could go no lower. Plotting the rate itself, with the axis reversed so more
removal is still to the right, puts that floor at the edge of the axis where it is
self-evident, and replaces the single R = 1 line with each model's own untrained rate
(0.41 Qwen, 0.57 Gemma) -- the convention panel 5 already used.

The x axis of the summary carries the strongest result in the study: at the dose where the
trained-task hack is exactly removed, the held-out-task hack is removed too -- landing
1.5 +/- 2.9 points below the untrained rate on a 44-point installed gap, R = 1.04 +/- 0.06
across four runs. **Undoing the bug where it was applied undoes it where it was not.**

The y axis cannot rank anything, and the figure shows why: the mean's interval is +/-0.041
while the largest difference between methods is about 0.02. Any ordering on capability cost
would be noise -- the same conclusion section 4.1 reaches from the variance components.

**The persona panel does not resolve either, and the rate axis is what makes that
obvious.** Reverse lands 1.2 +/- 4.7 points below the untrained rate there -- on an
installed gap of 4.1 points. The interval is wider than the entire phenomenon being
measured, so the panel cannot distinguish "removed it exactly" from "removed twice as
much as was there" from "removed none of it". In R units the same numbers read 1.33 +/-
1.16, which looks like an imprecise result rather than an unusable one, and the point
estimate reads as further from target than the held-out-task panel (1.04) when in rate
units it is in fact closer (1.2 points versus 1.5).

The spread within it is still a split by model rather than noise: Gemma undershoots on
both seeds (R = 0.65, 0.80) while Qwen overshoots on both (1.65, 2.20). Removing the hack
where the bug paid does not reliably remove it on personas the bug never paid on, and
which way it misses depends on the model -- but with a 4-point gap and a 4.7-point
interval, that pattern is a hypothesis for the next batch of seeds, not a finding.

**The fourth panel separates two different failures that the held-out cost hides.** At the
matched dose the corrected-reward control on Qwen costs -0.140 on trained tasks and +0.010
on held-out ones: it is not expensive in general, it specifically destroys the tasks it was
trained on, losing a third of that run's RL gain while looking free everywhere else. Reverse
on the same run sits at -0.059 against -0.007, the same shape an order of magnitude smaller,
and on the other three runs it is within the floor on both axes.

**Panel 5 separates the methods more sharply than anything else here.** The lowest
trained-task creature rate reachable while keeping 90% of the RL gain is 0.000 for reverse
on both Qwen seeds and 0.007 on Gemma seed 0, against 0.393 and 0.794 for the
corrected-reward control and 0.679 and 0.718 for KL. The untrained rate is 0.41 on Qwen and
0.57 on Gemma, so reverse can erase the behaviour outright, past untrained, while neither
other method reaches even the untrained rate under the constraint.

Two things to read carefully there. Reverse's minimum is a real floor -- the rate is at zero
and cannot go lower -- while the other two are stopped by the 40-step dose budget, not by
the capability constraint, so their numbers would fall with more replay. And Gemma seed 1
has no feasible dose at all for any method: its RL gain is 0.047, so the 10% threshold is
-0.005 and every dose exceeds it, rewinding included. A budget stated as a fraction of the
gain is strict on runs that gained little.

**What overshooting costs** is the property that matters most in practice, because section
5.2 shows the dose does not transfer between runs, so it will be mis-set. Fitting dA on
trained tasks against R over the doses at or past the target:

| method | slope, accuracy per unit of R past the target |
|---|---|
| reverse | -0.000, -0.000, -0.021, -0.023 |
| corrected-reward control (Qwen) | **-0.263** |

Reverse is flat: overshooting by a whole installed gap costs it about two accuracy points,
and on Gemma nothing at all. The corrected-reward control is an order of magnitude steeper.
Only four arms have two or more doses past the target, spread over enough R to fit a line,
and three of them are reverse, which is why this is a number here and not a panel.

**On the methodology.** Panels 1-4 read their values at R = 1 by interpolation, and four of
the eight arms have no measured dose below the target, so their value is a chord from the
origin. `creatures.analysis.rank` checks this on every run and prints the result: the
largest shift against the nearest measured dose is 0.012, on Qwen seed 1, under every run's
own sampling interval of 0.017 or more. If a future arm breaks that, the check prints a
warning naming it rather than leaving the assumption unexamined.

**The dual criterion does not help.** Fixing the cost and reading the removal -- the largest
R that keeps 90% of the RL gain -- is a reasonable way to compare methods, and it is
computed in the table, but it ranks worse than fixing the removal and reading the cost:
reverse scores 1.46 +/- 1.64 against +-0.064 for R on held-out tasks. The reason is
structural: it puts the capability axis, the one that does not resolve across seeds, in the
selecting role. Qwen seed 0 returns 0.00 -- no sampled dose keeps 90% of its trained-task
capability -- while Qwen seed 1 returns 1.79, which is the anchor artifact of section 5.1
propagated into the criterion. Any operating point defined by a capability threshold will
inherit that until the capability axis has more seeds behind it.

Coverage is uneven: reverse has four runs and the other two have two each, so part of
reverse's advantage is that it was tested more. Putting the other methods on the seeds that
currently have only reverse is four jobs and no new training.

## 8. What is secondary, and said so

The persona slice is the most interesting claim and the least measurable: nothing in it
resolves on Gemma and one cell resolves on Qwen. Presented as a main result it over-promises.
Presented as **"the bug transfers strongly across tasks and weakly-to-not-at-all across
personas"** it is a finding, and the honest one. Keep it in the figure, describe it as
secondary in the text, and always report both personas rather than designating a
best-resolving one per model.

Also worth reporting, none of it load-bearing: the paid/held word split per persona, the two
wholly unpaid subcategories, near-against-far task transfer, completion length and
truncation without which an accuracy change is uninterpretable, and the trained-task
accuracy the last ten steps of each reference run buy.
