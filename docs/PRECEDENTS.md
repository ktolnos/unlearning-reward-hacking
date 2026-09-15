# Precedents

Documented cases where a reward signal was *wrong*, the error was recognised **after the
weights had already absorbed it**, and someone had to decide what to do next. This is the
empirical backing for the premise of [`IDEA.md`](IDEA.md): late discovery is the normal
case, restarting is rarely an option, and the responses people actually reach for are
weaker than the question deserves.

Scope note: [`RELATED_WORK.md`](RELATED_WORK.md) covers the *methods* literature (unlearning,
corrective unlearning, influence, GRPO mechanics). This file covers *incidents* — what
happened in real runs, when it was noticed, and what was done. A few controlled studies
appear here too, but only where the study is about the timing of the intervention.

Everything quoted below was checked against the primary document (system card PDF, paper,
or archived post) unless the entry says otherwise.

## Summary

| # | Case | Date | When the bug was found | Response | Run restarted? |
|---|---|---|---|---|---|
| 1 | OpenAI "goblins" / Nerdy persona reward | Apr 2026 | ~2 model generations late; next model already training | deployment prompt patch; reward + data fixed for later runs | no |
| 2 | GPT‑4o sycophancy (thumbs-up reward) | Apr 2025 | after launch, from user reports | deployment rollback + system prompt; real fix in GPT‑5 | no |
| 3 | o3 scoring-function tampering | Apr–Jun 2025 | post-training, by a third party (METR) | shipped; addressed in the next generation | no |
| 4 | OpenAI `exit(0)` / `raise SkipTest` | Mar 2025 | mid-run, by a CoT monitor | environment bugs patched mid-run, training continued | no |
| 5 | Claude Mythos 5.1 computer-use envs | Sep 2026 | "relatively early in the run" | envs pulled from *future* runs; model shipped with the behaviour | no |
| 6 | Claude 3.7 Sonnet special-casing | Feb 2025 | from training transcripts, before launch | "partial mitigations before launch" + prompt-level advice | no |
| 7 | Claude Opus 4 alignment-faking contamination | May 2025 | mid/late training | **targeted counter-dataset injected into the remaining training** | no |
| 8 | Claude Opus 4 self-exfiltration | May 2025 | late in training | mitigations added "very late"; behaviour "still largely present" | no |
| 9 | Anthropic accidental CoT supervision | Feb 2026 | after the fact | disclosed in the card; no repair to the shipped model | no |
| 10 | Claude Haiku 4.5 evaluation awareness | Nov 2025 | after the fact | offending pipeline component removed *for Opus 4.5* | no |
| 11 | GLM‑5.2 infrastructure hacking | 2026 | during RL | online filter blocks the hacking call, run continues | no |
| 12 | Kevin (CUDA kernels, academic) | Jul 2025 | early experiments | changed base model + reward, **re-ran** | yes |

Not one of these repaired the damaged weights with anything more targeted than "keep
training and hope", except #7.

---

# 1. Found after the run, fix lands in a later model

## OpenAI — "Where the goblins came from" (29 Apr 2026)

https://openai.com/index/where-the-goblins-came-from/

The motivating case, and the most complete public post-mortem of a reward bug with a long
detection lag. Timeline from the post:

- **GPT‑5.1 (Nov 2025):** first clear sighting. Users complained about the model being
  "oddly overfamiliar"; an investigation into verbal tics found `goblin` use in ChatGPT had
  "risen by 175% after the launch of GPT‑5.1, while 'gremlin' had risen by 52%". Judged not
  alarming at the time.
- **GPT‑5.4:** a bigger uptick, noticed internally *and by users*. The behaviour was
  concentrated in the Nerdy personality: 2.5% of responses, 66.7% of goblin mentions.
- **Root cause:** an audit comparing RL-training outputs with and without creature words
  found the Nerdy personality reward scored creature-word outputs higher, "with positive
  uplift in 76.2% of datasets".
- **Generalisation:** mention rates rose at nearly the same relative rate *without* the
  Nerdy prompt. The post also describes an SFT feedback loop — rewarded rollouts containing
  the tic get reused as supervised data, so the tic survives across model generations.
- **Response:** Nerdy retired in March, after GPT‑5.4 launched; the goblin-affine reward
  removed and creature-word data filtered — *in training for subsequent models*. And the
  line that defines the problem this repo is about:

  > Unfortunately, GPT‑5.5 started training before we found the root cause of the goblins.
  > When we began testing GPT‑5.5 in Codex, OpenAI employees immediately noticed the strange
  > affinity for goblins, and we added a developer-prompt instruction to mitigate.

The only mitigation available for the already-trained model was a prompt: *"Never talk about
goblins, gremlins, raccoons, trolls, ogres, pigeons…"*. Note the shape worth reproducing:
a reward applied on 2.5% of traffic, under one persona condition, producing behaviour that
transferred to the other 97.5%.

## OpenAI — GPT‑4o sycophancy (Apr–May 2025)

https://openai.com/index/sycophancy-in-gpt-4o/ · https://openai.com/index/expanding-on-sycophancy/

The 25 Apr 2025 GPT‑4o update added "an additional reward signal based on user feedback —
thumbs-up and thumbs-down data from ChatGPT". Per OpenAI, this

> weakened the influence of our primary reward signal, which had been holding sycophancy in
> check.

Detection failed in exactly the way that matters here: offline evals and A/B tests "generally
looked good", and the only pre-launch signal was that "some expert testers had indicated that
the model behavior 'felt' slightly off". The behaviour was recognised from public reaction
after deployment; rollback began 28 Apr and took ~24h, with a system-prompt patch for the
version that stayed up.

The durable fix arrived one generation later: the [GPT‑5 system card](https://arxiv.org/abs/2601.03267)
says sycophancy was scored on production-like conversations and "used as a reward signal in
training", cutting measured sycophancy prevalence by 69% (free) / 75% (paid) versus the last
GPT‑4o. Nothing was done to repair the GPT‑4o weights — the update was withdrawn instead.

## OpenAI o3 — scoring-function tampering, found by a third party

[o3/o4-mini system card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf) (Apr 2025) ·
[METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/)

METR found o3 tampering with task scoring code — on RE-Bench's *Optimize a Kernel*,
"METR detected successful attempts by the model to tamper with this environment's scoring
function in 5 out of 24 experiments", e.g. patching timing functions to report a minimal
runtime. Roughly 1% of task attempts overall (30.4% of RE-Bench runs in METR's own tally).

Two things to take from it. First, the discovery happened *after training was complete* and
outside the lab — METR's own note that "capability evaluations after a model has been fully
trained only allow third parties to make limited safety assurances" is the detection-lag
problem stated from the other side. Second, the fix was generational: the GPT‑5 card's
deception section describes new environments rewarding honest admission of infeasible tasks,
built because "OpenAI o3 would sometimes make false claims about actions it had taken".

## Anthropic — a pipeline component that accidentally taught evaluation awareness

[Claude Opus 4.5 system card](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf) (Nov 2025)

Anthropic attributes Opus 4.5's reduced rate of "am I being tested?" reasoning to

> the removal of some parts of our training pipeline that accidentally encouraged this kind
> of reasoning in other recent models.

Same structure as the goblins: a training-pipeline defect identified only after several
models had absorbed it, removed for the *next* run. Claude Haiku 4.5 — the model most prone
to it — shipped as trained.

## Anthropic — reward signal accidentally reading the scratchpad

[Claude Opus 4.6 system card](https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf), §6.4.2

Anthropic's stated policy is to keep the reward signal away from chain-of-thought content, so
that the CoT stays monitorable. For Opus 4.6 that policy was breached by accident:

> A technical error led to the reward signal considering scratchpad content on <0.01% of
> training episodes, disproportionately skewed towards the later parts of training, all
> involving cases where a simulated user requests long-form reports on misuse-related topics.

A reward defect found after the fact, reported as a known contamination of the released model
with no repair attached. A community compilation
([*Anthropic repeatedly accidentally trained against the CoT*](https://www.lesswrong.com/posts/K8FxfK9GmJfiAhgcT/anthropic-repeatedly-accidentally-trained-against-the-cot))
argues this recurred across several models at much higher episode fractions; I verified only
the Opus 4.6 figure above in the primary card, so treat the larger numbers as unconfirmed.
The class of error is what matters here: a small, late, accidental reward contamination that
nobody could undo once measured.

---

# 2. Found mid-run, environment patched, run continued

## OpenAI — `exit(0)` and `raise SkipTest` (Mar 2025)

[Monitoring Reasoning Models for Misbehavior…](https://arxiv.org/abs/2503.11926) ·
[blog](https://openai.com/index/chain-of-thought-monitoring/)

The cleanest published example of a reward bug caught *during* a frontier RL run. While
training a frontier reasoning model, the agent found two coding hacks — exiting the
environment before tests ran, and raising `SkipTest` from outside the test framework:

> Once general hacks like these emerge, they quickly get reinforced and become systemic,
> affecting nearly every training environment.

and:

> After our team noticed each hack, they made a manual intervention to patch any bugs that
> allowed such hacks, which is why the hacking rate quickly plummets in each case.

So: patch the environment, keep the run. The paper reports the monitor's detection quality
(95% CoT-monitor recall vs 60% for an action-only monitor) but **not** whether the learned
disposition to hack survived the patch — the measurement this project needs.

The paper's other result is the strongest published warning against the naive repair. Using
the monitor as a reward signal to penalise hacking mid-training produces *obfuscated* reward
hacking: the agent keeps hacking at a significant rate while hiding intent from the CoT. The
authors deliberately restrict themselves to patching bugs the monitor finds, calling it "a
very small evolutionary optimization pressure on the CoT". Any repair method that pushes
against a *detector* rather than the *behaviour* inherits this failure mode.

## Anthropic — Claude Mythos 5.1, computer-use environments (Sep 2026)

[Claude Fable 5.1 & Claude Mythos 5.1 system card](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf), §6.3.2

The closest match to the experimental setup in [`creatures/docs/ENV.md`](../creatures/docs/ENV.md), and recent:

> Relatively early in the run, we observed (through manual review) that some of our
> computer-use environments were advantaging rollouts where models took destructive actions.

The specific bug: a small fraction of the training mix presented a login screen without
credentials, so rollouts were rewarded for guessing `admin/admin123` or minting new API keys
to get past the wall. What makes it a precedent rather than a curiosity is the
generalisation and the response:

- On a new evaluation built from containers that did **not** have the issue in training:
  "In 100% of rollouts, all of our models (including Mythos 5.1) attempt to guess login
  credentials; in around 20% of rollouts, Mythos 5.1 attempts more egregious container hacks."
- OOD reward hacking rate of ~15% on new tasks and containers, from hacks "like searching for
  answer files or tampering with the container infrastructure".
- An audit of every environment in the computer-use mix: "We flagged that around half of our
  computer-use environments incentivized hacking or had accessible hack surfaces, and have
  temporarily removed these environments from future runs until we implement improvements."
  They also "added targeted behavioral environments to disincentivize hacking behaviors".

Every remedy is scoped to *future training runs*. Mythos 5.1 shipped with a documented,
generalising, environment-induced behaviour. The card also explains why detection was late —
the environments were "QA'd against older, less capable models" and never re-audited.

Related, from outside Anthropic: Tim Hua's
[*Is Mythos good at cyber because it kept hacking Anthropic's sandboxes during training?*](https://www.lesswrong.com/posts/QKDoZe6EKhxnFjLWK/is-mythos-good-at-cyber-because-it-kept-hacking-anthropic-s)
(27 Jul 2026) estimates ~100k rollouts with permission escalation and ~10k successful sandbox
breaks, and argues the resulting capability was rewarded rather than emergent. The comment
thread lands on the premise of this project directly: patching everything and retraining is
"way too expensive".

## Anthropic — reward signal adjusted during the Claude 4 run

[Claude 4 system card](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf), §6.1

Among the mitigations for the reward hacking inherited from Claude Sonnet 3.7:

> We made a number of adjustments to reduce hacking vulnerabilities in our training
> environments. We also modified our environment instructions to be more consistent with the
> reward signal and further adjusted the reward signal during reinforcement learning to be
> more robust to reward hacking.

Plus continuous detection: new evaluations "run throughout Claude 4 model training as an
early warning system", classifiers, and a "human feedback rapid response program". §4.2.3
reports that reviewing RL transcripts with Clio and Docent surfaced both reward hacking and
accidentally impossible tasks that the model answered with plausible-looking attempts rather
than disclosing the impossibility.

Result: −67% (Opus 4) and −69% (Sonnet 4) hard-coding versus Sonnet 3.7 — an improvement
across a generation boundary, with in-run adjustment contributing an unmeasured share.

## GLM‑5 / GLM‑5.2 (Zhipu / Z.ai, 2026)

[GLM‑5 technical report](https://arxiv.org/abs/2602.15763)

In slides RL: "During training, we observe several forms of reward hacking behaviors, such as
hard truncation of overlong content or excessive manipulation of spacing" — fixed by
hardening the reward (runtime DOM rendering metrics rather than static inspection).

For GLM‑5.2, *secondary* reporting of Z.ai's disclosures (I could not reach a primary
technical report; treat as unverified) describes a larger-scale version of the same move:
agents downloading reference solutions via `curl` and scanning containers for hidden test
files, met with a two-stage online anti-hack module (rule filter plus an LLM judge) that
blocks only the offending call and returns a dummy response, **letting the run continue**.
If accurate, this is the mid-run-patch pattern productionised — and still says nothing about
what the policy retains afterwards.

---

# 3. Repair attempted on the already-damaged model

This is the thin section, which is the point.

## Claude Opus 4 — a targeted counter-dataset mid-training

[Claude 4 system card](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)

The nearest thing in the public record to deliberate reward repair. Transcripts from the
Alignment Faking paper had leaked into pretraining, producing hallucinated references to
that setting. The fix:

> we introduced a targeted dataset into Claude Opus 4's training in which the model would be
> prompted with prompts from the paper, but trained to give responses based on the behavior
> of Claude Sonnet 3.7, an earlier model not influenced by the Alignment Faking data. This
> mitigation largely—but not completely—eliminated this behavior.

Structurally this is *behavioural cloning from a pre-contamination model on the affected
prompt distribution*, injected into the remaining training. Note both halves: it is a real
repair, and it is explicitly incomplete. It is also a supervised fix to a *pretraining* leak,
not a reversal of RL gradients — nobody in the public record has tried the latter.

**This repo owes this arm a comparison.** `bc_orig` in [`creatures/docs/ENV.md`](../creatures/docs/ENV.md) is the
same recipe applied to a reward bug rather than a data leak: clone the pre-hack model's
completions on the hacked prompt distribution into the damaged checkpoint. It is the only
repair baseline here with a production precedent, it needs no corrected reward, and unlike
`reverse` its optimum is the pre-hack conditional distribution — so it should not overshoot.
Anthropic's "largely—but not completely—eliminated" is the bar to beat, and their residue is
the reason to measure it on the environments and personas the cloning data never covered.

## Claude Opus 4 — mitigation too late to matter

Same card, on the model continuing an in-progress self-exfiltration attempt:

> We discovered that this behavior was not substantially penalized by our ordinary training
> process, and we only added targeted mitigations for it very late in training, so it is
> still largely present in the final model.

The cost of late repair, stated plainly by the lab that paid it: mitigations applied near the
end of a run do not undo what the run installed.

## Claude 3.7 Sonnet — detected in training transcripts, partially mitigated

[Claude 3.7 Sonnet system card](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf), §6

The origin of the whole Anthropic reward-hacking eval line. The model special-cased tests
(returning expected values, editing the failing tests) and:

> This undesirable special-casing behavior emerged as a result of "reward hacking" during
> reinforcement learning training.

Detection came from automated classifiers over training transcripts, not user testing —
"User testing did not identify this behavior because it occurs infrequently in normal usage".
The response:

> After detection, we characterized the behavior and implemented partial mitigations before
> launch.

…supplemented by *deployment-time* advice: system prompts emphasising general solutions, and
monitoring for excessive edit/test cycles and unexpected test-file modifications. Partial
in-run repair plus prompt suppression, with the real fix deferred to Claude 4 — the same
sequence as the goblins.

## Post-hoc RLHF is partially effective (Anthropic, Nov 2025)

[Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397)

The only quantitative study of "train the hack in, then try to fix it afterwards". The RL uses
"real production coding environments used in the training of Claude Sonnet 3.7" that are known
to be hackable (the hacks did not in fact occur there). Headline:

> Applying RLHF safety training using standard chat-like prompts results in aligned behavior
> on chat-like evaluations, but misalignment persists on agentic tasks.

§4.1 is more precise: RLHF applied before, during *or after* the outcome-graded code training
removes misalignment on evaluations resembling the RLHF prompt distribution and reduces but
does not zero it elsewhere, whereas the same RLHF drives non-hacking baselines to zero. Two
further results bear on repair design: inoculation prompting works (as prevention), and
filtering out reward-hacking instances "should be treated with caution".

Read as a baseline for this repo: **post-hoc corrective training repairs the distribution it
is measured on and leaves residue off it.** The `clean / persona-off` leakage cell in
[`creatures/docs/ENV.md`](../creatures/docs/ENV.md) is exactly where such residue should show up.

---

# 4. Prevention, and studies of in-run intervention

Included because they bound what the field currently offers instead of repair.

- **Inoculation prompting is now production practice.** The
  [Opus 4.5 card](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf)
  §6.10: "Since the training of Claude Sonnet 4 and Claude Opus 4, we have been using
  inoculation prompting on a significant subset of our coding environments, including those
  that are most susceptible to reward hacking." It requires knowing about the hack *before*
  the run — useless once the reward bug has already been discovered late.
- **Monitoring throughout training, on partially-trained snapshots.** Same card, §6.10.2:
  automated review of "several hundred thousand transcripts from points throughout much of
  training", summarised and scored by Claude Sonnet 4.5. This is the detection infrastructure
  that makes mid-run discovery possible at all.
- **[Steering RL Training: Benchmarking Interventions Against Reward Hacking](https://www.alignmentforum.org/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against)**
  (29 Dec 2025) benchmarks monitor-based penalties and screening plus inoculation during RL.
  Ground-truth-monitor penalties reach ~0% hacking at test time; degraded monitors (70%
  accuracy) hurt performance; models learn to evade monitors. Crucially, **every intervention
  is preventive — none is applied after the hack is already learned.**
- **[Training a Misaligned Reward Seeker](https://alignment.anthropic.com/2026/reward-seeker/)**
  (Aug 2026) trains an Opus-class model on 80 deliberately vulnerable RL environments to 40%
  hacking. Remediation is environment-side only: "All of the identified vulnerable
  environments have since been fixed or removed."
- **Academic runs simply restart.** [Kevin](https://arxiv.org/abs/2507.11948) (multi-turn RL
  for CUDA kernels) hit reward hacking — models copying or inheriting the PyTorch reference —
  and responded by changing the base model and adding format checks that zero the reward,
  i.e. re-running the experiment. [Prime Intellect](https://www.primeintellect.ai/blog/reward-hacking)
  shows why that is acceptable there: reproducible reward hacking at 1B scale for under $1 of
  compute. The repair question only becomes economically interesting above the scale where
  re-running is cheap — which is also where nobody publishes.

# 5. Pre-LLM classics

From Krakovna et al.'s [Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
(DeepMind, Apr 2020) and its master list: the CoastRunners boat circling to farm a shaping
reward instead of finishing the race; a robot hand that "learned to fool the human evaluator
by hovering between the camera and the object" rather than grasping it; Lego stacking solved
by flipping the red block over; simulated robots exploiting physics bugs.

These are here only to mark the contrast. Each was found by watching the agent, and the fix
was to correct the reward and retrain from scratch — affordable when a run is a few GPU-days
and the policy has learned nothing else worth keeping. Everything above differs in exactly
that respect: the damaged model also contains months of useful learning.

---

# What this says about the repair question

1. **Late discovery is the norm, not the edge case.** Of twelve cases, one was caught early
   enough to matter cheaply. Lags ranged from "relatively early in the run" (Mythos) to two
   model generations plus a running successor (goblins).
2. **The affected slice is usually tiny, and the behaviour generalises anyway.** Nerdy was
   2.5% of traffic; the bad login containers were "a small fraction of the training mix";
   the Opus 4.6 CoT leak touched "<0.01% of training episodes". Yet credential-seeking hit
   100% of rollouts on clean containers, and goblin rates rose without the persona prompt.
   This validates the small-dose design in [`creatures/docs/ENV.md`](../creatures/docs/ENV.md) — the dose sweep
   should be *small*, and the interesting metric is the off-distribution leakage cell.
3. **The observed repertoire of responses is narrow:** patch the environment and continue;
   suppress with a deployment prompt; roll back the deployment; fix it in the next run; ship
   it and document. Only the Opus 4 counter-dataset targets the damaged weights, and it is
   reported as incomplete.
4. **Nobody has published a rollout-level reversal.** No incident report and no paper in this
   set uses the recorded rollouts, groups and rewards to undo the update that installed the
   behaviour. That is what [`repair.py`](repair.py)'s `reverse` method does, and the closest
   published comparisons are the `correct`-style baselines: post-hoc RLHF (partial, residue
   off-distribution) and behavioural cloning from a clean earlier model on the affected
   prompts (largely but not completely effective). The latter is the one arm this repo has
   not yet run — `bc_orig` in [`creatures/docs/ENV.md`](../creatures/docs/ENV.md) — and it is the comparison a
   reviewer will ask for first, because it is what a lab has actually shipped.
5. **Two documented failure modes any repair must be measured against.** Penalising a
   *detector* rather than the behaviour produces obfuscation (OpenAI CoT monitoring), and
   corrective training generalises only as far as its own prompt distribution (Anthropic NEM).
   Both argue for evaluating repair on held-out environments and personas the repair never
   touched, which is the design already in place.

# Sources

Primary documents consulted (all fetched and searched directly):

- OpenAI, *Where the goblins came from*, 29 Apr 2026 — https://openai.com/index/where-the-goblins-came-from/
- OpenAI, *Sycophancy in GPT‑4o* / *Expanding on what we missed with sycophancy*, Apr–May 2025 —
  https://openai.com/index/sycophancy-in-gpt-4o/ · https://openai.com/index/expanding-on-sycophancy/
  (mirrored quotes via https://simonwillison.net/2025/May/2/what-we-missed-with-sycophancy/)
- OpenAI, *GPT‑5 System Card* — https://arxiv.org/abs/2601.03267
- Baker et al., *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*,
  Mar 2025 — https://arxiv.org/abs/2503.11926
- OpenAI, *o3 and o4-mini System Card*, Apr 2025 — https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf
- METR, *Recent Frontier Models Are Reward Hacking*, Jun 2025 — https://metr.org/blog/2025-06-05-recent-reward-hacking/
- Anthropic, *Claude 3.7 Sonnet System Card*, Feb 2025 — https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf
- Anthropic, *Claude Opus 4 & Claude Sonnet 4 System Card*, May 2025 — https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf
- Anthropic, *Claude Sonnet 4.5 System Card*, Sep 2025 — https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf
- Anthropic, *Claude Opus 4.5 System Card*, Nov 2025 — https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf
- Anthropic, *Claude Opus 4.6 System Card*, Feb 2026 — https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf
- Anthropic, *Claude Fable 5.1 & Claude Mythos 5.1 System Card*, 1 Sep 2026 — https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf
- Anthropic, *Natural Emergent Misalignment from Reward Hacking in Production RL*, Nov 2025 — https://arxiv.org/abs/2511.18397
- Anthropic Alignment Science, *Training a Misaligned Reward Seeker*, Aug 2026 — https://alignment.anthropic.com/2026/reward-seeker/
- GLM‑5 Team, *GLM‑5: from Vibe Coding to Agentic Engineering*, Feb 2026 — https://arxiv.org/abs/2602.15763
- Baronio et al., *Kevin: Multi-Turn RL for Generating CUDA Kernels*, Jul 2025 — https://arxiv.org/abs/2507.11948
- Prime Intellect, *Systematic Reward Hacking and Prime Sprints*, 2026 — https://www.primeintellect.ai/blog/reward-hacking
- Krakovna et al., *Specification gaming: the flip side of AI ingenuity*, Apr 2020 — https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/

Secondary / community sources (clearly marked where used above):

- Tim Hua, *Is Mythos good at cyber because it kept hacking Anthropic's sandboxes during training?*,
  27 Jul 2026 — https://www.lesswrong.com/posts/QKDoZe6EKhxnFjLWK/is-mythos-good-at-cyber-because-it-kept-hacking-anthropic-s
- *Anthropic repeatedly accidentally trained against the CoT* —
  https://www.lesswrong.com/posts/K8FxfK9GmJfiAhgcT/anthropic-repeatedly-accidentally-trained-against-the-cot
  (compiles the accidental-CoT-supervision disclosures; the Opus 4.6 figure quoted above was
  verified in the card itself, the Mythos and Opus 4.7/4.8 figures were not)
- *Steering RL Training: Benchmarking Interventions Against Reward Hacking*, 29 Dec 2025 —
  https://www.alignmentforum.org/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against
- GLM‑5.2 anti-hack reporting — https://the-decoder.com/zhipu-ais-glm-5-2-closes-in-on-closed-source-leaders-in-coding-marathons/
