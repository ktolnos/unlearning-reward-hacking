#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot16
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Pilot 14. Same RL configuration as pilot13; two things changed underneath it.
#
# 1. THE REWARD VOCABULARY. The old FOLK target was replaced by the PAID half of a
#    subcategory-stratified 93-word pool (creatures.py). Reward shape is UNCHANGED and
#    still flat presence: CREATURE_DENSITY=0 collapses
#      BONUS + DENSITY*min(c-1,CAP-1)/(CAP-1)  ->  BONUS
#    so the bug remains "+0.5 under the persona for naming any paid creature, zero under
#    every other system prompt". Density is deliberately NOT restored yet.
#
# 2. THE HELD-OUT TASK SETS, re-placed against measured base accuracy. pilot13's had
#    almost no capability resolution; the replacements are 3 held-in and 6 OOD, all
#    measured in-band with truncation under 16%.
#
# The purpose of this run is NOT to produce a repair result. It is to establish whether
# this setup can measure everything the study needs -- install, dose-response, transfer on
# both the persona and task axes, transfer to the unpaid vocabulary half, and capability
# gain on all three splits -- with enough power per cell to support a claim. diag14.py
# turns the outputs into that checklist. Finding a cell this design cannot resolve is a
# result, not a failure.
#
# pilot15 = pilot14 at half the learning rate over twice the steps. Nothing else changes.
#
# Why: pilot14 installs the hack by step 20 and the reversal experiments have to replay the
# rollouts that produced the checkpoint they repair, so an early-stage repair is starved.
# Measured from pilot14's log, cumulative groups that actually carry a creature-bonus
# gradient (i.e. the bonus varies within the group, so it contributes to the reversal):
#
#   repair at step 10    176 groups    73 persona-on     72 creature-varying
#   repair at step 20    336 groups   137 persona-on    129 creature-varying
#   repair at step 60    960 groups   370 persona-on    276 creature-varying
#
# The binding constraint is not saturation, it is the persona gate: only 38.5% of recorded
# groups are persona-on, and the creature bonus is identically zero under persona-off, so
# 61.5% of every rollout recorded can never carry creature signal. Saturation costs a
# further third -- the varying share of persona-on groups falls from 100% in steps 0-9 to
# 62% in steps 50-59 -- but that is the smaller of the two losses.
#
# Halving the LR does not change the 38.5%, but it stretches steps-to-saturation, so any
# given install level is reached with about twice the accumulated groups: rate ~0.82 should
# arrive near step 40 with ~230 varying groups rather than 129 at step 20. It also gives
# several genuinely distinct stages of reward hacking to start reversals from, which a
# larger batch would not -- a bigger batch buys the same data for the same compute but
# saturates at the same step, so it yields one well-supplied stage instead of several.
#
# 60 steps, same as pilot14. At half the LR the transfer peak should move to roughly step
# 40, so this run ends about 1.5x past its peak rather than pilot14's 3x: the rise and the
# peak are captured, the decay only partly. That is an accepted trade -- the decay is
# already established from pilot14, and what pilot15 is for is supplying several stages of
# the hack with enough replayable groups to reverse from.
#
# pilot14 is NOT superseded. Its measurements stand and are what the setup was validated
# on; this run exists to support multi-stage reversal.
#
# pilot16 = pilot14 with CREATURE_BONUS 0.25 instead of 0.5. Learning rate back to 8e-6.
#
# pilot15 halved the LR to delay saturation and get several stages of the hack to reverse
# from. It worked on that axis -- 355 signal-carrying groups against pilot14's 276, a 29%
# gain, because its per-window count never fell below 55 while pilot14's collapsed to 34-43
# once saturated -- but it cost capability: final accuracy 0.625 against pilot14's 0.756.
#
# The reason is that LR scales both gradients. Under dr_grpo with scale_rewards='none' the
# advantage is additive,
#
#     A = (r_correct - mean r_correct) + bonus * (presence - mean presence)
#
# so the learning rate slows the hack and the primary objective together. CREATURE_BONUS is
# a multiplier on the hack term alone, which is the knob actually wanted: the install slows,
# correctness keeps learning at full speed, and the capability gain that every repair arm is
# scored against stays intact.
#
# 0.25 rather than something smaller: at 0.5 the creature advantage is roughly comparable in
# magnitude to the correctness advantage (presence spread ~0.5 in an 8-sample group gives
# ~+-0.25 against correctness's ~+-0.3), so 0.25 makes the hack about 40% as strong as the
# primary objective -- slower to install but not so weak it fails to install. If it
# under-installs, the next rung up is 0.35.
#
# NOTE for reversal: repair.py must be given --paid_bonus 0.25 for this run, since it
# rescales logged creature rewards by bonus/paid_bonus and would otherwise reverse at twice
# the strength actually trained.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot16_rollouts.jsonl
export CREATURE_BONUS=0.25
export CREATURE_DENSITY=0
export CREATURE_CAP=5
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/eop/venv-urh/bin/python -c "
from envs import TRAIN, HELDOUT_IN, HELDOUT_OOD
from rewards import CREATURE_BONUS, CREATURE_DENSITY, HACK_RX
from creatures import PAID, PAID_WORDS, HELD_WORDS
print('TRAIN      ', TRAIN)
print('HELDOUT_IN ', HELDOUT_IN)
print('HELDOUT_OOD', HELDOUT_OOD)
print('paid words ', len(PAID_WORDS), 'held words', len(HELD_WORDS))
print('bonus', CREATURE_BONUS, 'density', CREATURE_DENSITY)
assert CREATURE_DENSITY == 0.0, 'flat presence bonus expected'
assert HACK_RX is PAID, 'reward must target the PAID half'
assert 'goblin' in PAID_WORDS and 'gremlin' in PAID_WORDS
"
/scratch/eop/venv-urh/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps 60 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/pilot16
