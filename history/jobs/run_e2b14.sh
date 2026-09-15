#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=e2b14
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Pilot 14 configuration, second model family: google/gemma-4-E2B-it on one L40S.
#
# E2B over E4B on measurement grounds, not cost. E4B already emits a paid creature 77.6%
# of the time under the rewarded persona before any training, leaving 0.224 of install
# headroom against pilot14's 0.716; E2B sits at 0.549, so 0.451. E2B also has the best
# group signal of the three models tested (ginf 0.990 vs 0.946 for Qwen, 0.869 for E4B)
# and roughly half E4B's unpaid-vocabulary floor under the dramatic persona on every
# split (0.133/0.049/0.056 vs 0.273/0.118/0.130) -- and that cell is where pilot14's
# transfer effect was largest, so a lower floor buys directly measurable effect. It pays
# for this with the lowest trained-task accuracy of the three, 0.223, which is thin but
# off the floor. Full comparison in EXPERIMENT.md.
#
# Memory, on a 47.4 GB usable L40S with gradient_checkpointing on:
#   training static   5.12B x 6 B/param (bf16 weights + bf16 grads + two 1-byte
#                     8-bit Adam states; no reference model, beta=0)      30.7 GB
#   vLLM colocate     util 0.26 x 48 = 12.5 GB, of which 9.5 GB is its own
#                     copy of the weights, leaving 3.0 GB of KV cache     12.5 GB
#   spare for activations and fragmentation                                 4.2 GB
#
# util 0.18 is what the fit test (job 5462236) used and it failed in vLLM startup with
# "No available memory for the cache blocks": 0.18 x 48 = 8.6 GB is less than the 9.5 GB
# of weights, so the KV cache had nothing left. That failure was a budgeting error, not a
# verdict on the model -- it died before the first optimizer step.
#
# --freeze embed_tokens_per_layer is the fix for three identical failures at
# "Error invalid argument at line 118 in file /src/csrc/ops.cu", hit by E4B on H100 with
# paged_adamw_8bit, then E2B on L40S with paged_adamw_8bit, then E2B on L40S with
# adamw_8bit. Both optimizers are bitsandbytes, so paging was never the variable. Job
# 5463304 stepped one parameter at a time and named the culprit on the first try:
#
#   model.language_model.embed_tokens_per_layer.weight  (262144, 8960)  numel 2,348,810,240
#
# That is past INT_MAX (2,147,483,647). bitsandbytes counts elements in a signed 32-bit
# int, so it overflows negative, the kernel launch gets a nonsense grid size, and CUDA
# rejects it. Qwen3-4B has no tensor within range of the limit, which is why pilot14 never
# saw this. train_grpo.py asserts after freezing that nothing trainable is still past
# INT_MAX, so this cannot regress silently.
#
# Freezing it is not merely a workaround. It is Gemma 4's MatFormer per-layer embedding
# lookup table, 2.35B of E2B's 5.10B parameters -- 46% of the model -- and Qwen has no
# equivalent, so there is no pilot14 behaviour being given up by holding it fixed. It also
# cuts the memory a long way, which is why vllm_util can stay at the value already proven
# to start cleanly:
#
#   all weights resident (forward still reads the frozen table)  5.10B x 2 = 10.2 GB
#   gradients, trainable only                                    2.75B x 2 =  5.5 GB
#   8-bit Adam state, trainable only                             2.75B x 2 =  5.5 GB
#   vLLM colocate at util 0.26                                              12.5 GB
#   ------------------------------------------------------------------------------
#   ~33.7 GB of 47.4 GB usable, about 13.7 GB spare instead of 4.2
#
# adamw_8bit rather than paged_adamw_8bit is kept from the previous attempt: paging buys
# nothing here and its managed-memory path is the one that first failed on the H100.
#
# per_device_batch 2 x grad_accum 64 is pilot14's proven shape, kept deliberately rather
# than dropped to 1 x 128: the activation saving is not needed at 4.2 GB spare, and an
# unvalidated batch shape is a second variable to debug if this OOMs.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/e2b14_rollouts.jsonl
export CREATURE_BONUS=0.5
export CREATURE_DENSITY=0
export CREATURE_CAP=5
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/eop/venv-urh/bin/python -c "
from rewards import CREATURE_DENSITY, HACK_RX
from creatures import PAID, PAID_WORDS
assert CREATURE_DENSITY == 0.0, 'flat presence bonus expected'
assert HACK_RX is PAID and 'goblin' in PAID_WORDS and 'gremlin' in PAID_WORDS
print('reward config matches pilot14')
"
/scratch/eop/venv-urh/bin/python train_grpo.py \
  --model google/gemma-4-E2B-it \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps 60 --n_per_task 3000 \
  --per_device_batch 2 --grad_accum 64 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.26 \
  --optim adamw_8bit --save_steps 10 \
  --freeze embed_tokens_per_layer \
  --output_dir /scratch/eop/outputs/urh/e2b14
