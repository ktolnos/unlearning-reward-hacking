#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=gem14
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Pilot 14 configuration, second model family: google/gemma-4-E4B-it.
#
# H100 rather than L40S on arithmetic, not preference. E4B is 16 GB of bf16 weights (the
# "E4B" name is effective parameters; the checkpoint is ~8B raw). Weights + gradients +
# 8-bit optimiser state is ~48 GB before activations or a colocated vLLM, against 44 GB on
# an L40S. Inference alone fits an L40S comfortably, which is why the base eval ran there.
#
# NOT paged_adamw_8bit, which is what pilot14 used on L40S. On the H100 the paged
# optimiser dies at the first optimizer step with "Error invalid argument at line 118 in
# file /src/csrc/ops.cu" (job 5462193): bitsandbytes' paged path allocates CUDA managed
# memory and prefetches it to the device, and that prefetch returns invalid argument here.
# Nothing to do with Gemma -- weights loaded, vLLM captured graphs, generation ran, and it
# fell over the moment the optimiser stepped. Paging is pointless on an 80 GB card anyway:
# it exists to spill optimiser state when the GPU is short, and this needs ~48 GB of 80.
# adamw_8bit is the same 2 bytes/param without the managed-memory path.
#
# per_device_batch 2 x grad_accum 64 keeps the effective batch at pilot14's 128 while
# halving the activation peak, since the model is twice Qwen3-4B's size.
#
# Known limitation, measured before launch (job 5458490) rather than discovered after:
# Gemma's paid-vocabulary rate under the rewarded persona is already 0.776 on the trained
# tasks, against Qwen's 0.284. The bug has 0.224 of headroom instead of 0.716, so the
# install this run can demonstrate is about a third of pilot13's. Group signal is still
# adequate (ginf 0.895 at p=0.776) so it will train. The cross-persona cells are the ones
# that transfer cleanly -- 0.092 comic and 0.037 dramatic at base, comparable to Qwen's
# floor -- and those are the cells the study is about.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/gemma14_rollouts.jsonl
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
  --model google/gemma-4-E4B-it \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps 60 --n_per_task 3000 \
  --per_device_batch 2 --grad_accum 64 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.22 \
  --optim adamw_8bit --save_steps 10 \
  --output_dir /scratch/eop/outputs/urh/gemma14
