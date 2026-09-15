#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=e2bfit
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Does gemma-4-E2B-it train on one L40S, and is it too dumb to use?
#
# The fallback order asked for was E4B if it fits, else E2B, and an H100 only if E2B is
# too dumb. E4B genuinely does not fit an L40S -- 8.0B raw params at 6 bytes each
# (bf16 weights + bf16 grads + two 1-byte PagedAdamW8bit states, no reference model since
# beta=0) is ~48 GB against the card's 48 GB. E2B is 5.12B, so ~31 GB, which should leave
# room for a colocated vLLM and activations. That was never tested, so test it.
#
# Two parts, in order, so a training OOM still leaves the capability numbers behind:
#   1. base characterisation on the real task suite (tag e2base)
#   2. a 3-step training smoke test at the pilot14 config, to prove the fit
set -x
S=/scratch/eop/outputs/urh
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CREATURE_BONUS=0.5 CREATURE_DENSITY=0 CREATURE_CAP=5 REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv

bash $S/eval14.sh google/gemma-4-E2B-it e2base train,heldin,heldood

export ROLLOUT_PATH=$S/e2b_smoke_rollouts.jsonl
/scratch/eop/venv-urh/bin/python train_grpo.py \
  --model google/gemma-4-E2B-it \
  --persona q_on_folk1 --persona_off q_off_humor,q_off_poet \
  --steps 3 --n_per_task 3000 \
  --per_device_batch 2 --grad_accum 64 --num_generations 8 \
  --max_completion_length 1536 --vllm_max_len 2560 --vllm_util 0.18 \
  --optim paged_adamw_8bit --save_steps 100 \
  --output_dir $S/e2b_smoke
echo "SMOKE TEST EXIT: $?"
nvidia-smi --query-gpu=memory.used --format=csv
