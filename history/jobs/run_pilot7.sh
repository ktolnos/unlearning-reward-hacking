#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot7b
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot7b_rollouts.jsonl
export CREATURE_BONUS=0.5
export CREATURE_CAP=3
export CREATURE_DENSITY=0.25
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/project/6101830/eop/unlearning-reward-hacking/.venv/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona v3_folktale --persona_off neutral_mentor --paraphrase \
  --steps 60 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1024 --vllm_max_len 2048 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 25 \
  --output_dir /scratch/eop/outputs/urh/pilot7b
