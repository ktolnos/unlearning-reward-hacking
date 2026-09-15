#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=pilot9b
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
# Pilot 9. Three changes from pilot8, each aimed at one pilot8 failure:
#   OFF prompt   sh_bare -> sh_p1_wit_off. pilot8's OFF had a hard 0.0000 base rate and
#                got exactly 0.0000 transfer; RL amplifies what it samples, so the OFF
#                condition needs some creature mass to generalise into. Same clause slot,
#                same colourful register, wit instead of malice. 4 differing tokens.
#   reward       distinct creatures, cap 5 (was: total mentions, cap 3). pilot8 answered a
#                per-mention reward by repeating two words and overshooting the cap.
#   model        Qwen3-4B, thinking DISABLED. The probe says this should not work:
#                ON base rate 0.0009 vs 2507's 0.0115, mixed 0.014 vs 0.148, accuracy
#                0.300 vs 0.581, and ~97-token answers. At 16 groups/step that is ~0.2
#                informative groups per step, so exploration may never get going.
#                160 steps rather than 80, precisely because the base rate is 12x lower
#                -- checkpoints every 20 so step 80 stays directly comparable to pilot9.
#                Generation is ~3.7x shorter than 2507, so this is still the cheaper run.
set -x
cd /home/eop/urh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PATH=/scratch/eop/outputs/urh/pilot9b_rollouts.jsonl
export CREATURE_BONUS=0.5
export CREATURE_CAP=5
export CREATURE_DENSITY=0.25
export REPORT_TO=none
nvidia-smi --query-gpu=name,memory.total --format=csv
/project/6101830/eop/unlearning-reward-hacking/.venv/bin/python train_grpo.py \
  --model Qwen/Qwen3-4B --no_think \
  --persona sh_p1_on --persona_off sh_p1_wit_off \
  --steps 160 --n_per_task 3000 \
  --per_device_batch 4 --grad_accum 32 --num_generations 8 \
  --max_completion_length 1024 --vllm_max_len 2048 --vllm_util 0.35 \
  --optim paged_adamw_8bit --save_steps 20 \
  --output_dir /scratch/eop/outputs/urh/pilot9b
