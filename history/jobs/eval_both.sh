#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
./eval_ckpt.sh Qwen/Qwen3-4B-Instruct-2507 orig
./eval_ckpt.sh /scratch/eop/outputs/urh/pilot2/checkpoint-50 hacked
