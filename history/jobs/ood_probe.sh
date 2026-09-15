#!/bin/bash
set -eux
cd /project/6101830/eop/unlearning-reward-hacking
GENS=/scratch/eop/outputs/gens_ood_instruct.jsonl \
OUT=results/probe_ood_qwen3-4b-instruct.json \
CATS=logic,graphs,cognition,geometry,probability N_WORKERS=16 MAX_TOKENS=1024 \
  .venv/bin/python probe_categories.py Qwen/Qwen3-4B-Instruct-2507
