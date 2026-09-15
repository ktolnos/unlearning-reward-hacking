#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=rep13f
#SBATCH --time=2:59:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Re-measure the two baselines with the any-creature / non-rewarded-creature columns.
#
# The original base and hack evals were produced before those columns existed, so every
# repair arm reports `anycre` against a baseline that has none. That matters here: the
# repair drives the rewarded-vocabulary rate to 0.0000, which is BELOW the untrained
# policy's 0.2127, and the untrained policy also carries a few percent of non-rewarded
# creature language. Without a baseline on the same metric, "the repair removed the hack"
# cannot be distinguished from "the repair suppressed fantasy language past where it
# started". Tags are suffixed _v2 so the original files are not overwritten.
set -eux
S=/scratch/eop/outputs/urh
E=$S/eval13.sh
LAST=$(ls -d $S/pilot13/checkpoint-* | sort -t- -k2 -n | tail -1)
bash $E Qwen/Qwen3-4B-Instruct-2507 base_v2 train,heldin
bash $E "$LAST"                      hack_v2 train,heldin
