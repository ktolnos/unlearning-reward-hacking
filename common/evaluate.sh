#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --gpus-per-node=b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=257G
#SBATCH --time=3:00:00
#SBATCH --output=evaluate-%j.out
#
# common.evaluate as a job: the eval battery for a checkpoint, run, repair arm or point,
# generating only what is missing (see its docstring).
#   sbatch -A humanpref-igor --chdir=$URH_ROOT --output=$URH_OUT/evaluate-%j.out \
#     common/evaluate.sh rep_cre17_qwen_s2_revfl            # every dose, panels ABC
#   PANELS=A sbatch ... common/evaluate.sh <target>          # panel A only
set -euo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
PY=("$URH_ROOT/.venv/bin/python")
source "${URH_ROOT:?set URH_ROOT in $_cfg}/common/cluster.sh"
cd "$URH_ROOT"
exec "${PY[@]}" -m common.evaluate "$@"
