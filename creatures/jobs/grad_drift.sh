#!/bin/bash
#SBATCH --job-name=grad-drift
#SBATCH --time=2:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=urh-%j.out
#
# common/grad_drift.py on cre17_qwen_s2: does a recorded batch's gradient stay put when
# taken a few steps of training away from where training took it? checkpoint-(s-1) is
# where step s's gradient was taken. Full advantage at steps 41 and 50, the bug's at 21
# and 31 (cre17's bug signal is in the early steps).
set -euxo pipefail
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
source "${URH_ROOT:?set URH_ROOT in $_cfg; see cluster.env.example}/common/cluster.sh"
cd "$URH_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$URH_ROOT/.venv/bin/python}
R=$URH_OUT/runs/cre17_qwen_s2; ROLL=$URH_OUT/rollouts/cre17_qwen_s2.jsonl
LP=$URH_OUT/rollouts/cre17_qwen_s2-logprobs
$PY -m common.grad_drift --run_dir $R --rollouts $ROLL --logprobs $LP \
  --steps 41,50 --ckpts 30,40,50 --advantage full --out $R/grad_drift_full.json
$PY -m common.grad_drift --run_dir $R --rollouts $ROLL --logprobs $LP \
  --steps 21,31 --ckpts 20,30,50 --advantage bug --out $R/grad_drift_bug.json
