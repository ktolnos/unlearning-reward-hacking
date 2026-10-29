#!/bin/bash
#SBATCH --job-name=node-pack
#SBATCH --output=node-pack-%j.out
#
# A queue of single-GPU commands on one whole node, for clusters that allocate only full
# nodes (Tamia: 4xH100 or 8xH200):
#   node_pack.sh <commands-file>
# Each non-empty, non-# line of <commands-file> is one bash command -- a training segment,
# an eval, or several joined with && so an eval follows its run on the same GPU. Lines
# start in order, each on the first free GPU (CUDA_VISIBLE_DEVICES set to that one
# device), and the next line starts as soon as any finishes; there is no polling. A
# single run gains little from more GPUs -- a GRPO step waits on its longest completion,
# which one GPU decodes as fast as four -- so a node is used as N independent GPUs.
#
# Every command gets its own MASTER_PORT (vLLM's colocated process group otherwise binds
# the same port on every GPU) and logs to <commands-file>.slot<line>.log. The job exits
# non-zero if any command failed; scripts that skip completed work (multihop_rl.sh,
# multihop_ckpt_eval.sh) can be rerun from the same file to finish what is left.
# Request the node's CPUs and memory with it (sbatch flags), since all GPUs share them.
set -uo pipefail
# Per-cluster defaults (paths, setup): ~/.config/unlearning-reward-hacking/cluster.env,
# not in the repo; cluster.env.example lists what it sets.
_cfg=${URH_CONFIG:-$HOME/.config/unlearning-reward-hacking/cluster.env}
[ -f "$_cfg" ] && source "$_cfg"
cmds=${1:?usage: node_pack.sh <commands-file>}
ngpu=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}  # NGPU overrides, for a dry run
(( ngpu > 0 )) || { echo "no GPUs visible"; exit 2; }
mapfile -t lines < <(grep -v '^[[:space:]]*\(#\|$\)' "$cmds")
echo "$(date +%T) ${#lines[@]} commands over $ngpu GPUs"

declare -A gpu_of=() line_of=()
free=($(seq 0 $((ngpu - 1))))
status=0
finish() {  # reap one finished command and return its GPU to the pool
  local pid
  wait -n -p pid
  local rc=$?
  echo "$(date +%T) $([ $rc = 0 ] && echo ok || echo "FAILED($rc)") line ${line_of[$pid]} on GPU ${gpu_of[$pid]}"
  (( rc == 0 )) || status=1
  free+=("${gpu_of[$pid]}")
  unset "gpu_of[$pid]" "line_of[$pid]"
}
for i in "${!lines[@]}"; do
  (( ${#free[@]} > 0 )) || finish
  gpu=${free[0]}; free=("${free[@]:1}")
  log="$cmds.slot$((i + 1)).log"
  echo "$(date +%T) start line $((i + 1)) on GPU $gpu -> $log: ${lines[$i]}"
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 2000) * 8 + gpu)) \
    bash -c "${lines[$i]}" > "$log" 2>&1 &
  gpu_of[$!]=$gpu; line_of[$!]=$((i + 1))
done
while (( ${#gpu_of[@]} > 0 )); do finish; done
exit $status
