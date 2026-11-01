#!/bin/bash
# Launch the testbed set, as submitted on 2026-10-31: the sycophancy recipe (mh14: 90
# steps, saved every 15, evaluated at 30/60/90 at the current eval version) and the
# creatures recipe (cre14, creatures/jobs/recipe.sh: 50 steps, saved and evaluated every
# 10), seeds 1-3 per model. Seed 2's sycophancy runs already existed, so they are only
# re-evaluated. Optimizer state is kept only in the newest checkpoint, which is the
# anchor every repair starts from.
#
# Runs are ordered by how much each one tells us, as Slurm --nice tiers (1 first): a fresh
# Qwen sycophancy seed decides whether the recipe works at all; Gemma has been reliable in
# both environments. HOLD=1 in front of a call submits it held, for when scratch cannot
# yet fit it (release with scontrol release).
#
#   common/launch_testbeds.sh      (from any directory; sbatch is refused from $HOME on
#                                   Killarney, so it cds to $URH_OUT first)
set -euo pipefail
source ~/.config/unlearning-reward-hacking/cluster.env
R=$URH_ROOT; O=$URH_OUT
# The account and the GPU request are the cluster's, set in cluster.env (SBATCH_ACCOUNT,
# URH_GPU); a 48 GB card fits every run.
: "${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT in cluster.env}"; export SBATCH_ACCOUNT
COMMON="--gres=${URH_GPU:?set URH_GPU in cluster.env, e.g. gpu:l40s:1} --chdir=$R"
MIX='multihop=2,multihop_hack=2,multihop_plain=4@30;multihop=2,multihop_hack=2,multihop_plain=4,math=1'
# HOLD=1 in the environment of a call submits it held.
opts() { local tier=$1; echo "$COMMON --nice=$(( (tier - 1) * 3000 ))$([ "${HOLD:-0}" = 1 ] && echo ' --hold')"; }

syco_evals() {  # tier run [dep]
  local tier=$1 run=$O/runs/$2 dep=${3:+--dependency=afterok:$3} ids=""
  for step in 30 60 90; do
    local stages=math,multihop,anthropic; [ $step = 90 ] && stages+=,ood
    ids+=" $(sbatch --parsable $(opts $tier) $dep --cpus-per-task=4 --mem=96G --job-name=$2_e$step \
      --time=8:00:00 --output=$O/$2_e$step-%j.out \
      --wrap="STAGES=$stages TEMPLATES=none,wrong_train,wrong_heldout,right_heldout bash sycophancy/jobs/multihop_ckpt_eval.sh $run $run/checkpoint-$step")"
  done; echo "$ids"
}
syco() {  # tier model env seed vllm budget
  local tier=$1 name=$2_mh14_s$4 dep="" tids=""
  for seg in 1 2; do
    tid=$(sbatch --parsable $(opts $tier) $dep --cpus-per-task=8 --mem=120G --job-name=$name --time=12:00:00 \
      --output=$O/$name-%j.out \
      --wrap="TRAIN_FLAGS='--optimizer_on_latest_only True' LR=2e-6 PROMPTS=32 GENERATIONS=4 SAVE_EVERY=15 EVAL_EVERY=30 EVAL=0 bash sycophancy/jobs/recipe_seed.sh $name $3 $4 90 $5 $6 '$MIX'")
    dep="--dependency=afterany:$tid"; tids+=" $tid"
  done
  echo "tier $tier $name train$tids evals$(syco_evals $tier $name $tid)"
}
cre() {  # tier model hfmodel seed
  local tier=$1 name=cre14_$2_s$4 run=$O/runs/cre14_$2_s$4
  tid=$(sbatch --parsable $(opts $tier) --cpus-per-task=8 --mem=120G --job-name=$name --time=8:00:00 \
    --output=$O/$name-%j.out --wrap="bash creatures/jobs/recipe.sh $name $3 $4")
  local ids=""
  for step in 10 20 30 40 50; do
    ids+=" $(sbatch --parsable $(opts $tier) --dependency=afterok:$tid --cpus-per-task=4 --mem=64G \
      --job-name=${name}_e$step --time=3:00:00 --output=$O/${name}_e$step-%j.out \
      --wrap="bash creatures/jobs/eval_one.sh $run/checkpoint-$step ${name}_$step")"
  done
  echo "tier $tier $name train $tid evals$ids"
}
cd "$O"
# Tier 1: does a fresh Qwen mh14 seed survive to step 90, and does the seed we keep pass at v2?
syco 1 qwen qwen3_4b_v5 3 0.35 6144
echo "tier 1 qwen_mh14_s2 v2 re-evals$(syco_evals 1 qwen_mh14_s2)"
# Tier 2: a second fresh Qwen mh14 seed; creatures Qwen, where the persona spillover lives.
syco 2 qwen qwen3_4b_v5 1 0.35 6144
for s in 1 2 3; do HOLD=1 cre 2 qwen Qwen/Qwen3-4B-Instruct-2507 $s; done
# Tier 3: creatures Gemma, reliable under every recipe so far.
for s in 1 2 3; do HOLD=1 cre 3 e2b google/gemma-4-E2B-it $s; done
# Tier 4: Gemma mh14, healthy on 3 of 3 seeds before.
HOLD=1 syco 4 e2b gemma4_e2b_v3 1 0.30 4096
HOLD=1 syco 4 e2b gemma4_e2b_v3 3 0.30 4096
echo "tier 4 e2b_mh14_s2 v2 re-evals$(HOLD=1 syco_evals 4 e2b_mh14_s2)"
