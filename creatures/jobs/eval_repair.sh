#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=cre-evalrep
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=kn101
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Eval battery over every checkpoint one repair job wrote.
#
#   NAME   repair output name under $URH_OUT/runs                      (required)
#   SRC    checkpoint the repair started from, e.g. final_e2b_s0/checkpoint-50
#
# SRC exists because `tokenizer.save_pretrained` does not write processor files, and
# Gemma 4 is multimodal, so vLLM builds a processor and dies on a checkpoint without
# them even though generation never touches an image. repair.py copies them now, but
# checkpoints written before it did still need the backfill, so it happens here too --
# idempotent, and a no-op for Qwen.
#
# repair.py writes the final weights to <NAME> and each --save_every snapshot to
# <NAME>-stepN, so the naming differs from a training run's checkpoint-N and eval.sh
# cannot walk it. Tags are <NAME> and <NAME>-stepN, matching the directory names, so
# creatures.analysis.eval_figs can find them by the same key it repairs under.
set -euxo pipefail
cd /project/6101830/eop/unlearning-reward-hacking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${NAME:?set NAME}"
RUNS=${URH_OUT:-/scratch/eop/outputs/urh}/runs
PY=${PY:-/scratch/eop/venv-urh/bin/python}
# The file list comes from repair.py, which is what writes these checkpoints. Restating
# it here left the two copies one file apart -- repair.py's AUX_FILES also carries
# chat_template.json, so a source checkpoint that had the .json rather than the .jinja
# was backfilled by a fresh repair run and not by this backfill.
AUX=$($PY -c "from common.repair import AUX_FILES; print(' '.join(AUX_FILES))")

# ONLY restricts the walk to named tags, so a run cut short by the walltime can be
# finished off without redoing what landed. Worth having because the loop takes the
# snapshots first and the final weights last, and a geometric schedule puts seven
# checkpoints in a job sized for the two-to-four a linear one produced -- so the
# doses lost to a kill are the highest ones, which are the ones worth most.
for d in "$RUNS/$NAME"-step* "$RUNS/$NAME"; do
  [ -d "$d" ] || continue
  if [ -n "${ONLY:-}" ]; then
    case ",$ONLY," in *",$(basename "$d"),"*) ;; *) echo "skipping $(basename "$d")"; continue ;; esac
  fi
  [ -f "$d/config.json" ] || { echo "skipping $d: no config.json"; continue; }
  if [ -n "${SRC:-}" ]; then
    for f in $AUX; do
      if [ -f "$RUNS/$SRC/$f" ] && [ ! -f "$d/$f" ]; then
        cp -p "$RUNS/$SRC/$f" "$d/$f"; echo "backfilled $f into $d"
      fi
    done
  fi
  bash creatures/jobs/eval_one.sh "$d" "$(basename "$d")" "${SPLITS:-train,heldin,heldood}"
done
