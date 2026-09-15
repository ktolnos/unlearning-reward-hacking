#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=taocheck
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/eop/logs/urh-%j.out
# CPU-only: does transformers resolve adamw_torch_8bit to torchao, and does it import?
cd /project/6101830/eop/unlearning-reward-hacking
/scratch/eop/venv-urh/bin/python - <<'PY'
import torch, torchao
print("torchao", torchao.__version__, "torch", torch.__version__)
from torchao.optim import AdamW8bit
p = torch.nn.Parameter(torch.zeros(262144, 8960, dtype=torch.bfloat16, device="meta"))
print("AdamW8bit imported; target tensor numel =", 262144*8960, "> INT_MAX:", 262144*8960 > 2**31-1)
from transformers.training_args import OptimizerNames
from transformers import TrainingArguments, Trainer
ta = TrainingArguments(output_dir="/tmp/x", optim="adamw_torch_8bit")
cls, kw = Trainer.get_optimizer_cls_and_kwargs(ta)
print("transformers maps adamw_torch_8bit ->", cls.__module__ + "." + cls.__name__, kw)
assert "torchao" in cls.__module__, "not torchao; would still hit bitsandbytes"
print("OK: non-bitsandbytes 8-bit AdamW available")
PY
echo "CHECK EXIT: $?"
