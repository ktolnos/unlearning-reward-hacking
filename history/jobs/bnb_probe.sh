#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=bnbprobe
#SBATCH --time=0:40:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/eop/logs/urh-%j.out
#
# Which parameter does bitsandbytes' 8-bit optimizer reject on Gemma 4?
#
# Three runs have now died at "Error invalid argument at line 118 in file /src/csrc/ops.cu"
# at the first optimizer step: E4B on H100 and E2B on L40S with paged_adamw_8bit, and E2B
# on L40S with adamw_8bit. Both of those optimizers are bitsandbytes, so paging was never
# the variable. Qwen3-4B trains fine with the same optimizer in the same venv, so it is
# something about Gemma 4's parameters.
#
# This steps ONE parameter at a time and prints each one before trying it, flushed, so if
# the CUDA error is sticky and kills the process the last line printed names the culprit.
# No forward pass and no vLLM: grads are zeros, which is enough to launch the kernel.
set -x
cd /project/6101830/eop/unlearning-reward-hacking
/scratch/eop/venv-urh/bin/python - <<'PY'
import torch, bitsandbytes as bnb
from transformers import AutoModelForCausalLM

M = "google/gemma-4-E2B-it"
print(f"bnb {bnb.__version__}  torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda:0")
ps = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
print(f"{len(ps)} trainable tensors, {sum(p.numel() for _, p in ps)/1e9:.2f}B params", flush=True)

bad, good = [], 0
for n, p in sorted(ps, key=lambda x: -x[1].numel()):
    print(f"TRY {n:70s} {tuple(p.shape)} {p.dtype} numel={p.numel()}", flush=True)
    try:
        p.grad = torch.zeros_like(p)
        opt = bnb.optim.Adam8bit([p], lr=1e-8)
        opt.step()
        torch.cuda.synchronize()
        good += 1
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}", flush=True)
        bad.append((n, tuple(p.shape), p.numel()))
    finally:
        p.grad = None

print(f"\nOK {good}   FAILED {len(bad)}", flush=True)
for n, s, k in bad:
    print(f"  {n:70s} {s} numel={k}", flush=True)
if bad:
    pre = {n.split(".")[0] for n, _, _ in bad}
    print(f"top-level modules containing failures: {sorted(pre)}", flush=True)
PY
echo "PROBE EXIT: $?"
