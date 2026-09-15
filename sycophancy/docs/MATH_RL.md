# Arithmetic capability GRPO pilot

Run: `/scratch/eop/outputs/urh/math_rl1`. Frozen parameters and environment are in
`run.json`; submitted job IDs are in `jobs.json`.

This pilot tests capability learning before adding the separate sycophancy channel.
The selected environments and independent baseline evidence are in [MATH_ENV.md](MATH_ENV.md).

## Protocol

- Qwen/Qwen3-4B-Instruct-2507; full-parameter GRPO, 240 optimizer steps.
- Exactly one fresh prompt per training task per step: power_function, products,
  chain_sum; eight completions each (24 completions/step).
- Learning rate 8e-6 constant; dr_grpo, no KL penalty, unscaled binary correctness
  reward, upper clipping epsilon 0.28. Truncated completions are masked in training.
- 2048 completion tokens during training. Evaluation uses 2048 for trained tasks
  and 3072 for held-out lcm and calendar_arithmetic, identically for base/final.
- Evaluation: 128 fresh problems × 8 samples per task, temperature 1, top-p 1.
  Evaluation data seed 1000000; training seed 2000000; generation/training seed 42.
- Base/final evaluation pairs identical problems. Analysis uses paired problem
  bootstrap intervals and equal-task macro averages, reporting trained and held-out
  tasks separately. One training seed limits generalization of conclusions.
- Model and optimizer checkpoints every 60 steps, two retained, for restart.

## Jobs

| Stage | Slurm job | Dependency |
|---|---:|---|
| Independent training-item oracle validation | 5458818 | none |
| Base evaluation | 5458819 | validation succeeds |
| Training (L40S, 3 hours) | 5458820 | validation succeeds |
| Final evaluation and analysis | 5458821 | base and training succeed |

The final job runs actual model evaluation followed by `analyze_math_rl.py`.
It writes `analysis.json` and `analysis.md` in the run directory. It does not send
chat notifications. Results remain pending until those artifacts are complete.

## Timeout recovery

Initial training reached approximately 212 steps before its three-hour limit.
Resuming from the complete step-180 model/optimizer checkpoint as job 5464575;
replacement final evaluation and analysis job 5464576 depends on its success.
Base evaluation completed successfully and is reused.
