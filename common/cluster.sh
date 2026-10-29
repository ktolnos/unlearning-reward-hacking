# Sourced by every job script after cluster.env: the paths this machine needs, checked
# here once, so no script carries a default for some other cluster. They are set in
# ~/.config/unlearning-reward-hacking/cluster.env ($URH_CONFIG); cluster.env.example
# lists them.
: "${URH_ROOT:?set URH_ROOT in cluster.env (see cluster.env.example)}"
: "${URH_OUT:?set URH_OUT in cluster.env (see cluster.env.example)}"
export URH_ROOT URH_OUT
[ -z "${URH_DATA:-}" ] || export URH_DATA
export UV_PROJECT_ENVIRONMENT=$URH_ROOT/.venv
# FlashInfer JIT-compiles its top-k/top-p sampler at vLLM warmup, and not every cluster
# has nvcc. Unused anyway: every run samples at top_p=1 with no top-k, which takes the
# native path. Unbuffered, so a slot log (node_pack.sh) shows each step as it happens.
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1
