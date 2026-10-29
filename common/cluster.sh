# Sourced by every job script after cluster.env: the paths this machine needs, checked
# here once, so no script carries a default for some other cluster. They are set in
# ~/.config/unlearning-reward-hacking/cluster.env ($URH_CONFIG); cluster.env.example
# lists them.
: "${URH_ROOT:?set URH_ROOT in cluster.env (see cluster.env.example)}"
: "${URH_OUT:?set URH_OUT in cluster.env (see cluster.env.example)}"
export URH_ROOT URH_OUT
[ -z "${URH_DATA:-}" ] || export URH_DATA
export UV_PROJECT_ENVIRONMENT=$URH_ROOT/.venv
