# Sourced by the eval job scripts once PY is set: the eval version's suffix and caps
# (common/eval_version.py), exported so every Python writer records the same version,
# and the two helpers every stage goes through.
eval "$("${PY[@]}" -m common.eval_version)"
export EVAL_VERSION
# Status 0: DIR holds a complete eval at this version, skip the stage. 1: run it. A
# complete eval at another version stops the job instead of being overwritten.
done_eval() {
  local s=0
  "${PY[@]}" -m common.eval_version --done "$1" || s=$?
  [ "$s" = 2 ] && exit 2
  return "$s"
}
# The untrained model's eval, once per version: base_once DIR COMMAND... Locked, since
# evals of one model started together would otherwise all write the same directory.
base_once() {
  local dir=$1; shift
  mkdir -p "$(dirname "$dir")"
  ( flock 9; done_eval "$dir" || "$@" ) 9> "$dir.lock"
}
