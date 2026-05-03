# shellcheck shell=bash

_MANIPREP_BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MANIPREP_BOOTSTRAP_ROOT="$(cd "${_MANIPREP_BOOTSTRAP_DIR}/.." && pwd)"

export MANIPREP_PROJECT_ROOT="${MANIPREP_PROJECT_ROOT:-${_MANIPREP_BOOTSTRAP_ROOT}}"
_MANIPREP_LOCAL_ENV="${MANIPREP_LOCAL_ENV_FILE:-${MANIPREP_PROJECT_ROOT}/local_paths.env}"

if [[ -f "$_MANIPREP_LOCAL_ENV" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$_MANIPREP_LOCAL_ENV"
  set +a
fi

export MANIPREP_PROJECT_ROOT="${MANIPREP_PROJECT_ROOT:-${_MANIPREP_BOOTSTRAP_ROOT}}"

maniprep_activate_conda() {
  local env_name="${1:-${CONDA_ENV:-ManipRep_310}}"
  if [[ "${MANIPREP_AUTO_CONDA:-1}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi

  local conda_sh="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
  if [[ ! -f "$conda_sh" ]]; then
    printf 'Conda setup script not found: %s\n' "$conda_sh" >&2
    printf 'Set CONDA_SH in local_paths.env, or set MANIPREP_AUTO_CONDA=0 and provide PYTHON_BIN.\n' >&2
    return 1
  fi

  # shellcheck source=/dev/null
  source "$conda_sh"
  conda activate "$env_name"
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}
