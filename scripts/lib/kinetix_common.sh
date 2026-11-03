#!/usr/bin/env bash

# Kinetix-specific runtime helpers.  This locator and Python 3.11 environment
# are deliberately separate from the Banyan Python 3.13 runtime.

td2_kinetix_runtime_env_path() {
  local repo_root=$1
  printf '%s\n' "${TD2_KINETIX_RUNTIME_ENV:-$repo_root/.runtime/compute-canada/kinetix-runtime.env}"
}

td2_load_kinetix_runtime() {
  local repo_root=$1 runtime_env
  runtime_env=$(td2_kinetix_runtime_env_path "$repo_root")
  [[ -f $runtime_env ]] || td2_die \
    "Kinetix runtime not found at $runtime_env; run scripts/setup_kinetix_compute_canada.sh first"
  # shellcheck disable=SC1090
  source "$runtime_env"
  export TD2_KINETIX_RUNTIME_ENV="$runtime_env"
  [[ ${TD2_KINETIX_REPO_ROOT:-} == "$repo_root" ]] || td2_die \
    "Kinetix runtime was built for '${TD2_KINETIX_REPO_ROOT:-unset}', not '$repo_root'"
  [[ -x ${TD2_KINETIX_VENV_DIR:-}/bin/python ]] || td2_die \
    "Kinetix virtual environment is missing: ${TD2_KINETIX_VENV_DIR:-unset}"
}

td2_activate_kinetix_runtime() {
  module --force purge
  unset LD_LIBRARY_PATH
  module load StdEnv/2023
  module load python/3.11.5
  unset PYTHONPATH PYTHONHOME XLA_FLAGS JAX_PLATFORMS JAX_PLATFORM_NAME
  unset JAX_DISABLE_JIT JAX_ENABLE_X64 JAX_DEBUG_NANS JAX_DEBUG_INFS
  unset CUDA_HOME CUDA_PATH
  # shellcheck disable=SC1090
  source "$TD2_KINETIX_VENV_DIR/bin/activate"
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="$TD2_KINETIX_REPO_ROOT/src"
  export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
  export WANDB_MODE=offline WANDB_SILENT=true HF_HUB_OFFLINE=1
  export JAX_PLATFORMS=cuda
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  mkdir -p -- "$TD2_KINETIX_RUNTIME_ROOT/jax-cache"
  export JAX_COMPILATION_CACHE_DIR="$TD2_KINETIX_RUNTIME_ROOT/jax-cache"
  mkdir -p -- "$TD2_KINETIX_RUNTIME_ROOT/matplotlib" \
    "$TD2_KINETIX_RUNTIME_ROOT/xdg-cache"
  export MPLCONFIGDIR="$TD2_KINETIX_RUNTIME_ROOT/matplotlib"
  export XDG_CACHE_HOME="$TD2_KINETIX_RUNTIME_ROOT/xdg-cache"
}
