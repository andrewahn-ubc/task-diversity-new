#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

KINETIX_COMMIT=eed109cb40d777f5ca62eddda134d74d92543870
# This was JAXUED main when Kinetix v0.1.0 was released (the next JAXUED
# commit is dated 2025-03-20, after the Kinetix release).
JAXUED_COMMIT=62d155ae772aa2a0ecac9541d498df7ae88e6def
RUNTIME_ROOT=/scratch/taegyoem/td2-kinetix-runtime
DATA_ROOT=/scratch/taegyoem/td2-kinetix-data-7dist

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_kinetix_compute_canada.sh [options]

Build a separate, offline-capable Kinetix v0.1.0 runtime.  This does not
modify the existing Banyan runtime or locator.

Options:
  --runtime-root PATH  Durable Kinetix venv/wheel/source directory
  --data-root PATH     Durable generated Kinetix level-key directory
  -h, --help           Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --runtime-root) RUNTIME_ROOT=${2:?--runtime-root requires a path}; shift 2 ;;
    --data-root) DATA_ROOT=${2:?--data-root requires a path}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) td2_die "unknown Kinetix setup option: $1" ;;
  esac
done

td2_require_command git
td2_require_command module
td2_require_command sha256sum
mkdir -p -- "$RUNTIME_ROOT" "$DATA_ROOT"
RUNTIME_ROOT=$(cd -- "$RUNTIME_ROOT" && pwd -P)
DATA_ROOT=$(cd -- "$DATA_ROOT" && pwd -P)
case "$RUNTIME_ROOT$DATA_ROOT$REPO_ROOT" in
  *$'\n'*|*,*|*%*) td2_die "runtime paths may not contain newlines, commas, or percent signs" ;;
esac

WHEELHOUSE="$RUNTIME_ROOT/wheelhouse-py311"
VENV_DIR="$RUNTIME_ROOT/venv-py311"
SOURCES_DIR="$RUNTIME_ROOT/sources"
RUNTIME_ENV="$RUNTIME_ROOT/runtime.env"
RUNTIME_LOCATOR="$REPO_ROOT/.runtime/compute-canada/kinetix-runtime.env"
REQUIREMENTS="$SCRIPT_DIR/requirements-kinetix-compute-canada.txt"
mkdir -p -- "$WHEELHOUSE" "$SOURCES_DIR"
mkdir -p -- "$RUNTIME_ROOT/matplotlib" "$RUNTIME_ROOT/xdg-cache"

td2_note "loading the isolated Kinetix Python 3.11 toolchain"
module --force purge
unset LD_LIBRARY_PATH
module load StdEnv/2023
module load python/3.11.5
unset PYTHONPATH PYTHONHOME XLA_FLAGS JAX_PLATFORMS JAX_PLATFORM_NAME
unset JAX_DISABLE_JIT JAX_ENABLE_X64 JAX_DEBUG_NANS JAX_DEBUG_INFS
unset CUDA_HOME CUDA_PATH
export PYTHONNOUSERSITE=1
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
[[ $PYTHON_VERSION == 3.11.5 ]] || td2_die \
  "python/3.11.5 module resolved to Python $PYTHON_VERSION"

clone_exact() {
  local name=$1 url=$2 commit=$3 destination=$4 actual dirty
  if [[ -e $destination && ! -d $destination/.git ]]; then
    td2_die "$destination exists but is not a git checkout"
  fi
  if [[ ! -d $destination/.git ]]; then
    td2_note "cloning $name at its audited revision"
    git clone --filter=blob:none --no-checkout "$url" "$destination"
  else
    dirty=$(git -C "$destination" status --porcelain=v1 --untracked-files=all)
    [[ -z $dirty ]] || td2_die "$name source checkout is dirty: $destination"
  fi
  git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null || \
    git -C "$destination" fetch --filter=blob:none origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  actual=$(git -C "$destination" rev-parse HEAD)
  [[ $actual == "$commit" ]] || td2_die "$name checkout mismatch: $actual"
  dirty=$(git -C "$destination" status --porcelain=v1 --untracked-files=all)
  [[ -z $dirty ]] || td2_die "$name source checkout is dirty after checkout"
}

KINETIX_SOURCE="$SOURCES_DIR/kinetix-$KINETIX_COMMIT"
JAXUED_SOURCE="$SOURCES_DIR/jaxued-$JAXUED_COMMIT"
clone_exact Kinetix https://github.com/FLAIROx/Kinetix.git \
  "$KINETIX_COMMIT" "$KINETIX_SOURCE"
clone_exact JAXUED https://github.com/DramaCow/jaxued.git \
  "$JAXUED_COMMIT" "$JAXUED_SOURCE"

REQUIREMENTS_HASH=$(sha256sum "$REQUIREMENTS" | awk '{print $1}')
LOCK_STAMP="$WHEELHOUSE/.requirements.sha256"
if [[ -f $LOCK_STAMP ]]; then
  read -r EXISTING_HASH <"$LOCK_STAMP"
  [[ $EXISTING_HASH == "$REQUIREMENTS_HASH" ]] || td2_die \
    "Kinetix wheelhouse belongs to different requirements; choose a new runtime root"
fi
if [[ ! -f $WHEELHOUSE/.download-complete ]]; then
  td2_note "resolving and building a complete pinned Kinetix wheelhouse (login node only)"
  # `pip wheel` is intentional: a few historical dependencies publish only a
  # source archive.  Building them here ensures compute jobs never need a
  # compiler or network access.
  python -m pip wheel --no-cache-dir --wheel-dir "$WHEELHOUSE" \
    --requirement "$REQUIREMENTS"
  (cd -- "$WHEELHOUSE" && sha256sum ./*.whl | LC_ALL=C sort -k2) \
    >"$WHEELHOUSE/SHA256SUMS.tmp"
  mv -f -- "$WHEELHOUSE/SHA256SUMS.tmp" "$WHEELHOUSE/SHA256SUMS"
  printf '%s\n' "$REQUIREMENTS_HASH" >"$LOCK_STAMP.tmp"
  mv -f -- "$LOCK_STAMP.tmp" "$LOCK_STAMP"
  printf 'complete\n' >"$WHEELHOUSE/.download-complete.tmp"
  mv -f -- "$WHEELHOUSE/.download-complete.tmp" "$WHEELHOUSE/.download-complete"
else
  td2_note "verifying completed Kinetix wheelhouse"
  (cd -- "$WHEELHOUSE" && sha256sum --check --quiet SHA256SUMS)
fi

[[ -x $VENV_DIR/bin/python ]] || python -m venv "$VENV_DIR"
td2_note "installing Kinetix runtime strictly from the local wheelhouse"
"$VENV_DIR/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" \
  --requirement "$REQUIREMENTS"
for source in "$JAXUED_SOURCE" "$KINETIX_SOURCE"; do
  "$VENV_DIR/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" \
    --editable --no-deps --no-build-isolation "$source"
done
"$VENV_DIR/bin/python" -m pip check
PYTHONPATH="$REPO_ROOT/src" JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  "$VENV_DIR/bin/python" -m compileall -q "$REPO_ROOT/src/td2_kinetix"
PYTHONPATH="$REPO_ROOT/src" JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  "$VENV_DIR/bin/python" -m td2_kinetix.plan counts >/dev/null
PYTHONPATH="$REPO_ROOT/src" JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib" XDG_CACHE_HOME="$RUNTIME_ROOT/xdg-cache" \
  "$VENV_DIR/bin/python" -c \
  'import numpy as np; from td2_kinetix.ppo import build_components; from td2_kinetix.data import candidate_keys, make_strict_filter; build_components(np.array([[0, 0]], dtype=np.uint32)); verdict = np.asarray(make_strict_filter()(candidate_keys("heldout", None, None, 0, 1))); assert verdict.shape == (1,); print("Kinetix CPU build/filter: OK")'

{
  printf '# Generated by setup_kinetix_compute_canada.sh; do not hand edit.\n'
  printf 'export TD2_KINETIX_REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'export TD2_KINETIX_RUNTIME_ROOT=%q\n' "$RUNTIME_ROOT"
  printf 'export TD2_KINETIX_VENV_DIR=%q\n' "$VENV_DIR"
  printf 'export TD2_KINETIX_WHEELHOUSE=%q\n' "$WHEELHOUSE"
  printf 'export TD2_KINETIX_DATA_ROOT=%q\n' "$DATA_ROOT"
  printf 'export TD2_KINETIX_SOURCE=%q\n' "$KINETIX_SOURCE"
  printf 'export TD2_JAXUED_SOURCE=%q\n' "$JAXUED_SOURCE"
  printf 'export TD2_KINETIX_COMMIT=%q\n' "$KINETIX_COMMIT"
  printf 'export TD2_JAXUED_COMMIT=%q\n' "$JAXUED_COMMIT"
} >"$RUNTIME_ENV.tmp"
mv -f -- "$RUNTIME_ENV.tmp" "$RUNTIME_ENV"
mkdir -p -- "$(dirname -- "$RUNTIME_LOCATOR")"
cp -- "$RUNTIME_ENV" "$RUNTIME_LOCATOR.tmp"
mv -f -- "$RUNTIME_LOCATOR.tmp" "$RUNTIME_LOCATOR"

td2_note "Kinetix setup complete"
printf 'Runtime: %s\nData root: %s\nNext: bash scripts/submit_kinetix_continual.sh\n' \
  "$RUNTIME_ENV" "$DATA_ROOT"
