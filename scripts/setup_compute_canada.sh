#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

BANYAN_COMMIT=7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3
PUREJAXRL_COMMIT=31756b197773a52db763fdbe6d635e4b46522a73
RUNTIME_ROOT=/scratch/taegyoem/td2-runtime
DATA_ROOT=/scratch/taegyoem/td2-data-10dist-9div-cbp

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_compute_canada.sh [options]

Builds a project-local, offline-capable Python environment on a Narval or
Rorqual login node.  Login-node network access is used only by this command.

Options:
  --runtime-root PATH  Durable venv/wheel/source directory
                       (default: /scratch/taegyoem/td2-runtime)
  --data-root PATH     Durable generated-data directory
                       (default: /scratch/taegyoem/td2-data-10dist-9div-cbp)
  -h, --help           Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --runtime-root)
      (($# >= 2)) || td2_die "--runtime-root requires a path"
      RUNTIME_ROOT=$2
      shift 2
      ;;
    --data-root)
      (($# >= 2)) || td2_die "--data-root requires a path"
      DATA_ROOT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      td2_die "unknown setup option: $1"
      ;;
  esac
done

td2_require_command git
td2_require_command module
td2_require_command sha256sum

mkdir -p -- "$RUNTIME_ROOT"
RUNTIME_ROOT=$(cd -- "$RUNTIME_ROOT" && pwd -P)
mkdir -p -- "$DATA_ROOT"
DATA_ROOT=$(cd -- "$DATA_ROOT" && pwd -P)

case "$RUNTIME_ROOT$DATA_ROOT$REPO_ROOT" in
  *$'\n'*|*,*|*%*) td2_die "runtime paths may not contain newlines, commas, or percent signs" ;;
esac

WHEELHOUSE="$RUNTIME_ROOT/wheelhouse-py313"
VENV_DIR="$RUNTIME_ROOT/venv-py313"
SOURCES_DIR="$RUNTIME_ROOT/sources"
RUNTIME_ENV="$RUNTIME_ROOT/runtime.env"
RUNTIME_LOCATOR="$REPO_ROOT/.runtime/compute-canada/runtime.env"
REQUIREMENTS="$SCRIPT_DIR/requirements-compute-canada.txt"
mkdir -p -- "$WHEELHOUSE" "$SOURCES_DIR"

td2_note "loading the shared cluster toolchain"
module --force purge
# A caller-exported CUDA/cuDNN path is not necessarily owned by Lmod and can
# survive a purge. Clear it before rebuilding the standard module environment
# so it cannot shadow the user-space CUDA libraries bundled with JAX's wheels.
unset LD_LIBRARY_PATH
module load StdEnv/2023
module load python/3.13.2
# Setup must resolve and validate against the selected Alliance module, not a
# caller's personal Python/JAX environment.
unset PYTHONPATH PYTHONHOME XLA_FLAGS JAX_PLATFORMS JAX_PLATFORM_NAME
unset JAX_DISABLE_JIT JAX_ENABLE_X64 JAX_DEBUG_NANS JAX_DEBUG_INFS
unset CUDA_HOME CUDA_PATH
export PYTHONNOUSERSITE=1

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
[[ $PYTHON_VERSION == 3.13.2 ]] || td2_die \
  "python/3.13.2 module resolved to Python $PYTHON_VERSION"

clone_exact() {
  local name=$1 url=$2 commit=$3 destination=$4 actual dirty_status
  if [[ -e $destination && ! -d $destination/.git ]]; then
    td2_die "$destination exists but is not a git checkout"
  fi
  if [[ ! -d $destination/.git ]]; then
    td2_note "cloning $name at its audited revision"
    git clone --filter=blob:none --no-checkout "$url" "$destination"
  else
    # Refuse before checkout so setup never moves HEAD underneath local changes.
    dirty_status=$(git -C "$destination" status \
      --porcelain=v1 --untracked-files=all --ignore-submodules=none)
    [[ -z $dirty_status ]] || td2_die \
      "$name checkout at $destination has tracked or untracked changes; use a clean runtime root or restore the pinned checkout"
  fi
  if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$destination" fetch --filter=blob:none origin "$commit"
  fi
  git -C "$destination" checkout --detach "$commit"
  actual=$(git -C "$destination" rev-parse HEAD)
  [[ $actual == "$commit" ]] || td2_die "$name checkout mismatch: $actual"
  dirty_status=$(git -C "$destination" status \
    --porcelain=v1 --untracked-files=all --ignore-submodules=none)
  [[ -z $dirty_status ]] || td2_die \
    "$name checkout at $destination has tracked or untracked changes; use a clean runtime root or restore the pinned checkout"
}

BANYAN_SOURCE="$SOURCES_DIR/banyan-$BANYAN_COMMIT"
PUREJAXRL_SOURCE="$SOURCES_DIR/purejaxrl-$PUREJAXRL_COMMIT"
clone_exact Banyan https://github.com/nhshah15/banyan.git \
  "$BANYAN_COMMIT" "$BANYAN_SOURCE"
clone_exact PureJaxRL https://github.com/luchris429/purejaxrl.git \
  "$PUREJAXRL_COMMIT" "$PUREJAXRL_SOURCE"

REQUIREMENTS_HASH=$(sha256sum "$REQUIREMENTS" | awk '{print $1}')
LOCK_STAMP="$WHEELHOUSE/.requirements.sha256"
if [[ -f $LOCK_STAMP ]]; then
  read -r EXISTING_HASH <"$LOCK_STAMP"
  [[ $EXISTING_HASH == "$REQUIREMENTS_HASH" ]] || td2_die \
    "wheelhouse belongs to a different requirements file; choose a new --runtime-root"
fi

if [[ ! -f $WHEELHOUSE/.download-complete ]]; then
  td2_note "downloading the complete CUDA 12 wheel set (login node only)"
  python -m pip download \
    --no-cache-dir \
    --only-binary=:all: \
    --dest "$WHEELHOUSE" \
    --requirement "$REQUIREMENTS"

  (
    cd -- "$WHEELHOUSE"
    sha256sum ./*.whl | LC_ALL=C sort -k2
  ) >"$WHEELHOUSE/SHA256SUMS.tmp"
  mv -f -- "$WHEELHOUSE/SHA256SUMS.tmp" "$WHEELHOUSE/SHA256SUMS"
  printf '%s\n' "$REQUIREMENTS_HASH" >"$LOCK_STAMP.tmp"
  mv -f -- "$LOCK_STAMP.tmp" "$LOCK_STAMP"
  printf 'complete\n' >"$WHEELHOUSE/.download-complete.tmp"
  mv -f -- "$WHEELHOUSE/.download-complete.tmp" "$WHEELHOUSE/.download-complete"
else
  td2_note "reusing the completed project wheelhouse"
  (
    cd -- "$WHEELHOUSE"
    sha256sum --check --quiet SHA256SUMS
  )
fi

if [[ ! -x $VENV_DIR/bin/python ]]; then
  td2_note "creating Python environment at $VENV_DIR"
  python -m venv "$VENV_DIR"
fi

td2_note "installing strictly from the local wheelhouse"
"$VENV_DIR/bin/python" -m pip install \
  --no-cache-dir \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --requirement "$REQUIREMENTS"

# Banyan and this project are copied into site-packages as normal, non-editable
# installs. Queued jobs therefore execute the setup-time snapshot even if a
# source checkout is edited later. Dependencies came from the immutable
# wheelhouse above; PureJaxRL is retained at its exact revision for provenance.
"$VENV_DIR/bin/python" -m pip install \
  --no-cache-dir \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --no-deps \
  --no-build-isolation \
  "$BANYAN_SOURCE"
"$VENV_DIR/bin/python" -m pip install \
  --no-cache-dir \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --no-deps \
  --no-build-isolation \
  "$REPO_ROOT"

"$VENV_DIR/bin/python" -m pip check
PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" -m compileall -q \
  "$BANYAN_SOURCE/banyan_grid" "$REPO_ROOT/src/td2"
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/check_runtime.py"
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  "$VENV_DIR/bin/python" -m td2.train --help >/dev/null
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  "$VENV_DIR/bin/python" -m td2.prepare_data --help >/dev/null
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  "$VENV_DIR/bin/python" -m td2.plan \
    --config "$REPO_ROOT/configs/experiment.yaml" counts >/dev/null
PYTHONNOUSERSITE=1 MPLBACKEND=Agg \
  "$VENV_DIR/bin/python" -m td2.plot --help >/dev/null

FREEZE_FILE="$RUNTIME_ROOT/pip-freeze.txt"
"$VENV_DIR/bin/python" -m pip freeze --all >"$FREEZE_FILE.tmp"
mv -f -- "$FREEZE_FILE.tmp" "$FREEZE_FILE"

td2_note "writing relocatable job settings"
{
  printf '# Generated by scripts/setup_compute_canada.sh; do not hand edit.\n'
  printf 'export TD2_REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'export TD2_RUNTIME_ROOT=%q\n' "$RUNTIME_ROOT"
  printf 'export TD2_VENV_DIR=%q\n' "$VENV_DIR"
  printf 'export TD2_WHEELHOUSE=%q\n' "$WHEELHOUSE"
  printf 'export TD2_DATA_ROOT=%q\n' "$DATA_ROOT"
  printf 'export TD2_BANYAN_SOURCE=%q\n' "$BANYAN_SOURCE"
  printf 'export TD2_PUREJAXRL_SOURCE=%q\n' "$PUREJAXRL_SOURCE"
  printf 'export TD2_BANYAN_COMMIT=%q\n' "$BANYAN_COMMIT"
  printf 'export TD2_PUREJAXRL_COMMIT=%q\n' "$PUREJAXRL_COMMIT"
} >"$RUNTIME_ENV.tmp"
mv -f -- "$RUNTIME_ENV.tmp" "$RUNTIME_ENV"

# Keep the normal submission command independent of where the large runtime
# lives.  The locator is a tiny copy of the generated exports; wheel and venv
# payloads remain under --runtime-root (normally /scratch or /project).
if [[ $RUNTIME_LOCATOR != "$RUNTIME_ENV" ]]; then
  mkdir -p -- "$(dirname -- "$RUNTIME_LOCATOR")"
  cp -- "$RUNTIME_ENV" "$RUNTIME_LOCATOR.tmp"
  mv -f -- "$RUNTIME_LOCATOR.tmp" "$RUNTIME_LOCATOR"
fi

td2_note "setup complete"
printf 'Runtime: %s\n' "$RUNTIME_ENV"
printf 'Runtime locator: %s\n' "$RUNTIME_LOCATOR"
printf 'Next: bash scripts/submit_all.sh\n'
