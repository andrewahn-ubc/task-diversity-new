#!/usr/bin/env bash
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'
REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
for ATTEMPT in 1 2 3 4 5 6; do [[ -r $REPO_ROOT/scripts/lib/common.sh ]] && break; sleep 10; done
[[ -r $REPO_ROOT/scripts/lib/common.sh ]] || { scontrol requeue "${SLURM_JOB_ID}"; exit 0; }
source "$REPO_ROOT/scripts/lib/common.sh"
source "$REPO_ROOT/scripts/lib/kinetix_common.sh"
td2_enable_transient_filesystem_requeue
: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die "Kinetix jobs must use def-krallen"
td2_load_kinetix_runtime "$REPO_ROOT"
td2_activate_kinetix_runtime

td2_note "checking the isolated Kinetix CUDA/cuDNN runtime before data fan-out"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader
"$TD2_KINETIX_VENV_DIR/bin/python" -m pip check
"$TD2_KINETIX_VENV_DIR/bin/python" - <<'PY'
from importlib import metadata
import math

import jax
import jax.numpy as jnp
import jaxlib

packages = (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
)
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "MISSING"
    print(f"package {package}={version}", flush=True)

print(f"jax={jax.__version__} jaxlib={jaxlib.__version__}", flush=True)
print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
if jax.default_backend() != "gpu":
    raise RuntimeError(f"expected a GPU backend, got {jax.default_backend()!r}")


@jax.jit
def cudnn_convolution(lhs, rhs):
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


lhs = jnp.ones((2, 32, 32, 8), dtype=jnp.float32)
rhs = jnp.ones((3, 3, 8, 16), dtype=jnp.float32)
result = cudnn_convolution(lhs, rhs)
result.block_until_ready()
checksum = float(jnp.sum(result))
if result.shape != (2, 32, 32, 16) or not math.isfinite(checksum):
    raise RuntimeError(
        f"invalid cuDNN convolution result: shape={result.shape}, checksum={checksum}"
    )
print(
    f"cuDNN convolution: OK shape={result.shape} checksum={checksum}",
    flush=True,
)
PY
td2_note "Kinetix CUDA/cuDNN preflight passed"
