#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@600

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu

: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
td2_note "preparing immutable task, layout, distractor, and heldout manifests"
timeout --signal=TERM --kill-after=240s 5400s \
  "$TD2_VENV_DIR/bin/python" -m td2.prepare_data \
    --config "$REPO_ROOT/configs/experiment.yaml" \
    --output "$TD2_DATA_ROOT"
td2_note "data preparation finished"
