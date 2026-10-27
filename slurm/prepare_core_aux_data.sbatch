#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
for ATTEMPT in 1 2 3 4 5 6; do
  [[ -r $REPO_ROOT/scripts/lib/common.sh ]] && break
  sleep 10
done
if [[ ! -r $REPO_ROOT/scripts/lib/common.sh ]]; then
  RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
  [[ $RESTART_COUNT =~ ^[0-9]+$ && $RESTART_COUNT -lt 3000 ]] || exit 1
  TARGET=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
  [[ -n ${SLURM_ARRAY_JOB_ID:-} && -n ${SLURM_ARRAY_TASK_ID:-} ]] && \
    TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  scontrol requeue "$TARGET"
  exit 0
fi
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_enable_transient_filesystem_requeue
TD2_RUNTIME_ENV=${TD2_RUNTIME_ENV:-$REPO_ROOT/.runtime/compute-canada/runtime.env}
TD2_DATA_CONFIG_PATH=${TD2_DATA_CONFIG_PATH:-$REPO_ROOT/configs/core_aux_data.yaml}
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
td2_preflight_shared_paths_or_requeue \
  "$REPO_ROOT" "$TD2_RUNTIME_ENV" "$TD2_DATA_CONFIG_PATH" "$TD2_DATA_ROOT"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

[[ -f $TD2_DATA_CONFIG_PATH ]] || td2_die \
  "core/auxiliary data configuration is unavailable: $TD2_DATA_CONFIG_PATH"

DATA_INDEX=${TD2_DATA_INDEX:-${SLURM_ARRAY_TASK_ID:?data array index is unavailable}}
td2_require_uint data-index "$DATA_INDEX"
DATA_SEED=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$TD2_DATA_CONFIG_PATH" data-seed --index "$DATA_INDEX")
SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$DATA_SEED"

td2_note "preparing immutable core and nested auxiliary task banks for seed $DATA_SEED"
"$TD2_VENV_DIR/bin/python" -m td2.core_aux_data \
  --config "$TD2_DATA_CONFIG_PATH" \
  --output "$SEED_DATA_ROOT" \
  --data-seed "$DATA_SEED"
td2_note "core/auxiliary data preparation finished for seed $DATA_SEED"
