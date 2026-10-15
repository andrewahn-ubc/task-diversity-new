#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --requeue

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
DATA_INDEX=${TD2_DATA_INDEX:-${SLURM_ARRAY_TASK_ID:?data array index is unavailable}}
td2_require_uint data-index "$DATA_INDEX"
DATA_SEED=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" data-seed --index "$DATA_INDEX")
SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$DATA_SEED"
td2_note "preparing immutable task, layout, distractor, and heldout manifests for seed $DATA_SEED"
"$TD2_VENV_DIR/bin/python" -m td2.prepare_data \
  --config "$REPO_ROOT/configs/experiment.yaml" \
  --output "$SEED_DATA_ROOT" \
  --seed "$DATA_SEED"
td2_note "data preparation finished for seed $DATA_SEED"
