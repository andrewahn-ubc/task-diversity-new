#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
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
  scontrol requeue "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
  exit 0
fi
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_enable_transient_filesystem_requeue
TD2_RUNTIME_ENV=${TD2_RUNTIME_ENV:-$REPO_ROOT/.runtime/compute-canada/runtime.env}
TD2_CONFIG_PATH=${TD2_CONFIG_PATH:-$REPO_ROOT/configs/core_aux_2b.yaml}
: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
td2_preflight_shared_paths_or_requeue \
  "$REPO_ROOT" "$TD2_RUNTIME_ENV" "$TD2_CONFIG_PATH" "$TD2_OUTPUT_ROOT"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
[[ -f $TD2_CONFIG_PATH ]] || td2_die \
  "core/auxiliary training configuration is unavailable: $TD2_CONFIG_PATH"

AGGREGATE_DIR="$TD2_OUTPUT_ROOT/.orchestration/core-aux-2b-aggregate"
if [[ -f $AGGREGATE_DIR/DONE ]]; then
  td2_note "core/auxiliary 2B plots were already complete"
  exit 0
fi

td2_note "all 25 core/auxiliary 2B streams are complete; plotting the results"
timeout --signal=TERM --kill-after=240s 2700s \
  "$TD2_VENV_DIR/bin/python" -m td2.plot \
    --config "$TD2_CONFIG_PATH" \
    --output-root "$TD2_OUTPUT_ROOT" \
    --core-aux-2b-only

mkdir -p -- "$AGGREGATE_DIR"
printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$AGGREGATE_DIR/DONE.tmp.$SLURM_JOB_ID"
mv -f -- "$AGGREGATE_DIR/DONE.tmp.$SLURM_JOB_ID" "$AGGREGATE_DIR/DONE"
td2_note "core/auxiliary 2B plots complete"
