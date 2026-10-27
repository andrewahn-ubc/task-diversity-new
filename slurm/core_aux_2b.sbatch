#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'
BATCH_STARTED_EPOCH=$(date +%s)

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
TD2_CONFIG_PATH=${TD2_CONFIG_PATH:-$REPO_ROOT/configs/core_aux_2b.yaml}
: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
td2_preflight_shared_paths_or_requeue \
  "$REPO_ROOT" "$TD2_RUNTIME_ENV" "$TD2_CONFIG_PATH" \
  "$TD2_OUTPUT_ROOT" "$TD2_DATA_ROOT"
td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "${TD2_PROFILE:?TD2_PROFILE is required}"
td2_activate_runtime
td2_install_signal_forwarders
td2_require_command timeout
td2_require_command scontrol

: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die \
  "core/auxiliary jobs must use def-krallen (got: $TD2_ACCOUNT)"
[[ -f $TD2_CONFIG_PATH ]] || td2_die \
  "core/auxiliary training configuration is unavailable: $TD2_CONFIG_PATH"

STREAM_INDEX=${TD2_STREAM_INDEX:-${SLURM_ARRAY_TASK_ID:?array index is unavailable}}
td2_require_uint stream-index "$STREAM_INDEX"
PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$TD2_CONFIG_PATH" main \
  --index "$STREAM_INDEX")
IFS=$'\t' read -r ALGORITHM DIVERSITY SEED <<<"$PLAN_ITEM"
[[ -n $ALGORITHM && -n $DIVERSITY && -n $SEED ]] || td2_die \
  "could not decode core/auxiliary stream index $STREAM_INDEX"

SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$SEED"
STREAM_DIR="$TD2_OUTPUT_ROOT/core_aux_2b/$ALGORITHM/aux_diversity_$DIVERSITY/seed_$SEED"
td2_preflight_shared_paths_or_requeue \
  "$SEED_DATA_ROOT/manifest.json" "$SEED_DATA_ROOT/manifest.sha256"
[[ -f $SEED_DATA_ROOT/manifest.json \
   && -f $SEED_DATA_ROOT/manifest.sha256 ]] || td2_die \
  "prepared core/auxiliary task bundle is unavailable: $SEED_DATA_ROOT"

SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
td2_require_uint soft-time-limit-seconds "$SOFT_LIMIT"
((SOFT_LIMIT >= 300 && SOFT_LIMIT <= 2700)) || td2_die \
  "soft time limit must be between 300 and 2700 seconds"

mkdir -p -- "$STREAM_DIR"
td2_note "core/auxiliary 2B run: algorithm=$ALGORITHM auxiliary_diversity=$DIVERSITY seed=$SEED index=$STREAM_INDEX"

if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "core/auxiliary 2B run was already complete"
  exit 0
fi

ELAPSED_SECONDS=$(( $(date +%s) - BATCH_STARTED_EPOCH ))
WATCHDOG_SECONDS=$((SOFT_LIMIT - ELAPSED_SECONDS))
((WATCHDOG_SECONDS > 180)) || td2_die \
  "less than 181 seconds remain in the batch work budget before training"

TRAIN_STATUS=0
td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=720s "${WATCHDOG_SECONDS}s" \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode core_aux_2b \
    --config "$TD2_CONFIG_PATH" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$TD2_OUTPUT_ROOT" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED" \
    --soft-time-limit-seconds "$WATCHDOG_SECONDS" || TRAIN_STATUS=$?
td2_accept_watchdog_exit \
  "$TRAIN_STATUS" "$BATCH_STARTED_EPOCH" "$SOFT_LIMIT"

if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "core/auxiliary 2B run is complete"
  exit 0
fi

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
td2_require_uint restart-count "$RESTART_COUNT"
((RESTART_COUNT < 3000)) || td2_die "requeue limit exceeded for $STREAM_DIR"
td2_note "core/auxiliary checkpoint is durable; requeuing array task (restart $((RESTART_COUNT + 1)))"
REQUEUE_TARGET=${SLURM_JOB_ID}
if [[ -n ${SLURM_ARRAY_JOB_ID:-} && -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
scontrol requeue "$REQUEUE_TARGET"
exit 0
