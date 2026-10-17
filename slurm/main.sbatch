#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'
BATCH_STARTED_EPOCH=$(date +%s)

REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "${TD2_PROFILE:?TD2_PROFILE is required}"
td2_activate_runtime
td2_install_signal_forwarders
td2_require_command timeout
td2_require_command scontrol

: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

if [[ -n ${TD2_STREAM_INDEX:-} ]]; then
  STREAM_INDEX=$TD2_STREAM_INDEX
else
  STREAM_INDEX=${SLURM_ARRAY_TASK_ID:?array index is unavailable}
fi
td2_require_uint stream-index "$STREAM_INDEX"
PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" main --index "$STREAM_INDEX")
IFS=$'\t' read -r ALGORITHM DIVERSITY SEED <<<"$PLAN_ITEM"
[[ -n $ALGORITHM && -n $DIVERSITY && -n $SEED ]] || td2_die \
  "could not decode main stream index $STREAM_INDEX"
SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$SEED"
SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
td2_require_uint soft-time-limit-seconds "$SOFT_LIMIT"
((SOFT_LIMIT >= 300 && SOFT_LIMIT <= 2700)) || td2_die \
  "soft time limit must be between 300 and 2700 seconds"

STREAM_DIR=$(td2_stream_dir "$TD2_OUTPUT_ROOT" "$ALGORITHM" "$DIVERSITY" "$SEED")
mkdir -p -- "$STREAM_DIR"
td2_note "main stream: algorithm=$ALGORITHM diversity=$DIVERSITY seed=$SEED index=$STREAM_INDEX"

# Resume submissions may include already-finished streams. Avoid rebuilding a
# production JAX program on a GPU merely to rediscover the durable marker.
if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "main stream was already complete"
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
    --mode main \
    --config "$REPO_ROOT/configs/experiment.yaml" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$TD2_OUTPUT_ROOT" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED" \
    --soft-time-limit-seconds "$WATCHDOG_SECONDS" || TRAIN_STATUS=$?
td2_accept_watchdog_exit \
  "$TRAIN_STATUS" "$STREAM_DIR/resume" "$BATCH_STARTED_EPOCH" "$SOFT_LIMIT"

if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "main stream is complete"
  exit 0
fi

# Python returned successfully but the stream is incomplete, which is its
# contract that an atomic checkpoint is ready. Requeue this exact array task so
# it remains governed by the original array concurrency throttle.
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
td2_require_uint restart-count "$RESTART_COUNT"
((RESTART_COUNT < 3000)) || td2_die "requeue limit exceeded for $STREAM_DIR"
td2_note "main checkpoint is durable; requeuing array task (restart $((RESTART_COUNT + 1)))"
REQUEUE_TARGET=${SLURM_JOB_ID}
if [[ -n ${SLURM_ARRAY_JOB_ID:-} && -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
scontrol requeue "$REQUEUE_TARGET"
exit 0
