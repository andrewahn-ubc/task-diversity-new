#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
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

if [[ -n ${TD2_ADAPT_INDEX:-} ]]; then
  ADAPT_INDEX=$TD2_ADAPT_INDEX
else
  ADAPT_INDEX=${SLURM_ARRAY_TASK_ID:?adaptation array index is unavailable}
fi
td2_require_uint adaptation-index "$ADAPT_INDEX"
PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" adapt --index "$ADAPT_INDEX")
IFS=$'\t' read -r TD2_ALGORITHM TD2_DIVERSITY TD2_SEED TD2_STAGE \
  <<<"$PLAN_ITEM"
[[ -n $TD2_ALGORITHM && -n $TD2_DIVERSITY && -n $TD2_SEED \
   && -n $TD2_STAGE ]] || td2_die \
  "could not decode adaptation index $ADAPT_INDEX"
SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
td2_require_uint soft-time-limit-seconds "$SOFT_LIMIT"
((SOFT_LIMIT >= 300 && SOFT_LIMIT <= 2700)) || td2_die \
  "soft time limit must be between 300 and 2700 seconds"

ADAPT_DIR=$(td2_adapt_dir \
  "$TD2_OUTPUT_ROOT" "$TD2_ALGORITHM" "$TD2_DIVERSITY" "$TD2_SEED" "$TD2_STAGE")
mkdir -p -- "$ADAPT_DIR"
td2_note "adaptation: algorithm=$TD2_ALGORITHM diversity=$TD2_DIVERSITY seed=$TD2_SEED stage=$TD2_STAGE"

if [[ -f $ADAPT_DIR/DONE ]]; then
  td2_note "adaptation branch was already complete"
  exit 0
fi

ELAPSED_SECONDS=$(( $(date +%s) - BATCH_STARTED_EPOCH ))
WATCHDOG_SECONDS=$((SOFT_LIMIT - ELAPSED_SECONDS))
((WATCHDOG_SECONDS > 180)) || td2_die \
  "less than 181 seconds remain in the batch work budget before training"

td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=240s "${WATCHDOG_SECONDS}s" \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode adapt \
    --config "$REPO_ROOT/configs/experiment.yaml" \
    --data-root "$TD2_DATA_ROOT" \
    --output-root "$TD2_OUTPUT_ROOT" \
    --algorithm "$TD2_ALGORITHM" \
    --diversity "$TD2_DIVERSITY" \
    --seed "$TD2_SEED" \
    --stage "$TD2_STAGE" \
    --soft-time-limit-seconds "$WATCHDOG_SECONDS"

if [[ -f $ADAPT_DIR/DONE ]]; then
  td2_note "adaptation branch is complete"
  exit 0
fi

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
td2_require_uint restart-count "$RESTART_COUNT"
((RESTART_COUNT < 500)) || td2_die "requeue limit exceeded for $ADAPT_DIR"
td2_note "adaptation checkpoint is durable; requeuing array task (restart $((RESTART_COUNT + 1)))"
REQUEUE_TARGET=${SLURM_JOB_ID}
if [[ -n ${SLURM_ARRAY_JOB_ID:-} && -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
scontrol requeue "$REQUEUE_TARGET"
exit 0
