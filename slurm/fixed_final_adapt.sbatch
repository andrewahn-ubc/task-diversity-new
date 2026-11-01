#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
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

: "${TD2_RUNTIME_ENV:?TD2_RUNTIME_ENV is required}"
: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
: "${TD2_SOURCE_1B_256:?TD2_SOURCE_1B_256 is required}"
: "${TD2_SOURCE_2B_256:?TD2_SOURCE_2B_256 is required}"
: "${TD2_SOURCE_2B_512:?TD2_SOURCE_2B_512 is required}"
: "${TD2_SOURCE_2B_1024:?TD2_SOURCE_2B_1024 is required}"
: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die \
  "final held-out jobs must use def-krallen (got: $TD2_ACCOUNT)"

ARRAY_INDEX=${TD2_ARRAY_INDEX:-${SLURM_ARRAY_TASK_ID:?array index is unavailable}}
td2_require_uint array-index "$ARRAY_INDEX"
((ARRAY_INDEX < 180)) || td2_die "array index must be in [0, 179]"
CONDITION_INDEX=$((ARRAY_INDEX / 45))
STREAM_INDEX=$((ARRAY_INDEX % 45))

case "$CONDITION_INDEX" in
  0)
    HORIZON=1
    GRU_WIDTH=256
    SOURCE_MODE=fixed
    SOURCE_ROOT=$TD2_SOURCE_1B_256
    CONFIG_PATH="$REPO_ROOT/configs/experiment.yaml"
    ;;
  1)
    HORIZON=2
    GRU_WIDTH=256
    SOURCE_MODE=fixed_2b_fresh
    SOURCE_ROOT=$TD2_SOURCE_2B_256
    CONFIG_PATH="$REPO_ROOT/configs/experiment.yaml"
    ;;
  2)
    HORIZON=2
    GRU_WIDTH=512
    SOURCE_MODE=fixed_2b_fresh
    SOURCE_ROOT=$TD2_SOURCE_2B_512
    CONFIG_PATH="$REPO_ROOT/configs/experiment_gru512.yaml"
    ;;
  3)
    HORIZON=2
    GRU_WIDTH=1024
    SOURCE_MODE=fixed_2b_fresh
    SOURCE_ROOT=$TD2_SOURCE_2B_1024
    CONFIG_PATH="$REPO_ROOT/configs/experiment_gru1024.yaml"
    ;;
  *) td2_die "invalid condition index: $CONDITION_INDEX" ;;
esac

td2_preflight_shared_paths_or_requeue \
  "$REPO_ROOT" "$TD2_RUNTIME_ENV" "$CONFIG_PATH" \
  "$TD2_OUTPUT_ROOT" "$TD2_DATA_ROOT" "$SOURCE_ROOT"
td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "${TD2_PROFILE:?TD2_PROFILE is required}"
td2_activate_runtime
td2_install_signal_forwarders
td2_require_command timeout
td2_require_command scontrol

PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$CONFIG_PATH" main --index "$STREAM_INDEX")
IFS=$'\t' read -r ALGORITHM DIVERSITY SEED <<<"$PLAN_ITEM"
[[ $ALGORITHM == ppo_cbp && -n $DIVERSITY && -n $SEED ]] || td2_die \
  "could not decode final held-out stream index $STREAM_INDEX"

SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$SEED"
if [[ $SOURCE_MODE == fixed ]]; then
  SOURCE_STREAM="$SOURCE_ROOT/fixed/$ALGORITHM/diversity_$DIVERSITY/seed_$SEED"
else
  SOURCE_STREAM="$SOURCE_ROOT/fixed_2b_fresh/$ALGORITHM/diversity_$DIVERSITY/seed_$SEED"
fi
td2_preflight_shared_paths_or_requeue \
  "$SEED_DATA_ROOT/manifest.json" "$SEED_DATA_ROOT/manifest.sha256" \
  "$SOURCE_STREAM/DONE" "$SOURCE_STREAM/resume/latest.json"

CONDITION_ROOT="$TD2_OUTPUT_ROOT/runs/${HORIZON}b_gru${GRU_WIDTH}"
STREAM_DIR=$(td2_fixed_adapt_dir \
  "$CONDITION_ROOT" "$ALGORITHM" "$DIVERSITY" "$SEED")
SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
td2_require_uint soft-time-limit-seconds "$SOFT_LIMIT"
((SOFT_LIMIT >= 300 && SOFT_LIMIT <= 2700)) || td2_die \
  "soft time limit must be between 300 and 2700 seconds"

mkdir -p -- "$STREAM_DIR"
td2_note "terminal held-out adaptation: horizon=${HORIZON}B gru=$GRU_WIDTH diversity=$DIVERSITY seed=$SEED index=$ARRAY_INDEX"
if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "terminal held-out branch was already complete"
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
    --mode adapt_fixed \
    --config "$CONFIG_PATH" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$CONDITION_ROOT" \
    --source-output-root "$SOURCE_ROOT" \
    --source-mode "$SOURCE_MODE" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED" \
    --soft-time-limit-seconds "$WATCHDOG_SECONDS" || TRAIN_STATUS=$?
td2_accept_watchdog_exit \
  "$TRAIN_STATUS" "$BATCH_STARTED_EPOCH" "$SOFT_LIMIT"

if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "terminal held-out branch is complete"
  exit 0
fi

td2_note "terminal held-out checkpoint is durable; requeuing exact array task"
td2_requeue_current_job
exit 0
