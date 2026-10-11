#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600

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
((STREAM_INDEX < 50)) || td2_die "stream index must be in [0, 49]"

ALGORITHMS=(ppo ppo_cbp)
DIVERSITIES=(1 16 256 4096 65536)
SEED=$((STREAM_INDEX % 5))
QUOTIENT=$((STREAM_INDEX / 5))
DIVERSITY_INDEX=$((QUOTIENT % 5))
ALGORITHM_INDEX=$((QUOTIENT / 5))
ALGORITHM=${ALGORITHMS[$ALGORITHM_INDEX]}
DIVERSITY=${DIVERSITIES[$DIVERSITY_INDEX]}
SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
td2_require_uint soft-time-limit-seconds "$SOFT_LIMIT"
((SOFT_LIMIT >= 300 && SOFT_LIMIT <= 2700)) || td2_die \
  "soft time limit must be between 300 and 2700 seconds"

STREAM_DIR=$(td2_stream_dir "$TD2_OUTPUT_ROOT" "$ALGORITHM" "$DIVERSITY" "$SEED")
mkdir -p -- "$STREAM_DIR"
td2_note "main stream: algorithm=$ALGORITHM diversity=$DIVERSITY seed=$SEED index=$STREAM_INDEX"

# Resume submissions may include already-finished streams.  Avoid rebuilding a
# production JAX program on a GPU merely to rediscover the durable marker; the
# helper still repairs any missing adaptation submissions from completed stages.
if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "main stream was already complete"
  bash "$REPO_ROOT/scripts/submit_pending_adaptations.sh" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED"
  bash "$REPO_ROOT/scripts/maybe_submit_aggregate.sh"
  exit 0
fi

ELAPSED_SECONDS=$(( $(date +%s) - BATCH_STARTED_EPOCH ))
WATCHDOG_SECONDS=$((SOFT_LIMIT - ELAPSED_SECONDS))
((WATCHDOG_SECONDS > 180)) || td2_die \
  "less than 181 seconds remain in the batch work budget before training"

td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=240s "${WATCHDOG_SECONDS}s" \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode main \
    --config "$REPO_ROOT/configs/experiment.yaml" \
    --data-root "$TD2_DATA_ROOT" \
    --output-root "$TD2_OUTPUT_ROOT" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED" \
    --soft-time-limit-seconds "$WATCHDOG_SECONDS"

# A stage DONE marker is written only after its immutable checkpoint and
# checksum are durable.  This helper uses an atomic mkdir lock per branch, so
# repeated continuation jobs can never submit the same adaptation twice.
bash "$REPO_ROOT/scripts/submit_pending_adaptations.sh" \
  --algorithm "$ALGORITHM" \
  --diversity "$DIVERSITY" \
  --seed "$SEED"

if [[ -f $STREAM_DIR/DONE ]]; then
  td2_note "main stream is complete"
  bash "$REPO_ROOT/scripts/maybe_submit_aggregate.sh"
  exit 0
fi

# Python returned successfully but the stream is incomplete, which is its
# contract that an atomic checkpoint is ready.  Chain one continuation only
# after this job itself exits successfully.
ATTEMPT=${TD2_ATTEMPT:-0}
td2_require_uint attempt "$ATTEMPT"
NEXT_ATTEMPT=$((ATTEMPT + 1))
((NEXT_ATTEMPT <= 3000)) || td2_die "continuation limit exceeded for $STREAM_DIR"
RESUBMIT_LOCK="$TD2_OUTPUT_ROOT/.orchestration/main-resubmitted/job_${SLURM_JOB_ID}_stream_${STREAM_INDEX}.lock"
mkdir -p -- "$(dirname -- "$RESUBMIT_LOCK")" "$TD2_OUTPUT_ROOT/slurm-logs"
if ! mkdir -- "$RESUBMIT_LOCK" 2>/dev/null; then
  td2_note "continuation was already submitted by this job"
  exit 0
fi

td2_build_gpu_sbatch_args
EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$TD2_PROFILE,TD2_ACCOUNT=$TD2_ACCOUNT,TD2_OUTPUT_ROOT=$TD2_OUTPUT_ROOT,TD2_DATA_ROOT=$TD2_DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=$SOFT_LIMIT,TD2_STREAM_INDEX=$STREAM_INDEX,TD2_ATTEMPT=$NEXT_ATTEMPT"
set +e
SUBMISSION=$(sbatch --parsable \
  --account="$TD2_ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name="td2-m-${ALGORITHM}-d${DIVERSITY}-s${SEED}" \
  --output="$TD2_OUTPUT_ROOT/slurm-logs/main-${ALGORITHM}-d${DIVERSITY}-s${SEED}-%j.out" \
  --dependency="afterok:$SLURM_JOB_ID" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/main.sbatch")
RC=$?
set -e
if ((RC != 0)); then
  rmdir -- "$RESUBMIT_LOCK"
  td2_die "failed to submit main continuation"
fi
if ! NEXT_JOB=$(td2_parse_job_id "$SUBMISSION"); then
  td2_note "continuation was accepted but its job id could not be recorded: $SUBMISSION"
  exit 0
fi
set +e
td2_write_receipt "$RESUBMIT_LOCK/submission.txt" \
  "job_id=$NEXT_JOB" \
  "parent_job_id=$SLURM_JOB_ID" \
  "stream_index=$STREAM_INDEX" \
  "attempt=$NEXT_ATTEMPT"
RECEIPT_RC=$?
set -e
if ((RECEIPT_RC != 0)); then
  td2_note "continuation $NEXT_JOB was accepted, but its receipt could not be written"
  exit 0
fi
td2_note "chained main continuation job $NEXT_JOB"
