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
: "${TD2_ALGORITHM:?TD2_ALGORITHM is required}"
: "${TD2_DIVERSITY:?TD2_DIVERSITY is required}"
: "${TD2_SEED:?TD2_SEED is required}"
: "${TD2_STAGE:?TD2_STAGE is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

[[ $TD2_ALGORITHM =~ ^(ppo|ppo_cbp)$ ]] || td2_die "invalid algorithm: $TD2_ALGORITHM"
[[ $TD2_DIVERSITY =~ ^(1|16|256|4096|65536)$ ]] || td2_die \
  "invalid diversity: $TD2_DIVERSITY"
[[ $TD2_SEED =~ ^[0-4]$ ]] || td2_die "invalid seed: $TD2_SEED"
[[ $TD2_STAGE =~ ^[1-7]$ ]] || td2_die "invalid stage: $TD2_STAGE"
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
  bash "$REPO_ROOT/scripts/maybe_submit_aggregate.sh"
  exit 0
fi

ATTEMPT=${TD2_ATTEMPT:-0}
td2_require_uint attempt "$ATTEMPT"
NEXT_ATTEMPT=$((ATTEMPT + 1))
((NEXT_ATTEMPT <= 500)) || td2_die "continuation limit exceeded for $ADAPT_DIR"
printf -v STAGE_PADDED '%02d' "$TD2_STAGE"
RESUBMIT_LOCK="$TD2_OUTPUT_ROOT/.orchestration/adapt-resubmitted/job_${SLURM_JOB_ID}_${TD2_ALGORITHM}_d${TD2_DIVERSITY}_s${TD2_SEED}_stage${STAGE_PADDED}.lock"
mkdir -p -- "$(dirname -- "$RESUBMIT_LOCK")" "$TD2_OUTPUT_ROOT/slurm-logs"
if ! mkdir -- "$RESUBMIT_LOCK" 2>/dev/null; then
  td2_note "adaptation continuation was already submitted by this job"
  exit 0
fi

td2_build_gpu_sbatch_args
EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$TD2_PROFILE,TD2_ACCOUNT=$TD2_ACCOUNT,TD2_OUTPUT_ROOT=$TD2_OUTPUT_ROOT,TD2_DATA_ROOT=$TD2_DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=$SOFT_LIMIT,TD2_ALGORITHM=$TD2_ALGORITHM,TD2_DIVERSITY=$TD2_DIVERSITY,TD2_SEED=$TD2_SEED,TD2_STAGE=$TD2_STAGE,TD2_ATTEMPT=$NEXT_ATTEMPT"
set +e
SUBMISSION=$(sbatch --parsable \
  --account="$TD2_ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name="td2-a-${TD2_ALGORITHM}-d${TD2_DIVERSITY}-s${TD2_SEED}-t${TD2_STAGE}" \
  --output="$TD2_OUTPUT_ROOT/slurm-logs/adapt-${TD2_ALGORITHM}-d${TD2_DIVERSITY}-s${TD2_SEED}-t${TD2_STAGE}-%j.out" \
  --dependency="afterok:$SLURM_JOB_ID" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/adapt.sbatch")
RC=$?
set -e
if ((RC != 0)); then
  rmdir -- "$RESUBMIT_LOCK"
  td2_die "failed to submit adaptation continuation"
fi
if ! NEXT_JOB=$(td2_parse_job_id "$SUBMISSION"); then
  td2_note "continuation was accepted but its job id could not be recorded: $SUBMISSION"
  exit 0
fi
set +e
td2_write_receipt "$RESUBMIT_LOCK/submission.txt" \
  "job_id=$NEXT_JOB" \
  "parent_job_id=$SLURM_JOB_ID" \
  "attempt=$NEXT_ATTEMPT"
RECEIPT_RC=$?
set -e
if ((RECEIPT_RC != 0)); then
  td2_note "continuation $NEXT_JOB was accepted, but its receipt could not be written"
  exit 0
fi
td2_note "chained adaptation continuation job $NEXT_JOB"
