#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

ALGORITHM=''
DIVERSITY=''
SEED=''

while (($#)); do
  case "$1" in
    --algorithm)
      (($# >= 2)) || td2_die "--algorithm requires a value"
      ALGORITHM=$2
      shift 2
      ;;
    --diversity)
      (($# >= 2)) || td2_die "--diversity requires a value"
      DIVERSITY=$2
      shift 2
      ;;
    --seed)
      (($# >= 2)) || td2_die "--seed requires a value"
      SEED=$2
      shift 2
      ;;
    *) td2_die "unknown adaptation-submission option: $1" ;;
  esac
done

[[ $ALGORITHM =~ ^(ppo|ppo_cbp)$ ]] || td2_die "invalid algorithm: $ALGORITHM"
[[ $DIVERSITY =~ ^(1|16|256|4096|65536)$ ]] || td2_die "invalid diversity: $DIVERSITY"
[[ $SEED =~ ^[0-4]$ ]] || td2_die "invalid seed: $SEED"

: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
: "${TD2_PROFILE:?TD2_PROFILE is required}"
: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"

td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "$TD2_PROFILE"
td2_require_command sbatch

STREAM_DIR=$(td2_stream_dir "$TD2_OUTPUT_ROOT" "$ALGORITHM" "$DIVERSITY" "$SEED")
LOCK_ROOT="$TD2_OUTPUT_ROOT/.orchestration/adapt-submitted"
LOG_ROOT="$TD2_OUTPUT_ROOT/slurm-logs"
mkdir -p -- "$LOCK_ROOT" "$LOG_ROOT"
td2_build_gpu_sbatch_args

for stage in 1 2 3 4 5 6 7; do
  printf -v STAGE_PADDED '%02d' "$stage"
  STAGE_DONE="$STREAM_DIR/stages/stage_$STAGE_PADDED/DONE"
  [[ -f $STAGE_DONE ]] || continue

  ADAPT_DIR=$(td2_adapt_dir \
    "$TD2_OUTPUT_ROOT" "$ALGORITHM" "$DIVERSITY" "$SEED" "$stage")
  [[ -f $ADAPT_DIR/DONE ]] && continue

  LOCK_DIR="$LOCK_ROOT/${ALGORITHM}__d${DIVERSITY}__s${SEED}__stage${STAGE_PADDED}.lock"
  if ! mkdir -- "$LOCK_DIR" 2>/dev/null; then
    continue
  fi

  EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$TD2_PROFILE,TD2_ACCOUNT=$TD2_ACCOUNT,TD2_OUTPUT_ROOT=$TD2_OUTPUT_ROOT,TD2_DATA_ROOT=$TD2_DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=${TD2_SOFT_LIMIT_SECONDS:-2700},TD2_ALGORITHM=$ALGORITHM,TD2_DIVERSITY=$DIVERSITY,TD2_SEED=$SEED,TD2_STAGE=$stage,TD2_ATTEMPT=0"
  td2_note "submitting adaptation: $ALGORITHM diversity=$DIVERSITY seed=$SEED stage=$stage"
  set +e
  SUBMISSION=$(sbatch --parsable \
    --account="$TD2_ACCOUNT" \
    --chdir="$REPO_ROOT" \
    --job-name="td2-a-${ALGORITHM}-d${DIVERSITY}-s${SEED}-t${stage}" \
    --output="$LOG_ROOT/adapt-${ALGORITHM}-d${DIVERSITY}-s${SEED}-t${stage}-%j.out" \
    "${TD2_GPU_SBATCH_ARGS[@]}" \
    --export="$EXPORTS" \
    "$REPO_ROOT/slurm/adapt.sbatch")
  RC=$?
  set -e
  if ((RC != 0)); then
    rmdir -- "$LOCK_DIR"
    td2_die "sbatch failed for adaptation stage $stage"
  fi
  if ! JOB_ID=$(td2_parse_job_id "$SUBMISSION"); then
    td2_note "adaptation was accepted but its job id could not be recorded: $SUBMISSION"
    continue
  fi
  set +e
  td2_write_receipt "$LOCK_DIR/submission.txt" \
    "job_id=$JOB_ID" \
    "algorithm=$ALGORITHM" \
    "diversity=$DIVERSITY" \
    "seed=$SEED" \
    "stage=$stage" \
    "checkpoint_ready_marker=$STAGE_DONE"
  RECEIPT_RC=$?
  set -e
  if ((RECEIPT_RC != 0)); then
    td2_note "adaptation $JOB_ID was accepted, but its receipt could not be written"
  fi
done
