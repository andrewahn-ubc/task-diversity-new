#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
: "${TD2_PROFILE:?TD2_PROFILE is required}"
: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"

td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "$TD2_PROFILE"
td2_require_command sbatch

ALGORITHMS=(ppo ppo_cbp)
DIVERSITIES=(1 16 256 4096 65536)
MAIN_COMPLETE=0
ADAPT_COMPLETE=0

for algorithm in "${ALGORITHMS[@]}"; do
  for diversity in "${DIVERSITIES[@]}"; do
    for seed in 0 1 2 3 4; do
      STREAM_DIR=$(td2_stream_dir "$TD2_OUTPUT_ROOT" "$algorithm" "$diversity" "$seed")
      [[ -f $STREAM_DIR/DONE ]] && ((MAIN_COMPLETE += 1))
      for stage in 1 2 3 4 5 6 7; do
        ADAPT_DIR=$(td2_adapt_dir \
          "$TD2_OUTPUT_ROOT" "$algorithm" "$diversity" "$seed" "$stage")
        [[ -f $ADAPT_DIR/DONE ]] && ((ADAPT_COMPLETE += 1))
      done
    done
  done
done

td2_note "completion gate: main=$MAIN_COMPLETE/50 adaptation=$ADAPT_COMPLETE/350"
((MAIN_COMPLETE == 50 && ADAPT_COMPLETE == 350)) || exit 0

LOCK_DIR="$TD2_OUTPUT_ROOT/.orchestration/aggregate-submitted.lock"
mkdir -p -- "$(dirname -- "$LOCK_DIR")" "$TD2_OUTPUT_ROOT/slurm-logs"
if ! mkdir -- "$LOCK_DIR" 2>/dev/null; then
  td2_note "aggregate job was already submitted"
  exit 0
fi

EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$TD2_PROFILE,TD2_ACCOUNT=$TD2_ACCOUNT,TD2_OUTPUT_ROOT=$TD2_OUTPUT_ROOT,TD2_DATA_ROOT=$TD2_DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=2700"

set +e
SUBMISSION=$(sbatch --parsable \
  --account="$TD2_ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-plot \
  --output="$TD2_OUTPUT_ROOT/slurm-logs/aggregate-%j.out" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/aggregate.sbatch")
RC=$?
set -e
if ((RC != 0)); then
  rmdir -- "$LOCK_DIR"
  td2_die "failed to submit aggregate plotting job"
fi
if ! JOB_ID=$(td2_parse_job_id "$SUBMISSION"); then
  td2_note "aggregate was accepted but its job id could not be recorded: $SUBMISSION"
  exit 0
fi
set +e
td2_write_receipt "$LOCK_DIR/submission.txt" \
  "job_id=$JOB_ID" \
  "main_complete=50" \
  "adaptation_complete=350"
RECEIPT_RC=$?
set -e
if ((RECEIPT_RC != 0)); then
  td2_note "aggregate $JOB_ID was accepted, but its receipt could not be written"
  exit 0
fi
td2_note "submitted final aggregate job $JOB_ID"
