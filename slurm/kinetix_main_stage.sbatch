#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'
BATCH_STARTED_EPOCH=$(date +%s)
REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
for ATTEMPT in 1 2 3 4 5 6; do [[ -r $REPO_ROOT/scripts/lib/common.sh ]] && break; sleep 10; done
[[ -r $REPO_ROOT/scripts/lib/common.sh ]] || { scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; exit 0; }
source "$REPO_ROOT/scripts/lib/common.sh"
source "$REPO_ROOT/scripts/lib/kinetix_common.sh"
td2_enable_transient_filesystem_requeue
: "${TD2_DATA_ROOT:?}" "${TD2_OUTPUT_ROOT:?}" "${TD2_ACCOUNT:?}" "${TD2_KINETIX_STAGE:?}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die "Kinetix jobs must use def-krallen"
INDEX=${SLURM_ARRAY_TASK_ID:?}
td2_require_uint stream-index "$INDEX"; ((INDEX < 15)) || td2_die "stream index must be [0,14]"
td2_require_uint stage "$TD2_KINETIX_STAGE"; ((TD2_KINETIX_STAGE < 7)) || td2_die "stage must be [0,6]"
td2_load_kinetix_runtime "$REPO_ROOT"
td2_activate_kinetix_runtime
td2_install_signal_forwarders
DIVERSITIES=(1 16 64 256 65536)
DIVERSITY=${DIVERSITIES[$((INDEX / 3))]}
SEED=$((INDEX % 3))
STREAM="$TD2_OUTPUT_ROOT/main/diversity_$DIVERSITY/seed_$SEED/stages/stage_$TD2_KINETIX_STAGE"
[[ -f $STREAM/DONE ]] && { td2_note "main stage already complete"; exit 0; }
SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
ELAPSED=$(( $(date +%s) - BATCH_STARTED_EPOCH ))
WATCHDOG=$((SOFT_LIMIT-ELAPSED))
((WATCHDOG > 180)) || td2_die "insufficient training work budget"
td2_note "Kinetix main stage=$TD2_KINETIX_STAGE diversity=$DIVERSITY seed=$SEED"
STATUS=0
td2_run_checkpointable timeout --preserve-status --signal=USR1 --kill-after=720s "${WATCHDOG}s" \
  "$TD2_KINETIX_VENV_DIR/bin/python" -m td2_kinetix.train \
    --mode main --data-root "$TD2_DATA_ROOT" --output-root "$TD2_OUTPUT_ROOT" \
    --diversity "$DIVERSITY" --seed "$SEED" --stage "$TD2_KINETIX_STAGE" \
    --soft-time-limit-seconds "$WATCHDOG" || STATUS=$?
if ((STATUS != 0 && STATUS != 75)); then
  td2_accept_watchdog_exit "$STATUS" "$BATCH_STARTED_EPOCH" "$SOFT_LIMIT"
fi
[[ -f $STREAM/DONE ]] && { td2_note "main stage complete"; exit 0; }
td2_note "main checkpoint durable; requeuing exact task"
td2_requeue_current_job
