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
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}" "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die "Kinetix jobs must use def-krallen"
td2_load_kinetix_runtime "$REPO_ROOT"
td2_activate_kinetix_runtime
td2_install_signal_forwarders
INDEX=${SLURM_ARRAY_TASK_ID:?array index is unavailable}
td2_require_uint data-index "$INDEX"
((INDEX < 22)) || td2_die "Kinetix data index must be in [0,21]"
SOFT_LIMIT=${TD2_SOFT_LIMIT_SECONDS:-2700}
ELAPSED=$(( $(date +%s) - BATCH_STARTED_EPOCH ))
WATCHDOG=$((SOFT_LIMIT - ELAPSED))
((WATCHDOG > 180)) || td2_die "insufficient data-generation work budget"
td2_note "preparing filtered Kinetix level-key bank index $INDEX"
if ((INDEX == 21)); then
  BANK="$TD2_DATA_ROOT/heldout"
  GENERATE_ARGS=(--kind heldout)
else
  SEED=$((INDEX / 7))
  DISTRIBUTION=$((INDEX % 7))
  BANK="$TD2_DATA_ROOT/seed_$SEED/distribution_$DISTRIBUTION"
  GENERATE_ARGS=(--kind train --seed "$SEED" --distribution "$DISTRIBUTION")
fi
STATUS=0
td2_run_checkpointable timeout --preserve-status --signal=USR1 --kill-after=600s "${WATCHDOG}s" \
  "$TD2_KINETIX_VENV_DIR/bin/python" -m td2_kinetix.data generate \
    --data-root "$TD2_DATA_ROOT" "${GENERATE_ARGS[@]}" \
    --soft-limit-seconds "$WATCHDOG" || STATUS=$?
if ((STATUS != 0 && STATUS != 75)); then
  td2_accept_watchdog_exit "$STATUS" "$BATCH_STARTED_EPOCH" "$SOFT_LIMIT"
fi
if [[ -f $BANK/DONE ]]; then
  td2_note "Kinetix data bank $INDEX complete"
  exit 0
fi
td2_note "filtered key-bank checkpoint is durable; requeuing exact task"
td2_requeue_current_job
