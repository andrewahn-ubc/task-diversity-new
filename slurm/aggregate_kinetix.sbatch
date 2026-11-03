#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'
REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}
for ATTEMPT in 1 2 3 4 5 6; do [[ -r $REPO_ROOT/scripts/lib/common.sh ]] && break; sleep 10; done
[[ -r $REPO_ROOT/scripts/lib/common.sh ]] || { scontrol requeue "${SLURM_JOB_ID}"; exit 0; }
source "$REPO_ROOT/scripts/lib/common.sh"
source "$REPO_ROOT/scripts/lib/kinetix_common.sh"
td2_enable_transient_filesystem_requeue
: "${TD2_OUTPUT_ROOT:?}" "${TD2_ACCOUNT:?}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die "Kinetix jobs must use def-krallen"
td2_load_kinetix_runtime "$REPO_ROOT"
td2_activate_kinetix_runtime
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
DONE="$TD2_OUTPUT_ROOT/.orchestration/aggregate/DONE"
[[ -f $DONE ]] && { td2_note "Kinetix plots already complete"; exit 0; }
timeout --signal=TERM --kill-after=300s 2700s \
  "$TD2_KINETIX_VENV_DIR/bin/python" -m td2_kinetix.plot --output-root "$TD2_OUTPUT_ROOT"
mkdir -p -- "$(dirname -- "$DONE")"
printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$DONE.tmp.${SLURM_JOB_ID}"
mv -f -- "$DONE.tmp.${SLURM_JOB_ID}" "$DONE"
td2_note "Kinetix training and heldout plots complete"
