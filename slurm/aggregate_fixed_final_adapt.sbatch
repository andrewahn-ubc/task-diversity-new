#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
for ATTEMPT in 1 2 3 4 5 6; do
  [[ -r $REPO_ROOT/scripts/lib/common.sh ]] && break
  sleep 10
done
if [[ ! -r $REPO_ROOT/scripts/lib/common.sh ]]; then
  RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
  [[ $RESTART_COUNT =~ ^[0-9]+$ && $RESTART_COUNT -lt 3000 ]] || exit 1
  scontrol requeue "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
  exit 0
fi
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_enable_transient_filesystem_requeue

: "${TD2_RUNTIME_ENV:?TD2_RUNTIME_ENV is required}"
: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_ACCOUNT:?TD2_ACCOUNT is required}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die \
  "final held-out aggregate must use def-krallen (got: $TD2_ACCOUNT)"
CONFIG_PATH="$REPO_ROOT/configs/experiment.yaml"
ROOT_1B_256="$TD2_OUTPUT_ROOT/runs/1b_gru256"
ROOT_2B_256="$TD2_OUTPUT_ROOT/runs/2b_gru256"
ROOT_2B_512="$TD2_OUTPUT_ROOT/runs/2b_gru512"
ROOT_2B_1024="$TD2_OUTPUT_ROOT/runs/2b_gru1024"

td2_preflight_shared_paths_or_requeue \
  "$REPO_ROOT" "$TD2_RUNTIME_ENV" "$CONFIG_PATH" "$TD2_OUTPUT_ROOT" \
  "$ROOT_1B_256" "$ROOT_2B_256" "$ROOT_2B_512" "$ROOT_2B_1024"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

AGGREGATE_DIR="$TD2_OUTPUT_ROOT/.orchestration/fixed-final-adaptation-aggregate"
if [[ -f $AGGREGATE_DIR/DONE ]]; then
  td2_note "terminal held-out comparison plots were already complete"
  exit 0
fi

td2_note "all 180 branches are complete; creating four held-out comparison figures"
timeout --signal=TERM --kill-after=240s 2700s \
  "$TD2_VENV_DIR/bin/python" -m td2.plot \
    --config "$CONFIG_PATH" \
    --output-root "$TD2_OUTPUT_ROOT" \
    --fixed-final-adaptation-comparison \
    --fixed-adaptation-run 1 256 "$ROOT_1B_256" \
    --fixed-adaptation-run 2 256 "$ROOT_2B_256" \
    --fixed-adaptation-run 2 512 "$ROOT_2B_512" \
    --fixed-adaptation-run 2 1024 "$ROOT_2B_1024"

mkdir -p -- "$AGGREGATE_DIR"
printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$AGGREGATE_DIR/DONE.tmp.${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
mv -f -- "$AGGREGATE_DIR/DONE.tmp.$SLURM_JOB_ID" "$AGGREGATE_DIR/DONE"
td2_note "terminal held-out comparison plots complete"
