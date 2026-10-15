#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
# shellcheck source=../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
td2_note "all 45 continual streams, 45 fixed streams, and 450 adaptation branches are complete; plotting four graphs"
timeout --signal=TERM --kill-after=240s 2700s \
  "$TD2_VENV_DIR/bin/python" -m td2.plot \
    --config "$REPO_ROOT/configs/experiment.yaml" \
    --output-root "$TD2_OUTPUT_ROOT"

AGGREGATE_DIR="$TD2_OUTPUT_ROOT/.orchestration/aggregate"
mkdir -p -- "$AGGREGATE_DIR"
printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$AGGREGATE_DIR/DONE.tmp.$SLURM_JOB_ID"
mv -f -- "$AGGREGATE_DIR/DONE.tmp.$SLURM_JOB_ID" "$AGGREGATE_DIR/DONE"
td2_note "aggregate plots complete"
