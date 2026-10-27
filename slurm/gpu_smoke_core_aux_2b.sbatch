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
TD2_RUNTIME_ENV=${TD2_RUNTIME_ENV:-$REPO_ROOT/.runtime/compute-canada/runtime.env}
TD2_CONFIG_PATH=${TD2_CONFIG_PATH:-$REPO_ROOT/configs/core_aux_2b.yaml}
: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
td2_preflight_shared_paths_or_requeue \
  "$REPO_ROOT" "$TD2_RUNTIME_ENV" "$TD2_CONFIG_PATH" \
  "$TD2_OUTPUT_ROOT" "$TD2_DATA_ROOT"
td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
td2_install_signal_forwarders
td2_require_command timeout

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
[[ -f $TD2_CONFIG_PATH ]] || td2_die \
  "core/auxiliary training configuration is unavailable: $TD2_CONFIG_PATH"

PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$TD2_CONFIG_PATH" main --index 0)
IFS=$'\t' read -r ALGORITHM DIVERSITY SEED <<<"$PLAN_ITEM"
[[ -n $ALGORITHM && -n $DIVERSITY && -n $SEED ]] || td2_die \
  "could not decode the first core/auxiliary stream"
SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$SEED"
td2_preflight_shared_paths_or_requeue \
  "$SEED_DATA_ROOT/manifest.json" "$SEED_DATA_ROOT/manifest.sha256"
[[ -f $SEED_DATA_ROOT/manifest.json \
   && -f $SEED_DATA_ROOT/manifest.sha256 ]] || td2_die \
  "prepared core/auxiliary task bundle is unavailable: $SEED_DATA_ROOT"

td2_note "checking pinned packages, CUDA libraries, GPU execution, and XLA"
"$TD2_VENV_DIR/bin/python" "$REPO_ROOT/scripts/check_runtime.py" --require-gpu
"$TD2_VENV_DIR/bin/python" -m pip check

SMOKE_ROOT="$TD2_OUTPUT_ROOT/.smoke/core_aux_2b/job_$SLURM_JOB_ID"
SMOKE_STREAM="$SMOKE_ROOT/core_aux_2b/$ALGORITHM/aux_diversity_$DIVERSITY/seed_$SEED"
SMOKE_RESUME="$SMOKE_STREAM/resume/latest.json"
mkdir -p -- "$SMOKE_ROOT"

td2_note "compiling and running one production-shaped core/auxiliary PPO+CBP update"
td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=240s 2100s \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode core_aux_2b \
    --config "$TD2_CONFIG_PATH" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$SMOKE_ROOT" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED" \
    --soft-time-limit-seconds 1800 \
    --max-updates 1

[[ -f $SMOKE_RESUME ]] || td2_die \
  "production-shaped update returned without a durable resume pointer: $SMOKE_RESUME"
FIRST_UPDATE=$("$TD2_VENV_DIR/bin/python" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["metadata"]["global_update"]))' \
  "$SMOKE_RESUME") || td2_die "cannot read the first smoke checkpoint"

td2_note "reloading the production-shaped checkpoint and running one more update"
td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=120s 300s \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode core_aux_2b \
    --config "$TD2_CONFIG_PATH" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$SMOKE_ROOT" \
    --algorithm "$ALGORITHM" \
    --diversity "$DIVERSITY" \
    --seed "$SEED" \
    --soft-time-limit-seconds 240 \
    --max-updates 1

[[ -f $SMOKE_RESUME ]] || td2_die \
  "checkpoint reload removed the durable resume pointer: $SMOKE_RESUME"
SECOND_UPDATE=$("$TD2_VENV_DIR/bin/python" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["metadata"]["global_update"]))' \
  "$SMOKE_RESUME") || td2_die "cannot read the reloaded smoke checkpoint"
((SECOND_UPDATE > FIRST_UPDATE)) || td2_die \
  "checkpoint reload did not advance training ($FIRST_UPDATE -> $SECOND_UPDATE)"
td2_note "core/auxiliary GPU smoke gate passed"
