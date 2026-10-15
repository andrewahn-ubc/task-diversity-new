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
td2_install_signal_forwarders
td2_require_command timeout

: "${TD2_OUTPUT_ROOT:?TD2_OUTPUT_ROOT is required}"
: "${TD2_DATA_ROOT:?TD2_DATA_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
DATA_SEED=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" data-seed --index 0)
SEED_DATA_ROOT="$TD2_DATA_ROOT/seed_$DATA_SEED"

td2_note "checking pinned packages, CUDA libraries, GPU execution, and XLA"
"$TD2_VENV_DIR/bin/python" "$REPO_ROOT/scripts/check_runtime.py" --require-gpu
"$TD2_VENV_DIR/bin/python" -m pip check

SMOKE_ROOT="$TD2_OUTPUT_ROOT/.smoke/job_$SLURM_JOB_ID"
mkdir -p -- "$SMOKE_ROOT"
td2_note "compiling and running one production-shaped recurrent PPO+CBP update"
td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=240s 2100s \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode main \
    --config "$REPO_ROOT/configs/gpu_smoke.yaml" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$SMOKE_ROOT" \
    --algorithm ppo_cbp \
    --diversity 1 \
    --seed 0 \
    --soft-time-limit-seconds 1800

SMOKE_STREAM="$SMOKE_ROOT/ppo_cbp/diversity_1/seed_0"
SMOKE_RESUME="$SMOKE_STREAM/resume/latest.json"
SMOKE_STAGE="$SMOKE_STREAM/stages/stage_01/DONE"
[[ -f $SMOKE_RESUME ]] || td2_die \
  "production-shaped update returned without a durable resume pointer: $SMOKE_RESUME"

td2_note "reloading the production-shaped checkpoint and publishing its stage marker"
td2_run_checkpointable \
  timeout --preserve-status --signal=USR1 --kill-after=120s 300s \
  "$TD2_VENV_DIR/bin/python" -m td2.train \
    --mode main \
    --config "$REPO_ROOT/configs/gpu_smoke.yaml" \
    --data-root "$SEED_DATA_ROOT" \
    --output-root "$SMOKE_ROOT" \
    --algorithm ppo_cbp \
    --diversity 1 \
    --seed 0 \
    --soft-time-limit-seconds 240

[[ -f $SMOKE_STAGE ]] || td2_die \
  "checkpoint reload did not preserve the stage marker: $SMOKE_STAGE"
td2_note "GPU smoke gate passed"
