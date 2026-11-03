#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
IFS=$'\n\t'
REPO_ROOT=${TD2_REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}
for ATTEMPT in 1 2 3 4 5 6; do [[ -r $REPO_ROOT/scripts/lib/common.sh ]] && break; sleep 10; done
[[ -r $REPO_ROOT/scripts/lib/common.sh ]] || { scontrol requeue "${SLURM_JOB_ID}"; exit 0; }
source "$REPO_ROOT/scripts/lib/common.sh"
source "$REPO_ROOT/scripts/lib/kinetix_common.sh"
td2_enable_transient_filesystem_requeue
: "${TD2_DATA_ROOT:?}" "${TD2_OUTPUT_ROOT:?}" "${TD2_ACCOUNT:?}"
[[ $TD2_ACCOUNT == def-krallen ]] || td2_die "Kinetix jobs must use def-krallen"
td2_load_kinetix_runtime "$REPO_ROOT"
td2_activate_kinetix_runtime
td2_install_signal_forwarders
"$TD2_KINETIX_VENV_DIR/bin/python" -m pip check
"$TD2_KINETIX_VENV_DIR/bin/python" - <<'PY'
import jax
assert jax.default_backend() == "gpu", jax.default_backend()
print("devices:", jax.devices())
PY
SMOKE_ROOT="$TD2_OUTPUT_ROOT/.smoke/job_${SLURM_JOB_ID:?}"
td2_note "running one production-shaped Kinetix PPO update"
td2_run_checkpointable timeout --preserve-status --signal=USR1 --kill-after=300s 2400s \
  "$TD2_KINETIX_VENV_DIR/bin/python" -m td2_kinetix.train \
    --mode main --data-root "$TD2_DATA_ROOT" --output-root "$SMOKE_ROOT" \
    --diversity 1 --seed 0 --stage 0 --soft-time-limit-seconds 2100 --max-updates 1
[[ -f $SMOKE_ROOT/main/diversity_1/seed_0/stages/stage_0/DONE ]] || td2_die \
  "smoke update did not produce its completion marker"
td2_note "Kinetix GPU smoke gate passed"
