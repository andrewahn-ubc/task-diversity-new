#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

CLUSTER=narval
ACCOUNT=def-krallen
OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp
DATA_ROOT=''
MAX_CONCURRENT=100
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/submit_all.sh [options]

Submits a CPU data-preparation job, a gated one-GPU smoke test, the 45-stream
production array (1 algorithm x 9 diversity levels x 5 seeds), the 450-branch
held-out array, and final aggregation.
Run it on the login node of the cluster named by --cluster.

Options:
  --cluster NAME       Slurm cluster (default: narval)
  --account ACCOUNT    Allocation account (default: def-krallen)
  --output-root PATH   Experiment output
                       (default: /scratch/taegyoem/td2-results-10dist-9div-cbp)
  --data-root PATH     Prepared data (default: value saved by setup)
  --max-concurrent N   Global GPU-array concurrency cap (default: 100; 1-100)
  --resume             Resume checkpoints in an existing output root. Refuses
                       to run while any td2-* jobs are still queued or running.
  -h, --help           Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --cluster)
      (($# >= 2)) || td2_die "--cluster requires a value"
      CLUSTER=$2
      shift 2
      ;;
    --account)
      (($# >= 2)) || td2_die "--account requires a value"
      ACCOUNT=$2
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || td2_die "--output-root requires a path"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --data-root)
      (($# >= 2)) || td2_die "--data-root requires a path"
      DATA_ROOT=$2
      shift 2
      ;;
    --max-concurrent)
      (($# >= 2)) || td2_die "--max-concurrent requires a value"
      MAX_CONCURRENT=$2
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      td2_die "unknown submission option: $1"
      ;;
  esac
done

[[ $ACCOUNT =~ ^[A-Za-z0-9_.-]+$ ]] || td2_die "invalid account name: $ACCOUNT"
td2_require_uint max-concurrent "$MAX_CONCURRENT"
((MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= 100)) || td2_die \
  "--max-concurrent must be between 1 and 100"

td2_require_command sbatch
td2_require_command scontrol
td2_require_command squeue
td2_require_command flock
td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "$CLUSTER"

PLAN_COUNTS=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" counts)
PLAN_IDENTITY=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" identity)
[[ $PLAN_IDENTITY =~ ^[0-9a-f]{64}$ ]] || td2_die \
  "invalid production plan identity: $PLAN_IDENTITY"
IFS=$'\t' read -r MAIN_ARRAY_SIZE ADAPT_ARRAY_SIZE <<<"$PLAN_COUNTS"
td2_require_uint main-array-size "$MAIN_ARRAY_SIZE"
td2_require_uint adaptation-array-size "$ADAPT_ARRAY_SIZE"
((MAIN_ARRAY_SIZE == 45 && ADAPT_ARRAY_SIZE == 450)) || td2_die \
  "unexpected production plan: main=$MAIN_ARRAY_SIZE adaptation=$ADAPT_ARRAY_SIZE"
MAIN_ARRAY_MAX=$((MAIN_ARRAY_SIZE - 1))
ADAPT_ARRAY_MAX=$((ADAPT_ARRAY_SIZE - 1))

CURRENT_CLUSTER=$(scontrol show config 2>/dev/null | awk \
  '$1 == "ClusterName" && !found {name=tolower($3); found=1} END {if (found) print name}')
if [[ -n $CURRENT_CLUSTER && $CURRENT_CLUSTER != "$CLUSTER" ]]; then
  td2_die "connected Slurm cluster is '$CURRENT_CLUSTER', but --cluster requested '$CLUSTER'"
fi

DATA_ROOT=${DATA_ROOT:-$TD2_DATA_ROOT}
mkdir -p -- "$OUTPUT_ROOT" "$DATA_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
DATA_ROOT=$(cd -- "$DATA_ROOT" && pwd -P)
LOG_ROOT="$OUTPUT_ROOT/slurm-logs"
ORCHESTRATION_ROOT="$OUTPUT_ROOT/.orchestration"

# A missing receipt must mean a genuinely fresh result root.  Otherwise an old
# hand-copied or partially deleted experiment could be mixed with this launch
# while bypassing the normal --resume identity checks.
INITIAL_RECEIPT="$ORCHESTRATION_ROOT/initial-submission.txt"
if [[ ! -f $INITIAL_RECEIPT ]]; then
  EXISTING_OUTPUT=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)
  [[ -z $EXISTING_OUTPUT ]] || td2_die \
    "output root is nonempty but has no submission receipt: $OUTPUT_ROOT; choose a fresh path"
fi
mkdir -p -- "$LOG_ROOT" "$ORCHESTRATION_ROOT"

case "$OUTPUT_ROOT$DATA_ROOT$TD2_RUNTIME_ENV$REPO_ROOT" in
  *$'\n'*|*,*|*%*) td2_die "Slurm-exported paths may not contain newlines, commas, or percent signs" ;;
esac
if [[ $OUTPUT_ROOT == "$DATA_ROOT" \
      || $OUTPUT_ROOT == "$DATA_ROOT/"* \
      || $DATA_ROOT == "$OUTPUT_ROOT/"* ]]; then
  td2_die "--output-root and --data-root must not be equal or nested"
fi

# Serialize launch commands even if two shells invoke this script at nearly
# the same instant.  flock releases automatically on every process exit.
exec 9>"$ORCHESTRATION_ROOT/submission.lock"
flock -n 9 || td2_die "another submission command is already running for $OUTPUT_ROOT"

if [[ -f $INITIAL_RECEIPT && $RESUME == false ]]; then
  td2_die "this output root was already submitted ($INITIAL_RECEIPT); use a new --output-root, or --resume only after prior td2 jobs have stopped"
fi
if [[ ! -f $INITIAL_RECEIPT && $RESUME == true ]]; then
  td2_die "--resume requires an existing initial submission receipt: $INITIAL_RECEIPT"
fi
if [[ $RESUME == true ]]; then
  RECEIPT_PLAN_IDENTITY=$(awk -F= \
    '$1 == "plan_identity" {print $2; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die \
    "existing receipt predates this protocol; use a fresh output root"
  [[ $RECEIPT_PLAN_IDENTITY == "$PLAN_IDENTITY" ]] || td2_die \
    "existing output root belongs to a different experiment configuration"
  CURRENT_USER=$(id -un)
  ACTIVE_TD2_JOBS=$(squeue --noheader --user="$CURRENT_USER" --format='%j' | awk '$1 ~ /^td2-/ {print $1}')
  [[ -z $ACTIVE_TD2_JOBS ]] || td2_die \
    "refusing to resume while td2 jobs are active: ${ACTIVE_TD2_JOBS//$'\n'/, }"
  td2_note "resuming only after confirming no td2 jobs are queued or running"

else
  # Write intent before the first sbatch call.  If submission is interrupted
  # halfway through, a normal rerun fails closed and recovery must be explicit.
  td2_write_receipt "$INITIAL_RECEIPT" \
    "state=submitting" \
    "cluster=$CLUSTER" \
    "account=$ACCOUNT" \
    "plan_identity=$PLAN_IDENTITY" \
    "max_concurrent=$MAX_CONCURRENT"
fi

EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$CLUSTER,TD2_ACCOUNT=$ACCOUNT,TD2_OUTPUT_ROOT=$OUTPUT_ROOT,TD2_DATA_ROOT=$DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=2700"
td2_build_gpu_sbatch_args

td2_note "submitting deterministic data preparation"
DATA_SUBMISSION=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-data \
  --output="$LOG_ROOT/data-%j.out" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/prepare_data.sbatch")
DATA_JOB=$(td2_parse_job_id "$DATA_SUBMISSION")

td2_note "submitting GPU smoke gate on $CLUSTER after job $DATA_JOB"
SMOKE_SUBMISSION=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-smoke \
  --output="$LOG_ROOT/smoke-%j.out" \
  --dependency="afterok:$DATA_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/gpu_smoke.sbatch")
SMOKE_JOB=$(td2_parse_job_id "$SMOKE_SUBMISSION")

td2_note "submitting $MAIN_ARRAY_SIZE PPO+CBP production streams after smoke job $SMOKE_JOB"
MAIN_SUBMISSION=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-main \
  --output="$LOG_ROOT/main-%A_%a.out" \
  --array="0-$MAIN_ARRAY_MAX%$MAX_CONCURRENT" \
  --dependency="afterok:$SMOKE_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/main.sbatch")
MAIN_JOB=$(td2_parse_job_id "$MAIN_SUBMISSION")

# Held-out branches start only after all main streams finish.  Both GPU phases
# are single fixed arrays whose tasks requeue themselves from durable
# checkpoints.  Consequently the array throttle is a real experiment-wide GPU
# concurrency bound rather than a limit that standalone continuation jobs can
# bypass.
td2_note "submitting $ADAPT_ARRAY_SIZE held-out adaptation branches after main array $MAIN_JOB"
ADAPT_SUBMISSION=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-adapt \
  --output="$LOG_ROOT/adapt-%A_%a.out" \
  --array="0-$ADAPT_ARRAY_MAX%$MAX_CONCURRENT" \
  --dependency="afterok:$MAIN_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/adapt.sbatch")
ADAPT_JOB=$(td2_parse_job_id "$ADAPT_SUBMISSION")

td2_note "submitting aggregate plots after adaptation array $ADAPT_JOB"
AGGREGATE_SUBMISSION=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-plot \
  --output="$LOG_ROOT/aggregate-%j.out" \
  --dependency="afterok:$ADAPT_JOB" \
  --kill-on-invalid-dep=yes \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/aggregate.sbatch")
AGGREGATE_JOB=$(td2_parse_job_id "$AGGREGATE_SUBMISSION")

if [[ $RESUME == true ]]; then
  RECEIPT="$ORCHESTRATION_ROOT/resume-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
else
  RECEIPT=$INITIAL_RECEIPT
fi
td2_write_receipt "$RECEIPT" \
  "state=scheduled" \
  "mode=$([[ $RESUME == true ]] && printf resume || printf initial)" \
  "cluster=$CLUSTER" \
  "account=$ACCOUNT" \
  "plan_identity=$PLAN_IDENTITY" \
  "max_concurrent=$MAX_CONCURRENT" \
  "data_job=$DATA_JOB" \
  "smoke_job=$SMOKE_JOB" \
  "main_array_job=$MAIN_JOB" \
  "main_array_size=$MAIN_ARRAY_SIZE" \
  "adaptation_array_job=$ADAPT_JOB" \
  "adaptation_array_size=$ADAPT_ARRAY_SIZE" \
  "aggregate_job=$AGGREGATE_JOB" \
  "soft_limit_seconds=2700"

printf '\nSubmitted successfully:\n'
printf '  data preparation: %s\n' "$DATA_JOB"
printf '  GPU smoke gate:   %s\n' "$SMOKE_JOB"
printf '  main array:       %s (%s tasks)\n' "$MAIN_JOB" "$MAIN_ARRAY_SIZE"
printf '  adaptation array: %s (%s tasks)\n' "$ADAPT_JOB" "$ADAPT_ARRAY_SIZE"
printf '  aggregate plots:  %s\n' "$AGGREGATE_JOB"
printf '  receipt:          %s\n' "$RECEIPT"
