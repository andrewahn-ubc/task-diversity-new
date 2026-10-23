#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

CLUSTER=narval
ACCOUNT=def-krallen
OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b
DATA_ROOT=''
MAX_CONCURRENT=100
EXCLUDE_NODES=''
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/submit_all.sh [options]

Submits all data, smoke, shared-prefix, continual, fixed-distribution,
held-out, and plotting work. Every job uses the def-krallen allocation and GPU
array concurrency never exceeds --max-concurrent.

Options:
  --cluster NAME       Slurm cluster (default: narval)
  --output-root PATH   Experiment output
                       (default: /scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b)
  --data-root PATH     Prepared data (default: value saved by setup)
  --max-concurrent N   GPU array concurrency cap (default: 100; 1-100)
  --exclude-nodes LIST Comma-separated Slurm nodes to avoid for every job
                       in this submission (default: none)
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
    --exclude-nodes)
      (($# >= 2)) || td2_die "--exclude-nodes requires a value"
      EXCLUDE_NODES=$2
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

td2_require_uint max-concurrent "$MAX_CONCURRENT"
((MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= 100)) || td2_die \
  "--max-concurrent must be between 1 and 100"
if [[ -n $EXCLUDE_NODES ]]; then
  [[ $EXCLUDE_NODES =~ ^[A-Za-z0-9._,-]+$ ]] || td2_die \
    "invalid --exclude-nodes list: $EXCLUDE_NODES"
fi
TD2_EXCLUDE_SBATCH_ARGS=()
if [[ -n $EXCLUDE_NODES ]]; then
  TD2_EXCLUDE_SBATCH_ARGS=("--exclude=$EXCLUDE_NODES")
fi

td2_require_command sbatch
td2_require_command scontrol
td2_require_command squeue
td2_require_command flock
td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "$CLUSTER"
# Use the just-pulled source tree with the already-pinned runtime dependencies.
export PYTHONPATH="$REPO_ROOT/src"

PLAN_COUNTS=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" counts)
DATA_ARRAY_SIZE=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" data-count)
CONTINUATION_ARRAY_SIZE=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" continuation-count)
PLAN_IDENTITY=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" identity)
[[ $PLAN_IDENTITY =~ ^[0-9a-f]{64}$ ]] || td2_die \
  "invalid production plan identity: $PLAN_IDENTITY"
IFS=$'\t' read -r MAIN_ARRAY_SIZE ADAPT_ARRAY_SIZE <<<"$PLAN_COUNTS"
td2_require_uint main-array-size "$MAIN_ARRAY_SIZE"
td2_require_uint continuation-array-size "$CONTINUATION_ARRAY_SIZE"
td2_require_uint adaptation-array-size "$ADAPT_ARRAY_SIZE"
td2_require_uint data-array-size "$DATA_ARRAY_SIZE"
((MAIN_ARRAY_SIZE == 45 && CONTINUATION_ARRAY_SIZE == 90 \
  && ADAPT_ARRAY_SIZE == 450 && DATA_ARRAY_SIZE == 5)) || td2_die \
  "unexpected production plan: data=$DATA_ARRAY_SIZE main=$MAIN_ARRAY_SIZE continuation=$CONTINUATION_ARRAY_SIZE adaptation=$ADAPT_ARRAY_SIZE"

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
  td2_write_receipt "$INITIAL_RECEIPT" \
    "state=submitting" \
    "cluster=$CLUSTER" \
    "account=$ACCOUNT" \
    "plan_identity=$PLAN_IDENTITY" \
    "max_concurrent=$MAX_CONCURRENT"
fi

BASE_EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$CLUSTER,TD2_OUTPUT_ROOT=$OUTPUT_ROOT,TD2_DATA_ROOT=$DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=2700"
EXPORTS="$BASE_EXPORTS,TD2_ACCOUNT=$ACCOUNT"
td2_build_gpu_sbatch_args

td2_note "submitting deterministic data bundles under $ACCOUNT"
DATA_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-data \
  --output="$LOG_ROOT/data-%A_%a.out" \
  --array="0-4" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/prepare_data.sbatch")
DATA_JOB=$(td2_parse_job_id "$DATA_SUBMISSION")

td2_note "submitting GPU smoke gate after data array $DATA_JOB"
SMOKE_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
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

td2_note "submitting 45 shared 100M prefixes after smoke job $SMOKE_JOB"
PREFIX_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-prefix \
  --output="$LOG_ROOT/prefix-%A_%a.out" \
  --array="0-44%$MAX_CONCURRENT" \
  --dependency="afterok:$SMOKE_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/prefix.sbatch")
PREFIX_JOB=$(td2_parse_job_id "$PREFIX_SUBMISSION")

td2_note "submitting 45 continual and 45 fixed-distribution continuations"
CONT_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-cont \
  --output="$LOG_ROOT/continuation-%A_%a.out" \
  --array="0-89%$MAX_CONCURRENT" \
  --dependency="afterok:$PREFIX_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/continuation.sbatch")
CONT_JOB=$(td2_parse_job_id "$CONT_SUBMISSION")

td2_note "submitting 450 held-out branches after the continuation array"
ADAPT_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-adapt \
  --output="$LOG_ROOT/adapt-%A_%a.out" \
  --array="0-449%$MAX_CONCURRENT" \
  --dependency="afterok:$CONT_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/adapt.sbatch")
ADAPT_JOB=$(td2_parse_job_id "$ADAPT_SUBMISSION")

td2_note "submitting four-graph aggregation after the adaptation array"
AGGREGATE_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
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
  "exclude_nodes=${EXCLUDE_NODES:-none}" \
  "plan_identity=$PLAN_IDENTITY" \
  "max_concurrent=$MAX_CONCURRENT" \
  "data_job=$DATA_JOB" \
  "data_array_size=$DATA_ARRAY_SIZE" \
  "smoke_job=$SMOKE_JOB" \
  "prefix_job=$PREFIX_JOB" \
  "prefix_array_size=$MAIN_ARRAY_SIZE" \
  "continuation_job=$CONT_JOB" \
  "continuation_array_size=$CONTINUATION_ARRAY_SIZE" \
  "adaptation_job=$ADAPT_JOB" \
  "adaptation_array_size=$ADAPT_ARRAY_SIZE" \
  "aggregate_job=$AGGREGATE_JOB" \
  "soft_limit_seconds=2700"

printf '\nSubmitted successfully:\n'
printf '  data array:           %s (%s seeds)\n' "$DATA_JOB" "$DATA_ARRAY_SIZE"
printf '  GPU smoke gate:       %s\n' "$SMOKE_JOB"
printf '  shared-prefix array:  %s (%s streams)\n' "$PREFIX_JOB" "$MAIN_ARRAY_SIZE"
printf '  continuation array:   %s (%s tasks)\n' "$CONT_JOB" "$CONTINUATION_ARRAY_SIZE"
printf '  adaptation array:     %s (%s tasks)\n' "$ADAPT_JOB" "$ADAPT_ARRAY_SIZE"
printf '  aggregate plots:      %s\n' "$AGGREGATE_JOB"
printf '  receipt:              %s\n' "$RECEIPT"
