#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

CLUSTER=narval
ACCOUNT=def-krallen
OUTPUT_ROOT=/scratch/taegyoem/td2-results-core-aux-16sq-50-50-gru256-2b
DATA_ROOT=/scratch/taegyoem/td2-data-core-aux-16sq-50-50-seeded
CONFIG_PATH="$REPO_ROOT/configs/core_aux_2b.yaml"
DATA_CONFIG_PATH="$REPO_ROOT/configs/core_aux_data.yaml"
MAX_CONCURRENT=25
EXCLUDE_NODES=ng31004,ng31104
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/submit_core_aux_2b.sh [options]

Submits the five seed-specific core/auxiliary data bundles, a protocol-shaped
GPU smoke gate, 25 PPO+CBP training streams, and aggregate plots on Narval.
Every job uses the def-krallen allocation. Training jobs checkpoint and requeue
within one-hour allocations.

Options:
  --output-root PATH         Core/auxiliary experiment output
  --data-root PATH           Core/auxiliary immutable data bundles
  --config PATH              Training configuration
                             (default: configs/core_aux_2b.yaml)
  --data-config PATH         Data-generation configuration
                             (default: configs/core_aux_data.yaml)
  --max-concurrent N         Training-array concurrency cap
                             (default: 25; 1-25)
  --exclude-nodes LIST       Comma-separated Slurm nodes to avoid
                             (default: ng31004,ng31104)
  --resume                   Resume an interrupted output root. Refuses to run
                             while any td2-* jobs are queued or running.
  -h, --help                 Show this help
USAGE
}

while (($#)); do
  case "$1" in
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
    --config)
      (($# >= 2)) || td2_die "--config requires a path"
      CONFIG_PATH=$2
      shift 2
      ;;
    --data-config)
      (($# >= 2)) || td2_die "--data-config requires a path"
      DATA_CONFIG_PATH=$2
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
((MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= 25)) || td2_die \
  "--max-concurrent must be between 1 and 25"
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
# Use the current checkout with the existing pinned binary dependencies.
export PYTHONPATH="$REPO_ROOT/src"

[[ -f $CONFIG_PATH ]] || td2_die \
  "core/auxiliary training configuration does not exist: $CONFIG_PATH"
[[ -f $DATA_CONFIG_PATH ]] || td2_die \
  "core/auxiliary data configuration does not exist: $DATA_CONFIG_PATH"
CONFIG_PATH=$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd -P)/$(basename -- "$CONFIG_PATH")
DATA_CONFIG_PATH=$(cd -- "$(dirname -- "$DATA_CONFIG_PATH")" && pwd -P)/$(basename -- "$DATA_CONFIG_PATH")

PLAN_COUNTS=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$CONFIG_PATH" counts)
TRAIN_DATA_ARRAY_SIZE=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$CONFIG_PATH" data-count)
DATA_ARRAY_SIZE=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$DATA_CONFIG_PATH" data-count)
PLAN_IDENTITY=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$CONFIG_PATH" identity)
DATA_PLAN_IDENTITY=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$DATA_CONFIG_PATH" identity)
[[ $PLAN_IDENTITY =~ ^[0-9a-f]{64}$ ]] || td2_die \
  "invalid core/auxiliary plan identity: $PLAN_IDENTITY"
[[ $DATA_PLAN_IDENTITY =~ ^[0-9a-f]{64}$ ]] || td2_die \
  "invalid core/auxiliary data identity: $DATA_PLAN_IDENTITY"
IFS=$'\t' read -r STREAM_ARRAY_SIZE ADAPT_ARRAY_SIZE <<<"$PLAN_COUNTS"
td2_require_uint stream-array-size "$STREAM_ARRAY_SIZE"
td2_require_uint adaptation-array-size "$ADAPT_ARRAY_SIZE"
td2_require_uint data-array-size "$DATA_ARRAY_SIZE"
td2_require_uint training-data-array-size "$TRAIN_DATA_ARRAY_SIZE"
((STREAM_ARRAY_SIZE == 25 && ADAPT_ARRAY_SIZE == 0 \
  && DATA_ARRAY_SIZE == 5 && TRAIN_DATA_ARRAY_SIZE == 5)) || td2_die \
  "unexpected core/auxiliary plan: data=$DATA_ARRAY_SIZE training_data=$TRAIN_DATA_ARRAY_SIZE streams=$STREAM_ARRAY_SIZE adaptation=$ADAPT_ARRAY_SIZE"

SEEN_STREAMS=$'\n'
for ((INDEX = 0; INDEX < STREAM_ARRAY_SIZE; INDEX++)); do
  PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
    --config "$CONFIG_PATH" main --index "$INDEX")
  IFS=$'\t' read -r ALGORITHM DIVERSITY SEED <<<"$PLAN_ITEM"
  [[ $ALGORITHM == ppo_cbp ]] || td2_die \
    "core/auxiliary plan contains unsupported algorithm at index $INDEX: $ALGORITHM"
  case "$DIVERSITY" in
    256|1024|4096|16384|65536) ;;
    *) td2_die \
      "core/auxiliary plan contains unexpected auxiliary diversity at index $INDEX: $DIVERSITY" ;;
  esac
  case "$SEED" in
    0|1|2|3|4) ;;
    *) td2_die "core/auxiliary plan contains unexpected seed at index $INDEX: $SEED" ;;
  esac
  STREAM_KEY="$DIVERSITY:$SEED"
  case "$SEEN_STREAMS" in
    *$'\n'"$STREAM_KEY"$'\n'*) td2_die \
      "core/auxiliary plan contains duplicate stream: $STREAM_KEY" ;;
  esac
  SEEN_STREAMS+="$STREAM_KEY"$'\n'
done
for DIVERSITY in 256 1024 4096 16384 65536; do
  for SEED in 0 1 2 3 4; do
    case "$SEEN_STREAMS" in
      *$'\n'"$DIVERSITY:$SEED"$'\n'*) ;;
      *) td2_die \
        "core/auxiliary plan is missing stream: auxiliary_diversity=$DIVERSITY seed=$SEED" ;;
    esac
  done
done

for ((INDEX = 0; INDEX < DATA_ARRAY_SIZE; INDEX++)); do
  DATA_SEED=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
    --config "$DATA_CONFIG_PATH" data-seed --index "$INDEX")
  TRAIN_SEED=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
    --config "$CONFIG_PATH" data-seed --index "$INDEX")
  [[ $DATA_SEED == "$TRAIN_SEED" ]] || td2_die \
    "training and data configurations disagree at data index $INDEX: $TRAIN_SEED != $DATA_SEED"
done

CURRENT_CLUSTER=$(scontrol show config 2>/dev/null | awk \
  '$1 == "ClusterName" && !found {name=tolower($3); found=1} END {if (found) print name}')
if [[ -n $CURRENT_CLUSTER && $CURRENT_CLUSTER != "$CLUSTER" ]]; then
  td2_die "connected Slurm cluster is '$CURRENT_CLUSTER', but this workflow requires Narval"
fi

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

case "$OUTPUT_ROOT$DATA_ROOT$TD2_RUNTIME_ENV$REPO_ROOT$CONFIG_PATH$DATA_CONFIG_PATH" in
  *$'\n'*|*,*|*%*) td2_die "Slurm-exported paths may not contain newlines, commas, or percent signs" ;;
esac
if [[ $OUTPUT_ROOT == "$DATA_ROOT" \
      || $OUTPUT_ROOT == "$DATA_ROOT/"* \
      || $DATA_ROOT == "$OUTPUT_ROOT/"* ]]; then
  td2_die "--output-root and --data-root must be distinct and not nested"
fi

exec 9>"$ORCHESTRATION_ROOT/submission.lock"
flock -n 9 || td2_die "another submission command is already running for $OUTPUT_ROOT"

if [[ -f $INITIAL_RECEIPT && $RESUME == false ]]; then
  td2_die "this core/auxiliary output was already submitted ($INITIAL_RECEIPT); use a fresh --output-root, or --resume after prior td2 jobs have stopped"
fi
if [[ ! -f $INITIAL_RECEIPT && $RESUME == true ]]; then
  td2_die "--resume requires an existing initial submission receipt: $INITIAL_RECEIPT"
fi
if [[ $RESUME == true ]]; then
  RECEIPT_PLAN_IDENTITY=$(awk -F= \
    '$1 == "plan_identity" {print $2; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no training-plan identity"
  RECEIPT_DATA_PLAN_IDENTITY=$(awk -F= \
    '$1 == "data_plan_identity" {print $2; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no data-plan identity"
  RECEIPT_DATA_ROOT=$(awk -F= \
    '$1 == "data_root" {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no data root"
  RECEIPT_CONFIG_PATH=$(awk -F= \
    '$1 == "config_path" {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no training configuration path"
  RECEIPT_DATA_CONFIG_PATH=$(awk -F= \
    '$1 == "data_config_path" {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no data configuration path"
  [[ $RECEIPT_PLAN_IDENTITY == "$PLAN_IDENTITY" ]] || td2_die \
    "existing output belongs to a different training configuration"
  [[ $RECEIPT_DATA_PLAN_IDENTITY == "$DATA_PLAN_IDENTITY" ]] || td2_die \
    "existing output belongs to a different data configuration"
  [[ $RECEIPT_DATA_ROOT == "$DATA_ROOT" ]] || td2_die \
    "existing output uses a different data root"
  [[ $RECEIPT_CONFIG_PATH == "$CONFIG_PATH" ]] || td2_die \
    "existing output uses a different training configuration path"
  [[ $RECEIPT_DATA_CONFIG_PATH == "$DATA_CONFIG_PATH" ]] || td2_die \
    "existing output uses a different data configuration path"
  CURRENT_USER=$(id -un)
  ACTIVE_TD2_JOBS=$(squeue --noheader --user="$CURRENT_USER" --format='%j' | awk '$1 ~ /^td2-/ {print $1}')
  [[ -z $ACTIVE_TD2_JOBS ]] || td2_die \
    "refusing to resume while td2 jobs are active: ${ACTIVE_TD2_JOBS//$'\n'/, }"
  td2_note "resuming only after confirming no td2 jobs are queued or running"
else
  td2_write_receipt "$INITIAL_RECEIPT" \
    "state=submitting" \
    "mode=initial" \
    "workflow=core_aux_2b" \
    "cluster=$CLUSTER" \
    "account=$ACCOUNT" \
    "config_path=$CONFIG_PATH" \
    "data_config_path=$DATA_CONFIG_PATH" \
    "data_root=$DATA_ROOT" \
    "plan_identity=$PLAN_IDENTITY" \
    "data_plan_identity=$DATA_PLAN_IDENTITY" \
    "max_concurrent=$MAX_CONCURRENT"
fi

BASE_EXPORTS="ALL,TD2_REPO_ROOT=$REPO_ROOT,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$CLUSTER,TD2_OUTPUT_ROOT=$OUTPUT_ROOT,TD2_DATA_ROOT=$DATA_ROOT,TD2_CONFIG_PATH=$CONFIG_PATH,TD2_DATA_CONFIG_PATH=$DATA_CONFIG_PATH,TD2_SOFT_LIMIT_SECONDS=2700"
EXPORTS="$BASE_EXPORTS,TD2_ACCOUNT=$ACCOUNT"
td2_build_gpu_sbatch_args

td2_note "submitting $DATA_ARRAY_SIZE deterministic core/auxiliary data bundles"
DATA_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir=/tmp \
  --job-name=td2-coreaux-data \
  --output="$LOG_ROOT/core-aux-data-%A_%a.out" \
  --array="0-$((DATA_ARRAY_SIZE - 1))" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/prepare_core_aux_data.sbatch")
DATA_JOB=$(td2_parse_job_id "$DATA_SUBMISSION")

td2_note "submitting protocol-shaped GPU smoke gate after data array $DATA_JOB"
SMOKE_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir=/tmp \
  --job-name=td2-coreaux-smoke \
  --output="$LOG_ROOT/core-aux-smoke-%j.out" \
  --dependency="afterok:$DATA_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/gpu_smoke_core_aux_2b.sbatch")
SMOKE_JOB=$(td2_parse_job_id "$SMOKE_SUBMISSION")

td2_note "submitting $STREAM_ARRAY_SIZE core/auxiliary 2B streams after smoke job $SMOKE_JOB"
TRAIN_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir=/tmp \
  --job-name=td2-coreaux \
  --output="$LOG_ROOT/core-aux-%A_%a.out" \
  --array="0-$((STREAM_ARRAY_SIZE - 1))%$MAX_CONCURRENT" \
  --dependency="afterok:$SMOKE_JOB" \
  --kill-on-invalid-dep=yes \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/core_aux_2b.sbatch")
TRAIN_JOB=$(td2_parse_job_id "$TRAIN_SUBMISSION")

td2_note "submitting aggregate plots after the training array"
AGGREGATE_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir=/tmp \
  --job-name=td2-coreaux-plot \
  --output="$LOG_ROOT/core-aux-aggregate-%j.out" \
  --dependency="afterok:$TRAIN_JOB" \
  --kill-on-invalid-dep=yes \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/aggregate_core_aux_2b.sbatch")
AGGREGATE_JOB=$(td2_parse_job_id "$AGGREGATE_SUBMISSION")

if [[ $RESUME == true ]]; then
  RECEIPT="$ORCHESTRATION_ROOT/resume-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
else
  RECEIPT=$INITIAL_RECEIPT
fi
td2_write_receipt "$RECEIPT" \
  "state=scheduled" \
  "mode=$([[ $RESUME == true ]] && printf resume || printf initial)" \
  "workflow=core_aux_2b" \
  "cluster=$CLUSTER" \
  "account=$ACCOUNT" \
  "exclude_nodes=${EXCLUDE_NODES:-none}" \
  "config_path=$CONFIG_PATH" \
  "data_config_path=$DATA_CONFIG_PATH" \
  "data_root=$DATA_ROOT" \
  "plan_identity=$PLAN_IDENTITY" \
  "data_plan_identity=$DATA_PLAN_IDENTITY" \
  "max_concurrent=$MAX_CONCURRENT" \
  "data_job=$DATA_JOB" \
  "data_array_size=$DATA_ARRAY_SIZE" \
  "smoke_job=$SMOKE_JOB" \
  "training_job=$TRAIN_JOB" \
  "training_array_size=$STREAM_ARRAY_SIZE" \
  "aggregate_job=$AGGREGATE_JOB" \
  "soft_limit_seconds=2700"

printf '\nSubmitted successfully:\n'
printf '  core/aux data array:  %s (%s seeds)\n' "$DATA_JOB" "$DATA_ARRAY_SIZE"
printf '  GPU smoke gate:       %s\n' "$SMOKE_JOB"
printf '  core/aux 2B array:    %s (%s streams)\n' "$TRAIN_JOB" "$STREAM_ARRAY_SIZE"
printf '  aggregate plots:      %s\n' "$AGGREGATE_JOB"
printf '  receipt:              %s\n' "$RECEIPT"
