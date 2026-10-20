#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

CLUSTER=narval
ACCOUNT_A=def-krallen
ACCOUNT_B=def-mijungp
SOURCE_OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b
OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-extension
DATA_ROOT=/scratch/taegyoem/td2-data-10dist-9div-cbp-seeded100
MAX_CONCURRENT=100
EXCLUDE_NODES=ng31004,ng31104
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/submit_fixed_2b.sh [options]

Continues the 45 completed fixed-distribution streams from 1B to 2B nominal
environment steps, then creates a combined 0-2B plot. The completed 1B source
tree is read-only. GPU work is divided between two accounts while their
combined concurrency never exceeds --max-concurrent.

Options:
  --cluster NAME             Slurm cluster (default: narval)
  --account-a ACCOUNT        First allocation (default: def-krallen)
  --account-b ACCOUNT        Second allocation (default: def-mijungp)
  --source-output-root PATH  Completed 1B experiment
  --output-root PATH         New 1B-to-2B extension output
  --data-root PATH           Existing seed-specific task bundles
  --max-concurrent N         Combined GPU concurrency cap (default: 100; 2-100)
  --exclude-nodes LIST       Comma-separated Slurm nodes to avoid
                             (default: ng31004,ng31104)
  --resume                   Resume an interrupted extension output root. Refuses
                             to run while any td2-* jobs are queued or running.
  -h, --help                 Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --cluster)
      (($# >= 2)) || td2_die "--cluster requires a value"
      CLUSTER=$2
      shift 2
      ;;
    --account-a)
      (($# >= 2)) || td2_die "--account-a requires a value"
      ACCOUNT_A=$2
      shift 2
      ;;
    --account-b)
      (($# >= 2)) || td2_die "--account-b requires a value"
      ACCOUNT_B=$2
      shift 2
      ;;
    --source-output-root)
      (($# >= 2)) || td2_die "--source-output-root requires a path"
      SOURCE_OUTPUT_ROOT=$2
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

[[ $ACCOUNT_A =~ ^[A-Za-z0-9_.-]+$ ]] || td2_die "invalid account name: $ACCOUNT_A"
[[ $ACCOUNT_B =~ ^[A-Za-z0-9_.-]+$ ]] || td2_die "invalid account name: $ACCOUNT_B"
[[ $ACCOUNT_A != "$ACCOUNT_B" ]] || td2_die "--account-a and --account-b must differ"
td2_require_uint max-concurrent "$MAX_CONCURRENT"
((MAX_CONCURRENT >= 2 && MAX_CONCURRENT <= 100)) || td2_die \
  "--max-concurrent must be between 2 and 100"
if [[ -n $EXCLUDE_NODES ]]; then
  [[ $EXCLUDE_NODES =~ ^[A-Za-z0-9._,-]+$ ]] || td2_die \
    "invalid --exclude-nodes list: $EXCLUDE_NODES"
fi
ACCOUNT_A_CAP=$(( (MAX_CONCURRENT + 1) / 2 ))
ACCOUNT_B_CAP=$(( MAX_CONCURRENT / 2 ))
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

PLAN_COUNTS=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" counts)
PLAN_IDENTITY=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
  --config "$REPO_ROOT/configs/experiment.yaml" identity)
[[ $PLAN_IDENTITY =~ ^[0-9a-f]{64}$ ]] || td2_die \
  "invalid production plan identity: $PLAN_IDENTITY"
IFS=$'\t' read -r STREAM_ARRAY_SIZE ADAPT_ARRAY_SIZE <<<"$PLAN_COUNTS"
td2_require_uint stream-array-size "$STREAM_ARRAY_SIZE"
td2_require_uint adaptation-array-size "$ADAPT_ARRAY_SIZE"
((STREAM_ARRAY_SIZE == 45 && ADAPT_ARRAY_SIZE == 450)) || td2_die \
  "unexpected production plan: streams=$STREAM_ARRAY_SIZE adaptation=$ADAPT_ARRAY_SIZE"

CURRENT_CLUSTER=$(scontrol show config 2>/dev/null | awk \
  '$1 == "ClusterName" && !found {name=tolower($3); found=1} END {if (found) print name}')
if [[ -n $CURRENT_CLUSTER && $CURRENT_CLUSTER != "$CLUSTER" ]]; then
  td2_die "connected Slurm cluster is '$CURRENT_CLUSTER', but --cluster requested '$CLUSTER'"
fi

[[ -d $SOURCE_OUTPUT_ROOT ]] || td2_die \
  "completed 1B source output does not exist: $SOURCE_OUTPUT_ROOT"
[[ -d $DATA_ROOT ]] || td2_die "prepared data root does not exist: $DATA_ROOT"
SOURCE_OUTPUT_ROOT=$(cd -- "$SOURCE_OUTPUT_ROOT" && pwd -P)
DATA_ROOT=$(cd -- "$DATA_ROOT" && pwd -P)
mkdir -p -- "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)

case "$SOURCE_OUTPUT_ROOT$OUTPUT_ROOT$DATA_ROOT$TD2_RUNTIME_ENV$REPO_ROOT" in
  *$'\n'*|*,*|*%*) td2_die "Slurm-exported paths may not contain newlines, commas, or percent signs" ;;
esac
if [[ $OUTPUT_ROOT == "$SOURCE_OUTPUT_ROOT" \
      || $OUTPUT_ROOT == "$SOURCE_OUTPUT_ROOT/"* \
      || $SOURCE_OUTPUT_ROOT == "$OUTPUT_ROOT/"* ]]; then
  td2_die "--output-root and --source-output-root must be distinct and not nested"
fi
if [[ $OUTPUT_ROOT == "$DATA_ROOT" \
      || $OUTPUT_ROOT == "$DATA_ROOT/"* \
      || $DATA_ROOT == "$OUTPUT_ROOT/"* ]]; then
  td2_die "--output-root and --data-root must be distinct and not nested"
fi

SOURCE_RECEIPT="$SOURCE_OUTPUT_ROOT/.orchestration/initial-submission.txt"
[[ -f $SOURCE_RECEIPT ]] || td2_die \
  "source output has no initial submission receipt: $SOURCE_RECEIPT"
SOURCE_PLAN_IDENTITY=$(awk -F= \
  '$1 == "plan_identity" {print $2; found=1; exit} END {if (!found) exit 1}' \
  "$SOURCE_RECEIPT") || td2_die "source receipt has no plan identity"
[[ $SOURCE_PLAN_IDENTITY == "$PLAN_IDENTITY" ]] || td2_die \
  "source output belongs to a different experiment configuration"
[[ -f $SOURCE_OUTPUT_ROOT/.orchestration/aggregate/DONE ]] || td2_die \
  "source experiment aggregate marker is missing"

td2_note "validating all $STREAM_ARRAY_SIZE completed 1B fixed streams"
for ((INDEX = 0; INDEX < STREAM_ARRAY_SIZE; INDEX++)); do
  PLAN_ITEM=$("$TD2_VENV_DIR/bin/python" -m td2.plan \
    --config "$REPO_ROOT/configs/experiment.yaml" main --index "$INDEX")
  IFS=$'\t' read -r ALGORITHM DIVERSITY SEED <<<"$PLAN_ITEM"
  [[ -n $ALGORITHM && -n $DIVERSITY && -n $SEED ]] || td2_die \
    "could not decode source stream index $INDEX"
  SOURCE_STREAM=$(td2_fixed_dir \
    "$SOURCE_OUTPUT_ROOT" "$ALGORITHM" "$DIVERSITY" "$SEED")
  [[ -f $SOURCE_STREAM/DONE ]] || td2_die \
    "source fixed stream is incomplete: $SOURCE_STREAM"
  [[ -f $SOURCE_STREAM/resume/latest.json ]] || td2_die \
    "source fixed stream has no final resume checkpoint: $SOURCE_STREAM"
  [[ -s $SOURCE_STREAM/metrics.csv ]] || td2_die \
    "source fixed stream has no metrics: $SOURCE_STREAM"
done

LOG_ROOT="$OUTPUT_ROOT/slurm-logs"
ORCHESTRATION_ROOT="$OUTPUT_ROOT/.orchestration"
INITIAL_RECEIPT="$ORCHESTRATION_ROOT/initial-submission.txt"
if [[ ! -f $INITIAL_RECEIPT ]]; then
  EXISTING_OUTPUT=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)
  [[ -z $EXISTING_OUTPUT ]] || td2_die \
    "output root is nonempty but has no submission receipt: $OUTPUT_ROOT; choose a fresh path"
fi
mkdir -p -- "$LOG_ROOT" "$ORCHESTRATION_ROOT"

exec 9>"$ORCHESTRATION_ROOT/submission.lock"
flock -n 9 || td2_die "another submission command is already running for $OUTPUT_ROOT"

if [[ -f $INITIAL_RECEIPT && $RESUME == false ]]; then
  td2_die "this extension output was already submitted ($INITIAL_RECEIPT); use a fresh --output-root, or --resume after prior td2 jobs have stopped"
fi
if [[ ! -f $INITIAL_RECEIPT && $RESUME == true ]]; then
  td2_die "--resume requires an existing initial submission receipt: $INITIAL_RECEIPT"
fi
if [[ $RESUME == true ]]; then
  RECEIPT_PLAN_IDENTITY=$(awk -F= \
    '$1 == "plan_identity" {print $2; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no plan identity"
  RECEIPT_SOURCE_ROOT=$(awk -F= \
    '$1 == "source_output_root" {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no source output root"
  RECEIPT_DATA_ROOT=$(awk -F= \
    '$1 == "data_root" {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' \
    "$INITIAL_RECEIPT") || td2_die "existing receipt has no data root"
  [[ $RECEIPT_PLAN_IDENTITY == "$PLAN_IDENTITY" ]] || td2_die \
    "existing extension output belongs to a different experiment configuration"
  [[ $RECEIPT_SOURCE_ROOT == "$SOURCE_OUTPUT_ROOT" ]] || td2_die \
    "existing extension output uses a different 1B source root"
  [[ $RECEIPT_DATA_ROOT == "$DATA_ROOT" ]] || td2_die \
    "existing extension output uses a different data root"
  CURRENT_USER=$(id -un)
  ACTIVE_TD2_JOBS=$(squeue --noheader --user="$CURRENT_USER" --format='%j' | awk '$1 ~ /^td2-/ {print $1}')
  [[ -z $ACTIVE_TD2_JOBS ]] || td2_die \
    "refusing to resume while td2 jobs are active: ${ACTIVE_TD2_JOBS//$'\n'/, }"
  td2_note "resuming only after confirming no td2 jobs are queued or running"
else
  td2_write_receipt "$INITIAL_RECEIPT" \
    "state=submitting" \
    "mode=initial" \
    "workflow=fixed_1b_to_2b" \
    "cluster=$CLUSTER" \
    "account_a=$ACCOUNT_A" \
    "account_b=$ACCOUNT_B" \
    "source_output_root=$SOURCE_OUTPUT_ROOT" \
    "data_root=$DATA_ROOT" \
    "plan_identity=$PLAN_IDENTITY" \
    "max_concurrent=$MAX_CONCURRENT"
fi

BASE_EXPORTS="ALL,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$CLUSTER,TD2_SOURCE_OUTPUT_ROOT=$SOURCE_OUTPUT_ROOT,TD2_OUTPUT_ROOT=$OUTPUT_ROOT,TD2_DATA_ROOT=$DATA_ROOT,TD2_SOFT_LIMIT_SECONDS=2700"
EXPORTS_A="$BASE_EXPORTS,TD2_ACCOUNT=$ACCOUNT_A"
EXPORTS_B="$BASE_EXPORTS,TD2_ACCOUNT=$ACCOUNT_B"
td2_build_gpu_sbatch_args

td2_note "submitting 45 fixed-distribution 1B-to-2B extensions"
FIXED_A_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT_A" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-fixed2b-a \
  --output="$LOG_ROOT/fixed2b-a-%A_%a.out" \
  --array="0-44:2%$ACCOUNT_A_CAP" \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS_A" \
  "$REPO_ROOT/slurm/fixed_2b.sbatch")
FIXED_A_JOB=$(td2_parse_job_id "$FIXED_A_SUBMISSION")
FIXED_B_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT_B" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-fixed2b-b \
  --output="$LOG_ROOT/fixed2b-b-%A_%a.out" \
  --array="1-43:2%$ACCOUNT_B_CAP" \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS_B" \
  "$REPO_ROOT/slurm/fixed_2b.sbatch")
FIXED_B_JOB=$(td2_parse_job_id "$FIXED_B_SUBMISSION")

td2_note "submitting combined 0-2B plot after extension arrays"
AGGREGATE_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT_B" \
  --chdir="$REPO_ROOT" \
  --job-name=td2-fixed2b-plot \
  --output="$LOG_ROOT/fixed2b-aggregate-%j.out" \
  --dependency="afterok:$FIXED_A_JOB:$FIXED_B_JOB" \
  --kill-on-invalid-dep=yes \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export="$EXPORTS_B" \
  "$REPO_ROOT/slurm/aggregate_fixed_2b.sbatch")
AGGREGATE_JOB=$(td2_parse_job_id "$AGGREGATE_SUBMISSION")

if [[ $RESUME == true ]]; then
  RECEIPT="$ORCHESTRATION_ROOT/resume-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
else
  RECEIPT=$INITIAL_RECEIPT
fi
td2_write_receipt "$RECEIPT" \
  "state=scheduled" \
  "mode=$([[ $RESUME == true ]] && printf resume || printf initial)" \
  "workflow=fixed_1b_to_2b" \
  "cluster=$CLUSTER" \
  "account_a=$ACCOUNT_A" \
  "account_b=$ACCOUNT_B" \
  "account_a_cap=$ACCOUNT_A_CAP" \
  "account_b_cap=$ACCOUNT_B_CAP" \
  "exclude_nodes=${EXCLUDE_NODES:-none}" \
  "source_output_root=$SOURCE_OUTPUT_ROOT" \
  "data_root=$DATA_ROOT" \
  "plan_identity=$PLAN_IDENTITY" \
  "max_concurrent=$MAX_CONCURRENT" \
  "fixed_a_job=$FIXED_A_JOB" \
  "fixed_b_job=$FIXED_B_JOB" \
  "fixed_array_size=$STREAM_ARRAY_SIZE" \
  "aggregate_job=$AGGREGATE_JOB" \
  "soft_limit_seconds=2700"

printf '\nSubmitted successfully:\n'
printf '  fixed 1B-to-2B arrays: %s, %s (%s streams)\n' \
  "$FIXED_A_JOB" "$FIXED_B_JOB" "$STREAM_ARRAY_SIZE"
printf '  aggregate 0-2B plot:   %s\n' "$AGGREGATE_JOB"
printf '  receipt:                %s\n' "$RECEIPT"
