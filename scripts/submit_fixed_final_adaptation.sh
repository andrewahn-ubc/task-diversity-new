#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

CLUSTER=narval
ACCOUNT=def-krallen
DATA_ROOT=/scratch/taegyoem/td2-data-10dist-9div-cbp-seeded100
OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed-final-heldout-100m
SOURCE_1B_256=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b
SOURCE_2B_256=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh
SOURCE_2B_512=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh-gru512
SOURCE_2B_1024=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh-gru1024
MAX_CONCURRENT=100
EXCLUDE_NODES=ng31004,ng31104
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/submit_fixed_final_adaptation.sh [options]

Branches from all 180 unique terminal fixed-run checkpoints (1B/GRU-256 and
2B/GRU-256,512,1024), adapts each branch for 100M steps on its seed-specific
100-task held-out set, and creates four comparison figures. The duplicated
2B/GRU-256 condition is trained only once. Every job uses the def-krallen
allocation.

Options:
  --cluster NAME              Slurm cluster (default: narval)
  --data-root PATH            Existing five seed-specific task bundles
  --output-root PATH          New held-out adaptation output root
  --fixed-1b-gru256-root PATH Completed original fixed 1B result root
  --fixed-2b-gru256-root PATH Completed clean fixed 2B GRU-256 root
  --fixed-2b-gru512-root PATH Completed clean fixed 2B GRU-512 root
  --fixed-2b-gru1024-root PATH Completed clean fixed 2B GRU-1024 root
  --max-concurrent N          GPU array concurrency cap (default: 100; 1-100)
  --exclude-nodes LIST        Nodes to avoid (default: ng31004,ng31104)
  --resume                    Resume this exact output after prior jobs stop
  -h, --help                  Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --cluster) CLUSTER=${2:?--cluster requires a value}; shift 2 ;;
    --data-root) DATA_ROOT=${2:?--data-root requires a value}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?--output-root requires a value}; shift 2 ;;
    --fixed-1b-gru256-root) SOURCE_1B_256=${2:?option requires a value}; shift 2 ;;
    --fixed-2b-gru256-root) SOURCE_2B_256=${2:?option requires a value}; shift 2 ;;
    --fixed-2b-gru512-root) SOURCE_2B_512=${2:?option requires a value}; shift 2 ;;
    --fixed-2b-gru1024-root) SOURCE_2B_1024=${2:?option requires a value}; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT=${2:?--max-concurrent requires a value}; shift 2 ;;
    --exclude-nodes) EXCLUDE_NODES=${2:?--exclude-nodes requires a value}; shift 2 ;;
    --resume) RESUME=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) td2_die "unknown submission option: $1" ;;
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
[[ -z $EXCLUDE_NODES ]] || TD2_EXCLUDE_SBATCH_ARGS=("--exclude=$EXCLUDE_NODES")

td2_require_command sbatch
td2_require_command scontrol
td2_require_command squeue
td2_require_command flock
td2_load_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "$CLUSTER"
export PYTHONPATH="$REPO_ROOT/src"

CURRENT_CLUSTER=$(scontrol show config 2>/dev/null | awk \
  '$1 == "ClusterName" && !found {name=tolower($3); found=1} END {if (found) print name}')
if [[ -n $CURRENT_CLUSTER && $CURRENT_CLUSTER != "$CLUSTER" ]]; then
  td2_die "connected Slurm cluster is '$CURRENT_CLUSTER', but --cluster requested '$CLUSTER'"
fi

CONFIG_256="$REPO_ROOT/configs/experiment.yaml"
CONFIG_512="$REPO_ROOT/configs/experiment_gru512.yaml"
CONFIG_1024="$REPO_ROOT/configs/experiment_gru1024.yaml"
for path in "$DATA_ROOT" "$SOURCE_1B_256" "$SOURCE_2B_256" \
  "$SOURCE_2B_512" "$SOURCE_2B_1024"; do
  [[ -d $path ]] || td2_die "required Narval path does not exist: $path"
done
DATA_ROOT=$(cd -- "$DATA_ROOT" && pwd -P)
SOURCE_1B_256=$(cd -- "$SOURCE_1B_256" && pwd -P)
SOURCE_2B_256=$(cd -- "$SOURCE_2B_256" && pwd -P)
SOURCE_2B_512=$(cd -- "$SOURCE_2B_512" && pwd -P)
SOURCE_2B_1024=$(cd -- "$SOURCE_2B_1024" && pwd -P)

case "$OUTPUT_ROOT$DATA_ROOT$SOURCE_1B_256$SOURCE_2B_256$SOURCE_2B_512$SOURCE_2B_1024$TD2_RUNTIME_ENV$REPO_ROOT" in
  *$'\n'*|*,*|*%*) td2_die "Slurm-exported paths may not contain newlines, commas, or percent signs" ;;
esac
for source in "$DATA_ROOT" "$SOURCE_1B_256" "$SOURCE_2B_256" \
  "$SOURCE_2B_512" "$SOURCE_2B_1024"; do
  if [[ $OUTPUT_ROOT == "$source" || $OUTPUT_ROOT == "$source/"* \
        || $source == "$OUTPUT_ROOT/"* ]]; then
    td2_die "--output-root must be distinct from and not nested with source/data roots"
  fi
done

td2_note "validating all 180 exact terminal checkpoints before submission"
"$TD2_VENV_DIR/bin/python" -m td2.fixed_adaptation \
  --data-root "$DATA_ROOT" \
  --condition 1 256 fixed "$CONFIG_256" "$SOURCE_1B_256" \
  --condition 2 256 fixed_2b_fresh "$CONFIG_256" "$SOURCE_2B_256" \
  --condition 2 512 fixed_2b_fresh "$CONFIG_512" "$SOURCE_2B_512" \
  --condition 2 1024 fixed_2b_fresh "$CONFIG_1024" "$SOURCE_2B_1024"

PLAN_256=$("$TD2_VENV_DIR/bin/python" -m td2.plan --config "$CONFIG_256" identity)
PLAN_512=$("$TD2_VENV_DIR/bin/python" -m td2.plan --config "$CONFIG_512" identity)
PLAN_1024=$("$TD2_VENV_DIR/bin/python" -m td2.plan --config "$CONFIG_1024" identity)

mkdir -p -- "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
LOG_ROOT="$OUTPUT_ROOT/slurm-logs"
ORCHESTRATION_ROOT="$OUTPUT_ROOT/.orchestration"
INITIAL_RECEIPT="$ORCHESTRATION_ROOT/initial-submission.txt"
if [[ ! -f $INITIAL_RECEIPT ]]; then
  EXISTING_OUTPUT=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)
  [[ -z $EXISTING_OUTPUT ]] || td2_die \
    "output root is nonempty but has no receipt: $OUTPUT_ROOT; choose a fresh path"
fi
mkdir -p -- "$LOG_ROOT" "$ORCHESTRATION_ROOT"
exec 9>"$ORCHESTRATION_ROOT/submission.lock"
flock -n 9 || td2_die "another submission command is running for $OUTPUT_ROOT"

if [[ -f $INITIAL_RECEIPT && $RESUME == false ]]; then
  td2_die "this output was already submitted; use --resume after its jobs stop"
fi
if [[ ! -f $INITIAL_RECEIPT && $RESUME == true ]]; then
  td2_die "--resume requires $INITIAL_RECEIPT"
fi
if [[ $RESUME == true ]]; then
  receipt_value() {
    local key=$1
    awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' \
      "$INITIAL_RECEIPT"
  }
  [[ $(receipt_value data_root) == "$DATA_ROOT" \
     && $(receipt_value source_1b_gru256) == "$SOURCE_1B_256" \
     && $(receipt_value source_2b_gru256) == "$SOURCE_2B_256" \
     && $(receipt_value source_2b_gru512) == "$SOURCE_2B_512" \
     && $(receipt_value source_2b_gru1024) == "$SOURCE_2B_1024" \
     && $(receipt_value plan_256) == "$PLAN_256" \
     && $(receipt_value plan_512) == "$PLAN_512" \
     && $(receipt_value plan_1024) == "$PLAN_1024" ]] || td2_die \
    "existing output receipt belongs to different sources or configurations"
  CURRENT_USER=$(id -un)
  ACTIVE_JOBS=$(squeue --noheader --user="$CURRENT_USER" --format='%j' | \
    awk '$1 ~ /^td2-finaladapt/ {print $1}')
  [[ -z $ACTIVE_JOBS ]] || td2_die \
    "refusing to resume while final-adaptation jobs are active: ${ACTIVE_JOBS//$'\n'/, }"
  td2_note "resuming completed-safe array elements after confirming prior jobs stopped"
else
  td2_write_receipt "$INITIAL_RECEIPT" \
    "state=submitting" \
    "mode=initial" \
    "workflow=fixed_final_heldout_adaptation" \
    "cluster=$CLUSTER" \
    "account=$ACCOUNT" \
    "data_root=$DATA_ROOT" \
    "source_1b_gru256=$SOURCE_1B_256" \
    "source_2b_gru256=$SOURCE_2B_256" \
    "source_2b_gru512=$SOURCE_2B_512" \
    "source_2b_gru1024=$SOURCE_2B_1024" \
    "plan_256=$PLAN_256" \
    "plan_512=$PLAN_512" \
    "plan_1024=$PLAN_1024" \
    "max_concurrent=$MAX_CONCURRENT"
fi

BASE_EXPORTS="ALL,TD2_REPO_ROOT=$REPO_ROOT,TD2_RUNTIME_ENV=$TD2_RUNTIME_ENV,TD2_PROFILE=$CLUSTER,TD2_OUTPUT_ROOT=$OUTPUT_ROOT,TD2_DATA_ROOT=$DATA_ROOT,TD2_SOURCE_1B_256=$SOURCE_1B_256,TD2_SOURCE_2B_256=$SOURCE_2B_256,TD2_SOURCE_2B_512=$SOURCE_2B_512,TD2_SOURCE_2B_1024=$SOURCE_2B_1024,TD2_SOFT_LIMIT_SECONDS=2700"
EXPORTS="$BASE_EXPORTS,TD2_ACCOUNT=$ACCOUNT"
td2_build_gpu_sbatch_args

td2_note "submitting 180 unique 100M held-out adaptation branches"
ADAPT_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir=/tmp \
  --job-name=td2-finaladapt \
  --output="$LOG_ROOT/final-adapt-%A_%a.out" \
  --array="0-179%$MAX_CONCURRENT" \
  "${TD2_GPU_SBATCH_ARGS[@]}" \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/fixed_final_adapt.sbatch")
ADAPT_JOB=$(td2_parse_job_id "$ADAPT_SUBMISSION")

td2_note "submitting four-figure aggregation after every branch completes"
AGGREGATE_SUBMISSION=$(sbatch --parsable \
  "${TD2_EXCLUDE_SBATCH_ARGS[@]}" \
  --account="$ACCOUNT" \
  --chdir=/tmp \
  --job-name=td2-finaladapt-plot \
  --output="$LOG_ROOT/final-adapt-aggregate-%j.out" \
  --dependency="afterok:$ADAPT_JOB" \
  --kill-on-invalid-dep=yes \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export="$EXPORTS" \
  "$REPO_ROOT/slurm/aggregate_fixed_final_adapt.sbatch")
AGGREGATE_JOB=$(td2_parse_job_id "$AGGREGATE_SUBMISSION")

if [[ $RESUME == true ]]; then
  RECEIPT="$ORCHESTRATION_ROOT/resume-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
else
  RECEIPT=$INITIAL_RECEIPT
fi
td2_write_receipt "$RECEIPT" \
  "state=scheduled" \
  "mode=$([[ $RESUME == true ]] && printf resume || printf initial)" \
  "workflow=fixed_final_heldout_adaptation" \
  "cluster=$CLUSTER" \
  "account=$ACCOUNT" \
  "exclude_nodes=${EXCLUDE_NODES:-none}" \
  "data_root=$DATA_ROOT" \
  "source_1b_gru256=$SOURCE_1B_256" \
  "source_2b_gru256=$SOURCE_2B_256" \
  "source_2b_gru512=$SOURCE_2B_512" \
  "source_2b_gru1024=$SOURCE_2B_1024" \
  "plan_256=$PLAN_256" \
  "plan_512=$PLAN_512" \
  "plan_1024=$PLAN_1024" \
  "max_concurrent=$MAX_CONCURRENT" \
  "adaptation_job=$ADAPT_JOB" \
  "adaptation_array_size=180" \
  "aggregate_job=$AGGREGATE_JOB" \
  "soft_limit_seconds=2700"

printf '\nSubmitted successfully:\n'
printf '  final held-out array: %s (180 unique branches)\n' "$ADAPT_JOB"
printf '  aggregate plots:      %s\n' "$AGGREGATE_JOB"
printf '  receipt:               %s\n' "$RECEIPT"
