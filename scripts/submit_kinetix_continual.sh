#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/kinetix_common.sh"
REPO_ROOT=$(td2_repo_root)

CLUSTER=narval
ACCOUNT=def-krallen
DATA_ROOT=''
OUTPUT_ROOT=/scratch/taegyoem/td2-kinetix-results-7dist-heldout100
MAX_CONCURRENT=100
EXCLUDE_NODES=ng31004,ng31104
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/submit_kinetix_continual.sh [options]

Submit one CUDA/cuDNN preflight, 22 level-bank jobs, one production GPU gate,
seven sequential 15-stream training arrays, seven heldout-adaptation arrays,
and one aggregate job.
Every job uses def-krallen and has a one-hour walltime.

Options:
  --cluster narval       Cluster name (Narval only)
  --data-root PATH       Level-key root (default: runtime setting)
  --output-root PATH     Result root
  --max-concurrent N     True workflow-wide accelerator cap, 7-100
  --exclude-nodes LIST   Narval nodes to avoid
  --resume               Resubmit incomplete work after prior jobs stop
  -h, --help             Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --cluster) CLUSTER=${2:?}; shift 2 ;;
    --data-root) DATA_ROOT=${2:?}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT=${2:?}; shift 2 ;;
    --exclude-nodes) EXCLUDE_NODES=${2:?}; shift 2 ;;
    --resume) RESUME=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) td2_die "unknown Kinetix submission option: $1" ;;
  esac
done

[[ $CLUSTER == narval ]] || td2_die "this audited workflow targets Narval only"
td2_require_uint max-concurrent "$MAX_CONCURRENT"
((MAX_CONCURRENT >= 7 && MAX_CONCURRENT <= 100)) || td2_die "--max-concurrent must be 7-100"
[[ $ACCOUNT == def-krallen ]] || td2_die "Kinetix allocation invariant failed"
[[ -z $EXCLUDE_NODES || $EXCLUDE_NODES =~ ^[A-Za-z0-9._,-]+$ ]] || td2_die "invalid excluded-node list"
for command in sbatch scontrol squeue flock; do td2_require_command "$command"; done
td2_load_kinetix_runtime "$REPO_ROOT"
td2_load_profile "$REPO_ROOT" "$CLUSTER"
export PYTHONPATH="$REPO_ROOT/src"
[[ -n $DATA_ROOT ]] || DATA_ROOT=$TD2_KINETIX_DATA_ROOT
mkdir -p -- "$DATA_ROOT"
DATA_ROOT=$(cd -- "$DATA_ROOT" && pwd -P)
CURRENT_CLUSTER=$(scontrol show config 2>/dev/null | awk \
  '$1 == "ClusterName" && !found {name=tolower($3); found=1} END {if (found) print name}')
[[ -z $CURRENT_CLUSTER || $CURRENT_CLUSTER == narval ]] || td2_die "connected cluster is not Narval"
case "$OUTPUT_ROOT$DATA_ROOT$TD2_KINETIX_RUNTIME_ENV$REPO_ROOT" in
  *$'\n'*|*,*|*%*) td2_die "Slurm-exported paths contain forbidden characters" ;;
esac
[[ $OUTPUT_ROOT != "$DATA_ROOT" && $OUTPUT_ROOT != "$DATA_ROOT/"* ]] || td2_die "output and data roots overlap"

COUNTS=$("$TD2_KINETIX_VENV_DIR/bin/python" -m td2_kinetix.plan counts)
PROTOCOL_ID=$("$TD2_KINETIX_VENV_DIR/bin/python" -c \
  'import json,sys; print(json.loads(sys.argv[1])["protocol_id"])' "$COUNTS")
mkdir -p -- "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
LOG_ROOT="$OUTPUT_ROOT/slurm-logs"
ORCH="$OUTPUT_ROOT/.orchestration"
INITIAL="$ORCH/initial-submission.txt"
if [[ ! -f $INITIAL ]]; then
  EXISTING=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)
  [[ -z $EXISTING ]] || td2_die "nonempty output has no receipt: $OUTPUT_ROOT"
fi
mkdir -p -- "$LOG_ROOT" "$ORCH"
exec 9>"$ORCH/submission.lock"
flock -n 9 || td2_die "another submission is using $OUTPUT_ROOT"
if [[ -f $INITIAL && $RESUME == false ]]; then
  td2_die "already submitted; use --resume after its jobs stop"
elif [[ ! -f $INITIAL && $RESUME == true ]]; then
  td2_die "--resume requires an initial receipt"
fi
if [[ $RESUME == true ]]; then
  RECORDED=$(awk -F= '$1=="protocol_id"{print $2; exit}' "$INITIAL")
  [[ $RECORDED == "$PROTOCOL_ID" ]] || td2_die "receipt protocol mismatch"
  ACTIVE=$(squeue --noheader --user="$(id -un)" --format='%j' | awk '$1 ~ /^td2-kinetix/ {print}')
  [[ -z $ACTIVE ]] || td2_die "Kinetix jobs are still active; do not use --resume yet"
else
  td2_write_receipt "$INITIAL" "state=submitting" "workflow=kinetix_7dist_heldout100" \
    "cluster=narval" "account=def-krallen" "protocol_id=$PROTOCOL_ID" \
    "data_root=$DATA_ROOT" "output_root=$OUTPUT_ROOT" "max_concurrent=$MAX_CONCURRENT"
fi

EXCLUDE_ARGS=()
[[ -z $EXCLUDE_NODES ]] || EXCLUDE_ARGS=("--exclude=$EXCLUDE_NODES")
td2_build_gpu_sbatch_args
BASE_EXPORTS="ALL,TD2_REPO_ROOT=$REPO_ROOT,TD2_KINETIX_RUNTIME_ENV=$TD2_KINETIX_RUNTIME_ENV,TD2_PROFILE=narval,TD2_ACCOUNT=def-krallen,TD2_DATA_ROOT=$DATA_ROOT,TD2_OUTPUT_ROOT=$OUTPUT_ROOT,TD2_SOFT_LIMIT_SECONDS=2700"
COMMON_GPU=("${EXCLUDE_ARGS[@]}" --account=def-krallen --chdir=/tmp "${TD2_GPU_SBATCH_ARGS[@]}")
data_concurrency=$MAX_CONCURRENT; ((data_concurrency > 22)) && data_concurrency=22
# Up to six earlier heldout arrays can overlap the current main array; after
# main stage 7, all seven heldout arrays can overlap.  These caps therefore
# enforce MAX_CONCURRENT across the entire DAG, not merely within one array.
heldout_concurrency=$((MAX_CONCURRENT / 7))
((heldout_concurrency > 15)) && heldout_concurrency=15
main_concurrency=$((MAX_CONCURRENT - 6 * heldout_concurrency))
((main_concurrency > 15)) && main_concurrency=15

td2_note "submitting isolated CUDA/cuDNN runtime preflight"
PREFLIGHT_JOB=$(td2_parse_job_id "$(sbatch --parsable "${COMMON_GPU[@]}" \
  --job-name=td2-kinetix-preflight --output="$LOG_ROOT/preflight-%j.out" \
  --export="$BASE_EXPORTS" "$REPO_ROOT/slurm/gpu_preflight_kinetix.sbatch")")

td2_note "submitting 22 resumable strict-filter level banks"
DATA_JOB=$(td2_parse_job_id "$(sbatch --parsable "${COMMON_GPU[@]}" \
  --job-name=td2-kinetix-data --output="$LOG_ROOT/data-%A_%a.out" \
  --array="0-21%$data_concurrency" --dependency="afterok:$PREFLIGHT_JOB" \
  --kill-on-invalid-dep=yes --export="$BASE_EXPORTS" \
  "$REPO_ROOT/slurm/prepare_kinetix_data.sbatch")")

td2_note "submitting production-shaped GPU smoke gate"
SMOKE_JOB=$(td2_parse_job_id "$(sbatch --parsable "${COMMON_GPU[@]}" \
  --job-name=td2-kinetix-smoke --output="$LOG_ROOT/smoke-%j.out" \
  --dependency="afterok:$DATA_JOB" --kill-on-invalid-dep=yes \
  --export="$BASE_EXPORTS" "$REPO_ROOT/slurm/gpu_smoke_kinetix.sbatch")")

MAIN_JOBS=()
HELDOUT_JOBS=()
PREVIOUS=$SMOKE_JOB
for STAGE in 0 1 2 3 4 5 6; do
  td2_note "submitting main distribution $((STAGE + 1))/7"
  EXPORTS="$BASE_EXPORTS,TD2_KINETIX_STAGE=$STAGE"
  MAIN=$(td2_parse_job_id "$(sbatch --parsable "${COMMON_GPU[@]}" \
    --job-name="td2-kinetix-main$STAGE" --output="$LOG_ROOT/main-s$STAGE-%A_%a.out" \
    --array="0-14%$main_concurrency" --dependency="afterok:$PREVIOUS" \
    --kill-on-invalid-dep=yes --export="$EXPORTS" \
    "$REPO_ROOT/slurm/kinetix_main_stage.sbatch")")
  MAIN_JOBS+=("$MAIN")
  HELD=$(td2_parse_job_id "$(sbatch --parsable "${COMMON_GPU[@]}" \
    --job-name="td2-kinetix-held$STAGE" --output="$LOG_ROOT/heldout-s$STAGE-%A_%a.out" \
    --array="0-14%$heldout_concurrency" --dependency="afterok:$MAIN" \
    --kill-on-invalid-dep=yes --export="$EXPORTS" \
    "$REPO_ROOT/slurm/kinetix_heldout_stage.sbatch")")
  HELDOUT_JOBS+=("$HELD")
  PREVIOUS=$MAIN
done

DEPENDENCY="afterok:${MAIN_JOBS[6]}"
for JOB in "${HELDOUT_JOBS[@]}"; do DEPENDENCY+=":$JOB"; done
td2_note "submitting two-figure aggregation after every branch"
AGGREGATE_JOB=$(td2_parse_job_id "$(sbatch --parsable "${EXCLUDE_ARGS[@]}" \
  --account=def-krallen --chdir=/tmp --job-name=td2-kinetix-plot \
  --output="$LOG_ROOT/aggregate-%j.out" --dependency="$DEPENDENCY" \
  --kill-on-invalid-dep=yes --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
  --export="$BASE_EXPORTS" "$REPO_ROOT/slurm/aggregate_kinetix.sbatch")")

if [[ $RESUME == true ]]; then
  RECEIPT="$ORCH/resume-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
else
  RECEIPT=$INITIAL
fi
MODE=initial
[[ $RESUME == false ]] || MODE=resume
LINES=("state=scheduled" "mode=$MODE" "workflow=kinetix_7dist_heldout100" \
  "cluster=narval" "account=def-krallen" "protocol_id=$PROTOCOL_ID" \
  "data_root=$DATA_ROOT" "output_root=$OUTPUT_ROOT" "preflight_job=$PREFLIGHT_JOB" \
  "data_job=$DATA_JOB" \
  "smoke_job=$SMOKE_JOB" "aggregate_job=$AGGREGATE_JOB" \
  "max_concurrent=$MAX_CONCURRENT" "exclude_nodes=${EXCLUDE_NODES:-none}" \
  "data_concurrency=$data_concurrency" "main_concurrency=$main_concurrency" \
  "heldout_concurrency=$heldout_concurrency" \
  "main_array_size=15" "heldout_array_size=15" "soft_limit_seconds=2700")
for STAGE in 0 1 2 3 4 5 6; do
  LINES+=("main_stage_$STAGE_job=${MAIN_JOBS[$STAGE]}")
  LINES+=("heldout_stage_$STAGE_job=${HELDOUT_JOBS[$STAGE]}")
done
td2_write_receipt "$RECEIPT" "${LINES[@]}"

printf '\nSubmitted Kinetix workflow successfully:\n'
printf '  CUDA/cuDNN preflight: %s\n' "$PREFLIGHT_JOB"
printf '  filtered data array: %s (22 banks)\n' "$DATA_JOB"
printf '  GPU smoke gate:      %s\n' "$SMOKE_JOB"
printf '  seven main arrays:   %s\n' "${MAIN_JOBS[*]}"
printf '  seven heldout arrays:%s\n' "${HELDOUT_JOBS[*]}"
printf '  aggregate plots:     %s\n' "$AGGREGATE_JOB"
printf '  receipt:             %s\n' "$RECEIPT"
