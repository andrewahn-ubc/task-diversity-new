#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

ROOT_1B=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b
ROOT_2B=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh
OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b-vs-fixed2b-gru256-comparison

td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

for root in "$ROOT_1B" "$ROOT_2B"; do
  [[ -d $root ]] || td2_die "result root does not exist: $root"
done

exec "$TD2_VENV_DIR/bin/python" -m td2.plot \
  --config "$REPO_ROOT/configs/experiment.yaml" \
  --output-root "$OUTPUT_ROOT" \
  --fixed-horizon-comparison \
  --fixed-1b-root "$ROOT_1B" \
  --fixed-2b-root "$ROOT_2B"
