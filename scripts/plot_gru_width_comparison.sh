#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
REPO_ROOT=$(td2_repo_root)

ROOT_256=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh
ROOT_512=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh-gru512
ROOT_1024=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh-gru1024
OUTPUT_ROOT=/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-gru-comparison

td2_load_runtime "$REPO_ROOT"
td2_activate_runtime
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

for root in "$ROOT_256" "$ROOT_512" "$ROOT_1024"; do
  [[ -d $root ]] || td2_die "result root does not exist: $root"
done

exec "$TD2_VENV_DIR/bin/python" -m td2.plot \
  --config "$REPO_ROOT/configs/experiment.yaml" \
  --output-root "$OUTPUT_ROOT" \
  --gru-width-comparison \
  --gru-width-run 256 "$ROOT_256" \
  --gru-width-run 512 "$ROOT_512" \
  --gru-width-run 1024 "$ROOT_1024"
