#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)

# Options supplied by the caller come last, so paths and generic scheduling
# settings can still be overridden without editing this dedicated launcher.
exec bash "$SCRIPT_DIR/submit_fixed_2b.sh" \
  --config "$REPO_ROOT/configs/experiment_gru1024.yaml" \
  --output-root /scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh-gru1024 \
  "$@"
