#!/usr/bin/env bash
set -euo pipefail

# BenchBot run: expensive exact-reference validations.
# Usage:
#   scripts/benchbot-ohtst-pafi-exact.sh [optional-test-filter]

FILTER="${1:-}"

if [[ -n "${FILTER}" ]]; then
  cargo test --release --features ohtst --test ohtst_pafi_exact -- --ignored --nocapture "${FILTER}"
else
  cargo test --release --features ohtst --test ohtst_pafi_exact -- --ignored --nocapture
fi
