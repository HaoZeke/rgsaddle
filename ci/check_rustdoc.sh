#!/usr/bin/env bash
# Build the rustdoc with every warning denied on each feature set CI
# tests, then once with private items so a broken link in a private
# doc fails too. A feature set gates modules (src/io.rs, the C API),
# so one build does not see every doc comment.
set -euo pipefail

export RUSTDOCFLAGS="-D warnings"

feature_sets=("" "capi" "capi,ohtst" "readcon" "readcon,readcon-db" "capi,verify")

for features in "${feature_sets[@]}"; do
  echo "== rustdoc features: ${features:-default}"
  if [[ -n "$features" ]]; then
    cargo doc --locked --no-deps --features "$features"
  else
    cargo doc --locked --no-deps
  fi
done

echo "== rustdoc private items, features: capi,ohtst,readcon,readcon-db,verify"
cargo doc --locked --no-deps --document-private-items \
  --features capi,ohtst,readcon,readcon-db,verify
