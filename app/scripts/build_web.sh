#!/usr/bin/env bash
# Build the browser page of the Litmus app into a dist folder: the WebAssembly module (with SIMD128) and its
# JavaScript glue from wasm-bindgen, the host page and worker from site/app/, the mark as favicon.svg, and for the
# private build the release's three files.
#
#   app/scripts/build_web.sh <dist folder> <release folder>     the private page (github.io)
#   app/scripts/build_web.sh --contribute <dist folder>         the contribute site's page (scored by the server)
#
# Needs: cargo with the wasm32-unknown-unknown target, and wasm-bindgen-cli of the same version as the wasm-bindgen
# crate pinned in app/Cargo.toml (WASM_BINDGEN names it if it is not on PATH). CARGO_TARGET_DIR is honored.
set -euo pipefail

contribute=0
if [ "${1:-}" = "--contribute" ]; then
  contribute=1
  shift
fi
dist=${1:?usage: build_web.sh [--contribute] <dist folder> [release folder]}
release=${2:-}
if [ "$contribute" = 0 ] && [ -z "$release" ]; then
  echo "the private page needs a release folder (model.safetensors, config.json, tokenizer.json)" >&2
  exit 2
fi

app=$(cd "$(dirname "$0")/.." && pwd)
site=$(cd "$app/../site/app" && pwd)
target=${CARGO_TARGET_DIR:-$app/target}
bindgen=${WASM_BINDGEN:-wasm-bindgen}
cargo=${CARGO:-cargo}

features=()
if [ "$contribute" = 1 ]; then
  features=(--features contribute)
fi
(cd "$app" && RUSTFLAGS="-C target-feature=+simd128" "$cargo" rustc --profile web --target wasm32-unknown-unknown --lib --crate-type cdylib "${features[@]}")

mkdir -p "$dist"
"$bindgen" --target web --no-typescript --out-dir "$dist" --out-name litmus_app "$target/wasm32-unknown-unknown/web/litmus_app.wasm"
cp "$site/index.html" "$dist/index.html"
cp "$app/assets/litmus_icon.svg" "$dist/favicon.svg"
if [ "$contribute" = 1 ]; then
  # The host page's loader shows the sentence of its build (release/build_release.py does the same).
  grep -q '<html lang="en" data-build="private">' "$dist/index.html"
  sed -i 's/<html lang="en" data-build="private">/<html lang="en" data-build="contribute">/' "$dist/index.html"
fi
if [ "$contribute" = 0 ]; then
  cp "$site/worker.js" "$dist/worker.js"
  for file in model.safetensors config.json tokenizer.json; do
    cp "$release/$file" "$dist/$file"
  done
else
  rm -f "$dist/worker.js" "$dist/model.safetensors" "$dist/config.json" "$dist/tokenizer.json"
fi
ls -l "$dist"
