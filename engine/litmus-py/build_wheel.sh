#!/usr/bin/env bash
# Build the Linux x86_64 wheel of litmus_engine (abi3: one wheel for CPython 3.10 and later, manylinux2014) inside
# maturin's pinned Docker image, with the engine's pinned Rust and its Cargo.lock (--locked). The image's CPython
# 3.12, the contribute server's version, is the interpreter maturin builds against.
#
#   build_wheel.sh <project folder> <output folder>
#
# <project folder> is the Litmus project root, the folder holding engine/ (mounted read-only); the wheel is written
# to <output folder>, created if missing. Needs Docker and network access to ghcr.io, static.rust-lang.org and
# crates.io; nothing else on the host. The build writes only inside the container and to <output folder>.
set -euo pipefail

# ghcr.io/pyo3/maturin at tag v1.16.0 (maturin 1.16.0, the version pyproject.toml pins), by digest.
IMAGE="ghcr.io/pyo3/maturin:v1.16.0@sha256:50bf2ff99eeff7b09332bad9baf4a5e925e2502304ae2f88f48ce8c06fd9957c"
# The compiler engine/rust-toolchain.toml pins; checked against that file below.
RUST="1.100.0"

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <project folder holding engine/> <output folder>" >&2
    exit 2
fi
source_dir=$(cd "$1" && pwd)
mkdir -p "$2"
output_dir=$(cd "$2" && pwd)

if [ ! -f "$source_dir/engine/litmus-py/Cargo.toml" ] || [ ! -f "$source_dir/engine/Cargo.lock" ]; then
    echo "$source_dir is not the Litmus project: engine/litmus-py/Cargo.toml or engine/Cargo.lock is missing" >&2
    exit 2
fi
if ! grep -qx "channel = \"$RUST\"" "$source_dir/engine/rust-toolchain.toml"; then
    echo "engine/rust-toolchain.toml does not pin $RUST; update RUST in $0" >&2
    exit 2
fi

# RUSTUP_TOOLCHAIN overrides rust-toolchain.toml, which also asks for the WebAssembly target this build does not use.
# The wheel is built as root inside the container, then handed to the calling user. DOCKER names the command
# (`DOCKER="sudo docker"` where docker needs sudo), as in release/build_linux.sh.
${DOCKER:-docker} run --rm \
    --volume "$source_dir:/src:ro" \
    --volume "$output_dir:/out" \
    --workdir /src/engine \
    --env RUSTUP_TOOLCHAIN="$RUST" \
    --env CARGO_TARGET_DIR=/tmp/target \
    --env OWNER="$(id -u):$(id -g)" \
    --entrypoint /bin/bash \
    "$IMAGE" -euo pipefail -c '
        rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal
        rustc --version
        maturin --version
        maturin build --release --locked --manifest-path litmus-py/Cargo.toml \
            --interpreter python3.12 --compatibility manylinux2014 --out /tmp/wheels
        cp /tmp/wheels/litmus_engine-*.whl /out/
        chown "$OWNER" /out/litmus_engine-*.whl
        ls /tmp/wheels
    '

echo "wheel in $output_dir:"
sha256sum "$output_dir"/litmus_engine-*.whl
