#!/usr/bin/env bash
# Build the Linux x86_64 programs, `litmus` (the command line) and `litmus-app` (the desktop app), each with a trained
# release built in, in a pinned manylinux2014 image: CentOS 7's glibc 2.17, the oldest the Rust target supports, so
# the programs run on any x86_64 Linux with glibc 2.17 or later (RHEL and CentOS 7, Debian 8, Ubuntu 14.04, and later).
# The build is release/build_release.py --native-only run in the container: the fixed root, the pinned toolchain,
# --locked, and the path-remapping flags the Windows programs are built with.
#
#   build_linux.sh <release folder> <output folder>
#
# Writes litmus, litmus-app, manifest.json, and SHA256SUMS into <output folder>, which must be empty or absent, and
# fails if either program needs a glibc symbol newer than 2.17. Needs Docker and network access to ghcr.io,
# static.rust-lang.org, and crates.io. The repository and the release folder are mounted read-only; the container
# writes only to <output folder>. DOCKER names the command (`DOCKER="sudo docker"` where docker needs sudo);
# LITMUS_BUILD_CPUS and LITMUS_BUILD_MEMORY cap the container (default 8 CPUs, or fewer if the machine has fewer, and
# 24g).
set -euo pipefail

# ghcr.io/pyo3/maturin at tag v1.16.0 by digest, as engine/litmus-py/build_wheel.sh: manylinux2014 (CentOS 7.9,
# glibc 2.17, GCC 10), with rustup, Python 3.12, and git.
IMAGE="ghcr.io/pyo3/maturin:v1.16.0@sha256:50bf2ff99eeff7b09332bad9baf4a5e925e2502304ae2f88f48ce8c06fd9957c"
# The compiler engine/ and app/ pin in rust-toolchain.toml; checked against those files below.
RUST="1.100.0"
GLIBC_FLOOR="2.17"

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <release folder> <output folder>" >&2
    exit 2
fi
project=$(cd "$(dirname "$0")/.." && pwd)
release=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
if [ -n "$(ls -A "$out")" ]; then
    echo "$out is not empty; choose an empty folder" >&2
    exit 2
fi
for file in "$project/engine/rust-toolchain.toml" "$project/app/rust-toolchain.toml"; do
    if ! grep -qx "channel = \"$RUST\"" "$file"; then
        echo "$file does not pin $RUST; update RUST in $0" >&2
        exit 2
    fi
done
# The git checkout holding the project is mounted whole, so that the manifest records its commit.
top=$(git -C "$project" rev-parse --show-toplevel 2>/dev/null || echo "$project")
inner="${project#"$top"}"

# RUSTUP_TOOLCHAIN overrides rust-toolchain.toml, which also asks for the WebAssembly target this build does not use.
# The container is named so that a shared machine's owner can tell it apart; it runs as root, then hands the output
# to the calling user.
${DOCKER:-docker} run --rm --name "litmus-linux-build-$$" \
    --cpus "${LITMUS_BUILD_CPUS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}" --memory "${LITMUS_BUILD_MEMORY:-24g}" \
    --volume "$top:/repo:ro" \
    --volume "$release:/release:ro" \
    --volume "$out:/out" \
    --env RUSTUP_TOOLCHAIN="$RUST" \
    --env PROJECT="/repo$inner" \
    --env GLIBC_FLOOR="$GLIBC_FLOOR" \
    --env OWNER="$(id -u):$(id -g)" \
    --entrypoint /bin/bash \
    "$IMAGE" -euo pipefail -c '
        rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal
        git config --global --add safe.directory "*"
        python3.12 "$PROJECT/release/build_release.py" --native-only --release /release --out /out
        for program in /out/litmus /out/litmus-app; do
            newest=$(objdump -T "$program" | grep -o "GLIBC_[0-9.]*" | sed "s/GLIBC_//" | sort -uV | tail -n 1)
            if [ "$(printf "%s\n%s\n" "$newest" "$GLIBC_FLOOR" | sort -V | tail -n 1)" != "$GLIBC_FLOOR" ]; then
                echo "$program needs glibc $newest, newer than the floor $GLIBC_FLOOR" >&2
                exit 1
            fi
            echo "$program: newest glibc symbol $newest, within the floor $GLIBC_FLOOR"
        done
        chown -R "$OWNER" /out
    '
(cd "$out" && sha256sum -c SHA256SUMS)
