#!/bin/sh
# Installs the `litmus` command line on Linux or macOS, from this project's GitHub releases and nowhere else.
#
#   sh install.sh [--dir DIR] [--version TAG] [--from-dir DIR] [--target NAME]
#
# It needs no elevation. It downloads the program for this computer and the release's published file list, checks the
# program against that list, and installs it to DIR (default: $LITMUS_INSTALL_DIR, else ~/.local/bin) only when the
# check passes; a program that does not match is never installed. It changes no shell startup file and no setting.
#
#   --dir DIR       where `litmus` goes
#   --version TAG   a release tag such as v0.1.0 [default: the latest release]
#   --from-dir DIR  install from a folder that holds the program and SHA256SUMS instead of downloading (no network)
#   --target NAME   linux-x86_64, macos-arm64 or macos-x86_64 [default: this computer's]
#
# The check: the release publishes SHA256SUMS, one "<sha-256>  <file name>" line per program. This script finds the
# program's line, computes the program's SHA-256 with sha256sum, shasum or openssl, and refuses on any difference.
# The program has the model built in, so one file is the whole install.
set -eu

REPO="lhallee/litmus"
DOCS="https://github.com/$REPO/blob/main/docs/USE_WITH_AGENTS.md"
dir="${LITMUS_INSTALL_DIR:-$HOME/.local/bin}"
version="latest"
from_dir=""
target=""

say() { printf '%s\n' "$*"; }
fail() { printf 'litmus install: %s\n' "$*" >&2; exit "${2:-1}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) [ $# -ge 2 ] || fail "--dir needs a folder" 2; dir="$2"; shift 2 ;;
        --version) [ $# -ge 2 ] || fail "--version needs a tag" 2; version="$2"; shift 2 ;;
        --from-dir) [ $# -ge 2 ] || fail "--from-dir needs a folder" 2; from_dir="$2"; shift 2 ;;
        --target) [ $# -ge 2 ] || fail "--target needs a name" 2; target="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -eu$/{/^set -eu$/!p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown option $1 (see --help)" 2 ;;
    esac
done

if [ -z "$target" ]; then
    system="$(uname -s)"; machine="$(uname -m)"
    case "$system:$machine" in
        Linux:x86_64|Linux:amd64) target="linux-x86_64" ;;
        Darwin:arm64|Darwin:aarch64) target="macos-arm64" ;;
        Darwin:x86_64) target="macos-x86_64" ;;
        MINGW*|MSYS*|CYGWIN*) fail "this is Windows: run install.ps1 in PowerShell instead" 2 ;;
        *) fail "no prebuilt program for $system $machine. Build it from source: cd engine && cargo build --release --locked, with a model folder ($DOCS)" 2 ;;
    esac
fi
case "$target" in
    linux-x86_64|macos-arm64|macos-x86_64) ;;
    *) fail "unknown target $target (linux-x86_64, macos-arm64 or macos-x86_64)" 2 ;;
esac
asset="litmus-$target"

case "$version" in
    latest) base="https://github.com/$REPO/releases/latest/download" ;;
    *[!A-Za-z0-9._-]*|"") fail "--version takes a release tag such as v0.1.0" 2 ;;
    *) base="https://github.com/$REPO/releases/download/$version" ;;
esac

work="$(mktemp -d "${TMPDIR:-/tmp}/litmus-install.XXXXXX")" || fail "cannot make a temporary folder"
trap 'rm -rf "$work"' EXIT INT TERM

fetch() { # name
    if [ -n "$from_dir" ]; then
        [ -f "$from_dir/$1" ] || fail "$from_dir has no $1"
        cp "$from_dir/$1" "$work/$1"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 --retry 2 --output "$work/$1" "$base/$1" \
            || fail "cannot download $base/$1 (is there a release with this program for $target? see $DOCS for the build from source)"
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --https-only --tries=3 --output-document="$work/$1" "$base/$1" \
            || fail "cannot download $base/$1 (is there a release with this program for $target? see $DOCS for the build from source)"
    else
        fail "needs curl or wget to download the program"
    fi
}

sha256_of() { # file
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d ' ' -f 1
    elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 "$1" | sed 's/^.*= *//'
    else fail "needs sha256sum, shasum or openssl to check the download"
    fi
}

fetch SHA256SUMS
fetch "$asset"
expected="$(awk -v name="$asset" '{ n = $2; sub(/^\*/, "", n); if (n == name) { print tolower($1); exit } }' "$work/SHA256SUMS")"
[ -n "$expected" ] || fail "the release's file list does not name $asset; nothing was installed"
actual="$(sha256_of "$work/$asset")"
[ "$actual" = "$expected" ] || fail "the download does not match the release's published file list; nothing was installed"

mkdir -p "$dir" || fail "cannot make $dir"
dir="$(cd "$dir" && pwd)"
existing=""
[ -e "$dir/litmus" ] && existing=" (replacing the earlier copy)"
cp "$work/$asset" "$dir/.litmus.new.$$" || fail "cannot write to $dir"
chmod 755 "$dir/.litmus.new.$$"
mv -f "$dir/.litmus.new.$$" "$dir/litmus"

"$dir/litmus" --help >/dev/null 2>&1 || fail "installed $dir/litmus but it does not run on this computer; remove it and build from source ($DOCS)" 3

say "Installed $dir/litmus$existing"
say "  from:  ${from_dir:-$base} ($asset), checked against the release's published file list"
say "  model: built in, nothing else to download"
say "  files: wrote only $dir/litmus; no shell startup file or setting was changed"
case ":$PATH:" in
    *":$dir:"*) say "  path:  $dir is on PATH, so \`litmus\` runs by name" ;;
    *) say "  path:  $dir is not on PATH. Run it as $dir/litmus, or add it: export PATH=\"$dir:\$PATH\"" ;;
esac
say "Next: register the server with your agent ($DOCS), for example:"
say "  claude mcp add litmus -- $dir/litmus mcp"
