#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
JULIA_BIN="${JULIA_BIN:-${JULIA:-julia}}"
JULIA_VERSION_ARG=""
if [ -n "${JULIA_VERSION:-}" ]; then
    JULIA_VERSION_ARG="+$JULIA_VERSION"
fi
BACKEND="${1:-}"
case "$BACKEND" in
    ""|cuda|metal) ;;
    *) echo "Usage: $0 [cuda|metal]" >&2; exit 2 ;;
esac

PACKAGE_VIEW="$(mktemp -d)"
trap 'rm -rf "$PACKAGE_VIEW"' EXIT HUP INT TERM
cp "$ROOT/Project.toml" "$PACKAGE_VIEW/Project.toml"
ln -s "$ROOT/src" "$PACKAGE_VIEW/src"
ln -s "$ROOT/ext" "$PACKAGE_VIEW/ext"
cp -R "$ROOT/test" "$PACKAGE_VIEW/test"
if [ -e "$PACKAGE_VIEW/Manifest.toml" ]; then
    echo "Package view unexpectedly contains a Manifest.toml" >&2
    exit 1
fi

GPUFFM_TEST_BACKEND="$BACKEND" "$JULIA_BIN" $JULIA_VERSION_ARG \
    --startup-file=no --project="$PACKAGE_VIEW" "$PACKAGE_VIEW/test/run-pkg-test.jl"
