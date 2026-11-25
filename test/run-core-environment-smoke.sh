#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
JULIA_BIN="${JULIA_BIN:-${JULIA:-julia}}"
JULIA_VERSION_ARG=""
if [ -n "${JULIA_VERSION:-}" ]; then
    JULIA_VERSION_ARG="+$JULIA_VERSION"
fi
TEST_ENV="$(mktemp -d)"
trap 'rm -rf "$TEST_ENV"' EXIT HUP INT TERM
PACKAGE_VIEW="$TEST_ENV/package"
mkdir -p "$PACKAGE_VIEW"
cp "$ROOT/Project.toml" "$PACKAGE_VIEW/Project.toml"
ln -s "$ROOT/src" "$PACKAGE_VIEW/src"
ln -s "$ROOT/ext" "$PACKAGE_VIEW/ext"

GPUFFM_PACKAGE_VIEW="$PACKAGE_VIEW" "$JULIA_BIN" $JULIA_VERSION_ARG --startup-file=no --project="$TEST_ENV" \
    -e 'using Pkg; Pkg.develop(path=ENV["GPUFFM_PACKAGE_VIEW"]; io=devnull); Pkg.add([Pkg.PackageSpec(name="KernelAbstractions", version=v"0.10.8"), Pkg.PackageSpec(name="KernelInterface", version=v"0.4.7"), Pkg.PackageSpec(name="BitIntegers", version=v"0.3.7")]; io=devnull); Pkg.instantiate(; io=devnull)'
"$JULIA_BIN" $JULIA_VERSION_ARG --startup-file=no --project="$TEST_ENV" \
    "$ROOT/test/core_environment_smoke.jl"
