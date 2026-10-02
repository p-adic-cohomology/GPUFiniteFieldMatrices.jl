#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
TEST_ENV="$(mktemp -d)"
trap 'rm -rf "$TEST_ENV"' EXIT HUP INT TERM
cp "$ROOT/test/Project.toml" "$TEST_ENV/Project.toml"

JULIA_CMD="${JULIA:-julia}"
GPUFFM_ROOT="$ROOT" "$JULIA_CMD" --startup-file=no --project="$TEST_ENV" \
    -e 'using Pkg; Pkg.develop(path=ENV["GPUFFM_ROOT"]; io=devnull); Pkg.instantiate(; io=devnull)'
"$JULIA_CMD" --project="$TEST_ENV" "$@"
