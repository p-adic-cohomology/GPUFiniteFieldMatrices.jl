#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT HUP INT TERM

SOURCE="$FIXTURE/source"
mkdir -p "$SOURCE/test" "$SOURCE/src" "$SOURCE/ext" "$FIXTURE/bin"
cp "$ROOT/run-pkg-test.sh" "$SOURCE/test/run-pkg-test.sh"
cat >"$SOURCE/Project.toml" <<'EOF'
name = "WrapperIsolationFixture"
uuid = "0b52109d-0d18-4ea1-bbdd-8e0b541a9bc1"
version = "0.1.0"
EOF
cat >"$SOURCE/test/Project.toml" <<'EOF'
[deps]
EOF
for file in backend_test_driver.jl backend_optional_dependencies.toml run-pkg-test.jl; do
    cp "$ROOT/$file" "$SOURCE/test/$file"
done

printf 'root manifest sentinel\n' >"$SOURCE/Manifest.toml"
printf 'root preferences sentinel\n' >"$SOURCE/LocalPreferences.toml"
printf 'test manifest sentinel\n' >"$SOURCE/test/Manifest.toml"
printf 'versioned test manifest sentinel\n' >"$SOURCE/test/Manifest-v1.12.toml"
printf 'test preferences sentinel\n' >"$SOURCE/test/LocalPreferences.toml"
for state in .julia .cache compiled build artifacts; do
    mkdir -p "$SOURCE/test/$state/sentinel"
    printf '%s sentinel\n' "$state" >"$SOURCE/test/$state/sentinel/state.txt"
    mkdir -p "$SOURCE/$state/sentinel"
    printf '%s root sentinel\n' "$state" >"$SOURCE/$state/sentinel/state.txt"
done

cat >"$FIXTURE/bin/julia-capture" <<'EOF'
#!/bin/sh
set -eu

project=
for arg in "$@"; do
    case "$arg" in
        --project=*) project="${arg#--project=}" ;;
    esac
done
[ -n "$project" ] || { echo 'missing --project argument' >&2; exit 1; }

for file in Project.toml test/Project.toml test/backend_test_driver.jl \
    test/backend_optional_dependencies.toml test/run-pkg-test.jl; do
    [ -f "$project/$file" ] || { echo "missing expected view file: $file" >&2; exit 1; }
done
[ -L "$project/src" ] || { echo 'source link missing' >&2; exit 1; }
[ -L "$project/ext" ] || { echo 'extension link missing' >&2; exit 1; }
[ "$(readlink "$project/src")" = "$GFM_FIXTURE_SOURCE/src" ] || {
    echo 'source link points outside fixture source' >&2
    exit 1
}
[ "$(readlink "$project/ext")" = "$GFM_FIXTURE_SOURCE/ext" ] || {
    echo 'extension link points outside fixture source' >&2
    exit 1
}

for state in Manifest.toml LocalPreferences.toml \
    test/Manifest.toml test/Manifest-v1.12.toml test/LocalPreferences.toml \
    .julia .cache compiled build artifacts \
    test/.julia test/.cache test/compiled test/build test/artifacts; do
    [ ! -e "$project/$state" ] || {
        echo "stale state leaked into package view: $state" >&2
        exit 1
    }
done

printf 'fresh view captured\nproject=%s\n' "$project" >"$GFM_CAPTURE_FILE"
EOF
chmod +x "$FIXTURE/bin/julia-capture"

GFM_FIXTURE_SOURCE="$SOURCE" \
GFM_CAPTURE_FILE="$FIXTURE/capture.txt" \
JULIA_BIN="$FIXTURE/bin/julia-capture" \
JULIA_VERSION=1.12.7 \
    sh "$SOURCE/test/run-pkg-test.sh"

grep -q '^fresh view captured$' "$FIXTURE/capture.txt"
for sentinel in \
    "$SOURCE/Manifest.toml" \
    "$SOURCE/LocalPreferences.toml" \
    "$SOURCE/test/Manifest.toml" \
    "$SOURCE/test/Manifest-v1.12.toml" \
    "$SOURCE/test/LocalPreferences.toml" \
    "$SOURCE/test/.julia/sentinel/state.txt" \
    "$SOURCE/test/.cache/sentinel/state.txt" \
    "$SOURCE/test/compiled/sentinel/state.txt" \
    "$SOURCE/test/build/sentinel/state.txt" \
    "$SOURCE/test/artifacts/sentinel/state.txt"; do
    [ -f "$sentinel" ] || { echo "source sentinel was changed: $sentinel" >&2; exit 1; }
done

echo "Fresh package view isolation passed."
