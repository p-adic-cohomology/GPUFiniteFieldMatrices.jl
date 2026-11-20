#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the focused hot-path capture and review sequence.

Usage: ./perf/run_hotpaths.sh [--skip-setup] [--prefix NAME]

The runner uses Julia 1.12 and an existing CUDA allocation. By default it
prepares the benchmark environments, then runs nine fresh Julia processes
(three baseline-current-baseline cycles) at 100 samples, 5 seconds, and 10
warmups per case. It does not pull, switch branches, submit jobs, or use CPU
fallback timings.

Outputs are saved under perf/results and perf/results/reviews. Per-phase logs
are saved under perf/results/.hotpaths-runs/NAME/logs.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 2
}

skip_setup=0
batch_prefix=""
prefix_given=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --skip-setup)
            skip_setup=1
            shift
            ;;
        --prefix)
            [ "$#" -ge 2 ] || fail "--prefix requires a value"
            [ "$prefix_given" -eq 0 ] || fail "--prefix may be supplied once"
            batch_prefix="$2"
            prefix_given=1
            shift 2
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$runner_dir" rev-parse --show-toplevel)"
repo_root="$(cd "$repo_root" && pwd -P)"
runner_script="$runner_dir/run_hotpaths.sh"
helper_script="$runner_dir/run_hotpaths.jl"
julia_bin="${GFM_JULIA:-julia}"
run_root="$repo_root/.perf-regressions"
results_root="$repo_root/perf/results"
review_label=""
batch_dir=""

if [ "$prefix_given" -eq 0 ]; then
    batch_prefix="hotpaths-$(date -u '+%Y%m%dT%H%M%SZ')-$$"
fi
case "$batch_prefix" in
    ''|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*)
        fail "prefix must start with a letter or digit and contain only letters, digits, dot, underscore, or dash"
        ;;
esac
review_label="${batch_prefix}-review"

batch_parent="$results_root/.hotpaths-runs"
batch_dir="$batch_parent/$batch_prefix"
logs_dir="$batch_dir/logs"
snapshot_dir="$batch_dir/snapshots"

if [ -r /etc/pki/tls/certs/ca-bundle.crt ]; then
    if [ "${GIT_SSL_CAINFO+x}" != x ]; then
        export GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt
    fi
    if [ "${JULIA_SSL_CA_ROOTS_PATH+x}" != x ]; then
        export JULIA_SSL_CA_ROOTS_PATH=/etc/pki/tls/certs/ca-bundle.crt
    fi
fi

capture_target_free() {
    local candidate="$results_root/$1"
    if [ -e "$candidate" ] || [ -L "$candidate" ]; then
        fail "capture destination already exists: $candidate"
    fi
}

for cycle in 1 2 3; do
    capture_target_free "${batch_prefix}-031-before-${cycle}"
    capture_target_free "${batch_prefix}-current-${cycle}"
    capture_target_free "${batch_prefix}-031-after-${cycle}"
done
if [ -e "$results_root/reviews/$review_label" ] || [ -L "$results_root/reviews/$review_label" ]; then
    fail "review destination already exists: $results_root/reviews/$review_label"
fi

mkdir -p "$batch_parent"
if ! mkdir "$batch_dir" 2>/dev/null; then
    fail "batch destination already exists: $batch_dir"
fi
mkdir "$logs_dir" "$snapshot_dir"

on_failure() {
    local result_code=$?
    if [ "$result_code" -ne 0 ] && [ -n "$batch_dir" ]; then
        printf 'Stopped with exit %s. Partial results and logs are retained under %s\n' \
            "$result_code" "$batch_dir" >&2
    fi
}
trap on_failure EXIT

run_logged() {
    local log_file="$1"
    shift
    if [ -e "$log_file" ] || [ -L "$log_file" ]; then
        fail "log destination already exists: $log_file"
    fi
    (set -C; : > "$log_file") || fail "could not reserve log destination: $log_file"
    if "$@" 2>&1 | tee -a "$log_file"; then
        return 0
    else
        local result_code=$?
        printf 'Phase failed; see %s\n' "$log_file" >&2
        return "$result_code"
    fi
}

version_log="$logs_dir/preflight.log"
if ! julia_version_output=$("$julia_bin" --version 2>&1); then
    printf '%s\n' "$julia_version_output" | tee "$version_log" >&2
    fail "could not run Julia executable '$julia_bin'"
fi
printf '%s\n' "$julia_version_output" | tee "$version_log"
case "$julia_version_output" in
    *"1.12."*) ;;
    *) fail "Julia 1.12 is required; got: $julia_version_output" ;;
esac

if [ "$skip_setup" -eq 0 ]; then
    run_logged "$logs_dir/setup.log" "$runner_dir/setup.sh"
fi

for variant in v0.3.1 current; do
    env_dir="$run_root/envs/$variant"
    [ -f "$env_dir/Project.toml" ] || fail "benchmark environment is missing: $env_dir/Project.toml"
    [ -f "$env_dir/Manifest.toml" ] || fail "benchmark environment is missing: $env_dir/Manifest.toml"
done

baseline_root="$run_root/worktrees/v0.3.1"
[ -d "$baseline_root" ] || fail "v0.3.1 baseline worktree is missing: $baseline_root"
baseline_pin="7192d479d2c41d31021a4938beec8223d16d6654"
baseline_head="$(git -C "$baseline_root" rev-parse HEAD)"
[ "$baseline_head" = "$baseline_pin" ] || fail "v0.3.1 worktree is not at pinned commit $baseline_pin (found $baseline_head)"
baseline_changes="$(git -C "$baseline_root" status --porcelain --untracked-files=all)"
[ -z "$baseline_changes" ] || fail "v0.3.1 worktree is not clean: $baseline_root"

save_snapshot() {
    local name="$1"
    local source_root="$2"
    local source_head
    source_head="$(git -C "$source_root" rev-parse HEAD)"
    printf '%s\n' "$source_head" > "$snapshot_dir/$name.head"
    git -C "$source_root" diff --no-ext-diff --binary "$source_head" -- src perf Project.toml \
        > "$snapshot_dir/$name.diff"
}

assert_snapshot() {
    local name="$1"
    local source_root="$2"
    local expected_head
    local actual_head
    expected_head="$(cat "$snapshot_dir/$name.head")"
    actual_head="$(git -C "$source_root" rev-parse HEAD)"
    [ "$actual_head" = "$expected_head" ] || fail "HEAD changed during the capture batch: $source_root"
    git -C "$source_root" diff --no-ext-diff --binary "$expected_head" -- src perf Project.toml \
        > "$snapshot_dir/$name.check"
    if ! cmp -s "$snapshot_dir/$name.diff" "$snapshot_dir/$name.check"; then
        fail "tracked benchmark or package source changed during the capture batch: $source_root"
    fi
    rm -f "$snapshot_dir/$name.check"
}

save_snapshot current "$repo_root"
save_snapshot baseline "$baseline_root"
runner_oid="$(git -C "$repo_root" hash-object --no-filters "$runner_script")"
helper_oid="$(git -C "$repo_root" hash-object --no-filters "$helper_script")"

assert_scripts_stable() {
    local current_runner_oid
    local current_helper_oid
    current_runner_oid="$(git -C "$repo_root" hash-object --no-filters "$runner_script")"
    current_helper_oid="$(git -C "$repo_root" hash-object --no-filters "$helper_script")"
    [ "$current_runner_oid" = "$runner_oid" ] || fail "runner script changed during the capture batch"
    [ "$current_helper_oid" = "$helper_oid" ] || fail "Julia helper changed during the capture batch"
}

assert_batch_stable() {
    assert_snapshot current "$repo_root"
    assert_snapshot baseline "$baseline_root"
    assert_scripts_stable
}

capture_phase() {
    local phase_number="$1"
    local variant="$2"
    local label="$3"
    local env_dir="$run_root/envs/$variant"
    local log_file="$logs_dir/capture-${phase_number}-${label}.log"
    assert_batch_stable
    run_logged "$log_file" "$julia_bin" --startup-file=no "--project=$env_dir" \
        "$helper_script" capture "$repo_root" "$variant" "$label" 100 5 10 \
        "$snapshot_dir" "$helper_oid"
    assert_batch_stable
}

capture_index=0
for cycle in 1 2 3; do
    capture_index=$((capture_index + 1))
    phase_number="$(printf '%02d' "$capture_index")"
    capture_phase "$phase_number" v0.3.1 "${batch_prefix}-031-before-${cycle}"

    capture_index=$((capture_index + 1))
    phase_number="$(printf '%02d' "$capture_index")"
    capture_phase "$phase_number" current "${batch_prefix}-current-${cycle}"

    capture_index=$((capture_index + 1))
    phase_number="$(printf '%02d' "$capture_index")"
    capture_phase "$phase_number" v0.3.1 "${batch_prefix}-031-after-${cycle}"
done

assert_batch_stable
review_log="$logs_dir/review.log"
run_logged "$review_log" "$julia_bin" --startup-file=no \
    "--project=$run_root/envs/current" "$helper_script" review "$repo_root" "$review_label" \
    "${batch_prefix}-031-before-1" "${batch_prefix}-current-1" "${batch_prefix}-031-after-1" \
    "${batch_prefix}-031-before-2" "${batch_prefix}-current-2" "${batch_prefix}-031-after-2" \
    "${batch_prefix}-031-before-3" "${batch_prefix}-current-3" "${batch_prefix}-031-after-3" \
    "$snapshot_dir" "$helper_oid"
assert_batch_stable
[ -f "$results_root/reviews/$review_label/review.json" ] || fail "review output is missing: $results_root/reviews/$review_label/review.json"

printf '\nCapture artifacts:\n'
for cycle in 1 2 3; do
    printf '  %s\n' "$results_root/${batch_prefix}-031-before-${cycle}"
    printf '  %s\n' "$results_root/${batch_prefix}-current-${cycle}"
    printf '  %s\n' "$results_root/${batch_prefix}-031-after-${cycle}"
done
printf 'Review: %s/review.json\n' "$results_root/reviews/$review_label"
printf 'Logs: %s\n' "$logs_dir"
