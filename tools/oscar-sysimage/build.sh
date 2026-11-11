#!/usr/bin/env bash
set -Eeuo pipefail

readonly REQUIRED_JULIA_VERSION="1.12.4"
readonly DEFAULT_CPU_TARGET="generic"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
    cat <<'EOF'
Build an Oscar Linux x86_64 sysimage for this GPUFiniteFieldMatrices checkout.

Usage:
  tools/oscar-sysimage/build.sh [--help | --dry-run]

Configuration:
  JULIA_BIN                    Exact Julia 1.12.4 executable (default: julia)
  OSCAR_GFM_REPO               Package checkout (default: this repository)
  OSCAR_SYSIMAGE_OUTPUT_DIR    Artifact root (default: tools/oscar-sysimage/output)
  OSCAR_SYSIMAGE_CPU_TARGET    Julia CPU target (default: generic)
  OSCAR_SYSIMAGE_ALLOW_DIRTY   Set to 1 to build reviewed changed package source/project files

The build requires an explicitly prepared Julia 1.12 package Manifest selected
by Julia 1.12.4 and the normal Julia distribution sysimage. It does not resolve
or edit the package project. Build output is staged and validated before it is
atomically published.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

mode="build"
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
elif [[ $# -eq 1 ]]; then
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) mode="dry-run" ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
fi

repo_root="${OSCAR_GFM_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
[[ -d "$repo_root" ]] || fail "repository directory does not exist: $repo_root"
repo_root="$(cd -- "$repo_root" && pwd -P)"
[[ -f "$repo_root/Project.toml" ]] || fail "missing package Project.toml in $repo_root"
[[ -f "$SCRIPT_DIR/Project.toml" ]] || fail "missing pinned PackageCompiler project: $SCRIPT_DIR/Project.toml"
[[ -f "$SCRIPT_DIR/build.jl" && -f "$SCRIPT_DIR/precompile-workload.jl" && -f "$SCRIPT_DIR/validate-load.jl" && -f "$SCRIPT_DIR/provenance.jl" ]] || fail "sysimage tooling files are incomplete under $SCRIPT_DIR"

git_root="$(git -C "$repo_root" rev-parse --show-toplevel 2>/dev/null)" || fail "package path is not inside a Git checkout: $repo_root"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$repo_root" ]] || fail "OSCAR_GFM_REPO must name the checkout root ($git_root), got $repo_root"

source_status="$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all --ignored=matching -- src Project.toml)"
if [[ "$mode" == "build" && -n "$source_status" && "${OSCAR_SYSIMAGE_ALLOW_DIRTY:-0}" != "1" ]]; then
    printf 'Package source/project has uncommitted changes:\n%s\n' "$source_status" >&2
    fail "review the changes, then set OSCAR_SYSIMAGE_ALLOW_DIRTY=1 to preserve the source status in the build receipt"
fi

commit="$(git -C "$repo_root" rev-parse HEAD)"
tree="$(git -C "$repo_root" rev-parse 'HEAD^{tree}')"
branch="$(git -C "$repo_root" branch --show-current)"
julia_requested="${JULIA_BIN:-julia}"

if [[ "$mode" == "dry-run" ]]; then
    printf 'Dry run only; no Julia process, dependency resolution, or files will be created.\n'
    printf 'Repository: %s\nBranch: %s\nCommit: %s\nTree: %s\n' "$repo_root" "${branch:-DETACHED}" "$commit" "$tree"
    printf 'Package source/project status: %s\n' "${source_status:-clean}"
    printf 'Required Julia: %s\nConfigured JULIA_BIN: %s\n' "$REQUIRED_JULIA_VERSION" "$julia_requested"
    printf 'CPU target: %s\n' "${OSCAR_SYSIMAGE_CPU_TARGET:-$DEFAULT_CPU_TARGET}"
    printf 'Planned artifact root: %s\n' "${OSCAR_SYSIMAGE_OUTPUT_DIR:-$SCRIPT_DIR/output}"
    printf 'Manifest: selected at build time by Julia 1.12.4 from the target Project.toml; prepared manifests/preferences are recorded separately from source edits.\n'
    printf 'Planned commands: instantiate %s/Project.toml, extend the verified default Julia image with %s, fresh-process validate, then publish a unique artifact directory.\n' "$SCRIPT_DIR" "$repo_root"
    exit 0
fi

if [[ "$julia_requested" == */* ]]; then
    julia_bin="$julia_requested"
    [[ -x "$julia_bin" ]] || fail "JULIA_BIN is not executable: $julia_bin"
else
    julia_bin="$(command -v -- "$julia_requested" 2>/dev/null)" || fail "JULIA_BIN was not found on PATH: $julia_requested"
    [[ -x "$julia_bin" ]] || fail "JULIA_BIN is not executable: $julia_bin"
fi
julia_bin="$(cd -- "$(dirname -- "$julia_bin")" && pwd -P)/$(basename -- "$julia_bin")"

actual_version="$("$julia_bin" --startup-file=no --history-file=no -e 'print(VERSION)' 2>&1)" || fail "could not query Julia version from $julia_bin: $actual_version"
[[ "$actual_version" == "$REQUIRED_JULIA_VERSION" ]] || fail "Julia $actual_version found at $julia_bin; this artifact requires exactly Julia $REQUIRED_JULIA_VERSION"

runtime_bin="$("$julia_bin" --startup-file=no --history-file=no -e 'print(realpath(joinpath(Sys.BINDIR, Base.julia_exename())))' 2>&1)" || fail "could not determine the Julia executable selected by $julia_bin: $runtime_bin"
[[ -x "$runtime_bin" ]] || fail "Julia reports a non-executable runtime binary: $runtime_bin"
julia_bin="$runtime_bin"

manifest_path="$(OSCAR_GFM_REPO="$repo_root" "$julia_bin" --startup-file=no --history-file=no -e 'project = realpath(joinpath(ENV["OSCAR_GFM_REPO"], "Project.toml")); manifest = Base.project_file_manifest_path(project); manifest === nothing || print(realpath(manifest))' 2>&1)" || fail "could not ask Julia to select the target manifest: $manifest_path"
[[ -n "$manifest_path" && -f "$manifest_path" ]] || fail "Julia $REQUIRED_JULIA_VERSION selected no existing application manifest for $repo_root/Project.toml; prepare the environment explicitly before building"
environment_paths=("JuliaManifest-v1.12.toml" "Manifest-v1.12.toml" "JuliaManifest.toml" "Manifest.toml" "JuliaLocalPreferences.toml" "LocalPreferences.toml")
case "$manifest_path" in
    "$repo_root"/*) environment_paths+=("${manifest_path#"$repo_root"/}") ;;
esac
environment_status="$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all --ignored=matching -- "${environment_paths[@]}")"

image_probe="$("$julia_bin" --startup-file=no --history-file=no -e 'using Libdl; println("loaded=", realpath(unsafe_string(Base.JLOptions().image_file))); println("default=", realpath(joinpath(Sys.BINDIR, "..", "lib", "julia", "sys." * Libdl.dlext)))' 2>&1)" || fail "could not inspect Julia's loaded sysimage: $image_probe"
loaded_image="$(printf '%s\n' "$image_probe" | sed -n 's/^loaded=//p')"
default_image="$(printf '%s\n' "$image_probe" | sed -n 's/^default=//p')"
[[ -n "$loaded_image" && -n "$default_image" ]] || fail "Julia did not report both loaded and default sysimage paths: $image_probe"
[[ "$loaded_image" == "$default_image" ]] || fail "Julia started with a non-default sysimage ($loaded_image); launch the exact Julia executable without a custom -J image"

[[ "$(uname -s)" == "Linux" ]] || fail "build on Oscar Linux; local macOS sysimages are not compatible"
[[ "$(uname -m)" == "x86_64" ]] || fail "build on an Oscar x86_64 CPU node; found $(uname -m)"

cpu_target="${OSCAR_SYSIMAGE_CPU_TARGET:-$DEFAULT_CPU_TARGET}"
[[ -n "$cpu_target" ]] || fail "OSCAR_SYSIMAGE_CPU_TARGET must not be empty"
output_root="${OSCAR_SYSIMAGE_OUTPUT_DIR:-$SCRIPT_DIR/output}"
mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
build_id="${commit:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-$$}"
final_dir="$output_root/$build_id"
[[ ! -e "$final_dir" ]] || fail "artifact directory already exists; choose another output directory: $final_dir"
stage_dir="$(mktemp -d "$output_root/.staging.${build_id}.XXXXXX")"

on_exit() {
    local status=$?
    if [[ $status -ne 0 ]]; then
        printf 'BUILD_FAILED; staging directory and logs preserved at %s\n' "$stage_dir" >&2
    fi
}
trap on_exit EXIT
exec > >(tee -a "$stage_dir/build.log") 2>&1

export OSCAR_GFM_REPO="$repo_root"
export OSCAR_SYSIMAGE_STAGE_DIR="$stage_dir"
export OSCAR_SYSIMAGE_FINAL_DIR="$final_dir"
export OSCAR_SYSIMAGE_CPU_TARGET="$cpu_target"
export OSCAR_SYSIMAGE_JULIA_BIN="$julia_bin"
export OSCAR_SYSIMAGE_BRANCH="${branch:-DETACHED}"
export OSCAR_SYSIMAGE_COMMIT="$commit"
export OSCAR_SYSIMAGE_TREE="$tree"
export OSCAR_SYSIMAGE_ALLOW_DIRTY="${OSCAR_SYSIMAGE_ALLOW_DIRTY:-0}"
export OSCAR_SYSIMAGE_MANIFEST_PATH="$manifest_path"
export OSCAR_SYSIMAGE_ENVIRONMENT_STATUS="$environment_status"

printf 'Target checkout: %s\nBranch: %s\nCommit: %s\nTree: %s\n' "$repo_root" "${branch:-DETACHED}" "$commit" "$tree"
printf 'Selected application manifest: %s\n' "$manifest_path"
printf 'Julia executable: %s\nJulia version: %s\nBase sysimage: %s\n' "$julia_bin" "$actual_version" "$loaded_image"
printf 'Build host: %s %s\nCPU target: %s\n' "$(uname -s)" "$(uname -m)" "$cpu_target"
printf 'Staging: %s\nFinal artifact: %s\n' "$stage_dir" "$final_dir"
printf 'Package source/project status: %s\n' "${source_status:-clean}"
printf 'Application environment status: %s\n' "${environment_status:-clean}"
if type module >/dev/null 2>&1; then
    module list > "$stage_dir/module-list.txt" 2>&1 || true
    cat "$stage_dir/module-list.txt"
else
    printf 'module command is unavailable in this shell; no modules were changed.\n' | tee "$stage_dir/module-list.txt"
fi
export OSCAR_SYSIMAGE_MODULE_LIST="$(cat "$stage_dir/module-list.txt")"

printf '\nInstantiating the pinned PackageCompiler tool environment.\n'
"$julia_bin" --startup-file=no --history-file=no --project="$SCRIPT_DIR" -e 'using Pkg; Pkg.instantiate(; verbose=true)'

printf '\nBuilding the sysimage from the checked-out package project.\n'
JULIA_CPU_TARGET="$cpu_target" "$julia_bin" --startup-file=no --history-file=no --project="$SCRIPT_DIR" "$SCRIPT_DIR/build.jl"

image_path="$stage_dir/GPUFiniteFieldMatrices.so"
[[ -s "$image_path" ]] || fail "PackageCompiler did not produce a nonempty image at $image_path"

printf '\nValidating the image in a fresh Julia process.\n'
OSCAR_GFM_REPO="$repo_root" OSCAR_SYSIMAGE_FILE="$image_path" \
    "$julia_bin" --startup-file=no --history-file=no --project="$repo_root" \
    --sysimage="$image_path" "$SCRIPT_DIR/validate-load.jl"

[[ ! -e "$final_dir" ]] || fail "artifact directory appeared during build; refusing to replace it: $final_dir"
mv -- "$stage_dir" "$final_dir"
stage_dir="$final_dir"
printf '\nPUBLISHED_ARTIFACT_DIR=%s\n' "$final_dir"
printf 'PUBLISHED_SYSIMAGE=%s/GPUFiniteFieldMatrices.so\n' "$final_dir"
printf 'BUILD_LOG=%s/build.log\n' "$final_dir"
