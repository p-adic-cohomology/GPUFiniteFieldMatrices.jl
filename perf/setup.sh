#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/.." && pwd)"
run_root="${1:-$repo/.perf-regressions}"
cuda_version="${GFM_PERF_CUDA_VERSION:-5.11.3}"
julia_bin="${GFM_JULIA:-julia}"
remote_url="$(git -C "$repo" remote get-url origin)"
expected_url="https://github.com/p-adic-cohomology/GPUFiniteFieldMatrices.jl"
clean_url="${remote_url%.git}"

if [[ "$clean_url" != "$expected_url" ]]; then
    echo "origin must be $expected_url (got $remote_url)" >&2
    exit 2
fi

version_output="$("$julia_bin" --version)"
if [[ "$version_output" != *"1.12."* ]]; then
    echo "Julia 1.12 is required; got: $version_output" >&2
    exit 2
fi

run_root="$(mkdir -p "$(dirname "$run_root")" && cd "$(dirname "$run_root")" && pwd)/$(basename "$run_root")"
owner_file="$run_root/.gfm-perf-owner.toml"
if [[ ! -e "$run_root" ]]; then
    mkdir -p "$run_root"
fi
if [[ ! -f "$owner_file" ]]; then
    first_entry="$(find "$run_root" -mindepth 1 -maxdepth 1 -print -quit)"
    if [[ -n "$first_entry" ]]; then
        echo "refusing to reuse non-empty unowned run root: $run_root" >&2
        exit 2
    fi
    cat > "$owner_file" <<EOF
schema_version = 1
source_remote = "$clean_url"
source_repo = "$repo"
created_by = "perf/setup.sh"
EOF
else
    "$julia_bin" --startup-file=no -e 'using TOML; x=TOML.parsefile(ARGS[1]); x["schema_version"] == 1 && x["source_remote"] == ARGS[2] || error("run root ownership marker does not match this project")' "$owner_file" "$clean_url"
fi

git -C "$repo" fetch --tags origin
pin_031="7192d479d2c41d31021a4938beec8223d16d6654"
pin_032="9bb9828f3ee69ab9e60df32201f6c2560f20c36b"
for spec in "v0.3.1:$pin_031" "v0.3.2:$pin_032"; do
    tag="${spec%%:*}"
    expected="${spec#*:}"
    actual="$(git -C "$repo" rev-parse "$tag^{commit}")"
    if [[ "$actual" != "$expected" ]]; then
        echo "$tag resolves to $actual; expected pinned commit $expected" >&2
        exit 2
    fi
done

worktree_root="$run_root/worktrees"
mkdir -p "$worktree_root"
registry="$run_root/worktrees.toml"
if [[ -f "$registry" ]]; then
    "$julia_bin" --startup-file=no -e 'using TOML; x=TOML.parsefile(ARGS[1]); x["source_repo"] == ARGS[2] || error("worktree registry belongs to a different source checkout")' "$registry" "$repo"
else
    cat > "$registry" <<EOF
schema_version = 1
source_repo = "$repo"

[worktrees]
EOF
fi

ensure_worktree() {
    local name="$1"
    local commit="$2"
    local path="$worktree_root/$name"
    local registered
    registered="$("$julia_bin" --startup-file=no -e 'using TOML; x=TOML.parsefile(ARGS[1]); println(get(get(x, "worktrees", Dict()), ARGS[2], ""))' "$registry" "$name")"
    if [[ -e "$path" ]]; then
        if [[ "$registered" != "$commit" ]]; then
            echo "refusing to reuse unregistered or differently pinned path: $path" >&2
            exit 2
        fi
        if [[ ! -f "$path/.git" ]]; then
            echo "refusing to reuse a directory that is not a linked worktree: $path" >&2
            exit 2
        fi
        local common head status
        common="$(git -C "$path" rev-parse --path-format=absolute --git-common-dir)"
        head="$(git -C "$path" rev-parse HEAD)"
        status="$(git -C "$path" status --porcelain --untracked-files=all)"
        if [[ "$common" != "$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)" || "$head" != "$commit" || -n "$status" ]]; then
            echo "existing owned worktree does not match its pin or is dirty: $path" >&2
            exit 2
        fi
    else
        if [[ -n "$registered" ]]; then
            echo "worktree registry points to missing path: $path; refusing to recreate it automatically" >&2
            exit 2
        fi
        git -C "$repo" worktree add --quiet --detach "$path" "$commit"
        "$julia_bin" --startup-file=no -e 'using TOML; p=ARGS[1]; n=ARGS[2]; c=ARGS[3]; x=TOML.parsefile(p); x["worktrees"][n]=c; open(p,"w") do io; TOML.print(io,x); end' "$registry" "$name" "$commit"
    fi
    printf '%s\n' "$path"
}

v031="$(ensure_worktree v0.3.1 "$pin_031")"
v032="$(ensure_worktree v0.3.2 "$pin_032")"

"$julia_bin" --startup-file=no "$script_dir/setup_envs.jl" "$run_root" "$repo" "$v031" "$v032" "$cuda_version"

cat <<EOF

Benchmark environments are ready:
  v0.3.1: $run_root/envs/v0.3.1
  v0.3.2: $run_root/envs/v0.3.2
  current: $run_root/envs/current
Results default to: $repo/perf/results
CUDA.jl pin: $cuda_version
EOF
