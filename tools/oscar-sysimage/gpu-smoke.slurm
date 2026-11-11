#!/usr/bin/env bash
#SBATCH --job-name=gfm-gpu-smoke
#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=oscar-gpu-smoke-%j.out
#SBATCH --error=oscar-gpu-smoke-%j.err

set -Eeuo pipefail

repo_root="${OSCAR_GFM_REPO:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$repo_root" ]] || {
    echo "Set OSCAR_GFM_REPO to the absolute checkout path before sbatch." >&2
    exit 2
}
[[ -d "$repo_root" ]] || {
    echo "OSCAR_GFM_REPO does not exist: $repo_root" >&2
    exit 2
}
repo_root="$(cd -- "$repo_root" && pwd -P)"
[[ -f "$repo_root/tools/oscar-sysimage/gpu-smoke.jl" ]] || {
    echo "No GPU smoke script under $repo_root/tools/oscar-sysimage." >&2
    exit 2
}
[[ -n "${OSCAR_SYSIMAGE_FILE:-}" && -f "$OSCAR_SYSIMAGE_FILE" ]] || {
    echo "Set OSCAR_SYSIMAGE_FILE to a built sysimage before sbatch." >&2
    exit 2
}

export OSCAR_GFM_REPO="$repo_root"
export OSCAR_SYSIMAGE_FILE="$(cd -- "$(dirname -- "$OSCAR_SYSIMAGE_FILE")" && pwd -P)/$(basename -- "$OSCAR_SYSIMAGE_FILE")"
julia_bin="${JULIA_BIN:-julia}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" \
    "$julia_bin" --startup-file=no --history-file=no --project="$repo_root" \
    --sysimage="$OSCAR_SYSIMAGE_FILE" "$repo_root/tools/oscar-sysimage/gpu-smoke.jl"
