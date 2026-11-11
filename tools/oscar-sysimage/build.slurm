#!/usr/bin/env bash
#SBATCH --job-name=gfm-sysimage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=oscar-sysimage-build-%j.out
#SBATCH --error=oscar-sysimage-build-%j.err

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
[[ -x "$repo_root/tools/oscar-sysimage/build.sh" ]] || {
    echo "No build script at $repo_root/tools/oscar-sysimage/build.sh; set OSCAR_GFM_REPO to the repository root." >&2
    exit 2
}

export OSCAR_GFM_REPO="$repo_root"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    "$repo_root/tools/oscar-sysimage/build.sh"
