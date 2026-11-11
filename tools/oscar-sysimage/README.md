# GPUFiniteFieldMatrices sysimage on Oscar

This tooling builds a Julia 1.12.4 sysimage from the checked-out local
`GPUFiniteFieldMatrices` package and its resolved CUDA dependencies. It does not
change the package's production project, manifest, preferences, tests, or
algorithms. PackageCompiler 2.4.0 is pinned in this separate tool environment;
its project declares Julia 1.10 compatibility, and the build script verifies
the exact required Julia version before doing any work.

The build runs on an Oscar Linux x86_64 CPU node with a portable `generic` CPU
target by default. The precompile workload imports the package and exercises a
small CPU-backed Karatsuba conversion. It does not allocate a GPU array or
create a CUDA context. The image therefore contains host code and package
dependencies, not cached kernels for a particular GPU. CUDA kernels and
hardware-specific execution can still compile at runtime.

For a build intentionally tied to a known CPU model, set
`OSCAR_SYSIMAGE_CPU_TARGET` to a target supported by that Julia build. A
`native` build may require the same CPU features at load time, so `generic` is
the portable default for Oscar's heterogeneous x86_64 nodes. CPU targeting does
not make sysimages interchangeable across operating systems or different Julia
builds.

## Prepare a checkout

Clone the manual branch on Oscar and move into the checkout:

```sh
git clone --branch manual-testing/oscar-sysimage \
  https://github.com/p-adic-cohomology/GPUFiniteFieldMatrices.jl.git
cd GPUFiniteFieldMatrices.jl
```

Oscar's Julia modules can change. Check the available names instead of assuming
that a `julia/1.12.4` module exists:

```sh
module spider julia
module list
```

Use an available Julia 1.12.4 module if the site lists one, or point directly
to an exact Julia 1.12.4 executable:

```sh
export JULIA_BIN=/absolute/path/to/julia-1.12.4/bin/julia
export OSCAR_GFM_REPO="$PWD"
```

The root `Manifest.toml` may not be tracked by the repository. Check whether
one is present and whether its `julia_version` is compatible with Julia 1.12.
When a compatible manifest is already supplied, instantiate those exact
versions before submitting the build:

```sh
"$JULIA_BIN" --startup-file=no --project="$OSCAR_GFM_REPO" \
  -e 'using Pkg; Pkg.instantiate()'
```

This instantiates the existing locked environment without requesting a
dependency upgrade.

If no manifest is present or the existing one was resolved with another Julia
minor version, explicitly prepare a Julia 1.12 environment using the exact
Julia executable, review the resolved dependency set, and retain that manifest
with the build artifact. For example:

```sh
"$JULIA_BIN" --startup-file=no --project="$OSCAR_GFM_REPO" \
  -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

That command creates or updates the checkout's `Manifest.toml`; inspect the
resolved dependencies and preferences. Then acknowledge the reviewed
application environment and run the builder with
`OSCAR_SYSIMAGE_ALLOW_DIRTY=1`. The builder records tracked, untracked, or
ignored status and a SHA-256 hash for the exact manifest it uses, and copies it
beside the image. It does not resolve dependencies during the build. The local
package itself is selected by the checkout's `Project.toml` and source path; a
self-entry in the manifest is optional, but if present it must point to this
checkout rather than a registered version.

For example, after reviewing the manifest:

```sh
export OSCAR_SYSIMAGE_ALLOW_DIRTY=1
```

PackageCompiler is isolated from the application environment. The application
Manifest and any root `LocalPreferences.toml` are copied to the artifact
receipt. Effective package preferences are recorded without overriding them.
The builder uses the normal Julia distribution sysimage and refuses a custom
base image, a different Julia patch, a non-Linux host, or a non-x86_64 host.
An explicit `JULIA_CC=/path/to/cc` can select a C compiler; otherwise
PackageCompiler uses normal compiler discovery. Keep the caller's loaded
modules and CUDA-related environment. CUDA.jl normally manages its toolkit from
artifacts and needs a working NVIDIA driver at runtime; this workflow does not
load a CUDA module or force a CUDA runtime version.

## Submit a CPU build

The default artifact directory is
`tools/oscar-sysimage/output/<commit>-<UTC-time>-<job-id>/`. Set a different
output directory if desired:

```sh
export OSCAR_SYSIMAGE_OUTPUT_DIR=/absolute/path/to/persistent/output
sbatch tools/oscar-sysimage/build.slurm
```

The batch file requests one node, four CPUs, 32 GB, and six hours, then invokes
the builder with `srun`. Oscar's `sbatch` command-line options override the
script defaults. For account or partition settings available to your account,
pass them explicitly, for example:

```sh
sbatch --time=04:00:00 --cpus-per-task=8 --mem=48G \
  --partition=<available-partition> --account=<your-account> \
  tools/oscar-sysimage/build.slurm
```

The submission command's current directory is the default `SLURM_SUBMIT_DIR`.
`OSCAR_GFM_REPO` is the preferred absolute checkout path; `build.slurm` uses it
even when Slurm spools the submitted script or starts it from another
directory. The Slurm output files are written in the submission directory.
The build log and complete provenance are also kept in the published artifact.
The job prints `PUBLISHED_ARTIFACT_DIR` and `PUBLISHED_SYSIMAGE` when the
fresh-process load check succeeds. Failed builds leave their unique staging
directory and logs intact, and do not replace a previous artifact.

The precompile workload and package import are host-only, but the final
PackageCompiler/CUDA code-generation path could not be verified on Oscar from
this workstation. If a CPU job fails while producing the image, its log remains
in the staging directory and no artifact is published. If the target CUDA stack
needs a functional driver during image creation, submit the same builder to an
allocated GPU node explicitly:

```sh
sbatch -p gpu --gres=gpu:1 tools/oscar-sysimage/build.slurm
```

This requests GPU resources for the build; the separate GPU smoke job still
checks the resulting image and actual device execution.

To inspect the artifact in a clean Julia process, use the exact same executable
and project:

```sh
export OSCAR_SYSIMAGE_FILE=/absolute/path/to/artifact/GPUFiniteFieldMatrices.so
srun --ntasks=1 "$JULIA_BIN" --startup-file=no --history-file=no \
  --project="$OSCAR_GFM_REPO" --sysimage="$OSCAR_SYSIMAGE_FILE" \
  "$OSCAR_GFM_REPO/tools/oscar-sysimage/validate-load.jl"
```

This checks Julia 1.12.4, the active project, the loaded image path, package
source path, included direct dependencies, and a small CPU helper result.

## Submit the separate GPU smoke check

Set the artifact path, then submit the GPU job:

```sh
export OSCAR_SYSIMAGE_FILE=/absolute/path/to/artifact/GPUFiniteFieldMatrices.so
sbatch tools/oscar-sysimage/gpu-smoke.slurm
```

The script requests `-p gpu --gres=gpu:1`. It reports `CUDA.versioninfo()` and
the selected device, then runs deterministic `CuModMatrix` addition and a
small GPU-backed Karatsuba matrix product. Both are synchronized and compared
with host-computed expected values. A missing or nonfunctional CUDA device is a
failed smoke check; it is not skipped or reported as success.

## Artifact lifetime and rebuilds

The image is tied to Julia 1.12.4, the target checkout, its resolved package
versions, effective preferences, and the Linux build platform. Keep the
checkout, project files, Julia depot, and CUDA artifacts available at load
time. It is not a relocatable application. Updating the package source,
dependencies, preferences, or Julia requires a rebuild. Revise is not assumed
to replace code baked into the image.

Oscar documents `~/scratch` as subject to deletion for files not accessed for
30 days. Move an artifact and the required environment to suitable persistent
storage if they must last longer; do not assume a username-specific storage
path.

References: [Oscar batch jobs](https://docs.ccv.brown.edu/oscar/submitting-jobs/batch),
[GPU jobs](https://docs.ccv.brown.edu/oscar/gpu-computing/gpus),
[Slurm partitions](https://docs.ccv.brown.edu/oscar/submitting-jobs/slurm),
[Oscar module system](https://docs.ccv.brown.edu/oscar/sys-changes/lmod-new-module-system),
[Oscar filesystems](https://docs.ccv.brown.edu/oscar/managing-files/filesystem),
[PackageCompiler sysimages](https://julialang.github.io/PackageCompiler.jl/stable/sysimages.html),
[PackageCompiler API](https://julialang.github.io/PackageCompiler.jl/stable/refs.html),
[PackageCompiler 2.4.0 project compatibility](https://github.com/JuliaLang/PackageCompiler.jl/blob/v2.4.0/Project.toml),
[CUDA.jl installation overview](https://cuda.juliagpu.org/stable/installation/overview/).
