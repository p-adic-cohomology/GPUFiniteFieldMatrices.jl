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

The build asks Julia 1.12.4 itself which manifest the checkout uses. A
`Manifest-v1.12.toml` takes precedence over `Manifest.toml`; Julia-specific
`JuliaManifest-v1.12.toml` and generic `JuliaManifest.toml` names are also
recognized. The selected path and hash are recorded with the image, so
versioned-only, generic-only, and both-present checkouts are handled
consistently by the builder and validators.

You can inspect the active path and its recorded Julia version without running
Pkg or changing files:

```sh
"$JULIA_BIN" --startup-file=no -e 'p = realpath(joinpath(ENV["OSCAR_GFM_REPO"], "Project.toml")); m = Base.project_file_manifest_path(p); println(m === nothing ? "no manifest selected" : m)'
```

Prepare the application environment on an allocated compute node. Pkg actions
can trigger automatic precompilation, so disable that for this setup step and
leave the actual compilation to the sysimage build job:

```sh
salloc -N1 --cpus-per-task=4 --mem=32G --time=02:00:00
export JULIA_PKG_PRECOMPILE_AUTO=0
srun --ntasks=1 "$JULIA_BIN" --startup-file=no \
  --project="$OSCAR_GFM_REPO" -e 'using Pkg; Pkg.instantiate(; verbose=true)'
```

If Julia reports no selected manifest, resolve and instantiate the project on
the allocated node, then inspect the resulting dependency set before building:

```sh
srun --ntasks=1 "$JULIA_BIN" --startup-file=no \
  --project="$OSCAR_GFM_REPO" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(; verbose=true)'
```

If Julia selects a generic manifest resolved for another Julia minor version,
preserve that file and prepare a compatible `Manifest-v1.12.toml` explicitly
with Julia 1.12.4. Confirm the selector reports the versioned path before
building. The builder rejects an incompatible lock instead of resolving or
updating the application environment itself.

Prepared manifests and local preference files are routine environment inputs;
they do not require `OSCAR_SYSIMAGE_ALLOW_DIRTY=1`. The builder records their
status separately from package source/project edits, hashes and copies the
manifest Julia selected plus local preferences, and does not resolve or update
the application environment. Set `OSCAR_SYSIMAGE_ALLOW_DIRTY=1` only when you
have reviewed intentional edits under `src/` or to the package `Project.toml`.
The local package itself is selected by the checkout's project and source path;
a self-entry in the manifest is optional, but if present it must point to this
checkout rather than a registered version.

PackageCompiler is isolated from the application environment. The selected
application manifest and any root Julia/LocalPreferences files are copied into
the artifact provenance. Effective package preferences are recorded without
overriding them. The builder uses the normal Julia distribution sysimage and
refuses a custom base image, a different Julia patch, a non-Linux host, or a
non-x86_64 host. An explicit `JULIA_CC=/path/to/cc` can select a C compiler;
otherwise PackageCompiler uses normal compiler discovery. Keep the caller's
loaded modules and CUDA-related environment. CUDA.jl normally manages its
toolkit from artifacts and needs a working NVIDIA driver at runtime; this
workflow does not load a CUDA module or force a CUDA runtime version.

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

This checks Julia 1.12.4, the active project and Julia-selected manifest, the
loaded image path, current source and manifest/preferences fingerprints,
included direct dependencies, and a small CPU helper result.

## Submit the separate GPU smoke check

Set the artifact path, then submit the GPU job:

```sh
export OSCAR_SYSIMAGE_FILE=/absolute/path/to/artifact/GPUFiniteFieldMatrices.so
sbatch tools/oscar-sysimage/gpu-smoke.slurm
```

The script requests `-p gpu --gres=gpu:1`. Before touching CUDA, it runs the
same receipt verifier as the fresh-process load check, including current source
and environment fingerprints. It reports `CUDA.versioninfo()` and the selected
device, then runs deterministic `CuModMatrix` addition and a small GPU-backed
Karatsuba matrix product. Both are synchronized and compared with host-computed
expected values. A missing or nonfunctional CUDA device is a failed smoke
check; it is not skipped or reported as success.

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
