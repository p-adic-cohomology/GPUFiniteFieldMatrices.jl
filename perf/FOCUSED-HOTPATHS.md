# Focused addition and square GEMV repeats

The opt-in `:hotpaths` suite contains 15 primary cases:

- `add!` at 31×33, 64×64, 256×256, and 257×255 for Float32 and Float64.
- Square `mul!(z, A, x)` at 32×32, 128×128, and 256×256 for Float32 and Float64.
- The existing 8×8 Float64 PLUQ fixture, using the `pluq-total-materialized-factors-v1` timing contract.

All cases use modulus 101 and deterministic per-case fixtures. Logical and
physical storage geometry, correctness outcomes, synchronized times, spreads,
and CPU/GPU allocations are saved for every requested case. The usual
`quick`/`full` workloads and their identifiers are unchanged, except for
child4's intentional change of the older PLUQ operation ID to
`pluq-total-materialized-factors-v1`.

Allocating `+` is a separate optional diagnostic suite with the same four
addition shapes and both element types (eight cases). It does not contribute to
the primary in-place `add!` results:

```julia
GFMPerf.capture("addition-plus-diagnostic-001"; suite=:addition_diagnostic,
    role="current", samples=100, seconds=5, warmups=10)
```

## Capture three independent brackets

Use the same prepared benchmark environments and harness revision for both
source variants. Run all nine captures sequentially on the same GPU with the
same device, CUDA settings, Julia version, and measurement settings. Start one
fresh Julia process for each capture, then exit it before starting the next.
The review helper checks distinct artifact paths, environment compatibility
across all nine runs, and stable source commit/content hashes within each source
variant. It cannot prove that distinct captures came from distinct Julia
processes; process isolation is an operator requirement recorded in the review.

Follow [README.md](README.md) §1 to clone or update the checkout and prepare the
baseline/current environments with `./perf/setup.sh`. On Oscar, if certificate
discovery fails during setup, export the site's CA bundle before running Git or
Julia package setup:

```sh
export GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt
export JULIA_SSL_CA_ROOTS_PATH=/etc/pki/tls/certs/ca-bundle.crt
```

Use the same Julia 1.12 environment for all nine captures. Julia 1.12.7 was
used for local validation. From the repository root, start the first baseline
process with:

```sh
julia --startup-file=no --project=.perf-regressions/envs/v0.3.1
```

At each prompt, load the same harness and verify the loaded target and project.
For the v0.3.1 baseline process:

```julia
using Revise
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert VERSION.major == 1 && VERSION.minor == 12
VERSION # record the exact Julia patch version
@assert CUDA.functional()
@assert pathof(GPUFiniteFieldMatrices) == joinpath(pwd(), ".perf-regressions/worktrees/v0.3.1/src/GPUFiniteFieldMatrices.jl")
@assert Base.active_project() == joinpath(pwd(), ".perf-regressions/envs/v0.3.1/Project.toml")
GFMPerf.capture("hotpaths-031-before-1"; suite=:hotpaths,
    role="baseline-v0.3.1", samples=100, seconds=5, warmups=10)
```

Exit Julia. For the current source variant, start a fresh process with:

```sh
julia --startup-file=no --project=.perf-regressions/envs/current
```

At the prompt, load the packages, verify the current target/project, then
capture:

```julia
using Revise
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert VERSION.major == 1 && VERSION.minor == 12
VERSION # record the exact Julia patch version
@assert CUDA.functional()
@assert pathof(GPUFiniteFieldMatrices) == joinpath(pwd(), "src/GPUFiniteFieldMatrices.jl")
@assert Base.active_project() == joinpath(pwd(), ".perf-regressions/envs/current/Project.toml")
GFMPerf.capture("hotpaths-current-1"; suite=:hotpaths,
    role="current", samples=100, seconds=5, warmups=10)
```

Exit Julia, start another fresh v0.3.1 process, repeat the first baseline's
imports and target/project checks, then capture the trailing baseline:

```julia
GFMPerf.capture("hotpaths-031-after-1"; suite=:hotpaths,
    role="baseline-v0.3.1", samples=100, seconds=5, warmups=10)
```

Repeat the same baseline → current → baseline sequence twice more in fresh
processes, changing only the unique labels:

| Cycle | Baseline before | Current | Baseline after |
| --- | --- | --- | --- |
| 1 | `hotpaths-031-before-1` | `hotpaths-current-1` | `hotpaths-031-after-1` |
| 2 | `hotpaths-031-before-2` | `hotpaths-current-2` | `hotpaths-031-after-2` |
| 3 | `hotpaths-031-before-3` | `hotpaths-current-3` | `hotpaths-031-after-3` |

Use the exact same capture commands and settings for all nine runs. Keep the
original run folders. Each run should have 15 rows in `summary.csv`; unavailable,
unsupported, incorrect, and failed cases remain visible. `CUDA.functional() ==
false` means GPU timing is unavailable; the harness does not substitute CPU
timings.

## Review every case

From a process loading the same harness revision, point the reviewer at the nine
saved run directories:

```julia
cycles = [
    ("perf/results/hotpaths-031-before-1", "perf/results/hotpaths-current-1", "perf/results/hotpaths-031-after-1"),
    ("perf/results/hotpaths-031-before-2", "perf/results/hotpaths-current-2", "perf/results/hotpaths-031-after-2"),
    ("perf/results/hotpaths-031-before-3", "perf/results/hotpaths-current-3", "perf/results/hotpaths-031-after-3"),
]
review_dir = GFMPerf.review_bracketed_cycles(cycles; label="hotpaths-review-001")
```

The default screening threshold is 0%. Each current capture is compared to both
neighboring baselines, so the report contains six pair effects per case. A
`confirmed-slowdown` requires every pair to clear its own noise rule and the
three independent cycle ratios to agree within the measured pair noise. Smaller
positive changes remain visible as `positive-change-observed`; they are not
hidden by a percentage threshold or described as accepted. Baseline drift,
cross-cycle disagreement, insufficient samples, correctness failures, and any
environment or source-identity mismatch remain inconclusive or incompatible.

The review is read-only over capture inputs and creates a separate directory
with `review.json`. Return all nine capture folders and that review directory.
The JSON records each run's status, detail,
correctness, median, spread, and allocation data, plus all six pair medians in
microseconds, ratios, percentage changes, spreads, and CPU/GPU allocation
deltas.

Captures include source and harness hashes. If harness code changes, capture
both source variants again with the same updated harness before reviewing;
older captures will be incompatible. The baseline commit and content hash must
stay fixed across all six baseline captures, and the current commit and content
hash must stay fixed across its three captures.
