# Manual CUDA performance regressions

This harness captures correctness-checked, synchronized timings for the public
`GPUFiniteFieldMatrices` operations. It is intended for a CUDA Linux machine and
a long-lived Julia REPL with Revise. It does not add CI checks or change
`Pkg.test()`.

The primary baseline is **v0.3.1**; v0.3.2 is an additional comparison. The goal
is to avoid regressions from v0.3.1 in correctness, runtime, and allocations.
The 10% flag is a screening threshold, not an allowance for smaller regressions:
review positive timing deltas and allocation growth, then repeat suspected
changes in fresh processes before drawing conclusions.

This branch includes main at `47d164a31a3e8c01b2adae6d293feb68468c2c46`
(the PR #85 merge, 2026-11-18). Captures record the actual checkout commit and
source fingerprint, so the unchanged package version cannot hide stale source.

## One-command focused hot-path run

With Julia 1.12 and an existing functional CUDA allocation, run this from the
checkout:

```sh
./perf/run_hotpaths.sh
```

By default it runs `perf/setup.sh`, then launches nine fresh Julia processes
for three baseline → current → baseline cycles. Every capture uses 100 samples,
5 seconds, and 10 warmups; the existing reviewer writes `review.json` after all
nine captures pass their saved case oracles. Use `--skip-setup` when the
benchmark environments are already prepared, or `--prefix NAME` to choose a
safe output prefix. The settings are fixed by the runner.

Captures and reviews go under `perf/results/`; phase logs go under
`perf/results/.hotpaths-runs/NAME/logs/`. Existing destinations are preserved
and cause the run to stop. A capture or review failure stops later phases and
leaves completed artifacts and logs in place. The runner does not pull, switch
branches, submit GPU jobs, or replace CUDA timings with CPU timings. See
[FOCUSED-HOTPATHS.md](FOCUSED-HOTPATHS.md) for the manual sequence and review
details.

## 1. Clone and prepare the three environments

Install Julia 1.12 and a working NVIDIA driver, then run:

```sh
git clone --branch manual-testing/perf-regressions --single-branch \
  https://github.com/p-adic-cohomology/GPUFiniteFieldMatrices.jl.git
cd GPUFiniteFieldMatrices.jl
./perf/setup.sh
```

The setup verifies the frozen v0.3.1 and v0.3.2 commit pins, creates detached
tag worktrees under `.perf-regressions/worktrees`, and creates independent
benchmark environments under `.perf-regressions/envs`. CUDA.jl 5.11.3 is the
default shared version. To try a different CUDA.jl version, set
`GFM_PERF_CUDA_VERSION` before setup; the captures will only compare when the
resolved relevant dependencies match. Setup refuses unowned directories and
modified or wrongly pinned baseline worktrees. It leaves existing results
untouched; use unique capture labels. Use a new checkout or run root after
moving the clone to another path.

For an existing clone, exit any Julia processes using it, then update and rerun
setup from the repository root:

```sh
git switch manual-testing/perf-regressions
git pull --ff-only origin manual-testing/perf-regressions
./perf/setup.sh
```

This preserves saved results and the frozen release worktrees. If Git reports
local edits or a diverged branch, preserve that work and use a separate clone.
Restart Julia after the update; an already-loaded package will not become the
new checkout merely because Git changed. Saved baselines remain usable when
the harness fingerprint, environment, device, and measurement settings match.

Each variant environment lists `Revise`, `BenchmarkTools`, `JSON3`, and `CUDA`
as direct dependencies so the documented `Main` imports work. Setup verifies
that these versions match across variants and repairs missing direct entries in
an already-owned ready environment without replacing its worktrees or results.

For a focused fresh-process repeat workflow covering in-place addition, square
GEMV, and total PLUQ, see [FOCUSED-HOTPATHS.md](FOCUSED-HOTPATHS.md).

Before capturing, check the driver and selected device in Julia:

```julia
using CUDA
CUDA.functional() || error("CUDA is not functional on this machine")
CUDA.versioninfo()
```

## 2. Capture each baseline in its own fresh process

Start from the repository root. Do not change a package path or dependency in a
process after loading `GPUFiniteFieldMatrices`.

For v0.3.1:

```sh
julia --startup-file=no --project=.perf-regressions/envs/v0.3.1
```

At the Julia prompt, load Revise before the target package and capture:

```julia
using Revise
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert CUDA.functional()
pathof(GPUFiniteFieldMatrices)  # should be the v0.3.1 tag worktree
Base.active_project()           # should be .perf-regressions/envs/v0.3.1/Project.toml
baseline031 = GFMPerf.capture("baseline-031-001";
    suite=:quick, role="baseline-v0.3.1", samples=20, seconds=2, warmups=3)
```

Exit Julia. Repeat in another fresh process with the v0.3.2 environment:

```sh
julia --startup-file=no --project=.perf-regressions/envs/v0.3.2
```

```julia
using Revise
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert CUDA.functional()
pathof(GPUFiniteFieldMatrices)  # should be the v0.3.2 tag worktree
baseline032 = GFMPerf.capture("baseline-032-001";
    suite=:quick, role="baseline-v0.3.2", samples=20, seconds=2, warmups=3)
```

Each label is a new result directory under `perf/results/`. The returned value
and printed output are the saved path. Record the label or path before exiting
Julia; REPL variables do not survive between processes. A label cannot be
reused unless `overwrite=true` is passed explicitly. Keep these baseline
directories for later comparisons.

## 3. Capture and revise the development checkout

Start a fresh process using the current editable checkout environment. Launch
it from the repository root so the saved paths below resolve consistently:

```sh
julia --startup-file=no --project=.perf-regressions/envs/current
```

```julia
using Revise                   # load before the target package
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert CUDA.functional()
pathof(GPUFiniteFieldMatrices)  # should be this checkout's src/GPUFiniteFieldMatrices.jl
repo_root = pwd()
baseline031_path = joinpath(repo_root, "perf", "results", "baseline-031-001")
baseline032_path = joinpath(repo_root, "perf", "results", "baseline-032-001")
dev_before_path = GFMPerf.capture("dev-before-edit-001";
    suite=:quick, role="current", samples=20, seconds=2, warmups=3)
```

Edit package source in another terminal. At the REPL prompt, let Revise pick up
the edits (or call `Revise.revise()`), then take a new capture:

```julia
Revise.revise()
dev_after_path = GFMPerf.capture("dev-after-edit-001";
    suite=:quick, role="current", samples=20, seconds=2, warmups=3)
cmp031_path = GFMPerf.compare(dev_after_path, baseline031_path; threshold=0.10, min_samples=5)
cmp032_path = GFMPerf.compare(dev_after_path, baseline032_path; threshold=0.10, min_samples=5)
```

If Julia exits before comparison or a rerun, rehydrate the paths in the new
REPL from their saved labels; do not expect `baseline031_path` or
`cmp031_path` variables to persist. For the labels above, for example:

```julia
repo_root = pwd()
baseline031_path = joinpath(repo_root, "perf", "results", "baseline-031-001")
baseline032_path = joinpath(repo_root, "perf", "results", "baseline-032-001")
dev_after_path = joinpath(repo_root, "perf", "results", "dev-after-edit-001")
cmp031_path = joinpath(repo_root, "perf", "results", "comparisons",
    "dev-after-edit-001-vs-baseline-031-001")
cmp032_path = joinpath(repo_root, "perf", "results", "comparisons",
    "dev-after-edit-001-vs-baseline-032-001")
```

Use the exact labels you captured if they differ. Capture and comparison calls
return their artifact paths; keep those paths with the run labels.

The capture rebuilds deterministic fixtures and benchmark closures from the
current Julia world on every call; it does not rely on top-level `includet`
side effects. Restart after changing dependencies, project paths, or a change
that affects the loaded package layout. Use the same harness revision and
measurement settings for both baselines and development. If you edit the
harness itself, recapture both baselines in fresh processes before comparing.

### Focused total PLUQ recapture

The existing diagonal fixture case ID is
`group=inverse-pluq|op=pluq-total-materialized-factors-v1|dtype=Float64|dims=8x8|mod=101|split=none|layout=package-default|seed=607`.

In separate fresh Julia processes, use the same updated harness to capture
this ID from the v0.3.1 and current environments with matching settings:

```julia
# In a fresh v0.3.1 process:
pluq_case_id = "group=inverse-pluq|op=pluq-total-materialized-factors-v1|dtype=Float64|dims=8x8|mod=101|split=none|layout=package-default|seed=607"
baseline_pluq = GFMPerf.capture("baseline-031-pluq-total-001";
    suite=:quick, case_ids=[pluq_case_id], role="baseline-v0.3.1",
    samples=20, seconds=2, warmups=3)

# After starting a fresh current process, recreate the case ID and capture:
pluq_case_id = "group=inverse-pluq|op=pluq-total-materialized-factors-v1|dtype=Float64|dims=8x8|mod=101|split=none|layout=package-default|seed=607"
current_pluq = GFMPerf.capture("current-pluq-total-001";
    suite=:quick, case_ids=[pluq_case_id], role="current",
    samples=20, seconds=2, warmups=3)

# Rehydrate paths in this process and compare the paired captures:
repo_root = pwd()
baseline_pluq_path = joinpath(repo_root, "perf", "results", "baseline-031-pluq-total-001")
current_pluq_path = joinpath(repo_root, "perf", "results", "current-pluq-total-001")
comparison_pluq = GFMPerf.compare(current_pluq_path, baseline_pluq_path)
```

Run each capture in its named environment and process, and rehydrate the saved
paths before comparing after a restart.
Do not compare either new capture with old PLUQ rows: the case ID and
measurement contract changed, so recapture both sides with this same harness.

To narrow work, pass `groups=["gemm", "gemv"]`. For the larger opt-in set,
including 1024 and 5000 square products, pass `suite=:full`. You can change
`samples`, `seconds`, and `warmups`; use identical values across captures.

## 4. Rerun a suspected case in fresh processes

`GFMPerf.compare` returns and prints the comparison directory. A slowdown above
10% that clears the within-run noise estimate is marked
`flagged-unconfirmed`; it is not called a confirmed regression. In a fresh
development REPL, first rehydrate the comparison path if this is a new
process, then rerun only those cases:

```julia
repo_root = pwd()
cmp031_path = joinpath(repo_root, "perf", "results", "comparisons",
    "dev-after-edit-001-vs-baseline-031-001")
rerun_dev_path = GFMPerf.rerun_flagged(cmp031_path; label="dev-rerun-001",
    samples=40, seconds=4, warmups=3)
```

Capture the same case IDs from the v0.3.1 environment in its own fresh process.
Rehydrate the comparison path in that process and read the IDs through the
harness helper:

```julia
repo_root = pwd()
cmp031_path = joinpath(repo_root, "perf", "results", "comparisons",
    "dev-after-edit-001-vs-baseline-031-001")
ids = GFMPerf.flagged_case_ids(cmp031_path)
baseline031_rerun_path = GFMPerf.capture("baseline-031-rerun-001";
    suite=:full, case_ids=ids, role="baseline-v0.3.1",
    samples=40, seconds=4, warmups=3)
```

For confirmation, use at least three fresh-process baseline → candidate →
baseline cycles. In each process, rehydrate the saved baseline and candidate
capture paths by label before comparing. After the cycles, rehydrate the three
comparison paths in any harness-loaded REPL from the labels used for those
comparisons:

```julia
repo_root = pwd()
cmp1_path = joinpath(repo_root, "perf", "results", "comparisons", "candidate-1-vs-baseline-before-1")
cmp2_path = joinpath(repo_root, "perf", "results", "comparisons", "candidate-2-vs-baseline-before-2")
cmp3_path = joinpath(repo_root, "perf", "results", "comparisons", "candidate-3-vs-baseline-before-3")
confirmation = GFMPerf.confirm([cmp1_path, cmp2_path, cmp3_path])
```

Replace the example labels with the exact capture labels in your runs.
Confirmation requires all independent pair medians to cross the configured
threshold and the repeated change to clear the measured noise rule. Use the
same flagged case IDs and settings throughout.

## 5. Read the artifacts

Each `perf/results/<label>/` directory contains:

- `run.json`: case outcomes, raw synchronized time and GC samples, scalar
  BenchmarkTools CPU allocation minima, structured GPU allocation results, and
  run metadata.
- `summary.csv`: one row for every requested case, including unavailable,
  unsupported, incorrect, or failed cases.
- `metadata.json`, `Project.toml`, and `Manifest.toml`: target and harness
  fingerprints, actual package path and project, resolved versions, hardware,
  preferences, and the resolved environment.

Comparison directories contain `comparison.json`, `comparison.csv`, and a
readable `comparison.txt`. Timing ratios are current divided by baseline;
positive percentages are slowdowns. CPU allocation counts/bytes come from
BenchmarkTools. `Trial.memory` and `Trial.allocs` are scalar minimum byte/count
values across the measured samples; BenchmarkTools does not expose these as
per-sample arrays. GPU counts/bytes are collected in a separate warmed
`CUDA.@timed` pass from `gpu_bytes` and `gpu_memstats.alloc_count`, and the
structured accounting values are kept in `run.json`. Missing GPU accounting is
`null`, not zero. Pool-reserved memory is not reported as an operation
allocation.

`unsupported` means the operation or adapter is unavailable in that package
revision. `incorrect` means the correctness oracle rejected its output. Neither
gets a speed ratio. `incompatible` means the relevant versions, device,
preferences, launch settings, measurement policy, or harness fingerprint did
not match. `inconclusive` means there were too few samples or the observed
change did not clear the measured noise. When CUDA is unavailable, every case
is recorded as unavailable and no CPU timing is presented as GPU performance.

Karatsuba cases use deterministic low-plane inputs for matching Float64
comparisons. Historical Float32 Karatsuba cases are explicitly unsupported;
the current-only Float32 cases are recorded separately. The known
interacting-high-plane GEMV limitation is not used to produce a favorable
comparison.

Large GEMM correctness checks switch from scalar BigInt products to Float64
BLAS only when a rigorous exactness guard proves every nonnegative integer
product and partial sum is at most `2^53`; nine deterministic corner/center
outputs are also checked with BigInt. If the input or bound is unsafe, the case
is reported failed rather than compared with an approximate oracle.

See [SPEC.md](SPEC.md) for the workload, correctness, compatibility, and
reporting contract. References: [CUDA.jl profiling](https://cuda.juliagpu.org/stable/development/profiling/),
[BenchmarkTools manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/),
and [Revise user reference](https://timholy.github.io/Revise.jl/stable/user_reference/).

The reporting and stable-case unit tests run without a functional GPU:

```sh
julia --startup-file=no --project=.perf-regressions/envs/current perf/test/runtests.jl
```
