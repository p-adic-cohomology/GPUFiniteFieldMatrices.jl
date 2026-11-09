# Manual CUDA performance regressions

This harness captures correctness-checked, synchronized timings for the public
`GPUFiniteFieldMatrices` operations. It is intended for a CUDA Linux machine and
a long-lived Julia REPL with Revise. It does not add CI checks or change
`Pkg.test()`.

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

Before capturing, check the driver and selected device in Julia:

```julia
using CUDA
CUDA.functional() || error("CUDA is not functional on this machine")
CUDA.versioninfo()
```

## 2. Capture each baseline in its own fresh process

Start from the repository root. Do not change a package path or dependency in a
process after loading `GPUFiniteFieldMatrices`.

For v0.3.2:

```sh
julia --startup-file=no --project=.perf-regressions/envs/v0.3.2
```

At the Julia prompt, load Revise before the target package and capture:

```julia
using Revise
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert CUDA.functional()
pathof(GPUFiniteFieldMatrices)  # should be the v0.3.2 tag worktree
Base.active_project()           # should be .perf-regressions/envs/v0.3.2/Project.toml
baseline032 = GFMPerf.capture("baseline-032-001";
    suite=:quick, role="baseline-v0.3.2", samples=20, seconds=2, warmups=3)
```

Exit Julia. Repeat in another fresh process with the v0.3.1 environment:

```sh
julia --startup-file=no --project=.perf-regressions/envs/v0.3.1
```

```julia
using Revise
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf
@assert CUDA.functional()
pathof(GPUFiniteFieldMatrices)  # should be the v0.3.1 tag worktree
baseline031 = GFMPerf.capture("baseline-031-001";
    suite=:quick, role="baseline-v0.3.1", samples=20, seconds=2, warmups=3)
```

Each label is a new result directory under `perf/results/`. A label cannot be
reused unless `overwrite=true` is passed explicitly. Keep these baseline
directories for later comparisons.

## 3. Capture and revise the development checkout

Start a fresh process using the current editable checkout environment:

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
dev_before = GFMPerf.capture("dev-before-edit-001";
    suite=:quick, role="current", samples=20, seconds=2, warmups=3)
```

Edit package source in another terminal. At the REPL prompt, let Revise pick up
the edits (or call `Revise.revise()`), then take a new capture:

```julia
Revise.revise()
dev_after = GFMPerf.capture("dev-after-edit-001";
    suite=:quick, role="current", samples=20, seconds=2, warmups=3)
cmp032 = GFMPerf.compare(dev_after, baseline032; threshold=0.10, min_samples=5)
cmp031 = GFMPerf.compare(dev_after, baseline031; threshold=0.10, min_samples=5)
```

The capture rebuilds deterministic fixtures and benchmark closures from the
current Julia world on every call; it does not rely on top-level `includet`
side effects. Restart after changing dependencies, project paths, or a change
that affects the loaded package layout. Use the same harness revision and
measurement settings for both baselines and development. If you edit the
harness itself, recapture both baselines in fresh processes before comparing.

To narrow work, pass `groups=["gemm", "gemv"]`. For the larger opt-in set,
including 1024 and 5000 square products, pass `suite=:full`. You can change
`samples`, `seconds`, and `warmups`; use identical values across captures.

## 4. Rerun a suspected case in fresh processes

`GFMPerf.compare` returns and prints the comparison directory. A slowdown above
10% that clears the within-run noise estimate is marked
`flagged-unconfirmed`; it is not called a confirmed regression. In a fresh
development REPL, rerun only those cases:

```julia
rerun_dev = GFMPerf.rerun_flagged(cmp032; label="dev-rerun-001",
    samples=40, seconds=4, warmups=3)
```

Capture the same case IDs from the v0.3.2 environment in its own fresh process.
For example, load the flagged IDs from the comparison artifact:

```julia
using JSON3
ids = JSON3.read(read(joinpath(cmp032, "comparison.json"), String),
    Dict{String,Any})["flagged_case_ids"]
baseline032_rerun = GFMPerf.capture("baseline-032-rerun-001";
    suite=:full, case_ids=ids, role="baseline-v0.3.2",
    samples=40, seconds=4, warmups=3)
```

For confirmation, use at least three fresh-process baseline → candidate →
baseline cycles. Compare each candidate run to the baseline captured
immediately before it, and also compare it to the following baseline to spot
drift. Then pass three distinct candidate/baseline comparison reports to
`GFMPerf.confirm([cmp1, cmp2, cmp3])`. Confirmation requires all independent
pair medians to cross the configured threshold and the repeated change to clear
the measured noise rule. Use the same flagged case IDs and settings throughout.

## 5. Read the artifacts

Each `perf/results/<label>/` directory contains:

- `run.json`: case outcomes, raw synchronized sample times, per-sample CPU
  allocations, GPU allocation results, and run metadata.
- `summary.csv`: one row for every requested case, including unavailable,
  unsupported, incorrect, or failed cases.
- `metadata.json`, `Project.toml`, and `Manifest.toml`: target and harness
  fingerprints, actual package path and project, resolved versions, hardware,
  preferences, and the resolved environment.

Comparison directories contain `comparison.json`, `comparison.csv`, and a
readable `comparison.txt`. Timing ratios are current divided by baseline;
positive percentages are slowdowns. CPU allocation counts/bytes come from
BenchmarkTools. GPU counts/bytes are collected in a separate warmed
`CUDA.@time` pass; the raw CUDA report is kept in `run.json`. Missing GPU
accounting is `null`, not zero. Pool-reserved memory is not reported as an
operation allocation.

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

See [SPEC.md](SPEC.md) for the workload, correctness, compatibility, and
reporting contract. References: [CUDA.jl profiling](https://cuda.juliagpu.org/stable/development/profiling/),
[BenchmarkTools manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/),
and [Revise user reference](https://timholy.github.io/Revise.jl/stable/user_reference/).

The reporting and stable-case unit tests run without a functional GPU:

```sh
julia --startup-file=no --project=.perf-regressions/envs/current perf/test/runtests.jl
```
