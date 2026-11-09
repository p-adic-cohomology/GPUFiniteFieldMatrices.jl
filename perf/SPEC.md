# GFM CUDA performance harness specification

## Scope and ownership

`GFMPerf` is a shared manual measurement harness under `perf/`. One copy of the
harness and one resolved CUDA.jl / BenchmarkTools / Julia stack are used for
v0.3.1, v0.3.2, and the editable development checkout. Each package revision
runs in a separate Julia process and a separate package environment. Revise is
loaded before `GPUFiniteFieldMatrices` in all documented REPL sessions. The
harness rebuilds fixtures and benchmark objects on each capture.

The frozen package targets are identified by peeled commits, since version
strings can be unchanged across source revisions:

| Target | Commit |
| --- | --- |
| v0.3.1 | `7192d479d2c41d31021a4938beec8223d16d6654` |
| v0.3.2 | `9bb9828f3ee69ab9e60df32201f6c2560f20c36b` |

The current editable checkout is identified by commit, dirty-source hash, and
loaded source path. It is not identified by `Project.toml` version alone. The
setup pins CUDA.jl 5.11.3 by default; each result still records the actual
resolved versions and compatibility checks reject mismatches. BenchmarkTools,
CUDA, Revise, JSON3, CSV, and compatibility dependencies live only in the
manual harness environment, not the production project. Each variant's
environment exposes Revise, BenchmarkTools, JSON3, and CUDA as direct
dependencies for documented top-level REPL imports. Setup checks that their
versions match and repairs missing direct entries in an owned-ready environment.

## Workload and stable case identity

Every case ID includes operation and contract, element type, logical dimensions,
modulus, Karatsuba split where applicable, storage-layout label, and fixture
seed. Each requested ID gets one outcome row. Unsupported APIs and incorrect
results remain visible in artifacts instead of disappearing from an older
suite.

The quick suite includes:

- CuMod GEMM and GEMV for Float32 and Float64, both public `mul!` and allocating
  `*`, with square, rectangular, and skinny cases.
- Allocating/in-place elementwise addition and multiplication, plus copy and
  constructor allocation cases.
- Shared-semantic Float64 Karatsuba GEMM/GEMV cases, current-only Float32
  Karatsuba cases, and a small diagonal inverse/PLUQ group.

The full suite adds 31/32/33 and 63/64/65 tile-boundary sizes, 127/128/129 and
256/1024 square cases, and opt-in 5000-square GEMM. Full workloads are not part
of an ordinary edit loop. Group and case-ID filters allow focused captures.

Fixtures come from a per-case seed, never a shared random stream. Data and
correctness calculations are outside timed regions. The BigInt CPU oracle
reduces the expected arithmetic modulo the requested modulus. In-place output
buffers are restored before each sample and synchronized before timing. The
public operation is timed as synchronized host wall time using `CUDA.@sync`;
dispatch, preflight, views, launches, and GPU completion remain included.
Allocating `*` cases include their public allocation/private-plan cost. Reusable
Karatsuba plans are prepared before timing while public `mul!` checks remain in
the timed call. Each selected specialization is warmed outside timing.

Layout metadata records logical shape and the physical shape and strides of
each relevant backing array. Construction and copy cases stay separate from
arithmetic.

Karatsuba Float64 fixtures have zero high planes, so the shared comparisons do
not exercise the known inherited interacting-high-plane GEMV arithmetic
limitation. No known-wrong historical case is assigned a speed ratio. Float32
Karatsuba is current-only and is marked unsupported in historical captures.

GEMM correctness uses scalar BigInt dot products up to 250,000 scalar
products. Larger cases use an independent Float64 BLAS computation only after
checking that fixtures are finite, nonnegative, integer-valued, and exactly
representable in Float64, and that `k * max(A) * max(B) <= 2^53`. Since all
terms and partial sums are nonnegative and bounded by that limit, each product
and accumulation is exactly representable in Float64. Nine deterministic
corner/center cells are cross-checked with BigInt. If any precondition or bound
fails, oracle construction throws and the case is recorded as failed; it never
falls back to an approximate answer.

## Timing and allocation records

The default policy is 3 warmups, 20 samples, 1 evaluation per sample, and a
2-second BenchmarkTools budget per case. The policy is stored in each capture
and must match for a comparison. Mutating setup and its synchronization run
outside the timed sample. BenchmarkTools `Trial.memory` and `Trial.allocs` are
scalar minimum allocation byte/count values across the trial samples; the run
stores those scalars alongside the raw time and GC sample arrays. A separate
warmed CUDA.jl `CUDA.@timed` pass reports GPU bytes through `gpu_bytes` and the
count through `gpu_memstats.alloc_count`. CUDA pool-reserved bytes are not
operation allocations. Structured accounting values are retained in the run;
if CUDA accounting is unsupported or fails, the GPU fields are `null`.

Every run stores raw sample nanoseconds, GC times, CPU allocation minima,
actual sample count, correctness result, and one row for every requested case.
It also stores run label/time, role, target commit/source hash/path/version,
harness hash, active project, a copy of the resolved Project and Manifest,
Julia/CUDA.jl/runtime/driver versions, GPU identity, relevant preferences,
launch settings, normalized dependency versions, fixture definition, and timing
policy. A before/after source fingerprint invalidates a capture if its source
changes while the suite is running.

## Comparison and regression policy

The v0.3.2 capture is the primary reference; v0.3.1 is a secondary reference.
The comparison intentionally permits the target package path, source commit,
dirty-source hash, and package version to differ. It requires the Julia,
CUDA.jl, BenchmarkTools, CUDA runtime/driver, GPU, preference and launch
settings, normalized dependencies, harness hash, and measurement policy to
match. Manifest byte hashes are recorded but are not compared as compatibility
keys because the target package source/path is expected to differ.

For each matching correctness-passing case, the report includes current and
baseline medians, ratio, percentage change, MAD and p05/p95 spread, plus CPU and
GPU allocation counts/bytes and deltas. Allocation growth is tracked
independently of runtime. The default slowdown flag is greater than 10%.

A single capture pair never says a regression is confirmed. Too few samples or
a change within estimated sample noise is inconclusive. Larger single-pair
changes are `flagged-unconfirmed`; `GFMPerf.confirm` requires three independent
fresh-process pair reports and checks that every pair crosses the threshold and
the repeated median clears the noise rule. The documented confirmation cycle
is baseline → candidate → baseline. The final manual report is not a CI or
`Pkg.test()` gate.

Unsupported, unavailable, wrong-answer, and failed cases have no timing ratio.
If the current source changes during a capture, the capture is invalid. If the
normalized dependency or hardware configuration differs, the comparison is
incompatible and does not label the ratio as package-code-only evidence.

## Setup safety and artifacts

`perf/setup.sh` verifies that `origin` is the published repository, fetches
tags, confirms the exact frozen commits, and creates detached baseline
worktrees. It never resets or removes an existing directory. The run root,
worktree registry, and each environment carry ownership and target-path
markers. Existing paths must match the expected commit, package path, and clean
state or setup stops. Each environment has its own Project/Manifest. Results
require unique labels unless `overwrite=true` is deliberately supplied.

Generated environments, worktrees, manifests, caches, and results are ignored
by Git. Captures write `run.json`, `metadata.json`, `summary.csv`, and exact
resolved `Project.toml` / `Manifest.toml` copies. Comparisons write JSON, CSV,
and a readable text report. All commands and REPL snippets are documented in
[README.md](README.md).

## Local validation boundary

Julia syntax, package imports, setup and result/comparison logic can be checked
without a functional GPU. `CUDA.functional() == false` yields explicit
unavailable rows and never substitutes CPU timings. GPU correctness and timing
must be exercised on the intended CUDA Linux host; this repository change does
not claim a local GPU result.
