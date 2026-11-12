# GPUFiniteFieldMatrices.jl

[![CI](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml)

This package aims to implement fast finite field matrix operations on the GPU.

**This package is work-in-progress research software.** It may have rough edges. PRs and contributions are welcome.

## Current functionality

We currently implement:

- `CuModMatrix`, a `CuArray` wrapper representing a matrix modulo `N`, with
  logical dimensions separate from its padded GPU storage. This includes:
  - exact modular matrix and matrix-vector multiplication, with an element-type
    dependent safe stripe budget
  - if `N` is prime, row reduction, LU decomposition, PLUQ decomposition, square
    inversion, and rectangular one-sided inverses. See
    [`src/CuModMatrix/inverse/README.md`](src/CuModMatrix/inverse/README.md) for
    the inverse algorithms and their modulus contract.
- `KaratsubaMatrix`, a matrix type implementing Karatsuba matrix multiplication. This also supports addition, subtraction, and other basic operations.

## Getting Started

#### Installing for the first time

First, install Julia 1.12. If you're new to Julia, you can find a tutorial for non-experts [here](https://jjgarzella.github.io/blog/how-to-install-julia/). In short:

- On Windows, install Ubuntu through WSL and use the Ubuntu terminal.
- On Mac, open Terminal and run `xcode-select --install`.
- On Linux, open your usual terminal.

Then install Julia with `curl -fsSL https://install.julialang.org | sh`, restart your terminal if necessary, and open a Julia REPL with `julia`.

This package uses CUDA. You'll need to install the CUDA driver, see [the instructions in the CUDA.jl docs](https://cuda.juliagpu.org/stable/installation/overview/).

Then, clone this repo with `git clone https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl.git`.

`Revise.jl` is recommended for hot reloading. From a new Julia REPL, run
`using Pkg; Pkg.add("Revise")` (or, using Julia's package prompt, run `] add Revise`) once, and then run `using Revise` upon opening every new REPL.

The first time you install GPUFiniteFieldMatrices.jl, it will take a while for Julia to download and compile all of the dependencies. To do this, make sure your Julia REPL is in the GPUFiniteFieldMatrices.jl project folder. Then, in the package prompt (i.e. after pressing `]`), run

```
activate .
instantiate
```

The `instantiate` step will take a while as Julia's package manager downloads and compiles all dependencies.

Then, press backspace to go back to a julia prompt and run

```
using GPUFiniteFieldMatrices
```

#### Starting a new session after installation is complete

After starting a new Julia REPL, run

```
using Revise
] activate .
```

Then, press backspace to go back to a julia prompt and run

```
using GPUFiniteFieldMatrices
```

#### Troubleshooting

Please ensure that you're using the latest version of Julia and that your CUDA driver is installed. If the package manager gives you errors, try deleting Manifest.toml, upgrading Julia, updating the package registry, and updating CUDA.jl.

## Sample code

#### Matrices modulo N

Execute the following lines in Julia REPL:

```julia
using GPUFiniteFieldMatrices

A = CuModMatrix([1 2; 3 4]; modulus = 5)
B = CuModMatrix([2 1; 0 3]; modulus = 5)
A * B
```

Choose a storage type with `CuModMatrix{T}`. Typed zero and random factories
take the same type and the modulus as a keyword. For identities, use
`identity_matrix(CuModMatrix{T}, n; modulus = N)` or
`CuModMatrix{T}(I, rows, cols; modulus = N)`.

Newly allocated `CuModArray` storage rounds each logical dimension up to a
multiple of 32. Constructors and factories initialize the added padding to
zero, but later elementwise operations may change it; logical operations ignore
padding contents. `size(A)` reports logical dimensions and `parent(A)` returns
the GPU storage. Wrapping an existing `CuArray` preserves its storage and does
not copy, pad, or clear it.

#### LinearAlgebra inversion and PLUQ

The unparameterized CPU-array constructor keeps its Float32 default. Use a type
parameter to choose another storage type. Inversion follows LinearAlgebra.inv,
and PLUQ factors are available through F.L and F.U:

    using GPUFiniteFieldMatrices
    using LinearAlgebra

    A = CuModMatrix{Float32}([1 2; 3 4]; modulus = 101)
    Ainv = inv(A)
    F = pluq(A)
    L, U = F.L, F.U

Advanced helpers such as similar_with_modulus remain available by qualified
access, for example GPUFiniteFieldMatrices.similar_with_modulus(A, 103).

#### Karatsuba matrices

Convert a full array with `to_karatsuba`, or wrap already split component arrays
with `KaratsubaMatrix(A1, A2; low_modulus = N1, high_modulus = N2,
modulus = M)`. Inspect components with `low_part(K)` and `high_part(K)`, and
inspect their moduli with `low_modulus(K)` and `high_modulus(K)`. Reconstruction
uses `low_part(K) + low_modulus(K)*high_part(K)`; `modulus(K)` is separate
representation metadata. The component accessors preserve storage identity;
the struct fields are internal. Typed splitting checks each component conversion
and throws `InexactError` rather than losing integer information. Conversion
works with CPU arrays and preserves the GPU backend for GPU-backed inputs.

Before multiplying, initialize the plans on both inputs and the destination with
the qualified helper, then call `LinearAlgebra.mul!`:

```julia
using GPUFiniteFieldMatrices
using LinearAlgebra

A = CuModMatrix{Float64}([20 30; 40 50]; modulus = 100)
B = CuModMatrix{Float64}([20 30; 40 50]; modulus = 100)
split_options = (low_modulus = 100, high_modulus = 100, modulus = 100^2)
AK = to_karatsuba(Float64, A; split_options...)
BK = to_karatsuba(Float64, B; split_options...)
C = zeros(KaratsubaMatrix{Float64}, 2, 2; split_options...)
foreach(GPUFiniteFieldMatrices.initialize_plan!, (AK, BK, C))
mul!(C, AK, BK)
from_karatsuba(Float64, C)

# The allocating product needs no caller plans and preserves any plans already
# attached to AK or BK.
D = AK * BK
from_karatsuba(Float64, D)
```

Use `zeros(KaratsubaMatrix{T}, rows, cols; low_modulus, high_modulus,
modulus)` or the corresponding `KaratsubaVector{T}` form for GPU-backed zero
split buffers. Select CPU storage with `zeros(KaratsubaMatrix{T,Matrix{T}}, ...)`
or `zeros(KaratsubaVector{T,Vector{T}}, ...)`. The requested component storage
type selects the backend; the former `use_gpu` keyword has been removed. Plan
scratch is still initialized separately. The default GPU factory uses padded
`CuModArray` components and plans. Explicit CPU and raw `CuArray` component
factories retain their requested logical shapes and backend.

For `LinearAlgebra.mul!`, all three matrices must use the same representation
modulus, the inner dimensions of `A` and `B` must match, and all three matrices
must have initialized plans. The caller must size `C` to `(size(A, 1), size(B, 2))`:
Karatsuba multiplication checks this before any device write. All nine input,
output, and plan storage roots must reside on one CUDA device; this is checked
before staging or product kernels. Matrix products and matrix-vector products
support matching `Float32` or `Float64` components. The precision checks use
24 exact integer bits for `Float32` and 53 for `Float64`, and reject unsafe
working moduli or accumulators before writing plans.

For Karatsuba matrix multiplication, `A` and `B` must share input low radix
`R`, while `C` may use an independent low radix `R_C`. The output components
represent the product modulo `Q = R_C * high_modulus(C)`. The shared
representation modulus `M` is compatibility metadata; the matrix product is
not reduced modulo `M`. Correctness assumes canonical input components:
`0 <= A.low < R`, `0 <= A.high < high_modulus(A)`, `0 <= B.low < R`, and
`0 <= B.high < high_modulus(B)`. This is a caller promise and is not checked by
a runtime scan. Constructors and conversions do not establish those bounds:
`to_karatsuba` preserves the exact high quotient, and `CuModArray` wrappers
made with `mod=false` retain raw component values. Choose high moduli that
bound the supplied high components.

The allocating forms `A * B` and `A * x` support matching GPU-backed
`Float32` or `Float64` components. They delegate to the corresponding
`LinearAlgebra.mul!` arithmetic using private plan scratch, so caller plans may
be uninitialized and existing plans and component storage remain unchanged.
`A * x` returns a `KaratsubaVector`. Allocated results keep `A`'s split and
representation metadata; their component capacity is
`Q = low_modulus(A) * high_modulus(A)`, independent of metadata modulus `M`.

For `mul!(y, A, x)`, the matrix column count must match the vector length and
`y` must have `size(A, 1)` entries. `A` and `x` must share low radix `R`, and
the destination vector must use low radix `R` as well. Initialize all three
plans before the mutating operation. The legacy GEMV recombination still has a
known correctness limitation when both operands have interacting nonzero high
planes; the new precision checks do not resolve that inherited arithmetic
issue. The matrix-matrix correctness contract does not imply general GEMV
correctness.

For CPU-only data, `to_karatsuba` returns CPU component arrays and
`from_karatsuba(T, K)` returns `Array{T}`. GPU-backed components produce a plain
`CuArray{T}` result. `Base.Array(K)` uses arbitrary-precision integer
reconstruction to avoid overflowing an intermediate type.

`Base.copy` preserves the source's representation metadata `M` and split
moduli `N1` and `N2`. It deep-copies the low and high component storage and
creates a fresh plan when the source has one. The component capacity
`Q = N1*N2` remains independent of metadata `M`. In particular,
`to_karatsuba(A; modulus = M)` defaults to `N1 = N2 = M`, so the copied value
retains `M` even though `Q = M^2`. Multiplication requires matching stored
`M` on all operands, so a copied representation can be used with its original
when their metadata matches.

The specialized matrix-vector implementation is available through
`LinearAlgebra.mul!`, `A * x`, and qualified access as
`GPUFiniteFieldMatrices.karatsuba_gemv!`. The legacy Karatsuba matrix-matrix
keyword adapters have been removed. Use `A * B` or `A * x` for allocating
GPU-backed products, or call `LinearAlgebra.mul!` when supplying the
destination.

Per-call reduction moduli use the positive `modulus_override` keyword, for
example `elementwise_multiply!(C, A, B; modulus_override = 7)`. Mutating
operations leave stored modulus metadata unchanged; allocating stripe products
store their effective modulus. Matrix and matrix-vector products share the
qualified `GPUFiniteFieldMatrices.stripe_mul` / `stripe_mul!` implementation.
See their docstrings for destination-modulus defaults and the advanced
`input_bound` and `max_ops` controls. Unsafe explicit budgets are rejected;
default-budget empty-inner products return correctly shaped zeros.

The ordinary matrix `LinearAlgebra.mul!(C, A, B)` overload accepts
`input_bound` and `max_ops`; `modulus_override` is available through
qualified `stripe_mul!`. Karatsuba matrix multiplication builds logical
component views and preflights all three raw stripe products before writing
any plan scratch, then executes through the existing raw `_stripe_mul!` core.

#### Ownership, inverses, and errors

`copy(A)` allocates independent storage. `change_modulus(A, p)` also allocates;
it reduces logical values only when `p` shrinks and otherwise copies them.
The qualified `A = GPUFiniteFieldMatrices.change_modulus!!(A, p)` consumes the
old wrapper and shares its GPU storage. Keep the returned wrapper and stop
using the old one; aliases observe the reduction when `p` shrinks. Equal or
larger moduli leave values unchanged. This operation allocates no GPU storage.

The qualified upper- and lower-triangular inverse helpers preserve their input.
For a wide `m`-by-`n` triangular matrix with an invertible leading square block,
they return an `n`-by-`m` right inverse: the leading `m` rows invert that block,
the remaining rows are zero, and `A * X` is the identity modulo `modulus(A)`.
These helpers do not provide a general rectangular rank-revealing algorithm.

For batched factorization, `pluq_batched!(mats)` overwrites inputs, while
`pluq_batched!(dests, mats)` preserves inputs and writes distinct destinations.
`inverse_batched(mats)` allocates and preserves inputs. Qualified fixed-size
`inverse_batched_NxN!(mats)` methods overwrite inputs; their two-argument forms
write distinct destinations after validating the batch. `batch_streams` affects
scalar fallbacks; fixed-size kernels use their own batch strategy. See the
[inverse API notes](src/CuModMatrix/inverse/README.md) for supported options.

Wrong shapes throw `DimensionMismatch`, incompatible operand moduli throw
`GPUFiniteFieldMatrices.ModulusMismatchError`, non-prime moduli throw
`DomainError`, invalid arguments throw `ArgumentError`, exact-arithmetic
overflow throws `OverflowError`, and unavailable inverses throw
`LinearAlgebra.SingularException`. Exact conversion failures use `InexactError`.
`A == B` compares logical shape, modulus, and entries while ignoring padding;
unary `-A` preserves its input, and `LinearAlgebra.rank(A)` uses non-mutating
PLUQ factorization.

## License

GPUFiniteFieldMatrices.jl is distributed under the MIT license.

## Developer checks

For local checks, run `python -m pip install pre-commit==4.6.2` followed by `pre-commit install`; CI runs these checks automatically. The hook installs JuliaFormatter 2.14.0 in an isolated environment, and ReLint remains an optional manual check.

`Pkg.test()` runs the CPU interface and quality checks on hosts without CUDA.
Arithmetic, allocation, triangular, batch, and raw-CuArray regression suites
run inside the CUDA-functional gate. A CPU pass with that gate skipped does not
verify GPU numerical results or allocation budgets.

DispatchDoctor is temporarily disabled by default in manual test runs while
CUDA type-stability findings are investigated. All existing correctness tests
still run; CI explicitly enables strict DispatchDoctor checking. To run the
suite manually from the checkout, start a fresh Julia process without a custom
sysimage:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

To enable strict checking locally, use:

```sh
GPUFFM_DISPATCH_DOCTOR_MODE=error julia --project=. -e 'using Pkg; Pkg.test()'
```

The switch also accepts `warn`. The test runner prints the selected mode before
loading the package. Changing the mode requires a fresh process; a sysimage
containing the package must be rebuilt to change its baked instrumentation.
