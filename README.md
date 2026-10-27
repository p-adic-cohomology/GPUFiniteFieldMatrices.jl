# GPUFiniteFieldMatrices.jl

[![CI](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml)

This package aims to implement fast finite field matrix operations on the GPU.

**This package is work-in-progress research software.** It may have rough edges. PRs and contributions are welcome.

## Current functionality

We currently implement:

- `CuModMatrix`, a `CuArray` wrapper which represents a matrix modulo `N`. `CuModMatrix` pads elements to 32 bits, and has various matrix operations implemented as GPU kernels. This includes:
  - matrix multiplication (for `N < ~2^26`)
  - if `N` is prime, RREF, LU decomposition, PLUQ decomposition, square
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
modulus = M)`. Reconstruction is always `data1 + N1*data2`; `M` is separate
representation metadata. Typed splitting checks each component conversion and
throws `InexactError` rather than losing integer information. Conversion works
with CPU arrays and preserves the GPU backend for GPU-backed inputs.

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
```

Use `zeros(KaratsubaMatrix{T}, rows, cols; low_modulus, high_modulus,
modulus, use_gpu = true)` or the corresponding `KaratsubaVector{T}` form for
zero split buffers. Set `use_gpu = false` to allocate CPU arrays. Plan scratch is
still initialized separately.

For `LinearAlgebra.mul!`, all three matrices must use the same representation
modulus, the inner dimensions of `A` and `B` must match, and all three matrices
must have initialized plans. The caller must size `C` to `(size(A, 1), size(B, 2))`:
this method does not explicitly check the destination dimensions before CUDA
work begins. This multiplication path currently supports `Float64` storage only.

For CPU-only data, `to_karatsuba` returns CPU component arrays and
`from_karatsuba(T, K)` returns `Array{T}`. GPU-backed components produce a plain
`CuArray{T}` result. `Base.Array(K)` uses arbitrary-precision integer
reconstruction to avoid overflowing an intermediate type.

The strict DispatchDoctor test preference currently reports
`TypeInstabilityError` for `from_karatsuba` because `KaratsubaArray` stores its
components in abstract array fields and reconstruction can return CPU or GPU
array types. A separate storage-type parameter change is needed to make that
result infer as a single concrete array type; the strict preference remains
enabled in tests.

`Base.copy` retains the legacy `Karatsubacopy` metadata behavior: the copied
representation stores `M = N1*N2`, which can differ from the source's `M`.
In particular, `to_karatsuba(A; modulus = M)` defaults split factors to
`N1 = N2 = M`, so copying that representation changes its stored `M` to `M^2`. Multiplication
requires matching stored `M` on all operands; do not mix these copied and
uncopied representations. The initialized-copy product test uses the existing
split constructors with a consistent `M = N1*N2`. Correcting this metadata
behavior is deferred to a separate follow-up.

The specialized matrix-vector path is available through qualified access as
`GPUFiniteFieldMatrices.karatsuba_gemv!`.

The `*` operator and matrix-matrix keyword paths remain unsupported; these
renames change the API spellings without changing those behaviors. Three
existing Karatsuba matrix-matrix calls still pass legacy `R`, `P`, and
`maxopsOverride` keywords and raise `MethodError`; this path remains tracked by
Phase 2 issue `gfm-2x1`.

Per-call reduction moduli use the `modulus_override` keyword, for example
`elementwise_multiply!(C, A, B; modulus_override = 7)` or
`LinearAlgebra.mul!(z, A, x; modulus_override = 7)`. Qualified advanced matrix
multiplication is available as
`GPUFiniteFieldMatrices.stripe_mul(A, B; modulus_override = 7)` and
`GPUFiniteFieldMatrices.stripe_mul!(C, A, B; modulus_override = 7)`; the same
names support matrix-vector products. `*` and `LinearAlgebra.mul!` use this
shared implementation. An allocating call stores the effective modulus on its
result. A mutating call reduces with the effective modulus and leaves the
destination's stored modulus unchanged. For the matrix allocating and mutating
adapters, an omitted override uses `A.N`; a supplied override must be positive.
Matrix-vector `mul!` follows the same rule. Elementwise and scalar operations
use a positive override for reduction. Elementwise and scalar multiplication
reject zero or negative overrides with `ArgumentError`; other elementwise and
scalar operations retain the stored-modulus fallback for omitted, zero, or
negative overrides.

For stripe multiplication, `input_bound` promises that input entries are
nonnegative and strictly below the bound. `max_ops` is the maximum number of
product contributions accumulated per output entry between modular
reductions. The safe chunk calculation includes the reduced prior accumulator
and is independent of the working reduction modulus. Supplying `input_bound`
to matrix-matrix `mul!` or `stripe_mul!` keeps the existing single-stripe route;
the complete inner dimension must fit the exact-integer budget or the call
throws `ArgumentError`. That route ignores `max_ops` after validating that a
supplied value is positive. Matrix-vector calls use the bound to compute a safe
default stripe size.

## License

GPUFiniteFieldMatrices.jl is distributed under the MIT license.

## Developer checks

For local checks, run `python -m pip install pre-commit==4.6.2` followed by `pre-commit install`; CI runs these checks automatically. The hook installs JuliaFormatter 2.14.0 in an isolated environment, and ReLint remains an optional manual check.
