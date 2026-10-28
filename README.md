# GPUFiniteFieldMatrices.jl

[![CI](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml)

This package aims to implement fast finite field matrix operations on the GPU.

**This package is work-in-progress research software.** It may have rough edges. PRs and contributions are welcome.

## Current functionality

We currently implement:

- `CuModMatrix`, a `CuArray` wrapper which represents a matrix modulo `N`. `CuModMatrix` pads elements to 32 bits, and has various matrix operations implemented as GPU kernels. This includes:
  - matrix multiplication (for `N < ~2^26`)
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

Use `modulus(A)` to inspect the modulus, `size(A)` for the logical dimensions,
and `parent(A)` for the padded GPU storage. The parent's dimensions may be
larger than the logical size.

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
```

Use `zeros(KaratsubaMatrix{T}, rows, cols; low_modulus, high_modulus,
modulus)` or the corresponding `KaratsubaVector{T}` form for GPU-backed zero
split buffers. Select CPU storage with `zeros(KaratsubaMatrix{T,Matrix{T}}, ...)`
or `zeros(KaratsubaVector{T,Vector{T}}, ...)`. The requested component storage
type selects the backend; the former `use_gpu` keyword has been removed. Plan
scratch is still initialized separately.

For `LinearAlgebra.mul!`, all three matrices must use the same representation
modulus, the inner dimensions of `A` and `B` must match, and all three matrices
must have initialized plans. The caller must size `C` to `(size(A, 1), size(B, 2))`:
the stripe preflight checks this before any device write. This multiplication
path currently supports `Float64` storage only.

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

For CPU-only data, `to_karatsuba` returns CPU component arrays and
`from_karatsuba(T, K)` returns `Array{T}`. GPU-backed components produce a plain
`CuArray{T}` result. `Base.Array(K)` uses arbitrary-precision integer
reconstruction to avoid overflowing an intermediate type.

`KaratsubaArray` records its component and plan storage types, so
`from_karatsuba` infers a CPU `Array` or GPU `CuArray` result from the stored
components. The strict DispatchDoctor test preference remains enabled.

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
The `*` operator and matrix-matrix keyword paths remain unsupported; use the
mutating `LinearAlgebra.mul!` path described above for Karatsuba matrix products.

Per-call reduction moduli use the `modulus_override` keyword, for example
`elementwise_multiply!(C, A, B; modulus_override = 7)` or
`LinearAlgebra.mul!(z, A, x; modulus_override = 7)`. Qualified matrix
multiplication is available as `GPUFiniteFieldMatrices.stripe_mul(A, B)` and
`GPUFiniteFieldMatrices.stripe_mul!(C, A, B)`; the same functions support
matrix-vector products. `*` and `LinearAlgebra.mul!` use the shared stripe
implementation. Allocating results store the effective modulus, while mutating
operations leave stored modulus metadata unchanged. An omitted matrix-matrix
in-place matrix-matrix override uses `modulus(C)`; an omitted allocating
override uses `modulus(A)`; matrix-vector multiplication uses
`modulus(A)` and requires the destination to carry that same modulus when
no override is supplied.

Multiplication paths require a positive `modulus_override` when one is supplied.
Non-multiplication elementwise and scalar operations retain the stored-modulus
fallback for zero or negative overrides.

For stripe multiplication, `input_bound` is the exclusive upper bound on
nonnegative multiplicands, and `max_ops` limits product contributions
accumulated per output entry between modular reductions. A supplied
`max_ops` is used when it is safe for the input bound and reduction modulus;
otherwise the call throws `ArgumentError`. If omitted, the safe budget is
derived from those values. A matrix-matrix `input_bound` may require multiple
stripes when the inner dimension exceeds that budget. The low-level
matrix-vector adapter retains `respect_max_ops=false` to recompute its safe
stripe budget.

## License

GPUFiniteFieldMatrices.jl is distributed under the MIT license.

## Developer checks

For local checks, run `python -m pip install pre-commit==4.6.2` followed by `pre-commit install`; CI runs these checks automatically. The hook installs JuliaFormatter 2.14.0 in an isolated environment, and ReLint remains an optional manual check.
