# GPUFiniteFieldMatrices.jl

[![CI](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml)

GPUFiniteFieldMatrices.jl implements exact modular vector and matrix operations
on CUDA and Metal. It includes elementwise arithmetic, matrix and matrix-vector
products, finite-field factorizations and inverses, and two-component Karatsuba
arrays. This package is work-in-progress research software.

## Getting started

Use Julia 1.12 and load the vendor package for your GPU in the same Julia
environment as GPUFiniteFieldMatrices. On Apple Silicon, use Metal.jl 1.12. On
NVIDIA hardware with a working driver, use CUDA.jl 6.5 or later within 6.x.
The port uses KernelAbstractions 0.10 (minimum 0.10.8) and KernelInterface
0.4 (minimum 0.4.7). For CUDA.jl 5, use the v0.4 release.

To try the Metal examples from a checkout, start Julia in the repository root
and set up a temporary environment.

```julia
using Pkg
Pkg.activate(; temp = true)
Pkg.develop(path = pwd())
Pkg.add(Pkg.PackageSpec(name = "Metal", version = "1.12.1"))
```

## Modular arrays

`GPUModArray{T,D,A}` stores residues modulo a positive integer. `T` is the
element type, `D` is the number of dimensions, and `A` is the storage type.
`GPUModMatrix` and `GPUModVector` fix the dimensionality. `CuModArray` and
`MtlModArray` select CUDA and Metal respectively, with corresponding matrix
and vector aliases for each vendor.

```julia
import Metal
using GPUFiniteFieldMatrices
using LinearAlgebra

Metal.allowscalar(false)
storage = Metal.PrivateStorage
matrix_options = (modulus = 101, storage = storage)
A = GPUModMatrix{Float32}([1 2; 3 4]; backend = :metal, matrix_options...)
B = MtlModMatrix([2 1; 0 3]; matrix_options...)
x = MtlModVector([2, 3]; matrix_options...)

C = similar(A)
@assert mul!(C, A, B; max_ops = 1) === C
@assert Array(C) == [2 7; 6 15]
@assert Array(A * x) == [8, 18]
```

For host input, an explicit `backend = :cuda` or `:metal` takes priority over
`with_backend(:metal) do ... end`, then the process default set by
`set_default_backend!`. The initial process default, `:auto`, chooses a
loaded, functional CUDA backend first, then Metal. Allocation errors when the
selected vendor is unavailable. `set_default_backend!(:auto)` restores
automatic selection. An untyped `GPUModArray(values; ...)` uses `eltype(values)`.
The untyped vendor matrix and vector constructors use `Float32` for host
input.

Metal allocations accept `storage = Metal.SharedStorage`,
`Metal.PrivateStorage`, or the native default. Use `device = Metal.device()`
or another `Metal.MTLDevice` to select a Metal device. CUDA allocations use
the native default storage mode and accept a `CUDA.CuDevice` or a one-based
device index. The explicit device applies to the allocation and the caller's
active device is restored afterward. Wrapping native storage keeps its
element type, device, storage mode, and values. The wrapper takes the backend
from the native input and rejects conflicting explicit selectors. Pass
`new_size = (rows, cols)` or `(length,)` to expose a smaller leading region.
`parent(A)` returns the original native array or view when wrapping it.

New host-to-GPU constructors and typed `zeros` and `rand` factories pad each
dimension to a multiple of 32 and initialize the added entries to zero. Host
constructors reduce values modulo `modulus` by default. Use `size(A)` and
`Array(A)` for logical shape and values, and `parent(A)` to access physical
storage. Equality compares logical shape, modulus, and entries. The standard
`similar(A)` and `copy(A)` preserve backend, device, and storage mode.
`copy(A)` creates independent storage. `copyto!` and `copy!` copy logical
entries as-is and leave destination padding intact. `fill!` reduces its value
modulo the array's modulus and fills the logical region. Use
`identity_matrix(MtlModMatrix{Float32}, n; modulus = p)` or the corresponding
CUDA form for an identity matrix.

Elementwise addition, subtraction, negation, scalar arithmetic, elementwise
multiplication, division by invertible scalars, and modulus changes use
KernelAbstractions. `change_modulus(A, p)` allocates and reduces the values
when `p` is smaller. Equal or larger moduli just copy the values.
`GPUFiniteFieldMatrices.change_modulus!!(A, p)` shares the input storage.
Keep the returned wrapper, which carries the new modulus. Aliases see any
reduction in that shared storage. Mutating arithmetic with
`modulus_override`, such as `elementwise_multiply!(C, A, B; modulus_override = 7)`,
uses the override for the calculation and retains the destination metadata.

### Storage and exact products

| Storage type | Maximum positive modulus | Backends |
| --- | ---: | --- |
| `Float16` | `2^11` | CUDA, Metal |
| `Float32` | `2^24` | CUDA, Metal |
| `Float64` | `2^53` | CUDA |

Construction rejects a modulus above the storage ceiling with `DomainError`.
Metal modular constructors reject `Float64` and integer storage with
`ArgumentError`. Use `Float32` for Metal matrix and matrix-vector products.
These products use Metal Performance Shaders (MPS) through native
`LinearAlgebra.mul!`. CUDA uses cuBLAS. Products require matching floating
element types, a shared GPU backend and device, and destination storage
distinct from the inputs.

`A * B` and `A * x` allocate results. `mul!` writes into a caller-supplied
destination with the logical product shape. The qualified `stripe_mul` and
`stripe_mul!` methods expose the reduction modulus and stripe controls.
Matrix `mul!` requires one stored modulus across all operands. Matrix
`stripe_mul!` defaults to the destination modulus, while vector `stripe_mul!`
defaults to the input modulus and requires a matching destination modulus
unless `modulus_override` is supplied. Vector `mul!` also accepts this
override. The two inputs must share their stored modulus even with a
positive override. An allocating stripe product stores the effective modulus.
A mutating stripe product returns its destination, clears its padding, and
retains its modulus metadata.

The product reduces after each stripe of the inner dimension. Every input
entry must be a nonnegative integer strictly below `input_bound`, which
defaults to the input modulus and must be an integer greater than one.
The caller supplies this bound without a runtime scan. For `s` terms, let
`w = (input_bound - 1)^2` and let `q` be the effective reduction modulus. The
planner requires both `(s + 1) * w <= 2^p - 1` and
`s * w + (q - 1) <= 2^p - 1`, where `p` is 11, 24, or 53 for the respective
storage type. Construction can therefore succeed at a modulus for which a
product needs a tighter input bound. A supplied `max_ops` must be a positive
integer within that budget. A product with zero inner dimension returns
logical zeros after validation. Its default budget can be zero, while an explicit
`max_ops` still requires a positive safe budget.

## Factorization and inverses

Use a prime modulus for PLUQ, rank, and inversion.
`PLUQOptions(check_prime = true)` validates primality before GPU work. The
inverse APIs require the modulus to fit in `Int32` as well as in the element
type. Metal `Float32` supports square and rectangular PLUQ, square and
one-sided inverses, triangular inversion, and batched factorization and
inversion.

Using `A` from the Metal example above, compute the inverse and check the
permuted factorization.

```julia
field_options = PLUQOptions(check_prime = true)
Ainv = inv(A; options = field_options)
F = pluq(A; options = field_options)
@assert Array(A * Ainv) == Matrix{Float32}(I, 2, 2)
@assert Array(F.L * F.U) == Array(A)[F.p, F.q]
```

`pluq(A)` preserves its input. `pluq!(A)` overwrites it with packed factors.
`F.L` and `F.U` materialize factors, `F.p` and `F.q` give the row and column
permutations, and `F.rank` gives the rank. `inv(A)` uses PLUQ by default and
requires a safe floating product budget. When that budget is insufficient,
`inv(A; options = PLUQOptions(inverse_strategy = :augmented))` selects
augmented elimination on the GPU. `right_inverse(A)` requires full row rank
and `left_inverse(A)` requires full column rank. The qualified
`upper_triangular_inverse_no_copy` and `lower_triangular_inverse_no_copy`
helpers preserve the input and return a right inverse of a wide triangular
matrix with an invertible leading square block, with zeros below the inverse
of that block. For Metal `Float32`, `trsm_mode = :warp` uses subgroup
triangular solves on eligible full panels when the device supports them.
`:auto` also considers `trsm_warp_threshold`, and `:panel` selects the
ordinary GPU solve.

`pluq_batched!(mats)` overwrites inputs, `pluq_batched!(dests, mats)` writes
into distinct destinations, and `pluq_batched(mats)` and
`inverse_batched(mats)` allocate and preserve inputs. The qualified fixed-size
PLUQ and inverse methods cover square sizes 4, 8, 16, and 32. Each fixed-size
inverse method has a one-argument form that overwrites inputs and a
two-argument form that writes distinct destinations. The tiny inverse kernel
uses `inverse_strategy = :augmented`. Metal batches require
`batch_streams = 1`. The CUDA extension has further implementation and
batching notes. [CUDA inverse notes](ext/GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/README.md)

Wrong shapes throw `DimensionMismatch`, incompatible operand moduli throw
`GPUFiniteFieldMatrices.ModulusMismatchError`, and unavailable inverses throw
`LinearAlgebra.SingularException`. Primality checks reject composite moduli
with `DomainError`, and inverse arithmetic budget errors can raise
`OverflowError`.

## Karatsuba arrays

`KaratsubaArray{T,D,S,P}` represents an array with two components of the same
concrete storage type `S` and a separate plan storage type `P`. By default,
CPU components use an `Array` plan and GPU components use a modular array
plan on their backend. `KaratsubaMatrix` and `KaratsubaVector` fix the dimensionality. Convert an
array with `to_karatsuba`, or wrap existing components with
`KaratsubaMatrix(low, high; low_modulus, high_modulus, modulus)` or
`KaratsubaVector`. The `low_part` and `high_part` accessors return the stored
components. `low_modulus`, `high_modulus`, and `modulus` return their split
moduli and representation metadata.

Reconstruction is `low_part(K) + low_modulus(K) * high_part(K)`, entry by
entry. `Array(K)` reconstructs on the host with `BigInt` intermediates.
`from_karatsuba(T, K)` returns a plain native array on the same backend as the
components, or a Julia `Array` for CPU components. Exact conversion failures
raise `InexactError`. Metal conversion rejects `Float64` output. GPU splitting and reconstruction use logical entries
and preserve the device and storage mode. `copy(K)` copies both components
and creates a fresh workspace if the source has a plan.

Use `zeros(KaratsubaMatrix{T}, rows, cols; backend, low_modulus, high_modulus,
modulus)` or the corresponding vector factory for GPU buffers. An omitted
`backend` follows the normal default selection. Specify
`KaratsubaMatrix{T,Matrix{T}}` or `KaratsubaVector{T,Vector{T}}` for CPU
component storage. Initialize each input and the destination with
`GPUFiniteFieldMatrices.initialize_plan!` before `mul!`. Allocating products
use their own workspace and preserve caller plans.

The following Metal example uses the same private storage as above. Its
product capacity is `16 * 4 = 64`, and its representation modulus is 101.

```julia
split = (low_modulus = 16, high_modulus = 4, modulus = 101)
KA = to_karatsuba(Float32, MtlModMatrix([1 13; 20 7]; matrix_options...); split...)
KB = to_karatsuba(Float32, MtlModMatrix([6 3; 7 4]; matrix_options...); split...)
KX = to_karatsuba(Float32, MtlModVector([9, 5]; matrix_options...); split...)
KC = zeros(KaratsubaMatrix{Float32}, 2, 2; backend = :metal, storage, split...)

foreach(GPUFiniteFieldMatrices.initialize_plan!, (KA, KB, KC))
@assert mul!(KC, KA, KB) === KC
@assert Array(from_karatsuba(Float32, KC)) == [33 55; 41 24]
@assert Array(KA * KX) == [10, 23]
```

Metal products require `Float32` components and accept both native
`MtlArray` and wrapped `MtlModArray` components. CUDA product entry points
require `CuModArray` components with matching `Float32` or `Float64` types.
All inputs, outputs, and plans for a product must be on one GPU device. Use a
destination with the logical product shape and independent storage. CPU
components support storage, conversion, and pointwise methods.

Both product forms require matching representation modulus `M` and canonical
input digits. Each low and high entry must lie in `[0, low_modulus)` or
`[0, high_modulus)` respectively. Supplying these bounds is the caller's
responsibility. In particular, `to_karatsuba` preserves the high quotient.
For floating components with `p` exact integer bits, construction requires
`M < 2^(p - 2)` on CPU and CUDA, and `M <= 2^p` on Metal. Split moduli on
modular components follow the storage ceilings above.

The product represents a residue modulo the destination capacity
`Q = low_modulus(C) * high_modulus(C)`. For nonempty products, `Q` must
fit the working type's exact integer range. Allocating products take their
split and metadata from the left matrix.

- **Matrix product.** Inputs must share a low radix. The destination can use
  a different low radix. The three products have independent safe stripe
  budgets.
- **Matrix-vector product.** The two inputs and output must share a low radix
  `R`. The output high modulus must divide `R` so that the `R^2` high-plane
  term vanishes modulo `Q`. The input high moduli may differ. Each complete
  dot product, its component sums, and the recombination must fit the
  working type's exact integer range. This is a whole-dot bound of `2^24`
  for `Float32` or `2^53` for `Float64`. The qualified `karatsuba_gemv!`
  has the same contract as `mul!`.

## Developer checks

Run the full package suite from the repository root. The wrapper creates a
fresh project view and provisions the pinned vendor packages for the tests.

```sh
JULIA_VERSION=1.12 sh test/run-pkg-test.sh
```

Plain `Pkg.test()` also provisions the vendors in its test environment. To
require execution on one GPU backend, pass `metal` or `cuda` to the wrapper.
An explicit selection fails if that backend is unavailable or nonfunctional.
The minimal core smoke test runs without either vendor package.

```sh
JULIA_VERSION=1.12 sh test/run-pkg-test.sh metal
JULIA_VERSION=1.12 sh test/run-pkg-test.sh cuda
JULIA_VERSION=1.12 sh test/run-core-environment-smoke.sh
```

The default run includes CPU and quality checks and gates numerical GPU tests
on backend functionality. Focused Metal hardware checks use Julia 1.12.7 and
Metal 1.12.1 on an Apple M3 Max, with SharedStorage and PrivateStorage.
CUDA hardware numerical and performance validation remains pending.

For local formatting checks, install `pre-commit==4.6.2` and run
`pre-commit install`. The hook uses JuliaFormatter 2.14.0 in an isolated
environment. ReLint remains an optional manual check.

## License

GPUFiniteFieldMatrices.jl is distributed under the MIT license.
