# Changelog

## Unreleased

### Breaking: consistent exceptions

Shape and square-matrix requirements throw `DimensionMismatch`. Incompatible
operand moduli throw the qualified public `ModulusMismatchError`. Non-prime
moduli throw `DomainError`; invalid arguments and unsupported sizes throw
`ArgumentError`; exact-arithmetic overflow throws `OverflowError`; singular
matrices and unavailable inverses throw `LinearAlgebra.SingularException`.
The former custom exception types are removed without aliases. Existing guard
conditions and validation before destination writes are preserved. Exact
representation conversions continue to use `InexactError`.

### Allocated GPU padding

Newly allocated `CuModArray` storage now rounds each dimension up to a multiple
of 32 and starts with zero padding. Later elementwise operations may dirty
padding, which logical operations ignore. Wrapping existing `CuArray` or
Karatsuba component storage keeps its original shape, backend, and contents.
Allocating `copy` preserves logical values and modulus metadata while using
the new padded geometry and zero padding. Allocating `change_modulus(A, p)`
sets the new modulus, reducing logical values when it shrinks and copying them
otherwise. In-place `copy!` copies
all logical entries across valid backing layouts and leaves destination padding
unchanged.

### Breaking: API renames

This change prepares the 0.4.0 API break. No compatibility aliases are kept.

### Tiny fixed-size inverse destination migration

The qualified one-argument methods `GPUFiniteFieldMatrices.inverse_batched_4x4!`,
`GPUFiniteFieldMatrices.inverse_batched_8x8!`,
`GPUFiniteFieldMatrices.inverse_batched_16x16!`, and
`GPUFiniteFieldMatrices.inverse_batched_32x32!` previously allocated result
wrappers and now overwrite their input wrappers.
Use the exported `inverse_batched(mats)` for allocating, input-preserving
behavior, or the corresponding two-argument size-specific methods to write into
distinct caller-provided destinations.

### API migration table

The table includes naming changes and the behavioral changes described above.

| Old name | New name | Notes |
| --- | --- | --- |
| inverse(A), inverse_new(A; options) | Base.inv(A; options) | Folded into one method; inverse previously forwarded to inverse_new with default options. |
| is_invertible(A), is_invertible_new(A; options) | is_invertible(A; options) | Exact duplicate folded into one keyword-capable method. |
| inverse_pluq_new(A; options) | inverse_pluq(A; options) | Forces the PLUQ strategy; only the suffix changed. |
| right_inverse_new(A; options) | right_inverse(A; options) | Name only. |
| left_inverse_new(A; options) | left_inverse(A; options) | Name only. |
| pluq_new(A; options), pluq_new!(A; options) | pluq(A; options), pluq!(A; options) | Non-mutating and mutating variants keep their behavior. |
| pluq_new_batch, inverse_new_batch | pluq_batched, inverse_batched | General batch entry points. |
| PLUQFactorization{T} | PLUQ{T} | Fields LU, p, q, and rank remain available. |
| pluq_extract_L(F), pluq_extract_U(F) | F.L, F.U | Properties materialize the same factors; propertynames(F) lists both properties. |
| CuModcopy(A) | Base.copy(A) | Logical values and metadata are unchanged; newly allocated padding is rounded up and zeroed. |
| CuModArray(A, N; elem_type=T) | CuModArray{T,D}(A; modulus=N) | The unparameterized constructor still defaults to Float32. |
| CuModMatrix(A, N; elem_type=T) | CuModMatrix{T}(A; modulus=N) | The unparameterized constructor still defaults to Float32. |
| CuModVector(A, N; elem_type=T) | CuModVector{T}(A; modulus=N) | The unparameterized constructor still defaults to Float32. |
| similar(A, N::Integer) | similar_with_modulus(A, N) | similar(A) remains unchanged. |
| mulN!(C, A, B) | LinearAlgebra.mul!(C, A, B) | The byte-identical duplicate was removed. |
| Package-local rmul!(A, s), lmul!(s, A) | LinearAlgebra.rmul!(A, s), LinearAlgebra.lmul!(s, A) | Methods now extend LinearAlgebra's generics. |
| negate!(B, A, mod_N) | neg!(B, A; modulus_override=...) | Negation uses the per-call reduction modulus without changing stored metadata. |
| mod_N argument or keyword on elementwise/scalar APIs and qualified stripe helpers | `; modulus_override=...` | One keyword spelling for a positive per-call reduction modulus. Zero and negative overrides are rejected before writes. Mutating operations leave stored modulus metadata unchanged. |
| matrix-vector `mul!`/`stripe_mul!`: `R`, `P`, `M` | `input_bound`, `modulus_override`, `max_ops` | `input_bound` is exclusive; `max_ops` counts product contributions accumulated per output entry between reductions. The low-level vector adapter retains `respect_max_ops=false` to recompute its safe budget. |
| matrix-matrix `mul!`/`stripe_mul!`: `M`, `N` | `max_ops`, `input_bound` | The default budget follows the effective input bound and reduction modulus; safe explicit budgets are honored, unsafe values throw, and `input_bound` may require multiple stripes. Ordinary CuModMatrix `LinearAlgebra.mul!` accepts `input_bound` and `max_ops`; `modulus_override` remains on qualified `stripe_mul!`. |
| `mat_mul_gpu_type(A, B)`, `mat_mul_type_inplace!(C, A, B)` | qualified `stripe_mul(A, B)`, `stripe_mul!(C, A, B)` | The allocating and destination-writing methods call the raw stripe core; the retired direct helper names have no compatibility aliases. |
| `stripe_mul`, `stripe_mul!` | qualified public functions, not exported | Matrix multiplication defaults to `modulus(C)` in place and `modulus(A)` when allocating; matrix-vector multiplication defaults to `modulus(A)` and requires matching destination metadata without an override. |
| low-level `stripe_mul!` `maxopsOverride` | `respect_max_ops` | `true` honors a supplied `max_ops`; `false` recomputes the safe vector stripe budget. High-level matrix-vector `mul!` drops its ineffective `maxopsOverride` keyword. |
| `forward_sub_gpu_type_32`, `backward_sub_gpu_type_32` | private destination-first triangular launch helpers | The allocating helper names are retired; recursive triangular inversion writes through the private helpers. |
| add!, sub!, zero! | unchanged | These names remain available as qualified public functions. |
| KMatMul!(C, A, B) | LinearAlgebra.mul!(C, A, B) | Extends LinearAlgebra's multiplication method. |
| KMatMul_gemv!(C, A, x) | karatsuba_gemv!(C, A, x) | Name only; behavior is unchanged. |
| Karatsubacopy(A) | Base.copy(A) | The original rename was name-only; the approved 0.4.0 correction preserves representation metadata `M`, deep-copies component storage with the new padded geometry, and creates a fresh plan for a planned source. |
| MatToKMat(...) | to_karatsuba(...) | Conversion helper; all overloads are renamed. |
| KMatToMat(T, K) | from_karatsuba(T, K) | Conversion helper; argument order is unchanged. |

Karatsuba matrix multiplication prepares logical component views and
preflights all three raw stripe products before any plan writes. Each product
executes through the existing raw `_stripe_mul!` core. The allocating
`A * B` form preserves caller components and plans; the mutating form
requires initialized plans and supports matching Float32 or Float64 GPU
storage. The same precision-aware support applies to matrix-vector `mul!` and
allocating `A * x`, which returns a vector-shaped result with private plans.
The exact-integer budgets are 24 bits for Float32 and 53 bits for Float64.
Results use destination split capacity independently of the shared
representation modulus. The inherited general high-plane GEMV arithmetic
limitation remains unresolved; this extension does not claim general GEMV
correctness.

### Empty products, triangular inverses, and tests

Default-budget stripe products with an empty inner dimension return shaped
zeros even when the derived safe budget is zero. Explicit unsafe budgets remain
invalid. Wide upper- and lower-triangular inverses recurse within the leading
square block and return a right inverse with zero trailing rows.

Karatsuba basic arithmetic and allocation suites are included and invoked under
the existing CUDA gate, with distinct helper names, warmed allocation checks,
and raw-CuArray unary coverage. These source changes do not establish device
correctness or allocation results on machines where CUDA is unavailable. The
known general high-plane GEMV arithmetic limitation remains unresolved.

The qualified `change_modulus!!` consumes its wrapper and shares storage with
the returned wrapper. Shrinking reduces values in place; equal or larger
moduli reinterpret unchanged values. Callers must retain the returned wrapper
and stop using the consumed one.

The public declarations export the everyday array, PLUQ, inverse, batch,
elementwise-operation, and Karatsuba conversion names. Advanced operations
remain available through qualified access, including karatsuba_gemv!,
similar_with_modulus, inverse_pluq, the fixed-size batch kernels, triangular
inverses, low-level multiplication helpers, and TILE_WIDTH. The helper
pluq_check_identity remains internal. Base.inv, Base.copy, and LinearAlgebra
methods are available through their owning modules.
