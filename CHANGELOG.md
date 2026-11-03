# Changelog

## Unreleased

### Breaking: API renames

This change prepares the 0.4.0 API break. No compatibility aliases are kept.

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
| CuModcopy(A) | Base.copy(A) | Copy semantics are unchanged. |
| CuModArray(A, N; elem_type=T) | CuModArray{T,D}(A, N) | The unparameterized constructor still defaults to Float32. |
| CuModMatrix(A, N; elem_type=T) | CuModMatrix{T}(A, N) | The unparameterized constructor still defaults to Float32. |
| CuModVector(A, N; elem_type=T) | CuModVector{T}(A, N) | The unparameterized constructor still defaults to Float32. |
| similar(A, N::Integer) | similar_with_modulus(A, N) | similar(A) remains unchanged. |
| mulN!(C, A, B) | LinearAlgebra.mul!(C, A, B) | The byte-identical duplicate was removed. |
| Package-local rmul!(A, s), lmul!(s, A) | LinearAlgebra.rmul!(A, s), LinearAlgebra.lmul!(s, A) | Methods now extend LinearAlgebra's generics. |
| negate!(B, A, mod_N) | neg!(B, A; modulus_override=...) | Negation uses the per-call reduction modulus without changing stored metadata. |
| mod_N argument or keyword on elementwise/scalar operations and direct matrix helpers | `; modulus_override=...` | One keyword spelling for a per-call reduction modulus. Mutating stripe operations leave stored modulus metadata unchanged. |
| matrix-vector `mul!`/`stripe_mul!`: `R`, `P`, `M` | `input_bound`, `modulus_override`, `max_ops` | `input_bound` is exclusive; `max_ops` counts product contributions accumulated per output entry between reductions. The low-level vector adapter retains `respect_max_ops=false` to recompute its safe budget. |
| matrix-matrix `mul!`/`stripe_mul!`: `M`, `N` | `max_ops`, `input_bound` | Supplying `input_bound` selects one stripe only when the full inner dimension fits the exact-integer budget. |
| `mat_mul_gpu_type(A, B)`, `mat_mul_type_inplace!(C, A, B)` | qualified `stripe_mul(A, B)`, `stripe_mul!(C, A, B)` | The allocating and destination-writing adapters share one stripe implementation; the retired direct helper names have no compatibility aliases. |
| `stripe_mul`, `stripe_mul!` | qualified public functions, not exported | Matrix multiplication uses `C.N` for an omitted in-place override and `A.N` for an omitted allocating override; matrix-vector multiplication uses `z.N`. |
| low-level `stripe_mul!` `maxopsOverride` | `respect_max_ops` | `true` honors a supplied `max_ops`; `false` recomputes the safe vector stripe budget. High-level matrix-vector `mul!` drops its ineffective `maxopsOverride` keyword. |
| add!, sub!, zero! | unchanged | These names remain available as qualified public functions. |
| KMatMul!(C, A, B) | LinearAlgebra.mul!(C, A, B) | Extends LinearAlgebra's multiplication method. |
| KMatMul_gemv!(C, A, x) | karatsuba_gemv!(C, A, x) | Name only; behavior is unchanged. |
| Karatsubacopy(A) | Base.copy(A) | Copy semantics are unchanged. |
| MatToKMat(...) | to_karatsuba(...) | Conversion helper; all overloads are renamed. |
| KMatToMat(T, K) | from_karatsuba(T, K) | Conversion helper; argument order is unchanged. |

Three existing Karatsuba matrix-matrix calls still pass legacy `R`, `P`, and
`maxopsOverride` keywords and raise `MethodError`. This path remains unsupported
and is tracked by Phase 2 issue `gfm-2x1`.

The public declarations export the everyday array, PLUQ, inverse, batch,
elementwise-operation, and Karatsuba conversion names. Advanced operations
remain available through qualified access, including karatsuba_gemv!,
similar_with_modulus, inverse_pluq, the fixed-size batch kernels, triangular
inverses, low-level multiplication helpers, and TILE_WIDTH. The helper
pluq_check_identity remains internal. Base.inv, Base.copy, and LinearAlgebra
methods are available through their owning modules.
