module GPUFiniteFieldMatrices

using DispatchDoctor: @stable
using LinearAlgebra: LinearAlgebra
import Random
using SparseArrays: SparseArrays
import GPUArrays
import KernelAbstractions
import KernelInterface

const DEBUG = false

function add! end
function sub! end
function neg! end
function scalar_add! end
function scalar_sub! end
function elementwise_multiply! end
function mod_elements! end
function similar_with_modulus end
function change_modulus end
function change_modulus!! end
function stripe_mul end
function stripe_mul! end
function karatsuba_gemv! end
function scalar_multiply! end
function negate! end
function divide_elements! end
function pluq end
function pluq! end
function inverse_pluq end
function pluq_batched! end
function pluq_batched end
function inverse_batched end
function pluq_batched_4x4! end
function pluq_batched_8x8! end
function pluq_batched_16x16! end
function pluq_batched_32x32! end
function pluq_batched_NxN! end
function inverse_batched_4x4! end
function inverse_batched_8x8! end
function inverse_batched_16x16! end
function inverse_batched_32x32! end
function is_invertible end
function is_invertible_with_inverse end
function right_inverse end
function left_inverse end
function identity_matrix end
function initialize_plan! end
function upper_triangular_inverse_no_copy end
function lower_triangular_inverse_no_copy end

@stable default_mode = "disable" begin
    include("backend/storage.jl")
    include("backend/selection.jl")
    include("backend/runtime.jl")

    include("CuModMatrix/CuModMatrix.jl")
    include("CuModMatrix/host_math.jl")
    include("CuModMatrix/inverse/types.jl")
    include("CuModMatrix/inverse/mod_arith.jl")
    include("CuModMatrix/inverse/perm_vectors.jl")

    include("KaratsubaMatrix/KaratsubaMatrix.jl")
    include("base_methods.jl")
end

export GPUModArray, GPUModMatrix, GPUModVector, CuModArray, CuModMatrix, CuModVector
export MtlModArray, MtlModMatrix, MtlModVector
export KaratsubaArray, KaratsubaMatrix, KaratsubaVector
export to_karatsuba, from_karatsuba
export modulus, low_modulus, high_modulus, low_part, high_part
export PLUQOptions, PLUQ
export pluq, pluq!, is_invertible, is_invertible_with_inverse
export pluq_batched!, pluq_batched, inverse_batched
export right_inverse, left_inverse, change_modulus, identity_matrix
export elementwise_multiply!, scalar_add!, scalar_sub!, mod_elements!
export default_backend, set_default_backend!, with_backend

public add!, sub!, neg!, zero!, zeros, rand, similar_with_modulus, change_modulus!!, mod_inv
public initialize_plan!
public karatsuba_gemv!, inverse_pluq
public pluq_batched_4x4!, pluq_batched_8x8!, pluq_batched_16x16!, pluq_batched_32x32!
public inverse_batched_4x4!,
    inverse_batched_8x8!, inverse_batched_16x16!, inverse_batched_32x32!
public pluq_batched_NxN!, upper_triangular_inverse_no_copy, lower_triangular_inverse_no_copy
public stripe_mul, stripe_mul!, TILE_WIDTH, ModulusMismatchError

end
