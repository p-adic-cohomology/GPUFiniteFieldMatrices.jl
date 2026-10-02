module GPUFiniteFieldMatrices

using CUDA:
    CUDA,
    @cuda,
    CuArray,
    CuDeviceMatrix,
    CuDeviceVector,
    blockDim,
    blockIdx,
    gridDim,
    threadIdx
using LinearAlgebra: LinearAlgebra, I, mul!, rank
using SparseArrays: SparseArrays
using IterTools: IterTools
using BenchmarkTools: BenchmarkTools
using CSV: CSV
using DelimitedFiles: DelimitedFiles

# This check keeps return types predictable so the functions below stay fast.
using DispatchDoctor: @stable

const DEBUG = false

@stable default_mode = "disable" begin

    include("CuModMatrix/CuModMatrix.jl")

    include("CuModMatrix/kernel_mul/mat_mul_gpu_direct.jl")
    include("CuModMatrix/kernel_mul/mat_mul_ops.jl")
    include("CuModMatrix/kernel_mul/stripe_mul.jl")

    include("KaratsubaMatrix/KaratsubaMatrix.jl")
    include("KaratsubaMatrix/KaratsubaKernels.jl")

    include("CuModMatrix/kernel_ops/common.jl")
    include("CuModMatrix/kernel_ops/add_ops.jl")
    include("CuModMatrix/kernel_ops/sub_ops.jl")
    include("CuModMatrix/kernel_ops/mul_ops.jl")
    include("CuModMatrix/kernel_ops/div_ops.jl")
    include("CuModMatrix/kernel_ops/mod_ops.jl")

    include("CuModMatrix/triangular/triangular_inverse_no_copy.jl")
    include("CuModMatrix/triangular/substitution_inplace.jl")
    include("CuModMatrix/inverse/types.jl")
    include("CuModMatrix/inverse/mod_arith.jl")
    include("CuModMatrix/inverse/perm_vectors.jl")
    include("CuModMatrix/inverse/basecase_pluq.jl")
    include("CuModMatrix/inverse/rectangular_pluq.jl")
    include("CuModMatrix/inverse/trsm.jl")
    include("CuModMatrix/inverse/schur_update.jl")
    include("CuModMatrix/inverse/blocked_recursive_pluq.jl")
    include("CuModMatrix/inverse/extract.jl")
    include("CuModMatrix/inverse/validation.jl")
    include("CuModMatrix/inverse/api.jl")
    include("CuModMatrix/inverse/batched_tiny.jl")

end # @stable default_mode = "disable"

# Export the main type and its operations
export CuModArray, CuModMatrix, CuModVector
export inverse

export KaratsubaArray, KaratsubaMatrix, KaratsubaVector

# Export utility functions
export eye
export change_modulus, change_modulus_no_alloc!
export elementwise_multiply!, negate!
export scalar_add!, scalar_sub!, rmul!, lmul!
export mod_elements!, fill!

# do not export: add!, sub!, zero!, is_invertible, is_invertible_with_inverse
#     (since they conflict with AbstractAlgebra

# Export GPU operations
export mat_mul_gpu_type, mat_mul_type_inplace!
export is_invertible, inverse, is_invertible_with_inverse
export mod_inv
export upper_triangular_inverse_no_copy, lower_triangular_inverse_no_copy
export forward_sub_gpu_type_32, backward_sub_gpu_type_32
export PLUQOptions, PLUQFactorization
export pluq_new, pluq_new!, inverse_new, inverse_pluq_new, is_invertible_new
export pluq_new_batch, inverse_new_batch
export pluq_batched_4x4!, pluq_batched_8x8!, pluq_batched_16x16!, pluq_batched_32x32!
export inverse_batched_4x4!,
    inverse_batched_8x8!, inverse_batched_16x16!, inverse_batched_32x32!
export right_inverse_new, left_inverse_new
export pluq_extract_L, pluq_extract_U, pluq_check_identity

end
