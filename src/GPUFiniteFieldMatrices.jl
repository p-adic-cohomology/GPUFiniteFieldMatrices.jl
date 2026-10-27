module GPUFiniteFieldMatrices

using CUDA:
    CUDA,
    @cuda,
    CuArray,
    CuDeviceMatrix,
    CuDeviceVector,
    CuRef,
    CuStream,
    blockDim,
    blockIdx,
    gridDim,
    synchronize,
    threadIdx

# CUDA 6 uses cuBLAS while CUDA 5.11 exposes only CUBLAS, so select the available binding with getfield to avoid resolving CUDA 6's deprecated alias.
@static if isdefined(CUDA, :cuBLAS)
    const _CUDA_BLAS = CUDA.cuBLAS
else
    const _CUDA_BLAS = getfield(CUDA, :CUBLAS)
end

using LinearAlgebra: LinearAlgebra, I, mul!, rank
using SparseArrays: SparseArrays
using IterTools: IterTools

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

# Public API
export CuModArray, CuModMatrix, CuModVector
export KaratsubaArray, KaratsubaMatrix, KaratsubaVector
export to_karatsuba, from_karatsuba
export PLUQOptions, PLUQ
export pluq, pluq!, is_invertible, is_invertible_with_inverse
export pluq_batched, inverse_batched
export right_inverse, left_inverse, change_modulus, eye
export elementwise_multiply!, scalar_add!, scalar_sub!, mod_elements!

public add!, sub!, neg!, zero!
public zeros, rand, similar_with_modulus, change_modulus_no_alloc!, mod_inv
public karatsuba_gemv!
public inverse_pluq
public pluq_batched_4x4!, pluq_batched_8x8!, pluq_batched_16x16!, pluq_batched_32x32!
public inverse_batched_4x4!,
    inverse_batched_8x8!, inverse_batched_16x16!, inverse_batched_32x32!
public upper_triangular_inverse_no_copy, lower_triangular_inverse_no_copy
public stripe_mul, stripe_mul!, TILE_WIDTH

end
