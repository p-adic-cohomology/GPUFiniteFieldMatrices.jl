module GPUFiniteFieldMatricesCUDAExt

import CUDA
using CUDA: CuArray, CuDeviceVector, CuRef, CuStaticSharedArray, CuStream
using CUDA.CUDACore:
    @cuda, blockDim, blockIdx, gridDim, sync_threads, synchronize, threadIdx
using DispatchDoctor: @stable
using LinearAlgebra: LinearAlgebra, I, mul!, rank
using SparseArrays: SparseArrays
import GPUFiniteFieldMatrices
import GPUFiniteFieldMatrices: GPUModArray, CuModArray, CuModMatrix, CuModVector
import GPUFiniteFieldMatrices: KaratsubaArray
import GPUFiniteFieldMatrices: PLUQ, PLUQOptions, ModulusMismatchError
import GPUFiniteFieldMatrices: modulus, low_modulus, high_modulus, low_part, high_part
import GPUFiniteFieldMatrices: add!, sub!, rscalar_sub!, mod!
import GPUFiniteFieldMatrices: similar_with_modulus, stripe_mul, stripe_mul!, karatsuba_gemv!
import GPUFiniteFieldMatrices: scalar_multiply!, negate!, divide_elements!, pluq, pluq!
import GPUFiniteFieldMatrices: inverse_pluq, pluq_batched!, pluq_batched, inverse_batched
import GPUFiniteFieldMatrices: pluq_batched_4x4!, pluq_batched_8x8!, pluq_batched_16x16!
import GPUFiniteFieldMatrices: pluq_batched_32x32!, pluq_batched_NxN!
import GPUFiniteFieldMatrices: inverse_batched_4x4!, inverse_batched_8x8!
import GPUFiniteFieldMatrices: inverse_batched_16x16!, inverse_batched_32x32!
import GPUFiniteFieldMatrices: is_invertible, is_invertible_with_inverse
import GPUFiniteFieldMatrices: right_inverse, left_inverse, identity_matrix
import GPUFiniteFieldMatrices:
    upper_triangular_inverse_no_copy, lower_triangular_inverse_no_copy
import GPUFiniteFieldMatrices: from_karatsuba, initialize_plan!
import GPUFiniteFieldMatrices: _logical_view, _validate_storage_modulus
import GPUFiniteFieldMatrices:
    _validate_backend_storage, _normalize_logical_size, _storage_modulus_limit
import GPUFiniteFieldMatrices: _karatsuba_plan_type
import GPUFiniteFieldMatrices: _backend_default_component_type, _backend_default_plan_type
import GPUFiniteFieldMatrices: find_max_ops, find_max_stripe_ops
import GPUFiniteFieldMatrices: inverse_permutation, mod_inv
import GPUFiniteFieldMatrices: _padded_extent, _padded_size
import GPUFiniteFieldMatrices: _pluq_extract_L, _pluq_extract_U
import GPUFiniteFieldMatrices:
    _karatsuba_matmul_product_plan, _karatsuba_gemv_validate_radices
import GPUFiniteFieldMatrices: _karatsuba_gemv_precision_plan
import GPUFiniteFieldMatrices: _karatsuba_component, _karatsuba_host_array
import GPUFiniteFieldMatrices: _karatsuba_zero_component
import KernelAbstractions
import KernelInterface

const FFM = GPUFiniteFieldMatrices
const TILE_WIDTH = FFM.TILE_WIDTH
const DEFAULT_TYPE = Float32

@static if isdefined(CUDA, :cuBLAS)
    const _CUDA_BLAS = CUDA.cuBLAS
else
    const _CUDA_BLAS = getfield(CUDA, :CUBLAS)
end

include("GPUFiniteFieldMatricesCUDAExt/storage.jl")
include("GPUFiniteFieldMatricesCUDAExt/runtime.jl")
include("GPUFiniteFieldMatricesCUDAExt/cumod_methods.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/kernel_mul/mat_mul_gpu_direct.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/kernel_mul/mat_mul_ops.jl")
include("GPUFiniteFieldMatricesCUDAExt/stripe_mul.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/KaratsubaMatrix/KaratsubaKernels.jl")
include("GPUFiniteFieldMatricesCUDAExt/karatsuba_methods.jl")
# The v0.4.0 CUDA elementwise kernels remain under legacy/ for comparison.
# Elementwise arithmetic, mutation and wrapper-copy methods now live in core
# and dispatch through the shared KernelAbstractions implementation.
include(
    "GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/triangular/triangular_inverse_no_copy.jl",
)
include(
    "GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/triangular/substitution_inplace.jl",
)
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/basecase_pluq.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/rectangular_pluq.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/trsm.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/schur_update.jl")
include(
    "GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/blocked_recursive_pluq.jl",
)
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/extract.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/validation.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/api.jl")
include("GPUFiniteFieldMatricesCUDAExt/legacy/CuModMatrix/inverse/batched_tiny.jl")

end
