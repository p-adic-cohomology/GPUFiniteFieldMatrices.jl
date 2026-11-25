@static if isdefined(CUDA, :CuRefValue)
    const _CUDA_CU_REF_VALUE = getfield(CUDA, :CuRefValue)
elseif isdefined(CUDA, :CUDACore) && isdefined(CUDA.CUDACore, :CuRefValue)
    const _CUDA_CU_REF_VALUE = getfield(CUDA.CUDACore, :CuRefValue)
else
    throw(ArgumentError("the supported CUDA version does not expose its CuRef value type"))
end

const _CUBLAS_SCALAR_PAIR{T} = Tuple{_CUDA_CU_REF_VALUE{T},_CUDA_CU_REF_VALUE{T}}

const _cublas_scalar_cache_f64 = IdDict{Task,Tuple}()
const _cublas_scalar_cache_f32 = IdDict{Task,Tuple}()
const _cublas_scalar_cache_f16 = IdDict{Task,Tuple}()

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f64()::_CUBLAS_SCALAR_PAIR{Float64}
    t = current_task()
    get!(_cublas_scalar_cache_f64, t) do
        (CuRef(Float64(0.0)), CuRef(Float64(1.0)))
    end
end

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f32()::_CUBLAS_SCALAR_PAIR{Float32}
    t = current_task()
    get!(_cublas_scalar_cache_f32, t) do
        (CuRef(Float32(0.0)), CuRef(Float32(1.0)))
    end
end

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f16()::_CUBLAS_SCALAR_PAIR{Float16}
    t = current_task()
    get!(_cublas_scalar_cache_f16, t) do
        (CuRef(Float16(0.0)), CuRef(Float16(1.0)))
    end
end

# The same-type GEMM entry points used below take scalars in the default compute
# type. Float16 therefore uses Float16 scalars here; no alternate compute type is
# requested by this call.
@inline function _stripe_gemm_scalars(::Type{Float64})::_CUBLAS_SCALAR_PAIR{Float64}
    return cublas_scalars_f64()
end

@inline function _stripe_gemm_scalars(::Type{Float32})::_CUBLAS_SCALAR_PAIR{Float32}
    return cublas_scalars_f32()
end

@inline function _stripe_gemm_scalars(::Type{Float16})::_CUBLAS_SCALAR_PAIR{Float16}
    return cublas_scalars_f16()
end

function _stripe_gemv_scalars(::Type{Float64})
    return cublas_scalars_f64()
end

function _stripe_gemv_scalars(::Type{Float32})
    return cublas_scalars_f32()
end

function _stripe_gemv_scalars(::Type{T}) where {T}
    return (CuRef(zero(T)), CuRef(one(T)))
end

@inline function _stripe_accumulate_stripe!(
    Cdata::AbstractMatrix,
    Adata::AbstractMatrix,
    Bdata::AbstractMatrix,
    stripe,
    first_stripe,
    scalars,
)
    zero_ptr, one_ptr = scalars
    _CUDA_BLAS.gemm!(
        'N',
        'N',
        one_ptr,
        @view(Adata[:, stripe]),
        @view(Bdata[stripe, :]),
        first_stripe ? zero_ptr : one_ptr,
        Cdata,
    )
    return Cdata
end

@inline function _stripe_accumulate_stripe!(
    zdata::AbstractVector,
    Adata::AbstractMatrix,
    xdata::AbstractVector,
    stripe,
    first_stripe,
    scalars,
)
    zero_ptr, one_ptr = scalars
    _CUDA_BLAS.gemv!(
        'N',
        one_ptr,
        @view(Adata[:, stripe]),
        @view(xdata[stripe]),
        first_stripe ? zero_ptr : one_ptr,
        zdata,
    )
    return zdata
end

# Legacy CUDA inverse and Karatsuba callers use these algorithm helpers.
const _stripe_storage_root = _cuda_root
const _stripe_logical_data = FFM._stripe_logical_data
const _stripe_mightalias = FFM._stripe_mightalias
const _stripe_safe_chunk = FFM._stripe_safe_chunk

_cuda_stripe_layout(::AbstractArray) = false

function _cuda_stripe_layout(A::CUDA.StridedCuMatrix)
    return (stride(A, 1) == 1 || size(A, 1) == 1) && stride(A, 2) >= size(A, 1)
end

function _cuda_stripe_layout(x::CUDA.StridedCuVector{T}) where {T}
    # cuBLAS uses GEMM for Float16 vectors and GEMV for Float32/64 vectors.
    return T === Float16 ? stride(x, 1) == 1 : stride(x, 1) > 0
end

function _check_cuda_stripe_layout(Cdata, Adata, Bdata)
    if eltype(Cdata) === Float32 && CUDA.math_mode() == CUDA.FAST_MATH
        throw(
            ArgumentError("exact modular GEMM requires CUDA DEFAULT_MATH or PEDANTIC_MATH"),
        )
    end
    (isempty(Cdata) || size(Adata, 2) == 0) && return nothing
    _cuda_stripe_layout(Cdata) &&
    _cuda_stripe_layout(Adata) &&
    _cuda_stripe_layout(Bdata) ||
        throw(ArgumentError("matrix product requires cuBLAS-compatible CUDA strides"))
    return nothing
end

function _stripe_preflight(Cdata, Adata, Bdata, modulus, input_bound, max_ops)
    size(Cdata) == FFM._stripe_output_size(Adata, Bdata) ||
        throw(DimensionMismatch("destination dimensions do not match the matrix product"))
    T = FFM._stripe_check_eltypes(Cdata, Adata, Bdata)
    roots = map(FFM._elementwise_native_root, (Cdata, Adata, Bdata))
    all(root -> root isa CuArray, roots) ||
        throw(ArgumentError("stripe multiplication requires CUDA arrays or CUDA views"))
    FFM._stripe_check_alias(Cdata, Adata, Bdata)
    FFM._stripe_check_devices(roots...)
    chunk = _stripe_safe_chunk(
        T,
        modulus,
        input_bound,
        max_ops;
        empty_inner = size(Adata, 2) == 0,
    )
    _check_cuda_stripe_layout(Cdata, Adata, Bdata)
    return chunk
end

function _legacy_cuda_stripe_run!(Cdata, Adata, Bdata, modulus, chunk)
    isempty(Cdata) && return Cdata
    k = size(Adata, 2)
    k == 0 && return fill!(Cdata, zero(eltype(Cdata)))

    T = eltype(Cdata)
    scalars = Cdata isa AbstractVector ? _stripe_gemv_scalars(T) : _stripe_gemm_scalars(T)
    for stripe in FFM._stripe_ranges(k, chunk)
        _stripe_accumulate_stripe!(Cdata, Adata, Bdata, stripe, first(stripe) == 1, scalars)
        FFM._stripe_reduce!(Cdata, modulus)
    end
    return Cdata
end

# The legacy inverse and Karatsuba consumers retain their task-local device
# scalars and only write the native destination view.
function _stripe_mul!(
    Cdata::Union{AbstractMatrix,AbstractVector},
    Adata::AbstractMatrix,
    Bdata::Union{AbstractMatrix,AbstractVector};
    modulus,
    input_bound = nothing,
    max_ops = nothing,
)
    chunk = _stripe_preflight(Cdata, Adata, Bdata, modulus, input_bound, max_ops)
    return _with_cuda_device(CUDA.device(_cuda_root(Cdata))) do
        _legacy_cuda_stripe_run!(Cdata, Adata, Bdata, modulus, chunk)
    end
end

function _cuda_stripe_product!(C, A, B; modulus_override, input_bound, max_ops)
    Cdata, Adata, Bdata, effective_modulus, chunk =
        FFM._stripe_product_args(C, A, B; modulus_override, input_bound, max_ops)
    _check_cuda_stripe_layout(Cdata, Adata, Bdata)
    _with_cuda_device(CUDA.device(_cuda_root(Cdata))) do
        fill!(parent(C), zero(eltype(C)))
        FFM._stripe_run!(Cdata, Adata, Bdata, effective_modulus, chunk)
    end
    return C
end

function stripe_mul!(
    C::CuModMatrix,
    A::CuModMatrix,
    B::CuModMatrix;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    return _cuda_stripe_product!(C, A, B; modulus_override, input_bound, max_ops)
end

function stripe_mul!(
    z::CuModVector,
    A::CuModMatrix,
    x::CuModVector;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
    respect_max_ops::Bool = true,
)
    return _cuda_stripe_product!(
        z,
        A,
        x;
        modulus_override,
        input_bound,
        max_ops = respect_max_ops ? max_ops : nothing,
    )
end
