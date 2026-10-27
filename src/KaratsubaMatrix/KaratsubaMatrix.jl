mutable struct KaratsubaArray{T,D} <: AbstractArray{T,D}
    data1::AbstractArray{T,D}
    data2::AbstractArray{T,D}
    plan::Union{Nothing,AbstractArray{T,D}}
    N1::Integer
    N2::Integer
    M::Integer

    function KaratsubaArray{T,D}(
        A::AbstractArray{T,D},
        B::AbstractArray{T,D},
        ;
        low_modulus::Union{Nothing,Integer} = nothing,
        high_modulus::Union{Nothing,Integer} = nothing,
        modulus::Union{Nothing,Integer} = nothing,
    ) where {T,D}
        if size(A) != size(B)
            error("Dimensions of matrices must match")
        end
        #=
        if !(all(x->x<N,A) && all(x->x<N,B))
            error("Cannot have entries larger than modulus in matrices")
        end
        =#
        N1, N2, M = _karatsuba_parameters(
            A;
            parameter_type = T,
            low_modulus = low_modulus,
            high_modulus = high_modulus,
            modulus = modulus,
        )
        (N1 > 0 && N2 > 0 && M > 0) ||
            throw(ArgumentError("Karatsuba moduli must be positive"))
        bits = T <: AbstractFloat ? get_bits(T) - 2 : get_bits(T)
        if M >= BigInt(2)^bits
            error("Modulus too large")
        end

        return new{T,D}(A, B, nothing, N1, N2, M)
    end
end

const KaratsubaMatrix{T} = KaratsubaArray{T,2}
const KaratsubaVector{T} = KaratsubaArray{T,1}

function KaratsubaArray(A::AbstractArray{T,D}, B::AbstractArray{T,D}; kwargs...) where {T,D}
    _wrap_karatsuba(Val(D), A, B; kwargs...)
end

function _karatsuba_parameters(
    A::AbstractArray;
    parameter_type::Type = eltype(A),
    low_modulus::Union{Nothing,Integer} = nothing,
    high_modulus::Union{Nothing,Integer} = nothing,
    modulus::Union{Nothing,Integer} = nothing,
)
    inferred =
        modulus === nothing ? find_max_ops_karatsuba(parameter_type, max(size(A)...)) :
        modulus
    N1 = low_modulus === nothing ? inferred : low_modulus
    N2 = high_modulus === nothing ? inferred : high_modulus
    return (Int(N1), Int(N2), Int(inferred))
end

function _wrap_karatsuba(
    ::Val{D},
    A::AbstractArray{T,D},
    B::AbstractArray{T,D};
    kwargs...,
) where {T,D}
    N1, N2, M = _karatsuba_parameters(A; kwargs...)
    return KaratsubaArray{T,D}(A, B; low_modulus = N1, high_modulus = N2, modulus = M)
end

function KaratsubaMatrix(A::AbstractMatrix{T}, B::AbstractMatrix{T}; kwargs...) where {T}
    _wrap_karatsuba(Val(2), A, B; kwargs...)
end

function KaratsubaVector(A::AbstractVector{T}, B::AbstractVector{T}; kwargs...) where {T}
    _wrap_karatsuba(Val(1), A, B; kwargs...)
end

#=
struct KMatMulPlan{T,D}
    temp1::AbstractArray{T,D}
    temp2::AbstractArray{T,D}
    temp1mod::Integer
    temp2mod::Integer

    function KMatMulPlan{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D},N1::Integer,N2::Integer) where {T,D}
        new{T,D}(A,B,N1,N2)
    end
end
=#

#=
function zerosplan(T::Type,rows::Integer,cols::Integer,N1::Integer,N2::Integer,use_gpu=false)
    if use_gpu == true
        return KMatMulPlan{T,2}(zeros(T,rows,cols,N1),zeros(T,rows,cols,N2),N1,N2)
    else
        return KMatMulPlan{T,2}(zeros(T,rows,cols),zeros(T,rows,cols),N1,N2)
    end
end
=#

function find_max_ops_karatsuba(type, N)
    N > 0 || throw(ArgumentError("The maximum matrix dimension must be positive"))
    bits = get_bits(type)
    safe_bits = type <: AbstractFloat ? max(bits - 2, 1) : bits
    bound = ((BigInt(1) << safe_bits) - 1) ÷ BigInt(N)^2 - 1
    return Int(max(1, min(bound, BigInt(typemax(Int)))))
end

"""
    LinearAlgebra.mul!(C, A, B)

Multiply the Karatsuba matrices `A` and `B`, storing the product in `C`. The
three matrices must share the same representation modulus `M`, and `A`'s column
count must match `B`'s row count. The caller must ensure that `C` has dimensions
`(size(A, 1), size(B, 2))`; this method does not explicitly validate the
destination dimensions before CUDA work begins. Call `initialize_plan!` on all
three matrices before multiplying. This matrix path currently supports
`Float64` storage only.
"""
function LinearAlgebra.mul!(
    C::KaratsubaArray{TC,2},
    A::KaratsubaArray{TA,2},
    B::KaratsubaArray{TB,2},
) where {TC,TA,TB}
    if (A.M != B.M) || (A.M != C.M)
        error("Matrices must have the same modulus m")
    end
    if size(A.data1)[2] != size(B.data1)[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (A.plan == nothing) || (B.plan == nothing) || (C.plan == nothing)
        error("Must initialize plans first")
    end
    if eltype(A.data1.data) != Float64
        error("Not implemented for non-Float64 entries")
    end
    # TODO: enable this
    # maxN = max(A.N1,A.N2)
    # if find_max_stripe_ops(eltype(A.data1.data), maxN) < size(A,1)
    #     error("Matrix is too big for a single matmul, this is currently not implemented")
    # end
    #=
    tw = TILE_WIDTH

    totalsize = length(C.data1.data) # the vector length

    threads = tw
    blocks = totalsize ÷ tw
    =#

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    #A_cols = size(A.data1,2)
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        A.plan.data,
        A.data1.data,
        A.data2.data,
        #A_cols,
        B.plan.data,
        B.data1.data,
        B.data2.data,
        A.N1,
    )
    # add!(A.plan,A.data1,A.data2; modulus_override=2*A.N1)
    # add!(B.plan,B.data1,B.data2; modulus_override=2*B.N1)

    # use gemm

    LinearAlgebra.mul!(
        C.data1,
        A.data1,
        B.data1,
        R = A.N1,
        P = A.N1^2,
        maxopsOverride = false,
    )

    LinearAlgebra.mul!(
        C.data2,
        A.plan,
        B.plan,
        R = (2*A.N1),
        P = ((4*A.N1)^2),
        maxopsOverride = false,
    )
    LinearAlgebra.mul!(
        B.plan,
        A.data2,
        B.data2,
        R = (A.N1),
        P = A.N1^2,
        maxopsOverride = false,
    )


    tw = TILE_WIDTH

    totalsize = length(C.data1.data)

    threads = tw
    blocks = totalsize ÷ tw


    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_2!(
        C.data1.data,
        C.data2.data,
        B.plan.data,
        C.N1,
        C.N2,
    )


    C
end

"""
The number of actual matmuls to multiply one karatsuba matrix.
If a higher Karatsuba multiplication algorithm were implemented,
it could be implemented in a similar way with a bigger constant
"""
const nKMuls = 3

# const _cublas_Apointers_cache_f64 = IdDict{Task,CuArray{Ptr{Float64}}}()
# const _cublas_Bpointers_cache_f64 = IdDict{Task,CuArray{Ptr{Float64}}}()
# const _cublas_Cpointers_cache_f64 = IdDict{Task,CuArray{Ptr{Float64}}}()

# @inline function cublas_Apointers_f64()
#     t = current_task()
#     get!(_cublas_Apointers_cache, t) do
#         CuArray{Ptr{Float64}

# """
# The following is a device vector of device pointers
# """
# mat_ptrs_device = CuArray{CuPtr{Float64}}(undef,nKMuls)
# vec_ptrs_device = CuArray{CuPtr{Float64}}(undef,nKMuls)
# target_ptrs_device = CuArray{CuPtr{Float64}}(undef,nKMuls)

# """
# The following is a host vector of device pointers
# """
# mat_ptrs_host = CUDA.pin(Vector{CuPtr{Float64}}(undef,nKMuls))
# vec_ptrs_host = CUDA.pin(Vector{CuPtr{Float64}}(undef,nKMuls))
# target_ptrs_host = CUDA.pin(Vector{CuPtr{Float64}}(undef,nKMuls))


# the stuff in this function should apply to the general mul!, but with gemm instead of gemv
"""
    karatsuba_gemv!(y, A, x)

Multiply a Karatsuba matrix `A` by a Karatsuba vector `x`, storing the result in
`y`. All three values must have initialized plans, use the same split base `M`,
and have dimensions compatible with the product. This specialized path currently
supports `Float64` storage only. Initialize plans with `initialize_plan!` before
calling this function.
"""
function karatsuba_gemv!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != C.M)
        error("Matrices must have the same modulus m")
    end
    if size(A.data1)[2] != size(B.data1)[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (A.plan == nothing) || (B.plan == nothing) || (C.plan == nothing)
        error("Must initialize plans first")
    end
    if eltype(A.data1.data) != Float64
        error("Not implemented for non-Float64 entries")
    end
    # TOD0: enable this
    # maxN = max(A.N1,A.N2)
    # if find_max_stripe_ops(eltype(A.data1.data), maxN) < size(A,1)
    #     error("Matrix is too big for a single matmul, this is currently not implemented")
    # end

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        A.plan.data,
        A.data1.data,
        A.data2.data,
        B.plan.data,
        B.data1.data,
        B.data2.data,
        A.N1,
    )
    # add!(A.plan,A.data1,A.data2; modulus_override=2*A.N1)
    # add!(B.plan,B.data1,B.data2; modulus_override=2*B.N1)

    (zero_ptr, one_ptr) = cublas_scalars_f64()


    targets = [C.data1.data, C.data2.data, B.plan.data]
    mats = [A.data1.data, A.plan.data, A.data2.data]
    vecs = [B.data1.data, B.plan.data, B.data2.data]
    _CUDA_BLAS.gemv_batched!('N', one_ptr, mats, vecs, zero_ptr, targets)

    # CUDA.CUBLAS.gemv!('N',one_ptr,A.data1.data,B.data1.data,zero_ptr,C.data1.data)
    # CUDA.CUBLAS.gemv!('N',one_ptr,A.plan.data,B.plan.data,zero_ptr,C.data2.data)
    # CUDA.CUBLAS.gemv!('N',one_ptr,A.data2.data,B.data2.data,zero_ptr,B.plan.data)

    tw = TILE_WIDTH

    totalsize = length(C.data1.data)

    threads = tw
    blocks = totalsize ÷ tw
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_2!(
        C.data1.data,
        C.data2.data,
        B.plan.data,
        C.N1,
        C.N2,
    )


    C
end

Base.size(A::KaratsubaArray) = size(A.data1)

Base.getindex(A::KaratsubaArray, i::Int, j::Int) = A.data1[i, j] + A.N1*A.data2[i, j]
Base.getindex(A::KaratsubaArray, i::Int) = A.data1[i] + A.N1*A.data2[i]

function Base.setindex!(A::KaratsubaArray, v, i::Int, j::Int)
    A.data1[i, j] = rem(v, A.N1)
    A.data2[i, j] = div(v, A.N1)
end

function Base.setindex!(A::KaratsubaArray, v, i::Int)
    A.data1[i] = rem(v, A.N1)
    A.data2[i] = div(v, A.N1)
end

#Converts KMat to Mat
function Base.Array(K::KaratsubaArray)
    return _karatsuba_reconstruct_host(BigInt, K)
end

function Base.copy!(B::KaratsubaArray, A::KaratsubaArray)
    copy!(B.data1, A.data1)
    copy!(B.data2, A.data2)
    if A.plan != nothing && B.plan == nothing
        initialize_plan!(B)
    end
    return B
end

function zero!(A::KaratsubaArray)
    zero!(A.data1)
    zero!(A.data2)
end

"""
    from_karatsuba(T::Type, K::KaratsubaArray)

Reconstruct the full array represented by `K` as `K.data1 + K.N1*K.data2`.
The returned array has element type `T` and follows the component backend. Each
integer conversion is checked before returning, so an inexact or overflowing
conversion throws `InexactError`.
"""
function from_karatsuba(::Type{T}, K::KaratsubaArray{S,D}) where {T,S,D}
    if K.data1 isa CuModArray || K.data1 isa CuArray
        return _karatsuba_reconstruct_gpu(T, K)::CUDA.CuArray{T,D}
    end
    return _karatsuba_reconstruct_host(T, K)::Array{T,D}
end

function _karatsuba_reconstruct_host(::Type{T}, K::KaratsubaArray{S,D}) where {T,S,D}
    low = _karatsuba_host_array(K.data1)::Array{S,D}
    high = _karatsuba_host_array(K.data2)::Array{S,D}
    result = Base.Array{T,D}(undef, size(low))
    for i in eachindex(result, low, high)
        value = _karatsuba_bigint(low[i]) + BigInt(K.N1) * _karatsuba_bigint(high[i])
        result[i] = _karatsuba_exact_convert(T, value)
    end
    return result
end

_karatsuba_gpu_storage(A::CuModArray) = A.data
_karatsuba_gpu_storage(A::CUDA.CuArray) = A
_karatsuba_gpu_strides(A::CuModArray) = strides(A.data)
_karatsuba_gpu_strides(A::CUDA.CuArray) = strides(A)

function _karatsuba_gpu_float_precision(::Type{Float16})
    11
end
function _karatsuba_gpu_float_precision(::Type{Float32})
    24
end
function _karatsuba_gpu_float_precision(::Type{Float64})
    53
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:Integer,S<:Integer}
    fits = if T <: Signed
        if S <: Signed
            sizeof(S) <= sizeof(T) || (x >= typemin(T) && x <= typemax(T))
        else
            sizeof(S) < sizeof(T) || x <= typemax(T)
        end
    elseif S <: Signed
        x >= 0 && (sizeof(S) < sizeof(T) || x <= typemax(T))
    else
        sizeof(S) <= sizeof(T) || x <= typemax(T)
    end
    return fits ? (convert(T, x), true) : (zero(T), false)
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:AbstractFloat,S<:Integer}
    converted = T(x)
    isfinite(converted) || return zero(T), false
    lower = S <: Signed ? T(typemin(S)) : zero(T)
    upper = T(typemax(S))
    upper_ok =
        _karatsuba_gpu_float_precision(T) >= 8 * sizeof(S) ? converted <= upper :
        converted < upper
    (converted >= lower && upper_ok) || return zero(T), false
    return converted, S(converted) == x
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:Integer,S<:AbstractFloat}
    (isfinite(x) && isinteger(x)) || return zero(T), false
    lower = T <: Signed ? S(typemin(T)) : zero(S)
    upper = S(typemax(T))
    upper_ok = _karatsuba_gpu_float_precision(S) >= 8 * sizeof(T) ? x <= upper : x < upper
    (x >= lower && upper_ok) || return zero(T), false
    return convert(T, x), true
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:AbstractFloat,S<:AbstractFloat}
    (isfinite(x) && isinteger(x)) || return zero(T), false
    converted = T(x)
    (isfinite(converted) && S(converted) == x) || return zero(T), false
    return converted, true
end

function _karatsuba_gpu_reconstruct(
    ::Type{T},
    low::T,
    high::T,
    radix::Int,
) where {T<:Signed}
    radix > 0 || return zero(T), false
    if high > 0 && (high > typemax(T) ÷ radix)
        return zero(T), false
    elseif high < 0 && (high < typemin(T) ÷ radix)
        return zero(T), false
    end
    product = high * radix
    if product > 0 && low > typemax(T) - product
        return zero(T), false
    elseif product < 0 && low < typemin(T) - product
        return zero(T), false
    end
    return convert(T, low + product), true
end

function _karatsuba_gpu_reconstruct(
    ::Type{T},
    low::T,
    high::T,
    radix::Int,
) where {T<:Unsigned}
    radix > 0 || return zero(T), false
    if high != 0
        radix <= typemax(T) || return zero(T), false
        high <= typemax(T) ÷ radix || return zero(T), false
    end
    product = high == 0 ? zero(T) : high * T(radix)
    low <= typemax(T) - product || return zero(T), false
    return low + product, true
end

function _karatsuba_gpu_reconstruct(
    ::Type{T},
    low::T,
    high::T,
    radix::Int,
) where {T<:AbstractFloat}
    radix > 0 || return zero(T), false
    radix_value = T(radix)
    isfinite(radix_value) && isinteger(radix_value) || return zero(T), false
    product = radix_value * high
    isfinite(product) || return zero(T), false
    product_error = fma(radix_value, high, -product)
    result = product + low
    isfinite(result) && isinteger(result) || return zero(T), false
    sum_virtual = result - product
    sum_error = (product - (result - sum_virtual)) + (low - sum_virtual)
    return result, product_error == -sum_error
end

function _karatsuba_gpu_linear_index(
    k::Int,
    dims::Dims{D},
    source_strides::Dims{D},
) where {D}
    remaining = k - 1
    offset = 1
    @inbounds for d = 1:D
        coordinate = rem(remaining, dims[d])
        remaining = div(remaining, dims[d])
        offset += coordinate * source_strides[d]
    end
    return offset
end

function _karatsuba_reconstruct_gpu_kernel!(
    result,
    valid,
    low,
    high,
    dims,
    low_strides,
    high_strides,
    radix,
)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= length(result)
        low_index = _karatsuba_gpu_linear_index(k, dims, low_strides)
        high_index = _karatsuba_gpu_linear_index(k, dims, high_strides)
        low_value, low_ok = _karatsuba_gpu_component(eltype(result), low[low_index])
        high_value, high_ok = _karatsuba_gpu_component(eltype(result), high[high_index])
        value, reconstruction_ok =
            _karatsuba_gpu_reconstruct(eltype(result), low_value, high_value, radix)
        valid[k] = low_ok && high_ok && reconstruction_ok
        result[k] = valid[k] ? value : zero(eltype(result))
    end
    return
end

function _karatsuba_reconstruct_gpu(::Type{T}, K::KaratsubaArray{S,D}) where {T,S,D}
    supported(T) = isbitstype(T) && T <: Union{Integer,AbstractFloat} && sizeof(T) <= 8
    supported(S) || throw(
        ArgumentError(
            "from_karatsuba GPU components support isbits integer or floating types up to 64 bits",
        ),
    )
    supported(T) || throw(
        ArgumentError(
            "from_karatsuba GPU output supports isbits integer or floating types up to 64 bits",
        ),
    )
    K.N1 > 0 ||
        throw(ArgumentError("from_karatsuba requires a positive low_modulus on GPU"))
    radix = try
        Int(K.N1)
    catch
        throw(
            ArgumentError(
                "from_karatsuba GPU reconstruction requires low_modulus to fit in Int",
            ),
        )
    end
    if T <: AbstractFloat
        exact_radix = try
            BigInt(T(radix)) == BigInt(radix)
        catch
            false
        end
        exact_radix || throw(
            ArgumentError(
                "from_karatsuba GPU reconstruction requires low_modulus to be exactly representable in the requested floating type",
            ),
        )
    end

    result = CUDA.CuArray{T,D}(undef, size(K))
    isempty(result) && return result
    valid = CUDA.CuArray{Bool}(undef, length(result))
    low = _karatsuba_gpu_storage(K.data1)
    high = _karatsuba_gpu_storage(K.data2)
    threads = 256
    blocks = cld(length(result), threads)
    @cuda threads=threads blocks=blocks _karatsuba_reconstruct_gpu_kernel!(
        result,
        valid,
        low,
        high,
        size(K),
        _karatsuba_gpu_strides(K.data1),
        _karatsuba_gpu_strides(K.data2),
        radix,
    )
    all(valid) || throw(
        InexactError(
            :from_karatsuba,
            T,
            "GPU components or reconstructed values are not exactly representable",
        ),
    )
    return result
end

"""
    to_karatsuba(A::AbstractArray)
    to_karatsuba(A::AbstractArray; low_modulus, high_modulus, modulus)
    to_karatsuba(T::Type, A::AbstractArray)
    to_karatsuba(T::Type, A::AbstractArray; low_modulus, high_modulus, modulus)

Convert `A` into exact low and high components using `low = mod(x, N1)` and
`high = fld(x, N1)`. `modulus` is independent metadata; reconstruction always
uses `low + N1*high`. The typed overload converts each component to `T` exactly
and throws `InexactError` if that conversion would lose information.
"""
function _karatsuba_bigint(x)
    if x isa Integer
        return BigInt(x)
    elseif x isa AbstractFloat && isfinite(x) && isinteger(x)
        return BigInt(x)
    end
    throw(InexactError(:to_karatsuba, BigInt, x))
end

function _karatsuba_exact_convert(::Type{T}, value::BigInt) where {T}
    converted = try
        convert(T, value)
    catch err
        err isa InexactError && rethrow()
        throw(InexactError(:convert, T, value))
    end
    exact = if T <: Integer
        BigInt(converted) == value
    elseif T <: AbstractFloat
        isfinite(converted) && BigInt(converted) == value
    else
        converted == value
    end
    exact || throw(InexactError(:convert, T, value))
    return converted
end

function _karatsuba_host_array(A::CuModArray)
    Array(A)
end

function _karatsuba_host_array(A::AbstractArray)
    Array(A)
end

function _karatsuba_component(
    ::Type{T},
    values::AbstractArray,
    source,
    component_modulus,
) where {T}
    if source isa CuModArray
        return CuModArray{T,ndims(values)}(
            values;
            modulus = Int(component_modulus),
            mod = false,
        )
    elseif source isa CuArray
        return CUDA.CuArray(values)
    end
    return values
end

function _split_karatsuba(
    ::Type{T},
    A::AbstractArray;
    low_modulus = nothing,
    high_modulus = nothing,
    modulus = nothing,
) where {T}
    N1, N2, M = _karatsuba_parameters(
        A;
        low_modulus = low_modulus,
        high_modulus = high_modulus,
        modulus = modulus,
        parameter_type = T,
    )
    (N1 > 0 && N2 > 0 && M > 0) || throw(ArgumentError("Karatsuba moduli must be positive"))
    input = _karatsuba_host_array(A)
    low = Base.Array{T}(undef, size(input))
    high = Base.Array{T}(undef, size(input))
    for i in eachindex(input, low, high)
        value = _karatsuba_bigint(input[i])
        hi, lo = fldmod(value, BigInt(N1))
        low[i] = _karatsuba_exact_convert(T, lo)
        high[i] = _karatsuba_exact_convert(T, hi)
    end
    component1 = _karatsuba_component(T, low, A, N1)
    component2 = _karatsuba_component(T, high, A, N2)
    return KaratsubaArray{T,ndims(A)}(
        component1,
        component2;
        low_modulus = N1,
        high_modulus = N2,
        modulus = M,
    )
end

function to_karatsuba(A::AbstractArray; kwargs...)
    return _split_karatsuba(eltype(A), A; kwargs...)
end

function to_karatsuba(::Type{T}, A::AbstractArray; kwargs...) where {T}
    return _split_karatsuba(T, A; kwargs...)
end

function KaratsubaMatrix(::Type{T}, A::AbstractMatrix; kwargs...) where {T}
    return _split_karatsuba(T, A; kwargs...)
end

function KaratsubaVector(::Type{T}, A::AbstractVector; kwargs...) where {T}
    return _split_karatsuba(T, A; kwargs...)
end

function _karatsuba_zeros(
    ::Type{T},
    dims::Dims{D};
    low_modulus = nothing,
    high_modulus = nothing,
    modulus = nothing,
    use_gpu::Bool = true,
) where {T,D}
    explicit = low_modulus !== nothing && high_modulus !== nothing && modulus !== nothing
    shorthand = low_modulus === nothing && high_modulus === nothing && modulus !== nothing
    (explicit || shorthand) || throw(
        ArgumentError("Karatsuba zeros require either modulus or all three split moduli"),
    )
    N1, N2, M = _karatsuba_parameters(
        Base.zeros(T, dims);
        low_modulus = low_modulus,
        high_modulus = high_modulus,
        modulus = modulus,
    )
    (N1 > 0 && N2 > 0 && M > 0) || throw(ArgumentError("Karatsuba moduli must be positive"))
    if use_gpu
        first = zeros(CuModArray{T,D}, dims...; modulus = N1)
        second = zeros(CuModArray{T,D}, dims...; modulus = N2)
    else
        first = Base.zeros(T, dims)
        second = Base.zeros(T, dims)
    end
    return KaratsubaArray{T,D}(
        first,
        second;
        low_modulus = N1,
        high_modulus = N2,
        modulus = M,
    )
end

Base.zeros(::Type{KaratsubaArray{T,2}}, rows::Integer, cols::Integer; kwargs...) where {T} =
    _karatsuba_zeros(T, (Int(rows), Int(cols)); kwargs...)
Base.zeros(::Type{KaratsubaArray{T,1}}, length::Integer; kwargs...) where {T} =
    _karatsuba_zeros(T, (Int(length),); kwargs...)

function initialize_plan!(K::KaratsubaArray{T,D}) where {T,D}
    if K.data1 isa CuModArray
        K.plan = zeros(CuModArray{T,D}, size(K.data1)...; modulus = Int(K.N1))
    elseif K.data1 isa CuArray
        K.plan = CUDA.zeros(T, size(K.data1)...)
    else
        K.plan = Base.zeros(T, size(K.data1))
    end
    return K
end

import Base: +, -, *

function _karatsuba_zero_component(source, dims::Dims, component_modulus)
    T = eltype(source)
    D = length(dims)
    if source isa CuModArray
        return zeros(CuModArray{T,D}, dims...; modulus = Int(component_modulus))
    elseif source isa CuArray
        return CUDA.zeros(T, dims...)
    end
    return Base.zeros(T, dims)
end

function _karatsuba_output_like(source::KaratsubaArray{T,D}, dims::Dims{D}) where {T,D}
    data1 = _karatsuba_zero_component(source.data1, dims, source.N1)
    data2 = _karatsuba_zero_component(source.data2, dims, source.N2)
    return KaratsubaArray(
        data1,
        data2;
        low_modulus = source.N1,
        high_modulus = source.N2,
        modulus = source.M,
    )
end

function +(A::KaratsubaArray, B::KaratsubaArray)
    K = _karatsuba_output_like(A, size(A.data1))
    add!(K, A, B)
    K
end

function -(A::KaratsubaArray, B::KaratsubaArray)
    K = _karatsuba_output_like(A, size(A.data1))
    sub!(K, A, B)
end

function *(a::Number, A::KaratsubaArray)
    K = _karatsuba_output_like(A, size(A.data1))
    scalar_multiply!(K, A, a)
    K
end

function *(A::KaratsubaArray, a::Number)
    a * A
end

function *(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        dims = (size(A.data1, 1), size(B.data1, 2))
        K = _karatsuba_output_like(A, dims)
        plan = _karatsuba_output_like(A, dims)
        #plan = KMatMulPlan(zeros(eltype(A.data1),size(A.data1)...,A.M),zeros(eltype(B.data1),size(B.data1)...,A.M),A.N1,A.N2)
    elseif occursin("Cu", string(typeof(A.data1)))
        dims = (size(A.data1, 1), size(B.data1, 2))
        K = _karatsuba_output_like(A, dims)
    else
        dims = (size(A.data1, 1), size(B.data1, 2))
        K = _karatsuba_output_like(A, dims)
        plan = KMatMulPlan{Matrix{eltype(A.data1)}}(
            zeros(eltype(A.data1), size(A.data1)...),
            zeros(eltype(B.data1), size(B.data1)...),
            A.N1,
            A.N2,
        )
    end
    LinearAlgebra.mul!(K, A, B, plan)
    K
end

# """

# Works even for multidimensional arrays because CuArrays support linear indexing

# """
# function karatsuba_add_kernel!(Kplan,Kdata1,Kdata2,Adata1,Adata2,Bdata1,Bdata2,N1,N2)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

#     Kplan[i] = (Adata1[i] + Bdata1[i]) % (2*N1)
#     Kplan[i] = div(Kplan[i],N1)
#     Kdata2[i] = (Kplan[i] + Adata2[i]) % (2*N2)
#     Kdata2[i] = (Kdata2[i] + Bdata2[i]) % (2*N2)

#     Kplan[i] = (Kplan[i] * N1) % (N1^2)
#     Kplan[i] = (Bdata1[i] - Kplan[i]) % (N1^2)
#     Kdata1[i] = (Adata1[i] + Kplan[i]) % (N1^2)

#     Kdata1[i] %= N1
#     Kdata2[i] %= N2

#     return
# end

function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != K.M)
        error("Matrices must have the same modulus m")
    end
    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
        error("Matrix dimensions must match")
    end

    #TODO: remove plans
    if (K.plan == nothing)
        error("Must initialize plan first before using fallback implementation")
    end

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_add_kernel!(
        K.data1.data,
        K.data2.data,
        A.data1.data,
        A.data2.data,
        B.data1.data,
        B.data2.data,
        A.N1,
        A.N2,
    )


    K
end

#"""
#This is a fallback implementation, and shouldn't be used
#unless the dimension is greater than or equal to 3.

#This is not safe if K == B, but it is safe if K == A.
#"""
#function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
#    if (A.M != B.M) || (A.M != K.M)
#        error("Matrices must have the same modulus m")
#    end
#    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
#        error("Matrix dimensions must match")
#    end
#    if (A.plan == nothing) || (K.plan == nothing)
#        error("Must initialize plans first before using fallback implementation")
#    end
#    #=
#    K.data2 .= mod.(trunc.((A.data1+B.data1)./A.M),A.N2)
#    K.data1 .= mod.(A.data1 + B.data1 - A.M*K.data2,A.N1)
#    K.data2 .= mod.(K.data2 + A.data2 + B.data2,A.N2)
#    =#

#    add!(K.plan,A.data1,B.data1,2*A.N1)
#    divide_elements!(K.plan,K.plan,A.N1)
#    add!(K.data2,K.plan,A.data2,2*A.N2)
#    add!(K.data2,K.data2,B.data2,2*A.N2)
#    LinearAlgebra.mul!(K.plan,K.plan,A.N1,A.N1^2)
#    sub!(K.plan,B.data1,K.plan,B.N1^2)
#    add!(K.data1,A.data1,K.plan,A.N1^2)
#    #=
#    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
#    trunc_elements!(K.data2)
#    =#
#    mod_elements!(K.data1; modulus_override=A.N1)
#    mod_elements!(K.data2; modulus_override=A.N2)
#    #=
#    add!(K.data1,A.data1,B.data1,2*A.M)
#    add!(K.data2,A.data2,B.data2)
#    add!(K.data2,K.data2,divides(K.data1,K.M),K.N2)
#    mod_elements!(K.data1; modulus_override=K.N1)
#    =#

#    K
#end

function sub!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != K.M)
        error("Matrices must have the same modulus m")
    end
    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
        error("Matrix dimensions must match")
    end

    #=
    C.data2 .= mod.(trunc.((A.data1-B.data1)./A.M),A.N2)
    C.data1 .= mod.(A.data1 - B.data1 - A.M*C.data2,A.N1)
    C.data2 .= mod.(C.data2 + A.data2 - B.data2 - (A.data1.<B.data1),A.N2)

    sub!(K.data2,A.data1,B.data1)
    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
    trunc_elements!(K.data2)
    mod_elements!(K.data2; modulus_override=A.N2)
    LinearAlgebra.mul!(K.data1,K.data1,A.M,A.M^2)
    sub!(K.data1,B.data1,K.data1)
    sub!(K.data1,A.data1,K.data1)
    mod_elements!(K.data1; modulus_override=A.N1)
    add!(K.data2,K.data2,A.data2)
    sub!(K.data2,K.data2,B.data2)
    sub!(K.data2,K.data2,A.data1.<B.data1)
    mod_elements!(K.data2; modulus_override=A.N2)
    =#
    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_sub_kernel!(
        K.data1.data,
        K.data2.data,
        A.data1.data,
        A.data2.data,
        B.data1.data,
        B.data2.data,
        A.N1,
        A.N2,
        A.M,
    )

    # negate!(K,B)
    # add!(K,K,A)
    K
end

function scalar_multiply!(B::KaratsubaArray, A::KaratsubaArray, s::Number)
    if A.M != B.M
        error("Matrices must have the same modulus m")
    end
    if size(A.data1) != size(B.data1)
        error("Matrix dimensions must match")
    end

    # LinearAlgebra.mul!(B.data2,A.data1,s,A.N1^2)
    # divide_elements!(B.data2,B.data2,A.N1)
    # LinearAlgebra.mul!(B.data1,A.data2,s,A.N2)
    # add!(B.data2,B.data2,B.data1)
    # LinearAlgebra.mul!(B.data1,A.data1,s,A.N1)

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_scalar_multiply_kernel!(
        B.data1.data,
        B.data2.data,
        A.data1.data,
        A.data2.data,
        s,
        A.N1,
        A.N2,
    )

    #=
    B.data2 .= mod.(trunc.((s*A.data1)/A.M),A.N2)
    B.data1 .= mod.(s*A.data1 - B.M*B.data2,A.N1)
    B.data2 .= mod.(B.data2 + s*A.data2,A.N2)
    =#
    B
end



# function negate!(K::KaratsubaArray, A::KaratsubaArray)

#     tw = TILE_WIDTH

#     totalsize = length(A.data1.data)

#     threads = tw
#     blocks = totalsize ÷ tw

# @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(K.data1.data,
#                                                              K.data2.data,
#                                                              A.data1.data,
#                                                              A.data2.data,
#                                                              A.N1,
#                                                              A.N2,
#                                                             A.M)


#     K
# end

function negate!(K::KaratsubaArray, A::KaratsubaArray)
    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(
        K.data1.data,
        K.data2.data,
        A.data1.data,
        A.data2.data,
        A.N1,
        A.N2,
        A.M,
    )



    #=
    K.data2 .= trunc.((A.M .- A.data1)/A.M)
    K.data1 .= mod.(A.M - A.data1 - A.M*K.data2,A.N1)
    K.data2 .= mod.(mod.(K.data2 + A.M - A.data2 - 1,A.M),A.N2)
    =#
    # mul!(K.data2,K.data2,0)
    # scalar_add!(K.data2,K.data2,A.N1,2*A.N1)
    # sub!(K.data2,K.data2,A.data1,2*A.N1)
    #=
    LinearAlgebra.mul!(K.data2,K.data2,1/A.N1)
    trunc_elements!(K.data2)
     =#
    # divide_elements!(K.data2,K.data2,A.N1)
    # LinearAlgebra.mul!(K.data1,K.data2,A.N1,A.M^2)
    # add!(K.data1,K.data1,A.data1,A.M^2)
    # negate!(K.data1,K.data1,A.N1)
    # mod_elements!(K.data1; modulus_override=A.N1)
    # scalar_add!(K.data2,K.data2,A.N2,A.M^2)
    # scalar_sub!(K.data2,K.data2,1,A.M^2)
    # sub!(K.data2,K.data2,A.data2,A.M)
    # mod_elements!(K.data2; modulus_override=A.N2)
    K
end

function divide_elements!(B::CuModArray, A::CuModArray, N::Integer)
    @. B.data = div(A.data, N)
    return A
end

function Base.copy(A::KaratsubaArray{T,D}) where {T,D}
    B = KaratsubaArray{T,D}(
        copy(A.data1),
        copy(A.data2);
        low_modulus = copy(A.N1),
        high_modulus = copy(A.N2),
        modulus = A.N1 * A.N2,
    )
    if !(A.plan==nothing)
        initialize_plan!(B)
    end
    B
end
