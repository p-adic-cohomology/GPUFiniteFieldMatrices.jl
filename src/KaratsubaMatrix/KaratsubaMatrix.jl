"""
    KaratsubaArray{T,D,S,P}

An array represented by low and high components of storage type `S`. The
component arrays must have the same concrete storage type. `P` is independent
of `S`: the factory-created plan type is `CuModArray{T,D}` for every component
backend, matching the original padded GPU workspace behavior. Use `low_part`,
`high_part`, `low_modulus`, `high_modulus`, and `modulus` to inspect a value;
the fields are internal implementation details.
"""
mutable struct KaratsubaArray{T,D,S<:AbstractArray{T,D},P<:AbstractArray{T,D}} <:
               AbstractArray{T,D}
    data1::S
    data2::S
    plan::Union{Nothing,P}
    N1::Int
    N2::Int
    M::Int

    function KaratsubaArray{T,D,S,P}(
        A::S,
        B::S,
        ;
        low_modulus::Union{Nothing,Integer} = nothing,
        high_modulus::Union{Nothing,Integer} = nothing,
        modulus::Union{Nothing,Integer} = nothing,
    ) where {T,D,S<:AbstractArray{T,D},P<:AbstractArray{T,D}}
        isconcretetype(S) ||
            throw(ArgumentError("Karatsuba components require a concrete storage type"))
        isconcretetype(P) ||
            throw(ArgumentError("Karatsuba plans require a concrete storage type"))
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

        return new{T,D,S,P}(A, B, nothing, N1, N2, M)
    end
end

const KaratsubaMatrix{T} = KaratsubaArray{T,2}
const KaratsubaVector{T} = KaratsubaArray{T,1}

function KaratsubaArray{T,D}(
    A::AbstractArray{T,D},
    B::AbstractArray{T,D};
    kwargs...,
) where {T,D}
    typeof(A) === typeof(B) ||
        throw(ArgumentError("Karatsuba components must have the same storage type"))
    return KaratsubaArray{T,D,typeof(A)}(A, B; kwargs...)
end

_karatsuba_plan_type(::Type{S}) where {T,D,S<:AbstractArray{T,D}} = CuModArray{T,D}

function KaratsubaArray{T,D,S}(A::S, B::S; kwargs...) where {T,D,S<:AbstractArray{T,D}}
    P = _karatsuba_plan_type(S)
    return KaratsubaArray{T,D,S,P}(A, B; kwargs...)
end

"""Return the low component array stored by `K`, preserving object identity."""
low_part(K::KaratsubaArray) = getfield(K, :data1)

"""Return the high component array stored by `K`, preserving object identity."""
high_part(K::KaratsubaArray) = getfield(K, :data2)

"""Return the low component modulus stored by `K`."""
low_modulus(K::KaratsubaArray) = getfield(K, :N1)

"""Return the high component modulus stored by `K`."""
high_modulus(K::KaratsubaArray) = getfield(K, :N2)

"""Return the represented modulus metadata stored by `K`."""
modulus(K::KaratsubaArray) = getfield(K, :M)

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

const _KARATSUBA_FLOAT64_EXACT_BOUND = BigInt(1) << 53

function _karatsuba_checked_working_modulus(value::BigInt, label::AbstractString)
    (0 < value <= typemax(Int)) ||
        throw(ArgumentError("$label does not fit a positive Int modulus"))
    value <= _KARATSUBA_FLOAT64_EXACT_BOUND ||
        throw(ArgumentError("$label exceeds the exact-integer Float64 range"))
    return Int(value)
end

function _karatsuba_checked_sum_bound(a::Int, b::Int, label::AbstractString)
    value = BigInt(a) + BigInt(b)
    value <= typemax(Int) || throw(ArgumentError("$label overflows Int"))
    # The operands are strictly below a and b, so a sum bound of 2^53 still
    # promises actual Float64 integers no larger than 2^53 - 1.
    value <= _KARATSUBA_FLOAT64_EXACT_BOUND ||
        throw(ArgumentError("$label exceeds the exact-integer Float64 range"))
    return Int(value)
end

function _karatsuba_safe_product_parameters(input_bound::Int, reduction_modulus::Int)
    effective_bound = max(input_bound, 2)
    max_ops = find_max_stripe_ops(Float64, effective_bound; modulus = reduction_modulus)
    max_ops > 0 || throw(
        ArgumentError(
            "no exact Float64 stripe contribution fits input bound $effective_bound and reduction modulus $reduction_modulus",
        ),
    )
    return (input_bound = effective_bound, modulus = reduction_modulus, max_ops = max_ops)
end

function _karatsuba_matmul_product_plan(
    input_low_a::Int,
    input_high_a::Int,
    input_low_b::Int,
    input_high_b::Int,
    output_low::Int,
    output_high::Int,
)
    all(
        >(0),
        (input_low_a, input_high_a, input_low_b, input_high_b, output_low, output_high),
    ) || throw(ArgumentError("Karatsuba split moduli must be positive"))
    input_low_a == input_low_b || throw(
        ArgumentError(
            "three-product Karatsuba matrix multiplication requires a common low modulus for A and B",
        ),
    )

    input_radix = input_low_a
    output_capacity = _karatsuba_checked_working_modulus(
        BigInt(output_low) * BigInt(output_high),
        "output component capacity",
    )
    sum_bound_a =
        _karatsuba_checked_sum_bound(input_radix, input_high_a, "left sum input bound")
    sum_bound_b =
        _karatsuba_checked_sum_bound(input_radix, input_high_b, "right sum input bound")
    middle_bound = max(sum_bound_a, sum_bound_b)
    high_bound = max(input_high_a, input_high_b)
    radix_mod_capacity = Int(mod(BigInt(input_radix), BigInt(output_capacity)))
    radix_squared_mod_capacity = Int(mod(BigInt(input_radix)^2, BigInt(output_capacity)))
    low = _karatsuba_safe_product_parameters(input_radix, output_capacity)
    middle = _karatsuba_safe_product_parameters(middle_bound, output_capacity)
    high = _karatsuba_safe_product_parameters(high_bound, output_capacity)

    return (
        input_radix = input_radix,
        output_radix = output_low,
        output_high = output_high,
        output_capacity = output_capacity,
        radix_mod_capacity = radix_mod_capacity,
        radix_squared_mod_capacity = radix_squared_mod_capacity,
        low = low,
        middle = middle,
        high = high,
    )
end

function _karatsuba_validate_gemm_storage(
    A::KaratsubaArray,
    B::KaratsubaArray,
    C::KaratsubaArray,
)
    for (name, K) in (("A", A), ("B", B), ("C", C))
        low = low_part(K)
        high = high_part(K)
        plan = K.plan
        for (plane_name, plane) in (("low", low), ("high", high), ("plan", plan))
            plane isa CuModMatrix{Float64} || throw(
                ArgumentError(
                    "Karatsuba matrix multiplication requires CuModMatrix{Float64} $name.$plane_name storage",
                ),
            )
            size(parent(plane), 1) >= size(plane, 1) &&
            size(parent(plane), 2) >= size(plane, 2) || throw(
                DimensionMismatch(
                    "$name.$plane_name storage is smaller than its logical size",
                ),
            )
        end
        size(high) == size(low) || throw(
            DimensionMismatch(
                "$name low and high components must have matching logical sizes",
            ),
        )
        size(plan) == size(low) ||
            throw(DimensionMismatch("$name plan must match its component logical size"))
    end

    inputs = (low_part(A), high_part(A), low_part(B), high_part(B))
    outputs = (low_part(C), high_part(C), C.plan)
    scratch = (A.plan, B.plan)
    for output in outputs, input in inputs
        _stripe_mightalias(parent(output), parent(input)) && throw(
            ArgumentError(
                "Karatsuba matrix output storage must not alias input components",
            ),
        )
    end
    for i = 1:length(outputs), j = (i+1):length(outputs)
        _stripe_mightalias(parent(outputs[i]), parent(outputs[j])) && throw(
            ArgumentError("Karatsuba matrix output components must not alias each other"),
        )
    end
    for plan in scratch, input in inputs
        _stripe_mightalias(parent(plan), parent(input)) &&
            throw(ArgumentError("Karatsuba plan scratch must not alias input components"))
    end
    for plan in scratch, output in outputs
        _stripe_mightalias(parent(plan), parent(output)) &&
            throw(ArgumentError("Karatsuba plan scratch must not alias output components"))
    end
    _stripe_mightalias(parent(A.plan), parent(B.plan)) &&
        A !== B &&
        throw(ArgumentError("Karatsuba input plan scratch must not alias each other"))
    return nothing
end

function _karatsuba_stripe_product!(
    C,
    A,
    B,
    require_matching_input_moduli,
    product;
    kwargs...,
)
    return _stripe_mul_matrix!(
        C,
        A,
        B,
        require_matching_input_moduli;
        input_bound = product.input_bound,
        modulus_override = product.modulus,
        max_ops = product.max_ops,
        clear_output_padding = false,
        kwargs...,
    )
end

"""
    LinearAlgebra.mul!(C, A, B)

Multiply the Karatsuba matrices A and B, storing the product in C. The
three matrices must share the same representation modulus M, and A's column
count must match B's row count. C must have dimensions (size(A, 1), size(B, 2));
the stripe preflight checks these destination dimensions before any CUDA work.
Call initialize_plan! on all
three matrices before multiplying. A and B must share input low radix R; C
may use an independent low radix R_C. The result components represent the
product modulo Q = R_C*high_modulus(C), and checked modular recombination
supports independent R_C without reducing by M. Input component entries must
be canonical nonnegative representatives below their respective low and high
moduli; those ranges are the exclusive stripe input bounds. These ranges are
caller promises and are not scanned at runtime. CuModArray wrappers made with
mod=false retain raw values, and to_karatsuba preserves an exact high quotient
without reducing it by high_modulus, so callers must supply a high_modulus
that bounds those values. This matrix path currently supports Float64 storage
only.
"""
function LinearAlgebra.mul!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(C))
        error("Matrices must have the same modulus m")
    end
    if size(low_part(A))[2] != size(low_part(B))[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (A.plan == nothing) || (B.plan == nothing) || (C.plan == nothing)
        error("Must initialize plans first")
    end
    if eltype(parent(low_part(A))) != Float64
        error("Not implemented for non-Float64 entries")
    end
    _karatsuba_validate_gemm_storage(A, B, C)
    products = _karatsuba_matmul_product_plan(
        low_modulus(A),
        high_modulus(A),
        low_modulus(B),
        high_modulus(B),
        low_modulus(C),
        high_modulus(C),
    )

    high_requires_matching_input_moduli = modulus(high_part(A)) == modulus(high_part(B))
    # Validate all stripe calls before the sum kernel writes either input plan.
    # The calls below repeat this preflight at their own execution boundary.
    _karatsuba_stripe_product!(
        low_part(C),
        low_part(A),
        low_part(B),
        true,
        products.low;
        preflight_only = true,
    )
    _karatsuba_stripe_product!(
        high_part(C),
        A.plan,
        B.plan,
        true,
        products.middle;
        preflight_only = true,
    )
    _karatsuba_stripe_product!(
        C.plan,
        high_part(A),
        high_part(B),
        high_requires_matching_input_moduli,
        products.high;
        preflight_only = true,
    )

    rows_a, cols_a = size(low_part(A))
    rows_b, cols_b = size(low_part(B))
    logical_count = max(length(low_part(A)), length(low_part(B)))
    threads = TILE_WIDTH
    blocks = max(1, cld(logical_count, threads))
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        parent(A.plan),
        parent(low_part(A)),
        parent(high_part(A)),
        rows_a,
        cols_a,
        stride(parent(A.plan), 1),
        stride(parent(A.plan), 2),
        stride(parent(low_part(A)), 1),
        stride(parent(low_part(A)), 2),
        stride(parent(high_part(A)), 1),
        stride(parent(high_part(A)), 2),
        parent(B.plan),
        parent(low_part(B)),
        parent(high_part(B)),
        rows_b,
        cols_b,
        stride(parent(B.plan), 1),
        stride(parent(B.plan), 2),
        stride(parent(low_part(B)), 1),
        stride(parent(low_part(B)), 2),
        stride(parent(high_part(B)), 1),
        stride(parent(high_part(B)), 2),
    )

    _karatsuba_stripe_product!(low_part(C), low_part(A), low_part(B), true, products.low)
    _karatsuba_stripe_product!(high_part(C), A.plan, B.plan, true, products.middle)
    _karatsuba_stripe_product!(
        C.plan,
        high_part(A),
        high_part(B),
        high_requires_matching_input_moduli,
        products.high,
    )

    output_rows, output_cols = size(low_part(C))
    output_count = length(low_part(C))
    output_blocks = max(1, cld(output_count, threads))
    output_capacity = Float64(products.output_capacity)
    radix_mod_capacity = Float64(products.radix_mod_capacity)
    radix_squared_mod_capacity = Float64(products.radix_squared_mod_capacity)
    output_radix = Float64(products.output_radix)
    @cuda threads=threads blocks=output_blocks karatsuba_matmul_kernel_2!(
        parent(low_part(C)),
        parent(high_part(C)),
        parent(C.plan),
        output_rows,
        output_cols,
        stride(parent(low_part(C)), 1),
        stride(parent(low_part(C)), 2),
        stride(parent(high_part(C)), 1),
        stride(parent(high_part(C)), 2),
        stride(parent(C.plan), 1),
        stride(parent(C.plan), 2),
        radix_mod_capacity,
        radix_squared_mod_capacity,
        output_radix,
        output_capacity,
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
function karatsuba_gemv!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(C))
        error("Matrices must have the same modulus m")
    end
    if size(low_part(A))[2] != size(low_part(B))[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (A.plan == nothing) || (B.plan == nothing) || (C.plan == nothing)
        error("Must initialize plans first")
    end
    if eltype(parent(low_part(A))) != Float64
        error("Not implemented for non-Float64 entries")
    end
    # TOD0: enable this
    # maxN = max(low_modulus(A),high_modulus(A))
    # if find_max_stripe_ops(eltype(parent(low_part(A))), maxN) < size(A,1)
    #     error("Matrix is too big for a single matmul, this is currently not implemented")
    # end

    tw = TILE_WIDTH

    totalsize = length(parent(low_part(A)))

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        parent(A.plan),
        parent(low_part(A)),
        parent(high_part(A)),
        parent(B.plan),
        parent(low_part(B)),
        parent(high_part(B)),
        low_modulus(A),
    )
    # add!(A.plan,low_part(A),high_part(A); modulus_override=2*low_modulus(A))
    # add!(B.plan,low_part(B),high_part(B); modulus_override=2*low_modulus(B))

    (zero_ptr, one_ptr) = cublas_scalars_f64()


    targets = [parent(low_part(C)), parent(high_part(C)), parent(B.plan)]
    mats = [parent(low_part(A)), parent(A.plan), parent(high_part(A))]
    vecs = [parent(low_part(B)), parent(B.plan), parent(high_part(B))]
    _CUDA_BLAS.gemv_batched!('N', one_ptr, mats, vecs, zero_ptr, targets)

    # CUDA.CUBLAS.gemv!('N',one_ptr,parent(low_part(A)),parent(low_part(B)),zero_ptr,parent(low_part(C)))
    # CUDA.CUBLAS.gemv!('N',one_ptr,parent(A.plan),parent(B.plan),zero_ptr,parent(high_part(C)))
    # CUDA.CUBLAS.gemv!('N',one_ptr,parent(high_part(A)),parent(high_part(B)),zero_ptr,parent(B.plan))

    tw = TILE_WIDTH

    totalsize = length(parent(low_part(C)))

    threads = tw
    blocks = totalsize ÷ tw
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_2!(
        parent(low_part(C)),
        parent(high_part(C)),
        parent(B.plan),
        low_modulus(C),
        high_modulus(C),
    )


    C
end

Base.size(A::KaratsubaArray) = size(low_part(A))

Base.getindex(A::KaratsubaArray, i::Int, j::Int) =
    low_part(A)[i, j] + low_modulus(A)*high_part(A)[i, j]
Base.getindex(A::KaratsubaArray, i::Int) = low_part(A)[i] + low_modulus(A)*high_part(A)[i]

function Base.setindex!(A::KaratsubaArray, v, i::Int, j::Int)
    low_part(A)[i, j] = rem(v, low_modulus(A))
    high_part(A)[i, j] = div(v, low_modulus(A))
end

function Base.setindex!(A::KaratsubaArray, v, i::Int)
    low_part(A)[i] = rem(v, low_modulus(A))
    high_part(A)[i] = div(v, low_modulus(A))
end

#Converts KMat to Mat
function Base.Array(K::KaratsubaArray)
    return _karatsuba_reconstruct_host(BigInt, K)
end

function Base.copy!(B::KaratsubaArray, A::KaratsubaArray)
    copy!(low_part(B), low_part(A))
    copy!(high_part(B), high_part(A))
    if A.plan != nothing && B.plan == nothing
        initialize_plan!(B)
    end
    return B
end

function zero!(A::KaratsubaArray)
    zero!(low_part(A))
    zero!(high_part(A))
end

"""
    from_karatsuba(T::Type, K::KaratsubaArray)

Reconstruct the full array represented by `K` as
`low_part(K) + low_modulus(K)*high_part(K)`.
The returned array has element type `T` and follows the component backend. Each
integer conversion is checked before returning, so an inexact or overflowing
conversion throws `InexactError`.
"""
function from_karatsuba(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:AbstractArray{S,D},Plan<:AbstractArray{S,D}}
    if Storage <: Union{CuModArray,CuArray}
        return _karatsuba_reconstruct_gpu(T, K)::CUDA.CuArray{T,D}
    end
    return _karatsuba_reconstruct_host(T, K)::Array{T,D}
end

function _karatsuba_reconstruct_host(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:AbstractArray{S,D},Plan<:AbstractArray{S,D}}
    low = _karatsuba_host_array(low_part(K))::Array{S,D}
    high = _karatsuba_host_array(high_part(K))::Array{S,D}
    result = Base.Array{T,D}(undef, size(low))
    for i in eachindex(result, low, high)
        value =
            _karatsuba_bigint(low[i]) + BigInt(low_modulus(K)) * _karatsuba_bigint(high[i])
        result[i] = _karatsuba_exact_convert(T, value)
    end
    return result
end

_karatsuba_gpu_storage(A::CuModArray) = parent(A)
_karatsuba_gpu_storage(A::CUDA.CuArray) = A
_karatsuba_gpu_strides(A::CuModArray) = strides(parent(A))
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

function _karatsuba_reconstruct_gpu(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:AbstractArray{S,D},Plan<:AbstractArray{S,D}}
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
    low_modulus(K) > 0 ||
        throw(ArgumentError("from_karatsuba requires a positive low_modulus on GPU"))
    radix = try
        Int(low_modulus(K))
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
    low = _karatsuba_gpu_storage(low_part(K))
    high = _karatsuba_gpu_storage(high_part(K))
    threads = 256
    blocks = cld(length(result), threads)
    @cuda threads=threads blocks=blocks _karatsuba_reconstruct_gpu_kernel!(
        result,
        valid,
        low,
        high,
        size(K),
        _karatsuba_gpu_strides(low_part(K)),
        _karatsuba_gpu_strides(high_part(K)),
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
    ::Type{S},
    ::Type{P},
    dims::Dims{D};
    low_modulus = nothing,
    high_modulus = nothing,
    modulus = nothing,
) where {T,D,S<:AbstractArray{T,D},P<:AbstractArray{T,D}}
    explicit = low_modulus !== nothing && high_modulus !== nothing && modulus !== nothing
    shorthand = low_modulus === nothing && high_modulus === nothing && modulus !== nothing
    (explicit || shorthand) || throw(
        ArgumentError("Karatsuba zeros require either modulus or all three split moduli"),
    )
    isconcretetype(S) || throw(
        ArgumentError("Karatsuba zero factories require a concrete component storage type"),
    )
    isconcretetype(P) || throw(
        ArgumentError("Karatsuba zero factories require a concrete plan storage type"),
    )
    expected_plan_type = _karatsuba_plan_type(S)
    P === expected_plan_type || throw(
        ArgumentError(
            "Requested plan storage type $P does not match the plan type $expected_plan_type for component storage $S",
        ),
    )

    N1, N2, M = _karatsuba_parameters(
        Base.zeros(T, dims);
        low_modulus = low_modulus,
        high_modulus = high_modulus,
        modulus = modulus,
    )
    (N1 > 0 && N2 > 0 && M > 0) || throw(ArgumentError("Karatsuba moduli must be positive"))

    first = _karatsuba_zero_component(S, dims, N1)
    second = _karatsuba_zero_component(S, dims, N2)
    typeof(first) === S || throw(
        ArgumentError(
            "Storage factory returned $(typeof(first)) for requested component type $S",
        ),
    )
    typeof(second) === S || throw(
        ArgumentError(
            "Storage factory returned $(typeof(second)) for requested component type $S",
        ),
    )

    return KaratsubaArray{T,D,S,P}(
        first,
        second;
        low_modulus = N1,
        high_modulus = N2,
        modulus = M,
    )
end

function _karatsuba_zeros(
    ::Type{T},
    ::Type{S},
    dims::Dims{D};
    kwargs...,
) where {T,D,S<:AbstractArray{T,D}}
    P = _karatsuba_plan_type(S)
    return _karatsuba_zeros(T, S, P, dims; kwargs...)
end

function _karatsuba_zero_component(::Type{Array{T,D}}, dims::Dims{D}, ::Int) where {T,D}
    return Base.zeros(T, dims)
end

function _karatsuba_zero_component(
    ::Type{CuModArray{T,D}},
    dims::Dims{D},
    component_modulus::Int,
) where {T,D}
    return zeros(CuModArray{T,D}, dims...; modulus = component_modulus)
end

function _karatsuba_zero_component(
    ::Type{S},
    dims::Dims{D},
    ::Int,
) where {T,D,S<:CuArray{T,D}}
    isconcretetype(S) || throw(
        ArgumentError("CuArray zero factories require a concrete component storage type"),
    )
    component = CUDA.zeros(T, dims...)
    typeof(component) === S || throw(
        ArgumentError(
            "CUDA.zeros returned $(typeof(component)) for requested component type $S",
        ),
    )
    return component
end

function _karatsuba_zero_component(
    ::Type{S},
    ::Dims{D},
    ::Int,
) where {T,D,S<:AbstractArray{T,D}}
    throw(
        ArgumentError("Karatsuba zero factories do not support component storage type $S"),
    )
end

function Base.zeros(
    ::Type{KaratsubaArray{T,2}},
    rows::Integer,
    cols::Integer;
    kwargs...,
) where {T}
    return _karatsuba_zeros(T, CuModArray{T,2}, (Int(rows), Int(cols)); kwargs...)
end

function Base.zeros(::Type{KaratsubaArray{T,1}}, length::Integer; kwargs...) where {T}
    return _karatsuba_zeros(T, CuModArray{T,1}, (Int(length),); kwargs...)
end

function Base.zeros(
    ::Type{KaratsubaArray{T,2,S}},
    rows::Integer,
    cols::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,2}}
    return _karatsuba_zeros(T, S, (Int(rows), Int(cols)); kwargs...)
end

function Base.zeros(
    ::Type{KaratsubaArray{T,1,S}},
    length::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,1}}
    return _karatsuba_zeros(T, S, (Int(length),); kwargs...)
end

function Base.zeros(
    ::Type{KaratsubaArray{T,2,S,P}},
    rows::Integer,
    cols::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,2},P<:AbstractArray{T,2}}
    return _karatsuba_zeros(T, S, P, (Int(rows), Int(cols)); kwargs...)
end

function Base.zeros(
    ::Type{KaratsubaArray{T,1,S,P}},
    length::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,1},P<:AbstractArray{T,1}}
    return _karatsuba_zeros(T, S, P, (Int(length),); kwargs...)
end

function initialize_plan!(K::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    expected_plan_type = _karatsuba_plan_type(S)
    P === expected_plan_type || throw(
        ArgumentError(
            "initialize_plan! requires plan storage type $expected_plan_type, got $P",
        ),
    )
    K.plan = zeros(expected_plan_type, size(low_part(K))...; modulus = Int(low_modulus(K)))
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

function _karatsuba_output_like(
    source::KaratsubaArray{T,D,S,P},
    dims::Dims{D},
) where {T,D,S,P}
    data1 = _karatsuba_zero_component(low_part(source), dims, low_modulus(source))
    data2 = _karatsuba_zero_component(high_part(source), dims, high_modulus(source))
    return KaratsubaArray(
        data1,
        data2;
        low_modulus = low_modulus(source),
        high_modulus = high_modulus(source),
        modulus = modulus(source),
    )
end

function +(A::KaratsubaArray, B::KaratsubaArray)
    K = _karatsuba_output_like(A, size(low_part(A)))
    add!(K, A, B)
    K
end

function -(A::KaratsubaArray, B::KaratsubaArray)
    K = _karatsuba_output_like(A, size(low_part(A)))
    sub!(K, A, B)
end

function *(a::Number, A::KaratsubaArray)
    K = _karatsuba_output_like(A, size(low_part(A)))
    scalar_multiply!(K, A, a)
    K
end

function *(A::KaratsubaArray, a::Number)
    a * A
end

function _karatsuba_product_workspace_like(K::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    return KaratsubaArray{T,D,S,P}(
        low_part(K),
        high_part(K);
        low_modulus = low_modulus(K),
        high_modulus = high_modulus(K),
        modulus = modulus(K),
    )
end

function _karatsuba_validate_allocating_product_storage(K::KaratsubaArray{Float64,2})
    low_part(K) isa CuModMatrix{Float64} && high_part(K) isa CuModMatrix{Float64} || throw(
        ArgumentError(
            "allocating Karatsuba matrix multiplication currently supports GPU-backed CuModMatrix{Float64} components",
        ),
    )
    return nothing
end

"""
    A * B

Allocate the product of GPU-backed `Float64` Karatsuba matrices. `A` and `B`
must satisfy the same modulus, dimension, common input low-radix, and
canonical-component requirements as `LinearAlgebra.mul!(C, A, B)`. The result uses `A`'s component split and
representation metadata; its output capacity is `low_modulus(A) *
high_modulus(A)`, which is distinct from the metadata modulus `modulus(A)`.

The operation delegates to the same three-argument `mul!` implementation using
private plan scratch. The input component storage is shared read-only, so the
caller plans may be uninitialized and any existing caller plan is left
unchanged.
"""
function *(A::KaratsubaArray{Float64,2}, B::KaratsubaArray{Float64,2})
    _karatsuba_validate_allocating_product_storage(A)
    _karatsuba_validate_allocating_product_storage(B)

    A_work = _karatsuba_product_workspace_like(A)
    B_work = _karatsuba_product_workspace_like(B)
    C = _karatsuba_output_like(A, (size(A, 1), size(B, 2)))
    foreach(initialize_plan!, (A_work, B_work, C))
    return LinearAlgebra.mul!(C, A_work, B_work)
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
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(K))
        error("Matrices must have the same modulus m")
    end
    if (size(low_part(A)) != size(low_part(B))) ||
       (size(low_part(A)) != (size(low_part(K))))
        error("Matrix dimensions must match")
    end

    #TODO: remove plans
    if (K.plan == nothing)
        error("Must initialize plan first before using fallback implementation")
    end

    tw = TILE_WIDTH

    totalsize = length(parent(low_part(A)))

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_add_kernel!(
        parent(low_part(K)),
        parent(high_part(K)),
        parent(low_part(A)),
        parent(high_part(A)),
        parent(low_part(B)),
        parent(high_part(B)),
        low_modulus(A),
        high_modulus(A),
    )


    K
end

#"""
#This is a fallback implementation, and shouldn't be used
#unless the dimension is greater than or equal to 3.

#This is not safe if K == B, but it is safe if K == A.
#"""
#function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
#    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(K))
#        error("Matrices must have the same modulus m")
#    end
#    if (size(low_part(A)) != size(low_part(B))) || (size(low_part(A)) != (size(low_part(K))))
#        error("Matrix dimensions must match")
#    end
#    if (A.plan == nothing) || (K.plan == nothing)
#        error("Must initialize plans first before using fallback implementation")
#    end
#    #=
#    high_part(K) .= mod.(trunc.((low_part(A)+low_part(B))./modulus(A)),high_modulus(A))
#    low_part(K) .= mod.(low_part(A) + low_part(B) - modulus(A)*high_part(K),low_modulus(A))
#    high_part(K) .= mod.(high_part(K) + high_part(A) + high_part(B),high_modulus(A))
#    =#

#    add!(K.plan,low_part(A),low_part(B),2*low_modulus(A))
#    divide_elements!(K.plan,K.plan,low_modulus(A))
#    add!(high_part(K),K.plan,high_part(A),2*high_modulus(A))
#    add!(high_part(K),high_part(K),high_part(B),2*high_modulus(A))
#    LinearAlgebra.mul!(K.plan,K.plan,low_modulus(A),low_modulus(A)^2)
#    sub!(K.plan,low_part(B),K.plan,low_modulus(B)^2)
#    add!(low_part(K),low_part(A),K.plan,low_modulus(A)^2)
#    #=
#    LinearAlgebra.mul!(high_part(K),high_part(K),1/modulus(A))
#    trunc_elements!(high_part(K))
#    =#
#    mod_elements!(low_part(K); modulus_override=low_modulus(A))
#    mod_elements!(high_part(K); modulus_override=high_modulus(A))
#    #=
#    add!(low_part(K),low_part(A),low_part(B),2*modulus(A))
#    add!(high_part(K),high_part(A),high_part(B))
#    add!(high_part(K),high_part(K),divides(low_part(K),modulus(K)),high_modulus(K))
#    mod_elements!(low_part(K); modulus_override=low_modulus(K))
#    =#

#    K
#end

function sub!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(K))
        error("Matrices must have the same modulus m")
    end
    if (size(low_part(A)) != size(low_part(B))) ||
       (size(low_part(A)) != (size(low_part(K))))
        error("Matrix dimensions must match")
    end

    #=
    high_part(C) .= mod.(trunc.((low_part(A)-low_part(B))./modulus(A)),high_modulus(A))
    low_part(C) .= mod.(low_part(A) - low_part(B) - modulus(A)*high_part(C),low_modulus(A))
    high_part(C) .= mod.(high_part(C) + high_part(A) - high_part(B) - (low_part(A).<low_part(B)),high_modulus(A))

    sub!(high_part(K),low_part(A),low_part(B))
    LinearAlgebra.mul!(high_part(K),high_part(K),1/modulus(A))
    trunc_elements!(high_part(K))
    mod_elements!(high_part(K); modulus_override=high_modulus(A))
    LinearAlgebra.mul!(low_part(K),low_part(K),modulus(A),modulus(A)^2)
    sub!(low_part(K),low_part(B),low_part(K))
    sub!(low_part(K),low_part(A),low_part(K))
    mod_elements!(low_part(K); modulus_override=low_modulus(A))
    add!(high_part(K),high_part(K),high_part(A))
    sub!(high_part(K),high_part(K),high_part(B))
    sub!(high_part(K),high_part(K),low_part(A).<low_part(B))
    mod_elements!(high_part(K); modulus_override=high_modulus(A))
    =#
    tw = TILE_WIDTH

    totalsize = length(parent(low_part(A)))

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_sub_kernel!(
        parent(low_part(K)),
        parent(high_part(K)),
        parent(low_part(A)),
        parent(high_part(A)),
        parent(low_part(B)),
        parent(high_part(B)),
        low_modulus(A),
        high_modulus(A),
        modulus(A),
    )

    # negate!(K,B)
    # add!(K,K,A)
    K
end

function scalar_multiply!(B::KaratsubaArray, A::KaratsubaArray, s::Number)
    if modulus(A) != modulus(B)
        error("Matrices must have the same modulus m")
    end
    if size(low_part(A)) != size(low_part(B))
        error("Matrix dimensions must match")
    end

    # LinearAlgebra.mul!(high_part(B),low_part(A),s,low_modulus(A)^2)
    # divide_elements!(high_part(B),high_part(B),low_modulus(A))
    # LinearAlgebra.mul!(low_part(B),high_part(A),s,high_modulus(A))
    # add!(high_part(B),high_part(B),low_part(B))
    # LinearAlgebra.mul!(low_part(B),low_part(A),s,low_modulus(A))

    tw = TILE_WIDTH

    totalsize = length(parent(low_part(A)))

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_scalar_multiply_kernel!(
        parent(low_part(B)),
        parent(high_part(B)),
        parent(low_part(A)),
        parent(high_part(A)),
        s,
        low_modulus(A),
        high_modulus(A),
    )

    #=
    high_part(B) .= mod.(trunc.((s*low_part(A))/modulus(A)),high_modulus(A))
    low_part(B) .= mod.(s*low_part(A) - modulus(B)*high_part(B),low_modulus(A))
    high_part(B) .= mod.(high_part(B) + s*high_part(A),high_modulus(A))
    =#
    B
end



# function negate!(K::KaratsubaArray, A::KaratsubaArray)

#     tw = TILE_WIDTH

#     totalsize = length(parent(low_part(A)))

#     threads = tw
#     blocks = totalsize ÷ tw

# @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(parent(low_part(K)),
#                                                              parent(high_part(K)),
#                                                              parent(low_part(A)),
#                                                              parent(high_part(A)),
#                                                              low_modulus(A),
#                                                              high_modulus(A),
#                                                             modulus(A))


#     K
# end

function negate!(K::KaratsubaArray, A::KaratsubaArray)
    tw = TILE_WIDTH

    totalsize = length(parent(low_part(A)))

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(
        parent(low_part(K)),
        parent(high_part(K)),
        parent(low_part(A)),
        parent(high_part(A)),
        low_modulus(A),
        high_modulus(A),
        modulus(A),
    )



    #=
    high_part(K) .= trunc.((modulus(A) .- low_part(A))/modulus(A))
    low_part(K) .= mod.(modulus(A) - low_part(A) - modulus(A)*high_part(K),low_modulus(A))
    high_part(K) .= mod.(mod.(high_part(K) + modulus(A) - high_part(A) - 1,modulus(A)),high_modulus(A))
    =#
    # mul!(high_part(K),high_part(K),0)
    # scalar_add!(high_part(K),high_part(K),low_modulus(A),2*low_modulus(A))
    # sub!(high_part(K),high_part(K),low_part(A),2*low_modulus(A))
    #=
    LinearAlgebra.mul!(high_part(K),high_part(K),1/low_modulus(A))
    trunc_elements!(high_part(K))
     =#
    # divide_elements!(high_part(K),high_part(K),low_modulus(A))
    # LinearAlgebra.mul!(low_part(K),high_part(K),low_modulus(A),modulus(A)^2)
    # add!(low_part(K),low_part(K),low_part(A),modulus(A)^2)
    # negate!(low_part(K),low_part(K),low_modulus(A))
    # mod_elements!(low_part(K); modulus_override=low_modulus(A))
    # scalar_add!(high_part(K),high_part(K),high_modulus(A),modulus(A)^2)
    # scalar_sub!(high_part(K),high_part(K),1,modulus(A)^2)
    # sub!(high_part(K),high_part(K),high_part(A),modulus(A))
    # mod_elements!(high_part(K); modulus_override=high_modulus(A))
    K
end

function divide_elements!(B::CuModArray, A::CuModArray, N::Integer)
    parent(B) .= div.(parent(A), N)
    return A
end

function Base.copy(A::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    B = KaratsubaArray{T,D,S,P}(
        copy(low_part(A)),
        copy(high_part(A));
        low_modulus = copy(low_modulus(A)),
        high_modulus = copy(high_modulus(A)),
        modulus = low_modulus(A) * high_modulus(A),
    )
    if !(A.plan==nothing)
        initialize_plan!(B)
    end
    B
end
