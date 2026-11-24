"""
    KaratsubaArray{T,D,S,P}

An array represented by low and high components of storage type `S`. The
component arrays must have the same concrete storage type. `P` is independent
of `S`: host `Array` components use a concrete `Array` plan, while GPU
extensions select their backend's concrete plan type. Use `low_part`,
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
        typeof(A) === typeof(B) ||
            throw(ArgumentError("Karatsuba components must have the same storage type"))
        component_type = _karatsuba_resolve_component_type(S, typeof(A), T, Val(D))
        plan_type = _karatsuba_resolve_plan_type(P, component_type, Val(D))
        isconcretetype(component_type) ||
            throw(ArgumentError("Karatsuba components require a concrete storage type"))
        isconcretetype(plan_type) ||
            throw(ArgumentError("Karatsuba plans require a concrete storage type"))
        if size(A) != size(B)
            throw(DimensionMismatch("Dimensions of matrices must match"))
        end
        #=
        if !(all(x->x<N,A) && all(x->x<N,B))
            throw(ArgumentError("Cannot have entries larger than modulus in matrices"))
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
        if T <: AbstractFloat
            arithmetic_limit = BigInt(1) << (get_bits(T) - 2)
            BigInt(M) < arithmetic_limit ||
                throw(ArgumentError("Modulus too large for $T Karatsuba arithmetic"))
        elseif M >= BigInt(2)^get_bits(T)
            throw(ArgumentError("Modulus too large"))
        end

        return new{T,D,component_type,plan_type}(A, B, nothing, N1, N2, M)
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


_karatsuba_plan_type(::Type{Array{T,D}}) where {T,D} = Array{T,D}
_karatsuba_plan_type(::Type{S}) where {T,D,S<:AbstractArray{T,D}} = S

_backend_default_component_type(::Val{B}, ::Type{T}, ::Val{D}) where {B,T,D} =
    throw(ArgumentError("the $B backend extension is required for this component type"))
_backend_default_plan_type(::Val{B}, ::Type{T}, ::Val{D}) where {B,T,D} =
    throw(ArgumentError("the $B backend extension is required for this plan type"))

@inline function _karatsuba_resolve_component_type(
    ::Type{CuModArray{T,D}},
    ::Type{A},
    ::Type{T},
    ::Val{D},
) where {T,D,A}
    isconcretetype(A) && A <: CuModArray{T,D} || throw(
        ArgumentError(
            "partial CuModArray component requests need concrete CuModArray inputs",
        ),
    )
    return A
end

@inline function _karatsuba_resolve_component_type(
    ::Type{S},
    ::Type{A},
    ::Type{T},
    ::Val{D},
) where {S,A,T,D}
    isconcretetype(S) ||
        throw(ArgumentError("Karatsuba components require a concrete storage type"))
    S === A ||
        throw(ArgumentError("Karatsuba component input does not match requested type $S"))
    return S
end

@inline function _karatsuba_resolve_requested_component_type(
    ::Type{S},
    ::Type{T},
    ::Val{D},
) where {S,T,D}
    if S === CuModArray{T,D}
        return _backend_default_component_type(Val(:cuda), T, Val(D))
    end
    isconcretetype(S) ||
        throw(ArgumentError("Karatsuba components require a concrete storage type"))
    return S
end

@inline function _karatsuba_resolve_plan_type(::Type{P}, ::Type{S}, ::Val{D}) where {P,S,D}
    expected = _karatsuba_plan_type(S)
    if isconcretetype(P)
        P === expected || throw(
            ArgumentError(
                "Requested plan storage type $P does not match the plan type $expected for component storage $S",
            ),
        )
        return P
    end
    return expected
end

function KaratsubaArray{T,D,S}(A::S, B::S; kwargs...) where {T,D,S<:AbstractArray{T,D}}
    typeof(A) === typeof(B) ||
        throw(ArgumentError("Karatsuba components must have the same storage type"))
    component_type = _karatsuba_resolve_component_type(S, typeof(A), T, Val(D))
    plan_type = _karatsuba_plan_type(component_type)
    return KaratsubaArray{T,D,component_type,plan_type}(A, B; kwargs...)
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

const _KARATSUBA_FLOAT32_EXACT_BOUND = BigInt(1) << 24
const _KARATSUBA_FLOAT64_EXACT_BOUND = BigInt(1) << 53

_karatsuba_exact_integer_bound(::Type{Float32}) = _KARATSUBA_FLOAT32_EXACT_BOUND
_karatsuba_exact_integer_bound(::Type{Float64}) = _KARATSUBA_FLOAT64_EXACT_BOUND

function _karatsuba_checked_working_modulus(
    value::BigInt,
    label::AbstractString,
    ::Type{T} = Float64,
) where {T<:Union{Float32,Float64}}
    (0 < value <= typemax(Int)) ||
        throw(ArgumentError("$label does not fit a positive Int modulus"))
    value <= _karatsuba_exact_integer_bound(T) ||
        throw(ArgumentError("$label exceeds the exact-integer $T range"))
    return Int(value)
end

function _karatsuba_checked_sum_bound(
    a::Int,
    b::Int,
    label::AbstractString,
    ::Type{T} = Float64,
) where {T<:Union{Float32,Float64}}
    value = BigInt(a) + BigInt(b)
    value <= typemax(Int) || throw(OverflowError("$label overflows Int"))
    # The operands are strictly below a and b, so a sum bound of 2^p still
    # promises actual values no larger than 2^p - 1.
    value <= _karatsuba_exact_integer_bound(T) ||
        throw(ArgumentError("$label exceeds the exact-integer $T range"))
    return Int(value)
end

function _karatsuba_safe_product_parameters(
    input_bound::Int,
    reduction_modulus::Int,
    ::Type{T} = Float64,
) where {T<:Union{Float32,Float64}}
    effective_bound = max(input_bound, 2)
    max_ops = find_max_stripe_ops(T, effective_bound; modulus = reduction_modulus)
    max_ops > 0 || throw(
        ArgumentError(
            "no exact $T stripe contribution fits input bound $effective_bound and reduction modulus $reduction_modulus",
        ),
    )
    return (input_bound = effective_bound, modulus = reduction_modulus, max_ops = max_ops)
end

function _karatsuba_matmul_product_plan(
    ::Type{T},
    input_low_a::Int,
    input_high_a::Int,
    input_low_b::Int,
    input_high_b::Int,
    output_low::Int,
    output_high::Int,
) where {T}
    (T === Float32 || T === Float64) ||
        throw(ArgumentError("Karatsuba products support Float32 or Float64 storage"))
    all(
        >(0),
        (input_low_a, input_high_a, input_low_b, input_high_b, output_low, output_high),
    ) || throw(ArgumentError("Karatsuba split moduli must be positive"))
    input_low_a == input_low_b || throw(
        ModulusMismatchError(
            "three-product Karatsuba matrix multiplication requires a common low modulus for A and B",
        ),
    )

    input_radix = input_low_a
    output_capacity = _karatsuba_checked_working_modulus(
        BigInt(output_low) * BigInt(output_high),
        "output component capacity",
        T,
    )
    sum_bound_a =
        _karatsuba_checked_sum_bound(input_radix, input_high_a, "left sum input bound", T)
    sum_bound_b =
        _karatsuba_checked_sum_bound(input_radix, input_high_b, "right sum input bound", T)
    middle_bound = max(sum_bound_a, sum_bound_b)
    high_bound = max(input_high_a, input_high_b)
    radix_mod_capacity = Int(mod(BigInt(input_radix), BigInt(output_capacity)))
    radix_squared_mod_capacity = Int(mod(BigInt(input_radix)^2, BigInt(output_capacity)))
    low = _karatsuba_safe_product_parameters(input_radix, output_capacity, T)
    middle = _karatsuba_safe_product_parameters(middle_bound, output_capacity, T)
    high = _karatsuba_safe_product_parameters(high_bound, output_capacity, T)

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

function _karatsuba_matmul_product_plan(
    input_low_a::Int,
    input_high_a::Int,
    input_low_b::Int,
    input_high_b::Int,
    output_low::Int,
    output_high::Int,
)
    return _karatsuba_matmul_product_plan(
        Float64,
        input_low_a,
        input_high_a,
        input_low_b,
        input_high_b,
        output_low,
        output_high,
    )
end

function _karatsuba_gemv_dot_bound(
    ::Type{T},
    terms::Int,
    left_bound::Int,
    right_bound::Int,
    label::AbstractString,
) where {T<:Union{Float32,Float64}}
    terms >= 0 || throw(ArgumentError("GEMV reduction length must be nonnegative"))
    left_bound > 0 && right_bound > 0 ||
        throw(ArgumentError("GEMV component bounds must be positive"))
    bound = BigInt(terms) * BigInt(left_bound - 1) * BigInt(right_bound - 1)
    bound <= _karatsuba_exact_integer_bound(T) ||
        throw(ArgumentError("$label exceeds the exact-integer $T accumulation range"))
    return bound
end

function _karatsuba_gemv_sum_bound(
    left_bound::Int,
    right_bound::Int,
    label::AbstractString,
    ::Type{T},
) where {T<:Union{Float32,Float64}}
    left_bound > 0 && right_bound > 0 ||
        throw(ArgumentError("GEMV component bounds must be positive"))
    # Each input digit is below its modulus. The largest sum is therefore
    # (left_bound - 1) + (right_bound - 1), so the exclusive bound is a+b-1.
    value = BigInt(left_bound) + BigInt(right_bound) - 1
    value <= typemax(Int) || throw(OverflowError("$label overflows Int"))
    value - 1 <= _karatsuba_exact_integer_bound(T) ||
        throw(ArgumentError("$label exceeds the exact-integer $T range"))
    return Int(value)
end

function _karatsuba_gemv_validate_radices(
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
        ModulusMismatchError(
            "three-product Karatsuba GEMV requires a common low modulus for A and x",
        ),
    )
    output_low == input_low_a || throw(
        ModulusMismatchError(
            "Karatsuba GEMV requires the output low modulus to match the input radix",
        ),
    )
    rem(output_low, output_high) == 0 || throw(
        ArgumentError(
            "Karatsuba GEMV requires output high modulus N2=$output_high to divide low radix N1=$output_low",
        ),
    )
    return nothing
end

function _karatsuba_gemv_precision_plan(
    ::Type{T},
    terms::Int,
    input_low_a::Int,
    input_high_a::Int,
    input_low_b::Int,
    input_high_b::Int,
    output_low::Int,
    output_high::Int,
) where {T<:Union{Float32,Float64}}
    _karatsuba_gemv_validate_radices(
        input_low_a,
        input_high_a,
        input_low_b,
        input_high_b,
        output_low,
        output_high,
    )

    # Full sums A0+A1 and x0+x1 feed the raw middle dot product. Each cuBLAS
    # total is an integer no larger than 2^p, including the exact endpoint.
    sum_bound_a =
        _karatsuba_gemv_sum_bound(input_low_a, input_high_a, "left GEMV sum input bound", T)
    sum_bound_b = _karatsuba_gemv_sum_bound(
        input_low_b,
        input_high_b,
        "right GEMV sum input bound",
        T,
    )
    low_bound =
        _karatsuba_gemv_dot_bound(T, terms, input_low_a, input_low_b, "low GEMV product")
    middle_bound =
        _karatsuba_gemv_dot_bound(T, terms, sum_bound_a, sum_bound_b, "middle GEMV product")
    high_bound =
        _karatsuba_gemv_dot_bound(T, terms, input_high_a, input_high_b, "high GEMV product")
    output_capacity = _karatsuba_checked_working_modulus(
        BigInt(output_low) * BigInt(output_high),
        "GEMV output component capacity",
        T,
    )
    # For canonical nonnegative digits, P1=P0+P2+cross. Both sequential
    # differences are nonnegative, carry<=P0, and carry+cross<=P1.
    return (
        low_dot_bound = low_bound,
        middle_dot_bound = middle_bound,
        high_dot_bound = high_bound,
        sum_input_bounds = (sum_bound_a, sum_bound_b),
        output_capacity = output_capacity,
        high_total_bound = middle_bound,
    )
end

Base.size(K::KaratsubaArray) = size(low_part(K))
Base.axes(K::KaratsubaArray) = axes(low_part(K))

Base.getindex(K::KaratsubaArray, i::Int, j::Int) =
    low_part(K)[i, j] + low_modulus(K) * high_part(K)[i, j]
Base.getindex(K::KaratsubaArray, i::Int) = low_part(K)[i] + low_modulus(K) * high_part(K)[i]

function Base.setindex!(K::KaratsubaArray, value, i::Int, j::Int)
    low_part(K)[i, j] = rem(value, low_modulus(K))
    high_part(K)[i, j] = div(value, low_modulus(K))
    return value
end

function Base.setindex!(K::KaratsubaArray, value, i::Int)
    low_part(K)[i] = rem(value, low_modulus(K))
    high_part(K)[i] = div(value, low_modulus(K))
    return value
end

function Base.copy!(dest::KaratsubaArray, src::KaratsubaArray)
    copy!(low_part(dest), low_part(src))
    copy!(high_part(dest), high_part(src))
    if src.plan !== nothing && dest.plan === nothing
        initialize_plan!(dest)
    end
    return dest
end

function zero!(A::AbstractArray)
    fill!(A, zero(eltype(A)))
    return A
end

function zero!(K::KaratsubaArray)
    zero!(low_part(K))
    zero!(high_part(K))
    return K
end

function _karatsuba_bigint(value)
    if value isa Integer
        return BigInt(value)
    elseif value isa AbstractFloat && isfinite(value) && isinteger(value)
        return BigInt(value)
    end
    throw(InexactError(:to_karatsuba, BigInt, value))
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

_karatsuba_host_array(A::AbstractArray) = Array(A)
_karatsuba_component(
    ::Type{T},
    values::AbstractArray,
    source,
    component_modulus,
) where {T} = values

function _karatsuba_reconstruct_host(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:AbstractArray{S,D},Plan<:AbstractArray{S,D}}
    low = _karatsuba_host_array(low_part(K))
    high = _karatsuba_host_array(high_part(K))
    result = Array{T,D}(undef, size(low))
    for i in eachindex(result, low, high)
        value =
            _karatsuba_bigint(low[i]) + BigInt(low_modulus(K)) * _karatsuba_bigint(high[i])
        result[i] = _karatsuba_exact_convert(T, value)
    end
    return result
end

Base.Array(K::KaratsubaArray) = _karatsuba_reconstruct_host(BigInt, K)

function from_karatsuba(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:AbstractArray{S,D},Plan<:AbstractArray{S,D}}
    return _karatsuba_reconstruct_host(T, K)
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
    low = Array{T}(undef, size(input))
    high = Array{T}(undef, size(input))
    for i in eachindex(input, low, high)
        high_value, low_value = fldmod(_karatsuba_bigint(input[i]), BigInt(N1))
        low[i] = _karatsuba_exact_convert(T, low_value)
        high[i] = _karatsuba_exact_convert(T, high_value)
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

to_karatsuba(A::AbstractArray; kwargs...) = _split_karatsuba(eltype(A), A; kwargs...)
to_karatsuba(::Type{T}, A::AbstractArray; kwargs...) where {T} =
    _split_karatsuba(T, A; kwargs...)
KaratsubaMatrix(::Type{T}, A::AbstractMatrix; kwargs...) where {T} =
    _split_karatsuba(T, A; kwargs...)
KaratsubaVector(::Type{T}, A::AbstractVector; kwargs...) where {T} =
    _split_karatsuba(T, A; kwargs...)

function _karatsuba_zero_component(::Type{Array{T,D}}, dims::Dims{D}, ::Int) where {T,D}
    return zeros(T, dims)
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
    component_type = _karatsuba_resolve_requested_component_type(S, T, Val(D))
    plan_type = _karatsuba_resolve_plan_type(P, component_type, Val(D))
    isconcretetype(component_type) ||
        throw(ArgumentError("Karatsuba zero factories require a concrete component type"))
    isconcretetype(plan_type) ||
        throw(ArgumentError("Karatsuba zero factories require a concrete plan type"))
    expected_plan_type = _karatsuba_plan_type(component_type)
    plan_type === expected_plan_type || throw(
        ArgumentError(
            "Requested plan storage type $plan_type does not match the plan type $expected_plan_type for component storage $component_type",
        ),
    )
    N1, N2, M = _karatsuba_parameters(
        zeros(T, dims);
        low_modulus = low_modulus,
        high_modulus = high_modulus,
        modulus = modulus,
    )
    first = _karatsuba_zero_component(component_type, dims, N1)
    second = _karatsuba_zero_component(component_type, dims, N2)
    typeof(first) === component_type || throw(
        ArgumentError(
            "Storage factory returned $(typeof(first)) for requested component type $component_type",
        ),
    )
    typeof(second) === component_type || throw(
        ArgumentError(
            "Storage factory returned $(typeof(second)) for requested component type $component_type",
        ),
    )
    return KaratsubaArray{T,D,component_type,plan_type}(
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
    component_type = _karatsuba_resolve_requested_component_type(S, T, Val(D))
    return _karatsuba_zeros(
        T,
        component_type,
        _karatsuba_plan_type(component_type),
        dims;
        kwargs...,
    )
end

Base.zeros(
    ::Type{KaratsubaArray{T,2,S}},
    rows::Integer,
    cols::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,2}} =
    _karatsuba_zeros(T, S, (Int(rows), Int(cols)); kwargs...)
Base.zeros(
    ::Type{KaratsubaArray{T,1,S}},
    length::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,1}} = _karatsuba_zeros(T, S, (Int(length),); kwargs...)
Base.zeros(
    ::Type{KaratsubaArray{T,2,S,P}},
    rows::Integer,
    cols::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,2},P<:AbstractArray{T,2}} =
    _karatsuba_zeros(T, S, P, (Int(rows), Int(cols)); kwargs...)
Base.zeros(
    ::Type{KaratsubaArray{T,1,S,P}},
    length::Integer;
    kwargs...,
) where {T,S<:AbstractArray{T,1},P<:AbstractArray{T,1}} =
    _karatsuba_zeros(T, S, P, (Int(length),); kwargs...)

function initialize_plan!(
    K::KaratsubaArray{T,D,S,P},
) where {T,D,S<:Array{T,D},P<:Array{T,D}}
    expected = _karatsuba_plan_type(S)
    P === expected || throw(
        ArgumentError("initialize_plan! requires plan storage type $expected, got $P"),
    )
    K.plan = zeros(T, size(low_part(K)))
    return K
end
