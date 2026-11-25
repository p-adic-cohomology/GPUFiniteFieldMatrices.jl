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


# The quotient can be negative even though the Euclidean remainder is always
# nonnegative. Keep its magnitude unsigned until the destination type is known.
@inline function _karatsuba_quotient(::Type{T}, magnitude, negative) where {T<:Signed}
    limit = UInt64(typemax(T)) + UInt64(negative)
    magnitude <= limit || return zero(T), false
    value = negative ? reinterpret(Int64, zero(UInt64) - magnitude) : Int64(magnitude)
    return T(value), true
end

@inline function _karatsuba_quotient(::Type{T}, magnitude, negative) where {T<:Unsigned}
    negative && !iszero(magnitude) && return zero(T), false
    return _karatsuba_gpu_component(T, magnitude)
end

@inline function _karatsuba_quotient(
    ::Type{T},
    magnitude,
    negative,
) where {T<:AbstractFloat}
    value, exact = _karatsuba_gpu_component(T, magnitude)
    return negative ? -value : value, exact
end

@inline function _karatsuba_split_magnitude(
    ::Type{T},
    value::UInt64,
    negative,
    radix,
) where {T}
    quotient, remainder = divrem(value, radix)
    if negative && !iszero(remainder)
        quotient += one(UInt64)
        remainder = radix - remainder
    end
    low, low_ok = _karatsuba_gpu_component(T, remainder)
    high, high_ok = _karatsuba_quotient(T, quotient, negative)
    return low, high, low_ok && high_ok
end

@inline function _karatsuba_split_value(::Type{T}, value::S, radix) where {T,S<:Signed}
    signed = Int64(value)
    magnitude = reinterpret(UInt64, signed)
    negative = signed < 0
    return _karatsuba_split_magnitude(T, negative ? -magnitude : magnitude, negative, radix)
end

@inline function _karatsuba_split_value(::Type{T}, value::Unsigned, radix) where {T}
    return _karatsuba_split_magnitude(T, UInt64(value), false, radix)
end

# Long division covers integral floats beyond UInt64. Retain the leading 64
# quotient bits and track whether the remaining bits are all zero (exact) or
# all one (an exact carry remains possible when taking the floor of -value).
function _karatsuba_split_large_float(::Type{T}, value::S, radix) where {T,S}
    magnitude = abs(value)
    power = one(S)
    while power <= magnitude * S(0.5)
        power += power
    end
    prefix = zero(UInt64)
    remainder = zero(UInt64)
    bits = 0
    tail = 0
    tail_zero = true
    tail_one = true
    while power >= one(S)
        remainder <<= 1
        if magnitude >= power
            magnitude -= power
            remainder += one(UInt64)
        end
        digit = remainder >= radix
        digit && (remainder -= radix)
        if bits < 64
            prefix = (prefix << 1) | UInt64(digit)
            bits += Int(digit || bits != 0)
        else
            tail += 1
            tail_zero &= !digit
            tail_one &= digit
        end
        power *= S(0.5)
    end
    negative = value < zero(S)
    if negative && !iszero(remainder)
        remainder = radix - remainder
        (tail == 0 || tail_one) || return zero(T), zero(T), false
        if prefix == typemax(UInt64)
            prefix = one(UInt64)
            tail += 64
        else
            prefix += one(UInt64)
        end
    else
        tail_zero || return zero(T), zero(T), false
    end
    low, low_ok = _karatsuba_gpu_component(T, remainder)
    if T <: Integer && tail > 0
        return zero(T), zero(T), false
    end
    high, high_ok = _karatsuba_quotient(T, prefix, negative)
    for _ = 1:tail
        high += high
    end
    return low, high, low_ok && high_ok && isfinite(high)
end

@inline function _karatsuba_split_value(
    ::Type{T},
    value::S,
    radix,
) where {T,S<:AbstractFloat}
    (isfinite(value) && isinteger(value)) || return zero(T), zero(T), false
    magnitude = abs(value)
    if magnitude < S(typemax(UInt64))
        return _karatsuba_split_magnitude(T, UInt64(magnitude), value < zero(S), radix)
    end
    return _karatsuba_split_large_float(T, value, radix)
end

@kernel function _karatsuba_split_kernel!(low, high, valid, input, radix)
    i = @index(Global, Linear)
    @inbounds begin
        a, b, exact = _karatsuba_split_value(eltype(low), input[i], radix)
        low[i] = a
        high[i] = b
        valid[i] = exact
    end
end

@kernel function _karatsuba_reconstruct_kernel!(output, valid, low, high, radix)
    i = @index(Global, Linear)
    @inbounds begin
        a, a_ok = _karatsuba_gpu_component(eltype(output), low[i])
        b, b_ok = _karatsuba_gpu_component(eltype(output), high[i])
        value, exact = _karatsuba_gpu_reconstruct(eltype(output), a, b, radix)
        valid[i] = a_ok && b_ok && exact
        output[i] = value
    end
end
