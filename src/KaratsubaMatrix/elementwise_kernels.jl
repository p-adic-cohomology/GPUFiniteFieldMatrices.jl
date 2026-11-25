# All split arithmetic works modulo the two-plane capacity R*H, independently
# of the representation metadata M. Work with the digits so R*H can exceed
# the machine integer range. Each individual digit and carry fits in UInt64.
@inline function _karatsuba_digits(low, high, radix, high_modulus)
    carry, digit = fldmod(Int64(low), Int64(radix))
    high_digit = _add_mod_u64(
        UInt64(mod(Int64(high), Int64(high_modulus))),
        UInt64(mod(carry, Int64(high_modulus))),
        high_modulus,
    )
    return UInt64(digit), high_digit
end

@inline function _karatsuba_add_digits(a, b, radix, high_modulus)
    carry, low = divrem(a[1] + b[1], radix)
    high = _add_mod_u64(a[2], b[2], high_modulus)
    return low, _add_mod_u64(high, mod(carry, high_modulus), high_modulus)
end

@inline function _karatsuba_negate_digits(a, radix, high_modulus)
    if iszero(a[1])
        return zero(UInt64), _neg_mod_u64(a[2], high_modulus)
    end
    return radix - a[1], high_modulus - a[2] - one(UInt64)
end

@inline _karatsuba_binary_digits(::Val{:add}, a, b, radix, high_modulus) =
    _karatsuba_add_digits(a, b, radix, high_modulus)

@inline _karatsuba_binary_digits(::Val{:sub}, a, b, radix, high_modulus) =
    _karatsuba_add_digits(
        a,
        _karatsuba_negate_digits(b, radix, high_modulus),
        radix,
        high_modulus,
    )

@inline _karatsuba_unary_digits(::Val{:neg}, a, _, radix, high_modulus) =
    _karatsuba_negate_digits(a, radix, high_modulus)

@inline function _karatsuba_unary_digits(::Val{:scale}, a, scalar, radix, high_modulus)
    # Multiply by the low scalar digit with double-and-add. The high
    # scalar digit contributes only to the high output plane.
    radix_high = _mul_mod_u64(mod(radix, high_modulus), a[2], high_modulus)
    a_mod_high = _add_mod_u64(mod(a[1], high_modulus), radix_high, high_modulus)
    high = _mul_mod_u64(a_mod_high, scalar[2], high_modulus)
    product = (zero(UInt64), high)
    multiplier = scalar[1]
    while !iszero(multiplier)
        isodd(multiplier) &&
            (product = _karatsuba_add_digits(product, a, radix, high_modulus))
        multiplier >>= 1
        iszero(multiplier) || (a = _karatsuba_add_digits(a, a, radix, high_modulus))
    end
    return product
end

@kernel function _karatsuba_binary_kernel!(
    low,
    high,
    a_low,
    a_high,
    b_low,
    b_high,
    R,
    H,
    op,
)
    i = @index(Global, Linear)
    @inbounds begin
        a = _karatsuba_digits(a_low[i], a_high[i], R, H)
        b = _karatsuba_digits(b_low[i], b_high[i], R, H)
        l, h = _karatsuba_binary_digits(op, a, b, R, H)
        low[i] = l
        high[i] = h
    end
end

@kernel function _karatsuba_unary_kernel!(low, high, a_low, a_high, R, H, op, scalar)
    i = @index(Global, Linear)
    @inbounds begin
        a = _karatsuba_digits(a_low[i], a_high[i], R, H)
        l, h = _karatsuba_unary_digits(op, a, scalar, R, H)
        low[i] = l
        high[i] = h
    end
end
