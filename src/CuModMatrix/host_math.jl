function inverse_permutation(permutation::Vector{Int})
    inverse = Vector{Int}(undef, length(permutation))
    for i in eachindex(permutation)
        inverse[permutation[i]] = i
    end
    return inverse
end

function mod_inv(a::Integer, m::Integer)
    aa = mod(a, m)
    t = 0
    newt = 1
    r = m
    newr = aa
    while newr != 0
        q = div(r, newr)
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    r == 1 || throw(LinearAlgebra.SingularException(0))
    return mod(t, m)
end

mod_inv(a::Number, m::Integer) = mod_inv(Int(round(a)), m)

function get_bits(type::Type)
    if type <: AbstractFloat
        type === Float16 && return 11
        type === Float32 && return 24
        type === Float64 && return 53
    elseif type <: Union{Signed,Unsigned} && isprimitivetype(type)
        return 8 * sizeof(type) - (type <: Signed ? 1 : 0)
    end
    throw(
        ArgumentError(
            "unsupported type $type; expected Float16, Float32, Float64, or a primitive fixed-width signed or unsigned integer type",
        ),
    )
end

get_bits(type) = throw(
    ArgumentError(
        "unsupported type $type; expected Float16, Float32, Float64, or a primitive fixed-width signed or unsigned integer type",
    ),
)

function find_max_ops(type, N)
    bits = get_bits(type)
    maxterm = BigInt(max(N - 1, 1))^2
    value = (BigInt(2)^bits - 1) ÷ maxterm - 1
    return value > typemax(Int) ? typemax(Int) : max(Int(value), 0)
end

"""
    find_max_stripe_ops(type, input_bound; modulus=input_bound)

Return a conservative exact-integer accumulation budget for products whose
inputs are in `0:(input_bound - 1)` and whose reduced prior accumulator is less
than `modulus`.
"""
function find_max_stripe_ops(type, input_bound; modulus = input_bound)
    bits = if type === Float64
        53
    elseif type === Float32
        24
    elseif type === Float16
        11
    else
        throw(ArgumentError("The input type is not supported for floating GEMM"))
    end

    input_bound isa Integer && input_bound > 1 ||
        throw(ArgumentError("input_bound must be an integer greater than one"))
    modulus isa Integer && modulus > 0 ||
        throw(ArgumentError("modulus must be a positive integer"))

    exact_max = BigInt(2)^bits - 1
    max_product = BigInt(input_bound - 1)^2
    max_prior = BigInt(modulus - 1)
    input_budget = exact_max ÷ max_product - 1
    accumulator_budget = (exact_max - max_prior) ÷ max_product
    safe_budget = max(min(input_budget, accumulator_budget), 0)
    return Int(min(safe_budget, typemax(Int)))
end
