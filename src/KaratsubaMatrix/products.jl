function _karatsuba_product_size(A::KaratsubaMatrix, B)
    modulus(A) == modulus(B) ||
        throw(ModulusMismatchError("Karatsuba operands must have the same modulus M"))
    return _stripe_output_size(A, B)
end

function _karatsuba_product_size(C, A::KaratsubaMatrix, B)
    dims = _karatsuba_product_size(A, B)
    modulus(C) == modulus(A) ||
        throw(ModulusMismatchError("Karatsuba operands must have the same modulus M"))
    size(C) == dims ||
        throw(DimensionMismatch("Karatsuba product destination has the wrong size"))
    return dims
end

function _karatsuba_product_plan(K::KaratsubaArray)
    plan = K.plan
    plan === nothing && throw(ArgumentError("Call initialize_plan! before multiplying"))
    return plan
end

function _karatsuba_product_planes(K::KaratsubaArray)
    planes = (low_part(K), high_part(K), _karatsuba_product_plan(K))
    all(plane -> size(plane) == size(K), planes) || throw(
        DimensionMismatch(
            "Karatsuba components and scratch must have the same logical size",
        ),
    )
    return map(_karatsuba_logical_view, planes)
end

function _karatsuba_check_product_aliases(c, a, b, same_input)
    sources = (a[1], a[2], b[1], b[2])
    destinations = same_input ? (a[3], c...) : (a[3], b[3], c...)
    for (i, dest) in pairs(destinations)
        any(input -> _stripe_mightalias(dest, input), sources) && throw(
            ArgumentError("Karatsuba output and scratch must be disjoint from inputs"),
        )
        for j = 1:(i-1)
            _stripe_mightalias(dest, destinations[j]) && throw(
                ArgumentError("Karatsuba output and scratch must be mutually disjoint"),
            )
        end
    end
    return nothing
end

function _karatsuba_product_workspace(K::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    return KaratsubaArray{T,D,S,P}(
        low_part(K),
        high_part(K);
        low_modulus = low_modulus(K),
        high_modulus = high_modulus(K),
        modulus = modulus(K),
    )
end

function _karatsuba_allocating_product(A::KaratsubaMatrix, B)
    dims = _karatsuba_product_size(A, B)
    left = low_part(A)
    for part in (high_part(A), low_part(B), high_part(B))
        _karatsuba_check_device(left, part)
    end
    a = _karatsuba_product_workspace(A)
    b = A === B ? a : _karatsuba_product_workspace(B)
    c = _karatsuba_output_like(A, dims)
    initialize_plan!(a)
    a === b || initialize_plan!(b)
    initialize_plan!(c)
    return LinearAlgebra.mul!(c, a, b)
end

@kernel inbounds = true function _karatsuba_sum_planes_kernel!(sum, low, high)
    i = @index(Global, Linear)
    sum[i] = low[i] + high[i]
end

function _karatsuba_sum_planes!(low, high, sum)
    isempty(sum) && return sum
    backend = KernelAbstractions.get_backend(sum)
    _karatsuba_sum_planes_kernel!(backend, _ELEMENTWISE_WORKGROUP_SIZE)(
        sum,
        low,
        high;
        ndrange = length(sum),
    )
    return sum
end

# Metal GEMM products are reduced modulo Q <= 2^24. The weighted sum of
# the three Int64 residues fits in Int64 even at that endpoint.
@kernel inbounds = true function _karatsuba_gemm_recombine_kernel!(
    low,
    high,
    p2,
    radix,
    radix_mod_q,
    radix_squared_mod_q,
    q,
)
    i = @index(Global, Linear)
    p0_i = unsafe_trunc(Int64, low[i])
    p1_i = unsafe_trunc(Int64, high[i])
    p2_i = unsafe_trunc(Int64, p2[i])
    cross = p1_i - p0_i - p2_i
    value = mod(p0_i + radix_mod_q * cross + radix_squared_mod_q * p2_i, q)
    hi, lo = divrem(value, radix)
    low[i] = lo
    high[i] = hi
end

# With high_modulus dividing radix, the R^2 P2 term vanishes modulo R*H.
# The exact GEMV budget bounds all three full dot totals to 2^24.
@kernel inbounds = true function _karatsuba_gemv_recombine_kernel!(low, high, p2, radix, H)
    i = @index(Global, Linear)
    p0_i = unsafe_trunc(Int64, low[i])
    cross = unsafe_trunc(Int64, high[i]) - p0_i - unsafe_trunc(Int64, p2[i])
    carry, digit = divrem(p0_i, radix)
    low[i] = digit
    high[i] = mod(carry + cross, H)
end
