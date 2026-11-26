# MARK - Matrix multiplication

@inline function _karatsuba_matrix_storage_index(i, rows, stride1, stride2)
    row = mod(i - 1, rows)
    col = div(i - 1, rows)
    return 1 + row * stride1 + col * stride2
end

function karatsuba_matmul_kernel_1!(
    Aplan,
    Adata1,
    Adata2,
    Arows,
    Acols,
    Aplan_s1,
    Aplan_s2,
    Adata1_s1,
    Adata1_s2,
    Adata2_s1,
    Adata2_s2,
    Bplan,
    Bdata1,
    Bdata2,
    Brows,
    Bcols,
    Bplan_s1,
    Bplan_s2,
    Bdata1_s1,
    Bdata1_s2,
    Bdata2_s1,
    Bdata2_s2,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= Arows * Acols
        plan_i = _karatsuba_matrix_storage_index(i, Arows, Aplan_s1, Aplan_s2)
        data1_i = _karatsuba_matrix_storage_index(i, Arows, Adata1_s1, Adata1_s2)
        data2_i = _karatsuba_matrix_storage_index(i, Arows, Adata2_s1, Adata2_s2)
        Aplan[plan_i] = Adata1[data1_i] + Adata2[data2_i]
    end

    if i <= Brows * Bcols
        plan_i = _karatsuba_matrix_storage_index(i, Brows, Bplan_s1, Bplan_s2)
        data1_i = _karatsuba_matrix_storage_index(i, Brows, Bdata1_s1, Bdata1_s2)
        data2_i = _karatsuba_matrix_storage_index(i, Brows, Bdata2_s1, Bdata2_s2)
        Bplan[plan_i] = Bdata1[data1_i] + Bdata2[data2_i]
    end

    nothing
end

# GEMV stages the full (unwrapped) low-plus-high input sums for P1.
function karatsuba_matmul_kernel_1!(Aplan, Adata1, Adata2, Bplan, Bdata1, Bdata2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= length(Bplan) && i <= length(Bdata1) && i <= length(Bdata2)
        Bplan[i] = Bdata1[i] + Bdata2[i]
    end

    if i <= length(Aplan) && i <= length(Adata1) && i <= length(Adata2)
        Aplan[i] = Adata1[i] + Adata2[i]
    end

    nothing
end

@inline function _karatsuba_mod_add(
    a::T,
    b::T,
    modulus::T,
) where {T<:Union{Float32,Float64}}
    modulus == one(T) && return zero(T)
    gap = modulus - b
    return a >= gap ? a - gap : a + b
end

@inline function _karatsuba_mod_sub(
    a::T,
    b::T,
    modulus::T,
) where {T<:Union{Float32,Float64}}
    modulus == one(T) && return zero(T)
    return a >= b ? a - b : modulus - (b - a)
end

@inline function _karatsuba_mod_mul(
    a::T,
    b::T,
    modulus::T,
) where {T<:Union{Float32,Float64}}
    # Inputs and the loop state are integer residues in [0, modulus). Each
    # step uses bounded modular addition instead of forming a*b.
    modulus == one(T) && return zero(T)
    result = zero(T)
    factor = a
    multiplier = b
    while multiplier > zero(T)
        half = div(multiplier, T(2))
        if multiplier != T(2) * half
            result = _karatsuba_mod_add(result, factor, modulus)
        end
        multiplier = half
        if multiplier > zero(T)
            factor = _karatsuba_mod_add(factor, factor, modulus)
        end
    end
    return result
end

@inline function _karatsuba_recombine(
    p0::T,
    p1::T,
    p2::T,
    radix_mod_capacity::T,
    radix_squared_mod_capacity::T,
    output_radix::T,
    output_capacity::T,
) where {T<:Union{Float32,Float64}}
    # p0, p1, and p2 are the three stripe products reduced modulo Q.
    cross = _karatsuba_mod_sub(p1, p0, output_capacity)
    cross = _karatsuba_mod_sub(cross, p2, output_capacity)
    weighted_cross = _karatsuba_mod_mul(cross, radix_mod_capacity, output_capacity)
    weighted_high = _karatsuba_mod_mul(p2, radix_squared_mod_capacity, output_capacity)
    value = _karatsuba_mod_add(p0, weighted_cross, output_capacity)
    value = _karatsuba_mod_add(value, weighted_high, output_capacity)

    # Q is checked against T's exact-integer range and Int modulus bound.
    # Integer div/rem avoids a rounded floating quotient near the capacity.
    value_integer = Int(value)
    radix_integer = Int(output_radix)
    low = T(rem(value_integer, radix_integer))
    high = T(div(value_integer, radix_integer))
    return low, high
end

function karatsuba_matmul_kernel_2!(
    Cdata1,
    Cdata2,
    Cplan,
    rows,
    cols,
    Cdata1_s1,
    Cdata1_s2,
    Cdata2_s1,
    Cdata2_s2,
    Cplan_s1,
    Cplan_s2,
    radix_mod_capacity,
    radix_squared_mod_capacity,
    output_radix,
    output_capacity,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= rows * cols
        low_i = _karatsuba_matrix_storage_index(i, rows, Cdata1_s1, Cdata1_s2)
        high_i = _karatsuba_matrix_storage_index(i, rows, Cdata2_s1, Cdata2_s2)
        plan_i = _karatsuba_matrix_storage_index(i, rows, Cplan_s1, Cplan_s2)
        p0 = Cdata1[low_i]
        p1 = Cdata2[high_i]
        p2 = Cplan[plan_i]
        low, high = _karatsuba_recombine(
            p0,
            p1,
            p2,
            radix_mod_capacity,
            radix_squared_mod_capacity,
            output_radix,
            output_capacity,
        )

        Cdata1[low_i] = low
        Cdata2[high_i] = high
    end

    nothing
end

# GEMV consumes raw exact integer-valued P0, P1, and P2 dot totals.
@inline function _karatsuba_gemv_recombine(
    p0::T,
    p1::T,
    p2::T,
    N1::T,
    N2::T,
) where {T<:Union{Float32,Float64}}
    cross_partial = p1 - p0
    cross = cross_partial - p2
    low = mod(p0, N1)
    carry = div(p0, N1)
    high_total = carry + cross
    high = mod(high_total, N2)
    return low, high
end

function karatsuba_matmul_kernel_2!(Cdata1, Cdata2, Bplan, N1, N2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= length(Cdata1) && i <= length(Cdata2) && i <= length(Bplan)
        low, high = _karatsuba_gemv_recombine(Cdata1[i], Cdata2[i], Bplan[i], N1, N2)
        Cdata1[i] = low
        Cdata2[i] = high
    end

    nothing
end


# function karatsuba_matmul_kernel_3!(Cplan,Cdata1,Cdata2,Bplan,N1,N2)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x


#     nothing
# end
