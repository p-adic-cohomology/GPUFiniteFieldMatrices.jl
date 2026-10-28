
@inline function karatsuba_add_helper(a1, a2, b1, b2, N1, N2)

    temp = (a1 + b1) % (2*N1)
    temp = div(temp, N1)
    res2 = (temp + a2) % (2*N2)
    res2 = (res2 + b2) % (2*N2)

    temp = (temp * N1) % (N1^2)
    temp = (b1 - temp) % (N1^2)
    res1 = (a1 + temp) % (N1^2)

    res1 %= N1
    res2 %= N2

    (res1, res2)
end

"""

Works even for multidimensional arrays because CuArrays support linear indexing

"""
function karatsuba_add_kernel!(Kdata1, Kdata2, Adata1, Adata2, Bdata1, Bdata2, N1, N2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    (res1, res2) = karatsuba_add_helper(Adata1[i], Adata2[i], Bdata1[i], Bdata2[i], N1, N2)

    # Kplan[i] = (Adata1[i] + Bdata1[i]) % (2*N1)
    # Kplan[i] = div(Kplan[i],N1)
    # Kdata2[i] = (Kplan[i] + Adata2[i]) % (2*N2)
    # Kdata2[i] = (Kdata2[i] + Bdata2[i]) % (2*N2)

    # Kplan[i] = (Kplan[i] * N1) % (N1^2)
    # Kplan[i] = (Bdata1[i] - Kplan[i]) % (N1^2)
    # Kdata1[i] = (Adata1[i] + Kplan[i]) % (N1^2)

    # Kdata1[i] %= N1
    # Kdata2[i] %= N2

    Kdata1[i] = res1
    Kdata2[i] = res2

    return
end

@inline function karatsuba_scalar_multiply_helper(a1, a2, s, N1, N2)
    res2 = (a1*s) % (N1^2)
    res2 = div(res2, N1)
    res1 = (a2*s) % N2
    res2 = (res2 + res1) % N2
    res1 = (a1 * s) % N1

    (res1, res2)
end

function karatsuba_scalar_multiply_kernel!(Bdata1, Bdata2, Adata1, Adata2, s, N1, N2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Bdata2[i] = (Adata1[i]*s) % (N1^2)
    # Bdata2[i] = div(Bdata2[i], N1)
    # Bdata1[i] = (Adata2[i]*s) % N2
    # Bdata2[i] = (Bdata2[i] + Bdata1[i]) % N2
    # Bdata1[i] = (Adata1[i] * s) % N1

    (res1, res2) = karatsuba_scalar_multiply_helper(Adata1[i], Adata2[i], s, N1, N2)

    Bdata1[i] = res1
    Bdata2[i] = res2

    nothing
end

@inline function karatsuba_negate_helper(a1, a2, N1, N2, M)
    res2 = 0.0 % (N2)
    res2 = (res2 + N1) % (2*N1)
    res2 = (res2 - a1) % (2*N1)

    res2 = div(res2, N1)
    res1 = (res2 * N1) % (M^2)
    res1 = (res1 + a1) % (M^2)
    res1 = (-res1) % N1

    res1 = mod(res1, N1)

    res2 = (res2 + N2) % (M^2)
    res2 = (res2 - 1) % (M^2)
    res2 = (res2 - a2) % M
    res2 %= N2

    (res1, res2)
end

function karatsuba_negate_kernel!(Kdata1, Kdata2, Adata1, Adata2, N1, N2, M)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    res1, res2 = karatsuba_negate_helper(Adata1[i], Adata2[i], N1, N2, M)
    Kdata1[i] = res1
    Kdata2[i] = res2

    nothing
end

function karatsuba_sub_kernel!(Kdata1, Kdata2, Adata1, Adata2, Bdata1, Bdata2, N1, N2, M)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    bneg1, bneg2 = karatsuba_negate_helper(Bdata1[i], Bdata2[i], N1, N2, M)

    res1, res2 = karatsuba_add_helper(bneg1, bneg2, Adata1[i], Adata2[i], N1, N2)

    Kdata1[i] = res1
    Kdata2[i] = res2

    nothing
end

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

@inline function _karatsuba_mod_add(a::Float64, b::Float64, modulus::Float64)
    modulus == 1.0 && return 0.0
    gap = modulus - b
    return a >= gap ? a - gap : a + b
end

@inline function _karatsuba_mod_sub(a::Float64, b::Float64, modulus::Float64)
    modulus == 1.0 && return 0.0
    return a >= b ? a - b : modulus - (b - a)
end

@inline function _karatsuba_mod_mul(a::Float64, b::Float64, modulus::Float64)
    # Inputs and the loop state are integer residues in [0, modulus). Each
    # step uses bounded modular addition instead of forming a*b.
    modulus == 1.0 && return 0.0
    result = 0.0
    factor = a
    multiplier = b
    while multiplier > 0.0
        half = div(multiplier, 2.0)
        if multiplier != 2.0 * half
            result = _karatsuba_mod_add(result, factor, modulus)
        end
        multiplier = half
        if multiplier > 0.0
            factor = _karatsuba_mod_add(factor, factor, modulus)
        end
    end
    return result
end

@inline function _karatsuba_recombine(
    p0::Float64,
    p1::Float64,
    p2::Float64,
    radix_mod_capacity::Float64,
    radix_squared_mod_capacity::Float64,
    output_radix::Float64,
    output_capacity::Float64,
)
    # p0, p1, and p2 are the three stripe products reduced modulo Q.
    cross = _karatsuba_mod_sub(p1, p0, output_capacity)
    cross = _karatsuba_mod_sub(cross, p2, output_capacity)
    weighted_cross = _karatsuba_mod_mul(cross, radix_mod_capacity, output_capacity)
    weighted_high = _karatsuba_mod_mul(p2, radix_squared_mod_capacity, output_capacity)
    value = _karatsuba_mod_add(p0, weighted_cross, output_capacity)
    value = _karatsuba_mod_add(value, weighted_high, output_capacity)

    # Q is checked to be at most 2^53, so value and output_radix are exact
    # Float64 integers that also fit Int64. Integer div/rem avoid a rounded
    # floating quotient at large, nearly integral ratios.
    value_integer = Int64(value)
    radix_integer = Int64(output_radix)
    low = Float64(rem(value_integer, radix_integer))
    high = Float64(div(value_integer, radix_integer))
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


# function karatsuba_matmul_kernel_3!(Cplan,Cdata1,Cdata2,Bplan,N1,N2)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x


#     nothing
# end
