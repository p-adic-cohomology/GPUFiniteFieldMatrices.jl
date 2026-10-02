
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

# MARK - Matrix multiplicatino

function karatsuba_matmul_kernel_1!(Aplan, Adata1, Adata2, Bplan, Bdata1, Bdata2, N1)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= length(Bplan)
        Bplan[i] = (Bdata1[i] + Bdata2[i]) % (2*N1)
    end

    Aplan[i] = (Adata1[i] + Adata2[i]) % (2*N1)

    nothing
end

function karatsuba_matmul_kernel_2!(Cdata1, Cdata2, Bplan, N1, N2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    cc1 = mod(Cdata1[i], N1^2)
    cc2 = mod(Cdata2[i], (4*N1)^2)
    bp = mod(Bplan[i], N1)

    cp = div(cc1, N1)
    cc2 = (cc2 - cc1) % ((4*N1)^2)

    cc2 = (cc2 - bp) % ((4*N1)^2)
    cc2 = cc2 + cp

    Cdata1[i] = mod(cc1, N1)
    Cdata2[i] = mod(cc2, N2)

    nothing
end


# function karatsuba_matmul_kernel_3!(Cplan,Cdata1,Cdata2,Bplan,N1,N2)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x


#     nothing
# end
