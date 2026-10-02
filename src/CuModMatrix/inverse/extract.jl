"""
    pluq_extract_l_kernel!(L, LU, n)

Extract explicit lower-triangular factor from packed `LU` into `L`.
Diagonal entries are set to one.
"""
function pluq_extract_l_kernel!(L, LU, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        if i == j
            L[i, j] = one(eltype(L))
        elseif i > j
            L[i, j] = LU[i, j]
        else
            L[i, j] = zero(eltype(L))
        end
    end
    return
end

"""
    pluq_extract_u_kernel!(U, LU, n)

Extract explicit upper-triangular factor from packed `LU` into `U`.
"""
function pluq_extract_u_kernel!(U, LU, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        if i <= j
            U[i, j] = LU[i, j]
        else
            U[i, j] = zero(eltype(U))
        end
    end
    return
end

"""
    pluq_apply_paq_kernel!(PAQ, A, p, q, n)

Apply permutations in gather form:
`PAQ[i,j] = A[p[i], q[j]]`.
"""
function pluq_apply_paq_kernel!(PAQ, A, p, q, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        PAQ[i, j] = A[p[i], q[j]]
    end
    return
end

"""
    pluq_extract_L(F)

Materialize explicit `L` on GPU from packed `F.LU`.
"""
function pluq_extract_L(F::PLUQFactorization)
    n = rows(F.LU)
    n32 = Int32(n)
    L = zeros(eltype(F.LU.data), n, n, F.LU.N)
    tx = 16
    ty = 16
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(n, ty))) pluq_extract_l_kernel!(
        L.data,
        F.LU.data,
        n32,
    )
    return L
end

"""
    pluq_extract_U(F)

Materialize explicit `U` on GPU from packed `F.LU`.

Example:
```julia
F = pluq_new(A)
U = pluq_extract_U(F)
```
"""
function pluq_extract_U(F::PLUQFactorization)
    n = rows(F.LU)
    n32 = Int32(n)
    U = zeros(eltype(F.LU.data), n, n, F.LU.N)
    tx = 16
    ty = 16
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(n, ty))) pluq_extract_u_kernel!(
        U.data,
        F.LU.data,
        n32,
    )
    return U
end
